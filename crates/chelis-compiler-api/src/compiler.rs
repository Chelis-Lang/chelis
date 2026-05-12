use std::collections::{BTreeMap, HashMap, HashSet};

use chelis_backend_c::CodegenResult;
use chelis_backend_hip::HipCodegenResult;
use chelis_deep::Expr as DeepExpr;
use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStepOp, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{self, TensorValue as IrTensorValue};
use chelis_ir::lower::top_level_lowering_map;
use chelis_surf::ast::{
    BinOp, Decl, Expr, ImportKind, LetBinding, LetPattern, Literal, MatchArm, Param, Pattern,
    TypeExpr, UnaryOp, Variant, VariantFields,
};
use chelis_types::{CheckedProgram, errors::CheckError};

use crate::runtime::{
    RuntimeTensorValue, evaluate_host_program_filtered,
    evaluate_host_program_with_library_and_types, lookup_runtime_value_for_root,
    runtime_value_to_schema,
};
use crate::schema::{
    BatchRequest, BatchResult, BatchResultEnvelope, CheckResult, CompileRequest, CompileResult,
    CompileTarget, DecompileRequest, DecompileResult, DesugarRequest, DesugarResult, Diagnostic,
    EvalRequest, EvalResult, EvaluatedRoot, FitnessComponents, GeneratedFile, GradRequest,
    GradResult, LowerRequest, LowerResult, ParseRequest, ParseResult, SourceKind, Span,
    ValidateMode, ValidateRequest, ValidateResult, WireBinOp, WireDag, WireDagNode, WireDeepAtom,
    WireDeepExpr, WireDeepExprKind, WireDimExpr, WireDimInfo, WireFusedInput, WireFusedStep,
    WireFusedStepOp, WireImportKind, WireLetBinding, WireLetPattern, WireLiteral, WireMatchArm,
    WireMetaEntry, WireParam, WirePattern, WirePropertyOption, WireRecordExprField,
    WireRecordPatternField, WireRecordTypeField, WireRiscOp, WireSurfDecl, WireSurfExpr,
    WireSurfTypeExpr, WireTensorType, WireUnaryOp, WireVariant, WireVariantFields,
};

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime.h"
));
const BLAS_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_blas.h"
));
const HIP_RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_hip_runtime.h"
));

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionDim {
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionTensorSpec {
    pub name: String,
    pub dtype: String,
    pub dims: Vec<ExecutionDim>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledExecutionArtifact {
    pub compile_result: CompileResult,
    pub host_entry_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_entry_name: Option<String>,
    pub inputs: Vec<ExecutionTensorSpec>,
    pub outputs: Vec<ExecutionTensorSpec>,
    pub symbolic_dims: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CompilerError {
    pub stage: String,
    pub errors: Vec<Diagnostic>,
}

type Result<T> = std::result::Result<T, CompilerError>;

pub fn parse(request: ParseRequest) -> Result<ParseResult> {
    match request.source_kind {
        SourceKind::Surf => {
            let decls = parse_surf(&request.source)?;
            Ok(ParseResult {
                source_kind: SourceKind::Surf,
                surf_ast: Some(decls.iter().map(wire_decl).collect()),
                deep_ast: None,
            })
        }
        SourceKind::Deep => {
            let exprs = parse_deep(&request.source)?;
            Ok(ParseResult {
                source_kind: SourceKind::Deep,
                surf_ast: None,
                deep_ast: Some(exprs.iter().map(wire_deep_expr).collect()),
            })
        }
    }
}

pub fn desugar(request: DesugarRequest) -> Result<DesugarResult> {
    let decls = parse_surf(&request.source)?;
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    Ok(DesugarResult {
        deep_text: chelis_deep::printer::print_canonical(&deep_exprs),
        deep_ast: deep_exprs.iter().map(wire_deep_expr).collect(),
    })
}

pub fn check(request: crate::schema::CheckRequest) -> Result<CheckResult> {
    let deep_exprs = deep_exprs_from_source(request.source_kind, &request.source)?;
    let report = chelis_types::check_ir_fitness(&deep_exprs);
    Ok(CheckResult {
        score: report.score,
        components: FitnessComponents {
            parse: report.components.parse,
            structure: report.components.structure,
            names: report.components.names,
            types: report.components.types,
        },
        typed_nodes: report.typed_nodes,
        untyped_nodes: report.untyped_nodes,
        total_nodes: report.total_nodes,
        unresolved_names: report.unresolved_names,
        errors: report.errors.iter().map(check_error_diagnostic).collect(),
    })
}

pub fn lower(request: LowerRequest) -> Result<LowerResult> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    Ok(LowerResult {
        dag: wire_dag(&compiled.dag),
        named_roots: compiled
            .named_roots
            .into_iter()
            .map(|(k, v)| (k, v.0))
            .collect(),
    })
}

pub fn compile(request: CompileRequest) -> Result<CompileResult> {
    Ok(compile_for_execution(request)?.compile_result)
}

pub fn compile_for_execution(request: CompileRequest) -> Result<CompiledExecutionArtifact> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    let host_compiled = chelis_ir::host::lower_compiled_program(&compiled.checked);
    let func_name = request
        .entry_name
        .unwrap_or_else(|| "chelis_main".to_string());

    match request.target {
        CompileTarget::C => {
            if let Some(host_program) = host_compiled.host.as_ref()
                && (chelis_ir::host::host_program_requires_host_backend(host_program)
                    || compiled.dag.roots().is_empty())
            {
                let result = chelis_backend_c::codegen_host_program(host_program, &func_name);
                return Ok(compiled_execution_artifact(
                    request.target,
                    &func_name,
                    None,
                    compile_result_c(request.target, &func_name, &result),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ));
            }
            reject_unsized_named_dims(&compiled.dag, "c")?;
            let specialized = chelis_ir::specialize::specialize_for_blas(&compiled.dag);
            let fused = chelis_ir::fuse::fuse(&specialized);
            let result = chelis_backend_c::codegen_with_options(
                &fused,
                &func_name,
                chelis_backend_c::CodegenOptions {
                    use_blas: true,
                    ..chelis_backend_c::CodegenOptions::default()
                },
            );
            Ok(compiled_execution_artifact(
                request.target,
                &func_name,
                None,
                compile_result_c(request.target, &func_name, &result),
                execution_input_specs(&compiled.dag, &result.input_labels)?,
                execution_output_specs(&compiled.dag, &result.output_labels)?,
                result.symbolic_dims,
            ))
        }
        CompileTarget::Hip => {
            let host_requires_host_backend = host_compiled
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            let preferred_entry_dag = host_compiled
                .host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| {
                    chelis_ir::host::lower_named_tensor_entry_dag(&compiled.checked, name)
                });
            if compiled.dag.roots().is_empty()
                && preferred_entry_dag.is_none()
                && host_requires_host_backend
                && let Some(host_program) = host_compiled.host.as_ref()
            {
                let result = chelis_backend_c::codegen_host_program(host_program, &func_name);
                return Ok(compiled_execution_artifact(
                    request.target,
                    &func_name,
                    None,
                    compile_result_hip_host(request.target, &func_name, &result),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ));
            }
            let mut hip_dag = if !compiled.dag.roots().is_empty() {
                compiled.dag.clone()
            } else if let Some(entry_dag) = preferred_entry_dag {
                entry_dag
            } else {
                compiled.dag.clone()
            };
            hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
            reject_unsized_named_dims(&hip_dag, "hip")?;
            let specialized = chelis_ir::specialize::specialize_for_blas(&hip_dag);
            reject_unsupported_hip_ops(&specialized)?;
            let fused = chelis_ir::fuse::fuse(&specialized);
            let result = chelis_backend_hip::codegen_hip(&fused, &func_name);
            Ok(compiled_execution_artifact(
                request.target,
                &func_name,
                Some(format!("{func_name}_device")),
                compile_result_hip(request.target, &func_name, &result),
                execution_input_specs(&hip_dag, &result.input_labels)?,
                execution_output_specs(&hip_dag, &result.output_labels)?,
                result.symbolic_dims,
            ))
        }
    }
}

pub fn eval(request: EvalRequest) -> Result<EvalResult> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    eval_compiled(&compiled, request.bindings, None)
}

pub fn eval_selected(request: EvalRequest, selected_root_names: &[String]) -> Result<EvalResult> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    eval_compiled(&compiled, request.bindings, Some(selected_root_names))
}

/// Compile the source once, then evaluate it once per entry in `test_roots`,
/// returning a parallel Vec of per-root `(name, Result<EvalResult>)` pairs.
///
/// Each root is evaluated with the full `bindings` from the request re-used
/// unchanged. A failing root does not short-circuit the remaining roots — the
/// caller sees every root's outcome independently.
///
/// If the source fails to compile, the compile error is propagated to every
/// requested root. That keeps the return shape a 1:1 mapping with `test_roots`
/// (matching `chelis test`'s per-root reporting) and is easier for callers
/// than a mixed Result<Vec<_>> + compile-error channel.
///
/// **Deprecated since 0.3.0** — prefer
/// [`eval_many_in_context`] with a [`crate::CompiledContext`]. The
/// in-context path amortizes the library compile across every call so
/// per-invocation work drops to just the new source.
#[deprecated(
    since = "0.3.0",
    note = "use eval_many_in_context with a CompiledContext for ~5x faster amortized eval; see crates/chelis-compiler-api/src/compiler.rs::eval_many_in_context"
)]
pub fn eval_many(request: EvalRequest, test_roots: &[String]) -> Vec<(String, Result<EvalResult>)> {
    let compiled = match compile_source(request.source_kind, &request.source) {
        Ok(compiled) => compiled,
        Err(err) => {
            return test_roots
                .iter()
                .map(|name| (name.clone(), Err(err.clone())))
                .collect();
        }
    };

    test_roots
        .iter()
        .map(|name| {
            let roots_slice = std::slice::from_ref(name);
            let outcome = eval_compiled(&compiled, request.bindings.clone(), Some(roots_slice));
            (name.clone(), outcome)
        })
        .collect()
}

/// Opaque handle over a compiled program. Once prepared, any number of
/// `eval_root` calls share the same underlying Surf→Deep→DAG compile —
/// what the `chelis test` CLI needs to avoid paying the ~2.3s-per-test
/// recompilation that killed dev-loop ergonomics.
///
/// Cheap to clone (Arc-wrapped internals) and safe to Send into a worker
/// thread for per-test timeout isolation.
#[derive(Clone)]
pub struct PreparedEval {
    compiled: std::sync::Arc<CompiledSource>,
}

impl PreparedEval {
    /// Evaluate exactly one selected root. Other top-level non-fn bindings
    /// stay registered for lazy reference but are not eagerly evaluated.
    pub fn eval_root(
        &self,
        bindings: BTreeMap<String, crate::schema::TensorValue>,
        root: &str,
    ) -> Result<EvalResult> {
        let roots = [root.to_string()];
        eval_compiled(&self.compiled, bindings, Some(&roots))
    }
}

/// Compile a program once so downstream callers can cheaply evaluate
/// specific roots against it many times. Used by `chelis test` to share
/// one compile across every test in a file.
///
/// **Deprecated since 0.3.0** — prefer
/// [`prepare_eval_in_context`] with a [`crate::CompiledContext`]. The
/// in-context path keeps this function's amortization-across-roots
/// guarantee but additionally amortizes the library compile across
/// every CLI invocation that hits the same package, dropping cold
/// `chelis test` and `chelis eval` startup from O(library) to O(new
/// source). Retained for backward compatibility and as the
/// `LocalRegistry` fallback path inside `chelis test` until
/// `source_digests` gains `LocalRegistry` support.
#[deprecated(
    since = "0.3.0",
    note = "use prepare_eval_in_context with a CompiledContext for ~5x faster amortized eval; see crates/chelis-compiler-api/src/compiler.rs::prepare_eval_in_context"
)]
pub fn prepare_eval(request: EvalRequest) -> Result<PreparedEval> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    Ok(PreparedEval {
        compiled: std::sync::Arc::new(compiled),
    })
}

// ---- Phase G: in-context eval / check ----------------------------------
//
// `eval_in_context` / `check_in_context` / `eval_many_in_context` accept a
// pre-built `CompiledContext` (the library — chelis-std + reef deps + the
// package's own modules — already type-checked and lowered ONCE) plus the
// new source for this specific eval. Only the new source runs through
// parse/desugar/macro-expand + the C/D/E/F `_with_context` checker
// variants + the composing lowerer.

/// Strip `Decl::Module` wrappers — `compile_with_reef_graph` (and its
/// Phase G sibling `rewrite_entry_decls_with_reef_graph`) expect a flat
/// decl list. Mirrors the helper used by `chelis test`'s
/// `cmd_internal_test_file`.
fn flatten_module_decls(decls: &[Decl]) -> Vec<Decl> {
    let mut out = Vec::new();
    for decl in decls {
        match decl {
            Decl::Module { decls: inner, .. } => {
                out.extend(flatten_module_decls(inner));
            }
            other => out.push(other.clone()),
        }
    }
    out
}

fn compile_new_source_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
) -> Result<CompiledSource> {
    // Phase G inputs are always Surf — `compile_reef_context` already
    // resolved the package's library decls; the new source is whatever
    // the user typed into a `chelis eval --file` / `chelis test` worker /
    // `chelis check` call.
    let raw_new_decls = parse_surf(new_source)?;
    // Strip module wrappers and route through the reef name resolver so
    // bare references like `add` get rewritten to their internal-name
    // form (`mylib.math.add`) — matching what
    // `compile_with_reef_graph` does for the monolithic eval path. This
    // is what makes the new code's references resolve against the
    // library state stored in the `CompiledContext`.
    let flat_decls = flatten_module_decls(&raw_new_decls);
    let rewritten =
        chelis_reef::rewrite_entry_decls_with_reef_graph(&context.reef_state, &flat_decls)
            .map_err(|err| stage_error("reef", err, "reef_error"))?;
    let new_deep = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&rewritten),
        &chelis_macros::ExpansionOptions::default(),
    )
    .map_err(|err| stage_error("desugar", err.to_string(), "macro_error"))?
    .into_exprs();

    // Phase C: type-check new code against the library type env.
    let new_checked = chelis_types::check_ir_with_signature_context(
        &context.type_env,
        context.library_checked.signature_inference(),
        &new_deep,
    )
    .map_err(|report| CompilerError {
        stage: "check".to_string(),
        errors: report.errors.iter().map(check_error_diagnostic).collect(),
    })?;

    // Phase D: effects checker, library + new.
    let new_checked =
        chelis_effects::check_effects_with_context(&context.library_checked, &new_checked)
            .map_err(|errors| CompilerError {
                stage: "effects".to_string(),
                errors: errors
                    .iter()
                    .map(|error| Diagnostic {
                        kind: "effect_error".to_string(),
                        message: error.message.clone(),
                        severity: 0.8,
                        expected: None,
                        got: None,
                        suggestions: vec![],
                        span: None,
                    })
                    .collect(),
            })?;

    // Phase E: linearity checker, library + new.
    let new_checked =
        chelis_types::check_linearity_with_context(&context.library_checked, &new_checked)
            .map_err(|errors| CompilerError {
                stage: "linearity".to_string(),
                errors: errors.iter().map(check_error_diagnostic).collect(),
            })?;

    // Lower against the cached library DAG.
    let composed_dag =
        chelis_ir::lower::try_lower_program_with_context(&context.library_dag, &new_checked)
            .map_err(|diagnostic| {
                stage_error_with_span(
                    "lower",
                    diagnostic.to_string(),
                    "lower_error",
                    deep_span_to_schema(diagnostic.span),
                )
            })?;

    // Build the same CompiledSource shape `compile_source` produces, but
    // for the new code only — the library state lives in the composed
    // Dag and the type-env is unioned so eval-time name resolution
    // continues to find library symbols.
    let all_root_names =
        root_names_from_checked_exprs(new_checked.exprs(), new_checked.type_env(), false);
    // For the lowered-only filter (which feeds the tensor evaluator's
    // root list), use the context-aware lowering map so a new-code def
    // referencing a library function inherits the library's
    // lowered-vs-host classification — matching the monolithic
    // `lower_program(library + new)` decision byte-for-byte.
    let combined_lowered_names = chelis_ir::lower::top_level_lowering_map_with_context(
        &context.library_dag,
        new_checked.exprs(),
        new_checked.type_env(),
    );
    let new_tensor_root_names = root_names_from_checked_exprs_with_lowered_map(
        new_checked.exprs(),
        new_checked.type_env(),
        Some(&combined_lowered_names),
    );

    // The composed Dag's roots are [library_roots ..., new_roots ...].
    // Slice to the new-code tail so `tensor_root_names` aligns 1:1 with
    // the roots `eval_compiled` will iterate.
    let library_root_count = context.library_dag.dag.roots().len();
    let composed_roots = composed_dag.roots();
    let new_root_slice_start = library_root_count.min(composed_roots.len());
    let new_root_ids: Vec<NodeId> = composed_roots[new_root_slice_start..].to_vec();

    if !new_tensor_root_names.is_empty() && new_root_ids.len() != new_tensor_root_names.len() {
        return Err(stage_error(
            "lower",
            format!(
                "lowered new-code root count mismatch: expected {} named roots, got {}",
                new_tensor_root_names.len(),
                new_root_ids.len()
            ),
            "lower_error",
        ));
    }

    let named_roots = new_tensor_root_names
        .iter()
        .cloned()
        .zip(new_root_ids.iter().copied())
        .collect::<BTreeMap<_, _>>();

    let mut forward_nodes_by_name = named_roots.clone();
    for node in composed_dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            forward_nodes_by_name
                .entry(name.as_str().to_string())
                .or_insert(node.id);
        }
    }

    // Replace the composed Dag's roots vector with just the new-code
    // roots so `eval_compiled`'s iteration over `dag.roots()` aligns with
    // `tensor_root_names`. The composed Dag keeps every node (library +
    // new) so dependency lookups during evaluation still resolve.
    let mut dag_for_eval = composed_dag;
    dag_for_eval.set_roots(new_root_ids);

    // Phase G' — carry the library defs + lowered classification into
    // the runtime. Without this, the host evaluator's `top_level_defs`
    // would see only new code's defs and would error
    // `unknown runtime name pkg__chelis__std__Std__Time__is_leap_year`
    // on any new-code call into a library function.
    let library_runtime = LibraryRuntime {
        exprs: context.library_checked.annotated_exprs().to_vec(),
        type_env: context.library_checked.type_env().clone(),
        lowered_names: crate::runtime::library_lowered_names(
            context.library_checked.annotated_exprs(),
            context.library_checked.type_env(),
        ),
    };

    Ok(CompiledSource {
        checked: new_checked,
        dag: dag_for_eval,
        all_root_names,
        tensor_root_names: new_tensor_root_names,
        named_roots,
        forward_nodes_by_name,
        library_runtime: Some(library_runtime),
    })
}

/// Evaluate `new_source` against an existing `CompiledContext`. Equivalent
/// to `eval(EvalRequest { source: format(library + new_source), ... })`
/// for the produced root values, but the library-side compile work has
/// already been amortized across every prior `eval_in_context` call on
/// the same context.
pub fn eval_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
) -> Result<EvalResult> {
    let compiled = compile_new_source_in_context(context, new_source)?;
    eval_compiled(&compiled, BTreeMap::new(), None)
}

/// Type-/effects-/linearity-check `new_source` against an existing
/// `CompiledContext`. Returns a `CheckResult` with the same shape as
/// the existing `check` API. A successful result means the new code
/// composes cleanly with the library; errors carry new-code spans.
pub fn check_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
) -> Result<CheckResult> {
    let compiled = compile_new_source_in_context(context, new_source)?;
    // Mirror `check`'s shape: derive a fitness-style report from the
    // composed checked program. The total/typed counts only cover
    // new-code nodes — library nodes are checked once at context build
    // time and counted there.
    let total_nodes = compiled.checked.exprs().len();
    Ok(CheckResult {
        score: 1.0,
        components: FitnessComponents {
            parse: 1.0,
            structure: 1.0,
            names: 1.0,
            types: 1.0,
        },
        typed_nodes: total_nodes,
        untyped_nodes: 0,
        total_nodes,
        unresolved_names: vec![],
        errors: vec![],
    })
}

/// Compile `new_source` once against `context`, then evaluate it once per
/// entry in `roots`. Mirrors the contract of `eval_many` but on the
/// in-context API: the per-root iteration shares the new-source compile,
/// and a failing root does not short-circuit the others.
pub fn eval_many_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
    roots: &[String],
) -> Vec<(String, Result<EvalResult>)> {
    let compiled = match compile_new_source_in_context(context, new_source) {
        Ok(compiled) => compiled,
        Err(err) => {
            return roots
                .iter()
                .map(|name| (name.clone(), Err(err.clone())))
                .collect();
        }
    };

    roots
        .iter()
        .map(|name| {
            let roots_slice = std::slice::from_ref(name);
            let outcome = eval_compiled(&compiled, BTreeMap::new(), Some(roots_slice));
            (name.clone(), outcome)
        })
        .collect()
}

/// Opaque handle over `new_source` compiled against a `CompiledContext`.
/// Mirrors the [`PreparedEval`] handle but for the in-context path: any
/// number of `eval_root` calls share the single Surf→Deep→DAG compile of
/// the new source on top of the context's library snapshot.
///
/// Used by `chelis test`'s per-file worker: one compile per file (not per
/// test), with each test isolated through the worker-side per-test
/// timeout wrapper.
#[derive(Clone)]
pub struct PreparedEvalInContext {
    compiled: std::sync::Arc<CompiledSource>,
}

impl PreparedEvalInContext {
    /// Evaluate exactly one selected root against the prepared compile.
    pub fn eval_root(
        &self,
        bindings: BTreeMap<String, crate::schema::TensorValue>,
        root: &str,
    ) -> Result<EvalResult> {
        let roots = [root.to_string()];
        eval_compiled(&self.compiled, bindings, Some(&roots))
    }
}

/// Compile `new_source` once against `context` and return a handle so
/// downstream callers can cheaply evaluate specific roots against it many
/// times. The composed library + new-source pipeline runs once; each
/// per-root eval reuses the result.
pub fn prepare_eval_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
) -> Result<PreparedEvalInContext> {
    let compiled = compile_new_source_in_context(context, new_source)?;
    Ok(PreparedEvalInContext {
        compiled: std::sync::Arc::new(compiled),
    })
}

fn eval_compiled(
    compiled: &CompiledSource,
    bindings: BTreeMap<String, crate::schema::TensorValue>,
    selected_root_names: Option<&[String]>,
) -> Result<EvalResult> {
    let selected = selected_root_names.map(|roots| {
        roots
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<String>>()
    });
    if compiled.dag.roots().is_empty() && compiled.all_root_names.is_empty() {
        return Err(stage_error(
            "eval",
            "program produced no evaluable roots",
            "other",
        ));
    }

    let bindings = bindings
        .into_iter()
        .map(|(name, value)| {
            (
                name,
                IrTensorValue {
                    shape: value.shape,
                    data: value.data,
                },
            )
        })
        .collect::<HashMap<_, _>>();

    let roots = compiled
        .tensor_root_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            if selected.as_ref().is_some_and(|set| !set.contains(name)) {
                return None;
            }
            compiled.dag.roots().get(index).copied()
        })
        .collect::<Vec<_>>();
    let tensor_values = if roots.is_empty() {
        HashMap::new()
    } else {
        eval::eval_tensor_roots_with_strict(&compiled.dag, &roots, |name| {
            bindings.get(name).cloned()
        })
        .map_err(|message| stage_error("eval", message, "eval_error"))?
    };

    let mut tensor_values_by_name = HashMap::<String, RuntimeTensorValue>::new();
    for name in &compiled.tensor_root_names {
        if selected.as_ref().is_some_and(|set| !set.contains(name)) {
            continue;
        }
        let Some(node_id) = compiled.named_roots.get(name) else {
            continue;
        };
        let value = tensor_values.get(node_id).ok_or_else(|| {
            stage_error(
                "eval",
                format!("missing tensor root `{name}`"),
                "eval_error",
            )
        })?;
        let precision = compiled
            .dag
            .get(*node_id)
            .map(|node| node.output_type.precision)
            .ok_or_else(|| {
                stage_error("eval", format!("missing node {}", node_id.0), "eval_error")
            })?;
        tensor_values_by_name.insert(
            name.clone(),
            RuntimeTensorValue {
                value: value.clone(),
                precision,
            },
        );
    }

    // When a selected-roots filter is set (eval_selected / eval_many), push it
    // through to the host-program evaluator so only the selected non-fn
    // top-levels are eagerly evaluated. This is what lets a single compile
    // feed many per-test evaluations in `chelis test` without every test
    // paying for the others' module-init side effects.
    //
    // Phase G' — when `compiled.library_runtime` is `Some`, route through
    // the library-aware evaluator entry point so library `def`s (e.g.
    // chelis-std functions) are reachable from new-code calls. Without
    // this carry-over, the runtime errored
    // `unknown runtime name pkg__chelis__std__...`.
    let host_outcome = if let Some(library) = compiled.library_runtime.as_ref() {
        evaluate_host_program_with_library_and_types(
            &compiled.checked,
            &library.exprs,
            &library.type_env,
            Some(&library.lowered_names),
            &tensor_values_by_name,
            selected_root_names,
        )
    } else {
        evaluate_host_program_filtered(
            &compiled.checked,
            &tensor_values_by_name,
            selected_root_names,
        )
    }
    .map_err(|message| stage_error("eval", message, "eval_error"))?;

    let roots = compiled
        .all_root_names
        .iter()
        .enumerate()
        .filter(|(_, name)| selected.as_ref().is_none_or(|set| set.contains(*name)))
        .filter_map(|(index, name)| {
            let value = lookup_runtime_value_for_root(
                name,
                &host_outcome.host_bindings,
                &tensor_values_by_name,
            )?;
            let node_id = compiled
                .named_roots
                .get(name)
                .map(|id| id.0)
                .unwrap_or(index);
            Some((node_id, name.clone(), value))
        })
        .map(|(node_id, name, value)| {
            Ok(EvaluatedRoot {
                node_id,
                name: Some(name),
                value: runtime_value_to_schema(&value)
                    .map_err(|message| stage_error("eval", message, "eval_error"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(EvalResult {
        roots,
        transcript: host_outcome.transcript,
    })
}

pub fn grad(request: GradRequest) -> Result<GradResult> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    let output = compiled
        .forward_nodes_by_name
        .get(&request.output_name)
        .copied()
        .ok_or_else(|| unknown_name_error("grad", "output_name", &request.output_name))?;

    let wrt_nodes = request
        .wrt_names
        .iter()
        .map(|name| {
            compiled
                .forward_nodes_by_name
                .get(name)
                .copied()
                .ok_or_else(|| unknown_name_error("grad", "wrt_names", name))
        })
        .collect::<Result<Vec<_>>>()?;

    let grad_result = if request.fuse {
        chelis_ir::grad_then_fuse(&compiled.dag, output, &wrt_nodes)
    } else {
        chelis_ir::grad::grad_dag(&compiled.dag, output, &wrt_nodes)
    }
    .ok_or_else(|| {
        stage_error(
            "grad",
            "grad requires a scalar floating output and a valid unfused forward DAG",
            "grad_error",
        )
    })?;

    let grad_nodes_by_name = request
        .wrt_names
        .iter()
        .zip(wrt_nodes.iter())
        .filter_map(|(name, node)| {
            grad_result
                .grad_nodes
                .get(node)
                .copied()
                .map(|grad_node| (name.clone(), grad_node.0))
        })
        .collect();

    Ok(GradResult {
        dag: wire_dag(&grad_result.dag),
        output_node: grad_result.output_node.0,
        grad_nodes_by_name,
        forward_nodes_by_name: compiled
            .forward_nodes_by_name
            .into_iter()
            .map(|(name, node)| (name, node.0))
            .collect(),
    })
}

pub fn validate(request: ValidateRequest) -> Result<ValidateResult> {
    let result = match request.mode {
        ValidateMode::Surf => chelis_validate::validate_surf(&request.source),
        ValidateMode::Deep => chelis_validate::validate_deep(&request.source),
        ValidateMode::Desugar => chelis_validate::validate_desugared(&request.source),
    };

    result.map_err(|err| stage_error("validate", err.to_string(), "validation_error"))?;

    Ok(ValidateResult {
        mode: request.mode,
        valid: true,
    })
}

pub fn decompile(request: DecompileRequest) -> Result<DecompileResult> {
    let exprs = parse_deep(&request.source)?;
    Ok(DecompileResult {
        surf_text: chelis_surf::decompile::decompile_program(&exprs),
    })
}

pub fn batch(requests: Vec<BatchRequest>) -> BatchResultEnvelope {
    BatchResultEnvelope {
        results: requests
            .into_iter()
            .map(|request| match request {
                BatchRequest::Parse(req) => BatchResult::Parse(result_envelope(parse(req))),
                BatchRequest::Desugar(req) => BatchResult::Desugar(result_envelope(desugar(req))),
                BatchRequest::Check(req) => BatchResult::Check(result_envelope(check(req))),
                BatchRequest::Lower(req) => BatchResult::Lower(result_envelope(lower(req))),
                BatchRequest::Compile(req) => BatchResult::Compile(result_envelope(compile(req))),
                BatchRequest::Eval(req) => BatchResult::Eval(result_envelope(eval(req))),
                BatchRequest::Grad(req) => BatchResult::Grad(result_envelope(grad(req))),
                BatchRequest::Validate(req) => {
                    BatchResult::Validate(result_envelope(validate(req)))
                }
                BatchRequest::Decompile(req) => {
                    BatchResult::Decompile(result_envelope(decompile(req)))
                }
            })
            .collect(),
    }
}

pub fn result_envelope<T>(result: Result<T>) -> crate::schema::ApiEnvelope<T> {
    match result {
        Ok(value) => crate::schema::ApiEnvelope::success(value),
        Err(err) => crate::schema::ApiEnvelope::failure(err.stage, err.errors),
    }
}

struct CompiledSource {
    checked: CheckedProgram,
    dag: Dag,
    all_root_names: Vec<String>,
    tensor_root_names: Vec<String>,
    named_roots: BTreeMap<String, NodeId>,
    forward_nodes_by_name: BTreeMap<String, NodeId>,
    /// Phase G' — optional library context payload threaded into the
    /// host evaluator so library `def` names resolve at runtime when
    /// new code calls them. `None` on the monolithic `compile_source`
    /// path (no separate library to merge); `Some` on the in-context
    /// path produced by `compile_new_source_in_context`.
    library_runtime: Option<LibraryRuntime>,
}

/// The library payload threaded through `eval_compiled` so the host
/// evaluator's `top_level_defs` table can resolve library function
/// references when called from new code.
#[derive(Clone)]
struct LibraryRuntime {
    /// Library `def` annotated_exprs. Pulled into `top_level_defs`
    /// before the new-code defs so new-code can shadow on collision.
    exprs: Vec<DeepExpr>,
    /// Library-side type-env. Bucket 1 (`grad`/`vmap`/`realize` host
    /// runtime support) routes through `lower_subexpr_program`, which
    /// expects the merged library + new-code Deep type-env so a
    /// library-name reference inside a `grad` body resolves the same
    /// way it does in the monolithic compile.
    type_env: HashMap<String, DeepExpr>,
    /// Library-side lowered-vs-host classification. Threaded through
    /// so `evaluate_host_program_with_library`'s "is this a tensor
    /// root vs a host-init" decision is byte-identical to what the
    /// monolithic pipeline would have computed.
    lowered_names: HashMap<String, bool>,
}

fn compile_source(source_kind: SourceKind, source: &str) -> Result<CompiledSource> {
    let deep_exprs: Vec<DeepExpr> = match source_kind {
        SourceKind::Surf => {
            let decls = parse_surf(source)?;
            chelis_macros::expand_program(
                &chelis_surf::desugar::desugar_program(&decls),
                &chelis_macros::ExpansionOptions::default(),
            )
            .map_err(|err| stage_error("desugar", err.to_string(), "macro_error"))?
            .into_exprs()
        }
        SourceKind::Deep => parse_deep(source)?,
    };

    let checked = chelis_types::check_ir_program(&deep_exprs).map_err(|report| CompilerError {
        stage: "check".to_string(),
        errors: report.errors.iter().map(check_error_diagnostic).collect(),
    })?;
    let checked = chelis_effects::check_program(&checked).map_err(|errors| CompilerError {
        stage: "effects".to_string(),
        errors: errors
            .iter()
            .map(|error| Diagnostic {
                kind: "effect_error".to_string(),
                message: error.message.clone(),
                severity: 0.8,
                expected: None,
                got: None,
                suggestions: vec![],
                span: None,
            })
            .collect(),
    })?;
    let checked = chelis_types::check_linearity(&checked).map_err(|errors| CompilerError {
        stage: "linearity".to_string(),
        errors: errors.iter().map(check_error_diagnostic).collect(),
    })?;
    let all_root_names = root_names_from_checked_exprs(checked.exprs(), checked.type_env(), false);
    let tensor_root_names =
        root_names_from_checked_exprs(checked.exprs(), checked.type_env(), true);

    let dag = chelis_ir::lower::try_lower_program(&checked).map_err(|diagnostic| {
        stage_error_with_span(
            "lower",
            diagnostic.to_string(),
            "lower_error",
            deep_span_to_schema(diagnostic.span),
        )
    })?;

    if !tensor_root_names.is_empty() && dag.roots().len() != tensor_root_names.len() {
        return Err(stage_error(
            "lower",
            format!(
                "lowered root count mismatch: expected {} named roots, got {}",
                tensor_root_names.len(),
                dag.roots().len()
            ),
            "lower_error",
        ));
    }

    let named_roots = tensor_root_names
        .iter()
        .cloned()
        .zip(dag.roots().iter().copied())
        .collect::<BTreeMap<_, _>>();

    let mut forward_nodes_by_name = named_roots.clone();
    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            forward_nodes_by_name
                .entry(name.as_str().to_string())
                .or_insert(node.id);
        }
    }

    Ok(CompiledSource {
        checked,
        dag,
        all_root_names,
        tensor_root_names,
        named_roots,
        forward_nodes_by_name,
        library_runtime: None,
    })
}

fn deep_exprs_from_source(source_kind: SourceKind, source: &str) -> Result<Vec<DeepExpr>> {
    match source_kind {
        SourceKind::Surf => {
            let decls = parse_surf(source)?;
            chelis_macros::expand_program(
                &chelis_surf::desugar::desugar_program(&decls),
                &chelis_macros::ExpansionOptions::default(),
            )
            .map(|expanded| expanded.into_exprs())
            .map_err(|err| stage_error("desugar", err.to_string(), "macro_error"))
        }
        SourceKind::Deep => parse_deep(source),
    }
}

fn parse_surf(source: &str) -> Result<Vec<Decl>> {
    chelis_surf::parser::parse_str(source).map_err(|err| {
        stage_error_with_span(
            "parse",
            err.to_string(),
            "surf_parse_error",
            parse_error_span_surf(source, &err),
        )
    })
}

fn parse_deep(source: &str) -> Result<Vec<DeepExpr>> {
    chelis_deep::parser::parse_str(source).map_err(|err| {
        stage_error_with_span(
            "parse",
            err.to_string(),
            "deep_parse_error",
            parse_error_span_deep(&err),
        )
    })
}

fn root_names_from_checked_exprs(
    exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
    lowered_only: bool,
) -> Vec<String> {
    let lowered_names = lowered_only.then(|| top_level_lowering_map(exprs, type_env));
    let mut names = Vec::new();
    for expr in exprs {
        collect_checked_decl_names(expr, type_env, lowered_names.as_ref(), &mut names);
    }
    names
}

/// Phase G variant: same as [`root_names_from_checked_exprs`] but uses a
/// pre-computed `lowered_names` map (typically from
/// `top_level_lowering_map_with_context`) so library-context lowering
/// decisions feed the new-code's root-name filter.
fn root_names_from_checked_exprs_with_lowered_map(
    exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
    lowered_names: Option<&HashMap<String, bool>>,
) -> Vec<String> {
    let mut names = Vec::new();
    for expr in exprs {
        collect_checked_decl_names(expr, type_env, lowered_names, &mut names);
    }
    names
}

fn collect_checked_decl_names(
    expr: &DeepExpr,
    type_env: &HashMap<String, DeepExpr>,
    lowered_names: Option<&HashMap<String, bool>>,
    out: &mut Vec<String>,
) {
    let DeepExpr::List(list, _) = expr else {
        return;
    };
    let Some(tag) = list_tag(list) else {
        return;
    };
    match tag {
        "module" => {
            for child in list.elements.iter().skip(3) {
                collect_checked_decl_names(child, type_env, lowered_names, out);
            }
        }
        "def" => {
            if let Some(name) = list.elements.get(2).and_then(symbol_name) {
                if lowered_names.is_some_and(|map| !map.get(name).copied().unwrap_or(false)) {
                    return;
                }
                let body = list.elements.get(3);
                let ty = type_env
                    .get(name)
                    .or_else(|| body.and_then(expr_type_metadata));
                extend_root_names_from_value(name, ty, body, out);
            }
        }
        _ => {}
    }
}

fn extend_root_names_from_value(
    name: &str,
    ty: Option<&DeepExpr>,
    value: Option<&DeepExpr>,
    out: &mut Vec<String>,
) {
    if let Some(DeepExpr::List(list, _)) = ty
        && let Some(tag) = list_tag(list)
    {
        if tag == "t-fn" {
            extend_root_names_from_value(name, list.elements.last(), None, out);
            return;
        }
        if tag == "t-tuple" {
            for (index, child) in list.elements.iter().skip(2).enumerate() {
                extend_root_names_from_value(&format!("{name}.{index}"), Some(child), None, out);
            }
            return;
        }
    }
    if let Some(DeepExpr::List(list, _)) = value
        && list_tag(list) == Some("tuple")
    {
        for (index, child) in list.elements.iter().skip(2).enumerate() {
            extend_root_names_from_value(
                &format!("{name}.{index}"),
                expr_type_metadata(child),
                Some(child),
                out,
            );
        }
        return;
    }
    out.push(name.to_string());
}

fn expr_type_metadata(expr: &DeepExpr) -> Option<&DeepExpr> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(DeepExpr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn list_tag(list: &chelis_deep::List) -> Option<&str> {
    list.elements.first().and_then(symbol_name)
}

fn symbol_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(chelis_deep::Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn compiled_execution_artifact(
    _target: CompileTarget,
    host_entry_name: &str,
    device_entry_name: Option<String>,
    compile_result: CompileResult,
    inputs: Vec<ExecutionTensorSpec>,
    outputs: Vec<ExecutionTensorSpec>,
    symbolic_dims: Vec<String>,
) -> CompiledExecutionArtifact {
    CompiledExecutionArtifact {
        compile_result,
        host_entry_name: host_entry_name.to_string(),
        device_entry_name,
        inputs,
        outputs,
        symbolic_dims,
    }
}

fn compile_result_c(
    target: CompileTarget,
    func_name: &str,
    result: &CodegenResult,
) -> CompileResult {
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    CompileResult {
        target,
        entry_name: func_name.to_string(),
        files: vec![
            GeneratedFile {
                path: format!("{func_name}.c"),
                contents: result.c_source.clone(),
            },
            GeneratedFile {
                path: format!("{func_name}.h"),
                contents: result.h_header.clone(),
            },
            GeneratedFile {
                path: "chelis_runtime.h".to_string(),
                contents: RUNTIME_H.to_string(),
            },
            GeneratedFile {
                path: "chelis_blas.h".to_string(),
                contents: BLAS_H.to_string(),
            },
        ],
        compile_flags: toolchain.compile_flags,
        link_flags: toolchain.link_flags,
        peak_device_bytes_estimate: None,
    }
}

fn compile_result_hip(
    target: CompileTarget,
    func_name: &str,
    result: &HipCodegenResult,
) -> CompileResult {
    CompileResult {
        target,
        entry_name: func_name.to_string(),
        files: vec![
            GeneratedFile {
                path: format!("{func_name}_hip.cpp"),
                contents: result.c_source.clone(),
            },
            GeneratedFile {
                path: format!("{func_name}_hip.h"),
                contents: result.h_header.clone(),
            },
            GeneratedFile {
                path: "chelis_runtime.h".to_string(),
                contents: RUNTIME_H.to_string(),
            },
            GeneratedFile {
                path: "chelis_hip_runtime.h".to_string(),
                contents: HIP_RUNTIME_H.to_string(),
            },
        ],
        compile_flags: result.compile_flags.clone(),
        link_flags: result.link_flags.clone(),
        peak_device_bytes_estimate: result.peak_device_bytes_estimate,
    }
}

fn compile_result_hip_host(
    target: CompileTarget,
    func_name: &str,
    result: &CodegenResult,
) -> CompileResult {
    let mut toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    toolchain.compile_flags.retain(|flag| flag != "-fopenmp");
    toolchain.link_flags.retain(|flag| flag != "-fopenmp");
    CompileResult {
        target,
        entry_name: func_name.to_string(),
        files: vec![
            GeneratedFile {
                path: format!("{func_name}_hip.cpp"),
                contents: result.c_source.clone(),
            },
            GeneratedFile {
                path: format!("{func_name}_hip.h"),
                contents: result.h_header.clone(),
            },
            GeneratedFile {
                path: "chelis_runtime.h".to_string(),
                contents: RUNTIME_H.to_string(),
            },
            GeneratedFile {
                path: "chelis_hip_runtime.h".to_string(),
                contents: HIP_RUNTIME_H.to_string(),
            },
        ],
        compile_flags: toolchain.compile_flags,
        link_flags: toolchain.link_flags,
        peak_device_bytes_estimate: None,
    }
}

fn execution_input_specs(dag: &Dag, labels: &[String]) -> Result<Vec<ExecutionTensorSpec>> {
    let mut load_types = HashMap::<String, TensorType>::new();
    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            load_types
                .entry(name.as_str().to_string())
                .or_insert_with(|| node.output_type.clone());
        }
    }
    labels
        .iter()
        .map(|label| {
            let ty = load_types.get(label).ok_or_else(|| {
                stage_error(
                    "compile",
                    format!("generated code referenced input `{label}`, but the lowered IR has no matching load"),
                    "compile_error",
                )
            })?;
            Ok(execution_tensor_spec(label.clone(), ty))
        })
        .collect()
}

fn execution_output_specs(dag: &Dag, labels: &[String]) -> Result<Vec<ExecutionTensorSpec>> {
    let nodes = execution_output_nodes(dag);
    labels
        .iter()
        .zip(nodes)
        .map(|(label, node_id)| {
            let ty = &dag
                .get(node_id)
                .ok_or_else(|| {
                    stage_error(
                        "compile",
                        format!(
                            "generated code referenced output node {}, but the lowered IR has no matching node",
                            node_id.0
                        ),
                        "compile_error",
                    )
                })?
                .output_type;
            Ok(execution_tensor_spec(label.clone(), ty))
        })
        .collect()
}

fn execution_output_nodes(dag: &Dag) -> Vec<NodeId> {
    let mut seen = std::collections::HashSet::new();
    let mut nodes = Vec::new();

    for node in dag.nodes() {
        if let RiscOp::Store { name: _ } = &node.op
            && seen.insert(node.id)
        {
            nodes.push(node.id);
        }
    }

    let roots: Vec<NodeId> = if dag.roots().is_empty() {
        dag.nodes()
            .last()
            .map(|node| vec![node.id])
            .unwrap_or_default()
    } else {
        dag.roots().to_vec()
    };

    for root_id in roots {
        if seen.insert(root_id) {
            nodes.push(root_id);
        }
    }

    nodes
}

fn execution_tensor_spec(name: String, ty: &TensorType) -> ExecutionTensorSpec {
    ExecutionTensorSpec {
        name,
        dtype: ty.precision.name().to_string(),
        dims: ty
            .dims
            .iter()
            .map(|dim| match dim {
                DimInfo::Lit(size) => ExecutionDim {
                    name: None,
                    size: Some(*size),
                },
                DimInfo::Named(name, Some(size)) => ExecutionDim {
                    name: Some(name.clone()),
                    size: Some(*size),
                },
                DimInfo::Named(name, None) => ExecutionDim {
                    name: Some(name.clone()),
                    size: None,
                },
            })
            .collect(),
    }
}

fn reject_unsized_named_dims(dag: &Dag, target: &str) -> Result<()> {
    for node in dag.nodes() {
        for dim in &node.output_type.dims {
            if let DimInfo::Named(name, None) = dim {
                return Err(stage_error(
                    "compile",
                    format!(
                        "`chelis build --target {target}` does not yet support unresolved named dimensions; node {} uses symbolic dimension `{name}`",
                        node.id.0
                    ),
                    "unsupported_feature",
                ));
            }
        }
    }
    Ok(())
}

fn reject_unsupported_hip_ops(dag: &Dag) -> Result<()> {
    let sparse_index_nodes: HashSet<NodeId> = dag
        .nodes()
        .iter()
        .filter_map(|node| match node.op {
            RiscOp::Gather { .. } | RiscOp::ScatterAdd { .. } | RiscOp::Scatter { .. } => {
                node.inputs.get(1).copied()
            }
            _ => None,
        })
        .collect();

    for node in dag.nodes() {
        match &node.op {
            RiscOp::Pad { .. } => {
                return Err(stage_error(
                    "compile",
                    format!(
                        "`chelis build --target hip` does not yet support `pad`; lowered node {} requires it",
                        node.id.0
                    ),
                    "unsupported_feature",
                ));
            }
            RiscOp::Shrink { .. } => {
                return Err(stage_error(
                    "compile",
                    format!(
                        "`chelis build --target hip` does not yet support `shrink`; lowered node {} requires it",
                        node.id.0
                    ),
                    "unsupported_feature",
                ));
            }
            RiscOp::OneHot { .. } => {
                return Err(stage_error(
                    "compile",
                    format!(
                        "`chelis build --target hip` cannot compile internal one_hot node {}: \
                         the sparse gather recognizer must consume OneHot before backend emission",
                        node.id.0
                    ),
                    "unsupported_feature",
                ));
            }
            RiscOp::Gather { .. } => {
                let values = &dag.get(node.inputs[0]).unwrap().output_type;
                let index_node = dag.get(node.inputs[1]).unwrap();
                let indices = &index_node.output_type;
                if values.precision != chelis_types::types::Prim::F32
                    || node.output_type.precision != chelis_types::types::Prim::F32
                {
                    return Err(stage_error(
                        "compile",
                        format!(
                            "`chelis build --target hip` sparse gather supports f32 payloads only; \
                             node {} carries payload precision `{}` and output precision `{}`",
                            node.id.0,
                            values.precision.name(),
                            node.output_type.precision.name()
                        ),
                        "unsupported_feature",
                    ));
                }
                if !matches!(
                    indices.precision,
                    chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                ) {
                    return Err(stage_error(
                        "compile",
                        format!(
                            "`chelis build --target hip` sparse gather requires int32/int64 indices; \
                             node {} uses `{}`",
                            node.id.0,
                            indices.precision.name()
                        ),
                        "unsupported_feature",
                    ));
                }
                if !matches!(index_node.op, RiscOp::Load { .. }) {
                    return Err(stage_error(
                        "compile",
                        format!(
                            "`chelis build --target hip` sparse gather requires indices to be loaded input tensors in this milestone; \
                             node {} uses indices produced by {:?}. \
                             Non-load integer index producers need integer HIP codegen before they can feed sparse kernels safely.",
                            node.id.0, index_node.op
                        ),
                        "unsupported_feature",
                    ));
                }
            }
            RiscOp::ScatterAdd { .. } | RiscOp::Scatter { .. } => {
                let (label, payload_blocker) = match &node.op {
                    RiscOp::ScatterAdd { .. } => (
                        "scatter_add",
                        // Preserved verbatim from the pre-W2-A
                        // message so existing rejection tests (and
                        // downstream consumers grepping for the
                        // rationale phrase) keep working.
                        "f64 scatter_add needs backend-specific atomic support and is not in this milestone.",
                    ),
                    RiscOp::Scatter { .. } => (
                        "scatter_replace",
                        // For replace-scatter the limitation is not
                        // atomics — it's the single-thread serial
                        // kernel that the determinism rule requires.
                        // f64 support is a future widening of the
                        // serialized kernel, not an atomics
                        // question.
                        "f64 scatter_replace requires a widened serial last-write-wins kernel and is not in this milestone.",
                    ),
                    _ => unreachable!(),
                };
                let target = &dag.get(node.inputs[0]).unwrap().output_type;
                let index_node = dag.get(node.inputs[1]).unwrap();
                let indices = &index_node.output_type;
                let updates = &dag.get(node.inputs[2]).unwrap().output_type;
                if target.precision != chelis_types::types::Prim::F32
                    || updates.precision != chelis_types::types::Prim::F32
                    || node.output_type.precision != chelis_types::types::Prim::F32
                {
                    return Err(stage_error(
                        "compile",
                        format!(
                            "`chelis build --target hip` sparse {label} supports f32 payloads only; \
                             node {} carries target `{}`, updates `{}`, output `{}`. \
                             {payload_blocker}",
                            node.id.0,
                            target.precision.name(),
                            updates.precision.name(),
                            node.output_type.precision.name()
                        ),
                        "unsupported_feature",
                    ));
                }
                if !matches!(
                    indices.precision,
                    chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                ) {
                    return Err(stage_error(
                        "compile",
                        format!(
                            "`chelis build --target hip` sparse {label} requires int32/int64 indices; \
                             node {} uses `{}`",
                            node.id.0,
                            indices.precision.name()
                        ),
                        "unsupported_feature",
                    ));
                }
                if !matches!(index_node.op, RiscOp::Load { .. }) {
                    return Err(stage_error(
                        "compile",
                        format!(
                            "`chelis build --target hip` sparse {label} requires indices to be loaded input tensors in this milestone; \
                             node {} uses indices produced by {:?}. \
                             Non-load integer index producers need integer HIP codegen before they can feed sparse kernels safely.",
                            node.id.0, index_node.op
                        ),
                        "unsupported_feature",
                    ));
                }
            }
            _ => {}
        }
    }
    for node in dag.nodes() {
        match node.output_type.precision {
            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool => {}
            chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                if sparse_index_nodes.contains(&node.id)
                    && matches!(node.op, RiscOp::Load { .. }) => {}
            other => {
                return Err(stage_error(
                    "compile",
                    format!(
                        "`chelis build --target hip` DAG path only supports f32/bool tensors, \
                         plus loaded int32/int64 tensors when they are consumed as sparse indices; \
                         node {} carries precision `{}`. \
                         Rewrite the program to use f32 tensors or build it with `--target c` instead.",
                        node.id.0,
                        other.name()
                    ),
                    "unsupported_feature",
                ));
            }
        }
    }
    Ok(())
}

fn unknown_name_error(stage: &str, field: &str, name: &str) -> CompilerError {
    stage_error(
        stage,
        format!("unknown name `{name}` in `{field}`"),
        "unknown_name",
    )
}

pub(crate) fn stage_error(stage: &str, message: impl Into<String>, kind: &str) -> CompilerError {
    stage_error_with_span(stage, message, kind, None)
}

fn deep_span_to_schema(span: Option<chelis_deep::Span>) -> Option<Span> {
    span.map(|span| Span {
        offset: span.offset,
        len: span.len,
    })
}

fn stage_error_with_span(
    stage: &str,
    message: impl Into<String>,
    kind: &str,
    diagnostic_span: Option<Span>,
) -> CompilerError {
    CompilerError {
        stage: stage.to_string(),
        errors: vec![Diagnostic {
            kind: kind.to_string(),
            message: message.into(),
            severity: 1.0,
            expected: None,
            got: None,
            suggestions: Vec::new(),
            span: diagnostic_span,
        }],
    }
}

fn parse_error_span_surf(source: &str, err: &chelis_surf::parser::ParseError) -> Option<Span> {
    let offset = match err {
        chelis_surf::parser::ParseError::Lex(_) => return None,
        chelis_surf::parser::ParseError::UnexpectedEof => source.len(),
        chelis_surf::parser::ParseError::Expected { offset, .. }
        | chelis_surf::parser::ParseError::NonAssocChain { offset } => *offset,
    };
    Some(Span { offset, len: 0 })
}

fn parse_error_span_deep(err: &chelis_deep::parser::ParseError) -> Option<Span> {
    let offset = match err {
        chelis_deep::parser::ParseError::Lex(_) => return None,
        chelis_deep::parser::ParseError::UnexpectedEof { offset }
        | chelis_deep::parser::ParseError::Expected { offset, .. }
        | chelis_deep::parser::ParseError::EmptyList { offset } => *offset,
        chelis_deep::parser::ParseError::ForbiddenSpanChar { value_offset, .. } => *value_offset,
    };
    Some(Span { offset, len: 0 })
}

pub(crate) fn check_error_diagnostic(error: &CheckError) -> Diagnostic {
    Diagnostic {
        kind: format!("{:?}", error.kind),
        message: error.message.clone(),
        severity: error.severity,
        expected: error.expected.clone(),
        got: error.got.clone(),
        suggestions: error.suggestions.clone(),
        span: None,
    }
}

fn span(span: chelis_deep::Span) -> Span {
    Span {
        offset: span.offset,
        len: span.len,
    }
}

fn wire_decl(decl: &Decl) -> WireSurfDecl {
    match decl {
        Decl::Module {
            name,
            decls,
            span: s,
        } => WireSurfDecl::Module {
            name: name.clone(),
            decls: decls.iter().map(wire_decl).collect(),
            span: span(*s),
        },
        Decl::Import {
            module,
            kind,
            span: s,
        } => WireSurfDecl::Import {
            module: module.clone(),
            import_kind: wire_import_kind(kind),
            span: span(*s),
        },
        Decl::Sig {
            name, ty, span: s, ..
        } => WireSurfDecl::Sig {
            name: name.clone(),
            ty: wire_type_expr(ty),
            span: span(*s),
        },
        Decl::Dim { names, span: s } => WireSurfDecl::Dim {
            names: names.clone(),
            span: span(*s),
        },
        Decl::TypeDef {
            name,
            params,
            variants,
            span: s,
        } => WireSurfDecl::TypeDef {
            name: name.clone(),
            params: params.clone(),
            variants: variants.iter().map(wire_variant).collect(),
            span: span(*s),
        },
        Decl::TypeAlias {
            name,
            params,
            ty,
            span: s,
        } => WireSurfDecl::TypeAlias {
            name: name.clone(),
            params: params.clone(),
            ty: wire_type_expr(ty),
            span: span(*s),
        },
        Decl::MacroDef {
            name,
            params,
            body,
            span: s,
        } => WireSurfDecl::MacroDef {
            name: name.clone(),
            params: params.clone(),
            body: wire_expr(body),
            span: span(*s),
        },
        Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            body,
            span: s,
            ..
        } => WireSurfDecl::FunDef {
            name: name.clone(),
            dim_params: dim_params.clone(),
            params: params.iter().map(wire_param).collect(),
            ret_ty: ret_ty.as_ref().map(wire_type_expr),
            body: wire_expr(body),
            span: span(*s),
        },
        Decl::Property {
            name,
            params,
            preconditions,
            body,
            options,
            span: s,
        } => WireSurfDecl::Property {
            name: name.clone(),
            params: params.iter().map(wire_param).collect(),
            preconditions: preconditions.iter().map(wire_expr).collect(),
            body: wire_expr(body),
            options: options.iter().map(wire_property_option).collect(),
            span: span(*s),
        },
        Decl::LetDef {
            name,
            ty,
            value,
            span: s,
        } => WireSurfDecl::LetDef {
            name: name.clone(),
            ty: ty.as_ref().map(wire_type_expr),
            value: wire_expr(value),
            span: span(*s),
        },
        Decl::Export { names, span: s } => WireSurfDecl::Export {
            names: names.clone(),
            span: span(*s),
        },
    }
}

fn wire_property_option(option: &chelis_surf::ast::PropertyOption) -> WirePropertyOption {
    match option {
        chelis_surf::ast::PropertyOption::Tolerance(value, s) => WirePropertyOption::Tolerance {
            value: wire_expr(value),
            span: span(*s),
        },
        chelis_surf::ast::PropertyOption::Seed(value, s) => WirePropertyOption::Seed {
            value: wire_expr(value),
            span: span(*s),
        },
        chelis_surf::ast::PropertyOption::Samples(value, s) => WirePropertyOption::Samples {
            value: wire_expr(value),
            span: span(*s),
        },
    }
}

fn wire_import_kind(kind: &ImportKind) -> WireImportKind {
    match kind {
        ImportKind::Qualified => WireImportKind::Qualified,
        ImportKind::All => WireImportKind::All,
        ImportKind::Names(names) => WireImportKind::Names {
            names: names.clone(),
        },
    }
}

fn wire_variant(variant: &Variant) -> WireVariant {
    WireVariant {
        name: variant.name.clone(),
        fields: match &variant.fields {
            VariantFields::Positional(fields) => WireVariantFields::Positional {
                fields: fields.iter().map(wire_type_expr).collect(),
            },
            VariantFields::Record(fields) => WireVariantFields::Record {
                fields: fields
                    .iter()
                    .map(|(name, ty)| WireRecordTypeField {
                        name: name.clone(),
                        ty: wire_type_expr(ty),
                    })
                    .collect(),
            },
        },
        span: span(variant.span),
    }
}

fn wire_param(param: &Param) -> WireParam {
    WireParam {
        name: param.name.clone(),
        ty: param.ty.as_ref().map(wire_type_expr),
        span: span(param.span),
    }
}

fn wire_expr(expr: &Expr) -> WireSurfExpr {
    match expr {
        Expr::Lit(lit, s) => WireSurfExpr::Lit {
            literal: wire_literal(lit),
            span: span(*s),
        },
        Expr::Var(name, s) => WireSurfExpr::Var {
            name: name.clone(),
            span: span(*s),
        },
        Expr::Constructor(name, s) => WireSurfExpr::Constructor {
            name: name.clone(),
            span: span(*s),
        },
        Expr::Apply(func, args, s) => WireSurfExpr::Apply {
            func: Box::new(wire_expr(func)),
            args: args.iter().map(wire_expr).collect(),
            span: span(*s),
        },
        Expr::List(items, s) => WireSurfExpr::List {
            items: items.iter().map(wire_expr).collect(),
            span: span(*s),
        },
        Expr::Record(name, fields, s) => WireSurfExpr::Record {
            name: name.clone(),
            fields: fields
                .iter()
                .map(|(field, value)| WireRecordExprField {
                    name: field.clone(),
                    value: wire_expr(value),
                })
                .collect(),
            span: span(*s),
        },
        Expr::Access(inner, field, s) => WireSurfExpr::Access {
            expr: Box::new(wire_expr(inner)),
            field: field.clone(),
            span: span(*s),
        },
        Expr::TupleGet(inner, index, s) => WireSurfExpr::TupleGet {
            expr: Box::new(wire_expr(inner)),
            index: *index,
            span: span(*s),
        },
        Expr::Binary(op, lhs, rhs, s) => WireSurfExpr::Binary {
            op: wire_bin_op(*op),
            lhs: Box::new(wire_expr(lhs)),
            rhs: Box::new(wire_expr(rhs)),
            span: span(*s),
        },
        Expr::Unary(op, inner, s) => WireSurfExpr::Unary {
            op: wire_unary_op(*op),
            expr: Box::new(wire_expr(inner)),
            span: span(*s),
        },
        Expr::Pipe(inner, stages, s) => WireSurfExpr::Pipe {
            expr: Box::new(wire_expr(inner)),
            stages: stages.iter().map(wire_expr).collect(),
            span: span(*s),
        },
        Expr::If(cond, then_branch, else_branch, s) => WireSurfExpr::If {
            cond: Box::new(wire_expr(cond)),
            then_branch: Box::new(wire_expr(then_branch)),
            else_branch: Box::new(wire_expr(else_branch)),
            span: span(*s),
        },
        Expr::Match(inner, arms, s) => WireSurfExpr::Match {
            expr: Box::new(wire_expr(inner)),
            arms: arms.iter().map(wire_match_arm).collect(),
            span: span(*s),
        },
        Expr::Lambda(params, body, s) => WireSurfExpr::Lambda {
            params: params.iter().map(wire_param).collect(),
            body: Box::new(wire_expr(body)),
            span: span(*s),
        },
        Expr::Tuple(items, s) => WireSurfExpr::Tuple {
            items: items.iter().map(wire_expr).collect(),
            span: span(*s),
        },
        Expr::Cast(inner, ty, s) => WireSurfExpr::Cast {
            expr: Box::new(wire_expr(inner)),
            ty: ty.clone(),
            span: span(*s),
        },
        Expr::Grad(inner, wrt, s) => WireSurfExpr::Grad {
            expr: Box::new(wire_expr(inner)),
            wrt: wrt.clone(),
            span: span(*s),
        },
        Expr::Vmap(inner, axis, s) => WireSurfExpr::Vmap {
            expr: Box::new(wire_expr(inner)),
            axis: *axis,
            span: span(*s),
        },
        Expr::Jit(inner, s) => WireSurfExpr::Jit {
            expr: Box::new(wire_expr(inner)),
            span: span(*s),
        },
        Expr::Realize(inner, s) => WireSurfExpr::Realize {
            expr: Box::new(wire_expr(inner)),
            span: span(*s),
        },
        Expr::Copy(inner, s) => WireSurfExpr::Copy {
            expr: Box::new(wire_expr(inner)),
            span: span(*s),
        },
        Expr::Borrow(inner, s) => WireSurfExpr::Borrow {
            expr: Box::new(wire_expr(inner)),
            span: span(*s),
        },
        Expr::WithSeed(seed, body, s) => WireSurfExpr::WithSeed {
            seed: Box::new(wire_expr(seed)),
            body: Box::new(wire_expr(body)),
            span: span(*s),
        },
        Expr::WithDevice(device, body, s) => WireSurfExpr::WithDevice {
            device: Box::new(wire_expr(device)),
            body: Box::new(wire_expr(body)),
            span: span(*s),
        },
        Expr::Par(exprs, s) => WireSurfExpr::Par {
            exprs: exprs.iter().map(wire_expr).collect(),
            span: span(*s),
        },
        Expr::Annotate(inner, ty, s) => WireSurfExpr::Annotate {
            expr: Box::new(wire_expr(inner)),
            ty: wire_type_expr(ty),
            span: span(*s),
        },
        Expr::Block(bindings, body, s) => WireSurfExpr::Block {
            bindings: bindings.iter().map(wire_let_binding).collect(),
            body: Box::new(wire_expr(body)),
            span: span(*s),
        },
    }
}

fn wire_literal(lit: &Literal) -> WireLiteral {
    match lit {
        Literal::Int(value) => WireLiteral::Int { value: *value },
        Literal::Float(value) => WireLiteral::Float { value: *value },
        // Typed-suffix literals (spec §5.5): preserve the suffix across
        // the wire boundary so the receiving side sees the same type.
        Literal::TypedInt(value, suffix) => WireLiteral::TypedInt {
            value: *value,
            suffix: suffix.as_str().to_string(),
        },
        Literal::TypedFloat(value, suffix) => WireLiteral::TypedFloat {
            value: *value,
            suffix: suffix.as_str().to_string(),
        },
        Literal::Str(value) => WireLiteral::Str {
            value: value.clone(),
        },
        Literal::Bool(value) => WireLiteral::Bool { value: *value },
    }
}

fn wire_bin_op(op: BinOp) -> WireBinOp {
    match op {
        BinOp::Add => WireBinOp::Add,
        BinOp::Sub => WireBinOp::Sub,
        BinOp::Mul => WireBinOp::Mul,
        BinOp::Div => WireBinOp::Div,
        BinOp::Mod => WireBinOp::Mod,
        BinOp::Eq => WireBinOp::Eq,
        BinOp::Ne => WireBinOp::Ne,
        BinOp::Lt => WireBinOp::Lt,
        BinOp::Gt => WireBinOp::Gt,
        BinOp::Le => WireBinOp::Le,
        BinOp::Ge => WireBinOp::Ge,
        BinOp::And => WireBinOp::And,
        BinOp::Or => WireBinOp::Or,
    }
}

fn wire_unary_op(op: UnaryOp) -> WireUnaryOp {
    match op {
        UnaryOp::Neg => WireUnaryOp::Neg,
        UnaryOp::Not => WireUnaryOp::Not,
    }
}

fn wire_match_arm(arm: &MatchArm) -> WireMatchArm {
    WireMatchArm {
        pattern: wire_pattern(&arm.pattern),
        guard: arm.guard.as_ref().map(wire_expr),
        body: wire_expr(&arm.body),
        span: span(arm.span),
    }
}

fn wire_pattern(pattern: &Pattern) -> WirePattern {
    match pattern {
        Pattern::Wildcard(s) => WirePattern::Wildcard { span: span(*s) },
        Pattern::Var(name, s) => WirePattern::Var {
            name: name.clone(),
            span: span(*s),
        },
        Pattern::Lit(lit, s) => WirePattern::Lit {
            literal: wire_literal(lit),
            span: span(*s),
        },
        Pattern::Constructor(name, args, s) => WirePattern::Constructor {
            name: name.clone(),
            args: args.iter().map(wire_pattern).collect(),
            span: span(*s),
        },
        Pattern::Tuple(items, s) => WirePattern::Tuple {
            items: items.iter().map(wire_pattern).collect(),
            span: span(*s),
        },
        Pattern::Record(name, fields, s) => WirePattern::Record {
            name: name.clone(),
            fields: fields
                .iter()
                .map(|(name, pattern)| WireRecordPatternField {
                    name: name.clone(),
                    pattern: wire_pattern(pattern),
                })
                .collect(),
            span: span(*s),
        },
        Pattern::As(name, pattern, s) => WirePattern::As {
            name: name.clone(),
            pattern: Box::new(wire_pattern(pattern)),
            span: span(*s),
        },
    }
}

fn wire_let_binding(binding: &LetBinding) -> WireLetBinding {
    WireLetBinding {
        pattern: wire_let_pattern(&binding.pattern),
        ty: binding.ty.as_ref().map(wire_type_expr),
        value: wire_expr(&binding.value),
    }
}

fn wire_let_pattern(pattern: &LetPattern) -> WireLetPattern {
    match pattern {
        LetPattern::Var(name, s) => WireLetPattern::Var {
            name: name.clone(),
            span: span(*s),
        },
        LetPattern::Wildcard(s) => WireLetPattern::Wildcard { span: span(*s) },
        LetPattern::Tuple(items, s) => WireLetPattern::Tuple {
            items: items.iter().map(wire_let_pattern).collect(),
            span: span(*s),
        },
    }
}

fn wire_type_expr(ty: &TypeExpr) -> WireSurfTypeExpr {
    match ty {
        TypeExpr::Named(name, s) => WireSurfTypeExpr::Named {
            name: name.clone(),
            span: span(*s),
        },
        TypeExpr::Tensor(dims, precision, s) => WireSurfTypeExpr::Tensor {
            dims: dims.iter().map(wire_type_expr).collect(),
            precision: precision.clone(),
            span: span(*s),
        },
        TypeExpr::Arrow(args, ret, s) => WireSurfTypeExpr::Arrow {
            args: args.iter().map(wire_type_expr).collect(),
            ret: Box::new(wire_type_expr(ret)),
            span: span(*s),
        },
        TypeExpr::Ref(inner, s) => WireSurfTypeExpr::Ref {
            inner: Box::new(wire_type_expr(inner)),
            span: span(*s),
        },
        TypeExpr::App(name, args, s) => WireSurfTypeExpr::App {
            name: name.clone(),
            args: args.iter().map(wire_type_expr).collect(),
            span: span(*s),
        },
        TypeExpr::Tuple(items, s) => WireSurfTypeExpr::Tuple {
            items: items.iter().map(wire_type_expr).collect(),
            span: span(*s),
        },
        TypeExpr::Infer(s) => WireSurfTypeExpr::Infer { span: span(*s) },
    }
}

fn wire_deep_expr(expr: &DeepExpr) -> WireDeepExpr {
    match expr {
        DeepExpr::Atom(atom, s) => WireDeepExpr {
            kind: WireDeepExprKind::Atom {
                atom: match atom {
                    chelis_deep::Atom::Symbol(value) => WireDeepAtom::Symbol {
                        value: value.clone(),
                    },
                    chelis_deep::Atom::Int(value) => WireDeepAtom::Int { value: *value },
                    chelis_deep::Atom::Float(value) => WireDeepAtom::Float { value: *value },
                    chelis_deep::Atom::Str(value) => WireDeepAtom::Str {
                        value: value.clone(),
                    },
                    chelis_deep::Atom::Keyword(value) => WireDeepAtom::Keyword {
                        value: value.clone(),
                    },
                    chelis_deep::Atom::Bool(value) => WireDeepAtom::Bool { value: *value },
                },
            },
            span: Some(span(*s)),
        },
        DeepExpr::List(list, s) => WireDeepExpr {
            kind: WireDeepExprKind::List {
                elements: list.elements.iter().map(wire_deep_expr).collect(),
            },
            span: Some(span(*s)),
        },
        DeepExpr::Map(map, s) => WireDeepExpr {
            kind: WireDeepExprKind::Map {
                entries: map
                    .entries
                    .iter()
                    .map(|(key, value)| WireMetaEntry {
                        key: key.clone(),
                        value: wire_deep_expr(value),
                    })
                    .collect(),
            },
            span: Some(span(*s)),
        },
        DeepExpr::MetaExpr(meta, s) => WireDeepExpr {
            kind: WireDeepExprKind::MetaExpr {
                entries: meta
                    .entries
                    .iter()
                    .map(|(key, value)| WireMetaEntry {
                        key: key.clone(),
                        value: wire_deep_expr(value),
                    })
                    .collect(),
                expr: Box::new(wire_deep_expr(&meta.expr)),
            },
            span: Some(span(*s)),
        },
    }
}

fn wire_dag(dag: &Dag) -> WireDag {
    WireDag {
        nodes: dag.nodes().iter().map(wire_dag_node).collect(),
        roots: dag.roots().iter().map(|id| id.0).collect(),
    }
}

fn wire_dag_node(node: &chelis_ir::dag::DagNode) -> WireDagNode {
    WireDagNode {
        id: node.id.0,
        op: wire_op(&node.op),
        inputs: node.inputs.iter().map(|id| id.0).collect(),
        output_type: wire_tensor_type(&node.output_type),
    }
}

fn wire_tensor_type(ty: &TensorType) -> WireTensorType {
    WireTensorType {
        dims: ty.dims.iter().map(wire_dim).collect(),
        precision: ty.precision.name().to_string(),
    }
}

fn wire_dim(dim: &DimInfo) -> WireDimInfo {
    match dim {
        DimInfo::Named(name, size) => WireDimInfo::Named {
            name: name.clone(),
            size: *size,
        },
        DimInfo::Lit(size) => WireDimInfo::Lit { size: *size },
    }
}

fn wire_dim_expr(expr: &chelis_ir::dag::DimExpr) -> WireDimExpr {
    use chelis_ir::dag::DimExpr;
    match expr {
        DimExpr::Concrete(value) => WireDimExpr::Concrete { value: *value },
        DimExpr::Sym(name) => WireDimExpr::Sym { name: name.clone() },
        DimExpr::Mul(lhs, rhs) => WireDimExpr::Mul {
            lhs: Box::new(wire_dim_expr(lhs)),
            rhs: Box::new(wire_dim_expr(rhs)),
        },
        DimExpr::Div(lhs, rhs) => WireDimExpr::Div {
            lhs: Box::new(wire_dim_expr(lhs)),
            rhs: Box::new(wire_dim_expr(rhs)),
        },
    }
}

fn wire_op(op: &RiscOp) -> WireRiscOp {
    match op {
        RiscOp::Add => WireRiscOp::Add,
        RiscOp::Mul => WireRiscOp::Mul,
        RiscOp::CmpLt => WireRiscOp::CmpLt,
        RiscOp::MaxElem => WireRiscOp::MaxElem,
        RiscOp::Neg => WireRiscOp::Neg,
        RiscOp::Exp => WireRiscOp::Exp,
        RiscOp::Log => WireRiscOp::Log,
        RiscOp::Sin => WireRiscOp::Sin,
        RiscOp::Sqrt => WireRiscOp::Sqrt,
        RiscOp::Cos => WireRiscOp::Cos,
        RiscOp::Tan => WireRiscOp::Tan,
        RiscOp::Atan => WireRiscOp::Atan,
        RiscOp::Abs => WireRiscOp::Abs,
        RiscOp::Floor => WireRiscOp::Floor,
        RiscOp::Ceil => WireRiscOp::Ceil,
        RiscOp::UniformLike { low, high, seed } => WireRiscOp::UniformLike {
            low: *low,
            high: *high,
            seed: *seed,
        },
        RiscOp::Dropout { rate, seed } => WireRiscOp::Dropout {
            rate: *rate,
            seed: *seed,
        },
        RiscOp::Sum { axis, accumulator } => WireRiscOp::Sum {
            axis: *axis,
            accumulator: accumulator.name().to_string(),
        },
        RiscOp::MaxReduce { axis } => WireRiscOp::MaxReduce { axis: *axis },
        RiscOp::MinReduce { axis } => WireRiscOp::MinReduce { axis: *axis },
        RiscOp::ProdReduce { axis } => WireRiscOp::ProdReduce { axis: *axis },
        RiscOp::Argmax { axis } => WireRiscOp::Argmax { axis: *axis },
        RiscOp::Argmin { axis } => WireRiscOp::Argmin { axis: *axis },
        RiscOp::Reshape { new_shape } => WireRiscOp::Reshape {
            new_shape: new_shape.iter().map(wire_dim).collect(),
        },
        RiscOp::Permute { axes } => WireRiscOp::Permute { axes: axes.clone() },
        RiscOp::Expand { axis, size } => WireRiscOp::Expand {
            axis: *axis,
            size: size.to_string(),
        },
        RiscOp::OneHot { vocab } => WireRiscOp::OneHot { vocab: *vocab },
        RiscOp::Pad { padding, fill } => WireRiscOp::Pad {
            padding: padding.clone(),
            fill: *fill,
        },
        RiscOp::Shrink { bounds } => WireRiscOp::Shrink {
            bounds: bounds.clone(),
        },
        RiscOp::Stride { strides } => WireRiscOp::Stride {
            strides: strides.clone(),
        },
        RiscOp::Const { value } => WireRiscOp::Const { value: *value },
        RiscOp::Load { name } => WireRiscOp::Load {
            name: name.as_str().to_string(),
        },
        RiscOp::Store { name } => WireRiscOp::Store {
            name: name.as_str().to_string(),
        },
        RiscOp::Copy => WireRiscOp::Copy,
        RiscOp::Drop => WireRiscOp::Drop,
        RiscOp::Realize => WireRiscOp::Realize,
        RiscOp::Cast { new_precision } => WireRiscOp::Cast {
            new_precision: new_precision.name().to_string(),
        },
        RiscOp::FusedElem { ops } => WireRiscOp::FusedElem {
            ops: ops
                .iter()
                .map(|step| WireFusedStep {
                    op: match step.op {
                        FusedStepOp::Add => WireFusedStepOp::Add,
                        FusedStepOp::Mul => WireFusedStepOp::Mul,
                        FusedStepOp::MaxElem => WireFusedStepOp::MaxElem,
                        FusedStepOp::CmpLt => WireFusedStepOp::CmpLt,
                        FusedStepOp::Neg => WireFusedStepOp::Neg,
                        FusedStepOp::Exp => WireFusedStepOp::Exp,
                        FusedStepOp::Log => WireFusedStepOp::Log,
                        FusedStepOp::Sin => WireFusedStepOp::Sin,
                        FusedStepOp::Sqrt => WireFusedStepOp::Sqrt,
                        FusedStepOp::Cos => WireFusedStepOp::Cos,
                        FusedStepOp::Tan => WireFusedStepOp::Tan,
                        FusedStepOp::Atan => WireFusedStepOp::Atan,
                        FusedStepOp::Abs => WireFusedStepOp::Abs,
                        FusedStepOp::Floor => WireFusedStepOp::Floor,
                        FusedStepOp::Ceil => WireFusedStepOp::Ceil,
                    },
                    input_indices: step
                        .input_indices
                        .iter()
                        .map(|input| match input {
                            FusedInput::External(index) => {
                                WireFusedInput::External { index: *index }
                            }
                            FusedInput::PreviousStep(index) => {
                                WireFusedInput::PreviousStep { index: *index }
                            }
                        })
                        .collect(),
                })
                .collect(),
        },
        RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            accumulator,
        } => WireRiscOp::BlasMatmul {
            batch_dims: batch_dims.iter().map(wire_dim_expr).collect(),
            m: wire_dim_expr(m),
            n: wire_dim_expr(n),
            k: wire_dim_expr(k),
            accumulator: accumulator.name().to_string(),
        },
        RiscOp::Gather { axis } => WireRiscOp::Gather { axis: *axis },
        RiscOp::ScatterAdd { axis } => WireRiscOp::ScatterAdd { axis: *axis },
        RiscOp::Scatter { axis } => WireRiscOp::Scatter { axis: *axis },
    }
}

#[cfg(test)]
#[allow(deprecated)] // exercises eval_many for behavior parity; deprecation is for external callers
mod tests {
    use super::*;
    use crate::schema::ExecutionValue;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn copy_drop_context_fixture() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().join("myapp");
        fs::create_dir_all(root.join("src")).expect("mkdir src");
        fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib src");
        fs::write(
            root.join("reef.toml"),
            format!(
                "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n",
                crate::COMPILER_VERSION
            ),
        )
        .expect("write app reef.toml");
        fs::write(
            root.join("reef.lock"),
            format!(
                "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
                crate::COMPILER_VERSION
            ),
        )
        .expect("write app reef.lock");
        fs::write(
            root.join("src/main.ch"),
            "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n",
        )
        .expect("write app main");
        fs::write(
            root.join("mylib/reef.toml"),
            format!(
                "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Mylib\"\n",
                crate::COMPILER_VERSION
            ),
        )
        .expect("write lib reef.toml");
        fs::write(
            root.join("mylib/src/copy.ch"),
            "module Mylib.Copy\nexport (consume)\n\n\
             def consume(x: tensor[2, f32]) -> tensor[2, f32] = realize(x)\n",
        )
        .expect("write lib copy module");
        (dir, root)
    }

    fn tensor_type(dims: Vec<usize>, precision: chelis_types::types::Prim) -> TensorType {
        TensorType {
            dims: dims.into_iter().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    #[test]
    fn hip_sparse_gather_is_supported_with_integer_indices() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_type(vec![3, 2], chelis_types::types::Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::Int64),
            None,
        );
        let gather = dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, indices],
            tensor_type(vec![4, 2], chelis_types::types::Prim::F32),
            None,
        );
        dag.add_root(gather);

        reject_unsupported_hip_ops(&dag).expect("HIP should allow sparse gather");
    }

    #[test]
    fn hip_sparse_gather_rejects_non_load_integer_index_producer() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_type(vec![3, 2], chelis_types::types::Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::Int64),
            None,
        );
        let gather = dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, indices],
            tensor_type(vec![4, 2], chelis_types::types::Prim::F32),
            None,
        );
        dag.add_root(gather);

        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP should reject non-load integer index producers");
        let message = &err.errors[0].message;
        assert!(message.contains("indices to be loaded input tensors"));
        assert!(message.contains("Non-load integer index producers need integer HIP codegen"));
    }

    #[test]
    fn hip_sparse_scatter_add_rejects_non_load_integer_index_producer() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_type(vec![3, 2], chelis_types::types::Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::Int32),
            None,
        );
        let updates = dag.add_node(
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            tensor_type(vec![4, 2], chelis_types::types::Prim::F32),
            None,
        );
        let scatter = dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, indices, updates],
            tensor_type(vec![3, 2], chelis_types::types::Prim::F32),
            None,
        );
        dag.add_root(scatter);

        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP should reject non-load scatter_add index producers");
        let message = &err.errors[0].message;
        assert!(message.contains("indices to be loaded input tensors"));
        assert!(message.contains("Non-load integer index producers need integer HIP codegen"));
    }

    #[test]
    fn hip_sparse_scatter_f64_rejection_names_atomic_blocker() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_type(vec![3, 2], chelis_types::types::Prim::F64),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::Int64),
            None,
        );
        let updates = dag.add_node(
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            tensor_type(vec![4, 2], chelis_types::types::Prim::F64),
            None,
        );
        let scatter = dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, indices, updates],
            tensor_type(vec![3, 2], chelis_types::types::Prim::F64),
            None,
        );
        dag.add_root(scatter);

        let err = reject_unsupported_hip_ops(&dag).expect_err("HIP should reject f64 scatter_add");
        let message = &err.errors[0].message;
        assert!(message.contains("sparse scatter_add supports f32 payloads only"));
        assert!(message.contains("f64 scatter_add needs backend-specific atomic support"));
    }

    #[test]
    fn compile_source_keeps_all_lowered_tensor_roots() {
        let source = r#"
def logits(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[2, 2, f32] =
  matmul(x, w)

def loss(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[f32] =
  matmul(x, w) |> sum(1) |> mean(0)
"#;

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");

        assert_eq!(
            compiled.all_root_names,
            vec!["logits".to_string(), "loss".to_string()]
        );
        assert_eq!(
            compiled.named_roots.keys().cloned().collect::<Vec<_>>(),
            vec!["logits".to_string(), "loss".to_string()]
        );
        assert_eq!(compiled.dag.roots().len(), 2);
    }

    #[test]
    fn compile_source_excludes_host_only_roots_from_lowered_root_map() {
        let source = r#"
label = "mnist"

def logits(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[2, 2, f32] =
  matmul(x, w)
"#;

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");

        assert_eq!(
            compiled.all_root_names,
            vec!["label".to_string(), "logits".to_string()]
        );
        assert_eq!(
            compiled.named_roots.keys().cloned().collect::<Vec<_>>(),
            vec!["logits".to_string()]
        );
        assert_eq!(compiled.dag.roots().len(), 1);
    }

    #[test]
    fn compile_new_source_in_context_matches_monolithic_copy_insertion() {
        let monolithic = compile_source(
            SourceKind::Surf,
            r#"
def consume(x: tensor[2, f32]) -> tensor[2, f32] = realize(x)

def out(x: tensor[2, f32]) -> tensor[2, f32] = add(consume(x), consume(x))
"#,
        )
        .expect("monolithic compile");
        let monolithic_root = monolithic.named_roots["out"];
        let monolithic_summary = chelis_ir::analysis::analyze_function_copy_cost(
            &monolithic.dag,
            "out",
            monolithic_root,
        );

        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/copy-drop"), &root)
            .expect("compile context");
        let compiled = compile_new_source_in_context(
            &context,
            "module App.Eval\nimport Mylib.Copy (consume)\n\n\
             def out(x: tensor[2, f32]) -> tensor[2, f32] = add(consume(x), consume(x))\n",
        )
        .expect("compile new source in context");
        let context_root = compiled.named_roots["out"];
        let context_summary =
            chelis_ir::analysis::analyze_function_copy_cost(&compiled.dag, "out", context_root);

        assert_eq!(monolithic_summary.copy_count, 1);
        assert_eq!(context_summary.copy_count, monolithic_summary.copy_count);
        assert!(compiled.dag.roots().iter().all(|root| {
            !matches!(
                compiled.dag.get(*root).map(|node| &node.op),
                Some(chelis_ir::dag::RiscOp::Drop)
            )
        }));
    }

    #[test]
    fn compile_source_preserves_scalar_string_foundation_root_names() {
        let source = include_str!("../../../examples/scalar_string_foundation.ch");

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");

        assert_eq!(
            compiled.named_roots.keys().cloned().collect::<Vec<_>>(),
            Vec::<String>::new()
        );
        assert!(compiled.all_root_names.iter().any(|name| name == "status"));
        assert!(compiled.all_root_names.iter().any(|name| name == "loss"));
        assert!(
            compiled
                .all_root_names
                .iter()
                .any(|name| name == "should_stop")
        );
    }

    #[test]
    fn compile_source_accepts_typed_param_named_let() {
        let source = r#"
def id(let: int64) -> int64 = let
"#;

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");
        let deep = chelis_deep::printer::print_canonical(compiled.checked.exprs());
        assert!(deep.contains("^{:type (t-prim {} int64)} let"));
    }

    #[test]
    fn compile_emits_host_program_for_scalar_string_foundation() {
        let source = include_str!("../../../examples/scalar_string_foundation.ch");

        let result = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            target: CompileTarget::C,
            entry_name: Some("scalar_string_foundation".to_string()),
        })
        .expect("compile");

        let c_file = result
            .files
            .iter()
            .find(|file| file.path == "scalar_string_foundation.c")
            .expect("c file");
        assert!(c_file.contents.contains("int main(void)"));
        assert!(c_file.contents.contains("chelis_string_eq"));
        assert!(c_file.contents.contains("chelis_string_concat"));
    }

    #[test]
    fn compile_emits_generic_adt_runtime_calls_for_recursive_host_program() {
        let source = r#"
type Json =
  | JsonNull
  | JsonInt(int64)
  | JsonString(string)
  | JsonArray(List[Json])

def describe(value: Json) -> string =
  match value with {
    | JsonNull => "null"
    | JsonInt(n) => to_string(n)
    | JsonString(s) => s
    | JsonArray(items) => to_string(len(items))
  }

sample = JsonArray([JsonString("hi"), JsonInt(cast(3, int64))])
result = describe(sample)
"#;

        let result = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            target: CompileTarget::C,
            entry_name: Some("jsonish".to_string()),
        })
        .expect("compile");

        let c_file = result
            .files
            .iter()
            .find(|file| file.path == "jsonish.c")
            .expect("c file");
        assert!(
            c_file.contents.contains("chelis_adt_construct"),
            "expected generic ADT runtime construction in generated C, got:\n{}",
            c_file.contents
        );
        assert!(
            c_file.contents.contains("chelis_adt_get_tag"),
            "expected generic ADT runtime tag checks in generated C, got:\n{}",
            c_file.contents
        );
        assert!(
            !c_file.contents.contains("unsupported builtin"),
            "generated C must not fall back to unsupported builtin stubs:\n{}",
            c_file.contents
        );
    }

    #[test]
    fn eval_supports_shape_queries_with_tensor_bindings() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
x: tensor[2, 3, f32] = x
dims = (rank(x), shape(x, 1), numel(x))
"#
            .to_string(),
            bindings: BTreeMap::from([(
                "x".to_string(),
                crate::schema::TensorValue {
                    shape: vec![2, 3],
                    data: vec![0.0; 6],
                },
            )]),
        })
        .expect("eval");

        assert_eq!(result.roots.len(), 4);
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("dims.0")
                    && matches!(root.value, ExecutionValue::Int64 { value: 2 }))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("dims.1")
                    && matches!(root.value, ExecutionValue::Int64 { value: 3 }))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("dims.2")
                    && matches!(root.value, ExecutionValue::Int64 { value: 6 }))
        );
    }

    #[test]
    fn eval_supports_string_parse_helpers_and_option_matching() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
parsed = match to_int(" 42 ") with {
  | Some n => n
  | None => cast(0, int64)
}

cleaned = string_trim("  ckpt-42.safetensors  ")
progress = print(cleaned)
matches_path = and(
  string_starts_with(cleaned, "ckpt-"),
  string_contains(cleaned, "42")
)
result = if matches_path then parsed else cast(0, int64)
"#
            .to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");

        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("result")
                    && matches!(root.value, ExecutionValue::Int64 { value: 42 }))
        );
        assert_eq!(result.transcript, vec!["ckpt-42.safetensors".to_string()]);
    }

    #[test]
    fn eval_supports_integer_mod_and_bitwise_helpers() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
bits = bitxor(bitand(cast(7, int64), cast(3, int64)), shl(cast(1, int64), cast(2, int64)))
rem = mod(cast(17, int64), cast(5, int64))
shifted = shr(cast(8, int64), cast(1, int64))
"#
            .to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");

        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("bits")
                    && matches!(root.value, ExecutionValue::Int64 { value: 7 }))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("rem")
                    && matches!(root.value, ExecutionValue::Int64 { value: 2 }))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("shifted")
                    && matches!(root.value, ExecutionValue::Int64 { value: 4 }))
        );
    }

    #[test]
    fn eval_supports_recursive_top_level_defs() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
def sum_to(n: int64) -> int64 =
  if lte(n, cast(0, int64)) then cast(0, int64) else add(n, sum_to(sub(n, cast(1, int64))))

value = sum_to(cast(3, int64))
"#
            .to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");

        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("value")
                    && matches!(root.value, ExecutionValue::Int64 { value: 6 }))
        );
    }

    #[test]
    fn eval_supports_module_wrapped_host_defs() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
module Demo.Main

parsed = match to_int("7") with {
  | Some(value) => value
  | None => cast(0, int64)
}
label = if gt(parsed, cast(0, int64)) then "ready" else "waiting"
view = print(label)
"#
            .to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");

        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("label")
                    && matches!(&root.value, ExecutionValue::String { value } if value == "ready"))
        );
        assert_eq!(result.transcript, vec!["ready".to_string()]);
    }

    #[test]
    fn eval_supports_record_construction_and_field_access() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
type Date =
  | Date { year: int64, month: int64, day: int64 }

mk_date = Date { year: cast(2024, int64), month: cast(2, int64), day: cast(29, int64) }
year = mk_date.year
label = if eq(year, cast(2024, int64)) then "leap" else "plain"
"#
            .to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");

        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("year")
                    && matches!(root.value, ExecutionValue::Int64 { value: 2024 }))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("label")
                    && matches!(&root.value, ExecutionValue::String { value } if value == "leap"))
        );
    }

    #[test]
    fn compile_host_records_emit_runtime_adt_access() {
        let artifact = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: r#"
type Date =
  | Date { year: int64, month: int64, day: int64 }

mk_date = Date { year: cast(2024, int64), month: cast(2, int64), day: cast(29, int64) }
year = mk_date.year
"#
            .to_string(),
            target: CompileTarget::C,
            entry_name: Some("record_demo".to_string()),
        })
        .expect("compile");

        let c_file = artifact
            .files
            .iter()
            .find(|file| file.path.ends_with(".c"))
            .expect("generated c file");

        assert!(
            c_file.contents.contains("chelis_adt_construct"),
            "expected record construction through generic ADT runtime, got:\n{}",
            c_file.contents
        );
        assert!(
            c_file.contents.contains("chelis_adt_get_field"),
            "expected record field access through generic ADT runtime, got:\n{}",
            c_file.contents
        );
    }

    #[test]
    fn host_lowering_preserves_rich_host_function_types() {
        let compiled = compile_source(
            SourceKind::Surf,
            r#"
module Std.Test

type Json =
  | JsonNull
  | JsonString(string)
  | JsonArray(List[Json])
  | JsonObject(Dict[string, Json])

type Tokenizer =
  | BpeTokenizer(Dict[string, int64], Dict[string, int64], Dict[int64, string], int64)

def parse_line(line: string) -> Option[List[string]] =
  Some([])

def json_string(value: Option[Json]) -> Option[string] =
  match value with {
    | Some(inner) =>
        match inner with {
          | JsonString(text) => Some(text)
          | _ => None
        }
    | None => None
  }

def load_tokenizer(path: string) -> Option[Tokenizer] =
  Some(BpeTokenizer(dict_of([]), dict_of([]), dict_of([]), cast(0, int64)))
"#,
        )
        .expect("compile");

        let host = chelis_ir::host::lower_compiled_program(&compiled.checked)
            .host
            .expect("host lowering");
        let lowered = chelis_ir::lower::top_level_lowering_map(
            compiled.checked.exprs(),
            compiled.checked.type_env(),
        );
        let checked_text = chelis_deep::printer::print_canonical(compiled.checked.exprs());

        let find_ret = |suffix: &str| {
            host.functions
                .iter()
                .find(|function| function.name == suffix || function.name.ends_with(suffix))
                .map(|function| function.ret_ty.clone())
        };

        let available = host
            .functions
            .iter()
            .map(|function| format!("{} -> {:?}", function.name, function.ret_ty))
            .collect::<Vec<_>>();
        let lowered_debug = lowered
            .iter()
            .map(|(name, lowered)| format!("{name}={lowered}"))
            .collect::<Vec<_>>();

        assert_eq!(
            find_ret("parse_line"),
            Some(chelis_ir::host::HostType::Option(Box::new(
                chelis_ir::host::HostType::List(Box::new(chelis_ir::host::HostType::String))
            ))),
            "available functions: {available:#?}\nlowered: {lowered_debug:#?}\nchecked:\n{checked_text}"
        );
        assert_eq!(
            find_ret("json_string"),
            Some(chelis_ir::host::HostType::Option(Box::new(
                chelis_ir::host::HostType::String
            ))),
            "available functions: {available:#?}\nlowered: {lowered_debug:#?}\nchecked:\n{checked_text}"
        );
        assert_eq!(
            find_ret("load_tokenizer"),
            Some(chelis_ir::host::HostType::Option(Box::new(
                chelis_ir::host::HostType::Adt("Tokenizer".to_string(), Vec::new())
            ))),
            "available functions: {available:#?}\nlowered: {lowered_debug:#?}\nchecked:\n{checked_text}"
        );
    }

    #[test]
    fn eval_many_returns_per_root_results_preserving_order() {
        // Two tensor roots: `a` depends on named input `x` (provided), `b`
        // depends on named input `y` (not provided). The live-mask restricts
        // DAG evaluation to the selected root's subgraph, so the missing-input
        // error for `b` surfaces independently of `a`'s success.
        // eval_many must preserve input order and carry both outcomes.
        let source = r#"
a: tensor[2, f32] = a
b: tensor[2, f32] = b
"#;

        let mut bindings = BTreeMap::new();
        bindings.insert(
            "a".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: vec![1.0, 2.0],
            },
        );
        // `b` is intentionally omitted so that evaluating root `b` fails.

        let results = eval_many(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.to_string(),
                bindings,
            },
            &["a".to_string(), "b".to_string()],
        );

        assert_eq!(results.len(), 2, "expected one entry per requested root");
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");

        let a_result = results[0].1.as_ref().expect("a should succeed");
        assert!(
            a_result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("a")),
            "expected evaluated root `a`, got {:?}",
            a_result.roots
        );

        let b_err = results[1]
            .1
            .as_ref()
            .expect_err("b should fail because input `b` was not provided");
        assert_eq!(b_err.stage, "eval");
        assert!(
            b_err
                .errors
                .iter()
                .any(|diag| diag.message.contains('b') || diag.kind == "eval_error"),
            "expected eval error mentioning `b`, got {:?}",
            b_err.errors
        );
    }

    #[test]
    fn eval_many_reverse_order_also_isolates_failure() {
        // Negative-parity: run the same program with the roots in reverse
        // order. The successful root must still succeed after the failing
        // root — there must be no hidden shared state that outlives a
        // per-root eval_compiled call.
        let source = r#"
a: tensor[2, f32] = a
b: tensor[2, f32] = b
"#;

        let mut bindings = BTreeMap::new();
        bindings.insert(
            "a".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: vec![1.0, 2.0],
            },
        );

        let results = eval_many(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.to_string(),
                bindings,
            },
            &["b".to_string(), "a".to_string()],
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "b");
        assert_eq!(results[1].0, "a");
        assert!(results[0].1.is_err(), "b must still fail");
        assert!(
            results[1].1.is_ok(),
            "a must still succeed even after b failed first"
        );
    }

    #[test]
    fn eval_many_propagates_compile_error_per_root() {
        // When the source itself fails to compile, every requested root must
        // see the same compile error rather than one root silently swallowing
        // the failure.
        let source = "a = this_name_does_not_exist_anywhere";

        let results = eval_many(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.to_string(),
                bindings: BTreeMap::new(),
            },
            &["a".to_string(), "b".to_string()],
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");
        for (name, result) in &results {
            assert!(
                result.is_err(),
                "root `{name}` should inherit the compile error"
            );
        }
    }

    #[test]
    fn eval_rejects_negative_shape_axis_with_signed_diagnostic() {
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
axis = tensor_to_scalar(scalar_to_tensor(cast(-1, int32)))
bad = shape(scalar_to_tensor(cast(3, int64)), axis)
"#
            .to_string(),
            bindings: BTreeMap::new(),
        })
        .expect_err("negative axis should fail");

        assert_eq!(error.stage, "eval");
        assert!(
            error
                .errors
                .iter()
                .any(|diag| diag.message == "shape requires non-negative axis, got -1")
        );
    }
}
