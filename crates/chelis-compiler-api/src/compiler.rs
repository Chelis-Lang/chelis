use std::collections::{BTreeMap, HashMap};

use chelis_backend_c::CodegenResult;
use chelis_backend_hip::HipCodegenResult;
use chelis_deep::Expr as DeepExpr;
use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStepOp, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{self, TensorValue as IrTensorValue};
use chelis_surf::ast::{
    BinOp, Decl, Expr, ImportKind, LetBinding, LetPattern, Literal, MatchArm, Param, Pattern,
    TypeExpr, UnaryOp, Variant, VariantFields,
};
use chelis_types::errors::CheckError;

use crate::schema::{
    BatchRequest, BatchResult, BatchResultEnvelope, CheckResult, CompileRequest, CompileResult,
    CompileTarget, DecompileRequest, DecompileResult, DesugarRequest, DesugarResult, Diagnostic,
    EvalRequest, EvalResult, EvaluatedRoot, FitnessComponents, GeneratedFile, GradRequest,
    GradResult, LowerRequest, LowerResult, ParseRequest, ParseResult, SourceKind, Span,
    TensorValue, ValidateMode, ValidateRequest, ValidateResult, WireBinOp, WireDag, WireDagNode,
    WireDeepAtom, WireDeepExpr, WireDeepExprKind, WireDimInfo, WireFusedInput, WireFusedStep,
    WireFusedStepOp, WireImportKind, WireLetBinding, WireLetPattern, WireLiteral, WireMatchArm,
    WireMetaEntry, WireParam, WirePattern, WireRecordExprField, WireRecordPatternField,
    WireRecordTypeField, WireRiscOp, WireSurfDecl, WireSurfExpr, WireSurfTypeExpr, WireTensorType,
    WireUnaryOp, WireVariant, WireVariantFields,
};

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-c/runtime/chelis_runtime.h"
));
const RUNTIME_C: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-c/runtime/chelis_runtime.c"
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
    let report = chelis_types::check_phase0e_fitness(&deep_exprs);
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
    let func_name = request
        .entry_name
        .unwrap_or_else(|| "chelis_main".to_string());

    match request.target {
        CompileTarget::C => {
            reject_unsized_named_dims(&compiled.dag, "c")?;
            let fused = chelis_ir::fuse::fuse(&compiled.dag);
            let result = chelis_backend_c::codegen(&fused, &func_name);
            Ok(compiled_execution_artifact(
                request.target,
                &func_name,
                None,
                compile_result_c(request.target, &func_name, &result),
                execution_input_specs(&compiled.dag, &result.input_labels),
                execution_output_specs(&compiled.dag, &result.output_labels),
                result.symbolic_dims,
            ))
        }
        CompileTarget::Hip => {
            reject_unsized_named_dims(&compiled.dag, "hip")?;
            reject_unsupported_hip_ops(&compiled.dag)?;
            let fused = chelis_ir::fuse::fuse(&compiled.dag);
            let result = chelis_backend_hip::codegen_hip(&fused, &func_name);
            Ok(compiled_execution_artifact(
                request.target,
                &func_name,
                Some(format!("{func_name}_device")),
                compile_result_hip(request.target, &func_name, &result),
                execution_input_specs(&compiled.dag, &result.input_labels),
                execution_output_specs(&compiled.dag, &result.output_labels),
                result.symbolic_dims,
            ))
        }
    }
}

pub fn eval(request: EvalRequest) -> Result<EvalResult> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    if compiled.dag.roots().is_empty() {
        return Err(stage_error(
            "eval",
            "program produced no evaluable roots",
            "other",
        ));
    }

    let bindings = request
        .bindings
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

    let roots = compiled.dag.roots().to_vec();
    let values = eval::eval_tensor_roots_with_strict(&compiled.dag, &roots, |name| {
        bindings.get(name).cloned()
    })
    .map_err(|message| stage_error("eval", message, "eval_error"))?;

    Ok(EvalResult {
        roots: roots
            .into_iter()
            .map(|root| EvaluatedRoot {
                node_id: root.0,
                name: compiled.root_name_by_id.get(&root).cloned(),
                value: tensor_value(values.get(&root).expect("root value missing")),
            })
            .collect(),
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
    dag: Dag,
    named_roots: BTreeMap<String, NodeId>,
    root_name_by_id: HashMap<NodeId, String>,
    forward_nodes_by_name: BTreeMap<String, NodeId>,
}

type RootNameBuilder = Box<dyn FnOnce(&HashMap<String, DeepExpr>) -> Vec<String>>;

fn compile_source(source_kind: SourceKind, source: &str) -> Result<CompiledSource> {
    let (deep_exprs, root_name_builder): (Vec<DeepExpr>, RootNameBuilder) = match source_kind {
        SourceKind::Surf => {
            let decls = parse_surf(source)?;
            let deep_exprs = chelis_macros::expand_program(
                &chelis_surf::desugar::desugar_program(&decls),
                &chelis_macros::ExpansionOptions::default(),
            )
            .map_err(|err| stage_error("desugar", err.to_string(), "macro_error"))?
            .into_exprs();
            (
                deep_exprs,
                Box::new(move |type_env| lowered_root_names_from_surf(&decls, type_env)),
            )
        }
        SourceKind::Deep => {
            let deep_exprs = parse_deep(source)?;
            (
                deep_exprs.clone(),
                Box::new(move |type_env| lowered_root_names_from_deep(&deep_exprs, type_env)),
            )
        }
    };

    let checked =
        chelis_types::check_phase0e_program(&deep_exprs).map_err(|report| CompilerError {
            stage: "check".to_string(),
            errors: report.errors.iter().map(check_error_diagnostic).collect(),
        })?;
    let root_names = root_name_builder(checked.type_env());
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

    let dag = chelis_ir::lower::lower_program(&checked);

    if !root_names.is_empty() && dag.roots().len() != root_names.len() {
        return Err(stage_error(
            "lower",
            format!(
                "lowered root count mismatch: expected {} named roots, got {}",
                root_names.len(),
                dag.roots().len()
            ),
            "lower_error",
        ));
    }

    let named_roots = root_names
        .into_iter()
        .zip(dag.roots().iter().copied())
        .collect::<BTreeMap<_, _>>();

    let root_name_by_id = named_roots
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect::<HashMap<_, _>>();

    let mut forward_nodes_by_name = named_roots.clone();
    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            forward_nodes_by_name.entry(name.clone()).or_insert(node.id);
        }
    }

    Ok(CompiledSource {
        dag,
        named_roots,
        root_name_by_id,
        forward_nodes_by_name,
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

fn lowered_root_names_from_surf(
    decls: &[Decl],
    type_env: &HashMap<String, DeepExpr>,
) -> Vec<String> {
    let mut names = Vec::new();
    for decl in decls {
        collect_surf_decl_names(decl, type_env, &mut names);
    }
    names
}

fn collect_surf_decl_names(
    decl: &Decl,
    type_env: &HashMap<String, DeepExpr>,
    out: &mut Vec<String>,
) {
    match decl {
        Decl::FunDef { name, .. } | Decl::LetDef { name, .. } => {
            extend_root_names(name, type_env.get(name), out)
        }
        Decl::Module { decls, .. } => {
            for decl in decls {
                collect_surf_decl_names(decl, type_env, out);
            }
        }
        _ => {}
    }
}

fn lowered_root_names_from_deep(
    exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
) -> Vec<String> {
    let mut names = Vec::new();
    for expr in exprs {
        collect_deep_decl_names(expr, type_env, &mut names);
    }
    names
}

fn collect_deep_decl_names(
    expr: &DeepExpr,
    type_env: &HashMap<String, DeepExpr>,
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
                collect_deep_decl_names(child, type_env, out);
            }
        }
        "def" => {
            if let Some(name) = list.elements.get(2).and_then(symbol_name) {
                extend_root_names(name, type_env.get(name), out);
            }
        }
        _ => {}
    }
}

fn extend_root_names(name: &str, ty: Option<&DeepExpr>, out: &mut Vec<String>) {
    if let Some(DeepExpr::List(list, _)) = ty
        && let Some(tag) = list_tag(list)
    {
        if tag == "t-fn" {
            extend_root_names(name, list.elements.last(), out);
            return;
        }
        if tag == "t-tuple" {
            for (index, child) in list.elements.iter().skip(2).enumerate() {
                extend_root_names(&format!("{name}.{index}"), Some(child), out);
            }
            return;
        }
    }
    out.push(name.to_string());
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
                path: "chelis_runtime.c".to_string(),
                contents: RUNTIME_C.to_string(),
            },
        ],
        compile_flags: result.compile_flags.clone(),
        link_flags: result.link_flags.clone(),
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
                path: "chelis_runtime.c".to_string(),
                contents: RUNTIME_C.to_string(),
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

fn execution_input_specs(dag: &Dag, labels: &[String]) -> Vec<ExecutionTensorSpec> {
    let mut load_types = HashMap::<String, TensorType>::new();
    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            load_types
                .entry(name.clone())
                .or_insert_with(|| node.output_type.clone());
        }
    }
    labels
        .iter()
        .map(|label| {
            let ty = load_types
                .get(label)
                .unwrap_or_else(|| panic!("missing load type for `{label}`"));
            execution_tensor_spec(label.clone(), ty)
        })
        .collect()
}

fn execution_output_specs(dag: &Dag, labels: &[String]) -> Vec<ExecutionTensorSpec> {
    let nodes = execution_output_nodes(dag);
    labels
        .iter()
        .zip(nodes)
        .map(|(label, node_id)| {
            let ty = &dag
                .get(node_id)
                .unwrap_or_else(|| panic!("missing output node {}", node_id.0))
                .output_type;
            execution_tensor_spec(label.clone(), ty)
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
            _ => {}
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

fn stage_error(stage: &str, message: impl Into<String>, kind: &str) -> CompilerError {
    stage_error_with_span(stage, message, kind, None)
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
    };
    Some(Span { offset, len: 0 })
}

fn check_error_diagnostic(error: &CheckError) -> Diagnostic {
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

fn tensor_value(value: &IrTensorValue) -> TensorValue {
    TensorValue {
        shape: value.shape.clone(),
        data: value.data.clone(),
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
        Expr::Let(bindings, body, s) => WireSurfExpr::Let {
            bindings: bindings.iter().map(wire_let_binding).collect(),
            body: Box::new(wire_expr(body)),
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
        RiscOp::Dropout { rate, seed } => WireRiscOp::Dropout {
            rate: *rate,
            seed: *seed,
        },
        RiscOp::Sum { axis } => WireRiscOp::Sum { axis: *axis },
        RiscOp::MaxReduce { axis } => WireRiscOp::MaxReduce { axis: *axis },
        RiscOp::Reshape { new_shape } => WireRiscOp::Reshape {
            new_shape: new_shape.iter().map(wire_dim).collect(),
        },
        RiscOp::Permute { axes } => WireRiscOp::Permute { axes: axes.clone() },
        RiscOp::Expand { axis, size } => WireRiscOp::Expand {
            axis: *axis,
            size: size.to_string(),
        },
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
        RiscOp::Load { name } => WireRiscOp::Load { name: name.clone() },
        RiscOp::Store { name } => WireRiscOp::Store { name: name.clone() },
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
    }
}
