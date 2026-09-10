use crate::schema::numbers::{NonnegativeExtent, SourceInteger};
use chelis_deep::DeepTag;
use chelis_types::types::Prim;
use chelis_unord::{UnordMap, UnordSet};
use std::collections::{BTreeMap, BTreeSet};

use chelis_backend_c::CodegenResult;
use chelis_backend_hip::HipCodegenResult;
use chelis_deep::Expr as DeepExpr;
use chelis_ir::dag::{
    Dag, DimInfo, ExtremaKind, ExtremaOperand, FusedInput, FusedStepOp, NodeId, RiscOp, RtDim,
    TensorType,
};
use chelis_ir::eval;
use chelis_surf::ast::{
    BinOp, Decl, Expr, ImportKind, LetBinding, LetPattern, MatchArm, Param, Pattern, TypeExpr,
    UnaryOp, Variant, VariantFields,
};
use chelis_types::{
    CheckedProgram,
    errors::CheckError,
    manifest::{ManifestedProgram, RootEntry, RootManifest},
    types::{Lane, Target},
};
use sha2::{Digest, Sha256};

use crate::runtime::{
    RuntimeTensorValue, evaluate_host_program_filtered,
    evaluate_host_program_with_library_and_types, lookup_runtime_value_for_manifest_root,
    runtime_value_to_schema,
};
use crate::schema::{
    AddFunctionRequest, AddFunctionResult, AddPropertyRequest, AddPropertyResult, BatchRequest,
    BatchResult, BatchResultEnvelope, ChangeSignatureRequest, ChangeSignatureResult, CheckResult,
    CompileRequest, CompileResult, CompileTarget, DecompileRequest, DecompileResult,
    DeepCallGraphRequest, DeepCallGraphResult, DeepFunctionOutline, DeepOutlineRequest,
    DeepOutlineResult, DeepReference, DeepReferencesRequest, DeepReferencesResult, DesugarRequest,
    DesugarResult, Diagnostic, DiagnosticSpan, EvalRequest, EvalResult, EvaluatedRoot,
    FitnessComponents, GeneralKind, GeneratedFile, GradRequest, GradResult, LowerRequest,
    LowerResult, ParseRequest, ParseResult, RenameRequest, RenameResult, ReplaceFunctionRequest,
    ReplaceFunctionResult, RootManifestEntryResult, RootManifestResult, SourceKind, Span,
    ValidateMode, ValidateRequest, ValidateResult, WireBinOp, WireDag, WireDagNode,
    WireDagSchemaError, WireDimExpr, WireDimInfo, WireExtentWitnessSite, WireExtremaKind,
    WireExtremaOperand, WireFusedInput, WireFusedStep, WireFusedStepOp, WireImportKind,
    WireLetBinding, WireLetPattern, WireMatchArm, WireParam, WirePattern, WirePropertyOption,
    WireRecordExprField, WireRecordPatternField, WireRecordTypeField, WireRiscOp, WireRtAxis,
    WireRtDim, WireSurfDecl, WireSurfExpr, WireSurfTypeExpr, WireTensorType, WireTypeInvariant,
    WireUnaryOp, WireVariant, WireVariantFields,
};
use crate::schema::{stage_error, stage_error_with_span, unsupported_stage_error};
use crate::source_wire::{SourceWireResult, wire_deep_expr, wire_literal};

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime.h"
));
const RUNTIME_DTYPE_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime_dtype.h"
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
    pub size: Option<NonnegativeExtent>,
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
    /// Why the entry-scoped metadata lane declined this compilation, when
    /// it did. `None` means the lane claimed the compilation — or never
    /// ran: the lane is C-target-only, so e.g. a HIP artifact always
    /// carries `None`. `Some` means the lane did NOT produce the
    /// entry-scoped artifact; it does not by itself mean "no callable
    /// interface". On the legacy whole-DAG path the artifact can carry
    /// real (whole-program, merged) `inputs`/`outputs` alongside
    /// `Some(decline)` — reachable for `HasGlobals` (a fully-DAG-lowerable
    /// scalar-global program) and for `LoweringFailed`/`EmptyAfterDce`/
    /// `InputsOutsideParams` (the whole-program DAG can still have roots
    /// when the entry-scoped lowering fails). Consumers deciding
    /// callability must check `inputs`/`outputs`, not this field.
    ///
    /// [`EntryLaneDecline`] is deliberately `#[non_exhaustive]`: the
    /// #828/#830 tiering work will add variants, so downstream matches
    /// need a generic wildcard arm.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub entry_lane_decline: Option<EntryLaneDecline>,
}

#[derive(Debug, Clone)]
pub struct CompilerError {
    /// Output already produced by a failed evaluation, in execution order.
    /// Empty for failures before execution; never contains fabricated roots.
    pub transcript: Vec<String>,
    pub stage: String,
    pub errors: Vec<Diagnostic>,
}

/// `Diagnostic::kind` for an evaluation abandoned via a cancellation token
/// (chelis#914), as opposed to one that failed on its own merits.
///
/// The spelling is read off the sealed vocabulary rather than restated, so an
/// embedder comparing against this constant and a producer constructing
/// [`GeneralKind::Cancelled`] cannot drift apart.
pub const EVAL_CANCELLED_KIND: &str = chelis_vocab::DiagnosticKind::Cancelled.as_str();

impl CompilerError {
    /// Whether this error is a cancellation rather than a genuine failure.
    ///
    /// Prefer this over inspecting messages: it reads the structured
    /// `Diagnostic::kind` set at the eval-stage boundary, so it cannot be
    /// confused by a program whose own error text discusses cancellation.
    pub fn is_cancellation(&self) -> bool {
        self.errors
            .iter()
            .any(|diagnostic| diagnostic.kind() == chelis_vocab::DiagnosticKind::Cancelled)
    }
}

/// The error a front-end phase returns when the compile was abandoned
/// (chelis#930).
///
/// `stage` is the phase that was running or about to run, which is what the
/// operator wants to know ("it was still type-checking"). Classification is
/// structural — [`EVAL_CANCELLED_KIND`] — exactly as for the eval lanes; the
/// message carries the sentinel only for the CLI's `--timeout` boundary, which
/// sees flattened text rather than the typed error.
pub(crate) fn cancelled_stage_error(stage: &str) -> CompilerError {
    stage_error(
        stage,
        chelis_types::EVAL_CANCELLED_MSG,
        GeneralKind::Cancelled,
    )
}

/// Front-end phase boundary (chelis#930): abandon the compile if cancellation
/// has been requested.
///
/// Phase boundaries alone bound interrupt latency to one phase; the passes
/// that dominate a large front end also poll per top-level declaration
/// (`chelis-types`), which is what makes the bound useful rather than nominal.
pub(crate) fn bail_if_cancelled(stage: &str) -> Result<()> {
    if chelis_types::cancellation_requested() {
        return Err(cancelled_stage_error(stage));
    }
    Ok(())
}

/// Replace a phase's error with the cancellation error when the compile was
/// abandoned mid-phase (chelis#930).
///
/// A pass that stopped early reports whatever its truncated view implied —
/// unbound tail declarations, missing signatures. Those are artefacts of the
/// abandonment, not findings about the program, and reporting them would be
/// actively misleading (the user asked to stop; the program may be fine).
/// Cancellation is one-way, so this decision is deterministic: once the token
/// is tripped it stays tripped for the rest of the compile.
pub(crate) fn cancelled_or(stage: &str, error: CompilerError) -> CompilerError {
    if chelis_types::cancellation_requested() {
        cancelled_stage_error(stage)
    } else {
        error
    }
}

type Result<T> = std::result::Result<T, CompilerError>;

pub fn parse(request: ParseRequest) -> Result<ParseResult> {
    match request.source_kind {
        SourceKind::Surf => {
            let decls = parse_surf(&request.source)?;
            Ok(ParseResult {
                source_kind: SourceKind::Surf,
                surf_ast: Some(
                    decls
                        .iter()
                        .map(wire_decl)
                        .collect::<SourceWireResult<_>>()
                        .map_err(|message| {
                            stage_error("parse", &message, GeneralKind::ValidationError)
                        })?,
                ),
                deep_ast: None,
            })
        }
        SourceKind::Deep => {
            let exprs = parse_deep(&request.source)?;
            Ok(ParseResult {
                source_kind: SourceKind::Deep,
                surf_ast: None,
                deep_ast: Some(
                    exprs
                        .iter()
                        .map(wire_deep_expr)
                        .collect::<SourceWireResult<_>>()
                        .map_err(|message| {
                            stage_error("parse", &message, GeneralKind::ValidationError)
                        })?,
                ),
            })
        }
    }
}

pub fn desugar(request: DesugarRequest) -> Result<DesugarResult> {
    let decls = parse_surf(&request.source)?;
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    Ok(DesugarResult {
        deep_text: chelis_deep::printer::print_canonical(&deep_exprs),
        deep_ast: deep_exprs
            .iter()
            .map(wire_deep_expr)
            .collect::<SourceWireResult<_>>()
            .map_err(|message| stage_error("desugar", &message, GeneralKind::ValidationError))?,
    })
}

/// Replace one function's body in a Deep module with a new Deep body, returning
/// the canonical Deep of the changed def and the full rewritten module.
///
/// Deep-native and pure: it parses the `module` and `new_body` as Deep, runs
/// full whole-module `chelis check` on the rewritten module (the
/// `check_body_replacement` seam runs FITNESS, then TYPE, then EFFECTS, then
/// LINEARITY over the whole rewritten module), and returns canonical Deep on
/// success. Nothing is persisted; no file is written. On any failure it returns
/// a structured [`CompilerError`] whose stage names the rejecting pass.
pub fn replace_function_body(
    request: crate::schema::ReplaceFunctionBodyRequest,
) -> Result<crate::schema::ReplaceFunctionBodyResult> {
    // The new body must be exactly one Deep expression, stamped in the
    // RuntimeExpr role it will occupy. Zero or many is a parse-stage
    // rejection, not a check failure.
    let new_body = parse_one_deep_runtime_expr("replace", "new_body", &request.new_body)?;

    // The module goes through the stamped `.dp` ingress too: a malformed or
    // role-invalid `.dp` is a parse error.
    let module = parse_deep_authoring_module("replace", &request.module)?;

    // Whole-module body-replacement check: full `chelis check` of the rewritten
    // module. On rejection the error is tagged by the failing pass; on success
    // the report carries the full rewritten module the verdict equals.
    let report =
        crate::fragment::check_body_replacement(&module, &request.function_name, &new_body)
            .map_err(replacement_error_to_compiler_error)?;

    // The single rewritten def, for `changed_def_deep`. `spliced_function_def`
    // returns just the one `(def ...)` node; print it on its own line.
    let changed_def = chelis_deep::spliced_function_def(&module, &request.function_name, new_body)
        .map_err(|err| stage_error("replace", err.to_string(), GeneralKind::NameResolutionError))?;

    Ok(crate::schema::ReplaceFunctionBodyResult {
        changed_def_deep: chelis_deep::printer::print_expr(&changed_def),
        module_deep: chelis_deep::printer::print_canonical(report.validated_module.as_exprs()),
    })
}

/// Add one function declaration bundle to a Deep module.
///
/// The tool is Deep-native and pure. It accepts Deep text for the full module
/// and the new declaration bundle, performs only request-shape and insertion
/// target checks before editing, then runs the shared whole-module validation
/// gate over the rewritten module. Semantic failures such as duplicate `def`,
/// duplicate `defsig`, type errors, effects, and linearity violations are
/// surfaced from that whole-module pipeline.
pub fn add_function(request: AddFunctionRequest) -> Result<AddFunctionResult> {
    let module = parse_deep_authoring_module("add-function", &request.module)?;

    let new_decl_exprs = parse_deep_authoring_decls("add-function", &request.new_decls)?;
    let parsed = parse_add_function_decls(new_decl_exprs)?;

    let rewritten = chelis_deep::insert_function_decls(
        &module,
        &parsed.ordered_decls,
        request.insert_after_function.as_deref(),
    )
    .map_err(add_function_insert_error_to_compiler_error)?;

    let report = crate::fragment::check_whole_module_edit(rewritten)
        .map_err(edit_validation_error_to_compiler_error)?;

    Ok(AddFunctionResult {
        added_def_deep: chelis_deep::printer::print_expr(&parsed.def),
        added_defsig_deep: parsed.defsig.as_ref().map(chelis_deep::printer::print_expr),
        module_deep: chelis_deep::printer::print_canonical(report.as_exprs()),
    })
}

pub fn deep_outline(request: DeepOutlineRequest) -> Result<DeepOutlineResult> {
    let module = parse_deep_authoring_module("deep-outline", &request.module)?;
    let outline = chelis_deep::authoring::outline(&module)
        .map_err(|err| authoring_error_to_compiler_error("deep-outline", err))?;
    Ok(DeepOutlineResult {
        module_name: outline.module_name,
        exports: outline.exports,
        functions: outline
            .functions
            .into_iter()
            .map(|function| {
                let preimage_sha256 = sha256_hex(function.def_deep.as_bytes());
                DeepFunctionOutline {
                    name: function.name,
                    qualified_name: function.qualified_name,
                    params: function.params,
                    has_defsig: function.has_defsig,
                    body_path: function.body_path,
                    def_deep: function.def_deep,
                    defsig_deep: function.defsig_deep,
                    preimage_sha256,
                }
            })
            .collect(),
    })
}

pub fn deep_references(request: DeepReferencesRequest) -> Result<DeepReferencesResult> {
    let module = parse_deep_authoring_module("deep-references", &request.module)?;
    let refs = chelis_deep::authoring::references(&module, &request.symbol)
        .map_err(|err| authoring_error_to_compiler_error("deep-references", err))?;
    Ok(DeepReferencesResult {
        symbol: refs.symbol,
        references: refs
            .references
            .into_iter()
            .map(wire_deep_reference)
            .collect(),
    })
}

pub fn deep_call_graph(request: DeepCallGraphRequest) -> Result<DeepCallGraphResult> {
    let module = parse_deep_authoring_module("deep-call-graph", &request.module)?;
    let graph = chelis_deep::authoring::call_graph(&module)
        .map_err(|err| authoring_error_to_compiler_error("deep-call-graph", err))?;
    Ok(DeepCallGraphResult {
        edges: graph.edges.into_iter().map(wire_deep_reference).collect(),
    })
}

pub fn replace_function(request: ReplaceFunctionRequest) -> Result<ReplaceFunctionResult> {
    let module = parse_deep_authoring_module("replace-function", &request.module)?;
    check_preimage(
        &module,
        &request.function_name,
        request.preimage_sha256.as_deref(),
    )?;
    let new_decls = parse_deep_authoring_decls("replace-function", &request.new_decls)?;
    let edited =
        chelis_deep::authoring::replace_function(&module, &request.function_name, &new_decls)
            .map_err(|err| authoring_error_to_compiler_error("replace-function", err))?;
    let report = crate::fragment::check_whole_module_edit(edited.module)
        .map_err(edit_validation_error_to_compiler_error)?;
    Ok(ReplaceFunctionResult {
        replaced_def_deep: chelis_deep::printer::print_expr(&edited.replaced_def),
        replaced_defsig_deep: edited
            .replaced_defsig
            .as_ref()
            .map(chelis_deep::printer::print_expr),
        module_deep: chelis_deep::printer::print_canonical(report.as_exprs()),
    })
}

pub fn rename(request: RenameRequest) -> Result<RenameResult> {
    let module = parse_deep_authoring_module("rename", &request.module)?;
    check_preimage(
        &module,
        &request.function_name,
        request.preimage_sha256.as_deref(),
    )?;
    let edited =
        chelis_deep::authoring::rename_function(&module, &request.function_name, &request.new_name)
            .map_err(|err| authoring_error_to_compiler_error("rename", err))?;
    let report = crate::fragment::check_whole_module_edit(edited.module)
        .map_err(edit_validation_error_to_compiler_error)?;
    Ok(RenameResult {
        renamed_def_deep: chelis_deep::printer::print_expr(&edited.renamed_def),
        renamed_defsig_deep: edited
            .renamed_defsig
            .as_ref()
            .map(chelis_deep::printer::print_expr),
        module_deep: chelis_deep::printer::print_canonical(report.as_exprs()),
        renamed_references: edited
            .renamed_references
            .try_into()
            .map_err(|error| stage_error("rename", error, GeneralKind::Other))?,
    })
}

pub fn change_signature(request: ChangeSignatureRequest) -> Result<ChangeSignatureResult> {
    let module = parse_deep_authoring_module("change-signature", &request.module)?;
    check_preimage(
        &module,
        &request.function_name,
        request.preimage_sha256.as_deref(),
    )?;
    let new_defsig =
        parse_one_deep_authoring_decl("change-signature", "new_defsig", &request.new_defsig)?;
    let new_params = parse_one_deep_authoring_tagged(
        "change-signature",
        "new_params",
        &request.new_params,
        DeepTag::Params,
    )?;
    let param_renames: Vec<(String, String)> = request.param_renames.into_iter().collect();
    let edited = chelis_deep::authoring::change_signature(
        &module,
        &request.function_name,
        &new_defsig,
        &new_params,
        &request.argument_order,
        &param_renames,
    )
    .map_err(|err| authoring_error_to_compiler_error("change-signature", err))?;
    let report = crate::fragment::check_whole_module_edit(edited.module)
        .map_err(edit_validation_error_to_compiler_error)?;
    Ok(ChangeSignatureResult {
        changed_def_deep: chelis_deep::printer::print_expr(&edited.changed_def),
        changed_defsig_deep: chelis_deep::printer::print_expr(&edited.changed_defsig),
        module_deep: chelis_deep::printer::print_canonical(report.as_exprs()),
        rewritten_calls: edited
            .rewritten_calls
            .try_into()
            .map_err(|error| stage_error("change-signature", error, GeneralKind::Other))?,
    })
}

pub fn add_property(request: AddPropertyRequest) -> Result<AddPropertyResult> {
    let module = parse_deep_authoring_module("add-property", &request.module)?;
    let new_decl_exprs = parse_deep_authoring_decls("add-property", &request.new_decls)?;
    let parsed = parse_add_function_decls(new_decl_exprs)?;
    if !deep_def_has_role(&parsed.def, "property") {
        return Err(stage_error(
            "add-property",
            "`new_decls` def must carry `chelis_role: \"property\"`",
            GeneralKind::DeepDeclError,
        ));
    }
    let rewritten = chelis_deep::insert_function_decls(
        &module,
        &parsed.ordered_decls,
        request.insert_after_function.as_deref(),
    )
    .map_err(add_function_insert_error_to_compiler_error)?;
    let report = crate::fragment::check_whole_module_edit(rewritten)
        .map_err(edit_validation_error_to_compiler_error)?;
    Ok(AddPropertyResult {
        added_property_def_deep: chelis_deep::printer::print_expr(&parsed.def),
        added_defsig_deep: parsed.defsig.as_ref().map(chelis_deep::printer::print_expr),
        module_deep: chelis_deep::printer::print_canonical(report.as_exprs()),
    })
}

/// Stamped whole-module text ingress for the authoring APIs (chelis#1088).
///
/// A `module` field is a `.dp` program, so a top-level `(module ...)` wrapper
/// and bare declarations are both admissible and nothing else is. The
/// tag-vocabulary sweep that `parse_str_strict` used to supply is applied
/// explicitly afterwards, so an authoring request naming an unknown tag is
/// still rejected at its own boundary rather than reaching a rewriter.
fn parse_deep_authoring_module(stage: &str, source: &str) -> Result<Vec<DeepExpr>> {
    let exprs =
        chelis_deep::parse_and_stamp_file(source).map_err(|err| deep_ingress_error(stage, &err))?;
    require_valid_deep(stage, &exprs)?;
    Ok(exprs)
}

/// Stamped declaration-bundle text ingress for the authoring APIs
/// (chelis#1088). Every top-level form must be a declaration.
fn parse_deep_authoring_decls(stage: &str, source: &str) -> Result<Vec<DeepExpr>> {
    let exprs =
        chelis_deep::parse_and_stamp(source).map_err(|err| deep_ingress_error(stage, &err))?;
    require_valid_deep(stage, &exprs)?;
    Ok(exprs)
}

/// Stamped ingress for a field whose contract names one exact Deep tag.
fn parse_one_deep_authoring_tagged(
    stage: &str,
    field: &str,
    source: &str,
    expected: DeepTag,
) -> Result<DeepExpr> {
    let exprs = chelis_deep::parse_and_stamp_tagged(source, expected)
        .map_err(|err| deep_ingress_error(stage, &err))?;
    require_valid_deep(stage, &exprs)?;
    exactly_one_deep(stage, field, exprs)
}

/// Stamped ingress for a field that must be exactly one declaration.
fn parse_one_deep_authoring_decl(stage: &str, field: &str, source: &str) -> Result<DeepExpr> {
    let exprs = parse_deep_authoring_decls(stage, source)?;
    exactly_one_deep(stage, field, exprs)
}

/// Stamped ingress for a field that must be exactly one runtime expression.
fn parse_one_deep_runtime_expr(stage: &str, field: &str, source: &str) -> Result<DeepExpr> {
    let exprs = chelis_deep::parse_and_stamp_runtime_exprs(source)
        .map_err(|err| deep_ingress_error(stage, &err))?;
    require_valid_deep(stage, &exprs)?;
    exactly_one_deep(stage, field, exprs)
}

fn exactly_one_deep(stage: &str, field: &str, exprs: Vec<DeepExpr>) -> Result<DeepExpr> {
    let count = exprs.len();
    match exprs.into_iter().next() {
        Some(single) if count == 1 => Ok(single),
        _ => Err(stage_error(
            stage,
            format!("`{field}` must be exactly one Deep expression, got {count}"),
            GeneralKind::DeepParseError,
        )),
    }
}

/// Apply the Deep structural/vocabulary sweep the editing surfaces have
/// always run, now on the stamped carrier.
fn require_valid_deep(stage: &str, exprs: &[DeepExpr]) -> Result<()> {
    match chelis_deep::validate::validate(exprs).into_iter().next() {
        None => Ok(()),
        Some(warning) => Err(stage_error_with_span(
            stage,
            warning.message,
            GeneralKind::DeepParseError,
            // chelis#1395: `validate` reports a coordinate and no end, so
            // the location travels as a point rather than an invented range.
            Some(DiagnosticSpan::Point {
                offset: crate::schema::host_index(warning.offset),
            }),
        )),
    }
}

fn wire_deep_reference(reference: chelis_deep::authoring::Reference) -> DeepReference {
    DeepReference {
        caller: reference.caller,
        callee: reference.callee,
        path: reference.path,
    }
}

fn check_preimage(module: &[DeepExpr], function_name: &str, expected: Option<&str>) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = target_preimage_sha256(module, function_name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(stage_error(
            "preimage",
            format!(
                "preimage hash mismatch for `{function_name}`: expected {expected}, got {actual}"
            ),
            GeneralKind::PreimageMismatch,
        ))
    }
}

fn target_preimage_sha256(module: &[DeepExpr], function_name: &str) -> Result<String> {
    let outline = chelis_deep::authoring::outline(module)
        .map_err(|err| authoring_error_to_compiler_error("deep-outline", err))?;
    let function = outline
        .functions
        .iter()
        .find(|function| function.name == function_name || function.qualified_name == function_name)
        .ok_or_else(|| {
            stage_error(
                "name-resolution",
                format!("function `{function_name}` not found"),
                GeneralKind::NameResolutionError,
            )
        })?;
    Ok(sha256_hex(function.def_deep.as_bytes()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn authoring_error_to_compiler_error(
    default_stage: &str,
    error: chelis_deep::authoring::AuthoringError,
) -> CompilerError {
    use chelis_deep::authoring::AuthoringError;
    match error {
        AuthoringError::Resolve(err) => stage_error(
            "name-resolution",
            err.to_string(),
            GeneralKind::NameResolutionError,
        ),
        AuthoringError::NoModule | AuthoringError::MultipleModules { .. } => {
            stage_error(default_stage, error.to_string(), GeneralKind::DeepDeclError)
        }
        AuthoringError::Metadata(error) => {
            stage_error(default_stage, error.to_string(), GeneralKind::DeepDeclError)
        }
        AuthoringError::InvalidDecl(message) => {
            stage_error(default_stage, message, GeneralKind::DeepDeclError)
        }
        AuthoringError::DuplicateFunction { .. } => stage_error(
            "name-resolution",
            error.to_string(),
            GeneralKind::DuplicateName,
        ),
        AuthoringError::PreimageMismatch { .. } => {
            stage_error("preimage", error.to_string(), GeneralKind::PreimageMismatch)
        }
        AuthoringError::CascadeIncomplete { .. } => {
            stage_error("cascade", error.to_string(), GeneralKind::CascadeIncomplete)
        }
    }
}

struct ParsedAddFunctionDecls {
    ordered_decls: Vec<DeepExpr>,
    def: DeepExpr,
    defsig: Option<DeepExpr>,
}

fn parse_add_function_decls(exprs: Vec<DeepExpr>) -> Result<ParsedAddFunctionDecls> {
    if !(1..=2).contains(&exprs.len()) {
        return Err(stage_error(
            "add-function",
            format!(
                "`new_decls` must contain exactly one `(def ...)` and an optional matching `(defsig ...)`, got {} top-level expressions",
                exprs.len()
            ),
            GeneralKind::DeepDeclError,
        ));
    }

    let mut def: Option<DeepExpr> = None;
    let mut defsig: Option<DeepExpr> = None;
    let mut def_name: Option<String> = None;
    let mut defsig_name: Option<String> = None;

    for expr in &exprs {
        let Some(tag) = deep_expr_tag(expr) else {
            return Err(add_function_decl_error(
                "`new_decls` entries must be Deep declaration lists",
            ));
        };
        match tag {
            DeepTag::Def => {
                if def.is_some() {
                    return Err(add_function_decl_error(
                        "`new_decls` must contain exactly one `(def ...)`",
                    ));
                }
                let name = deep_decl_name(expr).ok_or_else(|| {
                    add_function_decl_error("new `(def ...)` must carry a symbol name")
                })?;
                if !deep_def_is_function(expr) {
                    return Err(add_function_decl_error(
                        "`chelis_add_function` only accepts function `(def ...)` declarations",
                    ));
                }
                def_name = Some(name.to_string());
                def = Some(expr.clone());
            }
            DeepTag::Defsig => {
                if defsig.is_some() {
                    return Err(add_function_decl_error(
                        "`new_decls` may contain at most one `(defsig ...)`",
                    ));
                }
                let name = deep_decl_name(expr).ok_or_else(|| {
                    add_function_decl_error("new `(defsig ...)` must carry a symbol name")
                })?;
                defsig_name = Some(name.to_string());
                defsig = Some(expr.clone());
            }
            other => {
                return Err(add_function_decl_error(format!(
                    "`chelis_add_function` accepts only `(def ...)` and optional `(defsig ...)`, got `({} ...)`",
                    other.as_str()
                )));
            }
        }
    }

    let Some(def) = def else {
        return Err(add_function_decl_error(
            "`new_decls` must contain exactly one `(def ...)`",
        ));
    };
    if let (Some(def_name), Some(defsig_name)) = (def_name.as_deref(), defsig_name.as_deref())
        && def_name != defsig_name
    {
        return Err(add_function_decl_error(format!(
            "new `(defsig ...)` names `{defsig_name}` but new `(def ...)` names `{def_name}`"
        )));
    }

    Ok(ParsedAddFunctionDecls {
        ordered_decls: exprs,
        def,
        defsig,
    })
}

fn add_function_decl_error(message: impl Into<String>) -> CompilerError {
    stage_error("add-function", message, GeneralKind::DeepDeclError)
}

fn add_function_insert_error_to_compiler_error(
    error: chelis_deep::InsertFunctionError,
) -> CompilerError {
    use chelis_deep::InsertFunctionError;
    match error {
        InsertFunctionError::InsertionTarget(err) => stage_error(
            "name-resolution",
            err.to_string(),
            GeneralKind::NameResolutionError,
        ),
        InsertFunctionError::NoModule
        | InsertFunctionError::MultipleModules { .. }
        | InsertFunctionError::InvalidStampedRewrite { .. } => stage_error(
            "add-function",
            error.to_string(),
            GeneralKind::DeepDeclError,
        ),
    }
}

fn edit_validation_error_to_compiler_error(
    error: crate::fragment::EditValidationError,
) -> CompilerError {
    use crate::fragment::EditValidationError;
    let (kind, location, deep_path) = match &error {
        EditValidationError::Type {
            location,
            deep_path,
            ..
        } => (GeneralKind::TypeError, *location, deep_path.clone()),
        EditValidationError::Effect {
            location,
            deep_path,
            ..
        } => (GeneralKind::EffectError, *location, deep_path.clone()),
        EditValidationError::Linearity {
            location,
            deep_path,
            ..
        } => (GeneralKind::LinearityError, *location, deep_path.clone()),
    };
    let mut diagnostic = Diagnostic::general(
        kind,
        error.message(),
        crate::schema::numbers::UnitInterval::new(1.0).expect("constant severity"),
    );
    diagnostic.span = location;
    diagnostic.deep_path = deep_path.map(wire_deep_error_path);
    CompilerError {
        transcript: Vec::new(),
        stage: error.stage().to_string(),
        errors: vec![diagnostic],
    }
}

/// Map a [`crate::fragment::ReplacementError`] to a structured
/// [`CompilerError`]. The stage and `kind` discriminate the rejecting pass so
/// the caller can branch on it; the forward-compatible `deep_path` slot is
/// threaded through (always `None` in L0).
fn replacement_error_to_compiler_error(error: crate::fragment::ReplacementError) -> CompilerError {
    use crate::fragment::ReplacementError;
    let (kind, location, deep_path) = match &error {
        ReplacementError::NameResolution { location, .. } => {
            (GeneralKind::NameResolutionError, *location, None)
        }
        ReplacementError::Type {
            location,
            deep_path,
            ..
        } => (GeneralKind::TypeError, *location, deep_path.clone()),
        ReplacementError::Effect {
            location,
            deep_path,
            ..
        } => (GeneralKind::EffectError, *location, deep_path.clone()),
        ReplacementError::Linearity {
            location,
            deep_path,
            ..
        } => (GeneralKind::LinearityError, *location, deep_path.clone()),
    };
    let mut diagnostic = Diagnostic::general(
        kind,
        error.message(),
        crate::schema::numbers::UnitInterval::new(1.0).expect("constant severity"),
    );
    diagnostic.span = location;
    diagnostic.deep_path = deep_path.map(wire_deep_error_path);
    CompilerError {
        transcript: Vec::new(),
        stage: error.stage().to_string(),
        errors: vec![diagnostic],
    }
}

/// Convert the internal forward-compatible [`crate::fragment::DeepErrorPath`]
/// into its wire form. Never reached in L0 (the inner `deep_path` is always
/// `None`); present so L2 provenance threading adds no new mapping seam. The
/// path renders as a dot-joined sequence of segments (`body.2.0`) so the wire
/// shape is a plain string rather than the internal `DeepPath` type.
fn wire_deep_error_path(path: crate::fragment::DeepErrorPath) -> crate::schema::WireDeepErrorPath {
    use chelis_deep::path::PathSegment;
    let rendered = path
        .path
        .segments()
        .iter()
        .map(|segment| match segment {
            PathSegment::Body => "body".to_string(),
            PathSegment::Child(index) => index.to_string(),
        })
        .collect::<Vec<_>>()
        .join(".");
    crate::schema::WireDeepErrorPath {
        def_qualified_name: path.def_qualified_name,
        path: rendered,
    }
}

pub fn check(request: crate::schema::CheckRequest) -> Result<CheckResult> {
    bail_if_cancelled("parse")?;
    let outcome = crate::pipeline::run_source(crate::pipeline::PipelineRequest {
        source_kind: request.source_kind,
        source: &request.source,
        entry: None,
        goal: crate::pipeline::PipelineGoal::TypeAnalysis,
    })
    .map_err(pipeline_rejection_to_compiler_error)
    .map_err(|error| cancelled_or("check", error))?;
    let crate::pipeline::PipelineOutcome::TypeAnalysis(analysis) = outcome else {
        unreachable!("the type-analysis goal returns only a type-analysis outcome")
    };
    let report = match analysis {
        chelis_types::TypeAnalysisOutcome::Rejected { fitness }
        | chelis_types::TypeAnalysisOutcome::Accepted { fitness, .. } => fitness,
    };
    // chelis#930 review: a cancelled walk must not become a CheckResult.
    // Every pass check_ir_fitness runs can now stop early on a tripped
    // token, so without this bail a cancelled check returns
    // Ok(score: 1.0, errors: []) computed over the truncated walk --
    // "perfect success" whose empty error list is a coverage artifact.
    // An embedder can reach this on the public API today by installing a
    // CancelToken (exported since chelis#914) and calling compiler::check.
    bail_if_cancelled("check")?;
    CheckResult::try_from_fitness(&report)
        .map_err(|error| stage_error("report", error, GeneralKind::Other))
}

pub fn lower(request: LowerRequest) -> Result<LowerResult> {
    let compiled = compile_source_scoped(
        request.source_kind,
        &request.source,
        request.entry.as_deref(),
        Target::Eval,
    )?;
    let dag = wire_dag(&compiled.dag)
        .map_err(|error| stage_error("schema", error, GeneralKind::Other))?;
    // WI-2 validate-on-consume: fail closed before this DAG crosses the
    // process edge to the client. A build that emits a `schema_version` it
    // cannot itself interpret must surface a typed `schema`-stage error, not
    // hand the client an untrusted surface and not panic.
    schema_stage_check(dag.validate_schema_version())?;
    Ok(LowerResult {
        dag,
        named_roots: compiled
            .named_roots
            .into_entries()
            .map(|(name, node)| (name.into_string(), crate::schema::host_index(node.0)))
            .collect(),
    })
}

/// Compile to generated source files: the C-SOURCE surface (tide's
/// `/compile`, cove's live pane, python's `chelis.compile()`).
///
/// This surface keeps the LEGACY whole-program contract: with no
/// `entry_name`, a multi-def program with no unambiguous entry emits the
/// whole program (never an "ambiguous entry" error), and an `entry_name`
/// naming no def is the output symbol (sanitized), not a selector error.
/// Entry-integrity strictness belongs to [`compile_for_execution`], the
/// callable surface, where a merged manifest is the #817 defect; here the
/// merged whole-program emission IS the product behavior.
pub fn compile(request: CompileRequest) -> Result<CompileResult> {
    Ok(compile_for_execution_impl(request, EntryStrictness::Legacy)?.compile_result)
}

/// Entry-integrity policy for the shared compile pipeline.
///
/// `compile()` (tide/cove/`chelis.compile`) and `compile_for_execution`
/// (`compile_and_load`) share one pipeline, so entry-selection rules added
/// for the callable surface would otherwise leak into the C-source surface
/// (that leak broke tide/cove default compiles of multi-def programs, and
/// let a `vmap` entry reach a debug assert). The policy is threaded as a
/// value, not read from the request, so it cannot be set over the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryStrictness {
    /// The callable surface: unknown `entry_name`, ambiguous default,
    /// and entry-lane declines that would fall through to merged
    /// whole-program metadata are loud errors.
    Strict,
    /// The C-source surface: legacy whole-program behavior with the
    /// decline reason recorded on the artifact.
    Legacy,
}

/// Map a caller-supplied `entry_name` to the emitted C symbol / file
/// stem, decoupling def *selection* from the emitted *symbol*.
///
/// A def literally named `main` is a valid `entry_name` (and a valid entry
/// def, see #817/#818), but emitting a C function named `main` collides
/// with the reserved program entry `int main(int, char**, char**)` and the
/// native toolchain rejects the translation unit. Rewriting the symbol
/// keeps the metadata scoped to the requested def while producing a
/// linkable artifact; the artifact's `host_entry_name` carries the
/// rewritten symbol so the loader (`dlsym`) stays consistent.
///
/// Any character that is not a valid C identifier character is replaced
/// with `_`; a symbol that would start with a digit (or be empty) is
/// prefixed. Ordinary names (`solve`, `dloss`, `jsonish`) are returned
/// unchanged, so existing output-name usage is preserved.
///
/// ASYMMETRY (Fix 2, #817): this sanitizing mapping is used by the HOST
/// lane, by the free-form pure-DAG lane, and by the LEGACY surface's
/// decline-fallthrough (whole-program emission on [`compile`] when the
/// entry lane declines). The free-form lane is every program that lowers
/// NO host program — a file with no top-level `def`s, or exactly one
/// fully-DAG-lowerable `def` (multiple lowerable defs get per-def host
/// wrappers, so a host program exists for them); there `entry_name`
/// becomes the output symbol after sanitization only, a public
/// `chelis build`/tide contract left intact. The host lane covers
/// everything the entry-scoped lane declines (see [`EntryLaneDecline`]:
/// top-level globals, scalar/`grad` entries, host-only programs). The
/// sanitization rewrites `main` -> `chelis_main`, non-identifier
/// characters -> `_`, and prefixes a digit-leading or empty name with
/// `chelis_`, but guards NEITHER a libc collision (`free`, `malloc`, …)
/// NOR the runtime's own `chelis_*` namespace (`chelis_runtime.h`
/// declares `chelis_tensor_release`, `chelis_tuple_get`, …): a single-def program
/// whose def is named `free`, compiled via the free-form path, still
/// emits `void free(...)`, and an `entry_name` of `chelis_tensor_release` emits
/// verbatim. Both gaps are documented in `spec/11-ffi.md` §5a. The
/// entry-scoped metadata lane — which claims multi-def tensor programs
/// and single-def programs whose body needs host lowering (e.g. `concat`)
/// — instead uses the fixed, collision-free
/// [`EXECUTION_ENTRY_C_SYMBOL`] (`chelis_main`), because there
/// `entry_name` is a def *selector* the user must pass and so cannot
/// avoid such names.
fn execution_c_symbol(entry_name: Option<&str>) -> String {
    let Some(name) = entry_name else {
        return "chelis_main".to_string();
    };
    let mut sym: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sym.is_empty() || sym.as_bytes()[0].is_ascii_digit() {
        sym.insert_str(0, "chelis_");
    }
    if sym == "main" {
        sym = "chelis_main".to_string();
    }
    sym
}

/// Emitted C symbol for the entry-scoped metadata lane (#817/#818).
///
/// `entry_name` is now a def *selector* the user must supply, so the selected
/// name can be anything — `main` (collides with the reserved program entry),
/// `free`/`malloc` (collide with libc), or a name in the runtime's own
/// `chelis_*` namespace (`chelis_runtime.h` declares `chelis_tensor_release`,
/// `chelis_tuple_get`, …). To be collision-free against ALL of those, the
/// entry-scoped artifact ALWAYS emits the fixed symbol `chelis_main`. This is
/// safe because each artifact is scoped to exactly one entry def, so there is
/// exactly one emitted entry function per translation unit; the metadata
/// scoping is carried by the entry DAG, not the symbol name. The artifact's
/// `host_entry_name` carries `chelis_main` so the loader (`dlsym`) and header
/// stay consistent. Contrast the raw legacy path in [`execution_c_symbol`],
/// which is left unchanged.
const EXECUTION_ENTRY_C_SYMBOL: &str = "chelis_main";

/// The tensor-signature top-level defs of a program, in source order —
/// the candidate set the entry-scoped metadata lane can select from.
fn tensor_signature_defs(host_program: &chelis_ir::host::ConcreteHostProgram) -> Vec<&str> {
    host_program
        .functions
        .iter()
        .filter(|function| {
            chelis_ir::host::function_has_tensor_signature(host_program, &function.name)
        })
        .map(|function| function.name.as_str())
        .collect()
}

fn project_host_program_to_entry(
    program: &chelis_ir::host::ConcreteHostProgram,
    entry: &str,
) -> Option<chelis_ir::host::ConcreteHostProgram> {
    use chelis_ir::host::{
        ConcreteHostCallback, ConcreteHostExpr, ConcreteHostExprKind, HostCallbackKind,
    };

    fn collect_callback(
        callback: &ConcreteHostCallback,
        bound: &UnordSet<String>,
        out: &mut UnordSet<String>,
    ) {
        match &callback.kind {
            HostCallbackKind::Named { function, .. } => {
                if !bound.contains(function) {
                    out.insert(function.clone());
                }
            }
            HostCallbackKind::Inline { params, body } => {
                let mut scoped = bound.clone();
                scoped.extend(params.iter().map(|param| param.name.clone()));
                collect_expr(body, &scoped, out);
            }
        }
    }

    fn collect_expr(expr: &ConcreteHostExpr, bound: &UnordSet<String>, out: &mut UnordSet<String>) {
        match &expr.kind {
            ConcreteHostExprKind::Call { function, args, .. } => {
                if !bound.contains(function) {
                    out.insert(function.clone());
                }
                for arg in args {
                    collect_expr(arg, bound, out);
                }
            }
            ConcreteHostExprKind::Var(name, _) => {
                if !bound.contains(name) {
                    out.insert(name.clone());
                }
            }
            ConcreteHostExprKind::Builtin { args, .. }
            | ConcreteHostExprKind::TensorCall { args, .. } => {
                for arg in args {
                    collect_expr(arg, bound, out);
                }
            }
            ConcreteHostExprKind::List(items, _) | ConcreteHostExprKind::Tuple(items, _) => {
                for item in items {
                    collect_expr(item, bound, out);
                }
            }
            ConcreteHostExprKind::AdtConstruct { fields, .. } => {
                for field in fields {
                    collect_expr(field, bound, out);
                }
            }
            ConcreteHostExprKind::AdtFieldAccess { base, .. } => collect_expr(base, bound, out),
            ConcreteHostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                collect_expr(cond, bound, out);
                collect_expr(then_expr, bound, out);
                collect_expr(else_expr, bound, out);
            }
            ConcreteHostExprKind::MatchOption {
                scrutinee,
                bind_name,
                some_expr,
                none_expr,
                ..
            } => {
                collect_expr(scrutinee, bound, out);
                let mut some_scope = bound.clone();
                some_scope.insert(bind_name.clone());
                collect_expr(some_expr, &some_scope, out);
                collect_expr(none_expr, bound, out);
            }
            ConcreteHostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ..
            } => {
                collect_expr(scrutinee, bound, out);
                for arm in arms {
                    let mut arm_scope = bound.clone();
                    arm_scope.extend(arm.bindings.iter().map(|binding| binding.name.clone()));
                    collect_expr(&arm.expr, &arm_scope, out);
                }
                if let Some(default_expr) = default_expr {
                    collect_expr(default_expr, bound, out);
                }
            }
            ConcreteHostExprKind::Let { bindings, body, .. } => {
                let mut scoped = bound.clone();
                for binding in bindings {
                    collect_expr(&binding.value, &scoped, out);
                    scoped.insert(binding.name.clone());
                }
                collect_expr(body, &scoped, out);
            }
            ConcreteHostExprKind::Map { callback, list, .. }
            | ConcreteHostExprKind::Filter { callback, list, .. }
            | ConcreteHostExprKind::Partition { callback, list, .. }
            | ConcreteHostExprKind::FlatMap { callback, list, .. } => {
                collect_callback(callback, bound, out);
                collect_expr(list, bound, out);
            }
            ConcreteHostExprKind::Fold {
                callback,
                init,
                list,
                ..
            }
            | ConcreteHostExprKind::Scan {
                callback,
                init,
                list,
                ..
            } => {
                collect_callback(callback, bound, out);
                collect_expr(init, bound, out);
                collect_expr(list, bound, out);
            }
            ConcreteHostExprKind::WithSeed { seed, body, .. } => {
                collect_expr(seed, bound, out);
                collect_expr(body, bound, out);
            }
            ConcreteHostExprKind::Int(_)
            | ConcreteHostExprKind::Float(_)
            | ConcreteHostExprKind::Bool(_)
            | ConcreteHostExprKind::String(_)
            | ConcreteHostExprKind::Unit => {}
        }
    }

    let function_names: UnordSet<&str> = program
        .functions
        .iter()
        .map(|function| function.name.as_str())
        .collect();
    if !function_names.contains(entry) {
        return None;
    }

    let mut reachable = UnordSet::from([entry.to_string()]);
    let mut pending = vec![entry.to_string()];
    while let Some(name) = pending.pop() {
        let function = program
            .functions
            .iter()
            .find(|function| function.name == name)
            .expect("pending host function comes from the program");
        let bound = function
            .params
            .iter()
            .map(|param| param.name.clone())
            .collect();
        let mut referenced = UnordSet::new();
        collect_expr(&function.body, &bound, &mut referenced);
        for referenced_name in referenced.into_sorted() {
            if function_names.contains(referenced_name.as_str())
                && reachable.insert(referenced_name.clone())
            {
                pending.push(referenced_name);
            }
        }
    }

    let functions: Vec<_> = program
        .functions
        .iter()
        .filter(|function| reachable.contains(&function.name))
        .cloned()
        .collect();
    let summary_rejections = functions
        .iter()
        .flat_map(|function| function.summary_rejections.iter().cloned())
        .collect();
    Some(chelis_ir::host::ConcreteHostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions,
        summary_rejections,
    })
}

/// Resolve the entry def used to *scope* compiled-execution metadata
/// (`inputs`/`outputs`), independent of the emitted C symbol.
///
/// `host_only` is the whole-program host-backend requirement. It matters
/// only for the *unmatched-name* and *ambiguous-default* branches: on the
/// STRICT (callable) surface those are loud errors for a clean tensor
/// program (a typo/ambiguity the caller can fix); on the LEGACY (C-source)
/// surface they decline instead, preserving tide/cove's whole-program
/// contract (`entry_name` as output symbol; multi-def default compiles).
/// For a host-requiring program the selection is just the file-stem symbol
/// threaded to the host lane (e.g. a `grad` module whose `entry_name` is
/// the program name, not a def name — see issue #309), so we return `None`
/// and let the host lane own it rather than erroring on either surface.
///
/// - explicit `entry_name` naming a def → scope to it (#817);
/// - explicit `entry_name` naming no def, clean tensor program → typo'd
///   selector: error listing the tensor-signature entry defs (Fix 3);
/// - explicit `entry_name` naming no def, host-requiring program → `None`
///   (host lane owns it; keeps #309 file-stem `entry_name` working);
/// - no `entry_name`, clean tensor program → prefer a tensor-signature
///   `main`; else if EXACTLY ONE tensor-signature def, use it; else error
///   listing the candidates and asking for an explicit `entry_name` (Fix 3
///   — no more silent "last def wins");
/// - no `entry_name`, host-requiring program → the preferred tensor entry
///   (`main`, else last), preserving existing host-lane selection.
fn resolve_execution_entry<'a>(
    entry_name: Option<&'a str>,
    host_program: &'a chelis_ir::host::ConcreteHostProgram,
    host_only: bool,
    strictness: EntryStrictness,
) -> Result<Option<&'a str>> {
    match entry_name {
        Some(name) => {
            if host_program
                .functions
                .iter()
                .any(|function| function.name == name)
            {
                Ok(Some(name))
            } else if host_only {
                // The `entry_name` is not a def in this host-requiring
                // program: it is the file-stem output symbol (e.g. #309's
                // `grad` module). The host lane owns emission; don't error.
                Ok(None)
            } else if strictness == EntryStrictness::Legacy {
                // C-source surface: `entry_name` naming no def is the
                // output symbol (tide's documented contract), not a typo'd
                // selector. Decline the entry lane and let the legacy
                // whole-program path emit under `execution_c_symbol`.
                Ok(None)
            } else {
                let available = tensor_signature_defs(host_program).join(", ");
                Err(stage_error(
                    "compile",
                    format!(
                        "unknown entry_name `{name}`; this program's entry defs are: \
                         {available}. Pass one of these as entry_name."
                    ),
                    GeneralKind::CompileError,
                ))
            }
        }
        None => {
            if host_only {
                // Host-requiring program: keep the historical selection
                // (`main` else last tensor-signature def). The host lane,
                // not the entry-scoped kernel, will emit it.
                Ok(chelis_ir::host::preferred_tensor_entry_name(host_program))
            } else {
                // Clean tensor program: Fix 3 default-selection rule.
                let candidates = tensor_signature_defs(host_program);
                if candidates.contains(&"main") {
                    Ok(Some("main"))
                } else if candidates.len() == 1 {
                    Ok(Some(candidates[0]))
                } else if candidates.is_empty() {
                    Ok(None)
                } else if strictness == EntryStrictness::Legacy {
                    // C-source surface: no unambiguous entry means the
                    // legacy whole-program emission, exactly as before the
                    // entry lane existed (tide/cove compile multi-def
                    // programs like examples/linreg.ch with no entry_name).
                    Ok(None)
                } else {
                    Err(stage_error(
                        "compile",
                        format!(
                            "ambiguous entry: this program has multiple entry defs \
                             ({}) and none named `main`. Pass entry_name to select one.",
                            candidates.join(", ")
                        ),
                        GeneralKind::CompileError,
                    ))
                }
            }
        }
    }
}

/// Why the entry-scoped compiled-execution lane declined a compilation and
/// left it to the host lane.
///
/// The decline is not an error: the host lane legitimately owns these
/// program classes. But it used to be *silent* — the artifact came back
/// with an empty callable interface and downstream surfaces could only
/// guess why. The reason is recorded on
/// [`CompiledExecutionArtifact::entry_lane_decline`] so consumers
/// (chelis-python's callable-interface guard today, the #828/#830 tiering
/// work next) can report or query the actual cause.
///
/// The enum is `#[non_exhaustive]` on purpose: the #828/#830 tiering work
/// will add finer-grained decline reasons, and downstream crates (e.g.
/// chelis-python's error reporting) must keep a wildcard arm with a
/// generic fallback message rather than break on a new variant.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum EntryLaneDecline {
    /// The SOURCE program has top-level (non-`def`) value bindings. The
    /// host lane owns those: [`chelis_ir::host::lower_named_tensor_entry_dag`]
    /// seeds its scope with the entry's params only, so a free reference to
    /// a global would lower as a phantom `Load` — silently demoting the
    /// global to a required caller-supplied runtime input — and an
    /// independent global's computation would vanish from the emitted
    /// artifact entirely. Declining restores the pre-#817 routing for this
    /// program class. Detection is source-level
    /// ([`chelis_ir::host::program_has_top_level_value_bindings`]), not
    /// `host_program.globals`: lowering drops a DAG-lowerable, uncaptured
    /// scalar global (e.g. `glb = 2.0`) from `host.globals` entirely, and
    /// exactly that shape used to slip past the lowered check.
    HasGlobals,
    /// No entry def resolved: either `entry_name` named no def in a
    /// host-requiring program (a file-stem output symbol, see issue #309)
    /// or the program has no tensor-signature def to default to.
    NoEntryResolved,
    /// The selected entry def's signature is not tensor-in/tensor-out
    /// (e.g. a scalar `f32 -> f32` def selected by name).
    NotTensorSignature { entry: String },
    /// The selected entry uses a `grad`/`vmap` form; the host lane owns
    /// multi-root grad-tuple emission (issue #309).
    GradLike { entry: String },
    /// `lower_named_tensor_entry_dag` could not lower the entry body.
    LoweringFailed { entry: String },
    /// The entry lowered, but its DAG had no roots after dead-code
    /// elimination.
    EmptyAfterDce { entry: String },
    /// Belt-and-braces: the entry DAG demanded input labels beyond the
    /// def's declared parameters, which would surface as phantom manifest
    /// inputs. `HasGlobals` catches every known cause of this before
    /// lowering; this guards against unknown ones.
    InputsOutsideParams { entry: String, extra: Vec<String> },
}

/// Loud error for an entry-lane decline on the STRICT (callable) surface.
///
/// On [`compile_for_execution`], a decline that would fall through to the
/// legacy whole-DAG emission means the returned manifest would merge every
/// def's inputs/outputs (the #817 defect) while quietly ignoring any
/// `entry_name`. Each reason maps to an actionable error instead. The
/// LEGACY surface ([`compile`]) never calls this: there the whole-program
/// emission is the product contract.
fn strict_entry_decline_error(reason: EntryLaneDecline) -> CompilerError {
    match reason {
        EntryLaneDecline::HasGlobals => stage_error(
            "compile",
            "this program has top-level (non-def) value bindings, so the compiled-execution \
             lane cannot scope a callable to one entry: the whole-program artifact would \
             merge every def's params and outputs into the manifest (the chelis#817 class). \
             Move the bindings into the entry def, or run the program through `eval`, which \
             supports top-level bindings.",
            GeneralKind::CompileError,
        ),
        EntryLaneDecline::GradLike { entry } => {
            unsupported_stage_error(chelis_types::unsupported::Unsupported::new(
                chelis_types::unsupported::UnsupportedKind::Construct(format!(
                    "a `grad`/`vmap` transform entry (`{entry}`)"
                )),
                "the compiled-execution lane (compile_and_load), which emits a single \
                 entry-scoped tensor kernel and does not yet lower transform entries \
                 standalone",
                chelis_types::unsupported::Stage::Codegen("c"),
                // chelis#1138 owns this capability: the entry-scoped
                // compiled lane declines grad/vmap transform entries
                // standalone (the chelis#817/#818 entry-scoping did not
                // extend to transform entries). Filed and re-pointed from
                // the provisional chelis#613 citation after the PR #1037
                // delta red team adjudicated that #613 (the legacy
                // whole-program build lane, different predicate) does not
                // govern this decline.
                chelis_types::unimplemented_rejection!(
                    1138,
                    "run the transform through `eval`, or select a non-transform def \
                     with `entry_name=`"
                ),
            ))
        }
        other => stage_error(
            "compile",
            format!(
                "the compiled-execution lane declined this program ({other:?}) and refuses \
                 to fall back to whole-program codegen, which would merge every def's \
                 inputs/outputs into the callable manifest (the chelis#817 class). Use \
                 `eval` to run the program, or restructure the entry to a plain \
                 tensor-in/tensor-out def."
            ),
            GeneralKind::CompileError,
        ),
    }
}

/// The entry lane's claim/decline decision for one compilation.
enum EntryLaneOutcome<'a> {
    /// The lane claims the compilation: emit `dag` scoped to `entry`.
    Claim { entry: &'a str, dag: Dag },
    /// The lane declines; the host lane owns the compilation.
    Decline(EntryLaneDecline),
}

/// Decide whether the entry-scoped compiled-execution lane (#817/#818)
/// claims this compilation, and if not, why (Fix 2 of the #819 review
/// round: the predicate is named and queryable instead of inline and
/// silent).
///
/// The lane claims the compilation iff ALL hold:
///   - the SOURCE program has NO top-level (non-`def`) value bindings
///     (`HasGlobals` otherwise) — see the variant doc for the
///     phantom-input / vanishing-global failure modes this prevents, and
///     for why the check is source-level rather than
///     `host_program.globals`;
///   - a single entry def resolves via [`resolve_execution_entry`]
///     (`NoEntryResolved` otherwise; on the STRICT surface a typo'd or
///     ambiguous selector in a clean tensor program is a loud `Err`, while
///     the LEGACY surface declines to whole-program emission);
///   - that def is tensor-signature (`NotTensorSignature` otherwise) — a
///     scalar/record/ADT entry stays on the host lane;
///   - the def does NOT use a `grad`/`vmap` form (`GradLike` otherwise) —
///     the host lane owns multi-root grad-tuple emission (#309), which
///     `lower_named_tensor_entry_dag` can technically lower but must not
///     here. NOTE: unlike `grad`, a `vmap` entry does NOT force the host
///     backend, so its `GradLike` decline reaches the legacy whole-DAG
///     fallthrough (a loud error on the strict surface, whole-program
///     emission on the legacy one);
///   - the def lowers to a DAG (`LoweringFailed`) that is non-empty after
///     DCE (`EmptyAfterDce`);
///   - the DAG's `Load` labels are a subset of the def's declared param
///     names (`InputsOutsideParams`) — an internal consistency guard that
///     should be unreachable once `HasGlobals` has declined, but declining
///     beats emitting phantom inputs if it ever fires.
///
/// A DAG-lowerable host-runtime builtin such as `concat` is NOT excluded —
/// it lowers cleanly (that is the #818 fix).
fn entry_lane_decision<'a>(
    entry_name: Option<&'a str>,
    checked: &CheckedProgram,
    host_program: &'a chelis_ir::host::ConcreteHostProgram,
    host_only: bool,
    strictness: EntryStrictness,
) -> Result<EntryLaneOutcome<'a>> {
    use EntryLaneOutcome::Decline;

    // The `HasGlobals` decline is keyed on SOURCE-LEVEL top-level value
    // bindings, not on the lowered `host_program.globals`: lowering drops
    // a DAG-lowerable, uncaptured value binding from `host.globals` when
    // the program emits no host `main` (`skip_for_lowered`), so a scalar
    // global like `glb = 2.0` in a multi-def program was invisible to the
    // lowered check — the lane claimed the program and `glb`'s computation
    // vanished from the emitted C. ANY top-level value binding declines,
    // regardless of how lowering classified it. The lowered check stays as
    // belt-and-braces (it is a strict subset of the source-level one).
    if chelis_ir::host::program_has_top_level_value_bindings(checked)
        || !host_program.globals.is_empty()
    {
        return Ok(Decline(EntryLaneDecline::HasGlobals));
    }
    let Some(entry) = resolve_execution_entry(entry_name, host_program, host_only, strictness)?
    else {
        return Ok(Decline(EntryLaneDecline::NoEntryResolved));
    };
    if !chelis_ir::host::function_has_tensor_signature(host_program, entry) {
        return Ok(Decline(EntryLaneDecline::NotTensorSignature {
            entry: entry.to_string(),
        }));
    }
    if chelis_ir::host::named_entry_uses_grad_like(checked, entry) {
        return Ok(Decline(EntryLaneDecline::GradLike {
            entry: entry.to_string(),
        }));
    }
    let Some(dag) = chelis_ir::host::lower_named_tensor_entry_dag(checked, entry) else {
        return Ok(Decline(EntryLaneDecline::LoweringFailed {
            entry: entry.to_string(),
        }));
    };
    let dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    if dag.roots().is_empty() {
        return Ok(Decline(EntryLaneDecline::EmptyAfterDce {
            entry: entry.to_string(),
        }));
    }
    let declared_params: UnordSet<&str> = host_program
        .functions
        .iter()
        .find(|function| function.name == entry)
        .map(|function| function.params.iter().map(|p| p.name.as_str()).collect())
        .unwrap_or_default();
    let mut extra: Vec<String> = Vec::new();
    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op
            && !declared_params.contains(name.as_str())
            && !extra.iter().any(|seen| seen == name.as_str())
        {
            extra.push(name.as_str().to_string());
        }
    }
    if !extra.is_empty() {
        return Ok(Decline(EntryLaneDecline::InputsOutsideParams {
            entry: entry.to_string(),
            extra,
        }));
    }
    Ok(EntryLaneOutcome::Claim { entry, dag })
}

/// Compile to a callable execution artifact: the CALLABLE surface
/// (python's `compile_and_load`). STRICT entry integrity: an unknown
/// `entry_name`, an ambiguous default on a multi-def program, or an
/// entry-lane decline that would otherwise fall through to merged
/// whole-program metadata (top-level value bindings, a `grad`/`vmap`
/// transform entry) is a loud error here, never a silently merged
/// manifest (#817) and never a debug assert. The C-source surface with
/// the legacy whole-program contract is [`compile`].
pub fn compile_for_execution(request: CompileRequest) -> Result<CompiledExecutionArtifact> {
    compile_for_execution_impl(request, EntryStrictness::Strict)
}

/// Opt-in observation of the same strict compilation as [`compile_for_execution`].
/// The callback sees the actual immutable ownership-verified emission payload.
/// The separate initial host snapshot is unverified and may contain functions
/// pruned from that payload; see [`crate::emission_observer::EmissionObservation`].
/// An observation is not success: later code generation or artifact construction
/// may still fail. No observer is installed globally or used by ordinary calls.
#[cfg(feature = "emission-observer")]
pub fn compile_for_execution_with_observer(
    request: CompileRequest,
    observer: &mut dyn FnMut(crate::emission_observer::EmissionObservation<'_>),
) -> Result<CompiledExecutionArtifact> {
    let compiled = compile_source_for_target(
        request.source_kind,
        &request.source,
        manifest_target(request.target),
    )?;
    execution_artifact_from_compiled_observed(
        compiled,
        request.target,
        request.entry_name.as_deref(),
        EntryStrictness::Strict,
        Some(observer),
    )
}

fn compile_for_execution_impl(
    request: CompileRequest,
    strictness: EntryStrictness,
) -> Result<CompiledExecutionArtifact> {
    let compiled = compile_source_for_target(
        request.source_kind,
        &request.source,
        manifest_target(request.target),
    )?;
    execution_artifact_from_compiled(
        compiled,
        request.target,
        request.entry_name.as_deref(),
        strictness,
    )
}

/// Compile `new_source` against an existing reef [`CompiledContext`] into a
/// callable execution artifact — the reef-aware analogue of
/// [`compile_for_execution`]. Library defs that `new_source` references
/// (e.g. a `Shoals.Pricing` import) resolve against the pre-linked context
/// instead of failing with `unbound variable`. The post-compile lowering /
/// codegen (including the entry-scoped metadata lane, #817/#818) is shared
/// with the monolithic path via [`execution_artifact_from_compiled`]. See
/// issue #816.
///
/// One deliberate divergence from the monolithic lane: this path is
/// entry-scoped from the start (`resolve_in_context_entry` slices the
/// new-code DAG to the selected root), so a top-level (non-`def`) value
/// binding in `new_source` does NOT decline compilation the way the
/// monolithic `entry_lane_decision` does (`HasGlobals`). The artifact is
/// scoped to the entry; an unreferenced sibling global's computation is
/// simply not part of it. Use [`eval_in_context`] for whole-program
/// semantics. (Documented in `spec/11-ffi.md` and the python README.)
pub fn compile_for_execution_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
    target: CompileTarget,
    entry_name: Option<&str>,
) -> Result<CompiledExecutionArtifact> {
    let compiled = compile_new_source_in_context(context, new_source, manifest_target(target))?;
    // The in-context lane is a callable surface: strict entry integrity.
    execution_artifact_from_compiled(compiled, target, entry_name, EntryStrictness::Strict)
}

/// Resolve and scope the entry for the in-context compiled path (#816).
/// (The monolithic path uses [`entry_lane_decision`] instead, which re-lowers
/// the named def from `compiled.checked` — that works only because the
/// monolithic `compiled.checked` holds the whole program. In-context it holds
/// new code only, so re-lowering a def that calls a library function would
/// fail — the reviewer-flagged Step-1 trap.)
///
/// In-context, a clean tensor entry lowers straight into `compiled.dag` as a
/// DAG root (its library calls already inlined by `compile_new_source_in_context`),
/// NOT as a host-program function — so it is selected by new-code tensor-root
/// name, and `compiled.dag` is sliced to that root and DCE'd. Scoping to the
/// entry's reachable subgraph is what keeps the metadata correct (only the
/// entry's inputs, #817) and keeps the dim/precision checks off unrelated
/// library helper nodes (which may carry polymorphic symbolic dims).
///
/// - explicit `entry_name` → the root matching it EXACTLY (the same rule as
///   the monolithic `resolve_execution_entry`); no match on a non-empty root
///   set errors listing the tensor entries. All selectable roots are the
///   compiled source's own defs with their bare names: library defs are
///   callable from the entry but are not themselves selectable entries;
/// - no `entry_name`, a tensor root named `main` → that root (the same `main`
///   preference `resolve_execution_entry` applies, so a file behaves the same
///   inside and outside a reef project);
/// - no `entry_name`, no `main`, exactly one tensor root → that root;
/// - no `entry_name`, no `main`, several tensor roots → error asking for
///   `entry_name` (no silent "merge every def", cf. #817);
/// - no tensor roots at all → `Ok(None)`; the caller then rejects with the
///   scalar/host-only guidance (a scalar or host-only entry).
///
/// Name-to-node resolution goes through the #1013 pipeline's typed
/// [`crate::pipeline::NamedRoots`] map, whose construction
/// (`NamedRoots::aligned` in `finish_lowering`) verifies the name/root
/// correspondence and rejects any count mismatch as a typed
/// `PipelineRejection::RootCount` before a `CompiledSource` can exist. An
/// earlier revision indexed `dag.roots()` positionally by the name's index
/// in `tensor_root_names` and carried a hand-rolled zero-names/rootful-DAG
/// invariant guard; both are superseded by the typed map (the mismatch
/// state is unrepresentable). `tensor_root_names` is still the SELECTABLE
/// SET and the source-order listing for error messages: its entries are
/// exactly the new-code defs, so linker-mangled library roots are never
/// selectable by construction.
fn resolve_in_context_entry<'a>(
    compiled: &'a CompiledSource,
    entry_name: Option<&str>,
) -> Result<Option<(&'a str, Dag)>> {
    let roots = &compiled.tensor_root_names;
    let list_entries = || {
        roots
            .iter()
            .map(|root| root.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let selected: &crate::pipeline::IrName = match entry_name {
        Some(name) => {
            // Exact match ONLY, mirroring the monolithic strict lane's
            // `resolve_execution_entry`. An earlier revision fell back to a
            // linker-mangled root whose name ends with `__<name>`, on the
            // theory that library-originated entries come back mangled. They
            // never do: `tensor_root_names` holds NEW-CODE roots only (the
            // pipeline slices the composed roots past the library count), and
            // new-code decl names are copied verbatim (`rewrite_eval_decl`)
            // with linker-format user decls hard-rejected. The suffix arm's
            // only reachable effect was silently compiling a DIFFERENT def:
            // an ordinary double-underscore def like `compute__solve` (legal
            // source) satisfied `entry_name = "solve"` with no diagnostic,
            // which is the #817 wrong-entry class this lane exists to close.
            if let Some(root) = roots.iter().find(|root| root.as_str() == name) {
                root
            } else if roots.is_empty() {
                return Ok(None);
            } else {
                return Err(stage_error(
                    "compile",
                    format!(
                        "unknown entry_name `{name}`; this program's tensor entries are: \
                         {}. Pass one of these as entry_name.",
                        list_entries()
                    ),
                    GeneralKind::CompileError,
                ));
            }
        }
        None => {
            // Default selection: prefer a tensor entry literally named `main`,
            // mirroring the monolithic `resolve_execution_entry`. Without this
            // a multi-def in-context file with a `main` erroneously reported
            // "ambiguous" while the same file compiled fine outside a project.
            if let Some(root) = roots.iter().find(|root| root.as_str() == "main") {
                root
            } else {
                match roots.as_slice() {
                    [] => return Ok(None),
                    [only] => only,
                    _ => {
                        return Err(stage_error(
                            "compile",
                            format!(
                                "ambiguous entry: this program has multiple tensor entries ({}) \
                                 and none named `main`. Pass entry_name to select one.",
                                list_entries()
                            ),
                            GeneralKind::CompileError,
                        ));
                    }
                }
            }
        }
    };
    let root = *compiled.named_roots.get(selected).ok_or_else(|| {
        stage_error(
            "compile",
            format!(
                "internal: in-context tensor entry `{selected}` has no node in the \
                 pipeline's named-root map; please report it."
            ),
            GeneralKind::CompileError,
        )
    })?;
    let mut scoped = compiled.dag.clone();
    scoped.set_roots(vec![root]);
    let scoped = chelis_ir::optimize::dead_code_eliminate(&scoped);
    Ok(Some((selected.as_str(), scoped)))
}

/// The branded HIP reef-context rejection (chelis#829), shared by the Hip
/// codegen arm and by chelis-python's EARLY guard in
/// `run_compile_and_load_job` (#822 review round 3, finding 3): the
/// rejection depends on nothing but the target and the presence of a reef
/// root, so callers reject BEFORE paying the context compile (the first
/// build of a real project is tens of seconds to minutes). Routed through
/// `Unsupported` so it carries the section C2 `unsupported:` brand and the
/// `unsupported_feature` kind; the #730 sweeps match on both, and a
/// `compile_error` here would read as internal desync rather than a
/// not-yet-implemented capability.
pub fn reef_context_hip_unsupported_error() -> CompilerError {
    unsupported_stage_error(chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Construct(
            "reef-context compilation (a `project_root=` with \
             reef-declared imports)"
                .to_string(),
        ),
        "the HIP backend, which does not yet apply the entry-scoped DAG \
         selection the C path uses and would merge every reef-linked \
         def's inputs/outputs into a single kernel instead of compiling \
         the requested entry",
        chelis_types::unsupported::Stage::Codegen("hip"),
        chelis_types::unimplemented_rejection!(
            829,
            "compile the entry with `target=\"c\"`, or run it through `eval`, \
             until HIP reef-context support lands"
        ),
    ))
}

/// The post-`compile_source` body shared by [`compile_for_execution`] and
/// [`compile_for_execution_in_context`]. It takes an already-`CompiledSource`
/// (monolithic OR in-context) and lowers + codegens the execution artifact,
/// including the entry-scoped metadata lane. Splitting this out is purely a
/// refactor: the monolithic path's behavior is byte-for-byte identical to the
/// pre-split inline body (the `execution_artifact_metadata` suite locks it).
fn execution_artifact_from_compiled(
    compiled: CompiledSource,
    target: CompileTarget,
    entry_name: Option<&str>,
    strictness: EntryStrictness,
) -> Result<CompiledExecutionArtifact> {
    execution_artifact_from_compiled_observed(
        compiled,
        target,
        entry_name,
        strictness,
        #[cfg(feature = "emission-observer")]
        None,
    )
}

fn execution_artifact_from_compiled_observed(
    compiled: CompiledSource,
    target: CompileTarget,
    entry_name: Option<&str>,
    strictness: EntryStrictness,
    #[cfg(feature = "emission-observer")] mut observer: Option<
        crate::emission_observer::Observer<'_>,
    >,
) -> Result<CompiledExecutionArtifact> {
    let build_target = BuildTarget::from(target);
    reject_host_only_builtins_before_host_lowering(compiled.checked(), build_target)?;
    let mut host_compiled = chelis_ir::host::try_lower_manifested_program(&compiled.program)
        .map_err(|diagnostic| {
            stage_error_with_span(
                "lower",
                diagnostic.to_string(),
                GeneralKind::LowerError,
                deep_span_to_diagnostic(diagnostic.span),
            )
        })?;
    // Preserve the actual lowering, not a second independently lowered program.
    // Ordinary compilation does not clone it, even with the feature enabled.
    #[cfg(feature = "emission-observer")]
    let observed_host = observer.as_ref().and_then(|_| host_compiled.host.clone());
    let func_name = execution_c_symbol(entry_name);

    // Reject host-runtime-only builtins early for any compiled-backend
    // target so both public compiler APIs preserve the owning builtin's
    // specific diagnostic. The fallible emitter independently rejects an
    // unknown compiled-lane builtin; this gate improves ordering and context,
    // and is not the correctness boundary. See spec/05-risc-primitives.md
    // §3.6 and spec/design/loud_unsupported.md §C6.3.
    if let Some(host_program) = host_compiled.host.as_ref() {
        reject_host_only_builtins(host_program, build_target)?;
        reject_eval_only_builtins(host_program, build_target)?;
    }

    match target {
        CompileTarget::C => {
            let host_only = host_compiled
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);

            // Entry-scoped compiled-execution metadata (#817, #818).
            //
            // The legacy path below compiles the WHOLE program: every
            // lowerable top-level `def`'s body becomes a DAG root and every
            // def's params become `Load`s, so `execution_input_specs` reports
            // all defs' inputs merged (#817). And when the entry's body uses a
            // syntactic host-runtime builtin such as `concat`, the def is
            // excluded from the pure DAG entirely, `roots()` is empty, and the
            // host-lane early-return hands back an EMPTY manifest (#818).
            //
            // Both are fixed by lowering the single resolved ENTRY def into a
            // standalone DAG and taking its metadata. The gate is scoped to
            // the ENTRY, not the whole program: the original `!host_only`
            // whole-program gate meant any host-flavored SIBLING def (or a
            // helper the entry calls that itself uses `concat`) disabled the
            // lane even when the requested entry lowers cleanly — the #817/#818
            // regression this PR round fixes.
            //
            // The claim/decline predicate lives in `entry_lane_decision` (see
            // its doc for the full rule set). Anything the entry lane declines
            // falls through to the host lane below, carrying the decline
            // reason on the artifact so downstream surfaces can report it
            // instead of guessing from an empty manifest. For a single-def
            // tensor program the scoped DAG equals the whole-program DAG, so
            // this is a no-op there.
            let mut entry_lane_decline = None;
            let scoped_entry = if compiled.library_runtime.is_some() {
                // In-context (#816): a clean tensor entry lowers straight into
                // `compiled.dag` as a DAG root — it is NOT a host-program
                // function, so `entry_lane_decision` (which reads the host
                // program) can't see it. Resolve against the new-code tensor
                // roots instead and slice `compiled.dag` to the entry's root,
                // so the metadata / dim checks run on the entry's reachable
                // subgraph only (not unreachable library helper nodes, which
                // may carry polymorphic symbolic dims). NOTE: this branch does
                // not consult `strictness` — `resolve_in_context_entry` is
                // strict by construction (exact-name errors, ambiguity errors,
                // no decline fallthrough), and no LEGACY caller routes
                // in-context today. A future `compile()`-in-context surface
                // must decide its own policy here rather than inherit this.
                //
                // Invariant guard, symmetric to Fix A in
                // `resolve_in_context_entry` (#822 review): a RESOLVED entry
                // whose scoped DAG comes back rootless (set_roots + DCE
                // emptied it) must fail loudly, not be silently dropped. The
                // previous `.filter(!roots.is_empty())` did exactly that
                // silent drop: with the claim gone, execution skips the
                // rootless in-context reject below (`compiled.dag.roots()` is
                // still non-empty — only the SCOPED dag is rootless) and
                // falls through to whole-DAG codegen with every new-code
                // root's inputs merged — the same #817 class Fix A closes,
                // via a different door. Should be unreachable: the root the
                // entry was scoped to IS a root, so DCE keeps it. The
                // legitimate paths are untouched: no entry resolved with a
                // rootless DAG still reaches the friendly scalar/host-only
                // reject below.
                match resolve_in_context_entry(&compiled, entry_name)? {
                    Some((entry, dag)) if dag.roots().is_empty() => {
                        return Err(stage_error(
                            "compile",
                            format!(
                                "internal: in-context entry `{entry}` resolved but its \
                                 entry-scoped DAG has no roots after scoping/DCE; refusing to \
                                 fall through to whole-program codegen, which would merge every \
                                 root's inputs (#817). This indicates a root-scoping vs DCE \
                                 mismatch; please report it."
                            ),
                            GeneralKind::CompileError,
                        ));
                    }
                    resolved => resolved,
                }
            } else if let Some(host_program) = host_compiled.host.as_ref() {
                match entry_lane_decision(
                    entry_name,
                    compiled.checked(),
                    host_program,
                    host_only,
                    strictness,
                )? {
                    EntryLaneOutcome::Claim { entry, dag } => Some((entry, dag)),
                    EntryLaneOutcome::Decline(reason) => {
                        entry_lane_decline = Some(reason);
                        None
                    }
                }
            } else {
                None
            };

            if let Some((_entry, entry_dag)) = scoped_entry {
                // Fix 2: the entry-scoped symbol is the fixed, collision-free
                // `chelis_main` so a def named `main`/`free`/`chelis_*` links.
                let entry_symbol = EXECUTION_ENTRY_C_SYMBOL;
                reject_unsupported_effect_ops(&entry_dag, BuildTarget::C)?;
                reject_symbolic_windowed_reduce(&entry_dag, BuildTarget::C)?;
                reject_unsupported_reduce_window_precision(&entry_dag, BuildTarget::C)?;
                reject_unsized_named_dims(&entry_dag, "c")?;
                let specialized =
                    chelis_ir::specialize::specialize_for_exact_arithmetic(&entry_dag);
                let fused = chelis_ir::fuse::fuse(&specialized);
                let options = chelis_backend_c::CodegenOptions {
                    use_blas: true,
                    ..chelis_backend_c::CodegenOptions::default()
                };
                let selected = chelis_backend_c::prepare_dag_for_codegen(fused, options);
                let verified = chelis_ir::ownership::verify_ownership(
                    chelis_ir::ownership::lower_dag_ownership(selected).map_err(|error| {
                        stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                    })?,
                )
                .map_err(|error| {
                    stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                })?;
                #[cfg(feature = "emission-observer")]
                crate::emission_observer::observe(
                    &mut observer,
                    &compiled.program,
                    observed_host.as_ref(),
                    crate::emission_observer::SelectedEmission::Dag {
                        unfused: &entry_dag,
                        selected: verified.emission(),
                    },
                );
                let result =
                    chelis_backend_c::codegen_with_options(verified, entry_symbol, options)
                        .map_err(unsupported_stage_error)?;
                return Ok(compiled_execution_artifact(
                    entry_symbol,
                    None,
                    compile_result_c(target, entry_symbol, &result),
                    manifest_result(&compiled.program),
                    execution_input_specs(&entry_dag, &result.input_labels)?,
                    execution_output_specs(&entry_dag, &result.output_labels)?,
                    result.symbolic_dims,
                ));
            }

            // In-context reef path (#816): a clean tensor entry lowers straight
            // into `compiled.dag` (its roots are the new-code tensor roots), so
            // the whole-program fallthrough below codegens it correctly — the
            // entry lane above declines because such a def is a DAG root, not a
            // host-program function. But when `compiled.dag` has NO roots, the
            // new source produced no tensor kernel at all: a scalar-signature
            // entry (`def main(s: f32, ...) -> f32`), a host-only entry, or one
            // whose body needs the host runtime. The monolithic fallthrough
            // would then codegen a root-less DAG into unbuildable C. Reject
            // cleanly with actionable guidance instead. `eval`/`eval_in_context`
            // still run these programs. (Gated on `library_runtime` so the
            // monolithic path — which routes these to the host lane below — is
            // byte-for-byte unchanged.)
            if compiled.library_runtime.is_some() && compiled.dag.roots().is_empty() {
                // Name the explicitly-requested entry when the user passed one:
                // "the selected entry" is opaque if they asked for `entry_name=foo`
                // and foo has no tensor form (review round 2).
                let selected = match entry_name {
                    Some(name) => format!("the selected entry `{name}`"),
                    None => "the selected entry".to_string(),
                };
                return Err(stage_error(
                    "compile",
                    format!(
                        "{selected} has no callable tensor-kernel form: it is scalar-\
                         signature (e.g. `def main(s: f32, ...) -> f32`), host-only (top-level \
                         bindings/globals, string/record/effect ops), or otherwise does not \
                         lower to a tensor entry. Give it a tensor-in/tensor-out signature by \
                         wrapping scalars as rank-1 tensors (`tensor[1, f32]`), or use `eval` to \
                         run it (`eval` supports scalar and host-only programs)."
                    ),
                    GeneralKind::CompileError,
                ));
            }

            if let Some(host_program) = host_compiled.host.as_ref()
                && (host_only || compiled.dag.roots().is_empty())
            {
                let projected_host_program = match (&entry_lane_decline, strictness) {
                    (
                        Some(EntryLaneDecline::NotTensorSignature { entry }),
                        EntryStrictness::Strict,
                    ) => Some(
                        project_host_program_to_entry(host_program, entry).ok_or_else(|| {
                            stage_error(
                                "compile",
                                format!(
                                    "internal: selected scalar entry `{entry}` disappeared \
                                     before host-program projection; please report it"
                                ),
                                GeneralKind::CompileError,
                            )
                        })?,
                    ),
                    _ => None,
                };
                let host_program = projected_host_program.as_ref().unwrap_or(host_program);
                reject_unsupported_effect_ops_in_host_program(host_program, BuildTarget::C)?;
                reject_unsupported_windowed_reductions_in_host_program(
                    host_program,
                    BuildTarget::C,
                )?;
                let scalar_only_globals = host_program
                    .globals
                    .iter()
                    .all(|global| matches!(&global.ty, chelis_ir::ConcreteHostType::Scalar(_)));
                let selected = projected_host_program
                    .unwrap_or_else(|| host_compiled.host.take().expect("host branch selected"));
                let selected = chelis_backend_c::prepare_host_program_for_codegen(selected)
                    .map_err(unsupported_stage_error)?;
                let verified = chelis_ir::ownership::verify_ownership(
                    chelis_ir::ownership::lower_host_ownership(&compiled.program, selected)
                        .map_err(|error| {
                            stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                        })?,
                )
                .map_err(|error| {
                    stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                })?;
                #[cfg(feature = "emission-observer")]
                crate::emission_observer::observe(
                    &mut observer,
                    &compiled.program,
                    observed_host.as_ref(),
                    crate::emission_observer::SelectedEmission::Host(verified.emission()),
                );
                let result = chelis_backend_c::codegen_host_program(&verified, &func_name)
                    .map_err(unsupported_stage_error)?;
                // Preserve a more specific host-emitter rejection (for
                // example the function-value ABI) when one exists. The
                // scalar-global shape is the one strict #817 fallback that
                // remains unsafe here: source-level detection sees the global,
                // while host lowering has no global product to expose. Real
                // host globals and GradLike entries retain their established
                // host-lane artifact with an explicit decline reason.
                if strictness == EntryStrictness::Strict
                    && matches!(entry_lane_decline, Some(EntryLaneDecline::HasGlobals))
                    && scalar_only_globals
                {
                    return Err(strict_entry_decline_error(EntryLaneDecline::HasGlobals));
                }
                let mut artifact = compiled_execution_artifact(
                    &func_name,
                    None,
                    compile_result_c(target, &func_name, &result),
                    manifest_result(&compiled.program),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                );
                artifact.entry_lane_decline = entry_lane_decline;
                return Ok(artifact);
            }
            reject_unsupported_effect_ops(&compiled.dag, BuildTarget::C)?;
            reject_symbolic_windowed_reduce(&compiled.dag, BuildTarget::C)?;
            reject_unsupported_reduce_window_precision(&compiled.dag, BuildTarget::C)?;
            reject_unsized_named_dims(&compiled.dag, "c")?;
            let specialized = chelis_ir::specialize::specialize_for_exact_arithmetic(&compiled.dag);
            let fused = chelis_ir::fuse::fuse(&specialized);
            let options = chelis_backend_c::CodegenOptions {
                use_blas: true,
                ..chelis_backend_c::CodegenOptions::default()
            };
            let selected = chelis_backend_c::prepare_dag_for_codegen(fused, options);
            let verified = chelis_ir::ownership::verify_ownership(
                chelis_ir::ownership::lower_dag_ownership(selected).map_err(|error| {
                    stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                })?,
            )
            .map_err(|error| {
                stage_error("ownership", error.to_string(), GeneralKind::CompileError)
            })?;
            #[cfg(feature = "emission-observer")]
            crate::emission_observer::observe(
                &mut observer,
                &compiled.program,
                observed_host.as_ref(),
                crate::emission_observer::SelectedEmission::Dag {
                    unfused: &compiled.dag,
                    selected: verified.emission(),
                },
            );
            let result = chelis_backend_c::codegen_with_options(verified, &func_name, options)
                .map_err(unsupported_stage_error)?;
            let mut artifact = compiled_execution_artifact(
                &func_name,
                None,
                compile_result_c(target, &func_name, &result),
                manifest_result(&compiled.program),
                execution_input_specs(&compiled.dag, &result.input_labels)?,
                execution_output_specs(&compiled.dag, &result.output_labels)?,
                result.symbolic_dims,
            );
            // On this legacy whole-DAG path a `Some(decline)` co-occurs with
            // a real (whole-program, merged) manifest — see the
            // `entry_lane_decline` field doc. Every decline reason that
            // leaves the whole-program DAG rooted can get here, including
            // `GradLike`: a `vmap` entry declines the lane but, unlike
            // `grad`, does NOT force the host backend, so it reaches this
            // path (an earlier revision asserted it could not, and a vmap
            // entry panicked every debug-built caller). On the STRICT
            // (callable) surface a merged manifest is the #817 defect, so
            // any decline here is a loud error; on the LEGACY (C-source)
            // surface the whole-program emission is the product contract
            // and the decline reason rides along on the artifact.
            if strictness == EntryStrictness::Strict
                && let Some(reason) = entry_lane_decline
            {
                return Err(strict_entry_decline_error(reason));
            }
            artifact.entry_lane_decline = entry_lane_decline;
            Ok(artifact)
        }
        CompileTarget::Hip => {
            // Reef-context (#816) HIP is not supported: the entry-scoped DAG
            // selection the C arm applies (`resolve_in_context_entry`, gated on
            // `library_runtime.is_some()`) is not implemented for HIP, which
            // still clones the whole composed DAG and treats `entry_name` only
            // as the emitted symbol. Compiling a multi-def reef context to HIP
            // would therefore merge every reef-linked def's inputs/outputs into
            // one kernel or ignore the requested entry — the #817/#818 class the
            // C path fixes. Reject explicitly with actionable guidance rather
            // than silently emit a mis-scoped kernel (an undocumented false
            // promise, since the public API accepts `CompileTarget` and Python
            // exposes `target="hip"`). Tracked as chelis#829 (apply the same
            // entry-scoped selection to HIP, then lift this reject).
            if compiled.library_runtime.is_some() {
                return Err(reef_context_hip_unsupported_error());
            }
            let host_requires_host_backend = host_compiled
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            let preferred_entry = host_compiled
                .host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name);
            let preferred_entry_is_host = match preferred_entry {
                Some(name) => {
                    crate::target_capability::hip_entry_lane(compiled.checked(), name)
                        .map_err(unsupported_stage_error)?
                        == chelis_types::types::Lane::Host
                }
                None => false,
            };
            let preferred_entry_dag = preferred_entry.and_then(|name| {
                chelis_ir::host::lower_named_tensor_entry_dag(compiled.checked(), name)
            });
            let has_host_roots = compiled
                .manifest()
                .entries
                .iter()
                .any(|entry| entry.lane == chelis_types::types::Lane::Host);
            if (has_host_roots
                || preferred_entry_is_host
                || (compiled.dag.roots().is_empty()
                    && preferred_entry_dag.is_none()
                    && host_requires_host_backend))
                && let Some(host_program) = host_compiled.host.as_ref()
            {
                reject_unsupported_effect_ops_in_host_program(host_program, BuildTarget::Hip)?;
                reject_unsupported_hip_ops_in_host_program(host_program)?;
                let selected = host_compiled.host.take().expect("host branch selected");
                let selected = chelis_backend_c::prepare_host_program_for_codegen(selected)
                    .map_err(unsupported_stage_error)?;
                let verified = chelis_ir::ownership::verify_ownership(
                    chelis_ir::ownership::lower_host_ownership(&compiled.program, selected)
                        .map_err(|error| {
                            stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                        })?,
                )
                .map_err(|error| {
                    stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                })?;
                #[cfg(feature = "emission-observer")]
                crate::emission_observer::observe(
                    &mut observer,
                    &compiled.program,
                    observed_host.as_ref(),
                    crate::emission_observer::SelectedEmission::Host(verified.emission()),
                );
                let result = chelis_backend_c::codegen_host_program(&verified, &func_name)
                    .map_err(unsupported_stage_error)?;
                return Ok(compiled_execution_artifact(
                    &func_name,
                    None,
                    compile_result_hip_host(target, &func_name, &result),
                    manifest_result(&compiled.program),
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
            reject_unsupported_effect_ops(&hip_dag, BuildTarget::Hip)?;
            reject_unsized_named_dims(&hip_dag, "hip")?;
            let specialized = chelis_ir::specialize::specialize_for_blas(&hip_dag);
            reject_unsupported_hip_ops(&specialized)?;
            let fused = chelis_ir::fuse::fuse(&specialized);
            let selected = chelis_backend_hip::prepare_dag_for_codegen(fused);
            let verified = chelis_ir::ownership::verify_ownership(
                chelis_ir::ownership::lower_dag_ownership(selected).map_err(|error| {
                    stage_error("ownership", error.to_string(), GeneralKind::CompileError)
                })?,
            )
            .map_err(|error| {
                stage_error("ownership", error.to_string(), GeneralKind::CompileError)
            })?;
            #[cfg(feature = "emission-observer")]
            crate::emission_observer::observe(
                &mut observer,
                &compiled.program,
                observed_host.as_ref(),
                crate::emission_observer::SelectedEmission::Dag {
                    unfused: &hip_dag,
                    selected: verified.emission(),
                },
            );
            let result = chelis_backend_hip::codegen_hip(verified, &func_name)
                .map_err(unsupported_stage_error)?;
            Ok(compiled_execution_artifact(
                &func_name,
                Some(format!("{func_name}_device")),
                compile_result_hip(target, &func_name, &result)?,
                manifest_result(&compiled.program),
                execution_input_specs(&hip_dag, &result.input_labels)?,
                execution_output_specs(&hip_dag, &result.output_labels)?,
                result.symbolic_dims,
            ))
        }
    }
}

pub fn eval(request: EvalRequest) -> Result<EvalResult> {
    eval_for_target(request, Target::Eval)
}

/// Evaluate through a manifest computed for `target`. Execution still uses
/// the local evaluator, but lane assignment, required inputs, and surfaced
/// roots are exactly the contract the requested backend would consume.
pub fn eval_for_target(request: EvalRequest, target: Target) -> Result<EvalResult> {
    let compiled = compile_source_for_target(request.source_kind, &request.source, target)?;
    eval_compiled(&compiled, request.bindings, None)
}

pub fn eval_selected(request: EvalRequest, selected_root_names: &[String]) -> Result<EvalResult> {
    eval_selected_for_target(request, selected_root_names, Target::Eval)
}

pub fn eval_selected_for_target(
    request: EvalRequest,
    selected_root_names: &[String],
    target: Target,
) -> Result<EvalResult> {
    let compiled = compile_source_for_target(request.source_kind, &request.source, target)?;
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
    target: Target,
) -> Result<CompiledSource> {
    // RFC v5 (RT-1 F2 bypass): the new entry decls are reef-rewritten
    // (`rewrite_entry_decls_with_reef_graph` mangles them) before this
    // checks them against the linked library, so the linker name format
    // is expected here.
    let _linked = chelis_types::install_linked_program_guard();
    // Phase G inputs are always Surf — `compile_reef_context` already
    // resolved the package's library decls; the new source is whatever
    // the user typed into a `chelis eval --file` / `chelis test` worker /
    // `chelis check` call.
    // chelis#930: same phase-boundary polling as `compile_source_scoped`. This
    // is the primary route for reef packages (`chelis eval --file` inside a
    // package, `chelis test` workers), so it is the path that matters most for
    // the library-heavy workloads in chelis#828.
    bail_if_cancelled("parse")?;
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
            .map_err(|err| stage_error("reef", err, GeneralKind::ReefError))?;
    compile_rewritten_decls_in_context(context, &rewritten, target)
}

/// Compile declarations whose module identity and imports have already been
/// resolved by Reef. Keeping this boundary separate prevents an isolated
/// multi-entry batch from being flattened back into the synthetic eval module
/// and rewritten a second time.
fn compile_rewritten_decls_in_context(
    context: &crate::context::CompiledContext,
    rewritten: &[Decl],
    target: Target,
) -> Result<CompiledSource> {
    let _linked = chelis_types::install_linked_program_guard();
    bail_if_cancelled("desugar")?;
    let prepared = crate::pipeline::prepare_surf_decls(rewritten, None).map_err(|error| {
        pipeline_rejection_to_compiler_error(crate::pipeline::PipelineRejection::Preparation(error))
    })?;
    bail_if_cancelled("check")?;

    // Type-check the new code once against the library context, then run the
    // shared effect and linearity transitions.
    let analysis =
        crate::pipeline::analyze_prepared_with_library(prepared, context.checked_library())
            .map_err(|report| {
                crate::compiler::check_errors_to_compiler_error("check", &report.errors)
            })
            .map_err(|error| cancelled_or("check", error))?;
    bail_if_cancelled("effects")?;
    let checked = crate::pipeline::complete_context_checks(analysis)
        .map_err(|rejection| pipeline_rejection_to_compiler_error(rejection.into()))
        .map_err(|error| cancelled_or("effects", error))?;
    bail_if_cancelled("linearity")?;
    bail_if_cancelled("lower")?;
    let lowered = crate::pipeline::lower_checked_with_context(
        checked,
        &context.library_dag,
        crate::pipeline::LoweringMode::AllowHostOnly,
    )
    .map_err(pipeline_rejection_to_compiler_error)
    .map_err(|error| cancelled_or("lower", error))?;
    let lowered_parts = lowered.into_parts();
    let (_, _, new_checked, root_metadata) = lowered_parts.checked.into_parts();
    let new_tensor_root_names = root_metadata.tensor_names().clone();
    let realizability = chelis_effects::realizability::infer_realizability(
        &new_checked,
        crate::target_capability::tensor_capable_prims(target),
    );
    let mut manifest =
        chelis_effects::realizability::compute_root_manifest(&new_checked, &realizability);
    route_tensor_inputs_from_dag(
        &mut manifest,
        &lowered_parts.dag,
        &lowered_parts.named_roots,
    );

    // Phase G' — carry the library defs + lowered classification into
    // the runtime. Without this, the host evaluator's `top_level_defs`
    // would see only new code's defs and would error
    // `unknown runtime name pkg__chelis__std__Std__Time__is_leap_year`
    // on any new-code call into a library function.
    let library_runtime = LibraryRuntime {
        checked: context.library_checked().clone(),
        lowered_names: crate::runtime::library_lowered_names(context.library_checked()),
    };

    Ok(CompiledSource {
        program: ManifestedProgram::new(new_checked, manifest, target),
        dag: lowered_parts.dag,
        tensor_root_names: new_tensor_root_names,
        named_roots: lowered_parts.named_roots,
        forward_node_index: lowered_parts.forward_node_index,
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
    eval_in_context_for_target(context, new_source, Target::Eval)
}

pub fn eval_in_context_for_target(
    context: &crate::context::CompiledContext,
    new_source: &str,
    target: Target,
) -> Result<EvalResult> {
    let compiled = compile_new_source_in_context(context, new_source, target)?;
    eval_compiled(&compiled, BTreeMap::new(), None)
}

/// Like [`eval_in_context`], but threads caller-supplied tensor `bindings`
/// into the evaluator instead of an empty map. This is the reef-aware
/// analogue of [`eval`] with bindings: it lets the Python `eval(...,
/// project_root=...)` path resolve library imports (issue #816) while still
/// binding the new source's free `Load`s to the caller's inputs.
pub fn eval_in_context_with_bindings(
    context: &crate::context::CompiledContext,
    new_source: &str,
    bindings: BTreeMap<String, crate::schema::TensorValue>,
) -> Result<EvalResult> {
    let compiled = compile_new_source_in_context(context, new_source, Target::Eval)?;
    eval_compiled(&compiled, bindings, None)
}

/// Type-/effects-/linearity-check `new_source` against an existing
/// `CompiledContext`. Returns a `CheckResult` with the same shape as
/// the existing `check` API. A successful result means the new code
/// composes cleanly with the library; errors carry new-code spans.
pub fn check_in_context(
    context: &crate::context::CompiledContext,
    new_source: &str,
) -> Result<CheckResult> {
    let compiled = compile_new_source_in_context(context, new_source, Target::Eval)?;
    // Mirror `check`'s shape: derive a fitness-style report from the
    // composed checked program. The total/typed counts only cover
    // new-code nodes — library nodes are checked once at context build
    // time and counted there.
    let total_nodes = compiled.checked().exprs().len();
    Ok(CheckResult {
        score: crate::schema::numbers::UnitInterval::new(1.0).expect("constant score"),
        components: FitnessComponents {
            parse: crate::schema::numbers::UnitInterval::new(1.0).expect("constant score"),
            structure: crate::schema::numbers::UnitInterval::new(1.0).expect("constant score"),
            names: crate::schema::numbers::UnitInterval::new(1.0).expect("constant score"),
            types: crate::schema::numbers::UnitInterval::new(1.0).expect("constant score"),
        },
        typed_nodes: total_nodes
            .try_into()
            .map_err(|error| stage_error("report", error, GeneralKind::Other))?,
        untyped_nodes: crate::schema::numbers::NonnegativeCount::new(0).expect("zero count"),
        total_nodes: total_nodes
            .try_into()
            .map_err(|error| stage_error("report", error, GeneralKind::Other))?,
        unresolved_names: vec![],
        inferred_signatures: None,
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
    let compiled = match compile_new_source_in_context(context, new_source, Target::Eval) {
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
    let compiled = compile_new_source_in_context(context, new_source, Target::Eval)?;
    Ok(PreparedEvalInContext {
        compiled: std::sync::Arc::new(compiled),
    })
}

/// Prepare an independently Reef-rewritten entry batch against a cached
/// context. The batch is consumed as linked declarations; it never re-enters
/// `rewrite_entry_decls_with_reef_graph` as one flat eval scope.
pub fn prepare_rewritten_entry_batch_in_context(
    context: &crate::context::CompiledContext,
    batch: &chelis_reef::RewrittenEntryBatch,
) -> Result<PreparedEvalInContext> {
    let compiled = compile_rewritten_decls_in_context(context, batch.declarations(), Target::Eval)?;
    Ok(PreparedEvalInContext {
        compiled: std::sync::Arc::new(compiled),
    })
}

fn eval_compiled(
    compiled: &CompiledSource,
    bindings: BTreeMap<String, crate::schema::TensorValue>,
    selected_root_names: Option<&[String]>,
) -> Result<EvalResult> {
    // A parameterized tensor entry is a callable declaration in the checked
    // manifest until evaluation selects it and supplies all of its runtime
    // inputs. Specialize that selection into a new manifested program before
    // consuming any roots, so the legacy in-context `main(x)` surface remains
    // manifest-authoritative instead of bypassing the phase boundary.
    let effective_program = manifested_program_for_eval(
        compiled,
        bindings.keys().map(String::as_str),
        selected_root_names,
    );
    let manifest = effective_program.manifest();
    let selected =
        selected_root_names.map(|roots| roots.iter().cloned().collect::<BTreeSet<String>>());
    let observed_entries = manifest
        .entries
        .iter()
        .filter(|entry| {
            selected.as_ref().is_none_or(|set| {
                set.contains(entry.name.as_str())
                    || set.iter().any(|root| {
                        entry
                            .name
                            .strip_prefix(root.as_str())
                            .is_some_and(|suffix| suffix.starts_with('.'))
                    })
            })
        })
        .collect::<Vec<_>>();
    let required_inputs = observed_entries
        .iter()
        .flat_map(|entry| entry.required_inputs.iter().cloned())
        .collect::<UnordSet<_>>();

    let bindings = bindings
        .into_iter()
        .filter(|(name, _)| required_inputs.contains(name))
        .map(|(name, value)| {
            let tensor = crate::decode::wire_tensor_to_ir(&value).map_err(|message| {
                stage_error(
                    "eval",
                    format!("binding `{name}`: {message}"),
                    GeneralKind::EvalError,
                )
            })?;
            Ok((name, tensor))
        })
        .collect::<Result<UnordMap<_, _>>>()?;

    let tensor_entries = observed_entries
        .iter()
        .copied()
        .filter(|entry| entry.lane == Lane::Tensor)
        .collect::<Vec<_>>();
    let roots = tensor_entries
        .iter()
        .map(|entry| {
            let name = crate::pipeline::IrName::new(entry.name.as_str());
            compiled.named_roots.get(&name).copied().ok_or_else(|| {
                unavailable_root_error(
                    entry,
                    "the Tensor-lane root is absent from the lowered named-root map",
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let tensor_values = if roots.is_empty() {
        UnordMap::new()
    } else {
        eval::eval_tensor_roots_with_strict(&compiled.dag, &roots, |name| {
            bindings.get(name).cloned()
        })
        .map_err(eval_stage_error)?
    };

    let mut tensor_values_by_name = UnordMap::<String, RuntimeTensorValue>::new();
    for entry in &tensor_entries {
        let name = crate::pipeline::IrName::new(entry.name.as_str());
        let node_id = compiled.named_roots.get(&name).ok_or_else(|| {
            unavailable_root_error(
                entry,
                "the Tensor-lane root is absent from the lowered named-root map",
            )
        })?;
        let value = tensor_values.get(node_id).ok_or_else(|| {
            unavailable_root_error(
                entry,
                "the Tensor evaluator returned no value for the owed root",
            )
        })?;
        let precision = compiled
            .dag
            .get(*node_id)
            .map(|node| node.output_type.precision)
            .ok_or_else(|| {
                stage_error(
                    "eval",
                    format!("missing node {}", node_id.0),
                    GeneralKind::EvalError,
                )
            })?;
        debug_assert_eq!(
            value.prim(),
            precision,
            "the DAG evaluator finalizes at the root's declared dtype"
        );
        tensor_values_by_name.insert(entry.name.clone(), RuntimeTensorValue::new(value.clone()));
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
    let host_selected_root_names = observed_entries
        .iter()
        .map(|entry| entry.def_name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let host_selected_root_names = Some(host_selected_root_names.as_slice());
    let manifested_lowered_names = manifest
        .entries
        .iter()
        .map(|entry| (entry.def_name.clone(), entry.lane == Lane::Tensor))
        .collect::<BTreeMap<_, _>>();
    let host_outcome = if let Some(library) = compiled.library_runtime.as_ref() {
        evaluate_host_program_with_library_and_types(
            compiled.checked(),
            Some(&library.checked),
            Some(&library.lowered_names),
            &tensor_values_by_name,
            host_selected_root_names,
            Some(&manifested_lowered_names),
        )
    } else {
        evaluate_host_program_filtered(
            compiled.checked(),
            &tensor_values_by_name,
            host_selected_root_names,
            Some(&manifested_lowered_names),
        )
    }
    .map_err(|failure| {
        let mut error = eval_stage_error(failure.message);
        error.transcript = failure.transcript;
        error
    })?;

    let preserve_transcript = |mut error: CompilerError| {
        error.transcript.clone_from(&host_outcome.transcript);
        error
    };
    let roots = observed_entries
        .iter()
        .copied()
        .enumerate()
        .map(|(index, entry)| {
            // An executed root's failure is already the runtime diagnostic.
            // Wrapping it as missing output corrupts the required trap line.
            if let Some(error) = host_outcome.host_root_errors.get(entry.def_name.as_str()) {
                return Err(eval_stage_error(error.clone()));
            }
            // A host-lane *zero-argument fn* root is the result of applying
            // the callable, never the closure stored in `host_bindings` when
            // another root resolved that declaration. Prefer the applied
            // root value, then fall back to ordinary host value bindings and
            // tensor-lane roots.
            let value = host_outcome
                .host_root_values
                .get(entry.def_name.as_str())
                .cloned()
                .and_then(|value| {
                    let synthetic_bindings = UnordMap::from([(entry.def_name.clone(), value)]);
                    lookup_runtime_value_for_manifest_root(
                        entry,
                        &synthetic_bindings,
                        &tensor_values_by_name,
                    )
                })
                .or_else(|| {
                    lookup_runtime_value_for_manifest_root(
                        entry,
                        &host_outcome.host_bindings,
                        &tensor_values_by_name,
                    )
                })
                .ok_or_else(|| {
                    let reason = host_outcome
                        .host_root_errors
                        .get(entry.def_name.as_str())
                        .map(|error| format!("the Host-lane nullary root failed: {error}"))
                        .unwrap_or_else(|| {
                            "the assigned lane returned no value for the owed root".to_string()
                        });
                    unavailable_root_error(entry, &reason)
                })?;
            let ir_name = crate::pipeline::IrName::new(entry.name.as_str());
            let node_id = compiled
                .named_roots
                .get(&ir_name)
                .map(|id| id.0)
                .unwrap_or(index);
            Ok((node_id, entry.name.clone(), value))
        })
        .collect::<Result<Vec<_>>>()
        .map_err(preserve_transcript)?
        .into_iter()
        .map(|(node_id, name, value)| {
            Ok(EvaluatedRoot {
                node_id: crate::schema::host_index(node_id),
                name: Some(name),
                // Render the display text HERE, where the runtime value's
                // dtype tags still exist; the wire `value` below cannot
                // carry them (chelis#732 P1, [05-OBS-1]).
                display: Some(crate::runtime::render_value(&value)),
                value: runtime_value_to_schema(&value).map_err(eval_stage_error)?,
            })
        })
        .collect::<Result<Vec<_>>>()
        .map_err(preserve_transcript)?;

    Ok(EvalResult {
        schema_version: crate::schema::EXECUTION_VALUE_SCHEMA_VERSION,
        roots,
        manifest: manifest_result(&effective_program),
        transcript: host_outcome.transcript,
    })
}

fn unavailable_root_error(entry: &RootEntry, reason: &str) -> CompilerError {
    let routing_reason = entry
        .reasons
        .first()
        .map(|reason| format!("; routing evidence: {reason:?}"))
        .unwrap_or_default();
    unsupported_stage_error(chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Construct(format!(
            "[05-UNS-1] unavailable root `{}`",
            entry.name
        )),
        format!("{:?} lane: {reason}{routing_reason}", entry.lane),
        chelis_types::unsupported::Stage::Runtime,
        chelis_types::unimplemented_rejection!(
            912,
            "this is a root-realization defect; file a bug with the program and target"
        ),
    ))
}

pub fn grad(request: GradRequest) -> Result<GradResult> {
    let compiled = compile_source(request.source_kind, &request.source)?;
    let output_name = crate::pipeline::IrName::new(request.output_name.as_str());
    let output = compiled
        .forward_node_index
        .get(&output_name)
        .copied()
        .ok_or_else(|| unknown_name_error("grad", "output_name", &request.output_name))?;

    let wrt_nodes = request
        .wrt_names
        .iter()
        .map(|name| {
            let ir_name = crate::pipeline::IrName::new(name.as_str());
            compiled
                .forward_node_index
                .get(&ir_name)
                .copied()
                .ok_or_else(|| unknown_name_error("grad", "wrt_names", name))
        })
        .collect::<Result<Vec<_>>>()?;

    // Issue #197: route through the *checked* AD entry points so a
    // non-differentiable op (argmax/argmin, floor/ceil,
    // scatter_replace) surfaces as a structured
    // `AdError::NotSupported` diagnostic with the offending op name
    // and a reason, instead of a generic "grad requires a scalar
    // floating output" message (or, worse, a silent zero gradient
    // for floor/ceil under the unchecked variant).
    let grad_result = if request.fuse {
        chelis_ir::grad_then_fuse_checked(&compiled.dag, output, &wrt_nodes)
    } else {
        chelis_ir::grad::grad_dag_checked(&compiled.dag, output, &wrt_nodes)
    }
    .map_err(|ad_err| stage_error("grad", ad_err.to_string(), GeneralKind::GradError))?;

    let grad_nodes_by_name = request
        .wrt_names
        .iter()
        .zip(wrt_nodes.iter())
        .filter_map(|(name, node)| {
            grad_result
                .grad_nodes
                .get(node)
                .copied()
                .map(|grad_node| (name.clone(), crate::schema::host_index(grad_node.0)))
        })
        .collect();

    let dag = wire_dag(&grad_result.dag)
        .map_err(|error| stage_error("schema", error, GeneralKind::Other))?;
    // WI-2 validate-on-consume: fail closed before the gradient DAG crosses
    // the process edge (same rationale as `lower`).
    schema_stage_check(dag.validate_schema_version())?;
    Ok(GradResult {
        dag,
        output_node: crate::schema::host_index(grad_result.output_node.0),
        grad_nodes_by_name,
        forward_nodes_by_name: compiled
            .forward_node_index
            .into_entries()
            .map(|(name, node)| (name.into_string(), crate::schema::host_index(node.0)))
            .collect(),
    })
}

pub fn validate(request: ValidateRequest) -> Result<ValidateResult> {
    let result = match request.mode {
        ValidateMode::Surf => chelis_validate::validate_surf(&request.source),
        ValidateMode::Deep => chelis_validate::validate_deep(&request.source),
        ValidateMode::Desugar => chelis_validate::validate_desugared(&request.source),
    };

    result.map_err(|err| stage_error("validate", err.to_string(), GeneralKind::ValidationError))?;

    Ok(ValidateResult {
        mode: request.mode,
        valid: true,
    })
}

pub fn decompile(request: DecompileRequest) -> Result<DecompileResult> {
    let exprs = parse_deep(&request.source)?;
    let surf_text = chelis_surf::decompile::try_decompile_program(&exprs).map_err(|error| {
        stage_error("decompile", error.to_string(), GeneralKind::ValidationError)
    })?;
    Ok(DecompileResult {
        surf_text: canonicalize_decompiled_surf(&surf_text)?,
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

pub(crate) fn pipeline_rejection_to_compiler_error(
    rejection: crate::pipeline::PipelineRejection,
) -> CompilerError {
    use crate::pipeline::{PipelineRejection, PreparationError};

    match rejection {
        PipelineRejection::Cancelled { stage } => cancelled_stage_error(stage),
        PipelineRejection::Preparation(PreparationError::SurfParse { source, error }) => {
            stage_error_with_span(
                "parse",
                error.to_string(),
                GeneralKind::SurfParseError,
                Some(parse_error_span_surf(&source, &error)),
            )
        }
        PipelineRejection::Preparation(PreparationError::DeepParse(error)) => {
            deep_ingress_error("parse", &error)
        }
        PipelineRejection::Preparation(PreparationError::Expansion(error)) => {
            stage_error("desugar", error.to_string(), GeneralKind::MacroError)
        }
        PipelineRejection::Type { fitness } => {
            crate::compiler::check_errors_to_compiler_error("check", &fitness.errors)
        }
        PipelineRejection::Effects { errors } => CompilerError {
            transcript: Vec::new(),
            stage: "effects".to_string(),
            errors: errors
                .iter()
                .map(|error| {
                    Diagnostic::general(
                        GeneralKind::EffectError,
                        error.message.clone(),
                        crate::schema::numbers::UnitInterval::new(0.8).expect("constant severity"),
                    )
                })
                .collect(),
        },
        PipelineRejection::Linearity { errors } => {
            crate::compiler::check_errors_to_compiler_error("linearity", &errors)
        }
        PipelineRejection::Lower(diagnostic) => stage_error_with_span(
            "lower",
            diagnostic.to_string(),
            GeneralKind::LowerError,
            deep_span_to_diagnostic(diagnostic.span),
        ),
        PipelineRejection::RootCount {
            context,
            expected,
            actual,
        } => {
            let subject = match context {
                crate::pipeline::RootCountContext::Program => "root",
                crate::pipeline::RootCountContext::NewCode => "new-code root",
            };
            stage_error(
                "lower",
                format!(
                    "lowered {subject} count mismatch: expected {expected} named roots, got {actual}"
                ),
                GeneralKind::LowerError,
            )
        }
    }
}

pub fn result_envelope<T>(result: Result<T>) -> crate::schema::ApiEnvelope<T> {
    match result {
        Ok(value) => crate::schema::ApiEnvelope::success(value),
        Err(err) => crate::schema::ApiEnvelope::failure(err.stage, err.errors),
    }
}

struct CompiledSource {
    program: ManifestedProgram,
    dag: Dag,
    // Callable function-entry selection is a separate surface from value-root
    // observation. Keep the pipeline's typed set for that API; eval/build
    // observation below consumes `program.manifest` exclusively.
    tensor_root_names: crate::pipeline::TensorRootNames,
    named_roots: crate::pipeline::NamedRoots,
    forward_node_index: crate::pipeline::ForwardNodeIndex,
    /// Phase G' — optional library context payload threaded into the
    /// host evaluator so library `def` names resolve at runtime when
    /// new code calls them. `None` on the monolithic `compile_source`
    /// path (no separate library to merge); `Some` on the in-context
    /// path produced by `compile_new_source_in_context`.
    library_runtime: Option<LibraryRuntime>,
}

impl CompiledSource {
    fn checked(&self) -> &CheckedProgram {
        self.program.checked()
    }

    fn manifest(&self) -> &RootManifest {
        self.program.manifest()
    }
}

fn manifest_result(program: &ManifestedProgram) -> RootManifestResult {
    RootManifestResult {
        target: program.target(),
        entries: program
            .manifest()
            .entries
            .iter()
            .map(|entry| RootManifestEntryResult {
                name: entry.name.clone(),
                lane: entry.lane,
                required_inputs: entry.required_inputs.iter().cloned().collect(),
            })
            .collect(),
        requires_main: program.manifest().requires_main(),
    }
}

/// Specialize callable tensor entries selected for evaluation into owed
/// roots once all authored parameters have bindings. The checked manifest
/// intentionally excludes parameterized declarations in the abstract; this
/// produces a new `ManifestedProgram` for the concrete evaluation request
/// rather than reaching around the manifest to the legacy named-root map.
fn manifested_program_for_eval<'a>(
    compiled: &CompiledSource,
    binding_names: impl Iterator<Item = &'a str>,
    selected_root_names: Option<&[String]>,
) -> ManifestedProgram {
    let available = binding_names.collect::<UnordSet<_>>();
    let candidate_names = selected_root_names
        .map(|names| {
            names
                .iter()
                .filter_map(|name| name.split('.').next())
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_else(|| BTreeSet::from(["main"]));
    let mut manifest = compiled.manifest().clone();
    let realizability = chelis_effects::realizability::infer_realizability(
        compiled.checked(),
        crate::target_capability::tensor_capable_prims(compiled.program.target()),
    );

    for candidate in candidate_names {
        if manifest
            .entries
            .iter()
            .any(|entry| entry.def_name == candidate)
        {
            continue;
        }
        let Some(signature) = compiled
            .checked()
            .signature_inference()
            .functions
            .get(candidate)
        else {
            continue;
        };
        if signature.params.is_empty() {
            continue;
        }
        let Some(function_ty) = compiled.checked().type_env().get(candidate) else {
            continue;
        };
        let Some(return_ty) = deep_tagged_children(function_ty)
            .filter(|(tag, _)| *tag == DeepTag::TFn)
            .and_then(|(_, children)| children.last())
            .cloned()
        else {
            continue;
        };
        let template = RootEntry {
            name: candidate.to_string(),
            path: Vec::new(),
            def_name: candidate.to_string(),
            ty: return_ty,
            lane: Lane::Tensor,
            required_inputs: BTreeSet::new(),
            reasons: Vec::new(),
        };
        let Some(body) = selected_callable_result_expr(compiled.checked(), candidate) else {
            continue;
        };
        let mut selected_entries = chelis_effects::realizability::expand_manifest_root(
            template,
            Some(body),
            compiled.checked().adt_registry(),
        );
        let selected_roots = selected_entries
            .iter()
            .filter_map(|entry| {
                compiled
                    .named_roots
                    .get(&crate::pipeline::IrName::new(entry.name.as_str()))
                    .copied()
            })
            .collect::<Vec<_>>();
        let parameter_names = signature
            .params
            .iter()
            .map(|param| param.name.clone())
            .collect::<BTreeSet<_>>();
        let live_parameter_names =
            chelis_effects::realizability::referenced_runtime_inputs(body, &parameter_names);
        if live_parameter_names.iter().any(|name| {
            signature
                .params
                .iter()
                .find(|param| param.name == *name)
                .is_none_or(|param| !type_is_tensor_runtime_input(&param.checked_type))
        }) {
            // EvalRequest bindings carry tensors. A live scalar/container
            // parameter therefore has not been supplied through this API and
            // the callable remains a declaration.
            continue;
        }
        let mut required_inputs = selected_roots
            .iter()
            .flat_map(|root| required_inputs_for_dag_root(&compiled.dag, *root))
            .collect::<BTreeSet<_>>();
        required_inputs.extend(live_parameter_names);
        if let Some(top_level_inputs) = realizability.required_inputs_by_def.get(candidate) {
            required_inputs.extend(top_level_inputs.iter().cloned());
        }
        if !required_inputs
            .iter()
            .all(|input| available.contains(input.as_str()))
        {
            continue;
        }
        let unsupported_prim = selected_roots.iter().find_map(|root| {
            compiled
                .dag
                .get(*root)
                .map(|node| node.output_type.precision)
                .filter(|prim| {
                    !crate::target_capability::tensor_capable_prims(compiled.program.target())
                        .contains(prim)
                })
        });
        let lane = if unsupported_prim.is_some() || selected_roots.len() != selected_entries.len() {
            Lane::Host
        } else {
            Lane::Tensor
        };
        for entry in &mut selected_entries {
            entry.lane = lane;
            entry.required_inputs.clone_from(&required_inputs);
            if let Some(prim) = unsupported_prim {
                entry
                    .reasons
                    .push(chelis_types::manifest::HostReason::PrecisionExceedsCapability { prim });
            } else if lane == Lane::Host {
                entry
                    .reasons
                    .push(chelis_types::manifest::HostReason::StructuralForm {
                        tag: "selected-callable-result".to_string(),
                    });
            }
        }
        manifest.entries.extend(selected_entries);
    }

    route_tensor_inputs_from_dag(&mut manifest, &compiled.dag, &compiled.named_roots);
    let declaration_order = checked_def_order(compiled.checked());
    manifest.entries.sort_by_key(|entry| {
        declaration_order
            .get(entry.def_name.as_str())
            .copied()
            .unwrap_or(usize::MAX)
    });
    ManifestedProgram::new(
        compiled.checked().clone(),
        manifest,
        compiled.program.target(),
    )
}

fn type_is_tensor_runtime_input(ty: &chelis_types::types::Type) -> bool {
    match ty {
        chelis_types::types::Type::Tensor(_, _) => true,
        chelis_types::types::Type::Ref(inner) => type_is_tensor_runtime_input(inner),
        _ => false,
    }
}

fn selected_callable_result_expr<'a>(
    program: &'a CheckedProgram,
    selected: &str,
) -> Option<&'a DeepExpr> {
    fn find<'a>(expr: &'a DeepExpr, selected: &str) -> Option<&'a DeepExpr> {
        let (tag, children) = deep_tagged_children(expr)?;
        if tag == DeepTag::Module {
            return children
                .iter()
                .skip(1)
                .find_map(|child| find(child, selected));
        }
        if tag != DeepTag::Def || children.first().and_then(symbol_name) != Some(selected) {
            return None;
        }
        let body = children.get(1)?;
        deep_tagged_children(body)
            .filter(|(body_tag, _)| *body_tag == DeepTag::Fn)
            .and_then(|(_, fn_children)| fn_children.last())
    }

    program
        .annotated_exprs()
        .iter()
        .find_map(|expr| find(expr, selected))
}

fn checked_def_order(program: &CheckedProgram) -> UnordMap<&str, usize> {
    fn collect<'a>(expr: &'a DeepExpr, names: &mut Vec<&'a str>) {
        let Some((tag, children)) = deep_tagged_children(expr) else {
            return;
        };
        if tag == DeepTag::Module {
            for child in children.iter().skip(1) {
                collect(child, names);
            }
        } else if tag == DeepTag::Def
            && let Some(name) = children.first().and_then(symbol_name)
        {
            names.push(name);
        }
    }

    let mut names = Vec::new();
    for expr in program.annotated_exprs() {
        collect(expr, &mut names);
    }
    names
        .into_iter()
        .enumerate()
        .map(|(index, name)| (name, index))
        .collect()
}

fn deep_tagged_children(expr: &DeepExpr) -> Option<(DeepTag, &[DeepExpr])> {
    match expr {
        DeepExpr::Node(node, _) => Some((node.tag(), node.children_slice())),
        DeepExpr::List(list, _) => {
            let tag = list.tag()?;
            let children = if list.elements.len() > 2
                && matches!(list.elements.get(1), Some(DeepExpr::Map(_, _)))
            {
                &list.elements[2..]
            } else if list.elements.len() > 1 {
                &list.elements[1..]
            } else {
                &[]
            };
            Some((tag, children))
        }
        _ => None,
    }
}

/// Replace the checked walk's conservative input set with the exact Load
/// closure of every root which has a lowered representation. Host-only roots
/// have no named DAG root and keep the checked top-level dependency closure.
/// Sibling roots cannot make an unrelated binding or symbolic dimension live.
fn route_tensor_inputs_from_dag(
    manifest: &mut RootManifest,
    dag: &Dag,
    named_roots: &crate::pipeline::NamedRoots,
) {
    let mut required_by_def = BTreeMap::<String, BTreeSet<String>>::new();
    for entry in &manifest.entries {
        if entry.lane != Lane::Tensor {
            continue;
        }
        let name = crate::pipeline::IrName::new(entry.name.as_str());
        let Some(root) = named_roots.get(&name).copied() else {
            continue;
        };
        required_by_def
            .entry(entry.def_name.clone())
            .or_default()
            .extend(required_inputs_for_dag_root(dag, root));
    }
    for entry in &mut manifest.entries {
        if entry.lane == Lane::Tensor
            && let Some(required_inputs) = required_by_def.get(entry.def_name.as_str())
        {
            entry.required_inputs.clone_from(required_inputs);
        }
    }
}

fn required_inputs_for_dag_root(dag: &Dag, root: NodeId) -> BTreeSet<String> {
    let mut stack = vec![root];
    let mut seen = UnordSet::new();
    let mut required = BTreeSet::new();
    while let Some(node_id) = stack.pop() {
        if !seen.insert(node_id) {
            continue;
        }
        let Some(node) = dag.get(node_id) else {
            continue;
        };
        if let RiscOp::Load { name } = &node.op {
            required.insert(name.as_str().to_string());
        }
        stack.extend(node.inputs.iter().copied());
    }
    required
}

/// The library payload threaded through `eval_compiled` so the host
/// evaluator's `top_level_defs` table can resolve library function
/// references when called from new code.
#[derive(Clone)]
struct LibraryRuntime {
    /// The checked proof, definitions and declarations travel together;
    /// imported calls consult the same kernel owner as generated C.
    checked: CheckedProgram,
    /// Library-side lowered-vs-host classification. Threaded through
    /// so `evaluate_host_program_with_library`'s "is this a tensor
    /// root vs a host-init" decision is byte-identical to what the
    /// monolithic pipeline would have computed.
    lowered_names: BTreeMap<String, bool>,
}

fn compile_source(source_kind: SourceKind, source: &str) -> Result<CompiledSource> {
    compile_source_for_target(source_kind, source, Target::Eval)
}

fn compile_source_for_target(
    source_kind: SourceKind,
    source: &str,
    target: Target,
) -> Result<CompiledSource> {
    compile_source_scoped(source_kind, source, None, target)
}

/// Like [`compile_source`], but when `entry` is `Some`, prune the expanded
/// program to the defs reachable from that named entry BEFORE the type
/// checker runs. This is the WI-3 entrypoint-isolation path: it lets a caller
/// extract one function from a module that also defines unrelated functions
/// referencing unresolved imports, without those unrelated functions blocking
/// the target's lowering. Pruning ([`crate::prune::prune_to_entry`]) drops
/// only genuinely-unreachable defs, so an unresolved symbol in the ENTRY's
/// own closure still surfaces here as a `check`-stage error.
fn compile_source_scoped(
    source_kind: SourceKind,
    source: &str,
    entry: Option<&str>,
    target: Target,
) -> Result<CompiledSource> {
    bail_if_cancelled("parse")?;
    let outcome = crate::pipeline::run_source(crate::pipeline::PipelineRequest {
        source_kind,
        source,
        entry,
        goal: crate::pipeline::PipelineGoal::Lower(crate::pipeline::LoweringMode::AllowHostOnly),
    })
    .map_err(pipeline_rejection_to_compiler_error)
    .map_err(|error| cancelled_or("check", error))?;
    let crate::pipeline::PipelineOutcome::Lowered(lowered) = outcome else {
        unreachable!("the lower goal returns only a lowered outcome")
    };
    bail_if_cancelled("lower")?;
    // chelis#1079: realizability is a production phase boundary. The target
    // is chosen before inference, and the checked program cannot proceed to
    // root observation without its manifest attached.
    let checked_program = lowered.checked().program();
    let realizability_result = chelis_effects::realizability::infer_realizability(
        checked_program,
        crate::target_capability::tensor_capable_prims(target),
    );
    let mut manifest = chelis_effects::realizability::compute_root_manifest(
        checked_program,
        &realizability_result,
    );
    let lowered_parts = lowered.into_parts();
    let (_, _, checked, root_metadata) = lowered_parts.checked.into_parts();
    route_tensor_inputs_from_dag(
        &mut manifest,
        &lowered_parts.dag,
        &lowered_parts.named_roots,
    );

    Ok(CompiledSource {
        program: ManifestedProgram::new(checked, manifest, target),
        dag: lowered_parts.dag,
        tensor_root_names: root_metadata.tensor_names().clone(),
        named_roots: lowered_parts.named_roots,
        forward_node_index: lowered_parts.forward_node_index,
        library_runtime: None,
    })
}

/// Does this Surf source contain at least one `import` declaration?
///
/// Used by reef-context auto-discovery (issue #816, review round 2): an
/// import-free source can only reference its own decls, so it never needs a
/// reef library context — it takes the bare self-contained compile path
/// exactly as pre-#816, avoiding both the whole-project context-compile cost
/// and coupling to unrelated sibling-file health. A source that DOES import is
/// the only one auto-discovery routes in-context. `import` decls inside
/// `module` wrappers are counted (they are flattened before resolution). A
/// source that fails to parse returns `false`: the bare path then surfaces the
/// real parse diagnostic, unchanged from pre-#816 behavior.
pub fn surf_source_has_import(source: &str) -> bool {
    match chelis_surf::parser::parse_str(source) {
        Ok(decls) => flatten_module_decls(&decls)
            .iter()
            .any(|decl| matches!(decl, Decl::Import { .. })),
        Err(_) => false,
    }
}

fn parse_surf(source: &str) -> Result<Vec<Decl>> {
    chelis_surf::parser::parse_str(source).map_err(|err| {
        stage_error_with_span(
            "parse",
            err.to_string(),
            GeneralKind::SurfParseError,
            Some(parse_error_span_surf(source, &err)),
        )
    })
}

/// The generic Deep text ingress for `parse` and `decompile` (chelis#1088).
///
/// Routes through the stamped `.dp` ingress, so a top-level form that is
/// neither a `(module ...)` wrapper nor a declaration is a loud ingress
/// rejection instead of an untyped `Expr::BareList` the consumer has to
/// re-diagnose. Tag-vocabulary validation is deliberately NOT applied here:
/// an unknown head stays an `Expr::UnknownForm` so the wire AST preserves it
/// and the checker owns the rejection.
fn parse_deep(source: &str) -> Result<Vec<DeepExpr>> {
    chelis_deep::parse_and_stamp_file(source).map_err(|err| deep_ingress_error("parse", &err))
}

/// Render a stamped-ingress failure as a staged compiler error.
///
/// Both halves are Deep ingress rejections, so both carry
/// [`GeneralKind::DeepParseError`]; the stamp half contributes the exact
/// offending span rather than the whole-input offset a re-wrapped parse
/// error would report.
fn deep_ingress_error(stage: &str, error: &chelis_deep::StampOrParseError) -> CompilerError {
    let span = match error {
        chelis_deep::StampOrParseError::Parse(parse_error) => {
            Some(parse_error_span_deep(parse_error))
        }
        // The stamp half carries a measured extent, so it reports a range
        // where the parse half above can only report a point (chelis#1395).
        chelis_deep::StampOrParseError::Stamp(stamp_error) => Some(DiagnosticSpan::Range {
            offset: crate::schema::host_index(stamp_error.span.offset),
            len: crate::schema::host_index(stamp_error.span.len),
        }),
    };
    stage_error_with_span(stage, error.to_string(), GeneralKind::DeepParseError, span)
}

fn canonicalize_decompiled_surf(source: &str) -> Result<String> {
    let decls = chelis_surf::parser::parse_str(source).map_err(|err| {
        stage_error_with_span(
            "decompile",
            format!("decompiler emitted Surf that the parser rejected: {err}"),
            GeneralKind::SurfParseError,
            Some(parse_error_span_surf(source, &err)),
        )
    })?;
    Ok(chelis_surf::format::format_program(&decls))
}

fn deep_expr_tag(expr: &DeepExpr) -> Option<DeepTag> {
    expr.tag()
}

fn deep_decl_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::List(list, _) => list.elements.get(2).and_then(symbol_name),
        DeepExpr::Node(node, _) => node.children_slice().first().and_then(symbol_name),
        _ => None,
    }
}

fn deep_def_is_function(expr: &DeepExpr) -> bool {
    matches!(deep_expr_tag(expr), Some(DeepTag::Def)) && chelis_deep::function_body(expr).is_some()
}

fn deep_def_has_role(expr: &DeepExpr, expected: &str) -> bool {
    if deep_expr_tag(expr) != Some(DeepTag::Def) {
        return false;
    }
    let meta = match expr {
        DeepExpr::List(list, _) => match list.elements.get(1) {
            Some(DeepExpr::Map(meta, _)) => meta,
            _ => return false,
        },
        DeepExpr::Node(node, _) => node.meta(),
        _ => return false,
    };
    meta.chelis_role()
        .is_some_and(|role| role.value() == expected)
}

fn symbol_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(chelis_deep::Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn compiled_execution_artifact(
    host_entry_name: &str,
    device_entry_name: Option<String>,
    mut compile_result: CompileResult,
    manifest: RootManifestResult,
    inputs: Vec<ExecutionTensorSpec>,
    outputs: Vec<ExecutionTensorSpec>,
    symbolic_dims: Vec<String>,
) -> CompiledExecutionArtifact {
    compile_result.manifest = manifest;
    CompiledExecutionArtifact {
        compile_result,
        host_entry_name: host_entry_name.to_string(),
        device_entry_name,
        inputs,
        outputs,
        symbolic_dims,
        // Set after construction on the C paths where the entry lane ran
        // and declined; `None` (the lane claimed it, or does not run on
        // this path) everywhere else.
        entry_lane_decline: None,
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
                path: "chelis_runtime_dtype.h".to_string(),
                contents: RUNTIME_DTYPE_H.to_string(),
            },
            GeneratedFile {
                path: "chelis_blas.h".to_string(),
                contents: BLAS_H.to_string(),
            },
        ],
        compile_flags: toolchain.compile_flags,
        link_flags: toolchain.link_flags,
        peak_device_bytes_estimate: None,
        manifest: RootManifestResult::default(),
    }
}

fn compile_result_hip(
    target: CompileTarget,
    func_name: &str,
    result: &HipCodegenResult,
) -> Result<CompileResult> {
    Ok(CompileResult {
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
                path: "chelis_runtime_dtype.h".to_string(),
                contents: RUNTIME_DTYPE_H.to_string(),
            },
            GeneratedFile {
                path: "chelis_hip_runtime.h".to_string(),
                contents: HIP_RUNTIME_H.to_string(),
            },
        ],
        compile_flags: result.compile_flags.clone(),
        link_flags: result.link_flags.clone(),
        peak_device_bytes_estimate: result
            .peak_device_bytes_estimate
            .map(TryInto::try_into)
            .transpose()
            .map_err(|error| stage_error("compile", error, GeneralKind::Other))?,
        manifest: RootManifestResult::default(),
    })
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
                path: "chelis_runtime_dtype.h".to_string(),
                contents: RUNTIME_DTYPE_H.to_string(),
            },
            GeneratedFile {
                path: "chelis_hip_runtime.h".to_string(),
                contents: HIP_RUNTIME_H.to_string(),
            },
        ],
        compile_flags: toolchain.compile_flags,
        link_flags: toolchain.link_flags,
        peak_device_bytes_estimate: None,
        manifest: RootManifestResult::default(),
    }
}

fn execution_input_specs(dag: &Dag, labels: &[String]) -> Result<Vec<ExecutionTensorSpec>> {
    let mut load_types = UnordMap::<String, TensorType>::new();
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
                    GeneralKind::CompileError,
                )
            })?;
            execution_tensor_spec(label.clone(), ty).map_err(|message| {
                stage_error("compile", message, GeneralKind::CompileError)
            })
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
                        GeneralKind::CompileError,
                    )
                })?
                .output_type;
            execution_tensor_spec(label.clone(), ty).map_err(|message| {
                stage_error("compile", message, GeneralKind::CompileError)
            })
        })
        .collect()
}

fn execution_output_nodes(dag: &Dag) -> Vec<NodeId> {
    let mut seen = chelis_unord::UnordSet::new();
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

fn execution_tensor_spec(name: String, ty: &TensorType) -> WireResult<ExecutionTensorSpec> {
    i32::try_from(ty.dims.len()).map_err(|_| "execution tensor rank exceeds int32".to_string())?;
    Ok(ExecutionTensorSpec {
        name,
        dtype: ty.precision.name().to_string(),
        dims: ty
            .dims
            .iter()
            .map(|dim| {
                Ok(match dim {
                    DimInfo::Lit(size) => ExecutionDim {
                        name: None,
                        size: Some(NonnegativeExtent::try_from(*size)?),
                    },
                    DimInfo::Named(name, Some(size)) => ExecutionDim {
                        name: Some(name.clone()),
                        size: Some(NonnegativeExtent::try_from(*size)?),
                    },
                    DimInfo::Named(name, None) => ExecutionDim {
                        name: Some(name.clone()),
                        size: None,
                    },
                })
            })
            .collect::<WireResult<Vec<_>>>()?,
    })
}

fn reject_unsized_named_dims(dag: &Dag, target: &'static str) -> Result<()> {
    for node in dag.nodes() {
        for dim in &node.output_type.dims {
            if let DimInfo::Named(name, None) = dim {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target {target}` does not yet support unresolved named dimensions; node {} uses symbolic dimension `{name}`",
                        node.id.0
                    ),
                    target,
                    chelis_types::unimplemented_rejection!(
                        600,
                        "dynamic output shapes do not yet have a backend representation"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// `reduce_window_*` over a runtime-symbolic windowed axis cannot be
/// lowered to a correct static output shape on the build path. The
/// windowed output extent is `floor((d - window) / stride) + 1` — always
/// strictly smaller than the input extent `d` unless `window == stride ==
/// 1` — and that expression is not representable in the `DimExpr` model
/// (no subtraction / floor). Lowering therefore leaves the windowed
/// output axis as an unsized symbolic dim, which the backend's
/// symbolic-dim binding then ties to the *input* extent at the same axis
/// index. The result is a silently mis-allocated output tensor and an
/// out-of-bounds window read: `chelis build` emits a kernel whose output
/// diverges from the IR evaluator / host runtime (which recompute the
/// shape from the concrete runtime extent and are correct).
///
/// Reject such a program at compile time with a clear `unsupported_feature`
/// error, per `spec/05-risc-primitives.md` §2.3.1 ("Statically-known
/// windowed extents required on the build path"). Only the *windowed*
/// (trailing `window_shape.len()`) axes are checked; the leading
/// pass-through axes may remain symbolic and bind correctly. The IR
/// evaluator and host runtime are unaffected and handle runtime-only
/// extents.
pub fn reject_symbolic_windowed_reduce(
    dag: &Dag,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    let target = target.as_str();
    for node in dag.nodes() {
        let RiscOp::ReduceWindow { window_shape, .. } = &node.op else {
            continue;
        };
        let dims = &node.output_type.dims;
        let leading = dims.len().saturating_sub(window_shape.len());
        for (offset, dim) in dims.iter().enumerate().skip(leading) {
            if let DimInfo::Named(name, None) = dim {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target {target}` requires statically-known \
                         windowed-axis extents for `reduce_window_*`; node {} windowed axis \
                         {offset} has runtime-only symbolic dimension `{name}`. The windowed \
                         output extent floor((d - window) / stride) + 1 is not representable \
                         for a runtime-only input extent, so the build cannot allocate a \
                         correct output. Window over a statically-sized axis, or pad the \
                         input to a concrete extent first. See spec/05-risc-primitives.md \
                         §2.3.1.",
                        node.id.0
                    ),
                    target,
                    chelis_types::unimplemented_rejection!(
                        600,
                        "the runtime-derived window output extent needs a dynamic output shape"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// `reduce_window_*` and its adjoint are f32-only in the C backend
/// today: `chelis_backend_c::emit::emit_reduce_window{,_grad}` route
/// through `fmaxf` / `fminf` / `float`-accumulator kernels with no
/// bf16/f16 convert-load path. The C backend admits bf16/f16 tensors
/// generally (other ops widen them via `chelis_<x>_to_f32` helpers), so without
/// this operation-specific guard a bf16/f16 `reduce_window_*` would reach the
/// emitter and abort with an `internal error` panic instead of a clean
/// diagnostic. Reject at compile time with an `unsupported_feature` error;
/// the emitter `panic!` stays as a defensive backstop. bf16/f16 widening is
/// follow-on work — see spec/05-risc-primitives.md §2.3.1.
pub fn reject_unsupported_reduce_window_precision(
    dag: &Dag,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    let target = target.as_str();
    for node in dag.nodes() {
        let (op_label, reducer) = match &node.op {
            RiscOp::ReduceWindow { reducer, .. } => ("reduce_window_*", reducer),
            RiscOp::ReduceWindowGrad { reducer, .. } => ("reduce_window_* adjoint", reducer),
            _ => continue,
        };
        let prec = node.output_type.precision;
        if prec != chelis_types::types::Prim::F32 {
            return Err(unsupported_gate_error(
                format!(
                    "`chelis build --target {target}` supports `{op_label}` (`{}`) on f32 \
                     tensors only; node {} carries precision `{}`. bf16/f16 windowed \
                     reductions are not yet lowered (no convert-load path); cast to f32 \
                     before the windowed reduction. See spec/05-risc-primitives.md §2.3.1.",
                    reducer.surf_name(),
                    node.id.0,
                    prec.name(),
                ),
                target,
                chelis_types::unimplemented_rejection!(
                    729,
                    "the dtype capability table and typed window kernels do not yet cover this cell"
                ),
            ));
        }
    }
    Ok(())
}

/// Host-only builtins that have no compiled-backend lowering. Calls
/// to these from a `chelis build` program must fail at compile time
/// with the owning early diagnostic. The emitter's Result boundary remains
/// the independent safety mechanism. Spec: `spec/05-risc-primitives.md` §3.6.
const HOST_ONLY_BUILTINS: &[&str] = &["tensor_scan"];

/// Closed target vocabulary for shared pre-codegen build gates.
///
/// Gate callers cannot pass an arbitrary string: every target spelling is
/// decoded once at the public boundary, then every shared policy consumes
/// this exhaustive enum. An unknown spelling is therefore an error, never a
/// request to skip the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildTarget {
    C,
    Hip,
    Metal,
}

impl BuildTarget {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::C => "c",
            Self::Hip => "hip",
            Self::Metal => "metal",
        }
    }
}

impl From<CompileTarget> for BuildTarget {
    fn from(target: CompileTarget) -> Self {
        match target {
            CompileTarget::C => Self::C,
            CompileTarget::Hip => Self::Hip,
        }
    }
}

const fn manifest_target(target: CompileTarget) -> Target {
    match target {
        CompileTarget::C => Target::C,
        CompileTarget::Hip => Target::Hip,
    }
}

impl TryFrom<&str> for BuildTarget {
    type Error = String;

    fn try_from(target: &str) -> std::result::Result<Self, Self::Error> {
        match target {
            "c" => Ok(Self::C),
            "hip" => Ok(Self::Hip),
            "metal" => Ok(Self::Metal),
            other => Err(format!(
                "unknown target '{other}': expected 'c', 'hip', or 'metal'"
            )),
        }
    }
}

fn host_only_builtin_error(name: &str, target: BuildTarget) -> CompilerError {
    let unsupported =
        chelis_types::unsupported::Unsupported::compiled_host_only_builtin(name, target.as_str());
    unsupported_stage_error(unsupported)
}

fn unsupported_gate_error(
    message: impl Into<String>,
    target: &'static str,
    authority: chelis_types::unsupported::RejectionAuthority,
) -> CompilerError {
    unsupported_stage_error(chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Construct(message.into()),
        format!("`chelis build --target {target}` early capability gate"),
        chelis_types::unsupported::Stage::Codegen(target),
        authority,
    ))
}

/// Reject direct host-runtime-only calls on checked Deep before host lowering
/// descends into their callback arguments. This preserves the owning builtin
/// diagnostic even when an argument is itself intentionally unrepresentable
/// in compiled code (for example `tensor_scan(..., fn (...), ...)`). The
/// concrete-HostProgram scan below remains the second boundary for aliases
/// and other shapes materialized by lowering.
pub fn reject_host_only_builtins_before_host_lowering(
    program: &CheckedProgram,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    if let Some(name) = chelis_ir::host::find_direct_builtin_call(program, HOST_ONLY_BUILTINS) {
        return Err(host_only_builtin_error(&name, target));
    }
    Ok(())
}

/// Eval/test-only builtins (`process_run`, the chelis#890 JSON family, the
/// chelis#903 CSV family) are rejected for every compiled target with the
/// same message the CLI build gate prints, so the public
/// `compile()`/`compile_for_execution()` APIs (the chelis-python path)
/// fail loudly instead of falling through to a generic codegen error
/// (chelis#891 review finding 13). The list lives in `chelis_ir::host`
/// and this gate is consumed by both public build paths.
pub fn reject_eval_only_builtins(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    if let Some(name) = chelis_ir::host::find_eval_only_host_builtin(program) {
        // Branded through `Unsupported` (section C2,
        // spec/design/loud_unsupported.md). Both public build paths call
        // this definition, keeping their diagnostics byte-compatible. The
        // stage tag names the ACTUAL rejecting lane (round-2 red-team
        // finding: a hardcoded "c" misstated the lane on HIP builds).
        return Err(unsupported_stage_error(
            chelis_types::unsupported::Unsupported::new(
                chelis_types::unsupported::UnsupportedKind::Builtin(name.to_string()),
                "compiled targets (the host interpreter's eval/test lanes only)",
                chelis_types::unsupported::Stage::Codegen(target.as_str()),
                chelis_types::deliberate_rejection!(
                    "[05-HOST-2]",
                    "run the program with `chelis eval` or `chelis test`, or remove the \
                     call before building (spec/05-risc-primitives.md section 3.7)"
                ),
            ),
        ));
    }
    Ok(())
}

pub fn reject_host_only_builtins(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    use chelis_ir::host::{
        ConcreteHostCallback, ConcreteHostExpr, ConcreteHostExprKind, HostCallbackKind,
    };

    // A higher-order helper's callback can itself reach a host-only
    // builtin (e.g. `map(fn (x) -> tensor_scan(...), xs)`). An *inline*
    // callback carries its body inline, so we descend into it. A *named*
    // callback refers to a top-level function by name; that function's
    // body is scanned separately when we walk `program.functions`, so we
    // do not need to chase the reference here.
    fn scan_callback(callback: &ConcreteHostCallback, found: &mut Option<String>) {
        if let HostCallbackKind::Inline { body, .. } = &callback.kind {
            scan_expr(body, found);
        }
    }

    fn scan_expr(expr: &ConcreteHostExpr, found: &mut Option<String>) {
        if found.is_some() {
            return;
        }
        match &expr.kind {
            ConcreteHostExprKind::Builtin { name, args, .. } => {
                if HOST_ONLY_BUILTINS.contains(&name.as_str()) {
                    *found = Some(name.clone());
                    return;
                }
                for arg in args {
                    scan_expr(arg, found);
                }
            }
            ConcreteHostExprKind::Call { args, .. } => {
                for arg in args {
                    scan_expr(arg, found);
                }
            }
            ConcreteHostExprKind::TensorCall { args, .. } => {
                for arg in args {
                    scan_expr(arg, found);
                }
            }
            ConcreteHostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                scan_expr(cond, found);
                scan_expr(then_expr, found);
                scan_expr(else_expr, found);
            }
            ConcreteHostExprKind::Let { bindings, body, .. } => {
                for binding in bindings {
                    scan_expr(&binding.value, found);
                }
                scan_expr(body, found);
            }
            ConcreteHostExprKind::List(items, _) | ConcreteHostExprKind::Tuple(items, _) => {
                for item in items {
                    scan_expr(item, found);
                }
            }
            ConcreteHostExprKind::AdtConstruct { fields, .. } => {
                for field in fields {
                    scan_expr(field, found);
                }
            }
            ConcreteHostExprKind::AdtFieldAccess { base, .. } => scan_expr(base, found),
            ConcreteHostExprKind::MatchOption {
                scrutinee,
                some_expr,
                none_expr,
                ..
            } => {
                scan_expr(scrutinee, found);
                scan_expr(some_expr, found);
                scan_expr(none_expr, found);
            }
            ConcreteHostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ..
            } => {
                scan_expr(scrutinee, found);
                for arm in arms {
                    scan_expr(&arm.expr, found);
                }
                if let Some(d) = default_expr {
                    scan_expr(d, found);
                }
            }
            ConcreteHostExprKind::Map { callback, list, .. }
            | ConcreteHostExprKind::Filter { callback, list, .. }
            | ConcreteHostExprKind::Partition { callback, list, .. }
            | ConcreteHostExprKind::FlatMap { callback, list, .. } => {
                scan_callback(callback, found);
                scan_expr(list, found);
            }
            ConcreteHostExprKind::Fold {
                callback,
                init,
                list,
                ..
            }
            | ConcreteHostExprKind::Scan {
                callback,
                init,
                list,
                ..
            } => {
                scan_callback(callback, found);
                scan_expr(init, found);
                scan_expr(list, found);
            }
            ConcreteHostExprKind::WithSeed { seed, body, .. } => {
                scan_expr(seed, found);
                scan_expr(body, found);
            }
            _ => {}
        }
    }

    // This walk is deliberately whole-program (every global value AND
    // every function body), NOT scoped to the build entry's reachable
    // call graph. That asymmetry with the reachability-scoped AD guard
    // in `runtime.rs::find_reachable_host_only_builtin_call` is
    // intentional: `chelis_backend_c::host_emit` emits *every*
    // `program.functions` entry unconditionally (no dead-code pruning),
    // so a `tensor_scan` call inside an otherwise-unreferenced helper still
    // reaches the C emitter's fallible builtin boundary. The gate walks the
    // same emitted set to preserve the earlier, builtin-specific diagnostic;
    // it is not the sole defense. The AD guard can scope to the transform
    // target because AD lowers only that target's subgraph. If backend
    // dead-function pruning lands later, this can be narrowed to the emitted
    // set in lockstep.
    let mut found: Option<String> = None;
    for global in &program.globals {
        scan_expr(&global.value, &mut found);
        if found.is_some() {
            break;
        }
    }
    if found.is_none() {
        for function in &program.functions {
            scan_expr(&function.body, &mut found);
            if found.is_some() {
                break;
            }
        }
    }

    if let Some(name) = found {
        return Err(host_only_builtin_error(&name, target));
    }

    Ok(())
}

/// chelis#616: whether a movement `(start, end)` bound pair is node-valued.
fn pair_has_node_bound(pair: &(RtDim, RtDim)) -> bool {
    pair.0.node_input().is_some() || pair.1.node_input().is_some()
}

/// Reject effectful DAG operations whose seeded evaluator semantics do not
/// yet have a compiled-backend implementation. This is the single policy
/// consumed by compiler-api and CLI build entry points; the backend emitters
/// retain independent fail-loud defenses per [05-UNS-4].
pub fn reject_unsupported_effect_ops(
    dag: &Dag,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    for node in dag.nodes() {
        if matches!(&node.op, RiscOp::Dropout { .. }) {
            return Err(unsupported_gate_error(
                format!("compiled `dropout` op at lowered node {}", node.id.0),
                target.as_str(),
                chelis_types::unimplemented_rejection!(
                    1192,
                    "compiled `dropout` kernels are not implemented; run this program with `chelis eval`"
                ),
            ));
        }
    }
    Ok(())
}

fn for_each_host_helper_dag(
    program: &chelis_ir::host::ConcreteHostProgram,
    mut visit: impl FnMut(&Dag) -> std::result::Result<(), CompilerError>,
) -> std::result::Result<(), CompilerError> {
    for helper in &program.global_tensor_helpers {
        visit(&helper.dag)?;
    }
    for function in &program.functions {
        for helper in &function.tensor_helpers {
            visit(&helper.dag)?;
        }
    }
    Ok(())
}

/// Reject unsupported effects in every tensor-helper DAG emitted with a host
/// program. The shared traversal keeps all callers aligned on helper scope.
pub fn reject_unsupported_effect_ops_in_host_program(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    for_each_host_helper_dag(program, |dag| reject_unsupported_effect_ops(dag, target))
}

/// Apply both C windowed-reduction gates to every tensor-helper DAG emitted
/// with a host program.
pub fn reject_unsupported_windowed_reductions_in_host_program(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: BuildTarget,
) -> std::result::Result<(), CompilerError> {
    for_each_host_helper_dag(program, |dag| {
        reject_symbolic_windowed_reduce(dag, target)?;
        reject_unsupported_reduce_window_precision(dag, target)
    })
}

fn guard_count_for_device(
    dag: &Dag,
    target: &'static str,
) -> std::result::Result<(), CompilerError> {
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Count { .. }) {
            return Err(unsupported_gate_error(
                format!(
                    "`chelis build --target {target}` does not support `count`; lowered node {} requires it. chelis#1291 owns the dedicated {target} kernel; use `--target c`.",
                    node.id.0
                ),
                target,
                chelis_types::unimplemented_rejection!(
                    1291,
                    "first-class count ships on eval and C-host/C-DAG in chelis#1287; chelis#1291 owns the dedicated HIP/Metal kernels"
                ),
            ));
        }
    }
    Ok(())
}

/// Reject Count in every tensor-helper DAG emitted with a HIP host program.
/// Other helper operations retain their C-host fallback semantics; full HIP
/// capability policy applies only to DAGs emitted as HIP device code.
pub fn reject_unsupported_hip_ops_in_host_program(
    program: &chelis_ir::host::ConcreteHostProgram,
) -> std::result::Result<(), CompilerError> {
    for_each_host_helper_dag(program, |dag| guard_count_for_device(dag, "hip"))
}

/// Reject Count in every tensor-helper DAG emitted with a Metal host program.
/// Other helper operations retain their C-host fallback semantics; full Metal
/// capability policy applies only to DAGs emitted as Metal device code.
pub fn reject_unsupported_metal_ops_in_host_program(
    program: &chelis_ir::host::ConcreteHostProgram,
) -> std::result::Result<(), CompilerError> {
    for_each_host_helper_dag(program, |dag| guard_count_for_device(dag, "metal"))
}

/// Metal-specific early capability policy. The IR verifier and backend
/// emitter independently enforce the same target boundary; this shared gate
/// provides the typed public diagnostic without allowing CLI/compiler-api
/// copies to drift.
pub fn reject_unsupported_metal_ops(dag: &Dag) -> std::result::Result<(), CompilerError> {
    guard_count_for_device(dag, "metal")?;
    for node in dag.nodes() {
        let direct_arithmetic = match &node.op {
            RiscOp::Sub => Some("sub"),
            RiscOp::MaxElem => Some("max_elem"),
            RiscOp::MinElem => Some("min_elem"),
            RiscOp::ExtremaAdjoint { .. } => Some("extrema adjoint"),
            RiscOp::FusedElem { ops }
                if ops.iter().any(|step| {
                    matches!(
                        step.op,
                        FusedStepOp::Sub | FusedStepOp::MaxElem | FusedStepOp::MinElem
                    )
                }) =>
            {
                Some("fused direct arithmetic")
            }
            _ => None,
        };
        if let Some(op) = direct_arithmetic {
            return Err(unsupported_gate_error(
                format!(
                    "`chelis build --target metal` does not yet support exact `{op}` at lowered node {}; use `--target c` or `--target hip` for the implemented cells",
                    node.id.0
                ),
                "metal",
                chelis_types::unimplemented_rejection!(
                    1306,
                    "the Metal direct-subtraction/extrema kernel and exact trap/bit-selection cells are not implemented"
                ),
            ));
        }

        if matches!(&node.op, RiscOp::Expand { size, .. } if size.node_input().is_some()) {
            return Err(unsupported_gate_error(
                format!(
                    "runtime (node-valued) `expand` extent at lowered node {}",
                    node.id.0
                ),
                "metal",
                chelis_types::unimplemented_rejection!(
                    1383,
                    "the Metal device scalar path for runtime expand extents is not implemented; use `--target c`"
                ),
            ));
        }

        let node_valued = match &node.op {
            RiscOp::Shrink { bounds } => bounds.iter().any(pair_has_node_bound),
            RiscOp::Pad { padding, .. } => padding.iter().any(pair_has_node_bound),
            RiscOp::Stride { strides } => {
                strides.iter().any(|stride| stride.node_input().is_some())
            }
            RiscOp::Reshape { new_shape } => new_shape.iter().any(|dim| dim.node_input().is_some()),
            _ => false,
        };
        if node_valued {
            return Err(unsupported_gate_error(
                format!(
                    "runtime (node-valued) movement bound or reshape target extent at lowered node {}",
                    node.id.0
                ),
                "metal",
                chelis_types::deliberate_rejection!(
                    "[05-MOV-1]",
                    "runtime movement bounds and reshape targets are defined on eval and C; use `--target c`"
                ),
            ));
        }

        match node.output_type.precision {
            chelis_types::types::Prim::F32
            | chelis_types::types::Prim::F16
            | chelis_types::types::Prim::Bf16
            | chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int16
            | chelis_types::types::Prim::Int32
            | chelis_types::types::Prim::Int64
            | chelis_types::types::Prim::Bool => {}
            chelis_types::types::Prim::F64 => {
                return Err(unsupported_gate_error(
                    format!("f64 value at lowered node {}", node.id.0),
                    "metal",
                    chelis_types::deliberate_rejection!(
                        "[04-TGT-1]",
                        "Apple Silicon GPUs lack FP64 ALUs; use `--target c` or `--target hip` for f64 workloads"
                    ),
                ));
            }
            other => {
                return Err(unsupported_gate_error(
                    format!(
                        "tensor precision `{}` at lowered node {}",
                        other.name(),
                        node.id.0
                    ),
                    "metal",
                    chelis_types::unimplemented_rejection!(
                        729,
                        "the Metal target dtype capability cell is not implemented; supported: f32/f16/bf16/int8/int16/int32/int64/bool (spec/04-type-system.md §1.1.3)"
                    ),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod metal_runtime_dim_reject_tests {
    use super::reject_unsupported_metal_ops;
    use chelis_ir::dag::{
        Dag, DimInfo, ExtremaKind, ExtremaOperand, FusedInput, FusedStep, FusedStepOp, NodeId,
        RiscOp, RtAxis, RtDim, TensorType,
    };
    use chelis_types::types::Prim;

    fn ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    fn dag_with_scalar() -> (Dag, NodeId, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&[4], Prim::F32),
            None,
        );
        let m = dag.add_node(
            RiscOp::Load { name: "m".into() },
            vec![],
            ty(&[], Prim::Int64),
            None,
        );
        (dag, x, m)
    }

    #[test]
    fn metal_seam_rejects_node_valued_shrink_bound() {
        let (mut dag, x, m) = dag_with_scalar();
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Node(1))],
            },
            vec![x, m],
            ty(&[4], Prim::F32),
            None,
        );
        let error = reject_unsupported_metal_ops(&dag)
            .expect_err("Metal seam must reject a node-valued shrink bound");
        let message = &error.errors[0].message;
        assert!(message.contains("--target c"), "{message}");
        assert!(message.contains("deliberate [05-MOV-1]"), "{message}");
    }

    #[test]
    fn metal_seam_rejects_node_valued_reshape_target() {
        let (mut dag, x, m) = dag_with_scalar();
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1)],
            },
            vec![x, m],
            ty(&[4], Prim::F32),
            None,
        );
        let error = reject_unsupported_metal_ops(&dag)
            .expect_err("Metal seam must reject a node-valued reshape target");
        let message = &error.errors[0].message;
        assert!(message.contains("--target c"), "{message}");
        assert!(message.contains("deliberate [05-MOV-1]"), "{message}");
    }

    #[test]
    fn metal_seam_accepts_literal_movement_and_reshape() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&[4], Prim::F32),
            None,
        );
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Lit(2))],
            },
            vec![x],
            ty(&[2], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(2), RtDim::Lit(1)],
            },
            vec![shrunk],
            ty(&[2, 1], Prim::F32),
            None,
        );
        reject_unsupported_metal_ops(&dag).expect("literal bounds must pass the Metal seam");
    }

    #[test]
    fn metal_seam_accepts_input_axis_expand_extent() {
        let mut dag = Dag::new();
        let value = dag.add_node(
            RiscOp::Load {
                name: "value".into(),
            },
            vec![],
            ty(&[], Prim::F32),
            None,
        );
        let witness = dag.add_node(
            RiscOp::Load {
                name: "witness".into(),
            },
            vec![],
            ty(&[4], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![value, witness],
            ty(&[4], Prim::F32),
            None,
        );

        reject_unsupported_metal_ops(&dag)
            .expect("InputAxis is a metadata read admitted by the Metal capability seam");
    }

    #[test]
    fn metal_seam_rejects_node_valued_expand_with_issue_1383_receipt() {
        let (mut dag, x, size) = dag_with_scalar();
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Node(1),
            },
            vec![x, size],
            ty(&[4, 4], Prim::F32),
            None,
        );

        let error = reject_unsupported_metal_ops(&dag)
            .expect_err("Metal must reject a device scalar expand extent");
        let message = &error.errors[0].message;
        assert!(message.contains("unimplemented chelis#1383:"), "{message}");
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
    }

    #[test]
    fn metal_seam_rejects_count_with_issue_1291_receipt() {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&[2, 3], Prim::Bool),
            None,
        );
        dag.add_node(
            RiscOp::Count { axes: vec![1] },
            vec![input],
            ty(&[2], Prim::Int64),
            None,
        );

        let error = reject_unsupported_metal_ops(&dag)
            .expect_err("Metal must reject Count until its dedicated kernel lands");
        let message = &error.errors[0].message;
        assert!(message.contains("unimplemented chelis#1291:"), "{message}");
        assert!(
            message.contains("count") && message.contains("--target c"),
            "{message}"
        );
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
    }

    fn direct_arithmetic_dag(op: RiscOp) -> Dag {
        let mut dag = Dag::new();
        let lhs = dag.add_node(
            RiscOp::Load { name: "lhs".into() },
            vec![],
            ty(&[4], Prim::F32),
            None,
        );
        let rhs = dag.add_node(
            RiscOp::Load { name: "rhs".into() },
            vec![],
            ty(&[4], Prim::F32),
            None,
        );
        let gradient = dag.add_node(
            RiscOp::Load {
                name: "gradient".into(),
            },
            vec![],
            ty(&[4], Prim::F32),
            None,
        );
        let inputs = match op {
            RiscOp::Relu => vec![lhs],
            RiscOp::ReluAdjoint => vec![lhs, gradient],
            RiscOp::ExtremaAdjoint { .. } => vec![lhs, rhs, gradient],
            _ => vec![lhs, rhs],
        };
        dag.add_node(op, inputs, ty(&[4], Prim::F32), None);
        dag
    }

    #[test]
    fn metal_seam_rejects_every_direct_arithmetic_identity_with_issue_1306() {
        let ops = [
            RiscOp::Sub,
            RiscOp::MaxElem,
            RiscOp::MinElem,
            RiscOp::ExtremaAdjoint {
                kind: ExtremaKind::Max,
                operand: ExtremaOperand::Left,
            },
            RiscOp::ExtremaAdjoint {
                kind: ExtremaKind::Min,
                operand: ExtremaOperand::Right,
            },
            RiscOp::FusedElem {
                ops: vec![FusedStep {
                    op: FusedStepOp::Sub,
                    input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
                }],
            },
            RiscOp::FusedElem {
                ops: vec![FusedStep {
                    op: FusedStepOp::MinElem,
                    input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
                }],
            },
        ];

        for op in ops {
            let dag = direct_arithmetic_dag(op);
            let error = reject_unsupported_metal_ops(&dag)
                .expect_err("Metal must reject every unimplemented direct arithmetic identity");
            let message = &error.errors[0].message;
            assert!(message.contains("unimplemented chelis#1306:"), "{message}");
        }
    }

    #[test]
    fn metal_seam_accepts_dedicated_relu() {
        for op in [RiscOp::Relu, RiscOp::ReluAdjoint] {
            reject_unsupported_metal_ops(&direct_arithmetic_dag(op))
                .expect("Metal ReLU must reach the typed kernel");
        }
    }
}

pub fn reject_unsupported_hip_ops(dag: &Dag) -> std::result::Result<(), CompilerError> {
    guard_count_for_device(dag, "hip")?;
    for node in dag.nodes() {
        let fused_direct_ops = match &node.op {
            RiscOp::FusedElem { ops } => Some(ops),
            _ => None,
        };
        let has_direct_sub = matches!(node.op, RiscOp::Sub)
            || fused_direct_ops
                .is_some_and(|ops| ops.iter().any(|step| matches!(step.op, FusedStepOp::Sub)));
        let has_direct_arithmetic = matches!(
            node.op,
            RiscOp::Sub | RiscOp::MaxElem | RiscOp::MinElem | RiscOp::ExtremaAdjoint { .. }
        ) || fused_direct_ops.is_some_and(|ops| {
            ops.iter().any(|step| {
                matches!(
                    step.op,
                    FusedStepOp::Sub | FusedStepOp::MaxElem | FusedStepOp::MinElem
                )
            })
        });

        if has_direct_sub && node.output_type.precision.is_integer() {
            return Err(unsupported_gate_error(
                format!(
                    "`chelis build --target hip` cannot execute checked `{}` subtraction at lowered node {} without a device numeric-trap channel; use `--target c`",
                    node.output_type.precision.name(),
                    node.id.0
                ),
                "hip",
                chelis_types::unimplemented_rejection!(
                    1306,
                    "checked signed-integer subtraction needs an exact HIP overflow-trap channel; the C target implements this cell"
                ),
            ));
        }
        if has_direct_arithmetic
            && matches!(
                node.output_type.precision,
                chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16
            )
        {
            return Err(unsupported_gate_error(
                format!(
                    "`chelis build --target hip` does not yet support exact `{}` direct arithmetic at lowered node {}",
                    node.output_type.precision.name(),
                    node.id.0
                ),
                "hip",
                chelis_types::unimplemented_rejection!(
                    1306,
                    "the HIP bf16/f16 direct-subtraction/extrema bit-preserving kernels are not implemented"
                ),
            ));
        }

        match &node.op {
            // `pad` / `shrink` are now implemented on the HIP backend
            // (typed per-output-element kernels, GPU==eval verified by the
            // `gpu_correctness` manual oracle). No reject arm: they fall
            // through to codegen.

            // `reduce_window_*` HIP codegen is excluded by [05-RWIN-2]. Reject
            // it cleanly here rather than letting it reach the launch-emit
            // `todo!`, which would abort the build with an `internal error`
            // panic. The C backend is canonical; use `--target c`.
            // [05-OP-6] demands identical eval-vs-compiled behavior, and
            // the HIP `cast` kernel is a raw device-side C++ conversion
            // with no trap guard at all. Emitting `cast_trunc` through it
            // would silently skip the Domain/Overflow traps, so the HIP
            // lane rejects loudly until the guarded kernels land.
            RiscOp::CastTrunc { .. } => {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target hip` does not support `cast_trunc`; \
                         lowered node {} requires it. The HIP cast kernels carry no \
                         numeric-trap guard, so the [05-OP-6] Domain/Overflow traps \
                         cannot be honored on device yet; use `--target c`.",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::unimplemented_rejection!(
                        759,
                        "the HIP cast kernels emit an unguarded device-side conversion, \
                         so the [05-OP-6] traps have no device implementation; the C \
                         target is canonical for the named cast ladder"
                    ),
                ));
            }
            RiscOp::ReduceWindow { .. } => {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target hip` does not support `reduce_window_*`; \
                         lowered node {} requires it. HIP windowed-reduction codegen is excluded \
                         (spec/05-risc-primitives.md §2.3.1); use `--target c`.",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-RWIN-2]",
                        "windowed reduction kernels are intentionally excluded from the HIP target; use the C target"
                    ),
                ));
            }
            RiscOp::ReduceWindowGrad { .. } => {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target hip` does not support the `reduce_window_*` \
                         adjoint (`ReduceWindowGrad`); lowered node {} requires it. HIP windowed-reduction codegen is \
                         excluded (spec/05-risc-primitives.md §2.3.1); use `--target c`.",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-RWIN-2]",
                        "windowed reduction adjoint kernels are intentionally excluded from the HIP target; use the C target"
                    ),
                ));
            }
            RiscOp::OneHot { .. } => {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target hip` cannot compile internal one_hot node {}: \
                         the sparse gather recognizer must consume OneHot before backend emission",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-SPARSE-2]",
                        "OneHot is internal-only and must be consumed before backend emission"
                    ),
                ));
            }
            RiscOp::Shape { .. }
            | RiscOp::ExtentWitness { .. }
            | RiscOp::CheckedReshapeExtent { .. }
            | RiscOp::CheckedUnitAxis { .. } => {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target hip` does not support the runtime `shape` \
                         value read; lowered node {} requires it. The C backend is canonical \
                         for runtime-dim reads (spec/05-risc-primitives.md [05-SHAPE-1]); use `--target c`.",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-SHAPE-1]",
                        "runtime shape-value reads are intentionally excluded from the HIP device lane; use the C target"
                    ),
                ));
            }
            RiscOp::Expand { size, .. } if size.node_input().is_some() => {
                return Err(unsupported_gate_error(
                    format!(
                        "runtime (node-valued) `expand` extent at lowered node {}",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::unimplemented_rejection!(
                        1298,
                        "the HIP device scalar path for runtime movement bounds is not implemented; use `--target c`"
                    ),
                ));
            }
            // [05-MOV-1]: node-valued (runtime) movement bounds are C-only.
            RiscOp::Shrink { bounds } if bounds.iter().any(pair_has_node_bound) => {
                return Err(unsupported_gate_error(
                    format!(
                        "runtime (node-valued) `shrink` bound at lowered node {}",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-MOV-1]",
                        "runtime movement bounds are defined on eval and C; use `--target c`"
                    ),
                ));
            }
            RiscOp::Pad { padding, .. } if padding.iter().any(pair_has_node_bound) => {
                return Err(unsupported_gate_error(
                    format!(
                        "runtime (node-valued) `pad` bound at lowered node {}",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-MOV-1]",
                        "runtime movement bounds are defined on eval and C; use `--target c`"
                    ),
                ));
            }
            RiscOp::Stride { strides } if strides.iter().any(|s| s.node_input().is_some()) => {
                return Err(unsupported_gate_error(
                    format!(
                        "runtime (node-valued) `stride` step at lowered node {}",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-MOV-1]",
                        "runtime movement bounds are defined on eval and C; use `--target c`"
                    ),
                ));
            }
            RiscOp::Reshape { new_shape } if new_shape.iter().any(|d| d.node_input().is_some()) => {
                return Err(unsupported_gate_error(
                    format!(
                        "runtime (node-valued) `reshape` target extent at lowered node {}",
                        node.id.0
                    ),
                    "hip",
                    chelis_types::deliberate_rejection!(
                        "[05-MOV-1]",
                        "runtime reshape targets are defined on eval and C; use `--target c`"
                    ),
                ));
            }
            RiscOp::Gather { .. } => {
                let values = &dag.get(node.inputs[0]).unwrap().output_type;
                let index_node = dag.get(node.inputs[1]).unwrap();
                let indices = &index_node.output_type;
                if values.precision != chelis_types::types::Prim::F32
                    || node.output_type.precision != chelis_types::types::Prim::F32
                {
                    return Err(unsupported_gate_error(
                        format!(
                            "`chelis build --target hip` sparse gather supports f32 payloads only; \
                             node {} carries payload precision `{}` and output precision `{}`",
                            node.id.0,
                            values.precision.name(),
                            node.output_type.precision.name()
                        ),
                        "hip",
                        chelis_types::unimplemented_rejection!(
                            729,
                            "the HIP sparse payload dtype cell is not implemented"
                        ),
                    ));
                }
                if !matches!(
                    indices.precision,
                    chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                ) {
                    return Err(unsupported_gate_error(
                        format!(
                            "`chelis build --target hip` sparse gather requires int32/int64 indices; \
                             node {} uses `{}`",
                            node.id.0,
                            indices.precision.name()
                        ),
                        "hip",
                        chelis_types::deliberate_rejection!(
                            "[05-SPARSE-1]",
                            "sparse indices must use the specified int32 or int64 dtype"
                        ),
                    ));
                }
                if !matches!(index_node.op, RiscOp::Load { .. }) {
                    return Err(unsupported_gate_error(
                        format!(
                            "`chelis build --target hip` sparse gather requires indices to be loaded input tensors in this milestone; \
                             node {} uses indices produced by {:?}. \
                             Non-load integer index producers need integer HIP codegen before they can feed sparse kernels safely.",
                            node.id.0, index_node.op
                        ),
                        "hip",
                        chelis_types::unimplemented_rejection!(
                            729,
                            "non-load sparse index producers need typed integer HIP kernels"
                        ),
                    ));
                }
            }
            RiscOp::ScatterAdd { .. } | RiscOp::Scatter { .. } | RiscOp::ScatterElements { .. } => {
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
                    RiscOp::ScatterElements { .. } => (
                        "scatter_elements",
                        // Element-wise scatter shares the serial
                        // last-write-wins kernel limitation; f64 is a
                        // future widening, not an atomics question.
                        "f64 scatter_elements requires a widened serial last-write-wins kernel and is not in this milestone.",
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
                    return Err(unsupported_gate_error(
                        format!(
                            "`chelis build --target hip` sparse {label} supports f32 payloads only; \
                             node {} carries target `{}`, updates `{}`, output `{}`. \
                             {payload_blocker}",
                            node.id.0,
                            target.precision.name(),
                            updates.precision.name(),
                            node.output_type.precision.name()
                        ),
                        "hip",
                        chelis_types::unimplemented_rejection!(
                            729,
                            "the HIP sparse payload dtype cell is not implemented"
                        ),
                    ));
                }
                if !matches!(
                    indices.precision,
                    chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                ) {
                    return Err(unsupported_gate_error(
                        format!(
                            "`chelis build --target hip` sparse {label} requires int32/int64 indices; \
                             node {} uses `{}`",
                            node.id.0,
                            indices.precision.name()
                        ),
                        "hip",
                        chelis_types::deliberate_rejection!(
                            "[05-SPARSE-1]",
                            "sparse indices must use the specified int32 or int64 dtype"
                        ),
                    ));
                }
                if !matches!(index_node.op, RiscOp::Load { .. }) {
                    return Err(unsupported_gate_error(
                        format!(
                            "`chelis build --target hip` sparse {label} requires indices to be loaded input tensors in this milestone; \
                             node {} uses indices produced by {:?}. \
                             Non-load integer index producers need integer HIP codegen before they can feed sparse kernels safely.",
                            node.id.0, index_node.op
                        ),
                        "hip",
                        chelis_types::unimplemented_rejection!(
                            729,
                            "non-load sparse index producers need typed integer HIP kernels"
                        ),
                    ));
                }
            }
            _ => {}
        }
    }

    // The shared gate follows the backend's exact dtype surface. f64 and
    // the integer family have typed kernel templates. bf16/f16 are narrower:
    // storage and hipBLAS matmul are implemented, and [05-OP-43]'s dedicated
    // ReLU identities have exact raw-bit kernels. Other compute nodes would
    // still reach an unsupported narrow-float path.
    let narrow_float_admissible: UnordSet<NodeId> = dag
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                node.output_type.precision,
                chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16
            )
        })
        .filter_map(|node| match &node.op {
            RiscOp::Load { .. }
            | RiscOp::Store { .. }
            | RiscOp::BlasMatmul { .. }
            | RiscOp::Relu
            | RiscOp::ReluAdjoint => Some(node.id),
            _ => None,
        })
        .collect();

    for node in dag.nodes() {
        match node.output_type.precision {
            chelis_types::types::Prim::F32
            | chelis_types::types::Prim::F64
            | chelis_types::types::Prim::Bool
            | chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int16
            | chelis_types::types::Prim::Int32
            | chelis_types::types::Prim::Int64 => {}
            chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16 => {
                if !narrow_float_admissible.contains(&node.id) {
                    let authority = if node.output_type.precision == chelis_types::types::Prim::F16
                    {
                        chelis_types::unimplemented_rejection!(
                            729,
                            "`f16` is implemented only for HIP tensor load/store, `BlasMatmul`, and the dedicated [05-OP-43] ReLU identities; this operation needs a typed bf16/f16 kernel (spec/04-type-system.md §5.7.1)"
                        )
                    } else {
                        chelis_types::unimplemented_rejection!(
                            729,
                            "`bf16` is implemented only for HIP tensor load/store, `BlasMatmul`, and the dedicated [05-OP-43] ReLU identities; this operation needs a typed bf16/f16 kernel (spec/04-type-system.md §5.7.1)"
                        )
                    };
                    return Err(unsupported_gate_error(
                        format!(
                            "narrow-float compute at lowered node {} (`{:?}` with `{}`)",
                            node.id.0,
                            node.op,
                            node.output_type.precision.name(),
                        ),
                        "hip",
                        authority,
                    ));
                }
            }
            other => {
                return Err(unsupported_gate_error(
                    format!(
                        "`chelis build --target hip` DAG path does not support tensor precision \
                         `{}` (node {}). Supported: f32/f64/bool plus the integer family \
                         (int8/int16/int32/int64), with bf16/f16 admitted on matmul, \
                         load/store, and dedicated ReLU nodes. See \
                         spec/04-type-system.md §5.7.1.",
                        other.name(),
                        node.id.0,
                    ),
                    "hip",
                    chelis_types::unimplemented_rejection!(
                        729,
                        "the HIP target dtype capability cell is not implemented"
                    ),
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
        GeneralKind::UnknownName,
    )
}

/// Lift an eval-lane failure message into a `CompilerError` at the one
/// boundary where lane-internal text becomes a structured diagnostic.
/// Cancellation gets its dedicated kind; genuine evaluation failures retain
/// the recovery suggestions attached by their owning diagnostic rules.
fn eval_stage_error(message: String) -> CompilerError {
    let kind = if chelis_types::is_cancellation(&message) {
        GeneralKind::Cancelled
    } else {
        GeneralKind::EvalError
    };
    let cast_domain = message.contains("numeric trap: domain in cast at");
    let mut error = stage_error("eval", message, kind);
    if cast_domain && let Some(diagnostic) = error.errors.first_mut() {
        diagnostic.suggestions.push(
            "fractional float-to-int conversion must state its rounding explicitly: \
             use `cast_trunc` to truncate toward zero ([05-OP-6]), or apply \
             `floor` or `round` before `cast`; the remaining named lossy cast \
             forms are tracked by chelis#759"
                .to_string(),
        );
    }
    error
}

/// Lift a [`WireDagSchemaError`] from validating a `WireDag` at a
/// process boundary into a typed `schema`-stage [`CompilerError`] (WI-2).
/// An unsupported schema version is rejected here -- surfaced to the client
/// as a normal failure envelope -- rather than panicking or silently handing
/// back a DAG whose surface this build cannot trust.
pub(crate) fn schema_stage_check(
    result: std::result::Result<(), WireDagSchemaError>,
) -> Result<()> {
    result.map_err(|err| {
        let (found, supported) = match &err {
            WireDagSchemaError::MissingSchemaVersion { supported } => {
                ("missing".to_string(), *supported)
            }
            WireDagSchemaError::UnsupportedSchemaVersion { found, supported } => {
                (found.to_string(), *supported)
            }
        };
        let mut error = stage_error("schema", err.to_string(), GeneralKind::UnknownSchemaVersion);
        if let Some(diagnostic) = error.errors.first_mut() {
            diagnostic.expected = Some(format!("schema_version = {supported}"));
            diagnostic.got = Some(found);
        }
        error
    })
}

/// A Deep node span as a DIAGNOSTIC location (chelis#1395).
///
/// Always a `Range`: `chelis_deep::Span` carries both an offset and a length,
/// so this producer never has to report a bare coordinate. The AST wire types
/// keep their own converter, `span`, because their spans are structurally
/// ranges and stay on `Span`.
fn deep_span_to_diagnostic(span: Option<chelis_deep::Span>) -> Option<DiagnosticSpan> {
    span.map(|span| DiagnosticSpan::Range {
        offset: crate::schema::host_index(span.offset),
        len: crate::schema::host_index(span.len),
    })
}

/// The byte offset a Surf lexer error carries.
///
/// Exhaustive by construction rather than a catch-all: a new `LexError`
/// variant stops this compiling until its coordinate is chosen, which is what
/// the previous `return None` arm silently avoided (chelis#1395). Every
/// variant carries one, so the return type is `usize`, not `Option`.
fn surf_lex_error_offset(err: &chelis_surf::lexer::LexError) -> usize {
    use chelis_surf::lexer::LexError as Lex;
    match err {
        Lex::UnterminatedString { offset }
        | Lex::InvalidEscape { offset, .. }
        | Lex::UnescapedControl { offset, .. }
        | Lex::InvalidNumber { offset, .. }
        | Lex::UnexpectedChar { offset, .. }
        | Lex::ReservedForFuture { offset, .. }
        | Lex::UnterminatedBlockComment { offset }
        | Lex::DeferredSuffix { offset, .. }
        | Lex::UnsignedSuffix { offset, .. }
        | Lex::IntegerSuffixOnFloat { offset, .. }
        | Lex::HexFloatSuffix { offset, .. }
        | Lex::UnknownSuffix { offset, .. } => *offset,
    }
}

/// The byte offset a Deep lexer error carries. See `surf_lex_error_offset`.
fn deep_lex_error_offset(err: &chelis_deep::lexer::LexError) -> usize {
    use chelis_deep::lexer::LexError as Lex;
    match err {
        Lex::UnterminatedString { offset }
        | Lex::InvalidEscape { offset, .. }
        | Lex::InvalidNumber { offset, .. }
        | Lex::UnexpectedChar { offset, .. }
        | Lex::DeferredSuffix { offset, .. }
        | Lex::UnsignedSuffix { offset, .. }
        | Lex::IntegerSuffixOnFloat { offset, .. }
        | Lex::HexFloatSuffix { offset, .. }
        | Lex::UnknownSuffix { offset, .. } => *offset,
    }
}

fn parse_error_span_surf(source: &str, err: &chelis_surf::parser::ParseError) -> DiagnosticSpan {
    let offset = match err {
        chelis_surf::parser::ParseError::Lex(lex) => surf_lex_error_offset(lex),
        chelis_surf::parser::ParseError::UnexpectedEof => source.len(),
        chelis_surf::parser::ParseError::Expected { offset, .. }
        | chelis_surf::parser::ParseError::ReservedWordBinding { offset, .. }
        | chelis_surf::parser::ParseError::NonAssocChain { offset }
        | chelis_surf::parser::ParseError::BareStatementInBlock { offset }
        | chelis_surf::parser::ParseError::SemicolonBlockSeparator { offset }
        | chelis_surf::parser::ParseError::NonCanonicalLiteral { offset, .. }
        | chelis_surf::parser::ParseError::NonFiniteLiteral { offset, .. }
        | chelis_surf::parser::ParseError::SignedMinimumMagnitudeRequiresNegation {
            offset, ..
        } => *offset,
    };
    DiagnosticSpan::Point {
        offset: crate::schema::host_index(offset),
    }
}

fn parse_error_span_deep(err: &chelis_deep::parser::ParseError) -> DiagnosticSpan {
    let offset = match err {
        chelis_deep::parser::ParseError::Lex(lex) => deep_lex_error_offset(lex),
        chelis_deep::parser::ParseError::UnexpectedEof { offset }
        | chelis_deep::parser::ParseError::Expected { offset, .. }
        | chelis_deep::parser::ParseError::EmptyList { offset } => *offset,
        chelis_deep::parser::ParseError::ForbiddenSpanChar { value_offset, .. } => *value_offset,
        chelis_deep::parser::ParseError::Metadata(error) => error.span.offset,
    };
    DiagnosticSpan::Point {
        offset: crate::schema::host_index(offset),
    }
}

/// Project a check diagnostic onto the wire carrier.
///
/// chelis#886 consolidated this with the CLI report's producer: there were
/// two independent 24-arm projections of the same type onto the same
/// carrier, which is the drift the issue exists to remove. The one
/// implementation lives beside `Diagnostic`; this keeps the name its callers
/// already use.
pub(crate) fn check_errors_to_compiler_error(stage: &str, errors: &[CheckError]) -> CompilerError {
    match errors
        .iter()
        .map(Diagnostic::try_from_check_error)
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(errors) => CompilerError {
            transcript: Vec::new(),
            stage: stage.to_string(),
            errors,
        },
        Err(error) => stage_error("report", error, GeneralKind::Other),
    }
}

fn span(span: chelis_deep::Span) -> Span {
    Span {
        offset: crate::schema::host_index(span.offset),
        len: crate::schema::host_index(span.len),
    }
}

fn wire_decl(decl: &Decl) -> SourceWireResult<WireSurfDecl> {
    Ok(match decl {
        Decl::Module {
            name,
            decls,
            span: s,
        } => WireSurfDecl::Module {
            name: name.clone(),
            decls: decls
                .iter()
                .map(wire_decl)
                .collect::<SourceWireResult<_>>()?,
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
            opaque,
            invariant,
            span: s,
        } => WireSurfDecl::TypeDef {
            name: name.clone(),
            params: params.clone(),
            variants: variants.iter().map(wire_variant).collect(),
            opaque: *opaque,
            invariant: invariant
                .as_ref()
                .map(|inv| -> SourceWireResult<_> {
                    Ok(WireTypeInvariant {
                        binder: inv.binder.clone(),
                        body: wire_expr(&inv.body)?,
                        span: span(inv.span),
                    })
                })
                .transpose()?,
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
            body: wire_expr(body)?,
            span: span(*s),
        },
        Decl::FunDef {
            name,
            type_binders,
            params,
            ret_ty,
            body,
            span: s,
            ..
        } => WireSurfDecl::FunDef {
            name: name.clone(),
            type_binders: type_binders
                .iter()
                .map(|binder| crate::schema::WireTypeBinder {
                    name: binder.name.clone(),
                    bound: binder.bound.map(|family| family.surf_name().to_string()),
                })
                .collect(),
            params: params.iter().map(wire_param).collect(),
            ret_ty: ret_ty.as_ref().map(wire_type_expr),
            body: wire_expr(body)?,
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
            preconditions: preconditions
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            body: wire_expr(body)?,
            options: options
                .iter()
                .map(wire_property_option)
                .collect::<SourceWireResult<_>>()?,
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
            value: wire_expr(value)?,
            span: span(*s),
        },
        Decl::Export { names, span: s } => WireSurfDecl::Export {
            names: names.clone(),
            span: span(*s),
        },
    })
}

fn wire_property_option(
    option: &chelis_surf::ast::PropertyOption,
) -> SourceWireResult<WirePropertyOption> {
    Ok(match option {
        chelis_surf::ast::PropertyOption::Tolerance(value, s) => WirePropertyOption::Tolerance {
            value: wire_expr(value)?,
            span: span(*s),
        },
        chelis_surf::ast::PropertyOption::Seed(value, s) => WirePropertyOption::Seed {
            value: wire_expr(value)?,
            span: span(*s),
        },
        chelis_surf::ast::PropertyOption::Samples(value, s) => WirePropertyOption::Samples {
            value: wire_expr(value)?,
            span: span(*s),
        },
        chelis_surf::ast::PropertyOption::Contract(id, s) => WirePropertyOption::Contract {
            id: id.clone(),
            span: span(*s),
        },
    })
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

fn wire_expr(expr: &Expr) -> SourceWireResult<WireSurfExpr> {
    Ok(match expr {
        Expr::Lit(lit, s) => WireSurfExpr::Lit {
            literal: wire_literal(lit)?,
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
            func: Box::new(wire_expr(func)?),
            args: args
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::List(items, s) => WireSurfExpr::List {
            items: items
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::Record(name, fields, s) => WireSurfExpr::Record {
            name: name.clone(),
            fields: fields
                .iter()
                .map(|(field, value)| {
                    Ok(WireRecordExprField {
                        name: field.clone(),
                        value: wire_expr(value)?,
                    })
                })
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::RecordUpdate(base, fields, s) => WireSurfExpr::RecordUpdate {
            base: Box::new(wire_expr(base)?),
            fields: fields
                .iter()
                .map(|(field, value)| {
                    Ok(WireRecordExprField {
                        name: field.clone(),
                        value: wire_expr(value)?,
                    })
                })
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::Access(inner, field, s) => WireSurfExpr::Access {
            expr: Box::new(wire_expr(inner)?),
            field: field.clone(),
            span: span(*s),
        },
        Expr::TupleGet(inner, index, s) => WireSurfExpr::TupleGet {
            expr: Box::new(wire_expr(inner)?),
            index: SourceInteger::new(*index),
            span: span(*s),
        },
        Expr::Binary(op, lhs, rhs, s) => WireSurfExpr::Binary {
            op: wire_bin_op(*op),
            lhs: Box::new(wire_expr(lhs)?),
            rhs: Box::new(wire_expr(rhs)?),
            span: span(*s),
        },
        Expr::Unary(op, inner, s) => WireSurfExpr::Unary {
            op: wire_unary_op(*op),
            expr: Box::new(wire_expr(inner)?),
            span: span(*s),
        },
        Expr::Pipe(inner, stages, s) => WireSurfExpr::Pipe {
            expr: Box::new(wire_expr(inner)?),
            stages: stages
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::If(cond, then_branch, else_branch, s) => WireSurfExpr::If {
            cond: Box::new(wire_expr(cond)?),
            then_branch: Box::new(wire_expr(then_branch)?),
            else_branch: Box::new(wire_expr(else_branch)?),
            span: span(*s),
        },
        Expr::Match(inner, arms, s) => WireSurfExpr::Match {
            expr: Box::new(wire_expr(inner)?),
            arms: arms
                .iter()
                .map(wire_match_arm)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::Lambda(params, body, s) => WireSurfExpr::Lambda {
            params: params.iter().map(wire_param).collect(),
            body: Box::new(wire_expr(body)?),
            span: span(*s),
        },
        Expr::Tuple(items, s) => WireSurfExpr::Tuple {
            items: items
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::Cast(inner, ty, mode, s) => WireSurfExpr::Cast {
            expr: Box::new(wire_expr(inner)?),
            ty: ty.clone(),
            mode: mode.deep_selector().map(str::to_string),
            span: span(*s),
        },
        Expr::Grad(inner, wrt, s) => WireSurfExpr::Grad {
            expr: Box::new(wire_expr(inner)?),
            wrt: wrt.clone(),
            span: span(*s),
        },
        Expr::Vmap(inner, axis, s) => WireSurfExpr::Vmap {
            expr: Box::new(wire_expr(inner)?),
            axis: axis.map(SourceInteger::new),
            span: span(*s),
        },
        Expr::Jit(inner, s) => WireSurfExpr::Jit {
            expr: Box::new(wire_expr(inner)?),
            span: span(*s),
        },
        Expr::Realize(inner, s) => WireSurfExpr::Realize {
            expr: Box::new(wire_expr(inner)?),
            span: span(*s),
        },
        Expr::Copy(inner, s) => WireSurfExpr::Copy {
            expr: Box::new(wire_expr(inner)?),
            span: span(*s),
        },
        Expr::Borrow(inner, s) => WireSurfExpr::Borrow {
            expr: Box::new(wire_expr(inner)?),
            span: span(*s),
        },
        Expr::WithSeed(seed, body, s) => WireSurfExpr::WithSeed {
            seed: Box::new(wire_expr(seed)?),
            body: Box::new(wire_expr(body)?),
            span: span(*s),
        },
        Expr::WithDevice(device, body, s) => WireSurfExpr::WithDevice {
            device: Box::new(wire_expr(device)?),
            body: Box::new(wire_expr(body)?),
            span: span(*s),
        },
        Expr::Par(exprs, s) => WireSurfExpr::Par {
            exprs: exprs
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::Do(exprs, s) => WireSurfExpr::Do {
            exprs: exprs
                .iter()
                .map(wire_expr)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Expr::Quote(expr, s) => WireSurfExpr::Quote {
            expr: Box::new(wire_expr(expr)?),
            span: span(*s),
        },
        Expr::Unquote(expr, s) => WireSurfExpr::Unquote {
            expr: Box::new(wire_expr(expr)?),
            span: span(*s),
        },
        Expr::Splice(expr, s) => WireSurfExpr::Splice {
            expr: Box::new(wire_expr(expr)?),
            span: span(*s),
        },
        Expr::Annotate(inner, ty, s) => WireSurfExpr::Annotate {
            expr: Box::new(wire_expr(inner)?),
            ty: wire_type_expr(ty),
            span: span(*s),
        },
        Expr::Block(bindings, body, s) => WireSurfExpr::Block {
            bindings: bindings
                .iter()
                .map(wire_let_binding)
                .collect::<SourceWireResult<_>>()?,
            body: Box::new(wire_expr(body)?),
            span: span(*s),
        },
    })
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

fn wire_match_arm(arm: &MatchArm) -> SourceWireResult<WireMatchArm> {
    Ok(WireMatchArm {
        pattern: wire_pattern(&arm.pattern)?,
        guard: arm.guard.as_ref().map(wire_expr).transpose()?,
        body: wire_expr(&arm.body)?,
        span: span(arm.span),
    })
}

fn wire_pattern(pattern: &Pattern) -> SourceWireResult<WirePattern> {
    Ok(match pattern {
        Pattern::Wildcard(s) => WirePattern::Wildcard { span: span(*s) },
        Pattern::Var(name, s) => WirePattern::Var {
            name: name.clone(),
            span: span(*s),
        },
        Pattern::Lit(lit, s) => WirePattern::Lit {
            literal: wire_literal(lit)?,
            span: span(*s),
        },
        Pattern::Constructor(name, args, s) => WirePattern::Constructor {
            name: name.clone(),
            args: args
                .iter()
                .map(wire_pattern)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Pattern::Tuple(items, s) => WirePattern::Tuple {
            items: items
                .iter()
                .map(wire_pattern)
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Pattern::Record(name, fields, s) => WirePattern::Record {
            name: name.clone(),
            fields: fields
                .iter()
                .map(|(name, pattern)| {
                    Ok(WireRecordPatternField {
                        name: name.clone(),
                        pattern: wire_pattern(pattern)?,
                    })
                })
                .collect::<SourceWireResult<_>>()?,
            span: span(*s),
        },
        Pattern::As(name, pattern, s) => WirePattern::As {
            name: name.clone(),
            pattern: Box::new(wire_pattern(pattern)?),
            span: span(*s),
        },
    })
}

fn wire_let_binding(binding: &LetBinding) -> SourceWireResult<WireLetBinding> {
    Ok(WireLetBinding {
        pattern: wire_let_pattern(&binding.pattern),
        ty: binding.ty.as_ref().map(wire_type_expr),
        value: wire_expr(&binding.value)?,
    })
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
        TypeExpr::DimensionLiteral(value, s) => WireSurfTypeExpr::DimensionLiteral {
            digits: value.to_string(),
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
        TypeExpr::RankSpread(name, s) => WireSurfTypeExpr::RankSpread {
            name: name.clone(),
            span: span(*s),
        },
    }
}

type WireResult<T> = std::result::Result<T, String>;

fn wire_extent(value: usize) -> WireResult<NonnegativeExtent> {
    NonnegativeExtent::try_from(value)
}

fn wire_axis(value: usize) -> WireResult<i32> {
    i32::try_from(value).map_err(|_| "wire axis exceeds int32".to_string())
}

fn wire_float_parameter(value: f64, precision: Prim) -> WireResult<chelis_types::ScalarValue> {
    if !value.is_finite() || !matches!(precision, Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64) {
        return Err(
            "wire random parameter requires a finite value at the active float dtype".to_string(),
        );
    }
    let scalar = chelis_types::scalar_from_f64("wire_parameter", precision, value)
        .map_err(|error| error.to_string())?;
    if scalar.as_f64_lossy().to_bits() != value.to_bits() {
        return Err(
            "IR random parameter is not an exact stored value of its active dtype".to_string(),
        );
    }
    Ok(scalar)
}

fn wire_dag(dag: &Dag) -> WireResult<WireDag> {
    let wire = WireDag {
        schema_version: crate::schema::WIRE_DAG_SCHEMA_VERSION,
        nodes: dag
            .nodes()
            .iter()
            .map(wire_dag_node)
            .collect::<WireResult<_>>()?,
        roots: dag
            .roots()
            .iter()
            .map(|id| crate::schema::host_index(id.0))
            .collect(),
    };
    wire.validate_wire_contract()
        .map_err(|error| error.to_string())?;
    Ok(wire)
}

fn wire_dag_node(node: &chelis_ir::dag::DagNode) -> WireResult<WireDagNode> {
    Ok(WireDagNode {
        shape_deps: node
            .shape_deps
            .iter()
            .map(|id| crate::schema::host_index(id.0))
            .collect(),
        span_id: node.span_id.clone(),
        merged_spans: node.merged_spans.clone(),
        id: crate::schema::host_index(node.id.0),
        op: wire_op(&node.op, node.output_type.precision)?,
        inputs: node
            .inputs
            .iter()
            .map(|id| crate::schema::host_index(id.0))
            .collect(),
        output_type: wire_tensor_type(&node.output_type)?,
    })
}

fn wire_tensor_type(ty: &TensorType) -> WireResult<WireTensorType> {
    Ok(WireTensorType {
        dims: ty.dims.iter().map(wire_dim).collect::<WireResult<_>>()?,
        precision: ty.precision.name().to_string(),
    })
}

fn wire_dim(dim: &DimInfo) -> WireResult<WireDimInfo> {
    Ok(match dim {
        DimInfo::Named(name, size) => WireDimInfo::Named {
            name: name.clone(),
            size: size.map(wire_extent).transpose()?,
        },
        DimInfo::Lit(size) => WireDimInfo::Lit {
            size: wire_extent(*size)?,
        },
    })
}

fn wire_dim_expr(expr: &chelis_ir::dag::DimExpr) -> WireResult<WireDimExpr> {
    use chelis_ir::dag::DimExpr;
    Ok(match expr {
        DimExpr::Concrete(value) => WireDimExpr::Concrete {
            value: wire_extent(*value)?,
        },
        DimExpr::Sym(name) => WireDimExpr::Sym { name: name.clone() },
        DimExpr::Mul(lhs, rhs) => WireDimExpr::Mul {
            lhs: Box::new(wire_dim_expr(lhs)?),
            rhs: Box::new(wire_dim_expr(rhs)?),
        },
        DimExpr::Div(lhs, rhs) => WireDimExpr::Div {
            lhs: Box::new(wire_dim_expr(lhs)?),
            rhs: Box::new(wire_dim_expr(rhs)?),
        },
    })
}

/// chelis#616: map a movement-op / reshape-target [`RtDim`] to its wire form.
fn wire_bound(b: &RtDim) -> WireResult<WireRtDim> {
    Ok(match b {
        RtDim::Lit(n) => WireRtDim::Lit {
            value: wire_extent(*n)?,
        },
        RtDim::ToEnd => WireRtDim::ToEnd,
        RtDim::Node(i) => WireRtDim::Node {
            input: crate::schema::host_index(*i),
        },
        RtDim::Sym(name) => WireRtDim::Sym { name: name.clone() },
        RtDim::InputAxis {
            tensor,
            axis: chelis_ir::dag::RtAxis::Lit(axis),
        } => WireRtDim::InputAxis {
            tensor: crate::schema::host_index(*tensor),
            axis: WireRtAxis::Lit { value: *axis },
        },
    })
}

fn wire_op(op: &RiscOp, precision: Prim) -> WireResult<WireRiscOp> {
    Ok(match op {
        RiscOp::Add => WireRiscOp::Add,
        RiscOp::Sub => WireRiscOp::Sub,
        RiscOp::Mul => WireRiscOp::Mul,
        RiscOp::Div => WireRiscOp::Div,
        RiscOp::FloorDiv => WireRiscOp::FloorDiv,
        RiscOp::TruncDiv => WireRiscOp::TruncDiv,
        RiscOp::Mod => WireRiscOp::Mod,
        RiscOp::CmpLt => WireRiscOp::CmpLt,
        RiscOp::MaxElem => WireRiscOp::MaxElem,
        RiscOp::MinElem => WireRiscOp::MinElem,
        RiscOp::ExtremaAdjoint { kind, operand } => WireRiscOp::ExtremaAdjoint {
            extrema: match kind {
                ExtremaKind::Max => WireExtremaKind::Max,
                ExtremaKind::Min => WireExtremaKind::Min,
            },
            operand: match operand {
                ExtremaOperand::Left => WireExtremaOperand::Left,
                ExtremaOperand::Right => WireExtremaOperand::Right,
            },
        },
        RiscOp::Relu => WireRiscOp::Relu,
        RiscOp::ReluAdjoint => WireRiscOp::ReluAdjoint,
        RiscOp::Neg => WireRiscOp::Neg,
        RiscOp::Recip => WireRiscOp::Recip,
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
        RiscOp::Round => WireRiscOp::Round,
        RiscOp::UniformLike { low, high, seed } => WireRiscOp::UniformLike {
            low: wire_float_parameter(*low, precision)?,
            high: wire_float_parameter(*high, precision)?,
            seed: *seed,
        },
        RiscOp::Dropout { rate, seed } => WireRiscOp::Dropout {
            rate: wire_float_parameter(*rate, precision)?,
            seed: *seed,
        },
        RiscOp::Sum { axis, accumulator } => WireRiscOp::Sum {
            axis: wire_axis(*axis)?,
            accumulator: accumulator.name().to_string(),
        },
        RiscOp::Count { axes } => WireRiscOp::Count {
            axes: axes
                .iter()
                .copied()
                .map(wire_axis)
                .collect::<WireResult<_>>()?,
        },
        RiscOp::MaxReduce { axis } => WireRiscOp::MaxReduce {
            axis: wire_axis(*axis)?,
        },
        RiscOp::MinReduce { axis } => WireRiscOp::MinReduce {
            axis: wire_axis(*axis)?,
        },
        RiscOp::ProdReduce { axis } => WireRiscOp::ProdReduce {
            axis: wire_axis(*axis)?,
        },
        RiscOp::ReduceWindow {
            reducer,
            window_shape,
            strides,
        } => WireRiscOp::ReduceWindow {
            reducer: match reducer {
                chelis_ir::dag::ReduceWindowKind::Max => "max".to_string(),
                chelis_ir::dag::ReduceWindowKind::Min => "min".to_string(),
                chelis_ir::dag::ReduceWindowKind::Sum => "sum".to_string(),
                chelis_ir::dag::ReduceWindowKind::Mean => "mean".to_string(),
            },
            window_shape: window_shape
                .iter()
                .copied()
                .map(wire_extent)
                .collect::<WireResult<_>>()?,
            strides: strides
                .iter()
                .copied()
                .map(wire_extent)
                .collect::<WireResult<_>>()?,
        },
        RiscOp::ReduceWindowGrad {
            reducer,
            window_shape,
            strides,
        } => WireRiscOp::ReduceWindowGrad {
            reducer: match reducer {
                chelis_ir::dag::ReduceWindowKind::Max => "max".to_string(),
                chelis_ir::dag::ReduceWindowKind::Min => "min".to_string(),
                chelis_ir::dag::ReduceWindowKind::Sum => "sum".to_string(),
                chelis_ir::dag::ReduceWindowKind::Mean => "mean".to_string(),
            },
            window_shape: window_shape
                .iter()
                .copied()
                .map(wire_extent)
                .collect::<WireResult<_>>()?,
            strides: strides
                .iter()
                .copied()
                .map(wire_extent)
                .collect::<WireResult<_>>()?,
        },
        RiscOp::Argmax { axis } => WireRiscOp::Argmax {
            axis: wire_axis(*axis)?,
        },
        RiscOp::Argmin { axis } => WireRiscOp::Argmin {
            axis: wire_axis(*axis)?,
        },
        RiscOp::Reshape { new_shape } => WireRiscOp::Reshape {
            new_shape: new_shape
                .iter()
                .map(wire_bound)
                .collect::<WireResult<_>>()?,
        },
        RiscOp::Permute { axes } => WireRiscOp::Permute {
            axes: axes
                .iter()
                .copied()
                .map(wire_axis)
                .collect::<WireResult<_>>()?,
        },
        RiscOp::Expand { axis, size } => WireRiscOp::Expand {
            axis: wire_axis(*axis)?,
            size: wire_bound(size)?,
        },
        RiscOp::OneHot { vocab } => WireRiscOp::OneHot {
            vocab: wire_extent(*vocab)?,
        },
        RiscOp::Pad { padding, fill } => WireRiscOp::Pad {
            padding: padding
                .iter()
                .map(|(s, e)| Ok((wire_bound(s)?, wire_bound(e)?)))
                .collect::<WireResult<_>>()?,
            fill: *fill,
        },
        RiscOp::Shrink { bounds } => WireRiscOp::Shrink {
            bounds: bounds
                .iter()
                .map(|(s, e)| Ok((wire_bound(s)?, wire_bound(e)?)))
                .collect::<WireResult<_>>()?,
        },
        RiscOp::Stride { strides } => WireRiscOp::Stride {
            strides: strides.iter().map(wire_bound).collect::<WireResult<_>>()?,
        },
        RiscOp::Const { value } => WireRiscOp::Const { value: *value },
        RiscOp::ConstTensor { data } => WireRiscOp::ConstTensor { data: data.clone() },
        RiscOp::Shape { axis } => WireRiscOp::Shape {
            axis: wire_axis(*axis)?,
        },
        RiscOp::ExtentWitness {
            site,
            parameter,
            axis: chelis_ir::dag::RtAxis::Lit(axis),
            requirements,
        } => WireRiscOp::ExtentWitness {
            site: match site {
                chelis_ir::dag::ExtentWitnessSite::Caller => WireExtentWitnessSite::Caller,
                chelis_ir::dag::ExtentWitnessSite::LocalExpand => {
                    WireExtentWitnessSite::LocalExpand
                }
            },
            parameter: parameter.clone(),
            axis: WireRtAxis::Lit { value: *axis },
            requirements: requirements
                .iter()
                .copied()
                .map(NonnegativeExtent::try_from)
                .collect::<WireResult<_>>()?,
        },
        RiscOp::CheckedReshapeExtent {
            claims,
            axis: chelis_ir::dag::RtAxis::Lit(axis),
        } => WireRiscOp::CheckedReshapeExtent {
            claims: claims.clone(),
            axis: WireRtAxis::Lit { value: *axis },
        },
        RiscOp::CheckedUnitAxis {
            axis: chelis_ir::dag::RtAxis::Lit(axis),
        } => WireRiscOp::CheckedUnitAxis {
            axis: WireRtAxis::Lit { value: *axis },
        },
        RiscOp::Load { name } => WireRiscOp::Load {
            name: name.as_str().to_string(),
        },
        RiscOp::Store { name } => WireRiscOp::Store {
            name: name.as_str().to_string(),
        },
        RiscOp::Copy => WireRiscOp::Copy,
        RiscOp::Drop => WireRiscOp::Drop,
        RiscOp::Realize => WireRiscOp::Realize,
        RiscOp::CastTrunc { new_precision } => WireRiscOp::CastTrunc {
            new_precision: new_precision.name().to_string(),
        },
        RiscOp::Cast { new_precision } => WireRiscOp::Cast {
            new_precision: new_precision.name().to_string(),
        },
        RiscOp::FusedElem { ops } => WireRiscOp::FusedElem {
            ops: ops
                .iter()
                .map(|step| WireFusedStep {
                    op: match step.op {
                        FusedStepOp::Add => WireFusedStepOp::Add,
                        FusedStepOp::Sub => WireFusedStepOp::Sub,
                        FusedStepOp::Mul => WireFusedStepOp::Mul,
                        FusedStepOp::Div => WireFusedStepOp::Div,
                        FusedStepOp::FloorDiv => WireFusedStepOp::FloorDiv,
                        FusedStepOp::TruncDiv => WireFusedStepOp::TruncDiv,
                        FusedStepOp::MaxElem => WireFusedStepOp::MaxElem,
                        FusedStepOp::MinElem => WireFusedStepOp::MinElem,
                        FusedStepOp::CmpLt => WireFusedStepOp::CmpLt,
                        FusedStepOp::Neg => WireFusedStepOp::Neg,
                        FusedStepOp::Recip => WireFusedStepOp::Recip,
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
                        FusedStepOp::Round => WireFusedStepOp::Round,
                    },
                    input_indices: step
                        .input_indices
                        .iter()
                        .map(|input| match input {
                            FusedInput::External(index) => WireFusedInput::External {
                                index: crate::schema::host_index(*index),
                            },
                            FusedInput::PreviousStep(index) => WireFusedInput::PreviousStep {
                                index: crate::schema::host_index(*index),
                            },
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
            batch_dims: batch_dims
                .iter()
                .map(wire_dim_expr)
                .collect::<WireResult<_>>()?,
            m: wire_dim_expr(m)?,
            n: wire_dim_expr(n)?,
            k: wire_dim_expr(k)?,
            accumulator: accumulator.name().to_string(),
        },
        RiscOp::Gather { axis } => WireRiscOp::Gather {
            axis: wire_axis(*axis)?,
        },
        RiscOp::ScatterAdd { axis } => WireRiscOp::ScatterAdd {
            axis: wire_axis(*axis)?,
        },
        RiscOp::Scatter { axis } => WireRiscOp::Scatter {
            axis: wire_axis(*axis)?,
        },
        RiscOp::ScatterElements { axis } => WireRiscOp::ScatterElements {
            axis: wire_axis(*axis)?,
        },
    })
}

#[cfg(test)]
use crate::schema::ExecutionValue;

#[cfg(test)]
#[path = "../../../tests/support/wire_values.rs"]
pub(crate) mod wire_values;

#[cfg(test)]
#[allow(deprecated)] // exercises eval_many for behavior parity; deprecation is for external callers
mod tests {
    use super::*;
    use crate::schema::ExecutionValue;
    use crate::schema::{WireDeepAtom, WireDeepExprKind};
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn native_wire_witness_fixture() -> Dag {
        use chelis_ir::dag::RtAxis;
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("rows".into(), None)],
                precision: Prim::F32,
            },
            Some("input-span".into()),
        );
        let requirements = [4, 4, 9]
            .into_iter()
            .map(|value| chelis_types::scalar_from_i64("load", Prim::Int64, value).unwrap())
            .collect();
        let witness = dag.add_node(
            RiscOp::ExtentWitness {
                site: chelis_ir::dag::ExtentWitnessSite::Caller,
                parameter: "x".into(),
                axis: RtAxis::Lit(0),
                requirements,
            },
            vec![input],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            Some("call-span".into()),
        );
        dag.node_mut(witness).unwrap().merged_spans = vec!["result-span".into()];
        let root = dag.add_node(
            RiscOp::Const {
                value: chelis_types::scalar_from_i64("load", Prim::Int64, 9).unwrap(),
            },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        dag.node_mut(root).unwrap().shape_deps = vec![witness];
        dag.add_root(root);
        dag
    }

    #[test]
    fn native_wire_witness_projection_preserves_exact_claims_and_provenance() {
        let dag = native_wire_witness_fixture();
        let projected = wire_dag(&dag).unwrap();
        let json = serde_json::to_value(&projected).unwrap();
        assert_eq!(json["schema_version"], 10);
        assert_eq!(
            json["nodes"][1]["op"]["requirements"],
            serde_json::json!([4, 4, 9])
        );
        assert_eq!(
            json["nodes"][1]["op"]["axis"],
            serde_json::json!({"axis":"lit","value":0})
        );
        assert_eq!(json["nodes"][1]["span_id"], "call-span");
        assert_eq!(
            json["nodes"][1]["merged_spans"],
            serde_json::json!(["result-span"])
        );
        assert_eq!(json["nodes"][2]["shape_deps"], serde_json::json!([1]));
        assert_eq!(json["nodes"][2]["span_id"], serde_json::Value::Null);
        let decoded = WireDag::from_validated_json(&json.to_string()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);
    }

    #[test]
    fn native_wire_witness_projection_rejects_invalid_requirements_and_edges() {
        use chelis_ir::dag::{NodeId, RtAxis};
        for (invalid, expected) in [
            (
                chelis_types::scalar_from_i64("load", Prim::Int64, -1).unwrap(),
                "dimension extent must be a nonnegative int64",
            ),
            (
                chelis_types::scalar_from_i64("load", Prim::Int32, 4).unwrap(),
                "source integer requires exact int64 dtype",
            ),
        ] {
            let mut dag = native_wire_witness_fixture();
            let RiscOp::ExtentWitness { requirements, .. } =
                &mut dag.node_mut(NodeId(1)).unwrap().op
            else {
                unreachable!()
            };
            requirements[0] = invalid;
            assert_eq!(wire_dag(&dag).unwrap_err(), expected);
        }
        for mutation in 0..4 {
            let mut dag = native_wire_witness_fixture();
            match mutation {
                0 => dag.node_mut(NodeId(2)).unwrap().shape_deps = vec![NodeId(2)],
                1 => dag.node_mut(NodeId(1)).unwrap().inputs.clear(),
                2 => dag.node_mut(NodeId(1)).unwrap().output_type.precision = Prim::F32,
                3 => {
                    let RiscOp::ExtentWitness { axis, .. } =
                        &mut dag.node_mut(NodeId(1)).unwrap().op
                    else {
                        unreachable!()
                    };
                    *axis = RtAxis::Lit(1);
                }
                _ => unreachable!(),
            }
            assert!(wire_dag(&dag).is_err(), "native mutation {mutation}");
        }
    }

    #[test]
    fn wire_producer_rejects_numeric_narrowing_instead_of_repairing_ir() {
        assert!(wire_float_parameter(f64::from(0.1_f32), Prim::F32).is_ok());
        assert!(wire_float_parameter(0.1_f64, Prim::F32).is_err());
        assert!(wire_float_parameter(f64::NAN, Prim::F64).is_err());
        assert!(wire_float_parameter(1.0, Prim::Int64).is_err());
        assert_eq!(wire_axis(0).unwrap(), 0);
        assert!(wire_axis(usize::MAX).is_err());
        assert_eq!(wire_extent(0).unwrap().get(), 0);
        #[cfg(target_pointer_width = "64")]
        assert!(wire_extent(usize::MAX).is_err());
    }

    /// chelis#1395 [04-FIT-16]: a lexer error carries a byte offset, so the
    /// carrier transports it.
    ///
    /// Both languages previously matched `ParseError::Lex(_) => return None`,
    /// discarding a coordinate the producer held -- the exact loss the tagged
    /// carrier exists to stop, left in place at the one producer that cannot
    /// reach the parser to be given a span.
    #[test]
    fn a_surf_lex_error_retains_its_point() {
        let source = "x=\"unterminated";
        let error = chelis_surf::parser::parse_str(source)
            .expect_err("an unterminated string is a lex error");
        assert!(
            matches!(error, chelis_surf::parser::ParseError::Lex(_)),
            "fixture must reach the lexer arm, got: {error:?}"
        );
        let span = parse_error_span_surf(source, &error);
        assert_eq!(
            span.extent(),
            None,
            "a lexer coordinate is a point, not a measured range"
        );
        assert_eq!(
            span,
            DiagnosticSpan::Point {
                offset: u64::try_from(source.find('"').expect("the fixture has a quote")).unwrap(),
            },
            "the point must be the offset the lexer reported"
        );
    }

    /// The Deep half of the same omission. Kept as a separate test because the
    /// two languages have separate `LexError` enums and separate projections;
    /// one passing proved nothing about the other.
    #[test]
    fn a_deep_lex_error_retains_its_point() {
        let source = "(x \"unterminated";
        // `parse_raw_str` is the lex-then-parse entry; the stamped entries
        // wrap the same `ParseError` in `StampOrParseError`.
        let error = chelis_deep::parser::parse_raw_str(source)
            .expect_err("an unterminated string is a lex error");
        assert!(
            matches!(error, chelis_deep::parser::ParseError::Lex(_)),
            "fixture must reach the lexer arm, got: {error:?}"
        );
        let span = parse_error_span_deep(&error);
        assert_eq!(span.extent(), None);
        assert_eq!(
            span,
            DiagnosticSpan::Point {
                offset: u64::try_from(source.find('"').expect("the fixture has a quote")).unwrap(),
            }
        );
    }

    #[test]
    fn decompile_maps_resugaring_rejection_to_the_validation_kind() {
        // The unknown form sits at a RuntimeExpr slot, so the stamped ingress
        // admits it as an `UnknownForm` and the resugaring boundary is what
        // rejects it (chelis#1088 moved the top-level spelling of this input
        // to an ingress rejection; see the sibling test below).
        let error = decompile(DecompileRequest {
            source: "(def {} value (future-form {} 1))".to_string(),
        })
        .expect_err("an unknown Deep form must fail closed during resugaring");

        assert_eq!(error.stage, "decompile");
        assert_eq!(error.errors.len(), 1);
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::ValidationError
        );
    }

    #[test]
    fn decompile_rejects_a_non_declaration_top_level_at_ingress() {
        // chelis#1088: `decompile` shares the stamped `.dp` ingress, so a
        // top-level form that is not a declaration never reaches resugaring.
        let error = decompile(DecompileRequest {
            source: "(future-form {} value)".to_string(),
        })
        .expect_err("a non-declaration top-level form must fail at ingress");

        assert_eq!(error.stage, "parse");
        assert_eq!(error.errors.len(), 1);
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::DeepParseError
        );
        assert!(
            error.errors[0].message.contains("expected declaration"),
            "{}",
            error.errors[0].message
        );
    }

    #[test]
    fn opaque_extension_wire_is_explicit_and_preserves_scalar_spelling() {
        let payload = "{type: false, type: (var {}), value: 1e-3f32}";
        let exprs = chelis_deep::parse_and_stamp_file(&format!(
            "(def {{tool_data: {payload}}} f (lit {{}} 1))"
        ))
        .unwrap();
        let wire = serde_json::to_value(wire_deep_expr(&exprs[0])).unwrap();
        let text = wire.to_string();
        assert!(text.contains("extension_data"), "{text}");
        assert!(text.contains(payload), "{text}");
        assert!(chelis_deep::parse_and_stamp_file("(def {type: false} f (lit {} 1))").is_err());
    }

    #[test]
    fn typed_deep_node_wire_bridge_preserves_the_complete_node_shape() {
        let exprs = chelis_deep::parse_and_stamp_file("(def {doc: \"test\"} root (var {} value))")
            .expect("typed Deep must parse and stamp");
        assert!(matches!(exprs.first(), Some(DeepExpr::Node(_, _))));

        let wire = wire_deep_expr(&exprs[0]).expect("finite Deep source must encode");
        let WireDeepExprKind::List { elements } = wire.kind else {
            panic!("a typed node must cross the wire as its canonical list shape");
        };
        assert_eq!(
            elements.len(),
            4,
            "def must retain tag, metadata, and children"
        );
        assert!(matches!(
            &elements[0].kind,
            WireDeepExprKind::Atom {
                atom: WireDeepAtom::Symbol { value }
            } if value == "def"
        ));
        let WireDeepExprKind::Map { entries } = &elements[1].kind else {
            panic!("def metadata must remain at wire element 1");
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "doc");
        assert!(matches!(
            &entries[0].value.kind,
            WireDeepExprKind::Atom {
                atom: WireDeepAtom::Str { value }
            } if value == "test"
        ));
        assert!(matches!(
            &elements[2].kind,
            WireDeepExprKind::Atom {
                atom: WireDeepAtom::Symbol { value }
            } if value == "root"
        ));
        let WireDeepExprKind::List {
            elements: value_elements,
        } = &elements[3].kind
        else {
            panic!("the def runtime child must remain a complete wire node");
        };
        assert!(matches!(
            &value_elements[0].kind,
            WireDeepExprKind::Atom {
                atom: WireDeepAtom::Symbol { value }
            } if value == "var"
        ));
        assert!(matches!(
            &value_elements[2].kind,
            WireDeepExprKind::Atom {
                atom: WireDeepAtom::Symbol { value }
            } if value == "value"
        ));
    }

    /// The three sanitization rules `spec/11-ffi.md` §5a documents for the
    /// legacy symbol mapping (reviewer N1: previously untested, so any of
    /// them could regress silently).
    #[test]
    fn execution_c_symbol_sanitization_rules() {
        // Reserved process entry is rewritten.
        assert_eq!(execution_c_symbol(Some("main")), "chelis_main");
        // Non-identifier characters map to `_`.
        assert_eq!(execution_c_symbol(Some("my-model")), "my_model");
        // Digit-leading and empty names gain the `chelis_` prefix.
        assert_eq!(execution_c_symbol(Some("2fast")), "chelis_2fast");
        assert_eq!(execution_c_symbol(Some("")), "chelis_");
        // Ordinary names pass through unchanged; the default is chelis_main.
        assert_eq!(execution_c_symbol(Some("solve")), "solve");
        assert_eq!(execution_c_symbol(None), "chelis_main");
    }

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
            "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n",
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

    fn hip_direct_arithmetic_dag(op: RiscOp, precision: chelis_types::types::Prim) -> Dag {
        let mut dag = Dag::new();
        let lhs = dag.add_node(
            RiscOp::Load { name: "lhs".into() },
            vec![],
            tensor_type(vec![4], precision),
            None,
        );
        let rhs = dag.add_node(
            RiscOp::Load { name: "rhs".into() },
            vec![],
            tensor_type(vec![4], precision),
            None,
        );
        let gradient = dag.add_node(
            RiscOp::Load {
                name: "gradient".into(),
            },
            vec![],
            tensor_type(vec![4], precision),
            None,
        );
        let inputs = match op {
            RiscOp::Relu => vec![lhs],
            RiscOp::ReluAdjoint => vec![lhs, gradient],
            RiscOp::ExtremaAdjoint { .. } => vec![lhs, rhs, gradient],
            _ => vec![lhs, rhs],
        };
        let result = dag.add_node(op, inputs, tensor_type(vec![4], precision), None);
        dag.add_root(result);
        dag
    }

    #[test]
    fn hip_seam_accepts_supported_direct_arithmetic_cells() {
        for precision in [
            chelis_types::types::Prim::F32,
            chelis_types::types::Prim::F64,
        ] {
            for op in [
                RiscOp::Sub,
                RiscOp::MaxElem,
                RiscOp::MinElem,
                RiscOp::ExtremaAdjoint {
                    kind: ExtremaKind::Max,
                    operand: ExtremaOperand::Left,
                },
                RiscOp::ExtremaAdjoint {
                    kind: ExtremaKind::Min,
                    operand: ExtremaOperand::Right,
                },
                RiscOp::Relu,
                RiscOp::ReluAdjoint,
            ] {
                reject_unsupported_hip_ops(&hip_direct_arithmetic_dag(op, precision))
                    .expect("HIP f32/f64 direct arithmetic must reach implemented codegen");
            }
        }

        for precision in [
            chelis_types::types::Prim::F32,
            chelis_types::types::Prim::F64,
            chelis_types::types::Prim::F16,
            chelis_types::types::Prim::Bf16,
        ] {
            for op in [RiscOp::Relu, RiscOp::ReluAdjoint] {
                reject_unsupported_hip_ops(&hip_direct_arithmetic_dag(op, precision))
                    .expect("HIP ReLU must reach the typed kernel at every float width");
            }
        }

        for precision in [
            chelis_types::types::Prim::Int8,
            chelis_types::types::Prim::Int16,
            chelis_types::types::Prim::Int32,
            chelis_types::types::Prim::Int64,
        ] {
            for op in [RiscOp::MaxElem, RiscOp::MinElem] {
                reject_unsupported_hip_ops(&hip_direct_arithmetic_dag(op, precision))
                    .expect("HIP signed-integer extrema must reach implemented codegen");
            }
        }
    }

    #[test]
    fn hip_seam_rejects_unimplemented_direct_arithmetic_cells_with_issue_1306() {
        for precision in [
            chelis_types::types::Prim::Int8,
            chelis_types::types::Prim::Int16,
            chelis_types::types::Prim::Int32,
            chelis_types::types::Prim::Int64,
        ] {
            let error =
                reject_unsupported_hip_ops(&hip_direct_arithmetic_dag(RiscOp::Sub, precision))
                    .expect_err("HIP integer subtraction needs a device trap channel");
            assert!(
                error.errors[0]
                    .message
                    .contains("unimplemented chelis#1306:"),
                "{}",
                error.errors[0].message
            );
        }

        for precision in [
            chelis_types::types::Prim::F16,
            chelis_types::types::Prim::Bf16,
        ] {
            for op in [
                RiscOp::Sub,
                RiscOp::MaxElem,
                RiscOp::MinElem,
                RiscOp::ExtremaAdjoint {
                    kind: ExtremaKind::Max,
                    operand: ExtremaOperand::Left,
                },
            ] {
                let error = reject_unsupported_hip_ops(&hip_direct_arithmetic_dag(op, precision))
                    .expect_err("HIP narrow-float direct arithmetic is not implemented");
                assert!(
                    error.errors[0]
                        .message
                        .contains("unimplemented chelis#1306:"),
                    "{}",
                    error.errors[0].message
                );
            }
        }

        let fused_sub = RiscOp::FusedElem {
            ops: vec![chelis_ir::dag::FusedStep {
                op: FusedStepOp::Sub,
                input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
            }],
        };
        for precision in [
            chelis_types::types::Prim::Int32,
            chelis_types::types::Prim::F16,
        ] {
            let error = reject_unsupported_hip_ops(&hip_direct_arithmetic_dag(
                fused_sub.clone(),
                precision,
            ))
            .expect_err("fused subtraction inherits the direct target disposition");
            assert!(
                error.errors[0]
                    .message
                    .contains("unimplemented chelis#1306:"),
                "{}",
                error.errors[0].message
            );
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
    fn hip_rejects_count_with_issue_1291_receipt() {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(vec![2, 3], chelis_types::types::Prim::Bool),
            None,
        );
        dag.add_node(
            RiscOp::Count { axes: vec![1] },
            vec![input],
            tensor_type(vec![2], chelis_types::types::Prim::Int64),
            None,
        );

        let error = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP must reject Count until its dedicated kernel lands");
        let message = &error.errors[0].message;
        assert!(message.contains("unimplemented chelis#1291:"), "{message}");
        assert!(
            message.contains("count") && message.contains("--target c"),
            "{message}"
        );
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
    }

    #[test]
    fn hip_seam_accepts_input_axis_expand_extent() {
        let mut dag = Dag::new();
        let value = dag.add_node(
            RiscOp::Load {
                name: "value".into(),
            },
            vec![],
            tensor_type(vec![], chelis_types::types::Prim::F32),
            None,
        );
        let witness = dag.add_node(
            RiscOp::Load {
                name: "witness".into(),
            },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: chelis_ir::dag::RtAxis::Lit(0),
                },
            },
            vec![value, witness],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );

        reject_unsupported_hip_ops(&dag)
            .expect("InputAxis is a metadata read implemented by HIP expand codegen");
    }

    #[test]
    fn hip_seam_rejects_node_valued_expand_with_issue_1298_receipt() {
        let mut dag = Dag::new();
        let value = dag.add_node(
            RiscOp::Load {
                name: "value".into(),
            },
            vec![],
            tensor_type(vec![], chelis_types::types::Prim::F32),
            None,
        );
        let size = dag.add_node(
            RiscOp::Load {
                name: "size".into(),
            },
            vec![],
            tensor_type(vec![], chelis_types::types::Prim::Int64),
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Node(1),
            },
            vec![value, size],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );

        let error = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP must reject a device scalar expand extent");
        let message = &error.errors[0].message;
        assert!(message.contains("unimplemented chelis#1298:"), "{message}");
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
    }

    #[test]
    fn hip_sparse_gather_rejects_float_indices_with_deciding_atom() {
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
            tensor_type(vec![4], chelis_types::types::Prim::F32),
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
            .expect_err("HIP must reject a sparse gather with float indices");
        assert!(err.errors[0].message.contains("deliberate [05-SPARSE-1]:"));
    }

    #[test]
    fn hip_internal_one_hot_rejection_names_specialization_invariant() {
        let mut dag = Dag::new();
        let one_hot = dag.add_node(
            RiscOp::OneHot { vocab: 4 },
            vec![],
            tensor_type(vec![2, 4], chelis_types::types::Prim::F32),
            None,
        );
        dag.add_root(one_hot);

        let err =
            reject_unsupported_hip_ops(&dag).expect_err("OneHot must not survive to HIP emission");
        assert!(err.errors[0].message.contains("deliberate [05-SPARSE-2]:"));
    }

    #[test]
    fn hip_shape_rejection_names_the_target_authority() {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );
        let shape = dag.add_node(
            RiscOp::Shape { axis: 0 },
            vec![input],
            tensor_type(vec![], chelis_types::types::Prim::Int32),
            None,
        );
        dag.add_root(shape);

        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP must reject a runtime shape value node");
        assert!(err.errors[0].message.contains("deliberate [05-SHAPE-1]:"));
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
            RiscOp::synth_const(
                tensor_type(vec![4], chelis_types::types::Prim::Int64).precision,
                0.0,
            ),
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
        assert!(message.contains("unimplemented chelis#729:"));
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
            RiscOp::synth_const(
                tensor_type(vec![4], chelis_types::types::Prim::Int32).precision,
                0.0,
            ),
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
        assert!(message.contains("unimplemented chelis#729:"));
    }

    // --- reduce_window: runtime-symbolic windowed axis is a build error ---
    //
    // Regression coverage for issue #261: `chelis build` must reject
    // `reduce_window_*` over a windowed axis whose extent is only known at
    // runtime, rather than silently bind the windowed output axis to the
    // input extent (which mis-allocates the output and emits an
    // out-of-bounds window read). See spec/05-risc-primitives.md §2.3.1.

    fn reduce_window_node_dag(out_dims: Vec<DimInfo>, window: Vec<usize>) -> Dag {
        use chelis_ir::dag::ReduceWindowKind;
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(vec![2, 8], chelis_types::types::Prim::F32),
            None,
        );
        let rw = dag.add_node(
            RiscOp::ReduceWindow {
                reducer: ReduceWindowKind::Max,
                window_shape: window,
                strides: vec![1],
            },
            vec![input],
            TensorType {
                dims: out_dims,
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        dag.add_root(rw);
        dag
    }

    #[test]
    fn reduce_window_rejects_runtime_symbolic_windowed_axis_dag() {
        // Leading axis sized, trailing (windowed) axis runtime-only.
        let dag = reduce_window_node_dag(
            vec![
                DimInfo::Named("batch".into(), Some(2)),
                DimInfo::Named("seq".into(), None),
            ],
            vec![2],
        );
        let err = reject_symbolic_windowed_reduce(&dag, BuildTarget::C)
            .expect_err("a runtime-only windowed axis must be rejected on the build path");
        let message = &err.errors[0].message;
        assert!(
            message.contains("requires statically-known"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("windowed axis 1"),
            "unexpected message: {message}"
        );
        assert!(message.contains("`seq`"), "unexpected message: {message}");
        assert!(message.contains("unimplemented chelis#600:"));
    }

    #[test]
    fn reduce_window_allows_symbolic_leading_axis_dag() {
        // A symbolic *leading* (pass-through) axis is fine; only the
        // windowed axes must be statically known.
        let dag = reduce_window_node_dag(
            vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(7)],
            vec![2],
        );
        reject_symbolic_windowed_reduce(&dag, BuildTarget::C)
            .expect("symbolic leading axis with a statically-sized windowed axis is allowed");
    }

    #[test]
    fn reduce_window_allows_statically_sized_windowed_axis_dag() {
        // Both a literal and a named-with-size windowed axis are allowed.
        let lit_dag = reduce_window_node_dag(vec![DimInfo::Lit(2), DimInfo::Lit(7)], vec![2]);
        reject_symbolic_windowed_reduce(&lit_dag, BuildTarget::C)
            .expect("literal windowed axis is allowed");

        let named_sized_dag = reduce_window_node_dag(
            vec![DimInfo::Lit(2), DimInfo::Named("h_out".into(), Some(7))],
            vec![2],
        );
        reject_symbolic_windowed_reduce(&named_sized_dag, BuildTarget::C)
            .expect("named-with-size windowed axis is allowed");
    }

    #[test]
    fn compile_rejects_reduce_window_over_runtime_symbolic_axis() {
        // End-to-end: `pad_sequences` yields a runtime-bound trailing
        // extent, so windowing over it cannot be lowered to a correct
        // static output shape. The build must fail rather than emit a
        // mis-allocated kernel whose output diverges from the evaluator
        // (issue #261). The host runtime / IR evaluator handle this case
        // correctly; only the ahead-of-time build path is restricted.
        let source = r#"
padded = pad_sequences([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], 0.0)
windowed = reduce_window_max(padded, [2i64], [1i64])
"#;
        let err = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            target: CompileTarget::C,
            entry_name: Some("rw_symbolic".to_string()),
        })
        .expect_err("build must reject reduce_window over a runtime-symbolic windowed axis");
        let message = &err.errors[0].message;
        assert!(
            message.contains("requires statically-known"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("reduce_window"),
            "unexpected message: {message}"
        );
    }

    fn reduce_window_dag_with_precision(prec: chelis_types::types::Prim) -> Dag {
        use chelis_ir::dag::ReduceWindowKind;
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(vec![1, 1, 4, 4], prec),
            None,
        );
        let rw = dag.add_node(
            RiscOp::ReduceWindow {
                reducer: ReduceWindowKind::Max,
                window_shape: vec![2, 2],
                strides: vec![1, 1],
            },
            vec![input],
            tensor_type(vec![1, 1, 3, 3], prec),
            None,
        );
        dag.add_root(rw);
        dag
    }

    // --- reduce_window: bf16/f16 is rejected on the C build path ---
    //
    // PR #261 review finding #1: the C backend admits bf16/f16 generally, but
    // the C `reduce_window_*` emitter is f32-only and `panic!`s on anything
    // else. Without this guard a bf16 windowed reduction aborts with an
    // `internal error` panic instead of a clean diagnostic.
    #[test]
    fn reduce_window_rejects_bf16_precision_on_c_build() {
        let bf16 = reduce_window_dag_with_precision(chelis_types::types::Prim::Bf16);
        let err = reject_unsupported_reduce_window_precision(&bf16, BuildTarget::C)
            .expect_err("bf16 reduce_window must be rejected on the C build path");
        let message = &err.errors[0].message;
        assert!(
            message.contains("f32") && message.contains("reduce_window"),
            "unexpected message: {message}"
        );
        assert!(message.contains("unimplemented chelis#729:"));
        assert_eq!(
            err.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );

        // f16 is rejected the same way; f32 is allowed.
        let f16 = reduce_window_dag_with_precision(chelis_types::types::Prim::F16);
        reject_unsupported_reduce_window_precision(&f16, BuildTarget::C)
            .expect_err("f16 reduce_window must be rejected on the C build path");
        let f32 = reduce_window_dag_with_precision(chelis_types::types::Prim::F32);
        reject_unsupported_reduce_window_precision(&f32, BuildTarget::C)
            .expect("f32 reduce_window must be allowed");
    }

    // --- reduce_window: HIP build rejects the node cleanly (no todo! panic) ---
    //
    // HIP windowed-reduction codegen is excluded by [05-RWIN-2].
    // The build must reject a `ReduceWindow` node with a clean
    // `unsupported_feature` error before it reaches the launch-emit `todo!`.
    #[test]
    fn hip_rejects_reduce_window_node_with_clean_message() {
        let dag = reduce_window_dag_with_precision(chelis_types::types::Prim::F32);
        let err =
            reject_unsupported_hip_ops(&dag).expect_err("HIP must reject reduce_window codegen");
        let message = &err.errors[0].message;
        assert!(
            message.contains("reduce_window") && message.contains("--target hip"),
            "unexpected message: {message}"
        );
        assert!(message.contains("deliberate [05-RWIN-2]:"));
        assert_eq!(
            err.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
    }

    #[test]
    fn hip_rejects_reduce_window_grad_with_the_same_target_authority() {
        use chelis_ir::dag::ReduceWindowKind;

        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );
        let cotangent = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            tensor_type(vec![3], chelis_types::types::Prim::F32),
            None,
        );
        let grad = dag.add_node(
            RiscOp::ReduceWindowGrad {
                reducer: ReduceWindowKind::Max,
                window_shape: vec![2],
                strides: vec![1],
            },
            vec![input, cotangent],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );
        dag.add_root(grad);

        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP must reject reduce_window adjoint codegen");
        let message = &err.errors[0].message;
        assert!(message.contains("ReduceWindowGrad"), "{message}");
        assert!(message.contains("deliberate [05-RWIN-2]:"), "{message}");
    }

    /// chelis#616: a runtime (node-valued) reshape target extent is C-only;
    /// the HIP seam must reject it cleanly before codegen, exactly like the
    /// node-valued movement-bound arms.
    #[test]
    fn hip_rejects_node_valued_reshape_target_with_clean_message() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );
        // A rank-0 integer Load, not a `Shape` read: the seam blanket-rejects
        // `RiscOp::Shape` first, and this test must exercise the reshape arm.
        let extent = dag.add_node(
            RiscOp::Load { name: "m".into() },
            vec![],
            tensor_type(vec![], chelis_types::types::Prim::Int32),
            None,
        );
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1)],
            },
            vec![x, extent],
            tensor_type(vec![4], chelis_types::types::Prim::F32),
            None,
        );
        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP must reject a node-valued reshape target");
        let message = &err.errors[0].message;
        assert!(
            message.contains("reshape") && message.contains("--target c"),
            "unexpected message: {message}"
        );
        assert!(message.contains("deliberate [05-MOV-1]:"));
        assert_eq!(
            err.errors[0].kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
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
        assert!(message.contains("unimplemented chelis#729:"));
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
            compiled
                .tensor_root_names
                .iter()
                .map(crate::pipeline::IrName::as_str)
                .collect::<Vec<_>>(),
            ["logits", "loss"]
        );
        assert_eq!(
            compiled
                .named_roots
                .keys()
                .map(crate::pipeline::IrName::as_str)
                .collect::<Vec<_>>(),
            ["logits", "loss"]
        );
        assert_eq!(compiled.dag.roots().len(), 2);
    }

    #[test]
    fn compile_source_arrow_form_def_appears_in_manifest() {
        // Issue #947: arrow-form `def n() -> T = body` must appear in
        // the production manifest so eval_compiled can surface it.
        let source = "def n() -> int32 = add(cast(20, int32), cast(22, int32))\n";
        let compiled = compile_source(SourceKind::Surf, source).expect("compile");
        assert!(
            compiled
                .manifest()
                .entries
                .iter()
                .any(|entry| entry.name == "n"),
            "arrow-form def `n` must be in the manifest; got: {:?}",
            compiled.manifest().entries
        );
    }

    /// chelis#864 / [05-OBS-1]: a nullary tensor function whose body
    /// changes dtype must carry the same stored source value through each
    /// inlined call root. The transcript evaluates `mk()` in the host lane,
    /// while the labeled `troot` is read from the lowered DAG. The F64 root
    /// assertion below is an explicit diagnosis control: the issue was not a
    /// stale root tag, but an F32 static-literal source that had never been
    /// materialized at f32 width before the widening cast.
    #[test]
    fn issue_864_cast_constructed_f64_root_agrees_with_host_value() {
        let source = "module M.Main\n\
            def mk() -> tensor[2, f64] = cast(to_tensor([0.1, 0.3]), f64)\n\
            shown = print(mk())\n\
            troot = mk()\n";

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");
        let root_id = *compiled
            .named_roots
            .get(&crate::pipeline::IrName::new("troot"))
            .expect("troot named root");
        let root = compiled.dag.get(root_id).expect("troot DAG node");
        assert_eq!(
            root.output_type.precision,
            chelis_types::types::Prim::F64,
            "the lowered call root must carry mk's declared f64 return dtype"
        );

        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");
        assert_eq!(
            result.transcript,
            vec!["tensor(shape=[2], data=[0.10000000149011612, 0.30000001192092896])"]
        );
        let troot = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("troot"))
            .expect("troot result");
        assert_eq!(
            troot.display.as_deref(),
            result.transcript.first().map(String::as_str),
            "labeled-root and transcript exits must render the same stored f64 bits"
        );
    }

    /// Negative-parity control for chelis#864: finalizing f32 literal
    /// ingress must not promote a genuinely f32 cast root.
    #[test]
    fn issue_864_f32_cast_root_keeps_f32_dtype() {
        let source = "module M.Main\n\
            def mk() -> tensor[2, f32] = cast(to_tensor([0.1, 0.3]), f32)\n\
            troot = mk()\n";
        let compiled = compile_source(SourceKind::Surf, source).expect("compile");
        let root_id = *compiled
            .named_roots
            .get(&crate::pipeline::IrName::new("troot"))
            .expect("troot named root");
        let root = compiled.dag.get(root_id).expect("troot DAG node");
        assert_eq!(
            root.output_type.precision,
            chelis_types::types::Prim::F32,
            "the dtype handoff must preserve a real f32 return"
        );
    }

    /// Negative-parity control for chelis#864: the F32 ingress repair is
    /// not a blanket f64-to-f32 narrowing. A direct f64 static literal
    /// tensor keeps the lexical f64 values in both eval exits.
    #[test]
    fn issue_864_direct_f64_literal_root_is_not_narrowed() {
        let source = "module M.Main\n\
            def mk() -> tensor[2, f64] = to_tensor([cast(0.1, f64), cast(0.3, f64)])\n\
            shown = print(mk())\n\
            troot = mk()\n";
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        })
        .expect("eval");
        assert_eq!(
            result.transcript,
            vec!["tensor(shape=[2], data=[0.1, 0.3])"],
            "a direct f64 literal tensor must not be narrowed to f32 ingress"
        );
        let troot = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("troot"))
            .expect("troot result");
        assert_eq!(
            troot.display.as_deref(),
            result.transcript.first().map(String::as_str)
        );
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
            compiled
                .manifest()
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["label"]
        );
        assert_eq!(
            compiled
                .named_roots
                .keys()
                .map(crate::pipeline::IrName::as_str)
                .collect::<Vec<_>>(),
            ["logits"]
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
        let monolithic_root = *monolithic
            .named_roots
            .get(&crate::pipeline::IrName::new("out"))
            .expect("out root");
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
            Target::Eval,
        )
        .expect("compile new source in context");
        let context_root = *compiled
            .named_roots
            .get(&crate::pipeline::IrName::new("out"))
            .expect("out root");
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

    // --- In-context compiled-execution coverage (issue #816, review round 2) ---
    //
    // These gate the `compile_for_execution_in_context` path that had ZERO
    // gated coverage before this round (the manual oracle needs the 0.16.1
    // toolchain + chelis-std registry). They use `copy_drop_context_fixture` —
    // a two-package path-dep reef project compiled fully in-process, no network,
    // no installed toolchain — so they run in the default gate. The library
    // exports `consume(x: tensor[2, f32]) -> tensor[2, f32] = realize(x)`
    // (identity), which the new-code entry calls across the module boundary:
    // the #816 scenario where `compiled.checked` holds new code only and the
    // called def lives in the linked library context.

    // Wrong-subgraph-slice + `main`-preference guard: a two-def in-context
    // source (`main` with ONE input calling the library fn, `second` with TWO
    // inputs) and NO `entry_name` must select `main` and scope the callable
    // metadata to it — exactly ONE input, not the union of every def's params
    // (the #817 regression) and not `second`.
    #[test]
    fn compile_for_execution_in_context_selects_main_and_scopes_metadata() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-main"), &root)
            .expect("compile context");
        let source = "module App.Entry\nimport Mylib.Copy (consume)\n\n\
             def main(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n\
             def second(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n";
        let artifact = compile_for_execution_in_context(&context, source, CompileTarget::C, None)
            .expect("in-context compile selects main");
        assert_eq!(
            artifact.inputs.len(),
            1,
            "entry must be scoped to `main` (1 input), not merged or `second`: {:?}",
            artifact.inputs
        );
        assert_eq!(artifact.outputs.len(), 1, "{:?}", artifact.outputs);
        let input_dims: Vec<Option<usize>> = artifact.inputs[0]
            .dims
            .iter()
            .map(|d| d.size.map(|size| usize::try_from(size.get()).unwrap()))
            .collect();
        assert_eq!(input_dims, vec![Some(2)], "main's input is tensor[2, f32]");
        let output_dims: Vec<Option<usize>> = artifact.outputs[0]
            .dims
            .iter()
            .map(|d| d.size.map(|size| usize::try_from(size.get()).unwrap()))
            .collect();
        assert_eq!(output_dims, vec![Some(2)]);
    }

    // Compiled-metadata-vs-eval agreement on the SAME in-context program: the
    // compiled artifact's output arity/shape must match what `eval_in_context`
    // actually computes, and the numeric value proves the correct library def
    // is invoked (identity `consume`, so `main([3, 4]) == [3, 4]`). The runtime
    // numeric-through-cc leg is the manual oracle's job (it needs
    // CHELIS_RUNTIME_DIR + a C toolchain); this gate proves eval and the
    // compiled interface agree without either.
    #[test]
    fn in_context_compiled_metadata_agrees_with_eval() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-eval"), &root)
            .expect("compile context");
        let source = "module App.Eval\nimport Mylib.Copy (consume)\n\n\
             def main(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n";

        let mut bindings = BTreeMap::new();
        bindings.insert(
            "x".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![3.0, 4.0]),
            },
        );
        let eval_result = eval_in_context_with_bindings(&context, source, bindings)
            .expect("eval in context succeeds");
        assert_eq!(eval_result.manifest.entries.len(), 1);
        assert_eq!(eval_result.manifest.entries[0].name, "main");
        assert_eq!(
            eval_result.manifest.entries[0].required_inputs,
            ["x".to_string()],
            "the concrete bound callable must be promoted into the consumed eval manifest"
        );
        let main_root = eval_result
            .roots
            .iter()
            .find(|r| r.name.as_deref() == Some("main"))
            .expect("main root present");
        let data = match &main_root.value {
            ExecutionValue::Tensor { value } => value.data.clone(),
            other => panic!("expected a tensor root, got {other:?}"),
        };
        assert_eq!(
            data,
            wire_values::storage_f32(vec![3.0, 4.0]),
            "identity consume(x) == x"
        );

        let artifact = compile_for_execution_in_context(&context, source, CompileTarget::C, None)
            .expect("in-context compile");
        assert_eq!(
            artifact.outputs.len(),
            1,
            "compiled output arity agrees with the single eval root"
        );
        let output_dims: Vec<Option<usize>> = artifact.outputs[0]
            .dims
            .iter()
            .map(|d| d.size.map(|size| usize::try_from(size.get()).unwrap()))
            .collect();
        assert_eq!(
            output_dims,
            vec![Some(2)],
            "compiled output shape agrees with the [2] eval value"
        );
    }

    #[test]
    fn manifested_callable_requires_only_reachable_runtime_inputs() {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            "live".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![3.0, 4.0]),
            },
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: "def main(dead: tensor[2, f32], live: tensor[2, f32]) \
                     -> tensor[2, f32] = add(live, live)\n"
                .to_string(),
            bindings,
        })
        .expect("a dead authored parameter is not a required runtime input");

        assert_eq!(result.manifest.entries.len(), 1);
        assert_eq!(result.manifest.entries[0].name, "main");
        assert_eq!(
            result.manifest.entries[0].required_inputs,
            ["live".to_string()]
        );
        assert_eq!(result.roots.len(), 1);
    }

    #[test]
    fn manifested_callable_expands_tuple_roots_with_originating_inputs() {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            "x".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![1.0, 2.0]),
            },
        );
        bindings.insert(
            "y".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![3.0, 4.0]),
            },
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: "def main(x: tensor[2, f32], y: tensor[2, f32]) \
                     -> (tensor[2, f32], tensor[2, f32]) = (x, y)\n"
                .to_string(),
            bindings,
        })
        .expect("a fully supplied tuple-returning callable is observable");

        let manifest = result
            .manifest
            .entries
            .iter()
            .map(|entry| (entry.name.as_str(), entry.required_inputs.as_slice()))
            .collect::<Vec<_>>();
        assert_eq!(
            manifest,
            vec![
                ("main.0", ["x".to_string(), "y".to_string()].as_slice()),
                ("main.1", ["x".to_string(), "y".to_string()].as_slice()),
            ],
            "[05-OBS-7..9] require selected callable products to use dotted \
             topology and inherit the originating call's reachable input closure"
        );
        assert_eq!(
            result
                .roots
                .iter()
                .map(|root| root.name.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("main.0"), Some("main.1")]
        );
    }

    #[test]
    fn manifested_callable_tuple_stays_a_declaration_when_an_input_is_missing() {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            "x".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![1.0, 2.0]),
            },
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: "def main(x: tensor[2, f32], y: tensor[2, f32]) \
                     -> (tensor[2, f32], tensor[2, f32]) = (x, y)\n"
                .to_string(),
            bindings,
        })
        .expect("an incompletely supplied callable remains a declaration");

        assert!(result.manifest.entries.is_empty());
        assert!(result.roots.is_empty());
    }

    #[test]
    fn manifested_callable_expands_static_record_adt_roots() {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            "x".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![1.0, 2.0]),
            },
        );
        bindings.insert(
            "y".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![3.0, 4.0]),
            },
        );
        let source = "type Pair =\n\
                     \x20 | Pair { left: tensor[2, f32], right: tensor[2, f32] }\n\n\
                     def main(x: tensor[2, f32], y: tensor[2, f32]) -> Pair = \
                     Pair { left: x, right: y }\n";
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings,
        })
        .expect("a fully supplied static-ADT callable is observable");

        assert_eq!(
            result
                .manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["main.left", "main.right"]
        );
        assert!(
            result
                .manifest
                .entries
                .iter()
                .all(|entry| { entry.required_inputs == vec!["x".to_string(), "y".to_string()] })
        );
        assert_eq!(
            result
                .roots
                .iter()
                .map(|root| root.name.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("main.left"), Some("main.right")]
        );
    }

    #[test]
    fn manifested_callable_static_adt_stays_a_declaration_when_an_input_is_missing() {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            "x".to_string(),
            crate::schema::TensorValue {
                shape: vec![2],
                data: wire_values::storage_f32(vec![1.0, 2.0]),
            },
        );
        let source = "type Pair =\n\
                     \x20 | Pair { left: tensor[2, f32], right: tensor[2, f32] }\n\n\
                     def main(x: tensor[2, f32], y: tensor[2, f32]) -> Pair = \
                     Pair { left: x, right: y }\n";
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings,
        })
        .expect("an incompletely supplied static-ADT callable remains a declaration");

        assert!(result.manifest.entries.is_empty());
        assert!(result.roots.is_empty());
    }

    // In-context scalar-entry rejection: a scalar-signature entry has no
    // callable tensor-kernel form, so `compile_for_execution_in_context` must
    // reject it with actionable wrap guidance (and name the requested
    // `entry_name` when one was passed) rather than emit unbuildable C.
    #[test]
    fn compile_for_execution_in_context_rejects_scalar_entry() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-scalar"), &root)
            .expect("compile context");
        let source = "def main(a: f32, b: f32) -> f32 = add(a, b)\n";
        let err =
            compile_for_execution_in_context(&context, source, CompileTarget::C, Some("main"))
                .expect_err("scalar in-context entry must be rejected");
        let message = &err.errors[0].message;
        assert!(
            message.contains("no callable tensor-kernel form"),
            "expected scalar-entry wrap guidance, got: {message}"
        );
        assert!(
            message.contains("`main`"),
            "rejection must name the requested entry_name, got: {message}"
        );
    }

    // Reef-context HIP is rejected as unsupported (chelis#829): the HIP arm
    // does not apply the entry-scoped DAG selection the C arm does, so rather
    // than silently emit a mis-scoped kernel it must reject with guidance
    // pointing at the C target. A clean tensor entry that compiles fine to C
    // in-context is used, so only the target differs.
    #[test]
    fn compile_for_execution_in_context_rejects_hip_target() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-hip"), &root)
            .expect("compile context");
        let source = "module App.Hip\nimport Mylib.Copy (consume)\n\n\
             def main(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n";
        // Sanity: the same source compiles in-context to C.
        compile_for_execution_in_context(&context, source, CompileTarget::C, None)
            .expect("in-context C compile succeeds");
        let err = compile_for_execution_in_context(&context, source, CompileTarget::Hip, None)
            .expect_err("reef-context HIP must be rejected");
        let message = &err.errors[0].message;
        // The remedy and the tracking issue are what this test is for. It
        // asserted the prose "only for the C target" until the rejection
        // moved onto the `Unsupported` channel, which restates the same
        // guidance as the actionable `target="c"` remediation clause. The
        // kind and the `unsupported:` brand are pinned separately, by
        // `compile_in_context_hip_rejects_as_branded_unsupported_feature`
        // in `tests/compiled_context.rs`.
        assert!(
            message.contains(r#"target="c""#) && message.contains("chelis#829"),
            "expected reef-context HIP rejection guidance naming the C target \
             and the tracking issue, got: {message}"
        );
    }

    // `main`-preference is a no-op when no def is named `main`: a single tensor
    // entry is still selected, and two non-`main` entries are ambiguous.
    #[test]
    fn resolve_in_context_entry_ambiguous_without_main() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-ambig"), &root)
            .expect("compile context");
        let source = "module App.Ambig\nimport Mylib.Copy (consume)\n\n\
             def first(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n\
             def other(y: tensor[2, f32]) -> tensor[2, f32] = realize(y)\n";
        let err = compile_for_execution_in_context(&context, source, CompileTarget::C, None)
            .expect_err("two non-main tensor entries are ambiguous");
        let message = &err.errors[0].message;
        assert!(
            message.contains("ambiguous entry") && message.contains("none named `main`"),
            "expected ambiguity guidance, got: {message}"
        );
    }

    // Reviewer finding 1 (#822 round 3): entry selection is EXACT-name only.
    // An earlier suffix-match fallback (`__<name>`) could never fire for its
    // stated purpose (library roots are never in `tensor_root_names`) and
    // instead silently compiled a DIFFERENT def: `def compute__solve` is
    // legal source and satisfied `entry_name = "solve"` with no diagnostic —
    // the #817 wrong-entry class. The selector must reject the near-miss
    // loudly, listing the real entries.
    #[test]
    fn resolve_in_context_entry_rejects_suffix_near_miss() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-suffix"), &root)
            .expect("compile context");
        let source = "module App.Sfx\nimport Mylib.Copy (consume)\n\n\
             def compute__solve(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n";
        let err =
            compile_for_execution_in_context(&context, source, CompileTarget::C, Some("solve"))
                .expect_err("a suffix near-miss must not silently select compute__solve");
        let message = &err.errors[0].message;
        assert!(
            message.contains("unknown entry_name `solve`") && message.contains("compute__solve"),
            "expected the exact-match unknown-entry error listing the real def, got: {message}"
        );
    }

    // Positive parity for the exact-match rule: the same def selected by its
    // real (double-underscore) name compiles and scopes correctly.
    #[test]
    fn resolve_in_context_entry_selects_double_underscore_def_by_exact_name() {
        let (_dir, root) = copy_drop_context_fixture();
        let context = crate::compile_reef_context(Path::new("/tmp/inctx-suffix-pos"), &root)
            .expect("compile context");
        let source = "module App.Sfx\nimport Mylib.Copy (consume)\n\n\
             def compute__solve(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n";
        let artifact = compile_for_execution_in_context(
            &context,
            source,
            CompileTarget::C,
            Some("compute__solve"),
        )
        .expect("exact name must select the def");
        assert_eq!(artifact.inputs.len(), 1, "{:?}", artifact.inputs);
        assert_eq!(artifact.inputs[0].name, "x");
    }

    // The old "Fix A" invariant guard (#822 review) rejected zero named
    // tensor roots alongside a rootful DAG, synthesized by clearing
    // `tensor_root_names` by hand. The #1013 pipeline made that state
    // unrepresentable: `NamedRoots::aligned` (in `finish_lowering`) verifies
    // the name/root correspondence at construction and rejects any count
    // mismatch as a typed `PipelineRejection::RootCount` before a
    // `CompiledSource` exists, and `resolve_in_context_entry` now resolves
    // name-to-node through that typed map rather than by positional index.
    // The guard and its synthesized-mismatch test are therefore retired; the
    // type-level lock lives in `pipeline::NamedRoots` and its tests.

    #[test]
    fn compile_source_preserves_scalar_string_foundation_root_names() {
        let source = include_str!("../../../examples/scalar_string_foundation.ch");

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");

        assert!(compiled.named_roots.is_empty());
        assert!(
            compiled
                .manifest()
                .entries
                .iter()
                .any(|entry| entry.name == "status")
        );
        assert!(
            compiled
                .manifest()
                .entries
                .iter()
                .any(|entry| entry.name == "loss")
        );
        assert!(
            compiled
                .manifest()
                .entries
                .iter()
                .any(|entry| entry.name == "should_stop")
        );
    }

    #[test]
    fn compile_source_accepts_typed_param_named_let() {
        let source = r#"
def id(let: int64) -> int64 = let
"#;

        let compiled = compile_source(SourceKind::Surf, source).expect("compile");
        let deep = chelis_deep::printer::print_canonical(compiled.checked().exprs());
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
type Jsonish =
  | JsonNull
  | JsonInt(int64)
  | JsonString(string)
  | JsonArray(List[Jsonish])

def describe(value: Jsonish) -> string =
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
                    data: wire_values::storage_f32(vec![0.0; 6]),
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
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(2)))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("dims.1")
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(3)))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("dims.2")
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(6)))
        );
    }

    #[test]
    fn eval_rejects_tagged_binding_dtype_substitution() {
        let err = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: "x: tensor[1, f32] = x\n".to_string(),
            bindings: BTreeMap::from([(
                "x".to_string(),
                crate::schema::TensorValue {
                    shape: vec![1],
                    data: serde_json::from_value(
                        serde_json::json!({"dtype":"f16","bits":["3e00"]}),
                    )
                    .unwrap(),
                },
            )]),
        })
        .expect_err("a tagged f16 binding must not be contextually cast to f32");
        let message = err
            .errors
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            message.contains("f16"),
            "source dtype must be named: {message}"
        );
        assert!(
            message.contains("f32"),
            "declared dtype must be named: {message}"
        );
        assert!(message.contains("casts are explicit"), "{message}");
    }

    #[test]
    fn eval_supports_string_parse_helpers_and_option_matching() {
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
parsed = match to_int(" 42 ") with {
  | Some(n) => n
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
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(42)))
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
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(7)))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("rem")
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(2)))
        );
        assert!(
            result
                .roots
                .iter()
                .any(|root| root.name.as_deref() == Some("shifted")
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(4)))
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
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(6)))
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
                    && matches!(root.value, ExecutionValue::Scalar { value } if value.get().prim() == Prim::Int64 && value.get().as_i64_exact() == Some(2024)))
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

type Jsonish =
  | JsonNull
  | JsonString(string)
  | JsonArray(List[Jsonish])
  | JsonObject(Dict[string, Jsonish])

type Tokenizer =
  | BpeTokenizer(Dict[string, int64], Dict[string, int64], Dict[int64, string], int64)

def parse_line(line: string) -> Option[List[string]] =
  Some([])

def json_string(value: Option[Jsonish]) -> Option[string] =
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

        let host = chelis_ir::host::try_lower_compiled_program(compiled.checked())
            .expect("checked host program must lower")
            .host
            .expect("host lowering");
        let lowered = chelis_ir::lower::top_level_lowering_map(
            compiled.checked().exprs(),
            compiled.checked().type_env(),
        );
        let checked_text = chelis_deep::printer::print_canonical(compiled.checked().exprs());

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
            Some(chelis_ir::host_type_state::ConcreteHostType::Option(
                Box::new(chelis_ir::host_type_state::ConcreteHostType::List(
                    Box::new(chelis_ir::host_type_state::ConcreteHostType::String)
                ))
            )),
            "available functions: {available:#?}\nlowered: {lowered_debug:#?}\nchecked:\n{checked_text}"
        );
        assert_eq!(
            find_ret("json_string"),
            Some(chelis_ir::host_type_state::ConcreteHostType::Option(
                Box::new(chelis_ir::host_type_state::ConcreteHostType::String)
            )),
            "available functions: {available:#?}\nlowered: {lowered_debug:#?}\nchecked:\n{checked_text}"
        );
        assert_eq!(
            find_ret("load_tokenizer"),
            Some(chelis_ir::host_type_state::ConcreteHostType::Option(
                Box::new(chelis_ir::host_type_state::ConcreteHostType::Adt(
                    "Tokenizer".to_string(),
                    Vec::new()
                ))
            )),
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
                data: wire_values::storage_f32(vec![1.0, 2.0]),
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
            b_err.errors.iter().any(|diag| {
                diag.message.contains('b') || diag.kind() == chelis_vocab::DiagnosticKind::EvalError
            }),
            "expected eval error mentioning `b`, got {:?}",
            b_err.errors
        );
    }

    #[test]
    fn manifested_tensor_roots_carry_only_their_own_lowered_inputs() {
        let compiled = compile_source(
            SourceKind::Surf,
            "a: tensor[2, f32] = a\nb: tensor[2, f32] = b\n",
        )
        .expect("compile independent input roots");
        let inputs_for = |name: &str| {
            compiled
                .manifest()
                .entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("missing manifest entry {name}"))
                .required_inputs
                .clone()
        };
        assert_eq!(inputs_for("a"), ["a".to_string()].into_iter().collect());
        assert_eq!(inputs_for("b"), ["b".to_string()].into_iter().collect());
    }

    #[test]
    fn manifested_host_root_inherits_only_its_top_level_input_dependency() {
        let compiled = compile_source_for_target(
            SourceKind::Surf,
            "input: tensor[2, f64] = input\nother: tensor[2, f64] = other\n\
             values = to_list(input)\n",
            Target::C,
        )
        .expect("compile Host root with one runtime input");
        let entry = compiled
            .manifest()
            .entries
            .iter()
            .find(|entry| entry.name == "values")
            .expect("values manifest entry");
        assert_eq!(entry.lane, Lane::Host);
        assert_eq!(
            entry.required_inputs,
            ["input".to_string()].into_iter().collect()
        );
    }

    #[test]
    fn unavailable_owed_root_fails_with_named_lane_and_authority() {
        let mut compiled = compile_source(SourceKind::Surf, "answer = cast(42, int32)\n")
            .expect("compile fault-injection fixture");
        let mut manifest = compiled.manifest().clone();
        let mut unavailable = manifest.entries[0].clone();
        unavailable.name = "missing_root".to_string();
        unavailable.lane = Lane::Tensor;
        manifest.entries.push(unavailable);
        compiled.program =
            ManifestedProgram::new(compiled.checked().clone(), manifest, Target::Eval);

        let error = eval_compiled(&compiled, BTreeMap::new(), None)
            .expect_err("an owed root absent from its assigned lane must fail loudly");
        let diagnostic = error.errors.first().expect("one typed diagnostic");
        assert_eq!(
            diagnostic.kind(),
            chelis_vocab::DiagnosticKind::UnsupportedFeature
        );
        assert!(diagnostic.message.contains("[05-UNS-1]"));
        assert!(diagnostic.message.contains("missing_root"));
        assert!(diagnostic.message.contains("Tensor lane"));
        assert!(diagnostic.message.contains("unimplemented chelis#912"));
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
                data: wire_values::storage_f32(vec![1.0, 2.0]),
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
        // The -1 is laundered through runtime arithmetic (`0 - 1`) so
        // the checker cannot see it and the runtime `shape` arm owns
        // the rejection. The original form `cast(-1, int32)` stopped
        // exercising this path with issue #308: the desugarer now
        // folds the sign into the literal (spec §5.6 position 4), so
        // the binding takes the same tensor lane the positive-literal
        // form `cast(1, int32)` always took, and the axis arrives as a
        // rank-0 tensor rather than a host int scalar.
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: r#"
axis = tensor_to_scalar(scalar_to_tensor(cast(0 - 1, int32)))
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

    // WI-2 validate-on-consume (WS-5 Part A). The schema-version check at the
    // WireDag process boundary must fail closed on an unsupported version with
    // a typed `schema`-stage error, accept a supported version, and never
    // silently pass an unknown surface through.

    fn wire_dag_at_version(version: u32) -> WireDag {
        WireDag {
            schema_version: version,
            nodes: Vec::new(),
            roots: Vec::new(),
        }
    }

    #[test]
    fn schema_stage_check_accepts_supported_version() {
        let dag = wire_dag_at_version(crate::schema::WIRE_DAG_SCHEMA_VERSION);
        schema_stage_check(dag.validate_schema_version())
            .expect("the supported schema version is accepted at the consume boundary");
    }

    #[test]
    fn schema_stage_check_rejects_legacy_lower_version() {
        let legacy = wire_dag_at_version(0);
        let error = schema_stage_check(legacy.validate_schema_version())
            .expect_err("a lower-than-current version is rejected");
        let diagnostic = &error.errors[0];
        assert_eq!(diagnostic.got.as_deref(), Some("0"));
        assert_eq!(
            diagnostic.expected.as_deref(),
            Some(
                format!(
                    "schema_version = {}",
                    crate::schema::WIRE_DAG_SCHEMA_VERSION
                )
                .as_str()
            )
        );
    }

    #[test]
    fn schema_stage_check_rejects_over_version_with_typed_error() {
        let future = crate::schema::WIRE_DAG_SCHEMA_VERSION + 1;
        let dag = wire_dag_at_version(future);
        let error = schema_stage_check(dag.validate_schema_version())
            .expect_err("an over-version DAG is rejected at the consume boundary");
        assert_eq!(
            error.stage, "schema",
            "rejection is a typed schema-stage error"
        );
        assert_eq!(error.errors.len(), 1, "exactly one diagnostic: {error:?}");
        let diagnostic = &error.errors[0];
        assert_eq!(
            diagnostic.kind(),
            chelis_vocab::DiagnosticKind::UnknownSchemaVersion
        );
        assert_eq!(
            diagnostic.got.as_deref(),
            Some(future.to_string().as_str()),
            "the diagnostic reports the unsupported version it was handed"
        );
        assert_eq!(
            diagnostic.expected.as_deref(),
            Some(
                format!(
                    "schema_version = {}",
                    crate::schema::WIRE_DAG_SCHEMA_VERSION
                )
                .as_str()
            ),
            "the diagnostic reports the supported ceiling"
        );
    }

    #[test]
    fn lower_happy_path_passes_schema_validation() {
        let result = lower(LowerRequest {
            source_kind: SourceKind::Surf,
            source: "a = (a : tensor[2, 3, f32])\n\
                     b = (b : tensor[3, 4, f32])\n\
                     out = (matmul(a, b) : tensor[2, 4, f32])\n"
                .to_string(),
            entry: None,
        })
        .expect("lower succeeds");
        // The DAG the server hands back is at the supported version, so the
        // boundary validation it just ran accepted it.
        assert_eq!(
            result.dag.schema_version,
            crate::schema::WIRE_DAG_SCHEMA_VERSION
        );
        result
            .dag
            .validate_schema_version()
            .expect("the produced DAG validates");
    }

    #[test]
    fn grad_happy_path_passes_schema_validation() {
        let result = grad(GradRequest {
            source_kind: SourceKind::Surf,
            source: "x = (x : tensor[f32])\n\
                     out = (mul(x, x) : tensor[f32])\n"
                .to_string(),
            output_name: "out".to_string(),
            wrt_names: vec!["x".to_string()],
            fuse: true,
        })
        .expect("grad succeeds");
        assert_eq!(
            result.dag.schema_version,
            crate::schema::WIRE_DAG_SCHEMA_VERSION
        );
        result
            .dag
            .validate_schema_version()
            .expect("the produced gradient DAG validates");
    }
}
