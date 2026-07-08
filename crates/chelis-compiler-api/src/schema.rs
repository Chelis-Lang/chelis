use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Surf,
    Deep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CompileTarget {
    C,
    Hip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidateMode {
    Surf,
    Deep,
    Desugar,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApiSuccess<T> {
    pub ok: bool,
    pub result: T,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApiFailure {
    pub ok: bool,
    pub stage: String,
    pub errors: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ApiEnvelope<T> {
    Success(ApiSuccess<T>),
    Failure(ApiFailure),
}

impl<T> ApiEnvelope<T> {
    pub fn success(result: T) -> Self {
        Self::Success(ApiSuccess { ok: true, result })
    }

    pub fn failure(stage: impl Into<String>, errors: Vec<Diagnostic>) -> Self {
        Self::Failure(ApiFailure {
            ok: false,
            stage: stage.into(),
            errors,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Diagnostic {
    pub kind: String,
    pub message: String,
    pub severity: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub got: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
    /// Forward-compatible Deep-address slot for the L2 authoring loop. The
    /// fragment body-replacement check (`chelis_replace_function_body`) will
    /// populate this with the Deep path of the offending node so a caller can
    /// pinpoint the rejected subtree without re-deriving it. It is `None`
    /// today (L0; provenance threading is L2 work), and every other tool
    /// leaves it `None`, so the `skip_serializing_if` keeps their wire output
    /// byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep_path: Option<WireDeepErrorPath>,
}

/// Wire form of `chelis_compiler_api::fragment::DeepErrorPath`: the Deep
/// address of an offending node, relative to the def it lives in. Serialized
/// as the dot-joined path string plus the owning def's qualified name so the
/// shape does not couple the wire surface to the internal `DeepPath` type. It
/// is never emitted in L0 (the inner `deep_path` is always `None` there); the
/// type exists so populating it in L2 adds no new field to [`Diagnostic`].
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WireDeepErrorPath {
    /// The qualified name of the def the address is relative to.
    pub def_qualified_name: String,
    /// The path from the def node to the offending subtree, as the canonical
    /// dot-joined `DeepPath` rendering.
    pub path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Span {
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TensorValue {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DictEntryValue {
    pub key: ExecutionValue,
    pub value: ExecutionValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionValue {
    Tensor {
        value: TensorValue,
    },
    Int64 {
        value: i64,
    },
    Float64 {
        value: f64,
    },
    Bool {
        value: bool,
    },
    String {
        value: String,
    },
    List {
        value: Vec<ExecutionValue>,
    },
    Dict {
        entries: Vec<DictEntryValue>,
    },
    Tuple {
        value: Vec<ExecutionValue>,
    },
    Adt {
        ctor: String,
        fields: Vec<ExecutionValue>,
    },
    Unit,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParseRequest {
    pub source_kind: SourceKind,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub source_kind: SourceKind,
    pub surf_ast: Option<Vec<WireSurfDecl>>,
    pub deep_ast: Option<Vec<WireDeepExpr>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DesugarRequest {
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesugarResult {
    pub deep_text: String,
    pub deep_ast: Vec<WireDeepExpr>,
}

/// Request for `chelis_replace_function_body`: replace one function's body in
/// a Deep module with a new Deep body expression, returning the canonical Deep
/// of the changed def and the full rewritten module. Deep-native: both
/// `module` and `new_body` are Deep s-expression text; there is no Surf
/// ingestion path. The tool is pure — it persists nothing and writes no file.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReplaceFunctionBodyRequest {
    /// The full module as canonical Deep (`.dp`) text.
    pub module: String,
    /// The qualified (or bare) name of the function whose body to replace.
    pub function_name: String,
    /// The new function body as a single Deep s-expression.
    pub new_body: String,
}

/// Result of a clean `chelis_replace_function_body`: the changed def and the
/// full rewritten module, both in canonical Deep. The L0 validation runs full
/// whole-module `chelis check` on `module_deep`, so the verdict EQUALS full
/// `chelis check` of the rewritten module by construction (the splice-faithfulness
/// gate locks that the rewrite is the module full check is run on); a returned
/// result is a module that type-, effect-, and linearity-checks. Closure-scoped
/// validation (caller-ward effect closure, the def's SCC for termination,
/// per-def for type) is the future optimization, not a fragment-scoped path that
/// could disagree.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReplaceFunctionBodyResult {
    /// Canonical Deep of just the changed `(def ...)` node.
    pub changed_def_deep: String,
    /// Canonical Deep of the full rewritten module.
    pub module_deep: String,
}

/// Request for `chelis_add_function`: insert a new Deep function declaration
/// bundle into a Deep module and return canonical Deep only if the rewritten
/// whole module validates. The bundle is Deep text containing exactly one
/// `(def ...)` and an optional matching `(defsig ...)`; no export declaration is
/// accepted here and the tool preserves existing exports unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddFunctionRequest {
    /// The full module as canonical Deep (`.dp`) text.
    pub module: String,
    /// Deep text containing the function declaration bundle to insert.
    pub new_decls: String,
    /// Optional qualified (or bare) function name. When present, the new bundle
    /// is inserted immediately after that function's existing def/defsig
    /// declaration bundle; otherwise it is appended to the module decl list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insert_after_function: Option<String>,
}

/// Result of a clean `chelis_add_function`: the inserted declarations and the
/// full rewritten module, all in canonical Deep. A returned result means the
/// post-insertion whole-module pipeline accepted the module.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddFunctionResult {
    /// Canonical Deep of the inserted `(def ...)` node.
    pub added_def_deep: String,
    /// Canonical Deep of the inserted `(defsig ...)` node, when one was
    /// authored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_defsig_deep: Option<String>,
    /// Canonical Deep of the full rewritten module.
    pub module_deep: String,
}

/// Query the stable Deep authoring outline for one module.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepOutlineRequest {
    /// The full module as canonical Deep (`.dp`) text.
    pub module: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepOutlineResult {
    pub module_name: String,
    pub exports: Vec<String>,
    pub functions: Vec<DeepFunctionOutline>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepFunctionOutline {
    pub name: String,
    pub qualified_name: String,
    pub params: Vec<String>,
    pub has_defsig: bool,
    pub body_path: String,
    pub def_deep: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defsig_deep: Option<String>,
    /// SHA-256 over the canonical Deep of this function's `(def ...)` node.
    /// Edit tools use this as an optional optimistic-concurrency preimage:
    /// mismatch means the target node is not the node the caller planned over,
    /// so the whole edit fails closed before validation.
    pub preimage_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepReferencesRequest {
    pub module: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepReferencesResult {
    pub symbol: String,
    pub references: Vec<DeepReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepCallGraphRequest {
    pub module: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepCallGraphResult {
    pub edges: Vec<DeepReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepReference {
    pub caller: String,
    pub callee: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReplaceFunctionRequest {
    pub module: String,
    pub function_name: String,
    pub new_decls: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preimage_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReplaceFunctionResult {
    pub replaced_def_deep: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replaced_defsig_deep: Option<String>,
    pub module_deep: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RenameRequest {
    pub module: String,
    pub function_name: String,
    pub new_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preimage_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RenameResult {
    pub renamed_def_deep: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renamed_defsig_deep: Option<String>,
    pub module_deep: String,
    pub renamed_references: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChangeSignatureRequest {
    pub module: String,
    pub function_name: String,
    pub new_defsig: String,
    pub new_params: String,
    /// Old parameter names in the order call-site arguments should appear
    /// after the signature change. A mismatch is a request-shape error. The
    /// cascade is fail-closed: every direct call in the pre-edit call graph is
    /// rewritten or the whole tool call fails.
    pub argument_order: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub param_renames: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preimage_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChangeSignatureResult {
    pub changed_def_deep: String,
    pub changed_defsig_deep: String,
    pub module_deep: String,
    pub rewritten_calls: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddPropertyRequest {
    pub module: String,
    pub new_decls: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insert_after_function: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddPropertyResult {
    pub added_property_def_deep: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_defsig_deep: Option<String>,
    pub module_deep: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CheckRequest {
    pub source_kind: SourceKind,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessComponents {
    pub parse: f64,
    pub structure: f64,
    pub names: f64,
    pub types: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub score: f64,
    pub components: FitnessComponents,
    pub typed_nodes: usize,
    pub untyped_nodes: usize,
    pub total_nodes: usize,
    pub unresolved_names: Vec<String>,
    pub errors: Vec<Diagnostic>,
}

/// Structured, machine-readable inferred-type tree for one inferred
/// function signature, emitted by `chelis check --show-inferred --json`.
///
/// This is the lossless counterpart to the human-facing `display_*`
/// strings already produced by the CLI's `format_cli_type`. A consumer
/// (e.g. Hull) can reconstruct a `chelis_types::Type` from this tree
/// directly instead of re-parsing a type printer. The shape mirrors
/// `chelis_types::types::Type` one variant at a time and is internally
/// tagged on `kind` so the JSON is self-describing.
///
/// Stability contract: every variant name here is pinned to its
/// `chelis_types::types::Type` source variant. Adding a new `Type`
/// variant is a breaking change to this wire shape and must add the
/// matching variant here in the same change set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireInferredType {
    /// `Type::Prim` — a scalar primitive (`f32`, `int64`, `bool`, ...).
    /// `name` is the canonical `Prim::name()` spelling.
    Prim { name: String },
    /// `Type::Fn` — function type. `args` are the parameter types in
    /// order; `ret` is the return type.
    Fn {
        args: Vec<WireInferredType>,
        ret: Box<WireInferredType>,
    },
    /// `Type::Ref` — a read-only non-owning borrow of `inner`.
    Ref { inner: Box<WireInferredType> },
    /// `Type::Tensor` — `dims` in order plus a `precision` slot.
    Tensor {
        dims: Vec<WireInferredDim>,
        precision: WireInferredPrecision,
    },
    /// `Type::Adt` — a named algebraic data type with optional type
    /// arguments (empty `args` for a nullary ADT).
    Adt {
        name: String,
        args: Vec<WireInferredType>,
    },
    /// `Type::Var` — an unresolved inference type variable. `id` is the
    /// raw `TypeVar` index, matching the `?N` display rendering.
    Var { id: u32 },
    /// `Type::Tuple` — an ordered tuple of element types.
    Tuple { items: Vec<WireInferredType> },
    /// `Type::Unit` — the unit type.
    Unit,
    /// `Type::Error` — the partial-inference error sentinel. Present so
    /// the structured tree never silently drops a node; a consumer
    /// should treat this as "type unknown due to an upstream error".
    Error,
}

/// Structured tensor dimension, mirroring `chelis_types::types::Dim`.
/// Internally tagged on `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireInferredDim {
    /// `Dim::Name` — a concrete named dimension (e.g. `batch`).
    Name { name: String },
    /// `Dim::Var` — a polymorphic dimension variable. `id` is the raw
    /// `DimVar` index, matching the `dN` display rendering.
    Var { id: u32 },
    /// `Dim::Lit` — a fixed numeric size.
    Lit { size: i64 },
    /// `Dim::Wildcard` — an unknown / dynamic dimension (the `*`
    /// display rendering).
    Wildcard,
    /// `Dim::Rank` — a rank variable standing for an entire shape vector
    /// (Tier-2 rank polymorphism). `id` is the raw `RankVar` index, matching
    /// the `..rN` display rendering.
    Rank { id: u32 },
}

/// Structured tensor precision slot, mirroring
/// `chelis_types::types::TensorPrec`. Internally tagged on `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireInferredPrecision {
    /// `TensorPrec::Concrete` — a resolved numeric primitive. `name`
    /// is the canonical `Prim::name()` spelling.
    Concrete { name: String },
    /// `TensorPrec::Var` — a still-polymorphic precision variable.
    /// `id` is the raw `TypeVar` index, matching the `?N` rendering.
    Var { id: u32 },
}

/// One inferred effect, mirroring `chelis_types::types::Effect`.
/// Internally tagged on `kind`. The `kind` discriminant is the
/// lowercase spelling; `Resource` additionally carries its `device`
/// string. Consumers that want the human Display spelling
/// (`Random`/`Accum`/`IO`/`Test`/`Resource("dev")`) can reconstruct it
/// from `kind` + `device`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireInferredEffect {
    /// `Effect::Random`.
    Random,
    /// `Effect::Accum`.
    Accum,
    /// `Effect::Io`.
    Io,
    /// `Effect::Test`.
    Test,
    /// `Effect::Resource(device)`.
    Resource { device: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LowerRequest {
    pub source_kind: SourceKind,
    pub source: String,
    /// Optional single-entry scoping: when set, lowering is restricted to the
    /// defs reachable from this named entry, dropping unrelated top-level
    /// functions before the type checker runs. This lets a caller extract one
    /// function from a module that also defines unrelated functions
    /// referencing unresolved imports, without those unrelated functions
    /// blocking the target's lowering. `None` lowers the whole program (the
    /// default, and the only behavior before this field existed). Pruning
    /// drops only genuinely-unreachable defs, so an unresolved symbol in the
    /// ENTRY's own closure still surfaces as a lowering error.
    ///
    /// `skip_serializing_if` keeps the wire output clean for the common
    /// whole-program case (no `"entry": null`), matching the other optional
    /// request fields in this module.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerResult {
    pub dag: WireDag,
    pub named_roots: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CompileRequest {
    pub source_kind: SourceKind,
    pub source: String,
    pub target: CompileTarget,
    #[serde(default)]
    pub entry_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedFile {
    pub path: String,
    pub contents: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub target: CompileTarget,
    pub entry_name: String,
    pub files: Vec<GeneratedFile>,
    pub compile_flags: Vec<String>,
    pub link_flags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_device_bytes_estimate: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EvalRequest {
    pub source_kind: SourceKind,
    pub source: String,
    #[serde(default)]
    pub bindings: BTreeMap<String, TensorValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatedRoot {
    pub node_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub value: ExecutionValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub roots: Vec<EvaluatedRoot>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transcript: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GradRequest {
    pub source_kind: SourceKind,
    pub source: String,
    pub output_name: String,
    pub wrt_names: Vec<String>,
    #[serde(default = "default_true")]
    pub fuse: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradResult {
    pub dag: WireDag,
    pub output_node: usize,
    pub grad_nodes_by_name: BTreeMap<String, usize>,
    pub forward_nodes_by_name: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ValidateRequest {
    pub mode: ValidateMode,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateResult {
    pub mode: ValidateMode,
    pub valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DecompileRequest {
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompileResult {
    pub surf_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatchRequest {
    Parse(ParseRequest),
    Desugar(DesugarRequest),
    Check(CheckRequest),
    Lower(LowerRequest),
    Compile(CompileRequest),
    Eval(EvalRequest),
    Grad(GradRequest),
    Validate(ValidateRequest),
    Decompile(DecompileRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestEnvelope {
    pub requests: Vec<BatchRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatchResult {
    Parse(ApiEnvelope<ParseResult>),
    Desugar(ApiEnvelope<DesugarResult>),
    Check(ApiEnvelope<CheckResult>),
    Lower(ApiEnvelope<LowerResult>),
    Compile(ApiEnvelope<CompileResult>),
    Eval(ApiEnvelope<EvalResult>),
    Grad(ApiEnvelope<GradResult>),
    Validate(ApiEnvelope<ValidateResult>),
    Decompile(ApiEnvelope<DecompileResult>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResultEnvelope {
    pub results: Vec<BatchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WireDeepExpr {
    pub kind: WireDeepExprKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDeepExprKind {
    Atom {
        atom: WireDeepAtom,
    },
    List {
        elements: Vec<WireDeepExpr>,
    },
    Map {
        entries: Vec<WireMetaEntry>,
    },
    MetaExpr {
        entries: Vec<WireMetaEntry>,
        expr: Box<WireDeepExpr>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDeepAtom {
    Symbol { value: String },
    Int { value: i64 },
    Float { value: f64 },
    Str { value: String },
    Keyword { value: String },
    Bool { value: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WireMetaEntry {
    pub key: String,
    pub value: WireDeepExpr,
}

/// Wire form of a declared opaque-type invariant (RFC D-SYNTAX).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTypeInvariant {
    pub binder: String,
    pub body: WireSurfExpr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireSurfDecl {
    Module {
        name: String,
        decls: Vec<WireSurfDecl>,
        span: Span,
    },
    Import {
        module: String,
        import_kind: WireImportKind,
        span: Span,
    },
    Sig {
        name: String,
        ty: WireSurfTypeExpr,
        span: Span,
    },
    Dim {
        names: Vec<String>,
        span: Span,
    },
    TypeDef {
        name: String,
        params: Vec<String>,
        variants: Vec<WireVariant>,
        opaque: bool,
        /// Declared invariant for an opaque type (RFC D-SYNTAX).
        /// Additive: absent on the wire for non-opaque or
        /// invariant-free types.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        invariant: Option<WireTypeInvariant>,
        span: Span,
    },
    TypeAlias {
        name: String,
        params: Vec<String>,
        ty: WireSurfTypeExpr,
        span: Span,
    },
    MacroDef {
        name: String,
        params: Vec<String>,
        body: WireSurfExpr,
        span: Span,
    },
    FunDef {
        name: String,
        dim_params: Vec<String>,
        params: Vec<WireParam>,
        ret_ty: Option<WireSurfTypeExpr>,
        body: WireSurfExpr,
        span: Span,
    },
    Property {
        name: String,
        params: Vec<WireParam>,
        preconditions: Vec<WireSurfExpr>,
        body: WireSurfExpr,
        options: Vec<WirePropertyOption>,
        span: Span,
    },
    LetDef {
        name: String,
        ty: Option<WireSurfTypeExpr>,
        value: WireSurfExpr,
        span: Span,
    },
    Export {
        names: Vec<String>,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WirePropertyOption {
    Tolerance { value: WireSurfExpr, span: Span },
    Seed { value: WireSurfExpr, span: Span },
    Samples { value: WireSurfExpr, span: Span },
    Contract { id: String, span: Span },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireImportKind {
    Qualified,
    All,
    Names { names: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireVariantFields {
    Positional { fields: Vec<WireSurfTypeExpr> },
    Record { fields: Vec<WireRecordTypeField> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireVariant {
    pub name: String,
    pub fields: WireVariantFields,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireParam {
    pub name: String,
    pub ty: Option<WireSurfTypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireSurfExpr {
    Lit {
        literal: WireLiteral,
        span: Span,
    },
    Var {
        name: String,
        span: Span,
    },
    Constructor {
        name: String,
        span: Span,
    },
    Apply {
        func: Box<WireSurfExpr>,
        args: Vec<WireSurfExpr>,
        span: Span,
    },
    List {
        items: Vec<WireSurfExpr>,
        span: Span,
    },
    Record {
        name: String,
        fields: Vec<WireRecordExprField>,
        span: Span,
    },
    Access {
        expr: Box<WireSurfExpr>,
        field: String,
        span: Span,
    },
    TupleGet {
        expr: Box<WireSurfExpr>,
        index: i64,
        span: Span,
    },
    Binary {
        op: WireBinOp,
        lhs: Box<WireSurfExpr>,
        rhs: Box<WireSurfExpr>,
        span: Span,
    },
    Unary {
        op: WireUnaryOp,
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    Pipe {
        expr: Box<WireSurfExpr>,
        stages: Vec<WireSurfExpr>,
        span: Span,
    },
    If {
        cond: Box<WireSurfExpr>,
        then_branch: Box<WireSurfExpr>,
        else_branch: Box<WireSurfExpr>,
        span: Span,
    },
    Match {
        expr: Box<WireSurfExpr>,
        arms: Vec<WireMatchArm>,
        span: Span,
    },
    Lambda {
        params: Vec<WireParam>,
        body: Box<WireSurfExpr>,
        span: Span,
    },
    Tuple {
        items: Vec<WireSurfExpr>,
        span: Span,
    },
    Cast {
        expr: Box<WireSurfExpr>,
        ty: String,
        span: Span,
    },
    Grad {
        expr: Box<WireSurfExpr>,
        #[serde(skip_serializing_if = "Option::is_none")]
        wrt: Option<Vec<String>>,
        span: Span,
    },
    Vmap {
        expr: Box<WireSurfExpr>,
        axis: Option<i64>,
        span: Span,
    },
    Jit {
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    Realize {
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    Copy {
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    Borrow {
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    WithSeed {
        seed: Box<WireSurfExpr>,
        body: Box<WireSurfExpr>,
        span: Span,
    },
    WithDevice {
        device: Box<WireSurfExpr>,
        body: Box<WireSurfExpr>,
        span: Span,
    },
    Par {
        exprs: Vec<WireSurfExpr>,
        span: Span,
    },
    Annotate {
        expr: Box<WireSurfExpr>,
        ty: WireSurfTypeExpr,
        span: Span,
    },
    Block {
        bindings: Vec<WireLetBinding>,
        body: Box<WireSurfExpr>,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireLiteral {
    Int {
        value: i64,
    },
    Float {
        value: f64,
    },
    /// Integer literal carrying an explicit precision suffix per spec
    /// §5.5. Suffix is one of `i8`/`i16`/`i32`/`i64`/`f32`/`f64`/`bf16`/`f16`.
    TypedInt {
        value: i64,
        suffix: String,
    },
    TypedFloat {
        value: f64,
        suffix: String,
    },
    Str {
        value: String,
    },
    Bool {
        value: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireUnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMatchArm {
    pub pattern: WirePattern,
    pub guard: Option<WireSurfExpr>,
    pub body: WireSurfExpr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WirePattern {
    Wildcard {
        span: Span,
    },
    Var {
        name: String,
        span: Span,
    },
    Lit {
        literal: WireLiteral,
        span: Span,
    },
    Constructor {
        name: String,
        args: Vec<WirePattern>,
        span: Span,
    },
    Tuple {
        items: Vec<WirePattern>,
        span: Span,
    },
    Record {
        name: String,
        fields: Vec<WireRecordPatternField>,
        span: Span,
    },
    As {
        name: String,
        pattern: Box<WirePattern>,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireLetBinding {
    pub pattern: WireLetPattern,
    pub ty: Option<WireSurfTypeExpr>,
    pub value: WireSurfExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireLetPattern {
    Var {
        name: String,
        span: Span,
    },
    Wildcard {
        span: Span,
    },
    Tuple {
        items: Vec<WireLetPattern>,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireSurfTypeExpr {
    Named {
        name: String,
        span: Span,
    },
    Tensor {
        dims: Vec<WireSurfTypeExpr>,
        precision: String,
        span: Span,
    },
    Arrow {
        args: Vec<WireSurfTypeExpr>,
        ret: Box<WireSurfTypeExpr>,
        span: Span,
    },
    Ref {
        inner: Box<WireSurfTypeExpr>,
        span: Span,
    },
    App {
        name: String,
        args: Vec<WireSurfTypeExpr>,
        span: Span,
    },
    Tuple {
        items: Vec<WireSurfTypeExpr>,
        span: Span,
    },
    Infer {
        span: Span,
    },
    /// `..r` rank-variable spread (Tier-2 rank polymorphism).
    RankSpread {
        name: String,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRecordExprField {
    pub name: String,
    pub value: WireSurfExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRecordTypeField {
    pub name: String,
    pub ty: WireSurfTypeExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRecordPatternField {
    pub name: String,
    pub pattern: WirePattern,
}

/// Current monotonic schema version of the serialized [`WireDag`] surface
/// (master plan WI-2, `spec/design/verification_stack_master_plan.md`
/// §4.1). Bump this whenever the wire shape of the IR DAG changes in a
/// way a pinned consumer (Beacon's transformers, an offline verifier)
/// must observe. Bumping is monotonic and never reused: a higher number
/// always means "newer than" a lower one.
///
/// Version history:
/// - `1`: initial pinned surface (46-variant `WireRiscOp`, additive
///   `accumulator` defaults on `Sum`/`BlasMatmul`).
/// - `2`: chelis#178 — added `WireRiscOp::FloorDiv` / `TruncDiv` and the
///   matching `WireFusedStepOp` variants for integer floor/truncating
///   division (`div` is now float-only). A producer may emit the new ops,
///   so a pinned consumer must observe the version bump.
pub const WIRE_DAG_SCHEMA_VERSION: u32 = 2;

/// Backwards-compat default for [`WireDag::schema_version`]. A wire
/// payload predating WI-2 carries no `schema_version`; it is the
/// pre-versioning surface, which is version `1`, so a missing field
/// deserializes to the current baseline. This keeps deserialize additive
/// (same rationale as [`default_sum_accumulator_name`]). The default is
/// applied by serde at deserialize time; the *validation* of the value
/// is a separate explicit step ([`WireDag::validate_schema_version`]),
/// not a `Deserialize` side effect, so a mismatch surfaces as a typed
/// [`WireDagSchemaError`] rather than a raw serde error.
fn default_wire_dag_schema_version() -> u32 {
    WIRE_DAG_SCHEMA_VERSION
}

/// A typed failure from validating a serialized [`WireDag`] against the
/// supported schema version (WI-2). This is deliberately its own error
/// type rather than a reuse of [`crate::decode::DecodeError`] (opaque-ADT
/// invariant decoding) or [`crate::cache_envelope::CacheError`]: the
/// concern is IR-DAG wire-surface compatibility, a distinct domain.
///
/// Policy: an unknown or mismatched schema version is **rejected**, never
/// silently accepted and never a panic. A consumer pinned to
/// [`WIRE_DAG_SCHEMA_VERSION`] that is handed a payload stamped with a
/// version it does not recognize (in practice, a *newer* version it
/// cannot interpret) must fail closed — interpreting an unknown surface
/// would risk reading a renamed or re-shaped field as if it were the old
/// one. A strictly-lower version is forward-compatible only up to the
/// additive-default guarantee; this check rejects anything greater than
/// the version this build supports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireDagSchemaError {
    /// The payload's `schema_version` is greater than the version this
    /// build supports, so its wire shape cannot be safely interpreted.
    UnknownSchemaVersion {
        /// The version stamped on the payload.
        found: u32,
        /// The newest version this build understands
        /// ([`WIRE_DAG_SCHEMA_VERSION`]).
        supported: u32,
    },
}

impl std::fmt::Display for WireDagSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireDagSchemaError::UnknownSchemaVersion { found, supported } => write!(
                f,
                "WireDag schema version {found} is newer than the supported \
                 version {supported}; this build cannot interpret it. Rebuild \
                 against a chelis that emits version {found} or lower."
            ),
        }
    }
}

impl std::error::Error for WireDagSchemaError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireDag {
    /// Monotonic schema version of this serialized DAG surface (WI-2).
    /// Emitted as [`WIRE_DAG_SCHEMA_VERSION`] by the producer; defaults to
    /// the current baseline when absent (pre-versioning payloads), and is
    /// validated explicitly on consume via
    /// [`WireDag::validate_schema_version`].
    #[serde(default = "default_wire_dag_schema_version")]
    pub schema_version: u32,
    pub nodes: Vec<WireDagNode>,
    pub roots: Vec<usize>,
}

impl WireDag {
    /// Validate this DAG's [`schema_version`](Self::schema_version)
    /// against the version this build supports (WI-2).
    ///
    /// Returns `Ok(())` for any version less than or equal to
    /// [`WIRE_DAG_SCHEMA_VERSION`] (a lower version is accepted under the
    /// additive-default guarantee), and a typed
    /// [`WireDagSchemaError::UnknownSchemaVersion`] for any greater
    /// version. Callers consuming a serialized `WireDag` from an
    /// untrusted or cross-version producer MUST call this before relying
    /// on the DAG's shape; deserialize alone does not validate the
    /// version (the field has a serde default), so skipping this check
    /// would silently accept an unknown surface.
    pub fn validate_schema_version(&self) -> Result<(), WireDagSchemaError> {
        if self.schema_version > WIRE_DAG_SCHEMA_VERSION {
            return Err(WireDagSchemaError::UnknownSchemaVersion {
                found: self.schema_version,
                supported: WIRE_DAG_SCHEMA_VERSION,
            });
        }
        Ok(())
    }

    /// Deserialize a `WireDag` from JSON and validate its schema version
    /// in one step (WI-2). This is the recommended consume path for a
    /// payload from another build or process: it fails closed on an
    /// unknown version with a typed [`WireDagSchemaError`] rather than
    /// returning a `WireDag` whose shape this build cannot trust.
    ///
    /// A serde parse failure surfaces as [`serde_json::Error`]; a
    /// version mismatch on an otherwise-parseable payload surfaces as
    /// [`WireDagSchemaError`]. The two failure classes are distinct so a
    /// caller can tell a malformed payload from a version-incompatible
    /// one.
    pub fn from_validated_json(json: &str) -> Result<Self, WireDagDecodeError> {
        let dag: WireDag = serde_json::from_str(json).map_err(WireDagDecodeError::Parse)?;
        dag.validate_schema_version()
            .map_err(WireDagDecodeError::Schema)?;
        Ok(dag)
    }
}

/// Combined failure type for [`WireDag::from_validated_json`]: either the
/// JSON did not parse, or it parsed but carries an unsupported schema
/// version. Kept distinct so a caller can branch on malformed-vs-
/// incompatible.
#[derive(Debug)]
pub enum WireDagDecodeError {
    /// The payload is not valid `WireDag` JSON.
    Parse(serde_json::Error),
    /// The payload parsed but its schema version is unsupported.
    Schema(WireDagSchemaError),
}

impl std::fmt::Display for WireDagDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireDagDecodeError::Parse(e) => write!(f, "WireDag JSON parse error: {e}"),
            WireDagDecodeError::Schema(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WireDagDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WireDagDecodeError::Parse(e) => Some(e),
            WireDagDecodeError::Schema(e) => Some(e),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireDagNode {
    pub id: usize,
    pub op: WireRiscOp,
    pub inputs: Vec<usize>,
    pub output_type: WireTensorType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensorType {
    pub dims: Vec<WireDimInfo>,
    pub precision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireDimInfo {
    Named { name: String, size: Option<usize> },
    Lit { size: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireDimExpr {
    Concrete {
        value: usize,
    },
    Sym {
        name: String,
    },
    Mul {
        lhs: Box<WireDimExpr>,
        rhs: Box<WireDimExpr>,
    },
    Div {
        lhs: Box<WireDimExpr>,
        rhs: Box<WireDimExpr>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFusedStep {
    pub op: WireFusedStepOp,
    pub input_indices: Vec<WireFusedInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFusedStepOp {
    Add,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    MaxElem,
    CmpLt,
    Neg,
    Recip,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    Round,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireFusedInput {
    External { index: usize },
    PreviousStep { index: usize },
}

/// chelis#616: wire form of `chelis_ir::dag::RtDim` for movement-op bounds.
/// `Node(i)` indexes the owning op's `inputs` (the rank-0 integer bound scalars);
/// `to_end` is the full-axis sentinel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "bound", rename_all = "snake_case")]
pub enum WireRtDim {
    Lit { value: usize },
    ToEnd,
    Node { input: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireRiscOp {
    Add,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    CmpLt,
    MaxElem,
    Neg,
    Recip,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    Round,
    UniformLike {
        low: f64,
        high: f64,
        seed: u64,
    },
    Dropout {
        rate: f64,
        seed: u64,
    },
    Sum {
        axis: usize,
        /// Accumulator precision, populated per spec/04-type-system.md
        /// §5.7.1. Defaults are resolved before lowering, so this is
        /// always concrete in the wire schema.
        #[serde(default = "default_sum_accumulator_name")]
        accumulator: String,
    },
    MaxReduce {
        axis: usize,
    },
    MinReduce {
        axis: usize,
    },
    ProdReduce {
        axis: usize,
    },
    ReduceWindow {
        /// One of "max" / "min" / "sum" / "mean", matching the Surf
        /// builtin name suffix and `chelis_ir::dag::ReduceWindowKind`.
        reducer: String,
        window_shape: Vec<usize>,
        strides: Vec<usize>,
    },
    /// Reverse-mode adjoint of `ReduceWindow` (`RiscOp::ReduceWindowGrad`).
    /// Carries the same `reducer` / window / stride contract; appears only
    /// in `grad`-lowered DAGs.
    ReduceWindowGrad {
        /// One of "max" / "min" / "sum" / "mean", as for `ReduceWindow`.
        reducer: String,
        window_shape: Vec<usize>,
        strides: Vec<usize>,
    },
    Argmax {
        axis: usize,
    },
    Argmin {
        axis: usize,
    },
    Reshape {
        new_shape: Vec<WireDimInfo>,
    },
    Permute {
        axes: Vec<usize>,
    },
    Expand {
        axis: usize,
        size: String,
    },
    OneHot {
        vocab: usize,
    },
    Pad {
        padding: Vec<(WireRtDim, WireRtDim)>,
        fill: f64,
    },
    Shrink {
        bounds: Vec<(WireRtDim, WireRtDim)>,
    },
    Stride {
        strides: Vec<WireRtDim>,
    },
    Const {
        value: f64,
    },
    ConstTensor {
        data: Vec<f64>,
    },
    Shape {
        axis: usize,
    },
    Load {
        name: String,
    },
    Store {
        name: String,
    },
    Copy,
    Drop,
    Realize,
    Cast {
        new_precision: String,
    },
    FusedElem {
        ops: Vec<WireFusedStep>,
    },
    BlasMatmul {
        batch_dims: Vec<WireDimExpr>,
        m: WireDimExpr,
        n: WireDimExpr,
        k: WireDimExpr,
        /// Accumulator precision per spec/04-type-system.md §5.7.1.
        /// Result precision matches operand precision; the wider
        /// accumulator is consumed inside the op.
        #[serde(default = "default_matmul_accumulator_name")]
        accumulator: String,
    },
    Gather {
        axis: usize,
    },
    ScatterAdd {
        axis: usize,
    },
    Scatter {
        axis: usize,
    },
    ScatterElements {
        axis: usize,
    },
}

/// Backwards-compat default for the `accumulator` field on
/// [`WireRiscOp::Sum`]. Old wire payloads predate the WS-A0 spec lock
/// (cc47e6d) and don't carry the field; default to `f32`, the
/// pre-WS-A0 implicit accumulator.
fn default_sum_accumulator_name() -> String {
    "f32".to_string()
}

/// Backwards-compat default for the `accumulator` field on
/// [`WireRiscOp::BlasMatmul`]. Same rationale as
/// [`default_sum_accumulator_name`].
fn default_matmul_accumulator_name() -> String {
    "f32".to_string()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_wire_dag() -> WireDag {
        WireDag {
            schema_version: WIRE_DAG_SCHEMA_VERSION,
            nodes: vec![],
            roots: vec![],
        }
    }

    // WI-2 positive: a `WireDag` carries its schema version, the version
    // survives a serialize -> deserialize round-trip, and the recovered
    // DAG validates against the supported version.
    #[test]
    fn wire_dag_carries_schema_version_and_round_trips() {
        let dag = empty_wire_dag();
        assert_eq!(dag.schema_version, WIRE_DAG_SCHEMA_VERSION);

        let json = serde_json::to_string(&dag).expect("serialize WireDag");
        // The version is actually emitted on the wire, not merely a
        // default-on-read.
        assert!(
            json.contains(&format!("\"schema_version\":{WIRE_DAG_SCHEMA_VERSION}")),
            "serialized WireDag must carry schema_version, got {json}"
        );

        let back: WireDag = serde_json::from_str(&json).expect("deserialize WireDag");
        assert_eq!(back.schema_version, WIRE_DAG_SCHEMA_VERSION);
        back.validate_schema_version()
            .expect("current-version DAG validates");

        // The combined consume path accepts it too.
        let validated = WireDag::from_validated_json(&json).expect("validated decode");
        assert_eq!(validated.schema_version, WIRE_DAG_SCHEMA_VERSION);
    }

    // WI-2 additive-default: a pre-versioning payload (no schema_version
    // field) deserializes to the current baseline rather than failing,
    // keeping the wire surface additive.
    #[test]
    fn wire_dag_missing_schema_version_defaults_to_baseline() {
        let legacy = r#"{"nodes":[],"roots":[]}"#;
        let dag: WireDag = serde_json::from_str(legacy).expect("legacy payload deserializes");
        assert_eq!(
            dag.schema_version, WIRE_DAG_SCHEMA_VERSION,
            "missing schema_version defaults to the current baseline"
        );
        dag.validate_schema_version()
            .expect("defaulted version validates");
    }

    // WI-2 negative twin: a payload stamped with a version NEWER than this
    // build supports is REJECTED with the typed
    // `WireDagSchemaError::UnknownSchemaVersion` — not silently accepted,
    // not a panic, and not a bare serde error (the field parses fine; the
    // version value is what is rejected).
    #[test]
    fn wire_dag_rejects_unknown_schema_version() {
        let future = WIRE_DAG_SCHEMA_VERSION + 1;
        let json = format!(r#"{{"schema_version":{future},"nodes":[],"roots":[]}}"#);

        // It still PARSES (additive serde) ...
        let dag: WireDag = serde_json::from_str(&json).expect("future payload parses");
        assert_eq!(dag.schema_version, future);

        // ... but explicit validation REJECTS it with the typed error.
        let err = dag
            .validate_schema_version()
            .expect_err("future schema version must be rejected");
        assert_eq!(
            err,
            WireDagSchemaError::UnknownSchemaVersion {
                found: future,
                supported: WIRE_DAG_SCHEMA_VERSION,
            },
            "rejection must be the typed UnknownSchemaVersion error"
        );

        // The combined consume path surfaces it as the Schema arm, not a
        // parse error and not a silent accept.
        match WireDag::from_validated_json(&json) {
            Err(WireDagDecodeError::Schema(WireDagSchemaError::UnknownSchemaVersion {
                found,
                supported,
            })) => {
                assert_eq!(found, future);
                assert_eq!(supported, WIRE_DAG_SCHEMA_VERSION);
            }
            other => panic!(
                "from_validated_json must reject a future version with a typed \
                 Schema error, got {other:?}"
            ),
        }
    }

    #[test]
    fn sparse_wire_risc_ops_round_trip_as_additive_variants() {
        let gather = serde_json::to_string(&WireRiscOp::Gather { axis: 1 }).unwrap();
        assert_eq!(gather, r#"{"kind":"gather","axis":1}"#);
        match serde_json::from_str::<WireRiscOp>(&gather).unwrap() {
            WireRiscOp::Gather { axis } => assert_eq!(axis, 1),
            other => panic!("expected gather wire op, got {other:?}"),
        }

        let scatter = serde_json::to_string(&WireRiscOp::ScatterAdd { axis: 0 }).unwrap();
        assert_eq!(scatter, r#"{"kind":"scatter_add","axis":0}"#);
        match serde_json::from_str::<WireRiscOp>(&scatter).unwrap() {
            WireRiscOp::ScatterAdd { axis } => assert_eq!(axis, 0),
            other => panic!("expected scatter_add wire op, got {other:?}"),
        }

        let scatter_replace = serde_json::to_string(&WireRiscOp::Scatter { axis: 2 }).unwrap();
        assert_eq!(scatter_replace, r#"{"kind":"scatter","axis":2}"#);
        match serde_json::from_str::<WireRiscOp>(&scatter_replace).unwrap() {
            WireRiscOp::Scatter { axis } => assert_eq!(axis, 2),
            other => panic!("expected scatter wire op, got {other:?}"),
        }

        let one_hot = serde_json::to_string(&WireRiscOp::OneHot { vocab: 7 }).unwrap();
        assert_eq!(one_hot, r#"{"kind":"one_hot","vocab":7}"#);
        match serde_json::from_str::<WireRiscOp>(&one_hot).unwrap() {
            WireRiscOp::OneHot { vocab } => assert_eq!(vocab, 7),
            other => panic!("expected one_hot wire op, got {other:?}"),
        }
    }

    #[test]
    fn wire_property_contract_option_round_trips() {
        let option = WirePropertyOption::Contract {
            id: "std.normal_cdf.reflection".to_string(),
            span: Span { offset: 7, len: 31 },
        };
        let json = serde_json::to_string(&option).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"contract","id":"std.normal_cdf.reflection","span":{"offset":7,"len":31}}"#
        );
        match serde_json::from_str::<WirePropertyOption>(&json).unwrap() {
            WirePropertyOption::Contract { id, span } => {
                assert_eq!(id, "std.normal_cdf.reflection");
                assert_eq!(span, Span { offset: 7, len: 31 });
            }
            other => panic!("expected contract option, got {other:?}"),
        }
    }

    #[test]
    fn wire_inferred_type_serializes_internally_tagged_and_round_trips() {
        // `ref(tensor[lit 4, concrete f32])` — pins the tag shape Hull
        // reads back.
        let ty = WireInferredType::Ref {
            inner: Box::new(WireInferredType::Tensor {
                dims: vec![WireInferredDim::Lit { size: 4 }],
                precision: WireInferredPrecision::Concrete {
                    name: "f32".to_string(),
                },
            }),
        };
        let json = serde_json::to_string(&ty).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"ref","inner":{"kind":"tensor","dims":[{"kind":"lit","size":4}],"precision":{"kind":"concrete","name":"f32"}}}"#
        );
        let back: WireInferredType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ty);
    }

    #[test]
    fn wire_inferred_effect_serializes_internally_tagged_and_round_trips() {
        let io = serde_json::to_string(&WireInferredEffect::Io).unwrap();
        assert_eq!(io, r#"{"kind":"io"}"#);
        assert_eq!(
            serde_json::from_str::<WireInferredEffect>(&io).unwrap(),
            WireInferredEffect::Io
        );

        let resource = serde_json::to_string(&WireInferredEffect::Resource {
            device: "gpu:0".to_string(),
        })
        .unwrap();
        assert_eq!(resource, r#"{"kind":"resource","device":"gpu:0"}"#);
        match serde_json::from_str::<WireInferredEffect>(&resource).unwrap() {
            WireInferredEffect::Resource { device } => assert_eq!(device, "gpu:0"),
            other => panic!("expected resource effect, got {other:?}"),
        }
    }

    // Negative parity: an unknown effect `kind` must be REJECTED, not
    // silently coerced. A consumer that mints a tag we don't define
    // should get a hard deserialize error, never a default variant.
    #[test]
    fn wire_inferred_effect_rejects_unknown_kind() {
        let result = serde_json::from_str::<WireInferredEffect>(r#"{"kind":"telepathy"}"#);
        assert!(
            result.is_err(),
            "unknown effect kind must not deserialize, got {result:?}"
        );
    }

    /// Issue #254: the `reduce_window_*` family wires to a single
    /// `ReduceWindow` variant carrying a stringly-typed `reducer`
    /// discriminator plus the window/stride vectors. Pin the JSON
    /// shape and the serialize → deserialize round-trip so a downstream
    /// consumer of the machine-facing DAG sees a stable contract.
    #[test]
    fn reduce_window_wire_op_round_trips_with_reducer_string() {
        for (reducer, kind) in [
            ("max", "max"),
            ("min", "min"),
            ("sum", "sum"),
            ("mean", "mean"),
        ] {
            let op = WireRiscOp::ReduceWindow {
                reducer: reducer.to_string(),
                window_shape: vec![2, 3],
                strides: vec![1, 2],
            };
            let json = serde_json::to_string(&op).unwrap();
            assert_eq!(
                json,
                format!(
                    r#"{{"kind":"reduce_window","reducer":"{kind}","window_shape":[2,3],"strides":[1,2]}}"#
                ),
            );
            match serde_json::from_str::<WireRiscOp>(&json).unwrap() {
                WireRiscOp::ReduceWindow {
                    reducer,
                    window_shape,
                    strides,
                } => {
                    assert_eq!(reducer, kind);
                    assert_eq!(window_shape, vec![2, 3]);
                    assert_eq!(strides, vec![1, 2]);
                }
                other => panic!("expected reduce_window wire op, got {other:?}"),
            }
        }
    }

    /// The `reduce_window_*` adjoint wires to a sibling `ReduceWindowGrad`
    /// variant carrying the same `reducer` / window / stride contract.
    /// Pin its JSON shape and round-trip so grad-lowered machine-facing
    /// DAGs have a stable wire form.
    #[test]
    fn reduce_window_grad_wire_op_round_trips_with_reducer_string() {
        for kind in ["max", "min", "sum", "mean"] {
            let op = WireRiscOp::ReduceWindowGrad {
                reducer: kind.to_string(),
                window_shape: vec![2, 3],
                strides: vec![1, 2],
            };
            let json = serde_json::to_string(&op).unwrap();
            assert_eq!(
                json,
                format!(
                    r#"{{"kind":"reduce_window_grad","reducer":"{kind}","window_shape":[2,3],"strides":[1,2]}}"#
                ),
            );
            match serde_json::from_str::<WireRiscOp>(&json).unwrap() {
                WireRiscOp::ReduceWindowGrad {
                    reducer,
                    window_shape,
                    strides,
                } => {
                    assert_eq!(reducer, kind);
                    assert_eq!(window_shape, vec![2, 3]);
                    assert_eq!(strides, vec![1, 2]);
                }
                other => panic!("expected reduce_window_grad wire op, got {other:?}"),
            }
        }
    }
}
