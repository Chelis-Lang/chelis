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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSuccess<T> {
    pub ok: bool,
    pub result: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiFailure {
    pub ok: bool,
    pub stage: String,
    pub errors: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireDeepExpr {
    pub kind: WireDeepExprKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDeepAtom {
    Symbol { value: String },
    Int { value: i64 },
    Float { value: f64 },
    Str { value: String },
    Keyword { value: String },
    Bool { value: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireDag {
    pub nodes: Vec<WireDagNode>,
    pub roots: Vec<usize>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireFusedInput {
    External { index: usize },
    PreviousStep { index: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireRiscOp {
    Add,
    Mul,
    Div,
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
        padding: Vec<(usize, usize)>,
        fill: f64,
    },
    Shrink {
        bounds: Vec<(usize, usize)>,
    },
    Stride {
        strides: Vec<usize>,
    },
    Const {
        value: f64,
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
