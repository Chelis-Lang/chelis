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
    Int { value: i64 },
    Float { value: f64 },
    Str { value: String },
    Bool { value: bool },
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
pub struct WireFusedStep {
    pub op: WireFusedStepOp,
    pub input_indices: Vec<WireFusedInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFusedStepOp {
    Add,
    Mul,
    MaxElem,
    CmpLt,
    Neg,
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
    CmpLt,
    MaxElem,
    Neg,
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
    Realize,
    Cast {
        new_precision: String,
    },
    FusedElem {
        ops: Vec<WireFusedStep>,
    },
}

fn default_true() -> bool {
    true
}
