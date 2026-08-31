use std::collections::BTreeMap;

use chelis_types::unsupported::Unsupported;
use chelis_types::{
    ScalarValue,
    types::{Lane, Prim, Target},
};
use chelis_vocab::DiagnosticKind;
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

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ApiSuccess<T> {
    pub ok: bool,
    pub result: T,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ApiFailure {
    pub ok: bool,
    pub stage: String,
    pub errors: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum ApiEnvelope<T> {
    Success(ApiSuccess<T>),
    Failure(ApiFailure),
}

impl<T> ApiEnvelope<T> {
    pub fn success(result: T) -> Self {
        Self::Success(ApiSuccess { ok: true, result })
    }

    /// Assemble a failure from producer diagnostics only.
    ///
    /// A free kind string is not a producer diagnostic:
    ///
    /// ```compile_fail
    /// use chelis_compiler_api::schema::ApiEnvelope;
    /// let _: ApiEnvelope<()> = ApiEnvelope::failure("compile", vec!["compile_error"]);
    /// ```
    ///
    /// Nor may the public unsupported variant be handed directly to this
    /// general envelope API; it must arrive through a typed `Unsupported`:
    ///
    /// ```compile_fail
    /// use chelis_compiler_api::schema::ApiEnvelope;
    /// use chelis_vocab::DiagnosticKind;
    /// let _: ApiEnvelope<()> = ApiEnvelope::failure(
    ///     "compile",
    ///     vec![DiagnosticKind::UnsupportedFeature],
    /// );
    /// ```
    pub fn failure(stage: impl Into<String>, errors: Vec<Diagnostic>) -> Self {
        Self::Failure(ApiFailure {
            ok: false,
            stage: stage.into(),
            errors,
        })
    }

    /// Build the HTTP request-decoding failure without exposing diagnostic
    /// construction to the transport crate.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::failure(
            "http",
            vec![Diagnostic::general(
                GeneralKind::InvalidRequest,
                message,
                1.0,
            )],
        )
    }
}

/// Consumer-only success envelope decoded from the public JSON surface.
///
/// This is intentionally distinct from [`ApiSuccess`]: deserialized input
/// cannot be reused as a compiler-produced envelope.
#[derive(Debug, Clone, Deserialize)]
pub struct WireApiSuccess<T> {
    pub ok: bool,
    pub result: T,
}

/// Consumer-only failure envelope decoded from the public JSON surface.
#[derive(Debug, Clone, Deserialize)]
pub struct WireApiFailure {
    pub ok: bool,
    pub stage: String,
    pub errors: Vec<WireDiagnostic>,
}

/// Complete read-only consumer shape for a Tide/compiler API response.
///
/// There is deliberately no conversion from this type into [`ApiEnvelope`].
///
/// ```compile_fail
/// use chelis_compiler_api::schema::{ApiEnvelope, WireApiEnvelope};
/// let wire: WireApiEnvelope<()> = serde_json::from_str(
///     r#"{"ok":true,"result":null}"#,
/// ).unwrap();
/// let _: ApiEnvelope<()> = wire.into();
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum WireApiEnvelope<T> {
    Success(WireApiSuccess<T>),
    Failure(WireApiFailure),
}

/// A compiler-produced diagnostic.
///
/// `kind` is private and this type intentionally implements `Serialize` but
/// not `Deserialize`. Every producer therefore crosses the typed constructor
/// boundary below. Reading wire JSON uses [`WireDiagnostic`] instead.
///
/// A foreign crate cannot forge a diagnostic with a struct literal:
///
/// ```compile_fail
/// use chelis_compiler_api::schema::Diagnostic;
/// let _ = Diagnostic {
///     kind: "unsupported_feature".to_owned(),
///     message: "forged".to_owned(),
///     severity: 1.0,
///     expected: None,
///     got: None,
///     suggestions: vec![],
///     span: None,
///     deep_path: None,
/// };
/// ```
///
/// Nor can wire input deserialize into the producer type:
///
/// ```compile_fail
/// use chelis_compiler_api::schema::Diagnostic;
/// let _: Diagnostic = serde_json::from_str(r#"{"kind":"unsupported_feature"}"#).unwrap();
/// ```
///
/// A deserialized consumer value cannot enter an envelope producer API:
///
/// ```compile_fail
/// use chelis_compiler_api::schema::{ApiEnvelope, Diagnostic, WireDiagnostic};
/// let wire: WireDiagnostic = serde_json::from_str(
///     r#"{"kind":"unsupported_feature","message":"forged","severity":1.0}"#,
/// ).unwrap();
/// let producer: Diagnostic = wire.into();
/// let _: ApiEnvelope<()> = ApiEnvelope::failure("compile", vec![producer]);
/// ```
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct Diagnostic {
    kind: String,
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

impl Diagnostic {
    pub fn kind(&self) -> DiagnosticKind {
        DiagnosticKind::decode(&self.kind)
            .expect("producer diagnostics are constructed from DiagnosticKind")
    }

    pub(crate) fn general(kind: GeneralKind, message: impl Into<String>, severity: f64) -> Self {
        let diagnostic_kind = kind.diagnostic_kind();
        debug_assert_eq!(GeneralKind::project(diagnostic_kind), Some(kind));
        Self::new(diagnostic_kind, message, severity)
    }

    fn unsupported(error: Unsupported) -> Self {
        Self::new(DiagnosticKind::UnsupportedFeature, error.to_string(), 1.0)
    }

    fn new(kind: DiagnosticKind, message: impl Into<String>, severity: f64) -> Self {
        Self {
            kind: kind.as_str().to_owned(),
            message: message.into(),
            severity,
            expected: None,
            got: None,
            suggestions: Vec::new(),
            span: None,
            deep_path: None,
        }
    }
}

/// Consumer-only representation of a diagnostic read from JSON.
///
/// This type deliberately has no conversion into [`Diagnostic`]. Callers may
/// inspect and validate the string, but cannot feed it back into a producer
/// envelope.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct WireDiagnostic {
    pub kind: String,
    pub message: String,
    pub severity: f64,
    pub expected: Option<String>,
    pub got: Option<String>,
    #[serde(default)]
    pub suggestions: Vec<String>,
    pub span: Option<Span>,
    pub deep_path: Option<WireDeepErrorPath>,
}

/// The producer projection of [`DiagnosticKind`]. It intentionally has no
/// `UnsupportedFeature` variant: a general producer cannot spell an
/// unsupported rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeneralKind {
    SurfParseError,
    DeepParseError,
    MacroError,
    NameResolutionError,
    DeepDeclError,
    DuplicateName,
    PreimageMismatch,
    CascadeIncomplete,
    TypeError,
    EffectError,
    LinearityError,
    LowerError,
    ReefError,
    EvalError,
    Cancelled,
    GradError,
    ValidationError,
    UnknownName,
    UnknownSchemaVersion,
    HashError,
    InvalidRequest,
    CompileError,
    Other,
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    UnboundVariable,
    UnknownConstructor,
    NotAFunction,
    NonExhaustiveMatch,
    OccursCheck,
    CastNonTensor,
    TupleIndexOutOfBounds,
    UseAfterConsume,
    UnconsumedLinear,
    InvalidBorrow,
    CycleDetected,
    UnsupportedTensorPrecision,
    DuplicateDefinition,
    DuplicateModule,
    OpaqueTypeViolation,
    ReservedLinkerName,
    BuiltinShadowing,
    UnknownForm,
    MalformedForm,
    CheckOther,
}

impl GeneralKind {
    const fn project(kind: DiagnosticKind) -> Option<Self> {
        match kind {
            DiagnosticKind::UnsupportedFeature => None,
            DiagnosticKind::SurfParseError => Some(Self::SurfParseError),
            DiagnosticKind::DeepParseError => Some(Self::DeepParseError),
            DiagnosticKind::MacroError => Some(Self::MacroError),
            DiagnosticKind::NameResolutionError => Some(Self::NameResolutionError),
            DiagnosticKind::DeepDeclError => Some(Self::DeepDeclError),
            DiagnosticKind::DuplicateName => Some(Self::DuplicateName),
            DiagnosticKind::PreimageMismatch => Some(Self::PreimageMismatch),
            DiagnosticKind::CascadeIncomplete => Some(Self::CascadeIncomplete),
            DiagnosticKind::TypeError => Some(Self::TypeError),
            DiagnosticKind::EffectError => Some(Self::EffectError),
            DiagnosticKind::LinearityError => Some(Self::LinearityError),
            DiagnosticKind::LowerError => Some(Self::LowerError),
            DiagnosticKind::ReefError => Some(Self::ReefError),
            DiagnosticKind::EvalError => Some(Self::EvalError),
            DiagnosticKind::Cancelled => Some(Self::Cancelled),
            DiagnosticKind::GradError => Some(Self::GradError),
            DiagnosticKind::ValidationError => Some(Self::ValidationError),
            DiagnosticKind::UnknownName => Some(Self::UnknownName),
            DiagnosticKind::UnknownSchemaVersion => Some(Self::UnknownSchemaVersion),
            DiagnosticKind::HashError => Some(Self::HashError),
            DiagnosticKind::InvalidRequest => Some(Self::InvalidRequest),
            DiagnosticKind::CompileError => Some(Self::CompileError),
            DiagnosticKind::GeneralOther => Some(Self::Other),
            DiagnosticKind::TypeMismatch => Some(Self::TypeMismatch),
            DiagnosticKind::PrecisionMismatch => Some(Self::PrecisionMismatch),
            DiagnosticKind::DimensionMismatch => Some(Self::DimensionMismatch),
            DiagnosticKind::ArityMismatch => Some(Self::ArityMismatch),
            DiagnosticKind::UnboundVariable => Some(Self::UnboundVariable),
            DiagnosticKind::UnknownConstructor => Some(Self::UnknownConstructor),
            DiagnosticKind::NotAFunction => Some(Self::NotAFunction),
            DiagnosticKind::NonExhaustiveMatch => Some(Self::NonExhaustiveMatch),
            DiagnosticKind::OccursCheck => Some(Self::OccursCheck),
            DiagnosticKind::CastNonTensor => Some(Self::CastNonTensor),
            DiagnosticKind::TupleIndexOutOfBounds => Some(Self::TupleIndexOutOfBounds),
            DiagnosticKind::UseAfterConsume => Some(Self::UseAfterConsume),
            DiagnosticKind::UnconsumedLinear => Some(Self::UnconsumedLinear),
            DiagnosticKind::InvalidBorrow => Some(Self::InvalidBorrow),
            DiagnosticKind::CycleDetected => Some(Self::CycleDetected),
            DiagnosticKind::UnsupportedTensorPrecision => Some(Self::UnsupportedTensorPrecision),
            DiagnosticKind::DuplicateDefinition => Some(Self::DuplicateDefinition),
            DiagnosticKind::DuplicateModule => Some(Self::DuplicateModule),
            DiagnosticKind::OpaqueTypeViolation => Some(Self::OpaqueTypeViolation),
            DiagnosticKind::ReservedLinkerName => Some(Self::ReservedLinkerName),
            DiagnosticKind::BuiltinShadowing => Some(Self::BuiltinShadowing),
            DiagnosticKind::UnknownForm => Some(Self::UnknownForm),
            DiagnosticKind::MalformedForm => Some(Self::MalformedForm),
            DiagnosticKind::CheckOther => Some(Self::CheckOther),
        }
    }

    const fn diagnostic_kind(self) -> DiagnosticKind {
        match self {
            Self::SurfParseError => DiagnosticKind::SurfParseError,
            Self::DeepParseError => DiagnosticKind::DeepParseError,
            Self::MacroError => DiagnosticKind::MacroError,
            Self::NameResolutionError => DiagnosticKind::NameResolutionError,
            Self::DeepDeclError => DiagnosticKind::DeepDeclError,
            Self::DuplicateName => DiagnosticKind::DuplicateName,
            Self::PreimageMismatch => DiagnosticKind::PreimageMismatch,
            Self::CascadeIncomplete => DiagnosticKind::CascadeIncomplete,
            Self::TypeError => DiagnosticKind::TypeError,
            Self::EffectError => DiagnosticKind::EffectError,
            Self::LinearityError => DiagnosticKind::LinearityError,
            Self::LowerError => DiagnosticKind::LowerError,
            Self::ReefError => DiagnosticKind::ReefError,
            Self::EvalError => DiagnosticKind::EvalError,
            Self::Cancelled => DiagnosticKind::Cancelled,
            Self::GradError => DiagnosticKind::GradError,
            Self::ValidationError => DiagnosticKind::ValidationError,
            Self::UnknownName => DiagnosticKind::UnknownName,
            Self::UnknownSchemaVersion => DiagnosticKind::UnknownSchemaVersion,
            Self::HashError => DiagnosticKind::HashError,
            Self::InvalidRequest => DiagnosticKind::InvalidRequest,
            Self::CompileError => DiagnosticKind::CompileError,
            Self::Other => DiagnosticKind::GeneralOther,
            Self::TypeMismatch => DiagnosticKind::TypeMismatch,
            Self::PrecisionMismatch => DiagnosticKind::PrecisionMismatch,
            Self::DimensionMismatch => DiagnosticKind::DimensionMismatch,
            Self::ArityMismatch => DiagnosticKind::ArityMismatch,
            Self::UnboundVariable => DiagnosticKind::UnboundVariable,
            Self::UnknownConstructor => DiagnosticKind::UnknownConstructor,
            Self::NotAFunction => DiagnosticKind::NotAFunction,
            Self::NonExhaustiveMatch => DiagnosticKind::NonExhaustiveMatch,
            Self::OccursCheck => DiagnosticKind::OccursCheck,
            Self::CastNonTensor => DiagnosticKind::CastNonTensor,
            Self::TupleIndexOutOfBounds => DiagnosticKind::TupleIndexOutOfBounds,
            Self::UseAfterConsume => DiagnosticKind::UseAfterConsume,
            Self::UnconsumedLinear => DiagnosticKind::UnconsumedLinear,
            Self::InvalidBorrow => DiagnosticKind::InvalidBorrow,
            Self::CycleDetected => DiagnosticKind::CycleDetected,
            Self::UnsupportedTensorPrecision => DiagnosticKind::UnsupportedTensorPrecision,
            Self::DuplicateDefinition => DiagnosticKind::DuplicateDefinition,
            Self::DuplicateModule => DiagnosticKind::DuplicateModule,
            Self::OpaqueTypeViolation => DiagnosticKind::OpaqueTypeViolation,
            Self::ReservedLinkerName => DiagnosticKind::ReservedLinkerName,
            Self::BuiltinShadowing => DiagnosticKind::BuiltinShadowing,
            Self::UnknownForm => DiagnosticKind::UnknownForm,
            Self::MalformedForm => DiagnosticKind::MalformedForm,
            Self::CheckOther => DiagnosticKind::CheckOther,
        }
    }
}

pub(crate) fn stage_error(
    stage: &str,
    message: impl Into<String>,
    kind: GeneralKind,
) -> crate::compiler::CompilerError {
    stage_error_with_span(stage, message, kind, None)
}

pub(crate) fn stage_error_with_span(
    stage: &str,
    message: impl Into<String>,
    kind: GeneralKind,
    span: Option<Span>,
) -> crate::compiler::CompilerError {
    let mut diagnostic = Diagnostic::general(kind, message, 1.0);
    diagnostic.span = span;
    crate::compiler::CompilerError {
        stage: stage.to_owned(),
        errors: vec![diagnostic],
    }
}

pub(crate) fn unsupported_stage_error(error: Unsupported) -> crate::compiler::CompilerError {
    crate::compiler::CompilerError {
        stage: "compile".to_owned(),
        errors: vec![Diagnostic::unsupported(error)],
    }
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

/// Execution-payload wire version (chelis#729 Phase 1, the section C3
/// storage decision's wire layer). Version history:
///
/// * v1 (implicit; no version field on the wire): `TensorValue.data` was
///   an untagged `Vec<f64>`, which cannot carry exact int64 above 2^53
///   (chelis#686) and erased every element dtype (chelis#685).
/// * v2: `TensorValue.data` is the tagged per-dtype [`TensorElements`]
///   payload below; numeric scalar leaves use exact-width
///   [`ExecutionValue`] variants instead of substituting `Int64`/`Float64`;
///   and [`EvalResult`] stamps `schema_version: 2`.
///
/// Mechanics (v1 compat DELETED at the chelis#729 rework): producers
/// always stamp the current version, and `schema_version` is REQUIRED on
/// decode and must equal this constant - a missing field is a loud serde
/// "missing field `schema_version`" error, and a `1` (or any other
/// value) is a loud error naming the field and both versions. Every
/// reader and writer of this payload is in-repo, so there is no
/// deployment that can legitimately present a version-less or v1
/// payload; per the chelis#730 closed-vocabulary doctrine (closed types
/// have no `Default` and no `Unknown`), the compat default was a spare
/// key to a door that should have exactly one. Tensor BINDINGS in
/// requests changed shape with v2, so a v1 client posting the old
/// bare-array `data` also fails loudly at serde (a type error at the
/// payload position), never a silent reinterpretation. This constant
/// governs the execution payload only; `WIRE_DAG_SCHEMA_VERSION` below
/// governs the `WireDag` surface and is independent (and, unlike this
/// one, has a genuinely external consumer - see its note).
pub const EXECUTION_VALUE_SCHEMA_VERSION: u32 = 2;

/// Field validator for [`EvalResult::schema_version`]: the field is
/// required and must equal [`EXECUTION_VALUE_SCHEMA_VERSION`]. The
/// error names the field so a stale producer is diagnosable from the
/// message alone.
fn require_execution_value_schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    use serde::de::Error;
    let version = u32::deserialize(deserializer)?;
    if version != EXECUTION_VALUE_SCHEMA_VERSION {
        return Err(D::Error::custom(format!(
            "unsupported `schema_version` {version} on the execution-value              payload: this reader supports exactly              {EXECUTION_VALUE_SCHEMA_VERSION} (the v1 compat path was              deleted at the chelis#729 rework; regenerate the payload              with a current producer)"
        )));
    }
    Ok(version)
}

/// Per-dtype tensor element payload (execution wire v2; the chelis#729
/// section C3 storage decision expressed at the wire layer). Integer
/// families carry exact integers at width; `f16`/`bf16` carry the EXACT
/// f64 images of the stored half-precision values (every half value is
/// exactly representable in f64, and JSON numbers carry f64 exactly);
/// bool carries true/false.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "dtype", content = "values", rename_all = "snake_case")]
pub enum TensorElements {
    F64(Vec<f64>),
    F32(Vec<f32>),
    F16(Vec<f64>),
    Bf16(Vec<f64>),
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Int16(Vec<i16>),
    Int8(Vec<i8>),
    Bool(Vec<bool>),
}

impl TensorElements {
    pub fn len(&self) -> usize {
        match self {
            TensorElements::F64(v) => v.len(),
            TensorElements::F32(v) => v.len(),
            TensorElements::F16(v) => v.len(),
            TensorElements::Bf16(v) => v.len(),
            TensorElements::Int64(v) => v.len(),
            TensorElements::Int32(v) => v.len(),
            TensorElements::Int16(v) => v.len(),
            TensorElements::Int8(v) => v.len(),
            TensorElements::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Widen every element to f64. Exact except for int64 magnitudes
    /// above 2^53, hence the lossy name (the section C3 read-side
    /// contract; consumers that need exact int64 match the variant).
    pub fn to_f64_lossy_vec(&self) -> Vec<f64> {
        match self {
            TensorElements::F64(v) => v.clone(),
            TensorElements::F32(v) => v.iter().map(|&x| x as f64).collect(),
            TensorElements::F16(v) => v.clone(),
            TensorElements::Bf16(v) => v.clone(),
            TensorElements::Int64(v) => v.iter().map(|&x| x as f64).collect(),
            TensorElements::Int32(v) => v.iter().map(|&x| x as f64).collect(),
            TensorElements::Int16(v) => v.iter().map(|&x| x as f64).collect(),
            TensorElements::Int8(v) => v.iter().map(|&x| x as f64).collect(),
            TensorElements::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        }
    }

    /// One element widened to f64 (same loss profile as
    /// [`Self::to_f64_lossy_vec`]).
    pub fn element_as_f64_lossy(&self, index: usize) -> f64 {
        match self {
            TensorElements::F64(v) => v[index],
            TensorElements::F32(v) => v[index] as f64,
            TensorElements::F16(v) => v[index],
            TensorElements::Bf16(v) => v[index],
            TensorElements::Int64(v) => v[index] as f64,
            TensorElements::Int32(v) => v[index] as f64,
            TensorElements::Int16(v) => v[index] as f64,
            TensorElements::Int8(v) => v[index] as f64,
            TensorElements::Bool(v) => {
                if v[index] {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Convenience f64 constructor for request builders and tests that
    /// carry plain float payloads.
    pub fn from_f64_vec(data: Vec<f64>) -> Self {
        TensorElements::F64(data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TensorValue {
    pub shape: Vec<usize>,
    pub data: TensorElements,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DictEntryValue {
    pub key: ExecutionValue,
    pub value: ExecutionValue,
}

/// Machine-facing execution value. Every numeric scalar variant names its
/// own dtype; the field width is either that dtype's exact Rust carrier or,
/// for f16/bf16, the exact f64 image of the stored reduced-width value.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionValue {
    Tensor {
        value: TensorValue,
    },
    Int8 {
        value: i8,
    },
    Int16 {
        value: i16,
    },
    Int32 {
        value: i32,
    },
    Int64 {
        value: i64,
    },
    Float16 {
        /// Exact f64 image of the stored IEEE binary16 value.
        value: f64,
    },
    Bfloat16 {
        /// Exact f64 image of the stored bfloat16 value.
        value: f64,
    },
    Float32 {
        value: f32,
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

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub score: f64,
    pub components: FitnessComponents,
    pub typed_nodes: usize,
    pub untyped_nodes: usize,
    pub total_nodes: usize,
    pub unresolved_names: Vec<String>,
    pub errors: Vec<Diagnostic>,
}

/// Consumer-side shape for `chelis check --json` output.
#[derive(Debug, Clone, Deserialize)]
pub struct WireCheckResult {
    pub score: f64,
    pub components: FitnessComponents,
    pub typed_nodes: usize,
    pub untyped_nodes: usize,
    pub total_nodes: usize,
    pub unresolved_names: Vec<String>,
    pub errors: Vec<WireDiagnostic>,
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
        args: Vec<WireInferredAdtArg>,
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

/// One nominal argument in structured inferred JSON. The enum is untagged so
/// ordinary type arguments retain their established `WireInferredType` object
/// shape; only dimension arguments add a new wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum WireInferredAdtArg {
    Type(WireInferredType),
    Dimension(WireInferredDimensionArg),
}

/// Closed wrapper that distinguishes a nominal dimension from an ordinary
/// inferred type without changing the existing type-argument encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireInferredDimensionArg {
    Dimension { dim: WireInferredDim },
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

/// Stable machine-facing projection of one checked root-manifest entry.
/// Internal type expressions and routing evidence stay on `RootEntry`; the
/// public wire carries only the facts consumers need to route observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootManifestEntryResult {
    pub name: String,
    pub lane: Lane,
    pub required_inputs: Vec<String>,
}

/// Target-carrying root contract returned by production eval/build APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootManifestResult {
    pub target: Target,
    pub entries: Vec<RootManifestEntryResult>,
    pub requires_main: bool,
}

impl Default for RootManifestResult {
    fn default() -> Self {
        Self {
            target: Target::Eval,
            entries: Vec::new(),
            requires_main: false,
        }
    }
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
    pub manifest: RootManifestResult,
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
    /// Human-rendered display text for this root, produced by the
    /// runtime's single [05-OBS-1] renderer while the dtype tags still
    /// exist (`ExecutionValue` cannot carry them: its tensor payload is an
    /// untagged `Vec<f64>`). In-process transport only - `#[serde(skip)]`
    /// keeps the machine-facing `--json` wire byte-identical, and a
    /// deserialized `EvalResult` carries `None` here. The CLI's
    /// labeled-root exit consumes this so it shares the transcript exit's
    /// renderer byte-for-byte (chelis#732 Phase 1).
    #[serde(skip)]
    pub display: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Execution-payload wire version (see
    /// [`EXECUTION_VALUE_SCHEMA_VERSION`]): producers stamp the current
    /// version; decode REQUIRES the field and rejects any other version
    /// loudly (the v1 compat default is gone, chelis#729 rework).
    #[serde(deserialize_with = "require_execution_value_schema_version")]
    pub schema_version: u32,
    pub roots: Vec<EvaluatedRoot>,
    #[serde(default)]
    pub manifest: RootManifestResult,
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct BatchResultEnvelope {
    pub results: Vec<BatchResult>,
}

/// Consumer-only counterpart to [`BatchResult`]. The `Check` arm uses the
/// read-only diagnostic graph all the way through the enclosing envelope.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireBatchResult {
    Parse(WireApiEnvelope<ParseResult>),
    Desugar(WireApiEnvelope<DesugarResult>),
    Check(WireApiEnvelope<WireCheckResult>),
    Lower(WireApiEnvelope<LowerResult>),
    Compile(WireApiEnvelope<CompileResult>),
    Eval(WireApiEnvelope<EvalResult>),
    Grad(WireApiEnvelope<GradResult>),
    Validate(WireApiEnvelope<ValidateResult>),
    Decompile(WireApiEnvelope<DecompileResult>),
}

/// Complete read-only consumer shape for the batch endpoint response.
#[derive(Debug, Clone, Deserialize)]
pub struct WireBatchResultEnvelope {
    pub results: Vec<WireBatchResult>,
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
    RecordUpdate {
        base: Box<WireSurfExpr>,
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
        /// The chelis#759 ladder rung ([05-OP-6]). Absent means the
        /// checked default, so a pre-`cast_trunc` payload still decodes.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mode: Option<String>,
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
    Do {
        exprs: Vec<WireSurfExpr>,
        span: Span,
    },
    Quote {
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    Unquote {
        expr: Box<WireSurfExpr>,
        span: Span,
    },
    Splice {
        expr: Box<WireSurfExpr>,
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
    DimensionLiteral {
        digits: String,
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
/// - `3`: chelis#616 — `WireRiscOp::Reshape::new_shape` changed from
///   `Vec<WireDimInfo>` to `Vec<WireRtDim>` (runtime reshape target
///   extents), and `WireRtDim` gained the `Sym` variant. A reshape target
///   now serializes as a bound-tagged value (`lit` / `node` / `sym`), not
///   a dim-info object, so a pinned consumer must observe the bump.
/// - `4`: chelis#729 rework (chelis#856, the fifth storage layer) —
///   `WireRiscOp::Const::value` changed from a bare f64 to the sealed
///   dtype-tagged scalar payload and `WireRiscOp::ConstTensor::data`
///   from `Vec<f64>` to the sealed per-dtype storage payload. Integer
///   constants now travel exact at width (no f64 collapse above 2^53)
///   and decoding finalizes through the dtype_semantics module
///   (finalize-on-decode; corrupt reduced-float images are a loud
///   decode error).
/// - `5`: chelis#878 — `WireRiscOp::Pad::fill` changed from a bare f64
///   capacity seam to the sealed dtype-tagged scalar payload.
/// - `6`: chelis#1287 — added the dedicated multi-axis
///   `WireRiscOp::Count` form and made the complete WireDag encoding exact:
///   the version stamp and all fields are explicit, with no legacy migration
///   or default-on-read spellings. Chelis#1306 added direct `Sub`, `MinElem`,
///   and `ExtremaAdjoint` identities plus matching fused-step identities to
///   that exact encoding.
pub const WIRE_DAG_SCHEMA_VERSION: u32 = 6;

/// A typed failure from validating a serialized [`WireDag`] against the
/// supported schema version (WI-2). This is deliberately its own error
/// type rather than a reuse of [`crate::decode::DecodeError`] (opaque-ADT
/// invariant decoding) or [`crate::cache_envelope::CacheError`]: the
/// concern is IR-DAG wire-surface compatibility, a distinct domain.
///
/// Policy: the version stamp is mandatory and must equal
/// [`WIRE_DAG_SCHEMA_VERSION`] exactly. Missing, older, and future versions
/// are all rejected before a `WireRiscOp` is decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireDagSchemaError {
    /// The payload omitted the mandatory version stamp.
    MissingSchemaVersion {
        /// The one version this build accepts.
        supported: u32,
    },
    /// The payload carries an older or newer version than this build accepts.
    UnsupportedSchemaVersion {
        /// The version stamped on the payload.
        found: u32,
        /// The one version this build accepts.
        supported: u32,
    },
}

impl std::fmt::Display for WireDagSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireDagSchemaError::MissingSchemaVersion { supported } => write!(
                f,
                "WireDag schema version is missing; this build requires explicit version {supported}"
            ),
            WireDagSchemaError::UnsupportedSchemaVersion { found, supported } => write!(
                f,
                "WireDag schema version {found} is unsupported; this build requires exactly version {supported}"
            ),
        }
    }
}

impl std::error::Error for WireDagSchemaError {}

/// A structurally invalid exact-version WireDag encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireDagContractError {
    message: String,
}

impl WireDagContractError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for WireDagContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for WireDagContractError {}

#[derive(Debug, Clone)]
pub struct WireDag {
    /// Exact schema version of this serialized DAG surface.
    pub schema_version: u32,
    pub nodes: Vec<WireDagNode>,
    pub roots: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct WireDagFields {
    schema_version: u32,
    nodes: Vec<WireDagNode>,
    roots: Vec<usize>,
}

#[derive(Serialize)]
struct WireDagFieldsRef<'a> {
    schema_version: u32,
    nodes: &'a [WireDagNode],
    roots: &'a [usize],
}

impl Serialize for WireDag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.validate_schema_version()
            .map_err(<S::Error as serde::ser::Error>::custom)?;
        self.validate_wire_contract()
            .map_err(<S::Error as serde::ser::Error>::custom)?;
        WireDagFieldsRef {
            schema_version: self.schema_version,
            nodes: &self.nodes,
            roots: &self.roots,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WireDag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if !value.is_object() {
            return Err(<D::Error as serde::de::Error>::custom(
                "WireDag root must be a JSON object",
            ));
        }
        let found = explicit_wire_dag_schema_version(&value)
            .map_err(<D::Error as serde::de::Error>::custom)?;
        validate_explicit_wire_dag_schema_version(found)
            .map_err(<D::Error as serde::de::Error>::custom)?;

        // The version gate above intentionally runs while nodes are still
        // untyped JSON. Only an exact v6 payload may construct WireRiscOp.
        let fields: WireDagFields =
            serde_json::from_value(value).map_err(<D::Error as serde::de::Error>::custom)?;
        let dag = Self {
            schema_version: fields.schema_version,
            nodes: fields.nodes,
            roots: fields.roots,
        };
        dag.validate_wire_contract()
            .map_err(<D::Error as serde::de::Error>::custom)?;
        Ok(dag)
    }
}

fn explicit_wire_dag_schema_version(
    value: &serde_json::Value,
) -> Result<Option<u32>, serde_json::Error> {
    value
        .get("schema_version")
        .cloned()
        .map(serde_json::from_value::<u32>)
        .transpose()
}

fn validate_explicit_wire_dag_schema_version(found: Option<u32>) -> Result<(), WireDagSchemaError> {
    let Some(found) = found else {
        return Err(WireDagSchemaError::MissingSchemaVersion {
            supported: WIRE_DAG_SCHEMA_VERSION,
        });
    };
    if found != WIRE_DAG_SCHEMA_VERSION {
        return Err(WireDagSchemaError::UnsupportedSchemaVersion {
            found,
            supported: WIRE_DAG_SCHEMA_VERSION,
        });
    }
    Ok(())
}

impl WireDag {
    /// Validate this DAG's [`schema_version`](Self::schema_version)
    /// against the version this build supports (WI-2).
    ///
    /// Returns `Ok(())` only for [`WIRE_DAG_SCHEMA_VERSION`]. Older and
    /// future versions both fail closed.
    pub fn validate_schema_version(&self) -> Result<(), WireDagSchemaError> {
        validate_explicit_wire_dag_schema_version(Some(self.schema_version))
    }

    /// Validate fields whose exact encoding depends on surrounding DAG shape.
    pub fn validate_wire_contract(&self) -> Result<(), WireDagContractError> {
        for (index, node) in self.nodes.iter().enumerate() {
            if let WireRiscOp::Pad { fill, .. } = &node.op {
                let output_prim =
                    Prim::parse_name(&node.output_type.precision).ok_or_else(|| {
                        WireDagContractError::new(format!(
                            "WireDag Pad node {} has unknown output dtype {}",
                            node.id, node.output_type.precision
                        ))
                    })?;
                if fill.prim() != output_prim {
                    return Err(WireDagContractError::new(format!(
                        "WireDag Pad fill dtype {} does not match node {} output dtype {}",
                        fill.prim().name(),
                        node.id,
                        output_prim.name()
                    )));
                }
            }

            let WireRiscOp::Count { axes } = &node.op else {
                continue;
            };
            if node.inputs.len() != 1 {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} requires exactly one input, found {}",
                    node.id,
                    node.inputs.len()
                )));
            }
            let input_id = node.inputs[0];
            let input = self.nodes[..index]
                .iter()
                .find(|candidate| candidate.id == input_id)
                .ok_or_else(|| {
                    WireDagContractError::new(format!(
                        "WireDag Count node {} input {input_id} does not resolve to an earlier node",
                        node.id
                    ))
                })?;
            if axes.is_empty() {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} axes must be non-empty",
                    node.id
                )));
            }
            if axes.windows(2).any(|pair| pair[0] <= pair[1]) {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} axes must be unique and strictly descending, found {axes:?}",
                    node.id
                )));
            }
            let input_rank = input.output_type.dims.len();
            if let Some(axis) = axes.iter().copied().find(|axis| *axis >= input_rank) {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} axis {axis} is out of range for input rank {input_rank}",
                    node.id
                )));
            }
            if input.output_type.precision != Prim::Bool.name() {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} input dtype must be bool, found {}",
                    node.id, input.output_type.precision
                )));
            }
            if node.output_type.precision != Prim::Int64.name() {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} output dtype must be int64, found {}",
                    node.id, node.output_type.precision
                )));
            }
            let expected_output_dims = input
                .output_type
                .dims
                .iter()
                .enumerate()
                .filter(|(axis, _)| !axes.contains(axis))
                .map(|(_, dim)| dim)
                .collect::<Vec<_>>();
            if expected_output_dims.len() != node.output_type.dims.len()
                || expected_output_dims
                    .iter()
                    .zip(&node.output_type.dims)
                    .any(|(expected, actual)| !wire_dim_info_equal(expected, actual))
            {
                return Err(WireDagContractError::new(format!(
                    "WireDag Count node {} output dimensions must equal input dimensions with axes removed",
                    node.id
                )));
            }
        }
        Ok(())
    }

    /// Deserialize a `WireDag` from JSON and validate its schema version
    /// in one step (WI-2). This is the recommended consume path for a
    /// payload from another build or process: it fails closed on an
    /// missing or mismatched version with a typed [`WireDagSchemaError`] rather than
    /// returning a `WireDag` whose shape this build cannot trust.
    ///
    /// A serde parse failure surfaces as [`serde_json::Error`]; a
    /// version mismatch on an otherwise-parseable payload surfaces as
    /// [`WireDagSchemaError`], and an invalid exact-version cross-node shape
    /// surfaces as [`WireDagContractError`].
    pub fn from_validated_json(json: &str) -> Result<Self, WireDagDecodeError> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(WireDagDecodeError::Parse)?;
        if !value.is_object() {
            return serde_json::from_value::<WireDagFields>(value)
                .map(|fields| Self {
                    schema_version: fields.schema_version,
                    nodes: fields.nodes,
                    roots: fields.roots,
                })
                .map_err(WireDagDecodeError::Parse);
        }
        let found = explicit_wire_dag_schema_version(&value).map_err(WireDagDecodeError::Parse)?;
        validate_explicit_wire_dag_schema_version(found).map_err(WireDagDecodeError::Schema)?;

        // Keep node JSON untyped until the exact schema stamp above succeeds.
        let fields: WireDagFields =
            serde_json::from_value(value).map_err(WireDagDecodeError::Parse)?;
        let dag = Self {
            schema_version: fields.schema_version,
            nodes: fields.nodes,
            roots: fields.roots,
        };
        dag.validate_wire_contract()
            .map_err(WireDagDecodeError::Contract)?;
        Ok(dag)
    }
}

fn wire_dim_info_equal(left: &WireDimInfo, right: &WireDimInfo) -> bool {
    match (left, right) {
        (WireDimInfo::Lit { size: left }, WireDimInfo::Lit { size: right }) => left == right,
        (
            WireDimInfo::Named {
                name: left_name,
                size: left_size,
            },
            WireDimInfo::Named {
                name: right_name,
                size: right_size,
            },
        ) => left_name == right_name && left_size == right_size,
        _ => false,
    }
}

/// Combined failure type for [`WireDag::from_validated_json`]. Parse,
/// exact-version, and cross-node contract failures remain distinct.
#[derive(Debug)]
pub enum WireDagDecodeError {
    /// The payload is not valid `WireDag` JSON.
    Parse(serde_json::Error),
    /// The payload parsed but its schema version is unsupported.
    Schema(WireDagSchemaError),
    /// The exact-version payload violates a cross-node wire invariant.
    Contract(WireDagContractError),
}

impl std::fmt::Display for WireDagDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireDagDecodeError::Parse(e) => write!(f, "WireDag JSON parse error: {e}"),
            WireDagDecodeError::Schema(e) => write!(f, "{e}"),
            WireDagDecodeError::Contract(e) => write!(f, "WireDag contract error: {e}"),
        }
    }
}

impl std::error::Error for WireDagDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WireDagDecodeError::Parse(e) => Some(e),
            WireDagDecodeError::Schema(e) => Some(e),
            WireDagDecodeError::Contract(e) => Some(e),
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
    Sub,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    MaxElem,
    MinElem,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireExtremaKind {
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireExtremaOperand {
    Left,
    Right,
}

/// chelis#616: wire form of `chelis_ir::dag::RtDim` for movement-op bounds
/// and reshape targets. `Node(i)` indexes the owning op's `inputs` (the
/// rank-0 integer bound scalars); `to_end` is the full-axis sentinel; `sym`
/// is a symbolic dim declared elsewhere (reshape targets only).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "bound", rename_all = "snake_case")]
pub enum WireRtDim {
    Lit { value: usize },
    ToEnd,
    Node { input: usize },
    Sym { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireRiscOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    CmpLt,
    MaxElem,
    MinElem,
    ExtremaAdjoint {
        extrema: WireExtremaKind,
        operand: WireExtremaOperand,
    },
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
        accumulator: String,
    },
    /// Dedicated exact multi-axis boolean cardinality reduction.
    ///
    /// `axes` is a complete, non-empty set of normalized original input
    /// positions in strictly descending order. [`WireDag`] validates the
    /// order and range against the referenced input before encoding or after
    /// exact-version decoding.
    Count {
        axes: Vec<usize>,
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
        new_shape: Vec<WireRtDim>,
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
        fill: ScalarValue,
    },
    Shrink {
        bounds: Vec<(WireRtDim, WireRtDim)>,
    },
    Stride {
        strides: Vec<WireRtDim>,
    },
    Const {
        /// Sealed dtype-tagged scalar (wire v4; finalize-on-decode).
        value: chelis_types::ScalarValue,
    },
    ConstTensor {
        /// Sealed per-dtype storage (wire v4; finalize-on-decode).
        data: chelis_types::TensorStorage,
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
    CastTrunc {
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

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_types::types::Prim;
    #[test]
    fn general_kind_projection_excludes_exactly_unsupported_feature() {
        for kind in DiagnosticKind::ALL {
            let projected = GeneralKind::project(kind);
            assert_eq!(
                projected.is_none(),
                kind == DiagnosticKind::UnsupportedFeature,
                "unexpected projection decision for {kind:?}"
            );
            if let Some(general) = projected {
                assert_eq!(general.diagnostic_kind(), kind);
            }
        }
    }

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

    #[test]
    fn wire_dag_missing_schema_version_is_rejected() {
        let versionless = r#"{"nodes":[],"roots":[]}"#;
        assert!(serde_json::from_str::<WireDag>(versionless).is_err());
        assert!(matches!(
            WireDag::from_validated_json(versionless),
            Err(WireDagDecodeError::Schema(
                WireDagSchemaError::MissingSchemaVersion {
                    supported: WIRE_DAG_SCHEMA_VERSION
                }
            ))
        ));
    }

    #[test]
    fn wire_dag_rejects_unknown_schema_version() {
        let future = WIRE_DAG_SCHEMA_VERSION + 1;
        let json = format!(r#"{{"schema_version":{future},"nodes":[],"roots":[]}}"#);

        assert!(serde_json::from_str::<WireDag>(&json).is_err());
        let dag = WireDag {
            schema_version: future,
            nodes: vec![],
            roots: vec![],
        };
        let err = dag
            .validate_schema_version()
            .expect_err("future schema version must be rejected");
        assert_eq!(
            err,
            WireDagSchemaError::UnsupportedSchemaVersion {
                found: future,
                supported: WIRE_DAG_SCHEMA_VERSION,
            },
            "rejection must be the typed UnsupportedSchemaVersion error"
        );

        match WireDag::from_validated_json(&json) {
            Err(WireDagDecodeError::Schema(WireDagSchemaError::UnsupportedSchemaVersion {
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
    fn wire_dag_v4_pad_fill_is_rejected_without_migration() {
        let legacy = r#"{
            "schema_version": 4,
            "nodes": [{
                "id": 0,
                "op": {"kind": "pad", "padding": [], "fill": 1.5},
                "inputs": [],
                "output_type": {"dims": [], "precision": "f32"}
            }],
            "roots": [0]
        }"#;
        assert!(matches!(
            WireDag::from_validated_json(legacy),
            Err(WireDagDecodeError::Schema(
                WireDagSchemaError::UnsupportedSchemaVersion {
                    found: 4,
                    supported: WIRE_DAG_SCHEMA_VERSION
                }
            ))
        ));
    }

    #[test]
    fn wire_dag_v6_rejects_raw_pad_fill() {
        let current_with_legacy_fill = r#"{
            "schema_version": 6,
            "nodes": [{
                "id": 0,
                "op": {"kind": "pad", "padding": [], "fill": 1.5},
                "inputs": [],
                "output_type": {"dims": [], "precision": "f32"}
            }],
            "roots": [0]
        }"#;
        assert!(
            matches!(
                WireDag::from_validated_json(current_with_legacy_fill),
                Err(WireDagDecodeError::Parse(_))
            ),
            "v6 must not retain a raw-number alternate Pad.fill spelling"
        );
    }

    #[test]
    fn wire_dag_legacy_versions_reject_typed_pad_fill_spelling() {
        let typed_fill = serde_json::to_value(
            chelis_types::scalar_from_f64("wire_pad_test", Prim::F32, 1.5)
                .expect("finite f32 fill"),
        )
        .expect("serialize typed fill");
        let legacy = serde_json::json!({
            "schema_version": 4,
            "nodes": [{
                "id": 0,
                "op": {"kind": "pad", "padding": [], "fill": typed_fill},
                "inputs": [],
                "output_type": {"dims": [], "precision": "f32"}
            }],
            "roots": [0]
        });
        assert!(matches!(
            WireDag::from_validated_json(&legacy.to_string()),
            Err(WireDagDecodeError::Schema(
                WireDagSchemaError::UnsupportedSchemaVersion {
                    found: 4,
                    supported: WIRE_DAG_SCHEMA_VERSION
                }
            ))
        ));
    }

    #[test]
    fn wire_dag_non_object_roots_reject_without_panicking() {
        for malformed in ["[]", "null", "42", r#""wire""#] {
            assert!(
                matches!(
                    WireDag::from_validated_json(malformed),
                    Err(WireDagDecodeError::Parse(_))
                ),
                "non-object root {malformed} must be a parse error"
            );
        }
    }

    #[test]
    fn wire_dag_rejects_malformed_schema_versions_before_op_decode() {
        for version in [r#""4""#, "-1", "4294967296"] {
            let json = format!(
                r#"{{
                    "schema_version": {version},
                    "nodes": [],
                    "roots": []
                }}"#
            );
            assert!(
                matches!(
                    WireDag::from_validated_json(&json),
                    Err(WireDagDecodeError::Parse(_))
                ),
                "malformed schema version {version} must not be treated as an absent v1 field"
            );
        }
    }

    #[test]
    fn wire_dag_v6_pad_fill_round_trips_exact_int64() {
        let exact = 9_007_199_254_740_993i64;
        let dag = WireDag {
            schema_version: WIRE_DAG_SCHEMA_VERSION,
            nodes: vec![WireDagNode {
                id: 0,
                op: WireRiscOp::Pad {
                    padding: vec![],
                    fill: chelis_types::scalar_from_i64("wire_pad_test", Prim::Int64, exact)
                        .expect("exact int64 fill"),
                },
                inputs: vec![],
                output_type: WireTensorType {
                    dims: vec![],
                    precision: "int64".to_string(),
                },
            }],
            roots: vec![0],
        };
        let json = serde_json::to_string(&dag).expect("serialize typed Pad");
        let decoded = WireDag::from_validated_json(&json).expect("decode typed Pad");
        match &decoded.nodes[0].op {
            WireRiscOp::Pad { fill, .. } => {
                assert_eq!(fill.prim(), Prim::Int64);
                assert_eq!(fill.as_i64_exact(), Some(exact));
            }
            other => panic!("expected Pad, got {other:?}"),
        }
    }

    #[test]
    fn wire_dag_v4_pad_rejects_before_inspecting_missing_precision() {
        let legacy = r#"{
            "schema_version": 4,
            "nodes": [{
                "id": 0,
                "op": {"kind": "pad", "padding": [], "fill": 1.5},
                "inputs": [],
                "output_type": {"dims": []}
            }],
            "roots": [0]
        }"#;
        assert!(
            matches!(
                WireDag::from_validated_json(legacy),
                Err(WireDagDecodeError::Schema(
                    WireDagSchemaError::UnsupportedSchemaVersion {
                        found: 4,
                        supported: WIRE_DAG_SCHEMA_VERSION
                    }
                ))
            ),
            "legacy schema rejection must precede node-field decoding"
        );
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
    fn direct_arithmetic_wire_identities_round_trip_without_surrogates() {
        let cases = [
            (WireRiscOp::Sub, r#"{"kind":"sub"}"#),
            (WireRiscOp::MinElem, r#"{"kind":"min_elem"}"#),
            (
                WireRiscOp::ExtremaAdjoint {
                    extrema: WireExtremaKind::Max,
                    operand: WireExtremaOperand::Left,
                },
                r#"{"kind":"extrema_adjoint","extrema":"max","operand":"left"}"#,
            ),
            (
                WireRiscOp::ExtremaAdjoint {
                    extrema: WireExtremaKind::Min,
                    operand: WireExtremaOperand::Right,
                },
                r#"{"kind":"extrema_adjoint","extrema":"min","operand":"right"}"#,
            ),
        ];

        for (op, expected) in cases {
            let encoded = serde_json::to_string(&op).expect("serialize direct arithmetic op");
            assert_eq!(encoded, expected);
            let decoded: WireRiscOp =
                serde_json::from_str(&encoded).expect("deserialize direct arithmetic op");
            assert_eq!(
                serde_json::to_string(&decoded).expect("re-serialize direct arithmetic op"),
                expected
            );
        }

        for (op, expected) in [
            (WireFusedStepOp::Sub, r#""sub""#),
            (WireFusedStepOp::MinElem, r#""min_elem""#),
        ] {
            let encoded = serde_json::to_string(&op).expect("serialize fused direct op");
            assert_eq!(encoded, expected);
            let decoded: WireFusedStepOp =
                serde_json::from_str(&encoded).expect("deserialize fused direct op");
            assert_eq!(
                serde_json::to_string(&decoded).expect("re-serialize fused direct op"),
                expected
            );
        }
    }

    #[test]
    fn direct_arithmetic_wire_schema_has_no_legacy_aliases() {
        for legacy in [
            r#"{"kind":"add_neg"}"#,
            r#"{"kind":"neg_add"}"#,
            r#"{"kind":"minimum"}"#,
            r#"{"kind":"min"}"#,
            r#"{"kind":"max_grad"}"#,
            r#"{"kind":"min_grad"}"#,
        ] {
            assert!(
                serde_json::from_str::<WireRiscOp>(legacy).is_err(),
                "legacy arithmetic alias must be rejected: {legacy}"
            );
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
