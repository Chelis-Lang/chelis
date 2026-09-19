mod artifact;
mod dag_domains;
mod directory;
mod envelopes;
mod execution;
pub mod numbers;
mod reports;
pub use artifact::{ArtifactAbiVersion, CompiledArtifactManifest};
pub use directory::{
    CheckDirectoryEntry, CheckDirectoryReport, EmptyWalk, EntryPath, UnrepresentablePath,
    WireCheckDirectoryEntry, WireCheckDirectoryReport, escaped_path,
};
pub use execution::NumericScalar;
use numbers::{NonnegativeCount, NonnegativeExtent, SourceFloat, SourceInteger, UnitInterval};

use std::collections::{BTreeMap, BTreeSet};

use chelis_types::unsupported::{Unsupported, UnsupportedIdentity};
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
    /// let _: ApiEnvelope<()> = ApiEnvelope::failure("compile".into(), vec!["compile_error"]);
    /// ```
    ///
    /// Nor may the public unsupported variant be handed directly to this
    /// general envelope API; it must arrive through a typed `Unsupported`:
    ///
    /// ```compile_fail
    /// use chelis_compiler_api::schema::ApiEnvelope;
    /// use chelis_vocab::DiagnosticKind;
    /// let _: ApiEnvelope<()> = ApiEnvelope::failure(
    ///     "compile".into(),
    ///     vec![DiagnosticKind::UnsupportedFeature],
    /// );
    /// ```
    pub fn failure(stage: String, errors: Vec<Diagnostic>) -> Self {
        Self::Failure(ApiFailure {
            ok: false,
            stage,
            errors,
        })
    }

    /// Build the HTTP request-decoding failure without exposing diagnostic
    /// construction to the transport crate.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::failure(
            "http".into(),
            vec![Diagnostic::general(
                GeneralKind::InvalidRequest,
                message,
                crate::schema::numbers::UnitInterval::new(1.0).expect("constant severity"),
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
#[derive(Debug, Clone)]
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
/// let _: ApiEnvelope<()> = ApiEnvelope::failure("compile".into(), vec![producer]);
/// ```
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct Diagnostic {
    kind: String,
    pub message: String,
    pub severity: UnitInterval,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub got: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<DiagnosticSpan>,
    /// Forward-compatible Deep-address slot for the L2 authoring loop. The
    /// fragment body-replacement check (`chelis_replace_function_body`) will
    /// populate this with the Deep path of the offending node so a caller can
    /// pinpoint the rejected subtree without re-deriving it. It is `None`
    /// today (L0; provenance threading is L2 work), and every other tool
    /// leaves it `None`, so the `skip_serializing_if` keeps their wire output
    /// byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep_path: Option<WireDeepErrorPath>,
    /// Opaque producer-supplied identity for the reported location, when the
    /// producer has one (chelis#886). Carried beside `span` rather than
    /// inside it because a producer may hold an identity without a resolved
    /// range, and [04-FIT-17] forbids inventing the range to fit.
    ///
    /// A `String`, so it adds no row to the §C6 wire numeric census.
    /// `skip_serializing_if` keeps every existing producer's bytes
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    /// In-process producer identity. Kept off the wire until unimplemented
    /// rows can carry their exact capability-table key.
    #[serde(skip)]
    #[schemars(skip)]
    pub(crate) unsupported: Option<Box<Unsupported>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedDiagnosticIdentity {
    pub brand: &'static str,
    pub kind: DiagnosticKind,
    pub payload: UnsupportedIdentity,
}

impl Diagnostic {
    pub fn kind(&self) -> DiagnosticKind {
        DiagnosticKind::decode(&self.kind)
            .expect("producer diagnostics are constructed from DiagnosticKind")
    }

    /// Wording-independent identity for a production unsupported diagnostic.
    pub fn unsupported_identity(&self) -> Option<UnsupportedDiagnosticIdentity> {
        self.unsupported.as_ref().map(|unsupported| {
            let payload = unsupported.identity();
            UnsupportedDiagnosticIdentity {
                brand: payload.brand,
                kind: self.kind(),
                payload,
            }
        })
    }

    pub(crate) fn general(
        kind: GeneralKind,
        message: impl Into<String>,
        severity: UnitInterval,
    ) -> Self {
        let diagnostic_kind = kind.diagnostic_kind();
        debug_assert_eq!(GeneralKind::project(diagnostic_kind), Some(kind));
        Self::new(diagnostic_kind, message, severity)
    }

    fn unsupported(error: Unsupported) -> Self {
        let mut diagnostic = Self::new(
            DiagnosticKind::UnsupportedFeature,
            error.to_string(),
            UnitInterval::new(1.0).expect("constant severity"),
        );
        if let Some(span) = error.span.as_deref() {
            diagnostic.span = match (span.offset, span.len) {
                (Some(offset), Some(len)) => Some(DiagnosticSpan::Range {
                    offset: host_index(offset),
                    len: host_index(len),
                }),
                (Some(offset), None) => Some(DiagnosticSpan::Point {
                    offset: host_index(offset),
                }),
                (None, _) => None,
            };
            diagnostic.span_id = span.span_id.clone();
        }
        diagnostic.unsupported = Some(Box::new(error));
        diagnostic
    }

    fn new(kind: DiagnosticKind, message: impl Into<String>, severity: UnitInterval) -> Self {
        Self {
            kind: kind.as_str().to_owned(),
            message: message.into(),
            severity,
            expected: None,
            got: None,
            suggestions: Vec::new(),
            span: None,
            deep_path: None,
            span_id: None,
            unsupported: None,
        }
    }
}

/// Projections of the checker's own diagnostic types onto this wire carrier
/// (chelis#886).
///
/// The `chelis check` report used to serialize `chelis_types::CheckError`
/// directly. That made a checker-internal type a numeric wire root -- its
/// public `severity: f64` sat outside the §C6 census, which is rooted here
/// -- and it spelled the kind from a Rust variant name, re-coupling the wire
/// to an identifier the sealed vocabulary exists to decouple it from.
///
/// Projecting instead of deriving fixes both at once. The carrier is this
/// already-enumerated type, so the census gains no numeric row; the kind is
/// `DiagnosticKind`'s governed spelling, so renaming a checker variant
/// cannot move the wire.
impl Diagnostic {
    /// Project a check diagnostic onto the wire carrier.
    pub fn try_from_check_error(error: &chelis_types::errors::CheckError) -> Result<Self, String> {
        Ok(Self {
            kind: check_error_kind(&error.kind).as_str().to_owned(),
            message: error.message.clone(),
            severity: UnitInterval::new(error.severity)?,
            expected: error.expected.clone(),
            got: error.got.clone(),
            // [04-FIT-15]: the field set does not vary by producing stage.
            // The embedding API's projection has always carried these; the
            // CLI report dropping them was the divergence, not this.
            suggestions: error.suggestions.clone(),
            span: check_error_span(error),
            deep_path: None,
            span_id: error.span_id.clone(),
            unsupported: None,
        })
    }

    /// Project an effect diagnostic onto the wire carrier.
    ///
    /// `EffectError` carries no severity of its own; the constant was
    /// inlined in the template this replaces.
    ///
    /// `suggestions` is carried across because [04-FIT-15] requires the field
    /// set not to vary by producing stage: the effect checker populates
    /// repair hints at six sites, and the template this replaces had no slot
    /// for them, so dropping them here would keep the very stage-dependence
    /// the atom forbids.
    pub fn from_effect_error(error: &chelis_effects::EffectError, severity: UnitInterval) -> Self {
        Self {
            kind: effect_error_kind(&error.kind).as_str().to_owned(),
            message: error.message.clone(),
            severity,
            expected: None,
            got: None,
            suggestions: error.suggestions.clone(),
            span: None,
            deep_path: None,
            span_id: None,
            unsupported: None,
        }
    }
}

/// The reported range, ONLY when the producer genuinely has one.
///
/// spec/04 [04-FIT-17]: where a producer holds a point or an opaque identity,
/// the serializer does not invent a length or a `0..0` range. `CheckError`
/// carries a byte offset and an opaque `span_id`; the id's canonical
/// `<source>:<start>..<end>` rendering is the only place a real end offset
/// exists, so the range is DERIVED from it and omitted when it cannot be.
///
/// The coordinate travels as `DiagnosticSpan::Point` whether or not an
/// identity accompanies it (chelis#1395). The carrier's variant tag records
/// whether an extent was measured, so [04-FIT-16]'s requirement that a
/// coordinate travel without an identity is met without weakening
/// [04-FIT-17]'s prohibition on inventing one: `Point` has no `len` field.
fn check_error_span(error: &chelis_types::errors::CheckError) -> Option<DiagnosticSpan> {
    // Always a point. `CheckError` carries `span_offset` and an opaque
    // `span_id`, and no measured extent -- so there is nothing here to build a
    // `Range` from.
    //
    // This previously recovered a length by parsing `N..M` out of `span_id`.
    // That is not provenance: `spec/03-deep-syntax.md` §1.1.1 makes external
    // span IDs opaque and their interpretation none of Chelis's concern, so a
    // foreign `octant:30..34` is a valid opaque identity that the parse turned
    // into a measured `Range { offset: 30, len: 4 }` nobody measured. A
    // numeric-looking identity is still an identity; spelling is not
    // provenance, and [04-FIT-17] forbids inventing an extent.
    //
    // A `Range` from this producer therefore waits on a typed field that
    // carries an extent Chelis itself measured. The stamp ingress path in
    // `compiler.rs` already has one and still reports `Range`.
    Some(DiagnosticSpan::Point {
        offset: host_index(error.span_offset?),
    })
}

/// Total map from the checker's kind to its governed vocabulary identity.
///
/// Exhaustive by construction: adding a `CheckErrorKind` variant does not
/// compile until its wire identity is chosen here.
fn check_error_kind(kind: &chelis_types::errors::CheckErrorKind) -> DiagnosticKind {
    use chelis_types::errors::CheckErrorKind as K;
    match kind {
        K::TypeMismatch => DiagnosticKind::TypeMismatch,
        K::PrecisionMismatch => DiagnosticKind::PrecisionMismatch,
        K::DimensionMismatch => DiagnosticKind::DimensionMismatch,
        K::ArityMismatch => DiagnosticKind::ArityMismatch,
        K::UnboundVariable { .. } => DiagnosticKind::UnboundVariable,
        K::UnknownConstructor { .. } => DiagnosticKind::UnknownConstructor,
        K::NotAFunction => DiagnosticKind::NotAFunction,
        K::NonExhaustiveMatch => DiagnosticKind::NonExhaustiveMatch,
        K::OccursCheck => DiagnosticKind::OccursCheck,
        K::CastNonTensor => DiagnosticKind::CastNonTensor,
        K::TupleIndexOutOfBounds => DiagnosticKind::TupleIndexOutOfBounds,
        K::UseAfterConsume => DiagnosticKind::UseAfterConsume,
        K::UnconsumedLinear => DiagnosticKind::UnconsumedLinear,
        K::InvalidBorrow => DiagnosticKind::InvalidBorrow,
        K::CycleDetected => DiagnosticKind::CycleDetected,
        K::UnsupportedTensorPrecision => DiagnosticKind::UnsupportedTensorPrecision,
        K::DuplicateDefinition => DiagnosticKind::DuplicateDefinition,
        K::DuplicateModule => DiagnosticKind::DuplicateModule,
        K::OpaqueTypeViolation => DiagnosticKind::OpaqueTypeViolation,
        K::ReservedLinkerName => DiagnosticKind::ReservedLinkerName,
        K::BuiltinShadowing => DiagnosticKind::BuiltinShadowing,
        K::UnknownForm => DiagnosticKind::UnknownForm,
        K::MalformedForm => DiagnosticKind::MalformedForm,
        K::Other => DiagnosticKind::CheckOther,
    }
}

/// Total map from the effect checker's kind to its governed identity.
fn effect_error_kind(kind: &chelis_effects::EffectErrorKind) -> DiagnosticKind {
    use chelis_effects::EffectErrorKind as K;
    match kind {
        K::UnhandledEffect => DiagnosticKind::UnhandledEffect,
        K::InvalidHandler => DiagnosticKind::InvalidHandler,
        K::BuildTargetMismatch => DiagnosticKind::BuildTargetMismatch,
        K::TypeTotality => DiagnosticKind::TypeTotality,
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
    pub severity: UnitInterval,
    pub expected: Option<String>,
    pub got: Option<String>,
    #[serde(default)]
    pub suggestions: Vec<String>,
    pub span: Option<DiagnosticSpan>,
    pub deep_path: Option<WireDeepErrorPath>,
    #[serde(default)]
    pub span_id: Option<String>,
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
            // chelis#886: the effect checker's kinds are not general-producer
            // kinds. They reach the wire only through
            // `Diagnostic::from_effect_error`, for the same reason
            // `UnsupportedFeature` is excluded above: a general producer
            // cannot spell a rejection it has no standing to make.
            DiagnosticKind::UnhandledEffect
            | DiagnosticKind::InvalidHandler
            | DiagnosticKind::BuildTargetMismatch
            | DiagnosticKind::TypeTotality => None,
            // chelis#1678: directory mode's walk failures reach the wire only
            // through `CheckDirectoryReport`, which is the one place that can
            // tell a walk failure from an empty corpus ([04-FIT-23],
            // [04-FIT-24]). chelis#1825's empty-test-selection kind is
            // produced only by the native test runner ([04-TEST-1..3]).
            DiagnosticKind::DirectoryWalkError
            | DiagnosticKind::EmptyCorpus
            | DiagnosticKind::EmptyTestSelection => None,
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
    span: Option<DiagnosticSpan>,
) -> crate::compiler::CompilerError {
    let mut diagnostic = Diagnostic::general(
        kind,
        message,
        crate::schema::numbers::UnitInterval::new(1.0).expect("constant severity"),
    );
    diagnostic.span = span;
    crate::compiler::CompilerError {
        transcript: Vec::new(),
        stage: stage.to_owned(),
        errors: vec![diagnostic],
    }
}

pub(crate) fn unsupported_stage_error(error: Unsupported) -> crate::compiler::CompilerError {
    crate::compiler::CompilerError {
        transcript: Vec::new(),
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
#[serde(deny_unknown_fields)]
pub struct Span {
    pub offset: u64,
    pub len: u64,
}

impl Span {
    /// Admit a foreign byte range for access to this local UTF-8 source.
    pub fn slice<'a>(&self, source: &'a str) -> Result<&'a str, String> {
        let end = self
            .offset
            .checked_add(self.len)
            .ok_or("source range endpoint exceeds u64")?;
        let start = usize::try_from(self.offset)
            .map_err(|_| "source offset exceeds host index capacity")?;
        let end =
            usize::try_from(end).map_err(|_| "source endpoint exceeds host index capacity")?;
        source.get(start..end).ok_or_else(|| {
            "source range is outside the buffer or splits a UTF-8 character".to_string()
        })
    }
}

/// Pure widening of an existing local index. A wider host must implement a
/// fallible producer path before it can build this wire implementation.
pub(crate) fn host_index(value: usize) -> u64 {
    const { assert!(usize::BITS <= u64::BITS) };
    value as u64
}

impl From<chelis_deep::Span> for Span {
    fn from(span: chelis_deep::Span) -> Self {
        Self {
            offset: host_index(span.offset),
            len: host_index(span.len),
        }
    }
}

/// A diagnostic's reported source location (chelis#1395).
///
/// Deliberately NOT `Span`. An AST node's span is structurally a range --
/// `WireParam`, `WireVariant`, `WireMatchArm` and `WireTypeInvariant` all
/// require one -- whereas a diagnostic's producer may hold only a coordinate.
/// Reusing `Span` there forces a choice between fabricating an extent, which
/// [04-FIT-17] forbids, and dropping the coordinate, which [04-FIT-16]
/// forbids.
///
/// The variant tag carries which of the two the producer measured, so the
/// prohibition is structural rather than a rule to remember: `Point` has no
/// `len` field to invent. This is the same carrier shape as `WireRtDim`,
/// whose `InputAxis.tensor` slot index is the wire census's registered
/// tagged transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "span", rename_all = "snake_case", deny_unknown_fields)]
pub enum DiagnosticSpan {
    /// The producer measured a range.
    Range { offset: u64, len: u64 },
    /// The producer held a coordinate and no extent.
    Point { offset: u64 },
}

impl DiagnosticSpan {
    /// The byte offset, which both variants carry.
    pub fn offset(self) -> u64 {
        match self {
            Self::Range { offset, .. } | Self::Point { offset } => offset,
        }
    }

    /// The measured extent, absent when the producer held only a coordinate.
    ///
    /// Named for [04-FIT-17]'s word rather than `len`: this is not a
    /// collection length, and calling it one invites `is_empty`, which would
    /// be meaningless for a source coordinate.
    pub fn extent(self) -> Option<u64> {
        match self {
            Self::Range { len, .. } => Some(len),
            Self::Point { .. } => None,
        }
    }
}

/// Execution-value schema version: exact stored-bit carriers, numeric scalar
/// tags, and checked int64 tensor shapes. Readers validate this stamp before
/// decoding values; older, missing and future versions have no fallback.
pub const EXECUTION_VALUE_SCHEMA_VERSION: u32 = 3;

/// The canonical sealed storage carrier, with the exact spec/10 bit codec.
pub type TensorElements = chelis_types::TensorStorage;

#[derive(Debug, Clone, JsonSchema)]
pub struct TensorValue {
    #[schemars(schema_with = "execution::shape_schema")]
    pub shape: Vec<i64>,
    pub data: TensorElements,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DictEntryValue {
    pub key: ExecutionValue,
    pub value: ExecutionValue,
}

/// Machine-facing execution value. Numeric descendants retain their sealed
/// dtype and stored bits; booleans have one separate execution spelling.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionValue {
    Tensor {
        value: TensorValue,
    },
    Scalar {
        value: NumericScalar,
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
/// validation (caller-ward effect closure, retained declaration dependencies,
/// and per-def type validation) is the future optimization, not a
/// fragment-scoped path that could disagree. Uniform base-case-free recursion
/// is checker-legal; backend support remains a separate chelis#730 boundary.
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
    pub renamed_references: NonnegativeCount,
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
    pub rewritten_calls: NonnegativeCount,
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
    pub parse: UnitInterval,
    pub structure: UnitInterval,
    pub names: UnitInterval,
    pub types: UnitInterval,
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub score: UnitInterval,
    pub components: FitnessComponents,
    pub typed_nodes: NonnegativeCount,
    pub untyped_nodes: NonnegativeCount,
    pub total_nodes: NonnegativeCount,
    pub unresolved_names: Vec<String>,
    /// Typed inferred-signature rows, omitted when the caller did not request them
    /// ([04-FIT-13]). Parameter references retain their owning list and exact indices.
    pub inferred_signatures: Option<Vec<WireInferredSignature>>,
    pub errors: Vec<Diagnostic>,
}

/// Consumer-side shape for `chelis check --json` output.
#[derive(Debug, Clone)]
pub struct WireCheckResult {
    pub score: UnitInterval,
    pub components: FitnessComponents,
    pub typed_nodes: NonnegativeCount,
    pub untyped_nodes: NonnegativeCount,
    pub total_nodes: NonnegativeCount,
    pub unresolved_names: Vec<String>,
    pub inferred_signatures: Option<Vec<WireInferredSignature>>,
    pub errors: Vec<WireDiagnostic>,
}

/// One function's inferred report, including its ordered parameter references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireInferredSignature {
    pub function: String,
    pub recursive_cycle: bool,
    pub checked_signature: String,
    pub display_signature: String,
    pub checked_signature_structured: WireInferredType,
    pub display_signature_structured: WireInferredType,
    pub effect_row: Vec<WireInferredEffect>,
    pub effect_row_display: Vec<String>,
    pub params: OrderedInferredParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireInferredParameter {
    pub index: u64,
    pub name: String,
    pub written: bool,
    pub inferred_read_only: bool,
    pub checked_type: String,
    pub display_type: String,
    pub checked_type_structured: WireInferredType,
    pub display_type_structured: WireInferredType,
}

/// Parameter references validated against their owning ordered list.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct OrderedInferredParameters(Vec<WireInferredParameter>);

impl OrderedInferredParameters {
    pub fn as_slice(&self) -> &[WireInferredParameter] {
        &self.0
    }
}

impl TryFrom<Vec<WireInferredParameter>> for OrderedInferredParameters {
    type Error = String;
    fn try_from(parameters: Vec<WireInferredParameter>) -> Result<Self, Self::Error> {
        for (position, parameter) in parameters.iter().enumerate() {
            let expected =
                u64::try_from(position).map_err(|_| "parameter position exceeds uint64")?;
            if parameter.index != expected {
                return Err(format!(
                    "inferred parameter index {} does not equal its owning list position {expected}",
                    parameter.index
                ));
            }
        }
        Ok(Self(parameters))
    }
}

impl<'de> Deserialize<'de> for OrderedInferredParameters {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<WireInferredParameter>::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
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
    /// `Type::Prim` — a scalar primitive (`f32`, `i64`, `bool`, ...).
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
    Lit { size: NonnegativeExtent },
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

#[derive(Debug, Clone)]
pub struct LowerResult {
    pub dag: WireDag,
    pub named_roots: BTreeMap<String, u64>,
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
    pub peak_device_bytes_estimate: Option<NonnegativeCount>,
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
    pub node_id: u64,
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

#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Execution-payload wire version (see
    /// [`EXECUTION_VALUE_SCHEMA_VERSION`]): producers stamp the current
    /// version; decode REQUIRES the field and rejects any other version
    /// loudly (the v1 compat default is gone, chelis#729 rework).
    pub schema_version: u32,
    pub roots: Vec<EvaluatedRoot>,
    pub manifest: RootManifestResult,
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

#[derive(Debug, Clone)]
pub struct GradResult {
    pub dag: WireDag,
    pub output_node: u64,
    pub grad_nodes_by_name: BTreeMap<String, u64>,
    pub forward_nodes_by_name: BTreeMap<String, u64>,
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
#[derive(Debug, Clone)]
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
    /// Producer data syntax, not a compiler expression. Re-ingestion validates
    /// its lexical data format before it can enter an ExtensionMap.
    ExtensionData {
        syntax: String,
    },
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
    Int { value: SourceInteger },
    Float { value: SourceFloat },
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
        type_binders: Vec<WireTypeBinder>,
        params: Vec<WireParam>,
        ret_ty: Option<WireSurfTypeExpr>,
        body: WireSurfExpr,
        span: Span,
    },
    Property {
        name: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        type_binders: Vec<WireTypeBinder>,
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

/// One entry of a declaration's `[..]` binder list. `bound` carries the
/// `spec/04-type-system.md` §5.9 dtype family when the binder declares one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTypeBinder {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound: Option<String>,
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
        index: SourceInteger,
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
        axis: Option<SourceInteger>,
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
        value: SourceInteger,
    },
    Float {
        value: SourceFloat,
    },
    /// Integer literal carrying an explicit precision suffix per spec
    /// §5.5. Suffix is one of `i8`/`i16`/`i32`/`i64`/`f32`/`f64`/`bf16`/`f16`.
    TypedInt {
        value: SourceInteger,
        suffix: String,
    },
    TypedFloat {
        value: SourceFloat,
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
///   Historical correction (chelis#1269): chelis#759 / PR #1144 subsequently
///   added `WireRiscOp::CastTrunc` (`kind: "cast_trunc"`, with
///   `new_precision`) while producers still stamped version 4. That addition
///   missed its required bump; consumers migrating from before #1144 must
///   also handle this operation. This records the omission, not an exception
///   to the version-bump rule.
/// - `5`: chelis#878 — `WireRiscOp::Pad::fill` changed from a bare f64
///   capacity seam to the sealed dtype-tagged scalar payload.
/// - `6`: chelis#1287 — added the dedicated multi-axis
///   `WireRiscOp::Count` form and made the complete WireDag encoding exact:
///   the version stamp and all fields are explicit, with no legacy migration
///   or default-on-read spellings. Chelis#1306 added direct `Sub`, `MinElem`,
///   and `ExtremaAdjoint` identities plus matching fused-step identities to
///   that exact encoding.
/// - `7`: chelis#1277 Slice A — `WireRiscOp::Expand::size` changed from a
///   display string to `WireRtDim`, and `WireRtDim` gained the structural
///   `InputAxis` metadata read. Chelis#1313 added the dedicated `Relu` and
///   `ReluAdjoint` identities to this unreleased exact schema.
/// - `8`: literal call witnesses and explicit invocation dependencies and
///   provenance. Requirement and dependency payloads used sealed int64 values.
/// - `9`: exact stored IEEE bits, active-dtype random parameters, fixed-width
///   numeric domains, and references validated in their declared owners.
///   Shape dependencies are exact u64 node identities and literal-witness
///   requirements use the fixed int64 extent carrier.
/// - `10`: checked reshape scalars and checked unit-axis refinements retain
///   independent actual/required values through graph transport; integer remainder
///   targets use the explicit `Mod` operation.
/// - `11`: chelis#1374/#1376 - `WireRiscOp::ExtentWitness` gained a mandatory
///   `claims` vector and one earlier-witness input per entry, so a declared
///   result's NAMED extent and a binder repeated across parameters transport as
///   obligations rather than being reconstructed. A consumer migrating from 10
///   must read the vector, the extra inputs, and each entry's
///   `requirement_declares` role, which the edges alone do not recover.
/// - `12`: result-claim witnesses carry a mandatory diagnostic label and
///   output axis. A producer's shape dependency names its exact declaring
///   extent; these obligations cannot be reconstructed from dimension names.
/// - `13`: literal-result claims have a distinct site role with one exact
///   requirement and one producer owner.
/// - `14`: local tensor-ascription claims carry mandatory authored identity,
///   binding, claim and axis fields, with exact literal or declaring-witness
///   forms and one initializer owner.
/// - `15`: comparison, logical, and conditional selection preserve their
///   direct identities as `Compare`, `Logical`, and `Where`; the standalone
///   `CmpLt` operation spelling is removed.
pub const WIRE_DAG_SCHEMA_VERSION: u32 = 15;

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
    pub roots: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
struct WireDagFields {
    schema_version: u32,
    nodes: Vec<WireDagNode>,
    roots: Vec<u64>,
}

#[derive(Serialize)]
struct WireDagFieldsRef<'a> {
    schema_version: u32,
    nodes: &'a [WireDagNode],
    roots: &'a [u64],
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
        let raw = Box::<serde_json::value::RawValue>::deserialize(deserializer)?;
        let found =
            envelopes::version(raw.get()).map_err(<D::Error as serde::de::Error>::custom)?;
        validate_explicit_wire_dag_schema_version(found)
            .map_err(<D::Error as serde::de::Error>::custom)?;
        let fields: WireDagFields =
            serde_json::from_str(raw.get()).map_err(<D::Error as serde::de::Error>::custom)?;
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
        dag_domains::validate(self)?;
        for (index, node) in self.nodes.iter().enumerate() {
            match &node.op {
                WireRiscOp::Compare { comparison } => {
                    let inputs = node
                        .inputs
                        .iter()
                        .map(|input_id| {
                            self.nodes[..index]
                                .iter()
                                .find(|candidate| candidate.id == *input_id)
                                .ok_or_else(|| {
                                    WireDagContractError::new(format!(
                                        "WireDag Compare node {} input {input_id} does not resolve to an earlier node",
                                        node.id
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if inputs.len() != 2 {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Compare node {} requires exactly two inputs",
                            node.id
                        )));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let same_shape = wire_node_shape_equal(&self.nodes, lhs, rhs);
                    let output_shape = wire_node_shape_equal(&self.nodes, lhs, node);
                    let ordered = matches!(
                        comparison,
                        WireComparisonKind::CmpLt
                            | WireComparisonKind::Lt
                            | WireComparisonKind::Gt
                            | WireComparisonKind::Gte
                            | WireComparisonKind::Lte
                    );
                    let operand = Prim::parse_interchange_name(&lhs.output_type.precision);
                    let valid_operand = operand.is_some_and(|prim| {
                        (prim.is_numeric() && prim.is_admissible_active()) || prim == Prim::Bool
                    });
                    let valid_ordered_operand = !ordered
                        || operand
                            .is_some_and(|prim| prim.is_numeric() && prim.is_admissible_active());
                    if lhs.output_type.precision != rhs.output_type.precision
                        || !same_shape
                        || !output_shape
                        || node.output_type.precision != Prim::Bool.interchange_name()
                        || !valid_operand
                        || !valid_ordered_operand
                    {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Compare node {} requires two same-shape, same-precision active numeric or bool operands and a same-shape Bool output; ordered comparisons require active numeric operands",
                            node.id
                        )));
                    }
                }
                WireRiscOp::Logical { logical } => {
                    let expected = match logical {
                        WireLogicalKind::And | WireLogicalKind::Or => 2,
                        WireLogicalKind::Not => 1,
                    };
                    let inputs = node
                        .inputs
                        .iter()
                        .map(|input_id| {
                            self.nodes[..index]
                                .iter()
                                .find(|candidate| candidate.id == *input_id)
                                .ok_or_else(|| {
                                    WireDagContractError::new(format!(
                                        "WireDag Logical node {} input {input_id} does not resolve to an earlier node",
                                        node.id
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let valid_inputs = inputs.iter().all(|input| {
                        input.output_type.precision == Prim::Bool.interchange_name()
                            && wire_node_shape_equal(&self.nodes, input, node)
                    });
                    if inputs.len() != expected
                        || node.output_type.precision != Prim::Bool.interchange_name()
                        || !valid_inputs
                    {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Logical node {} requires exactly {expected} same-shape bool input(s) and a Bool output",
                            node.id
                        )));
                    }
                }
                WireRiscOp::Where {} => {
                    let inputs = node
                        .inputs
                        .iter()
                        .map(|input_id| {
                            self.nodes[..index]
                                .iter()
                                .find(|candidate| candidate.id == *input_id)
                                .ok_or_else(|| {
                                    WireDagContractError::new(format!(
                                        "WireDag Where node {} input {input_id} does not resolve to an earlier node",
                                        node.id
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if inputs.len() != 3 {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Where node {} requires exactly three inputs",
                            node.id
                        )));
                    }
                    let condition = inputs[0];
                    let then_value = inputs[1];
                    let else_value = inputs[2];
                    let branch_precision =
                        Prim::parse_interchange_name(&then_value.output_type.precision);
                    if condition.output_type.precision != Prim::Bool.interchange_name()
                        || !wire_node_tensor_type_equal(&self.nodes, then_value, else_value)
                        || !wire_node_tensor_type_equal(&self.nodes, then_value, node)
                        || !wire_node_shape_equal(&self.nodes, condition, then_value)
                        || !branch_precision.is_some_and(|prim| prim.is_valid_tensor_precision())
                    {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Where node {} requires a same-shape Bool condition and exactly matching branch/output types",
                            node.id
                        )));
                    }
                }
                WireRiscOp::Mod => {
                    if node.inputs.len() != 2
                        || !Prim::parse_interchange_name(&node.output_type.precision)
                            .is_some_and(|prim| prim.is_integer())
                        || node.inputs.iter().any(|id| {
                            self.nodes[..index]
                                .iter()
                                .find(|input| input.id == *id)
                                .is_none_or(|input| {
                                    input.output_type.precision != node.output_type.precision
                                        || input.output_type.dims.len()
                                            != node.output_type.dims.len()
                                        || input
                                            .output_type
                                            .dims
                                            .iter()
                                            .zip(&node.output_type.dims)
                                            .any(|(actual, expected)| {
                                                !wire_dim_info_equal(actual, expected)
                                            })
                                })
                        })
                    {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Mod node {} requires two earlier inputs with its integer dtype and shape",
                            node.id
                        )));
                    }
                }
                WireRiscOp::CheckedReshapeExtent {
                    axis: WireRtAxis::Lit { value: axis },
                    claims,
                } => {
                    let scalar =
                        |ty: &WireTensorType| ty.dims.is_empty() && ty.precision == "int64";
                    if *axis < 0
                        || claims.is_empty()
                        || node.inputs.len() != claims.len() + 1
                        || !scalar(&node.output_type)
                        || node.inputs.iter().any(|id| {
                            usize::try_from(*id)
                                .ok()
                                .and_then(|id| self.nodes.get(id))
                                .is_none_or(|input| !scalar(&input.output_type))
                        })
                    {
                        return Err(WireDagContractError::new(format!(
                            "WireDag CheckedReshapeExtent node {} requires an earlier scalar int64 actual, one earlier scalar int64 input per nonempty claim, and a scalar int64 output",
                            node.id
                        )));
                    }
                }
                WireRiscOp::CheckedUnitAxis {
                    axis: WireRtAxis::Lit { value: axis },
                } => {
                    let valid = (|| {
                        let [input_id, witness_id] = node.inputs.as_slice() else {
                            return None;
                        };
                        if *input_id >= host_index(index) || *witness_id >= host_index(index) {
                            return None;
                        }
                        let input = self.nodes.get(usize::try_from(*input_id).ok()?)?;
                        let witness = self.nodes.get(usize::try_from(*witness_id).ok()?)?;
                        let WireRiscOp::ExtentWitness {
                            axis:
                                WireRtAxis::Lit {
                                    value: observed_axis,
                                },
                            requirements,
                            ..
                        } = &witness.op
                        else {
                            return None;
                        };
                        if input.id != *input_id
                            || witness.id != *witness_id
                            // wire v11: `inputs[0]` is the observed tensor and
                            // `inputs[1..]` are the witness's named-claim
                            // requirement edges, so the refinement ties to the
                            // FIRST input rather than to a sole input.
                            || witness.inputs.first() != Some(input_id)
                            || observed_axis != axis
                            || !requirements.iter().any(|value| value.get() == 1)
                        {
                            return None;
                        }
                        let axis = usize::try_from(*axis).ok()?;
                        if axis >= input.output_type.dims.len()
                            || input.output_type.precision != node.output_type.precision
                            || input.output_type.dims.len() != node.output_type.dims.len()
                        {
                            return None;
                        }
                        input
                            .output_type
                            .dims
                            .iter()
                            .zip(&node.output_type.dims)
                            .enumerate()
                            .all(|(index, (input, output))| {
                                if index == axis {
                                    return matches!(output, WireDimInfo::Lit { size } if size.get() == 1);
                                }
                                match (input, output) {
                                    (
                                        WireDimInfo::Lit { size: lhs },
                                        WireDimInfo::Lit { size: rhs },
                                    ) => lhs == rhs,
                                    (
                                        WireDimInfo::Named {
                                            name: lhs,
                                            size: lhs_size,
                                        },
                                        WireDimInfo::Named {
                                            name: rhs,
                                            size: rhs_size,
                                        },
                                    ) => lhs == rhs && lhs_size == rhs_size,
                                    _ => false,
                                }
                            })
                            .then_some(())
                    })()
                    .is_some();
                    if !valid {
                        return Err(WireDagContractError::new(format!(
                            "WireDag CheckedUnitAxis node {} requires its own tensor-axis witness with requirement one and only that axis refined",
                            node.id
                        )));
                    }
                }
                WireRiscOp::Expand { size, .. } => {
                    validate_wire_rt_dim(&self.nodes, index, node, size, false, true, "Expand")?;
                    let expected_inputs = match size {
                        WireRtDim::Lit { .. } => 1,
                        WireRtDim::Node { input: 1 } | WireRtDim::InputAxis { tensor: 1, .. } => 2,
                        WireRtDim::Node { input } | WireRtDim::InputAxis { tensor: input, .. } => {
                            return Err(WireDagContractError::new(format!(
                                "WireDag Expand node {} size must reference absolute input slot 1, found {input}",
                                node.id
                            )));
                        }
                        WireRtDim::ToEnd | WireRtDim::Sym { .. } => {
                            unreachable!("owner validation rejects forbidden Expand carriers")
                        }
                    };
                    if node.inputs.len() != expected_inputs {
                        return Err(WireDagContractError::new(format!(
                            "WireDag Expand node {} has {} inputs; size requires {expected_inputs}",
                            node.id,
                            node.inputs.len()
                        )));
                    }
                }
                WireRiscOp::Reshape { new_shape } => {
                    for dim in new_shape {
                        validate_wire_rt_dim(&self.nodes, index, node, dim, true, true, "Reshape")?;
                    }
                    validate_exact_wire_rt_dim_inputs(node, new_shape, "Reshape")?;
                }
                WireRiscOp::Pad { padding, .. } => {
                    for (start, end) in padding {
                        validate_wire_rt_dim(
                            &self.nodes,
                            index,
                            node,
                            start,
                            false,
                            false,
                            "Pad start",
                        )?;
                        validate_wire_rt_dim(
                            &self.nodes,
                            index,
                            node,
                            end,
                            false,
                            false,
                            "Pad end",
                        )?;
                    }
                    validate_exact_wire_rt_dim_inputs(
                        node,
                        padding.iter().flat_map(|(start, end)| [start, end]),
                        "Pad",
                    )?;
                }
                WireRiscOp::Shrink { bounds } => {
                    for (start, end) in bounds {
                        validate_wire_rt_dim(
                            &self.nodes,
                            index,
                            node,
                            start,
                            false,
                            false,
                            "Shrink start",
                        )?;
                        validate_wire_rt_dim(
                            &self.nodes,
                            index,
                            node,
                            end,
                            false,
                            false,
                            "Shrink end",
                        )?;
                        // chelis#1480, `spec/05` section 2.4.1: a `ToEnd` end
                        // is well formed only beside a `Lit(0)` start. The
                        // per-carrier validation above sees one bound at a
                        // time and cannot see the pairing, so the decoder
                        // checks it here, where both halves are in hand.
                        if matches!(end, WireRtDim::ToEnd)
                            && !matches!(start, WireRtDim::Lit { value } if value.get() == 0)
                        {
                            return Err(WireDagContractError::new(format!(
                                "WireDag Shrink node {} pairs the to_end carrier with a start \
                                 that is not literal 0, which is a malformed bound",
                                node.id
                            )));
                        }
                    }
                    validate_exact_wire_rt_dim_inputs(
                        node,
                        bounds.iter().flat_map(|(start, end)| [start, end]),
                        "Shrink",
                    )?;
                }
                WireRiscOp::Stride { strides } => {
                    for stride in strides {
                        validate_wire_rt_dim(
                            &self.nodes,
                            index,
                            node,
                            stride,
                            false,
                            false,
                            "Stride",
                        )?;
                    }
                    validate_exact_wire_rt_dim_inputs(node, strides, "Stride")?;
                }
                _ => {}
            }

            if let WireRiscOp::Pad { fill, .. } = &node.op {
                let output_prim = Prim::parse_interchange_name(&node.output_type.precision)
                    .ok_or_else(|| {
                        WireDagContractError::new(format!(
                            "WireDag Pad node {} has unknown output dtype {}",
                            node.id, node.output_type.precision
                        ))
                    })?;
                if fill.prim() != output_prim {
                    return Err(WireDagContractError::new(format!(
                        "WireDag Pad fill dtype {} does not match node {} output dtype {}",
                        fill.prim().interchange_name(),
                        node.id,
                        output_prim.interchange_name()
                    )));
                }
            }

            if matches!(&node.op, WireRiscOp::Relu | WireRiscOp::ReluAdjoint) {
                let expected_inputs = if matches!(&node.op, WireRiscOp::Relu) {
                    1
                } else {
                    2
                };
                if node.inputs.len() != expected_inputs {
                    return Err(WireDagContractError::new(format!(
                        "WireDag ReLU node {} requires exactly {expected_inputs} input(s), found {}",
                        node.id,
                        node.inputs.len()
                    )));
                }
                let inputs = node
                    .inputs
                    .iter()
                    .map(|input_id| {
                        self.nodes[..index]
                            .iter()
                            .find(|candidate| candidate.id == *input_id)
                            .ok_or_else(|| {
                                WireDagContractError::new(format!(
                                    "WireDag ReLU node {} input {input_id} does not resolve to an earlier node",
                                    node.id
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let Some(output_prim) = Prim::parse_interchange_name(&node.output_type.precision)
                else {
                    return Err(WireDagContractError::new(format!(
                        "WireDag ReLU node {} has unknown output dtype {}",
                        node.id, node.output_type.precision
                    )));
                };
                if !matches!(output_prim, Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64) {
                    return Err(WireDagContractError::new(format!(
                        "WireDag ReLU node {} output dtype must be float, found {}",
                        node.id, node.output_type.precision
                    )));
                }
                if inputs.iter().any(|input| {
                    input.output_type.precision != node.output_type.precision
                        || input.output_type.dims.len() != node.output_type.dims.len()
                        || input
                            .output_type
                            .dims
                            .iter()
                            .zip(&node.output_type.dims)
                            .any(|(actual, expected)| !wire_dim_info_equal(actual, expected))
                }) {
                    return Err(WireDagContractError::new(format!(
                        "WireDag ReLU node {} inputs must match its output shape and float dtype",
                        node.id
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
            if let Some(axis) = axes
                .iter()
                .copied()
                .find(|axis| usize::try_from(*axis).map_or(true, |axis| axis >= input_rank))
            {
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
            if node.output_type.precision != Prim::Int64.interchange_name() {
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
                .filter(|(axis, _)| i32::try_from(*axis).map_or(true, |axis| !axes.contains(&axis)))
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
        let found = envelopes::version(json).map_err(WireDagDecodeError::Parse)?;
        validate_explicit_wire_dag_schema_version(found).map_err(WireDagDecodeError::Schema)?;
        let fields: WireDagFields =
            serde_json::from_str(json).map_err(WireDagDecodeError::Parse)?;
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

fn validate_wire_rt_dim(
    nodes: &[WireDagNode],
    owner_index: usize,
    owner: &WireDagNode,
    dim: &WireRtDim,
    allow_sym: bool,
    allow_input_axis: bool,
    label: &str,
) -> Result<(), WireDagContractError> {
    let resolve_slot = |wire_slot: u64| -> Result<&WireDagNode, WireDagContractError> {
        let slot = usize::try_from(wire_slot).map_err(|_| {
            WireDagContractError::new("runtime input reference exceeds host capacity")
        })?;
        if slot == 0 || slot >= owner.inputs.len() {
            return Err(WireDagContractError::new(format!(
                "WireDag {label} node {} references invalid input slot {slot} (inputs len {})",
                owner.id,
                owner.inputs.len()
            )));
        }
        let source_id = owner.inputs[slot];
        nodes[..owner_index]
            .iter()
            .find(|candidate| candidate.id == source_id)
            .ok_or_else(|| {
                WireDagContractError::new(format!(
                    "WireDag {label} node {} input slot {slot} does not resolve to an earlier node",
                    owner.id
                ))
            })
    };

    match dim {
        WireRtDim::Lit { .. } => Ok(()),
        WireRtDim::ToEnd if label == "Shrink end" => Ok(()),
        WireRtDim::ToEnd => Err(WireDagContractError::new(format!(
            "WireDag {label} node {} forbids the to_end carrier",
            owner.id
        ))),
        WireRtDim::Sym { .. } if allow_sym => Ok(()),
        WireRtDim::Sym { .. } => Err(WireDagContractError::new(format!(
            "WireDag {label} node {} forbids the sym carrier",
            owner.id
        ))),
        WireRtDim::Node { input } => {
            let source = resolve_slot(*input)?;
            if !source.output_type.dims.is_empty() || source.output_type.precision != "int64" {
                return Err(WireDagContractError::new(format!(
                    "WireDag {label} node {} input slot {input} must be an exact rank-0 int64 extent scalar",
                    owner.id
                )));
            }
            Ok(())
        }
        WireRtDim::InputAxis { .. } if !allow_input_axis => {
            Err(WireDagContractError::new(format!(
                "WireDag {label} node {} forbids the input_axis carrier",
                owner.id
            )))
        }
        WireRtDim::InputAxis {
            tensor,
            axis: WireRtAxis::Lit { value },
        } => {
            let source = resolve_slot(*tensor)?;
            let axis = usize::try_from(*value).map_err(|_| {
                WireDagContractError::new(format!(
                    "WireDag {label} node {} InputAxis value {value} is not normalized",
                    owner.id
                ))
            })?;
            if axis >= source.output_type.dims.len() {
                return Err(WireDagContractError::new(format!(
                    "WireDag {label} node {} InputAxis {axis} is out of range for source rank {}",
                    owner.id,
                    source.output_type.dims.len()
                )));
            }
            Ok(())
        }
    }
}

fn validate_exact_wire_rt_dim_inputs<'a>(
    owner: &WireDagNode,
    dims: impl IntoIterator<Item = &'a WireRtDim>,
    label: &str,
) -> Result<(), WireDagContractError> {
    let owned = dims
        .into_iter()
        .filter_map(|dim| match dim {
            WireRtDim::Node { input } | WireRtDim::InputAxis { tensor: input, .. } => Some(*input),
            WireRtDim::Lit { .. } | WireRtDim::ToEnd | WireRtDim::Sym { .. } => None,
        })
        .collect::<BTreeSet<_>>();
    for input in 1..owner.inputs.len() {
        if !owned.contains(&host_index(input)) {
            return Err(WireDagContractError::new(format!(
                "WireDag {label} node {} has unowned runtime extent input slot {input}",
                owner.id
            )));
        }
    }
    Ok(())
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

fn wire_semantic_dim_info_equal(left: &WireDimInfo, right: &WireDimInfo) -> bool {
    match (left, right) {
        (WireDimInfo::Lit { size: left }, WireDimInfo::Lit { size: right }) => left == right,
        (
            WireDimInfo::Lit { size: left },
            WireDimInfo::Named {
                size: Some(right), ..
            },
        )
        | (
            WireDimInfo::Named {
                size: Some(left), ..
            },
            WireDimInfo::Lit { size: right },
        )
        | (
            WireDimInfo::Named {
                size: Some(left), ..
            },
            WireDimInfo::Named {
                size: Some(right), ..
            },
        ) => left == right,
        (
            WireDimInfo::Named {
                name: left_name,
                size: None,
            },
            WireDimInfo::Named {
                name: right_name,
                size: None,
            },
        ) => left_name == right_name,
        _ => false,
    }
}

fn wire_dim_is_anonymous(dim: &WireDimInfo) -> bool {
    matches!(
        dim,
        WireDimInfo::Named { name, size: None } if name.is_empty() || name == "*"
    )
}

fn wire_node_by_id(nodes: &[WireDagNode], id: u64) -> Option<&WireDagNode> {
    nodes.iter().find(|node| node.id == id)
}

fn wire_semantic_node_dim<'a>(
    nodes: &'a [WireDagNode],
    node: &'a WireDagNode,
    axis: usize,
    fuel: usize,
) -> Option<&'a WireDimInfo> {
    if fuel == 0 {
        return None;
    }
    let dim = node.output_type.dims.get(axis)?;
    if wire_dim_is_anonymous(dim) {
        if let Some(resolved) = node.shape_deps.iter().find_map(|source_id| {
            let source = wire_node_by_id(nodes, *source_id)?;
            (source.id < node.id && source.output_type.dims.len() == node.output_type.dims.len())
                .then(|| wire_semantic_node_dim(nodes, source, axis, fuel - 1))
                .flatten()
                .filter(|resolved| !wire_dim_is_anonymous(resolved))
        }) {
            return Some(resolved);
        }
        if matches!(node.op, WireRiscOp::Where { .. })
            && let Some(resolved) = node.inputs.iter().skip(1).find_map(|source_id| {
                let source = wire_node_by_id(nodes, *source_id)?;
                (source.id < node.id
                    && source.output_type.dims.len() == node.output_type.dims.len())
                .then(|| wire_semantic_node_dim(nodes, source, axis, fuel - 1))
                .flatten()
                .filter(|resolved| !wire_dim_is_anonymous(resolved))
            })
        {
            return Some(resolved);
        }
    }
    Some(dim)
}

fn wire_node_shape_equal(nodes: &[WireDagNode], left: &WireDagNode, right: &WireDagNode) -> bool {
    left.output_type.dims.len() == right.output_type.dims.len()
        && (0..left.output_type.dims.len()).all(|axis| {
            wire_semantic_node_dim(nodes, left, axis, nodes.len())
                .zip(wire_semantic_node_dim(nodes, right, axis, nodes.len()))
                .is_some_and(|(left, right)| wire_semantic_dim_info_equal(left, right))
        })
}

fn wire_node_tensor_type_equal(
    nodes: &[WireDagNode],
    left: &WireDagNode,
    right: &WireDagNode,
) -> bool {
    left.output_type.precision == right.output_type.precision
        && wire_node_shape_equal(nodes, left, right)
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
    /// Exact u64 references to earlier shape-only dependencies.
    pub shape_deps: Vec<u64>,
    #[serde(deserialize_with = "require_explicit_span")]
    pub span_id: Option<String>,
    pub merged_spans: Vec<String>,
    pub id: u64,
    pub op: WireRiscOp,
    pub inputs: Vec<u64>,
    pub output_type: WireTensorType,
}

fn require_explicit_span<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    Option::<String>::deserialize(deserializer)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensorType {
    pub dims: Vec<WireDimInfo>,
    pub precision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireDimInfo {
    Named {
        name: String,
        size: Option<NonnegativeExtent>,
    },
    Lit {
        size: NonnegativeExtent,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireDimExpr {
    Concrete {
        value: NonnegativeExtent,
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
    External { index: u64 },
    PreviousStep { index: u64 },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireComparisonKind {
    CmpLt,
    Lt,
    Eq,
    Neq,
    Gt,
    Gte,
    Lte,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireLogicalKind {
    And,
    Or,
    Not,
}

/// chelis#616: wire form of `chelis_ir::dag::RtDim` for movement-op bounds
/// and reshape targets. `Node(i)` indexes the owning op's `inputs` (the
/// rank-0 integer bound scalars); `to_end` is the full-axis sentinel; `sym`
/// is a symbolic dim declared elsewhere (reshape targets only).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "bound", rename_all = "snake_case")]
pub enum WireRtDim {
    Lit { value: NonnegativeExtent },
    ToEnd,
    Node { input: u64 },
    Sym { name: String },
    InputAxis { tensor: u64, axis: WireRtAxis },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "axis", rename_all = "snake_case")]
pub enum WireRtAxis {
    Lit { value: i32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireExtentWitnessSite {
    LiteralResultClaim,
    Caller,
    LocalExpand,
    ResultClaim {
        claim: String,
        axis: WireRtAxis,
    },
    LocalAscriptionClaim {
        ascription_id: u64,
        binding: String,
        claim: String,
        axis: WireRtAxis,
    },
}

/// One named equality a witness owes against another witness (wire v11).
///
/// `claim` is the dimension binder; `requirement_declares` says which of the
/// two witnesses declares it, so a consumer renders the declaring side first
/// without re-deriving the signature. Both fields are mandatory: a payload
/// missing either is a decoding error, and no default is supplied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireExtentClaim {
    pub claim: String,
    pub requirement_declares: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WireRiscOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    Mod,
    Compare {
        comparison: WireComparisonKind,
    },
    Logical {
        logical: WireLogicalKind,
    },
    Where {},
    MaxElem,
    MinElem,
    ExtremaAdjoint {
        extrema: WireExtremaKind,
        operand: WireExtremaOperand,
    },
    Relu,
    ReluAdjoint,
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
        low: ScalarValue,
        high: ScalarValue,
        seed: u64,
    },
    Dropout {
        rate: ScalarValue,
        seed: u64,
    },
    Sum {
        axis: i32,
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
        axes: Vec<i32>,
    },
    MaxReduce {
        axis: i32,
    },
    MinReduce {
        axis: i32,
    },
    ProdReduce {
        axis: i32,
    },
    ReduceWindow {
        /// One of "max" / "min" / "sum" / "mean", matching the Surf
        /// builtin name suffix and `chelis_ir::dag::ReduceWindowKind`.
        reducer: String,
        window_shape: Vec<NonnegativeExtent>,
        strides: Vec<NonnegativeExtent>,
    },
    /// Reverse-mode adjoint of `ReduceWindow` (`RiscOp::ReduceWindowGrad`).
    /// Carries the same `reducer` / window / stride contract; appears only
    /// in `grad`-lowered DAGs.
    ReduceWindowGrad {
        /// One of "max" / "min" / "sum" / "mean", as for `ReduceWindow`.
        reducer: String,
        window_shape: Vec<NonnegativeExtent>,
        strides: Vec<NonnegativeExtent>,
    },
    Argmax {
        axis: i32,
    },
    Argmin {
        axis: i32,
    },
    Reshape {
        new_shape: Vec<WireRtDim>,
    },
    Permute {
        axes: Vec<i32>,
    },
    Expand {
        axis: i32,
        size: WireRtDim,
    },
    OneHot {
        vocab: NonnegativeExtent,
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
        /// Sealed dtype-tagged scalar (wire v9; exact stored-bit transport).
        value: chelis_types::ScalarValue,
    },
    ConstTensor {
        /// Sealed per-dtype storage (wire v9; exact stored-bit transport).
        data: chelis_types::TensorStorage,
    },
    Shape {
        axis: i32,
    },
    ExtentWitness {
        site: WireExtentWitnessSite,
        parameter: String,
        axis: WireRtAxis,
        requirements: Vec<NonnegativeExtent>,
        claims: Vec<WireExtentClaim>,
    },
    CheckedReshapeExtent {
        claims: Vec<String>,
        axis: WireRtAxis,
    },
    CheckedUnitAxis {
        axis: WireRtAxis,
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
        axis: i32,
    },
    ScatterAdd {
        axis: i32,
    },
    Scatter {
        axis: i32,
    },
    ScatterElements {
        axis: i32,
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
    fn unsupported_sidecar_storage_is_pointer_sized() {
        let diagnostic = Diagnostic::unsupported(Unsupported::new(
            chelis_types::unsupported::UnsupportedKind::Builtin("softmax".into()),
            "`chelis build` host emission",
            chelis_types::unsupported::Stage::Codegen("c"),
            chelis_types::deliberate_rejection!(
                "[04-TOT-2]",
                "the checked builtin vocabulary and C expression vocabulary disagree"
            ),
        ));
        assert_eq!(
            std::mem::size_of_val(&diagnostic.unsupported),
            std::mem::size_of::<usize>(),
            "the off-wire typed sidecar must not inline Unsupported into every Diagnostic"
        );
    }

    /// The kinds a general producer has no standing to spell.
    ///
    /// `UnsupportedFeature` is the original member. chelis#886 adds the
    /// effect checker's four, which reach the wire only through
    /// `Diagnostic::from_effect_error`. Stated as a list so that adding a
    /// governed identity and quietly excluding it from general production
    /// has to be written down here. chelis#1678 adds directory mode's two,
    /// which only `CheckDirectoryReport` produces; chelis#1825 adds the
    /// native test runner's zero-selection diagnostic.
    const NON_GENERAL_KINDS: [DiagnosticKind; 8] = [
        DiagnosticKind::UnsupportedFeature,
        DiagnosticKind::UnhandledEffect,
        DiagnosticKind::InvalidHandler,
        DiagnosticKind::BuildTargetMismatch,
        DiagnosticKind::TypeTotality,
        DiagnosticKind::DirectoryWalkError,
        DiagnosticKind::EmptyCorpus,
        DiagnosticKind::EmptyTestSelection,
    ];

    #[test]
    fn general_kind_projection_excludes_exactly_the_non_general_kinds() {
        for kind in DiagnosticKind::ALL {
            let projected = GeneralKind::project(kind);
            assert_eq!(
                projected.is_none(),
                NON_GENERAL_KINDS.contains(&kind),
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
    fn extent_witness_wire_requires_claims_and_roundtrips_exactly() {
        let witness = WireRiscOp::ExtentWitness {
            site: WireExtentWitnessSite::Caller,
            parameter: "x".into(),
            axis: WireRtAxis::Lit { value: 0 },
            requirements: vec![NonnegativeExtent::new(4).unwrap()],
            claims: vec![WireExtentClaim {
                claim: "rows".into(),
                requirement_declares: true,
            }],
        };
        let json = serde_json::to_value(&witness).unwrap();
        let back: WireRiscOp = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(back).unwrap(), json);
        let mut missing = json;
        missing.as_object_mut().unwrap().remove("requirements");
        assert!(serde_json::from_value::<WireRiscOp>(missing).is_err());
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
    fn current_wire_dag_rejects_raw_pad_fill() {
        let current_with_legacy_fill = serde_json::json!({
            "schema_version": WIRE_DAG_SCHEMA_VERSION,
            "nodes": [{
                "shape_deps": [],
                "span_id": null,
                "merged_spans": [],
                "id": 0,
                "op": {"kind": "pad", "padding": [], "fill": 1.5},
                "inputs": [],
                "output_type": {"dims": [], "precision": "f32"}
            }],
            "roots": [0]
        })
        .to_string();
        assert!(
            matches!(
                WireDag::from_validated_json(&current_with_legacy_fill),
                Err(WireDagDecodeError::Parse(_))
            ),
            "the current schema must not retain a raw-number alternate Pad.fill spelling"
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
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
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

        let one_hot = serde_json::to_string(&WireRiscOp::OneHot {
            vocab: NonnegativeExtent::new(7).unwrap(),
        })
        .unwrap();
        assert_eq!(one_hot, r#"{"kind":"one_hot","vocab":7}"#);
        match serde_json::from_str::<WireRiscOp>(&one_hot).unwrap() {
            WireRiscOp::OneHot { vocab } => assert_eq!(vocab.get(), 7),
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
    fn relu_wire_identities_round_trip_as_distinct_exact_variants() {
        for (op, expected) in [
            (WireRiscOp::Relu, r#"{"kind":"relu"}"#),
            (WireRiscOp::ReluAdjoint, r#"{"kind":"relu_adjoint"}"#),
        ] {
            let encoded = serde_json::to_string(&op).expect("serialize relu identity");
            assert_eq!(encoded, expected);
            let decoded =
                serde_json::from_str::<WireRiscOp>(&encoded).expect("decode relu identity");
            assert_eq!(serde_json::to_string(&decoded).unwrap(), expected);
        }

        assert!(serde_json::from_str::<WireRiscOp>(r#"{"kind":"relu_surrogate"}"#).is_err());
    }

    #[test]
    fn relu_wire_contract_rejects_wrong_arity_dtype_and_shape() {
        let ty = |precision: &str, size: i64| WireTensorType {
            dims: vec![WireDimInfo::Lit {
                size: NonnegativeExtent::new(size).unwrap(),
            }],
            precision: precision.to_string(),
        };
        let load = |id, precision: &str, size| WireDagNode {
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
            id,
            op: WireRiscOp::Load {
                name: format!("input_{id}"),
            },
            inputs: vec![],
            output_type: ty(precision, size),
        };
        let validate = |op, inputs, output_type| {
            WireDag {
                schema_version: WIRE_DAG_SCHEMA_VERSION,
                nodes: vec![
                    load(0, "f32", 4),
                    load(1, "f32", 4),
                    WireDagNode {
                        shape_deps: vec![],
                        span_id: None,
                        merged_spans: vec![],
                        id: 2,
                        op,
                        inputs,
                        output_type,
                    },
                ],
                roots: vec![2],
            }
            .validate_wire_contract()
        };

        validate(WireRiscOp::Relu, vec![0], ty("f32", 4)).expect("valid ReLU wire node");
        validate(WireRiscOp::ReluAdjoint, vec![0, 1], ty("f32", 4))
            .expect("valid ReLU adjoint wire node");
        assert!(validate(WireRiscOp::Relu, vec![0, 1], ty("f32", 4)).is_err());
        assert!(validate(WireRiscOp::Relu, vec![0], ty("int32", 4)).is_err());
        assert!(validate(WireRiscOp::ReluAdjoint, vec![0, 1], ty("f32", 3)).is_err());
        assert!(validate(WireRiscOp::ReluAdjoint, vec![0, 99], ty("f32", 4)).is_err());
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
                dims: vec![WireInferredDim::Lit {
                    size: NonnegativeExtent::new(4).unwrap(),
                }],
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
                window_shape: [2, 3]
                    .into_iter()
                    .map(|value| NonnegativeExtent::new(value).unwrap())
                    .collect(),
                strides: [1, 2]
                    .into_iter()
                    .map(|value| NonnegativeExtent::new(value).unwrap())
                    .collect(),
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
                    assert_eq!(
                        window_shape
                            .into_iter()
                            .map(NonnegativeExtent::get)
                            .collect::<Vec<_>>(),
                        vec![2, 3]
                    );
                    assert_eq!(
                        strides
                            .into_iter()
                            .map(NonnegativeExtent::get)
                            .collect::<Vec<_>>(),
                        vec![1, 2]
                    );
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
                window_shape: [2, 3]
                    .into_iter()
                    .map(|value| NonnegativeExtent::new(value).unwrap())
                    .collect(),
                strides: [1, 2]
                    .into_iter()
                    .map(|value| NonnegativeExtent::new(value).unwrap())
                    .collect(),
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
                    assert_eq!(
                        window_shape
                            .into_iter()
                            .map(NonnegativeExtent::get)
                            .collect::<Vec<_>>(),
                        vec![2, 3]
                    );
                    assert_eq!(
                        strides
                            .into_iter()
                            .map(NonnegativeExtent::get)
                            .collect::<Vec<_>>(),
                        vec![1, 2]
                    );
                }
                other => panic!("expected reduce_window_grad wire op, got {other:?}"),
            }
        }
    }
}

#[cfg(test)]
mod diagnostic_projection_contract {
    use super::{Diagnostic, DiagnosticSpan, check_error_kind, effect_error_kind};
    use chelis_effects::EffectErrorKind as E;
    use chelis_types::errors::{CheckError, CheckErrorKind as K};

    /// Every check kind, so the pin below covers the whole enum rather than
    /// the three a CLI fixture happens to provoke. Adding a variant does not
    /// compile until `check_error_kind` gains an arm, which lands the author
    /// here.
    const ALL_CHECK_KINDS: [K; 24] = [
        K::TypeMismatch,
        K::PrecisionMismatch,
        K::DimensionMismatch,
        K::ArityMismatch,
        K::UnboundVariable {
            identifier: String::new(),
        },
        K::UnknownConstructor {
            identifier: String::new(),
        },
        K::NotAFunction,
        K::NonExhaustiveMatch,
        K::OccursCheck,
        K::CastNonTensor,
        K::TupleIndexOutOfBounds,
        K::UseAfterConsume,
        K::UnconsumedLinear,
        K::InvalidBorrow,
        K::CycleDetected,
        K::UnsupportedTensorPrecision,
        K::DuplicateDefinition,
        K::DuplicateModule,
        K::OpaqueTypeViolation,
        K::ReservedLinkerName,
        K::BuiltinShadowing,
        K::UnknownForm,
        K::MalformedForm,
        K::Other,
    ];

    /// The published spellings, restated independently of the projection.
    ///
    /// Reading them back through `check_error_kind` would agree with any
    /// edit to it, including a wrong one. Two tables disagree exactly when
    /// one moved without the other, which is what an accidental wire change
    /// looks like.
    fn expected_spelling(kind: &K) -> &'static str {
        match kind {
            K::TypeMismatch => "TypeMismatch",
            K::PrecisionMismatch => "PrecisionMismatch",
            K::DimensionMismatch => "DimensionMismatch",
            K::ArityMismatch => "ArityMismatch",
            K::UnboundVariable { .. } => "UnboundVariable",
            K::UnknownConstructor { .. } => "UnknownConstructor",
            K::NotAFunction => "NotAFunction",
            K::NonExhaustiveMatch => "NonExhaustiveMatch",
            K::OccursCheck => "OccursCheck",
            K::CastNonTensor => "CastNonTensor",
            K::TupleIndexOutOfBounds => "TupleIndexOutOfBounds",
            K::UseAfterConsume => "UseAfterConsume",
            K::UnconsumedLinear => "UnconsumedLinear",
            K::InvalidBorrow => "InvalidBorrow",
            K::CycleDetected => "CycleDetected",
            K::UnsupportedTensorPrecision => "UnsupportedTensorPrecision",
            K::DuplicateDefinition => "DuplicateDefinition",
            K::DuplicateModule => "DuplicateModule",
            K::OpaqueTypeViolation => "OpaqueTypeViolation",
            K::ReservedLinkerName => "ReservedLinkerName",
            K::BuiltinShadowing => "BuiltinShadowing",
            K::UnknownForm => "UnknownForm",
            K::MalformedForm => "MalformedForm",
            K::Other => "Other",
        }
    }

    fn check_error(kind: K, span_offset: Option<usize>, span_id: Option<&str>) -> CheckError {
        CheckError {
            kind,
            message: "m".to_string(),
            suggestions: Vec::new(),
            severity: 0.5,
            expected: None,
            got: None,
            span_offset,
            span_id: span_id.map(str::to_string),
        }
    }

    /// chelis#886: every check kind reaches the wire through its governed
    /// vocabulary identity, at the spelling the report has always published.
    #[test]
    fn every_check_kind_projects_to_its_pinned_governed_spelling() {
        for kind in &ALL_CHECK_KINDS {
            assert_eq!(
                check_error_kind(kind).as_str(),
                expected_spelling(kind),
                "published spelling changed for {kind:?}"
            );
        }
    }

    /// `CheckErrorKind::diagnostic_name` (chelis#1399) is a second published
    /// spelling of the same kind: it is what `chelis-e2e`'s snippet checker
    /// still emits, and what the CLI report emitted before this change. Two
    /// spellings of one wire field is the drift chelis#886 exists to remove,
    /// and until the remaining producer is converted the honest defence is to
    /// pin them equal. If this fails, one of the two moved alone and a
    /// consumer reading both surfaces now sees two names for one kind.
    #[test]
    fn the_governed_identity_agrees_with_the_checkers_own_spelling() {
        for kind in &ALL_CHECK_KINDS {
            assert_eq!(
                check_error_kind(kind).as_str(),
                kind.diagnostic_name(),
                "the governed identity and `diagnostic_name` disagree for {kind:?}"
            );
        }
    }

    /// Two check kinds sharing one identity would silently merge them on the
    /// wire, which is information loss a consumer cannot detect.
    #[test]
    fn the_projection_does_not_collapse_two_kinds_onto_one_identity() {
        let mut identities: Vec<&str> = ALL_CHECK_KINDS
            .iter()
            .map(|kind| check_error_kind(kind).as_str())
            .collect();
        let total = identities.len();
        identities.sort_unstable();
        identities.dedup();
        assert_eq!(identities.len(), total, "two check kinds share an identity");
    }

    #[test]
    fn every_effect_kind_projects_to_its_pinned_governed_spelling() {
        for (kind, spelling) in [
            (E::UnhandledEffect, "UnhandledEffect"),
            (E::InvalidHandler, "InvalidHandler"),
            (E::BuildTargetMismatch, "BuildTargetMismatch"),
            (E::TypeTotality, "TypeTotality"),
        ] {
            assert_eq!(effect_error_kind(&kind).as_str(), spelling);
        }
    }

    /// chelis#1395: a producer that held a coordinate and no extent reports
    /// it, rather than losing it for want of a length.
    ///
    /// Before the tagged carrier this returned `None`, dropping a coordinate
    /// the previous hand-assembled document published as `span_offset` --
    /// 17 of 219 diagnostics, per the corpus measurement recorded on
    /// chelis#1395 — cited rather than restated so a reader can check it.
    #[test]
    fn a_coordinate_without_an_identity_travels_as_a_point() {
        let derived =
            Diagnostic::try_from_check_error(&check_error(K::Other, Some(30), None)).unwrap();
        assert_eq!(derived.span, Some(DiagnosticSpan::Point { offset: 30 }));
        assert_eq!(derived.span.map(|span| span.extent()), Some(None));
    }

    /// [04-FIT-17]: a degenerate `start..start` identity is a coordinate the
    /// producer never measured an extent for, not a measured empty range.
    /// The distinction is the whole point of the atom -- a consumer cannot
    /// tell an invented `len: 0` from a real one.
    #[test]
    fn a_degenerate_identity_is_a_point_not_a_zero_width_range() {
        let derived =
            Diagnostic::try_from_check_error(&check_error(K::Other, Some(77), Some("surf:77..77")))
                .unwrap();
        assert_eq!(derived.span, Some(DiagnosticSpan::Point { offset: 77 }));
    }

    /// The carrier makes the prohibition structural: `Point` has no `len`
    /// field, so a fabricated extent is unrepresentable rather than merely
    /// forbidden. This pins the property that motivates the shape.
    #[test]
    fn a_point_cannot_carry_an_extent() {
        assert_eq!(DiagnosticSpan::Point { offset: 5 }.extent(), None);
        assert_eq!(
            DiagnosticSpan::Range { offset: 5, len: 2 }.extent(),
            Some(2)
        );
        assert_eq!(DiagnosticSpan::Point { offset: 5 }.offset(), 5);
        assert_eq!(DiagnosticSpan::Range { offset: 5, len: 2 }.offset(), 5);
    }

    /// The variant tag is what earns the census's tagged-transport class, so
    /// it is part of the wire contract rather than a serde detail.
    #[test]
    fn both_variants_carry_their_tag_on_the_wire() {
        let point =
            serde_json::to_string(&DiagnosticSpan::Point { offset: 30 }).expect("point serializes");
        assert_eq!(point, r#"{"span":"point","offset":30}"#);
        let range = serde_json::to_string(&DiagnosticSpan::Range { offset: 30, len: 4 })
            .expect("range serializes");
        assert_eq!(range, r#"{"span":"range","offset":30,"len":4}"#);
    }

    /// spec/04 [04-FIT-17] and `spec/03-deep-syntax.md` §1.1.1: a check
    /// diagnostic reports the coordinate it holds and never derives an extent
    /// from the identity beside it.
    ///
    /// `CheckError` carries `span_offset` and an opaque `span_id`, and no
    /// measured length. An earlier revision recovered one by parsing `N..M`
    /// out of the identity, which invents an extent for any identity that
    /// happens to be spelled that way -- including a foreign one, since §1.1.1
    /// makes external span IDs opaque and their interpretation none of
    /// Chelis's concern.
    #[test]
    fn a_check_diagnostic_reports_a_coordinate_never_a_derived_range() {
        // A producer with no location at all still reports nothing. `Point`
        // exists for a coordinate the producer HELD, never for one it lacked.
        assert!(
            Diagnostic::try_from_check_error(&check_error(K::Other, None, None))
                .unwrap()
                .span
                .is_none(),
            "no location at all must stay absent"
        );

        for (label, error) in [
            ("no identity at all", check_error(K::Other, Some(30), None)),
            (
                "an opaque identity carrying no range",
                check_error(K::Other, Some(30), Some("octant:theorem-7")),
            ),
            // The identity is Chelis's own spelling and still yields no
            // extent: reading one back out of a string is not a typed
            // producer path, so `surf:` earns no more trust here than
            // `octant:` does.
            (
                "a range-shaped identity with Chelis's own prefix",
                check_error(
                    K::UnboundVariable {
                        identifier: "x".to_string(),
                    },
                    Some(30),
                    Some("surf:30..34"),
                ),
            ),
            // The adversarial case: an EXTERNAL identity that merely looks
            // like a range. Spelling is not provenance, and a consumer must
            // not receive `len: 4` that no producer measured.
            (
                "a range-shaped identity from a foreign producer",
                check_error(K::Other, Some(30), Some("octant:30..34")),
            ),
            // A degenerate identity is a coordinate, not a measured empty
            // range; a consumer cannot tell an invented `len: 0` from a real
            // one.
            (
                "a degenerate range-shaped identity",
                check_error(K::Other, Some(30), Some("surf:30..30")),
            ),
            // The identity's range disagrees with the producer's own offset.
            // `span_offset` is the first-class coordinate; the identity still
            // travels in `span_id`, so a consumer can see the disagreement
            // rather than having it hidden.
            (
                "an identity whose range contradicts the offset",
                check_error(K::Other, Some(30), Some("surf:99..104")),
            ),
        ] {
            let span = Diagnostic::try_from_check_error(&error).unwrap().span;
            assert_eq!(
                span,
                Some(DiagnosticSpan::Point { offset: 30 }),
                "{label} must report the coordinate and no range"
            );
            assert_eq!(
                span.map(|span| span.extent()),
                Some(None),
                "{label} must carry no extent"
            );
        }
    }

    /// The identity survives even when the range does not, so omitting the
    /// range loses nothing a consumer previously had.
    #[test]
    fn an_opaque_identity_travels_without_a_range() {
        let projected = Diagnostic::try_from_check_error(&check_error(
            K::Other,
            Some(30),
            Some("octant:theorem-7"),
        ))
        .unwrap();
        assert_eq!(projected.span_id.as_deref(), Some("octant:theorem-7"));
        // Both travel: [04-FIT-16] requires the coordinate and the identity
        // to be independently carryable, so an opaque identity that yields no
        // range does not suppress the coordinate beside it.
        assert_eq!(projected.span, Some(DiagnosticSpan::Point { offset: 30 }));
    }
}
