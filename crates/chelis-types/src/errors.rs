//! Error types for the Chelis type checker.

use crate::unify::{TypeError, TypeErrorKind};

#[derive(Debug, Clone)]
pub struct CheckError {
    pub kind: CheckErrorKind,
    pub message: String,
    pub suggestions: Vec<String>,
    pub severity: f64,
    /// Expected type (for structured error reports).
    pub expected: Option<String>,
    /// Actual type found (for structured error reports).
    pub got: Option<String>,
}

#[derive(Debug, Clone)]
pub enum CheckErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    UnboundVariable,
    NotAFunction,
    NonExhaustiveMatch,
    OccursCheck,
    CastNonTensor,
    TupleIndexOutOfBounds,
    UseAfterConsume,
    UnconsumedLinear,
    InvalidBorrow,
    CycleDetected,
    /// A tensor type uses a precision the Phase 0f backend cannot represent
    /// (currently f16, bf16, f64, f8e4m3). Host scalar precisions are unaffected.
    UnsupportedTensorPrecision,
    /// A `deftype` or `typealias` reuses a name already bound by an
    /// earlier `deftype`, `typealias`, or prelude ADT in the same
    /// program. Type names share one flat namespace in `AdtRegistry`
    /// (keyed on bare strings), so silent last-write-wins would let a
    /// later declaration overwrite the original — propagating wrong
    /// constructor types, wrong variant sets, and (per
    /// `compute_tensor_carrying_adts` in `linearity.rs`) order-
    /// dependent borrow semantics. Reject the collision at declaration
    /// time instead.
    DuplicateDefinition,
    /// RFC v4b (RT-1 F2): a named module is opened by more than one
    /// `(module ...)` wrapper in the same check unit. Module identity
    /// is otherwise a forgeable string -- a second wrapper of an
    /// opaque type's defining module would construct and inspect the
    /// type as if it were inside. A named module may be opened at most
    /// once per check unit (the same ambiguity rationale as the
    /// named-module requirement for `@opaque`).
    DuplicateModule,
    /// RFC D-CHECK (spec/design/opaque_invariants_rfc.md): a type
    /// declared `@opaque` was constructed, inspected, forged, or
    /// reached through an unexported binding outside its defining
    /// module, or `@opaque` was declared outside a named module. The
    /// message names the type, the defining module, and the exported
    /// producers with signatures; location context is the enclosing
    /// def name embedded in the message.
    OpaqueTypeViolation,
    /// RFC v5 (RT-1 F2 bypass): a top-level declaration's binding name
    /// matches the reef package-linker's reserved internal-name format
    /// (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) in a program
    /// NOT produced by the linker. That format is the linker's private
    /// output; hand-authoring it forges module identity through the
    /// reef-stem channel (a flat program of mangled names self-keys to
    /// one module and constructs/inspects opaque types as in-module).
    ReservedLinkerName,
    Other,
}

impl CheckErrorKind {
    pub fn default_severity(&self) -> f64 {
        match self {
            CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch => 0.8,
            CheckErrorKind::ArityMismatch => 0.7,
            CheckErrorKind::UnboundVariable => 0.6,
            CheckErrorKind::NotAFunction => 0.5,
            CheckErrorKind::TypeMismatch => 0.5,
            CheckErrorKind::NonExhaustiveMatch => 0.7,
            CheckErrorKind::OccursCheck => 0.5,
            CheckErrorKind::CastNonTensor => 0.6,
            CheckErrorKind::TupleIndexOutOfBounds => 0.7,
            CheckErrorKind::UseAfterConsume => 0.9,
            CheckErrorKind::UnconsumedLinear => 0.9,
            CheckErrorKind::InvalidBorrow => 0.8,
            CheckErrorKind::CycleDetected => 0.9,
            CheckErrorKind::UnsupportedTensorPrecision => 0.8,
            CheckErrorKind::DuplicateDefinition => 0.9,
            CheckErrorKind::DuplicateModule => 0.9,
            CheckErrorKind::OpaqueTypeViolation => 0.8,
            CheckErrorKind::ReservedLinkerName => 0.9,
            CheckErrorKind::Other => 0.5,
        }
    }
}

impl CheckError {
    /// Create a new CheckError with default severity for its kind.
    pub fn new(kind: CheckErrorKind, message: String, suggestions: Vec<String>) -> Self {
        let severity = kind.default_severity();
        CheckError {
            kind,
            message,
            suggestions,
            severity,
            expected: None,
            got: None,
        }
    }

    /// Create with expected/got for structured error reports.
    pub fn with_types(
        kind: CheckErrorKind,
        message: String,
        expected: String,
        got: String,
        suggestions: Vec<String>,
    ) -> Self {
        let severity = kind.default_severity();
        CheckError {
            kind,
            message,
            suggestions,
            severity,
            expected: Some(expected),
            got: Some(got),
        }
    }
}

impl From<TypeError> for CheckError {
    fn from(te: TypeError) -> Self {
        let kind = match te.kind {
            TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
            TypeErrorKind::PrecisionMismatch => CheckErrorKind::PrecisionMismatch,
            TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
            TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
            TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
            TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
        };
        let severity = match &kind {
            CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch => 0.8,
            CheckErrorKind::ArityMismatch => 0.7,
            CheckErrorKind::UnboundVariable => 0.6,
            _ => 0.5,
        };
        let suggestions = match &kind {
            CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
            _ => vec![],
        };
        CheckError {
            kind,
            message: te.message,
            suggestions,
            severity,
            expected: None,
            got: None,
        }
    }
}
