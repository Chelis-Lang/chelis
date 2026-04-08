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
    InvalidBorrow,
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
            CheckErrorKind::InvalidBorrow => 0.8,
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
