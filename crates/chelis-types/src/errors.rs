//! Error types for the Chelis type checker.

use crate::unify::{TypeError, TypeErrorKind};

#[derive(Debug, Clone)]
pub struct CheckError {
    pub kind: CheckErrorKind,
    pub message: String,
    pub suggestions: Vec<String>,
    pub severity: f64,
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
        }
    }
}
