//! Error types for the Chelis type checker.

use crate::unify::{TypeError, TypeErrorKind};

#[derive(Debug, Clone)]
pub struct CheckError {
    pub kind: CheckErrorKind,
    pub message: String,
    pub suggestions: Vec<String>,
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
        CheckError {
            kind,
            message: te.message,
            suggestions: vec![],
        }
    }
}
