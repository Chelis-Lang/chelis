//! thiserror's `AsDisplay` (Aeneas.Std Thiserror.lean). `#[derive(thiserror::Error)]`
//! formats each interpolated field `{field}` as `field.as_display()`, which for a field
//! `&T` is `AsDisplay for &T`: the identity on the borrowed field. This fixture compiles
//! under the crate name `thiserror` with a verbatim copy of thiserror 2.0.18's
//! `src/display.rs` (without the `std` feature's `Path` impls), opaque as the dependency
//! is in production, so the item keeps its production path. The `Display` impls below
//! are thiserror-impl 2.0.18's expansion of the derives of chelis-ir
//! `ownership::error::OwnershipError` and `capacity_key::CapacityKeyBuildError` (a
//! subset of their variants).

pub mod display {
    use core::fmt::Display;

    #[doc(hidden)]
    pub trait AsDisplay<'a>: Sealed {
        // TODO: convert to generic associated type.
        // https://github.com/dtolnay/thiserror/pull/253
        type Target: Display;

        fn as_display(&'a self) -> Self::Target;
    }

    impl<'a, T> AsDisplay<'a> for &T
    where
        T: Display + ?Sized + 'a,
    {
        type Target = &'a T;

        fn as_display(&'a self) -> Self::Target {
            *self
        }
    }

    #[doc(hidden)]
    pub trait Sealed {}
    impl<T: Display + ?Sized> Sealed for &T {}
}

/// `#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]`, with
/// `#[error("exact capacity quotient has a statically zero divisor")] StaticZeroDivisor`,
/// `#[error("verified capacity source is malformed: {0}")] MalformedSource(String)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapacityKeyBuildError {
    StaticZeroDivisor,
    MalformedSource(String),
}

#[allow(unused_qualifications)]
#[automatically_derived]
impl ::core::fmt::Display for CapacityKeyBuildError {
    fn fmt(&self, __formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        #[allow(unused_imports)]
        use crate::display::AsDisplay as _;
        #[allow(unused_variables, deprecated, clippy::used_underscore_binding)]
        match self {
            CapacityKeyBuildError::StaticZeroDivisor {} => {
                __formatter.write_str("exact capacity quotient has a statically zero divisor")
            }
            CapacityKeyBuildError::MalformedSource(_0) => match (_0.as_display(),) {
                (__display0,) => ::core::write!(
                    __formatter,
                    "verified capacity source is malformed: {__display0}"
                ),
            },
        }
    }
}

/// `#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]`, with
/// `#[error("ownership lowering has no checked signature for `{function}`")]`,
/// `#[error("call to `{callee}` in `{unit}` supplies {supplied} arguments, expected {declared}")]`,
/// a field of a local error type, `#[error("capacity key in `{unit}`: {source}")]`, and a
/// boxed error of the same type, `#[error("in `{unit}`: {inner}")]` (its `Display` is
/// recursive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnershipError {
    MissingSignature { function: String },
    CallArityMismatch { unit: String, callee: String, supplied: usize, declared: usize },
    Capacity { unit: String, source: CapacityKeyBuildError },
    Nested { unit: String, inner: Box<OwnershipError> },
}

#[allow(unused_qualifications)]
#[automatically_derived]
impl ::core::fmt::Display for OwnershipError {
    fn fmt(&self, __formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        #[allow(unused_imports)]
        use crate::display::AsDisplay as _;
        #[allow(unused_variables, deprecated, clippy::used_underscore_binding)]
        match self {
            OwnershipError::MissingSignature { function } => match (function.as_display(),) {
                (__display_function,) => ::core::write!(
                    __formatter,
                    "ownership lowering has no checked signature for `{__display_function}`"
                ),
            },
            OwnershipError::CallArityMismatch { unit, callee, supplied, declared } => match (
                callee.as_display(),
                unit.as_display(),
                supplied.as_display(),
                declared.as_display(),
            ) {
                (__display_callee, __display_unit, __display_supplied, __display_declared) => {
                    ::core::write!(
                        __formatter,
                        "call to `{__display_callee}` in `{__display_unit}` supplies {__display_supplied} arguments, expected {__display_declared}"
                    )
                }
            },
            OwnershipError::Capacity { unit, source } => {
                match (unit.as_display(), source.as_display()) {
                    (__display_unit, __display_source) => ::core::write!(
                        __formatter,
                        "capacity key in `{__display_unit}`: {__display_source}"
                    ),
                }
            }
            OwnershipError::Nested { unit, inner } => {
                match (unit.as_display(), inner.as_display()) {
                    (__display_unit, __display_inner) => ::core::write!(
                        __formatter,
                        "in `{__display_unit}`: {__display_inner}"
                    ),
                }
            }
        }
    }
}

/// As the production callers: the error's message.
pub fn message(error: &OwnershipError) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolated_fields_are_displayed() {
        let missing = OwnershipError::MissingSignature { function: "main".to_string() };
        assert_eq!(message(&missing), "ownership lowering has no checked signature for `main`");
        let arity = OwnershipError::CallArityMismatch {
            unit: "u".to_string(),
            callee: "f".to_string(),
            supplied: 3,
            declared: 12,
        };
        assert_eq!(message(&arity), "call to `f` in `u` supplies 3 arguments, expected 12");
        let zero = OwnershipError::Capacity {
            unit: "k".to_string(),
            source: CapacityKeyBuildError::StaticZeroDivisor,
        };
        assert_eq!(
            message(&zero),
            "capacity key in `k`: exact capacity quotient has a statically zero divisor"
        );
        let malformed = OwnershipError::Capacity {
            unit: String::new(),
            source: CapacityKeyBuildError::MalformedSource("x/0".to_string()),
        };
        assert_eq!(
            message(&malformed),
            "capacity key in ``: verified capacity source is malformed: x/0"
        );
        let nested = OwnershipError::Nested {
            unit: "outer".to_string(),
            inner: Box::new(OwnershipError::Nested { unit: "inner".to_string(), inner: Box::new(zero) }),
        };
        assert_eq!(
            message(&nested),
            "in `outer`: in `inner`: capacity key in `k`: exact capacity quotient has a statically zero divisor"
        );
    }
}
