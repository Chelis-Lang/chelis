//! Shared numeric failure vocabulary and allocation-free line format.
//!
//! Each execution lane uses the [04-NUM-9] trap format.
//! Context belongs on separate lines before the final trap line.

use core::fmt;

/// The closed set of [04-NUM-9] numeric trap kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericTrapKind {
    /// An integer result outside the declared dtype range.
    Overflow,
    /// An input or result outside the operation or dtype domain.
    Domain,
    /// Integer division or remainder by a zero divisor.
    DivZero,
}

impl NumericTrapKind {
    pub const ALL: [Self; 3] = [Self::Overflow, Self::Domain, Self::DivZero];

    /// The frozen spelling inside a trap line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Overflow => "overflow",
            Self::Domain => "domain",
            Self::DivZero => "division by zero",
        }
    }
}

/// One [04-NUM-9] line: `numeric trap: <kind> in <op> at <dtype>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericTrapLine<'a> {
    pub kind: NumericTrapKind,
    /// The canonical name of the primitive that raised the trap.
    pub op: &'a str,
    /// The canonical name of the result dtype.
    pub dtype: &'a str,
}

impl NumericTrapLine<'_> {
    /// The frozen prefix of every numeric trap line.
    pub const PREFIX: &'static str = "numeric trap: ";
    /// The frozen separator before the operation name.
    pub const OPERATION_SEPARATOR: &'static str = " in ";
    /// The frozen separator before the dtype name.
    pub const DTYPE_SEPARATOR: &'static str = " at ";
}

impl fmt::Display for NumericTrapLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}{}{}{}{}",
            Self::PREFIX,
            self.kind.as_str(),
            Self::OPERATION_SEPARATOR,
            self.op,
            Self::DTYPE_SEPARATOR,
            self.dtype
        )
    }
}
