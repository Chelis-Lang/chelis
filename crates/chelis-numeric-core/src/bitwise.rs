//! Exact signed-width bitwise operation identities for [05-OP-47].

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum BitwiseKind {
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
}

impl BitwiseKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::And => "bitand",
            Self::Or => "bitor",
            Self::Xor => "bitxor",
            Self::ShiftLeft => "shl",
            Self::ShiftRight => "shr",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "bitand" => Some(Self::And),
            "bitor" => Some(Self::Or),
            "bitxor" => Some(Self::Xor),
            "shl" => Some(Self::ShiftLeft),
            "shr" => Some(Self::ShiftRight),
            _ => None,
        }
    }

    pub const fn is_shift(self) -> bool {
        matches!(self, Self::ShiftLeft | Self::ShiftRight)
    }
}
