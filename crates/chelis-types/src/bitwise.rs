//! Exact signed-width bitwise kernels, shared by host and tensor execution.
//! [05-OP-47] owns every identity in this closed family.

use crate::types::Prim;
use crate::{NumericKernelError, ScalarValue, TensorStorage, scalar_from_i64, tensor_from_scalars};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Debug, Clone, PartialEq)]
pub enum BitwiseError {
    Signature(NumericKernelError),
    NegativeShift(ScalarValue),
}

impl std::fmt::Display for BitwiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Signature(error) => error.fmt(f),
            Self::NegativeShift(amount) => write!(
                f,
                "shift amount must be non-negative, got {}",
                amount.as_i64_exact().expect("integer count")
            ),
        }
    }
}

impl std::error::Error for BitwiseError {}

fn signature(op: BitwiseKind, lhs: Prim, rhs: Prim) -> Result<(), BitwiseError> {
    for actual in [lhs, rhs] {
        if !actual.is_integer() {
            return Err(BitwiseError::Signature(NumericKernelError::WrongFamily {
                op: op.name(),
                expected: crate::NumericFamily::Int,
                actual,
            }));
        }
    }
    if lhs != rhs {
        return Err(BitwiseError::Signature(NumericKernelError::DtypeMismatch {
            op: op.name(),
            lhs,
            rhs,
        }));
    }
    Ok(())
}

/// Apply [05-OP-47] at the operands' exact declared width.
pub fn bitwise_scalar(
    op: BitwiseKind,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, BitwiseError> {
    signature(op, lhs.prim(), rhs.prim())?;
    let a = lhs.as_i64_exact().expect("integer lhs");
    let b = rhs.as_i64_exact().expect("integer rhs");
    let width = match lhs.prim() {
        Prim::Int8 => 8,
        Prim::Int16 => 16,
        Prim::Int32 => 32,
        Prim::Int64 => 64,
        _ => unreachable!("signature requires integers"),
    };
    if op.is_shift() && b < 0 {
        return Err(BitwiseError::NegativeShift(rhs));
    }
    let raw = match op {
        BitwiseKind::And => a & b,
        BitwiseKind::Or => a | b,
        BitwiseKind::Xor => a ^ b,
        BitwiseKind::ShiftLeft if b >= width => 0,
        BitwiseKind::ShiftRight if b >= width => {
            if a < 0 {
                -1
            } else {
                0
            }
        }
        BitwiseKind::ShiftLeft => a.wrapping_shl(b as u32),
        BitwiseKind::ShiftRight => a >> b,
    };
    let wrapped = match lhs.prim() {
        Prim::Int8 => i64::from(raw as i8),
        Prim::Int16 => i64::from(raw as i16),
        Prim::Int32 => i64::from(raw as i32),
        Prim::Int64 => raw,
        _ => unreachable!("signature requires integers"),
    };
    Ok(scalar_from_i64(op.name(), lhs.prim(), wrapped).expect("width-bounded bit pattern"))
}

/// Apply the same scalar semantics without passing integer values through a float.
pub fn bitwise_tensor(
    op: BitwiseKind,
    lhs: &TensorStorage,
    rhs: &TensorStorage,
) -> Result<TensorStorage, BitwiseError> {
    signature(op, lhs.prim(), rhs.prim())?;
    if lhs.len() != rhs.len() {
        return Err(BitwiseError::Signature(
            NumericKernelError::LengthMismatch {
                op: op.name(),
                lhs: lhs.len(),
                rhs: rhs.len(),
            },
        ));
    }
    let values = (0..lhs.len())
        .map(|i| bitwise_scalar(op, lhs.scalar_at(i), rhs.scalar_at(i)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(tensor_from_scalars(lhs.prim(), &values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_signed_width_patterns_and_boundary_counts() {
        for (prim, width, minimum) in [
            (Prim::Int8, 8, -128),
            (Prim::Int16, 16, -32768),
            (Prim::Int32, 32, -2147483648),
            (Prim::Int64, 64, i64::MIN),
        ] {
            for (op, a, b, expected) in [
                (BitwiseKind::And, -1, 6, 6),
                (BitwiseKind::Or, -8, 3, -5),
                (BitwiseKind::Xor, -1, 6, -7),
                (BitwiseKind::ShiftLeft, 1, width - 1, minimum),
                (BitwiseKind::ShiftLeft, -1, 1, -2),
                (BitwiseKind::ShiftLeft, -1, width, 0),
                (BitwiseKind::ShiftRight, -16, 2, -4),
                (BitwiseKind::ShiftRight, -1, width, -1),
                (BitwiseKind::ShiftRight, 1, width + 1, 0),
            ] {
                let lhs = scalar_from_i64("test", prim, a).unwrap();
                let rhs = scalar_from_i64("test", prim, b).unwrap();
                let actual = bitwise_scalar(op, lhs, rhs).unwrap();
                assert_eq!(actual.prim(), prim);
                assert_eq!(actual.as_i64_exact(), Some(expected));
                let actual = bitwise_tensor(
                    op,
                    &tensor_from_scalars(prim, &[lhs]),
                    &tensor_from_scalars(prim, &[rhs]),
                )
                .unwrap();
                assert_eq!(actual.scalar_at(0).as_i64_exact(), Some(expected));
            }
        }
        let lhs = scalar_from_i64("test", Prim::Int64, 9007199254740992).unwrap();
        let rhs = scalar_from_i64("test", Prim::Int64, 1).unwrap();
        assert_eq!(
            bitwise_scalar(BitwiseKind::Or, lhs, rhs)
                .unwrap()
                .as_i64_exact(),
            Some(9007199254740993)
        );
    }

    #[test]
    fn negative_counts_and_nonmatching_domains_are_errors() {
        for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
            let one = scalar_from_i64("test", prim, 1).unwrap();
            let negative = scalar_from_i64("test", prim, -1).unwrap();
            for op in [BitwiseKind::ShiftLeft, BitwiseKind::ShiftRight] {
                assert_eq!(
                    bitwise_scalar(op, one, negative),
                    Err(BitwiseError::NegativeShift(negative))
                );
                assert_eq!(
                    bitwise_tensor(
                        op,
                        &tensor_from_scalars(prim, &[one]),
                        &tensor_from_scalars(prim, &[negative])
                    )
                    .unwrap_err()
                    .to_string(),
                    "shift amount must be non-negative, got -1"
                );
            }
            let empty = tensor_from_scalars(prim, &[]);
            assert_eq!(
                bitwise_tensor(BitwiseKind::And, &empty, &empty)
                    .unwrap()
                    .len(),
                0
            );
            assert!(
                bitwise_tensor(BitwiseKind::And, &empty, &tensor_from_scalars(prim, &[one]))
                    .is_err()
            );
        }
        let one = scalar_from_i64("test", Prim::Int32, 1).unwrap();
        for other in [
            scalar_from_i64("test", Prim::Int64, 1).unwrap(),
            scalar_from_i64("test", Prim::Bool, 1).unwrap(),
            crate::scalar_from_f64("test", Prim::F32, 1.0).unwrap(),
        ] {
            for op in [
                BitwiseKind::And,
                BitwiseKind::Or,
                BitwiseKind::Xor,
                BitwiseKind::ShiftLeft,
                BitwiseKind::ShiftRight,
            ] {
                assert!(matches!(
                    bitwise_scalar(op, one, other),
                    Err(BitwiseError::Signature(_))
                ));
            }
        }
    }
}
