//! THE dtype semantics layer (chelis#729 Phases 1-2).
//!
//! Single source of truth for what a dtype MEANS: value set, finalize
//! (rounding / width / domain), and the trap contract. Implements
//! `spec/design/dtype_semantics.md` Part I (the normative contracts
//! sections C1, C2, and C3) and the ratified `spec/04-type-system.md`
//! section 9 atoms [04-NUM-1..6].
//!
//! The mechanism is privacy, not convention: [`ScalarValue`] and
//! [`TensorStorage`] have module-private representations, so producing a
//! numeric value without passing through the typed kernels or
//! [`finalize_scalar`] / [`finalize_tensor`] (or the ingress constructors
//! [`scalar_from_i64`] / [`scalar_from_f64`]) is a compile error, never a
//! review finding. Reads
//! are free-form (section C3: `as_f64_lossy` is explicitly named lossy,
//! `as_i64_exact` is checked); only CONSTRUCTION is gated. The one
//! deliberate hole is the `reuse_*` family below, whose doc contract is
//! "element-preserving ops only"; every use site cites it.
//!
//! ## Import discipline (decided, open question 4)
//!
//! This module stays import-clean with respect to the checker machinery:
//! it uses only [`Prim`] from `types`, [`ElementRef`] from the sibling
//! `observation` module (the two modules lift to the same future leaf
//! crate together; `faithful_observation.md` section C3.1 places the
//! formatter "beside the dtype definitions"), the `half` crate, and std.
//! No env / infer / errors imports, so a later crate lift stays
//! mechanical.
//!
//! ## Landed-signature notes (recorded in `dtype_semantics.md`)
//!
//! * Every constructor takes a leading `op: &'static str` so the section
//!   C2 trap message can name the operation without a side channel; the
//!   pre-implementation sketch omitted it.
//! * `finalize_tensor` returns [`TensorStorage`] (shape stays with the
//!   evaluator's `TensorValue`, which wraps this storage).
//! * `F16`/`Bf16` buffers store `half::f16` / `half::bf16` (both
//!   `repr(transparent)` over `u16`, the sketch's spelling).
//!
//! ## Trap-string status
//!
//! The message SHAPE below (`numeric trap: <kind> in <op> at <prim>`) is
//! the section C2 contract; the exact strings freeze at Phase 2 exit
//! together with the chelis#687 corpus. This first Phase 2 slice routes the
//! host runtime's basic arithmetic through the trap type. The remaining
//! cross-lane corpus and composed-operation naming decision freeze later in
//! the phase.

use crate::observation::ElementRef;
use crate::types::Prim;

/// The one numeric error type, identical in every lane
/// (`spec/design/dtype_semantics.md` section C2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericTrap {
    /// Integer result outside the dtype's range.
    Overflow { op: &'static str, prim: Prim },
    /// Value not a member of the dtype's set (fractional or non-finite
    /// into an integer width, non-0/1 into bool).
    Domain { op: &'static str, prim: Prim },
    /// Division/remainder by zero.
    DivZero { op: &'static str, prim: Prim },
}

impl std::fmt::Display for NumericTrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericTrap::Overflow { op, prim } => {
                write!(f, "numeric trap: overflow in {op} at {}", prim.name())
            }
            NumericTrap::Domain { op, prim } => {
                write!(f, "numeric trap: domain in {op} at {}", prim.name())
            }
            NumericTrap::DivZero { op, prim } => {
                write!(
                    f,
                    "numeric trap: division by zero in {op} at {}",
                    prim.name()
                )
            }
        }
    }
}

impl std::error::Error for NumericTrap {}

/// What kernels produce: the wide intermediate of section C1 (f64 for the
/// float family, i64 for the integer family and bool).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RawScalar {
    Int(i64),
    Float(f64),
}

/// Bulk form of [`RawScalar`] for tensor finalize.
#[derive(Debug, Clone, PartialEq)]
pub enum RawTensor {
    Int(Vec<i64>),
    Float(Vec<f64>),
}

/// Sealed per-dtype scalar bits. Private on purpose: the variant IS the
/// dtype, so a dtype/bits mismatch is unrepresentable, and construction
/// outside this module is a compile error (the section C3 privacy
/// contract).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Bits {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(half::f16),
    Bf16(half::bf16),
    F32(f32),
    F64(f64),
    Bool(bool),
}

/// A finalized scalar at its dtype's own width. Construct via
/// [`finalize_scalar`] / [`scalar_from_i64`] / [`scalar_from_f64`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarValue {
    bits: Bits,
}

impl ScalarValue {
    /// The dtype this value carries, from the storage variant itself.
    pub fn prim(&self) -> Prim {
        match self.bits {
            Bits::I8(_) => Prim::Int8,
            Bits::I16(_) => Prim::Int16,
            Bits::I32(_) => Prim::Int32,
            Bits::I64(_) => Prim::Int64,
            Bits::F16(_) => Prim::F16,
            Bits::Bf16(_) => Prim::Bf16,
            Bits::F32(_) => Prim::F32,
            Bits::F64(_) => Prim::F64,
            Bits::Bool(_) => Prim::Bool,
        }
    }

    /// Widen to f64. Exact for every float width, bool, and integers up
    /// to 2^53; EXPLICITLY LOSSY for int64 magnitudes above 2^53 (the
    /// section C3 read-side contract names the loss instead of hiding it).
    pub fn as_f64_lossy(&self) -> f64 {
        match self.bits {
            Bits::I8(v) => v as f64,
            Bits::I16(v) => v as f64,
            Bits::I32(v) => v as f64,
            Bits::I64(v) => v as f64,
            Bits::F16(v) => f64::from(v),
            Bits::Bf16(v) => f64::from(v),
            Bits::F32(v) => v as f64,
            Bits::F64(v) => v,
            Bits::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The exact i64 of an integer-family value (bool reads 0/1); `None`
    /// for floats, which have no exact integer reading in general.
    pub fn as_i64_exact(&self) -> Option<i64> {
        match self.bits {
            Bits::I8(v) => Some(v as i64),
            Bits::I16(v) => Some(v as i64),
            Bits::I32(v) => Some(v as i64),
            Bits::I64(v) => Some(v),
            Bits::Bool(v) => Some(if v { 1 } else { 0 }),
            Bits::F16(_) | Bits::Bf16(_) | Bits::F32(_) | Bits::F64(_) => None,
        }
    }

    /// The bool payload, or `None` for numeric dtypes.
    pub fn as_bool_exact(&self) -> Option<bool> {
        match self.bits {
            Bits::Bool(v) => Some(v),
            Bits::I8(_)
            | Bits::I16(_)
            | Bits::I32(_)
            | Bits::I64(_)
            | Bits::F16(_)
            | Bits::Bf16(_)
            | Bits::F32(_)
            | Bits::F64(_) => None,
        }
    }

    /// The observation-channel view of this value, at its own width, for
    /// `format_element` (section C4; the formatter is the only exit).
    pub fn element_ref(&self) -> ElementRef {
        match self.bits {
            Bits::I8(v) => ElementRef::I8(v),
            Bits::I16(v) => ElementRef::I16(v),
            Bits::I32(v) => ElementRef::I32(v),
            Bits::I64(v) => ElementRef::I64(v),
            Bits::F16(v) => ElementRef::F16(v),
            Bits::Bf16(v) => ElementRef::Bf16(v),
            Bits::F32(v) => ElementRef::F32(v),
            Bits::F64(v) => ElementRef::F64(v),
            Bits::Bool(v) => ElementRef::Bool(v),
        }
    }
}

/// Closed operation set for integer binary kernels. A consumer must map
/// its callable identity to one of these variants before arithmetic can
/// begin; there is no string or closure fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBinOp {
    Add,
    Sub,
    Mul,
    FloorDiv,
    TruncDiv,
    Rem,
    Max,
    Min,
}

impl IntBinOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::FloorDiv => "floor_div",
            Self::TruncDiv => "trunc_div",
            Self::Rem => "mod",
            Self::Max => "max_elem",
            Self::Min => "min_elem",
        }
    }
}

/// Closed operation set for integer unary kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntUnOp {
    Neg,
    Abs,
}

impl IntUnOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Abs => "abs",
        }
    }
}

/// Closed operation set for float binary kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Max,
    Min,
}

impl FloatBinOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::FloorDiv => "floor_div",
            Self::Max => "max_elem",
            Self::Min => "min_elem",
        }
    }
}

/// Closed operation set for float unary kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatUnOp {
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
    Relu,
    Sigmoid,
    Tanh,
    Silu,
    Gelu,
}

impl FloatUnOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Recip => "recip",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Sqrt => "sqrt",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Atan => "atan",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
        }
    }
}

/// Closed comparison operation set for finalized scalar values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Lte,
    Gte,
}

impl CompareOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "neq",
            Self::Lt => "cmplt",
            Self::Gt => "cmpgt",
            Self::Lte => "cmplte",
            Self::Gte => "cmpgte",
        }
    }
}

/// Arithmetic family required by a closed kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericFamily {
    Int,
    Float,
}

/// A kernel boundary failure. Arithmetic traps remain the single public
/// [`NumericTrap`] value; family and dtype mismatches indicate a consumer
/// violated its checker-proven call contract before arithmetic began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericKernelError {
    Trap(NumericTrap),
    WrongFamily {
        op: &'static str,
        expected: NumericFamily,
        actual: Prim,
    },
    DtypeMismatch {
        op: &'static str,
        lhs: Prim,
        rhs: Prim,
    },
    LengthMismatch {
        op: &'static str,
        lhs: usize,
        rhs: usize,
    },
}

impl From<NumericTrap> for NumericKernelError {
    fn from(value: NumericTrap) -> Self {
        Self::Trap(value)
    }
}

impl std::fmt::Display for NumericKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trap(trap) => trap.fmt(f),
            Self::WrongFamily {
                op,
                expected,
                actual,
            } => write!(
                f,
                "numeric kernel {op} expects {expected:?} operands, got {}",
                actual.name()
            ),
            Self::DtypeMismatch { op, lhs, rhs } => write!(
                f,
                "numeric kernel {op} expects matching dtypes, got {} and {}",
                lhs.name(),
                rhs.name()
            ),
            Self::LengthMismatch { op, lhs, rhs } => write!(
                f,
                "numeric kernel {op} expects matching buffer lengths, got {lhs} and {rhs}"
            ),
        }
    }
}

impl std::error::Error for NumericKernelError {}

fn require_family(
    op: &'static str,
    value: ScalarValue,
    expected: NumericFamily,
) -> Result<(), NumericKernelError> {
    let matches = match expected {
        NumericFamily::Int => value.prim().is_integer(),
        NumericFamily::Float => value.prim().is_float(),
    };
    if matches {
        Ok(())
    } else {
        Err(NumericKernelError::WrongFamily {
            op,
            expected,
            actual: value.prim(),
        })
    }
}

fn require_same_dtype(
    op: &'static str,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<(), NumericKernelError> {
    if lhs.prim() == rhs.prim() {
        Ok(())
    } else {
        Err(NumericKernelError::DtypeMismatch {
            op,
            lhs: lhs.prim(),
            rhs: rhs.prim(),
        })
    }
}

macro_rules! int_binop_at_width {
    ($op:expr, $prim:expr, $lhs:expr, $rhs:expr, $variant:ident) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let value = match $op {
            IntBinOp::Add => lhs.checked_add(rhs),
            IntBinOp::Sub => lhs.checked_sub(rhs),
            IntBinOp::Mul => lhs.checked_mul(rhs),
            IntBinOp::FloorDiv => {
                if rhs == 0 {
                    return Err(NumericTrap::DivZero {
                        op: $op.name(),
                        prim: $prim,
                    }
                    .into());
                }
                let quotient = lhs.checked_div(rhs).ok_or(NumericTrap::Overflow {
                    op: $op.name(),
                    prim: $prim,
                })?;
                let remainder = lhs % rhs;
                if remainder != 0 && (remainder < 0) != (rhs < 0) {
                    quotient.checked_sub(1)
                } else {
                    Some(quotient)
                }
            }
            IntBinOp::TruncDiv => {
                if rhs == 0 {
                    return Err(NumericTrap::DivZero {
                        op: $op.name(),
                        prim: $prim,
                    }
                    .into());
                }
                lhs.checked_div(rhs)
            }
            IntBinOp::Rem => {
                if rhs == 0 {
                    return Err(NumericTrap::DivZero {
                        op: $op.name(),
                        prim: $prim,
                    }
                    .into());
                }
                Some(if rhs == -1 { 0 } else { lhs % rhs })
            }
            IntBinOp::Max => Some(lhs.max(rhs)),
            IntBinOp::Min => Some(lhs.min(rhs)),
        }
        .ok_or(NumericTrap::Overflow {
            op: $op.name(),
            prim: $prim,
        })?;
        Ok(ScalarValue {
            bits: Bits::$variant(value),
        })
    }};
}

/// Perform one integer operation at the operand dtype's exact arithmetic
/// width, trapping before any narrowing or cross-family conversion.
pub fn int_binop(
    op: IntBinOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, NumericKernelError> {
    require_family(op.name(), lhs, NumericFamily::Int)?;
    require_family(op.name(), rhs, NumericFamily::Int)?;
    require_same_dtype(op.name(), lhs, rhs)?;
    match (lhs.bits, rhs.bits) {
        (Bits::I8(lhs), Bits::I8(rhs)) => int_binop_at_width!(op, Prim::Int8, lhs, rhs, I8),
        (Bits::I16(lhs), Bits::I16(rhs)) => {
            int_binop_at_width!(op, Prim::Int16, lhs, rhs, I16)
        }
        (Bits::I32(lhs), Bits::I32(rhs)) => {
            int_binop_at_width!(op, Prim::Int32, lhs, rhs, I32)
        }
        (Bits::I64(lhs), Bits::I64(rhs)) => {
            int_binop_at_width!(op, Prim::Int64, lhs, rhs, I64)
        }
        _ => unreachable!("family and dtype checks make the integer match exhaustive"),
    }
}

macro_rules! int_unop_at_width {
    ($op:expr, $prim:expr, $value:expr, $variant:ident) => {{
        let value = match $op {
            IntUnOp::Neg => $value.checked_neg(),
            IntUnOp::Abs => $value.checked_abs(),
        }
        .ok_or(NumericTrap::Overflow {
            op: $op.name(),
            prim: $prim,
        })?;
        Ok(ScalarValue {
            bits: Bits::$variant(value),
        })
    }};
}

/// Perform one integer unary operation at the operand dtype's exact
/// arithmetic width.
pub fn int_unop(op: IntUnOp, value: ScalarValue) -> Result<ScalarValue, NumericKernelError> {
    require_family(op.name(), value, NumericFamily::Int)?;
    match value.bits {
        Bits::I8(value) => int_unop_at_width!(op, Prim::Int8, value, I8),
        Bits::I16(value) => int_unop_at_width!(op, Prim::Int16, value, I16),
        Bits::I32(value) => int_unop_at_width!(op, Prim::Int32, value, I32),
        Bits::I64(value) => int_unop_at_width!(op, Prim::Int64, value, I64),
        _ => unreachable!("family check makes the integer match exhaustive"),
    }
}

fn apply_float_binop_f32(op: FloatBinOp, lhs: f32, rhs: f32) -> f32 {
    match op {
        FloatBinOp::Add => lhs + rhs,
        FloatBinOp::Sub => lhs - rhs,
        FloatBinOp::Mul => lhs * rhs,
        FloatBinOp::Div => lhs / rhs,
        FloatBinOp::FloorDiv => (lhs / rhs).floor(),
        FloatBinOp::Max => lhs.max(rhs),
        FloatBinOp::Min => lhs.min(rhs),
    }
}

fn apply_float_binop_f64(op: FloatBinOp, lhs: f64, rhs: f64) -> f64 {
    match op {
        FloatBinOp::Add => lhs + rhs,
        FloatBinOp::Sub => lhs - rhs,
        FloatBinOp::Mul => lhs * rhs,
        FloatBinOp::Div => lhs / rhs,
        FloatBinOp::FloorDiv => (lhs / rhs).floor(),
        FloatBinOp::Max => lhs.max(rhs),
        FloatBinOp::Min => lhs.min(rhs),
    }
}

/// Perform one float operation at f64 for f64 and f32 for every other
/// active float dtype, then finalize once into the operand storage width.
pub fn float_binop(
    op: FloatBinOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, NumericKernelError> {
    require_family(op.name(), lhs, NumericFamily::Float)?;
    require_family(op.name(), rhs, NumericFamily::Float)?;
    require_same_dtype(op.name(), lhs, rhs)?;
    let bits =
        match (lhs.bits, rhs.bits) {
            (Bits::F64(lhs), Bits::F64(rhs)) => Bits::F64(apply_float_binop_f64(op, lhs, rhs)),
            (Bits::F32(lhs), Bits::F32(rhs)) => Bits::F32(apply_float_binop_f32(op, lhs, rhs)),
            (Bits::F16(lhs), Bits::F16(rhs)) => Bits::F16(half::f16::from_f32(
                apply_float_binop_f32(op, lhs.to_f32(), rhs.to_f32()),
            )),
            (Bits::Bf16(lhs), Bits::Bf16(rhs)) => Bits::Bf16(half::bf16::from_f32(
                apply_float_binop_f32(op, lhs.to_f32(), rhs.to_f32()),
            )),
            _ => unreachable!("family and dtype checks make the float match exhaustive"),
        };
    Ok(ScalarValue { bits })
}

fn apply_float_unop_f32(op: FloatUnOp, value: f32) -> f32 {
    match op {
        FloatUnOp::Neg => -value,
        FloatUnOp::Recip => value.recip(),
        FloatUnOp::Exp => value.exp(),
        FloatUnOp::Log => value.ln(),
        FloatUnOp::Sin => value.sin(),
        FloatUnOp::Sqrt => value.sqrt(),
        FloatUnOp::Cos => value.cos(),
        FloatUnOp::Tan => value.tan(),
        FloatUnOp::Atan => value.atan(),
        FloatUnOp::Abs => value.abs(),
        FloatUnOp::Floor => value.floor(),
        FloatUnOp::Ceil => value.ceil(),
        FloatUnOp::Round => value.round_ties_even(),
        FloatUnOp::Relu => {
            if value > 0.0 {
                value
            } else {
                0.0
            }
        }
        FloatUnOp::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        FloatUnOp::Tanh => value.tanh(),
        FloatUnOp::Silu => value * (1.0 / (1.0 + (-value).exp())),
        FloatUnOp::Gelu => {
            #[allow(clippy::excessive_precision)]
            const C: f32 = 0.7978845608028654_f32;
            const K: f32 = 0.044715_f32;
            let inner = C * (value + K * value * value * value);
            0.5 * value * (1.0 + inner.tanh())
        }
    }
}

fn apply_float_unop_f64(op: FloatUnOp, value: f64) -> f64 {
    match op {
        FloatUnOp::Neg => -value,
        FloatUnOp::Recip => value.recip(),
        FloatUnOp::Exp => value.exp(),
        FloatUnOp::Log => value.ln(),
        FloatUnOp::Sin => value.sin(),
        FloatUnOp::Sqrt => value.sqrt(),
        FloatUnOp::Cos => value.cos(),
        FloatUnOp::Tan => value.tan(),
        FloatUnOp::Atan => value.atan(),
        FloatUnOp::Abs => value.abs(),
        FloatUnOp::Floor => value.floor(),
        FloatUnOp::Ceil => value.ceil(),
        FloatUnOp::Round => value.round_ties_even(),
        FloatUnOp::Relu => {
            if value > 0.0 {
                value
            } else {
                0.0
            }
        }
        FloatUnOp::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        FloatUnOp::Tanh => value.tanh(),
        FloatUnOp::Silu => value * (1.0 / (1.0 + (-value).exp())),
        FloatUnOp::Gelu => {
            const C: f64 = 0.7978845608028654;
            const K: f64 = 0.044715;
            let inner = C * (value + K * value * value * value);
            0.5 * value * (1.0 + inner.tanh())
        }
    }
}

/// Perform one float unary operation at the dtype's arithmetic width and
/// finalize once into its storage width.
pub fn float_unop(op: FloatUnOp, value: ScalarValue) -> Result<ScalarValue, NumericKernelError> {
    require_family(op.name(), value, NumericFamily::Float)?;
    let bits = match value.bits {
        Bits::F64(value) => Bits::F64(apply_float_unop_f64(op, value)),
        Bits::F32(value) => Bits::F32(apply_float_unop_f32(op, value)),
        Bits::F16(value) => Bits::F16(half::f16::from_f32(apply_float_unop_f32(
            op,
            value.to_f32(),
        ))),
        Bits::Bf16(value) => Bits::Bf16(half::bf16::from_f32(apply_float_unop_f32(
            op,
            value.to_f32(),
        ))),
        _ => unreachable!("family check makes the float match exhaustive"),
    };
    Ok(ScalarValue { bits })
}

macro_rules! compare_values {
    ($op:expr, $lhs:expr, $rhs:expr) => {
        match $op {
            CompareOp::Eq => $lhs == $rhs,
            CompareOp::Ne => $lhs != $rhs,
            CompareOp::Lt => $lhs < $rhs,
            CompareOp::Gt => $lhs > $rhs,
            CompareOp::Lte => $lhs <= $rhs,
            CompareOp::Gte => $lhs >= $rhs,
        }
    };
}

/// Compare finalized operands at their exact matching dtype. In
/// particular, int64 never crosses f64 and half values compare only after
/// their ingress finalization.
pub fn compare_scalars(
    op: CompareOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<bool, NumericKernelError> {
    require_same_dtype(op.name(), lhs, rhs)?;
    let result = match (lhs.bits, rhs.bits) {
        (Bits::I8(lhs), Bits::I8(rhs)) => compare_values!(op, lhs, rhs),
        (Bits::I16(lhs), Bits::I16(rhs)) => compare_values!(op, lhs, rhs),
        (Bits::I32(lhs), Bits::I32(rhs)) => compare_values!(op, lhs, rhs),
        (Bits::I64(lhs), Bits::I64(rhs)) => compare_values!(op, lhs, rhs),
        (Bits::F16(lhs), Bits::F16(rhs)) => {
            compare_values!(op, lhs.to_f32(), rhs.to_f32())
        }
        (Bits::Bf16(lhs), Bits::Bf16(rhs)) => {
            compare_values!(op, lhs.to_f32(), rhs.to_f32())
        }
        (Bits::F32(lhs), Bits::F32(rhs)) => compare_values!(op, lhs, rhs),
        (Bits::F64(lhs), Bits::F64(rhs)) => compare_values!(op, lhs, rhs),
        (Bits::Bool(lhs), Bits::Bool(rhs)) => compare_values!(op, lhs, rhs),
        _ => unreachable!("dtype check makes the comparison match exhaustive"),
    };
    Ok(result)
}

/// Sealed per-dtype element buffer (section C3's storage decision:
/// per-dtype buffers, not finalize-on-write over `Vec<f64>`; f64 storage
/// cannot represent exact int64 above 2^53, chelis#684).
#[derive(Debug, Clone, PartialEq)]
enum Buf {
    F64(Vec<f64>),
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    I8(Vec<i8>),
    /// 0/1, one byte per element; ends PR #79's bool-storage deferral.
    Bool(Vec<u8>),
}

/// Read-only borrowed view of a [`TensorStorage`] buffer at its own
/// width (section C3: reads are free-form; only construction is gated).
#[derive(Debug, Clone, Copy)]
pub enum StorageView<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    F16(&'a [half::f16]),
    Bf16(&'a [half::bf16]),
    I64(&'a [i64]),
    I32(&'a [i32]),
    I16(&'a [i16]),
    I8(&'a [i8]),
    /// 0/1 bytes.
    Bool(&'a [u8]),
}

/// A finalized element buffer at its dtype's own width. Construct via
/// [`finalize_tensor`] or the `reuse_*` element-preserving family.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorStorage {
    buf: Buf,
}

impl TensorStorage {
    /// The dtype of every element, from the buffer variant itself.
    pub fn prim(&self) -> Prim {
        match &self.buf {
            Buf::F64(_) => Prim::F64,
            Buf::F32(_) => Prim::F32,
            Buf::F16(_) => Prim::F16,
            Buf::Bf16(_) => Prim::Bf16,
            Buf::I64(_) => Prim::Int64,
            Buf::I32(_) => Prim::Int32,
            Buf::I16(_) => Prim::Int16,
            Buf::I8(_) => Prim::Int8,
            Buf::Bool(_) => Prim::Bool,
        }
    }

    pub fn len(&self) -> usize {
        match &self.buf {
            Buf::F64(v) => v.len(),
            Buf::F32(v) => v.len(),
            Buf::F16(v) => v.len(),
            Buf::Bf16(v) => v.len(),
            Buf::I64(v) => v.len(),
            Buf::I32(v) => v.len(),
            Buf::I16(v) => v.len(),
            Buf::I8(v) => v.len(),
            Buf::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrowed per-dtype view of the buffer for exact egress (the wire
    /// schema, typed consumers). A read-only API: no construction path
    /// exists through it.
    pub fn view(&self) -> StorageView<'_> {
        match &self.buf {
            Buf::F64(v) => StorageView::F64(v),
            Buf::F32(v) => StorageView::F32(v),
            Buf::F16(v) => StorageView::F16(v),
            Buf::Bf16(v) => StorageView::Bf16(v),
            Buf::I64(v) => StorageView::I64(v),
            Buf::I32(v) => StorageView::I32(v),
            Buf::I16(v) => StorageView::I16(v),
            Buf::I8(v) => StorageView::I8(v),
            Buf::Bool(v) => StorageView::Bool(v),
        }
    }

    /// The wide intermediate view of the whole buffer, per family: floats
    /// widen exactly to f64, integers exactly to i64, bool to 0/1 i64.
    /// This read is EXACT (no cross-family collapse); the lossy
    /// cross-family read is [`Self::to_f64_lossy_vec`].
    pub fn to_raw(&self) -> RawTensor {
        match &self.buf {
            Buf::F64(v) => RawTensor::Float(v.clone()),
            Buf::F32(v) => RawTensor::Float(v.iter().map(|&x| x as f64).collect()),
            Buf::F16(v) => RawTensor::Float(v.iter().map(|&x| f64::from(x)).collect()),
            Buf::Bf16(v) => RawTensor::Float(v.iter().map(|&x| f64::from(x)).collect()),
            Buf::I64(v) => RawTensor::Int(v.clone()),
            Buf::I32(v) => RawTensor::Int(v.iter().map(|&x| x as i64).collect()),
            Buf::I16(v) => RawTensor::Int(v.iter().map(|&x| x as i64).collect()),
            Buf::I8(v) => RawTensor::Int(v.iter().map(|&x| x as i64).collect()),
            Buf::Bool(v) => RawTensor::Int(v.iter().map(|&x| x as i64).collect()),
        }
    }

    /// Widen every element to f64. Exact except for int64 magnitudes
    /// above 2^53, hence the lossy name (section C3 read-side contract).
    pub fn to_f64_lossy_vec(&self) -> Vec<f64> {
        match &self.buf {
            Buf::F64(v) => v.clone(),
            Buf::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Buf::F16(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Buf::Bf16(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Buf::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Buf::I32(v) => v.iter().map(|&x| x as f64).collect(),
            Buf::I16(v) => v.iter().map(|&x| x as f64).collect(),
            Buf::I8(v) => v.iter().map(|&x| x as f64).collect(),
            Buf::Bool(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    /// The exact i64 vector of an integer-family buffer (bool reads 0/1);
    /// `None` for float buffers.
    pub fn to_i64_exact_vec(&self) -> Option<Vec<i64>> {
        match &self.buf {
            Buf::I64(v) => Some(v.clone()),
            Buf::I32(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Buf::I16(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Buf::I8(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Buf::Bool(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Buf::F64(_) | Buf::F32(_) | Buf::F16(_) | Buf::Bf16(_) => None,
        }
    }

    /// One element widened to f64 (same loss profile as
    /// [`Self::to_f64_lossy_vec`]). Panics on out-of-bounds like slice
    /// indexing.
    pub fn element_f64_lossy(&self, index: usize) -> f64 {
        match &self.buf {
            Buf::F64(v) => v[index],
            Buf::F32(v) => v[index] as f64,
            Buf::F16(v) => f64::from(v[index]),
            Buf::Bf16(v) => f64::from(v[index]),
            Buf::I64(v) => v[index] as f64,
            Buf::I32(v) => v[index] as f64,
            Buf::I16(v) => v[index] as f64,
            Buf::I8(v) => v[index] as f64,
            Buf::Bool(v) => v[index] as f64,
        }
    }

    /// One element as a sealed [`ScalarValue`] (element-preserving read;
    /// the element was finalized when the buffer was constructed).
    pub fn scalar_at(&self, index: usize) -> ScalarValue {
        let bits = match &self.buf {
            Buf::F64(v) => Bits::F64(v[index]),
            Buf::F32(v) => Bits::F32(v[index]),
            Buf::F16(v) => Bits::F16(v[index]),
            Buf::Bf16(v) => Bits::Bf16(v[index]),
            Buf::I64(v) => Bits::I64(v[index]),
            Buf::I32(v) => Bits::I32(v[index]),
            Buf::I16(v) => Bits::I16(v[index]),
            Buf::I8(v) => Bits::I8(v[index]),
            Buf::Bool(v) => Bits::Bool(v[index] != 0),
        };
        ScalarValue { bits }
    }

    /// The observation-channel view of one element, for `format_element`.
    pub fn element_ref(&self, index: usize) -> ElementRef {
        self.scalar_at(index).element_ref()
    }

    /// reuse_storage family (THE one deliberate construction hole,
    /// section C3): element-preserving gather. `output[i] =
    /// self[indices[i]]` with no re-finalize, legal ONLY for movement ops
    /// that provably preserve elements (reshape, permute, expand, shrink,
    /// stride, gather). Every use site cites this contract.
    pub fn reuse_gather(&self, indices: &[usize]) -> TensorStorage {
        fn pick<T: Copy>(src: &[T], indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&i| src[i]).collect()
        }
        let buf = match &self.buf {
            Buf::F64(v) => Buf::F64(pick(v, indices)),
            Buf::F32(v) => Buf::F32(pick(v, indices)),
            Buf::F16(v) => Buf::F16(pick(v, indices)),
            Buf::Bf16(v) => Buf::Bf16(pick(v, indices)),
            Buf::I64(v) => Buf::I64(pick(v, indices)),
            Buf::I32(v) => Buf::I32(pick(v, indices)),
            Buf::I16(v) => Buf::I16(pick(v, indices)),
            Buf::I8(v) => Buf::I8(pick(v, indices)),
            Buf::Bool(v) => Buf::Bool(pick(v, indices)),
        };
        TensorStorage { buf }
    }

    /// reuse_storage family (section C3, element-preserving only): gather
    /// with a fill for uncovered slots (pad). `map[i] = Some(j)` copies
    /// `self[j]`; `None` writes `fill`. Panics if `fill`'s dtype differs
    /// from the buffer's (a caller bug, mirroring `format_element`'s
    /// mismatch rule).
    pub fn reuse_fill_gather(&self, fill: &ScalarValue, map: &[Option<usize>]) -> TensorStorage {
        assert_eq!(
            fill.prim(),
            self.prim(),
            "reuse_fill_gather: fill dtype {} does not match buffer dtype {}",
            fill.prim().name(),
            self.prim().name()
        );
        fn place<T: Copy>(src: &[T], fill: T, map: &[Option<usize>]) -> Vec<T> {
            map.iter()
                .map(|slot| match slot {
                    Some(j) => src[*j],
                    None => fill,
                })
                .collect()
        }
        let buf = match (&self.buf, fill.bits) {
            (Buf::F64(v), Bits::F64(f)) => Buf::F64(place(v, f, map)),
            (Buf::F32(v), Bits::F32(f)) => Buf::F32(place(v, f, map)),
            (Buf::F16(v), Bits::F16(f)) => Buf::F16(place(v, f, map)),
            (Buf::Bf16(v), Bits::Bf16(f)) => Buf::Bf16(place(v, f, map)),
            (Buf::I64(v), Bits::I64(f)) => Buf::I64(place(v, f, map)),
            (Buf::I32(v), Bits::I32(f)) => Buf::I32(place(v, f, map)),
            (Buf::I16(v), Bits::I16(f)) => Buf::I16(place(v, f, map)),
            (Buf::I8(v), Bits::I8(f)) => Buf::I8(place(v, f, map)),
            (Buf::Bool(v), Bits::Bool(f)) => Buf::Bool(place(v, u8::from(f), map)),
            (buf_other, bits_other) => unreachable!(
                "reuse_fill_gather: prim equality was asserted above, yet buffer {:?} \
                 met fill {:?}",
                std::mem::discriminant(buf_other),
                std::mem::discriminant(&bits_other)
            ),
        };
        TensorStorage { buf }
    }

    /// reuse_storage family (section C3, element-preserving only):
    /// last-write-wins overwrite for the replace-scatter ops. Clones
    /// `self`, then writes `src[from]` into slot `to` for each `(to,
    /// from)` in order. Panics on dtype mismatch (caller bug).
    pub fn reuse_overwrite<I>(&self, src: &TensorStorage, writes: I) -> TensorStorage
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        assert_eq!(
            src.prim(),
            self.prim(),
            "reuse_overwrite: source dtype {} does not match target dtype {}",
            src.prim().name(),
            self.prim().name()
        );
        fn write<T: Copy>(
            dst: &mut [T],
            src: &[T],
            writes: impl IntoIterator<Item = (usize, usize)>,
        ) {
            for (to, from) in writes {
                dst[to] = src[from];
            }
        }
        let mut out = self.clone();
        match (&mut out.buf, &src.buf) {
            (Buf::F64(d), Buf::F64(s)) => write(d, s, writes),
            (Buf::F32(d), Buf::F32(s)) => write(d, s, writes),
            (Buf::F16(d), Buf::F16(s)) => write(d, s, writes),
            (Buf::Bf16(d), Buf::Bf16(s)) => write(d, s, writes),
            (Buf::I64(d), Buf::I64(s)) => write(d, s, writes),
            (Buf::I32(d), Buf::I32(s)) => write(d, s, writes),
            (Buf::I16(d), Buf::I16(s)) => write(d, s, writes),
            (Buf::I8(d), Buf::I8(s)) => write(d, s, writes),
            (Buf::Bool(d), Buf::Bool(s)) => write(d, s, writes),
            (dst_other, src_other) => unreachable!(
                "reuse_overwrite: prim equality was asserted above, yet target {:?} \
                 met source {:?}",
                std::mem::discriminant(&*dst_other),
                std::mem::discriminant(src_other)
            ),
        }
        out
    }
}

/// Round an exact i64 directly to bfloat16, once, with target-width
/// round-to-nearest-ties-to-even. Converting through f64 first is not
/// equivalent above 2^53: it can erase which side of a bf16 midpoint the
/// exact integer occupies and then manufacture a tie (chelis#729 Phase 1,
/// [04-NUM-14]).
fn bf16_from_i64_rne(value: i64) -> half::bf16 {
    if value == 0 {
        return half::bf16::from_f64(0.0);
    }

    let negative = value.is_negative();
    let magnitude = value.unsigned_abs();
    let top_bit = 63 - magnitude.leading_zeros();
    let rounded_magnitude = if top_bit <= 7 {
        magnitude
    } else {
        let shift = top_bit - 7;
        let mut significand = magnitude >> shift;
        let remainder_mask = (1_u64 << shift) - 1;
        let remainder = magnitude & remainder_mask;
        let halfway = 1_u64 << (shift - 1);
        if remainder > halfway || (remainder == halfway && significand & 1 == 1) {
            significand += 1;
        }
        significand << shift
    };

    // `rounded_magnitude` has at most eight significant bits, so this u64
    // to f64 conversion is exact even when the rounded result is 2^63.
    let exact_image = rounded_magnitude as f64;
    half::bf16::from_f64(if negative { -exact_image } else { exact_image })
}

fn require_same_storage_shape(
    op: &'static str,
    lhs: &TensorStorage,
    rhs: &TensorStorage,
) -> Result<(), NumericKernelError> {
    if lhs.len() != rhs.len() {
        return Err(NumericKernelError::LengthMismatch {
            op,
            lhs: lhs.len(),
            rhs: rhs.len(),
        });
    }
    if lhs.prim() != rhs.prim() {
        return Err(NumericKernelError::DtypeMismatch {
            op,
            lhs: lhs.prim(),
            rhs: rhs.prim(),
        });
    }
    Ok(())
}

fn zip_map<T: Copy, U>(lhs: &[T], rhs: &[T], mut op: impl FnMut(T, T) -> U) -> Vec<U> {
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs)| op(lhs, rhs))
        .collect()
}

fn zip_try_map<T: Copy, U, E>(
    lhs: &[T],
    rhs: &[T],
    mut op: impl FnMut(T, T) -> Result<U, E>,
) -> Result<Vec<U>, E> {
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs)| op(lhs, rhs))
        .collect()
}

macro_rules! int_tensor_binop_at_width {
    ($op:expr, $prim:expr, $lhs:expr, $rhs:expr, $variant:ident) => {{
        let overflow = || NumericTrap::Overflow {
            op: $op.name(),
            prim: $prim,
        };
        let div_zero = || NumericTrap::DivZero {
            op: $op.name(),
            prim: $prim,
        };
        let values = match $op {
            IntBinOp::Add => zip_try_map($lhs, $rhs, |lhs, rhs| {
                lhs.checked_add(rhs).ok_or_else(overflow)
            }),
            IntBinOp::Sub => zip_try_map($lhs, $rhs, |lhs, rhs| {
                lhs.checked_sub(rhs).ok_or_else(overflow)
            }),
            IntBinOp::Mul => zip_try_map($lhs, $rhs, |lhs, rhs| {
                lhs.checked_mul(rhs).ok_or_else(overflow)
            }),
            IntBinOp::FloorDiv => zip_try_map($lhs, $rhs, |lhs, rhs| {
                if rhs == 0 {
                    return Err(div_zero());
                }
                let quotient = lhs.checked_div(rhs).ok_or_else(overflow)?;
                let remainder = lhs % rhs;
                if remainder != 0 && (remainder < 0) != (rhs < 0) {
                    quotient.checked_sub(1).ok_or_else(overflow)
                } else {
                    Ok(quotient)
                }
            }),
            IntBinOp::TruncDiv => zip_try_map($lhs, $rhs, |lhs, rhs| {
                if rhs == 0 {
                    Err(div_zero())
                } else {
                    lhs.checked_div(rhs).ok_or_else(overflow)
                }
            }),
            IntBinOp::Rem => zip_try_map($lhs, $rhs, |lhs, rhs| {
                if rhs == 0 {
                    Err(div_zero())
                } else {
                    Ok(if rhs == -1 { 0 } else { lhs % rhs })
                }
            }),
            IntBinOp::Max => Ok(zip_map($lhs, $rhs, |lhs, rhs| lhs.max(rhs))),
            IntBinOp::Min => Ok(zip_map($lhs, $rhs, |lhs, rhs| lhs.min(rhs))),
        }?;
        Ok(TensorStorage {
            buf: Buf::$variant(values),
        })
    }};
}

/// Bulk integer binary kernel. Operation dispatch happens once outside a
/// monomorphized loop for the storage width, never once per element.
pub fn int_tensor_binop(
    op: IntBinOp,
    lhs: &TensorStorage,
    rhs: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    if !lhs.prim().is_integer() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Int,
            actual: lhs.prim(),
        });
    }
    if !rhs.prim().is_integer() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Int,
            actual: rhs.prim(),
        });
    }
    require_same_storage_shape(op.name(), lhs, rhs)?;
    match (&lhs.buf, &rhs.buf) {
        (Buf::I8(lhs), Buf::I8(rhs)) => {
            int_tensor_binop_at_width!(op, Prim::Int8, lhs, rhs, I8)
        }
        (Buf::I16(lhs), Buf::I16(rhs)) => {
            int_tensor_binop_at_width!(op, Prim::Int16, lhs, rhs, I16)
        }
        (Buf::I32(lhs), Buf::I32(rhs)) => {
            int_tensor_binop_at_width!(op, Prim::Int32, lhs, rhs, I32)
        }
        (Buf::I64(lhs), Buf::I64(rhs)) => {
            int_tensor_binop_at_width!(op, Prim::Int64, lhs, rhs, I64)
        }
        _ => unreachable!("family and dtype checks make the integer buffers exhaustive"),
    }
}

macro_rules! int_tensor_unop_at_width {
    ($op:expr, $prim:expr, $values:expr, $variant:ident) => {{
        let overflow = || NumericTrap::Overflow {
            op: $op.name(),
            prim: $prim,
        };
        let values: Result<Vec<_>, NumericTrap> = match $op {
            IntUnOp::Neg => $values
                .iter()
                .copied()
                .map(|value| value.checked_neg().ok_or_else(overflow))
                .collect(),
            IntUnOp::Abs => $values
                .iter()
                .copied()
                .map(|value| value.checked_abs().ok_or_else(overflow))
                .collect(),
        };
        Ok(TensorStorage {
            buf: Buf::$variant(values?),
        })
    }};
}

/// Bulk integer unary kernel with one operation dispatch per buffer.
pub fn int_tensor_unop(
    op: IntUnOp,
    value: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    if !value.prim().is_integer() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Int,
            actual: value.prim(),
        });
    }
    match &value.buf {
        Buf::I8(values) => int_tensor_unop_at_width!(op, Prim::Int8, values, I8),
        Buf::I16(values) => int_tensor_unop_at_width!(op, Prim::Int16, values, I16),
        Buf::I32(values) => int_tensor_unop_at_width!(op, Prim::Int32, values, I32),
        Buf::I64(values) => int_tensor_unop_at_width!(op, Prim::Int64, values, I64),
        _ => unreachable!("family check makes the integer buffer exhaustive"),
    }
}

fn float_vec_binop_f32<T: Copy>(
    op: FloatBinOp,
    lhs: &[T],
    rhs: &[T],
    to_f32: impl Fn(T) -> f32,
    from_f32: impl Fn(f32) -> T,
) -> Vec<T> {
    match op {
        FloatBinOp::Add => zip_map(lhs, rhs, |lhs, rhs| from_f32(to_f32(lhs) + to_f32(rhs))),
        FloatBinOp::Sub => zip_map(lhs, rhs, |lhs, rhs| from_f32(to_f32(lhs) - to_f32(rhs))),
        FloatBinOp::Mul => zip_map(lhs, rhs, |lhs, rhs| from_f32(to_f32(lhs) * to_f32(rhs))),
        FloatBinOp::Div => zip_map(lhs, rhs, |lhs, rhs| from_f32(to_f32(lhs) / to_f32(rhs))),
        FloatBinOp::FloorDiv => zip_map(lhs, rhs, |lhs, rhs| {
            from_f32((to_f32(lhs) / to_f32(rhs)).floor())
        }),
        FloatBinOp::Max => zip_map(lhs, rhs, |lhs, rhs| from_f32(to_f32(lhs).max(to_f32(rhs)))),
        FloatBinOp::Min => zip_map(lhs, rhs, |lhs, rhs| from_f32(to_f32(lhs).min(to_f32(rhs)))),
    }
}

fn float_vec_binop_f64(op: FloatBinOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    match op {
        FloatBinOp::Add => zip_map(lhs, rhs, |lhs, rhs| lhs + rhs),
        FloatBinOp::Sub => zip_map(lhs, rhs, |lhs, rhs| lhs - rhs),
        FloatBinOp::Mul => zip_map(lhs, rhs, |lhs, rhs| lhs * rhs),
        FloatBinOp::Div => zip_map(lhs, rhs, |lhs, rhs| lhs / rhs),
        FloatBinOp::FloorDiv => zip_map(lhs, rhs, |lhs, rhs| (lhs / rhs).floor()),
        FloatBinOp::Max => zip_map(lhs, rhs, f64::max),
        FloatBinOp::Min => zip_map(lhs, rhs, f64::min),
    }
}

/// Bulk float binary kernel with native f64/f32 arithmetic and one final
/// half-width conversion per element.
pub fn float_tensor_binop(
    op: FloatBinOp,
    lhs: &TensorStorage,
    rhs: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    if !lhs.prim().is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Float,
            actual: lhs.prim(),
        });
    }
    if !rhs.prim().is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Float,
            actual: rhs.prim(),
        });
    }
    require_same_storage_shape(op.name(), lhs, rhs)?;
    let buf = match (&lhs.buf, &rhs.buf) {
        (Buf::F64(lhs), Buf::F64(rhs)) => Buf::F64(float_vec_binop_f64(op, lhs, rhs)),
        (Buf::F32(lhs), Buf::F32(rhs)) => Buf::F32(float_vec_binop_f32(op, lhs, rhs, |x| x, |x| x)),
        (Buf::F16(lhs), Buf::F16(rhs)) => Buf::F16(float_vec_binop_f32(
            op,
            lhs,
            rhs,
            half::f16::to_f32,
            half::f16::from_f32,
        )),
        (Buf::Bf16(lhs), Buf::Bf16(rhs)) => Buf::Bf16(float_vec_binop_f32(
            op,
            lhs,
            rhs,
            half::bf16::to_f32,
            half::bf16::from_f32,
        )),
        _ => unreachable!("family and dtype checks make the float buffers exhaustive"),
    };
    Ok(TensorStorage { buf })
}

fn float_vec_unop_f32<T: Copy>(
    op: FloatUnOp,
    values: &[T],
    to_f32: impl Fn(T) -> f32,
    from_f32: impl Fn(f32) -> T,
) -> Vec<T> {
    macro_rules! map {
        ($body:expr) => {
            values
                .iter()
                .copied()
                .map(|value| from_f32($body(to_f32(value))))
                .collect()
        };
    }
    match op {
        FloatUnOp::Neg => map!(|x: f32| -x),
        FloatUnOp::Recip => map!(f32::recip),
        FloatUnOp::Exp => map!(f32::exp),
        FloatUnOp::Log => map!(f32::ln),
        FloatUnOp::Sin => map!(f32::sin),
        FloatUnOp::Sqrt => map!(f32::sqrt),
        FloatUnOp::Cos => map!(f32::cos),
        FloatUnOp::Tan => map!(f32::tan),
        FloatUnOp::Atan => map!(f32::atan),
        FloatUnOp::Abs => map!(f32::abs),
        FloatUnOp::Floor => map!(f32::floor),
        FloatUnOp::Ceil => map!(f32::ceil),
        FloatUnOp::Round => map!(f32::round_ties_even),
        FloatUnOp::Relu
        | FloatUnOp::Sigmoid
        | FloatUnOp::Tanh
        | FloatUnOp::Silu
        | FloatUnOp::Gelu => values
            .iter()
            .copied()
            .map(|value| from_f32(apply_float_unop_f32(op, to_f32(value))))
            .collect(),
    }
}

fn float_vec_unop_f64(op: FloatUnOp, values: &[f64]) -> Vec<f64> {
    macro_rules! map {
        ($body:expr) => {
            values.iter().copied().map($body).collect()
        };
    }
    match op {
        FloatUnOp::Neg => map!(|x: f64| -x),
        FloatUnOp::Recip => map!(f64::recip),
        FloatUnOp::Exp => map!(f64::exp),
        FloatUnOp::Log => map!(f64::ln),
        FloatUnOp::Sin => map!(f64::sin),
        FloatUnOp::Sqrt => map!(f64::sqrt),
        FloatUnOp::Cos => map!(f64::cos),
        FloatUnOp::Tan => map!(f64::tan),
        FloatUnOp::Atan => map!(f64::atan),
        FloatUnOp::Abs => map!(f64::abs),
        FloatUnOp::Floor => map!(f64::floor),
        FloatUnOp::Ceil => map!(f64::ceil),
        FloatUnOp::Round => map!(f64::round_ties_even),
        FloatUnOp::Relu
        | FloatUnOp::Sigmoid
        | FloatUnOp::Tanh
        | FloatUnOp::Silu
        | FloatUnOp::Gelu => values
            .iter()
            .copied()
            .map(|value| apply_float_unop_f64(op, value))
            .collect(),
    }
}

/// Bulk float unary kernel with one operation dispatch per buffer.
pub fn float_tensor_unop(
    op: FloatUnOp,
    value: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    if !value.prim().is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Float,
            actual: value.prim(),
        });
    }
    let buf = match &value.buf {
        Buf::F64(values) => Buf::F64(float_vec_unop_f64(op, values)),
        Buf::F32(values) => Buf::F32(float_vec_unop_f32(op, values, |x| x, |x| x)),
        Buf::F16(values) => Buf::F16(float_vec_unop_f32(
            op,
            values,
            half::f16::to_f32,
            half::f16::from_f32,
        )),
        Buf::Bf16(values) => Buf::Bf16(float_vec_unop_f32(
            op,
            values,
            half::bf16::to_f32,
            half::bf16::from_f32,
        )),
        _ => unreachable!("family check makes the float buffer exhaustive"),
    };
    Ok(TensorStorage { buf })
}

fn splat_storage(value: ScalarValue, len: usize) -> TensorStorage {
    let buf = match value.bits {
        Bits::I8(value) => Buf::I8(vec![value; len]),
        Bits::I16(value) => Buf::I16(vec![value; len]),
        Bits::I32(value) => Buf::I32(vec![value; len]),
        Bits::I64(value) => Buf::I64(vec![value; len]),
        Bits::F16(value) => Buf::F16(vec![value; len]),
        Bits::Bf16(value) => Buf::Bf16(vec![value; len]),
        Bits::F32(value) => Buf::F32(vec![value; len]),
        Bits::F64(value) => Buf::F64(vec![value; len]),
        Bits::Bool(value) => Buf::Bool(vec![u8::from(value); len]),
    };
    TensorStorage { buf }
}

pub fn int_tensor_scalar_binop(
    op: IntBinOp,
    tensor: &TensorStorage,
    scalar: ScalarValue,
) -> Result<TensorStorage, NumericKernelError> {
    int_tensor_binop(op, tensor, &splat_storage(scalar, tensor.len()))
}

pub fn int_scalar_tensor_binop(
    op: IntBinOp,
    scalar: ScalarValue,
    tensor: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    int_tensor_binop(op, &splat_storage(scalar, tensor.len()), tensor)
}

pub fn float_tensor_scalar_binop(
    op: FloatBinOp,
    tensor: &TensorStorage,
    scalar: ScalarValue,
) -> Result<TensorStorage, NumericKernelError> {
    float_tensor_binop(op, tensor, &splat_storage(scalar, tensor.len()))
}

pub fn float_scalar_tensor_binop(
    op: FloatBinOp,
    scalar: ScalarValue,
    tensor: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    float_tensor_binop(op, &splat_storage(scalar, tensor.len()), tensor)
}

fn compare_vec<T: Copy + PartialEq + PartialOrd>(op: CompareOp, lhs: &[T], rhs: &[T]) -> Vec<u8> {
    match op {
        CompareOp::Eq => zip_map(lhs, rhs, |lhs, rhs| u8::from(lhs == rhs)),
        CompareOp::Ne => zip_map(lhs, rhs, |lhs, rhs| u8::from(lhs != rhs)),
        CompareOp::Lt => zip_map(lhs, rhs, |lhs, rhs| u8::from(lhs < rhs)),
        CompareOp::Gt => zip_map(lhs, rhs, |lhs, rhs| u8::from(lhs > rhs)),
        CompareOp::Lte => zip_map(lhs, rhs, |lhs, rhs| u8::from(lhs <= rhs)),
        CompareOp::Gte => zip_map(lhs, rhs, |lhs, rhs| u8::from(lhs >= rhs)),
    }
}

/// Compare two finalized buffers at their shared storage width and return
/// sealed bool storage.
pub fn compare_tensors(
    op: CompareOp,
    lhs: &TensorStorage,
    rhs: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    require_same_storage_shape(op.name(), lhs, rhs)?;
    let values = match (&lhs.buf, &rhs.buf) {
        (Buf::I8(lhs), Buf::I8(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::I16(lhs), Buf::I16(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::I32(lhs), Buf::I32(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::I64(lhs), Buf::I64(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::F16(lhs), Buf::F16(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::Bf16(lhs), Buf::Bf16(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::F32(lhs), Buf::F32(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::F64(lhs), Buf::F64(rhs)) => compare_vec(op, lhs, rhs),
        (Buf::Bool(lhs), Buf::Bool(rhs)) => compare_vec(op, lhs, rhs),
        _ => unreachable!("dtype check makes the comparison buffers exhaustive"),
    };
    Ok(TensorStorage {
        buf: Buf::Bool(values),
    })
}

pub fn compare_tensor_scalar(
    op: CompareOp,
    tensor: &TensorStorage,
    scalar: ScalarValue,
) -> Result<TensorStorage, NumericKernelError> {
    compare_tensors(op, tensor, &splat_storage(scalar, tensor.len()))
}

pub fn compare_scalar_tensor(
    op: CompareOp,
    scalar: ScalarValue,
    tensor: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    compare_tensors(op, &splat_storage(scalar, tensor.len()), tensor)
}

/// Finalize one wide intermediate into `prim` per the section C1 table,
/// or trap. THE construction chokepoint for op results.
pub fn finalize_scalar(
    op: &'static str,
    prim: Prim,
    raw: RawScalar,
) -> Result<ScalarValue, NumericTrap> {
    let bits = match prim {
        Prim::F64 => Bits::F64(match raw {
            RawScalar::Float(x) => x,
            RawScalar::Int(i) => i as f64,
        }),
        Prim::F32 => Bits::F32(match raw {
            RawScalar::Float(x) => x as f32,
            RawScalar::Int(i) => i as f32,
        }),
        Prim::F16 => Bits::F16(match raw {
            RawScalar::Float(x) => half::f16::from_f64(x),
            RawScalar::Int(i) => half::f16::from_f64(i as f64),
        }),
        Prim::Bf16 => Bits::Bf16(match raw {
            RawScalar::Float(x) => half::bf16::from_f64(x),
            RawScalar::Int(i) => bf16_from_i64_rne(i),
        }),
        Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => {
            let wide = int_wide(op, prim, raw)?;
            let (lo, hi) = prim
                .integer_range()
                .expect("integer prims carry an integer_range");
            if wide < lo || wide > hi {
                return Err(NumericTrap::Overflow { op, prim });
            }
            match prim {
                Prim::Int8 => Bits::I8(wide as i8),
                Prim::Int16 => Bits::I16(wide as i16),
                Prim::Int32 => Bits::I32(wide as i32),
                Prim::Int64 => Bits::I64(wide),
                Prim::F32
                | Prim::F64
                | Prim::F16
                | Prim::Bf16
                | Prim::F8e4m3
                | Prim::Bool
                | Prim::String => unreachable!("outer match binds an integer prim"),
            }
        }
        Prim::Bool => {
            let wide = int_wide(op, prim, raw)?;
            match wide {
                0 => Bits::Bool(false),
                1 => Bits::Bool(true),
                _ => return Err(NumericTrap::Domain { op, prim }),
            }
        }
        // Not a runtime arm by contract (section C1): the checker rejects
        // f8e4m3 wherever it appears (spec/04-type-system.md section
        // 1.1.1), so no kernel can produce a value to finalize at it.
        Prim::F8e4m3 => panic!(
            "finalize_scalar: f8e4m3 is not in the active dtype set \
             (spec/04-type-system.md section 1.1.1); the checker rejects it, \
             so no numeric value can reach finalize at it (op {op})"
        ),
        Prim::String => panic!(
            "finalize_scalar: string is not a numeric dtype and has no \
             finalize semantics (op {op})"
        ),
    };
    Ok(ScalarValue { bits })
}

/// The exact integer reading of a wide intermediate for the integer/bool
/// rows: an `Int` passes through; a `Float` must be finite and integral
/// (else `Domain`) and inside i64 (else `Overflow`).
fn int_wide(op: &'static str, prim: Prim, raw: RawScalar) -> Result<i64, NumericTrap> {
    match raw {
        RawScalar::Int(i) => Ok(i),
        RawScalar::Float(x) => {
            if !x.is_finite() || x.fract() != 0.0 {
                return Err(NumericTrap::Domain { op, prim });
            }
            // Any integral f64 strictly below 2^63 is exactly
            // representable in i64; 2^63 itself (= the f64 image of many
            // out-of-range integers) is not.
            if x < -(2f64.powi(63)) || x >= 2f64.powi(63) {
                return Err(NumericTrap::Overflow { op, prim });
            }
            Ok(x as i64)
        }
    }
}

/// Ingress constructor for already-exact integer values (literals, host
/// boundary). Domain-checked; cannot trap for in-range input by
/// construction (section C3).
pub fn scalar_from_i64(op: &'static str, prim: Prim, v: i64) -> Result<ScalarValue, NumericTrap> {
    finalize_scalar(op, prim, RawScalar::Int(v))
}

/// Ingress constructor for already-exact float values (literals, host
/// boundary). Float targets apply the dtype's rounding; integer/bool
/// targets domain-check (section C3).
pub fn scalar_from_f64(op: &'static str, prim: Prim, v: f64) -> Result<ScalarValue, NumericTrap> {
    finalize_scalar(op, prim, RawScalar::Float(v))
}

/// THE authored cast ladder (the chelis#759 one-rule-per-direction
/// obligation, executed at the chelis#729 rework): the DEFAULT cast is
/// CHECKED on every eval surface. Per direction:
///
/// * any source -> float target: finalize (IEEE RNE at the target
///   width); total.
/// * integer/bool source -> integer target: exact value; out of the
///   target range TRAPS `Overflow` (no wrap).
/// * float source -> integer target: finalize only when finite and
///   integral; fractional/NaN/inf values TRAP `Domain`, while integral
///   values outside the target range TRAP `Overflow`.
/// * any source -> bool target: STRICT {0, 1} membership; an exact 0/1
///   encodes false/true, anything else TRAPS `Domain` (explicit over
///   implicit; comparisons already produce bool, and the corpus sweep
///   found no dependence on the old nonzero-to-1 encoding).
///
/// The named lossy/wrapping cast forms remain chelis#759's future
/// surface; this function is the checked DEFAULT. The compiled C lane
/// stays documented-divergent until chelis#729 Phase 3.
pub fn cast_raw(op: &'static str, raw: RawScalar, dst: Prim) -> Result<ScalarValue, NumericTrap> {
    match dst {
        Prim::F64
        | Prim::F32
        | Prim::F16
        | Prim::Bf16
        | Prim::Int8
        | Prim::Int16
        | Prim::Int32
        | Prim::Int64
        | Prim::Bool => finalize_scalar(op, dst, raw),
        Prim::F8e4m3 => panic!(
            "cast_raw: f8e4m3 is not in the active dtype set \
             (spec/04-type-system.md section 1.1.1); the checker rejects it, \
             so no cast can target it (op {op})"
        ),
        Prim::String => panic!("cast_raw: string is not a numeric dtype (op {op})"),
    }
}

/// [`cast_raw`] over a sealed scalar: the source family picks its exact
/// wide reading (integers and bool exactly through i64, floats through
/// their exact f64 image).
pub fn cast_scalar(
    op: &'static str,
    value: ScalarValue,
    dst: Prim,
) -> Result<ScalarValue, NumericTrap> {
    if value.prim() == dst {
        return Ok(value);
    }
    let raw = match value.as_i64_exact() {
        Some(i) => RawScalar::Int(i),
        None => RawScalar::Float(value.as_f64_lossy()),
    };
    cast_raw(op, raw, dst)
}

// ---------------------------------------------------------------------
// Serialization (chelis#729 rework: the FIFTH storage layer). The IR
// constant payloads (`RiscOp::Const`/`ConstTensor`) embed the sealed
// types, and `RiscOp` derives serde for the on-disk context/stdlib
// caches and the `WireDag` surface. The privacy contract survives the
// wire: `Deserialize` routes every inbound value through finalize
// (finalize-on-decode), and the reduced-width float images are
// round-trip-validated, so a corrupt or hand-forged payload is a LOUD
// decode error, never a silently renormalized value. Integer families
// travel exact at width; f16/bf16 travel as their exact f64 images
// (every half value is exactly representable in f64).
// ---------------------------------------------------------------------

/// Wire mirror of [`ScalarValue`]. Private: the only way in or out is
/// the serde impls below.
#[derive(serde::Serialize, serde::Deserialize)]
enum ScalarWire {
    F64(f64),
    F32(f32),
    /// Exact f64 image of the stored half value.
    F16(f64),
    /// Exact f64 image of the stored bfloat value.
    Bf16(f64),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    Bool(bool),
}

impl serde::Serialize for ScalarValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match self.bits {
            Bits::F64(v) => ScalarWire::F64(v),
            Bits::F32(v) => ScalarWire::F32(v),
            Bits::F16(v) => ScalarWire::F16(f64::from(v)),
            Bits::Bf16(v) => ScalarWire::Bf16(f64::from(v)),
            Bits::I64(v) => ScalarWire::I64(v),
            Bits::I32(v) => ScalarWire::I32(v),
            Bits::I16(v) => ScalarWire::I16(v),
            Bits::I8(v) => ScalarWire::I8(v),
            Bits::Bool(v) => ScalarWire::Bool(v),
        };
        wire.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ScalarValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let wire = ScalarWire::deserialize(deserializer)?;
        let value = match wire {
            ScalarWire::F64(v) => ScalarValue { bits: Bits::F64(v) },
            ScalarWire::F32(v) => ScalarValue { bits: Bits::F32(v) },
            ScalarWire::F16(image) => {
                let half = half::f16::from_f64(image);
                if f64::from(half) != image && !image.is_nan() {
                    return Err(D::Error::custom(format!(
                        "f16 wire image {image} is not an exact f16 value; \
                         refusing to renormalize a corrupt payload \
                         (chelis#729 section C3 finalize-on-decode)"
                    )));
                }
                ScalarValue {
                    bits: Bits::F16(half),
                }
            }
            ScalarWire::Bf16(image) => {
                let half = half::bf16::from_f64(image);
                if f64::from(half) != image && !image.is_nan() {
                    return Err(D::Error::custom(format!(
                        "bf16 wire image {image} is not an exact bf16 value; \
                         refusing to renormalize a corrupt payload \
                         (chelis#729 section C3 finalize-on-decode)"
                    )));
                }
                ScalarValue {
                    bits: Bits::Bf16(half),
                }
            }
            ScalarWire::I64(v) => ScalarValue { bits: Bits::I64(v) },
            ScalarWire::I32(v) => ScalarValue { bits: Bits::I32(v) },
            ScalarWire::I16(v) => ScalarValue { bits: Bits::I16(v) },
            ScalarWire::I8(v) => ScalarValue { bits: Bits::I8(v) },
            ScalarWire::Bool(v) => ScalarValue {
                bits: Bits::Bool(v),
            },
        };
        Ok(value)
    }
}

/// Wire mirror of [`TensorStorage`]. Private, same discipline as
/// [`ScalarWire`].
#[derive(serde::Serialize, serde::Deserialize)]
enum StorageWire {
    F64(Vec<f64>),
    F32(Vec<f32>),
    /// Exact f64 images of the stored half values.
    F16(Vec<f64>),
    /// Exact f64 images of the stored bfloat values.
    Bf16(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    I8(Vec<i8>),
    Bool(Vec<bool>),
}

impl serde::Serialize for TensorStorage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match &self.buf {
            Buf::F64(v) => StorageWire::F64(v.clone()),
            Buf::F32(v) => StorageWire::F32(v.clone()),
            Buf::F16(v) => StorageWire::F16(v.iter().map(|&h| f64::from(h)).collect()),
            Buf::Bf16(v) => StorageWire::Bf16(v.iter().map(|&h| f64::from(h)).collect()),
            Buf::I64(v) => StorageWire::I64(v.clone()),
            Buf::I32(v) => StorageWire::I32(v.clone()),
            Buf::I16(v) => StorageWire::I16(v.clone()),
            Buf::I8(v) => StorageWire::I8(v.clone()),
            Buf::Bool(v) => StorageWire::Bool(v.iter().map(|&b| b != 0).collect()),
        };
        wire.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for TensorStorage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let wire = StorageWire::deserialize(deserializer)?;
        let buf = match wire {
            StorageWire::F64(v) => Buf::F64(v),
            StorageWire::F32(v) => Buf::F32(v),
            StorageWire::F16(images) => {
                let mut out = Vec::with_capacity(images.len());
                for image in images {
                    let half = half::f16::from_f64(image);
                    if f64::from(half) != image && !image.is_nan() {
                        return Err(D::Error::custom(format!(
                            "f16 wire image {image} is not an exact f16 value; \
                             refusing to renormalize a corrupt payload \
                             (chelis#729 section C3 finalize-on-decode)"
                        )));
                    }
                    out.push(half);
                }
                Buf::F16(out)
            }
            StorageWire::Bf16(images) => {
                let mut out = Vec::with_capacity(images.len());
                for image in images {
                    let half = half::bf16::from_f64(image);
                    if f64::from(half) != image && !image.is_nan() {
                        return Err(D::Error::custom(format!(
                            "bf16 wire image {image} is not an exact bf16 value; \
                             refusing to renormalize a corrupt payload \
                             (chelis#729 section C3 finalize-on-decode)"
                        )));
                    }
                    out.push(half);
                }
                Buf::Bf16(out)
            }
            StorageWire::I64(v) => Buf::I64(v),
            StorageWire::I32(v) => Buf::I32(v),
            StorageWire::I16(v) => Buf::I16(v),
            StorageWire::I8(v) => Buf::I8(v),
            StorageWire::Bool(v) => Buf::Bool(v.into_iter().map(u8::from).collect()),
        };
        Ok(TensorStorage { buf })
    }
}

/// Bulk finalize: one monomorphized loop per dtype, never per-element
/// dynamic dispatch (the section C5 performance contract). Traps on the
/// first offending element.
pub fn finalize_tensor(
    op: &'static str,
    prim: Prim,
    raw: RawTensor,
) -> Result<TensorStorage, NumericTrap> {
    let buf = match prim {
        Prim::F64 => Buf::F64(match raw {
            RawTensor::Float(v) => v,
            RawTensor::Int(v) => v.into_iter().map(|i| i as f64).collect(),
        }),
        Prim::F32 => Buf::F32(match raw {
            RawTensor::Float(v) => v.into_iter().map(|x| x as f32).collect(),
            RawTensor::Int(v) => v.into_iter().map(|i| i as f32).collect(),
        }),
        Prim::F16 => Buf::F16(match raw {
            RawTensor::Float(v) => v.into_iter().map(half::f16::from_f64).collect(),
            RawTensor::Int(v) => v
                .into_iter()
                .map(|i| half::f16::from_f64(i as f64))
                .collect(),
        }),
        Prim::Bf16 => Buf::Bf16(match raw {
            RawTensor::Float(v) => v.into_iter().map(half::bf16::from_f64).collect(),
            RawTensor::Int(v) => v.into_iter().map(bf16_from_i64_rne).collect(),
        }),
        Prim::Int8 => Buf::I8(int_buf(op, prim, raw)?),
        Prim::Int16 => Buf::I16(int_buf(op, prim, raw)?),
        Prim::Int32 => Buf::I32(int_buf(op, prim, raw)?),
        Prim::Int64 => Buf::I64(int_buf(op, prim, raw)?),
        Prim::Bool => {
            let wides = match raw {
                RawTensor::Int(v) => v,
                RawTensor::Float(v) => v
                    .into_iter()
                    .map(|x| int_wide(op, prim, RawScalar::Float(x)))
                    .collect::<Result<Vec<i64>, NumericTrap>>()?,
            };
            let mut out = Vec::with_capacity(wides.len());
            for w in wides {
                match w {
                    0 => out.push(0u8),
                    1 => out.push(1u8),
                    _ => return Err(NumericTrap::Domain { op, prim }),
                }
            }
            Buf::Bool(out)
        }
        Prim::F8e4m3 => panic!(
            "finalize_tensor: f8e4m3 is not in the active dtype set \
             (spec/04-type-system.md section 1.1.1); the checker rejects it, \
             so no numeric buffer can reach finalize at it (op {op})"
        ),
        Prim::String => panic!(
            "finalize_tensor: string is not a numeric dtype and has no \
             finalize semantics (op {op})"
        ),
    };
    Ok(TensorStorage { buf })
}

/// Shared integer-width bulk finalize: exact wide reads, per-width range
/// check, narrow store.
fn int_buf<T>(op: &'static str, prim: Prim, raw: RawTensor) -> Result<Vec<T>, NumericTrap>
where
    T: TryFrom<i64>,
{
    let wides = match raw {
        RawTensor::Int(v) => v,
        RawTensor::Float(v) => v
            .into_iter()
            .map(|x| int_wide(op, prim, RawScalar::Float(x)))
            .collect::<Result<Vec<i64>, NumericTrap>>()?,
    };
    let (lo, hi) = prim
        .integer_range()
        .expect("integer prims carry an integer_range");
    let mut out = Vec::with_capacity(wides.len());
    for w in wides {
        if w < lo || w > hi {
            return Err(NumericTrap::Overflow { op, prim });
        }
        match T::try_from(w) {
            Ok(t) => out.push(t),
            Err(_) => return Err(NumericTrap::Overflow { op, prim }),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fin(prim: Prim, x: f64) -> Result<ScalarValue, NumericTrap> {
        finalize_scalar("test_op", prim, RawScalar::Float(x))
    }

    fn fin_i(prim: Prim, i: i64) -> Result<ScalarValue, NumericTrap> {
        finalize_scalar("test_op", prim, RawScalar::Int(i))
    }

    // ---- section C1 row: f64 (identity; specials preserved) ----

    #[test]
    fn f64_finalize_is_identity_and_preserves_specials() {
        assert_eq!(fin(Prim::F64, 0.1).unwrap().as_f64_lossy(), 0.1);
        let two_p53 = 9007199254740992.0;
        assert_eq!(fin(Prim::F64, two_p53).unwrap().as_f64_lossy(), two_p53);
        assert!(fin(Prim::F64, f64::NAN).unwrap().as_f64_lossy().is_nan());
        assert_eq!(
            fin(Prim::F64, f64::INFINITY).unwrap().as_f64_lossy(),
            f64::INFINITY
        );
        assert_eq!(
            fin(Prim::F64, f64::NEG_INFINITY).unwrap().as_f64_lossy(),
            f64::NEG_INFINITY
        );
        let z = fin(Prim::F64, -0.0).unwrap().as_f64_lossy();
        assert_eq!(z, 0.0);
        assert!(z.is_sign_negative(), "-0.0 must keep its sign");
        assert_eq!(fin_i(Prim::F64, 3).unwrap().as_f64_lossy(), 3.0);
    }

    // ---- section C1 row: f32 (RNE to 24-bit; overflow to inf) ----

    #[test]
    fn f32_finalize_rounds_to_nearest_even() {
        // The audit's add case: f32(0.1) + f32(0.2) computed wide in f64,
        // rounded ONCE, is the correct f32 sum.
        let wide = 0.1f32 as f64 + 0.2f32 as f64;
        let got = fin(Prim::F32, wide).unwrap();
        assert_eq!(got.prim(), Prim::F32);
        assert_eq!(got.as_f64_lossy(), 0.30000001192092896);
        // 2^24 + 1 is the first non-representable integer: ties-to-even
        // rounds down to 2^24.
        assert_eq!(
            fin(Prim::F32, 16777217.0).unwrap().as_f64_lossy(),
            16777216.0
        );
        // 2^24 + 3 rounds up to 2^24 + 4 (nearest, no tie).
        assert_eq!(
            fin(Prim::F32, 16777219.0).unwrap().as_f64_lossy(),
            16777220.0
        );
    }

    #[test]
    fn f32_finalize_overflows_to_signed_infinity_and_keeps_specials() {
        assert_eq!(fin(Prim::F32, 1e39).unwrap().as_f64_lossy(), f64::INFINITY);
        assert_eq!(
            fin(Prim::F32, -1e39).unwrap().as_f64_lossy(),
            f64::NEG_INFINITY
        );
        assert!(fin(Prim::F32, f64::NAN).unwrap().as_f64_lossy().is_nan());
        let z = fin(Prim::F32, -0.0).unwrap().as_f64_lossy();
        assert!(z == 0.0 && z.is_sign_negative());
    }

    #[test]
    fn f32_finalize_handles_subnormals() {
        // f32 min subnormal is 2^-149. Half of it ties to even (zero);
        // anything above half rounds up to the subnormal itself.
        let min_sub = 2f64.powi(-149);
        assert_eq!(fin(Prim::F32, min_sub).unwrap().as_f64_lossy(), min_sub);
        assert_eq!(fin(Prim::F32, min_sub / 2.0).unwrap().as_f64_lossy(), 0.0);
        assert_eq!(
            fin(Prim::F32, min_sub * 0.75).unwrap().as_f64_lossy(),
            min_sub
        );
    }

    // ---- section C1 row: f16 (RNE to 11-bit incl. subnormals) ----

    #[test]
    fn f16_finalize_rounds_to_nearest_even() {
        // 2048 + 1 = 2049 is unrepresentable; ties-to-even rounds DOWN to
        // 2048 (the narrow_dtype_matrix per-op lock's cell).
        assert_eq!(fin(Prim::F16, 2049.0).unwrap().as_f64_lossy(), 2048.0);
        // 2050 is representable and must survive (negative parity: the
        // rounding must not flatten representable neighbors).
        assert_eq!(fin(Prim::F16, 2050.0).unwrap().as_f64_lossy(), 2050.0);
        // 2051 ties between 2050 and 2052; even mantissa wins (2052).
        assert_eq!(fin(Prim::F16, 2051.0).unwrap().as_f64_lossy(), 2052.0);
        // The f16(0.1)^2 product cell from the scalar lock.
        let wide = f64::from(half::f16::from_f64(0.1)) * f64::from(half::f16::from_f64(0.1));
        assert_eq!(
            fin(Prim::F16, wide).unwrap().as_f64_lossy(),
            0.0099945068359375
        );
    }

    #[test]
    fn f16_finalize_overflows_to_infinity_at_the_ieee_boundary() {
        // The locked cell: mul(65504f16, 2f16) = inf.
        assert_eq!(
            fin(Prim::F16, 131008.0).unwrap().as_f64_lossy(),
            f64::INFINITY
        );
        // 65520 is the overflow threshold (max finite 65504 + ulp/2 with
        // an even-infinite tie); 65519.999.. still rounds to 65504.
        assert_eq!(fin(Prim::F16, 65519.9).unwrap().as_f64_lossy(), 65504.0);
        assert_eq!(
            fin(Prim::F16, 65520.0).unwrap().as_f64_lossy(),
            f64::INFINITY
        );
        assert_eq!(
            fin(Prim::F16, -65520.0).unwrap().as_f64_lossy(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn f16_finalize_handles_subnormals_and_specials() {
        // f16 min subnormal is 2^-24; half of it ties to even (zero).
        let min_sub = 2f64.powi(-24);
        assert_eq!(fin(Prim::F16, min_sub).unwrap().as_f64_lossy(), min_sub);
        assert_eq!(fin(Prim::F16, min_sub / 2.0).unwrap().as_f64_lossy(), 0.0);
        assert_eq!(
            fin(Prim::F16, min_sub * 0.75).unwrap().as_f64_lossy(),
            min_sub
        );
        assert!(fin(Prim::F16, f64::NAN).unwrap().as_f64_lossy().is_nan());
        let z = fin(Prim::F16, -0.0).unwrap().as_f64_lossy();
        assert!(z == 0.0 && z.is_sign_negative());
    }

    // ---- section C1 row: bf16 (RNE to 8-bit) ----

    #[test]
    fn bf16_finalize_rounds_to_nearest_even() {
        // 257 = 2^8 + 1 is the first non-representable integer; ties-to-
        // even rounds down to 256 (the narrow_dtype_matrix cell).
        assert_eq!(fin(Prim::Bf16, 257.0).unwrap().as_f64_lossy(), 256.0);
        assert_eq!(fin(Prim::Bf16, 0.75).unwrap().as_f64_lossy(), 0.75);
        let wide = f64::from(half::bf16::from_f64(0.1)) * f64::from(half::bf16::from_f64(0.1));
        assert_eq!(
            fin(Prim::Bf16, wide).unwrap().as_f64_lossy(),
            0.010009765625
        );
    }

    #[test]
    fn int64_to_bf16_rounds_once_at_the_target_width() {
        const LOWER: i64 = 4_611_686_018_427_387_904;
        const MIDPOINT: i64 = 4_629_700_416_936_869_888;
        const UPPER: i64 = 4_647_714_815_446_351_872;

        assert_eq!(
            fin_i(Prim::Bf16, MIDPOINT - 1).unwrap().as_f64_lossy(),
            LOWER as f64
        );
        assert_eq!(
            fin_i(Prim::Bf16, MIDPOINT).unwrap().as_f64_lossy(),
            LOWER as f64,
            "the exact midpoint ties to the even lower significand"
        );
        assert_eq!(
            fin_i(Prim::Bf16, MIDPOINT + 1).unwrap().as_f64_lossy(),
            UPPER as f64,
            "the first integer above the midpoint must not double-round through f64"
        );
        assert_eq!(
            fin_i(Prim::Bf16, -(MIDPOINT + 1)).unwrap().as_f64_lossy(),
            -(UPPER as f64)
        );

        let tensor =
            finalize_tensor("test_op", Prim::Bf16, RawTensor::Int(vec![MIDPOINT + 1])).unwrap();
        assert_eq!(tensor.element_f64_lossy(0), UPPER as f64);
    }

    #[test]
    fn bf16_finalize_overflows_to_infinity() {
        // bf16 max finite is ~3.39e38; 1e39 is beyond it.
        assert_eq!(fin(Prim::Bf16, 1e39).unwrap().as_f64_lossy(), f64::INFINITY);
        assert_eq!(
            fin(Prim::Bf16, -1e39).unwrap().as_f64_lossy(),
            f64::NEG_INFINITY
        );
        assert!(fin(Prim::Bf16, f64::NAN).unwrap().as_f64_lossy().is_nan());
    }

    // ---- section C1 rows: integer widths (trap at width; exact in range) ----

    #[test]
    fn integer_finalize_traps_overflow_at_every_width() {
        for (prim, max, min) in [
            (Prim::Int8, i8::MAX as i64, i8::MIN as i64),
            (Prim::Int16, i16::MAX as i64, i16::MIN as i64),
            (Prim::Int32, i32::MAX as i64, i32::MIN as i64),
        ] {
            assert_eq!(
                fin_i(prim, max + 1),
                Err(NumericTrap::Overflow {
                    op: "test_op",
                    prim
                }),
                "{} max + 1 must trap",
                prim.name()
            );
            assert_eq!(
                fin_i(prim, min - 1),
                Err(NumericTrap::Overflow {
                    op: "test_op",
                    prim
                }),
                "{} min - 1 must trap",
                prim.name()
            );
            // Negative parity: the extremes themselves are members and
            // must NOT trap (the just-below-overflow locks).
            assert_eq!(fin_i(prim, max).unwrap().as_i64_exact(), Some(max));
            assert_eq!(fin_i(prim, min).unwrap().as_i64_exact(), Some(min));
        }
        // int64 from an exact i64 wide value can never overflow.
        assert_eq!(
            fin_i(Prim::Int64, i64::MAX).unwrap().as_i64_exact(),
            Some(i64::MAX)
        );
        assert_eq!(
            fin_i(Prim::Int64, i64::MIN).unwrap().as_i64_exact(),
            Some(i64::MIN)
        );
    }

    #[test]
    fn integer_finalize_from_float_requires_integral_and_in_range() {
        // Fractional wide values are Domain traps, not truncations.
        assert_eq!(
            fin(Prim::Int32, 3.5),
            Err(NumericTrap::Domain {
                op: "test_op",
                prim: Prim::Int32
            })
        );
        // Non-finite wide values are Domain traps.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                fin(Prim::Int64, bad),
                Err(NumericTrap::Domain {
                    op: "test_op",
                    prim: Prim::Int64
                })
            );
        }
        // Integral and in range is exact.
        assert_eq!(fin(Prim::Int32, -7.0).unwrap().as_i64_exact(), Some(-7));
        // Integral but out of the width is Overflow.
        assert_eq!(
            fin(Prim::Int8, 200.0),
            Err(NumericTrap::Overflow {
                op: "test_op",
                prim: Prim::Int8
            })
        );
        // 2^63 exactly (the f64 image of i64 overflow) is out of range.
        assert_eq!(
            fin(Prim::Int64, 2f64.powi(63)),
            Err(NumericTrap::Overflow {
                op: "test_op",
                prim: Prim::Int64
            })
        );
        // The largest integral f64 strictly below 2^63 is in range.
        let below = 2f64.powi(63) - 1024.0;
        assert_eq!(
            fin(Prim::Int64, below).unwrap().as_i64_exact(),
            Some(9223372036854774784)
        );
    }

    // ---- section C1 row: bool ({0, 1}; Domain trap otherwise) ----

    #[test]
    fn bool_finalize_accepts_exactly_zero_and_one() {
        assert_eq!(fin_i(Prim::Bool, 0).unwrap().as_bool_exact(), Some(false));
        assert_eq!(fin_i(Prim::Bool, 1).unwrap().as_bool_exact(), Some(true));
        assert_eq!(fin(Prim::Bool, 1.0).unwrap().as_bool_exact(), Some(true));
        assert_eq!(
            fin_i(Prim::Bool, 2),
            Err(NumericTrap::Domain {
                op: "test_op",
                prim: Prim::Bool
            }),
            "bool add-style 1 + 1 = 2 must Domain-trap (chelis#726 value half)"
        );
        assert_eq!(
            fin(Prim::Bool, 0.5),
            Err(NumericTrap::Domain {
                op: "test_op",
                prim: Prim::Bool
            })
        );
        assert_eq!(
            fin_i(Prim::Bool, -1),
            Err(NumericTrap::Domain {
                op: "test_op",
                prim: Prim::Bool
            })
        );
    }

    // ---- section C1 row: f8e4m3 (unreachable by construction) ----

    #[test]
    #[should_panic(expected = "f8e4m3 is not in the active dtype set")]
    fn f8e4m3_finalize_is_not_a_runtime_arm() {
        let _ = fin(Prim::F8e4m3, 1.0);
    }

    #[test]
    #[should_panic(expected = "string is not a numeric dtype")]
    fn string_finalize_is_not_a_runtime_arm() {
        let _ = fin(Prim::String, 1.0);
    }

    // ---- ingress constructors ----

    #[test]
    fn ingress_constructors_domain_check_like_finalize() {
        assert_eq!(
            scalar_from_i64("ingress", Prim::Int8, 127)
                .unwrap()
                .as_i64_exact(),
            Some(127)
        );
        assert_eq!(
            scalar_from_i64("ingress", Prim::Int8, 200),
            Err(NumericTrap::Overflow {
                op: "ingress",
                prim: Prim::Int8
            })
        );
        assert_eq!(
            scalar_from_f64("ingress", Prim::F16, 2049.0)
                .unwrap()
                .as_f64_lossy(),
            2048.0
        );
        // int64 ingress from i64 is total (cannot trap by construction).
        assert_eq!(
            scalar_from_i64("ingress", Prim::Int64, 9007199254740993)
                .unwrap()
                .as_i64_exact(),
            Some(9007199254740993)
        );
    }

    // ---- tensor finalize ----

    #[test]
    fn finalize_tensor_keeps_exact_int64_above_2p53() {
        let s = finalize_tensor(
            "to_tensor",
            Prim::Int64,
            RawTensor::Int(vec![9007199254740993, 1]),
        )
        .unwrap();
        assert_eq!(s.prim(), Prim::Int64);
        assert_eq!(s.to_i64_exact_vec(), Some(vec![9007199254740993, 1]));
        // The named-lossy read collapses, which is exactly why it is
        // named lossy.
        assert_eq!(s.to_f64_lossy_vec(), vec![9007199254740992.0, 1.0]);
    }

    #[test]
    fn finalize_tensor_rounds_float_buffers_at_width() {
        let s = finalize_tensor("add", Prim::F16, RawTensor::Float(vec![2049.0, 0.75])).unwrap();
        assert_eq!(s.to_f64_lossy_vec(), vec![2048.0, 0.75]);
        let s = finalize_tensor(
            "add",
            Prim::F32,
            RawTensor::Float(vec![0.1f32 as f64 + 0.2f32 as f64]),
        )
        .unwrap();
        assert_eq!(s.to_f64_lossy_vec(), vec![0.30000001192092896]);
    }

    #[test]
    fn finalize_tensor_traps_on_the_first_bad_element() {
        assert_eq!(
            finalize_tensor("add", Prim::Int8, RawTensor::Int(vec![1, 200, 3])),
            Err(NumericTrap::Overflow {
                op: "add",
                prim: Prim::Int8
            })
        );
        assert_eq!(
            finalize_tensor("cmplt", Prim::Bool, RawTensor::Int(vec![1, 0, 2])),
            Err(NumericTrap::Domain {
                op: "cmplt",
                prim: Prim::Bool
            })
        );
        // Negative parity: all-in-range buffers never trap.
        assert!(finalize_tensor("add", Prim::Int8, RawTensor::Int(vec![-128, 127])).is_ok());
    }

    #[test]
    fn to_raw_is_exact_per_family() {
        let s = finalize_tensor("t", Prim::Int64, RawTensor::Int(vec![9007199254740993])).unwrap();
        assert_eq!(s.to_raw(), RawTensor::Int(vec![9007199254740993]));
        let s =
            finalize_tensor("t", Prim::F16, RawTensor::Float(vec![0.0099945068359375])).unwrap();
        assert_eq!(s.to_raw(), RawTensor::Float(vec![0.0099945068359375]));
        let s = finalize_tensor("t", Prim::Bool, RawTensor::Int(vec![1, 0])).unwrap();
        assert_eq!(s.to_raw(), RawTensor::Int(vec![1, 0]));
    }

    // ---- the reuse_* element-preserving hole ----

    #[test]
    fn reuse_gather_preserves_elements_bit_for_bit() {
        let s =
            finalize_tensor("t", Prim::Int64, RawTensor::Int(vec![9007199254740993, 5])).unwrap();
        let g = s.reuse_gather(&[1, 0, 0]);
        assert_eq!(
            g.to_i64_exact_vec(),
            Some(vec![5, 9007199254740993, 9007199254740993])
        );
    }

    #[test]
    fn reuse_fill_gather_places_fill_at_uncovered_slots() {
        let s = finalize_tensor("t", Prim::F32, RawTensor::Float(vec![1.5, 2.5])).unwrap();
        let fill = scalar_from_f64("t", Prim::F32, 9.0).unwrap();
        let p = s.reuse_fill_gather(&fill, &[None, Some(0), Some(1), None]);
        assert_eq!(p.to_f64_lossy_vec(), vec![9.0, 1.5, 2.5, 9.0]);
    }

    #[test]
    #[should_panic(expected = "fill dtype")]
    fn reuse_fill_gather_panics_on_dtype_mismatch() {
        let s = finalize_tensor("t", Prim::F32, RawTensor::Float(vec![1.5])).unwrap();
        let fill = scalar_from_f64("t", Prim::F64, 9.0).unwrap();
        let _ = s.reuse_fill_gather(&fill, &[None]);
    }

    #[test]
    fn reuse_overwrite_is_last_write_wins() {
        let t = finalize_tensor("t", Prim::Int8, RawTensor::Int(vec![0, 0, 0])).unwrap();
        let u = finalize_tensor("t", Prim::Int8, RawTensor::Int(vec![7, 9])).unwrap();
        let w = t.reuse_overwrite(&u, [(0usize, 0usize), (0, 1)]);
        assert_eq!(w.to_i64_exact_vec(), Some(vec![9, 0, 0]));
    }

    // ---- Phase 2 closed numeric kernels ([04-NUM-8], section C5) ----

    #[test]
    fn int_binop_is_exact_above_f64_mantissa_and_traps_at_each_width() {
        let exact = int_binop(
            IntBinOp::Add,
            scalar_from_i64("test", Prim::Int64, 9_007_199_254_740_992).unwrap(),
            scalar_from_i64("test", Prim::Int64, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(exact.as_i64_exact(), Some(9_007_199_254_740_993));

        for (prim, max) in [
            (Prim::Int8, i8::MAX as i64),
            (Prim::Int16, i16::MAX as i64),
            (Prim::Int32, i32::MAX as i64),
            (Prim::Int64, i64::MAX),
        ] {
            let err = int_binop(
                IntBinOp::Add,
                scalar_from_i64("test", prim, max).unwrap(),
                scalar_from_i64("test", prim, 1).unwrap(),
            )
            .unwrap_err();
            assert_eq!(
                err,
                NumericKernelError::Trap(NumericTrap::Overflow { op: "add", prim })
            );

            let in_range = int_binop(
                IntBinOp::Add,
                scalar_from_i64("test", prim, max - 1).unwrap(),
                scalar_from_i64("test", prim, 1).unwrap(),
            )
            .unwrap();
            assert_eq!(in_range.as_i64_exact(), Some(max));
        }
    }

    #[test]
    fn int_division_kernels_distinguish_divzero_overflow_and_rounding() {
        let i64v = |value| scalar_from_i64("test", Prim::Int64, value).unwrap();
        assert_eq!(
            int_binop(IntBinOp::FloorDiv, i64v(-7), i64v(3))
                .unwrap()
                .as_i64_exact(),
            Some(-3)
        );
        assert_eq!(
            int_binop(IntBinOp::TruncDiv, i64v(-7), i64v(3))
                .unwrap()
                .as_i64_exact(),
            Some(-2)
        );
        assert_eq!(
            int_binop(IntBinOp::Rem, i64v(-7), i64v(3))
                .unwrap()
                .as_i64_exact(),
            Some(-1)
        );
        assert_eq!(
            int_binop(IntBinOp::TruncDiv, i64v(1), i64v(0)).unwrap_err(),
            NumericKernelError::Trap(NumericTrap::DivZero {
                op: "trunc_div",
                prim: Prim::Int64
            })
        );
        assert_eq!(
            int_binop(IntBinOp::TruncDiv, i64v(i64::MIN), i64v(-1)).unwrap_err(),
            NumericKernelError::Trap(NumericTrap::Overflow {
                op: "trunc_div",
                prim: Prim::Int64
            })
        );
    }

    #[test]
    fn int_unop_traps_on_signed_min_and_keeps_in_range_values_exact() {
        for (prim, min) in [
            (Prim::Int8, i8::MIN as i64),
            (Prim::Int16, i16::MIN as i64),
            (Prim::Int32, i32::MIN as i64),
            (Prim::Int64, i64::MIN),
        ] {
            let value = scalar_from_i64("test", prim, min).unwrap();
            assert_eq!(
                int_unop(IntUnOp::Neg, value).unwrap_err(),
                NumericKernelError::Trap(NumericTrap::Overflow { op: "neg", prim })
            );
            let in_range = scalar_from_i64("test", prim, min + 1).unwrap();
            assert_eq!(
                int_unop(IntUnOp::Abs, in_range).unwrap().as_i64_exact(),
                Some(-(min + 1))
            );
        }
    }

    #[test]
    fn float_kernels_compute_at_declared_arithmetic_width() {
        // This f32 input is a one-ulp witness on macOS: native expf and
        // f64-exp-then-narrow differ. The contract assertion is against the
        // platform f32 operation itself, not a hard-coded libm result.
        let x = f32::from_bits(1_040_209_326);
        let input = scalar_from_f64("test", Prim::F32, f64::from(x)).unwrap();
        let got = float_unop(FloatUnOp::Exp, input).unwrap();
        assert_eq!(got.as_f64_lossy(), f64::from(x.exp()));

        let lhs = scalar_from_f64("test", Prim::F32, f64::from(0.1_f32)).unwrap();
        let rhs = scalar_from_f64("test", Prim::F32, f64::from(0.2_f32)).unwrap();
        let sum = float_binop(FloatBinOp::Add, lhs, rhs).unwrap();
        assert_eq!(sum.as_f64_lossy(), f64::from(0.1_f32 + 0.2_f32));

        for prim in [Prim::F16, Prim::Bf16] {
            let lhs = scalar_from_f64("test", prim, 2048.0).unwrap();
            let rhs = scalar_from_f64("test", prim, 1.0).unwrap();
            let sum = float_binop(FloatBinOp::Add, lhs, rhs).unwrap();
            assert_eq!(sum.prim(), prim);
            assert_eq!(sum.as_f64_lossy(), 2048.0);
        }
    }

    #[test]
    fn kernel_family_and_dtype_mismatches_fail_before_arithmetic() {
        let int = scalar_from_i64("test", Prim::Int32, 1).unwrap();
        let float = scalar_from_f64("test", Prim::F32, 1.0).unwrap();
        assert!(matches!(
            float_binop(FloatBinOp::Add, int, int),
            Err(NumericKernelError::WrongFamily {
                expected: NumericFamily::Float,
                ..
            })
        ));
        assert!(matches!(
            int_binop(IntBinOp::Add, int, float),
            Err(NumericKernelError::WrongFamily {
                expected: NumericFamily::Int,
                ..
            })
        ));

        let wider = scalar_from_i64("test", Prim::Int64, 1).unwrap();
        assert_eq!(
            int_binop(IntBinOp::Add, int, wider).unwrap_err(),
            NumericKernelError::DtypeMismatch {
                op: "add",
                lhs: Prim::Int32,
                rhs: Prim::Int64
            }
        );
    }

    #[test]
    fn comparisons_use_exact_integer_values_and_matching_float_widths() {
        let lo = scalar_from_i64("test", Prim::Int64, 9_007_199_254_740_992).unwrap();
        let hi = scalar_from_i64("test", Prim::Int64, 9_007_199_254_740_993).unwrap();
        assert!(compare_scalars(CompareOp::Lt, lo, hi).unwrap());
        assert!(!compare_scalars(CompareOp::Eq, lo, hi).unwrap());

        let f16_lo = scalar_from_f64("test", Prim::F16, 2048.0).unwrap();
        let f16_same = scalar_from_f64("test", Prim::F16, 2049.0).unwrap();
        assert!(compare_scalars(CompareOp::Eq, f16_lo, f16_same).unwrap());
        assert!(!compare_scalars(CompareOp::Lt, f16_lo, f16_same).unwrap());
    }

    #[test]
    fn tensor_kernels_are_exact_at_width_and_preserve_broadcast_order() {
        let lhs = finalize_tensor(
            "test",
            Prim::Int64,
            RawTensor::Int(vec![9_007_199_254_740_992, 7]),
        )
        .unwrap();
        let rhs = finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![1, 3])).unwrap();
        assert_eq!(
            int_tensor_binop(IntBinOp::Add, &lhs, &rhs)
                .unwrap()
                .to_i64_exact_vec(),
            Some(vec![9_007_199_254_740_993, 10])
        );

        let scalar = scalar_from_i64("test", Prim::Int64, 10).unwrap();
        assert_eq!(
            int_tensor_scalar_binop(IntBinOp::Sub, &rhs, scalar)
                .unwrap()
                .to_i64_exact_vec(),
            Some(vec![-9, -7])
        );
        assert_eq!(
            int_scalar_tensor_binop(IntBinOp::Sub, scalar, &rhs)
                .unwrap()
                .to_i64_exact_vec(),
            Some(vec![9, 7])
        );

        let overflow = finalize_tensor("test", Prim::Int8, RawTensor::Int(vec![0, 127])).unwrap();
        let one = scalar_from_i64("test", Prim::Int8, 1).unwrap();
        assert_eq!(
            int_tensor_scalar_binop(IntBinOp::Add, &overflow, one).unwrap_err(),
            NumericKernelError::Trap(NumericTrap::Overflow {
                op: "add",
                prim: Prim::Int8
            })
        );
    }

    #[test]
    fn tensor_float_and_comparison_kernels_do_not_widen_elements() {
        let x = f32::from_bits(1_040_209_326);
        let input =
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![f64::from(x)])).unwrap();
        assert_eq!(
            float_tensor_unop(FloatUnOp::Exp, &input)
                .unwrap()
                .to_f64_lossy_vec(),
            vec![f64::from(x.exp())]
        );

        let lo = finalize_tensor(
            "test",
            Prim::Int64,
            RawTensor::Int(vec![9_007_199_254_740_992]),
        )
        .unwrap();
        let hi = scalar_from_i64("test", Prim::Int64, 9_007_199_254_740_993).unwrap();
        assert_eq!(
            compare_tensor_scalar(CompareOp::Lt, &lo, hi)
                .unwrap()
                .to_i64_exact_vec(),
            Some(vec![1])
        );
        assert_eq!(
            compare_scalar_tensor(CompareOp::Lt, hi, &lo)
                .unwrap()
                .to_i64_exact_vec(),
            Some(vec![0])
        );
    }

    // ---- trap message shape (section C2; strings freeze at Phase 2) ----

    #[test]
    fn trap_messages_follow_the_branded_shape() {
        assert_eq!(
            NumericTrap::Overflow {
                op: "add",
                prim: Prim::Int8
            }
            .to_string(),
            "numeric trap: overflow in add at int8"
        );
        assert_eq!(
            NumericTrap::Domain {
                op: "cast",
                prim: Prim::Bool
            }
            .to_string(),
            "numeric trap: domain in cast at bool"
        );
        assert_eq!(
            NumericTrap::DivZero {
                op: "trunc_div",
                prim: Prim::Int64
            }
            .to_string(),
            "numeric trap: division by zero in trunc_div at int64"
        );
    }

    // ---- the checked cast ladder (chelis#759 one rule per direction;
    // executed at the chelis#729 rework). Positive AND negative parity
    // per direction, per the repo contract. ----

    #[test]
    fn cast_raw_to_float_finalizes_at_target_width() {
        // 2049 is the first f16-unrepresentable integer; RNE rounds to 2048.
        let v = cast_raw("cast", RawScalar::Float(2049.0), Prim::F16).unwrap();
        assert_eq!(v.as_f64_lossy(), 2048.0);
        // Exact-int source finalizes from the exact integer, same rule.
        let v = cast_raw("cast", RawScalar::Int(2049), Prim::F16).unwrap();
        assert_eq!(v.as_f64_lossy(), 2048.0);
        // int64 above 2^53 to f64 is the LOSSY-BY-DESIGN float direction
        // ([04-NUM-6]): RNE, never a trap.
        let v = cast_raw("cast", RawScalar::Int(9_007_199_254_740_993), Prim::F64).unwrap();
        assert_eq!(v.as_f64_lossy(), 9_007_199_254_740_992.0);
    }

    #[test]
    fn cast_raw_float_overflowing_target_goes_to_infinity_per_ieee() {
        // Float targets finalize per [04-NUM-2]: overflow is the correctly
        // signed infinity, not a trap (65504 is f16::MAX).
        let v = cast_raw("cast", RawScalar::Float(1.0e6), Prim::F16).unwrap();
        assert_eq!(v.as_f64_lossy(), f64::INFINITY);
    }

    #[test]
    fn cast_raw_int_to_narrower_int_in_range_is_exact() {
        let v = cast_raw("cast", RawScalar::Int(127), Prim::Int8).unwrap();
        assert_eq!(v.as_i64_exact(), Some(127));
        let v = cast_raw("cast", RawScalar::Int(-128), Prim::Int8).unwrap();
        assert_eq!(v.as_i64_exact(), Some(-128));
    }

    #[test]
    fn cast_raw_int_to_narrower_int_out_of_range_traps_overflow_not_wrap() {
        // Pre-rework this wrapped two's-complement (300 -> 44). The checked
        // default TRAPS; wrapping is chelis#759's future NAMED form.
        let err = cast_raw("cast", RawScalar::Int(300), Prim::Int8).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at int8");
        let err = cast_raw("cast", RawScalar::Int(-129), Prim::Int8).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at int8");
    }

    #[test]
    fn cast_raw_float_to_int_requires_an_integral_value() {
        let v = cast_raw("cast", RawScalar::Float(3.0), Prim::Int8).unwrap();
        assert_eq!(v.as_i64_exact(), Some(3));

        for fractional in [3.5, -3.5] {
            let err = cast_raw("cast", RawScalar::Float(fractional), Prim::Int8).unwrap_err();
            assert_eq!(err.to_string(), "numeric trap: domain in cast at int8");
        }
    }

    #[test]
    fn cast_raw_float_to_int_out_of_range_traps_overflow_not_saturate() {
        // Pre-rework this saturated (300.0 -> 127). The checked default
        // TRAPS; saturation is chelis#759's future NAMED form.
        let err = cast_raw("cast", RawScalar::Float(300.0), Prim::Int8).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at int8");
        let err = cast_raw("cast", RawScalar::Float(1.0e300), Prim::Int64).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at int64");
    }

    #[test]
    fn cast_raw_non_finite_float_to_int_traps_domain() {
        let err = cast_raw("cast", RawScalar::Float(f64::NAN), Prim::Int32).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: domain in cast at int32");
        let err = cast_raw("cast", RawScalar::Float(f64::INFINITY), Prim::Int32).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: domain in cast at int32");
    }

    #[test]
    fn cast_raw_to_bool_is_strict_zero_one_membership() {
        assert!(
            !cast_raw("cast", RawScalar::Int(0), Prim::Bool)
                .unwrap()
                .as_bool_exact()
                .unwrap()
        );
        assert!(
            cast_raw("cast", RawScalar::Int(1), Prim::Bool)
                .unwrap()
                .as_bool_exact()
                .unwrap()
        );
        assert!(
            cast_raw("cast", RawScalar::Float(1.0), Prim::Bool)
                .unwrap()
                .as_bool_exact()
                .unwrap()
        );
    }

    #[test]
    fn cast_raw_to_bool_rejects_everything_outside_zero_one() {
        // Pre-rework any nonzero encoded true. The checked default is
        // STRICT {0, 1}; counting idioms cast explicitly instead.
        for raw in [
            RawScalar::Int(2),
            RawScalar::Int(-1),
            RawScalar::Float(0.5),
            RawScalar::Float(f64::NAN),
        ] {
            let err = cast_raw("cast", raw, Prim::Bool).unwrap_err();
            assert_eq!(err.to_string(), "numeric trap: domain in cast at bool");
        }
    }

    #[test]
    fn cast_scalar_same_prim_is_identity_and_routes_exact_wides() {
        let v = scalar_from_i64("t", Prim::Int64, 9_007_199_254_740_993).unwrap();
        // Identity short-circuit.
        let same = cast_scalar("cast", v, Prim::Int64).unwrap();
        assert_eq!(same.as_i64_exact(), Some(9_007_199_254_740_993));
        // Exact integer wide: above 2^53 an int64 -> int64-family cast must
        // not launder through f64 (that laundering is the chelis#684 bug
        // shape this module exists to end).
        let narrowed = cast_scalar("cast", v, Prim::Int32).unwrap_err();
        assert_eq!(
            narrowed.to_string(),
            "numeric trap: overflow in cast at int32"
        );
    }
}
