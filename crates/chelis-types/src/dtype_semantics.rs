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
//! ## Frozen trap strings
//!
//! The public constants below are the exact [04-NUM-9] / section C2 message
//! grammar. The operation slot names the canonical primitive whose numeric
//! kernel raised the trap after lowering; composed evaluation forwards the
//! message unchanged.

use crate::activation::{ActivationGraph, DerivedActivation, lower_activation};
use crate::observation::ElementRef;
use crate::types::Prim;

/// Frozen prefix shared by every [04-NUM-9] numeric-trap diagnostic.
pub const NUMERIC_TRAP_PREFIX: &str = "numeric trap: ";

/// Why a key buffer or key scalar refuses every numeric read: a key has no
/// arithmetic, comparison, or cast (spec/04 §1.1), and the IR verifier keeps
/// keys out of every numeric operation, so reaching one is a compiler defect.
const KEY_HAS_NO_NUMERIC_READING: &str = "a random key has no numeric reading; the IR verifier keeps keys out of every numeric operation";
/// Frozen spelling of the [04-NUM-9] overflow kind.
pub const NUMERIC_TRAP_OVERFLOW_KIND: &str = "overflow";
/// Frozen spelling of the [04-NUM-9] domain kind.
pub const NUMERIC_TRAP_DOMAIN_KIND: &str = "domain";
/// Frozen spelling of the [04-NUM-9] division-by-zero kind.
pub const NUMERIC_TRAP_DIV_ZERO_KIND: &str = "division by zero";
/// Frozen separator before the canonical raising primitive name.
pub const NUMERIC_TRAP_OPERATION_SEPARATOR: &str = " in ";
/// Frozen separator before the finalized dtype name.
pub const NUMERIC_TRAP_DTYPE_SEPARATOR: &str = " at ";

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
                write!(
                    f,
                    "{NUMERIC_TRAP_PREFIX}{NUMERIC_TRAP_OVERFLOW_KIND}\
                     {NUMERIC_TRAP_OPERATION_SEPARATOR}{op}{NUMERIC_TRAP_DTYPE_SEPARATOR}{}",
                    prim.name()
                )
            }
            NumericTrap::Domain { op, prim } => {
                write!(
                    f,
                    "{NUMERIC_TRAP_PREFIX}{NUMERIC_TRAP_DOMAIN_KIND}\
                     {NUMERIC_TRAP_OPERATION_SEPARATOR}{op}{NUMERIC_TRAP_DTYPE_SEPARATOR}{}",
                    prim.name()
                )
            }
            NumericTrap::DivZero { op, prim } => {
                write!(
                    f,
                    "{NUMERIC_TRAP_PREFIX}{NUMERIC_TRAP_DIV_ZERO_KIND}\
                     {NUMERIC_TRAP_OPERATION_SEPARATOR}{op}{NUMERIC_TRAP_DTYPE_SEPARATOR}{}",
                    prim.name()
                )
            }
        }
    }
}

impl std::error::Error for NumericTrap {}

impl NumericTrap {
    /// Recognize one complete [04-NUM-9] trap line at the boundary where an
    /// evaluator's legacy text failure becomes a structured diagnostic.
    /// Context belongs on other lines; a CLI prefix or appended hint is not
    /// part of this grammar.
    pub fn is_canonical_line(line: &str) -> bool {
        let Some(body) = line.strip_prefix(NUMERIC_TRAP_PREFIX) else {
            return false;
        };
        let operation_and_prim = [
            NUMERIC_TRAP_OVERFLOW_KIND,
            NUMERIC_TRAP_DOMAIN_KIND,
            NUMERIC_TRAP_DIV_ZERO_KIND,
        ]
        .into_iter()
        .find_map(|kind| {
            body.strip_prefix(kind)?
                .strip_prefix(NUMERIC_TRAP_OPERATION_SEPARATOR)
        });
        let Some((operation, prim)) =
            operation_and_prim.and_then(|rest| rest.rsplit_once(NUMERIC_TRAP_DTYPE_SEPARATOR))
        else {
            return false;
        };
        !operation.is_empty()
            && operation
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            && (Prim::parse_name(prim).is_some() || prim == "key")
    }
}

#[cfg(test)]
mod numeric_trap_line_tests {
    use super::NumericTrap;

    #[test]
    fn canonical_line_excludes_cli_decoration_and_context() {
        assert!(NumericTrap::is_canonical_line(
            "numeric trap: domain in concat at i64"
        ));
        assert!(NumericTrap::is_canonical_line(
            "numeric trap: division by zero in floor_div at i32"
        ));
        for line in [
            "error: numeric trap: domain in concat at i64",
            "numeric trap: domain in concat at i64; hint: retry",
            "numeric trap: domain in concat at i64 trailing",
            "numeric trap: other in concat at i64",
            "numeric trap: domain in concat at imaginary",
            "numeric trap: domain in bad op at i64",
        ] {
            assert!(!NumericTrap::is_canonical_line(line), "{line}");
        }
    }
}

/// One elementwise trap paired with the row-major flat index that produced
/// it ([04-NUM-15]). Parallel lanes reduce this value by `flat_index`; a
/// sequential lane may use the same reduction without changing semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedTrapCandidate {
    pub flat_index: usize,
    pub trap: NumericTrap,
}

impl IndexedTrapCandidate {
    pub fn earlier(self, other: Self) -> Self {
        if self.flat_index <= other.flat_index {
            self
        } else {
            other
        }
    }
}

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

/// Closed semantic action for one checked `cast` source/target pair.
///
/// This is the executable projection of [04-NUM-14].  Backends may choose a
/// representation-specific implementation of the action, but they do not
/// maintain a separate roster of legal pairs and they may select
/// [`Self::Identity`] only on the exact same-`Prim` diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedCastKind {
    Identity,
    ExactToInteger,
    FloatToInteger,
    ExactToFloat,
    FloatToFloat,
    ExactToBool,
    FloatToBool,
}

/// Construction failure for a checked-cast plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedCastPlanError {
    UnsupportedSource(Prim),
    UnsupportedTarget(Prim),
}

impl std::fmt::Display for CheckedCastPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSource(prim) => write!(
                f,
                "checked cast source `{}` is not an active scalar dtype",
                prim.name()
            ),
            Self::UnsupportedTarget(prim) => write!(
                f,
                "checked cast target `{}` is not an active scalar dtype",
                prim.name()
            ),
        }
    }
}

impl std::error::Error for CheckedCastPlanError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckedCastFamily {
    Float,
    SignedInteger,
    Bool,
}

/// Exhaustive conversion plan for one active checked-cast pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedCastPlan {
    source: Prim,
    target: Prim,
    kind: CheckedCastKind,
}

impl CheckedCastPlan {
    pub fn new(source: Prim, target: Prim) -> Result<Self, CheckedCastPlanError> {
        let source_family = match source {
            Prim::F64 | Prim::F32 | Prim::F16 | Prim::Bf16 => CheckedCastFamily::Float,
            Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => {
                CheckedCastFamily::SignedInteger
            }
            Prim::Bool => CheckedCastFamily::Bool,
            Prim::F8e4m3 | Prim::String | Prim::Key => {
                return Err(CheckedCastPlanError::UnsupportedSource(source));
            }
        };
        let target_family = match target {
            Prim::F64 | Prim::F32 | Prim::F16 | Prim::Bf16 => CheckedCastFamily::Float,
            Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => {
                CheckedCastFamily::SignedInteger
            }
            Prim::Bool => CheckedCastFamily::Bool,
            Prim::F8e4m3 | Prim::String | Prim::Key => {
                return Err(CheckedCastPlanError::UnsupportedTarget(target));
            }
        };

        let kind = if source == target {
            CheckedCastKind::Identity
        } else {
            match (source_family, target_family) {
                (CheckedCastFamily::Float, CheckedCastFamily::Float) => {
                    CheckedCastKind::FloatToFloat
                }
                (CheckedCastFamily::Float, CheckedCastFamily::SignedInteger) => {
                    CheckedCastKind::FloatToInteger
                }
                (CheckedCastFamily::Float, CheckedCastFamily::Bool) => CheckedCastKind::FloatToBool,
                (CheckedCastFamily::SignedInteger, CheckedCastFamily::Float)
                | (CheckedCastFamily::Bool, CheckedCastFamily::Float) => {
                    CheckedCastKind::ExactToFloat
                }
                (CheckedCastFamily::SignedInteger, CheckedCastFamily::SignedInteger)
                | (CheckedCastFamily::Bool, CheckedCastFamily::SignedInteger) => {
                    CheckedCastKind::ExactToInteger
                }
                (CheckedCastFamily::SignedInteger, CheckedCastFamily::Bool) => {
                    CheckedCastKind::ExactToBool
                }
                (CheckedCastFamily::Bool, CheckedCastFamily::Bool) => {
                    unreachable!("bool -> bool is the exact identity pair")
                }
            }
        };

        Ok(Self {
            source,
            target,
            kind,
        })
    }

    pub fn source(self) -> Prim {
        self.source
    }

    pub fn target(self) -> Prim {
        self.target
    }

    pub fn kind(self) -> CheckedCastKind {
        self.kind
    }

    /// Apply this plan to a sealed scalar. A source mismatch is an internal
    /// caller violation: the plan and the value are produced from the same
    /// checked type edge and must never disagree.
    pub fn cast_scalar(
        self,
        op: &'static str,
        value: ScalarValue,
    ) -> Result<ScalarValue, NumericTrap> {
        assert_eq!(
            value.prim(),
            self.source,
            "checked-cast plan source does not match the sealed scalar"
        );
        if self.kind == CheckedCastKind::Identity {
            return Ok(value);
        }
        let raw = match value.as_i64_exact() {
            Some(i) => RawScalar::Int(i),
            None => RawScalar::Float(value.as_f64_lossy()),
        };
        finalize_scalar(op, self.target, raw)
    }

    fn cast_raw(self, op: &'static str, raw: RawScalar) -> Result<ScalarValue, NumericTrap> {
        let family_matches = match (self.source, raw) {
            (
                Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 | Prim::Bool,
                RawScalar::Int(_),
            ) => true,
            (Prim::F64 | Prim::F32 | Prim::F16 | Prim::Bf16, RawScalar::Float(_)) => true,
            (
                Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64
                | Prim::Bool
                | Prim::F64
                | Prim::F32
                | Prim::F16
                | Prim::Bf16,
                _,
            ) => false,
            (Prim::F8e4m3 | Prim::String | Prim::Key, _) => {
                unreachable!("unsupported source cannot construct a checked-cast plan")
            }
        };
        assert!(family_matches, "checked-cast raw source family mismatch");
        finalize_scalar(op, self.target, raw)
    }
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
    /// A random key ([05-RNG-2]): an opaque word, never a number.
    Key(RandomKey),
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
            Bits::Key(_) => Prim::Key,
        }
    }

    /// A key as a scalar value. A key has no literal, so this is the only
    /// way a key scalar is formed: from a key some key operation produced.
    pub fn from_key(key: RandomKey) -> Self {
        Self {
            bits: Bits::Key(key),
        }
    }

    /// The key payload, or `None` for every numeric and bool dtype.
    pub fn as_key(&self) -> Option<RandomKey> {
        match self.bits {
            Bits::Key(key) => Some(key),
            _ => None,
        }
    }

    /// Widen to f64. Exact for every float width, bool, and integers up
    /// to 2^53; EXPLICITLY LOSSY for i64 magnitudes above 2^53 (the
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
            Bits::Key(_) => panic!(
                "as_f64_lossy: a random key has no numeric reading; the IR verifier \
                 keeps keys out of every numeric operation"
            ),
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
            Bits::F16(_) | Bits::Bf16(_) | Bits::F32(_) | Bits::F64(_) | Bits::Key(_) => None,
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
            | Bits::F64(_)
            | Bits::Key(_) => None,
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
            Bits::Key(_) => {
                panic!("element_ref: a random key has no observation form (chelis#2413)")
            }
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
    Floor,
    Ceil,
    Round,
}

impl IntUnOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
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

/// Extrema selector used by the direct forward and reverse typed kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatExtremaOp {
    Max,
    Min,
}

/// Operand whose cotangent an extrema adjoint materializes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremaOperand {
    Left,
    Right,
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

    const fn is_activation(self) -> bool {
        matches!(self, Self::Relu | Self::Sigmoid | Self::Silu | Self::Gelu)
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

/// Closed reduction operation set for the Phase 2 dtype-keyed tensor
/// kernel. The variants carry the canonical primitive identity, so a trap
/// cannot be mislabeled by a consumer-provided string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorReduceOp {
    /// Global `sum`; its explicit accumulator is part of the IR contract.
    /// `result` is either that accumulator dtype or the source dtype after
    /// the host surface's specified final narrowing for f16/bf16.
    Sum {
        accumulator: Prim,
        result: Prim,
    },
    ProdReduce,
    MaxReduce,
    MinReduce,
    ReduceWindowSum,
    ReduceWindowMean,
    ReduceWindowMax,
    ReduceWindowMin,
}

impl TensorReduceOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sum { .. } => "sum",
            Self::ProdReduce => "prod_reduce",
            Self::MaxReduce => "max_reduce",
            Self::MinReduce => "min_reduce",
            Self::ReduceWindowSum => "reduce_window_sum",
            Self::ReduceWindowMean => "reduce_window_mean",
            Self::ReduceWindowMax => "reduce_window_max",
            Self::ReduceWindowMin => "reduce_window_min",
        }
    }
}

/// Closed exact-comparison reduction set. The result is an i64 index;
/// operands remain at their stored dtype throughout comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgReduceOp {
    Argmax,
    Argmin,
}

/// Closed reducer set for the reverse-mode window adjoint. Shape and window
/// membership remain consumer-owned; all value selection and overlap-add
/// arithmetic crosses the dtype-keyed kernel boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceWindowGradOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl ReduceWindowGradOp {
    pub const fn name(self) -> &'static str {
        "reduce_window_grad"
    }

    const fn forward_op(self) -> TensorReduceOp {
        match self {
            Self::Sum => TensorReduceOp::ReduceWindowSum,
            Self::Mean => TensorReduceOp::ReduceWindowMean,
            Self::Max => TensorReduceOp::ReduceWindowMax,
            Self::Min => TensorReduceOp::ReduceWindowMin,
        }
    }
}

impl ArgReduceOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Argmax => "argmax_reduce",
            Self::Argmin => "argmin_reduce",
        }
    }

    const fn compare(self) -> CompareOp {
        match self {
            Self::Argmax => CompareOp::Gt,
            Self::Argmin => CompareOp::Lt,
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
    InvalidReductionSignature {
        op: &'static str,
        input: Prim,
        accumulator: Prim,
        result: Prim,
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
            Self::InvalidReductionSignature {
                op,
                input,
                accumulator,
                result,
            } => write!(
                f,
                "numeric reduction {op} has invalid dtype signature: input {}, accumulator {}, result {}",
                input.name(),
                accumulator.name(),
                result.name()
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
            IntUnOp::Floor | IntUnOp::Ceil | IntUnOp::Round => Some($value),
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

fn select_float_max_first<T: Copy + PartialOrd>(lhs: T, rhs: T, is_nan: impl Fn(T) -> bool) -> T {
    if is_nan(lhs) {
        lhs
    } else if is_nan(rhs) || rhs > lhs {
        rhs
    } else {
        lhs
    }
}

fn select_float_min_first<T: Copy + PartialOrd>(lhs: T, rhs: T, is_nan: impl Fn(T) -> bool) -> T {
    if is_nan(lhs) {
        lhs
    } else if is_nan(rhs) || rhs < lhs {
        rhs
    } else {
        lhs
    }
}

fn extrema_selects_left<T: Copy + PartialOrd>(
    op: FloatExtremaOp,
    lhs: T,
    rhs: T,
    is_nan: impl Fn(T) -> bool,
) -> bool {
    if is_nan(lhs) {
        true
    } else if is_nan(rhs) {
        false
    } else {
        match op {
            FloatExtremaOp::Max => rhs <= lhs,
            FloatExtremaOp::Min => rhs >= lhs,
        }
    }
}

/// [04-NUM-2]: floating arithmetic and numeric conversion finalize every NaN
/// they produce to the dtype's canonical quiet NaN, dropping the input
/// payload and sign, so NaN bits never depend on the host ISA (arm64 yields
/// `0x7fc00000` for an invalid operation, x86 `0xffc00000`, and both
/// propagate input payloads). Only selection, which [05-OP-40] declares
/// bit-preserving, bypasses this. The f16 and bf16 lanes compute at f32 and
/// narrow the canonical f32 NaN to their own canonical encoding.
fn canonical_nan_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        value
    }
}

fn canonical_nan_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        value
    }
}

fn apply_float_binop_f32(op: FloatBinOp, lhs: f32, rhs: f32) -> f32 {
    let value = match op {
        FloatBinOp::Add => lhs + rhs,
        FloatBinOp::Sub => lhs - rhs,
        FloatBinOp::Mul => lhs * rhs,
        FloatBinOp::Div => lhs / rhs,
        FloatBinOp::FloorDiv => (lhs / rhs).floor(),
        FloatBinOp::Max => select_float_max_first(lhs, rhs, f32::is_nan),
        FloatBinOp::Min => select_float_min_first(lhs, rhs, f32::is_nan),
    };
    match op {
        FloatBinOp::Max | FloatBinOp::Min => value,
        _ => canonical_nan_f32(value),
    }
}

fn apply_float_binop_f64(op: FloatBinOp, lhs: f64, rhs: f64) -> f64 {
    let value = match op {
        FloatBinOp::Add => lhs + rhs,
        FloatBinOp::Sub => lhs - rhs,
        FloatBinOp::Mul => lhs * rhs,
        FloatBinOp::Div => lhs / rhs,
        FloatBinOp::FloorDiv => (lhs / rhs).floor(),
        FloatBinOp::Max => select_float_max_first(lhs, rhs, f64::is_nan),
        FloatBinOp::Min => select_float_min_first(lhs, rhs, f64::is_nan),
    };
    match op {
        FloatBinOp::Max | FloatBinOp::Min => value,
        _ => canonical_nan_f64(value),
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
    let bits = match (lhs.bits, rhs.bits) {
        (Bits::F64(lhs), Bits::F64(rhs)) => Bits::F64(apply_float_binop_f64(op, lhs, rhs)),
        (Bits::F32(lhs), Bits::F32(rhs)) => Bits::F32(apply_float_binop_f32(op, lhs, rhs)),
        (Bits::F16(lhs), Bits::F16(rhs)) => Bits::F16(match op {
            FloatBinOp::Max => select_float_max_first(lhs, rhs, half::f16::is_nan),
            FloatBinOp::Min => select_float_min_first(lhs, rhs, half::f16::is_nan),
            _ => half::f16::from_f32(apply_float_binop_f32(op, lhs.to_f32(), rhs.to_f32())),
        }),
        (Bits::Bf16(lhs), Bits::Bf16(rhs)) => Bits::Bf16(match op {
            FloatBinOp::Max => select_float_max_first(lhs, rhs, half::bf16::is_nan),
            FloatBinOp::Min => select_float_min_first(lhs, rhs, half::bf16::is_nan),
            _ => half::bf16::from_f32(apply_float_binop_f32(op, lhs.to_f32(), rhs.to_f32())),
        }),
        _ => unreachable!("family and dtype checks make the float match exhaustive"),
    };
    Ok(ScalarValue { bits })
}

fn apply_float_unop_f32(op: FloatUnOp, value: f32) -> f32 {
    canonical_nan_f32(match op {
        FloatUnOp::Neg => -value,
        FloatUnOp::Recip => value.recip(),
        FloatUnOp::Exp => chelis_crmath::exp_f32(value),
        FloatUnOp::Log => chelis_crmath::log_f32(value),
        FloatUnOp::Sin => chelis_crmath::sin_f32(value),
        FloatUnOp::Sqrt => value.sqrt(),
        FloatUnOp::Cos => chelis_crmath::cos_f32(value),
        FloatUnOp::Tan => chelis_crmath::tan_f32(value),
        FloatUnOp::Atan => chelis_crmath::atan_f32(value),
        FloatUnOp::Tanh => chelis_crmath::tanh_f32(value),
        FloatUnOp::Abs => value.abs(),
        FloatUnOp::Floor => value.floor(),
        FloatUnOp::Ceil => value.ceil(),
        FloatUnOp::Round => value.round_ties_even(),
        FloatUnOp::Relu | FloatUnOp::Sigmoid | FloatUnOp::Silu | FloatUnOp::Gelu => {
            unreachable!("derived activations decompose before the unary primitive kernel")
        }
    })
}

fn apply_float_unop_f64(op: FloatUnOp, value: f64) -> f64 {
    canonical_nan_f64(match op {
        FloatUnOp::Neg => -value,
        FloatUnOp::Recip => value.recip(),
        FloatUnOp::Exp => chelis_crmath::exp_f64(value),
        FloatUnOp::Log => chelis_crmath::log_f64(value),
        FloatUnOp::Sin => chelis_crmath::sin_f64(value),
        FloatUnOp::Sqrt => value.sqrt(),
        FloatUnOp::Cos => chelis_crmath::cos_f64(value),
        FloatUnOp::Tan => chelis_crmath::tan_f64(value),
        FloatUnOp::Atan => chelis_crmath::atan_f64(value),
        FloatUnOp::Tanh => chelis_crmath::tanh_f64(value),
        FloatUnOp::Abs => value.abs(),
        FloatUnOp::Floor => value.floor(),
        FloatUnOp::Ceil => value.ceil(),
        FloatUnOp::Round => value.round_ties_even(),
        FloatUnOp::Relu | FloatUnOp::Sigmoid | FloatUnOp::Silu | FloatUnOp::Gelu => {
            unreachable!("derived activations decompose before the unary primitive kernel")
        }
    })
}

fn activation_constant(
    op: FloatUnOp,
    prim: Prim,
    value: f64,
) -> Result<ScalarValue, NumericKernelError> {
    scalar_from_f64(op.name(), prim, value).map_err(Into::into)
}

#[cfg(test)]
thread_local! {
    static SCALAR_ACTIVATION_CALL_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

fn float_activation(op: FloatUnOp, value: ScalarValue) -> Result<ScalarValue, NumericKernelError> {
    #[cfg(test)]
    SCALAR_ACTIVATION_CALL_COUNT.with(|count| count.set(count.get() + 1));

    let prim = value.prim();
    if op == FloatUnOp::Relu {
        return float_binop(FloatBinOp::Max, value, activation_constant(op, prim, 0.0)?);
    }
    let activation = DerivedActivation::from_float_unop(op)
        .expect("float_activation requires an activation selector");
    lower_activation(&mut ScalarActivationGraph { op, prim }, activation, value)
}

/// Evaluates each section 3.3 primitive on finalized scalars as it is reached.
struct ScalarActivationGraph {
    op: FloatUnOp,
    prim: Prim,
}

impl ActivationGraph for ScalarActivationGraph {
    type Value = ScalarValue;
    type Error = NumericKernelError;

    fn constant(&mut self, value: f64) -> Result<ScalarValue, NumericKernelError> {
        activation_constant(self.op, self.prim, value)
    }

    fn unary(&mut self, op: FloatUnOp, x: ScalarValue) -> Result<ScalarValue, NumericKernelError> {
        float_unop(op, x)
    }

    fn binary(
        &mut self,
        op: FloatBinOp,
        lhs: ScalarValue,
        rhs: ScalarValue,
    ) -> Result<ScalarValue, NumericKernelError> {
        float_binop(op, lhs, rhs)
    }
}

/// Perform one float unary primitive at the dtype's arithmetic width and
/// finalize once into its storage width. Derived activations execute their
/// specified Tier-2 composition, finalizing each constituent primitive.
pub fn float_unop(op: FloatUnOp, value: ScalarValue) -> Result<ScalarValue, NumericKernelError> {
    require_family(op.name(), value, NumericFamily::Float)?;
    if op.is_activation() {
        return float_activation(op, value);
    }
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
/// particular, i64 never crosses f64 and half values compare only after
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
/// cannot represent exact i64 above 2^53, chelis#684).
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
    /// Random keys ([05-RNG-2]); constructed only by the key kernels.
    Key(Vec<RandomKey>),
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
    /// Opaque random keys.
    Key(&'a [RandomKey]),
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
            Buf::Key(_) => Prim::Key,
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
            Buf::Key(v) => v.len(),
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
            Buf::Key(v) => StorageView::Key(v),
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
            Buf::Key(_) => panic!("to_raw: {KEY_HAS_NO_NUMERIC_READING}"),
        }
    }

    /// Widen every element to f64. Exact except for i64 magnitudes
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
            Buf::Key(_) => panic!("to_f64_lossy_vec: {KEY_HAS_NO_NUMERIC_READING}"),
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
            Buf::F64(_) | Buf::F32(_) | Buf::F16(_) | Buf::Bf16(_) | Buf::Key(_) => None,
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
            Buf::Key(_) => panic!("element_f64_lossy: {KEY_HAS_NO_NUMERIC_READING}"),
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
            Buf::Key(v) => Bits::Key(v[index]),
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
            Buf::Key(v) => Buf::Key(pick(v, indices)),
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
            (Buf::Key(v), Bits::Key(f)) => Buf::Key(place(v, f, map)),
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
            (Buf::Key(d), Buf::Key(s)) => write(d, s, writes),
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

fn round_shift_even_u64(value: u64, shift: u32) -> u64 {
    match shift {
        0 => value,
        1..=63 => {
            let quotient = value >> shift;
            let remainder = value & ((1_u64 << shift) - 1);
            let halfway = 1_u64 << (shift - 1);
            quotient + u64::from(remainder > halfway || (remainder == halfway && quotient & 1 == 1))
        }
        64 => u64::from(value > (1_u64 << 63)),
        _ => 0,
    }
}

/// Round one binary64 value directly into a binary16-shaped IEEE encoding.
///
/// The source significand remains intact until the one target-width
/// round-to-nearest-ties-to-even step. In particular, this must not delegate
/// through f32 or a converter which truncates the low binary64 significand
/// bits before rounding: doing so loses which side of an f16/bf16 midpoint the
/// original f64 occupied ([04-NUM-14]).
fn f64_to_ieee16_bits(value: f64, exponent_bits: u32, mantissa_bits: u32, bias: i32) -> u16 {
    let source = value.to_bits();
    let sign = ((source >> 48) & 0x8000) as u16;
    let source_exponent = ((source >> 52) & 0x7ff) as u32;
    let source_mantissa = source & 0x000f_ffff_ffff_ffff;
    let target_exponent_max = (1_u32 << exponent_bits) - 1;
    let target_exponent_bits = (target_exponent_max << mantissa_bits) as u16;

    if source_exponent == 0x7ff {
        if source_mantissa == 0 {
            return sign | target_exponent_bits;
        }
        // Every NaN remains a NaN. A canonical quiet payload also avoids
        // architecture-dependent signaling-NaN behavior at this boundary.
        return sign | target_exponent_bits | (1_u16 << (mantissa_bits - 1));
    }
    if source_exponent == 0 {
        // Every finite binary64 subnormal is below half of the least f16 or
        // bf16 subnormal. This includes signed zero and preserves its sign.
        return sign;
    }

    let mut exponent = source_exponent as i32 - 1023;
    let significand = (1_u64 << 52) | source_mantissa;
    let minimum_exponent = 1 - bias;
    let maximum_exponent = target_exponent_max as i32 - 1 - bias;
    if exponent > maximum_exponent {
        return sign | target_exponent_bits;
    }

    if exponent >= minimum_exponent {
        let mut rounded = round_shift_even_u64(significand, 52 - mantissa_bits);
        if rounded == (1_u64 << (mantissa_bits + 1)) {
            rounded >>= 1;
            exponent += 1;
            if exponent > maximum_exponent {
                return sign | target_exponent_bits;
            }
        }
        let target_exponent = ((exponent + bias) as u16) << mantissa_bits;
        let target_mantissa = (rounded & ((1_u64 << mantissa_bits) - 1)) as u16;
        return sign | target_exponent | target_mantissa;
    }

    let shift = (52 - mantissa_bits) + (minimum_exponent - exponent) as u32;
    let rounded = round_shift_even_u64(significand, shift);
    sign | rounded as u16
}

fn i64_to_ieee16_bits(value: i64, exponent_bits: u32, mantissa_bits: u32, bias: i32) -> u16 {
    let sign = if value.is_negative() { 0x8000 } else { 0 };
    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return sign;
    }

    let mut exponent = 63 - magnitude.leading_zeros();
    let target_exponent_max = (1_u32 << exponent_bits) - 1;
    let maximum_exponent = target_exponent_max as i32 - 1 - bias;
    if exponent as i32 > maximum_exponent {
        return sign | ((target_exponent_max << mantissa_bits) as u16);
    }

    let mut rounded = if exponent > mantissa_bits {
        round_shift_even_u64(magnitude, exponent - mantissa_bits)
    } else {
        magnitude << (mantissa_bits - exponent)
    };
    if rounded == (1_u64 << (mantissa_bits + 1)) {
        rounded >>= 1;
        exponent += 1;
        if exponent as i32 > maximum_exponent {
            return sign | ((target_exponent_max << mantissa_bits) as u16);
        }
    }
    let target_exponent = ((exponent as i32 + bias) as u16) << mantissa_bits;
    let target_mantissa = (rounded & ((1_u64 << mantissa_bits) - 1)) as u16;
    sign | target_exponent | target_mantissa
}

/// Canonical one-step binary64-to-f16 conversion ([04-NUM-14]).
pub fn f16_from_f64_rne(value: f64) -> half::f16 {
    half::f16::from_bits(f64_to_ieee16_bits(value, 5, 10, 15))
}

/// Canonical one-step binary64-to-bf16 conversion ([04-NUM-14]).
pub fn bf16_from_f64_rne(value: f64) -> half::bf16 {
    half::bf16::from_bits(f64_to_ieee16_bits(value, 8, 7, 127))
}

fn f16_from_i64_rne(value: i64) -> half::f16 {
    half::f16::from_bits(i64_to_ieee16_bits(value, 5, 10, 15))
}

/// Round an exact i64 directly to bfloat16, once, with target-width
/// round-to-nearest-ties-to-even. Converting through f64 first is not
/// equivalent above 2^53: it can erase which side of a bf16 midpoint the
/// exact integer occupies and then manufacture a tie (chelis#729 Phase 1,
/// [04-NUM-14]).
fn bf16_from_i64_rne(value: i64) -> half::bf16 {
    half::bf16::from_bits(i64_to_ieee16_bits(value, 8, 7, 127))
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
            IntUnOp::Floor | IntUnOp::Ceil | IntUnOp::Round => Ok($values.to_vec()),
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

fn float_vec_binop_f32<T: Copy + PartialOrd>(
    op: FloatBinOp,
    lhs: &[T],
    rhs: &[T],
    to_f32: impl Fn(T) -> f32,
    from_f32: impl Fn(f32) -> T,
) -> Vec<T> {
    match op {
        FloatBinOp::Add
        | FloatBinOp::Sub
        | FloatBinOp::Mul
        | FloatBinOp::Div
        | FloatBinOp::FloorDiv => zip_map(lhs, rhs, |lhs, rhs| {
            from_f32(apply_float_binop_f32(op, to_f32(lhs), to_f32(rhs)))
        }),
        FloatBinOp::Max => zip_map(lhs, rhs, |lhs, rhs| {
            select_float_max_first(lhs, rhs, |value| to_f32(value).is_nan())
        }),
        FloatBinOp::Min => zip_map(lhs, rhs, |lhs, rhs| {
            select_float_min_first(lhs, rhs, |value| to_f32(value).is_nan())
        }),
    }
}

fn float_vec_binop_f64(op: FloatBinOp, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    match op {
        FloatBinOp::Add
        | FloatBinOp::Sub
        | FloatBinOp::Mul
        | FloatBinOp::Div
        | FloatBinOp::FloorDiv => zip_map(lhs, rhs, |lhs, rhs| apply_float_binop_f64(op, lhs, rhs)),
        FloatBinOp::Max => zip_map(lhs, rhs, |lhs, rhs| {
            select_float_max_first(lhs, rhs, f64::is_nan)
        }),
        FloatBinOp::Min => zip_map(lhs, rhs, |lhs, rhs| {
            select_float_min_first(lhs, rhs, f64::is_nan)
        }),
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

/// Materialize one operand cotangent for the exact [05-OP-40] extrema
/// selection rule. Each output element is the complete incoming cotangent
/// when that operand was selected, otherwise exact positive zero at the
/// stored dtype. Selection reads the forward operands without re-encoding
/// them, including NaN payloads and signed-zero ties.
pub fn float_extrema_adjoint(
    op: FloatExtremaOp,
    operand: ExtremaOperand,
    lhs: &TensorStorage,
    rhs: &TensorStorage,
    cotangent: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    const NAME: &str = "extrema_adjoint";
    for value in [lhs, rhs, cotangent] {
        if !value.prim().is_float() {
            return Err(NumericKernelError::WrongFamily {
                op: NAME,
                expected: NumericFamily::Float,
                actual: value.prim(),
            });
        }
    }
    require_same_storage_shape(NAME, lhs, rhs)?;
    require_same_storage_shape(NAME, lhs, cotangent)?;

    macro_rules! route {
        ($lhs:expr, $rhs:expr, $g:expr, $variant:ident, $zero:expr, $is_nan:expr) => {{
            let values = $lhs
                .iter()
                .copied()
                .zip($rhs.iter().copied())
                .zip($g.iter().copied())
                .map(|((lhs, rhs), g)| {
                    let left = extrema_selects_left(op, lhs, rhs, $is_nan);
                    if (operand == ExtremaOperand::Left) == left {
                        g
                    } else {
                        $zero
                    }
                })
                .collect();
            Ok(TensorStorage {
                buf: Buf::$variant(values),
            })
        }};
    }

    match (&lhs.buf, &rhs.buf, &cotangent.buf) {
        (Buf::F16(lhs), Buf::F16(rhs), Buf::F16(g)) => {
            route!(lhs, rhs, g, F16, half::f16::ZERO, half::f16::is_nan)
        }
        (Buf::Bf16(lhs), Buf::Bf16(rhs), Buf::Bf16(g)) => {
            route!(lhs, rhs, g, Bf16, half::bf16::ZERO, half::bf16::is_nan)
        }
        (Buf::F32(lhs), Buf::F32(rhs), Buf::F32(g)) => {
            route!(lhs, rhs, g, F32, 0.0_f32, f32::is_nan)
        }
        (Buf::F64(lhs), Buf::F64(rhs), Buf::F64(g)) => {
            route!(lhs, rhs, g, F64, 0.0_f64, f64::is_nan)
        }
        _ => unreachable!("same-dtype checks make float extrema adjoint exhaustive"),
    }
}

/// Evaluate [05-OP-43] ReLU without re-encoding retained elements. Negative
/// values become exact positive zero; positive values, both signed zeros, and
/// NaNs retain their stored bits exactly.
pub fn float_relu(input: &TensorStorage) -> Result<TensorStorage, NumericKernelError> {
    const NAME: &str = "relu";
    if !input.prim().is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: NAME,
            expected: NumericFamily::Float,
            actual: input.prim(),
        });
    }

    macro_rules! apply {
        ($values:expr, $variant:ident, $zero:expr) => {{
            let values = $values
                .iter()
                .copied()
                .map(|value| if value < $zero { $zero } else { value })
                .collect();
            Ok(TensorStorage {
                buf: Buf::$variant(values),
            })
        }};
    }

    match &input.buf {
        Buf::F16(values) => apply!(values, F16, half::f16::ZERO),
        Buf::Bf16(values) => apply!(values, Bf16, half::bf16::ZERO),
        Buf::F32(values) => apply!(values, F32, 0.0_f32),
        Buf::F64(values) => apply!(values, F64, 0.0_f64),
        _ => unreachable!("family check makes float relu exhaustive"),
    }
}

/// Evaluate [05-OP-43]'s dedicated ReLU adjoint. The complete incoming
/// cotangent is selected only where `0 < input`; every other row constructs
/// exact positive zero, including both input zeros and NaN. Selection rather
/// than multiplication is required so rejected infinite and NaN cotangents do
/// not contaminate zero rows.
pub fn float_relu_adjoint(
    input: &TensorStorage,
    cotangent: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    const NAME: &str = "relu_adjoint";
    for value in [input, cotangent] {
        if !value.prim().is_float() {
            return Err(NumericKernelError::WrongFamily {
                op: NAME,
                expected: NumericFamily::Float,
                actual: value.prim(),
            });
        }
    }
    require_same_storage_shape(NAME, input, cotangent)?;

    macro_rules! route {
        ($input:expr, $g:expr, $variant:ident, $zero:expr) => {{
            let values = $input
                .iter()
                .copied()
                .zip($g.iter().copied())
                .map(|(input, g)| if $zero < input { g } else { $zero })
                .collect();
            Ok(TensorStorage {
                buf: Buf::$variant(values),
            })
        }};
    }

    match (&input.buf, &cotangent.buf) {
        (Buf::F16(input), Buf::F16(g)) => route!(input, g, F16, half::f16::ZERO),
        (Buf::Bf16(input), Buf::Bf16(g)) => route!(input, g, Bf16, half::bf16::ZERO),
        (Buf::F32(input), Buf::F32(g)) => route!(input, g, F32, 0.0_f32),
        (Buf::F64(input), Buf::F64(g)) => route!(input, g, F64, 0.0_f64),
        _ => unreachable!("same-dtype checks make float relu adjoint exhaustive"),
    }
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
                .map(|value| from_f32(canonical_nan_f32($body(to_f32(value)))))
                .collect()
        };
    }
    match op {
        FloatUnOp::Neg => map!(|x: f32| -x),
        FloatUnOp::Recip => map!(f32::recip),
        FloatUnOp::Exp => map!(chelis_crmath::exp_f32),
        FloatUnOp::Log => map!(chelis_crmath::log_f32),
        FloatUnOp::Sin => map!(chelis_crmath::sin_f32),
        FloatUnOp::Sqrt => map!(f32::sqrt),
        FloatUnOp::Cos => map!(chelis_crmath::cos_f32),
        FloatUnOp::Tan => map!(chelis_crmath::tan_f32),
        FloatUnOp::Atan => map!(chelis_crmath::atan_f32),
        FloatUnOp::Tanh => map!(chelis_crmath::tanh_f32),
        FloatUnOp::Abs => map!(f32::abs),
        FloatUnOp::Floor => map!(f32::floor),
        FloatUnOp::Ceil => map!(f32::ceil),
        FloatUnOp::Round => map!(f32::round_ties_even),
        FloatUnOp::Relu | FloatUnOp::Sigmoid | FloatUnOp::Silu | FloatUnOp::Gelu => {
            unreachable!("derived activations decompose before the unary tensor kernel")
        }
    }
}

fn float_vec_unop_f64(op: FloatUnOp, values: &[f64]) -> Vec<f64> {
    macro_rules! map {
        ($body:expr) => {
            values
                .iter()
                .copied()
                .map(|value| canonical_nan_f64($body(value)))
                .collect()
        };
    }
    match op {
        FloatUnOp::Neg => map!(|x: f64| -x),
        FloatUnOp::Recip => map!(f64::recip),
        FloatUnOp::Exp => map!(chelis_crmath::exp_f64),
        FloatUnOp::Log => map!(chelis_crmath::log_f64),
        FloatUnOp::Sin => map!(chelis_crmath::sin_f64),
        FloatUnOp::Sqrt => map!(f64::sqrt),
        FloatUnOp::Cos => map!(chelis_crmath::cos_f64),
        FloatUnOp::Tan => map!(chelis_crmath::tan_f64),
        FloatUnOp::Atan => map!(chelis_crmath::atan_f64),
        FloatUnOp::Tanh => map!(chelis_crmath::tanh_f64),
        FloatUnOp::Abs => map!(f64::abs),
        FloatUnOp::Floor => map!(f64::floor),
        FloatUnOp::Ceil => map!(f64::ceil),
        FloatUnOp::Round => map!(f64::round_ties_even),
        FloatUnOp::Relu | FloatUnOp::Sigmoid | FloatUnOp::Silu | FloatUnOp::Gelu => {
            unreachable!("derived activations decompose before the unary tensor kernel")
        }
    }
}

trait ActivationElement: Copy {
    fn constant(value: f64) -> Self;
    fn unary(self, op: FloatUnOp) -> Self;
    fn binary(self, op: FloatBinOp, rhs: Self) -> Self;
}

// Each activation step is the same primitive kernel the scalar lane uses, so
// the tensor lane cannot drift from it, including [04-NUM-2]'s NaN
// finalization.
macro_rules! impl_native_activation_element {
    ($ty:ty, $unop:ident, $binop:ident) => {
        impl ActivationElement for $ty {
            fn constant(value: f64) -> Self {
                value as Self
            }

            fn unary(self, op: FloatUnOp) -> Self {
                $unop(op, self)
            }

            fn binary(self, op: FloatBinOp, rhs: Self) -> Self {
                $binop(op, self, rhs)
            }
        }
    };
}

impl_native_activation_element!(f32, apply_float_unop_f32, apply_float_binop_f32);
impl_native_activation_element!(f64, apply_float_unop_f64, apply_float_binop_f64);

macro_rules! impl_reduced_activation_element {
    ($ty:ty) => {
        impl ActivationElement for $ty {
            fn constant(value: f64) -> Self {
                Self::from_f64(value)
            }

            fn unary(self, op: FloatUnOp) -> Self {
                Self::from_f32(apply_float_unop_f32(op, self.to_f32()))
            }

            fn binary(self, op: FloatBinOp, rhs: Self) -> Self {
                Self::from_f32(apply_float_binop_f32(op, self.to_f32(), rhs.to_f32()))
            }
        }
    };
}

impl_reduced_activation_element!(half::f16);
impl_reduced_activation_element!(half::bf16);

/// Evaluates each section 3.3 primitive on one tensor element as it is reached.
struct ElementActivationGraph<T>(std::marker::PhantomData<T>);

impl<T: ActivationElement> ActivationGraph for ElementActivationGraph<T> {
    type Value = T;
    type Error = std::convert::Infallible;

    fn constant(&mut self, value: f64) -> Result<T, Self::Error> {
        Ok(T::constant(value))
    }

    fn unary(&mut self, op: FloatUnOp, x: T) -> Result<T, Self::Error> {
        Ok(x.unary(op))
    }

    fn binary(&mut self, op: FloatBinOp, lhs: T, rhs: T) -> Result<T, Self::Error> {
        Ok(lhs.binary(op, rhs))
    }
}

fn float_vec_activation<T: ActivationElement>(op: FloatUnOp, values: &[T]) -> Vec<T> {
    let activation = DerivedActivation::from_float_unop(op)
        .expect("float_vec_activation requires a section 3.3 activation selector");
    let mut graph = ElementActivationGraph(std::marker::PhantomData);
    values
        .iter()
        .map(
            |&value| match lower_activation(&mut graph, activation, value) {
                Ok(result) => result,
                Err(never) => match never {},
            },
        )
        .collect()
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
    if op == FloatUnOp::Relu {
        return float_relu(value);
    }
    if op.is_activation() {
        let buf = match &value.buf {
            Buf::F64(values) => Buf::F64(float_vec_activation(op, values)),
            Buf::F32(values) => Buf::F32(float_vec_activation(op, values)),
            Buf::F16(values) => Buf::F16(float_vec_activation(op, values)),
            Buf::Bf16(values) => Buf::Bf16(float_vec_activation(op, values)),
            _ => unreachable!("family check makes the float buffer exhaustive"),
        };
        return Ok(TensorStorage { buf });
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

fn reduction_arithmetic_prim(prim: Prim) -> Option<Prim> {
    match prim {
        Prim::F16 | Prim::Bf16 => Some(Prim::F32),
        Prim::F32 | Prim::F64 | Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => Some(prim),
        Prim::Bool | Prim::String | Prim::Key | Prim::F8e4m3 => None,
    }
}

fn valid_sum_accumulator(input: Prim, accumulator: Prim) -> bool {
    matches!(
        (input, accumulator),
        (Prim::F16 | Prim::Bf16, Prim::F32 | Prim::F64)
            | (Prim::F32, Prim::F32 | Prim::F64)
            | (Prim::F64, Prim::F64)
            | (Prim::Int8 | Prim::Int16, Prim::Int32 | Prim::Int64)
            | (Prim::Int32, Prim::Int32 | Prim::Int64)
            | (Prim::Int64, Prim::Int64)
    )
}

fn reduction_signature(
    op: TensorReduceOp,
    input: Prim,
) -> Result<(Prim, Prim), NumericKernelError> {
    let (accumulator, result) = match op {
        TensorReduceOp::Sum {
            accumulator,
            result,
        } => (accumulator, result),
        TensorReduceOp::ProdReduce
        | TensorReduceOp::MaxReduce
        | TensorReduceOp::MinReduce
        | TensorReduceOp::ReduceWindowSum
        | TensorReduceOp::ReduceWindowMax
        | TensorReduceOp::ReduceWindowMin => {
            let accumulator = reduction_arithmetic_prim(input).ok_or(
                NumericKernelError::InvalidReductionSignature {
                    op: op.name(),
                    input,
                    accumulator: input,
                    result: input,
                },
            )?;
            (accumulator, input)
        }
        TensorReduceOp::ReduceWindowMean => {
            let accumulator = reduction_arithmetic_prim(input).ok_or(
                NumericKernelError::InvalidReductionSignature {
                    op: op.name(),
                    input,
                    accumulator: input,
                    result: input,
                },
            )?;
            if !input.is_float() {
                return Err(NumericKernelError::InvalidReductionSignature {
                    op: op.name(),
                    input,
                    accumulator,
                    result: input,
                });
            }
            (accumulator, input)
        }
    };

    let same_family = (input.is_integer() && accumulator.is_integer() && result.is_integer())
        || (input.is_float() && accumulator.is_float() && result.is_float());
    let valid = match op {
        TensorReduceOp::Sum { .. } => {
            valid_sum_accumulator(input, accumulator)
                && same_family
                && if input.is_integer() {
                    result == accumulator
                } else {
                    result == input || result == accumulator
                }
        }
        _ => {
            same_family && reduction_arithmetic_prim(input) == Some(accumulator) && result == input
        }
    };
    if !valid {
        return Err(NumericKernelError::InvalidReductionSignature {
            op: op.name(),
            input,
            accumulator,
            result,
        });
    }
    Ok((accumulator, result))
}

fn scalar_at_reduction_width(
    op: TensorReduceOp,
    input: &TensorStorage,
    index: usize,
    accumulator: Prim,
) -> Result<ScalarValue, NumericKernelError> {
    let value = input.scalar_at(index);
    if value.prim() == accumulator {
        return Ok(value);
    }
    if value.prim().is_integer() && accumulator.is_integer() {
        return scalar_from_i64(
            op.name(),
            accumulator,
            value
                .as_i64_exact()
                .expect("integer-family reduction input reads exactly"),
        )
        .map_err(Into::into);
    }
    if value.prim().is_float() && accumulator.is_float() {
        return scalar_from_f64(op.name(), accumulator, value.as_f64_lossy()).map_err(Into::into);
    }
    Err(NumericKernelError::WrongFamily {
        op: op.name(),
        expected: if accumulator.is_integer() {
            NumericFamily::Int
        } else {
            NumericFamily::Float
        },
        actual: value.prim(),
    })
}

fn reduction_seed(
    op: TensorReduceOp,
    prim: Prim,
    integer: i64,
    float: f64,
) -> Result<ScalarValue, NumericKernelError> {
    if prim.is_integer() {
        scalar_from_i64(op.name(), prim, integer).map_err(Into::into)
    } else {
        scalar_from_f64(op.name(), prim, float).map_err(Into::into)
    }
}

macro_rules! checked_reduce_int {
    ($op:expr, $prim:expr, $lhs:expr, $rhs:expr, $method:ident, $variant:ident) => {{
        let value = $lhs.$method($rhs).ok_or(NumericTrap::Overflow {
            op: $op.name(),
            prim: $prim,
        })?;
        Ok(ScalarValue {
            bits: Bits::$variant(value),
        })
    }};
}

fn reduction_add(
    op: TensorReduceOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, NumericKernelError> {
    require_same_dtype(op.name(), lhs, rhs)?;
    match (lhs.bits, rhs.bits) {
        (Bits::I8(lhs), Bits::I8(rhs)) => {
            checked_reduce_int!(op, Prim::Int8, lhs, rhs, checked_add, I8)
        }
        (Bits::I16(lhs), Bits::I16(rhs)) => {
            checked_reduce_int!(op, Prim::Int16, lhs, rhs, checked_add, I16)
        }
        (Bits::I32(lhs), Bits::I32(rhs)) => {
            checked_reduce_int!(op, Prim::Int32, lhs, rhs, checked_add, I32)
        }
        (Bits::I64(lhs), Bits::I64(rhs)) => {
            checked_reduce_int!(op, Prim::Int64, lhs, rhs, checked_add, I64)
        }
        (Bits::F32(lhs), Bits::F32(rhs)) => Ok(ScalarValue {
            bits: Bits::F32(lhs + rhs),
        }),
        (Bits::F64(lhs), Bits::F64(rhs)) => Ok(ScalarValue {
            bits: Bits::F64(lhs + rhs),
        }),
        _ => unreachable!("validated reduction accumulators are exact-width int or f32/f64"),
    }
}

fn reduction_mul(
    op: TensorReduceOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, NumericKernelError> {
    require_same_dtype(op.name(), lhs, rhs)?;
    match (lhs.bits, rhs.bits) {
        (Bits::I8(lhs), Bits::I8(rhs)) => {
            checked_reduce_int!(op, Prim::Int8, lhs, rhs, checked_mul, I8)
        }
        (Bits::I16(lhs), Bits::I16(rhs)) => {
            checked_reduce_int!(op, Prim::Int16, lhs, rhs, checked_mul, I16)
        }
        (Bits::I32(lhs), Bits::I32(rhs)) => {
            checked_reduce_int!(op, Prim::Int32, lhs, rhs, checked_mul, I32)
        }
        (Bits::I64(lhs), Bits::I64(rhs)) => {
            checked_reduce_int!(op, Prim::Int64, lhs, rhs, checked_mul, I64)
        }
        (Bits::F32(lhs), Bits::F32(rhs)) => Ok(ScalarValue {
            bits: Bits::F32(lhs * rhs),
        }),
        (Bits::F64(lhs), Bits::F64(rhs)) => Ok(ScalarValue {
            bits: Bits::F64(lhs * rhs),
        }),
        _ => unreachable!("validated reduction accumulators are exact-width int or f32/f64"),
    }
}

fn reduction_div_float(
    op: TensorReduceOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, NumericKernelError> {
    require_same_dtype(op.name(), lhs, rhs)?;
    let prim = lhs.prim();
    match (lhs.bits, rhs.bits) {
        (Bits::F32(lhs), Bits::F32(rhs)) => Ok(ScalarValue {
            bits: Bits::F32(lhs / rhs),
        }),
        (Bits::F64(lhs), Bits::F64(rhs)) => Ok(ScalarValue {
            bits: Bits::F64(lhs / rhs),
        }),
        _ => Err(NumericKernelError::InvalidReductionSignature {
            op: op.name(),
            input: prim,
            accumulator: prim,
            result: prim,
        }),
    }
}

fn reduction_extreme(
    op: TensorReduceOp,
    lhs: ScalarValue,
    rhs: ScalarValue,
    take_max: bool,
) -> Result<ScalarValue, NumericKernelError> {
    require_same_dtype(op.name(), lhs, rhs)?;
    let compare = if take_max {
        CompareOp::Gt
    } else {
        CompareOp::Lt
    };
    if compare_scalars(compare, rhs, lhs)? {
        Ok(rhs)
    } else {
        Ok(lhs)
    }
}

fn scalar_is_nan(value: ScalarValue) -> bool {
    value.prim().is_float() && value.as_f64_lossy().is_nan()
}

fn reduce_sum_group(
    op: TensorReduceOp,
    input: &TensorStorage,
    group: &[usize],
    accumulator: Prim,
) -> Result<ScalarValue, NumericKernelError> {
    let leaves = group
        .iter()
        .map(|&index| scalar_at_reduction_width(op, input, index, accumulator))
        .collect::<Result<Vec<_>, _>>()?;
    match checked_adjacent_pair_fold(leaves, |left, right| reduction_add(op, left, right))? {
        Some(value) => Ok(value),
        None => reduction_seed(op, accumulator, 0, 0.0),
    }
}

fn reduce_group(
    op: TensorReduceOp,
    input: &TensorStorage,
    group: &[usize],
    accumulator: Prim,
) -> Result<ScalarValue, NumericKernelError> {
    match op {
        TensorReduceOp::Sum { .. } => reduce_sum_group(op, input, group, accumulator),
        TensorReduceOp::ReduceWindowSum => reduce_sum_group(op, input, group, accumulator),
        TensorReduceOp::ReduceWindowMean => {
            if group.is_empty() {
                return Err(NumericTrap::Domain {
                    op: op.name(),
                    prim: input.prim(),
                }
                .into());
            }
            let sum = reduce_sum_group(op, input, group, accumulator)?;
            let divisor = reduction_seed(op, accumulator, group.len() as i64, group.len() as f64)?;
            reduction_div_float(op, sum, divisor)
        }
        TensorReduceOp::ProdReduce => {
            let mut acc = reduction_seed(op, accumulator, 1, 1.0)?;
            for &index in group {
                acc = reduction_mul(
                    op,
                    acc,
                    scalar_at_reduction_width(op, input, index, accumulator)?,
                )?;
            }
            Ok(acc)
        }
        TensorReduceOp::MaxReduce
        | TensorReduceOp::MinReduce
        | TensorReduceOp::ReduceWindowMax
        | TensorReduceOp::ReduceWindowMin => {
            let Some((&first_index, remaining)) = group.split_first() else {
                return Err(NumericTrap::Domain {
                    op: op.name(),
                    prim: input.prim(),
                }
                .into());
            };
            let take_max = matches!(
                op,
                TensorReduceOp::MaxReduce | TensorReduceOp::ReduceWindowMax
            );
            let mut acc = input.scalar_at(first_index);
            if scalar_is_nan(acc) {
                return Ok(acc);
            }
            for &index in remaining {
                let value = input.scalar_at(index);
                if scalar_is_nan(value) {
                    return Ok(value);
                }
                acc = reduction_extreme(op, acc, value, take_max)?;
            }
            Ok(acc)
        }
    }
}

fn reduction_result_scalar(
    op: TensorReduceOp,
    value: ScalarValue,
    result: Prim,
) -> Result<ScalarValue, NumericKernelError> {
    if value.prim() == result {
        return Ok(value);
    }
    if value.prim().is_integer() && result.is_integer() {
        scalar_from_i64(
            op.name(),
            result,
            value
                .as_i64_exact()
                .expect("integer accumulator result reads exactly"),
        )
        .map_err(Into::into)
    } else if value.prim().is_float() && result.is_float() {
        scalar_from_f64(op.name(), result, value.as_f64_lossy()).map_err(Into::into)
    } else {
        Err(NumericKernelError::InvalidReductionSignature {
            op: op.name(),
            input: value.prim(),
            accumulator: value.prim(),
            result,
        })
    }
}

/// Fold one level at a time by combining adjacent pairs and carrying an odd
/// final element unchanged. This is the canonical balanced reduction tree for
/// [05-OP-29]; leaf order is exactly the caller-provided group order.
fn checked_adjacent_pair_fold<T, E>(
    mut level: Vec<T>,
    mut combine: impl FnMut(T, T) -> Result<T, E>,
) -> Result<Option<T>, E> {
    while level.len() > 1 {
        let mut source = level.into_iter();
        let mut next = Vec::with_capacity(source.len().div_ceil(2));
        while let Some(left) = source.next() {
            let value = match source.next() {
                Some(right) => combine(left, right)?,
                None => left,
            };
            next.push(value);
        }
        level = next;
    }
    Ok(level.pop())
}

fn checked_count_add(left: i64, right: i64) -> Result<i64, NumericKernelError> {
    left.checked_add(right)
        .ok_or(NumericKernelError::Trap(NumericTrap::Overflow {
            op: "count",
            prim: Prim::Int64,
        }))
}

/// Count true elements in explicitly ordered groups through the closed typed
/// kernel boundary. The input must use exact Bool storage and the result is
/// exact i64. Empty groups produce the specified zero identity.
pub fn count_tensor_groups(
    input: &TensorStorage,
    groups: &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError> {
    if input.prim() != Prim::Bool {
        return Err(NumericKernelError::InvalidReductionSignature {
            op: "count",
            input: input.prim(),
            accumulator: Prim::Int64,
            result: Prim::Int64,
        });
    }

    let mut values = Vec::with_capacity(groups.len());
    for group in groups {
        let leaves = group
            .iter()
            .map(|&index| {
                i64::from(
                    input
                        .scalar_at(index)
                        .as_bool_exact()
                        .expect("sealed Bool storage contains exact bool elements"),
                )
            })
            .collect();
        // This is the [05-OP-29] empty selected-extent identity, not a
        // permissive fallback. Keep the branch explicit and auditable.
        #[allow(
            clippy::manual_unwrap_or,
            clippy::manual_unwrap_or_default,
            reason = "Count's explicit empty-group identity must remain auditable"
        )]
        let value = match checked_adjacent_pair_fold(leaves, checked_count_add)? {
            Some(value) => value,
            None => 0,
        };
        values.push(value);
    }
    finalize_tensor("count", Prim::Int64, RawTensor::Int(values)).map_err(Into::into)
}

/// Reduce explicitly ordered groups of input indices through one closed,
/// dtype-keyed kernel. Callers own shape/index planning; arithmetic order,
/// accumulator width, trap identity, and final storage are enforced here.
pub fn reduce_tensor_groups(
    op: TensorReduceOp,
    input: &TensorStorage,
    groups: &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError> {
    let (accumulator, result) = reduction_signature(op, input.prim())?;
    let values = groups
        .iter()
        .map(|group| {
            reduce_group(op, input, group, accumulator)
                .and_then(|value| reduction_result_scalar(op, value, result))
        })
        .collect::<Result<Vec<_>, NumericKernelError>>()?;
    let raw = if result.is_integer() {
        RawTensor::Int(
            values
                .iter()
                .map(|value| {
                    value
                        .as_i64_exact()
                        .expect("integer reduction results read exactly")
                })
                .collect(),
        )
    } else {
        RawTensor::Float(values.iter().map(ScalarValue::as_f64_lossy).collect())
    };
    finalize_tensor(op.name(), result, raw).map_err(Into::into)
}

/// One [05-OP-33] addition at the operands' arithmetic width: integers
/// exactly, trapping on overflow under `op`'s name; f32 and f64 natively; f16
/// and bf16 through f32, rounded once into storage.
fn width_add(
    op: &'static str,
    lhs: ScalarValue,
    rhs: ScalarValue,
) -> Result<ScalarValue, NumericKernelError> {
    let sum = if lhs.prim().is_integer() {
        int_binop(IntBinOp::Add, lhs, rhs)
    } else {
        float_binop(FloatBinOp::Add, lhs, rhs)
    };
    sum.map_err(|err| match err {
        NumericKernelError::Trap(NumericTrap::Overflow { prim, .. }) => {
            NumericTrap::Overflow { op, prim }.into()
        }
        other => other,
    })
}

/// Convert within one numeric family where the value set of `value` is a
/// subset of `prim`'s (an accumulator widening), or round a float once into
/// a narrower storage dtype.
fn within_family(
    op: &'static str,
    value: ScalarValue,
    prim: Prim,
) -> Result<ScalarValue, NumericKernelError> {
    if value.prim() == prim {
        Ok(value)
    } else if value.prim().is_integer() && prim.is_integer() {
        let exact = value.as_i64_exact().expect("integer elements read exactly");
        scalar_from_i64(op, prim, exact).map_err(Into::into)
    } else if value.prim().is_float() && prim.is_float() {
        scalar_from_f64(op, prim, value.as_f64_lossy()).map_err(Into::into)
    } else {
        Err(NumericKernelError::WrongFamily {
            op,
            expected: if prim.is_integer() {
                NumericFamily::Int
            } else {
                NumericFamily::Float
            },
            actual: value.prim(),
        })
    }
}

/// [05-OP-33] `cumsum` over explicitly ordered lanes. In lane order an
/// exact-zero accumulator at §5.7.1's default sum-accumulator dtype adds each
/// input, and each prefix is stored at `sum_result(p, default(p))`. Every
/// input index belongs to exactly one lane; callers own the axis planning.
pub fn cumsum_tensor_lanes(
    input: &TensorStorage,
    lanes: &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError> {
    const OP: &str = "cumsum";
    let prim = input.prim();
    let signature = prim
        .default_reduce_sum_accumulator()
        .and_then(|accumulator| {
            prim.default_reduce_sum_result_precision()
                .map(|result| (accumulator, result))
        });
    let (accumulator, result) = match (prim.is_integer() || prim.is_float(), signature) {
        (true, Ok(signature)) => signature,
        _ => {
            return Err(NumericKernelError::InvalidReductionSignature {
                op: OP,
                input: prim,
                accumulator: prim,
                result: prim,
            });
        }
    };
    let zero = if accumulator.is_integer() {
        scalar_from_i64(OP, accumulator, 0)?
    } else {
        scalar_from_f64(OP, accumulator, 0.0)?
    };
    let mut values = vec![within_family(OP, zero, result)?; input.len()];
    for lane in lanes {
        let mut running = zero;
        for &index in lane {
            let leaf = within_family(OP, input.scalar_at(index), accumulator)?;
            running = width_add(OP, running, leaf)?;
            values[index] = within_family(OP, running, result)?;
        }
    }
    Ok(tensor_from_scalars(result, &values))
}

/// [05-OP-33] add-mode `scatter`. `leaves[d]` lists, in row-major update
/// order, the update indices that target destination `d`. A targeted
/// destination's leaf sequence is its base value followed by those updates,
/// combined by the canonical adjacent-pair balanced tree at the operand
/// arithmetic width; an untargeted destination keeps its base value.
pub fn scatter_add_tensor_groups(
    base: &TensorStorage,
    updates: &TensorStorage,
    leaves: &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError> {
    const OP: &str = "scatter";
    let prim = base.prim();
    if !(prim.is_integer() || prim.is_float()) {
        return Err(NumericKernelError::InvalidReductionSignature {
            op: OP,
            input: prim,
            accumulator: prim,
            result: prim,
        });
    }
    if updates.prim() != prim {
        return Err(NumericKernelError::DtypeMismatch {
            op: OP,
            lhs: prim,
            rhs: updates.prim(),
        });
    }
    if leaves.len() != base.len() {
        return Err(NumericKernelError::LengthMismatch {
            op: OP,
            lhs: base.len(),
            rhs: leaves.len(),
        });
    }
    let values = leaves
        .iter()
        .enumerate()
        .map(|(destination, targeting)| {
            let level = std::iter::once(base.scalar_at(destination))
                .chain(targeting.iter().map(|&index| updates.scalar_at(index)))
                .collect();
            checked_adjacent_pair_fold(level, |left, right| width_add(OP, left, right))
                .map(|value| value.expect("every leaf sequence starts with its base value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(tensor_from_scalars(prim, &values))
}

/// Reduce explicitly ordered groups to exact i64 winner indices. Values
/// are compared at their stored dtype and never cross binary64 for integer
/// inputs; first-seen wins ties.
pub fn arg_reduce_tensor_groups(
    op: ArgReduceOp,
    input: &TensorStorage,
    groups: &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError> {
    if !input.prim().is_integer() && !input.prim().is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Float,
            actual: input.prim(),
        });
    }
    let mut indices = Vec::with_capacity(groups.len());
    for group in groups {
        if group.is_empty() {
            return Err(NumericTrap::Domain {
                op: op.name(),
                prim: Prim::Int64,
            }
            .into());
        }
        let mut best_value = input.scalar_at(group[0]);
        let mut best_index = 0i64;
        for (axis_index, &input_index) in group.iter().enumerate().skip(1) {
            let candidate = input.scalar_at(input_index);
            if !scalar_is_nan(best_value)
                && (scalar_is_nan(candidate)
                    || compare_scalars(op.compare(), candidate, best_value)?)
            {
                best_value = candidate;
                best_index = axis_index as i64;
            }
        }
        indices.push(best_index);
    }
    finalize_tensor(op.name(), Prim::Int64, RawTensor::Int(indices)).map_err(Into::into)
}

/// Apply the reverse-mode window adjoint to explicitly ordered source-index
/// groups. Each group corresponds to one cotangent element. Selection,
/// scaling, overlap accumulation, and finalization all execute at the input
/// dtype's [04-NUM-8] arithmetic width.
pub fn reduce_window_grad_tensor_groups(
    op: ReduceWindowGradOp,
    input: &TensorStorage,
    cotangent: &TensorStorage,
    groups: &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError> {
    let prim = input.prim();
    if !prim.is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: op.name(),
            expected: NumericFamily::Float,
            actual: prim,
        });
    }
    if cotangent.prim() != prim {
        return Err(NumericKernelError::DtypeMismatch {
            op: op.name(),
            lhs: prim,
            rhs: cotangent.prim(),
        });
    }
    if groups.len() != cotangent.len() {
        return Err(NumericKernelError::LengthMismatch {
            op: op.name(),
            lhs: groups.len(),
            rhs: cotangent.len(),
        });
    }

    let forward_op = op.forward_op();
    let accumulator =
        reduction_arithmetic_prim(prim).ok_or(NumericKernelError::InvalidReductionSignature {
            op: op.name(),
            input: prim,
            accumulator: prim,
            result: prim,
        })?;
    let mut output = vec![Vec::<ScalarValue>::new(); input.len()];

    for (group_index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(NumericTrap::Domain {
                op: forward_op.name(),
                prim,
            }
            .into());
        }
        let mut contribution =
            scalar_at_reduction_width(forward_op, cotangent, group_index, accumulator)?;
        if op == ReduceWindowGradOp::Mean {
            let divisor = reduction_seed(
                forward_op,
                accumulator,
                group.len() as i64,
                group.len() as f64,
            )?;
            contribution = reduction_div_float(forward_op, contribution, divisor)?;
        }

        match op {
            ReduceWindowGradOp::Sum | ReduceWindowGradOp::Mean => {
                for &input_index in group {
                    output[input_index].push(contribution);
                }
            }
            ReduceWindowGradOp::Max | ReduceWindowGradOp::Min => {
                if let Some(&first_nan) = group
                    .iter()
                    .find(|&&input_index| scalar_is_nan(input.scalar_at(input_index)))
                {
                    output[first_nan].push(contribution);
                    continue;
                }

                let extreme = reduce_group(forward_op, input, group, accumulator)?;
                let mut selected = Vec::new();
                for &input_index in group {
                    if compare_scalars(CompareOp::Eq, input.scalar_at(input_index), extreme)? {
                        selected.push(input_index);
                    }
                }
                let divisor = reduction_seed(
                    forward_op,
                    accumulator,
                    i64::try_from(selected.len()).expect("window tie count fits i64"),
                    selected.len() as f64,
                )?;
                let share = reduction_div_float(forward_op, contribution, divisor)?;
                for input_index in selected {
                    output[input_index].push(share);
                }
            }
        }
    }

    let values = output
        .into_iter()
        .map(|contributions| {
            let value = match checked_adjacent_pair_fold(contributions, |left, right| {
                reduction_add(forward_op, left, right)
            })? {
                Some(value) => value,
                None => reduction_seed(forward_op, accumulator, 0, 0.0)?,
            };
            reduction_result_scalar(forward_op, value, prim)
        })
        .collect::<Result<Vec<_>, NumericKernelError>>()?;
    Ok(tensor_from_scalars(prim, &values))
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
        Bits::Key(value) => Buf::Key(vec![value; len]),
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

/// Round a compile-time float bound to `prim`'s exact value, the way the
/// evaluator does when it stores one.
///
/// What a float-target `cast` does to a statically resolvable literal.
///
/// Random controls do NOT come through here: they are operands the IR lanes
/// evaluate and `StagedBound` (C) folds, and both reach the same
/// `finalize_scalar` chokepoint via `cast_raw`/`cast_scalar`.
///
/// chelis#2316: both of them used to recurse THROUGH a `cast` and keep the
/// innermost literal, on the premise recorded in `extract_f64_value`'s own
/// doc comment that "a float-target cast preserves the numeric value". That
/// holds for f32 and f64 and is false for every narrowing float target, so
/// `cast(cast(0.30000001, f16), f32)` baked `0.30000001` where the program
/// declares `0.300048828125` — a silent bound substitution of ~1600 f32 ULPs
/// that both compiled lanes made identically while `eval` rounded correctly.
///
/// Returns `None` for a non-float target, matching the fold sites' existing
/// contract that an integer-target cast is left unresolved and goes loud
/// rather than baking a guessed truncation (chelis#776).
pub fn round_float_bound(prim: Prim, value: f64) -> Option<f64> {
    if !prim.is_float() {
        return None;
    }
    finalize_scalar("cast", prim, RawScalar::Float(value))
        .ok()
        .map(|scalar| scalar.as_f64_lossy())
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
            RawScalar::Float(x) => f16_from_f64_rne(x),
            RawScalar::Int(i) => f16_from_i64_rne(i),
        }),
        Prim::Bf16 => Bits::Bf16(match raw {
            RawScalar::Float(x) => bf16_from_f64_rne(x),
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
                | Prim::String
                | Prim::Key => unreachable!("outer match binds an integer prim"),
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
        Prim::Key => panic!(
            "finalize_scalar: a random key is not a numeric dtype and has no \
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

/// Return whether `value` is exactly representable by the float dtype
/// `prim`. This is a value-set predicate, not a conversion: it performs no
/// cast and therefore cannot hide an integer through a rounded f64 image.
/// Non-float dtypes return `false`.
pub fn integer_is_exactly_representable(value: i64, prim: Prim) -> bool {
    let precision = match prim {
        Prim::F64 => 53,
        Prim::F32 => 24,
        Prim::F16 => 11,
        Prim::Bf16 => 8,
        Prim::Int64
        | Prim::Int32
        | Prim::Int16
        | Prim::Int8
        | Prim::Bool
        | Prim::F8e4m3
        | Prim::String
        | Prim::Key => return false,
    };
    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return true;
    }
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    significant_bits <= precision
        || magnitude.trailing_zeros() >= significant_bits.saturating_sub(precision)
}

/// Validated fixed parameters shared by the borrowed evaluator kernel and
/// native emission. Construction checks the input dtype and stored rate;
/// neither construction nor inspection draws, allocates a tensor or grants
/// permission to replay a source site.
#[derive(Debug, Clone, Copy)]
pub struct DropoutParameters {
    rate: ScalarValue,
}

impl DropoutParameters {
    pub fn new(prim: Prim, rate: ScalarValue) -> Result<Self, NumericKernelError> {
        if !prim.is_float() {
            return Err(NumericKernelError::WrongFamily {
                op: "dropout",
                expected: NumericFamily::Float,
                actual: prim,
            });
        }
        if rate.prim() != prim {
            return Err(NumericKernelError::DtypeMismatch {
                op: "dropout",
                lhs: prim,
                rhs: rate.prim(),
            });
        }
        let wide_rate = rate.as_f64_lossy();
        if !wide_rate.is_finite() || !(0.0..1.0).contains(&wide_rate) {
            return Err(NumericTrap::Domain {
                op: "dropout",
                prim,
            }
            .into());
        }
        Ok(Self { rate })
    }

    pub fn rate(self) -> ScalarValue {
        self.rate
    }

    /// The denominator is finalized at the storage dtype, before division.
    /// Keeping this operation here prevents native preparation from silently
    /// using a wide subtraction or reciprocal multiplication instead.
    pub fn denominator(self) -> Result<ScalarValue, NumericKernelError> {
        let one = cast_raw("dropout", RawScalar::Int(1), self.rate.prim())?;
        float_binop(FloatBinOp::Sub, one, self.rate)
    }
}

/// The key of one random draw: an opaque, structurally non-numeric carrier
/// that a random kernel receives in place of any ambient stream.
///
/// A key has no arithmetic, comparison, or cast. It is formed only by
/// [`RandomKey::from_seed`] ([05-OP-69]) and by the [05-RNG-2] derivations
/// [`RandomKey::split`], [`RandomKey::fold_in`] and [`RandomKey::split_n`].
/// [`RandomKey::bits`] exists so that native backends can port the kernels
/// bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RandomKey {
    bits: u64,
}

impl RandomKey {
    /// `[05-OP-69]` `key_from_seed`: the key whose bits are the i64 seed's
    /// two's-complement bits, with no mixing.
    pub fn from_seed(seed: ScalarValue) -> Result<Self, NumericKernelError> {
        match seed.bits {
            Bits::I64(seed) => Ok(Self { bits: seed as u64 }),
            _ => Err(NumericKernelError::DtypeMismatch {
                op: "key_from_seed",
                lhs: Prim::Int64,
                rhs: seed.prim(),
            }),
        }
    }

    /// `[05-RNG-2]`'s `derive(k, j) = splitmix64(k XOR rotl64(splitmix64(j), 29))`.
    pub fn derive(self, j: u64) -> Self {
        Self {
            bits: random_derive(self.bits, j),
        }
    }

    /// `[05-OP-70]` `split_key`: `(derive(k, 0), derive(k, 1))`.
    pub fn split(self) -> (Self, Self) {
        (self.derive(0), self.derive(1))
    }

    /// `[05-OP-72]` `fold_in`: `derive(derive(k, 2), n)` with the i64 `n`
    /// read as its two's-complement bits.
    pub fn fold_in(self, n: ScalarValue) -> Result<Self, NumericKernelError> {
        match n.bits {
            Bits::I64(n) => Ok(self.fold_in_word(n as u64)),
            _ => Err(NumericKernelError::DtypeMismatch {
                op: "fold_in",
                lhs: Prim::Int64,
                rhs: n.prim(),
            }),
        }
    }

    /// `[05-OP-71]` `split_keys`: row `j` is `derive(derive(k, 2), j)`.
    pub fn split_n(self, count: usize) -> Vec<Self> {
        (0..count as u64).map(|j| self.fold_in_word(j)).collect()
    }

    fn fold_in_word(self, n: u64) -> Self {
        self.derive(2).derive(n)
    }

    /// The key word, for native ports of the kernels below.
    pub fn bits(self) -> u64 {
        self.bits
    }
}

/// Which half of `[05-OP-70]`'s pair a split produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyHalf {
    /// `derive(k, 0)`.
    Left,
    /// `derive(k, 1)`.
    Right,
}

impl TensorStorage {
    /// Key storage from keys a key kernel produced. A key has no literal, so
    /// this is the only construction path for a key buffer.
    pub fn from_keys(keys: Vec<RandomKey>) -> Self {
        TensorStorage {
            buf: Buf::Key(keys),
        }
    }

    /// The keys of a key buffer, or `None` for every numeric and bool buffer.
    pub fn keys(&self) -> Option<&[RandomKey]> {
        match &self.buf {
            Buf::Key(keys) => Some(keys),
            _ => None,
        }
    }
}

fn key_operand<'a>(
    op: &'static str,
    storage: &'a TensorStorage,
) -> Result<&'a [RandomKey], NumericKernelError> {
    storage.keys().ok_or(NumericKernelError::DtypeMismatch {
        op,
        lhs: Prim::Key,
        rhs: storage.prim(),
    })
}

/// `[05-OP-69]` element by element over a `tensor[D, i64]` of seeds.
pub fn key_from_seed_storage(seeds: &TensorStorage) -> Result<TensorStorage, NumericKernelError> {
    let keys = (0..seeds.len())
        .map(|index| RandomKey::from_seed(seeds.scalar_at(index)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TensorStorage::from_keys(keys))
}

/// One half of `[05-OP-70]` element by element over a `tensor[D, key]`.
pub fn split_key_storage(
    keys: &TensorStorage,
    half: KeyHalf,
) -> Result<TensorStorage, NumericKernelError> {
    let index = match half {
        KeyHalf::Left => 0,
        KeyHalf::Right => 1,
    };
    let keys = key_operand("split_key", keys)?;
    Ok(TensorStorage::from_keys(
        keys.iter().map(|key| key.derive(index)).collect(),
    ))
}

/// `[05-OP-72]` element by element over a `tensor[D, key]` and a
/// `tensor[D, i64]` of exactly equal length; there is no broadcasting.
pub fn fold_in_storage(
    keys: &TensorStorage,
    ns: &TensorStorage,
) -> Result<TensorStorage, NumericKernelError> {
    let keys = key_operand("fold_in", keys)?;
    if keys.len() != ns.len() {
        return Err(NumericKernelError::LengthMismatch {
            op: "fold_in",
            lhs: keys.len(),
            rhs: ns.len(),
        });
    }
    let folded = keys
        .iter()
        .enumerate()
        .map(|(index, key)| key.fold_in(ns.scalar_at(index)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TensorStorage::from_keys(folded))
}

/// `[05-OP-71]` over a `tensor[D, key]`: the result is `tensor[D ++ [count],
/// key]` in row-major order, so element `(i, j)` is row `j` of key `i`. The
/// caller admits that result's extents under `[05-OP-33]` first, as the DAG
/// evaluator does: this kernel allocates every one of its keys.
pub fn split_keys_storage(
    keys: &TensorStorage,
    count: usize,
) -> Result<TensorStorage, NumericKernelError> {
    let keys = key_operand("split_keys", keys)?;
    Ok(TensorStorage::from_keys(
        keys.iter().flat_map(|key| key.split_n(count)).collect(),
    ))
}

/// The row of flat element `index` when `len` elements split into
/// `rows` equal rows, and the element's index within its row. A batched
/// draw (`spec/10-serialization.md` §3.2) keys row `b` by `keys[b]` and
/// numbers the row's elements from zero.
fn batched_row(index: usize, len: usize, rows: usize) -> (usize, u64) {
    let row_len = len / rows;
    (index / row_len, (index % row_len) as u64)
}

fn require_row_split(op: &'static str, len: usize, rows: usize) -> Result<(), NumericKernelError> {
    // Zero rows split only zero elements.
    let divides = len.is_multiple_of(rows);
    if divides {
        Ok(())
    } else {
        Err(NumericKernelError::LengthMismatch {
            op,
            lhs: len,
            rhs: rows,
        })
    }
}

/// Validated, borrowed input to the pure `[05-OP-37]` dropout value kernel.
///
/// Preparation performs every dtype/rate guard without drawing or allocating.
/// The caller takes its draw's key only after `new` succeeds, then supplies
/// that key to `apply`. This kernel owns no ambient random state and grants no
/// replay provenance.
///
/// ```
/// use chelis_types::dtype_semantics::{PreparedDropout, RandomKey};
/// fn draw(prepared: &PreparedDropout<'_>, key: RandomKey) {
///     let _ = prepared.apply(key);
/// }
/// ```
///
/// The kernel no longer takes a seed and an ordinal: its key is the only
/// thing it knows about the stream it draws from.
///
/// ```compile_fail
/// use chelis_types::dtype_semantics::PreparedDropout;
/// fn draw(prepared: &PreparedDropout<'_>) {
///     let _ = prepared.apply(42, 0);
/// }
/// ```
///
/// ```compile_fail
/// use chelis_types::dtype_semantics::{DropoutParameters, PreparedDropout};
/// fn bypass<'a>(input: &'a chelis_types::TensorStorage, rate: chelis_types::ScalarValue)
///     -> PreparedDropout<'a> {
///     let parameters = DropoutParameters::new(input.prim(), rate).unwrap();
///     PreparedDropout { input, parameters }
/// }
/// ```
///
/// ```compile_fail
/// use chelis_types::dtype_semantics::DropoutParameters;
/// fn bypass(rate: chelis_types::ScalarValue) -> DropoutParameters {
///     DropoutParameters { rate }
/// }
/// ```
#[derive(Debug)]
pub struct PreparedDropout<'a> {
    input: &'a TensorStorage,
    parameters: DropoutParameters,
}

impl<'a> PreparedDropout<'a> {
    pub fn new(input: &'a TensorStorage, rate: ScalarValue) -> Result<Self, NumericKernelError> {
        Ok(Self {
            input,
            parameters: DropoutParameters::new(input.prim(), rate)?,
        })
    }

    /// Evaluate the finalized sub/div graph with a pure keyed mask. A second
    /// application with the same key is suitable for pathwise input replay;
    /// the graph's key edge, not this numerical function, authorizes it.
    pub fn apply(&self, key: RandomKey) -> Result<TensorStorage, NumericKernelError> {
        let prim = self.input.prim();
        let wide_rate = self.parameters.rate().as_f64_lossy();
        let zero = cast_raw("dropout", RawScalar::Int(0), prim)?;
        let denominator = self.parameters.denominator()?;
        let mut output = Vec::with_capacity(self.input.len());
        for index in 0..self.input.len() {
            let exact_unit = random_unit(key.bits, index as u64);
            let arithmetic_unit = if prim == Prim::F64 {
                exact_unit
            } else {
                f64::from(exact_unit as f32)
            };
            output.push(if arithmetic_unit < wide_rate {
                zero
            } else {
                float_binop(FloatBinOp::Div, self.input.scalar_at(index), denominator)?
            });
        }
        Ok(tensor_from_scalars(prim, &output))
    }
}

/// Validated `[05-OP-8]` controls: the output dtype `p` and its two bounds.
///
/// Construction performs the atom's checks before any draw: `p` is an active
/// float dtype, both bounds share one dtype that widens exactly into `p`'s
/// arithmetic width (f64 for `p = f64`, f32 otherwise), both are finite,
/// `low <= high`, and `high - low` is finite at that width. The bounds'
/// dtype is `p` or f32; f32 is the checker's current bound signature for
/// every `p` (chelis#1295), which the atom's `p`-dtype bounds replace.
#[derive(Debug, Clone, Copy)]
pub struct UniformLikeParameters {
    prim: Prim,
    low: ScalarValue,
    high: ScalarValue,
}

impl UniformLikeParameters {
    pub fn new(
        prim: Prim,
        low: ScalarValue,
        high: ScalarValue,
    ) -> Result<Self, NumericKernelError> {
        if !prim.is_float() {
            return Err(NumericKernelError::WrongFamily {
                op: "uniform_like",
                expected: NumericFamily::Float,
                actual: prim,
            });
        }
        if low.prim() != high.prim() {
            return Err(NumericKernelError::DtypeMismatch {
                op: "uniform_like",
                lhs: low.prim(),
                rhs: high.prim(),
            });
        }
        if low.prim() != prim && low.prim() != Prim::F32 {
            return Err(NumericKernelError::DtypeMismatch {
                op: "uniform_like",
                lhs: prim,
                rhs: low.prim(),
            });
        }
        let parameters = Self { prim, low, high };
        let (low, high) = (low.as_f64_lossy(), high.as_f64_lossy());
        let finite_difference = if prim == Prim::F64 {
            (high - low).is_finite()
        } else {
            ((high as f32) - (low as f32)).is_finite()
        };
        if !low.is_finite() || !high.is_finite() || low > high || !finite_difference {
            return Err(NumericTrap::Domain {
                op: "uniform_like",
                prim,
            }
            .into());
        }
        Ok(parameters)
    }

    pub fn prim(self) -> Prim {
        self.prim
    }

    pub fn low(self) -> ScalarValue {
        self.low
    }

    pub fn high(self) -> ScalarValue {
        self.high
    }

    /// The `[05-OP-8]` sample of flat element `index` of the draw keyed by
    /// `key`, stored at `p`. f64 computes the one f64 fused multiply-add from
    /// the bounds' exact f64 images; every other `p` computes the one f32
    /// fused multiply-add from their exact f32 images and `round_f32(u)`,
    /// then finalizes once to `p`.
    fn sample(self, key: RandomKey, index: u64) -> Result<ScalarValue, NumericKernelError> {
        let unit = random_unit(key.bits, index);
        let (low, high) = (self.low.as_f64_lossy(), self.high.as_f64_lossy());
        let value = if self.prim == Prim::F64 {
            (high - low).mul_add(unit, low)
        } else {
            let (low, high) = (low as f32, high as f32);
            f64::from((high - low).mul_add(unit as f32, low))
        };
        scalar_from_f64("uniform_like", self.prim, value).map_err(Into::into)
    }
}

/// Validated input to the pure `[05-OP-8]` sampler for a template of `len`
/// elements. The template's element values are never observed.
///
/// ```
/// use chelis_types::dtype_semantics::{PreparedUniformLike, RandomKey};
/// fn draw(prepared: &PreparedUniformLike, key: RandomKey) {
///     let _ = prepared.apply(key);
/// }
/// ```
///
/// ```compile_fail
/// use chelis_types::dtype_semantics::uniform_sample;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PreparedUniformLike {
    len: usize,
    parameters: UniformLikeParameters,
}

impl PreparedUniformLike {
    pub fn new(
        prim: Prim,
        len: usize,
        low: ScalarValue,
        high: ScalarValue,
    ) -> Result<Self, NumericKernelError> {
        Ok(Self {
            len,
            parameters: UniformLikeParameters::new(prim, low, high)?,
        })
    }

    pub fn parameters(&self) -> UniformLikeParameters {
        self.parameters
    }

    /// Fill the template's element count with the draw keyed by `key`.
    pub fn apply(&self, key: RandomKey) -> Result<TensorStorage, NumericKernelError> {
        let values = (0..self.len)
            .map(|index| self.parameters.sample(key, index as u64))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(tensor_from_scalars(self.parameters.prim, &values))
    }
}

/// Which `[05-OP-8]` bound a pathwise adjoint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformBound {
    Low,
    High,
}

/// The `[05-OP-8]` pathwise adjoint of one bound under the forward draw keyed
/// by `key`, for the output cotangent `cotangent` of dtype `p`.
///
/// In increasing row-major order element `i` contributes `g_i * (1 - u_i)` to
/// `low` and `g_i * u_i` to `high`, where `u_i` is the forward draw's unit at
/// the arithmetic width (the exact unit for `p = f64`, `round_f32(u)`
/// otherwise). Every primitive executes at that arithmetic width (f64 for
/// `p = f64`, f32 otherwise, the width `[05-OP-8]`'s forward affine uses), the
/// contributions combine by the canonical adjacent-pair balanced tree, and
/// the sum narrows once to `p`. An empty cotangent contributes positive zero.
pub fn uniform_like_bound_adjoint(
    cotangent: &TensorStorage,
    key: RandomKey,
    bound: UniformBound,
) -> Result<ScalarValue, NumericKernelError> {
    uniform_like_bound_adjoint_by(cotangent, bound, |index| {
        random_unit(key.bits, index as u64)
    })
}

/// [`uniform_like_bound_adjoint`] of a row-batched draw whose rows share one
/// bound: the cotangent's elements split into `keys.len()` equal rows, the
/// unit of flat element `i` comes from its row's key at its index within the
/// row, and every contribution joins one canonical balanced tree in
/// increasing row-major order.
pub fn uniform_like_bound_adjoint_rows(
    cotangent: &TensorStorage,
    keys: &[RandomKey],
    bound: UniformBound,
) -> Result<ScalarValue, NumericKernelError> {
    let len = cotangent.len();
    require_row_split("uniform_like", len, keys.len())?;
    uniform_like_bound_adjoint_by(cotangent, bound, |index| {
        let (row, element) = batched_row(index, len, keys.len());
        random_unit(keys[row].bits, element)
    })
}

fn uniform_like_bound_adjoint_by(
    cotangent: &TensorStorage,
    bound: UniformBound,
    unit_at: impl Fn(usize) -> f64,
) -> Result<ScalarValue, NumericKernelError> {
    let prim = cotangent.prim();
    if !prim.is_float() {
        return Err(NumericKernelError::WrongFamily {
            op: "uniform_like",
            expected: NumericFamily::Float,
            actual: prim,
        });
    }
    let value = if prim == Prim::F64 {
        let leaves = (0..cotangent.len())
            .map(|index| {
                let unit = unit_at(index);
                let weight = match bound {
                    UniformBound::Low => 1.0 - unit,
                    UniformBound::High => unit,
                };
                cotangent.scalar_at(index).as_f64_lossy() * weight
            })
            .collect::<Vec<_>>();
        checked_adjacent_pair_fold(leaves, |left, right| {
            Ok::<_, NumericKernelError>(left + right)
        })?
        .unwrap_or(0.0)
    } else {
        let leaves = (0..cotangent.len())
            .map(|index| {
                let unit = unit_at(index) as f32;
                let weight = match bound {
                    UniformBound::Low => 1.0f32 - unit,
                    UniformBound::High => unit,
                };
                (cotangent.scalar_at(index).as_f64_lossy() as f32) * weight
            })
            .collect::<Vec<_>>();
        f64::from(
            checked_adjacent_pair_fold(leaves, |left, right| {
                Ok::<_, NumericKernelError>(left + right)
            })?
            .unwrap_or(0.0),
        )
    };
    scalar_from_f64("uniform_like", prim, value).map_err(Into::into)
}

// [05-RNG-1]'s `splitmix64`: add the golden gamma, xor-shift 30 and
// multiply, xor-shift 27 and multiply, then xor-shift 31, all modulo 2^64.
fn random_splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// `[05-RNG-2]`'s `derive(k, j) = splitmix64(k XOR rotl64(splitmix64(j), 29))`.
/// A finalized mix, never a bare XOR of per-index terms, so a chained fold is
/// not symmetric in its indices (the chelis#2408 class).
fn random_derive(key: u64, j: u64) -> u64 {
    random_splitmix64(key ^ random_splitmix64(j).rotate_left(29))
}

/// `[05-RNG-2]`'s source word for flat element `index` of the draw keyed by
/// `key`: `word(key, index) = splitmix64(key XOR rotl64(splitmix64(index), 41))`,
/// which [05-RNG-1] reads for element `index`.
///
/// This is the one kernel boundary of randomness (chelis#2408): every Rust
/// lane derives a draw's element words from its key here, and the emitted C
/// and HIP samplers are ports of this function. A key is all a sampler knows
/// about the stream it draws from.
fn random_word(key: u64, index: u64) -> u64 {
    random_splitmix64(key ^ random_splitmix64(index).rotate_left(41))
}

/// `[05-RNG-1]`'s unit value: the word's high 53 bits over `2^53`, which f64
/// represents exactly.
fn random_unit(key: u64, index: u64) -> f64 {
    ((random_word(key, index) >> 11) as f64) / ((1u64 << 53) as f64)
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
/// surface; this function is the checked DEFAULT. Untyped literal staging
/// has no declared source width, so it selects the representative active
/// dtype for the raw family and then consumes the same exhaustive plan as
/// sealed values and backend emitters.
pub fn cast_raw(op: &'static str, raw: RawScalar, dst: Prim) -> Result<ScalarValue, NumericTrap> {
    let source = match raw {
        RawScalar::Int(_) => Prim::Int64,
        RawScalar::Float(_) => Prim::F64,
    };
    CheckedCastPlan::new(source, dst)
        .unwrap_or_else(|error| panic!("cast_raw: {error} (op {op})"))
        .cast_raw(op, raw)
}

/// [`cast_raw`] over a sealed scalar: the source family picks its exact
/// wide reading (integers and bool exactly through i64, floats through
/// their exact f64 image).
pub fn cast_scalar(
    op: &'static str,
    value: ScalarValue,
    dst: Prim,
) -> Result<ScalarValue, NumericTrap> {
    CheckedCastPlan::new(value.prim(), dst)
        .unwrap_or_else(|error| panic!("cast_scalar: {error} (op {op})"))
        .cast_scalar(op, value)
}

/// The NAMED truncating cast of [05-OP-6] (the chelis#759 ladder's
/// float-to-integer rung): the explicit escape hatch from the checked
/// default's `Domain` trap on a fractional value.
///
/// A finite source yields its integer part truncated toward zero,
/// finalized at the target width through [`finalize_scalar`]; a truncated
/// integer outside the target range TRAPS `Overflow` (never wraps, never
/// saturates). A non-finite source (`NaN`, `+-inf`) TRAPS `Domain`:
/// truncation of a non-finite value has no integer meaning.
///
/// The float source and integer target are a CHECK-time contract
/// (`spec/05-risc-primitives.md` [05-OP-6]); every other pair is a type
/// error, so no program can reach the panicking arms below.
pub fn cast_trunc_raw(
    op: &'static str,
    raw: RawScalar,
    dst: Prim,
) -> Result<ScalarValue, NumericTrap> {
    assert_integer_trunc_target(op, dst);
    let x = match raw {
        RawScalar::Float(x) => x,
        RawScalar::Int(_) => panic!(
            "cast_trunc_raw: an integer source has no truncating cast \
             ([05-OP-6] is float-to-integer only); the checker rejects it, \
             so no program can reach this arm (op {op})"
        ),
    };
    if !x.is_finite() {
        return Err(NumericTrap::Domain { op, prim: dst });
    }
    // `trunc` discards the fractional part toward zero and is exact on
    // every finite f64, so the single finalize below is the only place a
    // value can be rejected: `int_wide` sees a finite integral float and
    // the width check raises `Overflow`.
    finalize_scalar(op, dst, RawScalar::Float(x.trunc()))
}

/// [`cast_trunc_raw`] over a sealed scalar: a float source reads through
/// its exact f64 image (every active float width is exactly representable
/// in f64, so the reading is lossless and the truncation is the only
/// value change).
pub fn cast_trunc_scalar(
    op: &'static str,
    value: ScalarValue,
    dst: Prim,
) -> Result<ScalarValue, NumericTrap> {
    assert_float_trunc_source(op, value.prim());
    cast_trunc_raw(op, RawScalar::Float(value.as_f64_lossy()), dst)
}

/// Bulk [`cast_trunc_raw`]: one truncation pass, then the monomorphized
/// [`finalize_tensor`] loop (the section C5 performance contract). Traps
/// on the first offending element, identically to the scalar surface.
pub fn cast_trunc_tensor(
    op: &'static str,
    raw: RawTensor,
    dst: Prim,
) -> Result<TensorStorage, NumericTrap> {
    assert_integer_trunc_target(op, dst);
    let values = match raw {
        RawTensor::Float(v) => v,
        RawTensor::Int(_) => panic!(
            "cast_trunc_tensor: an integer source has no truncating cast \
             ([05-OP-6] is float-to-integer only); the checker rejects it, \
             so no program can reach this arm (op {op})"
        ),
    };
    // IN-ORDER first-offender, element by element through the scalar
    // kernel: the compiled C lane is a per-element loop over the same
    // guard, so this is what makes the two lanes agree on the trap KIND
    // for a buffer carrying more than one kind of offender, as
    // [05-OP-6]'s identical-lanes clause requires.
    //
    // The bulk `finalize_tensor` path cannot express this: `int_buf`
    // domain-checks the WHOLE buffer before it width-checks any of it,
    // so `[300.9, NaN] -> i8` raises Domain there while C raises
    // Overflow at element 0. Only i64 (where out-of-range and
    // non-finite are both caught in the same pass) is unaffected.
    // `cast_value`'s checked rung is per-element for the same reason;
    // agreeing with the other lane outranks the section C5 bulk-loop
    // note here.
    let mut wides = Vec::with_capacity(values.len());
    for x in values {
        let value = cast_trunc_raw(op, RawScalar::Float(x), dst)?;
        wides.push(
            value
                .as_i64_exact()
                .expect("an integer target stores an exact i64"),
        );
    }
    finalize_tensor(op, dst, RawTensor::Int(wides))
}

/// [05-OP-6] target contract: integer widths only. `bool` is excluded by
/// [04-NUM-4] and a float target would be a widening, not a truncation.
fn assert_integer_trunc_target(op: &'static str, dst: Prim) {
    match dst {
        Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => {}
        Prim::F64 | Prim::F32 | Prim::F16 | Prim::Bf16 | Prim::Bool | Prim::F8e4m3 => panic!(
            "cast_trunc: `{}` is not an integer target; [05-OP-6] is \
             float-to-integer only and the checker rejects every other \
             target (op {op})",
            dst.name()
        ),
        Prim::String => panic!("cast_trunc: string is not a numeric dtype (op {op})"),
        Prim::Key => panic!("cast_trunc: a random key is not a numeric dtype (op {op})"),
    }
}

/// [05-OP-6] source contract: float widths only.
fn assert_float_trunc_source(op: &'static str, src: Prim) {
    match src {
        Prim::F64 | Prim::F32 | Prim::F16 | Prim::Bf16 => {}
        Prim::Int8
        | Prim::Int16
        | Prim::Int32
        | Prim::Int64
        | Prim::Bool
        | Prim::F8e4m3
        | Prim::String
        | Prim::Key => panic!(
            "cast_trunc: `{}` is not a float source; [05-OP-6] is \
             float-to-integer only and the checker rejects every other \
             source (op {op})",
            src.name()
        ),
    }
}

// Stored-value serialization is a checked bit transport, separate from arithmetic
// finalization. The private child module retains access to the sealed carriers.
mod wire_codec;
pub use wire_codec::{KeyBits, execution_storage};

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
            RawTensor::Float(v) => v.into_iter().map(f16_from_f64_rne).collect(),
            RawTensor::Int(v) => v.into_iter().map(f16_from_i64_rne).collect(),
        }),
        Prim::Bf16 => Buf::Bf16(match raw {
            RawTensor::Float(v) => v.into_iter().map(bf16_from_f64_rne).collect(),
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
        Prim::Key => panic!(
            "finalize_tensor: a random key has no literal or raw ingress; only the \
             key kernels construct key storage (op {op})"
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

/// Build sealed storage from ALREADY-finalized scalars, exactly.
///
/// The transport companion of [`finalize_tensor`] for values that carry
/// their dtype with them ([04-NUM-11]: a value survives transport at its
/// declared dtype; chelis#1116). Insertion is not a numeric op: no
/// rounding, no widening intermediate, bits in as bits out. Every
/// element must already carry `prim` - the caller finalizes or casts
/// first - so a mismatched element is a caller bug and panics, mirroring
/// the `reuse_*` family's dtype-mismatch contract.
pub fn tensor_from_scalars(prim: Prim, values: &[ScalarValue]) -> TensorStorage {
    fn read<T>(prim: Prim, v: &ScalarValue, out: Option<T>) -> T {
        match out {
            Some(x) => x,
            None => panic!(
                "tensor_from_scalars: element dtype {} does not match tensor \
                 dtype {} (finalize or cast the element first; insertion \
                 performs no conversion)",
                v.prim().name(),
                prim.name()
            ),
        }
    }
    let buf = match prim {
        Prim::F64 => Buf::F64(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::F64(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::F32 => Buf::F32(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::F32(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::F16 => Buf::F16(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::F16(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::Bf16 => Buf::Bf16(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::Bf16(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::Int64 => Buf::I64(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::I64(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::Int32 => Buf::I32(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::I32(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::Int16 => Buf::I16(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::I16(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::Int8 => Buf::I8(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::I8(x) => Some(x),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::Bool => Buf::Bool(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::Bool(b) => Some(u8::from(b)),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
        Prim::F8e4m3 => panic!(
            "tensor_from_scalars: f8e4m3 is not in the active dtype set \
             (spec/04-type-system.md section 1.1.1); no finalized value can \
             carry it"
        ),
        Prim::String => panic!(
            "tensor_from_scalars: string is not a numeric dtype and has no \
             tensor storage"
        ),
        Prim::Key => Buf::Key(
            values
                .iter()
                .map(|v| {
                    read(
                        prim,
                        v,
                        match v.bits {
                            Bits::Key(key) => Some(key),
                            _ => None,
                        },
                    )
                })
                .collect(),
        ),
    };
    TensorStorage { buf }
}

#[cfg(test)]
// Tests only: Rust std functions on the clippy disallowed list compute
// reference or input values here; the list holds production code to
// chelis-crmath (chelis#2957).
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn fin(prim: Prim, x: f64) -> Result<ScalarValue, NumericTrap> {
        finalize_scalar("test_op", prim, RawScalar::Float(x))
    }

    fn fin_i(prim: Prim, i: i64) -> Result<ScalarValue, NumericTrap> {
        finalize_scalar("test_op", prim, RawScalar::Int(i))
    }

    #[test]
    fn exact_integer_representability_is_total_at_every_float_width() {
        for (prim, precision) in [
            (Prim::Bf16, 8u32),
            (Prim::F16, 11),
            (Prim::F32, 24),
            (Prim::F64, 53),
        ] {
            let boundary = 1i64 << precision;
            for value in [0, 1, -1, boundary - 1, boundary, -boundary] {
                assert!(
                    integer_is_exactly_representable(value, prim),
                    "{} must represent {value} exactly",
                    prim.name()
                );
            }
            assert!(!integer_is_exactly_representable(boundary + 1, prim));
            assert!(!integer_is_exactly_representable(-(boundary + 1), prim));
            assert!(integer_is_exactly_representable(boundary * 2, prim));
        }
        assert!(integer_is_exactly_representable(i64::MIN, Prim::F64));
        assert!(!integer_is_exactly_representable(i64::MAX, Prim::F64));
        assert!(!integer_is_exactly_representable(1, Prim::Int64));
        assert!(!integer_is_exactly_representable(1, Prim::Bool));
        assert!(!integer_is_exactly_representable(1, Prim::String));
        assert!(!integer_is_exactly_representable(1, Prim::F8e4m3));
    }

    #[test]
    fn dropout_parameters_preserve_the_finalized_subtraction() {
        for (prim, expected) in [
            (Prim::F16, 0.89990234375),
            (Prim::Bf16, 0.8984375),
            (Prim::F32, f64::from(f32::from_bits(0x3f66_6666))),
            (Prim::F64, f64::from_bits(0x3fec_cccc_cccc_cccd)),
        ] {
            let rate = scalar_from_f64("test", prim, 0.1).unwrap();
            let parameters = DropoutParameters::new(prim, rate).unwrap();
            assert_eq!(parameters.rate(), rate);
            let denominator = parameters.denominator().unwrap();
            assert_eq!(denominator.prim(), prim);
            assert_eq!(denominator.as_f64_lossy(), expected, "{}", prim.name());
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                assert_ne!(denominator.as_f64_lossy(), 1.0 - rate.as_f64_lossy());
            }
        }
    }

    #[test]
    fn dropout_parameters_do_not_need_a_dummy_tensor_to_check_ingress() {
        let rate = scalar_from_f64("test", Prim::F32, 0.5).unwrap();
        assert!(matches!(
            DropoutParameters::new(Prim::Int32, rate),
            Err(NumericKernelError::WrongFamily { .. })
        ));
        assert!(matches!(
            DropoutParameters::new(Prim::F64, rate),
            Err(NumericKernelError::DtypeMismatch { .. })
        ));
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            for value in [-0.5, 1.0, f64::NAN, f64::INFINITY] {
                let rate = scalar_from_f64("test", prim, value).unwrap();
                assert!(matches!(
                    DropoutParameters::new(prim, rate),
                    Err(NumericKernelError::Trap(NumericTrap::Domain { .. }))
                ));
            }
            let zero = scalar_from_f64("test", prim, -0.0).unwrap();
            let parameters = DropoutParameters::new(prim, zero).unwrap();
            assert!(parameters.rate().as_f64_lossy().is_sign_negative());
            assert_eq!(parameters.denominator().unwrap().as_f64_lossy(), 1.0);
        }
    }

    #[test]
    fn prepared_dropout_rejects_invalid_ingress_even_when_empty() {
        let integer = finalize_tensor("test", Prim::Int32, RawTensor::Int(vec![])).unwrap();
        let half = scalar_from_f64("test", Prim::F32, 0.5).unwrap();
        assert!(matches!(
            PreparedDropout::new(&integer, half),
            Err(NumericKernelError::WrongFamily { .. })
        ));
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            for count in [0, 2] {
                let input =
                    finalize_tensor("test", prim, RawTensor::Float(vec![1.0; count])).unwrap();
                let other = if prim == Prim::F64 {
                    Prim::F32
                } else {
                    Prim::F64
                };
                let wrong = scalar_from_f64("test", other, 0.5).unwrap();
                assert!(matches!(
                    PreparedDropout::new(&input, wrong),
                    Err(NumericKernelError::DtypeMismatch { .. })
                ));
                for value in [-0.5, 1.0, 2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let rate = scalar_from_f64("test", prim, value).unwrap();
                    let error = PreparedDropout::new(&input, rate).unwrap_err();
                    assert_eq!(
                        error,
                        NumericKernelError::Trap(NumericTrap::Domain {
                            op: "dropout",
                            prim
                        })
                    );
                }
            }
        }
    }

    /// `[05-OP-69]`'s key of an i64 seed.
    fn seed_key(seed: i64) -> RandomKey {
        RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, seed).unwrap()).unwrap()
    }

    #[test]
    fn prepared_dropout_rounds_the_unit_before_comparing_and_finalizes_division() {
        // key_ref.py: word(key(45), 0) = 3cf6dc70d6515d83, whose unit rounds
        // up to the f32 0x3e73db72.
        let key = seed_key(45);
        let rate = f32::from_bits(0x3e73_db72);
        assert_eq!(random_word(key.bits(), 0), 0x3cf6_dc70_d651_5d83);
        let unit = random_unit(key.bits(), 0);
        assert_eq!((unit as f32).to_bits(), rate.to_bits());
        assert!(
            unit < f64::from(rate),
            "ideal-rational comparison would drop"
        );
        let input = finalize_tensor("test", Prim::F32, RawTensor::Float(vec![1.0])).unwrap();
        let prepared = PreparedDropout::new(
            &input,
            scalar_from_f64("test", Prim::F32, f64::from(rate)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            prepared.apply(key).unwrap().scalar_at(0).as_f64_lossy(),
            f64::from(1.0f32 / (1.0f32 - rate))
        );
        let input = finalize_tensor(
            "test",
            Prim::F32,
            RawTensor::Float(vec![f64::from(f32::from_bits(0x3f80_0005))]),
        )
        .unwrap();
        let prepared =
            PreparedDropout::new(&input, scalar_from_f64("test", Prim::F32, 0.1).unwrap()).unwrap();
        let result = prepared.apply(key).unwrap().scalar_at(0).as_f64_lossy() as f32;
        assert_eq!(result.to_bits(), 0x3f8e_38e9);
        assert_ne!(result.to_bits(), 0x3f8e_38ea, "reciprocal-multiply mutant");
    }

    #[test]
    fn prepared_dropout_is_pure_keyed_and_preserves_stored_zero_signs() {
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            let rate = scalar_from_f64("test", prim, 0.5).unwrap();
            let input = finalize_tensor("test", prim, RawTensor::Float(vec![-0.0; 2])).unwrap();
            let prepared = PreparedDropout::new(&input, rate).unwrap();
            // key_ref.py: unit(key(44), 0) = 0.4215 < 0.5 <= unit(key(44), 1) = 0.7762,
            // so element 0 is dropped and element 1 is kept.
            let first = prepared.apply(seed_key(44)).unwrap();
            assert_eq!(first, prepared.apply(seed_key(44)).unwrap());
            assert_eq!(first.scalar_at(0).as_f64_lossy().to_bits(), 0);
            assert_eq!(
                first.scalar_at(1).as_f64_lossy().to_bits(),
                (-0.0f64).to_bits()
            );
            assert_eq!(
                input.scalar_at(0).as_f64_lossy().to_bits(),
                (-0.0f64).to_bits()
            );
            let empty = finalize_tensor("test", prim, RawTensor::Float(vec![])).unwrap();
            assert!(
                PreparedDropout::new(&empty, rate)
                    .unwrap()
                    .apply(seed_key(-1))
                    .unwrap()
                    .is_empty()
            );
        }
    }

    // [05-RNG-2]'s `word` and [05-RNG-1]'s unit, transcribed from the spec
    // text for the kernel tests below.
    fn spec_unit(key_bits: u64, index: u64) -> f64 {
        fn splitmix64(x: u64) -> u64 {
            let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        let word = splitmix64(key_bits ^ splitmix64(index).rotate_left(41));
        (word >> 11) as f64 / (1_u64 << 53) as f64
    }

    #[test]
    fn the_draw_word_is_the_05_rng_2_word() {
        // key_ref.py's worked values: word(key(7), 0) and its unit.
        assert_eq!(random_word(seed_key(7).bits(), 0), 0x2065_4588_fcd2_5740);
        assert_eq!(
            random_unit(seed_key(7).bits(), 0).to_bits(),
            0.126_545_282_310_709_38_f64.to_bits()
        );
        for seed in [0, 42, -1, i64::MIN, i64::MAX] {
            let root = seed_key(seed);
            let (left, right) = root.split();
            for key in [root, left, right] {
                for index in [0, 1, 5, 1 << 40, u64::MAX] {
                    assert_eq!(
                        random_unit(key.bits(), index).to_bits(),
                        spec_unit(key.bits(), index).to_bits(),
                        "key {:#x} index {index}",
                        key.bits()
                    );
                }
            }
        }
    }

    // chelis#2408: the retired mixing computed `seed ^ c*G ^ i*G`, so element i
    // of draw c equalled element c of draw i, and every diagonal element was
    // the same seed-only value. Draw c here is the draw keyed by row c of
    // `split_keys` ([05-OP-71]).
    #[test]
    fn draw_c_element_i_is_not_draw_i_element_c() {
        const N: usize = 32;
        for seed in [42, -1] {
            let rows = seed_key(seed).split_n(N);
            let unit = |c: usize, i: usize| random_unit(rows[c].bits(), i as u64).to_bits();
            for c in 0..N {
                for i in (c + 1)..N {
                    assert_ne!(unit(c, i), unit(i, c), "seed {seed} ({c}, {i})");
                }
            }
            let diagonal = (0..N).map(|c| unit(c, c)).collect::<BTreeSet<_>>();
            assert_eq!(diagonal.len(), N, "seed {seed}");
        }
    }

    fn f32_scalar(value: f32) -> ScalarValue {
        scalar_from_f64("test", Prim::F32, f64::from(value)).unwrap()
    }

    fn uniform_element(prim: Prim, low: f32, high: f32, key: RandomKey, index: usize) -> f64 {
        PreparedUniformLike::new(prim, index + 1, f32_scalar(low), f32_scalar(high))
            .unwrap()
            .apply(key)
            .unwrap()
            .scalar_at(index)
            .as_f64_lossy()
    }

    #[test]
    fn uniform_sampler_is_the_05_op_8_affine_of_the_spec_unit() {
        for key in [seed_key(42), seed_key(-1).split().1] {
            for (low, high) in [(2.0f32, 5.0f32), (-1.0, 3.0), (0.0, 1.0)] {
                for index in 0..16 {
                    let unit = spec_unit(key.bits(), index);
                    let narrow = (high - low).mul_add(unit as f32, low);
                    let wide = (f64::from(high) - f64::from(low)).mul_add(unit, f64::from(low));
                    for (prim, expected) in [
                        (Prim::F64, wide),
                        (Prim::F32, f64::from(narrow)),
                        (Prim::F16, f64::from(half::f16::from_f32(narrow))),
                        (Prim::Bf16, f64::from(half::bf16::from_f32(narrow))),
                    ] {
                        let value = uniform_element(prim, low, high, key, index as usize);
                        assert_eq!(
                            value.to_bits(),
                            expected.to_bits(),
                            "{prim:?} [{low}, {high}) key {:#x} index {index}",
                            key.bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uniform_sampler_dispatches_at_the_output_dtype_width() {
        let low = 2.0f32;
        let high = 7.0f32;
        let key = seed_key(42);
        let index = 4;
        let f32_value = uniform_element(Prim::F32, low, high, key, index);
        let f64_value = uniform_element(Prim::F64, low, high, key, index);
        assert_ne!(
            f32_value.to_bits(),
            f64_value.to_bits(),
            "the two widths must not share a post-hoc f32 sampler"
        );
        for prim in [Prim::F16, Prim::Bf16] {
            let storage = PreparedUniformLike::new(prim, 5, f32_scalar(low), f32_scalar(high))
                .unwrap()
                .apply(key)
                .unwrap();
            assert_eq!(storage.prim(), prim);
            assert!((low as f64..high as f64).contains(&storage.scalar_at(index).as_f64_lossy()));
        }
        assert!(matches!(
            PreparedUniformLike::new(Prim::Int32, 1, f32_scalar(low), f32_scalar(high)),
            Err(NumericKernelError::WrongFamily { .. })
        ));
    }

    // chelis#2413: the kernels take a key and nothing else about the stream.
    // The dropout and uniform values under a key are the atoms' values at
    // that key, recomputed here from the spec text (`spec_unit` above) rather
    // than from any kernel helper.
    #[test]
    fn kernels_under_a_key_reproduce_the_spec_stream() {
        for seed in [0, 42, -1, i64::MIN, 7] {
            let root = seed_key(seed);
            let (left, right) = root.split();
            let mut keys = vec![root, left, right];
            keys.extend(root.split_n(2));
            for key in keys {
                for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
                    let values = (0..24)
                        .map(|i| 1.0 + f64::from(i) / 8.0)
                        .collect::<Vec<_>>();
                    let input = finalize_tensor("test", prim, RawTensor::Float(values)).unwrap();
                    let rate = scalar_from_f64("test", prim, 0.375).unwrap();
                    let output = PreparedDropout::new(&input, rate)
                        .unwrap()
                        .apply(key)
                        .unwrap();
                    let denominator = float_binop(
                        FloatBinOp::Sub,
                        scalar_from_f64("test", prim, 1.0).unwrap(),
                        rate,
                    )
                    .unwrap();
                    for index in 0..input.len() {
                        let unit = spec_unit(key.bits(), index as u64);
                        let unit = if prim == Prim::F64 {
                            unit
                        } else {
                            f64::from(unit as f32)
                        };
                        let expected = if unit < 0.375 {
                            0.0
                        } else {
                            float_binop(FloatBinOp::Div, input.scalar_at(index), denominator)
                                .unwrap()
                                .as_f64_lossy()
                        };
                        assert_eq!(
                            output.scalar_at(index).as_f64_lossy().to_bits(),
                            expected.to_bits(),
                            "dropout {prim:?} key {:#x} index {index}",
                            key.bits()
                        );
                    }
                    let (low, high) = (-1.5f32, 2.25f32);
                    let sampled =
                        PreparedUniformLike::new(prim, 24, f32_scalar(low), f32_scalar(high))
                            .unwrap()
                            .apply(key)
                            .unwrap();
                    for index in 0..24 {
                        let unit = spec_unit(key.bits(), index);
                        let expected = if prim == Prim::F64 {
                            (f64::from(high) - f64::from(low)).mul_add(unit, f64::from(low))
                        } else {
                            scalar_from_f64(
                                "test",
                                prim,
                                f64::from((high - low).mul_add(unit as f32, low)),
                            )
                            .unwrap()
                            .as_f64_lossy()
                        };
                        assert_eq!(
                            sampled.scalar_at(index as usize).as_f64_lossy().to_bits(),
                            expected.to_bits(),
                            "uniform {prim:?} key {:#x} index {index}",
                            key.bits()
                        );
                    }
                }
            }
        }
    }

    // [05-OP-8]'s bound adjoint, transcribed from the atom: contributions in
    // row-major order at the arithmetic width, combined level by level in
    // adjacent pairs with an odd trailing element carried up unchanged, then
    // narrowed once to `p`.
    fn spec_bound_adjoint(prim: Prim, cotangent: &[f64], key_bits: u64, high: bool) -> f64 {
        fn tree<T: Copy + std::ops::Add<Output = T>>(mut level: Vec<T>, zero: T) -> T {
            if level.is_empty() {
                return zero;
            }
            while level.len() > 1 {
                level = (0..level.len().div_ceil(2))
                    .map(|pair| match level.get(2 * pair + 1) {
                        Some(right) => level[2 * pair] + *right,
                        None => level[2 * pair],
                    })
                    .collect();
            }
            level[0]
        }
        if prim == Prim::F64 {
            let leaves = cotangent
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let u = spec_unit(key_bits, i as u64);
                    g * if high { u } else { 1.0 - u }
                })
                .collect();
            tree(leaves, 0.0)
        } else {
            let leaves = cotangent
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let u = spec_unit(key_bits, i as u64) as f32;
                    (*g as f32) * if high { u } else { 1.0 - u }
                })
                .collect();
            let wide = f64::from(tree(leaves, 0.0f32));
            scalar_from_f64("test", prim, wide).unwrap().as_f64_lossy()
        }
    }

    #[test]
    fn uniform_bound_adjoint_is_the_05_op_8_pathwise_transcription() {
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            for len in [0_usize, 1, 2, 3, 7, 16, 33] {
                let values = (0..len)
                    .map(|i| (i as f64 - 3.0) * 0.625 + 0.1)
                    .collect::<Vec<_>>();
                let cotangent =
                    finalize_tensor("test", prim, RawTensor::Float(values.clone())).unwrap();
                let stored = (0..len)
                    .map(|i| cotangent.scalar_at(i).as_f64_lossy())
                    .collect::<Vec<_>>();
                for key in [seed_key(42), seed_key(-1).split().0] {
                    for (bound, high) in [(UniformBound::Low, false), (UniformBound::High, true)] {
                        let actual = uniform_like_bound_adjoint(&cotangent, key, bound).unwrap();
                        assert_eq!(actual.prim(), prim);
                        let expected = spec_bound_adjoint(prim, &stored, key.bits(), high);
                        assert_eq!(
                            actual.as_f64_lossy().to_bits(),
                            expected.to_bits(),
                            "{prim:?} len {len} {bound:?} key {:#x}",
                            key.bits()
                        );
                    }
                }
            }
        }
        let integer = finalize_tensor("test", Prim::Int32, RawTensor::Int(vec![1])).unwrap();
        assert!(matches!(
            uniform_like_bound_adjoint(&integer, seed_key(0), UniformBound::Low),
            Err(NumericKernelError::WrongFamily { .. })
        ));
    }

    // [05-OP-8]: the bounds are validated before any draw, at the arithmetic
    // width, with equal bounds admitted; the bounds carry `p` or the checker's
    // current f32 signature (chelis#1295) and nothing else.
    #[test]
    fn uniform_parameters_validate_the_bounds_at_the_arithmetic_width() {
        let domain = |prim| {
            Err::<(), _>(NumericKernelError::Trap(NumericTrap::Domain {
                op: "uniform_like",
                prim,
            }))
        };
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            for (low, high) in [
                (1.0f32, 0.5f32),
                (f32::NAN, 1.0),
                (0.0, f32::INFINITY),
                (f32::NEG_INFINITY, 0.0),
            ] {
                assert_eq!(
                    UniformLikeParameters::new(prim, f32_scalar(low), f32_scalar(high)).map(|_| ()),
                    domain(prim),
                    "{prim:?} [{low}, {high})"
                );
            }
            let equal = PreparedUniformLike::new(prim, 3, f32_scalar(0.5), f32_scalar(0.5))
                .unwrap()
                .apply(seed_key(42))
                .unwrap();
            for index in 0..3 {
                assert_eq!(equal.scalar_at(index).as_f64_lossy(), 0.5);
            }
        }
        // Finite f32 bounds whose difference overflows f32 but not f64: the
        // f64 draw computes in f64 and is valid; every narrower `p` computes
        // in f32 and traps.
        let (low, high) = (f32_scalar(-3.0e38), f32_scalar(3.0e38));
        assert!(UniformLikeParameters::new(Prim::F64, low, high).is_ok());
        for prim in [Prim::F16, Prim::Bf16, Prim::F32] {
            assert_eq!(
                UniformLikeParameters::new(prim, low, high).map(|_| ()),
                domain(prim)
            );
        }
        let f64_bound = scalar_from_f64("test", Prim::F64, 0.5).unwrap();
        let f16_bound = scalar_from_f64("test", Prim::F16, 0.5).unwrap();
        assert!(UniformLikeParameters::new(Prim::F64, f64_bound, f64_bound).is_ok());
        assert!(UniformLikeParameters::new(Prim::F16, f16_bound, f16_bound).is_ok());
        for (prim, low, high) in [
            (Prim::F32, f64_bound, f64_bound),
            (Prim::Bf16, f16_bound, f16_bound),
            (Prim::F64, f32_scalar(0.5), f64_bound),
        ] {
            assert!(
                matches!(
                    UniformLikeParameters::new(prim, low, high),
                    Err(NumericKernelError::DtypeMismatch { .. })
                ),
                "{prim:?} {:?} {:?}",
                low.prim(),
                high.prim()
            );
        }
    }

    // ---- chelis#1116: exact insertion from already-finalized scalars ----

    #[test]
    fn tensor_from_scalars_inserts_f16_bits_exactly() {
        let v = fin(Prim::F16, 0.1).unwrap();
        let storage = tensor_from_scalars(Prim::F16, &[v, v]);
        assert_eq!(storage.prim(), Prim::F16);
        assert_eq!(storage.scalar_at(0), v);
        assert_eq!(storage.scalar_at(1), v);
    }

    #[test]
    fn tensor_from_scalars_preserves_negative_zero_sign() {
        let v = fin(Prim::F32, -0.0).unwrap();
        let storage = tensor_from_scalars(Prim::F32, &[v]);
        assert_eq!(storage.element_f64_lossy(0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn tensor_from_scalars_keeps_int64_above_2p53_exact() {
        let v = fin_i(Prim::Int64, 9_007_199_254_740_993).unwrap();
        let storage = tensor_from_scalars(Prim::Int64, &[v]);
        assert_eq!(
            storage.to_i64_exact_vec(),
            Some(vec![9_007_199_254_740_993])
        );
    }

    #[test]
    fn tensor_from_scalars_stores_bool_bytes() {
        let t = fin_i(Prim::Bool, 1).unwrap();
        let f = fin_i(Prim::Bool, 0).unwrap();
        let storage = tensor_from_scalars(Prim::Bool, &[t, f]);
        assert_eq!(storage.scalar_at(0).as_bool_exact(), Some(true));
        assert_eq!(storage.scalar_at(1).as_bool_exact(), Some(false));
    }

    #[test]
    #[should_panic(expected = "does not match tensor")]
    fn tensor_from_scalars_rejects_mismatched_element_dtype() {
        let v = fin(Prim::F32, 1.0).unwrap();
        let _ = tensor_from_scalars(Prim::F64, &[v]);
    }

    // ---- chelis#1123 red-team finding 1: integer ingest single-rounds ----

    #[test]
    fn int_ingest_into_f32_single_rounds_not_via_f64() {
        // Verified images differ for this value: single rounding i64->f32
        // yields bits 0x5A800001; double rounding through f64 first yields
        // 0x5A800000. Finalize takes the single-rounding path ([04-NUM-1]:
        // one rounding, at the declared width).
        let single = fin_i(Prim::F32, 18_014_399_583_223_809).unwrap();
        let bits = match single.element_ref() {
            ElementRef::F32(v) => v.to_bits(),
            other => panic!("expected f32 storage, got {other:?}"),
        };
        assert_eq!(bits, 0x5A80_0001, "finalize must single-round i64->f32");
        let double = ((18_014_399_583_223_809i64 as f64) as f32).to_bits();
        assert_eq!(
            double, 0x5A80_0000,
            "control: the double-rounded image is a different f32"
        );
    }

    #[test]
    fn int_ingest_into_bf16_single_rounds_not_via_f64() {
        let x: i64 = 18_084_767_253_659_649;
        let single = fin_i(Prim::Bf16, x).unwrap().as_f64_lossy();
        let double = f64::from(half::bf16::from_f64(x as f64));
        assert_ne!(
            single, double,
            "bf16 integer ingest must not round through f64"
        );
        // Correct rounding is nearest: the single-rounded image sits
        // strictly closer to the true integer than the double-rounded one.
        let err_single = (x - single as i64).abs();
        let err_double = (x - double as i64).abs();
        assert!(err_single < err_double);
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
        let wide = f64::from(f16_from_f64_rne(0.1)) * f64::from(f16_from_f64_rne(0.1));
        assert_eq!(
            fin(Prim::F16, wide).unwrap().as_f64_lossy(),
            0.0099945068359375
        );
    }

    #[test]
    fn direct_f64_to_f16_rounds_once_without_an_intermediate_f32() {
        const JUST_BELOW_MIDPOINT: f64 = 52847.99970178839;

        assert_eq!(f16_from_f64_rne(JUST_BELOW_MIDPOINT).to_bits(), 0x7a73);
        assert_eq!(
            f16_from_f64_rne(f64::from(JUST_BELOW_MIDPOINT as f32)).to_bits(),
            0x7a74,
            "an explicit f32 source first lands on the midpoint, then ties upward"
        );
    }

    #[test]
    fn f64_to_f16_rounds_once_without_an_intermediate_f32() {
        const JUST_BELOW_MIDPOINT: f64 = 52847.99970178839;

        assert_eq!(
            fin(Prim::F16, JUST_BELOW_MIDPOINT).unwrap().as_f64_lossy(),
            52832.0,
            "the original f64 lies below the f16 midpoint"
        );
        assert_eq!(
            fin(Prim::F16, f64::from(JUST_BELOW_MIDPOINT as f32))
                .unwrap()
                .as_f64_lossy(),
            52864.0,
            "an explicit f32 source first lands on the midpoint, then ties upward"
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
        let wide = f64::from(bf16_from_f64_rne(0.1)) * f64::from(bf16_from_f64_rne(0.1));
        assert_eq!(
            fin(Prim::Bf16, wide).unwrap().as_f64_lossy(),
            0.010009765625
        );
    }

    #[test]
    fn direct_f64_to_bf16_keeps_all_source_bits_until_the_rne_step() {
        const JUST_ABOVE_MIDPOINT: f64 = 1.0039062500000002;

        assert_eq!(bf16_from_f64_rne(JUST_ABOVE_MIDPOINT).to_bits(), 0x3f81);
        assert_eq!(
            bf16_from_f64_rne(f64::from(JUST_ABOVE_MIDPOINT as f32)).to_bits(),
            0x3f80,
            "an explicit f32 source first lands on the midpoint, then ties downward"
        );
    }

    #[test]
    fn f64_to_bf16_rounds_once_without_discarding_source_bits() {
        const JUST_ABOVE_MIDPOINT: f64 = 1.0039062500000002;

        assert_eq!(
            fin(Prim::Bf16, JUST_ABOVE_MIDPOINT).unwrap().as_f64_lossy(),
            1.0078125,
            "the low f64 mantissa bits put the source above the bf16 midpoint"
        );
        assert_eq!(
            fin(Prim::Bf16, f64::from(JUST_ABOVE_MIDPOINT as f32))
                .unwrap()
                .as_f64_lossy(),
            1.0,
            "an explicit f32 source first lands on the midpoint, then ties downward"
        );
    }

    #[test]
    fn direct_reduced_float_rounding_obeys_every_finite_midpoint() {
        fn check(last_finite: u16, image: impl Fn(u16) -> f64, round: impl Fn(f64) -> u16) {
            for lower_bits in 0..last_finite {
                let upper_bits = lower_bits + 1;
                let midpoint = (image(lower_bits) + image(upper_bits)) / 2.0;
                let below = f64::from_bits(midpoint.to_bits() - 1);
                let above = f64::from_bits(midpoint.to_bits() + 1);
                let tie_bits = if lower_bits & 1 == 0 {
                    lower_bits
                } else {
                    upper_bits
                };

                assert_eq!(round(below), lower_bits, "below midpoint {lower_bits:#06x}");
                assert_eq!(round(midpoint), tie_bits, "at midpoint {lower_bits:#06x}");
                assert_eq!(round(above), upper_bits, "above midpoint {lower_bits:#06x}");

                assert_eq!(
                    round(-below),
                    0x8000 | lower_bits,
                    "negative below-magnitude midpoint {lower_bits:#06x}"
                );
                assert_eq!(
                    round(-midpoint),
                    0x8000 | tie_bits,
                    "negative midpoint {lower_bits:#06x}"
                );
                assert_eq!(
                    round(-above),
                    0x8000 | upper_bits,
                    "negative above-magnitude midpoint {lower_bits:#06x}"
                );
            }
        }

        check(
            0x7bff,
            |bits| f64::from(half::f16::from_bits(bits)),
            |value| f16_from_f64_rne(value).to_bits(),
        );
        check(
            0x7f7f,
            |bits| f64::from(half::bf16::from_bits(bits)),
            |value| bf16_from_f64_rne(value).to_bits(),
        );
    }

    #[test]
    fn production_finalizers_obey_every_finite_reduced_float_midpoint() {
        fn check(prim: Prim, last_finite: u16, image: impl Fn(u16) -> f64) {
            for lower_bits in 0..last_finite {
                let upper_bits = lower_bits + 1;
                let midpoint = (image(lower_bits) + image(upper_bits)) / 2.0;
                let below = f64::from_bits(midpoint.to_bits() - 1);
                let above = f64::from_bits(midpoint.to_bits() + 1);
                let tie_bits = if lower_bits & 1 == 0 {
                    lower_bits
                } else {
                    upper_bits
                };
                let values = [below, midpoint, above, -below, -midpoint, -above];
                let expected = [
                    lower_bits,
                    tie_bits,
                    upper_bits,
                    0x8000 | lower_bits,
                    0x8000 | tie_bits,
                    0x8000 | upper_bits,
                ];

                let scalar_bits: Vec<u16> = values
                    .into_iter()
                    .map(|value| {
                        let value =
                            finalize_scalar("midpoint_finalization", prim, RawScalar::Float(value))
                                .unwrap();
                        match value.bits {
                            Bits::F16(value) => value.to_bits(),
                            Bits::Bf16(value) => value.to_bits(),
                            _ => unreachable!("the requested dtype fixes the storage variant"),
                        }
                    })
                    .collect();
                assert_eq!(
                    scalar_bits, expected,
                    "scalar {prim:?} finalization at midpoint below {lower_bits:#06x}"
                );

                let tensor = finalize_tensor(
                    "midpoint_finalization",
                    prim,
                    RawTensor::Float(values.to_vec()),
                )
                .unwrap();
                let tensor_bits: Vec<u16> = match tensor.buf {
                    Buf::F16(values) => values.into_iter().map(half::f16::to_bits).collect(),
                    Buf::Bf16(values) => values.into_iter().map(half::bf16::to_bits).collect(),
                    _ => unreachable!("the requested dtype fixes the storage variant"),
                };
                assert_eq!(
                    tensor_bits, expected,
                    "tensor {prim:?} finalization at midpoint below {lower_bits:#06x}"
                );
            }
        }

        check(Prim::F16, 0x7bff, |bits| {
            f64::from(half::f16::from_bits(bits))
        });
        check(Prim::Bf16, 0x7f7f, |bits| {
            f64::from(half::bf16::from_bits(bits))
        });
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
        // i64 from an exact i64 wide value can never overflow.
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
        // i64 ingress from i64 is total (cannot trap by construction).
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
    fn integer_floor_ceil_round_are_storage_exact_identities() {
        for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
            let (lo, hi) = prim.integer_range().expect("integer range");
            for op in [IntUnOp::Floor, IntUnOp::Ceil, IntUnOp::Round] {
                for value in [lo, -1, 0, 1, hi] {
                    let scalar = scalar_from_i64("test", prim, value).unwrap();
                    assert_eq!(
                        int_unop(op, scalar).unwrap(),
                        scalar,
                        "{} must preserve {} at {}",
                        op.name(),
                        value,
                        prim.name()
                    );
                }
                let storage =
                    finalize_tensor("test", prim, RawTensor::Int(vec![lo, -1, 0, 1, hi])).unwrap();
                assert_eq!(
                    int_tensor_unop(op, &storage).unwrap(),
                    storage,
                    "{} tensor kernel must be identity at {}",
                    op.name(),
                    prim.name()
                );
            }
        }
    }

    #[test]
    fn float_kernels_compute_at_declared_arithmetic_width() {
        // [05-OP-46]: f32 `exp` is the correctly rounded f32 operation, which
        // chelis-crmath computes, not an f64 exp narrowed afterwards. (This
        // input was a one-ulp witness where macOS libm expf misrounds.)
        let x = f32::from_bits(1_040_209_326);
        let input = scalar_from_f64("test", Prim::F32, f64::from(x)).unwrap();
        let got = float_unop(FloatUnOp::Exp, input).unwrap();
        assert_eq!(got.as_f64_lossy(), f64::from(chelis_crmath::exp_f32(x)));

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
            vec![f64::from(chelis_crmath::exp_f32(x))]
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

    #[test]
    fn tensor_activations_do_not_dispatch_through_the_scalar_kernel_per_element() {
        let scalar = scalar_from_f64("test", Prim::F32, 0.5).unwrap();
        SCALAR_ACTIVATION_CALL_COUNT.with(|count| count.set(0));
        float_unop(FloatUnOp::Gelu, scalar).unwrap();
        SCALAR_ACTIVATION_CALL_COUNT.with(|count| {
            assert!(
                count.get() > 0,
                "the scalar control must exercise the instrumented dispatcher"
            );
            count.set(0);
        });

        let input = finalize_tensor("test", Prim::F32, RawTensor::Float(vec![0.5; 4])).unwrap();
        float_tensor_unop(FloatUnOp::Gelu, &input).unwrap();
        SCALAR_ACTIVATION_CALL_COUNT.with(|count| {
            assert_eq!(
                count.get(),
                0,
                "the tensor kernel must dispatch once per buffer, not once per element"
            );
        });
    }

    fn one_group(len: usize) -> Vec<Vec<usize>> {
        vec![(0..len).collect()]
    }

    #[test]
    fn count_tensor_groups_uses_bool_input_int64_output_and_empty_identity() {
        let input =
            finalize_tensor("test", Prim::Bool, RawTensor::Int(vec![1, 0, 1, 1, 0])).unwrap();
        let groups = vec![vec![0, 2, 3], vec![1, 4], vec![]];
        let output = count_tensor_groups(&input, &groups).expect("typed Count kernel");
        assert_eq!(output.prim(), Prim::Int64);
        assert_eq!(output.to_i64_exact_vec(), Some(vec![3, 0, 0]));
    }

    #[test]
    fn count_tensor_groups_rejects_non_bool_storage_before_arithmetic() {
        let input = finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![0, 1])).unwrap();
        assert_eq!(
            count_tensor_groups(&input, &one_group(2)),
            Err(NumericKernelError::InvalidReductionSignature {
                op: "count",
                input: Prim::Int64,
                accumulator: Prim::Int64,
                result: Prim::Int64,
            })
        );
    }

    #[test]
    fn adjacent_pair_fold_preserves_canonical_tree() {
        let leaves = ["a", "b", "c", "d", "e"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let tree = checked_adjacent_pair_fold(leaves, |left, right| {
            Ok::<_, std::convert::Infallible>(format!("({left}+{right})"))
        })
        .expect("infallible trace fold")
        .expect("non-empty trace");
        assert_eq!(tree, "(((a+b)+(c+d))+e)");
    }

    #[test]
    fn count_add_traps_int64_overflow() {
        assert_eq!(checked_count_add(3, 4), Ok(7));
        assert_eq!(
            checked_count_add(i64::MAX, 1),
            Err(NumericKernelError::Trap(NumericTrap::Overflow {
                op: "count",
                prim: Prim::Int64,
            }))
        );
    }

    #[test]
    fn window_reductions_compute_at_each_declared_arithmetic_width() {
        let cases = [
            (Prim::F16, vec![2048.0, 1.0, -2048.0], 1.0),
            (Prim::Bf16, vec![256.0, 1.0, -256.0], 1.0),
            (Prim::F32, vec![16_777_216.0, 1.0, -16_777_216.0], 0.0),
            (Prim::F64, vec![16_777_216.0, 1.0, -16_777_216.0], 1.0),
        ];
        for (prim, values, expected) in cases {
            let input = finalize_tensor("test", prim, RawTensor::Float(values)).unwrap();
            let output =
                reduce_tensor_groups(TensorReduceOp::ReduceWindowSum, &input, &one_group(3))
                    .unwrap();
            assert_eq!(
                output.to_f64_lossy_vec(),
                vec![expected],
                "{} must use its declared arithmetic width",
                prim.name()
            );
        }
    }

    #[test]
    fn window_integer_reductions_trap_intermediate_overflow_at_every_width() {
        for (prim, max) in [
            (Prim::Int8, i64::from(i8::MAX)),
            (Prim::Int16, i64::from(i16::MAX)),
            (Prim::Int32, i64::from(i32::MAX)),
            (Prim::Int64, i64::MAX),
        ] {
            let input = finalize_tensor("test", prim, RawTensor::Int(vec![max, 1, -1])).unwrap();
            assert_eq!(
                reduce_tensor_groups(TensorReduceOp::ReduceWindowSum, &input, &one_group(3),),
                Err(NumericKernelError::Trap(NumericTrap::Overflow {
                    op: "reduce_window_sum",
                    prim,
                })),
                "{} max + 1 must trap before the later -1",
                prim.name()
            );

            let control =
                finalize_tensor("test", prim, RawTensor::Int(vec![max - 1, 1, -1])).unwrap();
            assert_eq!(
                reduce_tensor_groups(TensorReduceOp::ReduceWindowSum, &control, &one_group(3),)
                    .unwrap()
                    .to_i64_exact_vec(),
                Some(vec![max - 1])
            );
        }
    }

    #[test]
    fn global_sum_uses_explicit_accumulator_and_canonical_order() {
        let i8 = finalize_tensor(
            "test",
            Prim::Int8,
            RawTensor::Int(vec![i64::from(i8::MAX), 1, -1]),
        )
        .unwrap();
        let widened = reduce_tensor_groups(
            TensorReduceOp::Sum {
                accumulator: Prim::Int32,
                result: Prim::Int32,
            },
            &i8,
            &one_group(3),
        )
        .unwrap();
        assert_eq!(widened.prim(), Prim::Int32);
        assert_eq!(widened.to_i64_exact_vec(), Some(vec![i64::from(i8::MAX)]));

        let i32 = finalize_tensor(
            "test",
            Prim::Int32,
            RawTensor::Int(vec![i64::from(i32::MAX), 1, -1]),
        )
        .unwrap();
        assert_eq!(
            reduce_tensor_groups(
                TensorReduceOp::Sum {
                    accumulator: Prim::Int32,
                    result: Prim::Int32,
                },
                &i32,
                &one_group(3),
            ),
            Err(NumericKernelError::Trap(NumericTrap::Overflow {
                op: "sum",
                prim: Prim::Int32,
            }))
        );
    }

    #[test]
    fn global_sum_traps_only_at_canonical_adjacent_pairs() {
        for prim in [Prim::Int32, Prim::Int64] {
            let (_, max) = prim.integer_range().unwrap();
            let op = TensorReduceOp::Sum {
                accumulator: prim,
                result: prim,
            };
            let trapping = finalize_tensor(
                "test",
                prim,
                RawTensor::Int(vec![max, 1, 0, 0, -max, -1, 0, 0]),
            )
            .unwrap();
            assert_eq!(
                reduce_tensor_groups(op, &trapping, &one_group(8)),
                Err(NumericKernelError::Trap(NumericTrap::Overflow {
                    op: "sum",
                    prim
                }))
            );
            let valid = finalize_tensor(
                "test",
                prim,
                RawTensor::Int(vec![max, -max, 0, 0, 1, -1, 0, 0]),
            )
            .unwrap();
            assert_eq!(
                reduce_tensor_groups(op, &valid, &one_group(8))
                    .unwrap()
                    .to_i64_exact_vec(),
                Some(vec![0])
            );
        }
    }

    #[test]
    fn reduction_kernel_rejects_implicit_or_narrow_accumulator_signatures() {
        let input = finalize_tensor("test", Prim::Int32, RawTensor::Int(vec![1, 2])).unwrap();
        assert_eq!(
            reduce_tensor_groups(
                TensorReduceOp::Sum {
                    accumulator: Prim::Int16,
                    result: Prim::Int16,
                },
                &input,
                &one_group(2),
            ),
            Err(NumericKernelError::InvalidReductionSignature {
                op: "sum",
                input: Prim::Int32,
                accumulator: Prim::Int16,
                result: Prim::Int16,
            })
        );
    }

    #[test]
    fn arg_reductions_compare_int64_exactly_and_keep_first_ties() {
        let input = finalize_tensor(
            "test",
            Prim::Int64,
            RawTensor::Int(vec![
                9_007_199_254_740_992,
                9_007_199_254_740_993,
                9_007_199_254_740_993,
            ]),
        )
        .unwrap();
        assert_eq!(
            arg_reduce_tensor_groups(ArgReduceOp::Argmax, &input, &one_group(3))
                .unwrap()
                .to_i64_exact_vec(),
            Some(vec![1])
        );
    }

    #[test]
    fn value_and_window_extrema_preserve_first_nan_and_equal_value_bits() {
        let first_nan = f32::from_bits(0xffc1_2345);
        let later_nan = f32::from_bits(0x7fc5_4321);
        let input = TensorStorage {
            buf: Buf::F32(vec![first_nan, 1.0, later_nan]),
        };
        for op in [
            TensorReduceOp::MaxReduce,
            TensorReduceOp::MinReduce,
            TensorReduceOp::ReduceWindowMax,
            TensorReduceOp::ReduceWindowMin,
        ] {
            let output = reduce_tensor_groups(op, &input, &one_group(3)).unwrap();
            let StorageView::F32(values) = output.view() else {
                panic!("f32 extrema must return f32 storage");
            };
            assert_eq!(
                values[0].to_bits(),
                first_nan.to_bits(),
                "{} must preserve the first NaN payload and sign",
                op.name()
            );
        }

        let zeros = TensorStorage {
            buf: Buf::F32(vec![-0.0, 0.0]),
        };
        for op in [
            TensorReduceOp::MaxReduce,
            TensorReduceOp::MinReduce,
            TensorReduceOp::ReduceWindowMax,
            TensorReduceOp::ReduceWindowMin,
        ] {
            let output = reduce_tensor_groups(op, &zeros, &one_group(2)).unwrap();
            let StorageView::F32(values) = output.view() else {
                panic!("f32 extrema must return f32 storage");
            };
            assert_eq!(
                values[0].to_bits(),
                (-0.0f32).to_bits(),
                "{} must preserve the first representation among equal values",
                op.name()
            );
        }
    }

    #[test]
    fn reductions_without_empty_identities_trap_domain() {
        let float = finalize_tensor("test", Prim::F32, RawTensor::Float(vec![])).unwrap();
        for op in [
            TensorReduceOp::MaxReduce,
            TensorReduceOp::MinReduce,
            TensorReduceOp::ReduceWindowMean,
            TensorReduceOp::ReduceWindowMax,
            TensorReduceOp::ReduceWindowMin,
        ] {
            assert_eq!(
                reduce_tensor_groups(op, &float, &[vec![]]),
                Err(NumericKernelError::Trap(NumericTrap::Domain {
                    op: op.name(),
                    prim: Prim::F32,
                }))
            );
        }
        for op in [ArgReduceOp::Argmax, ArgReduceOp::Argmin] {
            assert_eq!(
                arg_reduce_tensor_groups(op, &float, &[vec![]]),
                Err(NumericKernelError::Trap(NumericTrap::Domain {
                    op: op.name(),
                    prim: Prim::Int64,
                }))
            );
        }
    }

    #[test]
    fn arg_reductions_select_the_lowest_nan_index() {
        let input = TensorStorage {
            buf: Buf::F32(vec![5.0, f32::from_bits(0xffc1_2345), 9.0, f32::NAN]),
        };
        for op in [ArgReduceOp::Argmax, ArgReduceOp::Argmin] {
            assert_eq!(
                arg_reduce_tensor_groups(op, &input, &one_group(4))
                    .unwrap()
                    .to_i64_exact_vec(),
                Some(vec![1])
            );
        }
    }

    #[test]
    fn window_sum_and_mean_use_the_canonical_adjacent_pair_tree() {
        let input = finalize_tensor(
            "test",
            Prim::F32,
            RawTensor::Float(vec![-1.0e-7, 3.0, 16_777_216.0, -33_554_432.0]),
        )
        .unwrap();
        assert_eq!(
            reduce_tensor_groups(TensorReduceOp::ReduceWindowSum, &input, &one_group(4))
                .unwrap()
                .to_f64_lossy_vec(),
            vec![-16_777_213.0]
        );
        assert_eq!(
            reduce_tensor_groups(TensorReduceOp::ReduceWindowMean, &input, &one_group(4))
                .unwrap()
                .to_f64_lossy_vec(),
            vec![-4_194_303.25]
        );
    }

    #[test]
    fn window_grad_overlap_add_uses_each_declared_float_arithmetic_width() {
        let groups = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4]];
        for (prim, cotangents, expected_center) in [
            (Prim::F16, vec![2048.0, 1.0, -2048.0], 1.0),
            (Prim::Bf16, vec![256.0, 1.0, -256.0], 1.0),
            (Prim::F32, vec![16_777_216.0, 1.0, -16_777_216.0], 0.0),
            (
                Prim::F64,
                vec![9_007_199_254_740_992.0, 1.0, -9_007_199_254_740_992.0],
                0.0,
            ),
        ] {
            let input = finalize_tensor("test", prim, RawTensor::Float(vec![0.0; 5])).unwrap();
            let cotangent = finalize_tensor("test", prim, RawTensor::Float(cotangents)).unwrap();
            let output = reduce_window_grad_tensor_groups(
                ReduceWindowGradOp::Sum,
                &input,
                &cotangent,
                &groups,
            )
            .unwrap();
            assert_eq!(
                output.element_f64_lossy(2),
                expected_center,
                "{} overlap-add must use its arithmetic width",
                prim.name()
            );
        }
    }

    #[test]
    fn window_grad_kernel_splits_ties_and_rejects_dtype_mismatch() {
        let input =
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![2.0, 2.0, 2.0])).unwrap();
        let cotangent =
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![4.0, 4.0])).unwrap();
        let groups = vec![vec![0, 1], vec![1, 2]];
        assert_eq!(
            reduce_window_grad_tensor_groups(ReduceWindowGradOp::Max, &input, &cotangent, &groups,)
                .unwrap()
                .to_f64_lossy_vec(),
            vec![2.0, 4.0, 2.0]
        );

        let wrong = finalize_tensor("test", Prim::F64, RawTensor::Float(vec![4.0, 4.0])).unwrap();
        assert_eq!(
            reduce_window_grad_tensor_groups(ReduceWindowGradOp::Sum, &input, &wrong, &groups,),
            Err(NumericKernelError::DtypeMismatch {
                op: "reduce_window_grad",
                lhs: Prim::F32,
                rhs: Prim::F64,
            })
        );
    }

    #[test]
    fn window_grad_routes_first_nan_and_balances_overlap_add() {
        let input = TensorStorage {
            buf: Buf::F32(vec![
                f32::from_bits(0xffc1_2345),
                f32::from_bits(0x7fc5_4321),
                2.0,
            ]),
        };
        let cotangent =
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0, 5.0])).unwrap();
        let groups = vec![vec![0, 1], vec![1, 2]];
        assert_eq!(
            reduce_window_grad_tensor_groups(ReduceWindowGradOp::Max, &input, &cotangent, &groups,)
                .unwrap()
                .to_f64_lossy_vec(),
            vec![3.0, 5.0, 0.0]
        );

        let single = finalize_tensor("test", Prim::F32, RawTensor::Float(vec![0.0])).unwrap();
        let overlap = finalize_tensor(
            "test",
            Prim::F32,
            RawTensor::Float(vec![-1.0e-7, 3.0, 16_777_216.0, -33_554_432.0]),
        )
        .unwrap();
        assert_eq!(
            reduce_window_grad_tensor_groups(
                ReduceWindowGradOp::Sum,
                &single,
                &overlap,
                &[vec![0], vec![0], vec![0], vec![0]],
            )
            .unwrap()
            .to_f64_lossy_vec(),
            vec![-16_777_213.0]
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
            "numeric trap: overflow in add at i8"
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
            "numeric trap: division by zero in trunc_div at i64"
        );
    }

    // ---- the checked cast ladder (chelis#759 one rule per direction;
    // executed at the chelis#729 rework). Positive AND negative parity
    // per direction, per the repo contract. ----

    #[test]
    fn checked_cast_plan_covers_the_active_prim_product_without_false_identity() {
        let active = [
            Prim::F64,
            Prim::F32,
            Prim::F16,
            Prim::Bf16,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ];

        for source in active {
            for target in active {
                let plan = CheckedCastPlan::new(source, target)
                    .expect("every active checked-cast pair has a plan");
                assert_eq!(plan.source(), source);
                assert_eq!(plan.target(), target);
                assert_eq!(
                    plan.kind() == CheckedCastKind::Identity,
                    source == target,
                    "identity is legal exactly on the equal-Prim diagonal: {} -> {}",
                    source.name(),
                    target.name()
                );
            }
        }
    }

    #[test]
    fn checked_cast_plan_rejects_deferred_and_non_numeric_prims_on_each_axis() {
        for unsupported in [Prim::F8e4m3, Prim::String] {
            assert_eq!(
                CheckedCastPlan::new(unsupported, Prim::F32),
                Err(CheckedCastPlanError::UnsupportedSource(unsupported))
            );
            assert_eq!(
                CheckedCastPlan::new(Prim::F32, unsupported),
                Err(CheckedCastPlanError::UnsupportedTarget(unsupported))
            );
        }
    }

    #[test]
    fn checked_cast_plan_applies_the_source_family_instead_of_only_the_target() {
        let float_to_int = CheckedCastPlan::new(Prim::F32, Prim::Int8).unwrap();
        assert_eq!(float_to_int.kind(), CheckedCastKind::FloatToInteger);
        let err = float_to_int
            .cast_scalar("cast", scalar_from_f64("test", Prim::F32, 3.5).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            NumericTrap::Domain {
                op: "cast",
                prim: Prim::Int8
            }
        );

        let int_to_float = CheckedCastPlan::new(Prim::Int64, Prim::Bf16).unwrap();
        assert_eq!(int_to_float.kind(), CheckedCastKind::ExactToFloat);
        let source = scalar_from_i64("test", Prim::Int64, 4_629_700_416_936_869_889).unwrap();
        assert_eq!(
            int_to_float.cast_scalar("cast", source).unwrap(),
            cast_scalar("cast", source, Prim::Bf16).unwrap()
        );
    }

    #[test]
    fn checked_cast_plan_negative_matrix_covers_every_applicable_trap_class() {
        let floats = [Prim::F64, Prim::F32, Prim::F16, Prim::Bf16];
        let integers = [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64];

        for source in floats {
            let fractional = scalar_from_f64("test", source, 1.5).unwrap();
            let non_finite = scalar_from_f64("test", source, f64::NAN).unwrap();
            for target in integers {
                let plan = CheckedCastPlan::new(source, target).unwrap();
                assert_eq!(
                    plan.cast_scalar("cast", fractional),
                    Err(NumericTrap::Domain {
                        op: "cast",
                        prim: target
                    }),
                    "fractional {} -> {}",
                    source.name(),
                    target.name()
                );
                assert_eq!(
                    plan.cast_scalar("cast", non_finite),
                    Err(NumericTrap::Domain {
                        op: "cast",
                        prim: target
                    }),
                    "non-finite {} -> {}",
                    source.name(),
                    target.name()
                );
            }
        }

        for (source, finite_max) in [
            (Prim::F64, f64::MAX),
            (Prim::F32, f64::from(f32::MAX)),
            (Prim::F16, f64::from(half::f16::MAX)),
            (Prim::Bf16, f64::from(half::bf16::MAX)),
        ] {
            let value = scalar_from_f64("test", source, finite_max).unwrap();
            for target in integers {
                let target_max = target.integer_range().unwrap().1 as f64;
                if finite_max > target_max {
                    assert_eq!(
                        CheckedCastPlan::new(source, target)
                            .unwrap()
                            .cast_scalar("cast", value),
                        Err(NumericTrap::Overflow {
                            op: "cast",
                            prim: target
                        }),
                        "finite overflow {} -> {}",
                        source.name(),
                        target.name()
                    );
                }
            }
        }

        for source in integers {
            let source_max = source.integer_range().unwrap().1;
            let value = scalar_from_i64("test", source, source_max).unwrap();
            for target in integers {
                let target_max = target.integer_range().unwrap().1;
                if source_max > target_max {
                    assert_eq!(
                        CheckedCastPlan::new(source, target)
                            .unwrap()
                            .cast_scalar("cast", value),
                        Err(NumericTrap::Overflow {
                            op: "cast",
                            prim: target
                        }),
                        "narrowing {} -> {}",
                        source.name(),
                        target.name()
                    );
                }
            }
        }

        for source in integers {
            let invalid = scalar_from_i64("test", source, 2).unwrap();
            assert_eq!(
                CheckedCastPlan::new(source, Prim::Bool)
                    .unwrap()
                    .cast_scalar("cast", invalid),
                Err(NumericTrap::Domain {
                    op: "cast",
                    prim: Prim::Bool
                }),
                "strict bool membership from {}",
                source.name()
            );
        }
        for source in floats {
            let invalid = scalar_from_f64("test", source, 0.5).unwrap();
            assert_eq!(
                CheckedCastPlan::new(source, Prim::Bool)
                    .unwrap()
                    .cast_scalar("cast", invalid),
                Err(NumericTrap::Domain {
                    op: "cast",
                    prim: Prim::Bool
                }),
                "strict bool membership from {}",
                source.name()
            );
        }
    }

    #[test]
    fn cast_raw_to_float_finalizes_at_target_width() {
        // 2049 is the first f16-unrepresentable integer; RNE rounds to 2048.
        let v = cast_raw("cast", RawScalar::Float(2049.0), Prim::F16).unwrap();
        assert_eq!(v.as_f64_lossy(), 2048.0);
        // Exact-int source finalizes from the exact integer, same rule.
        let v = cast_raw("cast", RawScalar::Int(2049), Prim::F16).unwrap();
        assert_eq!(v.as_f64_lossy(), 2048.0);
        // i64 above 2^53 to f64 is the LOSSY-BY-DESIGN float direction
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
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at i8");
        let err = cast_raw("cast", RawScalar::Int(-129), Prim::Int8).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at i8");
    }

    #[test]
    fn cast_raw_float_to_int_requires_an_integral_value() {
        let v = cast_raw("cast", RawScalar::Float(3.0), Prim::Int8).unwrap();
        assert_eq!(v.as_i64_exact(), Some(3));

        for fractional in [3.5, -3.5] {
            let err = cast_raw("cast", RawScalar::Float(fractional), Prim::Int8).unwrap_err();
            assert_eq!(err.to_string(), "numeric trap: domain in cast at i8");
        }
    }

    #[test]
    fn cast_raw_float_to_int_out_of_range_traps_overflow_not_saturate() {
        // Pre-rework this saturated (300.0 -> 127). The checked default
        // TRAPS; saturation is chelis#759's future NAMED form.
        let err = cast_raw("cast", RawScalar::Float(300.0), Prim::Int8).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at i8");
        let err = cast_raw("cast", RawScalar::Float(1.0e300), Prim::Int64).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: overflow in cast at i64");
    }

    #[test]
    fn cast_raw_non_finite_float_to_int_traps_domain() {
        let err = cast_raw("cast", RawScalar::Float(f64::NAN), Prim::Int32).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: domain in cast at i32");
        let err = cast_raw("cast", RawScalar::Float(f64::INFINITY), Prim::Int32).unwrap_err();
        assert_eq!(err.to_string(), "numeric trap: domain in cast at i32");
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
        // Exact integer wide: above 2^53 an i64 -> i64-family cast must
        // not launder through f64 (that laundering is the chelis#684 bug
        // shape this module exists to end).
        let narrowed = cast_scalar("cast", v, Prim::Int32).unwrap_err();
        assert_eq!(
            narrowed.to_string(),
            "numeric trap: overflow in cast at i32"
        );
    }

    #[test]
    fn direct_subtraction_uses_the_exact_stored_width_at_every_signed_dtype() {
        macro_rules! assert_width {
            ($prim:expr, $variant:ident, $min:expr, $max:expr) => {{
                let representable = int_binop(
                    IntBinOp::Sub,
                    ScalarValue {
                        bits: Bits::$variant(-1),
                    },
                    ScalarValue {
                        bits: Bits::$variant($min),
                    },
                )
                .expect("-1 - MIN is representable as MAX at the same width");
                assert_eq!(
                    representable,
                    ScalarValue {
                        bits: Bits::$variant($max),
                    },
                    "{} representable boundary",
                    $prim.name()
                );

                let overflow = int_binop(
                    IntBinOp::Sub,
                    ScalarValue {
                        bits: Bits::$variant($min),
                    },
                    ScalarValue {
                        bits: Bits::$variant(1),
                    },
                )
                .expect_err("MIN - 1 must trap as direct subtraction overflow");
                assert_eq!(
                    overflow,
                    NumericKernelError::Trap(NumericTrap::Overflow {
                        op: "sub",
                        prim: $prim,
                    }),
                    "{} true overflow",
                    $prim.name()
                );
            }};
        }

        assert_width!(Prim::Int8, I8, i8::MIN, i8::MAX);
        assert_width!(Prim::Int16, I16, i16::MIN, i16::MAX);
        assert_width!(Prim::Int32, I32, i32::MIN, i32::MAX);
        assert_width!(Prim::Int64, I64, i64::MIN, i64::MAX);
    }

    #[test]
    fn direct_subtraction_canonicalizes_float_nan_at_every_storage_width() {
        fn assert_scalar_bits(actual: ScalarValue, expected: ScalarValue) {
            match (actual.bits, expected.bits) {
                (Bits::F16(actual), Bits::F16(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                (Bits::Bf16(actual), Bits::Bf16(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                (Bits::F32(actual), Bits::F32(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                (Bits::F64(actual), Bits::F64(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                _ => panic!("test cases must compare one matching float dtype"),
            }
        }

        let scalar_cases = [
            (
                ScalarValue {
                    bits: Bits::F16(half::f16::from_bits(0xfe55)),
                },
                ScalarValue {
                    bits: Bits::F16(half::f16::from_f32(1.0)),
                },
                ScalarValue {
                    bits: Bits::F16(half::f16::from_bits(0x7e00)),
                },
            ),
            (
                ScalarValue {
                    bits: Bits::Bf16(half::bf16::from_bits(0xffe5)),
                },
                ScalarValue {
                    bits: Bits::Bf16(half::bf16::from_f32(1.0)),
                },
                ScalarValue {
                    bits: Bits::Bf16(half::bf16::from_bits(0x7fc0)),
                },
            ),
            (
                ScalarValue {
                    bits: Bits::F32(f32::from_bits(0xffc5_4321)),
                },
                ScalarValue {
                    bits: Bits::F32(1.0),
                },
                ScalarValue {
                    bits: Bits::F32(f32::from_bits(0x7fc0_0000)),
                },
            ),
            (
                ScalarValue {
                    bits: Bits::F64(f64::from_bits(0xfff8_abcd_1234_5678)),
                },
                ScalarValue {
                    bits: Bits::F64(1.0),
                },
                ScalarValue {
                    bits: Bits::F64(f64::from_bits(0x7ff8_0000_0000_0000)),
                },
            ),
        ];
        for (lhs, rhs, expected) in scalar_cases {
            assert_scalar_bits(float_binop(FloatBinOp::Sub, lhs, rhs).unwrap(), expected);
        }

        let tensor_cases = [
            (
                TensorStorage {
                    buf: Buf::F16(vec![half::f16::from_bits(0xfe55)]),
                },
                TensorStorage {
                    buf: Buf::F16(vec![half::f16::from_f32(1.0)]),
                },
                TensorStorage {
                    buf: Buf::F16(vec![half::f16::from_bits(0x7e00)]),
                },
            ),
            (
                TensorStorage {
                    buf: Buf::Bf16(vec![half::bf16::from_bits(0xffe5)]),
                },
                TensorStorage {
                    buf: Buf::Bf16(vec![half::bf16::from_f32(1.0)]),
                },
                TensorStorage {
                    buf: Buf::Bf16(vec![half::bf16::from_bits(0x7fc0)]),
                },
            ),
        ];
        for (lhs, rhs, expected) in tensor_cases {
            match (
                float_tensor_binop(FloatBinOp::Sub, &lhs, &rhs).unwrap().buf,
                expected.buf,
            ) {
                (Buf::F16(actual), Buf::F16(expected)) => {
                    assert_eq!(actual[0].to_bits(), expected[0].to_bits())
                }
                (Buf::Bf16(actual), Buf::Bf16(expected)) => {
                    assert_eq!(actual[0].to_bits(), expected[0].to_bits())
                }
                _ => panic!("test cases must compare one matching reduced float dtype"),
            }
        }
        match float_tensor_binop(
            FloatBinOp::Sub,
            &TensorStorage {
                buf: Buf::F32(vec![f32::from_bits(0xffc5_4321)]),
            },
            &TensorStorage {
                buf: Buf::F32(vec![1.0]),
            },
        )
        .unwrap()
        .buf
        {
            Buf::F32(actual) => assert_eq!(actual[0].to_bits(), 0x7fc0_0000),
            _ => panic!("f32 subtraction must retain f32 storage"),
        }
        match float_tensor_binop(
            FloatBinOp::Sub,
            &TensorStorage {
                buf: Buf::F64(vec![f64::from_bits(0xfff8_abcd_1234_5678)]),
            },
            &TensorStorage {
                buf: Buf::F64(vec![1.0]),
            },
        )
        .unwrap()
        {
            TensorStorage {
                buf: Buf::F64(actual),
            } => {
                assert_eq!(actual[0].to_bits(), 0x7ff8_0000_0000_0000)
            }
            _ => panic!("f64 subtraction must retain f64 storage"),
        }
    }

    /// chelis#2964 (C8): [04-NUM-2] finalizes every NaN that floating
    /// arithmetic or numeric conversion produces to the dtype's canonical
    /// quiet NaN, dropping input payload and sign. Selection (`max`/`min`)
    /// is bit-preserving by [05-OP-40] and is not covered here.
    #[test]
    fn every_nan_producing_float_op_finalizes_canonical_nan_at_every_storage_width() {
        fn scalar(prim: Prim, bits: u64) -> ScalarValue {
            let bits = match prim {
                Prim::F16 => Bits::F16(half::f16::from_bits(bits as u16)),
                Prim::Bf16 => Bits::Bf16(half::bf16::from_bits(bits as u16)),
                Prim::F32 => Bits::F32(f32::from_bits(bits as u32)),
                Prim::F64 => Bits::F64(f64::from_bits(bits)),
                _ => unreachable!("float widths only"),
            };
            ScalarValue { bits }
        }
        fn raw_bits(value: ScalarValue) -> u64 {
            match value.bits {
                Bits::F16(value) => u64::from(value.to_bits()),
                Bits::Bf16(value) => u64::from(value.to_bits()),
                Bits::F32(value) => u64::from(value.to_bits()),
                Bits::F64(value) => value.to_bits(),
                _ => unreachable!("float widths only"),
            }
        }
        // (prim, canonical, [negative quiet payload NaN, signaling NaN],
        // one, +inf, -inf, zero, minus one)
        let widths = [
            (
                Prim::F16,
                0x7e00,
                [0xfe55, 0x7c01],
                0x3c00,
                0x7c00,
                0xfc00,
                0,
                0xbc00,
            ),
            (
                Prim::Bf16,
                0x7fc0,
                [0xffe5, 0x7f81],
                0x3f80,
                0x7f80,
                0xff80,
                0,
                0xbf80,
            ),
            (
                Prim::F32,
                0x7fc0_0000,
                [0xffc5_4321, 0x7f81_2345],
                0x3f80_0000,
                0x7f80_0000,
                0xff80_0000,
                0,
                0xbf80_0000,
            ),
            (
                Prim::F64,
                0x7ff8_0000_0000_0000,
                [0xfff8_abcd_1234_5678, 0x7ff0_1234_5678_9abc],
                0x3ff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0,
                0xbff0_0000_0000_0000,
            ),
        ];
        let arithmetic = [
            FloatBinOp::Add,
            FloatBinOp::Sub,
            FloatBinOp::Mul,
            FloatBinOp::Div,
            FloatBinOp::FloorDiv,
        ];
        let unary = [
            FloatUnOp::Neg,
            FloatUnOp::Recip,
            FloatUnOp::Exp,
            FloatUnOp::Log,
            FloatUnOp::Sin,
            FloatUnOp::Sqrt,
            FloatUnOp::Cos,
            FloatUnOp::Tan,
            FloatUnOp::Atan,
            FloatUnOp::Abs,
            FloatUnOp::Floor,
            FloatUnOp::Ceil,
            FloatUnOp::Round,
            FloatUnOp::Sigmoid,
            FloatUnOp::Tanh,
            FloatUnOp::Silu,
            FloatUnOp::Gelu,
        ];
        for (prim, canonical, nans, one, inf, neg_inf, zero, minus_one) in widths {
            let name = prim.name();
            let mut binary_cases = Vec::new();
            for op in arithmetic {
                for nan in nans {
                    binary_cases.push((op, nan, one));
                    binary_cases.push((op, one, nan));
                }
            }
            // Invalid operations on non-NaN operands: x86 produces the
            // negative default NaN for these, arm64 the positive one.
            binary_cases.extend([
                (FloatBinOp::Add, inf, neg_inf),
                (FloatBinOp::Sub, inf, inf),
                (FloatBinOp::Mul, zero, inf),
                (FloatBinOp::Mul, neg_inf, zero),
                (FloatBinOp::Div, zero, zero),
                (FloatBinOp::Div, inf, neg_inf),
                (FloatBinOp::FloorDiv, zero, zero),
            ]);
            for (op, lhs, rhs) in binary_cases {
                let got = raw_bits(float_binop(op, scalar(prim, lhs), scalar(prim, rhs)).unwrap());
                assert_eq!(
                    got, canonical,
                    "{name} {op:?}({lhs:#x}, {rhs:#x}) gave {got:#x}"
                );
                let tensor = float_tensor_binop(
                    op,
                    &tensor_from_scalars(prim, &[scalar(prim, lhs)]),
                    &tensor_from_scalars(prim, &[scalar(prim, rhs)]),
                )
                .unwrap();
                let got = raw_bits(tensor.scalar_at(0));
                assert_eq!(
                    got, canonical,
                    "{name} tensor {op:?}({lhs:#x}, {rhs:#x}) gave {got:#x}"
                );
            }
            let mut unary_cases = Vec::new();
            for op in unary {
                for nan in nans {
                    unary_cases.push((op, nan));
                }
            }
            unary_cases.extend([
                (FloatUnOp::Sqrt, minus_one),
                (FloatUnOp::Log, minus_one),
                (FloatUnOp::Sin, inf),
                (FloatUnOp::Cos, neg_inf),
                (FloatUnOp::Tan, inf),
            ]);
            for (op, input) in unary_cases {
                let got = raw_bits(float_unop(op, scalar(prim, input)).unwrap());
                assert_eq!(got, canonical, "{name} {op:?}({input:#x}) gave {got:#x}");
                let tensor =
                    float_tensor_unop(op, &tensor_from_scalars(prim, &[scalar(prim, input)]))
                        .unwrap();
                let got = raw_bits(tensor.scalar_at(0));
                assert_eq!(
                    got, canonical,
                    "{name} tensor {op:?}({input:#x}) gave {got:#x}"
                );
            }
        }
    }

    #[test]
    fn scalar_extrema_preserve_first_nan_and_lhs_signed_zero_bits_at_every_float_width() {
        fn assert_same_bits(actual: ScalarValue, expected: ScalarValue) {
            match (actual.bits, expected.bits) {
                (Bits::F16(actual), Bits::F16(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                (Bits::Bf16(actual), Bits::Bf16(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                (Bits::F32(actual), Bits::F32(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                (Bits::F64(actual), Bits::F64(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits())
                }
                _ => panic!("float extrema must preserve the expected dtype"),
            }
        }

        macro_rules! assert_width {
            ($variant:ident, $float:ty, $left_nan:expr, $right_nan:expr) => {{
                let left_nan = ScalarValue {
                    bits: Bits::$variant(<$float>::from_bits($left_nan)),
                };
                let right_nan = ScalarValue {
                    bits: Bits::$variant(<$float>::from_bits($right_nan)),
                };
                let one = ScalarValue {
                    bits: Bits::$variant(<$float>::from_f32(1.0)),
                };
                assert_same_bits(
                    float_binop(FloatBinOp::Max, left_nan, one).unwrap(),
                    left_nan,
                );
                assert_same_bits(
                    float_binop(FloatBinOp::Min, left_nan, one).unwrap(),
                    left_nan,
                );
                assert_same_bits(
                    float_binop(FloatBinOp::Max, one, right_nan).unwrap(),
                    right_nan,
                );
                assert_same_bits(
                    float_binop(FloatBinOp::Min, one, right_nan).unwrap(),
                    right_nan,
                );
                assert_same_bits(
                    float_binop(FloatBinOp::Max, left_nan, right_nan).unwrap(),
                    left_nan,
                );
                assert_same_bits(
                    float_binop(FloatBinOp::Min, left_nan, right_nan).unwrap(),
                    left_nan,
                );

                let negative_zero = ScalarValue {
                    bits: Bits::$variant(<$float>::from_f32(-0.0)),
                };
                let positive_zero = ScalarValue {
                    bits: Bits::$variant(<$float>::from_f32(0.0)),
                };
                assert_same_bits(
                    float_binop(FloatBinOp::Max, negative_zero, positive_zero).unwrap(),
                    negative_zero,
                );
                assert_same_bits(
                    float_binop(FloatBinOp::Min, positive_zero, negative_zero).unwrap(),
                    positive_zero,
                );
            }};
        }

        assert_width!(F16, half::f16, 0xfe01, 0x7e55);
        assert_width!(Bf16, half::bf16, 0xffc1, 0x7fe5);

        let left_nan = ScalarValue {
            bits: Bits::F32(f32::from_bits(0xffc1_2345)),
        };
        let right_nan = ScalarValue {
            bits: Bits::F32(f32::from_bits(0x7fc5_4321)),
        };
        let one = ScalarValue {
            bits: Bits::F32(1.0),
        };
        assert_same_bits(
            float_binop(FloatBinOp::Max, left_nan, one).unwrap(),
            left_nan,
        );
        assert_same_bits(
            float_binop(FloatBinOp::Min, one, right_nan).unwrap(),
            right_nan,
        );
        assert_same_bits(
            float_binop(FloatBinOp::Max, left_nan, right_nan).unwrap(),
            left_nan,
        );
        assert_same_bits(
            float_binop(
                FloatBinOp::Max,
                ScalarValue {
                    bits: Bits::F32(-0.0),
                },
                ScalarValue {
                    bits: Bits::F32(0.0),
                },
            )
            .unwrap(),
            ScalarValue {
                bits: Bits::F32(-0.0),
            },
        );

        let left_nan = ScalarValue {
            bits: Bits::F64(f64::from_bits(0xfff8_1234_5678_9abc)),
        };
        let right_nan = ScalarValue {
            bits: Bits::F64(f64::from_bits(0x7ff8_abcd_1234_5678)),
        };
        let one = ScalarValue {
            bits: Bits::F64(1.0),
        };
        assert_same_bits(
            float_binop(FloatBinOp::Max, left_nan, one).unwrap(),
            left_nan,
        );
        assert_same_bits(
            float_binop(FloatBinOp::Min, one, right_nan).unwrap(),
            right_nan,
        );
        assert_same_bits(
            float_binop(FloatBinOp::Min, left_nan, right_nan).unwrap(),
            left_nan,
        );
        assert_same_bits(
            float_binop(
                FloatBinOp::Min,
                ScalarValue {
                    bits: Bits::F64(0.0),
                },
                ScalarValue {
                    bits: Bits::F64(-0.0),
                },
            )
            .unwrap(),
            ScalarValue {
                bits: Bits::F64(0.0),
            },
        );
    }

    #[test]
    fn tensor_extrema_select_stored_operands_without_reencoding_payloads() {
        macro_rules! assert_width {
            ($variant:ident, $float:ty, $left_nan:expr, $right_nan:expr) => {{
                let left = TensorStorage {
                    buf: Buf::$variant(vec![
                        <$float>::from_bits($left_nan),
                        <$float>::from_f32(-0.0),
                        <$float>::from_f32(4.0),
                    ]),
                };
                let right = TensorStorage {
                    buf: Buf::$variant(vec![
                        <$float>::from_bits($right_nan),
                        <$float>::from_f32(0.0),
                        <$float>::from_f32(2.0),
                    ]),
                };
                let max = float_tensor_binop(FloatBinOp::Max, &left, &right).unwrap();
                let min = float_tensor_binop(FloatBinOp::Min, &left, &right).unwrap();
                match (max.buf, min.buf) {
                    (Buf::$variant(max), Buf::$variant(min)) => {
                        assert_eq!(max[0].to_bits(), $left_nan);
                        assert_eq!(min[0].to_bits(), $left_nan);
                        assert_eq!(max[1].to_bits(), <$float>::from_f32(-0.0).to_bits());
                        assert_eq!(min[1].to_bits(), <$float>::from_f32(-0.0).to_bits());
                        assert_eq!(max[2], <$float>::from_f32(4.0));
                        assert_eq!(min[2], <$float>::from_f32(2.0));
                    }
                    _ => unreachable!("same-dtype kernels preserve their storage variant"),
                }
            }};
        }

        assert_width!(F16, half::f16, 0xfe01, 0x7e55);
        assert_width!(Bf16, half::bf16, 0xffc1, 0x7fe5);

        let left = TensorStorage {
            buf: Buf::F32(vec![f32::from_bits(0xffc1_2345), -0.0, 4.0]),
        };
        let right = TensorStorage {
            buf: Buf::F32(vec![f32::from_bits(0x7fc5_4321), 0.0, 2.0]),
        };
        let max = float_tensor_binop(FloatBinOp::Max, &left, &right).unwrap();
        let min = float_tensor_binop(FloatBinOp::Min, &left, &right).unwrap();
        match (max.buf, min.buf) {
            (Buf::F32(max), Buf::F32(min)) => {
                assert_eq!(max[0].to_bits(), 0xffc1_2345);
                assert_eq!(min[0].to_bits(), 0xffc1_2345);
                assert_eq!(max[1].to_bits(), (-0.0_f32).to_bits());
                assert_eq!(min[1].to_bits(), (-0.0_f32).to_bits());
                assert_eq!(max[2], 4.0);
                assert_eq!(min[2], 2.0);
            }
            _ => unreachable!("f32 kernels preserve f32 storage"),
        }

        let left = TensorStorage {
            buf: Buf::F64(vec![f64::from_bits(0xfff8_1234_5678_9abc), -0.0, 4.0]),
        };
        let right = TensorStorage {
            buf: Buf::F64(vec![f64::from_bits(0x7ff8_abcd_1234_5678), 0.0, 2.0]),
        };
        let max = float_tensor_binop(FloatBinOp::Max, &left, &right).unwrap();
        let min = float_tensor_binop(FloatBinOp::Min, &left, &right).unwrap();
        match (max.buf, min.buf) {
            (Buf::F64(max), Buf::F64(min)) => {
                assert_eq!(max[0].to_bits(), 0xfff8_1234_5678_9abc);
                assert_eq!(min[0].to_bits(), 0xfff8_1234_5678_9abc);
                assert_eq!(max[1].to_bits(), (-0.0_f64).to_bits());
                assert_eq!(min[1].to_bits(), (-0.0_f64).to_bits());
                assert_eq!(max[2], 4.0);
                assert_eq!(min[2], 2.0);
            }
            _ => unreachable!("f64 kernels preserve f64 storage"),
        }
    }

    #[test]
    fn extrema_adjoint_routes_complete_cotangents_by_the_exact_forward_rule() {
        macro_rules! assert_width {
            (
                $variant:ident,
                $left_nan:expr,
                $right_nan:expr,
                $zero:expr,
                $one:expr,
                $two:expr,
                $three:expr,
                $four:expr
            ) => {{
                let lhs = TensorStorage {
                    buf: Buf::$variant(vec![$left_nan, $one, $four, $zero]),
                };
                let rhs = TensorStorage {
                    buf: Buf::$variant(vec![$right_nan, $right_nan, $two, $zero]),
                };
                let g = TensorStorage {
                    buf: Buf::$variant(vec![$one, $two, $three, $four]),
                };

                assert_eq!(
                    float_extrema_adjoint(
                        FloatExtremaOp::Max,
                        ExtremaOperand::Left,
                        &lhs,
                        &rhs,
                        &g,
                    )
                    .unwrap(),
                    TensorStorage {
                        buf: Buf::$variant(vec![$one, $zero, $three, $four]),
                    }
                );
                assert_eq!(
                    float_extrema_adjoint(
                        FloatExtremaOp::Max,
                        ExtremaOperand::Right,
                        &lhs,
                        &rhs,
                        &g,
                    )
                    .unwrap(),
                    TensorStorage {
                        buf: Buf::$variant(vec![$zero, $two, $zero, $zero]),
                    }
                );
                assert_eq!(
                    float_extrema_adjoint(
                        FloatExtremaOp::Min,
                        ExtremaOperand::Left,
                        &lhs,
                        &rhs,
                        &g,
                    )
                    .unwrap(),
                    TensorStorage {
                        buf: Buf::$variant(vec![$one, $zero, $zero, $four]),
                    }
                );
                assert_eq!(
                    float_extrema_adjoint(
                        FloatExtremaOp::Min,
                        ExtremaOperand::Right,
                        &lhs,
                        &rhs,
                        &g,
                    )
                    .unwrap(),
                    TensorStorage {
                        buf: Buf::$variant(vec![$zero, $two, $three, $zero]),
                    }
                );
            }};
        }

        assert_width!(
            F16,
            half::f16::from_bits(0xfe01),
            half::f16::from_bits(0x7e55),
            half::f16::ZERO,
            half::f16::ONE,
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0)
        );
        assert_width!(
            Bf16,
            half::bf16::from_bits(0xffc1),
            half::bf16::from_bits(0x7fe5),
            half::bf16::ZERO,
            half::bf16::ONE,
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(3.0),
            half::bf16::from_f32(4.0)
        );
        assert_width!(
            F32,
            f32::from_bits(0xffc1_2345),
            f32::from_bits(0x7fc5_4321),
            0.0_f32,
            1.0_f32,
            2.0_f32,
            3.0_f32,
            4.0_f32
        );
        assert_width!(
            F64,
            f64::from_bits(0xfff8_1234_5678_9abc),
            f64::from_bits(0x7ff8_abcd_1234_5678),
            0.0_f64,
            1.0_f64,
            2.0_f64,
            3.0_f64,
            4.0_f64
        );
    }

    #[test]
    fn relu_preserves_positive_nan_and_negative_zero_bits_at_every_float_width() {
        macro_rules! assert_width {
            ($variant:ident, $nan:expr, $neg_zero:expr, $negative:expr, $positive:expr, $zero:expr) => {{
                let input = TensorStorage {
                    buf: Buf::$variant(vec![$nan, $neg_zero, $negative, $positive]),
                };
                let output = float_relu(&input).unwrap();
                let expected = [$nan, $neg_zero, $zero, $positive];
                let Buf::$variant(actual) = &output.buf else {
                    unreachable!()
                };
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );

                let legacy = float_tensor_unop(FloatUnOp::Relu, &input).unwrap();
                let Buf::$variant(legacy) = &legacy.buf else {
                    unreachable!()
                };
                assert_eq!(
                    legacy
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "the legacy activation entry must route through the exact selector"
                );

                for (index, expected) in expected.iter().enumerate() {
                    let ScalarValue {
                        bits: Bits::$variant(scalar),
                    } = float_unop(FloatUnOp::Relu, input.scalar_at(index)).unwrap()
                    else {
                        unreachable!()
                    };
                    assert_eq!(
                        scalar.to_bits(),
                        expected.to_bits(),
                        "the scalar activation entry must keep selected operand bits"
                    );
                }
            }};
        }

        assert_width!(
            F16,
            half::f16::from_bits(0xfe01),
            half::f16::NEG_ZERO,
            half::f16::from_f32(-1.0),
            half::f16::from_bits(1),
            half::f16::ZERO
        );
        assert_width!(
            Bf16,
            half::bf16::from_bits(0xffc1),
            half::bf16::NEG_ZERO,
            half::bf16::from_f32(-1.0),
            half::bf16::from_bits(1),
            half::bf16::ZERO
        );
        assert_width!(
            F32,
            f32::from_bits(0xffc1_2345),
            -0.0_f32,
            -1.0_f32,
            f32::from_bits(1),
            0.0_f32
        );
        assert_width!(
            F64,
            f64::from_bits(0xfff8_1234_5678_9abc),
            -0.0_f64,
            -1.0_f64,
            f64::from_bits(1),
            0.0_f64
        );
    }

    #[test]
    fn relu_adjoint_selects_complete_cotangent_only_for_strictly_positive_inputs() {
        macro_rules! assert_width {
            ($variant:ident, $nan:expr, $neg_zero:expr, $negative:expr, $positive:expr, $subnormal:expr, $g_nan:expr, $g_inf:expr, $g_neg_zero:expr, $zero:expr) => {{
                let input = TensorStorage {
                    buf: Buf::$variant(vec![
                        $nan, $neg_zero, $zero, $negative, $positive, $subnormal,
                    ]),
                };
                let cotangent = TensorStorage {
                    buf: Buf::$variant(vec![
                        $g_nan,
                        $g_inf,
                        $g_neg_zero,
                        $g_nan,
                        $g_neg_zero,
                        $g_inf,
                    ]),
                };
                assert_eq!(
                    float_relu_adjoint(&input, &cotangent).unwrap(),
                    TensorStorage {
                        buf: Buf::$variant(vec![$zero, $zero, $zero, $zero, $g_neg_zero, $g_inf]),
                    }
                );
            }};
        }

        assert_width!(
            F16,
            half::f16::from_bits(0xfe01),
            half::f16::NEG_ZERO,
            half::f16::from_f32(-1.0),
            half::f16::ONE,
            half::f16::from_bits(1),
            half::f16::from_bits(0x7e55),
            half::f16::INFINITY,
            half::f16::NEG_ZERO,
            half::f16::ZERO
        );
        assert_width!(
            Bf16,
            half::bf16::from_bits(0xffc1),
            half::bf16::NEG_ZERO,
            half::bf16::from_f32(-1.0),
            half::bf16::ONE,
            half::bf16::from_bits(1),
            half::bf16::from_bits(0x7fe5),
            half::bf16::INFINITY,
            half::bf16::NEG_ZERO,
            half::bf16::ZERO
        );
        assert_width!(
            F32,
            f32::from_bits(0xffc1_2345),
            -0.0_f32,
            -1.0_f32,
            1.0_f32,
            f32::from_bits(1),
            f32::from_bits(0x7fc5_4321),
            f32::INFINITY,
            -0.0_f32,
            0.0_f32
        );
        assert_width!(
            F64,
            f64::from_bits(0xfff8_1234_5678_9abc),
            -0.0_f64,
            -1.0_f64,
            1.0_f64,
            f64::from_bits(1),
            f64::from_bits(0x7ff8_abcd_1234_5678),
            f64::INFINITY,
            -0.0_f64,
            0.0_f64
        );
    }

    #[test]
    fn relu_kernels_reject_non_float_and_mismatched_cotangent_storage() {
        let integers = finalize_tensor("test", Prim::Int32, RawTensor::Int(vec![1])).unwrap();
        let floats = finalize_tensor("test", Prim::F32, RawTensor::Float(vec![1.0])).unwrap();
        let doubles = finalize_tensor("test", Prim::F64, RawTensor::Float(vec![1.0])).unwrap();

        assert!(matches!(
            float_relu(&integers),
            Err(NumericKernelError::WrongFamily { op: "relu", .. })
        ));
        assert!(matches!(
            float_relu_adjoint(&integers, &integers),
            Err(NumericKernelError::WrongFamily {
                op: "relu_adjoint",
                ..
            })
        ));
        assert!(matches!(
            float_relu_adjoint(&floats, &doubles),
            Err(NumericKernelError::DtypeMismatch {
                op: "relu_adjoint",
                ..
            })
        ));
    }
}

#[cfg(test)]
mod round_float_bound_tests {
    use super::{Prim, round_float_bound};

    /// The chelis#2316 repro, at the value level. `cast(0.30000001, f16)` is
    /// `0.300048828125`, and widening that to f32 keeps it — so the chain's
    /// value is NOT the literal it started from. Both compiled fold sites used
    /// to return the literal.
    #[test]
    fn f16_then_f32_keeps_the_f16_rounding() {
        let inner = round_float_bound(Prim::F16, 0.30000001).expect("f16 is a float target");
        assert_eq!(inner, 0.300048828125);
        let widened = round_float_bound(Prim::F32, inner).expect("f32 is a float target");
        assert_eq!((widened as f32).to_bits(), 0x3e99a000);
        // What the broken fold baked instead: the untouched literal, whose f32
        // image is 0x3e99999a. That is the exact constant chelis#2316 found in
        // the emitted `chelis_uniform_sample_f32` call.
        assert_ne!((widened as f32).to_bits(), 0x3e99999a_u32);
    }

    /// The high bound from the same repro.
    #[test]
    fn f16_then_f32_high_bound_matches_the_declared_value() {
        let inner = round_float_bound(Prim::F16, 0.90000001).expect("f16 is a float target");
        let widened = round_float_bound(Prim::F32, inner).expect("f32 is a float target");
        assert_eq!(widened, 0.89990234375);
        assert_eq!((widened as f32).to_bits(), 0x3f666000);
    }

    /// bf16 has fewer mantissa bits than f16, so it rounds further. This is
    /// why the bf16 program diverged more than the f16 one.
    #[test]
    fn bf16_rounds_further_than_f16() {
        let f16 = round_float_bound(Prim::F16, 0.30000001).expect("float target");
        let bf16 = round_float_bound(Prim::Bf16, 0.30000001).expect("float target");
        assert_ne!(f16, bf16);
        assert!(
            (bf16 - 0.30000001f64).abs() > (f16 - 0.30000001f64).abs(),
            "bf16 {bf16} should be further from the source than f16 {f16}",
        );
    }

    /// DISPOSITION LOCK: a target that cannot narrow the value returns it
    /// unchanged, which is why a lone `cast(lit, f32)` bound always agreed
    /// across lanes even with the broken fold.
    #[test]
    fn a_non_narrowing_target_is_the_identity() {
        let v = 0.30000001192092896_f64; // already exactly f32-representable
        assert_eq!(round_float_bound(Prim::F32, v), Some(v));
        assert_eq!(round_float_bound(Prim::F64, v), Some(v));
    }

    /// Ties-to-even at the target width, per [04-NUM-2] / [04-NUM-14].
    #[test]
    fn rounding_is_ties_to_even_not_truncation() {
        // A value just above an f16 midpoint must round up, not truncate down.
        let up = round_float_bound(Prim::F16, 0.30004884).expect("float target");
        assert!(
            up >= 0.300048828125,
            "expected round-to-nearest, got truncation: {up}",
        );
    }

    /// NEGATIVE PARITY (chelis#776): an integer target is left unresolved so
    /// the caller goes loud rather than baking a guessed truncation. Both fold
    /// sites depend on this `None`.
    #[test]
    fn integer_and_bool_targets_are_unresolved() {
        for prim in [
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ] {
            assert_eq!(
                round_float_bound(prim, 2.7),
                None,
                "{prim:?} must not resolve as a float bound",
            );
        }
    }

    /// Non-finite values are not bounds the emitter can bake; they must not
    /// silently become a finite number.
    #[test]
    fn non_finite_values_do_not_become_finite() {
        for v in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            if let Some(rounded) = round_float_bound(Prim::F32, v) {
                assert!(
                    !rounded.is_finite(),
                    "non-finite {v} must not round to the finite {rounded}",
                );
            }
        }
    }
}
