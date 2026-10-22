use chelis_vocab::RuntimeDType;
use core::fmt;

/// Numeric precision types.
///
/// The active numeric primitive set is pinned by `spec/04-type-system.md` §1.1:
/// `f32`, `f64`, `bf16`, `f16`, `i8`, `i16`, `i32`, `i64`, plus `bool`
/// and `string`. The `f8e4m3` variant is reserved per §1.1.1 but is **not
/// active** in this dtype build-out cycle: parse paths still produce
/// `Prim::F8e4m3` so producers can be diagnosed precisely, but every
/// admissibility predicate below excludes it. See [`Prim::is_admissible_active`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Prim {
    F32,
    F64,
    F16,
    Bf16,
    /// Deferred per `spec/04-type-system.md` §1.1.1. Not part of the active
    /// numeric primitive set; rejected at check time wherever it appears.
    F8e4m3,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
    String,
    /// A random key (spec/04 §1.1, [05-RNG-2]): an active tensor element
    /// dtype that is structurally non-numeric, with no arithmetic,
    /// comparison, cast, or literal carrier. A `tensor[D, key]` is stored
    /// with runtime dtype `key`. It is not one of the nine active data element
    /// dtypes ([`Prim::is_data_element_dtype`]), so an operation scheme admits
    /// it only where its own atom names `key`. Every key value is produced by
    /// a graph node or enters as an input. It has no source spelling yet; its
    /// interchange spelling is `key`.
    Key,
}

// Canonical identity order for order-free collection indices. This is byte
// order over the public dtype spelling, not a semantic numeric ordinal.
impl PartialOrd for Prim {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Prim {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.name().cmp(other.name())
    }
}

impl Prim {
    /// The active float dtypes of `spec/04-type-system.md` §1.1, the one list
    /// [`Prim::is_float`] and every float-family enumeration read.
    pub const ACTIVE_FLOATS: [Prim; 4] = [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16];

    /// The active signed integer dtypes of §1.1, narrowest first: the one list
    /// [`Prim::is_integer`] and every integer-family enumeration read.
    pub const ACTIVE_INTEGERS: [Prim; 4] = [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64];

    /// Parse a primitive type from its canonical name.
    pub fn parse_name(s: &str) -> Option<Prim> {
        match s {
            "f32" => Some(Prim::F32),
            "f64" => Some(Prim::F64),
            "f16" => Some(Prim::F16),
            "bf16" => Some(Prim::Bf16),
            "f8e4m3" => Some(Prim::F8e4m3),
            "i8" => Some(Prim::Int8),
            "i16" => Some(Prim::Int16),
            "i32" => Some(Prim::Int32),
            "i64" => Some(Prim::Int64),
            "bool" => Some(Prim::Bool),
            "string" => Some(Prim::String),
            "key" => Some(Prim::Key),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Prim::F32 => "f32",
            Prim::F64 => "f64",
            Prim::F16 => "f16",
            Prim::Bf16 => "bf16",
            Prim::F8e4m3 => "f8e4m3",
            Prim::Int8 => "i8",
            Prim::Int16 => "i16",
            Prim::Int32 => "i32",
            Prim::Int64 => "i64",
            Prim::Bool => "bool",
            Prim::String => "string",
            Prim::Key => "key",
        }
    }

    /// Stable ecosystem spelling used by versioned JSON and ABI-facing
    /// interchange surfaces. This is deliberately separate from [`Self::name`]:
    /// Chelis source and Deep use `i*`, while existing external payloads keep
    /// their `int*` identities.
    pub fn interchange_name(&self) -> &'static str {
        match self {
            Prim::Int8 => "int8",
            Prim::Int16 => "int16",
            Prim::Int32 => "int32",
            Prim::Int64 => "int64",
            _ => self.name(),
        }
    }

    /// Parse the stable ecosystem spelling used by versioned interchange
    /// formats. Language and Deep ingress must use [`Self::parse_name`].
    pub fn parse_interchange_name(s: &str) -> Option<Prim> {
        match s {
            "int8" => Some(Prim::Int8),
            "int16" => Some(Prim::Int16),
            "int32" => Some(Prim::Int32),
            "int64" => Some(Prim::Int64),
            "i8" | "i16" | "i32" | "i64" => None,
            _ => Self::parse_name(s),
        }
    }

    /// Map an active, runtime-storable primitive to the shared ABI dtype.
    /// Deferred and non-tensor primitives fail instead of borrowing another
    /// dtype's tag.
    pub fn runtime_dtype(self) -> Result<RuntimeDType, RuntimeDTypeMappingError> {
        match self {
            Prim::F32 => Ok(RuntimeDType::F32),
            Prim::F64 => Ok(RuntimeDType::F64),
            Prim::F16 => Ok(RuntimeDType::F16),
            Prim::Bf16 => Ok(RuntimeDType::Bf16),
            Prim::Int8 => Ok(RuntimeDType::I8),
            Prim::Int16 => Ok(RuntimeDType::I16),
            Prim::Int32 => Ok(RuntimeDType::I32),
            Prim::Int64 => Ok(RuntimeDType::I64),
            Prim::Bool => Ok(RuntimeDType::Bool),
            Prim::Key => Ok(RuntimeDType::Key),
            Prim::F8e4m3 | Prim::String => Err(RuntimeDTypeMappingError { prim: self }),
        }
    }

    /// The language dtype a runtime ABI dtype stores, the inverse of
    /// [`Self::runtime_dtype`]. Every runtime dtype has one, so a runtime
    /// dtype added without its language dtype fails to compile here.
    pub fn from_runtime_dtype(dtype: RuntimeDType) -> Prim {
        match dtype {
            RuntimeDType::F32 => Prim::F32,
            RuntimeDType::F64 => Prim::F64,
            RuntimeDType::F16 => Prim::F16,
            RuntimeDType::Bf16 => Prim::Bf16,
            RuntimeDType::I8 => Prim::Int8,
            RuntimeDType::I16 => Prim::Int16,
            RuntimeDType::I32 => Prim::Int32,
            RuntimeDType::I64 => Prim::Int64,
            RuntimeDType::Bool => Prim::Bool,
            RuntimeDType::Key => Prim::Key,
        }
    }

    /// True for the active float dtypes per `spec/04-type-system.md` §1.1.
    /// `f8e4m3` is deferred (§1.1.1) and is NOT a float for any active
    /// classification purpose.
    pub fn is_float(&self) -> bool {
        Self::ACTIVE_FLOATS.contains(self)
    }

    /// True for any numeric precision in the active set, including `f8e4m3`
    /// (so deferred-dtype rejection sites can still treat it as numeric for
    /// surface diagnostics). `Bool`, `String` and `Key` are not numeric.
    pub fn is_numeric(&self) -> bool {
        !matches!(self, Prim::Bool | Prim::String | Prim::Key)
    }

    /// True for all signed integer dtypes in the active set per §1.1.
    pub fn is_integer(&self) -> bool {
        Self::ACTIVE_INTEGERS.contains(self)
    }

    /// The representable `[min, max]` range of a signed integer width, or
    /// `None` for a non-integer primitive. The single type-system source for
    /// the per-width integer range.
    pub fn integer_range(&self) -> Option<(i64, i64)> {
        Some(match self {
            Prim::Int8 => (i8::MIN as i64, i8::MAX as i64),
            Prim::Int16 => (i16::MIN as i64, i16::MAX as i64),
            Prim::Int32 => (i32::MIN as i64, i32::MAX as i64),
            Prim::Int64 => (i64::MIN, i64::MAX),
            _ => return None,
        })
    }

    /// The integer fuzz-sampling `[min, max]` window: the `[-1000, 1000]`
    /// convenience range clamped to the width's [`integer_range`], or `None`
    /// for a non-integer primitive. The SINGLE source every prove-path
    /// integer sampler (the obligation engine, the property runner, the
    /// injection path, and the CLI fuzz sampler) shares, so a narrow width
    /// (e.g. i8) samples in `[-128, 127]` everywhere -- never an
    /// unrepresentable value that would yield a spurious counterexample.
    pub fn integer_fuzz_bounds(&self) -> Option<(i64, i64)> {
        let (lo, hi) = self.integer_range()?;
        Some((lo.max(-1000), hi.min(1000)))
    }

    /// True if this primitive is in the **active** numeric/scalar set per
    /// `spec/04-type-system.md` §1.1. Excludes the deferred `f8e4m3`
    /// (§1.1.1). Use this predicate as the canonical "is this dtype
    /// admitted in this cycle?" check across the type checker, IR builder,
    /// and backends.
    pub fn is_admissible_active(&self) -> bool {
        !matches!(self, Prim::F8e4m3)
    }

    /// Whether this precision is one of spec §1.1's ten active tensor element
    /// dtypes: f32, f64, bf16, f16, i8, i16, i32, i64, bool, and key. The
    /// deferred `f8e4m3` (§1.1.1) is rejected. This says only that a
    /// `tensor[D, p]` exists; an operation scheme that admits every element
    /// dtype admits [`Prim::is_data_element_dtype`]'s nine, and `key` only
    /// where its own atom names it.
    ///
    /// Note: backend support for the reduced floats (`f16`, `bf16`) is
    /// staged separately in WS-A1/A2/A3; the type checker admits them here
    /// because the spec lists them as active dtypes. Backends that cannot
    /// yet emit them are expected to produce their own targeted diagnostic
    /// rather than let them slip through silently.
    pub fn is_valid_tensor_precision(&self) -> bool {
        self.is_data_element_dtype() || matches!(self, Prim::Key)
    }

    /// Whether this precision is one of spec §1.1's nine active data element
    /// dtypes: the active tensor element dtypes other than `key`, that is the
    /// eight numeric dtypes and bool. A domain written as every active tensor
    /// element dtype is exactly this set.
    pub fn is_data_element_dtype(&self) -> bool {
        matches!(
            self,
            Prim::F32
                | Prim::F64
                | Prim::Bf16
                | Prim::F16
                | Prim::Bool
                | Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64
        )
    }

    /// Whether this precision is a legitimate target for `cast(scalar, p)`.
    /// Mirrors `is_data_element_dtype` in this cycle: the host scalar lane
    /// carries the same active data dtype set per spec §1.1, and the deferred
    /// `f8e4m3` (§1.1.1) is rejected here too. A key has no cast.
    pub fn is_valid_scalar_cast_target(&self) -> bool {
        matches!(
            self,
            Prim::F32
                | Prim::F64
                | Prim::Bf16
                | Prim::F16
                | Prim::Bool
                | Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64
        )
    }

    /// Resolve the spec/04 section 5.7.1 default reduce-sum accumulator precision.
    /// The type checker, IR, and runtime use this one policy.
    ///
    /// - bf16 / f16  → f32
    /// - f32         → f32 (operand-matching)
    /// - f64         → f64 (operand-matching)
    /// - i8 / i16 → i32
    /// - i32       → i32 (operand-matching)
    /// - i64       → i64 (operand-matching)
    ///
    /// Returns `Err` for non-numeric operands and for the deferred
    /// `f8e4m3` (§1.1.1).
    pub fn default_reduce_sum_accumulator(self) -> Result<Prim, ReductionPolicyError> {
        Ok(match self {
            Prim::Bf16 | Prim::F16 => Prim::F32,
            Prim::F32 => Prim::F32,
            Prim::F64 => Prim::F64,
            Prim::Int8 | Prim::Int16 => Prim::Int32,
            Prim::Int32 => Prim::Int32,
            Prim::Int64 => Prim::Int64,
            Prim::Bool => return Err(ReductionPolicyError::BoolOperand),
            Prim::F8e4m3 => return Err(ReductionPolicyError::DeferredOperand),
            Prim::String => return Err(ReductionPolicyError::StringOperand),
            Prim::Key => return Err(ReductionPolicyError::RandomKeyOperand),
        })
    }

    /// spec/04 §5.7.1 `sum_result(p, a)` for this operand `p` and an
    /// admitted accumulator `a`: the operand dtype for `bf16` and `f16`,
    /// otherwise the accumulator dtype.
    pub fn sum_result_precision(self, accumulator: Prim) -> Prim {
        match self {
            Prim::Bf16 | Prim::F16 => self,
            _ => accumulator,
        }
    }

    /// Resolve the spec/04-type-system.md §5.7.1 user-facing
    /// **result** precision of `reduce_sum` for this operand precision,
    /// per the "Result precision" column of the §5.7.1 table:
    ///
    /// - bf16 / f16  → operand precision (f32 accumulator consumed
    ///   inside the op and downcast on output)
    /// - f32         → f32
    /// - f64         → f64
    /// - i8 / i16 → i32 (accumulator precision)
    /// - i32       → i32
    /// - i64       → i64
    ///
    /// This is the user-visible result precision. The IR-level Sum
    /// node's `output_type.precision` is the accumulator precision;
    /// lowering inserts a `Cast` for the bf16/f16 row to recover the
    /// operand-precision result.
    pub fn default_reduce_sum_result_precision(self) -> Result<Prim, ReductionPolicyError> {
        match self {
            Prim::Bf16 => Ok(Prim::Bf16),
            Prim::F16 => Ok(Prim::F16),
            other => other.default_reduce_sum_accumulator(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeDTypeMappingError {
    pub prim: Prim,
}

impl fmt::Display for RuntimeDTypeMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "primitive `{}` has no runtime dtype", self.prim.name())
    }
}

impl core::error::Error for RuntimeDTypeMappingError {}

/// A reduction policy failure without owned text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionPolicyError {
    BoolOperand,
    DeferredOperand,
    StringOperand,
    RandomKeyOperand,
}

impl fmt::Display for ReductionPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BoolOperand => "reduce_sum is not defined on bool tensors; cast to i32 first",
            Self::DeferredOperand => {
                "reduce_sum: operand dtype `f8e4m3` is deferred per \
                     spec/04-type-system.md §1.1.1 and is not part of the \
                     active numeric primitive set"
            }
            Self::StringOperand => "reduce_sum is not defined for string operands",
            Self::RandomKeyOperand => "reduce_sum is not defined for random key operands",
        })
    }
}

impl core::error::Error for ReductionPolicyError {}
