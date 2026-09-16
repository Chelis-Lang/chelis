//! Internal type representation for the Chelis type checker.
//!
//! These are the checker's working types — NOT the Deep AST nodes.
//! They mirror the Deep t-* tags but are Rust-native for efficient unification.

use std::collections::BTreeSet;
use std::fmt;

use chelis_vocab::RuntimeDType;
use serde::{Deserialize, Serialize};

use crate::errors::ErrorWitness;

/// A unique identifier for a type variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypeVar(pub u32);

/// A semantic domain attached to a quantified type variable.
///
/// Unlike a callee-name check, a restriction is part of the polymorphic
/// scheme itself. Instantiation installs it on the fresh inference variable,
/// ordinary unification propagates it through aliases, and generalization
/// re-quantifies it on wrappers and higher-order values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeVarRestriction {
    /// `spec/04-type-system.md` §5.9 `Float`: the variable may instantiate
    /// only at an active float primitive (`f16`, `bf16`, `f32`, `f64`).
    ActiveFloat,
    /// §5.9 `Int`: only at an active signed integer primitive (`i8`,
    /// `i16`, `i32`, `i64`).
    ActiveInt,
    /// §5.9 `Numeric`: the union of [`TypeVarRestriction::ActiveFloat`] and
    /// [`TypeVarRestriction::ActiveInt`]. `bool` and `string` are excluded,
    /// as are the §1.1.1 reserved spellings.
    ActiveNumeric,
    /// An operation argument is a scalar or tensor with float precision.
    /// This is a value constraint, not the surface `Float` type-argument bound.
    FloatValue,
    /// An operation argument is a scalar or tensor with signed integer precision.
    IntValue,
    /// An operation argument is a scalar or tensor with numeric precision.
    NumericValue,
}

impl TypeVarRestriction {
    /// The `spec/04-type-system.md` §5.9 family name, as diagnostics and the
    /// Surf surface spell it.
    pub fn family_name(self) -> &'static str {
        match self {
            TypeVarRestriction::ActiveFloat | TypeVarRestriction::FloatValue => "Float",
            TypeVarRestriction::ActiveInt | TypeVarRestriction::IntValue => "Int",
            TypeVarRestriction::ActiveNumeric | TypeVarRestriction::NumericValue => "Numeric",
        }
    }

    /// The family's membership, spelled for a diagnostic reader who has not
    /// read §5.9.
    pub fn membership_gloss(self) -> &'static str {
        match self {
            TypeVarRestriction::ActiveFloat => "the active float dtypes",
            TypeVarRestriction::ActiveInt => "the active signed integer dtypes",
            TypeVarRestriction::ActiveNumeric => "the active numeric dtypes",
            TypeVarRestriction::FloatValue => "float scalars or tensors",
            TypeVarRestriction::IntValue => "signed integer scalars or tensors",
            TypeVarRestriction::NumericValue => "numeric scalars or tensors",
        }
    }

    /// Whether `prim` is a member of this family.
    ///
    /// Membership follows §1.1's active set through [`Prim::is_float`] and
    /// [`Prim::is_integer`], both of which already exclude the §1.1.1
    /// deferred `f8e4m3`. `Prim::is_numeric` is deliberately NOT used for
    /// `ActiveNumeric`: it admits `f8e4m3` so that rejection sites can
    /// describe it, and a bound must not admit a dtype §1.1 does not.
    pub fn admits(self, prim: Prim) -> bool {
        match self.precision_family() {
            TypeVarRestriction::ActiveFloat => prim.is_float(),
            TypeVarRestriction::ActiveInt => prim.is_integer(),
            TypeVarRestriction::ActiveNumeric => prim.is_float() || prim.is_integer(),
            _ => unreachable!("precision_family returns a primitive dtype family"),
        }
    }

    pub(crate) fn is_value_constraint(self) -> bool {
        matches!(self, Self::FloatValue | Self::IntValue | Self::NumericValue)
    }

    pub(crate) fn precision_family(self) -> Self {
        match self {
            Self::FloatValue => Self::ActiveFloat,
            Self::IntValue => Self::ActiveInt,
            Self::NumericValue => Self::ActiveNumeric,
            family => family,
        }
    }

    pub(crate) fn for_value(self) -> Self {
        match self.precision_family() {
            Self::ActiveFloat => Self::FloatValue,
            Self::ActiveInt => Self::IntValue,
            Self::ActiveNumeric => Self::NumericValue,
            _ => unreachable!("precision_family returns a primitive dtype family"),
        }
    }

    /// The family both bounds admit, or `None` when they share no dtype.
    ///
    /// [04-DTYPE-2]: unifying two bounded variables yields the intersection
    /// of their families. `Numeric` is the join of the other two, so every
    /// non-empty intersection is itself one of the three families and the
    /// only empty case is `Float` against `Int`.
    pub fn intersect(self, other: TypeVarRestriction) -> Option<TypeVarRestriction> {
        use TypeVarRestriction::{ActiveFloat, ActiveInt, ActiveNumeric};
        let family = match (self.precision_family(), other.precision_family()) {
            (ActiveFloat, ActiveFloat) => Some(ActiveFloat),
            (ActiveInt, ActiveInt) => Some(ActiveInt),
            (ActiveNumeric, ActiveNumeric) => Some(ActiveNumeric),
            (ActiveNumeric, ActiveFloat) | (ActiveFloat, ActiveNumeric) => Some(ActiveFloat),
            (ActiveNumeric, ActiveInt) | (ActiveInt, ActiveNumeric) => Some(ActiveInt),
            (ActiveFloat, ActiveInt) | (ActiveInt, ActiveFloat) => None,
            _ => unreachable!("precision_family returns a primitive dtype family"),
        }?;
        Some(
            if self.is_value_constraint() && other.is_value_constraint() {
                family.for_value()
            } else {
                family
            },
        )
    }
}

/// A unique identifier for a dimension variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DimVar(pub u32);

/// A unique identifier for a *rank* variable — a `Dim::Rank` stands for an
/// entire shape vector (Tier-2 rank polymorphism, `spec/design/rank_polymorphism.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RankVar(pub u32);

/// Numeric precision types.
///
/// The active numeric primitive set is pinned by `spec/04-type-system.md` §1.1:
/// `f32`, `f64`, `bf16`, `f16`, `i8`, `i16`, `i32`, `i64`, plus `bool`
/// and `string`. The `f8e4m3` variant is reserved per §1.1.1 but is **not
/// active** in this dtype build-out cycle: parse paths still produce
/// `Prim::F8e4m3` so producers can be diagnosed precisely, but every
/// admissibility predicate below excludes it. See [`Prim::is_admissible_active`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
}

// Canonical identity order for order-free collection indices. This is byte
// order over the public dtype spelling, not a semantic numeric ordinal.
impl PartialOrd for Prim {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Prim {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name().cmp(other.name())
    }
}

impl Prim {
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
            Prim::F8e4m3 | Prim::String => Err(RuntimeDTypeMappingError { prim: self }),
        }
    }

    /// True for the active float dtypes per `spec/04-type-system.md` §1.1.
    /// `f8e4m3` is deferred (§1.1.1) and is NOT a float for any active
    /// classification purpose.
    pub fn is_float(&self) -> bool {
        matches!(self, Prim::F32 | Prim::F64 | Prim::F16 | Prim::Bf16)
    }

    /// True for any numeric precision in the active set, including `f8e4m3`
    /// (so deferred-dtype rejection sites can still treat it as numeric for
    /// surface diagnostics). `Bool` and `String` are not numeric.
    pub fn is_numeric(&self) -> bool {
        !matches!(self, Prim::Bool | Prim::String)
    }

    /// True for all signed integer dtypes in the active set per §1.1.
    pub fn is_integer(&self) -> bool {
        matches!(self, Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64)
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

    /// Whether this precision is valid as the element type of a tensor.
    /// Per spec §1.1 the active tensor element set is f32, f64, bf16, f16,
    /// i8, i16, i32, i64, and bool. The deferred `f8e4m3` (§1.1.1)
    /// is rejected.
    ///
    /// Note: backend support for the reduced floats (`f16`, `bf16`) is
    /// staged separately in WS-A1/A2/A3; the type checker admits them here
    /// because the spec lists them as active dtypes. Backends that cannot
    /// yet emit them are expected to produce their own targeted diagnostic
    /// rather than let them slip through silently.
    pub fn is_valid_tensor_precision(&self) -> bool {
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
    /// Mirrors `is_valid_tensor_precision` in this cycle: the host scalar
    /// lane carries the same active dtype set per spec §1.1, and the
    /// deferred `f8e4m3` (§1.1.1) is rejected here too.
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

    /// Resolve the spec/04-type-system.md §5.7.1 default reduce-sum
    /// **accumulator** precision for this operand precision. Mirrors
    /// `chelis_ir::dag::RiscOp::default_reduce_sum_accumulator` so the
    /// type checker can resolve the same rule without a backward
    /// dependency from `chelis-types` on `chelis-ir`.
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
    pub fn default_reduce_sum_accumulator(self) -> Result<Prim, String> {
        Ok(match self {
            Prim::Bf16 | Prim::F16 => Prim::F32,
            Prim::F32 => Prim::F32,
            Prim::F64 => Prim::F64,
            Prim::Int8 | Prim::Int16 => Prim::Int32,
            Prim::Int32 => Prim::Int32,
            Prim::Int64 => Prim::Int64,
            Prim::Bool => {
                return Err(
                    "reduce_sum is not defined on bool tensors; cast to i32 first".to_string(),
                );
            }
            Prim::F8e4m3 => {
                return Err("reduce_sum: operand dtype `f8e4m3` is deferred per \
                     spec/04-type-system.md §1.1.1 and is not part of the \
                     active numeric primitive set"
                    .to_string());
            }
            Prim::String => {
                return Err("reduce_sum is not defined for string operands".to_string());
            }
        })
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
    pub fn default_reduce_sum_result_precision(self) -> Result<Prim, String> {
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

impl std::error::Error for RuntimeDTypeMappingError {}

/// A tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// Concrete named dimension (e.g., batch, hidden).
    Name(String),
    /// Polymorphic dimension variable (e.g., a, b).
    Var(DimVar),
    /// Fixed numeric size (e.g., 512).
    Lit(i64),
    /// Wildcard — unknown/dynamic dimension.
    Wildcard,
    /// Rank variable — a *name-preserving spread* standing for a run of dims
    /// (rank polymorphism). Tier-2 used it only as the sole element of a shape
    /// (`tensor[..r, p]`); Tier-3 allows it interleaved with concrete anchors
    /// (`tensor[..pre, seq, ..post, p]`). Structural invariant: a given
    /// `RankVar` appears at most once per tensor dim list, and two spreads are
    /// never *split* against a ground while both unbound (an undetermined
    /// boundary — rejected at unification). A spread binds to the actual named
    /// dims it covers, so names are preserved, not erased. Eliminated by
    /// monomorphization; no `Dim::Rank` reaches a backend.
    Rank(RankVar),
}

/// Tensor element precision slot.
///
/// Per `spec/04-type-system.md` §5.8 and `spec/02-surf-syntax.md`, the
/// precision slot of a tensor type may be either a concrete primitive
/// (e.g. `f32`, `i32`) or a sig-bound type variable (precision
/// polymorphism, WS-A5). After monomorphization every reachable
/// tensor must carry `TensorPrec::Concrete(_)`; backends assert this
/// invariant at lowering time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorPrec {
    /// A concrete numeric primitive; the only shape backends accept.
    Concrete(Prim),
    /// A sig-quantified type variable. Resolved by unification, then
    /// substituted to `Concrete(_)` by `Subst::apply` once bound.
    Var(TypeVar),
}

impl TensorPrec {
    /// Returns the concrete primitive if this slot is already resolved,
    /// or `None` if it is still a type variable. Backends and IR
    /// builders that must have a concrete dtype call this and treat
    /// `None` as a monomorphization bug.
    pub fn as_concrete(&self) -> Option<Prim> {
        match self {
            TensorPrec::Concrete(p) => Some(*p),
            TensorPrec::Var(_) => None,
        }
    }

    /// Convenience: short rendering of the slot, suitable for diagnostics.
    /// Concrete precisions render as their canonical name; vars render
    /// as `?N` matching `Type::Var` formatting.
    pub fn render(&self) -> String {
        match self {
            TensorPrec::Concrete(p) => p.name().to_string(),
            TensorPrec::Var(v) => format!("?{}", v.0),
        }
    }

    /// Diagnostic-only alias for [`TensorPrec::render`]; matches the
    /// `Prim::name()` ergonomic for call sites that previously took a
    /// bare `Prim`. Returns an owned `String` because var precisions
    /// have no `'static` representation.
    pub fn name(&self) -> String {
        self.render()
    }

    /// True iff this precision is concretely a float per
    /// [`Prim::is_float`]. A polymorphic precision var returns `false`:
    /// a not-yet-resolved precision is not yet known to be float, so
    /// any "this op needs a float" check should not silently accept a
    /// `Var` slot.
    pub fn is_float(&self) -> bool {
        matches!(self, TensorPrec::Concrete(p) if p.is_float())
    }

    /// True iff this precision is concretely an integer per
    /// [`Prim::is_integer`]. `Var` returns `false` for the same reason
    /// as [`TensorPrec::is_float`].
    pub fn is_integer(&self) -> bool {
        matches!(self, TensorPrec::Concrete(p) if p.is_integer())
    }

    /// True iff this precision is concretely numeric per
    /// [`Prim::is_numeric`]. `Var` returns `false`.
    pub fn is_numeric(&self) -> bool {
        matches!(self, TensorPrec::Concrete(p) if p.is_numeric())
    }

    /// Resolve the spec/04-type-system.md §5.7.1 reduce_sum result
    /// precision. Forwards to [`Prim::default_reduce_sum_result_precision`]
    /// for `Concrete`. For `Var`, returns an error: a polymorphic
    /// precision must be resolved (or rejected) before the reduce-sum
    /// resolution rule applies.
    pub fn default_reduce_sum_result_precision(self) -> Result<Prim, String> {
        match self {
            TensorPrec::Concrete(p) => p.default_reduce_sum_result_precision(),
            TensorPrec::Var(v) => Err(format!(
                "reduce_sum: operand precision is still polymorphic (?{}); a sig-bound \
                 type variable in the precision slot must be resolved by unification \
                 before reduce_sum's accumulator/result rule can be applied",
                v.0
            )),
        }
    }
}

impl From<Prim> for TensorPrec {
    fn from(p: Prim) -> Self {
        TensorPrec::Concrete(p)
    }
}

/// One argument of a nominal type application.
///
/// Keeping dimensions out of [`Type`] makes an extent impossible to consume
/// as an ordinary type while preserving the declared argument order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NominalArg {
    Type(Type),
    Dimension(Dim),
}

/// Checker-owned kind of a nominal declaration parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NominalParamKind {
    Type,
    Dimension,
}

impl NominalArg {
    pub fn as_type(&self) -> Option<&Type> {
        match self {
            Self::Type(ty) => Some(ty),
            Self::Dimension(_) => None,
        }
    }

    pub fn as_dimension(&self) -> Option<&Dim> {
        match self {
            Self::Type(_) => None,
            Self::Dimension(dim) => Some(dim),
        }
    }
}

impl From<Type> for NominalArg {
    fn from(value: Type) -> Self {
        Self::Type(value)
    }
}

impl fmt::Display for NominalArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(ty) => ty.fmt(f),
            Self::Dimension(dim) => dim.fmt(f),
        }
    }
}

/// Chelis type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    /// Primitive type (f32, i32, bool, etc.).
    Prim(Prim),
    /// Function type: args → return.
    Fn(Vec<Type>, Box<Type>),
    /// Read-only non-owning borrow of a value.
    Ref(Box<Type>),
    /// Tensor type: dimensions + precision slot. Precision is
    /// `TensorPrec::Concrete(_)` for fully-monomorphized tensors and
    /// `TensorPrec::Var(_)` for sig-quantified precision polymorphism
    /// per `spec/04-type-system.md` §5.8 (WS-A5).
    Tensor(Vec<Dim>, TensorPrec),
    /// Algebraic data type whose arguments are all ordinary types.
    Adt(String, Vec<Type>),
    /// Algebraic data type with at least one dimension-kinded argument.
    /// Keeping this representation distinct makes a dimension impossible to
    /// consume through an ordinary `Type::Adt` argument path.
    KindedAdt(String, Vec<NominalArg>),
    /// Type variable (for inference).
    Var(TypeVar),
    /// Tuple type.
    Tuple(Vec<Type>),
    /// Unit type.
    Unit,
    /// Error sentinel -- used for partial inference past errors. Carries a
    /// zero-sized [`ErrorWitness`](crate::errors::ErrorWitness) that can only
    /// be minted by `crate::errors::report` (which pushes a diagnostic) or
    /// `crate::errors::propagate` (cascade from an existing witness), so a
    /// silent `Type::Error` is unconstructible outside the diagnostics module
    /// (spec/design/checker_totality.md §C3, chelis#731 Phase 2).
    Error(ErrorWitness),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Effect {
    Random,
    Accum,
    Io,
    /// Chelis-native testing effect. Pinned at the root, no handler.
    /// Any function that (directly or transitively) calls a `test_assert_*`
    /// builtin acquires this effect, preventing assertions from silently
    /// leaking into `! {}` production code.
    Test,
    Resource(String),
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => f.write_str("Random"),
            Self::Accum => f.write_str("Accum"),
            Self::Io => f.write_str("IO"),
            Self::Test => f.write_str("Test"),
            Self::Resource(device) => write!(f, "Resource(\"{device}\")"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectSet {
    effects: BTreeSet<Effect>,
}

impl EffectSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    pub fn remove(&mut self, effect: &Effect) {
        self.effects.remove(effect);
    }

    pub fn contains(&self, effect: &Effect) -> bool {
        self.effects.contains(effect)
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    pub fn extend(&mut self, other: &EffectSet) {
        self.effects.extend(other.effects.iter().cloned());
    }

    pub fn iter(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }
}

impl FromIterator<Effect> for EffectSet {
    fn from_iter<T: IntoIterator<Item = Effect>>(iter: T) -> Self {
        let mut set = Self::new();
        for effect in iter {
            set.insert(effect);
        }
        set
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rendered: Vec<String> = self.effects.iter().map(ToString::to_string).collect();
        write!(f, "{{{}}}", rendered.join(", "))
    }
}

/// The checked operand/result relation of a first-class collection operation.
///
/// Builtin schemes own these contracts. Aliasing, higher-order passage,
/// import, and serialization retain them; applying the value consumes and
/// decides them. A newly authored wrapper does not create one from its body
/// because [04-INF-9] requires an explicit sufficient parameter contract.
///
/// Each variant holds the route's operands and its result in FIXED fields.
/// A rule tag plus an operand vector would describe `(List, List)` membership
/// and lose the equations the rule also imposes -- that `concat`'s two lists
/// share an element type and that its result is a list at that same type --
/// so a mixed-element call, or one whose result was annotated to something
/// the rule never produces, would satisfy the carried constraint.
///
/// The carried types are ordinary types, so [`crate::env::Env::instantiate`]'s
/// renaming substitution rewrites any dimension or rank variable they mention
/// exactly as it rewrites the scheme body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CollectionConstraint {
    /// `len(operand) -> i64`: the operand is a `List` or a `Dict`.
    Len { operand: Type, result: Type },
    /// `index(list, index) -> result`: `list` is a `List[e]`, `index` is
    /// exactly `i64`, and `result` is `e`.
    Index {
        list: Type,
        index: Type,
        result: Type,
    },
    /// `append(list, value) -> result`: `list` is a `List[e]`, `value` is an
    /// `e`, and `result` is `List[e]`.
    Append {
        list: Type,
        value: Type,
        result: Type,
    },
    /// `concat(lhs, rhs) -> result`, both spec/04 §4.5.4 overloads: two
    /// `List[e]` giving `List[e]`, or a `List[tensor[..]]` and an `i32`
    /// axis giving a tensor.
    Concat { lhs: Type, rhs: Type, result: Type },
}

impl CollectionConstraint {
    /// The builtin this constraint belongs to, for diagnostics.
    pub fn builtin(&self) -> &'static str {
        match self {
            Self::Len { .. } => "len",
            Self::Index { .. } => "index",
            Self::Append { .. } => "append",
            Self::Concat { .. } => "concat",
        }
    }

    /// The operand positions, in the order the rule reads them. The result is
    /// deliberately absent: an unresolved RESULT is what the rule computes,
    /// not what it waits on.
    pub fn operands(&self) -> Vec<&Type> {
        match self {
            Self::Len { operand, .. } => vec![operand],
            Self::Index { list, index, .. } => vec![list, index],
            Self::Append { list, value, .. } => vec![list, value],
            Self::Concat { lhs, rhs, .. } => vec![lhs, rhs],
        }
    }

    /// Every type this constraint carries, operands and result alike. Used to
    /// decide which variables a pending constraint keeps monomorphic.
    pub fn carried_types(&self) -> Vec<&Type> {
        match self {
            Self::Len { operand, result } => vec![operand, result],
            Self::Index {
                list,
                index,
                result,
            } => vec![list, index, result],
            Self::Append {
                list,
                value,
                result,
            } => vec![list, value, result],
            Self::Concat { lhs, rhs, result } => vec![lhs, rhs, result],
        }
    }

    /// The result the suspended call already handed its consumer.
    pub fn result(&self) -> &Type {
        match self {
            Self::Len { result, .. }
            | Self::Index { result, .. }
            | Self::Append { result, .. }
            | Self::Concat { result, .. } => result,
        }
    }

    /// Rewrite every carried type with `f`. Instantiation passes the
    /// quantifier renaming; discharge passes the current substitution.
    pub fn map_types(&self, f: impl Fn(&Type) -> Type) -> Self {
        match self {
            Self::Len { operand, result } => Self::Len {
                operand: f(operand),
                result: f(result),
            },
            Self::Index {
                list,
                index,
                result,
            } => Self::Index {
                list: f(list),
                index: f(index),
                result: f(result),
            },
            Self::Append {
                list,
                value,
                result,
            } => Self::Append {
                list: f(list),
                value: f(value),
                result: f(result),
            },
            Self::Concat { lhs, rhs, result } => Self::Concat {
                lhs: f(lhs),
                rhs: f(rhs),
                result: f(result),
            },
        }
    }
}

/// A polymorphic type scheme: ∀ tvars, dvars. body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scheme {
    pub tvars: Vec<TypeVar>,
    /// Domain restrictions for quantified type variables. Entries are kept
    /// in quantifier order for deterministic serialization.
    #[serde(default)]
    pub tvar_restrictions: Vec<(TypeVar, TypeVarRestriction)>,
    pub dvars: Vec<DimVar>,
    /// Quantified rank variables (Tier-2 rank polymorphism). Usually empty.
    #[serde(default)]
    pub rvars: Vec<RankVar>,
    /// Checked collection-operation relations transported by this function
    /// value and renamed at each instantiation. Usually empty.
    ///
    /// Deliberately NOT `#[serde(default)]`. A default would let a scheme
    /// persisted before this field existed decode as unconstrained and erase a
    /// checked function value's contract on the reuse path. A payload without
    /// this field must fail to decode; the TypeEnv format version makes that
    /// an obsolete-snapshot error.
    pub constraints: Vec<CollectionConstraint>,
    pub body: Type,
}

impl Scheme {
    /// A monomorphic scheme (no quantified variables).
    pub fn mono(ty: Type) -> Scheme {
        Scheme {
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            constraints: vec![],
            body: ty,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Prim(p) => write!(f, "{}", p.name()),
            Type::Fn(args, ret) => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "({}) -> {}", arg_strs.join(", "), ret)
            }
            Type::Ref(inner) => write!(f, "&{inner}"),
            Type::Tensor(dims, prec) => {
                let dim_strs: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                write!(f, "tensor[{}, {}]", dim_strs.join(", "), prec.render())
            }
            Type::Adt(name, args) if args.is_empty() => write!(f, "{name}"),
            Type::Adt(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name} {}", arg_strs.join(" "))
            }
            Type::KindedAdt(name, args) if args.is_empty() => write!(f, "{name}"),
            Type::KindedAdt(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(ToString::to_string).collect();
                write!(f, "{name} {}", arg_strs.join(" "))
            }
            Type::Var(v) => write!(f, "?{}", v.0),
            Type::Tuple(ts) => {
                let strs: Vec<String> = ts.iter().map(|t| t.to_string()).collect();
                write!(f, "({})", strs.join(", "))
            }
            Type::Unit => write!(f, "()"),
            Type::Error(_) => write!(f, "<error>"),
        }
    }
}

impl fmt::Display for Dim {
    /// User-facing rendering of a tensor dimension, matching the CLI surface
    /// (`format_cli_dim`): a named dim by its name, a dim variable as `d<id>`, a
    /// literal by its value, the wildcard as `*`, and a rank spread as `..r<id>`.
    /// Used by `Type`'s `Display` so diagnostics read `tensor[..r0, seq, ..r1, f32]`
    /// instead of the internal `Debug` form `Rank(RankVar(0)), Name("seq")`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Name(name) => write!(f, "{name}"),
            Dim::Var(v) => write!(f, "d{}", v.0),
            Dim::Lit(value) => write!(f, "{value}"),
            Dim::Wildcard => write!(f, "*"),
            Dim::Rank(r) => write!(f, "..r{}", r.0),
        }
    }
}

#[cfg(test)]
mod prim_classification_tests {
    //! Pin every `Prim` variant against the active-set classification rules
    //! in `spec/04-type-system.md` §1.1 and §1.1.1. Acceptance test (a) of
    //! WS-A0; protects against an active dtype silently sliding back into
    //! the deferred set or vice versa.
    use super::*;

    /// Closed enumeration of every `Prim` variant. If a variant is added
    /// or removed and this list is not updated, every per-variant test
    /// below fails — that is the intended invariant lock.
    const ALL_PRIMS: &[Prim] = &[
        Prim::F32,
        Prim::F64,
        Prim::F16,
        Prim::Bf16,
        Prim::F8e4m3,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
        Prim::String,
    ];

    #[test]
    fn is_float_matches_active_float_set() {
        for prim in ALL_PRIMS {
            let expected = matches!(prim, Prim::F32 | Prim::F64 | Prim::F16 | Prim::Bf16);
            assert_eq!(
                prim.is_float(),
                expected,
                "is_float({prim:?}) disagrees with §1.1 active float set; \
                 f8e4m3 is deferred per §1.1.1 and must NOT be a float here"
            );
        }
    }

    #[test]
    fn is_integer_matches_active_integer_set() {
        for prim in ALL_PRIMS {
            let expected = matches!(prim, Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64);
            assert_eq!(
                prim.is_integer(),
                expected,
                "is_integer({prim:?}) disagrees with §1.1 active integer set"
            );
        }
    }

    #[test]
    fn is_numeric_excludes_bool_and_string() {
        for prim in ALL_PRIMS {
            let expected = !matches!(prim, Prim::Bool | Prim::String);
            assert_eq!(
                prim.is_numeric(),
                expected,
                "is_numeric({prim:?}) disagrees with §1.1 numeric set"
            );
        }
    }

    #[test]
    fn is_valid_tensor_precision_matches_active_set() {
        // Active dtype set per §1.1, including f16/bf16 (active per WS-0
        // spec lock cc47e6d). Excludes the deferred f8e4m3 (§1.1.1) and
        // String (no tensor element representation).
        for prim in ALL_PRIMS {
            let expected = matches!(
                prim,
                Prim::F32
                    | Prim::F64
                    | Prim::Bf16
                    | Prim::F16
                    | Prim::Bool
                    | Prim::Int8
                    | Prim::Int16
                    | Prim::Int32
                    | Prim::Int64
            );
            assert_eq!(
                prim.is_valid_tensor_precision(),
                expected,
                "is_valid_tensor_precision({prim:?}) disagrees with §1.1 \
                 active tensor element set"
            );
        }
    }

    #[test]
    fn is_valid_scalar_cast_target_matches_active_set() {
        // Per §1.1: scalar cast targets mirror the tensor element set in
        // this cycle (no host-only widths beyond the active dtype set).
        // Deferred f8e4m3 is rejected per §1.1.1.
        for prim in ALL_PRIMS {
            let expected = matches!(
                prim,
                Prim::F32
                    | Prim::F64
                    | Prim::Bf16
                    | Prim::F16
                    | Prim::Bool
                    | Prim::Int8
                    | Prim::Int16
                    | Prim::Int32
                    | Prim::Int64
            );
            assert_eq!(
                prim.is_valid_scalar_cast_target(),
                expected,
                "is_valid_scalar_cast_target({prim:?}) disagrees with §1.1 \
                 active scalar cast set"
            );
        }
    }

    #[test]
    fn is_admissible_active_excludes_only_f8e4m3() {
        for prim in ALL_PRIMS {
            let expected = !matches!(prim, Prim::F8e4m3);
            assert_eq!(
                prim.is_admissible_active(),
                expected,
                "is_admissible_active({prim:?}) disagrees with §1.1.1: only \
                 f8e4m3 is deferred, every other variant is admitted"
            );
        }
    }

    #[test]
    fn parse_name_round_trips_through_name() {
        for prim in ALL_PRIMS {
            let name = prim.name();
            assert_eq!(
                Prim::parse_name(name),
                Some(*prim),
                "parse_name({name:?}) must round-trip {prim:?}"
            );
        }
    }

    #[test]
    fn int16_is_an_active_dtype() {
        // Pin the new variant explicitly so a refactor cannot silently
        // remove it without removing this assertion.
        assert!(Prim::Int16.is_admissible_active());
        assert!(Prim::Int16.is_integer());
        assert!(Prim::Int16.is_numeric());
        assert!(Prim::Int16.is_valid_tensor_precision());
        assert!(Prim::Int16.is_valid_scalar_cast_target());
        assert!(!Prim::Int16.is_float());
        assert_eq!(Prim::Int16.name(), "i16");
        assert_eq!(Prim::parse_name("i16"), Some(Prim::Int16));
    }

    #[test]
    fn f8e4m3_is_deferred_not_active() {
        // §1.1.1 invariants: f8e4m3 still parses (so producers can be
        // diagnosed) but no admissibility predicate accepts it.
        assert_eq!(Prim::parse_name("f8e4m3"), Some(Prim::F8e4m3));
        assert!(!Prim::F8e4m3.is_admissible_active());
        assert!(!Prim::F8e4m3.is_float());
        assert!(!Prim::F8e4m3.is_integer());
        assert!(!Prim::F8e4m3.is_valid_tensor_precision());
        assert!(!Prim::F8e4m3.is_valid_scalar_cast_target());
    }
}

/// Counter for generating fresh type and dimension variables.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VarGen {
    next_tvar: u32,
    next_dvar: u32,
    #[serde(default)]
    next_rvar: u32,
}

/// The next fresh identifier in each solver-variable class.
///
/// Level transitions record these watermarks instead of attaching a level to
/// every variable. A variable's mint level is therefore the final transition
/// whose corresponding watermark is no greater than its identifier.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct VarWatermarks {
    pub(crate) next_tvar: u32,
    pub(crate) next_dvar: u32,
    pub(crate) next_rvar: u32,
}

impl VarGen {
    pub(crate) fn watermarks(&self) -> VarWatermarks {
        VarWatermarks {
            next_tvar: self.next_tvar,
            next_dvar: self.next_dvar,
            next_rvar: self.next_rvar,
        }
    }

    pub fn fresh_tvar(&mut self) -> TypeVar {
        let v = TypeVar(self.next_tvar);
        self.next_tvar += 1;
        v
    }

    pub fn fresh_dvar(&mut self) -> DimVar {
        let v = DimVar(self.next_dvar);
        self.next_dvar += 1;
        v
    }

    /// Fresh rank variable (Tier-2 rank polymorphism).
    pub fn fresh_rvar(&mut self) -> RankVar {
        let v = RankVar(self.next_rvar);
        self.next_rvar += 1;
        v
    }

    pub fn fresh_type(&mut self) -> Type {
        Type::Var(self.fresh_tvar())
    }

    pub fn fresh_dim(&mut self) -> Dim {
        Dim::Var(self.fresh_dvar())
    }
}

// ─── Root boundary types (issue #912) ────────────────────────────────────────

/// Compilation/evaluation target. Determines which backend's capability set
/// is used for realizability inference and manifest computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Target {
    /// The Rust tensor evaluator (stores f64, supports all prims).
    Eval,
    /// The C backend (f32/bool/bf16/f16/i32/i64; rejects f64).
    C,
    /// The HIP/ROCm backend.
    Hip,
    /// The Metal backend.
    Metal,
}

/// Lane assignment for a top-level root. Determined by realizability inference
/// from per-builtin declarations, per-tag declarations, and def-level
/// precision checks against backend capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lane {
    /// Realized through the tensor DAG path (lowering → codegen).
    Tensor,
    /// Realized through the host runtime path (interpreter or host C emission).
    Host,
}

#[cfg(test)]
mod dtype_family_bound_tests {
    //! Lock `spec/04-type-system.md` §5.9's family membership and
    //! [04-DTYPE-2]'s intersection rule against the active dtype set.

    use super::*;

    const EVERY_PRIM: [Prim; 11] = [
        Prim::F32,
        Prim::F64,
        Prim::F16,
        Prim::Bf16,
        Prim::F8e4m3,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
        Prim::String,
    ];

    fn admitted(restriction: TypeVarRestriction) -> Vec<Prim> {
        EVERY_PRIM
            .into_iter()
            .filter(|prim| restriction.admits(*prim))
            .collect()
    }

    #[test]
    fn float_family_is_exactly_the_four_active_floats() {
        assert_eq!(
            admitted(TypeVarRestriction::ActiveFloat),
            vec![Prim::F32, Prim::F64, Prim::F16, Prim::Bf16]
        );
    }

    #[test]
    fn int_family_is_exactly_the_four_active_signed_integers() {
        assert_eq!(
            admitted(TypeVarRestriction::ActiveInt),
            vec![Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64]
        );
    }

    #[test]
    fn numeric_family_is_the_union_of_float_and_int() {
        let mut union = admitted(TypeVarRestriction::ActiveFloat);
        union.extend(admitted(TypeVarRestriction::ActiveInt));
        let mut numeric = admitted(TypeVarRestriction::ActiveNumeric);
        numeric.sort_by_key(|prim| format!("{prim:?}"));
        union.sort_by_key(|prim| format!("{prim:?}"));
        assert_eq!(numeric, union);
    }

    #[test]
    fn no_family_admits_bool_string_or_a_reserved_dtype() {
        // §1.1.1's deferred `f8e4m3` is numeric for diagnostic purposes but
        // is not an active dtype, so no bound may admit it.
        for restriction in [
            TypeVarRestriction::ActiveFloat,
            TypeVarRestriction::ActiveInt,
            TypeVarRestriction::ActiveNumeric,
        ] {
            for prim in [Prim::Bool, Prim::String, Prim::F8e4m3] {
                assert!(
                    !restriction.admits(prim),
                    "{} must not admit {prim:?}",
                    restriction.family_name()
                );
            }
        }
    }

    #[test]
    fn intersection_narrows_to_the_common_family() {
        use TypeVarRestriction::{ActiveFloat, ActiveInt, ActiveNumeric};
        assert_eq!(ActiveFloat.intersect(ActiveFloat), Some(ActiveFloat));
        assert_eq!(ActiveInt.intersect(ActiveInt), Some(ActiveInt));
        assert_eq!(ActiveNumeric.intersect(ActiveNumeric), Some(ActiveNumeric));
        assert_eq!(ActiveNumeric.intersect(ActiveFloat), Some(ActiveFloat));
        assert_eq!(ActiveFloat.intersect(ActiveNumeric), Some(ActiveFloat));
        assert_eq!(ActiveNumeric.intersect(ActiveInt), Some(ActiveInt));
        assert_eq!(ActiveInt.intersect(ActiveNumeric), Some(ActiveInt));
    }

    #[test]
    fn float_and_int_have_an_empty_intersection() {
        assert_eq!(
            TypeVarRestriction::ActiveFloat.intersect(TypeVarRestriction::ActiveInt),
            None
        );
        assert_eq!(
            TypeVarRestriction::ActiveInt.intersect(TypeVarRestriction::ActiveFloat),
            None
        );
    }

    #[test]
    fn an_intersection_admits_exactly_the_shared_dtypes() {
        for left in [
            TypeVarRestriction::ActiveFloat,
            TypeVarRestriction::ActiveInt,
            TypeVarRestriction::ActiveNumeric,
        ] {
            for right in [
                TypeVarRestriction::ActiveFloat,
                TypeVarRestriction::ActiveInt,
                TypeVarRestriction::ActiveNumeric,
            ] {
                let shared: Vec<Prim> = EVERY_PRIM
                    .into_iter()
                    .filter(|prim| left.admits(*prim) && right.admits(*prim))
                    .collect();
                match left.intersect(right) {
                    Some(merged) => assert_eq!(admitted(merged), shared),
                    None => assert!(
                        shared.is_empty(),
                        "{}/{} share {shared:?} but intersect to nothing",
                        left.family_name(),
                        right.family_name()
                    ),
                }
            }
        }
    }

    #[test]
    fn family_names_match_the_surf_spelling() {
        assert_eq!(TypeVarRestriction::ActiveFloat.family_name(), "Float");
        assert_eq!(TypeVarRestriction::ActiveInt.family_name(), "Int");
        assert_eq!(TypeVarRestriction::ActiveNumeric.family_name(), "Numeric");
    }
}
