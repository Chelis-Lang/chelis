//! Internal type representation for the Chelis type checker.
//!
//! These are the checker's working types — NOT the Deep AST nodes.
//! They mirror the Deep t-* tags but are Rust-native for efficient unification.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A unique identifier for a type variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVar(pub u32);

/// A unique identifier for a dimension variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DimVar(pub u32);

/// Numeric precision types.
///
/// The active numeric primitive set is pinned by `spec/04-type-system.md` §1.1:
/// `f32`, `f64`, `bf16`, `f16`, `int8`, `int16`, `int32`, `int64`, plus `bool`
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

impl Prim {
    /// Parse a primitive type from its canonical name.
    pub fn parse_name(s: &str) -> Option<Prim> {
        match s {
            "f32" => Some(Prim::F32),
            "f64" => Some(Prim::F64),
            "f16" => Some(Prim::F16),
            "bf16" => Some(Prim::Bf16),
            "f8e4m3" => Some(Prim::F8e4m3),
            "int8" => Some(Prim::Int8),
            "int16" => Some(Prim::Int16),
            "int32" => Some(Prim::Int32),
            "int64" => Some(Prim::Int64),
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
            Prim::Int8 => "int8",
            Prim::Int16 => "int16",
            Prim::Int32 => "int32",
            Prim::Int64 => "int64",
            Prim::Bool => "bool",
            Prim::String => "string",
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
    /// int8, int16, int32, int64, and bool. The deferred `f8e4m3` (§1.1.1)
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
    /// - int8 / int16 → int32
    /// - int32       → int32 (operand-matching)
    /// - int64       → int64 (operand-matching)
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
                    "reduce_sum is not defined on bool tensors; cast to int32 first".to_string(),
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
    ///                 inside the op and downcast on output)
    /// - f32         → f32
    /// - f64         → f64
    /// - int8 / int16 → int32 (accumulator precision)
    /// - int32       → int32
    /// - int64       → int64
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
}

/// Chelis type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    /// Primitive type (f32, int32, bool, etc.).
    Prim(Prim),
    /// Function type: args → return.
    Fn(Vec<Type>, Box<Type>),
    /// Read-only non-owning borrow of a value.
    Ref(Box<Type>),
    /// Tensor type: dimensions + precision.
    Tensor(Vec<Dim>, Prim),
    /// Algebraic data type: name + type arguments.
    Adt(String, Vec<Type>),
    /// Type variable (for inference).
    Var(TypeVar),
    /// Tuple type.
    Tuple(Vec<Type>),
    /// Unit type.
    Unit,
    /// Error sentinel — used for partial inference past errors.
    Error,
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

/// A polymorphic type scheme: ∀ tvars, dvars. body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scheme {
    pub tvars: Vec<TypeVar>,
    pub dvars: Vec<DimVar>,
    pub body: Type,
}

impl Scheme {
    /// A monomorphic scheme (no quantified variables).
    pub fn mono(ty: Type) -> Scheme {
        Scheme {
            tvars: vec![],
            dvars: vec![],
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
                let dim_strs: Vec<String> = dims.iter().map(|d| format!("{d:?}")).collect();
                write!(f, "tensor[{}, {}]", dim_strs.join(", "), prec.name())
            }
            Type::Adt(name, args) if args.is_empty() => write!(f, "{name}"),
            Type::Adt(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name} {}", arg_strs.join(" "))
            }
            Type::Var(v) => write!(f, "?{}", v.0),
            Type::Tuple(ts) => {
                let strs: Vec<String> = ts.iter().map(|t| t.to_string()).collect();
                write!(f, "({})", strs.join(", "))
            }
            Type::Unit => write!(f, "unit"),
            Type::Error => write!(f, "<error>"),
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
        assert_eq!(Prim::Int16.name(), "int16");
        assert_eq!(Prim::parse_name("int16"), Some(Prim::Int16));
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
}

impl VarGen {
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

    pub fn fresh_type(&mut self) -> Type {
        Type::Var(self.fresh_tvar())
    }

    pub fn fresh_dim(&mut self) -> Dim {
        Dim::Var(self.fresh_dvar())
    }
}
