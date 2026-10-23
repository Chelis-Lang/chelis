//! Internal type representation for the Chelis type checker.
//!
//!
//! These are the checker's working types — NOT the Deep AST nodes.
//!
//! They mirror the Deep t-* tags but are Rust-native for efficient unification.
//!

use alloc::collections::BTreeSet;
use alloc::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::fmt;

use chelis_numeric_core::prim::Prim;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A unique identifier for a type variable.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TypeVar(pub u32);

/// A semantic domain attached to a quantified type variable.
///
///
/// Unlike a callee-name check, a restriction is part of the polymorphic.
///
/// scheme itself. Instantiation installs it on the fresh inference variable,.
///
/// ordinary unification propagates it through aliases, and generalization.
///
/// re-quantifies it on wrappers and higher-order values.
///
/// `spec/04-type-system.md` §5.9's explicit dtype set, as a bitmask over the.
///
/// eight active numeric dtypes of §1.1 in declaration order.
///
///
/// A bitmask rather than a collection so [`TypeVarRestriction`] stays `Copy`,.
///
/// and so [04-DTYPE-2]'s intersection is a bitwise `and` that cannot.
///
/// disagree with itself.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PrimSet(u8);

impl PrimSet {
    /// The §1.1 declaration order the bits follow, and spec/03 §6.2's.
    ///
    /// canonical member order.
    ///
    pub const ORDER: [Prim; 8] = [
        Prim::F32,
        Prim::F64,
        Prim::Bf16,
        Prim::F16,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
    ];

    /// The empty set. [04-DTYPE-2] makes an empty bound a declaration error,.
    ///
    /// so this exists to be rejected, never to be installed.
    ///
    pub const EMPTY: PrimSet = PrimSet(0);

    fn bit(prim: Prim) -> Option<u8> {
        PrimSet::ORDER
            .iter()
            .position(|candidate| *candidate == prim)
            .map(|index| 1u8 << index)
    }

    /// Build a set from members, ignoring any dtype §1.1 admits into no.
    ///
    /// family. The caller rejects an empty or non-numeric member list.
    ///
    pub fn from_members(members: impl IntoIterator<Item = Prim>) -> PrimSet {
        let mut bits = 0u8;
        for prim in members {
            if let Some(bit) = PrimSet::bit(prim) {
                bits |= bit;
            }
        }
        PrimSet(bits)
    }

    /// Whether `prim` is a member.
    ///
    pub fn contains(self, prim: Prim) -> bool {
        PrimSet::bit(prim).is_some_and(|bit| self.0 & bit != 0)
    }

    /// Whether the set admits nothing.
    ///
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The members, in §1.1 declaration order.
    ///
    pub fn members(self) -> impl Iterator<Item = Prim> {
        PrimSet::ORDER
            .into_iter()
            .filter(move |prim| self.contains(*prim))
    }

    /// [04-DTYPE-2]'s intersection of two sets.
    ///
    pub fn intersect(self, other: PrimSet) -> PrimSet {
        PrimSet(self.0 & other.0)
    }

    /// The set a family admits, so a family can be intersected with a set.
    ///
    /// without duplicating §1.1 membership here.
    ///
    pub fn of_family(family: TypeVarRestriction) -> PrimSet {
        PrimSet::from_members(
            PrimSet::ORDER
                .into_iter()
                .filter(|prim| family.admits(*prim)),
        )
    }

    /// The §5.9 spelling, `{d1, d2}` in canonical order.
    ///
    pub fn spelling(self) -> String {
        format!(
            "{{{}}}",
            self.members()
                .map(|prim| prim.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TypeVarRestriction {
    /// `spec/04-type-system.md` §5.9 `Float`: the variable may instantiate.
    ///
    /// only at an active float primitive (`f16`, `bf16`, `f32`, `f64`).
    ///
    ActiveFloat,
    /// §5.9 `Int`: only at an active signed integer primitive (`i8`,.
    ///
    /// `i16`, `i32`, `i64`).
    ///
    ActiveInt,
    /// §5.9 `Numeric`: the union of [`TypeVarRestriction::ActiveFloat`] and.
    ///
    /// [`TypeVarRestriction::ActiveInt`]. `bool` and `string` are excluded,.
    ///
    /// as are the §1.1.1 reserved spellings.
    ///
    ActiveNumeric,
    /// An operation argument is a scalar or tensor with float precision.
    ///
    /// This is a value constraint, not the surface `Float` type-argument bound.
    ///
    FloatValue,
    /// An operation argument is a scalar or tensor with signed integer precision.
    ///
    IntValue,
    /// An operation argument is a scalar or tensor with numeric precision.
    ///
    NumericValue,
    /// §5.9's explicit dtype set: the variable may instantiate only at a.
    ///
    /// listed member. Unlike a family, this does not widen when §1.1.
    ///
    /// activates a dtype.
    ///
    ActiveSet(PrimSet),
}

impl TypeVarRestriction {
    /// The `spec/04-type-system.md` §5.9 family name, as diagnostics and the.
    ///
    /// Surf surface spell it.
    ///
    /// `None` for §5.9's explicit set, which is not a family and has no.
    ///
    /// family name.
    ///
    ///
    /// This is deliberately fallible. An infallible accessor returning `""`.
    ///
    /// let a set reach eight diagnostics as an empty name, which is the.
    ///
    /// silent-fallback class AGENTS.md's "Do Not Trust Green" section says to.
    ///
    /// audit. Returning `Option` makes every caller decide, and the compiler.
    ///
    /// enumerate them. Prefer [`TypeVarRestriction::bound_spelling`] or.
    ///
    /// [`TypeVarRestriction::bound_description`] in a diagnostic.
    ///
    pub fn family_name(self) -> Option<&'static str> {
        match self {
            TypeVarRestriction::ActiveFloat | TypeVarRestriction::FloatValue => Some("Float"),
            TypeVarRestriction::ActiveInt | TypeVarRestriction::IntValue => Some("Int"),
            TypeVarRestriction::ActiveNumeric | TypeVarRestriction::NumericValue => Some("Numeric"),
            TypeVarRestriction::ActiveSet(_) => None,
        }
    }

    /// The §5.9 spelling a diagnostic should name: a family name, or the.
    ///
    /// set's canonical `{d1, d2}` form.
    ///
    pub fn bound_spelling(self) -> String {
        match self {
            TypeVarRestriction::ActiveSet(set) => set.spelling(),
            family => family
                .family_name()
                .expect("a non-set restriction names a family")
                .to_string(),
        }
    }

    /// How a diagnostic names this bound's *kind* and spelling together.
    ///
    ///
    /// The family rendering of this value is byte-identical to the wording.
    ///
    /// that predates §5.9's set form. That is not the same as the composed.
    ///
    /// MESSAGE being byte-identical: a sentence that hoists "dtype families".
    ///
    /// as a shared plural across two operands churns if each operand.
    ///
    /// describes itself. `unify.rs::describe_bound_clash` is the one such.
    ///
    /// site, and it keeps the original plural for the all-families case.
    ///
    pub fn bound_description(self) -> String {
        match self {
            TypeVarRestriction::ActiveSet(set) => format!("dtype set `{}`", set.spelling()),
            family => format!(
                "dtype family `{}`",
                family
                    .family_name()
                    .expect("a non-set restriction names a family")
            ),
        }
    }

    /// Whether this bound is one of §5.9's three families rather than an.
    ///
    /// explicit set. A value constraint is not a §5.9 bound at all, so it is.
    ///
    /// not a family either.
    ///
    pub fn is_family(self) -> bool {
        !matches!(self, TypeVarRestriction::ActiveSet(_)) && !self.is_value_constraint()
    }

    /// Whether this is a §5.9 declaration bound rather than an operation's.
    ///
    /// value constraint. Prefer this over listing the family variants: a gate.
    ///
    /// written as `matches!(r, ActiveFloat | ActiveInt | ActiveNumeric)` is.
    ///
    /// one a new bound form falls straight out of, which is how chelis#2443.
    ///
    /// shipped two defects of the same class.
    ///
    pub fn is_declaration_bound(self) -> bool {
        !self.is_value_constraint()
    }

    /// Whether every dtype this bound admits is a float. True for `Float`,.
    ///
    /// for a set of floats, and false for a mixed set.
    ///
    pub fn admits_only_floats(self) -> bool {
        let admitted = PrimSet::of_family(self.precision_family());
        !admitted.is_empty() && admitted.members().all(|prim| prim.is_float())
    }

    /// Whether every dtype this bound admits is one `family` admits.
    ///
    pub fn admits_only_within(self, family: TypeVarRestriction) -> bool {
        let admitted = PrimSet::of_family(self.precision_family());
        !admitted.is_empty() && admitted.members().all(|prim| family.admits(prim))
    }

    /// Whether every dtype this bound admits is a signed integer.
    ///
    pub fn admits_only_integers(self) -> bool {
        let admitted = PrimSet::of_family(self.precision_family());
        !admitted.is_empty() && admitted.members().all(|prim| prim.is_integer())
    }

    /// The family's membership, spelled for a diagnostic reader who has not.
    ///
    /// read §5.9.
    ///
    pub fn membership_gloss(self) -> &'static str {
        match self {
            TypeVarRestriction::ActiveFloat => "the active float dtypes",
            TypeVarRestriction::ActiveInt => "the active signed integer dtypes",
            TypeVarRestriction::ActiveNumeric => "the active numeric dtypes",
            TypeVarRestriction::FloatValue => "float scalars or tensors",
            TypeVarRestriction::IntValue => "signed integer scalars or tensors",
            TypeVarRestriction::NumericValue => "numeric scalars or tensors",
            TypeVarRestriction::ActiveSet(_) => "the dtypes the bound lists",
        }
    }

    /// Whether `prim` is a member of this family.
    ///
    ///
    /// Membership follows §1.1's active set through [`Prim::is_float`] and.
    ///
    /// [`Prim::is_integer`], both of which already exclude the §1.1.1.
    ///
    /// deferred `f8e4m3`. `Prim::is_numeric` is deliberately NOT used for.
    ///
    /// `ActiveNumeric`: it admits `f8e4m3` so that rejection sites can.
    ///
    /// describe it, and a bound must not admit a dtype §1.1 does not.
    ///
    pub fn admits(self, prim: Prim) -> bool {
        match self.precision_family() {
            TypeVarRestriction::ActiveFloat => prim.is_float(),
            TypeVarRestriction::ActiveInt => prim.is_integer(),
            TypeVarRestriction::ActiveNumeric => prim.is_float() || prim.is_integer(),
            // A set lists its members, so it does not consult §1.1.
            //
            // membership and does not widen when §1.1 does.
            //
            TypeVarRestriction::ActiveSet(set) => set.contains(prim),
            _ => unreachable!("precision_family returns a primitive dtype family"),
        }
    }

    pub fn is_value_constraint(self) -> bool {
        matches!(self, Self::FloatValue | Self::IntValue | Self::NumericValue)
    }

    pub fn precision_family(self) -> Self {
        match self {
            Self::FloatValue => Self::ActiveFloat,
            Self::IntValue => Self::ActiveInt,
            Self::NumericValue => Self::ActiveNumeric,
            family => family,
        }
    }

    pub fn for_value(self) -> Self {
        match self.precision_family() {
            Self::ActiveFloat => Self::FloatValue,
            Self::ActiveInt => Self::IntValue,
            Self::ActiveNumeric => Self::NumericValue,
            // A set has no value-constraint counterpart: §5.9 bounds a.
            //
            // declaration's binder, not an operation's operand.
            //
            set @ Self::ActiveSet(_) => set,
            _ => unreachable!("precision_family returns a primitive dtype family"),
        }
    }

    /// The family both bounds admit, or `None` when they share no dtype.
    ///
    ///
    /// Whether every dtype this bound admits is also admitted by `wider`.
    ///
    ///
    /// [04-INF-9] needs a generic body's necessary restriction to follow.
    ///
    /// from the declared contract, so a declaration satisfies its body when.
    ///
    /// its bound is at least as narrow as the body's requirement. Exact.
    ///
    /// equality was enough while every bound was one of three families.
    ///
    /// §5.9's set form makes the subset relation the operative one, since.
    ///
    /// `{f32, f64}` satisfies a `Float` requirement without equalling it.
    ///
    pub fn is_at_least_as_narrow_as(self, wider: TypeVarRestriction) -> bool {
        let mine = PrimSet::of_family(self.precision_family());
        let theirs = PrimSet::of_family(wider.precision_family());
        !mine.is_empty() && mine.intersect(theirs) == mine
    }

    /// [04-DTYPE-2]: unifying two bounded variables yields the intersection.
    ///
    /// of the dtypes their bounds admit. Between families, `Numeric` is the.
    ///
    /// join of the other two, so every non-empty intersection is itself one.
    ///
    /// of the three families and the only empty case is `Float` against.
    ///
    /// `Int`. With §5.9's explicit set on either side the result is a set,.
    ///
    /// since an intersection that no longer denotes a family is a set.
    ///
    pub fn intersect(self, other: TypeVarRestriction) -> Option<TypeVarRestriction> {
        use TypeVarRestriction::{ActiveFloat, ActiveInt, ActiveNumeric, ActiveSet};
        let (lhs, rhs) = (self.precision_family(), other.precision_family());
        if matches!(lhs, ActiveSet(_)) || matches!(rhs, ActiveSet(_)) {
            let members = PrimSet::of_family(lhs).intersect(PrimSet::of_family(rhs));
            // A set has no value-constraint counterpart, so `for_value` is.
            //
            // not applied here: §5.9 bounds a declaration's binder.
            //
            return (!members.is_empty()).then_some(ActiveSet(members));
        }
        let family = match (lhs, rhs) {
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
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DimVar(pub u32);

/// A unique identifier for a *rank* variable — a `Dim::Rank` stands for an.
///
/// entire shape vector (Tier-2 rank polymorphism, `spec/design/rank_polymorphism.md`).
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RankVar(pub u32);

/// A tensor dimension.
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Dim {
    /// Concrete named dimension (e.g., batch, hidden).
    ///
    Name(String),
    /// Polymorphic dimension variable (e.g., a, b).
    ///
    Var(DimVar),
    /// Fixed numeric size (e.g., 512).
    ///
    Lit(i64),
    /// Wildcard — unknown/dynamic dimension.
    ///
    Wildcard,
    /// Rank variable — a *name-preserving spread* standing for a run of dims.
    ///
    /// (rank polymorphism). Tier-2 used it only as the sole element of a shape.
    ///
    /// (`tensor[..r, p]`). Tier-3 allows it interleaved with concrete anchors.
    ///
    /// (`tensor[..pre, seq, ..post, p]`). Structural invariant: a given.
    ///
    /// `RankVar` appears at most once per tensor dim list, and two spreads are.
    ///
    /// never *split* against a ground while both unbound (an undetermined.
    ///
    /// boundary — rejected at unification). A spread binds to the actual named.
    ///
    /// dims it covers, so names are preserved, not erased. Eliminated by.
    ///
    /// monomorphization. no `Dim::Rank` reaches a backend.
    ///
    Rank(RankVar),
}

/// Tensor element precision slot.
///
///
/// Per `spec/04-type-system.md` §5.8 and `spec/02-surf-syntax.md`, the.
///
/// precision slot of a tensor type may be either a concrete primitive.
///
/// (e.g. `f32`, `i32`) or a sig-bound type variable (precision.
///
/// polymorphism, WS-A5). After monomorphization every reachable.
///
/// tensor must carry `TensorPrec::Concrete(_)`. backends assert this.
///
/// invariant at lowering time.
///
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TensorPrec {
    /// A concrete numeric primitive. the only shape backends accept.
    ///
    Concrete(Prim),
    /// A sig-quantified type variable. Resolved by unification, then.
    ///
    /// substituted to `Concrete(_)` by `Subst::apply` once bound.
    ///
    Var(TypeVar),
}

impl TensorPrec {
    /// Returns the concrete primitive if this slot is already resolved,.
    ///
    /// or `None` if it is still a type variable. Backends and IR.
    ///
    /// builders that must have a concrete dtype call this and treat.
    ///
    /// `None` as a monomorphization bug.
    ///
    pub fn as_concrete(&self) -> Option<Prim> {
        match self {
            TensorPrec::Concrete(p) => Some(*p),
            TensorPrec::Var(_) => None,
        }
    }

    /// Convenience: short rendering of the slot, suitable for diagnostics.
    ///
    /// Concrete precisions render as their canonical name. vars render.
    ///
    /// as `?N` matching `Type::Var` formatting.
    ///
    pub fn render(&self) -> String {
        match self {
            TensorPrec::Concrete(p) => p.name().to_string(),
            TensorPrec::Var(v) => format!("?{}", v.0),
        }
    }

    /// Diagnostic-only alias for [`TensorPrec::render`]. matches the.
    ///
    /// `Prim::name()` ergonomic for call sites that previously took a.
    ///
    /// bare `Prim`. Returns an owned `String` because var precisions.
    ///
    /// have no `'static` representation.
    ///
    pub fn name(&self) -> String {
        self.render()
    }

    /// True iff this precision is concretely a float per.
    ///
    /// [`Prim::is_float`]. A polymorphic precision var returns `false`:.
    ///
    /// a not-yet-resolved precision is not yet known to be float, so.
    ///
    /// any "this op needs a float" check should not silently accept a.
    ///
    /// `Var` slot.
    ///
    pub fn is_float(&self) -> bool {
        matches!(self, TensorPrec::Concrete(p) if p.is_float())
    }

    /// True iff this precision is concretely an integer per.
    ///
    /// [`Prim::is_integer`]. `Var` returns `false` for the same reason.
    ///
    /// as [`TensorPrec::is_float`].
    ///
    pub fn is_integer(&self) -> bool {
        matches!(self, TensorPrec::Concrete(p) if p.is_integer())
    }

    /// True iff this precision is concretely numeric per.
    ///
    /// [`Prim::is_numeric`]. `Var` returns `false`.
    ///
    pub fn is_numeric(&self) -> bool {
        matches!(self, TensorPrec::Concrete(p) if p.is_numeric())
    }

    /// Resolve the spec/04-type-system.md §5.7.1 reduce_sum result.
    ///
    /// precision. Forwards to [`Prim::default_reduce_sum_result_precision`].
    ///
    /// for `Concrete`. For `Var`, returns an error: a polymorphic.
    ///
    /// precision must be resolved (or rejected) before the reduce-sum.
    ///
    /// resolution rule applies.
    ///
    pub fn default_reduce_sum_result_precision(self) -> Result<Prim, String> {
        match self {
            TensorPrec::Concrete(p) => p
                .default_reduce_sum_result_precision()
                .map_err(|error| error.to_string()),
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
///
/// Keeping dimensions out of [`Type`] makes an extent impossible to consume.
///
/// as an ordinary type while preserving the declared argument order.
///
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NominalArg<E> {
    Type(Type<E>),
    Dimension(Dim),
}

/// Checker-owned kind of a nominal declaration parameter.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NominalParamKind {
    Type,
    Dimension,
}

impl<E> NominalArg<E> {
    pub fn as_type(&self) -> Option<&Type<E>> {
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

impl<E> From<Type<E>> for NominalArg<E> {
    fn from(value: Type<E>) -> Self {
        Self::Type(value)
    }
}

impl<E> fmt::Display for NominalArg<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(ty) => ty.fmt(f),
            Self::Dimension(dim) => dim.fmt(f),
        }
    }
}

/// Chelis type.
///
///
/// Every recursive position is shared behind an `Arc`, so cloning a type.
///
/// copies one pointer per child instead of the child's whole tree, and a.
///
/// type nested `n` deep that every enclosing node records costs `O(n)` to.
///
/// keep rather than `O(n^2)` (chelis#2622). The children are immutable once.
///
/// built, so the sharing is never observable. `Arc` serializes, compares, and formats.
///
/// as its contents, so the wire forms, equality, and `Debug` output are the.
///
/// same as for owned children.
///
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Type<E> {
    /// Primitive type (f32, i32, bool, etc.).
    ///
    Prim(Prim),
    /// Function type: args → return.
    ///
    Fn(Arc<[Type<E>]>, Arc<Type<E>>),
    /// Read-only non-owning borrow of a value.
    ///
    Ref(Arc<Type<E>>),
    /// Tensor type: dimensions + precision slot. Precision is.
    ///
    /// `TensorPrec::Concrete(_)` for fully-monomorphized tensors and.
    ///
    /// `TensorPrec::Var(_)` for sig-quantified precision polymorphism.
    ///
    /// per `spec/04-type-system.md` §5.8 (WS-A5).
    ///
    Tensor(Vec<Dim>, TensorPrec),
    /// Algebraic data type whose arguments are all ordinary types.
    ///
    Adt(String, Arc<[Type<E>]>),
    /// Algebraic data type with at least one dimension-kinded argument.
    ///
    /// Keeping this representation distinct makes a dimension impossible to.
    ///
    /// consume through an ordinary `Type::Adt` argument path.
    ///
    KindedAdt(String, Arc<[NominalArg<E>]>),
    /// Type variable (for inference).
    ///
    Var(TypeVar),
    /// Tuple type.
    ///
    Tuple(Arc<[Type<E>]>),
    /// Unit type.
    ///
    Unit,
    /// An error sentinel with explicit evidence from its caller.
    ///
    /// Hosted callers use their private diagnostic witness.
    /// The core does not create or convert that witness.
    Error(E),
}

impl<E> Type<E> {
    /// A function type from its parameter types and result type.
    ///
    pub fn function(params: Vec<Type<E>>, result: Type<E>) -> Self {
        Self::Fn(params.into(), Arc::new(result))
    }

    /// A read-only borrow of `referent`.
    ///
    pub fn reference(referent: Type<E>) -> Self {
        Self::Ref(Arc::new(referent))
    }

    /// A tuple of `items`.
    ///
    pub fn tuple(items: Vec<Type<E>>) -> Self {
        Self::Tuple(items.into())
    }

    /// An algebraic data type whose arguments are all ordinary types.
    ///
    pub fn adt(name: String, args: Vec<Type<E>>) -> Self {
        Self::Adt(name, args.into())
    }

    /// An algebraic data type with at least one dimension-kinded argument.
    ///
    pub fn kinded_adt(name: String, args: Vec<NominalArg<E>>) -> Self {
        Self::KindedAdt(name, args.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Effect {
    Accum,
    Io,
    /// Chelis-native testing effect. Pinned at the root, no handler.
    ///
    /// Any function that (directly or transitively) calls a `test_assert_*`.
    ///
    /// builtin gets this effect, preventing assertions from silently.
    ///
    /// leaking into `! {}` production code.
    ///
    Test,
    Resource(String),
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accum => f.write_str("Accum"),
            Self::Io => f.write_str("IO"),
            Self::Test => f.write_str("Test"),
            Self::Resource(device) => write!(f, "Resource(\"{device}\")"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// The checked operand/result relation of a first-class builtin operation.
///
///
/// Builtin schemes own these contracts. Aliasing, higher-order passage,.
///
/// import, and serialization retain them. applying the value consumes and.
///
/// decides them. A newly authored wrapper does not create one from its body.
///
/// because [04-INF-9] needs an explicit enough parameter contract.
///
///
/// Each variant holds the route's operands and its result in FIXED fields.
///
/// A rule tag plus an operand vector would describe `(List, List)` membership.
///
/// and lose the equations the rule also imposes -- that `concat`'s two lists.
///
/// share an element type and that its result is a list at that same type --.
///
/// so a mixed-element call, or one whose result was annotated to something.
///
/// the rule never produces, would satisfy the carried constraint.
///
///
/// Scheme instantiation renames the variables in these types and the scheme body together.
///
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CollectionConstraint<E> {
    /// [05-OP-20..22]: float scalar/tensor to bool on the same surface.
    ///
    FloatClassify {
        kind: chelis_numeric_core::dtype_semantics::FloatClassify,
        operand: Type<E>,
        result: Type<E>,
    },
    /// `len(operand) -> i64`: the operand is a `List` or a `Dict`.
    ///
    Len { operand: Type<E>, result: Type<E> },
    /// `index(list, index) -> result`: `list` is a `List[e]`, `index` is.
    ///
    /// exactly `i64`, and `result` is `e`.
    ///
    Index {
        list: Type<E>,
        index: Type<E>,
        result: Type<E>,
    },
    /// `append(list, value) -> result`: `list` is a `List[e]`, `value` is an.
    ///
    /// `e`, and `result` is `List[e]`.
    ///
    Append {
        list: Type<E>,
        value: Type<E>,
        result: Type<E>,
    },
    /// `concat(lhs, rhs) -> result`, both spec/04 §4.5.4 overloads: two.
    ///
    /// `List[e]` giving `List[e]`, or a `List[tensor[..]]` and an `i32`.
    ///
    /// axis giving a tensor.
    ///
    Concat {
        lhs: Type<E>,
        rhs: Type<E>,
        result: Type<E>,
    },
    /// [05-OP-69]: scalar i64 -> key, or tensor[D,i64] -> tensor[D,key].
    ///
    KeyFromSeed { operand: Type<E>, result: Type<E> },
    /// [05-OP-70]: each half preserves the key operand's surface and shape.
    ///
    SplitKey { operand: Type<E>, result: Type<E> },
    /// [05-OP-71]: append one runtime count axis to a key's shape.
    ///
    SplitKeys {
        operand: Type<E>,
        count: Type<E>,
        result: Type<E>,
    },
    /// [05-OP-72]: equal scalar/tensor surfaces and exactly equal shapes.
    ///
    FoldIn {
        operand: Type<E>,
        index: Type<E>,
        result: Type<E>,
    },
}

impl<E> CollectionConstraint<E> {
    /// Whether deciding this relation at a value's application sees every.
    ///
    /// operand fact the direct call's checker reads ([04-INF-9]). `concat`.
    ///
    /// receives the call's axis expression and visible list elements, and.
    ///
    /// `split_keys` its statically folded count, as call evidence.
    ///
    /// (`CollectionCallEvidence`). The other relations read operand types.
    ///
    /// alone. A relation that a direct checker reads more of than this.
    ///
    /// declares cannot make its builtin a value.
    ///
    pub fn transports_static_operand_rule(&self) -> bool {
        match self {
            Self::FloatClassify { .. }
            | Self::SplitKeys { .. }
            | Self::Len { .. }
            | Self::Index { .. }
            | Self::Append { .. }
            | Self::Concat { .. }
            | Self::KeyFromSeed { .. }
            | Self::SplitKey { .. }
            | Self::FoldIn { .. } => true,
        }
    }

    /// The builtin this constraint belongs to, for diagnostics.
    ///
    pub fn builtin(&self) -> &'static str {
        match self {
            Self::FloatClassify { kind, .. } => kind.name(),
            Self::KeyFromSeed { .. } => "key_from_seed",
            Self::SplitKey { .. } => "split_key",
            Self::SplitKeys { .. } => "split_keys",
            Self::FoldIn { .. } => "fold_in",
            Self::Len { .. } => "len",
            Self::Index { .. } => "index",
            Self::Append { .. } => "append",
            Self::Concat { .. } => "concat",
        }
    }

    /// The operand positions, in the order the rule reads them. The result is.
    ///
    /// deliberately absent: an unresolved RESULT is what the rule computes,.
    ///
    /// not what it waits on.
    ///
    pub fn operands(&self) -> Vec<&Type<E>> {
        match self {
            Self::FloatClassify { operand, .. }
            | Self::KeyFromSeed { operand, .. }
            | Self::SplitKey { operand, .. } => vec![operand],
            Self::SplitKeys { operand, count, .. } => vec![operand, count],
            Self::FoldIn { operand, index, .. } => vec![operand, index],
            Self::Len { operand, .. } => vec![operand],
            Self::Index { list, index, .. } => vec![list, index],
            Self::Append { list, value, .. } => vec![list, value],
            Self::Concat { lhs, rhs, .. } => vec![lhs, rhs],
        }
    }

    /// Every type this constraint carries, operands and result alike. Used to.
    ///
    /// decide which variables a pending constraint keeps monomorphic.
    ///
    pub fn carried_types(&self) -> Vec<&Type<E>> {
        match self {
            Self::FloatClassify {
                operand, result, ..
            }
            | Self::KeyFromSeed { operand, result }
            | Self::SplitKey { operand, result } => {
                vec![operand, result]
            }
            Self::SplitKeys {
                operand,
                count,
                result,
            } => vec![operand, count, result],
            Self::FoldIn {
                operand,
                index,
                result,
            } => vec![operand, index, result],
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
    ///
    pub fn result(&self) -> &Type<E> {
        match self {
            Self::FloatClassify { result, .. }
            | Self::KeyFromSeed { result, .. }
            | Self::SplitKey { result, .. }
            | Self::SplitKeys { result, .. }
            | Self::FoldIn { result, .. }
            | Self::Len { result, .. }
            | Self::Index { result, .. }
            | Self::Append { result, .. }
            | Self::Concat { result, .. } => result,
        }
    }

    /// Rewrite every carried type with `f`. Instantiation passes the.
    ///
    /// quantifier renaming. discharge passes the current substitution.
    ///
    pub fn map_types(&self, f: impl Fn(&Type<E>) -> Type<E>) -> Self {
        match self {
            Self::FloatClassify {
                kind,
                operand,
                result,
            } => Self::FloatClassify {
                kind: *kind,
                operand: f(operand),
                result: f(result),
            },
            Self::KeyFromSeed { operand, result } => Self::KeyFromSeed {
                operand: f(operand),
                result: f(result),
            },
            Self::SplitKey { operand, result } => Self::SplitKey {
                operand: f(operand),
                result: f(result),
            },
            Self::SplitKeys {
                operand,
                count,
                result,
            } => Self::SplitKeys {
                operand: f(operand),
                count: f(count),
                result: f(result),
            },
            Self::FoldIn {
                operand,
                index,
                result,
            } => Self::FoldIn {
                operand: f(operand),
                index: f(index),
                result: f(result),
            },
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

/// A result equation retains its direction through function generalization.
///
/// It checks a produced value. it cannot supply an unresolved Grad input.
///
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ResultConstraint<E> {
    Annotation {
        actual: Type<E>,
        declared: Type<E>,
    },
    Join {
        inputs: Vec<Type<E>>,
        result: Type<E>,
    },
}

impl<E> ResultConstraint<E> {
    pub fn map_types(&self, mut map: impl FnMut(&Type<E>) -> Type<E>) -> Self {
        match self {
            Self::Annotation { actual, declared } => Self::Annotation {
                actual: map(actual),
                declared: map(declared),
            },
            Self::Join { inputs, result } => Self::Join {
                inputs: inputs.iter().map(&mut map).collect(),
                result: map(result),
            },
        }
    }

    pub fn types(&self) -> Vec<&Type<E>> {
        match self {
            Self::Annotation { actual, declared } => vec![actual, declared],
            Self::Join { inputs, result } => {
                inputs.iter().chain(core::iter::once(result)).collect()
            }
        }
    }
}

/// The input type and result equations before result-only inference solved.
///
/// a helper's published signature. Each use instantiates these together.
///
/// Quantifiers are closed over the scope components of these equations.
///
/// unquantified variables retain identity with their inference producer even.
///
/// when the conventional solved signature no longer mentions them.
///
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResultOrigin<E> {
    pub body: Type<E>,
    pub tvars: Vec<TypeVar>,
    /// Restrictions belong to these raw quantifiers. the published scheme.
    ///
    /// has a separately normalized restriction ledger for its solved body.
    ///
    pub tvar_restrictions: Vec<(TypeVar, TypeVarRestriction)>,
    pub dvars: Vec<DimVar>,
    pub rvars: Vec<RankVar>,
    pub equations: Vec<ResultConstraint<E>>,
}

impl<E: Clone> ResultOrigin<E> {
    pub fn aggregate(
        body: &Type<E>,
        tvars: &[TypeVar],
        dvars: &[DimVar],
        rvars: &[RankVar],
        vg: &mut VarGen,
    ) -> Option<Self> {
        let Type::Fn(params, result) = body else {
            return None;
        };
        let mut occurrences = alloc::collections::BTreeMap::<TypeVar, Vec<Type<E>>>::new();
        fn split<E: Clone>(
            ty: &Type<E>,
            variables: &[TypeVar],
            occurrences: &mut alloc::collections::BTreeMap<TypeVar, Vec<Type<E>>>,
            vg: &mut VarGen,
        ) -> Type<E> {
            match ty {
                Type::Var(var) if variables.contains(var) => {
                    let fresh = vg.fresh_type();
                    occurrences.entry(*var).or_default().push(fresh.clone());
                    fresh
                }
                Type::Fn(args, ret) => Type::Fn(
                    args.iter()
                        .map(|ty| split(ty, variables, occurrences, vg))
                        .collect(),
                    Arc::new(split(ret, variables, occurrences, vg)),
                ),
                Type::Tuple(items) => Type::Tuple(
                    items
                        .iter()
                        .map(|ty| split(ty, variables, occurrences, vg))
                        .collect(),
                ),
                Type::Adt(name, args) => Type::Adt(
                    name.clone(),
                    args.iter()
                        .map(|ty| split(ty, variables, occurrences, vg))
                        .collect(),
                ),
                Type::KindedAdt(name, args) => Type::KindedAdt(
                    name.clone(),
                    args.iter()
                        .map(|arg| match arg {
                            NominalArg::Type(ty) => {
                                NominalArg::Type(split(ty, variables, occurrences, vg))
                            }
                            NominalArg::Dimension(_) => arg.clone(),
                        })
                        .collect(),
                ),
                Type::Ref(inner) => Type::reference(split(inner, variables, occurrences, vg)),
                Type::Tensor(dims, TensorPrec::Var(var)) if variables.contains(var) => {
                    let Type::Var(fresh) = split(&Type::Var(*var), variables, occurrences, vg)
                    else {
                        unreachable!()
                    };
                    Type::Tensor(dims.clone(), TensorPrec::Var(fresh))
                }
                _ => ty.clone(),
            }
        }
        let params = params
            .iter()
            .map(|ty| split(ty, tvars, &mut occurrences, vg))
            .collect();
        let mut quantified = tvars.to_vec();
        let equations = occurrences
            .into_iter()
            .map(|(var, inputs)| {
                quantified.extend(inputs.iter().filter_map(|ty| match ty {
                    Type::Var(var) => Some(*var),
                    _ => None,
                }));
                ResultConstraint::Join {
                    inputs,
                    result: Type::Var(var),
                }
            })
            .collect::<Vec<_>>();
        (!equations.is_empty()).then(|| Self {
            body: Type::Fn(params, result.clone()),
            tvars: quantified,
            tvar_restrictions: Vec::new(),
            dvars: dvars.to_vec(),
            rvars: rvars.to_vec(),
            equations,
        })
    }
}

/// A polymorphic type scheme: ∀ tvars, dvars. body.
///
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(serialize = "E: Serialize", deserialize = "E: Deserialize<'de>"))
)]
pub struct Scheme<E> {
    /// Mandatory on the wire: old snapshots must not erase result origin.
    ///
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "deserialize_result_origin")
    )]
    pub result_origin: Option<ResultOrigin<E>>,
    pub tvars: Vec<TypeVar>,
    /// Domain restrictions for quantified type variables. Entries are kept.
    ///
    /// in quantifier order for deterministic serialization.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    pub tvar_restrictions: Vec<(TypeVar, TypeVarRestriction)>,
    pub dvars: Vec<DimVar>,
    /// Quantified rank variables (Tier-2 rank polymorphism). Usually empty.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    pub rvars: Vec<RankVar>,
    /// Checked builtin-operation relations transported by this function.
    ///
    /// value and renamed at each instantiation. Usually empty.
    ///
    ///
    /// Deliberately NOT `#[cfg_attr(feature = "serde", serde(default))]`. A default would let a scheme.
    ///
    /// persisted before this field existed decode as unconstrained and erase a.
    ///
    /// checked function value's contract on the reuse path. A payload without.
    ///
    /// this field must fail to decode. the TypeEnv format version makes that.
    ///
    /// an obsolete-snapshot error.
    ///
    pub constraints: Vec<CollectionConstraint<E>>,
    pub body: Type<E>,
}

#[cfg(feature = "serde")]
fn deserialize_result_origin<'de, E: Deserialize<'de>, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ResultOrigin<E>>, D::Error> {
    Option::<ResultOrigin<E>>::deserialize(deserializer)
}

impl<E> Scheme<E> {
    /// A monomorphic scheme (no quantified variables).
    ///
    pub fn mono(ty: Type<E>) -> Scheme<E> {
        Scheme {
            result_origin: None,
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            constraints: vec![],
            body: ty,
        }
    }
}

impl<E> fmt::Display for Type<E> {
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
    /// User-facing rendering of a tensor dimension, matching the CLI surface.
    ///
    /// (`format_cli_dim`): a named dim by its name, a dim variable as `d<id>`, a.
    ///
    /// literal by its value, the wildcard as `*`, and a rank spread as `..r<id>`.
    ///
    /// Used by `Type`'s `Display` so diagnostics read `tensor[..r0, seq, ..r1, f32]`.
    ///
    /// instead of the internal `Debug` form `Rank(RankVar(0)), Name("seq")`.
    ///
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

/// Counter for generating fresh type and dimension variables.
///
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VarGen {
    next_tvar: u32,
    next_dvar: u32,
    #[cfg_attr(feature = "serde", serde(default))]
    next_rvar: u32,
}

/// The next fresh identifier in each solver-variable class.
///
///
/// Level transitions record these watermarks instead of attaching a level to.
///
/// every variable. A variable's mint level is therefore the final transition.
///
/// whose corresponding watermark is no greater than its identifier.
///
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VarWatermarks {
    pub next_tvar: u32,
    pub next_dvar: u32,
    pub next_rvar: u32,
}

impl VarGen {
    pub fn watermarks(&self) -> VarWatermarks {
        VarWatermarks {
            next_tvar: self.next_tvar,
            next_dvar: self.next_dvar,
            next_rvar: self.next_rvar,
        }
    }

    /// Reserve an external type identifier before fresh allocation.
    pub fn reserve_tvar(&mut self, var: TypeVar) {
        self.next_tvar = self.next_tvar.max(
            var.0
                .checked_add(1)
                .expect("type variable identifier space exhausted"),
        );
    }

    /// Reserve an external dimension identifier before fresh allocation.
    pub fn reserve_dvar(&mut self, var: DimVar) {
        self.next_dvar = self.next_dvar.max(
            var.0
                .checked_add(1)
                .expect("dimension variable identifier space exhausted"),
        );
    }

    /// Reserve an external rank identifier before fresh allocation.
    pub fn reserve_rvar(&mut self, var: RankVar) {
        self.next_rvar = self.next_rvar.max(
            var.0
                .checked_add(1)
                .expect("rank variable identifier space exhausted"),
        );
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
    ///
    pub fn fresh_rvar(&mut self) -> RankVar {
        let v = RankVar(self.next_rvar);
        self.next_rvar += 1;
        v
    }

    pub fn fresh_type<E>(&mut self) -> Type<E> {
        Type::Var(self.fresh_tvar())
    }

    pub fn fresh_dim(&mut self) -> Dim {
        Dim::Var(self.fresh_dvar())
    }
}

#[cfg(test)]
mod dtype_family_bound_tests {
    //! Lock `spec/04-type-system.md` §5.9's family membership and.
    //!
    //! [04-DTYPE-2]'s intersection rule against the active dtype set.
    //!

    use super::*;

    const EVERY_PRIM: [Prim; 12] = [
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
        Prim::Key,
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
        // §1.1.1's deferred `f8e4m3` is numeric for diagnostic purposes but.
        //
        // is not an active dtype, so no bound may admit it.
        //
        for restriction in [
            TypeVarRestriction::ActiveFloat,
            TypeVarRestriction::ActiveInt,
            TypeVarRestriction::ActiveNumeric,
        ] {
            for prim in [Prim::Bool, Prim::String, Prim::Key, Prim::F8e4m3] {
                assert!(
                    !restriction.admits(prim),
                    "{:?} must not admit {prim:?}",
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
                        "{:?}/{:?} share {shared:?} but intersect to nothing",
                        left.family_name(),
                        right.family_name()
                    ),
                }
            }
        }
    }

    #[test]
    fn family_names_match_the_surf_spelling() {
        assert_eq!(TypeVarRestriction::ActiveFloat.family_name(), Some("Float"));
        assert_eq!(TypeVarRestriction::ActiveInt.family_name(), Some("Int"));
        assert_eq!(
            TypeVarRestriction::ActiveNumeric.family_name(),
            Some("Numeric")
        );
    }
}
