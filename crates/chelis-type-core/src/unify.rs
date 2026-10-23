//! Unification for the allocation-based portable type model.
//!
//! # Alias resolution
//!
//! Type, dimension, and rank bindings retain their logical alias edges.
//! Ordinary resolution follows each chain iteratively and caches its terminal owner.
//! Scheme resolution reads logical edges and stops at quantified identifiers.
//! Every logical mutation clears the transient lookup cache.
//!
//! Private stores support shared resolution through an immutable receiver.
//! Store guards do not cross recursive or cross-store calls.
//! Interrupted protected operations reject later access.
//! Clones and serialized snapshots omit transient lookup results.

use crate::store::{Bindings, Store};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use chelis_unord::{PersistentMap, PersistentSet, UnordMap, UnordSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::model::*;
use chelis_numeric_core::prim::Prim;

// Counts the type nodes `Subst::apply` visits, for the chelis#2622 scaling.
//
// oracle: checking a nested program applies the substitution to each node's.
//
// type at most once per stamp, never once per enclosing level.
//
#[cfg(any(test, feature = "test-instrumentation"))]
std::thread_local! {
    static APPLY_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(any(test, feature = "test-instrumentation"))]
pub fn take_apply_visits() -> usize {
    APPLY_VISITS.with(|visits| visits.replace(0))
}

/// Applied shared children for [`Subst::apply_memoized`], keyed by the.
///
/// address of the child's allocation. Each entry holds the original child as.
///
/// well as its applied form, so the address cannot be freed and reused while.
///
/// the memo lives. The memo is only ever looked up, never iterated.
///
pub struct ApplyMemo<E> {
    children: AppliedShared<Type<E>>,
    lists: AppliedShared<[Type<E>]>,
    arguments: AppliedShared<[NominalArg<E>]>,
}

/// Original and applied form of each shared child, by allocation address.
///
type AppliedShared<T> = UnordMap<usize, (Arc<T>, Arc<T>, Applied)>;

/// What applying the substitution to one type established about the result.
///
#[derive(Debug, Clone, Copy)]
struct Applied {
    /// The result holds no type, dimension, rank or precision variable.
    ///
    ground: bool,
    /// The result is the type that was applied to, unchanged.
    ///
    same: bool,
}

impl Applied {
    const GROUND: Self = Self {
        ground: true,
        same: true,
    };

    fn and(self, other: Self) -> Self {
        Self {
            ground: self.ground && other.ground,
            same: self.same && other.same,
        }
    }

    fn changed(self) -> Self {
        Self {
            same: false,
            ..self
        }
    }
}

/// A dimension the substitution cannot change: no variable of either kind.
///
fn dim_is_ground(dim: &Dim) -> bool {
    matches!(dim, Dim::Name(_) | Dim::Lit(_) | Dim::Wildcard)
}

/// Shared children that applying found ground, by allocation address.
///
/// (chelis#3819).
///
///
/// Applying a substitution to a ground type returns an equal type and.
///
/// touches no binding: there is no variable to resolve, so there is no chain.
///
/// to compress. [`Subst::apply`] therefore returns a recorded child as it.
///
/// stands instead of walking it. Without that, a checker that applies the.
///
/// substitution to a type at each of the levels that build it walks every.
///
/// finished subtree again at each level, which is quadratic in the depth.
///
///
/// Whether a type is ground does not depend on the substitution, so every.
///
/// copy of a substitution shares one store. Each entry holds its child, so.
///
/// its address cannot be freed and reused while it is recorded. The store.
///
/// sweeps the children only it still holds whenever it doubled since.
///
/// the last sweep, so it stays within a constant factor of the ground.
///
/// children that are live. It is only ever looked up, never iterated, apart.
///
/// from that sweep, whose outcome does not depend on its order.
///
#[derive(Clone)]
struct GroundTypes<E>(Arc<Store<GroundStore<E>>>);

impl<E> core::fmt::Debug for GroundTypes<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("GroundTypes")
    }
}

struct GroundStore<E> {
    addresses: UnordSet<usize>,
    held: Vec<GroundShared<E>>,
    sweep_at: usize,
}

/// The smallest store that is swept.
///
const GROUND_SWEEP_FLOOR: usize = 4096;

enum GroundShared<E> {
    Child(Arc<Type<E>>),
    List(Arc<[Type<E>]>),
    Arguments(Arc<[NominalArg<E>]>),
}

impl<E: Clone + PartialEq> GroundShared<E> {
    fn address(&self) -> usize {
        match self {
            Self::Child(child) => Arc::as_ptr(child).addr(),
            Self::List(items) => Arc::as_ptr(items).cast::<Type<E>>().addr(),
            Self::Arguments(arguments) => Arc::as_ptr(arguments).cast::<NominalArg<E>>().addr(),
        }
    }

    fn held_elsewhere(&self) -> bool {
        match self {
            Self::Child(child) => Arc::strong_count(child) > 1,
            Self::List(items) => Arc::strong_count(items) > 1,
            Self::Arguments(arguments) => Arc::strong_count(arguments) > 1,
        }
    }
}

impl<E: Clone + PartialEq> GroundTypes<E> {
    fn holds(&self, address: usize) -> bool {
        self.0.with(|store| store.addresses.contains(&address))
    }

    fn record(&self, shared: GroundShared<E>) {
        self.0.with(|store| {
            if !store.addresses.insert(shared.address()) {
                return;
            }
            store.held.push(shared);
            if store.held.len() >= store.sweep_at.max(GROUND_SWEEP_FLOOR) {
                store.held.retain(GroundShared::held_elsewhere);
                store.addresses = store.held.iter().map(GroundShared::address).collect();
                store.sweep_at = 2 * store.held.len();
            }
        })
    }
}

const INTERNAL_VMAP_AXIS_PREFIX: &str = "\0vmap-axis:";

fn mapped_axis_annotation(axis: usize) -> String {
    format!("{INTERNAL_VMAP_AXIS_PREFIX}{axis}")
}

fn parse_mapped_axis_annotation(annotation: &str) -> Option<usize> {
    annotation
        .strip_prefix(INTERNAL_VMAP_AXIS_PREFIX)?
        .parse()
        .ok()
}

/// A level change and the variable-generator state at which it took effect.
///
/// Transitions are append-only between persisted-context resumptions.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
enum LevelTransitionKind {
    Enter,
    Leave,
    Resume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct LevelTransition {
    kind: LevelTransitionKind,
    level: u32,
    watermarks: VarWatermarks,
}

/// The ordered enter, leave and resume transitions of one solver, kept.
///
/// persistent so a speculative copy of the substitution shares the log.
///
/// instead of copying it: the log grows with every scope the module enters,.
///
/// and a copy is taken per trial unification (chelis#2536). Serialized as.
///
/// the plain sequence of transitions.
///
#[derive(Debug, Clone, Default)]
struct TransitionLog {
    entries: PersistentMap<u32, LevelTransition>,
}

impl TransitionLog {
    fn push(&mut self, transition: LevelTransition) {
        let position = u32::try_from(self.entries.len()).expect("the transition log fits u32");
        self.entries.insert(position, transition);
    }

    #[cfg(any(test, feature = "test-instrumentation"))]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    /// The last transition whose watermark `watermark_of` is at most `id`.
    ///
    /// Watermarks never decrease along the log, so those transitions are a.
    ///
    /// prefix of it.
    ///
    fn last_at_or_below(
        &self,
        id: u32,
        watermark_of: impl Fn(&LevelTransition) -> u32,
    ) -> Option<&LevelTransition> {
        self.entries
            .last_where(|_, transition| watermark_of(transition) <= id)
            .map(|(_, transition)| transition)
    }
}

#[cfg(feature = "serde")]
impl Serialize for TransitionLog {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.entries
                .to_sorted()
                .into_iter()
                .map(|(_, transition)| transition),
        )
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for TransitionLog {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut log = Self::default();
        for transition in Vec::<LevelTransition>::deserialize(deserializer)? {
            log.push(transition);
        }
        Ok(log)
    }
}

#[derive(Debug, Clone, Copy)]
enum VarClass {
    Type,
    Dim,
    Rank,
}

/// Linear token proving which child level must be left next.
///
///
/// The token is deliberately neither `Copy` nor `Clone`: every successful.
///
/// enter must have exactly one matching leave, and leaves must be LIFO.
///
#[derive(Debug)]
pub struct LevelToken {
    parent_level: u32,
    child_level: u32,
}

/// Type error produced during unification.
///
#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    /// A checked family restriction failed, including before its operand.
    ///
    /// got a concrete type. Rendered publicly as PrecisionMismatch.
    ///
    DtypeFamilyMismatch,
    /// [04-LIN-10]: a function's type parameter was instantiated at a.
    ///
    /// key-carrying type. Rendered publicly as KeyReuse, the key-affinity.
    ///
    /// diagnostic. `value_binding` names the generic when it is a value.
    ///
    /// binding rather than a function, such as a generalized `let` binding,.
    ///
    /// whose repair is an ascription rather than a concrete key parameter.
    ///
    KeyInstantiation {
        value_binding: Option<String>,
    },
    DimensionMismatch,
    ArityMismatch,
    OccursCheck,
    NotAFunction,
}

/// [04-LIN-10]: the generic a key-free type variable stands for a parameter.
///
/// of, so the diagnostic names it. Both parts are optional: a variable.
///
/// generalized before its binding is named has no generic yet, and a.
///
/// parameter the program never spelled has no binder name, and neither is.
///
/// invented ([04-FIT-10]).
///
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GenericParameter {
    /// The definition or `let` binding whose scheme quantified the variable.
    ///
    pub generic: Option<String>,
    /// The authored binder the variable instantiates, when there is one.
    ///
    pub binder: Option<String>,
    /// Whether `generic` is a value binding, one whose type is not a function.
    ///
    /// type, such as `e = Nil`. Its repair ascribes the binding a type with no.
    ///
    /// type parameter, or writes the value where it is used.
    ///
    pub value: bool,
}

impl GenericParameter {
    fn describe(&self) -> String {
        match (&self.generic, &self.binder) {
            (Some(generic), Some(binder)) => {
                format!("type parameter `{binder}` of generic `{generic}`")
            }
            (Some(generic), None) => format!("an inferred type parameter of generic `{generic}`"),
            (None, Some(binder)) => format!("type parameter `{binder}` of a generic function"),
            (None, None) => "a type parameter of a generic function".to_string(),
        }
    }
}

/// A derived observation, never a replacement dimension for propagation.
///
/// Labels, known constraints, and current authored rigidity are independent.
///
/// Deliberately has no equality or conversion to `Dim`: each operation chooses.
///
/// its own proof/admission policy, and keeps the original ID in surviving axes.
///
pub struct DimObservation {
    constraint: Dim,
    name: Option<String>,
    protected: bool,
}

impl DimObservation {
    pub fn known_extent(&self) -> Option<i64> {
        match self.constraint {
            Dim::Lit(n) => Some(n),
            _ => None,
        }
    }

    /// The literal category of policies that admit named runtime claims.
    ///
    /// (expand/reshape/concat), not every internally known constraint.
    ///
    pub fn literal_extent(&self) -> Option<i64> {
        if self.name.is_none() {
            self.known_extent()
        } else {
            None
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn variable(&self) -> Option<DimVar> {
        match self.constraint {
            Dim::Var(v) => Some(v),
            _ => None,
        }
    }
    pub fn rank(&self) -> Option<RankVar> {
        match self.constraint {
            Dim::Rank(v) => Some(v),
            _ => None,
        }
    }
    pub fn is_wildcard(&self) -> bool {
        matches!(self.constraint, Dim::Wildcard)
    }
    pub fn is_protected(&self) -> bool {
        self.protected
    }
}

/// Type, dimension, and rank substitutions with deferred solver obligations.
///
/// Logical binding tables use persistent storage for speculative clones.
/// Mutable stores remain independent, while clones share immutable ground children.
/// Private mutexes protect transient compression through an immutable receiver.
/// Use [`Subst::insert_type`] and [`Subst::insert_dim`] for new bindings.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "E: Clone + serde::Serialize",
        deserialize = "E: Clone + serde::Deserialize<'de>"
    ))
)]
pub struct Subst<E> {
    #[cfg_attr(feature = "serde", serde(skip))]
    result_constraints: Store<Vec<crate::model::ResultConstraint<E>>>,
    types: Store<Bindings<TypeVar, Type<E>>>,
    /// Semantic domains attached to unresolved type variables. This is.
    ///
    /// serialized with reusable checking contexts: a constrained function.
    ///
    /// value must not become unconstrained after a cache round trip.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    tvar_restrictions: Store<PersistentMap<TypeVar, TypeVarRestriction>>,
    /// [04-LIN-10]: the type variables that stand for a function's type.
    ///
    /// parameter, each with the generic it belongs to, so a violation names.
    ///
    /// it. Generalization marks every variable it quantifies, instantiation.
    ///
    /// copies the mark to the fresh variable, and binding a marked variable.
    ///
    /// refuses a key-carrying type and marks every variable left inside the.
    ///
    /// type it is bound to.
    ///
    ///
    /// Kept apart from [`Self::tvar_restrictions`] on purpose: those are dtype.
    ///
    /// families that many sites read as "this variable is a dtype", which a.
    ///
    /// variable that may still become a tuple, a list or a function is not.
    ///
    /// Serialized, so a library's generics keep the rule across a context.
    ///
    /// round trip.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    key_free_tvars: Store<PersistentMap<TypeVar, GenericParameter>>,
    /// Spec/04 section 8.4.1: the data types of this check whose fields carry.
    ///
    /// a key, installed once every `type` declaration is registered and read.
    ///
    /// when a [`Self::key_free_tvars`] variable is bound to a data type.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    key_carrying_adts: PersistentSet<String>,
    dims: Store<Bindings<DimVar, Dim>>,
    /// Named-axis view of a checker dimension identity, plus reserved.
    ///
    /// compiler-owned annotations on synthetic dimension identities. Schemes.
    ///
    /// retain these IDs and instantiation copies their annotations to fresh.
    ///
    /// IDs. This mandatory snapshot field is not evidence that arbitrary.
    ///
    /// equal names have equal sizes.
    ///
    dimension_labels: Store<PersistentMap<DimVar, String>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    protected_dimensions: PersistentSet<DimVar>,
    /// chelis#1801: the alias CLASSES that met a runtime extent `*`.
    ///
    /// without being bound to it, keyed by the class root.
    ///
    ///
    /// `unify_dim`'s wildcard arm deliberately leaves such a variable free.
    ///
    /// (the #143 invariant documented on that function), so a later concrete.
    ///
    /// argument in the same signature can still constrain it. That leaves no.
    ///
    /// record of the meeting at all, and `spec/04-type-system.md` section 3.2.
    ///
    /// needs one: a variable an application's instantiation minted, that met.
    ///
    /// `*` and that no argument bound, denotes that runtime extent.
    ///
    ///
    /// Why a property of the CLASS rather than of one variable. Unification.
    ///
    /// identifies variables, and which member roots a class changes while the.
    ///
    /// call unifies: `unify_dim` resolves both operands through.
    ///
    /// `constraint_dim` before it matches, so the meeting is recorded on.
    ///
    /// whatever rooted the class at that moment, and a later `bind_dvar` in.
    ///
    /// the same call can re-root the class over it. Asking one member, or.
    ///
    /// two, then misses a meeting recorded on a third. chelis#1925's rounds 1.
    ///
    /// and 2 were both exactly that miss, one alias member apart. The flag.
    ///
    /// therefore lives on the root and is OR-merged onto the new root at.
    ///
    /// every merge ([`Subst::merge_dvar_class_flags`]), and the absorbing.
    ///
    /// site asks the class once.
    ///
    ///
    /// Monotone within a pass and never cleared: dimension variables are.
    ///
    /// minted fresh and never reused, so an entry cannot become stale. Not.
    ///
    /// serialized: transient per-pass bookkeeping, like the deferred ledgers.
    ///
    /// below.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    wildcard_touched_classes: PersistentSet<DimVar>,
    /// chelis#1801: the alias classes a DECLARED dimension of the definition.
    ///
    /// under check pins, keyed and merged exactly like the set above.
    ///
    ///
    /// This is half of section 3.2's own exclusion rather than an extra.
    ///
    /// guard: a literal or name that an argument of the same application.
    ///
    /// binds to the class is a CLAIM on the runtime extent, checked by a.
    ///
    /// section 4.7 guard, and absorbing the class to `*` would erase the.
    ///
    /// claim. The other half needs no ledger. A name or literal the class is.
    ///
    /// BOUND to is reported by `constraint_dim`, which the absorbing site.
    ///
    /// asks first. what `constraint_dim` cannot report is a declared binder,.
    ///
    /// because `bind_dvar`'s name arm records it as a label and leaves the.
    ///
    /// class unbound. `protected_dimensions` holds those binders, and this.
    ///
    /// set is what keeps that answer stable when a merge or a compose demotes.
    ///
    /// the protected member out of the root position.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    binder_pinned_classes: PersistentSet<DimVar>,
    #[cfg_attr(feature = "serde", serde(skip))]
    label_unification_depth: usize,
    /// Only labels written by the active outer unification need rollback.
    ///
    /// Interior mutability also covers scheme instantiation through `&Subst`.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    dimension_label_undo: Store<Vec<(DimVar, Option<String>)>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    refined_type_bindings: Vec<(TypeVar, Type<E>)>,
    /// Rank-variable bindings: a `RankVar` binds to the *entire* shape vector.
    ///
    /// it stands for (Tier-2 rank polymorphism). A binding to `[Dim::Rank(r2)]`.
    ///
    /// is a rank-to-rank alias resolved transitively by `resolve_rvar`.
    ///
    ranks: Store<Bindings<RankVar, Vec<Dim>>>,
    /// Issue #256 soundness ledger. The `borrow` inference arm accepts a.
    ///
    /// borrow whose inner type is still an unresolved `Type::Var`,.
    ///
    /// deferring the tensor-or-carrier classification to next.
    ///
    /// unification (the surrounding `&tensor[..]` parameter pins it).
    ///
    /// That deferral is only sound when the variable is *eventually*.
    ///
    /// pinned to a tensor or tensor-carrying type. When the consumer is.
    ///
    /// itself fully polymorphic (e.g. `consume_any[a](t: a)`), the.
    ///
    /// variable is never pinned and a genuinely-non-tensor value would.
    ///
    /// slip past every gate. Each deferred borrow records the inner.
    ///
    /// `TypeVar` here. after a def body's inference completes, the.
    ///
    /// driver resolves each one against the now-complete substitution.
    ///
    /// and rejects any that did not become a tensor or tensor carrier.
    ///
    /// Not serialized: this is transient per-pass bookkeeping, drained.
    ///
    /// by the inference driver, and never part of a persisted context.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    deferred_borrow_vars: Store<Vec<TypeVar>>,
    /// chelis#3180, mirroring `deferred_borrow_vars`: `drop` sites whose.
    ///
    /// operand type was still an unresolved `Type::Var` when the call was.
    ///
    /// inferred (an unannotated lambda parameter pinned only by a later.
    ///
    /// call). The driver drains these per declaration and refuses any that.
    ///
    /// resolved to a borrow, since [05-OP-67] makes the operand owned. Not.
    ///
    /// serialized: transient per-pass bookkeeping.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    deferred_drop_operands: Store<Vec<(TypeVar, Option<crate::decisions::SourceLabel>)>>,
    /// RFC D-CHECK deferred-access ledger, mirroring.
    ///
    /// `deferred_borrow_vars`: `access`/`record-update` sites whose.
    ///
    /// target type was still an unresolved `Type::Var` when inference.
    ///
    /// ran (e.g. an unannotated lambda parameter pinned only by a.
    ///
    /// later call). The driver drains these per def and re-checks each.
    ///
    /// against the final substitution so a target pinned to an opaque.
    ///
    /// ADT defined in another module is still rejected. Not.
    ///
    /// serialized: transient per-pass bookkeeping.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    deferred_opaque_uses: Store<Vec<(TypeVar, DeferredOpaqueUse)>>,
    /// chelis#1489 suspended-operand ledger. A set of checked sites rejected.
    ///
    /// an unresolved `Type::Var` outright, which made them sensitive to WHEN.
    ///
    /// inference resolved a variable rather than to whether the program was.
    ///
    /// well typed: 0.18.6 changed that timing and they fired ~50x more often.
    ///
    /// on an unchanged corpus.
    ///
    ///
    /// What suspends here is `copy`, `cast`, and the ten csv host-lane slots.
    ///
    /// — ~97% of the measured occurrences. `round_to` is NOT among them: see.
    ///
    /// `unify_host_slot_eager`.
    ///
    ///
    /// `gather`, `scatter`, `scatter_replace` and `trace` record too, as.
    ///
    /// [`DeferredOperandGate::ShapeRoute`]. Their results are shape functions of.
    ///
    /// the operand, so that variant carries the arm's other evidence -- the RAW.
    ///
    /// axis, the indices, and `scatter`'s updates operand -- and discharge.
    ///
    /// replays the whole arm rather than the one helper call it appears to make.
    ///
    ///
    /// `concat` and `diagonal` do NOT record. They compute a new extent.
    ///
    /// arithmetically (a sum, a minimum), their helpers encode a dim they.
    ///
    /// cannot compute with as `Dim::Wildcard`, and a dim can still be a.
    ///
    /// variable when the operand's TYPE variable binds -- so deferring them let.
    ///
    /// a false declared extent check, build and run. They stay on their eager.
    ///
    /// paths. `the_arithmetic_routes_are_not_converted` pins the exclusion.
    ///
    ///
    /// Recording rather than tolerating is deliberate. The ~71 sibling gates.
    ///
    /// return on an unresolved operand and forget it, which is sound for them.
    ///
    /// but not here: measured, a tolerant `cast` lets.
    ///
    /// `def go[t](x: t) -> i32 = cast(x, i32)` check at 1.0 and BUILD,.
    ///
    /// with the backend choosing a dtype for the never-resolved `t`. A.
    ///
    /// variable that is never bound is still rejected.
    ///
    ///
    /// The `Copy` and `Cast` entries carry the result variable their call.
    ///
    /// returned, and discharge unifies the eager arm's own answer into it.
    ///
    /// `HostSlot` carries no result and discharge unifies its expected type.
    ///
    /// against the operand instead. Both are load-bearing: a.
    ///
    /// revision that returned the OPERAND's variable made a deferred.
    ///
    /// `copy(&t)` type as `&tensor` where an eager one is `tensor`, so the.
    ///
    /// expression's type depended on when the operand resolved — the very.
    ///
    /// sensitivity this ledger exists to delete.
    ///
    ///
    /// Not serialized: transient per-pass bookkeeping.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    deferred_tensor_operands: Store<Vec<(TypeVar, DeferredOperandGate<E>)>>,
    /// Shared children known to be ground, which applying returns without.
    ///
    /// walking them. Shared by every copy of this substitution and not.
    ///
    /// serialized. see [`GroundTypes`].
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    ground_types: GroundTypes<E>,
    /// Verdicts from suspended operand constraints that discharged badly.
    ///
    /// (chelis#1489).
    ///
    ///
    /// Discharge happens inside unification, which has no `DiagnosticSink`, so.
    ///
    /// the failure is recorded here and rendered by the per-def reporting pass.
    ///
    /// Only failures land here. a constraint that discharges cleanly leaves no.
    ///
    /// trace beyond the unification it doed.
    ///
    ///
    /// Not serialized: transient per-pass bookkeeping.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    operand_gate_failures: Store<Vec<OperandGateFailure<E>>>,
    /// chelis#1654: checked collection-operation contracts instantiated during.
    ///
    /// this inference pass.
    ///
    ///
    /// Each scheme instantiation gets an opaque ID and its lexical inference.
    ///
    /// level. An application captures the exact IDs its callee produced,.
    ///
    /// attaches that call's evidence, and discharges only those instances after.
    ///
    /// argument unification. A relation merely carried through an alias,.
    ///
    /// aggregate, or higher-order result remains transportable until.
    ///
    /// generalization moves it back onto the resulting scheme.
    ///
    ///
    /// This is the single lifecycle owner for both variable-bearing and fully.
    ///
    /// monomorphic relations. In particular, no settled side queue can decide.
    ///
    /// tensor concat before an application supplies its axis and element.
    ///
    /// extents. Not serialized: schemes are the durable representation.
    ///
    #[cfg_attr(feature = "serde", serde(skip))]
    collection_contracts: Store<Vec<CollectionContractInstance<E>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    next_collection_contract_id: Store<u64>,
    /// Current lexical generalization level. Serialized because a cloned.
    ///
    /// checking context must preserve in-flight transactional state.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    current_level: u32,
    /// Ordered enter/leave/resume watermarks used to recover mint levels.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    level_transitions: TransitionLog,
    /// Sparse overrides for variables unified into an older scope.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    lowered_tvar_levels: PersistentMap<TypeVar, u32>,
    #[cfg_attr(feature = "serde", serde(default))]
    lowered_dvar_levels: PersistentMap<DimVar, u32>,
    #[cfg_attr(feature = "serde", serde(default))]
    lowered_rvar_levels: PersistentMap<RankVar, u32>,
    /// IDs below these floors came from an earlier persisted check and are.
    ///
    /// always level zero in the resumed check.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    resume_floors: VarWatermarks,
}

/// A suspended operand constraint that discharged to a rejection.
///
/// (chelis#1489).
///
///
/// Carries the data the diagnostic needs and none of the rendering: discharge.
///
/// runs inside unification, and the per-def reporting pass is what turns this.
///
/// into a `CheckError` (it is also what holds the declared type-parameter.
///
/// names the message may want).
///
#[derive(Debug, Clone)]
pub enum OperandGateFailure<E> {
    /// The operand settled to something the gate does not accept.
    ///
    Rejected {
        gate: DeferredOperandGate<E>,
        resolved: Type<E>,
    },
    /// The gate accepted the operand, but the result the call had already.
    ///
    /// handed its consumer cannot be the result the settled operand produces.
    ///
    ResultMismatch {
        gate: DeferredOperandGate<E>,
        expected: Type<E>,
        settled: Type<E>,
    },
    /// The gate's own decision function rejected the settled operand with a.
    ///
    /// specific error -- an unsupported cast precision, say -- rather than a.
    ///
    /// generic "wrong shape of type".
    ///
    Decision {
        error: crate::decisions::DecisionFailure,
    },
}

/// Which operand constraint suspended its decision (chelis#1489), and what it.
///
/// needs to decide once the operand is bound.
///
///
/// Not `Copy`: the host-slot arm carries the expected type and the producer's.
///
/// own slot wording, so discharge can re-decide without re-running the call.
///
#[derive(Debug, Clone, PartialEq)]
pub enum DeferredOperandGate<E> {
    /// `copy` with an unresolved operand. Carries the result variable the call.
    ///
    /// returned: `copy(&t)` yields `t`, not `&t`, so discharge must unify the.
    ///
    /// eager arm's own answer into it rather than let the operand's type stand.
    ///
    Copy {
        result: Box<Type<E>>,
        location: Option<crate::decisions::SourceLabel>,
    },
    /// `cast`/`cast_trunc` whose SOURCE was unresolved. Carries the target.
    ///
    /// precision, the mode, and the result variable the call returned, which.
    ///
    /// discharge unifies against once `cast`'s own decision function is called.
    ///
    /// with the settled source type.
    ///
    ///
    /// `borrowed` records that the source was `&v` over the unresolved `v`.
    ///
    /// (chelis#3101). Discharge then decides the type the eager `borrow` arm.
    ///
    /// gives `&settled` ([`crate::decisions::borrow::settled_borrow_type`]), as the.
    ///
    /// eager cast would see it.
    ///
    Cast {
        target: chelis_numeric_core::prim::Prim,
        mode: chelis_deep::CastMode,
        borrowed: bool,
        result: Box<Type<E>>,
        location: Option<crate::decisions::SourceLabel>,
    },
    /// chelis#2534: a checked `cast` whose TARGET is a declaration's dtype.
    ///
    /// binder and whose SOURCE was unresolved. Carries the target binder and.
    ///
    /// the result variable the call returned. discharge decides the settled.
    ///
    /// source with the same function the eager arm calls.
    ///
    CastToBinder {
        target: crate::model::TypeVar,
        result: Box<Type<E>>,
        location: Option<crate::decisions::SourceLabel>,
    },
    /// A host-lane slot that unifies against a fixed expected type:.
    ///
    /// the ten csv routes, which funnel through `unify_host_slot`. Carries.
    ///
    /// what the slot expected so discharge can re-decide without re-running.
    ///
    /// the call.
    ///
    ///
    /// `round_to` deliberately does NOT reach here. This variant carries ONE.
    ///
    /// expected type, and `round_to` accepts more than one. routing it here.
    ///
    /// made a later-bound operand reject against the single type this carries.
    ///
    /// It is on `unify_host_slot_eager` instead.
    ///
    HostSlot {
        fname: String,
        description: String,
        expected: Box<Type<E>>,
    },
    /// A shape-computing route whose result is a function of the operand's.
    ///
    /// shape: `gather`, `scatter`, `scatter_replace` and `trace` (chelis#1489.
    ///
    /// gate 3). `concat` and `diagonal` are deliberately excluded. see the.
    ///
    /// deferred-operand ledger's doc.
    ///
    ///
    /// The payload is the WHOLE call minus its operand, and.
    ///
    /// `infer::shape_route_result` is the one function that decides it. The.
    ///
    /// eager arm builds this same payload and calls that same function with a.
    ///
    /// resolved operand, so there is no second implementation of any route --.
    ///
    /// this is a deferral of one decision, not a copy of the arm's tail.
    ///
    ShapeRoute {
        route: ShapeRoute<E>,
        result: Box<Type<E>>,
    },
    /// A `sum`-family result (`sum`, `cumsum`, `trace`, `einsum`) over a.
    ///
    /// tensor whose precision is still an inference variable whose dtypes.
    ///
    /// have no single `sum_result(p, default(p))` (spec/04 §5.7.1). Keyed on.
    ///
    /// that precision variable. When it binds, discharge decides the result.
    ///
    /// precision with the same function the eager arm calls and unifies the.
    ///
    /// result tensor, at `dims`, into the type the call published. A variable.
    ///
    /// still unbound at the declaration boundary is decided over its bound.
    ///
    /// there ([`Subst::decide_pending_sum_results`]).
    ///
    SumResult {
        op: String,
        dims: Vec<crate::model::Dim>,
        result: Box<Type<E>>,
        location: Option<crate::decisions::SourceLabel>,
    },
}

pub type CollectionContractId = u64;

#[derive(Debug, Clone)]
struct CollectionContractInstance<E> {
    id: CollectionContractId,
    level: u32,
    constraint: CollectionConstraint<E>,
    state: CollectionContractState,
}

#[derive(Debug, Clone)]
enum CollectionContractState {
    Transport,
    Consumed {
        evidence: Option<crate::decisions::collection::CollectionCallEvidence>,
    },
}

impl<E: Clone + PartialEq> ShapeRoute<E> {
    /// Whether this route's arm normalizes one borrow off its operand.
    ///
    ///
    /// `gather` and `trace` read their operand through `type_for_readonly_check`,.
    ///
    /// which strips one `Type::Ref`. `scatter` and `scatter_replace` pass.
    ///
    /// `subst.apply` and reject a borrowed operand. Stripping unconditionally in.
    ///
    /// the shared decision widened those in the ACCEPTING direction -- and, when.
    ///
    /// `concat` still went through here, a borrowed `concat` passed its element.
    ///
    /// guard with empty per-element dims and made the concat axis a wildcard, so.
    ///
    /// any declared extent was admitted.
    ///
    ///
    /// The strip is a property of the route, so it is recorded as one.
    ///
    pub fn normalizes_borrow(&self) -> bool {
        match self {
            Self::Gather { op, .. } => op == "gather",
            Self::Trace { .. } => true,
        }
    }

    /// The builtin's name, for the diagnostic.
    ///
    pub fn op(&self) -> &str {
        match self {
            Self::Gather { op, .. } => op,
            Self::Trace { .. } => "trace",
        }
    }
}

/// A shape-computing call with its operand left out (chelis#1489).
///
///
/// Carries every input the decision needs EXCEPT the operand, so the same.
///
/// value decides the call eagerly and after a deferral.
///
///
/// Two fields are deliberately raw rather than pre-computed, because.
///
/// pre-computing them needs the operand:.
///
///
/// - `raw_axis` is the axis as written. The eager resolvers normalize a.
///
///   negative axis against the operand's RANK and yield 0 when the operand is.
///
///   not a concrete tensor, so an axis resolved before the operand settled.
///
///   would be wrong. It is normalized inside the decision instead.
///
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeRoute<E> {
    /// `gather`, and `scatter`/`scatter_replace` when `updates` is present.
    ///
    Gather {
        op: String,
        indices: Box<Type<E>>,
        raw_axis: Option<i64>,
        updates: Option<Box<Type<E>>>,
        /// `scatter`'s mode string, exactly as written. Validated inside the.
        ///
        /// decision: validating it in the arm meant the deferred path skipped.
        ///
        /// it and demoted a type error to a runtime abort.
        ///
        mode: Option<String>,
    },
    /// `trace`, which takes an axis PAIR.
    ///
    ///
    /// `diagonal` also takes one and is deliberately NOT here: it computes the.
    ///
    /// minimum of two extents, and when a dim it reads is still a variable at.
    ///
    /// discharge its helper falls back to `Dim::Wildcard`, which the declared.
    ///
    /// signature then narrows to any extent at all. It stays on its eager path.
    ///
    /// until that helper stops encoding "unknown" as permissive. see chelis#1489.
    ///
    Trace {
        raw_axis1: Option<i64>,
        raw_axis2: Option<i64>,
    },
}

impl<E: Clone + PartialEq> DeferredOperandGate<E> {
    /// The result the suspended call handed its consumer, if it handed one.
    ///
    ///
    /// `copy` and `cast` return a fresh variable that discharge unifies with the.
    ///
    /// decided type. a host slot returns its builtin's own fixed result type and.
    ///
    /// carries none. A shape route returns a fresh variable for the same reason.
    ///
    /// `copy` does -- its result is a function of a shape nobody knows yet.
    ///
    ///
    /// Answering this is not optional bookkeeping. `Env::generalize` refuses to.
    ///
    /// quantify anything this returns (chelis#1489, #1832): a result variable.
    ///
    /// that is generalized before its gate discharges gives every use of a.
    ///
    /// `let`-bound name its own instance, and the declared result is then never.
    ///
    /// checked against what the call produces. Returning `None` for a variant.
    ///
    /// that does hand out a fresh variable reopens that hole for that variant.
    ///
    /// The match is deliberately exhaustive with no wildcard arm so a new gate.
    ///
    /// cannot be added without deciding this.
    ///
    fn result(&self) -> Option<&Type<E>> {
        match self {
            Self::Copy { result, .. }
            | Self::Cast { result, .. }
            | Self::CastToBinder { result, .. }
            | Self::ShapeRoute { result, .. }
            | Self::SumResult { result, .. } => Some(result.as_ref()),
            Self::HostSlot { .. } => None,
        }
    }

    pub fn location(&self) -> Option<&crate::decisions::SourceLabel> {
        match self {
            Self::Copy { location, .. }
            | Self::Cast { location, .. }
            | Self::CastToBinder { location, .. }
            | Self::SumResult { location, .. } => location.as_ref(),
            Self::HostSlot { .. } | Self::ShapeRoute { .. } => None,
        }
    }

    fn attach_location(
        &self,
        mut error: crate::decisions::DecisionFailure,
    ) -> crate::decisions::DecisionFailure {
        if error
            .location
            .as_ref()
            .is_none_or(|label| label.span_offset.is_none() && label.span_id.is_none())
        {
            error.location = self.location().cloned();
        }
        error
    }

    pub fn expected_operand(&self) -> String {
        match self {
            Self::Copy { .. } => "an operand of determined type".to_string(),
            Self::ShapeRoute { .. } => "tensor".to_string(),
            Self::Cast { .. } => "tensor or numeric/bool scalar".to_string(),
            Self::CastToBinder { .. } => "numeric or bool scalar".to_string(),
            Self::HostSlot { description, .. } => description.clone(),
            Self::SumResult { .. } => "a dtype with a default sum accumulator".to_string(),
        }
    }

    // A consumer may constrained the provisional result's dtype before.
    //
    // this producer settles. Preserve that family's error. the producer did.
    //
    // not violate its own operand contract (a valid cast can produce an int).
    //
    fn reconcile_result(&self, result: &Type<E>, settled: Type<E>, subst: &mut Subst<E>) {
        if let Err(error) = unify(result, &settled, subst) {
            if matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch) {
                subst.record_operand_gate_failure(OperandGateFailure::Decision {
                    error: error.into(),
                });
            } else {
                let expected = subst.apply(result);
                subst.record_operand_gate_failure(OperandGateFailure::ResultMismatch {
                    gate: self.clone(),
                    expected,
                    settled,
                });
            }
        }
    }

    /// Settle this constraint against the type its operand was just bound to.
    ///
    /// (chelis#1489).
    ///
    ///
    /// Every arm calls the SAME decision function its eager counterpart calls.
    ///
    /// -- `copy_result_from_source`, `cast_result_from_settled_source`, or the.
    ///
    /// slot unification -- so there is no second implementation of any gate to.
    ///
    /// disagree with the first. The unification of `result` is what stops the.
    ///
    /// fresh variable the eager call handed its consumer from staying.
    ///
    /// unconstrained.
    ///
    ///
    /// `resolved` is never a `Type::Var`: the caller only discharges a bound.
    ///
    /// variable, and re-aliases instead when a variable was bound to another.
    ///
    fn discharge(self, resolved: &Type<E>, subst: &mut Subst<E>) {
        match self {
            Self::Copy { ref result, .. } => {
                let settled = crate::decisions::borrow::copy_result_from_source(resolved);
                self.reconcile_result(result, settled, subst);
            }
            Self::Cast {
                target,
                mode,
                borrowed,
                ref result,
                ..
            } => {
                let source = if borrowed {
                    match crate::decisions::borrow::settled_borrow_type(resolved.clone()) {
                        Some(source) => source,
                        // The eager `borrow` arm refuses this operand and the.
                        //
                        // eager cast then sees an error type and adds nothing.
                        //
                        // The deferred borrow's own recheck.
                        //
                        // (`validate_deferred_borrow_vars`) reports that.
                        //
                        // refusal here, so the cast adds nothing either.
                        //
                        None => return,
                    }
                } else {
                    resolved.clone()
                };
                match crate::decisions::cast::cast_result_from_settled_source(
                    source, target, mode, subst,
                ) {
                    Ok(settled) => {
                        self.reconcile_result(result, settled, subst);
                    }
                    Err(error) => subst.record_operand_gate_failure(OperandGateFailure::Decision {
                        error: self.attach_location(*error),
                    }),
                }
            }
            Self::CastToBinder {
                target, ref result, ..
            } => {
                match crate::decisions::cast::binder_cast_result_from_settled_source(
                    resolved.clone(),
                    target,
                    subst,
                ) {
                    Ok(settled) => {
                        self.reconcile_result(result, settled, subst);
                    }
                    Err(error) => subst.record_operand_gate_failure(OperandGateFailure::Decision {
                        error: self.attach_location(*error),
                    }),
                }
            }
            Self::HostSlot { ref expected, .. } => {
                if unify(expected.as_ref(), resolved, subst).is_err() {
                    subst.record_operand_gate_failure(OperandGateFailure::Rejected {
                        gate: self.clone(),
                        resolved: resolved.clone(),
                    });
                }
            }
            Self::ShapeRoute {
                ref route,
                ref result,
            } => {
                match crate::decisions::shape::shape_route_result(
                    route,
                    resolved,
                    crate::decisions::shape::SumResultSlot::Existing(result),
                    None,
                    subst,
                ) {
                    Ok((settled, updates)) => {
                        // The updates equation `scatter` imposes is part of the.
                        //
                        // arm, not of the helper. Replaying only the helper.
                        //
                        // would drop it.
                        //
                        if let Some(expected_updates) = updates
                            && let ShapeRoute::Gather {
                                updates: Some(actual),
                                ..
                            } = route
                            && let Err(te) = unify(&expected_updates, actual.as_ref(), subst)
                        {
                            // Record the unification error itself, not a generic.
                            //
                            // result mismatch. The eager arm reports `te.into()`,.
                            //
                            // which carries `DimensionMismatch`. collapsing that.
                            //
                            // onto `TypeMismatch` would change a published.
                            //
                            // diagnostic kind by inference order -- the wire.
                            //
                            // change `kind()`'s own doc forbids.
                            //
                            subst.record_operand_gate_failure(OperandGateFailure::Decision {
                                error: te.into(),
                            });
                            return;
                        }
                        self.reconcile_result(result, settled, subst);
                    }
                    Err(message) => {
                        subst.record_operand_gate_failure(OperandGateFailure::Decision {
                            error: crate::decisions::DecisionFailure::new(
                                crate::decisions::DecisionFailureKind::TypeMismatch,
                                message,
                                Vec::new(),
                            ),
                        })
                    }
                }
            }
            Self::SumResult {
                ref op,
                ref dims,
                ref result,
                ..
            } => match crate::decisions::shape::settled_sum_result_precision(op, resolved) {
                Ok(precision) => {
                    self.reconcile_result(
                        result,
                        Type::Tensor(dims.clone(), TensorPrec::Concrete(precision)),
                        subst,
                    );
                }
                Err(message) => subst.record_operand_gate_failure(OperandGateFailure::Decision {
                    error: self.attach_location(crate::decisions::DecisionFailure::new(
                        crate::decisions::DecisionFailureKind::TypeMismatch,
                        message,
                        Vec::new(),
                    )),
                }),
            },
        }
    }

    /// The rejection this gate emits once the operand is known to be wrong.
    ///
    ///
    /// The wording each gate used when it decided eagerly, with two.
    ///
    /// exceptions. The eager host-slot rejection appends macro provenance ("in.
    ///
    /// expansion of ...") from the node, and discharge has no node, so a.
    ///
    /// host-slot rejection inside a macro expansion loses that suffix. And.
    ///
    /// `subject_for` substitutes a backticked declared type-parameter name.
    ///
    /// (``got `t` ``) where the eager arm printed the internal identity --.
    ///
    /// that one is [04-FIT-9] and is asserted by.
    ///
    /// `a_never_resolved_declared_parameter_is_named_not_numbered`.
    ///
    pub fn message(&self, subject: &str) -> String {
        match self {
            // `copy` accepts every settled operand, so its one rejection is.
            //
            // the operand that never settles: whether it is a borrow decides.
            //
            // the result.
            //
            Self::Copy { .. } => format!(
                "copy requires its operand's type to be determined at the call, because \
                 whether it is a borrow decides the result (spec/04-type-system.md \
                 section 8.2), got {subject}"
            ),
            Self::Cast { .. } => format!("cast requires tensor or prim type, got {subject}"),
            Self::CastToBinder { .. } => format!(
                "cast to a quantified scalar dtype requires a numeric or bool scalar, got {subject}"
            ),
            Self::HostSlot {
                fname, description, ..
            } => format!("{fname} expects {description}, got {subject}"),
            Self::ShapeRoute { route, .. } => {
                format!("{} expects tensor input, got {subject}", route.op())
            }
            Self::SumResult { op, .. } => format!(
                "{op} has no single sum_result(p, default(p)) (spec/04-type-system.md \
                 section 5.7.1) for precision {subject}"
            ),
        }
    }

    /// What to call this constraint's call in a diagnostic.
    ///
    pub fn noun(&self) -> &str {
        match self {
            Self::Copy { .. } => "copy",
            Self::Cast { .. } | Self::CastToBinder { .. } => "cast",
            Self::HostSlot { fname, .. } => fname,
            Self::ShapeRoute { route, .. } => route.op(),
            Self::SumResult { op, .. } => op,
        }
    }

    /// The diagnostic kind the gate emitted when it decided eagerly.
    ///
    ///
    /// Deferring must not change the kind: a published vocabulary identity is.
    ///
    /// something consumers count by name (chelis#1334 tallies these), so.
    ///
    /// collapsing gates onto a different kind would be a wire change smuggled.
    ///
    /// in behind a timing fix.
    ///
    pub fn kind(&self) -> crate::decisions::DecisionFailureKind {
        use crate::decisions::DecisionFailureKind as Kind;
        match self {
            Self::Cast { .. } | Self::CastToBinder { .. } => Kind::CastNonTensor,
            Self::Copy { .. }
            | Self::HostSlot { .. }
            | Self::ShapeRoute { .. }
            | Self::SumResult { .. } => Kind::TypeMismatch,
        }
    }

    /// The repair hint, where the eager path carried one.
    ///
    pub fn suggestions(&self) -> Vec<String> {
        match self {
            Self::Copy { .. } => {
                vec!["Declare the copied value's type, as `T` or as the borrow `&T`".to_string()]
            }
            _ => Vec::new(),
        }
    }
}

/// Decide one consumed collection contract and reconcile its result.
///
///
/// `Unresolved` leaves the owning contract instance live until the current.
///
/// application or declaration boundary supplies the missing evidence. Every.
///
/// settled verdict removes the instance before entering this function, so.
///
/// result unification cannot rediscover or cross-contaminate another call.
///
enum CollectionDischarge<E> {
    Unresolved,
    Settled(Option<Type<E>>),
}

fn discharge_collection_constraint<E: Clone + PartialEq>(
    constraint: &crate::model::CollectionConstraint<E>,
    evidence: Option<&crate::decisions::collection::CollectionCallEvidence>,
    subst: &mut Subst<E>,
) -> CollectionDischarge<E> {
    match crate::decisions::collection::decide_collection_constraint(constraint, evidence, subst) {
        Ok(None) => CollectionDischarge::Unresolved,
        Ok(Some(decision)) => {
            let produced = decision.produced_result();
            match decision.publish(subst) {
                Ok(()) => CollectionDischarge::Settled(Some(
                    produced.unwrap_or_else(|| constraint.result().clone()),
                )),
                Err(error) => {
                    subst.record_operand_gate_failure(OperandGateFailure::Decision {
                        error: error.into(),
                    });
                    CollectionDischarge::Settled(None)
                }
            }
        }
        Err(message) => {
            subst.record_operand_gate_failure(OperandGateFailure::Decision {
                error: crate::decisions::DecisionFailure::new(
                    crate::decisions::DecisionFailureKind::TypeMismatch,
                    message,
                    Vec::new(),
                ),
            });
            CollectionDischarge::Settled(None)
        }
    }
}

/// Whether `ty` still contains the function value governed by `constraint`.
///
///
/// The relation's fixed operand/result fields reconstruct its callable type.
///
/// Exact structural containment works for fully monomorphic contracts, while.
///
/// the ordinary shared inference variables make the same test true for.
///
/// polymorphic aliases and higher-order results after substitution.
///
pub fn collection_contract_visible_in_type<E: Clone + PartialEq>(
    constraint: &CollectionConstraint<E>,
    ty: &Type<E>,
) -> bool {
    let callable = collection_contract_callable_type(constraint);
    fn contains<E: Clone + PartialEq>(ty: &Type<E>, callable: &Type<E>) -> bool {
        if ty == callable {
            return true;
        }
        match ty {
            Type::Fn(parameters, result) => {
                parameters
                    .iter()
                    .any(|parameter| contains(parameter, callable))
                    || contains(result, callable)
            }
            Type::Ref(inner) => contains(inner, callable),
            Type::Adt(_, arguments) | Type::Tuple(arguments) => arguments
                .iter()
                .any(|argument| contains(argument, callable)),
            Type::KindedAdt(_, arguments) => arguments
                .iter()
                .any(|argument| matches!(argument, NominalArg::Type(ty) if contains(ty, callable))),
            Type::Var(_) | Type::Prim(_) | Type::Tensor(_, _) | Type::Unit | Type::Error(_) => {
                false
            }
        }
    }
    contains(ty, &callable)
}

pub fn collection_contract_callable_type<E: Clone + PartialEq>(
    constraint: &CollectionConstraint<E>,
) -> Type<E> {
    if let CollectionConstraint::FloatClassify {
        operand, result, ..
    } = constraint
    {
        return Type::function(vec![Type::reference(operand.clone())], result.clone());
    }
    Type::Fn(
        constraint.operands().into_iter().cloned().collect(),
        Arc::new(constraint.result().clone()),
    )
}

fn collect_non_callable_tvars<E: Clone + PartialEq>(ty: &Type<E>, vars: &mut UnordSet<TypeVar>) {
    match ty {
        Type::Var(var) => {
            vars.insert(*var);
        }
        Type::Fn(_, _) => {}
        Type::Ref(inner) => collect_non_callable_tvars(inner, vars),
        Type::Tensor(_, TensorPrec::Var(var)) => {
            vars.insert(*var);
        }
        Type::Adt(_, arguments) | Type::Tuple(arguments) => {
            for argument in arguments.iter() {
                collect_non_callable_tvars(argument, vars);
            }
        }
        Type::KindedAdt(_, arguments) => {
            for argument in arguments.iter() {
                if let NominalArg::Type(ty) = argument {
                    collect_non_callable_tvars(ty, vars);
                }
            }
        }
        Type::Prim(_) | Type::Tensor(_, TensorPrec::Concrete(_)) | Type::Unit | Type::Error(_) => {}
    }
}

/// Whether an unresolved transported function argument became tied to a.
///
/// non-callable argument of this application. This is the higher-order.
///
/// `invoke(len, x)` edge: `identity(len)` returns the callable and is handled.
///
/// by `collection_contract_visible_in_type`, while `ignore(len, x)` leaves no.
///
/// shared variable and may discard the unused capability.
///
fn collection_contract_tied_to_arguments<E: Clone + PartialEq>(
    constraint: &CollectionConstraint<E>,
    arguments: &[Type<E>],
    subst: &Subst<E>,
) -> bool {
    let mut argument_vars = UnordSet::default();
    for argument in arguments {
        collect_non_callable_tvars(&subst.apply(argument), &mut argument_vars);
    }
    constraint.operands().into_iter().any(|operand| {
        crate::scheme::free_tvars(&subst.apply(operand))
            .into_iter()
            .any(|variable| argument_vars.contains(&variable))
    })
}

/// Which deferred use shape registered a ledger entry (determines the.
///
/// violation action text at validation time).
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredOpaqueUse {
    /// `(access target field)` with an unresolved target.
    ///
    Access,
    /// `(record-update target kv...)` with an unresolved target.
    ///
    RecordUpdate,
}

/// An ordinary map holding the same entries as `map`, for a snapshot that.
///
/// leaves the substitution.
///
fn sorted_copy<K: Ord + Clone, V: Clone>(map: &PersistentMap<K, V>) -> UnordMap<K, V> {
    map.to_sorted()
        .into_iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

impl<E: Clone> Clone for Subst<E> {
    fn clone(&self) -> Self {
        self.dimension_label_undo.with(|_| ());
        self.ground_types.0.with(|_| ());
        Subst {
            result_constraints: Store::new(self.result_constraints.with(|value| value.clone())),
            types: Store::new(self.types.with(|value| value.clone())),
            tvar_restrictions: Store::new(self.tvar_restrictions.with(|value| value.clone())),
            key_free_tvars: Store::new(self.key_free_tvars.with(|value| value.clone())),
            key_carrying_adts: self.key_carrying_adts.clone(),
            dims: Store::new(self.dims.with(|value| value.clone())),
            dimension_labels: Store::new(self.dimension_labels.with(|value| value.clone())),
            protected_dimensions: self.protected_dimensions.clone(),
            wildcard_touched_classes: self.wildcard_touched_classes.clone(),
            binder_pinned_classes: self.binder_pinned_classes.clone(),
            label_unification_depth: 0,
            dimension_label_undo: Store::new(Vec::new()),
            refined_type_bindings: Vec::new(),
            ranks: Store::new(self.ranks.with(|value| value.clone())),
            deferred_borrow_vars: Store::new(self.deferred_borrow_vars.with(|value| value.clone())),
            deferred_drop_operands: Store::new(
                self.deferred_drop_operands.with(|value| value.clone()),
            ),
            deferred_tensor_operands: Store::new(
                self.deferred_tensor_operands.with(|value| value.clone()),
            ),
            collection_contracts: Store::new(self.collection_contracts.with(|value| value.clone())),
            next_collection_contract_id: Store::new(
                self.next_collection_contract_id.with(|value| *value),
            ),
            operand_gate_failures: Store::new(
                self.operand_gate_failures.with(|value| value.clone()),
            ),
            deferred_opaque_uses: Store::new(self.deferred_opaque_uses.with(|value| value.clone())),
            current_level: self.current_level,
            level_transitions: self.level_transitions.clone(),
            lowered_tvar_levels: self.lowered_tvar_levels.clone(),
            lowered_dvar_levels: self.lowered_dvar_levels.clone(),
            lowered_rvar_levels: self.lowered_rvar_levels.clone(),
            resume_floors: self.resume_floors,
            ground_types: self.ground_types.clone(),
        }
    }
}

impl<E> Subst<E> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of logical type bindings.
    pub fn types_len(&self) -> usize {
        self.types.with(|bindings| bindings.len())
    }

    /// Number of logical dimension bindings.
    pub fn dims_len(&self) -> usize {
        self.dims.with(|bindings| bindings.len())
    }

    /// Number of logical rank bindings.
    pub fn ranks_len(&self) -> usize {
        self.ranks.with(|bindings| bindings.len())
    }

    fn assert_complete(&self) {
        self.result_constraints.with(|_| ());
        self.types.with(|_| ());
        self.tvar_restrictions.with(|_| ());
        self.key_free_tvars.with(|_| ());
        self.dims.with(|_| ());
        self.dimension_labels.with(|_| ());
        self.dimension_label_undo.with(|_| ());
        self.ranks.with(|_| ());
        self.deferred_borrow_vars.with(|_| ());
        self.deferred_drop_operands.with(|_| ());
        self.deferred_opaque_uses.with(|_| ());
        self.deferred_tensor_operands.with(|_| ());
        self.operand_gate_failures.with(|_| ());
        self.collection_contracts.with(|_| ());
        self.next_collection_contract_id.with(|_| ());
        self.ground_types.0.with(|_| ());
    }
}

impl<E: Clone + PartialEq> Subst<E> {
    pub fn record_result_constraint(&self, constraint: crate::model::ResultConstraint<E>) {
        self.result_constraints.with(|value| value.push(constraint));
    }

    pub fn take_result_constraints(&self) -> Vec<crate::model::ResultConstraint<E>> {
        self.result_constraints.with(core::mem::take)
    }

    pub fn current_level(&self) -> u32 {
        self.current_level
    }

    /// Enter a lexical inference scope and record the first IDs minted there.
    ///
    pub fn enter_level(&mut self, var_gen: &VarGen) -> LevelToken {
        let parent_level = self.current_level;
        let child_level = parent_level
            .checked_add(1)
            .expect("type-checker generalization level overflow");
        self.current_level = child_level;
        self.level_transitions.push(LevelTransition {
            kind: LevelTransitionKind::Enter,
            level: child_level,
            watermarks: var_gen.watermarks(),
        });
        LevelToken {
            parent_level,
            child_level,
        }
    }

    /// Leave the most recently entered lexical inference scope.
    ///
    pub fn leave_level(&mut self, token: LevelToken, var_gen: &VarGen) {
        assert_eq!(
            self.current_level, token.child_level,
            "generalization levels must be left in LIFO order"
        );
        self.current_level = token.parent_level;
        self.level_transitions.push(LevelTransition {
            kind: LevelTransitionKind::Leave,
            level: token.parent_level,
            watermarks: var_gen.watermarks(),
        });
    }

    /// Normalize a persisted solver before it is reused for a new check.
    ///
    /// Existing IDs become level-zero imports. only work doed after this.
    ///
    /// point contributes level metadata to the serialized context.
    ///
    pub fn resume_for_new_check(&mut self, var_gen: &VarGen) {
        self.protected_dimensions.clear();
        // chelis#1801: per-pass evidence, cleared at the pass boundary for the.
        //
        // same reason the protected set is. A resumed check reads it back.
        //
        // empty anyway (it is `serde(skip)`). this covers a live reuse.
        //
        self.wildcard_touched_classes.clear();
        self.binder_pinned_classes.clear();
        assert_eq!(
            self.current_level, 0,
            "a persisted type environment cannot resume inside an inference scope"
        );
        let floors = var_gen.watermarks();
        self.resume_floors = floors;
        self.level_transitions.clear();
        let stale_tvars = self
            .lowered_tvar_levels
            .to_sorted()
            .into_iter()
            .filter_map(|(var, _)| (var.0 < floors.next_tvar).then_some(*var))
            .collect::<Vec<_>>();
        for var in stale_tvars {
            self.lowered_tvar_levels.remove(&var);
        }
        let stale_dvars = self
            .lowered_dvar_levels
            .to_sorted()
            .into_iter()
            .filter_map(|(var, _)| (var.0 < floors.next_dvar).then_some(*var))
            .collect::<Vec<_>>();
        for var in stale_dvars {
            self.lowered_dvar_levels.remove(&var);
        }
        let stale_rvars = self
            .lowered_rvar_levels
            .to_sorted()
            .into_iter()
            .filter_map(|(var, _)| (var.0 < floors.next_rvar).then_some(*var))
            .collect::<Vec<_>>();
        for var in stale_rvars {
            self.lowered_rvar_levels.remove(&var);
        }
        self.level_transitions.push(LevelTransition {
            kind: LevelTransitionKind::Resume,
            level: 0,
            watermarks: floors,
        });
    }

    fn mint_level(&self, id: u32, class: VarClass) -> u32 {
        let floor = match class {
            VarClass::Type => self.resume_floors.next_tvar,
            VarClass::Dim => self.resume_floors.next_dvar,
            VarClass::Rank => self.resume_floors.next_rvar,
        };
        if id < floor {
            return 0;
        }
        self.level_transitions
            .last_at_or_below(id, |transition| match class {
                VarClass::Type => transition.watermarks.next_tvar,
                VarClass::Dim => transition.watermarks.next_dvar,
                VarClass::Rank => transition.watermarks.next_rvar,
            })
            .map_or(0, |transition| transition.level)
    }

    pub fn level_of_tvar(&self, var: TypeVar) -> u32 {
        self.lowered_tvar_levels
            .get(&var)
            .copied()
            .unwrap_or_else(|| self.mint_level(var.0, VarClass::Type))
    }

    pub fn level_of_dvar(&self, var: DimVar) -> u32 {
        self.lowered_dvar_levels
            .get(&var)
            .copied()
            .unwrap_or_else(|| self.mint_level(var.0, VarClass::Dim))
    }

    pub fn level_of_rvar(&self, var: RankVar) -> u32 {
        self.lowered_rvar_levels
            .get(&var)
            .copied()
            .unwrap_or_else(|| self.mint_level(var.0, VarClass::Rank))
    }

    fn lower_tvar_to(&mut self, var: TypeVar, level: u32) {
        if level < self.level_of_tvar(var) {
            self.lowered_tvar_levels.insert(var, level);
        }
    }

    fn lower_dvar_to(&mut self, var: DimVar, level: u32) {
        if level < self.level_of_dvar(var) {
            self.lowered_dvar_levels.insert(var, level);
        }
    }

    fn lower_rvar_to(&mut self, var: RankVar, level: u32) {
        if level < self.level_of_rvar(var) {
            self.lowered_rvar_levels.insert(var, level);
        }
    }

    /// Lower every variable reachable through the post-substitution type.
    ///
    fn lower_type_to(&mut self, ty: &Type<E>, level: u32) {
        let ty = self.apply(ty);
        for var in crate::scheme::free_tvars(&ty) {
            self.lower_tvar_to(var, level);
        }
        for var in crate::scheme::free_dvars(&ty) {
            self.lower_dvar_to(var, level);
        }
        for var in crate::scheme::free_rvars(&ty) {
            self.lower_rvar_to(var, level);
        }
    }

    fn lower_dim_to(&mut self, dim: &Dim, level: u32) {
        match self.apply_dim(dim) {
            Dim::Var(var) => self.lower_dvar_to(var, level),
            Dim::Rank(var) => self.lower_rvar_to(var, level),
            Dim::Lit(_) | Dim::Name(_) | Dim::Wildcard => {}
        }
    }

    fn lower_ground_rank_to(&mut self, dims: &[Dim], level: u32) {
        for dim in dims {
            self.lower_dim_to(dim, level);
        }
    }

    pub fn lower_type_to_current(&mut self, ty: &Type<E>) {
        self.lower_type_to(ty, self.current_level);
    }

    /// Lower every variable reachable through `ty` to the enclosing `level`,.
    ///
    /// so no scope entered since generalizes over it.
    ///
    pub fn lower_type_to_level(&mut self, ty: &Type<E>, level: u32) {
        self.lower_type_to(ty, level);
    }

    #[cfg(any(test, feature = "test-instrumentation"))]
    pub fn level_metadata_counts(&self) -> (usize, usize, usize, usize) {
        (
            self.level_transitions.len(),
            self.lowered_tvar_levels.len(),
            self.lowered_dvar_levels.len(),
            self.lowered_rvar_levels.len(),
        )
    }

    /// Snapshot of the type-variable bindings (cloned out of the lock).
    ///
    /// Useful for serialization, tests, and read-only inspection.
    ///
    pub fn types_snapshot(&self) -> UnordMap<TypeVar, Type<E>> {
        self.types.with(|value| sorted_copy(value))
    }

    /// Snapshot of the dim-variable bindings.
    ///
    pub fn dims_snapshot(&self) -> UnordMap<DimVar, Dim> {
        self.dims.with(|value| sorted_copy(value))
    }

    /// Record a type-variable binding through ordinary unification.
    ///
    ///
    /// This public mutation seam enforces occurs checks, semantic domains,.
    ///
    /// deferred constraints, and restriction transfer exactly like every.
    ///
    /// checker-created binding. It must never write the map directly.
    ///
    pub fn insert_type(&mut self, v: TypeVar, ty: Type<E>) -> Result<(), TypeError> {
        unify(&Type::Var(v), &ty, self)
    }

    /// Publish a type binding after [`bind_tvar`] has validated the complete.
    ///
    /// transaction. Keeping the raw map write private prevents callers from.
    ///
    /// bypassing semantic domains through the public API.
    ///
    fn record_validated_type_binding(&mut self, v: TypeVar, ty: Type<E>) {
        self.types.with(|value| value.insert(v, ty));
    }

    /// Attach a semantic domain to an unresolved inference variable,.
    ///
    /// narrowing to the intersection with any domain it already carries.
    ///
    ///
    /// `spec/04-type-system.md` [04-DTYPE-2]: identifying two bounded.
    ///
    /// variables yields the intersection of their families, and an empty.
    ///
    /// intersection is a `PrecisionMismatch`. Narrowing here rather than.
    ///
    /// overwriting is what keeps a `Numeric` alias from widening a `Float`.
    ///
    /// variable back out to every numeric dtype.
    ///
    pub fn narrow_tvar_restriction(
        &self,
        v: TypeVar,
        restriction: TypeVarRestriction,
    ) -> Result<(), TypeError> {
        self.tvar_restrictions.with(|restrictions| {
            let narrowed = match restrictions.get(&v).copied() {
                Some(existing) => merge_tvar_restrictions(existing, restriction)?,
                None => restriction,
            };
            restrictions.insert(v, narrowed);
            Ok(())
        })
    }

    /// Restriction currently attached to an unresolved variable, if any.
    ///
    pub fn tvar_restriction(&self, v: TypeVar) -> Option<TypeVarRestriction> {
        self.tvar_restrictions.with(|value| value.get(&v).copied())
    }

    fn tvar_restrictions_snapshot(&self) -> UnordMap<TypeVar, TypeVarRestriction> {
        self.tvar_restrictions.with(|value| sorted_copy(value))
    }

    fn remove_tvar_restriction(&self, v: TypeVar) {
        self.tvar_restrictions.with(|value| value.remove(&v));
    }

    /// [04-LIN-10]: mark `v` as standing for a function's type parameter. A.
    ///
    /// variable already marked keeps its generic, so a violation names the.
    ///
    /// first generic the variable was found to belong to.
    ///
    pub fn forbid_key_instantiation(&self, v: TypeVar, origin: GenericParameter) {
        self.key_free_tvars.with(|value| {
            value.get_or_insert_with(v, || origin);
        });
    }

    /// The generic `v` stands for a type parameter of, when it is marked.
    ///
    pub fn key_free_origin(&self, v: TypeVar) -> Option<GenericParameter> {
        self.key_free_tvars.with(|value| value.get(&v).cloned())
    }

    /// Name the generic that a binding's scheme belongs to on each of its.
    ///
    /// quantified variables, with the authored binder each one instantiates.
    ///
    /// when `binders` records it. An authored binder of this binding always.
    ///
    /// names the variable. A variable this binding did not spell keeps the.
    ///
    /// name of a generic it came from, so `t = (dup, 1)` then `(t.0)(k)` names.
    ///
    /// `dup`'s parameter rather than the tuple.
    ///
    pub fn name_generic_parameters(
        &self,
        scheme: &Scheme<E>,
        generic: &str,
        binders: &UnordMap<TypeVar, String>,
    ) {
        let binders = binders.to_sorted();
        let origin_scope = scheme
            .result_origin
            .as_ref()
            .map(|origin| crate::result_scope::ResultScope::new(&origin.equations));
        let closed_key_variables = crate::scheme::closed_key_relation_variables(scheme);
        self.key_free_tvars.with_detached(|marks| {
            // Equation-local origin quantifiers are not more generic.
            //
            // parameters. An explicitly authored binder can still name every.
            //
            // representative of its equality component, including raw origins.
            //
            let mut variables = scheme.tvars.clone();
            if let Some(origin) = &scheme.result_origin {
                for variable in &origin.tvars {
                    if !variables.contains(variable) {
                        variables.push(*variable);
                    }
                }
            }
            for tv in &variables {
                // A closed checked key relation is not an authored unconstrained.
                //
                // type parameter. Preserve any older generic mark, however: a.
                //
                // relation cannot erase the restriction of an enclosing binder.
                //
                if closed_key_variables.contains(tv) && !marks.contains_key(tv) {
                    continue;
                }
                let binder = binders
                    .iter()
                    .find(|(declared, _)| {
                        **declared == *tv
                            || self.apply(&Type::Var(**declared)) == Type::Var(*tv)
                            || origin_scope.as_ref().is_some_and(|scope| {
                                scope.shares_type_component(**declared, *tv)
                                    || match self.apply(&Type::Var(**declared)) {
                                        Type::Var(resolved) => {
                                            scope.shares_type_component(resolved, *tv)
                                        }
                                        _ => false,
                                    }
                            })
                    })
                    .map(|(_, name)| (*name).clone());
                if binder.is_none() && !scheme.tvars.contains(tv) {
                    continue;
                }
                let named_elsewhere = marks
                    .get(tv)
                    .is_some_and(|existing| existing.generic.is_some());
                if binder.is_none() && named_elsewhere {
                    continue;
                }
                marks.insert(
                    *tv,
                    GenericParameter {
                        generic: Some(generic.to_string()),
                        binder,
                        value: !matches!(scheme.body, Type::Fn(_, _)),
                    },
                );
            }
        })
    }

    /// Install spec/04 section 8.4.1's key-carrying data types for this check.
    ///
    pub fn set_key_carrying_adts(&mut self, adts: UnordSet<String>) {
        self.key_carrying_adts = adts.into_sorted().into_iter().collect();
    }

    /// Record a new dim-variable binding.
    ///
    pub fn insert_dim(&mut self, v: DimVar, dim: Dim) {
        self.dims.with(|value| value.insert(v, dim));
    }

    /// Record a rank-variable binding: `r` stands for the whole shape `dims`.
    ///
    pub fn insert_rank(&mut self, r: RankVar, dims: Vec<Dim>) {
        self.ranks.with(|value| value.insert(r, dims));
    }

    pub fn static_dim_product(&self, dims: &[Dim]) -> Option<i128> {
        dims.iter().try_fold(1_i128, |product, dim| {
            let value = self.observe_dim(dim).literal_extent()?;
            product.checked_mul(i128::from(value))
        })
    }

    /// Compare two fully-static dimension products without a fixed-width.
    ///
    /// multiplication. Pairwise GCD cancellation is exact for any number of.
    ///
    /// i64 factors, so a large known shape never degrades into "unknown".
    ///
    pub fn static_dim_products_match(&self, lhs: &[Dim], rhs: &[Dim]) -> Option<bool> {
        fn values<E: Clone + PartialEq>(subst: &Subst<E>, dims: &[Dim]) -> Option<Vec<i64>> {
            dims.iter()
                .map(|dim| subst.observe_dim(dim).literal_extent())
                .collect()
        }

        fn sign_and_factors(values: &[i64]) -> (bool, Vec<u64>) {
            let negative = values.iter().filter(|value| **value < 0).count() % 2 == 1;
            let factors = values.iter().map(|value| value.unsigned_abs()).collect();
            (negative, factors)
        }

        fn gcd(mut a: u64, mut b: u64) -> u64 {
            while b != 0 {
                let remainder = a % b;
                a = b;
                b = remainder;
            }
            a
        }

        let lhs = values(self, lhs)?;
        let rhs = values(self, rhs)?;
        let lhs_zero = lhs.contains(&0);
        let rhs_zero = rhs.contains(&0);
        if lhs_zero || rhs_zero {
            return Some(lhs_zero && rhs_zero);
        }

        let (lhs_negative, mut lhs_factors) = sign_and_factors(&lhs);
        let (rhs_negative, mut rhs_factors) = sign_and_factors(&rhs);
        if lhs_negative != rhs_negative {
            return Some(false);
        }
        for left in &mut lhs_factors {
            for right in &mut rhs_factors {
                let divisor = gcd(*left, *right);
                *left /= divisor;
                *right /= divisor;
            }
        }
        Some(
            lhs_factors.iter().all(|factor| *factor == 1)
                && rhs_factors.iter().all(|factor| *factor == 1),
        )
    }
    /// Snapshot of the rank-variable bindings.
    ///
    pub fn ranks_snapshot(&self) -> UnordMap<RankVar, Vec<Dim>> {
        self.ranks.with(|value| sorted_copy(value))
    }

    /// Resolve a rank variable to the shape vector it stands for, chasing.
    ///
    /// rank-to-rank aliases. Returns `[Dim::Rank(r)]` (the unbound variable).
    ///
    /// when `r` has no binding. The returned dims are not themselves.
    ///
    /// substitution-applied. callers run them through `apply_dim`.
    ///
    fn resolve_rvar(&self, start: RankVar) -> Vec<Dim> {
        self.ranks.with(|map| {
            if let Some(owner) = map.resolved.get(&start) {
                return map
                    .get(owner)
                    .cloned()
                    .unwrap_or_else(|| vec![Dim::Rank(*owner)]);
            }
            let mut chain = Vec::new();
            let mut current = start;
            let max_steps = map.len() + 1;
            let terminal = loop {
                if chain.len() >= max_steps {
                    break vec![Dim::Rank(current)];
                }
                match map.get(&current) {
                    None => break vec![Dim::Rank(current)],
                    Some(bound) => match bound.as_slice() {
                        [Dim::Rank(next)] if *next != current => {
                            chain.push(current);
                            current = *next;
                        }
                        [Dim::Rank(_)] => break vec![Dim::Rank(current)],
                        _ => break bound.clone(),
                    },
                }
            };
            if chain.len() > 1 && chain.len() < max_steps {
                for rank in chain {
                    map.resolved.insert(rank, current);
                }
            }
            terminal
        })
    }

    /// Suspend an operand decision on the variable that has to be bound.
    ///
    /// before it can be made (chelis#1489). Unification discharges it at that.
    ///
    /// binding. see the `deferred_tensor_operands` field doc.
    ///
    pub fn record_deferred_tensor_operand(&self, v: TypeVar, gate: DeferredOperandGate<E>) {
        self.deferred_tensor_operands
            .with(|value| value.push((v, gate)));
    }

    /// Every variable a PENDING gate waits on or hands out (chelis#1489).
    ///
    ///
    /// Consulted by `Env::generalize`, which must not quantify any of them.
    ///
    ///
    /// The result: a suspended `copy`/`cast` hands its consumer a fresh result.
    ///
    /// variable and ties it to the operand only through this ledger --.
    ///
    /// invisibly to levels. When an unannotated `let` generalized that.
    ///
    /// variable, every use of the bound name got its own unconstrained.
    ///
    /// instance, and discharge later bound only the original: a declared.
    ///
    /// result was never checked against what the call produces, and a false.
    ///
    /// signature checked and ran.
    ///
    ///
    /// The operand: the gate is suspended on this one variable, and only its.
    ///
    /// binding discharges the gate. When an unannotated `let` generalized it,.
    ///
    /// every application of the bound lambda bound a fresh instance instead,.
    ///
    /// the gate stayed on the never-bound template, and the declaration.
    ///
    /// boundary rejected a well-typed program as unresolved (chelis#2584:.
    ///
    /// `g = fn (y) -> cast(y, p)` then `g(x)`). Keeping it monomorphic is what.
    ///
    /// [04-INF-1] needs of a lambda carrying a pending obligation, so its.
    ///
    /// first application binds the operand and discharges the gate.
    ///
    ///
    /// ALL free variables are returned, not just a top-level type variable. A.
    ///
    /// pending result can be partly unified before it discharges -- `g` meeting.
    ///
    /// a `tensor[?d, 3, f32]` expectation makes it `tensor[?d, 3, f32]` -- and.
    ///
    /// quantifying `?d` reopens the same hole.
    ///
    pub fn pending_gate_vars(&self) -> (UnordSet<TypeVar>, UnordSet<DimVar>, UnordSet<RankVar>) {
        self.deferred_tensor_operands.with_detached(|ledger| {
            let mut tvars = UnordSet::default();
            let mut dvars = UnordSet::default();
            let mut rvars = UnordSet::default();
            // Transported checked builtin values remain polymorphic. A.
            //
            // consumed relation, however, is a pending body obligation just like.
            //
            // an operand gate: generalization must not detach its template from.
            //
            // the first application that supplies the operand/result evidence.
            //
            let consumed = self.collection_contracts.with_detached(|value| {
                value
                    .iter()
                    .filter(|instance| {
                        matches!(instance.state, CollectionContractState::Consumed { .. })
                    })
                    .map(|instance| instance.constraint.clone())
                    .collect::<Vec<_>>()
            });
            if ledger.is_empty() && consumed.is_empty() {
                return (tvars, dvars, rvars);
            }
            for (operand, gate) in ledger.iter() {
                let operand = self.apply(&Type::Var(*operand));
                let result = gate.result().map(|result| self.apply(result));
                for pending in core::iter::once(&operand).chain(result.as_ref()) {
                    tvars.extend(crate::scheme::free_tvars(pending));
                    dvars.extend(crate::scheme::free_dvars(pending));
                    rvars.extend(crate::scheme::free_rvars(pending));
                }
            }
            for constraint in consumed {
                for carried in constraint.carried_types() {
                    let pending = self.apply(carried);
                    tvars.extend(crate::scheme::free_tvars(&pending));
                    dvars.extend(crate::scheme::free_dvars(&pending));
                    rvars.extend(crate::scheme::free_rvars(&pending));
                }
            }
            (tvars, dvars, rvars)
        })
    }

    /// chelis#2584, [04-INF-1]: whether a pending gate waits on or hands out a.
    ///
    /// variable of `ty` that generalization at the current level would.
    ///
    /// quantify. A `let`-bound lambda for which this holds carries a deferred.
    ///
    /// obligation, so the whole lambda stays monomorphic until its first.
    ///
    /// application, and every later use has that same instantiation. an.
    ///
    /// unchecked second parameter does not become polymorphic beside a.
    ///
    /// checked one.
    ///
    pub fn has_generalizable_pending_gate(&self, ty: &Type<E>) -> bool {
        let (tvars, dvars, rvars) = self.pending_gate_vars();
        if tvars.is_empty() && dvars.is_empty() && rvars.is_empty() {
            return false;
        }
        let ty = self.apply(ty);
        let level = self.current_level();
        crate::scheme::free_tvars(&ty)
            .iter()
            .any(|v| tvars.contains(v) && self.level_of_tvar(*v) > level)
            || crate::scheme::free_dvars(&ty)
                .iter()
                .any(|v| dvars.contains(v) && self.level_of_dvar(*v) > level)
            || crate::scheme::free_rvars(&ty)
                .iter()
                .any(|v| rvars.contains(v) && self.level_of_rvar(*v) > level)
    }

    /// Record one fresh use of a checked collection-operation contract.
    ///
    pub fn record_collection_contract(
        &self,
        constraint: CollectionConstraint<E>,
    ) -> CollectionContractId {
        let id = self.next_collection_contract_id.with(|next| {
            let id = *next;
            *next = next
                .checked_add(1)
                .expect("collection contract identity overflow");
            id
        });
        self.collection_contracts.with(|value| {
            value.push(CollectionContractInstance {
                id,
                level: self.current_level,
                constraint,
                state: CollectionContractState::Transport,
            })
        });
        id
    }

    /// Mark the start of an expression's contract instantiations.
    ///
    pub fn collection_contract_mark(&self) -> CollectionContractId {
        self.next_collection_contract_id.with(|value| *value)
    }

    /// Live contract instances minted since `mark`, in instantiation order.
    ///
    pub fn collection_contract_ids_since(
        &self,
        mark: CollectionContractId,
    ) -> Vec<CollectionContractId> {
        self.collection_contracts.with(|value| {
            value
                .iter()
                .filter_map(|instance| (instance.id >= mark).then_some(instance.id))
                .collect()
        })
    }

    /// Transportable child-scope instances considered by `Env::generalize`.
    ///
    pub fn pending_collection_contracts(
        &self,
    ) -> Vec<(CollectionContractId, u32, CollectionConstraint<E>)> {
        self.collection_contracts.with_detached(|value| {
            value
                .iter()
                .filter(|instance| matches!(instance.state, CollectionContractState::Transport))
                .map(|instance| {
                    (
                        instance.id,
                        instance.level,
                        instance.constraint.map_types(|carried| self.apply(carried)),
                    )
                })
                .collect()
        })
    }

    /// Move or discard exact transport instances selected by generalization.
    ///
    pub fn take_collection_contracts(&self, removals: &[CollectionContractId]) {
        if removals.is_empty() {
            return;
        }
        self.collection_contracts
            .with(|value| value.retain(|instance| !removals.contains(&instance.id)));
    }

    /// Drop the scheme copies instantiated for a direct syntactic builtin call.
    ///
    pub fn discard_collection_contracts(&self, ids: &[CollectionContractId]) {
        self.take_collection_contracts(ids);
    }

    /// Whether these exact callee instances include tensor concat.
    ///
    pub fn collection_contracts_include_concat(
        &self,
        ids: &[CollectionContractId],
        callees: &[Type<E>],
    ) -> bool {
        let callees = callees.iter().map(|ty| self.apply(ty)).collect::<Vec<_>>();
        self.collection_contracts.with_detached(|value| {
            value.iter().any(|instance| {
                ids.contains(&instance.id)
                    && matches!(instance.constraint, CollectionConstraint::Concat { .. })
                    && callees.contains(&collection_contract_callable_type(
                        &instance.constraint.map_types(|carried| self.apply(carried)),
                    ))
            })
        })
    }

    /// Consume the exact checked contracts instantiated by this callee.
    ///
    pub fn prepare_collection_contract_call(
        &self,
        ids: &[CollectionContractId],
        callees: &[Type<E>],
        tensor_concat: Option<crate::decisions::collection::TensorConcatCallEvidence>,
        split_keys_count: Option<i64>,
    ) {
        let callees = callees.iter().map(|ty| self.apply(ty)).collect::<Vec<_>>();
        self.collection_contracts.with_detached(|contracts| {
            for instance in contracts.iter_mut() {
                if !ids.contains(&instance.id) {
                    continue;
                }
                let normalized = instance.constraint.map_types(|carried| self.apply(carried));
                if !callees.contains(&collection_contract_callable_type(&normalized)) {
                    continue;
                }
                let evidence = match instance.constraint {
                    CollectionConstraint::Concat { .. } => tensor_concat
                        .clone()
                        .map(crate::decisions::collection::CollectionCallEvidence::TensorConcat),
                    CollectionConstraint::SplitKeys { .. } => Some(
                        crate::decisions::collection::CollectionCallEvidence::SplitKeysCount(
                            split_keys_count,
                        ),
                    ),
                    _ => None,
                };
                instance.state = CollectionContractState::Consumed { evidence };
            }
        })
    }

    /// A before body call can wait on the operand this application.
    ///
    /// just bound. Revisit only consumed instances. transported callable.
    ///
    /// values still belong to their own applications/generalization.
    ///
    fn replay_consumed_collection_contracts_before(&mut self, mark: CollectionContractId) {
        loop {
            let ids = self.collection_contracts.with_detached(|value| {
                value
                    .iter()
                    .filter(|instance| {
                        instance.id < mark
                            && matches!(instance.state, CollectionContractState::Consumed { .. })
                    })
                    .map(|instance| instance.id)
                    .collect::<Vec<_>>()
            });
            let mut settled = false;
            for id in ids {
                let Some(instance) = self.take_collection_contract_instance(id) else {
                    continue;
                };
                let CollectionContractState::Consumed { evidence } = &instance.state else {
                    unreachable!("selected consumed relation");
                };
                match discharge_collection_constraint(&instance.constraint, evidence.as_ref(), self)
                {
                    CollectionDischarge::Unresolved => {
                        self.restore_collection_contract_instance(instance)
                    }
                    CollectionDischarge::Settled(_) => settled = true,
                }
            }
            if !settled {
                break;
            }
        }
    }

    /// Finish one application's contract lifecycle after ordinary call.
    ///
    /// unification tied all argument and result variables together.
    ///
    ///
    /// Exact callee instances are consumed. Contracts instantiated by.
    ///
    /// function-valued arguments remain transport when the application result.
    ///
    /// still contains their callable. otherwise a now-decidable relation is.
    ///
    /// checked and an unresolved ignored value is discarded. No instance from.
    ///
    /// before `mark` is examined.
    ///
    pub fn finish_collection_contract_application(
        &mut self,
        mark: CollectionContractId,
        arguments: &[Type<E>],
        result: &Type<E>,
        related_results: &[Type<E>],
    ) -> Option<Type<E>> {
        self.replay_consumed_collection_contracts_before(mark);
        let ids = self.collection_contract_ids_since(mark);
        let applied_result = self.apply(result);
        let mut checked_result = None;
        for id in ids {
            let Some(instance) = self.take_collection_contract_instance(id) else {
                continue;
            };
            match instance.state.clone() {
                CollectionContractState::Consumed { evidence } => {
                    match discharge_collection_constraint(
                        &instance.constraint,
                        evidence.as_ref(),
                        self,
                    ) {
                        CollectionDischarge::Unresolved => {
                            self.restore_collection_contract_instance(instance);
                        }
                        CollectionDischarge::Settled(Some(settled)) => {
                            if let Some(previous) = &checked_result {
                                let _ = unify(previous, &settled, self);
                            } else {
                                checked_result = Some(settled);
                            }
                        }
                        CollectionDischarge::Settled(None) => {}
                    }
                }
                CollectionContractState::Transport => {
                    let normalized = instance.constraint.map_types(|carried| self.apply(carried));
                    if collection_contract_visible_in_type(&normalized, &applied_result)
                        || related_results.iter().any(|ty| {
                            collection_contract_visible_in_type(&normalized, &self.apply(ty))
                        })
                    {
                        self.restore_collection_contract_instance(instance);
                    } else if collection_contract_tied_to_arguments(
                        &instance.constraint,
                        arguments,
                        self,
                    ) {
                        match discharge_collection_constraint(&instance.constraint, None, self) {
                            CollectionDischarge::Unresolved => {
                                let mut consumed = instance;
                                consumed.state =
                                    CollectionContractState::Consumed { evidence: None };
                                self.restore_collection_contract_instance(consumed);
                            }
                            CollectionDischarge::Settled(_) => {}
                        }
                    } else {
                        // A checked function argument not returned by the.
                        //
                        // outer call is either consumed by the callee's type.
                        //
                        // equations or ignored. A decidable relation owns its.
                        //
                        // verdict. an unresolved ignored value owns no future.
                        //
                        // application and is deliberately discarded.
                        //
                        let _discarded =
                            discharge_collection_constraint(&instance.constraint, None, self);
                    }
                }
            }
        }
        self.replay_consumed_collection_contracts_before(mark);
        checked_result
    }

    /// Cancel every contract instance created while a failed application was.
    ///
    /// inferred. This is the explicit cleanup edge that prevents a rejected.
    ///
    /// call from lending evidence to a later valid one.
    ///
    pub fn cancel_collection_contract_application(&self, mark: CollectionContractId) {
        self.collection_contracts
            .with(|value| value.retain(|instance| instance.id < mark));
    }

    fn take_collection_contract_instance(
        &self,
        id: CollectionContractId,
    ) -> Option<CollectionContractInstance<E>> {
        self.collection_contracts.with_detached(|contracts| {
            let index = contracts.iter().position(|instance| instance.id == id)?;
            Some(contracts.remove(index))
        })
    }

    fn restore_collection_contract_instance(&self, instance: CollectionContractInstance<E>) {
        self.collection_contracts.with_detached(|contracts| {
            let index = contracts
                .iter()
                .position(|current| current.id > instance.id)
                .unwrap_or(contracts.len());
            contracts.insert(index, instance);
        })
    }

    /// Drain unresolved consumed relations and detached child-scope transport.
    ///
    /// at a declaration boundary. Transport born in the active recursive.
    ///
    /// component's own level stays live until component generalization.
    ///
    pub fn take_boundary_collection_contracts(&self) -> Vec<CollectionConstraint<E>> {
        let current_level = self.current_level;
        self.collection_contracts.with_detached(|contracts| {
            let mut unresolved = Vec::new();
            contracts.retain(|instance| match instance.state {
                CollectionContractState::Consumed { .. } => {
                    unresolved.push(instance.constraint.map_types(|carried| self.apply(carried)));
                    false
                }
                CollectionContractState::Transport
                    if current_level == 0 || instance.level > current_level =>
                {
                    false
                }
                CollectionContractState::Transport => true,
            });
            unresolved
        })
    }

    /// Record a discharge failure for the per-def reporting pass.
    ///
    /// (chelis#1489).
    ///
    pub fn record_operand_gate_failure(&self, failure: OperandGateFailure<E>) {
        self.operand_gate_failures.with(|value| value.push(failure));
    }

    /// Drain the discharge failures. Called once per def body, with.
    ///
    /// [`Self::take_deferred_tensor_operands`], by the reporting pass.
    ///
    pub fn take_operand_gate_failures(&self) -> Vec<OperandGateFailure<E>> {
        self.operand_gate_failures.with(core::mem::take)
    }

    /// Take every constraint suspended on `v`, leaving the rest of the ledger.
    ///
    /// in place (chelis#1489).
    ///
    ///
    /// Discharge removes a constraint BEFORE deciding it, so a decision that.
    ///
    /// unifies -- and therefore re-enters this -- cannot rediscover the.
    ///
    /// constraint it is in the middle of discharging.
    ///
    ///
    /// The ledger does NOT strictly shrink: binding a variable to another.
    ///
    /// VARIABLE re-suspends the obligation on the target, so an entry can be.
    ///
    /// removed and re-added. Termination rests on the alias chain being.
    ///
    /// acyclic -- `bind_tvar_inner` rejects self-binding and `occurs_in`.
    ///
    /// rejects cycles -- not on a shrinking count.
    ///
    fn take_operand_gates_on(&self, v: TypeVar) -> Vec<DeferredOperandGate<E>> {
        self.deferred_tensor_operands.with_detached(|ledger| {
            let mut taken = Vec::new();
            ledger.retain(|(tv, gate)| {
                if *tv == v {
                    taken.push(gate.clone());
                    false
                } else {
                    true
                }
            });
            taken
        })
    }

    /// Re-suspend `gate` on `target` because `v` was bound to it rather than.
    ///
    /// to a concrete type (chelis#1489).
    ///
    ///
    /// Identifying two variables must carry the obligation across, exactly as.
    ///
    /// the shape ledgers' `merge_alias` does. dropping it here would silently.
    ///
    /// un-defer the constraint.
    ///
    /// Remove and decide every suspended scalar `Cast` or `CastToBinder` gate.
    ///
    /// whose operand now resolves to a variable with a declared dtype-family.
    ///
    /// bound, when that bound alone settles it (chelis#2151, chelis#2534). See.
    ///
    /// [`discharge_bounded_scalar_casts`].
    ///
    ///
    /// Almost every binding runs with an empty ledger, so it returns early.
    ///
    /// without applying the substitution in that case.
    ///
    #[allow(clippy::type_complexity)]
    fn take_bounded_scalar_cast_gates(
        &self,
    ) -> Vec<(
        DeferredOperandGate<E>,
        Result<Type<E>, Box<crate::decisions::DecisionFailure>>,
    )> {
        self.deferred_tensor_operands.with_detached(|ledger| {
            if ledger.is_empty() {
                return Vec::new();
            }
            let mut decided = Vec::new();
            ledger.retain(|(tv, gate)| {
                if !matches!(
                    gate,
                    DeferredOperandGate::Cast { .. } | DeferredOperandGate::CastToBinder { .. }
                ) {
                    return true;
                }
                let Type::Var(operand) = self.apply(&Type::Var(*tv)) else {
                    return true;
                };
                // Any §5.9 declaration bound discharges this gate, including the.
                //
                // explicit set form. Listing the three families rejected a.
                //
                // set-bounded scalar `cast` that every equivalent family accepted.
                //
                // (chelis#2443 round 2).
                //
                let Some(bound) = self
                    .tvar_restriction(operand)
                    .filter(|restriction| restriction.is_declaration_bound())
                else {
                    return true;
                };
                let decision = match gate {
                    // A borrowed source is decided only once it binds, as the.
                    //
                    // eager `borrow` arm decides it (chelis#3101).
                    //
                    DeferredOperandGate::Cast { borrowed: true, .. } => None,
                    DeferredOperandGate::Cast { target, mode, .. } => {
                        crate::decisions::cast::bounded_scalar_cast_result(bound, *target, *mode)
                    }
                    // chelis#2534: every member of a numeric bound is a scalar.
                    //
                    // source a binder-target `cast` admits ([05-OP-63]).
                    //
                    DeferredOperandGate::CastToBinder { target, .. } => {
                        Some(Ok(Type::Var(*target)))
                    }
                    _ => None,
                };
                match decision {
                    Some(decision) => {
                        decided.push((gate.clone(), decision));
                        false
                    }
                    None => true,
                }
            });
            decided
        })
    }

    fn realias_operand_gate(&self, target: TypeVar, gate: DeferredOperandGate<E>) {
        self.deferred_tensor_operands
            .with(|value| value.push((target, gate)));
    }

    /// Drain the deferred tensor-operand ledger. Called once per def body's.
    ///
    /// inference so one def's deferrals cannot leak into the next.
    ///
    /// Decide every `sum`-family result still waiting on an unbound precision.
    ///
    /// variable at the declaration boundary (spec/04 §5.7.1, [04-INF-6]).
    ///
    ///
    /// The variable never binds after this point, so the result must be one.
    ///
    /// type at every dtype it admits: the variable itself when sum_result.
    ///
    /// keeps each one, one concrete dtype when sum_result maps them all.
    ///
    /// there, and otherwise a type error naming the operation and.
    ///
    /// sum_result. Returns whether any gate was decided, so the caller can.
    ///
    /// replay shape checks that waited on these results.
    ///
    ///
    /// `only` limits the decision to gates whose variable resolves to one of.
    ///
    /// those roots: the authored binders, which never bind, can be decided.
    ///
    /// before the boundary's own derivations run, while an inference variable.
    ///
    /// waits until they had the chance to bind it.
    ///
    pub fn decide_pending_sum_results(&mut self, only: Option<&[TypeVar]>) -> bool {
        let pending = {
            self.deferred_tensor_operands.with_detached(|ledger| {
                let mut pending = Vec::new();
                let selected = |tv: TypeVar| {
                    only.is_none_or(|roots| match self.apply(&Type::Var(tv)) {
                        Type::Var(root) => roots.contains(&root),
                        _ => true,
                    })
                };
                ledger.retain(|(tv, gate)| {
                    if matches!(gate, DeferredOperandGate::SumResult { .. }) && selected(*tv) {
                        pending.push((*tv, gate.clone()));
                        false
                    } else {
                        true
                    }
                });
                pending
            })
        };
        let decided = !pending.is_empty();
        for (variable, gate) in pending {
            let DeferredOperandGate::SumResult {
                ref op,
                ref dims,
                ref result,
                ..
            } = gate
            else {
                unreachable!("only sum-result gates were taken");
            };
            match self.apply(&Type::Var(variable)) {
                Type::Var(root) => {
                    match crate::decisions::shape::bound_sum_result_precision(op, root, self) {
                        Ok(precision) => gate.reconcile_result(
                            result,
                            Type::Tensor(dims.clone(), precision),
                            self,
                        ),
                        Err(message) => {
                            self.record_operand_gate_failure(OperandGateFailure::Decision {
                                error: gate.attach_location(
                                    crate::decisions::DecisionFailure::new(
                                        crate::decisions::DecisionFailureKind::TypeMismatch,
                                        message,
                                        Vec::new(),
                                    ),
                                ),
                            })
                        }
                    }
                }
                resolved => gate.clone().discharge(&resolved, self),
            }
        }
        decided
    }

    pub fn take_deferred_tensor_operands(&self) -> Vec<(TypeVar, DeferredOperandGate<E>)> {
        self.deferred_tensor_operands.with(core::mem::take)
    }

    /// Issue #256: record a borrow site whose inner type was still an.
    ///
    /// unresolved `Type::Var` when the `borrow` inference arm ran. The.
    ///
    /// driver drains these after a def body's inference completes and.
    ///
    /// re-checks each against the final substitution. See the.
    ///
    /// `deferred_borrow_vars` field doc for the soundness rationale.
    ///
    pub fn record_deferred_borrow_var(&self, v: TypeVar) {
        self.deferred_borrow_vars.with(|value| value.push(v));
    }

    /// Drain the deferred-borrow ledger, returning every recorded.
    ///
    /// `TypeVar`. Called once per def body's inference by the driver so.
    ///
    /// the ledger does not leak deferred sites across defs.
    ///
    pub fn take_deferred_borrow_vars(&self) -> Vec<TypeVar> {
        self.deferred_borrow_vars.with(core::mem::take)
    }

    /// chelis#3180: record a `drop` whose operand type was still an.
    ///
    /// unresolved `Type::Var`. See the `deferred_drop_operands` field doc.
    ///
    pub fn record_deferred_drop_operand(
        &self,
        v: TypeVar,
        location: Option<crate::decisions::SourceLabel>,
    ) {
        self.deferred_drop_operands
            .with(|value| value.push((v, location)));
    }

    /// Drain the deferred-drop ledger. called once per declaration by the.
    ///
    /// driver, mirroring `take_deferred_borrow_vars`.
    ///
    pub fn take_deferred_drop_operands(
        &self,
    ) -> Vec<(TypeVar, Option<crate::decisions::SourceLabel>)> {
        self.deferred_drop_operands.with(core::mem::take)
    }

    /// RFC D-CHECK: record an `access`/`record-update` site whose.
    ///
    /// target type was still an unresolved `Type::Var` when inference.
    ///
    /// ran. See the `deferred_opaque_uses` field doc.
    ///
    pub fn record_deferred_opaque_use(&self, v: TypeVar, use_kind: DeferredOpaqueUse) {
        self.deferred_opaque_uses
            .with(|value| value.push((v, use_kind)));
    }

    /// Drain the deferred-access ledger. called once per def body's.
    ///
    /// inference by the driver, mirroring `take_deferred_borrow_vars`.
    ///
    pub fn take_deferred_opaque_uses(&self) -> Vec<(TypeVar, DeferredOpaqueUse)> {
        self.deferred_opaque_uses.with(core::mem::take)
    }

    /// Resolve a type variable to its terminal binding (a non-Var, or.
    ///
    /// an unbound Var). Logical aliases remain intact.
    ///
    /// The transient cache stores each chain's terminal owner.
    ///
    fn resolve_tvar(&self, start: TypeVar) -> Type<E> {
        self.types.with(|map| {
            if let Some(owner) = map.resolved.get(&start) {
                return map.get(owner).cloned().unwrap_or(Type::Var(*owner));
            }
            let mut chain = Vec::new();
            let mut current = start;
            let terminal = loop {
                match map.get(&current) {
                    None => break Type::Var(current),
                    Some(Type::Var(next)) if *next == current => break Type::Var(current),
                    Some(Type::Var(next)) => {
                        chain.push(current);
                        current = *next;
                    }
                    Some(other) => {
                        chain.push(current);
                        break other.clone();
                    }
                }
            };
            if chain.len() > 1 {
                for variable in chain {
                    map.resolved.insert(variable, current);
                }
            }
            terminal
        })
    }

    /// Resolve a dim variable to its terminal binding. Same iterative-.
    ///
    /// walk-then-compress shape as [`Subst::resolve_tvar`].
    ///
    fn resolve_dvar(&self, start: DimVar) -> Dim {
        self.dims.with(|map| {
            if let Some(owner) = map.resolved.get(&start) {
                return map.get(owner).cloned().unwrap_or(Dim::Var(*owner));
            }
            let mut chain = Vec::new();
            let mut current = start;
            let terminal = loop {
                match map.get(&current) {
                    None => break Dim::Var(current),
                    Some(Dim::Var(next)) if *next == current => break Dim::Var(current),
                    Some(Dim::Var(next)) => {
                        chain.push(current);
                        current = *next;
                    }
                    Some(other) => {
                        chain.push(current);
                        break other.clone();
                    }
                }
            };
            if chain.len() > 1 {
                for variable in chain {
                    map.resolved.insert(variable, current);
                }
            }
            terminal
        })
    }

    fn place_mapped_axes(mut base: Vec<Dim>, mut mapped: Vec<(usize, DimVar, Dim)>) -> Vec<Dim> {
        let deferred = |var, resolved: Dim| match resolved {
            Dim::Var(_) => resolved,
            _ => Dim::Var(var),
        };
        mapped.sort_by_key(|(axis, var, _)| (*axis, var.0));
        if base.iter().any(|dim| matches!(dim, Dim::Rank(_))) {
            // An unresolved spread has no physical width yet. Keep the.
            //
            // compiler-owned identities present. row unification uses their.
            //
            // recorded axes rather than this storage order.
            //
            base.extend(
                mapped
                    .into_iter()
                    .map(|(_, var, resolved)| deferred(var, resolved)),
            );
            return base;
        }
        for (axis, var, resolved) in mapped {
            if axis <= base.len() {
                base.insert(axis, resolved);
            } else {
                // Bounds checking belongs to vmap inference/application. An.
                //
                // unresolved stored transform must remain representable until.
                //
                // that boundary reports the user-facing error.
                //
                base.push(deferred(var, resolved));
            }
        }
        base
    }

    fn apply_tensor_dims(&self, dims: &[Dim]) -> Vec<Dim> {
        let mut base = Vec::with_capacity(dims.len());
        let mut mapped = Vec::new();
        for dim in dims {
            match dim {
                Dim::Var(var) if self.mapped_axis(*var).is_some() => {
                    let axis = self.mapped_axis(*var).expect("guarded mapped axis");
                    let resolved = self.resolve_dvar(*var);
                    let resolved = if resolved == Dim::Var(*var) {
                        Dim::Var(*var)
                    } else {
                        self.apply_dim(&resolved)
                    };
                    mapped.push((axis, *var, resolved));
                }
                Dim::Rank(rank) => {
                    for resolved in self.resolve_rvar(*rank) {
                        base.push(self.apply_dim(&resolved));
                    }
                }
                _ => base.push(self.apply_dim(dim)),
            }
        }
        Self::place_mapped_axes(base, mapped)
    }

    fn apply_tensor_dims_excluding(
        &self,
        dims: &[Dim],
        quantified_dvars: &chelis_unord::UnordSet<DimVar>,
        quantified_rvars: &chelis_unord::UnordSet<RankVar>,
    ) -> Vec<Dim> {
        let mut base = Vec::with_capacity(dims.len());
        let mut mapped = Vec::new();
        for dim in dims {
            match dim {
                Dim::Var(var) if self.mapped_axis(*var).is_some() => {
                    let axis = self.mapped_axis(*var).expect("guarded mapped axis");
                    let resolved = self.apply_dim_excluding(dim, quantified_dvars);
                    mapped.push((axis, *var, resolved));
                }
                Dim::Var(var) if quantified_dvars.contains(var) => base.push(dim.clone()),
                Dim::Rank(var) if quantified_rvars.contains(var) => base.push(dim.clone()),
                Dim::Rank(var) => {
                    for resolved in self.resolve_rvar_excluding(*var, quantified_rvars) {
                        base.push(self.apply_dim_excluding(&resolved, quantified_dvars));
                    }
                }
                _ => base.push(self.apply_dim_excluding(dim, quantified_dvars)),
            }
        }
        Self::place_mapped_axes(base, mapped)
    }

    /// Apply this substitution to a type, resolving all bound variables.
    ///
    /// Path-compresses any chains of length ≥ 2 it encounters so future.
    ///
    /// lookups land in O(1).
    ///
    pub fn apply(&self, ty: &Type<E>) -> Type<E> {
        self.apply_with(ty, None)
    }

    /// [`Subst::apply`] for one of many types that share subtrees, such as.
    ///
    /// the types recorded for every node of one declaration: a node's type.
    ///
    /// shares its children with the types recorded for the nodes inside it.
    ///
    /// `memo` answers a shared child it applied, so applying to.
    ///
    /// every recorded type visits each distinct subtree once instead of once.
    ///
    /// per enclosing record (chelis#2622).
    ///
    ///
    /// The result is the one `apply` returns. The substitution must not.
    ///
    /// gain a binding while `memo` is in use. Path compression is `apply`'s.
    ///
    /// only effect on it, and compression leaves every variable on a resolved.
    ///
    /// chain pointing at that chain's end, whichever variables are resolved.
    ///
    /// first and however often, so the substitution ends in the state that.
    ///
    /// applying to each type in full would leave.
    ///
    pub fn apply_memoized(&self, ty: &Type<E>, memo: &mut ApplyMemo<E>) -> Type<E> {
        self.apply_with(ty, Some(memo))
    }

    fn apply_with(&self, ty: &Type<E>, memo: Option<&mut ApplyMemo<E>>) -> Type<E> {
        self.apply_tracked(ty, memo).0
    }

    /// Apply to `ty`, and say whether the result is ground and whether it is.
    ///
    /// `ty` itself, so a shared child that applying leaves as it was can be.
    ///
    /// returned rather than copied.
    ///
    fn apply_tracked(
        &self,
        ty: &Type<E>,
        mut memo: Option<&mut ApplyMemo<E>>,
    ) -> (Type<E>, Applied) {
        #[cfg(any(test, feature = "test-instrumentation"))]
        APPLY_VISITS.with(|visits| visits.set(visits.get() + 1));
        match ty {
            Type::Var(v) => match self.resolve_tvar(*v) {
                Type::Var(variable) => (
                    Type::Var(variable),
                    Applied {
                        ground: false,
                        same: variable == *v,
                    },
                ),
                bound => {
                    let (applied, outcome) = self.apply_tracked(&bound, memo);
                    (applied, outcome.changed())
                }
            },
            Type::Fn(args, ret) => {
                let (args, args_outcome) = self.apply_list(args, memo.as_deref_mut());
                let (ret, ret_outcome) = self.apply_child(ret, memo);
                (Type::Fn(args, ret), args_outcome.and(ret_outcome))
            }
            Type::Ref(inner) => {
                let (inner, outcome) = self.apply_child(inner, memo);
                (Type::Ref(inner), outcome)
            }
            Type::Tensor(dims, prec) => {
                let applied_dims = self.apply_tensor_dims(dims);
                let applied_prec = self.apply_tensor_prec(prec);
                let outcome = Applied {
                    ground: applied_dims.iter().all(dim_is_ground)
                        && matches!(applied_prec, TensorPrec::Concrete(_)),
                    same: applied_dims == *dims && applied_prec == *prec,
                };
                (Type::Tensor(applied_dims, applied_prec), outcome)
            }
            Type::Adt(name, args) => {
                let (args, outcome) = self.apply_list(args, memo);
                (Type::Adt(name.clone(), args), outcome)
            }
            Type::KindedAdt(name, args) => {
                let (args, outcome) = self.apply_arguments(args, memo);
                (Type::KindedAdt(name.clone(), args), outcome)
            }
            Type::Tuple(ts) => {
                let (ts, outcome) = self.apply_list(ts, memo);
                (Type::Tuple(ts), outcome)
            }
            Type::Prim(_) | Type::Unit | Type::Error(_) => (ty.clone(), Applied::GROUND),
        }
    }

    fn apply_child(
        &self,
        child: &Arc<Type<E>>,
        mut memo: Option<&mut ApplyMemo<E>>,
    ) -> (Arc<Type<E>>, Applied) {
        if self.ground_types.holds(Arc::as_ptr(child).addr()) {
            return (child.clone(), Applied::GROUND);
        }
        let key = Arc::as_ptr(child).addr();
        if let Some((_, applied, outcome)) =
            memo.as_deref().and_then(|memo| memo.children.get(&key))
        {
            return (applied.clone(), *outcome);
        }
        let (applied, outcome) = self.apply_tracked(child, memo.as_deref_mut());
        let applied = if outcome.same {
            child.clone()
        } else {
            Arc::new(applied)
        };
        if outcome.ground {
            self.ground_types
                .record(GroundShared::Child(applied.clone()));
        }
        if let Some(memo) = memo {
            memo.children
                .insert(key, (child.clone(), applied.clone(), outcome));
        }
        (applied, outcome)
    }

    fn apply_list(
        &self,
        items: &Arc<[Type<E>]>,
        mut memo: Option<&mut ApplyMemo<E>>,
    ) -> (Arc<[Type<E>]>, Applied) {
        if self
            .ground_types
            .holds(Arc::as_ptr(items).cast::<Type<E>>().addr())
        {
            return (items.clone(), Applied::GROUND);
        }
        let key = Arc::as_ptr(items).cast::<Type<E>>().addr();
        if let Some((_, applied, outcome)) = memo.as_deref().and_then(|memo| memo.lists.get(&key)) {
            return (applied.clone(), *outcome);
        }
        let mut outcome = Applied::GROUND;
        let applied_items = items
            .iter()
            .map(|item| {
                let (applied, item_outcome) = self.apply_tracked(item, memo.as_deref_mut());
                outcome = outcome.and(item_outcome);
                applied
            })
            .collect::<Vec<_>>();
        let applied = if outcome.same {
            items.clone()
        } else {
            Arc::from(applied_items)
        };
        if outcome.ground {
            self.ground_types
                .record(GroundShared::List(applied.clone()));
        }
        if let Some(memo) = memo {
            memo.lists
                .insert(key, (items.clone(), applied.clone(), outcome));
        }
        (applied, outcome)
    }

    fn apply_arguments(
        &self,
        arguments: &Arc<[NominalArg<E>]>,
        mut memo: Option<&mut ApplyMemo<E>>,
    ) -> (Arc<[NominalArg<E>]>, Applied) {
        if self
            .ground_types
            .holds(Arc::as_ptr(arguments).cast::<NominalArg<E>>().addr())
        {
            return (arguments.clone(), Applied::GROUND);
        }
        let key = Arc::as_ptr(arguments).cast::<NominalArg<E>>().addr();
        if let Some((_, applied, outcome)) =
            memo.as_deref().and_then(|memo| memo.arguments.get(&key))
        {
            return (applied.clone(), *outcome);
        }
        let mut outcome = Applied::GROUND;
        let applied_arguments = arguments
            .iter()
            .map(|argument| match argument {
                NominalArg::Type(ty) => {
                    let (applied, argument_outcome) = self.apply_tracked(ty, memo.as_deref_mut());
                    outcome = outcome.and(argument_outcome);
                    NominalArg::Type(applied)
                }
                NominalArg::Dimension(dim) => {
                    let applied = self.apply_dim(dim);
                    outcome = outcome.and(Applied {
                        ground: dim_is_ground(&applied),
                        same: applied == *dim,
                    });
                    NominalArg::Dimension(applied)
                }
            })
            .collect::<Vec<_>>();
        let applied = if outcome.same {
            arguments.clone()
        } else {
            Arc::from(applied_arguments)
        };
        if outcome.ground {
            self.ground_types
                .record(GroundShared::Arguments(applied.clone()));
        }
        if let Some(memo) = memo {
            memo.arguments
                .insert(key, (arguments.clone(), applied.clone(), outcome));
        }
        (applied, outcome)
    }

    /// Apply this substitution to a scheme body without substituting through.
    ///
    /// the scheme's universally quantified variables.
    ///
    ///
    /// A global inference substitution can contain bindings whose numeric IDs.
    ///
    /// coincide with a quantified variable in an environment scheme. Those.
    ///
    /// bindings belong to an instantiation, not to the scheme itself. Applying.
    ///
    /// them while computing the environment's free variables turns a.
    ///
    /// quantified dimension into a free one and prevents later.
    ///
    /// generalization (chelis#968).
    ///
    pub fn apply_scheme(&self, scheme: &Scheme<E>) -> Type<E> {
        let quantified_tvars = scheme.tvars.iter().copied().collect();
        let quantified_dvars = scheme.dvars.iter().copied().collect();
        let quantified_rvars = scheme.rvars.iter().copied().collect();
        self.apply_excluding(
            &scheme.body,
            &quantified_tvars,
            &quantified_dvars,
            &quantified_rvars,
        )
    }

    fn apply_excluding(
        &self,
        ty: &Type<E>,
        quantified_tvars: &chelis_unord::UnordSet<TypeVar>,
        quantified_dvars: &chelis_unord::UnordSet<DimVar>,
        quantified_rvars: &chelis_unord::UnordSet<RankVar>,
    ) -> Type<E> {
        match ty {
            Type::Var(v) if quantified_tvars.contains(v) => ty.clone(),
            Type::Var(v) => {
                let resolved = self.resolve_tvar_excluding(*v, quantified_tvars);
                if resolved == Type::Var(*v) {
                    resolved
                } else {
                    self.apply_excluding(
                        &resolved,
                        quantified_tvars,
                        quantified_dvars,
                        quantified_rvars,
                    )
                }
            }
            Type::Fn(args, ret) => Type::Fn(
                args.iter()
                    .map(|arg| {
                        self.apply_excluding(
                            arg,
                            quantified_tvars,
                            quantified_dvars,
                            quantified_rvars,
                        )
                    })
                    .collect(),
                Arc::new(self.apply_excluding(
                    ret,
                    quantified_tvars,
                    quantified_dvars,
                    quantified_rvars,
                )),
            ),
            Type::Ref(inner) => Type::reference(self.apply_excluding(
                inner,
                quantified_tvars,
                quantified_dvars,
                quantified_rvars,
            )),
            Type::Tensor(dims, prec) => {
                let resolved_dims =
                    self.apply_tensor_dims_excluding(dims, quantified_dvars, quantified_rvars);
                let resolved_prec = match prec {
                    TensorPrec::Var(var) if quantified_tvars.contains(var) => prec.clone(),
                    TensorPrec::Var(var) => {
                        match self.resolve_tvar_excluding(*var, quantified_tvars) {
                            Type::Prim(prim) => TensorPrec::Concrete(prim),
                            Type::Var(resolved) => TensorPrec::Var(resolved),
                            _ => prec.clone(),
                        }
                    }
                    TensorPrec::Concrete(_) => prec.clone(),
                };
                Type::Tensor(resolved_dims, resolved_prec)
            }
            Type::Adt(name, args) => Type::Adt(
                name.clone(),
                args.iter()
                    .map(|arg| {
                        self.apply_excluding(
                            arg,
                            quantified_tvars,
                            quantified_dvars,
                            quantified_rvars,
                        )
                    })
                    .collect(),
            ),
            Type::KindedAdt(name, args) => Type::KindedAdt(
                name.clone(),
                args.iter()
                    .map(|argument| match argument {
                        NominalArg::Type(ty) => NominalArg::Type(self.apply_excluding(
                            ty,
                            quantified_tvars,
                            quantified_dvars,
                            quantified_rvars,
                        )),
                        NominalArg::Dimension(Dim::Var(var)) if quantified_dvars.contains(var) => {
                            argument.clone()
                        }
                        NominalArg::Dimension(Dim::Rank(var)) if quantified_rvars.contains(var) => {
                            argument.clone()
                        }
                        NominalArg::Dimension(dim) => {
                            NominalArg::Dimension(self.apply_dim_excluding(dim, quantified_dvars))
                        }
                    })
                    .collect(),
            ),
            Type::Tuple(items) => Type::Tuple(
                items
                    .iter()
                    .map(|item| {
                        self.apply_excluding(
                            item,
                            quantified_tvars,
                            quantified_dvars,
                            quantified_rvars,
                        )
                    })
                    .collect(),
            ),
            Type::Prim(_) | Type::Unit | Type::Error(_) => ty.clone(),
        }
    }

    fn resolve_tvar_excluding(
        &self,
        start: TypeVar,
        quantified: &chelis_unord::UnordSet<TypeVar>,
    ) -> Type<E> {
        self.types.with(|map| {
            let mut current = start;
            let max_steps = map.len() + 1;
            for _ in 0..max_steps {
                if quantified.contains(&current) {
                    return Type::Var(current);
                }
                match map.get(&current) {
                    None => return Type::Var(current),
                    Some(Type::Var(next)) if *next != current => current = *next,
                    Some(Type::Var(_)) => return Type::Var(current),
                    Some(other) => return other.clone(),
                }
            }
            Type::Var(current)
        })
    }

    fn apply_dim_excluding(&self, dim: &Dim, quantified: &chelis_unord::UnordSet<DimVar>) -> Dim {
        let Dim::Var(start) = dim else {
            return dim.clone();
        };
        self.dims.with(|map| {
            let mut current = *start;
            let max_steps = map.len() + 1;
            for _ in 0..max_steps {
                if quantified.contains(&current) {
                    return Dim::Var(current);
                }
                match map.get(&current) {
                    None => return Dim::Var(current),
                    Some(Dim::Var(next)) if *next != current => current = *next,
                    Some(Dim::Var(_)) => return Dim::Var(current),
                    Some(other) => return other.clone(),
                }
            }
            Dim::Var(current)
        })
    }

    fn resolve_rvar_excluding(
        &self,
        start: RankVar,
        quantified: &chelis_unord::UnordSet<RankVar>,
    ) -> Vec<Dim> {
        self.ranks.with(|map| {
            let mut current = start;
            let max_steps = map.len() + 1;
            for _ in 0..max_steps {
                if quantified.contains(&current) {
                    return vec![Dim::Rank(current)];
                }
                match map.get(&current) {
                    None => return vec![Dim::Rank(current)],
                    Some(bound) => match bound.as_slice() {
                        [Dim::Rank(next)] if *next != current => current = *next,
                        [Dim::Rank(_)] => return vec![Dim::Rank(current)],
                        _ => return bound.clone(),
                    },
                }
            }
            vec![Dim::Rank(current)]
        })
    }

    /// Apply this substitution to a dimension. Path-compresses chains.
    ///
    pub fn apply_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => {
                if self.mapped_axis(*v).is_some() {
                    // The tensor-shape applicator must see the transform.
                    //
                    // boundary before it can place the resolved batch.
                    //
                    // identity at its requested axis.
                    //
                    return Dim::Var(*v);
                }
                let resolved = self.resolve_dvar(*v);
                if self.dimension_label(*v).is_some() && !matches!(resolved, Dim::Var(_)) {
                    // Keep the label's identity available to aliases and.
                    //
                    // name-sensitive operations, even after a concrete call.
                    //
                    Dim::Var(*v)
                } else {
                    resolved
                }
            }
            _ => dim.clone(),
        }
    }

    pub fn constraint_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => self.resolve_dvar(*v),
            _ => dim.clone(),
        }
    }

    /// Resolve the shape run currently assigned to a rank variable, preserving.
    ///
    /// a sole unbound/aliased `Dim::Rank` identity for rigidity checks.
    ///
    pub fn constraint_rank(&self, rank: RankVar) -> Vec<Dim> {
        self.resolve_rvar(rank)
            .into_iter()
            .map(|dim| self.constraint_dim(&dim))
            .collect()
    }

    pub fn observe_dim(&self, dim: &Dim) -> DimObservation {
        let constraint = self.constraint_dim(dim);
        let name = match dim {
            Dim::Var(v) => self.dimension_label(*v),
            Dim::Name(name) => Some(name.clone()),
            _ => None,
        }
        .or_else(|| match &constraint {
            Dim::Name(name) => Some(name.clone()),
            _ => None,
        });
        let protected = matches!(&constraint, Dim::Var(v) if self.is_protected_dimension(*v));
        DimObservation {
            constraint,
            name,
            protected,
        }
    }

    pub fn dimension_label(&self, v: DimVar) -> Option<String> {
        self.raw_dimension_annotation(v)
            .filter(|annotation| parse_mapped_axis_annotation(annotation).is_none())
    }

    fn raw_dimension_annotation(&self, v: DimVar) -> Option<String> {
        let root = self.resolve_dvar(v);
        self.dimension_labels.with(|labels| {
            labels
                .get(&v)
                .or_else(|| match root {
                    Dim::Var(root) => labels.get(&root),
                    _ => None,
                })
                .cloned()
        })
    }

    pub fn mapped_axis(&self, v: DimVar) -> Option<usize> {
        self.dimension_labels.with(|labels| {
            labels
                .get(&v)
                .and_then(|annotation| parse_mapped_axis_annotation(annotation))
        })
    }

    pub fn mark_mapped_axis(&self, var: DimVar, axis: usize) {
        self.set_dimension_label(var, mapped_axis_annotation(axis));
    }

    pub fn copy_dimension_label(&self, old: DimVar, fresh: DimVar) {
        if let Some(annotation) = self.raw_dimension_annotation(old) {
            self.set_dimension_label(fresh, annotation);
        }
    }

    fn set_dimension_label(&self, var: DimVar, label: String) {
        self.dimension_labels.with_detached(|labels| {
            if labels.get(&var) == Some(&label) {
                return;
            }
            let previous = labels.insert(var, label);
            if self.label_unification_depth != 0 {
                self.dimension_label_undo
                    .with(|undo| undo.push((var, previous)));
            }
        });
    }

    fn finish_dimension_label_writes(&mut self, rollback: bool) {
        self.dimension_label_undo.with_detached(|undo| {
            if rollback && !undo.is_empty() {
                self.dimension_labels.with(|labels| {
                    for (var, previous) in undo.drain(..).rev() {
                        if let Some(label) = previous {
                            labels.insert(var, label);
                        } else {
                            labels.remove(&var);
                        }
                    }
                });
            } else {
                undo.clear();
            }
        });
    }

    pub fn semantic_dim(&self, dim: &Dim) -> Dim {
        if let Dim::Var(v) = dim
            && self.mapped_axis(*v).is_none()
            && let Some(label) = self.dimension_label(*v)
        {
            return Dim::Name(label);
        }
        self.constraint_dim(dim)
    }

    /// Semantic annotation/name-query view. Never use this before the authored.
    ///
    /// result guards: projecting a labelled identity to Name erases rigidity.
    ///
    pub fn semantic_type(&self, ty: &Type<E>) -> Type<E> {
        self.semantic_view(&self.apply(ty))
    }

    /// The semantic view of a type `apply` resolved. `apply`.
    ///
    /// resolves every child, so the children need only the dimension.
    ///
    /// projection. applying again at each level would rebuild every subtree.
    ///
    /// once per enclosing level, quadratic in the type's depth (chelis#2622).
    ///
    fn semantic_view(&self, ty: &Type<E>) -> Type<E> {
        match ty {
            Type::Tensor(dims, prec) => Type::Tensor(
                dims.iter().map(|d| self.semantic_dim(d)).collect(),
                prec.clone(),
            ),
            Type::Fn(args, ret) => Type::Fn(
                args.iter().map(|a| self.semantic_view(a)).collect(),
                Arc::new(self.semantic_view(ret)),
            ),
            Type::Ref(inner) => Type::reference(self.semantic_view(inner)),
            Type::Tuple(items) => {
                Type::Tuple(items.iter().map(|t| self.semantic_view(t)).collect())
            }
            Type::Adt(name, args) => Type::Adt(
                name.clone(),
                args.iter().map(|t| self.semantic_view(t)).collect(),
            ),
            Type::KindedAdt(name, args) => Type::KindedAdt(
                name.clone(),
                args.iter()
                    .map(|a| match a {
                        NominalArg::Type(t) => NominalArg::Type(self.semantic_view(t)),
                        NominalArg::Dimension(d) => NominalArg::Dimension(self.semantic_dim(d)),
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    /// The variable that currently roots `v`'s alias class.
    ///
    ///
    /// A class already bound to a concrete dimension has no variable root and.
    ///
    /// answers `v` itself. That case never reaches the class flags: the.
    ///
    /// absorbing site skips it on `constraint_dim`'s answer first, because a.
    ///
    /// bound class denotes what it is bound to.
    ///
    fn dvar_class_root(&self, v: DimVar) -> DimVar {
        match self.resolve_dvar(v) {
            Dim::Var(root) => root,
            _ => v,
        }
    }

    /// chelis#1801: record that `v`'s alias class met a runtime extent `*`.
    ///
    /// without being bound to it. Written only by [`unify_dim`]'s.
    ///
    /// wildcard-against-variable arm. read only by the application rule that.
    ///
    /// absorbs the extent.
    ///
    pub fn note_wildcard_touch(&mut self, v: DimVar) {
        let root = self.dvar_class_root(v);
        self.wildcard_touched_classes.insert(root);
    }

    /// chelis#1801: whether `v`'s alias class met a runtime extent `*`.
    ///
    ///
    /// This answers only "did the meeting happen". Whether the class still.
    ///
    /// denotes that extent is a separate question the caller asks of the.
    ///
    /// substitution, because an argument in the same call may bound it.
    ///
    /// to a literal after the meeting.
    ///
    pub fn dvar_class_met_wildcard(&self, v: DimVar) -> bool {
        self.wildcard_touched_classes
            .contains(&self.dvar_class_root(v))
    }

    /// chelis#1801: whether a declared dimension pins `v`'s alias class.
    ///
    ///
    /// `def outer(s: tensor[seq, f32])` binds the class to the named.
    ///
    /// dimension `seq`, which is section 3.2's exclusion, and it reaches that.
    ///
    /// exclusion through the rigidity ledger rather than through a `Dim::Name`.
    ///
    /// unification: the name arm in `bind_dvar` records the name as a label.
    ///
    /// and binds nothing. The set carries the same answer for a class whose.
    ///
    /// protected member a merge or a compose demoted out of the root.
    ///
    pub fn dvar_class_is_binder_pinned(&self, v: DimVar) -> bool {
        let root = self.dvar_class_root(v);
        self.binder_pinned_classes.contains(&root) || self.protected_dimensions.contains(&root)
    }

    /// chelis#1801: carry both class flags from `from`'s class onto `into`'s.
    ///
    ///
    /// Called by [`bind_dvar`] before every merge. A merged class keeps one.
    ///
    /// root, so evidence recorded on the other root would otherwise become.
    ///
    /// unreachable, which is the defect chelis#1925's rounds 1 and 2 both.
    ///
    /// reported.
    ///
    fn merge_dvar_class_flags(&mut self, from: DimVar, into: DimVar) {
        let source = self.dvar_class_root(from);
        let target = self.dvar_class_root(into);
        if source == target {
            return;
        }
        if self.wildcard_touched_classes.contains(&source) {
            self.wildcard_touched_classes.insert(target);
        }
        if self.binder_pinned_classes.contains(&source)
            || self.protected_dimensions.contains(&source)
        {
            self.binder_pinned_classes.insert(target);
        }
    }

    /// chelis#1801: re-canonicalize both operands' class evidence through the.
    ///
    /// COMPOSED alias graph.
    ///
    ///
    /// Reached from [`Self::compose`], and a set union is the wrong operation.
    ///
    /// here. `compose_bindings` merges only the binding maps, so composition.
    ///
    /// can JOIN two classes that were separate in either operand: a flag.
    ///
    /// keyed to the variable that rooted its class before the compose is then.
    ///
    /// keyed to a member that no longer roots it, and the query on the new.
    ///
    /// root answers false. Every incoming key is therefore re-resolved.
    ///
    /// against the composed graph and re-keyed to the root it now has, which.
    ///
    /// is exactly what the label pass in [`Self::compose`] does and for the.
    ///
    /// same reason.
    ///
    ///
    /// Protection is folded in on the way through. The pin query reads.
    ///
    /// `protected_dimensions` at the root, so a protected member that the.
    ///
    /// compose demoted out of the root position would otherwise stop pinning.
    ///
    /// its class. Carrying the other operand's protection here is also the.
    ///
    /// conservative direction: it can only withhold an absorption, never.
    ///
    /// erase a claim.
    ///
    fn recanonicalize_class_flags(&mut self, left: &Subst<E>, right: &Subst<E>) {
        let mut touched: Vec<DimVar> = Vec::new();
        let mut pinned: Vec<DimVar> = Vec::new();
        for source in [left, right] {
            touched.extend(source.wildcard_touched_classes.to_sorted().iter().copied());
            pinned.extend(source.binder_pinned_classes.to_sorted().iter().copied());
            pinned.extend(source.protected_dimensions.to_sorted().iter().copied());
        }
        self.wildcard_touched_classes.clear();
        self.binder_pinned_classes.clear();
        for v in touched {
            let root = self.dvar_class_root(v);
            self.wildcard_touched_classes.insert(root);
        }
        for v in pinned {
            let root = self.dvar_class_root(v);
            self.binder_pinned_classes.insert(root);
        }
    }

    pub fn protect_dimensions(&mut self, vars: impl IntoIterator<Item = DimVar>) {
        for v in vars {
            self.protected_dimensions.insert(v);
        }
    }

    /// Current-check authored rigidity, never historical snapshot metadata.
    ///
    /// Instantiation freshens quantified IDs without copying protection.
    ///
    /// lexical captures keep their canonical ID. Both live and decoded imports.
    ///
    /// clear historical protection in `resume_for_new_check` before inference.
    ///
    pub fn is_protected_dimension(&self, var: DimVar) -> bool {
        matches!(self.resolve_dvar(var), Dim::Var(root) if self.protected_dimensions.contains(&root))
    }

    pub fn validate_dimension_labels(&self, var_gen: &VarGen) -> Result<(), &'static str> {
        for (var, label) in self
            .dimension_labels
            .with(|value| value.clone())
            .to_sorted()
        {
            if var.0 >= var_gen.watermarks().next_dvar {
                return Err("dimension label refers to an unallocated checker identity");
            }
            if parse_mapped_axis_annotation(label).is_some() {
                continue;
            }
            if label.is_empty() || label == "_" || label.chars().any(char::is_whitespace) {
                return Err("invalid dimension label in checker snapshot");
            }
        }
        Ok(())
    }

    fn type_binding_owner(&self, start: TypeVar) -> TypeVar {
        self.types.with(|map| {
            if let Some(owner) = map.resolved.get(&start) {
                return *owner;
            }
            let mut current = start;
            let mut chain = Vec::new();
            while let Some(Type::Var(next)) = map.get(&current) {
                if *next == current {
                    break;
                }
                chain.push(current);
                current = *next;
            }
            if chain.len() > 1 {
                for variable in chain {
                    map.resolved.insert(variable, current);
                }
            }
            current
        })
    }

    /// Apply this substitution to a tensor precision slot.
    ///
    ///
    /// Resolves a `TensorPrec::Var` through the type-variable.
    ///
    /// substitution chain. If the resolved binding is a concrete.
    ///
    /// primitive, the slot collapses to `TensorPrec::Concrete(_)`. If.
    ///
    /// the binding is itself another type variable, we keep the slot.
    ///
    /// as `TensorPrec::Var(target_var)` so the tensor stays.
    ///
    /// well-formed (the precision slot must be a precision, not an.
    ///
    /// arbitrary `Type`). Any non-prim, non-var binding (e.g. a.
    ///
    /// downstream unification error that bound the slot var to.
    ///
    /// `Type::Fn` or `Type::Tensor`) collapses to `Type::Error` at.
    ///
    /// the surrounding `Type` level — but here we conservatively.
    ///
    /// retain the original var. the unifier surfaces the mismatch.
    ///
    pub fn apply_tensor_prec(&self, prec: &TensorPrec) -> TensorPrec {
        match prec {
            TensorPrec::Concrete(_) => prec.clone(),
            TensorPrec::Var(v) => match self.resolve_tvar(*v) {
                Type::Prim(p) => TensorPrec::Concrete(p),
                Type::Var(v2) => TensorPrec::Var(v2),
                _ => prec.clone(),
            },
        }
    }

    /// Compose: apply `other` to all bindings in self, then merge.
    ///
    ///
    /// Composition is transactional because independent substitutions can.
    ///
    /// carry a binding and a semantic restriction for the same variable.
    ///
    /// Restrictions from both operands are canonicalized through the merged.
    ///
    /// binding graph. a forbidden concrete resolution rejects the complete.
    ///
    /// compose and leaves `self` unchanged.
    ///
    pub fn compose(&mut self, other: &Subst<E>) -> Result<(), TypeError> {
        let mut trial = self.clone();
        other.assert_complete();
        let trial = self.types.with_detached(|_| {
            trial.compose_bindings(other);
            // chelis#1801: `compose_bindings` merges the BINDINGS. the class.
            //
            // evidence is separate and has to be re-canonicalized through the.
            //
            // composed alias graph, never copied, because the compose can join.
            //
            // two classes that were separate in either operand.
            //
            trial.recanonicalize_class_flags(self, other);
            // Re-canonicalize BOTH operands' labels through the composed alias.
            //
            // graph. merely copying incoming entries misses a newly joined class.
            //
            let mut incoming = Vec::new();
            let mut mapped_axes = Vec::new();
            for source in [&*self, other] {
                for (v, annotation) in source
                    .dimension_labels
                    .with(|value| value.clone())
                    .to_sorted()
                {
                    if parse_mapped_axis_annotation(annotation).is_some() {
                        mapped_axes.push((*v, annotation.clone()));
                    } else {
                        incoming.push((*v, annotation.clone()));
                    }
                }
            }
            trial.dimension_labels.with(|value| value.clear());
            for (v, label) in incoming {
                let target = trial.constraint_dim(&Dim::Var(v));
                trial.dimension_labels.with_detached(|labels| {
                    let root = if let Dim::Var(root) = target { root } else { v };
                    if matches!(&target, Dim::Name(name) if name != &label)
                        || [v, root]
                            .iter()
                            .any(|id| labels.get(id).is_some_and(|old| old != &label))
                    {
                        return Err(TypeError {
                            kind: TypeErrorKind::DimensionMismatch,
                            message: format!("conflicting dimension labels while composing {v:?}"),
                        });
                    }
                    labels.insert(v, label.clone());
                    labels.insert(root, label);
                    Ok(())
                })?;
            }
            for (v, annotation) in mapped_axes {
                trial.dimension_labels.with_detached(|labels| {
                    if let Some(existing) = labels.get(&v)
                        && existing != &annotation
                    {
                        return Err(TypeError {
                            kind: TypeErrorKind::DimensionMismatch,
                            message: format!(
                                "conflicting mapped-axis annotations while composing {v:?}"
                            ),
                        });
                    }
                    labels.insert(v, annotation);
                    Ok(())
                })?;
            }

            let mut restrictions = self.tvar_restrictions_snapshot();
            for (var, incoming) in other.tvar_restrictions_snapshot().into_sorted() {
                let narrowed = match restrictions.get(&var).copied() {
                    Some(existing) => merge_tvar_restrictions(existing, incoming)?,
                    None => incoming,
                };
                restrictions.insert(var, narrowed);
            }

            trial.tvar_restrictions.with(|value| value.clear());
            for (source, restriction) in restrictions.into_sorted() {
                let resolved = trial.resolve_tvar(source);
                ensure_tvar_restriction(restriction, &resolved, &trial)?;
                if let Type::Var(target) = resolved {
                    trial.narrow_tvar_restriction(target, restriction)?;
                }
            }

            // [04-LIN-10]: both operands' key-free marks hold in the composed.
            //
            // graph, so each is re-checked against what its variable now resolves.
            //
            // to, exactly like a dtype family above.
            //
            let incoming_marks = other
                .key_free_tvars
                .with(|value| value.clone())
                .into_sorted();
            for (source, origin) in incoming_marks {
                trial.forbid_key_instantiation(source, origin);
            }
            let marks = trial
                .key_free_tvars
                .with(|value| value.clone())
                .into_sorted();
            for (source, origin) in marks {
                let resolved = trial.resolve_tvar(source);
                ensure_key_free(&origin, &resolved, &resolved, &trial)?;
            }
            trial
                .key_carrying_adts
                .extend(other.key_carrying_adts.to_sorted().into_iter().cloned());
            Ok(trial)
        })?;

        *self = trial;
        Ok(())
    }

    /// Project semantic domains from an inferred implementation type onto a.
    ///
    /// structurally corresponding declared type.
    ///
    ///
    /// Most declared definitions use ordinary unification, which transfers.
    ///
    /// restrictions while binding matching variables. A small set of exact.
    ///
    /// stdlib contracts deliberately retains its declared type when unrelated.
    ///
    /// shape relations are not yet procedurally inferable. This projection.
    ///
    /// preserves the semantic domains learned while checking those bodies.
    ///
    /// without weakening them into an unconstrained exported function value.
    ///
    /// The operation is transactional so a later conflicting slot cannot.
    ///
    /// publish an earlier partial transfer.
    ///
    pub fn project_tvar_restrictions(
        &mut self,
        inferred: &Type<E>,
        declared: &Type<E>,
    ) -> Result<(), TypeError> {
        let mut trial = self.clone();
        trial.project_tvar_restrictions_inner(inferred, declared)?;
        *self = trial;
        Ok(())
    }

    fn project_tvar_restrictions_inner(
        &mut self,
        inferred: &Type<E>,
        declared: &Type<E>,
    ) -> Result<(), TypeError> {
        let inferred = self.apply(inferred);
        let declared = self.apply(declared);
        match (&inferred, &declared) {
            (Type::Var(source), target) => {
                let Some(restriction) = self.tvar_restriction(*source) else {
                    return Ok(());
                };
                ensure_tvar_restriction(restriction, target, self)?;
                if let Type::Var(target) = target {
                    self.narrow_tvar_restriction(*target, restriction)?;
                }
                Ok(())
            }
            (Type::Fn(inferred_args, inferred_ret), Type::Fn(declared_args, declared_ret))
                if inferred_args.len() == declared_args.len() =>
            {
                for (inferred, declared) in inferred_args.iter().zip(declared_args.iter()) {
                    self.project_tvar_restrictions_inner(inferred, declared)?;
                }
                self.project_tvar_restrictions_inner(inferred_ret, declared_ret)
            }
            (Type::Ref(inferred), Type::Ref(declared)) => {
                self.project_tvar_restrictions_inner(inferred, declared)
            }
            (
                Type::Tensor(inferred_dims, inferred_prec),
                Type::Tensor(declared_dims, declared_prec),
            ) if inferred_dims.len() == declared_dims.len() => {
                let inferred = match inferred_prec {
                    TensorPrec::Concrete(prim) => Type::Prim(*prim),
                    TensorPrec::Var(var) => Type::Var(*var),
                };
                let declared = match declared_prec {
                    TensorPrec::Concrete(prim) => Type::Prim(*prim),
                    TensorPrec::Var(var) => Type::Var(*var),
                };
                self.project_tvar_restrictions_inner(&inferred, &declared)
            }
            (Type::Adt(inferred_name, inferred_args), Type::Adt(declared_name, declared_args))
                if inferred_name == declared_name && inferred_args.len() == declared_args.len() =>
            {
                for (inferred, declared) in inferred_args.iter().zip(declared_args.iter()) {
                    self.project_tvar_restrictions_inner(inferred, declared)?;
                }
                Ok(())
            }
            (
                Type::KindedAdt(inferred_name, inferred_args),
                Type::KindedAdt(declared_name, declared_args),
            ) if inferred_name == declared_name && inferred_args.len() == declared_args.len() => {
                for (inferred, declared) in inferred_args.iter().zip(declared_args.iter()) {
                    if let (NominalArg::Type(inferred), NominalArg::Type(declared)) =
                        (inferred, declared)
                    {
                        self.project_tvar_restrictions_inner(inferred, declared)?;
                    }
                }
                Ok(())
            }
            (Type::Tuple(inferred), Type::Tuple(declared)) if inferred.len() == declared.len() => {
                for (inferred, declared) in inferred.iter().zip(declared.iter()) {
                    self.project_tvar_restrictions_inner(inferred, declared)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn compose_bindings(&mut self, other: &Subst<E>) {
        // Note: `other.apply` / `other.apply_dim` lock `other`'s maps.
        //
        // we must not hold a lock on `other` simultaneously. This.
        //
        // helper runs on a clone, so it is distinct even for `s.compose(&s)`.
        //
        {
            self.types.with_detached(|self_types| {
                let vars = self_types
                    .to_sorted()
                    .into_iter()
                    .map(|(var, _)| *var)
                    .collect::<Vec<_>>();
                for var in vars {
                    let val = self_types
                        .get_mut(&var)
                        .expect("collected type variable remains present");
                    *val = other.apply(val);
                }
            })
        }
        {
            self.dims.with_detached(|self_dims| {
                let vars = self_dims
                    .to_sorted()
                    .into_iter()
                    .map(|(var, _)| *var)
                    .collect::<Vec<_>>();
                for var in vars {
                    let val = self_dims
                        .get_mut(&var)
                        .expect("collected dimension variable remains present");
                    *val = other.constraint_dim(val);
                }
            })
        }
        {
            // Rank bindings are `Dim::Rank`-free runs (enforced by `bind_rvar`),.
            //
            // so applying `other` is a per-dim `apply_dim`. Omitting this would.
            //
            // silently drop rank substitutions through a compose — a Tier-3.
            //
            // footgun since ranks are now load-bearing.
            self.ranks.with_detached(|self_ranks| {
                let vars = self_ranks
                    .to_sorted()
                    .into_iter()
                    .map(|(var, _)| *var)
                    .collect::<Vec<_>>();
                for var in vars {
                    let val = self_ranks
                        .get_mut(&var)
                        .expect("collected rank variable remains present");
                    *val = val.iter().map(|d| other.apply_dim(d)).collect();
                }
            })
        }
        let other_types = other.types_snapshot();
        let other_dims = other.dims_snapshot();
        let other_ranks = other.ranks_snapshot();
        {
            self.types.with_detached(|self_types| {
                for (k, v) in other_types.into_sorted() {
                    self_types.get_or_insert_with(k, || v);
                }
            })
        }
        {
            self.dims.with_detached(|self_dims| {
                for (k, v) in other_dims.into_sorted() {
                    self_dims.get_or_insert_with(k, || v);
                }
            })
        }
        {
            self.ranks.with_detached(|self_ranks| {
                for (k, v) in other_ranks.into_sorted() {
                    self_ranks.get_or_insert_with(k, || v);
                }
            })
        }
        // `deferred_borrow_vars` is intentionally NOT merged: it is a transient.
        //
        // per-def ledger (issue #256), drained after each body's inference, not.
        //
        // part of the substitution's logical content.
        //
    }
}

/// Combine the bounds of two variables being identified.
///
///
/// [04-DTYPE-2] makes this the family intersection rather than equality: a.
///
/// `Numeric`-bounded variable may legitimately be identified with a `Float`.
///
/// one, and the result admits only floats. Only `Float` against `Int` is.
///
/// empty, and an empty intersection is a `PrecisionMismatch` naming both.
///
/// families.
///
/// How a clash between two bounds is named.
///
///
/// The all-families form is byte-identical to the wording that predates.
///
/// §5.9's set form, which a per-operand `bound_description()` would not be:.
///
/// the original sentence hoists "dtype families" as a shared plural, so.
///
/// describing each side separately doubles the phrase. Nothing pinned that.
///
/// text, so it churned silently until red-team round 3 measured it.
///
/// differentially against the base.
///
fn describe_bound_clash(left: TypeVarRestriction, right: TypeVarRestriction) -> String {
    if left.is_family() && right.is_family() {
        return format!(
            "dtype families `{}` and `{}`",
            left.bound_spelling(),
            right.bound_spelling()
        );
    }
    format!(
        "{} and {}",
        left.bound_description(),
        right.bound_description()
    )
}

fn merge_tvar_restrictions(
    existing: TypeVarRestriction,
    incoming: TypeVarRestriction,
) -> Result<TypeVarRestriction, TypeError> {
    existing.intersect(incoming).ok_or_else(|| TypeError {
        kind: TypeErrorKind::DtypeFamilyMismatch,
        message: format!(
            "{} share no active dtype, so the type variables they bound cannot be the same type",
            describe_bound_clash(existing, incoming)
        ),
    })
}
/// Unify two types, producing a substitution or a type error.
///
pub fn unify<E: Clone + PartialEq>(
    t1: &Type<E>,
    t2: &Type<E>,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    let outer = subst.label_unification_depth == 0;
    subst.label_unification_depth += 1;
    let result = unify_preserving_owners(t1, t2, subst);
    subst.label_unification_depth -= 1;
    if outer {
        subst.finish_dimension_label_writes(result.is_err());
        if result.is_err() {
            subst.types.with_detached(|types| {
                for (v, old) in subst.refined_type_bindings.drain(..).rev() {
                    types.insert(v, old);
                }
            })
        } else {
            subst.refined_type_bindings.clear();
        }
    }
    result
}

fn unify_preserving_owners<E: Clone + PartialEq>(
    t1: &Type<E>,
    t2: &Type<E>,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    let owner1 = if let Type::Var(v) = t1 {
        Some(subst.type_binding_owner(*v))
    } else {
        None
    };
    let owner2 = if let Type::Var(v) = t2 {
        Some(subst.type_binding_owner(*v))
    } else {
        None
    };
    // Resolve only the outer variable. Recursion must still see a shared T.
    //
    // under Ref/Fn rather than an eagerly copied tensor payload.
    //
    let t1 = if matches!(t1, Type::Var(_)) {
        subst.apply(t1)
    } else {
        t1.clone()
    };
    let t2 = if matches!(t2, Type::Var(_)) {
        subst.apply(t2)
    } else {
        t2.clone()
    };
    let result = unify_resolved(&t1, &t2, subst);
    if result.is_ok() {
        for (owner, old, other) in [(owner1, &t1, &t2), (owner2, &t2, &t1)] {
            if let (Some(owner), Type::Tensor(dims, prec), Type::Tensor(other_dims, _)) =
                (owner, old, other)
                && dims.len() == other_dims.len()
            {
                let refined: Vec<_> = dims
                    .iter()
                    .zip(other_dims)
                    .map(|(d, other)| {
                        if matches!(d, Dim::Name(_))
                            && matches!(other, Dim::Var(v) if subst.dimension_label(*v).is_some())
                        {
                            subst.apply_dim(other)
                        } else {
                            d.clone()
                        }
                    })
                    .collect();
                if &refined != dims {
                    subst.refined_type_bindings.push((owner, old.clone()));
                    subst
                        .types
                        .with(|value| value.insert(owner, Type::Tensor(refined, prec.clone())));
                }
            }
        }
    }
    result
}

fn unify_resolved<E: Clone + PartialEq>(
    t1: &Type<E>,
    t2: &Type<E>,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    match (&t1, &t2) {
        // Same type — trivially unified.
        //
        (Type::Prim(p1), Type::Prim(p2)) if p1 == p2 => Ok(()),
        (Type::Prim(p1), Type::Prim(p2)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!("precision mismatch: {} vs {}", p1.name(), p2.name()),
        }),

        (Type::Unit, Type::Unit) => Ok(()),

        (Type::Ref(inner1), Type::Ref(inner2)) => unify(inner1, inner2, subst),

        // Type variable binding.
        //
        (Type::Var(v), _) => bind_tvar(*v, t2, subst),
        (_, Type::Var(v)) => bind_tvar(*v, t1, subst),

        // Function types.
        //
        (Type::Fn(args1, ret1), Type::Fn(args2, ret2)) => {
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "function arity mismatch: {} args vs {} args",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify(a1, a2, subst)?;
            }
            unify(ret1, ret2, subst)?;
            // chelis#339: after the signature's rank spreads are bound, a.
            //
            // result-introduced axis name (a named-axis expand insert) must.
            //
            // not also be covered by a spread binding — the monomorphized.
            //
            // result row would carry the same dim name twice with two.
            //
            // different extents, making every later by-name lookup ambiguous.
            //
            check_introduced_name_rank_collision(args1, ret1, subst)?;
            check_introduced_name_rank_collision(args2, ret2, subst)
        }

        // Tensor types.
        //
        (Type::Tensor(dims1, p1), Type::Tensor(dims2, p2)) => {
            unify_tensor_prec(p1, p2, subst)?;
            // Rank polymorphism: a `Dim::Rank` is a *name-preserving spread*.
            //
            // standing for a run of dims. A shape is `Rank? (Name Rank?)*`.
            //
            // (no two spreads adjacent). Resolve both shapes (expanding any.
            //
            // bound spread to its run), then dispatch on how many spreads each.
            //
            // side carries. With at most one spread per gap and a named anchor.
            //
            // find each interior split, unification stays unitary — there.
            //
            // is one most-general binding. See.
            //
            // `spec/design/rank_polymorphism.md` §Unification.
            //
            let d1 = resolve_shape(dims1, subst);
            let d2 = resolve_shape(dims2, subst);
            let s1 = d1.iter().filter(|d| matches!(d, Dim::Rank(_))).count();
            let s2 = d2.iter().filter(|d| matches!(d, Dim::Rank(_))).count();
            match (s1, s2) {
                // Both ground (Tier-1 / fully-monomorphic): length + element-wise.
                //
                (0, 0) => {
                    let d1 = canonicalize_ground_mapped_axes(&d1, subst)?;
                    let d2 = canonicalize_ground_mapped_axes(&d2, subst)?;
                    if d1.len() != d2.len() {
                        return Err(TypeError {
                            kind: TypeErrorKind::DimensionMismatch,
                            message: format!(
                                "tensor rank mismatch: {} dims vs {} dims",
                                d1.len(),
                                d2.len()
                            ),
                        });
                    }
                    for (a, b) in d1.iter().zip(d2.iter()) {
                        unify_dim(a, b, subst)?;
                    }
                    Ok(())
                }
                // Exactly one side carries spreads: split the ground side.
                //
                (_, 0) => unify_row_against_ground_with_mapped_axes(&d1, &d2, subst),
                (0, _) => unify_row_against_ground_with_mapped_axes(&d2, &d1, subst),
                // Both carry spreads: only structurally-identical rows unify.
                //
                (_, _) => unify_rows_with_mapped_axes(&d1, &d2, subst),
            }
        }

        // ADT types.
        //
        (Type::Adt(n1, args1), Type::Adt(n2, args2)) => {
            if n1 != n2 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch,
                    message: format!("type mismatch: {n1} vs {n2}"),
                });
            }
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "ADT type argument count mismatch for {n1}: {} vs {}",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify(a1, a2, subst)?;
            }
            Ok(())
        }
        (Type::KindedAdt(n1, args1), Type::KindedAdt(n2, args2)) => {
            if n1 != n2 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch,
                    message: format!("type mismatch: {n1} vs {n2}"),
                });
            }
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "ADT type argument count mismatch for {n1}: {} vs {}",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                match (a1, a2) {
                    (NominalArg::Type(t1), NominalArg::Type(t2)) => unify(t1, t2, subst)?,
                    (NominalArg::Dimension(d1), NominalArg::Dimension(d2)) => {
                        unify_dim(d1, d2, subst)?
                    }
                    _ => {
                        return Err(TypeError {
                            kind: TypeErrorKind::TypeMismatch,
                            message: format!("nominal argument kind mismatch for {n1}"),
                        });
                    }
                }
            }
            Ok(())
        }

        // Tuple types.
        //
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            if ts1.len() != ts2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!("tuple length mismatch: {} vs {}", ts1.len(), ts2.len()),
                });
            }
            for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                unify(t1, t2, subst)?;
            }
            Ok(())
        }

        // Error propagation: unifying with Error trivially succeeds so.
        //
        // downstream call sites do not fan out a cascade of secondary.
        //
        // diagnostics from a single upstream error. Call sites that.
        //
        // need a precision/shape match against a non-Error declared.
        //
        // type must check for the Error sentinel themselves and surface.
        //
        // the mismatch explicitly (e.g. the def-body vs declared-sig.
        //
        // unify in `infer.rs` does this for WS-A5 RT-3a F1: when the.
        //
        // body collapses to Error but the declared type is concrete, we.
        //
        // still emit a declared-versus-inferred mismatch so the user sees.
        //
        // the unresolved declared shape).
        //
        (Type::Error(_), _) | (_, Type::Error(_)) => Ok(()),

        // Everything else is a mismatch.
        //
        _ => Err(TypeError {
            kind: TypeErrorKind::TypeMismatch,
            message: format!("type mismatch: {t1} vs {t2}"),
        }),
    }
}

/// Unify two tensor precision slots per `spec/04-type-system.md` §5.8.
///
/// (WS-A5 precision polymorphism).
///
///
/// - `Concrete(p1)` and `Concrete(p2)` unify only when `p1 == p2`.
///
/// - `Var(v)` unifies with `Concrete(p)` by binding `v` to `Type::Prim(p)`.
///
///   in the substitution. further references through that var resolve to.
///
///   the concrete prim via `Subst::apply_tensor_prec`.
///
/// - Two `Var`s unify through the ordinary type unifier, preserving any.
///
///   binding already established in an ordinary type position.
///
pub fn unify_tensor_prec<E: Clone + PartialEq>(
    p1: &TensorPrec,
    p2: &TensorPrec,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    // A precision variable is also an ordinary type variable ([04-INF-6]).
    //
    // Resolve its complete binding before treating it as a free precision:.
    //
    // `apply_tensor_prec` deliberately retains a variable bound to a composite.
    //
    // type, which would let `bind_tvar` overwrite that binding below.
    //
    let resolve = |prec: &TensorPrec| -> Result<TensorPrec, TypeError> {
        match prec {
            TensorPrec::Concrete(_) => Ok(prec.clone()),
            TensorPrec::Var(v) => match subst.apply(&Type::Var(*v)) {
                Type::Prim(p) => Ok(TensorPrec::Concrete(p)),
                Type::Var(root) => Ok(TensorPrec::Var(root)),
                other => Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch,
                    message: format!("tensor precision must be a primitive, found {other}"),
                }),
            },
        }
    };
    let p1 = resolve(p1)?;
    let p2 = resolve(p2)?;
    match (&p1, &p2) {
        (TensorPrec::Concrete(a), TensorPrec::Concrete(b)) if a == b => Ok(()),
        (TensorPrec::Concrete(a), TensorPrec::Concrete(b)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: match crate::decisions::shape::sum_result_widening_note::<E>(
                None,
                &Type::Prim(*a),
                &Type::Prim(*b),
            ) {
                Some(note) => {
                    format!(
                        "tensor precision mismatch: {} vs {}; {note}",
                        a.name(),
                        b.name()
                    )
                }
                None => format!("tensor precision mismatch: {} vs {}", a.name(), b.name()),
            },
        }),
        (TensorPrec::Var(v), TensorPrec::Concrete(p)) => {
            unify(&Type::Var(*v), &Type::Prim(*p), subst)
        }
        (TensorPrec::Concrete(p), TensorPrec::Var(v)) => {
            unify(&Type::Prim(*p), &Type::Var(*v), subst)
        }
        (TensorPrec::Var(v1), TensorPrec::Var(v2)) => {
            if v1 == v2 {
                Ok(())
            } else {
                unify(&Type::Var(*v1), &Type::Var(*v2), subst)
            }
        }
    }
}

/// Unify two dimensions.
///
///
/// Wildcard ↔ Var invariant: `unify_dim(Wildcard, Var(v))` succeeds.
///
/// without binding `v`. The `(Wildcard, _) | (_, Wildcard) => Ok(())`.
///
/// arm matches before the `(Var(v), _) => bind_dvar(...)` arm and.
///
/// returns `Ok(())` with no side effects, so the dim var stays free.
///
/// This is intentional — binding `v := Wildcard` would freeze `v` and.
///
/// make a later concrete arg in the same sig unable to constrain it.
///
/// (`apply_dim` would resolve `Var(v) → Wildcard` and the permissive.
///
/// Wildcard arm would silently accept any value). Leaving `v` free.
///
/// lets a concrete arg in any later position bind it, after which a.
///
/// different concrete value trips the `Lit ↔ Lit` mismatch as the sig.
///
/// demands.
///
///
/// Name ↔ Lit invariant (issue Chelis-Lang/chelis#219, Option A): a.
///
/// concrete-but-named slot (`Dim::Name("batch")`) accepts a concrete.
///
/// literal (`Dim::Lit(2)`) at the call site without binding any.
///
/// substitution. Names are preserved in diagnostics. they do not.
///
/// impose a distinct-from-literal constraint. This eliminates the.
///
/// asymmetry whereby `Var <-> Lit` was accepted at call sites but.
///
/// `Name <-> Lit` was rejected, which blocked stdlib sigs like.
///
/// `def f(x: tensor[batch, hidden, f32])` from being called with.
///
/// concrete-shaped inputs (e.g. `f(to_tensor([[1.0, 2.0, 3.0]], f32))`).
///
/// The relaxation is narrow: distinct `Name <-> Name` and.
///
/// distinct `Lit <-> Lit` continue to be rejected, and the.
///
/// `Var <-> Lit` cross-position contract is unaffected.
///
pub fn unify_dim<E: Clone + PartialEq>(
    d1: &Dim,
    d2: &Dim,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    let name1 = subst.semantic_dim(d1);
    let name2 = subst.semantic_dim(d2);
    if let (Dim::Name(a), Dim::Name(b)) = (&name1, &name2)
        && a != b
    {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!("dimension mismatch: {name1:?} vs {name2:?}"),
        });
    }
    let d1 = subst.constraint_dim(d1);
    let d2 = subst.constraint_dim(d2);

    match (&d1, &d2) {
        (Dim::Name(n1), Dim::Name(n2)) if n1 == n2 => Ok(()),
        (Dim::Lit(l1), Dim::Lit(l2)) if l1 == l2 => Ok(()),
        // chelis#1801 / spec/04-type-system.md section 3.2: record the.
        //
        // meeting before the permissive arm swallows it. The variable is.
        //
        // still deliberately left FREE here (the Wildcard/Var invariant.
        //
        // above), so a later concrete argument in the same signature can.
        //
        // constrain it. `infer_app` decides what an unconstrained one.
        //
        // denotes once the whole call has unified.
        //
        (Dim::Wildcard, Dim::Var(v)) | (Dim::Var(v), Dim::Wildcard) => {
            subst.note_wildcard_touch(*v);
            Ok(())
        }
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => Ok(()),
        // Issue #219 Option A: Name and Lit unify without binding any.
        //
        // substitution. The Name carries a label for diagnostics, the.
        //
        // Lit carries the concrete value. nothing flows into `subst`.
        //
        (Dim::Name(_), Dim::Lit(_)) | (Dim::Lit(_), Dim::Name(_)) => Ok(()),
        (Dim::Var(v), _) => bind_dvar(*v, &d2, subst),
        (_, Dim::Var(v)) => bind_dvar(*v, &d1, subst),
        _ => Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!("dimension mismatch: {d1:?} vs {d2:?}"),
        }),
    }
}

/// Bind `v`, then settle every operand constraint that waited on it.
///
/// (chelis#1489).
///
///
/// The wrapper exists so discharge cannot be skipped: `bind_tvar_inner` has.
///
/// more than one success path, and an earlier design that decided these.
///
/// constraints in a separate end-of-inference pass was order-dependent in.
///
/// exactly the way this issue is about. Binding a variable is the event that.
///
/// makes a suspended decision decidable, so that is where the decision is.
///
/// made, and nothing schedules it.
///
fn bind_tvar<E: Clone + PartialEq>(
    v: TypeVar,
    ty: &Type<E>,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    bind_tvar_inner(v, ty, subst)?;
    discharge_operand_gates(v, subst);
    discharge_bounded_scalar_casts(subst);
    Ok(())
}

/// Settle the suspended scalar `cast`/`cast_trunc` gates that a binding has.
///
/// made decidable without binding their operand to a type (chelis#2151).
///
///
/// A binding is also the event through which a variable GAINS a declared.
///
/// dtype-family bound: identified with a bounded variable, or with a rigid.
///
/// authored binder. Such a variable may never be bound to a concrete type, so.
///
/// [`discharge_operand_gates`] would only carry its gate from one variable to.
///
/// the next, and the per-def pass would reject a well-typed cast. [04-DTYPE-2].
///
/// makes the bound enough to decide it: the operand is a scalar primitive of.
///
/// that family at every instantiation.
///
///
/// The decision is [`crate::decisions::cast::bounded_scalar_cast_result`].
///
/// This is its only call site. A cast whose operand already carries the bound.
///
/// is suspended like any other, and it settles at the next binding, which the.
///
/// suspended cast's own result variable guarantees. The cast and the bound are.
///
/// therefore settled together by whichever arrives second, and the verdict.
///
/// cannot depend on which operand inference visits first. Only what later.
///
/// bindings cannot change is decided here (every acceptance, and a rejection.
///
/// that reads only the target). a gate it declines stays suspended.
///
fn discharge_bounded_scalar_casts<E: Clone + PartialEq>(subst: &mut Subst<E>) {
    for (gate, decision) in subst.take_bounded_scalar_cast_gates() {
        let (DeferredOperandGate::Cast { ref result, .. }
        | DeferredOperandGate::CastToBinder { ref result, .. }) = gate
        else {
            continue;
        };
        match decision {
            Ok(settled) => gate.reconcile_result(result, settled, subst),
            Err(error) => subst.record_operand_gate_failure(OperandGateFailure::Decision {
                error: gate.attach_location(*error),
            }),
        }
    }
}

/// Settle the constraints suspended on `v`, now that it is bound.
///
///
/// Reached only from [`bind_tvar`]. Records failures rather than returning.
///
/// them: a discharge failure is a diagnostic about the program, not a.
///
/// unification error, and turning it into one would abort the surrounding.
///
/// unification and lose every other constraint waiting on this binding.
///
fn discharge_operand_gates<E: Clone + PartialEq>(v: TypeVar, subst: &mut Subst<E>) {
    let gates = subst.take_operand_gates_on(v);
    if gates.is_empty() {
        return;
    }
    let resolved = subst.apply(&Type::Var(v));
    for gate in gates {
        // Still a variable: `v` was identified with another variable rather.
        //
        // than given a type. Carry the obligation across so it discharges when.
        //
        // THAT variable binds.
        //
        if let Type::Var(target) = resolved {
            if target != v {
                subst.realias_operand_gate(target, gate);
                continue;
            }
            // Bound to itself is not a binding. leave it suspended for the.
            //
            // per-def pass to report as never-resolved.
            //
            subst.realias_operand_gate(v, gate);
            continue;
        }
        gate.discharge(&resolved, subst);
    }
}

fn bind_tvar_inner<E: Clone + PartialEq>(
    v: TypeVar,
    ty: &Type<E>,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    // A caller must never replace an established binding. Reconcile through.
    //
    // the ordinary unifier so aliases and composite types retain their.
    //
    // existing constraints even if a new binding path is added later.
    //
    let existing = subst.resolve_tvar(v);
    if existing != Type::Var(v) {
        return unify(&existing, ty, subst);
    }
    if let Type::Var(v2) = ty
        && *v2 == v
    {
        return Ok(()); // same variable
    }
    if occurs_in(v, ty, subst) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!("infinite type: ?{} occurs in {ty}", v.0),
        });
    }

    let source_restriction = subst.tvar_restriction(v);
    let target_restriction = match ty {
        Type::Var(target) => subst.tvar_restriction(*target),
        _ => None,
    };
    // Both endpoints keep their bound: the identified variable admits only.
    //
    // the dtypes both families admit ([04-DTYPE-2]). `.or()` would silently.
    //
    // discard the narrower of the two.
    //
    let merged_restriction = match (source_restriction, target_restriction) {
        (Some(source), Some(target)) => Some(merge_tvar_restrictions(source, target)?),
        (found, None) | (None, found) => found,
    };
    if let Some(restriction) = source_restriction {
        ensure_tvar_restriction(restriction, ty, subst)?;
    }
    if let Some(origin) = subst.key_free_origin(v) {
        ensure_key_free(&origin, ty, ty, subst)?;
    }

    // An older variable that becomes bound to a younger composite makes all.
    //
    // reachable variables part of the older scope.
    //
    let target_level = subst.level_of_tvar(v);
    subst.lower_type_to(ty, target_level);
    subst.record_validated_type_binding(v, ty.clone());
    subst.remove_tvar_restriction(v);
    if let (Some(restriction), Type::Var(target)) = (merged_restriction, ty) {
        subst.narrow_tvar_restriction(*target, restriction)?;
    }
    Ok(())
}

/// [04-LIN-10]: check that a function's type parameter is not instantiated at.
///
/// a key-carrying type (spec/04 section 8.4.1), and carry the rule onto every.
///
/// variable left inside `ty`, which the parameter's instantiation now.
///
/// includes. `whole` is the complete type the parameter is bound to, for the.
///
/// diagnostic.
///
///
/// Function types carry no key, so a function-typed instantiation passes and.
///
/// its parameter and result variables stay unmarked: the function's own.
///
/// scheme, not this parameter, decides whether a key may reach them.
///
fn ensure_key_free<E: Clone + PartialEq>(
    origin: &GenericParameter,
    ty: &Type<E>,
    whole: &Type<E>,
    subst: &Subst<E>,
) -> Result<(), TypeError> {
    let refuse = || TypeError {
        kind: TypeErrorKind::KeyInstantiation {
            value_binding: origin.generic.clone().filter(|_| origin.value),
        },
        message: format!(
            "{} cannot be instantiated at the key-carrying type `{}`: a function's type \
             parameter never stands for a key or a value that carries one, because a generic \
             body may use its value more than once (spec/04-type-system.md [04-LIN-10])",
            origin.describe(),
            subst.apply(whole)
        ),
    };
    match ty {
        Type::Var(var) => match subst.apply(&Type::Var(*var)) {
            Type::Var(root) => {
                subst.forbid_key_instantiation(root, origin.clone());
                Ok(())
            }
            resolved => ensure_key_free(origin, &resolved, whole, subst),
        },
        Type::Prim(Prim::Key) => Err(refuse()),
        Type::Prim(_) | Type::Unit | Type::Fn(_, _) | Type::Error(_) => Ok(()),
        Type::Tensor(_, TensorPrec::Concrete(prim)) => {
            ensure_key_free(origin, &Type::Prim(*prim), whole, subst)
        }
        Type::Tensor(_, TensorPrec::Var(var)) => {
            ensure_key_free(origin, &Type::Var(*var), whole, subst)
        }
        Type::Ref(inner) => ensure_key_free(origin, inner, whole, subst),
        Type::Tuple(elements) => elements
            .iter()
            .try_for_each(|element| ensure_key_free(origin, element, whole, subst)),
        Type::Adt(name, arguments) => {
            if subst.key_carrying_adts.contains(name) {
                return Err(refuse());
            }
            arguments
                .iter()
                .try_for_each(|argument| ensure_key_free(origin, argument, whole, subst))
        }
        Type::KindedAdt(name, arguments) => {
            if subst.key_carrying_adts.contains(name) {
                return Err(refuse());
            }
            arguments.iter().try_for_each(|argument| match argument {
                NominalArg::Type(argument) => ensure_key_free(origin, argument, whole, subst),
                NominalArg::Dimension(_) => Ok(()),
            })
        }
    }
}

/// Check one instantiation of a bounded type variable ([04-DTYPE-2]).
///
///
/// An unresolved variable and a witnessed error both pass: the first is.
///
/// narrowed instead by [`Subst::narrow_tvar_restriction`], and the second.
///
/// already owns a diagnostic.
///
fn ensure_tvar_restriction<E: Clone + PartialEq>(
    restriction: TypeVarRestriction,
    ty: &Type<E>,
    subst: &Subst<E>,
) -> Result<(), TypeError> {
    let family = restriction.bound_description();
    let gloss = restriction.membership_gloss();
    match ty {
        Type::Ref(inner) if restriction.is_value_constraint() => {
            ensure_tvar_restriction(restriction, inner, subst)
        }
        Type::Tensor(_, precision) if restriction.is_value_constraint() => {
            let family = restriction.precision_family();
            match precision {
                TensorPrec::Concrete(prim) => {
                    ensure_tvar_restriction(family, &Type::Prim(*prim), subst)
                }
                TensorPrec::Var(var) => {
                    let resolved = subst.apply(&Type::Var(*var));
                    if let Type::Var(var) = resolved {
                        subst.narrow_tvar_restriction(var, family)
                    } else {
                        ensure_tvar_restriction(family, &resolved, subst)
                    }
                }
            }
        }
        Type::Var(_) | Type::Error(_) => Ok(()),
        Type::Prim(prim) if restriction.admits(*prim) => Ok(()),
        Type::Prim(prim) => Err(TypeError {
            kind: TypeErrorKind::DtypeFamilyMismatch,
            message: {
                let message = format!(
                    "type variable bounded by {family} ({gloss}) cannot be instantiated at `{}`",
                    prim.name()
                );
                let small_integer =
                    restriction.admits(Prim::Int8) || restriction.admits(Prim::Int16);
                match crate::decisions::shape::sum_result_bound_note(small_integer, *prim) {
                    Some(note) => format!("{message}; {note}"),
                    None => message,
                }
            },
        }),
        other => Err(TypeError {
            kind: TypeErrorKind::DtypeFamilyMismatch,
            message: format!(
                "type variable bounded by {family} ({gloss}) cannot be instantiated at `{other}`"
            ),
        }),
    }
}

fn bind_dvar<E: Clone + PartialEq>(
    v: DimVar,
    dim: &Dim,
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    if let Dim::Name(name) = dim
        && (subst.protected_dimensions.contains(&v) || subst.dimension_label(v).is_some())
    {
        subst.set_dimension_label(v, name.clone());
        // chelis#1801 note: this arm records an authored name as a LABEL and.
        //
        // binds nothing, so it is the one place a name reaches a class.
        //
        // without `constraint_dim` being able to report it, and it carries no.
        //
        // pin of its own. Not because the operand must come from the.
        //
        // definition under check: `def named(w: tensor[batch, f32])` keeps.
        //
        // `batch` as a `Dim::Name` in its published scheme, so instantiating.
        //
        // it anywhere contributes one. The route is the arm's OWN.
        //
        // precondition. It fires only when `v` is protected, which the pin.
        //
        // reads directly, or already labelled, and a label is never created.
        //
        // from nothing: the three production writers of `set_dimension_label`.
        //
        // are `copy_dimension_label`, which propagates an existing label to a.
        //
        // fresh instantiation id, and the two below, which each need an.
        //
        // existing label or protection. Every label therefore traces back to.
        //
        // a declared binder.
        //
        //
        // What that argument does NOT establish is that the labelled variable.
        //
        // is itself protected, because the instantiation copy lands a label on.
        //
        // a fresh unprotected id. chelis#1925's round 3 measured the gap.
        //
        // instead of arguing it: instrumenting all 196 `.ch` files in the.
        //
        // corpus plus 32 probes found no unpinned class reaching this arm, and.
        //
        // the labelled-then-named state, constructed four ways, binds through.
        //
        // the fall-through below so `constraint_dim` reports it.
        //
        // `a_name_arm_class_is_pinned_by_protection_not_by_a_second_ledger`.
        //
        // locks what the arm leaves behind and says where the pin goes if a.
        //
        // program ever reaches that state.
        //
        return Ok(());
    }
    if let Dim::Var(other) = dim
        && *other != v
    {
        if subst.protected_dimensions.contains(&v) && !subst.protected_dimensions.contains(other) {
            return bind_dvar(*other, &Dim::Var(v), subst);
        }
        if let Some(label) = subst
            .dimension_label(v)
            .or_else(|| subst.dimension_label(*other))
        {
            if subst.mapped_axis(v).is_none() {
                subst.set_dimension_label(v, label.clone());
            }
            if subst.mapped_axis(*other).is_none() {
                subst.set_dimension_label(*other, label);
            }
        }
    }
    if let Dim::Var(v2) = dim
        && *v2 == v
    {
        return Ok(());
    }
    if occurs_in_dim(v, dim, subst) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!("infinite dimension: d{} occurs in {dim:?}", v.0),
        });
    }
    // chelis#1801: the merge below leaves the union with one root, so the.
    //
    // class properties have to arrive there first. A name or literal the.
    //
    // class is BOUND to needs no flag: `constraint_dim` reports the binding,.
    //
    // and the absorbing site skips on that answer before it asks anything.
    //
    // else.
    //
    if let Dim::Var(other) = dim {
        subst.merge_dvar_class_flags(v, *other);
    }
    let target_level = subst.level_of_dvar(v);
    subst.lower_dim_to(dim, target_level);
    subst.insert_dim(v, dim.clone());
    Ok(())
}

/// Fully resolve a tensor's dim list under the current substitution: each.
///
/// bound `Dim::Rank` expands to the run it stands for and every other dim is.
///
/// `apply_dim`'d. an unbound (or terminal-rank) spread stays a `Dim::Rank`.
///
/// This is the per-shape form of `Subst::apply`'s tensor arm, used by the.
///
/// rank-unification dispatch so a row shape arrives with only *unbound*.
///
/// spreads interleaved among concrete dims.
///
fn resolve_shape<E: Clone + PartialEq>(dims: &[Dim], subst: &Subst<E>) -> Vec<Dim> {
    let mut out = Vec::with_capacity(dims.len());
    for d in dims {
        match d {
            Dim::Rank(r) => {
                for rd in subst.resolve_rvar(*r) {
                    out.push(subst.apply_dim(&rd));
                }
            }
            _ => out.push(subst.apply_dim(d)),
        }
    }
    out
}

fn mapped_axis_dim<E: Clone + PartialEq>(dim: &Dim, subst: &Subst<E>) -> Option<(usize, DimVar)> {
    let Dim::Var(var) = dim else {
        return None;
    };
    subst.mapped_axis(*var).map(|axis| (axis, *var))
}

fn mapped_axis_error(axis: usize, unbatched_rank: usize) -> TypeError {
    TypeError {
        kind: TypeErrorKind::DimensionMismatch,
        message: format!("vmap axis {axis} is out of bounds for rank {unbatched_rank} tensor"),
    }
}

fn canonicalize_ground_mapped_axes<E: Clone + PartialEq>(
    dims: &[Dim],
    subst: &Subst<E>,
) -> Result<Vec<Dim>, TypeError> {
    let mut base = Vec::with_capacity(dims.len());
    let mut mapped = Vec::new();
    for dim in dims {
        if let Some(marker) = mapped_axis_dim(dim, subst) {
            mapped.push(marker);
        } else {
            base.push(dim.clone());
        }
    }
    mapped.sort_by_key(|(axis, var)| (*axis, var.0));
    for pair in mapped.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(TypeError {
                kind: TypeErrorKind::DimensionMismatch,
                message: format!("two vmap batch dimensions claim mapped axis {}", pair[0].0),
            });
        }
    }
    let unbatched_rank = base.len();
    for (axis, var) in mapped {
        if axis > base.len() {
            return Err(mapped_axis_error(axis, unbatched_rank));
        }
        base.insert(axis, Dim::Var(var));
    }
    Ok(base)
}

fn split_row_mapped_axes<E: Clone + PartialEq>(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst<E>,
) -> Result<(Vec<Dim>, Vec<Dim>), TypeError> {
    let mut clean_row = Vec::with_capacity(row.len());
    let mut mapped = Vec::new();
    for dim in row {
        if let Some(marker) = mapped_axis_dim(dim, subst) {
            mapped.push(marker);
        } else {
            clean_row.push(dim.clone());
        }
    }
    mapped.sort_by_key(|(axis, var)| (*axis, var.0));
    for pair in mapped.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(TypeError {
                kind: TypeErrorKind::DimensionMismatch,
                message: format!("two vmap batch dimensions claim mapped axis {}", pair[0].0),
            });
        }
    }

    let mut clean_ground = ground.to_vec();
    let unbatched_rank = ground.len().saturating_sub(mapped.len());
    for (axis, var) in &mapped {
        let Some(batch) = ground.get(*axis) else {
            return Err(mapped_axis_error(*axis, unbatched_rank));
        };
        // Bind the compiler-owned batch identity in this direction so an.
        //
        // authored operand identity remains the class root and retains its.
        //
        // user-facing label.
        //
        unify_dim(&Dim::Var(*var), batch, subst)?;
    }
    for (axis, _) in mapped.into_iter().rev() {
        clean_ground.remove(axis);
    }
    Ok((clean_row, clean_ground))
}

fn unify_row_against_ground_with_mapped_axes<E: Clone + PartialEq>(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    let ground = canonicalize_ground_mapped_axes(ground, subst)?;
    let (row, ground) = split_row_mapped_axes(row, &ground, subst)?;
    unify_row_against_ground(&row, &ground, subst)
}

fn unify_rows_with_mapped_axes<E: Clone + PartialEq>(
    left: &[Dim],
    right: &[Dim],
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    let mut left_row = Vec::with_capacity(left.len());
    let mut right_row = Vec::with_capacity(right.len());
    let mut left_mapped = Vec::new();
    let mut right_mapped = Vec::new();
    for dim in left {
        if let Some(marker) = mapped_axis_dim(dim, subst) {
            left_mapped.push(marker);
        } else {
            left_row.push(dim.clone());
        }
    }
    for dim in right {
        if let Some(marker) = mapped_axis_dim(dim, subst) {
            right_mapped.push(marker);
        } else {
            right_row.push(dim.clone());
        }
    }
    left_mapped.sort_by_key(|(axis, var)| (*axis, var.0));
    right_mapped.sort_by_key(|(axis, var)| (*axis, var.0));
    if left_mapped
        .iter()
        .map(|(axis, _)| axis)
        .ne(right_mapped.iter().map(|(axis, _)| axis))
    {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: "cannot unify vectorized rank-spread rows with different mapped axes"
                .to_string(),
        });
    }
    for ((_, left_var), (_, right_var)) in left_mapped.iter().zip(&right_mapped) {
        unify_dim(&Dim::Var(*left_var), &Dim::Var(*right_var), subst)?;
    }
    unify_row_against_row(&left_row, &right_row, subst)
}

/// chelis#339 named-axis expand: a signature whose result rows *introduce* an.
///
/// axis name (present in a result tensor row but in no parameter row — the.
///
/// expand-inserted axis) must not have that same name covered by one of its.
///
/// rank-spread bindings at this unification. The symbolic collision check in.
///
/// `check_named_expand_signature` can only see the visible row. a caller.
///
/// whose spread-covered axes include the inserted name would otherwise.
///
/// monomorphize to a result row carrying the same dim name twice with two.
///
/// different extents — every later by-name axis lookup becomes ambiguous and.
///
/// the declared type misstates the runtime shape. Hard error, never a guessed.
///
/// alias (spec/04-type-system.md §4.5.3).
///
fn check_introduced_name_rank_collision<E: Clone + PartialEq>(
    args: &[Type<E>],
    ret: &Type<E>,
    subst: &Subst<E>,
) -> Result<(), TypeError> {
    fn collect<E: Clone + PartialEq>(
        ty: &Type<E>,
        names: &mut Vec<String>,
        rvars: &mut Vec<RankVar>,
    ) {
        match ty {
            Type::Tensor(dims, _) => {
                for d in dims {
                    match d {
                        Dim::Name(n) => names.push(n.clone()),
                        Dim::Rank(r) => rvars.push(*r),
                        _ => {}
                    }
                }
            }
            Type::Ref(inner) => collect(inner, names, rvars),
            Type::Fn(fn_args, fn_ret) => {
                for a in fn_args.iter() {
                    collect(a, names, rvars);
                }
                collect(fn_ret, names, rvars);
            }
            _ => {}
        }
    }

    let mut ret_names = Vec::new();
    let mut ret_rvars = Vec::new();
    collect(ret, &mut ret_names, &mut ret_rvars);
    if ret_names.is_empty() || ret_rvars.is_empty() {
        return Ok(());
    }
    let mut param_names = Vec::new();
    let mut param_rvars = Vec::new();
    for a in args {
        collect(a, &mut param_names, &mut param_rvars);
    }
    let introduced: Vec<&String> = ret_names
        .iter()
        .filter(|n| !param_names.contains(n))
        .collect();
    if introduced.is_empty() {
        return Ok(());
    }
    for rv in ret_rvars {
        for bound in resolve_shape(&[Dim::Rank(rv)], subst) {
            if let Dim::Name(n) = &subst.semantic_dim(&bound)
                && introduced.contains(&n)
            {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: format!(
                        "rank-spread monomorphization collides with an inserted axis name: the \
                         spread binds an operand axis named `{n}`, but the callee's result also \
                         introduces an axis named `{n}` (a named-axis expand insert); the \
                         monomorphized result would carry the same dim name twice with two \
                         different extents, making later by-name axis lookups ambiguous \
                         (spec/04-type-system.md \u{00a7}4.5.3). Rename the inserted axis or \
                         the operand axis."
                    ),
                });
            }
        }
    }
    Ok(())
}

/// The shared diagnostic for an undetermined split between two adjacent.
///
/// rank spreads (the non-unitary case the decidable fragment excludes).
///
fn adjacent_spread_error() -> TypeError {
    TypeError {
        kind: TypeErrorKind::DimensionMismatch,
        message: "two adjacent rank spreads cannot be split against a concrete shape; the \
                  boundary between them is undetermined (outside the decidable fragment, \
                  spec/04-type-system.md \u{00a7}4.5.3)"
            .to_string(),
    }
}

/// Unify a *row* shape (≥1 spread, no two spreads adjacent) against a *ground*.
///
/// shape (no spreads). Walks the row left to right: a fixed anchor matches the.
///
/// ground positionally. a spread is bound to the run of ground dims up to the.
///
/// next anchor, found by that anchor's name (the §4.2 name-preserving rule).
///
/// Each interior split anchor must be a `Dim::Name` present **exactly once** in.
///
/// the remaining ground — a missing, ambiguous, or non-named anchor is a hard.
///
/// error, never a guessed split. With one spread per gap and named splits,.
///
/// every split point is forced, so this is unitary.
///
fn unify_row_against_ground<E: Clone + PartialEq>(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    let n = ground.len();
    let mut gi = 0usize; // ground cursor
    let mut ri = 0usize; // row cursor
    while ri < row.len() {
        match &row[ri] {
            Dim::Rank(r) => {
                let rest = &row[ri + 1..];
                match rest.iter().position(|d| !matches!(d, Dim::Rank(_))) {
                    // The spread is immediately followed by a named anchor:.
                    //
                    // find that name in the remaining ground to fix the split.
                    //
                    Some(0) => {
                        let semantic_anchor = subst.semantic_dim(&rest[0]);
                        let name = match &semantic_anchor {
                            Dim::Name(s) => s,
                            other => {
                                return Err(TypeError {
                                    kind: TypeErrorKind::DimensionMismatch,
                                    message: format!(
                                        "rank-spread anchor must be a named dimension to locate the \
                                         split, got {other:?} (spec/04-type-system.md \u{00a7}4.5.3)"
                                    ),
                                });
                            }
                        };
                        let hits: Vec<usize> = (gi..n)
                            .filter(|&j| matches!(subst.semantic_dim(&ground[j]), Dim::Name(g) if &g == name))
                            .collect();
                        match hits.as_slice() {
                            [split] => {
                                bind_rvar(*r, &ground[gi..*split], subst)?;
                                gi = *split;
                            }
                            [] => {
                                return Err(TypeError {
                                    kind: TypeErrorKind::DimensionMismatch,
                                    message: format!(
                                        "rank-spread operand carries no named `{name}` axis; a \
                                         fully-literal or differently-named operand cannot locate \
                                         the axis (spec/04-type-system.md \u{00a7}4.5.3)"
                                    ),
                                });
                            }
                            many => {
                                return Err(TypeError {
                                    kind: TypeErrorKind::DimensionMismatch,
                                    message: format!(
                                        "named axis `{name}` is ambiguous: it appears {} times in \
                                         the operand shape (spec/04-type-system.md \u{00a7}4.5.3)",
                                        many.len()
                                    ),
                                });
                            }
                        }
                        ri += 1;
                    }
                    // A second spread sits between this spread and the next.
                    //
                    // anchor (or runs to the end): the split between two.
                    //
                    // adjacent spreads is undetermined — the non-unitary case.
                    //
                    Some(_) => {
                        return Err(adjacent_spread_error());
                    }
                    None if rest.is_empty() => {
                        // Trailing spread: absorb the rest of the ground.
                        //
                        bind_rvar(*r, &ground[gi..n], subst)?;
                        gi = n;
                        ri += 1;
                    }
                    None => {
                        // `rest` is one or more further spreads with no anchor.
                        //
                        return Err(adjacent_spread_error());
                    }
                }
            }
            anchor => {
                if gi >= n {
                    return Err(TypeError {
                        kind: TypeErrorKind::DimensionMismatch,
                        message: format!(
                            "tensor rank too small: the rank-spread shape has more fixed anchor \
                             dimensions than the operand's {n} dimensions"
                        ),
                    });
                }
                unify_dim(anchor, &ground[gi], subst)?;
                gi += 1;
                ri += 1;
            }
        }
    }
    if gi != n {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!(
                "tensor shape mismatch: {} operand dimension(s) left unmatched after the \
                 rank-spread shape was consumed",
                n - gi
            ),
        });
    }
    Ok(())
}

/// Unify two row shapes that *both* carry spreads. Restricted to.
///
/// structurally-identical rows (same length, same per-position kind): spreads.
///
/// alias pairwise, named anchors unify by name. Two rows with a *different*.
///
/// anchor structure are associative/string unification (not unitary) and are.
///
/// rejected — the body of a rank-poly def only ever produces an identical-row.
///
/// self-check, and call sites are row-vs-ground.
///
fn unify_row_against_row<E: Clone + PartialEq>(
    d1: &[Dim],
    d2: &[Dim],
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    if d1.len() != d2.len() {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!(
                "cannot unify rank-spread shapes with differing anchor structure ({} vs {} slots); \
                 two spreads with an undetermined split are outside the decidable fragment \
                 (spec/04-type-system.md \u{00a7}4.5.3)",
                d1.len(),
                d2.len()
            ),
        });
    }
    for (a, b) in d1.iter().zip(d2.iter()) {
        match (a, b) {
            (Dim::Rank(r1), Dim::Rank(r2)) => {
                if r1 != r2 {
                    let target_level = subst.level_of_rvar(*r1);
                    subst.lower_rvar_to(*r2, target_level);
                    subst.insert_rank(*r1, vec![Dim::Rank(*r2)]);
                }
            }
            (Dim::Rank(_), _) | (_, Dim::Rank(_)) => {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: "cannot unify a rank spread against a concrete dimension; the two \
                              rank-spread shapes have differing structure \
                              (spec/04-type-system.md \u{00a7}4.5.3)"
                        .to_string(),
                });
            }
            _ => unify_dim(a, b, subst)?,
        }
    }
    Ok(())
}

/// Bind a rank variable to a concrete shape *run*, with an occurs/nesting.
///
/// check: the bound run must be `Dim::Rank`-free. A run containing any spread.
///
/// would either make the rank infinite (its own occurrence) or smuggle a.
///
/// second spread into a single spread's binding (the non-unitary case).
///
/// Rank-to-rank aliasing is handled separately by `unify_row_against_row`.
///
fn bind_rvar<E: Clone + PartialEq>(
    r: RankVar,
    dims: &[Dim],
    subst: &mut Subst<E>,
) -> Result<(), TypeError> {
    if let Some(bad) = dims.iter().find(|d| matches!(d, Dim::Rank(_))) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!(
                "rank variable r{} cannot bind to a shape containing a rank spread ({bad:?})",
                r.0
            ),
        });
    }
    let target_level = subst.level_of_rvar(r);
    subst.lower_ground_rank_to(dims, target_level);
    subst.insert_rank(r, dims.to_vec());
    Ok(())
}

/// Does `v` occur in `ty` once the substitution is applied?.
///
///
/// The substitution is applied once, to the whole type, and the result is.
///
/// walked as it stands: every variable left in an applied type is unbound,.
///
/// so applying again at each level of the walk would find nothing new and.
///
/// would cost the size of each subtree once per enclosing level, which is.
///
/// cubic over a chain of bindings (chelis#3819).
///
fn occurs_in<E: Clone + PartialEq>(v: TypeVar, ty: &Type<E>, subst: &Subst<E>) -> bool {
    let ty = subst.apply(ty);
    occurs_in_applied(v, &ty, subst)
}

fn occurs_in_applied<E: Clone + PartialEq>(v: TypeVar, ty: &Type<E>, subst: &Subst<E>) -> bool {
    match ty {
        Type::Var(v2) => *v2 == v,
        Type::Fn(args, ret) => {
            args.iter().any(|a| occurs_in_applied(v, a, subst)) || occurs_in_applied(v, ret, subst)
        }
        Type::Ref(inner) => occurs_in_applied(v, inner, subst),
        Type::Tensor(_, prec) => match prec {
            // Dims do not carry type vars (they have their own DimVar lane).
            //
            // Precision slot CAN carry a TypeVar (WS-A5 precision polymorphism).
            //
            // include it in the occurs check so an attempt to unify.
            //
            // ?v with `tensor[..., ?v]` is caught as an infinite type.
            //
            TensorPrec::Concrete(_) => false,
            // Applying keeps a precision variable bound to a non-primitive.
            //
            // type as it is, so this one is resolved where it stands.
            //
            TensorPrec::Var(v2) => occurs_in(v, &Type::Var(*v2), subst),
        },
        Type::Adt(_, args) => args.iter().any(|a| occurs_in_applied(v, a, subst)),
        Type::KindedAdt(_, args) => args.iter().any(|argument| {
            argument
                .as_type()
                .is_some_and(|ty| occurs_in_applied(v, ty, subst))
        }),
        Type::Tuple(ts) => ts.iter().any(|t| occurs_in_applied(v, t, subst)),
        Type::Prim(_) | Type::Unit | Type::Error(_) => false,
    }
}

fn occurs_in_dim<E: Clone + PartialEq>(v: DimVar, dim: &Dim, subst: &Subst<E>) -> bool {
    let dim = subst.apply_dim(dim);
    matches!(dim, Dim::Var(v2) if v2 == v)
}

impl<E> Default for ApplyMemo<E> {
    fn default() -> Self {
        Self {
            children: Default::default(),
            lists: Default::default(),
            arguments: Default::default(),
        }
    }
}

impl<E> Default for GroundTypes<E> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<E> Default for GroundStore<E> {
    fn default() -> Self {
        Self {
            addresses: Default::default(),
            held: Default::default(),
            sweep_at: Default::default(),
        }
    }
}

impl<E> Default for Subst<E> {
    fn default() -> Self {
        Self {
            result_constraints: Default::default(),
            types: Default::default(),
            tvar_restrictions: Default::default(),
            key_free_tvars: Default::default(),
            key_carrying_adts: Default::default(),
            dims: Default::default(),
            dimension_labels: Default::default(),
            protected_dimensions: Default::default(),
            wildcard_touched_classes: Default::default(),
            binder_pinned_classes: Default::default(),
            label_unification_depth: Default::default(),
            dimension_label_undo: Default::default(),
            refined_type_bindings: Default::default(),
            ranks: Default::default(),
            deferred_borrow_vars: Default::default(),
            deferred_drop_operands: Default::default(),
            deferred_opaque_uses: Default::default(),
            deferred_tensor_operands: Default::default(),
            ground_types: Default::default(),
            operand_gate_failures: Default::default(),
            collection_contracts: Default::default(),
            next_collection_contract_id: Default::default(),
            current_level: Default::default(),
            level_transitions: Default::default(),
            lowered_tvar_levels: Default::default(),
            lowered_dvar_levels: Default::default(),
            lowered_rvar_levels: Default::default(),
            resume_floors: Default::default(),
        }
    }
}

#[cfg(test)]
#[path = "solver_tests.rs"]
mod tests;
