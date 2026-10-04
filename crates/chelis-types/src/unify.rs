//! Unification algorithm for the Chelis type checker.
//!
//! Solves type equations by binding type variables and dimension variables.
//!
//! ## Path compression
//!
//! `Subst` stores type / dim variable bindings as union-find chains. When a
//! caller resolves a variable via [`Subst::apply`] or [`Subst::apply_dim`],
//! the resolver iteratively follows the chain to its terminal value AND
//! writes the resolved value back into the substitution so future lookups
//! land in O(1). The maps live behind a `Mutex` so this in-place
//! compression remains available to callers that hold an immutable
//! borrow of `Subst` (annotation passes, `Env::generalize`, occurs checks).
//!
//! Without path compression, deeply-nested expressions like Cons chains
//! over wildcard tensor literals (e.g.,
//! `pad_sequences_to([[10×4 floats]], ...)`) accumulate dim-variable
//! chains 1000+ links long; resolving every dim through a 1000-link
//! chain on every type walk drives ~2 billion `apply_dim` calls and
//! pushes phase-0e annotation from ~2s into 10+ minutes for chelis-std's
//! `nn/embedding.ch`. See the gdb backtrace recorded in this commit's
//! body for the canonical reproducer.

use chelis_unord::{UnordMap, UnordSet};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::types::*;

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
/// Transitions are append-only between persisted-context resumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum LevelTransitionKind {
    Enter,
    Leave,
    Resume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LevelTransition {
    kind: LevelTransitionKind,
    level: u32,
    watermarks: VarWatermarks,
}

#[derive(Debug, Clone, Copy)]
enum VarClass {
    Type,
    Dim,
    Rank,
}

/// Linear token proving which child level must be left next.
///
/// The token is deliberately neither `Copy` nor `Clone`: every successful
/// enter must have exactly one matching leave, and leaves must be LIFO.
#[derive(Debug)]
pub(crate) struct LevelToken {
    parent_level: u32,
    child_level: u32,
}

/// Type error produced during unification.
#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    /// A checked family restriction failed, including before its operand
    /// acquired a concrete type. Rendered publicly as PrecisionMismatch.
    DtypeFamilyMismatch,
    /// [04-LIN-10]: a function's type parameter was instantiated at a
    /// key-carrying type. Rendered publicly as KeyReuse, the key-affinity
    /// diagnostic. `value_binding` names the generic when it is a value
    /// binding rather than a function, such as a generalized `let` binding,
    /// whose repair is an ascription rather than a concrete key parameter.
    KeyInstantiation {
        value_binding: Option<String>,
    },
    DimensionMismatch,
    ArityMismatch,
    OccursCheck,
    NotAFunction,
}

/// [04-LIN-10]: the generic a key-free type variable stands for a parameter
/// of, so the diagnostic names it. Both parts are optional: a variable
/// generalized before its binding is named has no generic yet, and a
/// parameter the program never spelled has no binder name, and neither is
/// invented ([04-FIT-10]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenericParameter {
    /// The definition or `let` binding whose scheme quantified the variable.
    pub generic: Option<String>,
    /// The authored binder the variable instantiates, when there is one.
    pub binder: Option<String>,
    /// Whether `generic` is a value binding, one whose type is not a function
    /// type, such as `e = Nil`. Its repair ascribes the binding a type with no
    /// type parameter, or writes the value where it is used.
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
/// Labels, known constraints, and current authored rigidity are independent.
/// Deliberately has no equality or conversion to `Dim`: each operation chooses
/// its own proof/admission policy, and keeps the original ID in surviving axes.
pub(crate) struct DimObservation {
    constraint: Dim,
    name: Option<String>,
    protected: bool,
}

impl DimObservation {
    pub(crate) fn known_extent(&self) -> Option<i64> {
        match self.constraint {
            Dim::Lit(n) => Some(n),
            _ => None,
        }
    }

    /// The literal category of policies that admit named runtime claims
    /// (expand/reshape/concat), not every internally known constraint.
    pub(crate) fn literal_extent(&self) -> Option<i64> {
        if self.name.is_none() {
            self.known_extent()
        } else {
            None
        }
    }

    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub(crate) fn variable(&self) -> Option<DimVar> {
        match self.constraint {
            Dim::Var(v) => Some(v),
            _ => None,
        }
    }
    pub(crate) fn rank(&self) -> Option<RankVar> {
        match self.constraint {
            Dim::Rank(v) => Some(v),
            _ => None,
        }
    }
    pub(crate) fn is_wildcard(&self) -> bool {
        matches!(self.constraint, Dim::Wildcard)
    }
    pub(crate) fn is_protected(&self) -> bool {
        self.protected
    }
}

/// Substitution: maps type variables to types and dim variables to dims.
///
/// The maps are wrapped in `Mutex` so [`Subst::apply`] and
/// [`Subst::apply_dim`] can perform in-place path compression while
/// keeping the public method receiver `&self`. The Mutex is uncontended
/// in normal use (each `compile_new_source_in_context` call clones the
/// substitution into its own thread-local state), so the locking cost
/// is one atomic compare-exchange per call. Use [`Subst::insert_type`] /
/// [`Subst::insert_dim`] to record new bindings during unification.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Subst {
    #[serde(skip)]
    result_constraints: Mutex<Vec<crate::types::ResultConstraint>>,
    types: Mutex<UnordMap<TypeVar, Type>>,
    /// Semantic domains attached to unresolved type variables. This is
    /// serialized with reusable checking contexts: a constrained function
    /// value must not become unconstrained after a cache round trip.
    #[serde(default)]
    tvar_restrictions: Mutex<UnordMap<TypeVar, TypeVarRestriction>>,
    /// [04-LIN-10]: the type variables that stand for a function's type
    /// parameter, each with the generic it belongs to, so a violation names
    /// it. Generalization marks every variable it quantifies, instantiation
    /// copies the mark to the fresh variable, and binding a marked variable
    /// refuses a key-carrying type and marks every variable left inside the
    /// type it is bound to.
    ///
    /// Kept apart from [`Self::tvar_restrictions`] on purpose: those are dtype
    /// families that many sites read as "this variable is a dtype", which a
    /// variable that may still become a tuple, a list or a function is not.
    /// Serialized, so a library's generics keep the rule across a context
    /// round trip.
    #[serde(default)]
    key_free_tvars: Mutex<UnordMap<TypeVar, GenericParameter>>,
    /// Spec/04 section 8.4.1: the data types of this check whose fields carry
    /// a key, installed once every `type` declaration is registered and read
    /// when a [`Self::key_free_tvars`] variable is bound to a data type.
    #[serde(default)]
    key_carrying_adts: UnordSet<String>,
    dims: Mutex<UnordMap<DimVar, Dim>>,
    /// Named-axis view of a checker dimension identity, plus reserved
    /// compiler-owned annotations on synthetic dimension identities. Schemes
    /// retain these IDs and instantiation copies their annotations to fresh
    /// IDs. This mandatory snapshot field is not evidence that arbitrary
    /// equal names have equal sizes.
    dimension_labels: Mutex<UnordMap<DimVar, String>>,
    #[serde(skip)]
    protected_dimensions: UnordSet<DimVar>,
    /// chelis#1801: the alias CLASSES that have met a runtime extent `*`
    /// without being bound to it, keyed by the class root.
    ///
    /// `unify_dim`'s wildcard arm deliberately leaves such a variable free
    /// (the #143 invariant documented on that function), so a later concrete
    /// argument in the same signature can still constrain it. That leaves no
    /// record of the meeting at all, and `spec/04-type-system.md` section 3.2
    /// needs one: a variable an application's instantiation minted, that met
    /// `*` and that no argument bound, denotes that runtime extent.
    ///
    /// Why a property of the CLASS rather than of one variable. Unification
    /// identifies variables, and which member roots a class changes while the
    /// call unifies: `unify_dim` resolves both operands through
    /// `constraint_dim` before it matches, so the meeting is recorded on
    /// whatever rooted the class at that moment, and a later `bind_dvar` in
    /// the same call can re-root the class over it. Asking one member, or
    /// two, then misses a meeting recorded on a third. chelis#1925's rounds 1
    /// and 2 were both exactly that miss, one alias member apart. The flag
    /// therefore lives on the root and is OR-merged onto the new root at
    /// every merge ([`Subst::merge_dvar_class_flags`]), and the absorbing
    /// site asks the class once.
    ///
    /// Monotone within a pass and never cleared: dimension variables are
    /// minted fresh and never reused, so an entry cannot become stale. Not
    /// serialized: transient per-pass bookkeeping, like the deferred ledgers
    /// below.
    #[serde(skip)]
    wildcard_touched_classes: UnordSet<DimVar>,
    /// chelis#1801: the alias classes a DECLARED dimension of the definition
    /// under check pins, keyed and merged exactly like the set above.
    ///
    /// This is half of section 3.2's own exclusion rather than an extra
    /// guard: a literal or name that an argument of the same application
    /// binds to the class is a CLAIM on the runtime extent, checked by a
    /// section 4.7 guard, and absorbing the class to `*` would erase the
    /// claim. The other half needs no ledger. A name or literal the class is
    /// BOUND to is reported by `constraint_dim`, which the absorbing site
    /// asks first; what `constraint_dim` cannot report is a declared binder,
    /// because `bind_dvar`'s name arm records it as a label and leaves the
    /// class unbound. `protected_dimensions` holds those binders, and this
    /// set is what keeps that answer stable when a merge or a compose demotes
    /// the protected member out of the root position.
    #[serde(skip)]
    binder_pinned_classes: UnordSet<DimVar>,
    #[serde(skip)]
    label_unification_depth: usize,
    /// Only labels written by the active outer unification need rollback.
    /// Interior mutability also covers scheme instantiation through `&Subst`.
    #[serde(skip)]
    dimension_label_undo: Mutex<Vec<(DimVar, Option<String>)>>,
    #[serde(skip)]
    refined_type_bindings: Vec<(TypeVar, Type)>,
    /// Rank-variable bindings: a `RankVar` binds to the *entire* shape vector
    /// it stands for (Tier-2 rank polymorphism). A binding to `[Dim::Rank(r2)]`
    /// is a rank-to-rank alias resolved transitively by `resolve_rvar`.
    ranks: Mutex<UnordMap<RankVar, Vec<Dim>>>,
    /// Issue #256 soundness ledger. The `borrow` inference arm accepts a
    /// borrow whose inner type is still an unresolved `Type::Var`,
    /// deferring the tensor-or-carrier classification to subsequent
    /// unification (the surrounding `&tensor[..]` parameter pins it).
    /// That deferral is only sound when the variable is *eventually*
    /// pinned to a tensor or tensor-carrying type. When the consumer is
    /// itself fully polymorphic (e.g. `consume_any[a](t: a)`), the
    /// variable is never pinned and a genuinely-non-tensor value would
    /// slip past every gate. Each deferred borrow records the inner
    /// `TypeVar` here; after a def body's inference completes, the
    /// driver resolves each one against the now-complete substitution
    /// and rejects any that did not become a tensor or tensor carrier.
    /// Not serialized: this is transient per-pass bookkeeping, drained
    /// by the inference driver, and never part of a persisted context.
    #[serde(skip)]
    deferred_borrow_vars: Mutex<Vec<TypeVar>>,
    /// RFC D-CHECK deferred-access ledger, mirroring
    /// `deferred_borrow_vars`: `access`/`record-update` sites whose
    /// target type was still an unresolved `Type::Var` when inference
    /// ran (e.g. an unannotated lambda parameter pinned only by a
    /// later call). The driver drains these per def and re-checks each
    /// against the final substitution so a target pinned to an opaque
    /// ADT defined in another module is still rejected. Not
    /// serialized: transient per-pass bookkeeping.
    #[serde(skip)]
    deferred_opaque_uses: Mutex<Vec<(TypeVar, DeferredOpaqueUse)>>,
    /// chelis#1489 suspended-operand ledger. A set of checked sites rejected
    /// an unresolved `Type::Var` outright, which made them sensitive to WHEN
    /// inference resolved a variable rather than to whether the program was
    /// well typed: 0.18.6 changed that timing and they fired ~50x more often
    /// on an unchanged corpus.
    ///
    /// What suspends here is `copy`, `cast`, and the ten csv host-lane slots
    /// — ~97% of the measured occurrences. `round_to` is NOT among them: see
    /// `unify_host_slot_eager`.
    ///
    /// `gather`, `scatter`, `scatter_replace` and `trace` record too, as
    /// [`DeferredOperandGate::ShapeRoute`]. Their results are shape functions of
    /// the operand, so that variant carries the arm's other evidence -- the RAW
    /// axis, the indices, and `scatter`'s updates operand -- and discharge
    /// replays the whole arm rather than the one helper call it appears to make.
    ///
    /// `concat` and `diagonal` do NOT record. They compute a new extent
    /// arithmetically (a sum, a minimum), their helpers encode a dim they
    /// cannot compute with as `Dim::Wildcard`, and a dim can still be a
    /// variable when the operand's TYPE variable binds -- so deferring them let
    /// a false declared extent check, build and run. They stay on their eager
    /// paths; `the_arithmetic_routes_are_not_converted` pins the exclusion.
    ///
    /// Recording rather than tolerating is deliberate. The ~71 sibling gates
    /// return on an unresolved operand and forget it, which is sound for them
    /// but not here: measured, a tolerant `cast` lets
    /// `def go[t](x: t) -> i32 = cast(x, i32)` check at 1.0 and BUILD,
    /// with the backend choosing a dtype for the never-resolved `t`. A
    /// variable that is never bound is still rejected.
    ///
    /// The `Copy` and `Cast` entries carry the result variable their call
    /// returned, and discharge unifies the eager arm's own answer into it;
    /// `HostSlot` carries no result and discharge unifies its expected type
    /// against the operand instead. Both are load-bearing: a
    /// revision that returned the OPERAND's variable made a deferred
    /// `copy(&t)` type as `&tensor` where an eager one is `tensor`, so the
    /// expression's type depended on when the operand resolved — the very
    /// sensitivity this ledger exists to delete.
    ///
    /// Not serialized: transient per-pass bookkeeping.
    #[serde(skip)]
    deferred_tensor_operands: Mutex<Vec<(TypeVar, DeferredOperandGate)>>,
    /// Verdicts from suspended operand constraints that discharged badly
    /// (chelis#1489).
    ///
    /// Discharge happens inside unification, which has no `DiagnosticSink`, so
    /// the failure is recorded here and rendered by the per-def reporting pass.
    /// Only failures land here; a constraint that discharges cleanly leaves no
    /// trace beyond the unification it performed.
    ///
    /// Not serialized: transient per-pass bookkeeping.
    #[serde(skip)]
    operand_gate_failures: Mutex<Vec<OperandGateFailure>>,
    /// chelis#1654: checked collection-operation contracts instantiated during
    /// this inference pass.
    ///
    /// Each scheme instantiation gets an opaque ID and its lexical inference
    /// level. An application captures the exact IDs its callee produced,
    /// attaches that call's evidence, and discharges only those instances after
    /// argument unification. A relation merely carried through an alias,
    /// aggregate, or higher-order result remains transportable until
    /// generalization moves it back onto the resulting scheme.
    ///
    /// This is the single lifecycle owner for both variable-bearing and fully
    /// monomorphic relations. In particular, no settled side queue can decide
    /// tensor concat before an application supplies its axis and element
    /// extents. Not serialized: schemes are the durable representation.
    #[serde(skip)]
    collection_contracts: Mutex<Vec<CollectionContractInstance>>,
    #[serde(skip)]
    next_collection_contract_id: Mutex<u64>,
    /// Current lexical generalization level. Serialized because a cloned
    /// checking context must preserve in-flight transactional state.
    #[serde(default)]
    current_level: u32,
    /// Ordered enter/leave/resume watermarks used to recover mint levels.
    #[serde(default)]
    level_transitions: Vec<LevelTransition>,
    /// Sparse overrides for variables unified into an older scope.
    #[serde(default)]
    lowered_tvar_levels: UnordMap<TypeVar, u32>,
    #[serde(default)]
    lowered_dvar_levels: UnordMap<DimVar, u32>,
    #[serde(default)]
    lowered_rvar_levels: UnordMap<RankVar, u32>,
    /// IDs below these floors came from an earlier persisted check and are
    /// always level zero in the resumed check.
    #[serde(default)]
    resume_floors: VarWatermarks,
}

/// A suspended operand constraint that discharged to a rejection
/// (chelis#1489).
///
/// Carries the data the diagnostic needs and none of the rendering: discharge
/// runs inside unification, and the per-def reporting pass is what turns this
/// into a `CheckError` (it is also what holds the declared type-parameter
/// names the message may want).
#[derive(Debug, Clone)]
pub(crate) enum OperandGateFailure {
    /// The operand settled to something the gate does not accept.
    Rejected {
        gate: DeferredOperandGate,
        resolved: Type,
    },
    /// The gate accepted the operand, but the result the call had already
    /// handed its consumer cannot be the result the settled operand produces.
    ResultMismatch {
        gate: DeferredOperandGate,
        expected: Type,
        settled: Type,
    },
    /// The gate's own decision function rejected the settled operand with a
    /// specific error -- an unsupported cast precision, say -- rather than a
    /// generic "wrong shape of type".
    Decision { error: crate::errors::CheckError },
}

/// Which operand constraint suspended its decision (chelis#1489), and what it
/// needs to decide once the operand is bound.
///
/// Not `Copy`: the host-slot arm carries the expected type and the producer's
/// own slot wording, so discharge can re-decide without re-running the call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DeferredOperandGate {
    /// `copy` with an unresolved operand. Carries the result variable the call
    /// returned: `copy(&t)` yields `t`, not `&t`, so discharge must unify the
    /// eager arm's own answer into it rather than let the operand's type stand.
    Copy {
        result: Box<Type>,
        location: Option<crate::deep_type::TypeDiagnosticLocation>,
    },
    /// `cast`/`cast_trunc` whose SOURCE was unresolved. Carries the target
    /// precision, the mode, and the result variable the call returned, which
    /// discharge unifies against once `cast`'s own decision function is called
    /// with the settled source type.
    Cast {
        target: crate::types::Prim,
        mode: chelis_deep::CastMode,
        result: Box<Type>,
        location: Option<crate::deep_type::TypeDiagnosticLocation>,
    },
    /// chelis#2534: a checked `cast` whose TARGET is a declaration's dtype
    /// binder and whose SOURCE was unresolved. Carries the target binder and
    /// the result variable the call returned; discharge decides the settled
    /// source with the same function the eager arm calls.
    CastToBinder {
        target: crate::types::TypeVar,
        result: Box<Type>,
        location: Option<crate::deep_type::TypeDiagnosticLocation>,
    },
    /// A host-lane slot that unifies against a fixed expected type:
    /// the ten csv routes, which funnel through `unify_host_slot`. Carries
    /// what the slot expected so discharge can re-decide without re-running
    /// the call.
    ///
    /// `round_to` deliberately does NOT reach here. This variant carries ONE
    /// expected type, and `round_to` accepts more than one; routing it here
    /// made a later-bound operand reject against the single type this carries.
    /// It is on `unify_host_slot_eager` instead.
    HostSlot {
        fname: String,
        description: String,
        expected: Box<Type>,
    },
    /// A shape-computing route whose result is a function of the operand's
    /// shape: `gather`, `scatter`, `scatter_replace` and `trace` (chelis#1489
    /// gate 3). `concat` and `diagonal` are deliberately excluded; see the
    /// deferred-operand ledger's doc.
    ///
    /// The payload is the WHOLE call minus its operand, and
    /// `infer::shape_route_result` is the one function that decides it. The
    /// eager arm builds this same payload and calls that same function with a
    /// resolved operand, so there is no second implementation of any route --
    /// this is a deferral of one decision, not a copy of the arm's tail.
    ShapeRoute {
        route: ShapeRoute,
        result: Box<Type>,
    },
}

pub(crate) type CollectionContractId = u64;

#[derive(Debug, Clone)]
struct CollectionContractInstance {
    id: CollectionContractId,
    level: u32,
    constraint: CollectionConstraint,
    state: CollectionContractState,
}

#[derive(Debug, Clone)]
enum CollectionContractState {
    Transport,
    Consumed {
        tensor_concat: Option<crate::infer::TensorConcatCallEvidence>,
    },
}

impl ShapeRoute {
    /// Whether this route's arm normalizes one borrow off its operand.
    ///
    /// `gather` and `trace` read their operand through `type_for_readonly_check`,
    /// which strips one `Type::Ref`; `scatter` and `scatter_replace` pass
    /// `subst.apply` and reject a borrowed operand. Stripping unconditionally in
    /// the shared decision widened those in the ACCEPTING direction -- and, when
    /// `concat` still went through here, a borrowed `concat` passed its element
    /// guard with empty per-element dims and made the concat axis a wildcard, so
    /// any declared extent was admitted.
    ///
    /// The strip is a property of the route, so it is recorded as one.
    pub(crate) fn normalizes_borrow(&self) -> bool {
        match self {
            Self::Gather { op, .. } => op == "gather",
            Self::Trace { .. } => true,
        }
    }

    /// The builtin's name, for the diagnostic.
    pub(crate) fn op(&self) -> &str {
        match self {
            Self::Gather { op, .. } => op,
            Self::Trace { .. } => "trace",
        }
    }
}

/// A shape-computing call with its operand left out (chelis#1489).
///
/// Carries every input the decision needs EXCEPT the operand, so the same
/// value decides the call eagerly and after a deferral.
///
/// Two fields are deliberately raw rather than pre-computed, because
/// pre-computing them needs the operand:
///
/// - `raw_axis` is the axis as written. The eager resolvers normalize a
///   negative axis against the operand's RANK and yield 0 when the operand is
///   not a concrete tensor, so an axis resolved before the operand settled
///   would be wrong. It is normalized inside the decision instead.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ShapeRoute {
    /// `gather`, and `scatter`/`scatter_replace` when `updates` is present.
    Gather {
        op: String,
        indices: Box<Type>,
        raw_axis: Option<i64>,
        updates: Option<Box<Type>>,
        /// `scatter`'s mode string, exactly as written. Validated inside the
        /// decision: validating it in the arm meant the deferred path skipped
        /// it and demoted a type error to a runtime abort.
        mode: Option<String>,
    },
    /// `trace`, which takes an axis PAIR.
    ///
    /// `diagonal` also takes one and is deliberately NOT here: it computes the
    /// minimum of two extents, and when a dim it reads is still a variable at
    /// discharge its helper falls back to `Dim::Wildcard`, which the declared
    /// signature then narrows to any extent at all. It stays on its eager path
    /// until that helper stops encoding "unknown" as permissive; see chelis#1489.
    Trace {
        raw_axis1: Option<i64>,
        raw_axis2: Option<i64>,
    },
}

impl DeferredOperandGate {
    /// The result the suspended call handed its consumer, if it handed one.
    ///
    /// `copy` and `cast` return a fresh variable that discharge unifies with the
    /// decided type; a host slot returns its builtin's own fixed result type and
    /// carries none. A shape route returns a fresh variable for the same reason
    /// `copy` does -- its result is a function of a shape nobody knows yet.
    ///
    /// Answering this is not optional bookkeeping. `Env::generalize` refuses to
    /// quantify anything this returns (chelis#1489, #1832): a result variable
    /// that is generalized before its gate discharges gives every use of a
    /// `let`-bound name its own instance, and the declared result is then never
    /// checked against what the call produces. Returning `None` for a variant
    /// that does hand out a fresh variable reopens that hole for that variant.
    /// The match is deliberately exhaustive with no wildcard arm so a new gate
    /// cannot be added without deciding this.
    fn result(&self) -> Option<&Type> {
        match self {
            Self::Copy { result, .. }
            | Self::Cast { result, .. }
            | Self::CastToBinder { result, .. }
            | Self::ShapeRoute { result, .. } => Some(result.as_ref()),
            Self::HostSlot { .. } => None,
        }
    }

    pub(crate) fn location(&self) -> Option<&crate::deep_type::TypeDiagnosticLocation> {
        match self {
            Self::Copy { location, .. }
            | Self::Cast { location, .. }
            | Self::CastToBinder { location, .. } => location.as_ref(),
            Self::HostSlot { .. } | Self::ShapeRoute { .. } => None,
        }
    }

    fn attach_location(&self, error: crate::errors::CheckError) -> crate::errors::CheckError {
        match self.location() {
            Some(location) => location.attach(error),
            None => error,
        }
    }

    pub(crate) fn expected_operand(&self) -> String {
        match self {
            Self::Copy { .. } | Self::ShapeRoute { .. } => "tensor".to_string(),
            Self::Cast { .. } => "tensor or numeric/bool scalar".to_string(),
            Self::CastToBinder { .. } => "numeric or bool scalar".to_string(),
            Self::HostSlot { description, .. } => description.clone(),
        }
    }

    // A consumer may have constrained the provisional result's dtype before
    // this producer settles. Preserve that family's error; the producer did
    // not violate its own operand contract (a valid cast can produce an int).
    fn reconcile_result(&self, result: &Type, settled: Type, subst: &mut Subst) {
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

    /// Settle this constraint against the type its operand was just bound to
    /// (chelis#1489).
    ///
    /// Every arm calls the SAME decision function its eager counterpart calls
    /// -- `copy_result_from_source`, `cast_result_from_settled_source`, or the
    /// slot unification -- so there is no second implementation of any gate to
    /// disagree with the first. The unification of `result` is what stops the
    /// fresh variable the eager call handed its consumer from staying
    /// unconstrained.
    ///
    /// `resolved` is never a `Type::Var`: the caller only discharges a bound
    /// variable, and re-aliases instead when a variable was bound to another.
    fn discharge(self, resolved: &Type, subst: &mut Subst) {
        match self {
            Self::Copy { ref result, .. } => {
                match crate::infer::expr::copy_result_from_source(resolved) {
                    Some(settled) => {
                        self.reconcile_result(result, settled, subst);
                    }
                    None => subst.record_operand_gate_failure(OperandGateFailure::Rejected {
                        gate: self.clone(),
                        resolved: resolved.clone(),
                    }),
                }
            }
            Self::Cast {
                target,
                mode,
                ref result,
                ..
            } => {
                match crate::infer::expr_record::cast_result_from_settled_source(
                    resolved.clone(),
                    target,
                    mode,
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
            Self::CastToBinder {
                target, ref result, ..
            } => {
                match crate::infer::expr_record::binder_cast_result_from_settled_source(
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
                match crate::infer::shape_route_result(route, resolved, subst) {
                    Ok((settled, updates)) => {
                        // The updates equation `scatter` imposes is part of the
                        // arm, not of the helper. Replaying only the helper
                        // would drop it.
                        if let Some(expected_updates) = updates
                            && let ShapeRoute::Gather {
                                updates: Some(actual),
                                ..
                            } = route
                            && let Err(te) = unify(&expected_updates, actual.as_ref(), subst)
                        {
                            // Record the unification error itself, not a generic
                            // result mismatch. The eager arm reports `te.into()`,
                            // which carries `DimensionMismatch`; collapsing that
                            // onto `TypeMismatch` would change a published
                            // diagnostic kind by inference order -- the wire
                            // change `kind()`'s own doc forbids.
                            subst.record_operand_gate_failure(OperandGateFailure::Decision {
                                error: te.into(),
                            });
                            return;
                        }
                        self.reconcile_result(result, settled, subst);
                    }
                    Err(message) => {
                        subst.record_operand_gate_failure(OperandGateFailure::Decision {
                            error: crate::errors::CheckError::new(
                                crate::errors::CheckErrorKind::TypeMismatch,
                                message,
                                Vec::new(),
                            ),
                        })
                    }
                }
            }
        }
    }

    /// The rejection this gate emits once the operand is known to be wrong.
    ///
    /// The wording each gate used when it decided eagerly, with two
    /// exceptions. The eager host-slot rejection appends macro provenance ("in
    /// expansion of ...") from the node, and discharge has no node, so a
    /// host-slot rejection inside a macro expansion loses that suffix. And
    /// `subject_for` substitutes a backticked declared type-parameter name
    /// (``got `t` ``) where the eager arm printed the internal identity --
    /// that one is [04-FIT-9] and is asserted by
    /// `a_never_resolved_declared_parameter_is_named_not_numbered`.
    pub(crate) fn message(&self, subject: &str) -> String {
        match self {
            Self::Copy { .. } => format!("copy requires tensor input, got {subject}"),
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
        }
    }

    /// What to call this constraint's call in a diagnostic.
    pub(crate) fn noun(&self) -> &str {
        match self {
            Self::Copy { .. } => "copy",
            Self::Cast { .. } | Self::CastToBinder { .. } => "cast",
            Self::HostSlot { fname, .. } => fname,
            Self::ShapeRoute { route, .. } => route.op(),
        }
    }

    /// The diagnostic kind the gate emitted when it decided eagerly.
    ///
    /// Deferring must not change the kind: a published vocabulary identity is
    /// something consumers count by name (chelis#1334 tallies these), so
    /// collapsing gates onto a different kind would be a wire change smuggled
    /// in behind a timing fix.
    pub(crate) fn kind(&self) -> crate::errors::CheckErrorKind {
        use crate::errors::CheckErrorKind as Kind;
        match self {
            Self::Cast { .. } | Self::CastToBinder { .. } => Kind::CastNonTensor,
            Self::Copy { .. } | Self::HostSlot { .. } | Self::ShapeRoute { .. } => {
                Kind::TypeMismatch
            }
        }
    }

    /// The repair hint, where the eager path carried one.
    pub(crate) fn suggestions(&self) -> Vec<String> {
        match self {
            Self::Copy { .. } => vec!["Wrap only tensor values in copy".to_string()],
            _ => Vec::new(),
        }
    }
}

/// Decide one consumed collection contract and reconcile its result.
///
/// `Unresolved` leaves the owning contract instance live until the current
/// application or declaration boundary supplies the missing evidence. Every
/// settled verdict removes the instance before entering this function, so
/// result unification cannot rediscover or cross-contaminate another call.
enum CollectionDischarge {
    Unresolved,
    Settled(Option<Type>),
}

fn discharge_collection_constraint(
    constraint: &crate::types::CollectionConstraint,
    tensor_concat: Option<&crate::infer::TensorConcatCallEvidence>,
    subst: &mut Subst,
) -> CollectionDischarge {
    match crate::infer::decide_collection_constraint(constraint, tensor_concat, subst) {
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
                error: crate::errors::CheckError::new(
                    crate::errors::CheckErrorKind::TypeMismatch,
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
/// The relation's fixed operand/result fields reconstruct its callable type.
/// Exact structural containment works for fully monomorphic contracts, while
/// the ordinary shared inference variables make the same test true for
/// polymorphic aliases and higher-order results after substitution.
pub(crate) fn collection_contract_visible_in_type(
    constraint: &CollectionConstraint,
    ty: &Type,
) -> bool {
    let callable = collection_contract_callable_type(constraint);
    fn contains(ty: &Type, callable: &Type) -> bool {
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

pub(crate) fn collection_contract_callable_type(constraint: &CollectionConstraint) -> Type {
    Type::Fn(
        constraint.operands().into_iter().cloned().collect(),
        Box::new(constraint.result().clone()),
    )
}

fn collect_non_callable_tvars(ty: &Type, vars: &mut UnordSet<TypeVar>) {
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
            for argument in arguments {
                collect_non_callable_tvars(argument, vars);
            }
        }
        Type::KindedAdt(_, arguments) => {
            for argument in arguments {
                if let NominalArg::Type(ty) = argument {
                    collect_non_callable_tvars(ty, vars);
                }
            }
        }
        Type::Prim(_) | Type::Tensor(_, TensorPrec::Concrete(_)) | Type::Unit | Type::Error(_) => {}
    }
}

/// Whether an unresolved transported function argument became tied to a
/// non-callable argument of this application. This is the higher-order
/// `invoke(len, x)` edge: `identity(len)` returns the callable and is handled
/// by `collection_contract_visible_in_type`, while `ignore(len, x)` leaves no
/// shared variable and may discard the unused capability.
fn collection_contract_tied_to_arguments(
    constraint: &CollectionConstraint,
    arguments: &[Type],
    subst: &Subst,
) -> bool {
    let mut argument_vars = UnordSet::default();
    for argument in arguments {
        collect_non_callable_tvars(&subst.apply(argument), &mut argument_vars);
    }
    constraint.operands().into_iter().any(|operand| {
        crate::env::free_tvars(&subst.apply(operand))
            .into_iter()
            .any(|variable| argument_vars.contains(&variable))
    })
}

/// Which deferred use shape registered a ledger entry (determines the
/// violation action text at validation time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredOpaqueUse {
    /// `(access target field)` with an unresolved target.
    Access,
    /// `(record-update target kv...)` with an unresolved target.
    RecordUpdate,
}

impl Clone for Subst {
    fn clone(&self) -> Self {
        Subst {
            result_constraints: Mutex::new(
                self.result_constraints
                    .lock()
                    .expect("result constraints poisoned")
                    .clone(),
            ),
            types: Mutex::new(self.types.lock().expect("subst.types poisoned").clone()),
            tvar_restrictions: Mutex::new(
                self.tvar_restrictions
                    .lock()
                    .expect("subst.tvar_restrictions poisoned")
                    .clone(),
            ),
            key_free_tvars: Mutex::new(
                self.key_free_tvars
                    .lock()
                    .expect("subst.key_free_tvars poisoned")
                    .clone(),
            ),
            key_carrying_adts: self.key_carrying_adts.clone(),
            dims: Mutex::new(self.dims.lock().expect("subst.dims poisoned").clone()),
            dimension_labels: Mutex::new(
                self.dimension_labels
                    .lock()
                    .expect("dimension labels poisoned")
                    .clone(),
            ),
            protected_dimensions: self.protected_dimensions.clone(),
            wildcard_touched_classes: self.wildcard_touched_classes.clone(),
            binder_pinned_classes: self.binder_pinned_classes.clone(),
            label_unification_depth: 0,
            dimension_label_undo: Mutex::new(Vec::new()),
            refined_type_bindings: Vec::new(),
            ranks: Mutex::new(self.ranks.lock().expect("subst.ranks poisoned").clone()),
            deferred_borrow_vars: Mutex::new(
                self.deferred_borrow_vars
                    .lock()
                    .expect("subst.deferred_borrow_vars poisoned")
                    .clone(),
            ),
            deferred_tensor_operands: Mutex::new(
                self.deferred_tensor_operands
                    .lock()
                    .expect("subst.deferred_tensor_operands poisoned")
                    .clone(),
            ),
            collection_contracts: Mutex::new(
                self.collection_contracts
                    .lock()
                    .expect("subst.collection_contracts poisoned")
                    .clone(),
            ),
            next_collection_contract_id: Mutex::new(
                *self
                    .next_collection_contract_id
                    .lock()
                    .expect("subst.next_collection_contract_id poisoned"),
            ),
            operand_gate_failures: Mutex::new(
                self.operand_gate_failures
                    .lock()
                    .expect("subst.operand_gate_failures poisoned")
                    .clone(),
            ),
            deferred_opaque_uses: Mutex::new(
                self.deferred_opaque_uses
                    .lock()
                    .expect("subst.deferred_opaque_uses poisoned")
                    .clone(),
            ),
            current_level: self.current_level,
            level_transitions: self.level_transitions.clone(),
            lowered_tvar_levels: self.lowered_tvar_levels.clone(),
            lowered_dvar_levels: self.lowered_dvar_levels.clone(),
            lowered_rvar_levels: self.lowered_rvar_levels.clone(),
            resume_floors: self.resume_floors,
        }
    }
}

impl Subst {
    pub(crate) fn record_result_constraint(&self, constraint: crate::types::ResultConstraint) {
        self.result_constraints
            .lock()
            .expect("result constraints poisoned")
            .push(constraint);
    }

    pub(crate) fn take_result_constraints(&self) -> Vec<crate::types::ResultConstraint> {
        std::mem::take(
            &mut *self
                .result_constraints
                .lock()
                .expect("result constraints poisoned"),
        )
    }
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn current_level(&self) -> u32 {
        self.current_level
    }

    /// Enter a lexical inference scope and record the first IDs minted there.
    pub(crate) fn enter_level(&mut self, var_gen: &VarGen) -> LevelToken {
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
    pub(crate) fn leave_level(&mut self, token: LevelToken, var_gen: &VarGen) {
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
    /// Existing IDs become level-zero imports; only work performed after this
    /// point contributes level metadata to the serialized context.
    pub(crate) fn resume_for_new_check(&mut self, var_gen: &VarGen) {
        self.protected_dimensions.clear();
        // chelis#1801: per-pass evidence, cleared at the pass boundary for the
        // same reason the protected set is. A resumed check reads it back
        // empty anyway (it is `serde(skip)`); this covers a live reuse.
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
        let transition_index = self.level_transitions.partition_point(|transition| {
            let watermark = match class {
                VarClass::Type => transition.watermarks.next_tvar,
                VarClass::Dim => transition.watermarks.next_dvar,
                VarClass::Rank => transition.watermarks.next_rvar,
            };
            watermark <= id
        });
        transition_index
            .checked_sub(1)
            .map_or(0, |index| self.level_transitions[index].level)
    }

    pub(crate) fn level_of_tvar(&self, var: TypeVar) -> u32 {
        self.lowered_tvar_levels
            .get(&var)
            .copied()
            .unwrap_or_else(|| self.mint_level(var.0, VarClass::Type))
    }

    pub(crate) fn level_of_dvar(&self, var: DimVar) -> u32 {
        self.lowered_dvar_levels
            .get(&var)
            .copied()
            .unwrap_or_else(|| self.mint_level(var.0, VarClass::Dim))
    }

    pub(crate) fn level_of_rvar(&self, var: RankVar) -> u32 {
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
    fn lower_type_to(&mut self, ty: &Type, level: u32) {
        let ty = self.apply(ty);
        for var in crate::env::free_tvars(&ty) {
            self.lower_tvar_to(var, level);
        }
        for var in crate::env::free_dvars(&ty) {
            self.lower_dvar_to(var, level);
        }
        for var in crate::env::free_rvars(&ty) {
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

    pub(crate) fn lower_type_to_current(&mut self, ty: &Type) {
        self.lower_type_to(ty, self.current_level);
    }

    /// Lower every variable reachable through `ty` to the enclosing `level`,
    /// so no scope entered since generalizes over it.
    pub(crate) fn lower_type_to_level(&mut self, ty: &Type, level: u32) {
        self.lower_type_to(ty, level);
    }

    #[cfg(test)]
    pub(crate) fn level_metadata_counts(&self) -> (usize, usize, usize, usize) {
        (
            self.level_transitions.len(),
            self.lowered_tvar_levels.len(),
            self.lowered_dvar_levels.len(),
            self.lowered_rvar_levels.len(),
        )
    }

    /// Snapshot of the type-variable bindings (cloned out of the lock).
    /// Useful for serialization, tests, and read-only inspection.
    pub fn types_snapshot(&self) -> UnordMap<TypeVar, Type> {
        self.types.lock().expect("subst.types poisoned").clone()
    }

    /// Snapshot of the dim-variable bindings.
    pub fn dims_snapshot(&self) -> UnordMap<DimVar, Dim> {
        self.dims.lock().expect("subst.dims poisoned").clone()
    }

    /// Number of type-variable bindings currently in the substitution.
    pub fn types_len(&self) -> usize {
        self.types.lock().expect("subst.types poisoned").len()
    }

    /// Number of dim-variable bindings currently in the substitution.
    pub fn dims_len(&self) -> usize {
        self.dims.lock().expect("subst.dims poisoned").len()
    }

    /// Record a type-variable binding through ordinary unification.
    ///
    /// This public mutation seam enforces occurs checks, semantic domains,
    /// deferred constraints, and restriction transfer exactly like every
    /// checker-created binding. It must never write the map directly.
    pub fn insert_type(&mut self, v: TypeVar, ty: Type) -> Result<(), TypeError> {
        unify(&Type::Var(v), &ty, self)
    }

    /// Publish a type binding after [`bind_tvar`] has validated the complete
    /// transaction. Keeping the raw map write private prevents callers from
    /// bypassing semantic domains through the public API.
    fn record_validated_type_binding(&mut self, v: TypeVar, ty: Type) {
        self.types
            .lock()
            .expect("subst.types poisoned")
            .insert(v, ty);
    }

    /// Attach a semantic domain to an unresolved inference variable,
    /// narrowing to the intersection with any domain it already carries.
    ///
    /// `spec/04-type-system.md` [04-DTYPE-2]: identifying two bounded
    /// variables yields the intersection of their families, and an empty
    /// intersection is a `PrecisionMismatch`. Narrowing here rather than
    /// overwriting is what keeps a `Numeric` alias from widening a `Float`
    /// variable back out to every numeric dtype.
    pub(crate) fn narrow_tvar_restriction(
        &self,
        v: TypeVar,
        restriction: TypeVarRestriction,
    ) -> Result<(), TypeError> {
        let mut restrictions = self
            .tvar_restrictions
            .lock()
            .expect("subst.tvar_restrictions poisoned");
        let narrowed = match restrictions.get(&v).copied() {
            Some(existing) => merge_tvar_restrictions(existing, restriction)?,
            None => restriction,
        };
        restrictions.insert(v, narrowed);
        Ok(())
    }

    /// Restriction currently attached to an unresolved variable, if any.
    pub fn tvar_restriction(&self, v: TypeVar) -> Option<TypeVarRestriction> {
        self.tvar_restrictions
            .lock()
            .expect("subst.tvar_restrictions poisoned")
            .get(&v)
            .copied()
    }

    fn tvar_restrictions_snapshot(&self) -> UnordMap<TypeVar, TypeVarRestriction> {
        self.tvar_restrictions
            .lock()
            .expect("subst.tvar_restrictions poisoned")
            .clone()
    }

    fn remove_tvar_restriction(&self, v: TypeVar) {
        self.tvar_restrictions
            .lock()
            .expect("subst.tvar_restrictions poisoned")
            .remove(&v);
    }

    /// [04-LIN-10]: mark `v` as standing for a function's type parameter. A
    /// variable already marked keeps its generic, so a violation names the
    /// first generic the variable was found to belong to.
    pub(crate) fn forbid_key_instantiation(&self, v: TypeVar, origin: GenericParameter) {
        self.key_free_tvars
            .lock()
            .expect("subst.key_free_tvars poisoned")
            .entry(v)
            .or_insert(origin);
    }

    /// The generic `v` stands for a type parameter of, when it is marked.
    pub(crate) fn key_free_origin(&self, v: TypeVar) -> Option<GenericParameter> {
        self.key_free_tvars
            .lock()
            .expect("subst.key_free_tvars poisoned")
            .get(&v)
            .cloned()
    }

    /// Name the generic that a binding's scheme belongs to on each of its
    /// quantified variables, with the authored binder each one instantiates
    /// when `binders` records it. An authored binder of this binding always
    /// names the variable. A variable this binding did not spell keeps the
    /// name of a generic it came from, so `t = (dup, 1)` then `(t.0)(k)` names
    /// `dup`'s parameter rather than the tuple.
    pub(crate) fn name_generic_parameters(
        &self,
        scheme: &Scheme,
        generic: &str,
        binders: &UnordMap<TypeVar, String>,
    ) {
        let binders = binders.to_sorted();
        let origin_scope = scheme
            .result_origin
            .as_ref()
            .map(|origin| crate::result_scope::ResultScope::new(&origin.equations));
        let closed_key_variables = crate::env::closed_key_relation_variables(scheme);
        let mut marks = self
            .key_free_tvars
            .lock()
            .expect("subst.key_free_tvars poisoned");
        // Equation-local origin quantifiers are not additional generic
        // parameters. An explicitly authored binder can still name every
        // representative of its equality component, including raw origins.
        let mut variables = scheme.tvars.clone();
        if let Some(origin) = &scheme.result_origin {
            for variable in &origin.tvars {
                if !variables.contains(variable) {
                    variables.push(*variable);
                }
            }
        }
        for tv in &variables {
            // A closed checked key relation is not an authored unconstrained
            // type parameter. Preserve any older generic mark, however: a
            // relation cannot erase the restriction of an enclosing binder.
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
    }

    /// Install spec/04 section 8.4.1's key-carrying data types for this check.
    pub(crate) fn set_key_carrying_adts(&mut self, adts: UnordSet<String>) {
        self.key_carrying_adts = adts;
    }

    /// Record a new dim-variable binding.
    pub fn insert_dim(&mut self, v: DimVar, dim: Dim) {
        self.dims
            .lock()
            .expect("subst.dims poisoned")
            .insert(v, dim);
    }

    /// Record a rank-variable binding: `r` stands for the whole shape `dims`.
    pub fn insert_rank(&mut self, r: RankVar, dims: Vec<Dim>) {
        self.ranks
            .lock()
            .expect("subst.ranks poisoned")
            .insert(r, dims);
    }

    pub(crate) fn static_dim_product(&self, dims: &[Dim]) -> Option<i128> {
        dims.iter().try_fold(1_i128, |product, dim| {
            let value = self.observe_dim(dim).literal_extent()?;
            product.checked_mul(i128::from(value))
        })
    }

    /// Compare two fully-static dimension products without a fixed-width
    /// multiplication. Pairwise GCD cancellation is exact for any number of
    /// i64 factors, so a large known shape never degrades into "unknown".
    pub(crate) fn static_dim_products_match(&self, lhs: &[Dim], rhs: &[Dim]) -> Option<bool> {
        fn values(subst: &Subst, dims: &[Dim]) -> Option<Vec<i64>> {
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
    pub fn ranks_snapshot(&self) -> UnordMap<RankVar, Vec<Dim>> {
        self.ranks.lock().expect("subst.ranks poisoned").clone()
    }

    /// Number of rank-variable bindings currently in the substitution.
    pub fn ranks_len(&self) -> usize {
        self.ranks.lock().expect("subst.ranks poisoned").len()
    }

    /// Resolve a rank variable to the shape vector it stands for, chasing
    /// rank-to-rank aliases. Returns `[Dim::Rank(r)]` (the unbound variable)
    /// when `r` has no binding. The returned dims are not themselves
    /// substitution-applied; callers run them through `apply_dim`.
    fn resolve_rvar(&self, start: RankVar) -> Vec<Dim> {
        let map = self.ranks.lock().expect("subst.ranks poisoned");
        let mut current = start;
        // Bound the alias chase. A chain cannot exceed the number of bound
        // ranks; a pathological cyclic alias (e.g. a swapped `..a`/`..b`
        // return sig that aliases a:=b and b:=a) is broken here by returning
        // the current rank rather than looping forever.
        let max_steps = map.len() + 1;
        for _ in 0..max_steps {
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
    }

    /// Suspend an operand decision on the variable that has to be bound
    /// before it can be made (chelis#1489). Unification discharges it at that
    /// binding; see the `deferred_tensor_operands` field doc.
    pub(crate) fn record_deferred_tensor_operand(&self, v: TypeVar, gate: DeferredOperandGate) {
        self.deferred_tensor_operands
            .lock()
            .expect("subst.deferred_tensor_operands poisoned")
            .push((v, gate));
    }

    /// Every variable a PENDING gate waits on or hands out (chelis#1489).
    ///
    /// Consulted by `Env::generalize`, which must not quantify any of them.
    ///
    /// The result: a suspended `copy`/`cast` hands its consumer a fresh result
    /// variable and ties it to the operand only through this ledger --
    /// invisibly to levels. When an unannotated `let` generalized that
    /// variable, every use of the bound name got its own unconstrained
    /// instance, and discharge later bound only the original: a declared
    /// result was never checked against what the call produces, and a false
    /// signature checked and ran.
    ///
    /// The operand: the gate is suspended on this one variable, and only its
    /// binding discharges the gate. When an unannotated `let` generalized it,
    /// every application of the bound lambda bound a fresh instance instead,
    /// the gate stayed on the never-bound template, and the declaration
    /// boundary rejected a well-typed program as unresolved (chelis#2584:
    /// `g = fn (y) -> cast(y, p)` then `g(x)`). Keeping it monomorphic is what
    /// [04-INF-1] requires of a lambda carrying a pending obligation, so its
    /// first application binds the operand and discharges the gate.
    ///
    /// ALL free variables are returned, not just a top-level type variable. A
    /// pending result can be partly unified before it discharges -- `g` meeting
    /// a `tensor[?d, 3, f32]` expectation makes it `tensor[?d, 3, f32]` -- and
    /// quantifying `?d` reopens the same hole.
    pub(crate) fn pending_gate_vars(
        &self,
    ) -> (UnordSet<TypeVar>, UnordSet<DimVar>, UnordSet<RankVar>) {
        let ledger = self
            .deferred_tensor_operands
            .lock()
            .expect("subst.deferred_tensor_operands poisoned");
        let mut tvars = UnordSet::default();
        let mut dvars = UnordSet::default();
        let mut rvars = UnordSet::default();
        // Almost every `generalize` runs with an empty ledger -- a suspended
        // operand decision is the exception, not the rule -- so leave without
        // applying the substitution or walking a type in that case.
        if ledger.is_empty() {
            return (tvars, dvars, rvars);
        }
        for (operand, gate) in ledger.iter() {
            let operand = self.apply(&Type::Var(*operand));
            let result = gate.result().map(|result| self.apply(result));
            for pending in std::iter::once(&operand).chain(result.as_ref()) {
                tvars.extend(crate::env::free_tvars(pending));
                dvars.extend(crate::env::free_dvars(pending));
                rvars.extend(crate::env::free_rvars(pending));
            }
        }
        (tvars, dvars, rvars)
    }

    /// chelis#2584, [04-INF-1]: whether a pending gate waits on or hands out a
    /// variable of `ty` that generalization at the current level would
    /// quantify. A `let`-bound lambda for which this holds carries a deferred
    /// obligation, so the whole lambda stays monomorphic until its first
    /// application, and every later use has that same instantiation; an
    /// unchecked second parameter does not become polymorphic beside a
    /// checked one.
    pub(crate) fn has_generalizable_pending_gate(&self, ty: &Type) -> bool {
        let (tvars, dvars, rvars) = self.pending_gate_vars();
        if tvars.is_empty() && dvars.is_empty() && rvars.is_empty() {
            return false;
        }
        let ty = self.apply(ty);
        let level = self.current_level();
        crate::env::free_tvars(&ty)
            .iter()
            .any(|v| tvars.contains(v) && self.level_of_tvar(*v) > level)
            || crate::env::free_dvars(&ty)
                .iter()
                .any(|v| dvars.contains(v) && self.level_of_dvar(*v) > level)
            || crate::env::free_rvars(&ty)
                .iter()
                .any(|v| rvars.contains(v) && self.level_of_rvar(*v) > level)
    }

    /// Record one fresh use of a checked collection-operation contract.
    pub(crate) fn record_collection_contract(
        &self,
        constraint: CollectionConstraint,
    ) -> CollectionContractId {
        let id = {
            let mut next = self
                .next_collection_contract_id
                .lock()
                .expect("subst.next_collection_contract_id poisoned");
            let id = *next;
            *next = next
                .checked_add(1)
                .expect("collection contract identity overflow");
            id
        };
        self.collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned")
            .push(CollectionContractInstance {
                id,
                level: self.current_level,
                constraint,
                state: CollectionContractState::Transport,
            });
        id
    }

    /// Mark the beginning of an expression's contract instantiations.
    pub(crate) fn collection_contract_mark(&self) -> CollectionContractId {
        *self
            .next_collection_contract_id
            .lock()
            .expect("subst.next_collection_contract_id poisoned")
    }

    /// Live contract instances minted since `mark`, in instantiation order.
    pub(crate) fn collection_contract_ids_since(
        &self,
        mark: CollectionContractId,
    ) -> Vec<CollectionContractId> {
        self.collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned")
            .iter()
            .filter_map(|instance| (instance.id >= mark).then_some(instance.id))
            .collect()
    }

    /// Transportable child-scope instances considered by `Env::generalize`.
    pub(crate) fn pending_collection_contracts(
        &self,
    ) -> Vec<(CollectionContractId, u32, CollectionConstraint)> {
        self.collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned")
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
    }

    /// Move or discard exact transport instances selected by generalization.
    pub(crate) fn take_collection_contracts(&self, removals: &[CollectionContractId]) {
        if removals.is_empty() {
            return;
        }
        self.collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned")
            .retain(|instance| !removals.contains(&instance.id));
    }

    /// Drop the scheme copies instantiated for a direct syntactic builtin call.
    pub(crate) fn discard_collection_contracts(&self, ids: &[CollectionContractId]) {
        self.take_collection_contracts(ids);
    }

    /// Whether these exact callee instances include tensor concat.
    pub(crate) fn collection_contracts_include_concat(
        &self,
        ids: &[CollectionContractId],
        callees: &[Type],
    ) -> bool {
        let callees = callees.iter().map(|ty| self.apply(ty)).collect::<Vec<_>>();
        self.collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned")
            .iter()
            .any(|instance| {
                ids.contains(&instance.id)
                    && matches!(instance.constraint, CollectionConstraint::Concat { .. })
                    && callees.contains(&collection_contract_callable_type(
                        &instance.constraint.map_types(|carried| self.apply(carried)),
                    ))
            })
    }

    /// Consume the exact checked contracts instantiated by this callee.
    pub(crate) fn prepare_collection_contract_call(
        &self,
        ids: &[CollectionContractId],
        callees: &[Type],
        tensor_concat: Option<crate::infer::TensorConcatCallEvidence>,
    ) {
        let callees = callees.iter().map(|ty| self.apply(ty)).collect::<Vec<_>>();
        let mut contracts = self
            .collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned");
        for instance in contracts.iter_mut() {
            if !ids.contains(&instance.id) {
                continue;
            }
            let normalized = instance.constraint.map_types(|carried| self.apply(carried));
            if !callees.contains(&collection_contract_callable_type(&normalized)) {
                continue;
            }
            let evidence = matches!(instance.constraint, CollectionConstraint::Concat { .. })
                .then(|| tensor_concat.clone())
                .flatten();
            instance.state = CollectionContractState::Consumed {
                tensor_concat: evidence,
            };
        }
    }

    /// Finish one application's contract lifecycle after ordinary call
    /// unification has tied all argument and result variables together.
    ///
    /// Exact callee instances are consumed. Contracts instantiated by
    /// function-valued arguments remain transport when the application result
    /// still contains their callable; otherwise a now-decidable relation is
    /// checked and an unresolved ignored value is discarded. No instance from
    /// before `mark` is examined.
    pub(crate) fn finish_collection_contract_application(
        &mut self,
        mark: CollectionContractId,
        arguments: &[Type],
        result: &Type,
        related_results: &[Type],
    ) -> Option<Type> {
        let ids = self.collection_contract_ids_since(mark);
        let applied_result = self.apply(result);
        let mut checked_result = None;
        for id in ids {
            let Some(instance) = self.take_collection_contract_instance(id) else {
                continue;
            };
            match instance.state.clone() {
                CollectionContractState::Consumed { tensor_concat } => {
                    match discharge_collection_constraint(
                        &instance.constraint,
                        tensor_concat.as_ref(),
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
                                consumed.state = CollectionContractState::Consumed {
                                    tensor_concat: None,
                                };
                                self.restore_collection_contract_instance(consumed);
                            }
                            CollectionDischarge::Settled(_) => {}
                        }
                    } else {
                        // A checked function argument not returned by the
                        // outer call is either consumed by the callee's type
                        // equations or ignored. A decidable relation owns its
                        // verdict; an unresolved ignored value owns no future
                        // application and is deliberately discarded.
                        let _discarded =
                            discharge_collection_constraint(&instance.constraint, None, self);
                    }
                }
            }
        }
        checked_result
    }

    /// Cancel every contract instance created while a failed application was
    /// inferred. This is the explicit cleanup edge that prevents a rejected
    /// call from lending evidence to a later valid one.
    pub(crate) fn cancel_collection_contract_application(&self, mark: CollectionContractId) {
        self.collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned")
            .retain(|instance| instance.id < mark);
    }

    fn take_collection_contract_instance(
        &self,
        id: CollectionContractId,
    ) -> Option<CollectionContractInstance> {
        let mut contracts = self
            .collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned");
        let index = contracts.iter().position(|instance| instance.id == id)?;
        Some(contracts.remove(index))
    }

    fn restore_collection_contract_instance(&self, instance: CollectionContractInstance) {
        let mut contracts = self
            .collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned");
        let index = contracts
            .iter()
            .position(|current| current.id > instance.id)
            .unwrap_or(contracts.len());
        contracts.insert(index, instance);
    }

    /// Drain unresolved consumed relations and detached child-scope transport
    /// at a declaration boundary. Transport born in the active recursive
    /// component's own level stays live until component generalization.
    pub(crate) fn take_boundary_collection_contracts(&self) -> Vec<CollectionConstraint> {
        let current_level = self.current_level;
        let mut contracts = self
            .collection_contracts
            .lock()
            .expect("subst.collection_contracts poisoned");
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
    }

    /// Record a discharge failure for the per-def reporting pass
    /// (chelis#1489).
    pub(crate) fn record_operand_gate_failure(&self, failure: OperandGateFailure) {
        self.operand_gate_failures
            .lock()
            .expect("subst.operand_gate_failures poisoned")
            .push(failure);
    }

    /// Drain the discharge failures. Called once per def body, with
    /// [`Self::take_deferred_tensor_operands`], by the reporting pass.
    pub(crate) fn take_operand_gate_failures(&self) -> Vec<OperandGateFailure> {
        std::mem::take(
            &mut *self
                .operand_gate_failures
                .lock()
                .expect("subst.operand_gate_failures poisoned"),
        )
    }

    /// Take every constraint suspended on `v`, leaving the rest of the ledger
    /// in place (chelis#1489).
    ///
    /// Discharge removes a constraint BEFORE deciding it, so a decision that
    /// unifies -- and therefore re-enters this -- cannot rediscover the
    /// constraint it is in the middle of discharging.
    ///
    /// The ledger does NOT strictly shrink: binding a variable to another
    /// VARIABLE re-suspends the obligation on the target, so an entry can be
    /// removed and re-added. Termination rests on the alias chain being
    /// acyclic -- `bind_tvar_inner` rejects self-binding and `occurs_in`
    /// rejects cycles -- not on a shrinking count.
    fn take_operand_gates_on(&self, v: TypeVar) -> Vec<DeferredOperandGate> {
        let mut ledger = self
            .deferred_tensor_operands
            .lock()
            .expect("subst.deferred_tensor_operands poisoned");
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
    }

    /// Re-suspend `gate` on `target` because `v` was bound to it rather than
    /// to a concrete type (chelis#1489).
    ///
    /// Identifying two variables must carry the obligation across, exactly as
    /// the shape ledgers' `merge_alias` does; dropping it here would silently
    /// un-defer the constraint.
    /// Remove and decide every suspended scalar `Cast` or `CastToBinder` gate
    /// whose operand now resolves to a variable with a declared dtype-family
    /// bound, when that bound alone settles it (chelis#2151, chelis#2534). See
    /// [`discharge_bounded_scalar_casts`].
    ///
    /// Almost every binding runs with an empty ledger, so it returns early
    /// without applying the substitution in that case.
    #[allow(clippy::type_complexity)]
    fn take_bounded_scalar_cast_gates(
        &self,
    ) -> Vec<(
        DeferredOperandGate,
        Result<Type, Box<crate::errors::CheckError>>,
    )> {
        let mut ledger = self
            .deferred_tensor_operands
            .lock()
            .expect("subst.deferred_tensor_operands poisoned");
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
            // Any §5.9 declaration bound discharges this gate, including the
            // explicit set form. Listing the three families rejected a
            // set-bounded scalar `cast` that every equivalent family accepted
            // (chelis#2443 round 2).
            let Some(bound) = self
                .tvar_restriction(operand)
                .filter(|restriction| restriction.is_declaration_bound())
            else {
                return true;
            };
            let decision = match gate {
                DeferredOperandGate::Cast { target, mode, .. } => {
                    crate::infer::expr_record::bounded_scalar_cast_result(bound, *target, *mode)
                }
                // chelis#2534: every member of a numeric bound is a scalar
                // source a binder-target `cast` admits ([05-OP-63]).
                DeferredOperandGate::CastToBinder { target, .. } => Some(Ok(Type::Var(*target))),
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
    }

    fn realias_operand_gate(&self, target: TypeVar, gate: DeferredOperandGate) {
        self.deferred_tensor_operands
            .lock()
            .expect("subst.deferred_tensor_operands poisoned")
            .push((target, gate));
    }

    /// Drain the deferred tensor-operand ledger. Called once per def body's
    /// inference so one def's deferrals cannot leak into the next.
    pub(crate) fn take_deferred_tensor_operands(&self) -> Vec<(TypeVar, DeferredOperandGate)> {
        std::mem::take(
            &mut *self
                .deferred_tensor_operands
                .lock()
                .expect("subst.deferred_tensor_operands poisoned"),
        )
    }

    /// Issue #256: record a borrow site whose inner type was still an
    /// unresolved `Type::Var` when the `borrow` inference arm ran. The
    /// driver drains these after a def body's inference completes and
    /// re-checks each against the final substitution. See the
    /// `deferred_borrow_vars` field doc for the soundness rationale.
    pub fn record_deferred_borrow_var(&self, v: TypeVar) {
        self.deferred_borrow_vars
            .lock()
            .expect("subst.deferred_borrow_vars poisoned")
            .push(v);
    }

    /// Drain the deferred-borrow ledger, returning every recorded
    /// `TypeVar`. Called once per def body's inference by the driver so
    /// the ledger does not leak deferred sites across defs.
    pub fn take_deferred_borrow_vars(&self) -> Vec<TypeVar> {
        std::mem::take(
            &mut *self
                .deferred_borrow_vars
                .lock()
                .expect("subst.deferred_borrow_vars poisoned"),
        )
    }

    /// RFC D-CHECK: record an `access`/`record-update` site whose
    /// target type was still an unresolved `Type::Var` when inference
    /// ran. See the `deferred_opaque_uses` field doc.
    pub fn record_deferred_opaque_use(&self, v: TypeVar, use_kind: DeferredOpaqueUse) {
        self.deferred_opaque_uses
            .lock()
            .expect("subst.deferred_opaque_uses poisoned")
            .push((v, use_kind));
    }

    /// Drain the deferred-access ledger; called once per def body's
    /// inference by the driver, mirroring `take_deferred_borrow_vars`.
    pub fn take_deferred_opaque_uses(&self) -> Vec<(TypeVar, DeferredOpaqueUse)> {
        std::mem::take(
            &mut *self
                .deferred_opaque_uses
                .lock()
                .expect("subst.deferred_opaque_uses poisoned"),
        )
    }

    /// Resolve a type variable to its terminal binding (a non-Var, or
    /// an unbound Var). Iteratively walks the chain and rewrites every
    /// link directly to the terminal so future lookups are O(1).
    fn resolve_tvar(&self, start: TypeVar) -> Type {
        let mut map = self.types.lock().expect("subst.types poisoned");
        // Phase 1: walk to the terminal, recording the chain.
        let mut chain: Vec<TypeVar> = Vec::new();
        let mut current = start;
        let terminal = loop {
            match map.get(&current) {
                None => break Type::Var(current),
                Some(Type::Var(v)) if *v == current => break Type::Var(current),
                Some(Type::Var(v)) => {
                    chain.push(current);
                    current = *v;
                }
                Some(other) => {
                    chain.push(current);
                    break other.clone();
                }
            }
        };
        // Keep the terminal binding's owner: a shared tensor type may later
        // acquire a protected dimension identity while retaining its name.
        if chain.len() > 1 {
            for v in chain {
                if v != current {
                    map.insert(v, Type::Var(current));
                }
            }
        }
        terminal
    }

    /// Resolve a dim variable to its terminal binding. Same iterative-
    /// walk-then-compress shape as [`Subst::resolve_tvar`].
    fn resolve_dvar(&self, start: DimVar) -> Dim {
        let mut map = self.dims.lock().expect("subst.dims poisoned");
        let mut chain: Vec<DimVar> = Vec::new();
        let mut current = start;
        let terminal = loop {
            match map.get(&current) {
                None => break Dim::Var(current),
                Some(Dim::Var(v)) if *v == current => break Dim::Var(current),
                Some(Dim::Var(v)) => {
                    chain.push(current);
                    current = *v;
                }
                Some(other) => {
                    chain.push(current);
                    break other.clone();
                }
            }
        };
        if chain.len() > 1 {
            for v in chain {
                map.insert(v, terminal.clone());
            }
        }
        terminal
    }

    fn place_mapped_axes(mut base: Vec<Dim>, mut mapped: Vec<(usize, DimVar, Dim)>) -> Vec<Dim> {
        let deferred = |var, resolved: Dim| match resolved {
            Dim::Var(_) => resolved,
            _ => Dim::Var(var),
        };
        mapped.sort_by_key(|(axis, var, _)| (*axis, var.0));
        if base.iter().any(|dim| matches!(dim, Dim::Rank(_))) {
            // An unresolved spread has no physical width yet. Keep the
            // compiler-owned identities present; row unification uses their
            // recorded axes rather than this storage order.
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
                // Bounds checking belongs to vmap inference/application. An
                // unresolved stored transform must remain representable until
                // that boundary reports the user-facing error.
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
    /// Path-compresses any chains of length ≥ 2 it encounters so future
    /// lookups land in O(1).
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => match self.resolve_tvar(*v) {
                Type::Var(variable) => Type::Var(variable),
                bound => self.apply(&bound),
            },
            Type::Fn(args, ret) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                let ret = Box::new(self.apply(ret));
                Type::Fn(args, ret)
            }
            Type::Ref(inner) => Type::Ref(Box::new(self.apply(inner))),
            Type::Tensor(dims, prec) => {
                Type::Tensor(self.apply_tensor_dims(dims), self.apply_tensor_prec(prec))
            }
            Type::Adt(name, args) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                Type::Adt(name.clone(), args)
            }
            Type::KindedAdt(name, args) => Type::KindedAdt(
                name.clone(),
                args.iter()
                    .map(|argument| match argument {
                        NominalArg::Type(ty) => NominalArg::Type(self.apply(ty)),
                        NominalArg::Dimension(dim) => NominalArg::Dimension(self.apply_dim(dim)),
                    })
                    .collect(),
            ),
            Type::Tuple(ts) => {
                let ts = ts.iter().map(|t| self.apply(t)).collect();
                Type::Tuple(ts)
            }
            Type::Prim(_) | Type::Unit | Type::Error(_) => ty.clone(),
        }
    }

    /// Apply this substitution to a scheme body without substituting through
    /// the scheme's universally quantified variables.
    ///
    /// A global inference substitution can contain bindings whose numeric IDs
    /// coincide with a quantified variable in an environment scheme. Those
    /// bindings belong to an instantiation, not to the scheme itself. Applying
    /// them while computing the environment's free variables turns a
    /// quantified dimension into a free one and prevents later
    /// generalization (chelis#968).
    pub fn apply_scheme(&self, scheme: &Scheme) -> Type {
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
        ty: &Type,
        quantified_tvars: &chelis_unord::UnordSet<TypeVar>,
        quantified_dvars: &chelis_unord::UnordSet<DimVar>,
        quantified_rvars: &chelis_unord::UnordSet<RankVar>,
    ) -> Type {
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
                Box::new(self.apply_excluding(
                    ret,
                    quantified_tvars,
                    quantified_dvars,
                    quantified_rvars,
                )),
            ),
            Type::Ref(inner) => Type::Ref(Box::new(self.apply_excluding(
                inner,
                quantified_tvars,
                quantified_dvars,
                quantified_rvars,
            ))),
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
    ) -> Type {
        let map = self.types.lock().expect("subst.types poisoned");
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
    }

    fn apply_dim_excluding(&self, dim: &Dim, quantified: &chelis_unord::UnordSet<DimVar>) -> Dim {
        let Dim::Var(start) = dim else {
            return dim.clone();
        };
        let map = self.dims.lock().expect("subst.dims poisoned");
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
    }

    fn resolve_rvar_excluding(
        &self,
        start: RankVar,
        quantified: &chelis_unord::UnordSet<RankVar>,
    ) -> Vec<Dim> {
        let map = self.ranks.lock().expect("subst.ranks poisoned");
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
    }

    /// Apply this substitution to a dimension. Path-compresses chains.
    pub fn apply_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => {
                if self.mapped_axis(*v).is_some() {
                    // The tensor-shape applicator must see the transform
                    // boundary before it can place the resolved batch
                    // identity at its requested axis.
                    return Dim::Var(*v);
                }
                let resolved = self.resolve_dvar(*v);
                if self.dimension_label(*v).is_some() && !matches!(resolved, Dim::Var(_)) {
                    // Keep the label's identity available to aliases and
                    // name-sensitive operations, even after a concrete call.
                    Dim::Var(*v)
                } else {
                    resolved
                }
            }
            _ => dim.clone(),
        }
    }

    pub(crate) fn constraint_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => self.resolve_dvar(*v),
            _ => dim.clone(),
        }
    }

    /// Resolve the shape run currently assigned to a rank variable, preserving
    /// a sole unbound/aliased `Dim::Rank` identity for rigidity checks.
    pub(crate) fn constraint_rank(&self, rank: RankVar) -> Vec<Dim> {
        self.resolve_rvar(rank)
            .into_iter()
            .map(|dim| self.constraint_dim(&dim))
            .collect()
    }

    pub(crate) fn observe_dim(&self, dim: &Dim) -> DimObservation {
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

    pub(crate) fn dimension_label(&self, v: DimVar) -> Option<String> {
        self.raw_dimension_annotation(v)
            .filter(|annotation| parse_mapped_axis_annotation(annotation).is_none())
    }

    fn raw_dimension_annotation(&self, v: DimVar) -> Option<String> {
        let root = self.resolve_dvar(v);
        let labels = self
            .dimension_labels
            .lock()
            .expect("dimension labels poisoned");
        labels
            .get(&v)
            .or_else(|| match root {
                Dim::Var(root) => labels.get(&root),
                _ => None,
            })
            .cloned()
    }

    pub(crate) fn mapped_axis(&self, v: DimVar) -> Option<usize> {
        let labels = self
            .dimension_labels
            .lock()
            .expect("dimension labels poisoned");
        labels
            .get(&v)
            .and_then(|annotation| parse_mapped_axis_annotation(annotation))
    }

    pub(crate) fn mark_mapped_axis(&self, var: DimVar, axis: usize) {
        self.set_dimension_label(var, mapped_axis_annotation(axis));
    }

    pub(crate) fn copy_dimension_label(&self, old: DimVar, fresh: DimVar) {
        if let Some(annotation) = self.raw_dimension_annotation(old) {
            self.set_dimension_label(fresh, annotation);
        }
    }

    fn set_dimension_label(&self, var: DimVar, label: String) {
        // The only nested lock order is labels -> undo. Rollback has exclusive
        // access to Subst and uses get_mut, so it never acquires the reverse.
        let mut labels = self
            .dimension_labels
            .lock()
            .expect("dimension labels poisoned");
        if labels.get(&var) == Some(&label) {
            return;
        }
        let previous = labels.insert(var, label);
        if self.label_unification_depth != 0 {
            self.dimension_label_undo
                .lock()
                .expect("dimension label undo poisoned")
                .push((var, previous));
        }
    }

    fn finish_dimension_label_writes(&mut self, rollback: bool) {
        let undo = self
            .dimension_label_undo
            .get_mut()
            .expect("dimension label undo poisoned");
        if rollback && !undo.is_empty() {
            let labels = self
                .dimension_labels
                .get_mut()
                .expect("dimension labels poisoned");
            for (var, previous) in undo.drain(..).rev() {
                if let Some(label) = previous {
                    labels.insert(var, label);
                } else {
                    labels.remove(&var);
                }
            }
        } else {
            undo.clear();
        }
    }

    pub(crate) fn semantic_dim(&self, dim: &Dim) -> Dim {
        if let Dim::Var(v) = dim
            && self.mapped_axis(*v).is_none()
            && let Some(label) = self.dimension_label(*v)
        {
            return Dim::Name(label);
        }
        self.constraint_dim(dim)
    }

    /// Semantic annotation/name-query view. Never use this before the authored
    /// result guards: projecting a labelled identity to Name erases rigidity.
    pub(crate) fn semantic_type(&self, ty: &Type) -> Type {
        match self.apply(ty) {
            Type::Tensor(dims, prec) => {
                Type::Tensor(dims.iter().map(|d| self.semantic_dim(d)).collect(), prec)
            }
            Type::Fn(args, ret) => Type::Fn(
                args.iter().map(|a| self.semantic_type(a)).collect(),
                Box::new(self.semantic_type(&ret)),
            ),
            Type::Ref(inner) => Type::Ref(Box::new(self.semantic_type(&inner))),
            Type::Tuple(items) => {
                Type::Tuple(items.iter().map(|t| self.semantic_type(t)).collect())
            }
            Type::Adt(name, args) => {
                Type::Adt(name, args.iter().map(|t| self.semantic_type(t)).collect())
            }
            Type::KindedAdt(name, args) => Type::KindedAdt(
                name,
                args.iter()
                    .map(|a| match a {
                        NominalArg::Type(t) => NominalArg::Type(self.semantic_type(t)),
                        NominalArg::Dimension(d) => NominalArg::Dimension(self.semantic_dim(d)),
                    })
                    .collect(),
            ),
            other => other,
        }
    }

    /// The variable that currently roots `v`'s alias class.
    ///
    /// A class already bound to a concrete dimension has no variable root and
    /// answers `v` itself. That case never reaches the class flags: the
    /// absorbing site skips it on `constraint_dim`'s answer first, because a
    /// bound class denotes what it is bound to.
    fn dvar_class_root(&self, v: DimVar) -> DimVar {
        match self.resolve_dvar(v) {
            Dim::Var(root) => root,
            _ => v,
        }
    }

    /// chelis#1801: record that `v`'s alias class met a runtime extent `*`
    /// without being bound to it. Written only by [`unify_dim`]'s
    /// wildcard-against-variable arm; read only by the application rule that
    /// absorbs the extent.
    pub(crate) fn note_wildcard_touch(&mut self, v: DimVar) {
        let root = self.dvar_class_root(v);
        self.wildcard_touched_classes.insert(root);
    }

    /// chelis#1801: whether `v`'s alias class has met a runtime extent `*`.
    ///
    /// This answers only "did the meeting happen". Whether the class still
    /// denotes that extent is a separate question the caller asks of the
    /// substitution, because an argument in the same call may have bound it
    /// to a literal after the meeting.
    pub(crate) fn dvar_class_met_wildcard(&self, v: DimVar) -> bool {
        self.wildcard_touched_classes
            .contains(&self.dvar_class_root(v))
    }

    /// chelis#1801: whether a declared dimension pins `v`'s alias class.
    ///
    /// `def outer(s: tensor[seq, f32])` binds the class to the named
    /// dimension `seq`, which is section 3.2's exclusion, and it reaches that
    /// exclusion through the rigidity ledger rather than through a `Dim::Name`
    /// unification: the name arm in `bind_dvar` records the name as a label
    /// and binds nothing. The set carries the same answer for a class whose
    /// protected member a merge or a compose demoted out of the root.
    pub(crate) fn dvar_class_is_binder_pinned(&self, v: DimVar) -> bool {
        let root = self.dvar_class_root(v);
        self.binder_pinned_classes.contains(&root) || self.protected_dimensions.contains(&root)
    }

    /// chelis#1801: carry both class flags from `from`'s class onto `into`'s.
    ///
    /// Called by [`bind_dvar`] before every merge. A merged class keeps one
    /// root, so evidence recorded on the other root would otherwise become
    /// unreachable, which is the defect chelis#1925's rounds 1 and 2 both
    /// reported.
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

    /// chelis#1801: re-canonicalize both operands' class evidence through the
    /// COMPOSED alias graph.
    ///
    /// Reached from [`Self::compose`], and a set union is the wrong operation
    /// here. `compose_bindings` merges only the binding maps, so composition
    /// can JOIN two classes that were separate in either operand: a flag
    /// keyed to the variable that rooted its class before the compose is then
    /// keyed to a member that no longer roots it, and the query on the new
    /// root answers false. Every incoming key is therefore re-resolved
    /// against the composed graph and re-keyed to the root it now has, which
    /// is exactly what the label pass in [`Self::compose`] does and for the
    /// same reason.
    ///
    /// Protection is folded in on the way through. The pin query reads
    /// `protected_dimensions` at the root, so a protected member that the
    /// compose demoted out of the root position would otherwise stop pinning
    /// its class. Carrying the other operand's protection here is also the
    /// conservative direction: it can only withhold an absorption, never
    /// erase a claim.
    fn recanonicalize_class_flags(&mut self, left: &Subst, right: &Subst) {
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

    pub(crate) fn protect_dimensions(&mut self, vars: impl IntoIterator<Item = DimVar>) {
        for v in vars {
            self.protected_dimensions.insert(v);
        }
    }

    /// Current-check authored rigidity, never historical snapshot metadata.
    /// Instantiation freshens quantified IDs without copying protection;
    /// lexical captures keep their canonical ID. Both live and decoded imports
    /// clear historical protection in `resume_for_new_check` before inference.
    pub(crate) fn is_protected_dimension(&self, var: DimVar) -> bool {
        matches!(self.resolve_dvar(var), Dim::Var(root) if self.protected_dimensions.contains(&root))
    }

    pub(crate) fn validate_dimension_labels(&self, var_gen: &VarGen) -> Result<(), &'static str> {
        for (var, label) in self
            .dimension_labels
            .lock()
            .expect("dimension labels poisoned")
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

    fn type_binding_owner(&self, mut v: TypeVar) -> TypeVar {
        let map = self.types.lock().expect("subst.types poisoned");
        while let Some(Type::Var(next)) = map.get(&v) {
            if *next == v {
                break;
            }
            v = *next;
        }
        v
    }

    /// Apply this substitution to a tensor precision slot.
    ///
    /// Resolves a `TensorPrec::Var` through the type-variable
    /// substitution chain. If the resolved binding is a concrete
    /// primitive, the slot collapses to `TensorPrec::Concrete(_)`. If
    /// the binding is itself another type variable, we keep the slot
    /// as `TensorPrec::Var(target_var)` so the tensor stays
    /// well-formed (the precision slot must be a precision, not an
    /// arbitrary `Type`). Any non-prim, non-var binding (e.g. a
    /// downstream unification error that bound the slot var to
    /// `Type::Fn` or `Type::Tensor`) collapses to `Type::Error` at
    /// the surrounding `Type` level — but here we conservatively
    /// retain the original var; the unifier surfaces the mismatch.
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
    /// Composition is transactional because independent substitutions can
    /// carry a binding and a semantic restriction for the same variable.
    /// Restrictions from both operands are canonicalized through the merged
    /// binding graph; a forbidden concrete resolution rejects the complete
    /// compose and leaves `self` unchanged.
    pub fn compose(&mut self, other: &Subst) -> Result<(), TypeError> {
        let mut trial = self.clone();
        trial.compose_bindings(other);
        // chelis#1801: `compose_bindings` merges the BINDINGS; the class
        // evidence is separate and has to be re-canonicalized through the
        // composed alias graph, never copied, because the compose can join
        // two classes that were separate in either operand.
        trial.recanonicalize_class_flags(self, other);
        // Re-canonicalize BOTH operands' labels through the composed alias
        // graph; merely copying incoming entries misses a newly joined class.
        let mut incoming = Vec::new();
        let mut mapped_axes = Vec::new();
        for source in [&*self, other] {
            for (v, annotation) in source
                .dimension_labels
                .lock()
                .expect("dimension labels poisoned")
                .to_sorted()
            {
                if parse_mapped_axis_annotation(annotation).is_some() {
                    mapped_axes.push((*v, annotation.clone()));
                } else {
                    incoming.push((*v, annotation.clone()));
                }
            }
        }
        trial
            .dimension_labels
            .lock()
            .expect("dimension labels poisoned")
            .clear();
        for (v, label) in incoming {
            let target = trial.constraint_dim(&Dim::Var(v));
            let mut labels = trial
                .dimension_labels
                .lock()
                .expect("dimension labels poisoned");
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
        }
        for (v, annotation) in mapped_axes {
            let mut labels = trial
                .dimension_labels
                .lock()
                .expect("dimension labels poisoned");
            if let Some(existing) = labels.get(&v)
                && existing != &annotation
            {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: format!("conflicting mapped-axis annotations while composing {v:?}"),
                });
            }
            labels.insert(v, annotation);
        }

        let mut restrictions = self.tvar_restrictions_snapshot();
        for (var, incoming) in other.tvar_restrictions_snapshot().into_sorted() {
            let narrowed = match restrictions.get(&var).copied() {
                Some(existing) => merge_tvar_restrictions(existing, incoming)?,
                None => incoming,
            };
            restrictions.insert(var, narrowed);
        }

        trial
            .tvar_restrictions
            .lock()
            .expect("subst.tvar_restrictions poisoned")
            .clear();
        for (source, restriction) in restrictions.into_sorted() {
            let resolved = trial.resolve_tvar(source);
            ensure_tvar_restriction(restriction, &resolved, &trial)?;
            if let Type::Var(target) = resolved {
                trial.narrow_tvar_restriction(target, restriction)?;
            }
        }

        // [04-LIN-10]: both operands' key-free marks hold in the composed
        // graph, so each is re-checked against what its variable now resolves
        // to, exactly like a dtype family above.
        let incoming_marks = other
            .key_free_tvars
            .lock()
            .expect("subst.key_free_tvars poisoned")
            .clone()
            .into_sorted();
        for (source, origin) in incoming_marks {
            trial.forbid_key_instantiation(source, origin);
        }
        let marks = trial
            .key_free_tvars
            .lock()
            .expect("subst.key_free_tvars poisoned")
            .clone()
            .into_sorted();
        for (source, origin) in marks {
            let resolved = trial.resolve_tvar(source);
            ensure_key_free(&origin, &resolved, &resolved, &trial)?;
        }
        trial
            .key_carrying_adts
            .merge(other.key_carrying_adts.clone());

        *self = trial;
        Ok(())
    }

    /// Project semantic domains from an inferred implementation type onto a
    /// structurally corresponding declared type.
    ///
    /// Most declared definitions use ordinary unification, which transfers
    /// restrictions while binding matching variables. A small set of exact
    /// stdlib contracts deliberately retains its declared type when unrelated
    /// shape relations are not yet procedurally inferable. This projection
    /// preserves the semantic domains learned while checking those bodies
    /// without weakening them into an unconstrained exported function value.
    /// The operation is transactional so a later conflicting slot cannot
    /// publish an earlier partial transfer.
    pub(crate) fn project_tvar_restrictions(
        &mut self,
        inferred: &Type,
        declared: &Type,
    ) -> Result<(), TypeError> {
        let mut trial = self.clone();
        trial.project_tvar_restrictions_inner(inferred, declared)?;
        *self = trial;
        Ok(())
    }

    fn project_tvar_restrictions_inner(
        &mut self,
        inferred: &Type,
        declared: &Type,
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
                for (inferred, declared) in inferred_args.iter().zip(declared_args) {
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
                for (inferred, declared) in inferred_args.iter().zip(declared_args) {
                    self.project_tvar_restrictions_inner(inferred, declared)?;
                }
                Ok(())
            }
            (
                Type::KindedAdt(inferred_name, inferred_args),
                Type::KindedAdt(declared_name, declared_args),
            ) if inferred_name == declared_name && inferred_args.len() == declared_args.len() => {
                for (inferred, declared) in inferred_args.iter().zip(declared_args) {
                    if let (NominalArg::Type(inferred), NominalArg::Type(declared)) =
                        (inferred, declared)
                    {
                        self.project_tvar_restrictions_inner(inferred, declared)?;
                    }
                }
                Ok(())
            }
            (Type::Tuple(inferred), Type::Tuple(declared)) if inferred.len() == declared.len() => {
                for (inferred, declared) in inferred.iter().zip(declared) {
                    self.project_tvar_restrictions_inner(inferred, declared)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn compose_bindings(&mut self, other: &Subst) {
        // Note: `other.apply` / `other.apply_dim` lock `other`'s maps;
        // we must not be holding a lock on `other` simultaneously. This
        // helper runs on a clone, so it is distinct even for `s.compose(&s)`.
        {
            let mut self_types = self.types.lock().expect("subst.types poisoned");
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
        }
        {
            let mut self_dims = self.dims.lock().expect("subst.dims poisoned");
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
        }
        {
            // Rank bindings are `Dim::Rank`-free runs (enforced by `bind_rvar`),
            // so applying `other` is a per-dim `apply_dim`. Omitting this would
            // silently drop rank substitutions through a compose — a Tier-3
            // footgun since ranks are now load-bearing.
            let mut self_ranks = self.ranks.lock().expect("subst.ranks poisoned");
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
        }
        let other_types = other.types_snapshot();
        let other_dims = other.dims_snapshot();
        let other_ranks = other.ranks_snapshot();
        {
            let mut self_types = self.types.lock().expect("subst.types poisoned");
            for (k, v) in other_types.into_sorted() {
                self_types.entry(k).or_insert(v);
            }
        }
        {
            let mut self_dims = self.dims.lock().expect("subst.dims poisoned");
            for (k, v) in other_dims.into_sorted() {
                self_dims.entry(k).or_insert(v);
            }
        }
        {
            let mut self_ranks = self.ranks.lock().expect("subst.ranks poisoned");
            for (k, v) in other_ranks.into_sorted() {
                self_ranks.entry(k).or_insert(v);
            }
        }
        // `deferred_borrow_vars` is intentionally NOT merged: it is a transient
        // per-def ledger (issue #256), drained after each body's inference, not
        // part of the substitution's logical content.
    }
}

/// Combine the bounds of two variables being identified.
///
/// [04-DTYPE-2] makes this the family intersection rather than equality: a
/// `Numeric`-bounded variable may legitimately be identified with a `Float`
/// one, and the result admits only floats. Only `Float` against `Int` is
/// empty, and an empty intersection is a `PrecisionMismatch` naming both
/// families.
/// How a clash between two bounds is named.
///
/// The all-families form is byte-identical to the wording that predates
/// §5.9's set form, which a per-operand `bound_description()` would not be:
/// the original sentence hoists "dtype families" as a shared plural, so
/// describing each side separately doubles the phrase. Nothing pinned that
/// text, so it churned silently until red-team round 3 measured it
/// differentially against the base.
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
pub fn unify(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    let outer = subst.label_unification_depth == 0;
    subst.label_unification_depth += 1;
    let result = unify_preserving_owners(t1, t2, subst);
    subst.label_unification_depth -= 1;
    if outer {
        subst.finish_dimension_label_writes(result.is_err());
        if result.is_err() {
            let mut types = subst.types.lock().expect("subst.types poisoned");
            for (v, old) in subst.refined_type_bindings.drain(..).rev() {
                types.insert(v, old);
            }
        } else {
            subst.refined_type_bindings.clear();
        }
    }
    result
}

fn unify_preserving_owners(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
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
    // Resolve only the outer variable. Recursion must still see a shared T
    // under Ref/Fn rather than an eagerly copied tensor payload.
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
                        .lock()
                        .expect("subst.types poisoned")
                        .insert(owner, Type::Tensor(refined, prec.clone()));
                }
            }
        }
    }
    result
}

fn unify_resolved(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    match (&t1, &t2) {
        // Same type — trivially unified
        (Type::Prim(p1), Type::Prim(p2)) if p1 == p2 => Ok(()),
        (Type::Prim(p1), Type::Prim(p2)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!("precision mismatch: {} vs {}", p1.name(), p2.name()),
        }),

        (Type::Unit, Type::Unit) => Ok(()),

        (Type::Ref(inner1), Type::Ref(inner2)) => unify(inner1, inner2, subst),

        // Type variable binding
        (Type::Var(v), _) => bind_tvar(*v, t2, subst),
        (_, Type::Var(v)) => bind_tvar(*v, t1, subst),

        // Function types
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
            // chelis#339: after the signature's rank spreads are bound, a
            // result-introduced axis name (a named-axis expand insert) must
            // not also be covered by a spread binding — the monomorphized
            // result row would carry the same dim name twice with two
            // different extents, making every later by-name lookup ambiguous.
            check_introduced_name_rank_collision(args1, ret1, subst)?;
            check_introduced_name_rank_collision(args2, ret2, subst)
        }

        // Tensor types
        (Type::Tensor(dims1, p1), Type::Tensor(dims2, p2)) => {
            unify_tensor_prec(p1, p2, subst)?;
            // Rank polymorphism: a `Dim::Rank` is a *name-preserving spread*
            // standing for a run of dims. A shape is `Rank? (Name Rank?)*`
            // (no two spreads adjacent). Resolve both shapes (expanding any
            // bound spread to its run), then dispatch on how many spreads each
            // side carries. With at most one spread per gap and a named anchor
            // locating each interior split, unification stays unitary — there
            // is one most-general binding. See
            // `spec/design/rank_polymorphism.md` §Unification.
            let d1 = resolve_shape(dims1, subst);
            let d2 = resolve_shape(dims2, subst);
            let s1 = d1.iter().filter(|d| matches!(d, Dim::Rank(_))).count();
            let s2 = d2.iter().filter(|d| matches!(d, Dim::Rank(_))).count();
            match (s1, s2) {
                // Both ground (Tier-1 / fully-monomorphic): length + element-wise.
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
                (_, 0) => unify_row_against_ground_with_mapped_axes(&d1, &d2, subst),
                (0, _) => unify_row_against_ground_with_mapped_axes(&d2, &d1, subst),
                // Both carry spreads: only structurally-identical rows unify.
                (_, _) => unify_rows_with_mapped_axes(&d1, &d2, subst),
            }
        }

        // ADT types
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
            for (a1, a2) in args1.iter().zip(args2) {
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

        // Tuple types
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

        // Error propagation: unifying with Error trivially succeeds so
        // downstream call sites do not fan out a cascade of secondary
        // diagnostics from a single upstream error. Call sites that
        // require a precision/shape match against a non-Error declared
        // type must check for the Error sentinel themselves and surface
        // the mismatch explicitly (e.g. the def-body vs declared-sig
        // unify in `infer.rs` does this for WS-A5 RT-3a F1: when the
        // body collapses to Error but the declared type is concrete, we
        // still emit a declared-versus-inferred mismatch so the user sees
        // the unresolved declared shape).
        (Type::Error(_), _) | (_, Type::Error(_)) => Ok(()),

        // Everything else is a mismatch
        _ => Err(TypeError {
            kind: TypeErrorKind::TypeMismatch,
            message: format!("type mismatch: {t1} vs {t2}"),
        }),
    }
}

/// Unify two tensor precision slots per `spec/04-type-system.md` §5.8
/// (WS-A5 precision polymorphism).
///
/// - `Concrete(p1)` and `Concrete(p2)` unify only when `p1 == p2`.
/// - `Var(v)` unifies with `Concrete(p)` by binding `v` to `Type::Prim(p)`
///   in the substitution; further references through that var resolve to
///   the concrete prim via `Subst::apply_tensor_prec`.
/// - Two `Var`s unify by linking them at the type-variable level
///   (delegated to `bind_tvar`), keeping the precision slot's monomorphic
///   structure consistent with the rest of the type system.
pub fn unify_tensor_prec(
    p1: &TensorPrec,
    p2: &TensorPrec,
    subst: &mut Subst,
) -> Result<(), TypeError> {
    let p1 = subst.apply_tensor_prec(p1);
    let p2 = subst.apply_tensor_prec(p2);
    match (&p1, &p2) {
        (TensorPrec::Concrete(a), TensorPrec::Concrete(b)) if a == b => Ok(()),
        (TensorPrec::Concrete(a), TensorPrec::Concrete(b)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!("tensor precision mismatch: {} vs {}", a.name(), b.name()),
        }),
        (TensorPrec::Var(v), TensorPrec::Concrete(p)) => bind_tvar(*v, &Type::Prim(*p), subst),
        (TensorPrec::Concrete(p), TensorPrec::Var(v)) => bind_tvar(*v, &Type::Prim(*p), subst),
        (TensorPrec::Var(v1), TensorPrec::Var(v2)) => {
            if v1 == v2 {
                Ok(())
            } else {
                bind_tvar(*v1, &Type::Var(*v2), subst)
            }
        }
    }
}

/// Unify two dimensions.
///
/// Wildcard ↔ Var invariant: `unify_dim(Wildcard, Var(v))` succeeds
/// without binding `v`. The `(Wildcard, _) | (_, Wildcard) => Ok(())`
/// arm matches before the `(Var(v), _) => bind_dvar(...)` arm and
/// returns `Ok(())` with no side effects, so the dim var stays free.
/// This is intentional — binding `v := Wildcard` would freeze `v` and
/// make a later concrete arg in the same sig unable to constrain it
/// (`apply_dim` would resolve `Var(v) → Wildcard` and the permissive
/// Wildcard arm would silently accept any value). Leaving `v` free
/// lets a concrete arg in any later position bind it, after which a
/// different concrete value trips the `Lit ↔ Lit` mismatch as the sig
/// demands.
///
/// Name ↔ Lit invariant (issue Chelis-Lang/chelis#219, Option A): a
/// concrete-but-named slot (`Dim::Name("batch")`) accepts a concrete
/// literal (`Dim::Lit(2)`) at the call site without binding any
/// substitution. Names are preserved in diagnostics; they do not
/// impose a distinct-from-literal constraint. This eliminates the
/// asymmetry whereby `Var <-> Lit` was accepted at call sites but
/// `Name <-> Lit` was rejected, which blocked stdlib sigs like
/// `def f(x: tensor[batch, hidden, f32])` from being called with
/// concrete-shaped inputs (e.g. `f(to_tensor([[1.0, 2.0, 3.0]]))`).
/// The relaxation is narrow: distinct `Name <-> Name` and
/// distinct `Lit <-> Lit` continue to be rejected, and the
/// `Var <-> Lit` cross-position contract is unaffected.
pub fn unify_dim(d1: &Dim, d2: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
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
        // chelis#1801 / spec/04-type-system.md section 3.2: record the
        // meeting before the permissive arm swallows it. The variable is
        // still deliberately left FREE here (the Wildcard/Var invariant
        // above), so a later concrete argument in the same signature can
        // constrain it; `infer_app` decides what an unconstrained one
        // denotes once the whole call has unified.
        (Dim::Wildcard, Dim::Var(v)) | (Dim::Var(v), Dim::Wildcard) => {
            subst.note_wildcard_touch(*v);
            Ok(())
        }
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => Ok(()),
        // Issue #219 Option A: Name and Lit unify without binding any
        // substitution. The Name carries a label for diagnostics, the
        // Lit carries the concrete value; nothing flows into `subst`.
        (Dim::Name(_), Dim::Lit(_)) | (Dim::Lit(_), Dim::Name(_)) => Ok(()),
        (Dim::Var(v), _) => bind_dvar(*v, &d2, subst),
        (_, Dim::Var(v)) => bind_dvar(*v, &d1, subst),
        _ => Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!("dimension mismatch: {d1:?} vs {d2:?}"),
        }),
    }
}

/// Bind `v`, then settle every operand constraint that was waiting on it
/// (chelis#1489).
///
/// The wrapper exists so discharge cannot be skipped: `bind_tvar_inner` has
/// more than one success path, and an earlier design that decided these
/// constraints in a separate end-of-inference pass was order-dependent in
/// exactly the way this issue is about. Binding a variable is the event that
/// makes a suspended decision decidable, so that is where the decision is
/// made, and nothing schedules it.
fn bind_tvar(v: TypeVar, ty: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    bind_tvar_inner(v, ty, subst)?;
    discharge_operand_gates(v, subst);
    discharge_bounded_scalar_casts(subst);
    Ok(())
}

/// Settle the suspended scalar `cast`/`cast_trunc` gates that a binding has
/// made decidable without binding their operand to a type (chelis#2151).
///
/// A binding is also the event through which a variable GAINS a declared
/// dtype-family bound: identified with a bounded variable, or with a rigid
/// authored binder. Such a variable may never be bound to a concrete type, so
/// [`discharge_operand_gates`] would only carry its gate from one variable to
/// the next, and the per-def pass would reject a well-typed cast. [04-DTYPE-2]
/// makes the bound enough to decide it: the operand is a scalar primitive of
/// that family at every instantiation.
///
/// The decision is [`crate::infer::expr_record::bounded_scalar_cast_result`].
/// This is its only call site. A cast whose operand already carries the bound
/// is suspended like any other, and it settles at the next binding, which the
/// suspended cast's own result variable guarantees. The cast and the bound are
/// therefore settled together by whichever arrives second, and the verdict
/// cannot depend on which operand inference visits first. Only what later
/// bindings cannot change is decided here (every acceptance, and a rejection
/// that reads only the target); a gate it declines stays suspended.
fn discharge_bounded_scalar_casts(subst: &mut Subst) {
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
/// Reached only from [`bind_tvar`]. Records failures rather than returning
/// them: a discharge failure is a diagnostic about the program, not a
/// unification error, and turning it into one would abort the surrounding
/// unification and lose every other constraint waiting on this binding.
fn discharge_operand_gates(v: TypeVar, subst: &mut Subst) {
    let gates = subst.take_operand_gates_on(v);
    if gates.is_empty() {
        return;
    }
    let resolved = subst.apply(&Type::Var(v));
    for gate in gates {
        // Still a variable: `v` was identified with another variable rather
        // than given a type. Carry the obligation across so it discharges when
        // THAT variable binds.
        if let Type::Var(target) = resolved {
            if target != v {
                subst.realias_operand_gate(target, gate);
                continue;
            }
            // Bound to itself is not a binding; leave it suspended for the
            // per-def pass to report as never-resolved.
            subst.realias_operand_gate(v, gate);
            continue;
        }
        gate.discharge(&resolved, subst);
    }
}

fn bind_tvar_inner(v: TypeVar, ty: &Type, subst: &mut Subst) -> Result<(), TypeError> {
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
    // Both endpoints keep their bound: the identified variable admits only
    // the dtypes both families admit ([04-DTYPE-2]). `.or()` would silently
    // discard the narrower of the two.
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

    // An older variable that becomes bound to a younger composite makes all
    // reachable variables part of the older scope.
    let target_level = subst.level_of_tvar(v);
    subst.lower_type_to(ty, target_level);
    subst.record_validated_type_binding(v, ty.clone());
    subst.remove_tvar_restriction(v);
    if let (Some(restriction), Type::Var(target)) = (merged_restriction, ty) {
        subst.narrow_tvar_restriction(*target, restriction)?;
    }
    Ok(())
}

/// [04-LIN-10]: check that a function's type parameter is not instantiated at
/// a key-carrying type (spec/04 section 8.4.1), and carry the rule onto every
/// variable left inside `ty`, which the parameter's instantiation now
/// includes. `whole` is the complete type the parameter is bound to, for the
/// diagnostic.
///
/// Function types carry no key, so a function-typed instantiation passes and
/// its parameter and result variables stay unmarked: the function's own
/// scheme, not this parameter, decides whether a key may reach them.
fn ensure_key_free(
    origin: &GenericParameter,
    ty: &Type,
    whole: &Type,
    subst: &Subst,
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
/// An unresolved variable and a witnessed error both pass: the first is
/// narrowed instead by [`Subst::narrow_tvar_restriction`], and the second
/// already owns a diagnostic.
fn ensure_tvar_restriction(
    restriction: TypeVarRestriction,
    ty: &Type,
    subst: &Subst,
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
            message: format!(
                "type variable bounded by {family} ({gloss}) cannot be instantiated at `{}`",
                prim.name()
            ),
        }),
        other => Err(TypeError {
            kind: TypeErrorKind::DtypeFamilyMismatch,
            message: format!(
                "type variable bounded by {family} ({gloss}) cannot be instantiated at `{other}`"
            ),
        }),
    }
}

fn bind_dvar(v: DimVar, dim: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
    if let Dim::Name(name) = dim
        && (subst.protected_dimensions.contains(&v) || subst.dimension_label(v).is_some())
    {
        subst.set_dimension_label(v, name.clone());
        // chelis#1801 note: this arm records an authored name as a LABEL and
        // binds nothing, so it is the one place a name reaches a class
        // without `constraint_dim` being able to report it, and it carries no
        // pin of its own. Not because the operand must come from the
        // definition under check: `def named(w: tensor[batch, f32])` keeps
        // `batch` as a `Dim::Name` in its published scheme, so instantiating
        // it anywhere contributes one. The route is the arm's OWN
        // precondition. It fires only when `v` is protected, which the pin
        // reads directly, or already labelled, and a label is never created
        // from nothing: the three production writers of `set_dimension_label`
        // are `copy_dimension_label`, which propagates an existing label to a
        // fresh instantiation id, and the two below, which each require an
        // existing label or protection. Every label therefore traces back to
        // a declared binder.
        //
        // What that argument does NOT establish is that the labelled variable
        // is itself protected, because the instantiation copy lands a label on
        // a fresh unprotected id. chelis#1925's round 3 measured the gap
        // instead of arguing it: instrumenting all 196 `.ch` files in the
        // corpus plus 32 probes found no unpinned class reaching this arm, and
        // the labelled-then-named state, constructed four ways, binds through
        // the fall-through below so `constraint_dim` reports it.
        // `a_name_arm_class_is_pinned_by_protection_not_by_a_second_ledger`
        // locks what the arm leaves behind and says where the pin goes if a
        // program ever reaches that state.
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
    // chelis#1801: the merge below leaves the union with one root, so the
    // class properties have to arrive there first. A name or literal the
    // class is BOUND to needs no flag: `constraint_dim` reports the binding,
    // and the absorbing site skips on that answer before it asks anything
    // else.
    if let Dim::Var(other) = dim {
        subst.merge_dvar_class_flags(v, *other);
    }
    let target_level = subst.level_of_dvar(v);
    subst.lower_dim_to(dim, target_level);
    subst.insert_dim(v, dim.clone());
    Ok(())
}

/// Fully resolve a tensor's dim list under the current substitution: each
/// bound `Dim::Rank` expands to the run it stands for and every other dim is
/// `apply_dim`'d; an unbound (or terminal-rank) spread stays a `Dim::Rank`.
/// This is the per-shape form of `Subst::apply`'s tensor arm, used by the
/// rank-unification dispatch so a row shape arrives with only *unbound*
/// spreads interleaved among concrete dims.
fn resolve_shape(dims: &[Dim], subst: &Subst) -> Vec<Dim> {
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

fn mapped_axis_dim(dim: &Dim, subst: &Subst) -> Option<(usize, DimVar)> {
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

fn canonicalize_ground_mapped_axes(dims: &[Dim], subst: &Subst) -> Result<Vec<Dim>, TypeError> {
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

fn split_row_mapped_axes(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst,
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
        // Bind the compiler-owned batch identity in this direction so an
        // authored operand identity remains the class root and retains its
        // user-facing label.
        unify_dim(&Dim::Var(*var), batch, subst)?;
    }
    for (axis, _) in mapped.into_iter().rev() {
        clean_ground.remove(axis);
    }
    Ok((clean_row, clean_ground))
}

fn unify_row_against_ground_with_mapped_axes(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst,
) -> Result<(), TypeError> {
    let ground = canonicalize_ground_mapped_axes(ground, subst)?;
    let (row, ground) = split_row_mapped_axes(row, &ground, subst)?;
    unify_row_against_ground(&row, &ground, subst)
}

fn unify_rows_with_mapped_axes(
    left: &[Dim],
    right: &[Dim],
    subst: &mut Subst,
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

/// chelis#339 named-axis expand: a signature whose result rows *introduce* an
/// axis name (present in a result tensor row but in no parameter row — the
/// expand-inserted axis) must not have that same name covered by one of its
/// rank-spread bindings at this unification. The symbolic collision check in
/// `check_named_expand_signature` can only see the visible row; a caller
/// whose spread-covered axes include the inserted name would otherwise
/// monomorphize to a result row carrying the same dim name twice with two
/// different extents — every later by-name axis lookup becomes ambiguous and
/// the declared type misstates the runtime shape. Hard error, never a guessed
/// alias (spec/04-type-system.md §4.5.3).
fn check_introduced_name_rank_collision(
    args: &[Type],
    ret: &Type,
    subst: &Subst,
) -> Result<(), TypeError> {
    fn collect(ty: &Type, names: &mut Vec<String>, rvars: &mut Vec<RankVar>) {
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
                for a in fn_args {
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

/// The shared diagnostic for an undetermined split between two adjacent
/// rank spreads (the non-unitary case the decidable fragment excludes).
fn adjacent_spread_error() -> TypeError {
    TypeError {
        kind: TypeErrorKind::DimensionMismatch,
        message: "two adjacent rank spreads cannot be split against a concrete shape; the \
                  boundary between them is undetermined (outside the decidable fragment, \
                  spec/04-type-system.md \u{00a7}4.5.3)"
            .to_string(),
    }
}

/// Unify a *row* shape (≥1 spread, no two spreads adjacent) against a *ground*
/// shape (no spreads). Walks the row left to right: a fixed anchor matches the
/// ground positionally; a spread is bound to the run of ground dims up to the
/// next anchor, located by that anchor's name (the §4.2 name-preserving rule).
/// Each interior split anchor must be a `Dim::Name` present **exactly once** in
/// the remaining ground — a missing, ambiguous, or non-named anchor is a hard
/// error, never a guessed split. With one spread per gap and named splits,
/// every split point is forced, so this is unitary.
fn unify_row_against_ground(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst,
) -> Result<(), TypeError> {
    let n = ground.len();
    let mut gi = 0usize; // ground cursor
    let mut ri = 0usize; // row cursor
    while ri < row.len() {
        match &row[ri] {
            Dim::Rank(r) => {
                let rest = &row[ri + 1..];
                match rest.iter().position(|d| !matches!(d, Dim::Rank(_))) {
                    // The spread is immediately followed by a named anchor:
                    // locate that name in the remaining ground to fix the split.
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
                    // A second spread sits between this spread and the next
                    // anchor (or runs to the end): the split between two
                    // adjacent spreads is undetermined — the non-unitary case.
                    Some(_) => {
                        return Err(adjacent_spread_error());
                    }
                    None if rest.is_empty() => {
                        // Trailing spread: absorb the rest of the ground.
                        bind_rvar(*r, &ground[gi..n], subst)?;
                        gi = n;
                        ri += 1;
                    }
                    None => {
                        // `rest` is one or more further spreads with no anchor.
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

/// Unify two row shapes that *both* carry spreads. Restricted to
/// structurally-identical rows (same length, same per-position kind): spreads
/// alias pairwise, named anchors unify by name. Two rows with a *different*
/// anchor structure are associative/string unification (not unitary) and are
/// rejected — the body of a rank-poly def only ever produces an identical-row
/// self-check, and call sites are row-vs-ground.
fn unify_row_against_row(d1: &[Dim], d2: &[Dim], subst: &mut Subst) -> Result<(), TypeError> {
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

/// Bind a rank variable to a concrete shape *run*, with an occurs/nesting
/// check: the bound run must be `Dim::Rank`-free. A run containing any spread
/// would either make the rank infinite (its own occurrence) or smuggle a
/// second spread into a single spread's binding (the non-unitary case).
/// Rank-to-rank aliasing is handled separately by `unify_row_against_row`.
fn bind_rvar(r: RankVar, dims: &[Dim], subst: &mut Subst) -> Result<(), TypeError> {
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

fn occurs_in(v: TypeVar, ty: &Type, subst: &Subst) -> bool {
    let ty = subst.apply(ty);
    match &ty {
        Type::Var(v2) => *v2 == v,
        Type::Fn(args, ret) => {
            args.iter().any(|a| occurs_in(v, a, subst)) || occurs_in(v, ret, subst)
        }
        Type::Ref(inner) => occurs_in(v, inner, subst),
        Type::Tensor(_, prec) => match prec {
            // Dims do not carry type vars (they have their own DimVar lane).
            // Precision slot CAN carry a TypeVar (WS-A5 precision polymorphism);
            // include it in the occurs check so an attempt to unify
            // ?v with `tensor[..., ?v]` is caught as an infinite type.
            TensorPrec::Concrete(_) => false,
            TensorPrec::Var(v2) => *v2 == v,
        },
        Type::Adt(_, args) => args.iter().any(|a| occurs_in(v, a, subst)),
        Type::KindedAdt(_, args) => args
            .iter()
            .any(|argument| argument.as_type().is_some_and(|ty| occurs_in(v, ty, subst))),
        Type::Tuple(ts) => ts.iter().any(|t| occurs_in(v, t, subst)),
        Type::Prim(_) | Type::Unit | Type::Error(_) => false,
    }
}

fn occurs_in_dim(v: DimVar, dim: &Dim, subst: &Subst) -> bool {
    let dim = subst.apply_dim(dim);
    matches!(dim, Dim::Var(v2) if v2 == v)
}

#[cfg(test)]
mod tests {
    #[test]
    fn substitution_resolves_variables_inside_a_late_bound_constructor() {
        let outer = Type::Var(TypeVar(100));
        let inner = Type::Var(TypeVar(101));
        let mut subst = Subst::new();
        unify(&outer, &Type::Tuple(vec![inner.clone()]), &mut subst).unwrap();
        unify(&inner, &Type::Prim(Prim::F32), &mut subst).unwrap();
        assert_eq!(
            subst.apply(&outer),
            Type::Tuple(vec![Type::Prim(Prim::F32)])
        );
        assert!(
            unify(
                &outer,
                &Type::Tuple(vec![Type::Prim(Prim::Bool)]),
                &mut subst
            )
            .is_err()
        );
    }
    use super::*;

    #[test]
    fn protected_label_aliases_keep_identity_after_compression_and_literal_binding() {
        let mut subst = Subst::new();
        let root = DimVar(0);
        subst.protect_dimensions([root]);
        unify_dim(&Dim::Var(root), &Dim::Name("fixed".into()), &mut subst).unwrap();
        for i in 1..5 {
            unify_dim(&Dim::Var(DimVar(i)), &Dim::Var(DimVar(i - 1)), &mut subst).unwrap();
        }
        assert_eq!(subst.apply_dim(&Dim::Var(DimVar(4))), Dim::Var(root));
        unify_dim(&Dim::Var(root), &Dim::Lit(2), &mut subst).unwrap();
        assert_eq!(subst.constraint_dim(&Dim::Var(DimVar(4))), Dim::Lit(2));
        assert_eq!(
            subst.semantic_dim(&Dim::Var(DimVar(4))),
            Dim::Name("fixed".into())
        );
        assert!(matches!(subst.apply_dim(&Dim::Var(DimVar(4))), Dim::Var(_)));
        assert!(unify_dim(&Dim::Var(DimVar(4)), &Dim::Lit(3), &mut subst).is_err());
    }

    #[test]
    fn protected_labels_do_not_leak_from_a_failed_type_unification() {
        let mut subst = Subst::new();
        subst.protect_dimensions([DimVar(0)]);
        let tensor = |dim| Type::Tensor(vec![dim], TensorPrec::Concrete(Prim::F32));
        let left = Type::Tuple(vec![tensor(Dim::Var(DimVar(0))), Type::Unit]);
        let right = Type::Tuple(vec![tensor(Dim::Name("fixed".into())), tensor(Dim::Lit(1))]);
        assert!(unify(&left, &right, &mut subst).is_err());
        assert_eq!(subst.dimension_label(DimVar(0)), None);
        unify_dim(&Dim::Var(DimVar(0)), &Dim::Name("other".into()), &mut subst).unwrap();
    }

    #[test]
    fn dimension_label_undo_replays_repeated_copy_writes_and_skips_noops() {
        let mut subst = Subst::new();
        subst.set_dimension_label(DimVar(0), "fixed".into());
        subst.set_dimension_label(DimVar(1), "other".into());
        subst.set_dimension_label(DimVar(2), "before".into());
        subst.label_unification_depth = 1;
        // Exercise the &Subst writer used by scheme instantiation, including
        // repeated writes to both a pre-existing entry and a fresh entry.
        for target in [DimVar(2), DimVar(3)] {
            subst.copy_dimension_label(DimVar(0), target);
            subst.copy_dimension_label(DimVar(0), target);
            subst.copy_dimension_label(DimVar(1), target);
        }
        assert_eq!(subst.dimension_label_undo.lock().unwrap().len(), 4);
        subst.label_unification_depth = 0;
        subst.finish_dimension_label_writes(true);
        assert_eq!(subst.dimension_label(DimVar(2)).as_deref(), Some("before"));
        assert_eq!(subst.dimension_label(DimVar(3)), None);
        assert!(subst.dimension_label_undo.lock().unwrap().is_empty());
    }

    #[test]
    fn protected_label_nested_alias_failure_preserves_prior_success() {
        let mut subst = Subst::new();
        subst.protect_dimensions([DimVar(0), DimVar(1)]);
        let tensor = |d| Type::Tensor(vec![d], TensorPrec::Concrete(Prim::F32));
        unify(
            &tensor(Dim::Var(DimVar(0))),
            &tensor(Dim::Name("fixed".into())),
            &mut subst,
        )
        .unwrap();
        assert!(subst.dimension_label_undo.lock().unwrap().is_empty());
        let lhs = Type::Tuple(vec![
            Type::Tuple(vec![
                tensor(Dim::Var(DimVar(1))),
                tensor(Dim::Var(DimVar(2))),
            ]),
            Type::Unit,
        ]);
        let rhs = Type::Tuple(vec![
            Type::Tuple(vec![
                tensor(Dim::Name("other".into())),
                tensor(Dim::Var(DimVar(1))),
            ]),
            tensor(Dim::Lit(1)),
        ]);
        assert!(unify(&lhs, &rhs, &mut subst).is_err());
        assert_eq!(subst.dimension_label(DimVar(0)).as_deref(), Some("fixed"));
        assert_eq!(subst.dimension_label(DimVar(1)), None);
        assert_eq!(subst.dimension_label(DimVar(2)), None);
        assert_eq!(subst.label_unification_depth, 0);
        assert!(subst.dimension_label_undo.lock().unwrap().is_empty());
        // Existing dim aliases are outside this narrow label rollback; retry
        // must nevertheless be free to install a different semantic label.
        unify(
            &tensor(Dim::Var(DimVar(1))),
            &tensor(Dim::Name("retry".into())),
            &mut subst,
        )
        .unwrap();
        assert_eq!(subst.dimension_label(DimVar(2)).as_deref(), Some("retry"));
        assert!(subst.dimension_label_undo.lock().unwrap().is_empty());
    }

    #[test]
    fn dimension_label_journal_is_absent_from_clones_and_wire() {
        let mut subst = Subst::new();
        subst.label_unification_depth = 1;
        subst.set_dimension_label(DimVar(0), "fixed".into());
        assert_eq!(subst.dimension_label_undo.lock().unwrap().len(), 1);
        let cloned = subst.clone();
        assert_eq!(cloned.label_unification_depth, 0);
        assert!(cloned.dimension_label_undo.lock().unwrap().is_empty());
        let bytes = bincode::serialize(&subst).unwrap();
        assert_eq!(bytes, bincode::serialize(&cloned).unwrap());
        let decoded: Subst = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.label_unification_depth, 0);
        assert!(decoded.dimension_label_undo.lock().unwrap().is_empty());
        subst.label_unification_depth = 0;
        subst.finish_dimension_label_writes(true);
        assert_eq!(subst.dimension_label(DimVar(0)), None);
        for independent in [&cloned, &decoded] {
            assert_eq!(
                independent.dimension_label(DimVar(0)).as_deref(),
                Some("fixed")
            );
        }
    }

    #[test]
    fn mapped_axis_annotation_survives_clone_wire_and_compose() {
        let mapped = DimVar(0);
        let mut subst = Subst::new();
        subst.mark_mapped_axis(mapped, 3);
        assert_eq!(subst.mapped_axis(mapped), Some(3));
        assert_eq!(
            subst.dimension_label(mapped),
            None,
            "compiler transform metadata must never become a user dimension name"
        );

        let cloned = subst.clone();
        let decoded: Subst = bincode::deserialize(&bincode::serialize(&subst).unwrap()).unwrap();
        for carrier in [&cloned, &decoded] {
            assert_eq!(carrier.mapped_axis(mapped), Some(3));
        }

        let mut receiver = Subst::new();
        receiver.compose(&subst).unwrap();
        assert_eq!(receiver.mapped_axis(mapped), Some(3));
        subst.compose(&Subst::new()).unwrap();
        assert_eq!(subst.mapped_axis(mapped), Some(3));
    }

    #[test]
    fn mapped_axis_is_removed_before_every_anchored_spread_split() {
        for axis in 0..=4 {
            let batch = DimVar(0);
            let pre = RankVar(0);
            let post = RankVar(1);
            let mut subst = Subst::new();
            subst.mark_mapped_axis(batch, axis);
            let row = Type::Tensor(
                vec![
                    Dim::Rank(pre),
                    Dim::Name("seq".into()),
                    Dim::Rank(post),
                    Dim::Var(batch),
                ],
                TensorPrec::Concrete(Prim::F32),
            );
            let authored = vec![
                Dim::Name("left".into()),
                Dim::Name("inner".into()),
                Dim::Name("seq".into()),
                Dim::Name("right".into()),
            ];
            let mut ground = authored.clone();
            ground.insert(axis, Dim::Name("batch".into()));
            let ground = Type::Tensor(ground, TensorPrec::Concrete(Prim::F32));

            unify(&row, &ground, &mut subst)
                .unwrap_or_else(|error| panic!("axis {axis}: {error:?}"));
            assert_eq!(
                subst.resolve_rvar(pre),
                authored[..2],
                "axis {axis}: mapped batch leaked into the leading spread"
            );
            assert_eq!(
                subst.resolve_rvar(post),
                authored[3..],
                "axis {axis}: mapped batch leaked into the trailing spread"
            );
            assert_eq!(
                subst.resolve_dvar(batch),
                Dim::Name("batch".into()),
                "axis {axis}: mapped batch lost its actual extent identity"
            );
        }
    }

    #[test]
    fn protected_labels_are_freshened_but_captured_ids_are_not() {
        let mut vg = VarGen::default();
        let quantified = vg.fresh_dvar();
        let capture = vg.fresh_dvar();
        let alias = vg.fresh_dvar();
        let alias_chain = vg.fresh_dvar();
        let mut subst = Subst::new();
        subst.protect_dimensions([quantified, capture]);
        for v in [quantified, capture] {
            unify_dim(&Dim::Var(v), &Dim::Name("fixed".into()), &mut subst).unwrap();
        }
        // Exercise both alias operand orders; binding prefers the protected
        // representative, including after path compression.
        unify_dim(&Dim::Var(capture), &Dim::Var(alias), &mut subst).unwrap();
        unify_dim(&Dim::Var(alias_chain), &Dim::Var(alias), &mut subst).unwrap();
        assert!(subst.is_protected_dimension(alias_chain));
        assert_eq!(
            subst.constraint_dim(&Dim::Var(alias_chain)),
            Dim::Var(capture)
        );
        let scheme = Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![quantified],
            rvars: vec![],
            body: Type::Tensor(
                vec![Dim::Var(quantified), Dim::Var(alias_chain)],
                TensorPrec::Concrete(Prim::F32),
            ),
        };
        let mut env = crate::env::Env::new();
        env.bind("f".into(), scheme);
        let a = env.instantiate(env.lookup("f").unwrap(), &mut vg, &subst);
        let b = env.instantiate(env.lookup("f").unwrap(), &mut vg, &subst);
        let (Type::Tensor(a, _), Type::Tensor(b, _)) = (a, b) else {
            panic!("tensor")
        };
        assert_ne!(a[0], b[0]);
        for dims in [&a, &b] {
            let Dim::Var(fresh) = dims[0] else {
                panic!("fresh dimension")
            };
            assert!(!subst.is_protected_dimension(fresh));
            let Dim::Var(captured) = dims[1] else {
                panic!("captured dimension")
            };
            assert!(subst.is_protected_dimension(captured));
            assert_eq!(subst.constraint_dim(&dims[1]), Dim::Var(capture));
        }
        unify_dim(&a[0], &Dim::Lit(2), &mut subst).unwrap();
        unify_dim(&b[0], &Dim::Lit(3), &mut subst).unwrap();
        assert_eq!(subst.semantic_dim(&a[0]), Dim::Name("fixed".into()));
        assert_eq!(subst.semantic_dim(&b[0]), Dim::Name("fixed".into()));
    }

    #[test]
    fn protected_dimension_queries_agree_after_live_and_decoded_resume() {
        let mut vg = VarGen::default();
        let historical = vg.fresh_dvar();
        let alias = vg.fresh_dvar();
        let mut published = Subst::new();
        published.protect_dimensions([historical]);
        unify_dim(
            &Dim::Var(historical),
            &Dim::Name("fixed".into()),
            &mut published,
        )
        .unwrap();
        unify_dim(&Dim::Var(alias), &Dim::Var(historical), &mut published).unwrap();
        let mut live = published.clone();
        let mut decoded: Subst =
            bincode::deserialize(&bincode::serialize(&published).unwrap()).unwrap();
        // Raw mid-check serialization intentionally omits active protection.
        // The public context entry point resumes BOTH paths before inference.
        assert!(live.is_protected_dimension(alias));
        assert!(!decoded.is_protected_dimension(alias));
        for subst in [&mut live, &mut decoded] {
            subst.resume_for_new_check(&vg);
            assert!(!subst.is_protected_dimension(alias));
            assert_eq!(
                subst.semantic_dim(&Dim::Var(alias)),
                Dim::Name("fixed".into())
            );
        }
        let current = vg.fresh_dvar();
        let current_alias = vg.fresh_dvar();
        for subst in [&mut live, &mut decoded] {
            subst.protect_dimensions([current]);
            unify_dim(&Dim::Var(current), &Dim::Name("fixed".into()), subst).unwrap();
            unify_dim(&Dim::Var(current), &Dim::Var(current_alias), subst).unwrap();
            assert!(subst.is_protected_dimension(current_alias));
            assert!(!subst.is_protected_dimension(alias));
        }
        assert_eq!(
            bincode::serialize(&live).unwrap(),
            bincode::serialize(&decoded).unwrap()
        );
    }

    #[test]
    fn protected_label_pending_result_stays_monomorphic_in_both_generalizers() {
        let mut vg = VarGen::default();
        let mut subst = Subst::new();
        let level = subst.enter_level(&vg);
        let operand = vg.fresh_tvar();
        let dim = vg.fresh_dvar();
        let rank = vg.fresh_rvar();
        let precision = vg.fresh_tvar();
        subst.protect_dimensions([dim]);
        unify_dim(&Dim::Var(dim), &Dim::Name("fixed".into()), &mut subst).unwrap();
        let result = Type::Tensor(
            vec![Dim::Var(dim), Dim::Rank(rank)],
            TensorPrec::Var(precision),
        );
        subst.record_deferred_tensor_operand(
            operand,
            DeferredOperandGate::Copy {
                result: Box::new(result.clone()),
                location: None,
            },
        );
        subst.leave_level(level, &vg);
        // generalize-sweep-oracle executes and compares the reference algorithm.
        let scheme = crate::env::Env::new().generalize(&result, &subst);
        assert!(scheme.dvars.is_empty());
        assert!(scheme.rvars.is_empty());
        assert!(scheme.tvars.is_empty());
    }

    #[test]
    fn protected_label_shared_type_refinement_rolls_back_with_failed_retry() {
        let mut subst = Subst::new();
        let tv = TypeVar(0);
        let dv = DimVar(0);
        subst.protect_dimensions([dv]);
        let tensor = |d| Type::Tensor(vec![d], TensorPrec::Concrete(Prim::F32));
        unify(
            &Type::Var(tv),
            &tensor(Dim::Name("fixed".into())),
            &mut subst,
        )
        .unwrap();
        let lhs = Type::Tuple(vec![Type::Var(tv), Type::Unit]);
        let rhs = Type::Tuple(vec![tensor(Dim::Var(dv)), tensor(Dim::Lit(1))]);
        assert!(unify(&lhs, &rhs, &mut subst).is_err());
        assert_eq!(
            subst.apply(&Type::Var(tv)),
            tensor(Dim::Name("fixed".into()))
        );
        assert_eq!(subst.dimension_label(dv), None);
        unify_dim(&Dim::Var(dv), &Dim::Name("other".into()), &mut subst).unwrap();
    }

    #[test]
    fn protected_label_is_a_rank_anchor_without_erasing_its_identity() {
        let mut subst = Subst::new();
        let dv = DimVar(0);
        subst.protect_dimensions([dv]);
        unify_dim(&Dim::Var(dv), &Dim::Name("fixed".into()), &mut subst).unwrap();
        let row = [Dim::Rank(RankVar(0)), Dim::Name("fixed".into())];
        let ground = [Dim::Name("batch".into()), Dim::Var(dv)];
        unify_row_against_ground(&row, &ground, &mut subst).unwrap();
        assert_eq!(subst.constraint_dim(&Dim::Var(dv)), Dim::Var(dv));
        assert_eq!(
            subst.resolve_rvar(RankVar(0)),
            vec![Dim::Name("batch".into())]
        );
    }

    #[test]
    fn protected_label_in_rank_spread_cannot_collide_with_inserted_name() {
        let mut subst = Subst::new();
        let dv = DimVar(0);
        subst.protect_dimensions([dv]);
        unify_dim(&Dim::Var(dv), &Dim::Name("fixed".into()), &mut subst).unwrap();
        subst.insert_rank(RankVar(0), vec![Dim::Var(dv)]);
        let tensor = |dims| Type::Tensor(dims, TensorPrec::Concrete(Prim::F32));
        let args = [tensor(vec![Dim::Rank(RankVar(0))])];
        let ret = tensor(vec![Dim::Rank(RankVar(0)), Dim::Name("fixed".into())]);
        assert!(check_introduced_name_rank_collision(&args, &ret, &subst).is_err());
    }

    #[test]
    fn protected_label_compose_rejects_conflicting_alias_classes_transactionally() {
        let mut left = Subst::new();
        let mut right = Subst::new();
        left.protect_dimensions([DimVar(0)]);
        right.protect_dimensions([DimVar(1)]);
        unify_dim(&Dim::Var(DimVar(0)), &Dim::Name("fixed".into()), &mut left).unwrap();
        unify_dim(&Dim::Var(DimVar(1)), &Dim::Name("other".into()), &mut right).unwrap();
        right.insert_dim(DimVar(0), Dim::Var(DimVar(1)));
        assert!(left.compose(&right).is_err());
        assert_eq!(
            left.constraint_dim(&Dim::Var(DimVar(0))),
            Dim::Var(DimVar(0))
        );
        assert_eq!(
            left.semantic_dim(&Dim::Var(DimVar(0))),
            Dim::Name("fixed".into())
        );
    }

    fn var_gen() -> VarGen {
        VarGen::default()
    }

    #[test]
    fn level_transitions_restore_parent_mint_levels_for_every_id_class() {
        let mut vg = var_gen();
        let mut subst = Subst::new();

        let child = subst.enter_level(&vg);
        let child_t = vg.fresh_tvar();
        let child_d = vg.fresh_dvar();
        let child_r = vg.fresh_rvar();
        assert_eq!(subst.level_of_tvar(child_t), 1);
        assert_eq!(subst.level_of_dvar(child_d), 1);
        assert_eq!(subst.level_of_rvar(child_r), 1);

        subst.leave_level(child, &vg);
        let parent_t = vg.fresh_tvar();
        let parent_d = vg.fresh_dvar();
        let parent_r = vg.fresh_rvar();
        assert_eq!(subst.level_of_tvar(parent_t), 0);
        assert_eq!(subst.level_of_dvar(parent_d), 0);
        assert_eq!(subst.level_of_rvar(parent_r), 0);

        let sibling = subst.enter_level(&vg);
        let sibling_t = vg.fresh_tvar();
        let sibling_d = vg.fresh_dvar();
        let sibling_r = vg.fresh_rvar();
        assert_eq!(subst.level_of_tvar(sibling_t), 1);
        assert_eq!(subst.level_of_dvar(sibling_d), 1);
        assert_eq!(subst.level_of_rvar(sibling_r), 1);
        subst.leave_level(sibling, &vg);
        assert_eq!(subst.current_level(), 0);
        assert_eq!(subst.level_of_tvar(child_t), 1);
        assert_eq!(subst.level_of_dvar(child_d), 1);
        assert_eq!(subst.level_of_rvar(child_r), 1);
        assert_eq!(subst.level_of_tvar(parent_t), 0);
        assert_eq!(subst.level_of_dvar(parent_d), 0);
        assert_eq!(subst.level_of_rvar(parent_r), 0);
    }

    #[test]
    fn binding_older_type_to_younger_composite_lowers_every_reachable_class() {
        let mut vg = var_gen();
        let mut subst = Subst::new();
        let older = vg.fresh_tvar();
        let mut env = crate::env::Env::new();
        env.bind("outer".to_string(), Scheme::mono(Type::Var(older)));
        let inner = subst.enter_level(&vg);
        let younger_t = vg.fresh_tvar();
        let younger_d = vg.fresh_dvar();
        let younger_r = vg.fresh_rvar();

        let composite = Type::Tuple(vec![
            Type::Var(younger_t),
            Type::Tensor(
                vec![Dim::Var(younger_d), Dim::Rank(younger_r)],
                TensorPrec::Var(younger_t),
            ),
        ]);
        unify(&Type::Var(older), &composite, &mut subst).expect("composite bind");

        assert_eq!(subst.level_of_tvar(younger_t), 0);
        assert_eq!(subst.level_of_dvar(younger_d), 0);
        assert_eq!(subst.level_of_rvar(younger_r), 0);
        subst.leave_level(inner, &vg);
        let escaped = env.generalize(&composite, &subst);
        assert!(escaped.tvars.is_empty());
        assert!(escaped.dvars.is_empty());
        assert!(escaped.rvars.is_empty());
    }

    #[test]
    fn dimension_rank_and_rank_alias_bindings_lower_younger_variables() {
        let mut vg = var_gen();
        let mut subst = Subst::new();
        let older_d = vg.fresh_dvar();
        let older_ground_r = vg.fresh_rvar();
        let older_alias_r = vg.fresh_rvar();
        let mut env = crate::env::Env::new();
        env.bind(
            "outer".to_string(),
            Scheme::mono(Type::Tuple(vec![
                Type::Tensor(vec![Dim::Var(older_d)], TensorPrec::Concrete(Prim::F32)),
                Type::Tensor(
                    vec![Dim::Rank(older_ground_r)],
                    TensorPrec::Concrete(Prim::F32),
                ),
                Type::Tensor(
                    vec![Dim::Rank(older_alias_r)],
                    TensorPrec::Concrete(Prim::F32),
                ),
            ])),
        );
        let inner = subst.enter_level(&vg);
        let younger_d_alias = vg.fresh_dvar();
        let younger_d_in_rank = vg.fresh_dvar();
        let younger_r = vg.fresh_rvar();

        unify_dim(&Dim::Var(older_d), &Dim::Var(younger_d_alias), &mut subst)
            .expect("dimension alias");
        bind_rvar(older_ground_r, &[Dim::Var(younger_d_in_rank)], &mut subst)
            .expect("ground rank bind");
        unify_row_against_row(
            &[Dim::Rank(older_alias_r)],
            &[Dim::Rank(younger_r)],
            &mut subst,
        )
        .expect("rank alias");

        assert_eq!(subst.level_of_dvar(younger_d_alias), 0);
        assert_eq!(subst.level_of_dvar(younger_d_in_rank), 0);
        assert_eq!(subst.level_of_rvar(younger_r), 0);
        subst.leave_level(inner, &vg);
        let escaped = env.generalize(
            &Type::Tuple(vec![
                Type::Tensor(
                    vec![Dim::Var(younger_d_alias)],
                    TensorPrec::Concrete(Prim::F32),
                ),
                Type::Tensor(
                    vec![Dim::Var(younger_d_in_rank)],
                    TensorPrec::Concrete(Prim::F32),
                ),
                Type::Tensor(vec![Dim::Rank(younger_r)], TensorPrec::Concrete(Prim::F32)),
            ]),
            &subst,
        );
        assert!(escaped.dvars.is_empty());
        assert!(escaped.rvars.is_empty());
    }

    #[test]
    fn expanded_rank_dimensions_and_precision_aliases_are_lowered() {
        let mut vg = var_gen();
        let mut subst = Subst::new();
        let older_type = vg.fresh_tvar();
        let older_precision = vg.fresh_tvar();
        let mut env = crate::env::Env::new();
        env.bind(
            "outer".to_string(),
            Scheme::mono(Type::Tuple(vec![
                Type::Var(older_type),
                Type::Tensor(vec![], TensorPrec::Var(older_precision)),
            ])),
        );
        let inner = subst.enter_level(&vg);
        let younger_rank = vg.fresh_rvar();
        let younger_dim = vg.fresh_dvar();
        let younger_precision = vg.fresh_tvar();

        bind_rvar(younger_rank, &[Dim::Var(younger_dim)], &mut subst)
            .expect("inner ground rank bind");
        unify(
            &Type::Var(older_type),
            &Type::Tensor(
                vec![Dim::Rank(younger_rank)],
                TensorPrec::Concrete(Prim::F32),
            ),
            &mut subst,
        )
        .expect("expanded rank reaches the older type");
        unify_tensor_prec(
            &TensorPrec::Var(older_precision),
            &TensorPrec::Var(younger_precision),
            &mut subst,
        )
        .expect("precision alias");

        assert_eq!(subst.level_of_dvar(younger_dim), 0);
        assert_eq!(subst.level_of_tvar(younger_precision), 0);
        subst.leave_level(inner, &vg);
        let escaped = env.generalize(
            &Type::Tuple(vec![
                Type::Tensor(vec![Dim::Var(younger_dim)], TensorPrec::Concrete(Prim::F32)),
                Type::Tensor(vec![], TensorPrec::Var(younger_precision)),
            ]),
            &subst,
        );
        assert!(escaped.tvars.is_empty());
        assert!(escaped.dvars.is_empty());
    }

    #[test]
    fn tensor_precision_variables_generalize_at_an_ordinary_boundary() {
        let mut vg = var_gen();
        let mut subst = Subst::new();
        let boundary = subst.enter_level(&vg);
        let precision = vg.fresh_tvar();
        let linked = vg.fresh_tvar();
        unify_tensor_prec(
            &TensorPrec::Var(precision),
            &TensorPrec::Var(linked),
            &mut subst,
        )
        .expect("precision variables link");
        subst.leave_level(boundary, &vg);

        let scheme = crate::env::Env::new()
            .generalize(&Type::Tensor(vec![], TensorPrec::Var(precision)), &subst);
        assert_eq!(scheme.tvars, vec![linked]);

        let instantiated = crate::env::Env::new().instantiate(&scheme, &mut vg, &subst);
        let Type::Tensor(_, TensorPrec::Var(fresh_precision)) = instantiated else {
            panic!("precision instantiation must remain a tensor precision variable");
        };
        unify_tensor_prec(
            &TensorPrec::Var(fresh_precision),
            &TensorPrec::Concrete(Prim::F32),
            &mut subst,
        )
        .expect("precision variable binds to a concrete precision");
        assert_eq!(
            subst.apply_tensor_prec(&TensorPrec::Var(fresh_precision)),
            TensorPrec::Concrete(Prim::F32)
        );
    }

    /// chelis#1417: identifying a `Numeric`-bounded variable with a
    /// `Float`-bounded one keeps the INTERSECTION, in both orders.
    ///
    /// This is the direction test, not the conflict test. The two acceptance
    /// tests that exercise narrowing through a program detect
    /// `merge_tvar_restrictions` returning `Err`, so a reversion that merely
    /// widens survives them; this one asserts the surviving family IS `Float`
    /// and that `i32` is consequently rejected.
    ///
    /// It does not isolate either mechanism, and measurement rather than
    /// reasoning says so. `bind_tvar`'s `merged_restriction` and
    /// `narrow_tvar_restriction` each independently suffice to produce the
    /// narrowing, so reverting either ALONE leaves this green (and leaves all
    /// 1460 crate tests green); only reverting BOTH reds it. Neither is dead
    /// code — both sit on live paths — but neither is individually necessary
    /// for this behavior, so no single test can discriminate them. Five
    /// program shapes, including the reversed order, were tried and none
    /// discriminates either.
    #[test]
    fn identifying_a_numeric_variable_with_a_float_one_keeps_float() {
        for numeric_first in [true, false] {
            let mut vg = var_gen();
            let mut subst = Subst::new();
            let numeric = vg.fresh_tvar();
            let float = vg.fresh_tvar();
            subst
                .narrow_tvar_restriction(numeric, TypeVarRestriction::ActiveNumeric)
                .expect("fresh variable takes a bound");
            subst
                .narrow_tvar_restriction(float, TypeVarRestriction::ActiveFloat)
                .expect("fresh variable takes a bound");

            let (left, right) = if numeric_first {
                (numeric, float)
            } else {
                (float, numeric)
            };
            unify(&Type::Var(left), &Type::Var(right), &mut subst)
                .expect("Float is a subset of Numeric, so the two are compatible");

            // Whichever variable survives the binding must admit floats only.
            let surviving = match subst.resolve_tvar(left) {
                Type::Var(v) => v,
                other => panic!("expected a variable, got {other}"),
            };
            assert_eq!(
                subst.tvar_restriction(surviving),
                Some(TypeVarRestriction::ActiveFloat),
                "identifying Numeric with Float must narrow to Float, not keep \
                 Numeric (numeric_first = {numeric_first})"
            );

            // And the narrowing is observable: i32 is in Numeric but not in
            // Float, so it must now be rejected.
            let error = unify(&Type::Var(surviving), &Type::Prim(Prim::Int32), &mut subst)
                .expect_err("a narrowed variable must reject a non-float dtype");
            assert!(
                matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch)
                    && error.message.contains("Float"),
                "expected a Float PrecisionMismatch, got: {}",
                error.message
            );
        }
    }

    #[test]
    fn active_float_restrictions_propagate_through_aliases_and_generalization() {
        let mut vg = var_gen();
        let mut subst = Subst::new();
        let boundary = subst.enter_level(&vg);
        let restricted = vg.fresh_tvar();
        let alias = vg.fresh_tvar();
        subst
            .narrow_tvar_restriction(restricted, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        unify(&Type::Var(restricted), &Type::Var(alias), &mut subst)
            .expect("restriction must flow through an ordinary type-variable alias");
        subst.leave_level(boundary, &vg);

        let scheme = crate::env::Env::new().generalize(&Type::Var(alias), &subst);
        assert_eq!(scheme.tvars.len(), 1);
        assert_eq!(
            scheme.tvar_restrictions,
            vec![(scheme.tvars[0], TypeVarRestriction::ActiveFloat)]
        );

        let mut call_subst = Subst::new();
        let instantiated = crate::env::Env::new().instantiate(&scheme, &mut vg, &call_subst);
        let Type::Var(call_var) = instantiated else {
            panic!("generalized alias must instantiate to a fresh variable");
        };
        let error = unify(
            &Type::Var(call_var),
            &Type::Prim(Prim::Bool),
            &mut call_subst,
        )
        .expect_err("instantiated alias must retain its active-float domain");
        assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
        assert!(error.message.contains("active float dtype"));
    }

    #[test]
    fn operation_value_restrictions_survive_serialization_and_late_tensor_binding() {
        for restriction in [
            TypeVarRestriction::FloatValue,
            TypeVarRestriction::IntValue,
            TypeVarRestriction::NumericValue,
        ] {
            let mut vg = var_gen();
            let value = vg.fresh_tvar();
            let scheme = Scheme {
                result_origin: None,
                tvars: vec![value],
                tvar_restrictions: vec![(value, restriction)],
                dvars: vec![],
                rvars: vec![],
                constraints: vec![],
                body: Type::Var(value),
            };
            let scheme: Scheme =
                bincode::deserialize(&bincode::serialize(&scheme).unwrap()).unwrap();
            assert_eq!(scheme.tvar_restrictions, vec![(value, restriction)]);
            for (dtype, accepts) in [
                (Prim::F32, restriction != TypeVarRestriction::IntValue),
                (Prim::Int32, restriction != TypeVarRestriction::FloatValue),
                (Prim::Bool, false),
            ] {
                let mut subst = Subst::new();
                let instantiated = crate::env::Env::new().instantiate(&scheme, &mut vg, &subst);
                let precision = vg.fresh_tvar();
                unify(
                    &instantiated,
                    &Type::Tensor(vec![], TensorPrec::Var(precision)),
                    &mut subst,
                )
                .unwrap();
                assert_eq!(
                    unify(&Type::Var(precision), &Type::Prim(dtype), &mut subst).is_ok(),
                    accepts,
                    "{restriction:?} at {dtype:?}"
                );
            }
        }
    }

    #[test]
    fn authored_dtype_bounds_still_reject_whole_tensor_types() {
        let mut vg = var_gen();
        for restriction in [
            TypeVarRestriction::ActiveFloat,
            TypeVarRestriction::ActiveInt,
            TypeVarRestriction::ActiveNumeric,
        ] {
            let mut subst = Subst::new();
            let value = vg.fresh_tvar();
            subst.narrow_tvar_restriction(value, restriction).unwrap();
            assert!(
                unify(
                    &Type::Var(value),
                    &Type::Tensor(vec![], TensorPrec::Concrete(Prim::F32)),
                    &mut subst
                )
                .is_err()
            );
        }
    }

    #[test]
    fn compose_preserves_active_float_restrictions_from_both_operands() {
        let receiver_var = TypeVar(80_001);
        let other_var = TypeVar(80_002);
        let shared_var = TypeVar(80_003);
        let mut receiver = Subst::new();
        receiver
            .narrow_tvar_restriction(receiver_var, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        receiver
            .narrow_tvar_restriction(shared_var, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        let other = Subst::new();
        other
            .narrow_tvar_restriction(other_var, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        other
            .narrow_tvar_restriction(shared_var, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");

        receiver
            .compose(&other)
            .expect("identical and independent restrictions compose");

        assert_eq!(
            receiver.tvar_restriction(receiver_var),
            Some(TypeVarRestriction::ActiveFloat),
            "compose must retain the receiver's restriction"
        );
        assert_eq!(
            receiver.tvar_restriction(other_var),
            Some(TypeVarRestriction::ActiveFloat),
            "compose must merge the right operand's restriction"
        );
        assert_eq!(
            receiver.tvar_restriction(shared_var),
            Some(TypeVarRestriction::ActiveFloat),
            "the same restriction on a shared key must compose idempotently"
        );
    }

    #[test]
    fn compose_canonicalizes_active_float_restrictions_through_alias_chains() {
        let source = TypeVar(80_004);
        let middle = TypeVar(80_005);
        let terminal = TypeVar(80_006);
        let mut receiver = Subst::new();
        receiver
            .narrow_tvar_restriction(source, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        receiver
            .insert_type(source, Type::Var(middle))
            .expect("restricted source aliases an unresolved variable");
        let mut other = Subst::new();
        other
            .insert_type(middle, Type::Var(terminal))
            .expect("unrestricted alias chain is valid");

        receiver
            .compose(&other)
            .expect("a restriction follows the composed alias chain");
        assert_eq!(receiver.apply(&Type::Var(source)), Type::Var(terminal));
        assert_eq!(
            receiver.tvar_restriction(terminal),
            Some(TypeVarRestriction::ActiveFloat)
        );

        receiver
            .insert_type(terminal, Type::Prim(Prim::F64))
            .expect("the composed restriction accepts an active float");
        assert_eq!(receiver.apply(&Type::Var(source)), Type::Prim(Prim::F64));
        assert_eq!(receiver.tvar_restriction(terminal), None);
    }

    #[test]
    fn compose_rejects_forbidden_bindings_transactionally_in_either_operand() {
        let restricted_in_receiver = TypeVar(80_007);
        let mut receiver = Subst::new();
        receiver
            .narrow_tvar_restriction(restricted_in_receiver, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        let mut other = Subst::new();
        other
            .insert_type(restricted_in_receiver, Type::Prim(Prim::Int64))
            .expect("the independent substitution does not know the restriction");
        let receiver_types_before = receiver.types_snapshot();
        let receiver_restrictions_before = receiver.tvar_restrictions_snapshot();

        let error = receiver
            .compose(&other)
            .expect_err("the merged substitution must enforce the receiver restriction");
        assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
        assert!(error.message.contains("active float dtype"));
        assert_eq!(receiver.types_snapshot(), receiver_types_before);
        assert_eq!(
            receiver.tvar_restrictions_snapshot(),
            receiver_restrictions_before
        );

        let restricted_in_other = TypeVar(80_008);
        let mut receiver = Subst::new();
        receiver
            .insert_type(restricted_in_other, Type::Prim(Prim::Int16))
            .expect("the receiver does not yet know the restriction");
        let other = Subst::new();
        other
            .narrow_tvar_restriction(restricted_in_other, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        let receiver_types_before = receiver.types_snapshot();
        let receiver_restrictions_before = receiver.tvar_restrictions_snapshot();

        let error = receiver
            .compose(&other)
            .expect_err("the merged substitution must enforce the right restriction");
        assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
        assert!(error.message.contains("active float dtype"));
        assert_eq!(receiver.types_snapshot(), receiver_types_before);
        assert_eq!(
            receiver.tvar_restrictions_snapshot(),
            receiver_restrictions_before
        );
    }

    #[test]
    fn direct_type_insertion_cannot_bypass_active_float_restrictions() {
        let restricted = TypeVar(80_009);
        let mut subst = Subst::new();
        subst
            .narrow_tvar_restriction(restricted, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");

        let error = subst
            .insert_type(restricted, Type::Prim(Prim::Int32))
            .expect_err("a forbidden direct insertion must fail explicitly");

        assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
        assert!(error.message.contains("active float dtype"));
        assert!(error.message.contains("i32"));
        assert_eq!(subst.apply(&Type::Var(restricted)), Type::Var(restricted));
        assert_eq!(
            subst.tvar_restriction(restricted),
            Some(TypeVarRestriction::ActiveFloat),
            "a rejected direct binding must preserve the restriction ledger"
        );
    }

    #[test]
    fn direct_type_insertion_transfers_and_resolves_active_float_restrictions() {
        let restricted = TypeVar(80_010);
        let alias = TypeVar(80_011);
        let mut subst = Subst::new();
        subst
            .narrow_tvar_restriction(restricted, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");

        subst
            .insert_type(restricted, Type::Var(alias))
            .expect("an unresolved alias preserves the semantic domain");
        assert_eq!(subst.tvar_restriction(restricted), None);
        assert_eq!(
            subst.tvar_restriction(alias),
            Some(TypeVarRestriction::ActiveFloat)
        );

        subst
            .insert_type(alias, Type::Prim(Prim::F16))
            .expect("an active float satisfies the transferred restriction");
        assert_eq!(subst.apply(&Type::Var(restricted)), Type::Prim(Prim::F16));
        assert_eq!(subst.tvar_restriction(alias), None);
    }

    #[test]
    fn structural_projection_preserves_domains_across_an_exact_signature_boundary() {
        let inferred_precision = TypeVar(80_012);
        let declared_precision = TypeVar(80_013);
        let mut subst = Subst::new();
        subst
            .narrow_tvar_restriction(inferred_precision, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        let inferred_tensor = Type::Tensor(vec![Dim::Lit(2)], TensorPrec::Var(inferred_precision));
        let declared_tensor = Type::Tensor(vec![Dim::Lit(2)], TensorPrec::Var(declared_precision));
        let inferred = Type::Fn(
            vec![
                Type::Ref(Box::new(inferred_tensor)),
                Type::Var(inferred_precision),
            ],
            Box::new(Type::Unit),
        );
        let declared = Type::Fn(
            vec![
                Type::Ref(Box::new(declared_tensor)),
                Type::Var(declared_precision),
            ],
            Box::new(Type::Unit),
        );

        subst
            .project_tvar_restrictions(&inferred, &declared)
            .expect("matching signature positions carry the inferred domain");
        assert_eq!(
            subst.tvar_restriction(declared_precision),
            Some(TypeVarRestriction::ActiveFloat)
        );
        let error = subst
            .insert_type(declared_precision, Type::Prim(Prim::Bool))
            .expect_err("the projected declaration must reject a non-float");
        assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
    }

    #[test]
    fn structural_projection_is_transactional_when_a_later_slot_conflicts() {
        let first_source = TypeVar(80_014);
        let second_source = TypeVar(80_015);
        let first_target = TypeVar(80_016);
        let mut subst = Subst::new();
        subst
            .narrow_tvar_restriction(first_source, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        subst
            .narrow_tvar_restriction(second_source, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");
        let inferred = Type::Tuple(vec![Type::Var(first_source), Type::Var(second_source)]);
        let declared = Type::Tuple(vec![Type::Var(first_target), Type::Prim(Prim::Int32)]);
        let restrictions_before = subst.tvar_restrictions_snapshot();

        let error = subst
            .project_tvar_restrictions(&inferred, &declared)
            .expect_err("a forbidden declared slot rejects the whole projection");
        assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
        assert_eq!(subst.tvar_restrictions_snapshot(), restrictions_before);
        assert_eq!(subst.tvar_restriction(first_target), None);
    }

    #[test]
    fn pp1_monomorphic_binding_is_lowered_before_a_sibling_boundary() {
        let mut vg = var_gen();
        let mut subst = Subst::new();
        let pp1_level = subst.enter_level(&vg);
        let pp1_var = vg.fresh_tvar();
        subst.leave_level(pp1_level, &vg);
        subst.lower_type_to_current(&Type::Var(pp1_var));

        let mut env = crate::env::Env::new();
        env.bind(
            "pending_shape".to_string(),
            Scheme::mono(Type::Var(pp1_var)),
        );
        let sibling_level = subst.enter_level(&vg);
        let sibling_var = vg.fresh_tvar();
        subst.leave_level(sibling_level, &vg);
        let scheme = env.generalize(
            &Type::Tuple(vec![Type::Var(pp1_var), Type::Var(sibling_var)]),
            &subst,
        );
        assert_eq!(scheme.tvars, vec![sibling_var]);
    }

    #[test]
    fn unify_same_prim() {
        let mut s = Subst::new();
        assert!(unify(&Type::Prim(Prim::F32), &Type::Prim(Prim::F32), &mut s).is_ok());
    }

    #[test]
    fn unify_different_prim_fails() {
        let mut s = Subst::new();
        let err = unify(&Type::Prim(Prim::F32), &Type::Prim(Prim::Bf16), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_tvar_binds() {
        let mut g = var_gen();
        let v = g.fresh_type();
        let mut s = Subst::new();
        assert!(unify(&v, &Type::Prim(Prim::F32), &mut s).is_ok());
        assert_eq!(s.apply(&v), Type::Prim(Prim::F32));
    }

    #[test]
    fn unify_fn_types() {
        let mut s = Subst::new();
        let f1 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        let f2 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        assert!(unify(&f1, &f2, &mut s).is_ok());
    }

    #[test]
    fn unify_fn_arity_mismatch() {
        let mut s = Subst::new();
        let f1 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        let f2 = Type::Fn(
            vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)],
            Box::new(Type::Prim(Prim::F32)),
        );
        let err = unify(&f1, &f2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::ArityMismatch));
    }

    fn tprec(p: Prim) -> TensorPrec {
        TensorPrec::Concrete(p)
    }

    #[test]
    fn unify_tensor_same_dims() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
            tprec(Prim::F32),
        );
        let t2 = Type::Tensor(
            vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
            tprec(Prim::F32),
        );
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn unify_tensor_dim_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("seq".into())], tprec(Prim::F32));
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_tensor_precision_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_dim_var_binds() {
        let mut g = var_gen();
        let dv = g.fresh_dvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Var(dv)], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        assert!(unify(&t1, &t2, &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Name("batch".into()));
    }

    // chelis#258 Tier-2 rank polymorphism: the rank-variable unification arm.

    #[test]
    fn unify_rank_binds_whole_shape() {
        let mut g = var_gen();
        let rv = g.fresh_rvar();
        let mut s = Subst::new();
        let rank_tensor = Type::Tensor(vec![Dim::Rank(rv)], tprec(Prim::F32));
        let concrete = Type::Tensor(
            vec![Dim::Name("a".into()), Dim::Name("b".into())],
            tprec(Prim::F32),
        );
        assert!(unify(&rank_tensor, &concrete, &mut s).is_ok());
        // The rank var now stands for the whole 2-dim shape; applying expands it.
        assert_eq!(s.apply(&rank_tensor), concrete);
    }

    #[test]
    fn unify_rank_to_rank_aliases() {
        let mut g = var_gen();
        let r1 = g.fresh_rvar();
        let r2 = g.fresh_rvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Rank(r1)], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Rank(r2)], tprec(Prim::F32));
        assert!(unify(&t1, &t2, &mut s).is_ok());
        // Binding one to a concrete shape resolves both (alias chased).
        let concrete = Type::Tensor(vec![Dim::Name("n".into())], tprec(Prim::F32));
        assert!(unify(&t2, &concrete, &mut s).is_ok());
        assert_eq!(s.apply(&t1), concrete);
    }

    #[test]
    fn unify_rank_precision_still_checked() {
        // The precision slot unifies independently of the rank binding.
        let mut g = var_gen();
        let rv = g.fresh_rvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Rank(rv)], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("a".into())], tprec(Prim::Int32));
        assert!(
            unify(&t1, &t2, &mut s).is_err(),
            "f32 vs i32 precision must fail even with a rank var"
        );
    }

    // chelis#258 Tier-3 rank polymorphism: name-preserving multi-spread
    // unification (`tensor[..pre, seq, ..post]`).

    fn name(n: &str) -> Dim {
        Dim::Name(n.into())
    }

    #[test]
    fn unify_rank_middle_split_preserves_names() {
        // `tensor[..pre, seq, ..post]` vs `tensor[batch, seq, hidden]` splits
        // at the unique named anchor: pre:=[batch], post:=[hidden]. The output
        // `tensor[..pre, ..post]` then carries the surviving names through.
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(
            vec![name("batch"), name("seq"), name("hidden")],
            tprec(Prim::F32),
        );
        assert!(unify(&row, &ground, &mut s).is_ok());
        let out = Type::Tensor(vec![Dim::Rank(pre), Dim::Rank(post)], tprec(Prim::F32));
        assert_eq!(
            s.apply(&out),
            Type::Tensor(vec![name("batch"), name("hidden")], tprec(Prim::F32)),
            "output drops `seq`, keeps batch/hidden by name"
        );
    }

    #[test]
    fn unify_rank_multi_anchor_split() {
        // Two named anchors, three spreads: `[..a, seq, ..b, head, ..c]`.
        let mut g = var_gen();
        let (a, b, c) = (g.fresh_rvar(), g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![
                Dim::Rank(a),
                name("seq"),
                Dim::Rank(b),
                name("head"),
                Dim::Rank(c),
            ],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(
            vec![
                name("batch"),
                name("seq"),
                name("kv"),
                name("head"),
                name("dim"),
            ],
            tprec(Prim::F32),
        );
        assert!(unify(&row, &ground, &mut s).is_ok());
        let out = Type::Tensor(
            vec![Dim::Rank(a), Dim::Rank(b), Dim::Rank(c)],
            tprec(Prim::F32),
        );
        assert_eq!(
            s.apply(&out),
            Type::Tensor(
                vec![name("batch"), name("kv"), name("dim")],
                tprec(Prim::F32)
            ),
        );
    }

    #[test]
    fn unify_rank_anchor_absent_rejected() {
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(vec![name("batch"), name("hidden")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
        assert!(
            err.message.contains("no named `seq` axis"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unify_rank_anchor_ambiguous_rejected() {
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(vec![name("seq"), name("x"), name("seq")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(err.message.contains("ambiguous"), "{}", err.message);
    }

    #[test]
    fn unify_rank_literal_operand_rejected() {
        // Name↔Lit hazard: a fully-literal operand carries no name to locate
        // the anchor — hard reject, never a guessed reduction.
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(
            vec![Dim::Lit(2), Dim::Lit(768), Dim::Lit(4)],
            tprec(Prim::F32),
        );
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(
            err.message.contains("no named `seq` axis"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unify_rank_too_small_rejected() {
        // Two leading anchors but a rank-1 operand: not enough dims.
        let mut g = var_gen();
        let r = g.fresh_rvar();
        let mut s = Subst::new();
        let row = Type::Tensor(vec![name("a"), name("b"), Dim::Rank(r)], tprec(Prim::F32));
        let ground = Type::Tensor(vec![name("a")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(err.message.contains("rank too small"), "{}", err.message);
    }

    #[test]
    fn unify_two_adjacent_spreads_rejected() {
        // Two adjacent spreads split against a ground is undetermined.
        let mut g = var_gen();
        let (a, b) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(vec![Dim::Rank(a), Dim::Rank(b)], tprec(Prim::F32));
        let ground = Type::Tensor(vec![name("x"), name("y")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(
            err.message.contains("two adjacent rank spreads"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unify_identical_rows_ok() {
        // The body output `[..pre, ..post]` vs the declared return
        // `[..pre, ..post]` (same vars) is the identical-row self-check.
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        assert!(unify(&row, &row.clone(), &mut s).is_ok());
    }

    #[test]
    fn unify_distinct_anchor_rows_rejected() {
        let mut g = var_gen();
        let (a, b) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let r1 = Type::Tensor(
            vec![Dim::Rank(a), name("seq"), Dim::Rank(b)],
            tprec(Prim::F32),
        );
        let r2 = Type::Tensor(
            vec![Dim::Rank(a), name("head"), Dim::Rank(b)],
            tprec(Prim::F32),
        );
        assert!(unify(&r1, &r2, &mut s).is_err());
    }

    #[test]
    fn unify_trailing_anchor_by_name() {
        // `[..pre, seq]` vs `[a, b, seq]`: pre:=[a,b], seq at the tail.
        let mut g = var_gen();
        let pre = g.fresh_rvar();
        let mut s = Subst::new();
        let row = Type::Tensor(vec![Dim::Rank(pre), name("seq")], tprec(Prim::F32));
        let ground = Type::Tensor(vec![name("a"), name("b"), name("seq")], tprec(Prim::F32));
        assert!(unify(&row, &ground, &mut s).is_ok());
        let out = Type::Tensor(vec![Dim::Rank(pre)], tprec(Prim::F32));
        assert_eq!(
            s.apply(&out),
            Type::Tensor(vec![name("a"), name("b")], tprec(Prim::F32)),
        );
    }

    #[test]
    fn unify_name_with_lit_accepts() {
        // Issue Chelis-Lang/chelis#219 Option A: a Name slot on the
        // left and a concrete Lit on the right unify without binding
        // any substitution.
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Name("batch".into()), &Dim::Lit(2), &mut s).is_ok());
        // Subst stays untouched — the Name carries diagnostics, the
        // Lit carries the value, nothing flows in.
        assert_eq!(s.dims_len(), 0);
    }

    #[test]
    fn unify_name_with_lit_accepts_symmetric() {
        // Issue #219 Option A: the symmetric direction (Lit on left,
        // Name on right) also unifies.
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Lit(3), &Dim::Name("hidden".into()), &mut s).is_ok());
        assert_eq!(s.dims_len(), 0);
    }

    #[test]
    fn unify_name_distinct_names_still_errors() {
        // Issue #219 regression-lock: the Name <-> Lit relaxation
        // must not bleed into Name <-> Name. Distinct symbolic names
        // continue to surface as DimensionMismatch (this is the
        // dim-polymorphism rigidity rule from §4.4).
        let mut s = Subst::new();
        let err =
            unify_dim(&Dim::Name("batch".into()), &Dim::Name("seq".into()), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_lit_distinct_still_errors() {
        // Issue #219 regression-lock: distinct Lit <-> Lit still
        // errors. The permissive arm is narrowly Name <-> Lit.
        let mut s = Subst::new();
        let err = unify_dim(&Dim::Lit(2), &Dim::Lit(3), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_dim_transitive_binding_resolves() {
        let mut g = var_gen();
        let d0 = g.fresh_dvar();
        let d1 = g.fresh_dvar();
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Var(d0), &Dim::Var(d1), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(d1), &Dim::Name("batch".into()), &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(d0)), Dim::Name("batch".into()));
    }

    #[test]
    fn unify_wildcard_matches_anything() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn unify_wildcard_with_var_leaves_var_free() {
        // chelis#143 invariant: `unify_dim(Wildcard, Var(v))` must
        // succeed without binding `v`. Binding `v := Wildcard` would
        // freeze the dim var; a later concrete arg in the same sig
        // couldn't then constrain `v` because `apply_dim` would
        // resolve `Var(v) → Wildcard` and the permissive Wildcard arm
        // would silently accept any concrete value.
        //
        // The current implementation satisfies this because both
        // wildcard arms match before the `(Var(_), _) => bind_dvar(...)`
        // arm (Rust `match` is first-match-wins) and neither binds the
        // dim var. chelis#1801 added the explicit
        // `(Wildcard, Var(v)) | (Var(v), Wildcard)` arm ahead of the
        // permissive one; it RECORDS the meeting in
        // `wildcard_touched_classes` and still leaves `v` free, which is
        // what lets the concrete binding below still happen. This test
        // pins the property in case a future refactor reorders the arms
        // or makes either wildcard arm bind.
        let mut g = var_gen();
        let dv = g.fresh_dvar();
        let mut s = Subst::new();

        // Wildcard ↔ Var: succeeds, var remains free.
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(dv), &mut s).is_ok());
        assert_eq!(
            s.apply_dim(&Dim::Var(dv)),
            Dim::Var(dv),
            "Wildcard ↔ Var must not bind the var to Wildcard",
        );

        // Symmetric direction: Var ↔ Wildcard also leaves the var
        // free.
        let dv2 = g.fresh_dvar();
        assert!(unify_dim(&Dim::Var(dv2), &Dim::Wildcard, &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv2)), Dim::Var(dv2));

        // A subsequent concrete unification with the still-free var
        // binds it to the concrete dim.
        assert!(unify_dim(&Dim::Var(dv), &Dim::Lit(2), &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Lit(2));

        // And once the var is concretely bound, a conflicting concrete
        // value trips DimensionMismatch — the cross-position contract
        // the sig promised is now enforced end to end.
        let err = unify_dim(&Dim::Var(dv), &Dim::Lit(3), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn a_wildcard_meeting_a_class_is_recorded_and_survives_clone_and_compose() {
        // chelis#1801. `unify_dim`'s wildcard arm leaves the variable free
        // by design, so without this record nothing downstream can tell a
        // variable that met a runtime extent from one that met nothing. The
        // application rule reads the record to decide what an unconstrained
        // instantiation variable denotes (spec/04-type-system.md section
        // 3.2).
        //
        // Evidentiary status: a mechanism test for a mechanism this change
        // introduces. There is no base-sha reading to compare against; what
        // it proves is that each of the three carriers keeps the evidence.
        let mut g = var_gen();
        let touched = g.fresh_dvar();
        let untouched = g.fresh_dvar();
        let mut s = Subst::new();

        assert!(!s.dvar_class_met_wildcard(touched));
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(touched), &mut s).is_ok());
        assert!(
            s.dvar_class_met_wildcard(touched),
            "the Wildcard-then-Var direction records the meeting",
        );
        assert!(
            !s.dvar_class_met_wildcard(untouched),
            "and records nothing about a variable that met nothing",
        );

        // The symmetric direction records too.
        let mut symmetric = Subst::new();
        assert!(unify_dim(&Dim::Var(touched), &Dim::Wildcard, &mut symmetric).is_ok());
        assert!(symmetric.dvar_class_met_wildcard(touched));

        // A wildcard against a wildcard names no variable, so it records
        // nothing: the permissive arm is still what handles it.
        let mut both_wild = Subst::new();
        assert!(unify_dim(&Dim::Wildcard, &Dim::Wildcard, &mut both_wild).is_ok());
        assert!(!both_wild.dvar_class_met_wildcard(touched));

        // `Subst::clone` carries the record. A clone that dropped it would
        // silently un-record the meeting for every checking context that
        // clones the substitution into its own state.
        let cloned = s.clone();
        assert!(cloned.dvar_class_met_wildcard(touched));
        assert!(!cloned.dvar_class_met_wildcard(untouched));

        // `Subst::compose` carries it in BOTH directions: from the receiver
        // (through the trial clone) and from the argument (through
        // `absorb_wildcard_touches`). The argument direction is the one a
        // compose can lose, because `compose_bindings` merges only the
        // binding maps.
        let mut receiver = Subst::new();
        receiver.compose(&s).expect("composing a rename is valid");
        assert!(
            receiver.dvar_class_met_wildcard(touched),
            "compose adopts the argument's wildcard-meeting evidence",
        );

        let mut carrier = s.clone();
        carrier
            .compose(&Subst::new())
            .expect("composing an empty substitution is valid");
        assert!(
            carrier.dvar_class_met_wildcard(touched),
            "and keeps its own",
        );
    }

    #[test]
    fn a_recorded_meeting_does_not_by_itself_mean_the_class_is_still_free() {
        // chelis#1801 negative parity for the test above, and the reason the
        // absorbing site asks the substitution a second question. The record
        // is monotone: it says the meeting happened, never that the variable
        // is still unconstrained. A literal in a later argument of the same
        // call binds the variable, and `constraint_dim` is what reports that.
        //
        // `apply_dim` cannot: it deliberately answers `Var(v)` for a
        // LABELLED variable that resolved to a concrete dim, so an absorber
        // written against it would overwrite the literal with `*`.
        //
        // Evidentiary status: a disposition lock on which predicate answers
        // "is this variable still unbound", not a regression test. The state
        // is built here directly. Swapping the absorbing site to `apply_dim`
        // was measured and changed no source-level reading, so no source
        // fixture pins this; what it pins is the contract the absorbing site
        // relies on.
        let mut g = var_gen();
        let dv = g.fresh_dvar();
        let mut s = Subst::new();
        s.set_dimension_label(dv, "k".to_string());

        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(dv), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(dv), &Dim::Lit(3), &mut s).is_ok());

        assert!(
            s.dvar_class_met_wildcard(dv),
            "the meeting is still recorded after the literal binding",
        );
        assert_eq!(
            s.constraint_dim(&Dim::Var(dv)),
            Dim::Lit(3),
            "but the variable is bound, so it denotes the literal",
        );
        assert_eq!(
            s.apply_dim(&Dim::Var(dv)),
            Dim::Var(dv),
            "`apply_dim` keeps the label's identity and cannot answer this",
        );
    }

    #[test]
    fn a_class_keeps_its_wildcard_evidence_when_a_later_binding_re_roots_it() {
        // chelis#1925 round 2, at the unit level. `unify_dim` resolves both
        // operands through `constraint_dim` before it matches, so the meeting
        // is recorded on whatever rooted the class at that moment, and a
        // later `bind_dvar` in the same call re-roots the class over it. Ask
        // the variable, or two ends of the class, and a meeting recorded on a
        // third member is invisible; the class that carried it is then
        // generalized instead of denoting the extent it met.
        //
        // Evidentiary status: regression test for the merge this change
        // introduces. Making `merge_dvar_class_flags` a no-op fails the loop
        // below, which is how it was checked.
        let mut g = var_gen();
        let (first, middle, last) = (g.fresh_dvar(), g.fresh_dvar(), g.fresh_dvar());
        let mut s = Subst::new();

        // The meeting happens while `middle` still roots its own class.
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(middle), &mut s).is_ok());
        assert!(s.dvar_class_met_wildcard(middle));

        // Two more members join and the class re-roots away from `middle`.
        assert!(unify_dim(&Dim::Var(middle), &Dim::Var(last), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(first), &Dim::Var(middle), &mut s).is_ok());

        for member in [first, middle, last] {
            assert!(
                s.dvar_class_met_wildcard(member),
                "every member answers for the class, whichever member roots it",
            );
        }
    }

    #[test]
    fn an_authored_binder_pins_its_class_even_after_it_meets_a_runtime_extent() {
        // Negative parity for the row above, and the exclusion
        // `spec/04-type-system.md` section 3.2 states: a class an argument
        // binds to a named dimension is a CLAIM on the runtime extent,
        // checked by a section 4.7 guard, so absorbing it to `*` would erase
        // the claim. The meeting is still recorded; the pin is what stops the
        // absorption, and both answers have to survive the same merge.
        //
        // Evidentiary status: disposition lock on the two predicates the
        // absorbing site asks. The source-level twin is
        // `an_enclosing_binder_survives_a_wildcard_in_its_own_alias_class`.
        let mut g = var_gen();
        let binder = g.fresh_dvar();
        let minted = g.fresh_dvar();
        let mut s = Subst::new();
        s.protect_dimensions([binder]);
        s.set_dimension_label(binder, "seq".to_string());

        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(minted), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(minted), &Dim::Var(binder), &mut s).is_ok());

        assert!(
            s.dvar_class_met_wildcard(minted),
            "the class still carries the meeting after the binder joins it",
        );
        assert!(
            s.dvar_class_is_binder_pinned(minted),
            "and the authored binder pins the class against absorption",
        );
        assert!(
            s.dvar_class_is_binder_pinned(binder),
            "asked from either member",
        );
    }

    #[test]
    fn a_name_arm_class_is_pinned_by_protection_not_by_a_second_ledger() {
        // chelis#1925 round 2, the first of the paths that pin or union
        // WITHOUT a plain `insert_dim`. `bind_dvar`'s name arm records the
        // name as a LABEL and returns, so `constraint_dim` still answers the
        // variable and the absorbing site has no binding to skip on.
        //
        // The arm needs no ledger of its own. It fires only when the variable
        // is protected or already labelled, and a `Dim::Name` operand arises
        // only from the definition under check, whose binders are protected.
        // The protected half is therefore the whole reachable half, and it
        // pins through the rigidity ledger.
        //
        // Evidentiary status, per assertion. The protected rows are a
        // disposition lock on the pin the absorbing site relies on. The last
        // row records a state no source program reaches, measured rather than
        // assumed: `def named(w: tensor[batch, f32])` passed as a polymorphic
        // argument reads `() -> tensor[batch, f32]` and executes identically
        // with and without a pin on this arm, because the reachable spelling
        // BINDS the name rather than labelling it, and `constraint_dim`
        // reports that. If a change ever makes the labelled-but-unprotected
        // state reachable, this row is where the missing pin goes.
        let mut g = var_gen();
        let protected = g.fresh_dvar();
        let mut s = Subst::new();
        s.protect_dimensions([protected]);

        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(protected), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(protected), &Dim::Name("seq".to_string()), &mut s).is_ok());

        assert_eq!(
            s.constraint_dim(&Dim::Var(protected)),
            Dim::Var(protected),
            "the name arm binds nothing, so the class still reads as unbound",
        );
        assert!(s.dvar_class_met_wildcard(protected));
        assert!(
            s.dvar_class_is_binder_pinned(protected),
            "and the declared binder pins it against absorption",
        );

        let labelled = g.fresh_dvar();
        let mut unreached = Subst::new();
        unreached.set_dimension_label(labelled, "seq".to_string());
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(labelled), &mut unreached).is_ok());
        assert!(
            unify_dim(
                &Dim::Var(labelled),
                &Dim::Name("seq".to_string()),
                &mut unreached
            )
            .is_ok()
        );
        assert!(
            !unreached.dvar_class_is_binder_pinned(labelled),
            "a labelled but UNPROTECTED variable is not pinned by this arm, and no \
             source program reaches that state today",
        );
    }

    #[test]
    fn a_swapped_binding_direction_still_lands_the_evidence_on_the_merged_root() {
        // The second such path. When `v` is protected and the other operand
        // is not, `bind_dvar` recurses with the operands SWAPPED, so the
        // union direction flips. The merge therefore hooks the actual insert
        // rather than the entry; hooking the entry would key the evidence to
        // a variable that does not root the merged class.
        //
        // Evidentiary status: disposition lock on where the hook sits.
        let mut g = var_gen();
        let binder = g.fresh_dvar();
        let minted = g.fresh_dvar();
        let mut s = Subst::new();
        s.protect_dimensions([binder]);

        // The meeting is recorded on the unprotected member, and the
        // protected one is unified FIRST, which is the direction that
        // recurses.
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(minted), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(binder), &Dim::Var(minted), &mut s).is_ok());

        assert!(
            s.dvar_class_met_wildcard(binder),
            "the meeting survives the swap, asked from the protected member",
        );
        assert!(s.dvar_class_met_wildcard(minted), "and from the other one");
        assert!(
            s.dvar_class_is_binder_pinned(minted),
            "and the protected member pins the class it was merged into",
        );
    }

    #[test]
    fn a_literal_claim_against_a_named_dimension_leaves_no_absorbable_class() {
        // The third such path. `unify_dim`'s `(Name, Lit)` arm returns `Ok`
        // binding nothing (issue #219 Option A), so a literal claim against a
        // named dimension produces no union event either. It needs no hook:
        // by the time that arm matches, `constraint_dim` has already resolved
        // both operands to non-variables, so no class is left for an
        // application to absorb, and the class the name came from was pinned
        // when the declared binder reached it.
        //
        // Evidentiary status: a reachability receipt for that claim rather
        // than a lock on a repair. It is the row to re-run if the arm ever
        // starts binding.
        let mut g = var_gen();
        let named = g.fresh_dvar();
        let bystander = g.fresh_dvar();
        let mut s = Subst::new();
        s.protect_dimensions([named]);
        s.set_dimension_label(named, "batch".to_string());

        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(named), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(named), &Dim::Name("batch".to_string()), &mut s).is_ok());
        assert!(
            s.dvar_class_is_binder_pinned(named),
            "the declared binder pinned the class before any literal arrived",
        );

        // The literal claim now meets the NAME, not the variable.
        assert!(unify_dim(&Dim::Name("batch".to_string()), &Dim::Lit(3), &mut s).is_ok());
        assert!(
            s.dvar_class_is_binder_pinned(named),
            "and the literal arm leaves the pin where it was",
        );
        assert!(
            !s.dvar_class_met_wildcard(bystander),
            "recording nothing about any other class",
        );
    }

    #[test]
    fn compose_rekeys_class_evidence_onto_the_root_the_composed_graph_gives_it() {
        // The fourth such path, and the one a set union gets wrong.
        // `compose_bindings` merges only the binding maps, so a compose can
        // JOIN two classes that were separate in either operand. Copying the
        // flags across leaves the meeting keyed to a variable that no longer
        // roots the joined class, and the query on the new root answers
        // false.
        //
        // Evidentiary status: regression test for the re-canonicalization.
        // Replacing `recanonicalize_class_flags` with the set union it
        // supersedes fails the second-to-last assertion, which is how it was
        // checked.
        let mut g = var_gen();
        let met = g.fresh_dvar();
        let joiner = g.fresh_dvar();

        // The left operand knows the meeting and roots that class at `met`.
        let mut left = Subst::new();
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(met), &mut left).is_ok());
        assert!(left.dvar_class_met_wildcard(met));

        // The right operand knows an aliasing the left never saw, and it
        // re-roots that class onto `joiner`.
        let mut right = Subst::new();
        assert!(unify_dim(&Dim::Var(met), &Dim::Var(joiner), &mut right).is_ok());
        assert!(!right.dvar_class_met_wildcard(joiner));

        left.compose(&right).expect("composing an alias is valid");

        assert_eq!(
            left.constraint_dim(&Dim::Var(met)),
            Dim::Var(joiner),
            "the compose joined the classes and `joiner` roots the result",
        );
        assert!(
            left.dvar_class_met_wildcard(joiner),
            "so the meeting has to be readable from the new root",
        );
        assert!(
            left.dvar_class_met_wildcard(met),
            "and from the member it was recorded on",
        );
    }

    #[test]
    fn compose_keeps_the_binder_pin_when_it_demotes_the_protected_member() {
        // The pin's half of the path above, and the reason
        // `binder_pinned_classes` exists at all rather than the query reading
        // the rigidity ledger alone.
        //
        // `bind_dvar` never demotes a protected member: when the protected
        // side would bind to an unprotected one it recurses with the operands
        // swapped, so protection stays at the root and the merge needs no
        // help. `compose_bindings` has no such rule, so a compose CAN put an
        // unprotected variable at the root of a class a declared binder pins,
        // and the set is what keeps that answer reachable.
        //
        // Evidentiary status: regression test for the protection fold in
        // `recanonicalize_class_flags`. Removing that fold fails the last
        // assertion, which is how it was checked. Removing the same fold from
        // `merge_dvar_class_flags` fails nothing today, for the swap reason
        // above; it is kept because a class this compose pinned can still be
        // merged afterwards, and the set is then the only carrier.
        let mut g = var_gen();
        let binder = g.fresh_dvar();
        let joiner = g.fresh_dvar();

        let mut left = Subst::new();
        left.protect_dimensions([binder]);
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(binder), &mut left).is_ok());
        assert!(
            left.dvar_class_is_binder_pinned(binder),
            "the declared binder pins its own class before the compose",
        );

        // The other operand re-roots that class onto an unprotected variable,
        // which is the demotion `bind_dvar` would have refused.
        let mut right = Subst::new();
        assert!(unify_dim(&Dim::Var(binder), &Dim::Var(joiner), &mut right).is_ok());

        left.compose(&right).expect("composing an alias is valid");

        assert_eq!(
            left.constraint_dim(&Dim::Var(binder)),
            Dim::Var(joiner),
            "the compose demoted the protected member out of the root",
        );
        assert!(
            left.dvar_class_is_binder_pinned(joiner),
            "and the pin has to survive it, or the absorption would erase the claim",
        );
    }

    #[test]
    fn unify_tensor_rank_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("a".into())], tprec(Prim::F32));
        let t2 = Type::Tensor(
            vec![Dim::Name("a".into()), Dim::Name("b".into())],
            tprec(Prim::F32),
        );
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_dim_cannot_recover_lit_from_two_var_tensors() {
        // chelis#158 cause partition: when two tensors both arrive at
        // a shared-dim sig with `Var(d)` (or `Wildcard`) dims and
        // never with a concrete `Lit(_)` source, `unify_dim` cannot
        // manufacture a mismatch. This is the unify-layer half of
        // chelis#158's root cause: `to_tensor` and `pad_sequences_to`
        // emit shape-erased tensors that reach the sig with no `Lit`
        // in any position; unify_dim correctly succeeds on every pair
        // and the sig's shared dim var stays free.
        //
        // The fix for chelis#158 must inject a `Lit` somewhere
        // UPSTREAM of unify (most likely in the desugarer when
        // `to_tensor` is applied to a statically-known list literal).
        // This test pins the unify-layer contract so any future
        // "unify_dim should be smarter" proposal can be measured
        // against the constraint that follows: from Var/Wildcard
        // inputs alone, no concrete dim can be inferred.
        let mut g = var_gen();
        let d_sig = g.fresh_dvar();

        // Two sig-instantiated tensor params that share `d_sig`.
        let sig_arg1 = Type::Tensor(vec![Dim::Var(d_sig)], tprec(Prim::F32));
        let sig_arg2 = Type::Tensor(vec![Dim::Var(d_sig)], tprec(Prim::F32));

        // Two caller tensors, both shape-erased to fresh `Var(d_caller_*)`
        // (modeling two independent to_tensor outputs).
        let d_caller_a = g.fresh_dvar();
        let d_caller_b = g.fresh_dvar();
        let caller_a = Type::Tensor(vec![Dim::Var(d_caller_a)], tprec(Prim::F32));
        let caller_b = Type::Tensor(vec![Dim::Var(d_caller_b)], tprec(Prim::F32));

        let mut s = Subst::new();
        // Simulate `pair_id(caller_a, caller_b)` where pair_id's sig
        // is `&tensor[d_sig, f32] -> &tensor[d_sig, f32] -> ...`.
        assert!(unify(&sig_arg1, &caller_a, &mut s).is_ok());
        assert!(unify(&sig_arg2, &caller_b, &mut s).is_ok());

        // After both unifications, `d_sig` is bound to (some chain
        // of) dim vars but never to a concrete `Lit`.
        let resolved = s.apply_dim(&Dim::Var(d_sig));
        assert!(
            matches!(resolved, Dim::Var(_) | Dim::Wildcard),
            "sig dim var resolved to a concrete value despite no \
             Lit source in either caller; this would mean unify_dim \
             manufactured a constraint from nothing, which is the \
             wrong place to fix chelis#158. Got: {resolved:?}"
        );

        // Symmetric: replacing one caller's Var with Wildcard (the
        // actual shape-erased symptom from to_tensor) also leaves
        // d_sig unable to bind to a Lit. No mismatch is producible.
        let mut s2 = Subst::new();
        let caller_a_wild = Type::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
        let caller_b_wild = Type::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
        assert!(unify(&sig_arg1, &caller_a_wild, &mut s2).is_ok());
        assert!(unify(&sig_arg2, &caller_b_wild, &mut s2).is_ok());
        let resolved2 = s2.apply_dim(&Dim::Var(d_sig));
        assert!(
            matches!(resolved2, Dim::Var(_) | Dim::Wildcard),
            "with Wildcard caller dims, d_sig also cannot resolve to \
             a concrete Lit. Got: {resolved2:?}. The fix for \
             chelis#158 must introduce a Lit at the desugar/builder \
             layer."
        );
    }

    // === WS-A5 precision polymorphism unification ===

    #[test]
    fn static_dim_product_comparison_is_exact_beyond_i128() {
        let s = Subst::new();
        let max = Dim::Lit(i64::MAX);
        let lhs = vec![max.clone(), max.clone(), max.clone(), Dim::Lit(2)];
        let rhs = vec![Dim::Lit(2), max.clone(), max.clone(), max.clone()];
        let unequal = vec![max.clone(), max.clone(), max, Dim::Lit(3)];

        assert_eq!(s.static_dim_product(&lhs), None);
        assert_eq!(s.static_dim_products_match(&lhs, &rhs), Some(true));
        assert_eq!(s.static_dim_products_match(&lhs, &unequal), Some(false));
    }

    #[test]
    fn unify_tensor_prec_var_binds_to_concrete() {
        // tensor[batch, ?p] vs tensor[batch, f32]
        // ?p must bind to f32 in the substitution.
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        unify(&t1, &t2, &mut s).expect("var precision should bind");
        assert_eq!(s.apply(&Type::Var(pv)), Type::Prim(Prim::F32));
        // After substitution, the var-precision tensor reads as f32 too.
        match s.apply(&t1) {
            Type::Tensor(_, TensorPrec::Concrete(Prim::F32)) => {}
            other => panic!("apply should resolve precision var to f32, got {other}"),
        }
    }

    #[test]
    fn unify_tensor_prec_concrete_binds_var_other_side() {
        // Symmetric: concrete on left, var on right.
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        unify(&t1, &t2, &mut s).expect("var precision should bind");
        assert_eq!(s.apply(&Type::Var(pv)), Type::Prim(Prim::Bf16));
    }

    #[test]
    fn unify_two_tensor_prec_vars_link() {
        // Two precision vars unify by linking; the link survives later
        // binding through either var.
        let mut g = var_gen();
        let p1 = g.fresh_tvar();
        let p2 = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(p1));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(p2));
        unify(&t1, &t2, &mut s).expect("two prec vars should link");
        // Now bind one of them to a concrete prim and confirm the other
        // resolves to the same prim through the link.
        let t3 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F64));
        unify(&t1, &t3, &mut s).expect("link + bind should work");
        assert_eq!(s.apply(&Type::Var(p1)), Type::Prim(Prim::F64));
        assert_eq!(s.apply(&Type::Var(p2)), Type::Prim(Prim::F64));
    }

    #[test]
    fn unify_tensor_prec_var_then_conflicting_concrete_errors() {
        // Once ?p is bound to f32, unifying tensor[..., ?p] with
        // tensor[..., bf16] must error with PrecisionMismatch.
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t_var = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        let t_f32 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        let t_bf16 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
        unify(&t_var, &t_f32, &mut s).expect("first call binds prec var to f32");
        let err = unify(&t_var, &t_bf16, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_same_prec_var_with_itself_succeeds() {
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        unify(&t1, &t2, &mut s).expect("self-unify must succeed");
    }

    #[test]
    fn apply_tensor_prec_resolves_through_chain() {
        // Build a substitution that puts ?p1 -> ?p2 -> Prim::F32.
        let mut g = var_gen();
        let p1 = g.fresh_tvar();
        let p2 = g.fresh_tvar();
        let mut s = Subst::new();
        unify(&Type::Var(p1), &Type::Var(p2), &mut s).unwrap();
        unify(&Type::Var(p2), &Type::Prim(Prim::F32), &mut s).unwrap();
        // apply_tensor_prec on either var must collapse to Concrete(F32).
        assert_eq!(
            s.apply_tensor_prec(&TensorPrec::Var(p1)),
            TensorPrec::Concrete(Prim::F32)
        );
        assert_eq!(
            s.apply_tensor_prec(&TensorPrec::Var(p2)),
            TensorPrec::Concrete(Prim::F32)
        );
    }

    #[test]
    fn unify_adt_same() {
        let mut s = Subst::new();
        let a1 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        let a2 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        assert!(unify(&a1, &a2, &mut s).is_ok());
    }

    #[test]
    fn unify_adt_name_mismatch() {
        let mut s = Subst::new();
        let a1 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        let a2 = Type::Adt("Result".into(), vec![Type::Prim(Prim::F32)]);
        assert!(unify(&a1, &a2, &mut s).is_err());
    }

    #[test]
    fn occurs_check_prevents_infinite_type() {
        let mut g = var_gen();
        let v = g.fresh_tvar();
        let mut s = Subst::new();
        // Try to unify ?0 with List(?0) — should fail with occurs check
        let list_v = Type::Adt("List".into(), vec![Type::Var(v)]);
        let err = unify(&Type::Var(v), &list_v, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::OccursCheck));
    }

    #[test]
    fn error_type_unifies_with_anything() {
        // Per WS-A5 RT-3a F1 escalation: the permissive rule is retained
        // here so a single upstream error does not fan out a cascade of
        // secondary diagnostics from one root cause; the silent
        // passthrough at the def-body vs declared-sig boundary is
        // closed at the call site in `infer.rs`, not by tightening the
        // unification rule.
        let mut s = Subst::new();
        let err = crate::errors::error_sentinel_for_test();
        assert!(unify(&err, &Type::Prim(Prim::F32), &mut s).is_ok());
        assert!(unify(&Type::Prim(Prim::F32), &err, &mut s).is_ok());
    }

    #[test]
    fn unify_tuple_types() {
        let mut s = Subst::new();
        let t1 = Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::Int32)]);
        let t2 = Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::Int32)]);
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn transitive_binding() {
        let mut g = var_gen();
        let a = g.fresh_tvar();
        let b = g.fresh_tvar();
        let mut s = Subst::new();
        // a = b, b = f32 → a = f32
        unify(&Type::Var(a), &Type::Var(b), &mut s).unwrap();
        unify(&Type::Var(b), &Type::Prim(Prim::F32), &mut s).unwrap();
        assert_eq!(s.apply(&Type::Var(a)), Type::Prim(Prim::F32));
    }
}
