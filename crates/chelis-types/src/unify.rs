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

use chelis_unord::UnordMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::types::*;

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
    DimensionMismatch,
    ArityMismatch,
    OccursCheck,
    NotAFunction,
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
    types: Mutex<UnordMap<TypeVar, Type>>,
    /// Semantic domains attached to unresolved type variables. This is
    /// serialized with reusable checking contexts: a constrained function
    /// value must not become unconstrained after a cache round trip.
    #[serde(default)]
    tvar_restrictions: Mutex<UnordMap<TypeVar, TypeVarRestriction>>,
    dims: Mutex<UnordMap<DimVar, Dim>>,
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
    /// chelis#942: positional `expand` has two legal output shapes until a
    /// consumer fixes the result rank. The constraint stays attached to the
    /// unresolved result variable so ordinary unification can select either
    /// the same-rank replacement or rank-increasing insertion shape without
    /// letting an unrelated tensor shape slip through. Reusable library
    /// contexts serialize this obligation so a downstream checking unit can
    /// make the first shape-bearing choice after a cache round trip.
    deferred_expand_constraints: Mutex<DeferredExpandConstraints>,
    /// Element-count obligations imposed by `reshape` while a positional
    /// `expand` input is unresolved. The reshape output itself is immediately
    /// shape-bearing from its shape list (spec/04 §4.7.2-§4.7.3); only this
    /// relation to the input remains deferred.
    deferred_reshape_constraints: Mutex<DeferredReshapeConstraints>,
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

/// The two-shape obligation carried by an unresolved positional `expand`
/// result (spec/04-type-system.md §4.7.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferredExpandConstraint {
    pub input_dims: Vec<Dim>,
    pub input_prec: TensorPrec,
    pub axis: usize,
    pub size: Dim,
}

/// A deferred shape obligation attached to an unresolved result.
///
/// One ledger carries both kinds so they share the source ordinals, the alias
/// merge, the settlement order, and the serialized form that chelis#1341's
/// Phase A already established.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum DeferredShapeObligation {
    /// The positional-`expand` two-shape choice of §4.7.2.
    PositionalExpand(DeferredExpandConstraint),
    /// A comparison result mirrors its operand's shape at `bool`
    /// (`spec/05-risc-primitives.md` [05-OP-36] makes the operand and result
    /// shapes one equation). The mirror supplies no evidence back to
    /// `source`; a shape independently supplied to the result fixes it.
    ShapeMirror { source: TypeVar, prec: TensorPrec },
}

impl DeferredShapeObligation {
    fn positional_expand(&self) -> Option<&DeferredExpandConstraint> {
        match self {
            Self::PositionalExpand(constraint) => Some(constraint),
            Self::ShapeMirror { .. } => None,
        }
    }
}

/// Preorder position in the canonical Deep program. This is deliberately
/// distinct from `TypeVar`: inference allocates type variables in callee-first
/// dependency order, while positional-expand settlement is source ordered.
mod deferred_order {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    pub(crate) struct SourceOrdinal(u64);

    impl SourceOrdinal {
        pub(crate) const fn new(value: u64) -> Self {
            Self(value)
        }

        const fn get(self) -> u64 {
            self.0
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub(super) enum DeferredSourcePosition {
        Program(SourceOrdinal),
        Imported {
            first_reference: Option<SourceOrdinal>,
            library_ordinal: u64,
        },
    }

    impl DeferredSourcePosition {
        fn sort_key(self) -> (u8, u64, u64) {
            match self {
                Self::Program(ordinal) => (0, ordinal.get(), 0),
                Self::Imported {
                    first_reference: Some(ordinal),
                    library_ordinal,
                } => (0, ordinal.get(), library_ordinal.saturating_add(1)),
                Self::Imported {
                    first_reference: None,
                    library_ordinal,
                } => (1, library_ordinal, 0),
            }
        }

        /// Rebase a completed checking unit into a reusable library layer.
        /// Existing imports belong to earlier composed layers and therefore
        /// precede every obligation authored by the unit being completed.
        /// First-reference positions are deliberately ignored here: they are
        /// local to the check that just ended and will be recorded anew by the
        /// next downstream consumer.
        fn composition_key(self) -> (u8, u64) {
            match self {
                Self::Imported {
                    library_ordinal, ..
                } => (0, library_ordinal),
                Self::Program(ordinal) => (1, ordinal.get()),
            }
        }

        fn note_reference(&mut self, at: SourceOrdinal) {
            if let Self::Imported {
                first_reference, ..
            } = self
            {
                // Dependency inference can visit a later textual reference first.
                // The precomputed ordinal, rather than visit order, owns this choice.
                *first_reference = Some(first_reference.map_or(at, |current| current.min(at)));
            }
        }

        #[cfg(test)]
        pub(super) fn observed_ordinal(self) -> u64 {
            match self {
                Self::Program(ordinal) => ordinal.get(),
                Self::Imported {
                    first_reference: Some(ordinal),
                    ..
                } => ordinal.get(),
                Self::Imported {
                    first_reference: None,
                    library_ordinal,
                } => library_ordinal,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub(super) struct Sourced<T> {
        pub(super) position: DeferredSourcePosition,
        pub(super) obligation: T,
    }

    impl<T> Sourced<T> {
        fn program(ordinal: SourceOrdinal, obligation: T) -> Self {
            Self {
                position: DeferredSourcePosition::Program(ordinal),
                obligation,
            }
        }
    }

    fn prepare_store_for_downstream<T>(store: &mut UnordMap<TypeVar, Vec<Sourced<T>>>) {
        let mut positions = store
            .to_sorted()
            .into_iter()
            .flat_map(|(var, constraints)| {
                let var_id = var.0;
                constraints
                    .iter()
                    .enumerate()
                    .map(move |(constraint_index, entry)| {
                        (
                            entry.position.composition_key(),
                            var_id,
                            constraint_index,
                            *var,
                        )
                    })
            })
            .collect::<Vec<_>>();
        positions.sort_by_key(|(position, var_id, constraint_index, _)| {
            (*position, *var_id, *constraint_index)
        });
        for (library_ordinal, (_, _, constraint_index, var)) in positions.into_iter().enumerate() {
            let entry = &mut store
                .get_mut(&var)
                .expect("recorded deferred variable remains present")[constraint_index];
            entry.position = DeferredSourcePosition::Imported {
                first_reference: None,
                library_ordinal: library_ordinal as u64,
            };
        }
        let vars = store
            .to_sorted()
            .into_iter()
            .map(|(var, _)| *var)
            .collect::<Vec<_>>();
        for var in vars {
            let constraints = store
                .get_mut(&var)
                .expect("collected deferred variable remains present");
            constraints.sort_by_key(|entry| entry.position.sort_key());
        }
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    #[serde(transparent)]
    pub(super) struct DeferredExpandConstraints(
        UnordMap<TypeVar, Vec<Sourced<DeferredShapeObligation>>>,
    );

    impl DeferredExpandConstraints {
        pub(super) fn insert(
            &mut self,
            var: TypeVar,
            ordinal: SourceOrdinal,
            obligation: DeferredShapeObligation,
        ) {
            let obligations = self.0.entry(var).or_default();
            obligations.push(Sourced::program(ordinal, obligation));
            obligations.sort_by_key(|entry| entry.position.sort_key());
        }

        pub(super) fn contains(&self, var: TypeVar) -> bool {
            self.0.contains_key(&var)
        }

        /// Whether `var` still owes a positional-`expand` choice, as opposed
        /// to owing only a mirror of some other result's choice.
        pub(super) fn owes_positional_expand(&self, var: TypeVar) -> bool {
            self.0.get(&var).is_some_and(|obligations| {
                obligations
                    .iter()
                    .any(|entry| entry.obligation.positional_expand().is_some())
            })
        }

        pub(super) fn get(&self, var: TypeVar) -> Option<Vec<Sourced<DeferredShapeObligation>>> {
            self.0.get(&var).cloned()
        }

        pub(super) fn take(
            &mut self,
            var: TypeVar,
        ) -> Option<Vec<Sourced<DeferredShapeObligation>>> {
            self.0.remove(&var)
        }

        pub(super) fn merge_alias(&mut self, from: TypeVar, to: TypeVar) {
            let Some(mut moved) = self.0.remove(&from) else {
                return;
            };
            let target = self.0.entry(to).or_default();
            target.append(&mut moved);
            target.sort_by_key(|entry| entry.position.sort_key());
        }

        pub(super) fn settlement_order(&self) -> Vec<TypeVar> {
            let mut vars = self
                .0
                .to_sorted()
                .into_iter()
                .filter_map(|(var, constraints)| {
                    constraints
                        .first()
                        .map(|first| (first.position.sort_key(), var.0, *var))
                })
                .collect::<Vec<_>>();
            vars.sort_by_key(|entry| (entry.0, entry.1));
            vars.into_iter().map(|(_, _, var)| var).collect()
        }

        pub(super) fn prepare_for_downstream(&mut self) {
            prepare_store_for_downstream(&mut self.0);
        }

        pub(super) fn note_reference(&mut self, vars: &[TypeVar], at: SourceOrdinal) {
            for var in vars {
                let Some(constraints) = self.0.get_mut(var) else {
                    continue;
                };
                for entry in &mut *constraints {
                    entry.position.note_reference(at);
                }
                constraints.sort_by_key(|entry| entry.position.sort_key());
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub(super) struct DeferredReshapeConstraint {
        pub(super) target_dims: Vec<Dim>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    #[serde(transparent)]
    pub(super) struct DeferredReshapeConstraints(
        UnordMap<TypeVar, Vec<Sourced<DeferredReshapeConstraint>>>,
    );

    impl DeferredReshapeConstraints {
        pub(super) fn insert(
            &mut self,
            var: TypeVar,
            ordinal: SourceOrdinal,
            target_dims: Vec<Dim>,
        ) {
            let obligations = self.0.entry(var).or_default();
            obligations.push(Sourced::program(
                ordinal,
                DeferredReshapeConstraint { target_dims },
            ));
            obligations.sort_by_key(|entry| entry.position.sort_key());
        }

        pub(super) fn contains(&self, var: TypeVar) -> bool {
            self.0.contains_key(&var)
        }

        #[cfg(test)]
        pub(super) fn get(&self, var: TypeVar) -> Vec<Sourced<DeferredReshapeConstraint>> {
            self.0.get(&var).cloned().unwrap_or_default()
        }

        pub(super) fn take(&mut self, var: TypeVar) -> Vec<Sourced<DeferredReshapeConstraint>> {
            self.0.remove(&var).unwrap_or_default()
        }

        pub(super) fn merge_alias(&mut self, from: TypeVar, to: TypeVar) {
            let Some(mut moved) = self.0.remove(&from) else {
                return;
            };
            let target = self.0.entry(to).or_default();
            target.append(&mut moved);
            target.sort_by_key(|entry| entry.position.sort_key());
        }

        pub(super) fn prepare_for_downstream(&mut self) {
            prepare_store_for_downstream(&mut self.0);
        }

        pub(super) fn note_reference(&mut self, vars: &[TypeVar], at: SourceOrdinal) {
            for var in vars {
                let Some(constraints) = self.0.get_mut(var) else {
                    continue;
                };
                for entry in &mut *constraints {
                    entry.position.note_reference(at);
                }
                constraints.sort_by_key(|entry| entry.position.sort_key());
            }
        }
    }
}

pub(crate) use deferred_order::SourceOrdinal;
use deferred_order::{
    DeferredExpandConstraints, DeferredReshapeConstraint, DeferredReshapeConstraints, Sourced,
};

/// Independently resolved shape evidence a rule supplies about a deferred
/// positional-`expand` result (`spec/04-type-system.md` §4.7.2).
#[derive(Debug, Clone, Copy)]
pub(crate) enum ShapeEvidence<'a> {
    /// A formed expected tensor type: a declared result, an ascription, an
    /// already-instantiated parameter or generic field, a branch join with
    /// independently resolved shape evidence, or a builtin relation with an
    /// independently resolved operand equation.
    Type(&'a Type),
    /// A `reshape` target list. §4.7.3's element-count relation keeps only
    /// the candidate forms that list admits, and the ordinal records where
    /// the relation was written for source-ordered settlement.
    ElementCount {
        target_dims: &'a [Dim],
        ordinal: SourceOrdinal,
    },
}

/// The disposition an inference rule that consumes a tensor takes on a
/// deferred positional-`expand` result. Exactly one applies; C3 of
/// `spec/design/runtime_extents.md` states which context takes which.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DeferralAction<'a> {
    /// The context supplies an independently fixed rank or shape equation,
    /// so it selects the unique candidate satisfying that equation.
    Constrain(ShapeEvidence<'a>),
    /// The context carries the same unresolved monomorphic candidate without
    /// requiring either rank. It adds no evidence and cannot select or clone
    /// the choice, so the obligation survives unchanged.
    Propagate,
    /// A freeze point §4.7.2 names: a concrete tensor type is required and
    /// no independent constraint selected a candidate.
    Freeze,
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
            types: Mutex::new(self.types.lock().expect("subst.types poisoned").clone()),
            tvar_restrictions: Mutex::new(
                self.tvar_restrictions
                    .lock()
                    .expect("subst.tvar_restrictions poisoned")
                    .clone(),
            ),
            dims: Mutex::new(self.dims.lock().expect("subst.dims poisoned").clone()),
            ranks: Mutex::new(self.ranks.lock().expect("subst.ranks poisoned").clone()),
            deferred_borrow_vars: Mutex::new(
                self.deferred_borrow_vars
                    .lock()
                    .expect("subst.deferred_borrow_vars poisoned")
                    .clone(),
            ),
            deferred_opaque_uses: Mutex::new(
                self.deferred_opaque_uses
                    .lock()
                    .expect("subst.deferred_opaque_uses poisoned")
                    .clone(),
            ),
            deferred_expand_constraints: Mutex::new(
                self.deferred_expand_constraints
                    .lock()
                    .expect("subst.deferred_expand_constraints poisoned")
                    .clone(),
            ),
            deferred_reshape_constraints: Mutex::new(
                self.deferred_reshape_constraints
                    .lock()
                    .expect("subst.deferred_reshape_constraints poisoned")
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Compile-fail-only opaque view of the real expand store. The fixture
    /// proves that enabling its inspection seam still exposes no raw map API.
    #[cfg(feature = "hash-order-compile-probe")]
    pub fn deferred_expand_store_compile_probe(&self) -> impl std::fmt::Debug + '_ {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
    }

    /// Compile-fail-only opaque view of the real reshape store.
    #[cfg(feature = "hash-order-compile-probe")]
    pub fn deferred_reshape_store_compile_probe(&self) -> impl std::fmt::Debug + '_ {
        self.deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
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
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .prepare_for_downstream();
        self.deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .prepare_for_downstream();
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

    /// Attach a positional-expand obligation to its unresolved result.
    pub(crate) fn record_deferred_expand_constraint(
        &self,
        v: TypeVar,
        ordinal: SourceOrdinal,
        constraint: DeferredExpandConstraint,
    ) {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .insert(
                v,
                ordinal,
                DeferredShapeObligation::PositionalExpand(constraint),
            );
    }

    /// Attach a comparison result's shape mirror to its own result variable.
    ///
    /// The mirror is recorded only while `source` still owes a choice. A
    /// comparison over resolved operands computes its result directly.
    pub(crate) fn record_deferred_shape_mirror(
        &self,
        v: TypeVar,
        ordinal: SourceOrdinal,
        source: TypeVar,
        prec: TensorPrec,
    ) {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .insert(
                v,
                ordinal,
                DeferredShapeObligation::ShapeMirror { source, prec },
            );
    }

    /// The tensor precision an unresolved result will settle at, read off the
    /// obligation that will decide it. A mirror settles at `bool`.
    pub(crate) fn deferred_tensor_precision(&self, v: TypeVar) -> Option<TensorPrec> {
        let obligations = self
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .get(v)?;
        let expand_precision = obligations
            .iter()
            .find_map(|entry| entry.obligation.positional_expand())
            .map(|constraint| constraint.input_prec.clone());
        // A result owing only a mirror settles at `bool`, because that is what
        // the comparison whose shape it mirrors produces.
        Some(expand_precision.unwrap_or(TensorPrec::Concrete(Prim::Bool)))
    }

    /// Whether `v` is an unresolved output carrying a positional-expand
    /// obligation. Generalization uses this to keep one produced value
    /// monomorphic: two consumers may not independently choose two shapes for
    /// the same binding.
    pub(crate) fn has_deferred_expand_constraint(&self, v: TypeVar) -> bool {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .contains(v)
    }

    /// Whether `v` still owes a positional-`expand` choice of its own. A
    /// result that only mirrors another's choice settles after every expand
    /// has, so the freeze loop separates the two.
    fn owes_positional_expand(&self, v: TypeVar) -> bool {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .owes_positional_expand(v)
    }

    /// Whether `v` carries any deferred shape relation and must therefore
    /// remain monomorphic until that relation is selected.
    pub fn has_deferred_shape_constraint(&self, v: TypeVar) -> bool {
        self.has_deferred_expand_constraint(v)
            || self
                .deferred_reshape_constraints
                .lock()
                .expect("subst.deferred_reshape_constraints poisoned")
                .contains(v)
    }

    /// The one executor for the three C3 actions. Every inference rule that
    /// consumes a tensor names its action here, so a rule that consumes one
    /// without a disposition is visible rather than a silent fall-through.
    ///
    /// `Ok(None)` means `v` carried no positional-`expand` obligation, or the
    /// action deliberately left the choice open; `Ok(Some(ty))` is the type
    /// the action settled on.
    pub(crate) fn settle_deferred_tensor(
        &mut self,
        v: TypeVar,
        action: DeferralAction<'_>,
    ) -> Result<Option<Type>, TypeError> {
        match action {
            DeferralAction::Constrain(ShapeEvidence::Type(expected)) => {
                self.constrain_deferred_tensor(v, expected)
            }
            DeferralAction::Constrain(ShapeEvidence::ElementCount {
                target_dims,
                ordinal,
            }) => self.resolve_deferred_expand_for_reshape(v, ordinal, target_dims.to_vec()),
            // Propagation inspects nothing and mutates nothing: the obligation
            // stays exactly where it was. The call exists so a reader can see
            // that the rule chose this action rather than falling through
            // undecided.
            DeferralAction::Propagate => Ok(None),
            DeferralAction::Freeze => self.materialize_deferred_expand_default(v),
        }
    }

    /// Select the candidate an expected tensor type admits, as one
    /// transaction across both deferred ledgers.
    ///
    /// The trial clone is required: a rejected candidate may already have
    /// bound dimension variables or lowered levels before a later obligation
    /// fails, so no part of a rejected trial may leak into `self`.
    fn constrain_deferred_tensor(
        &mut self,
        v: TypeVar,
        expected: &Type,
    ) -> Result<Option<Type>, TypeError> {
        if matches!(expected, Type::Var(_) | Type::Error(_)) {
            // A variable supplies no independent equation, and a witnessed
            // error already owns its diagnostic.
            return Ok(None);
        }
        let has_expand = self.has_deferred_expand_constraint(v);
        let has_reshape = self
            .deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .contains(v);
        if !has_expand && !has_reshape {
            return Ok(None);
        }
        let mut trial = self.clone();
        let expand_constraints = trial
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .take(v)
            .unwrap_or_default();
        for entry in expand_constraints {
            match entry.obligation {
                DeferredShapeObligation::PositionalExpand(constraint) => {
                    let canonical = constraint.canonical_for_output(expected)?;
                    unify(&canonical, expected, &mut trial)?;
                }
                DeferredShapeObligation::ShapeMirror { source, prec } => {
                    // [05-OP-36]: the operand shape, the result shape, and the
                    // `bool` result dtype are one equation, so a shape supplied
                    // to the result fixes the operand.
                    let Type::Tensor(dims, _) = expected else {
                        return Err(TypeError {
                            kind: TypeErrorKind::TypeMismatch,
                            message: format!(
                                "a comparison over a tensor operand produces a tensor result, \
                                 got {expected}"
                            ),
                        });
                    };
                    unify(
                        expected,
                        &Type::Tensor(dims.clone(), TensorPrec::Concrete(Prim::Bool)),
                        &mut trial,
                    )?;
                    unify(
                        &Type::Var(source),
                        &Type::Tensor(dims.clone(), prec.clone()),
                        &mut trial,
                    )?;
                }
            }
        }
        let reshape_constraints = trial
            .deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .take(v);
        trial.validate_deferred_reshape_constraints(expected, &reshape_constraints)?;
        bind_tvar(v, expected, &mut trial)?;
        *self = trial;
        Ok(Some(self.apply(expected)))
    }

    /// Materialize the context-free positional-expand default for `v`.
    /// Existing-axis calls choose same-rank replacement; `axis == rank` has
    /// no replacement form and therefore chooses trailing insertion. Every
    /// obligation attached through alias unification must agree.
    fn materialize_deferred_expand_default(
        &mut self,
        v: TypeVar,
    ) -> Result<Option<Type>, TypeError> {
        let constraints = self
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .get(v);
        let Some(constraints) = constraints else {
            return Ok(None);
        };

        // Aliasing through an ordinary consumer can attach more than one
        // expand obligation to the same result variable. In that case the
        // consumer has imposed equality between the results, so choose the
        // first legal shape of the first producer that satisfies every
        // obligation. Candidate order preserves the context-free rule:
        // same-rank replacement precedes insertion when both are legal.
        // Probe on a clone because a failed tensor unification may have
        // already bound dimension variables before discovering a later
        // mismatch; rejected candidates must not mutate the real state.
        let Some(earliest) = constraints
            .iter()
            .find_map(|entry| entry.obligation.positional_expand())
        else {
            // The result owes no choice of its own, only a mirror of one.
            return self.settle_shape_mirror(v);
        };
        let mut first_rejection = None;
        for candidate in earliest.candidate_types()? {
            let mut trial = self.clone();
            match unify(&Type::Var(v), &candidate, &mut trial) {
                Ok(()) => {
                    *self = trial;
                    return Ok(Some(self.apply(&candidate)));
                }
                Err(error) => {
                    first_rejection.get_or_insert(error);
                }
            }
        }
        Err(first_rejection.expect("deferred expand constraint set has at least one candidate"))
    }

    /// Settle a result that only mirrors another result's shape.
    ///
    /// A mirror adds no evidence, so it has nothing to select: it reads the
    /// shape its source settled on and reproduces it at `bool`. The freeze
    /// loop runs this only after every positional-`expand` obligation has
    /// settled, so an open source here is an internal ordering defect rather
    /// than a program error.
    fn settle_shape_mirror(&mut self, v: TypeVar) -> Result<Option<Type>, TypeError> {
        let obligations = self
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .get(v);
        let Some(obligations) = obligations else {
            return Ok(None);
        };
        let mut settled = None;
        for entry in &obligations {
            let DeferredShapeObligation::ShapeMirror { source, .. } = entry.obligation else {
                continue;
            };
            let Type::Tensor(dims, _) = self.apply(&Type::Var(source)) else {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch,
                    message: format!(
                        "internal: a comparison result's shape mirror settled before its \
                         operand ?{} did",
                        source.0
                    ),
                });
            };
            settled = Some(Type::Tensor(dims, TensorPrec::Concrete(Prim::Bool)));
        }
        let Some(result) = settled else {
            return Ok(None);
        };
        unify(&Type::Var(v), &result, self)?;
        Ok(Some(self.apply(&result)))
    }

    /// Publish a `reshape` result immediately while retaining only its
    /// element-count relation to an unresolved positional-expand input.
    fn resolve_deferred_expand_for_reshape(
        &mut self,
        input_var: TypeVar,
        ordinal: SourceOrdinal,
        target_dims: Vec<Dim>,
    ) -> Result<Option<Type>, TypeError> {
        let constraints = self
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .get(input_var);
        let Some(constraints) = constraints else {
            return Ok(None);
        };

        let Some(earliest) = constraints
            .iter()
            .find_map(|entry| entry.obligation.positional_expand())
        else {
            return Ok(None);
        };
        let precision = earliest.input_prec.clone();
        let mut first_rejection = None;
        let mut compatible_candidates = Vec::new();
        for candidate in earliest.candidate_types()? {
            let mut trial = self.clone();
            trial.record_deferred_reshape_constraint(input_var, ordinal, target_dims.clone());
            match unify(&Type::Var(input_var), &candidate, &mut trial) {
                Ok(()) => compatible_candidates.push((candidate, trial)),
                Err(error) => {
                    first_rejection.get_or_insert(error);
                }
            }
        }
        match compatible_candidates.len() {
            1 => {
                let (_, trial) = compatible_candidates
                    .pop()
                    .expect("one compatible candidate");
                *self = trial;
            }
            0 => {
                return Err(first_rejection
                    .expect("every deferred expand candidate was rejected by reshape"));
            }
            _ => self.record_deferred_reshape_constraint(input_var, ordinal, target_dims.clone()),
        }
        Ok(Some(self.apply(&Type::Tensor(target_dims, precision))))
    }

    fn record_deferred_reshape_constraint(
        &self,
        input_var: TypeVar,
        ordinal: SourceOrdinal,
        target_dims: Vec<Dim>,
    ) {
        self.deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .insert(input_var, ordinal, target_dims);
    }

    pub(crate) fn static_dim_product(&self, dims: &[Dim]) -> Option<i128> {
        dims.iter().try_fold(1_i128, |product, dim| {
            let Dim::Lit(value) = self.apply_dim(dim) else {
                return None;
            };
            product.checked_mul(i128::from(value))
        })
    }

    /// Compare two fully-static dimension products without a fixed-width
    /// multiplication. Pairwise GCD cancellation is exact for any number of
    /// i64 factors, so a large known shape never degrades into "unknown".
    pub(crate) fn static_dim_products_match(&self, lhs: &[Dim], rhs: &[Dim]) -> Option<bool> {
        fn values(subst: &Subst, dims: &[Dim]) -> Option<Vec<i64>> {
            dims.iter()
                .map(|dim| match subst.apply_dim(dim) {
                    Dim::Lit(value) => Some(value),
                    _ => None,
                })
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

    #[cfg(test)]
    fn deferred_reshape_constraints_for_input(
        &self,
        input_var: TypeVar,
    ) -> Vec<Sourced<DeferredReshapeConstraint>> {
        self.deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .get(input_var)
    }

    fn validate_deferred_reshape_constraints(
        &self,
        input: &Type,
        constraints: &[Sourced<DeferredReshapeConstraint>],
    ) -> Result<(), TypeError> {
        let Type::Tensor(input_dims, _) = self.apply(input) else {
            return Err(TypeError {
                kind: TypeErrorKind::TypeMismatch,
                message: format!("reshape expects tensor input, got {}", self.apply(input)),
            });
        };
        for constraint in constraints {
            if self.static_dim_products_match(&input_dims, &constraint.obligation.target_dims)
                == Some(false)
            {
                let input_numel = self.static_dim_product(&input_dims);
                let target_numel = self.static_dim_product(&constraint.obligation.target_dims);
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: match (target_numel, input_numel) {
                        (Some(target), Some(input)) => format!(
                            "reshape target has {target} elements but input tensor has {input}"
                        ),
                        _ => "reshape target element count does not match input tensor".to_string(),
                    },
                });
            }
        }
        Ok(())
    }

    /// Resolve every positional-expand result whose shape remained
    /// unconstrained through the complete inference schedule. Consumers must
    /// get the first opportunity to select either legal shape; only the
    /// program freeze point invokes this fallback so no unresolved result can
    /// escape into checked annotations.
    pub(crate) fn materialize_deferred_expand_defaults(&mut self) -> Result<(), TypeError> {
        let vars = self
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .settlement_order();
        for var in vars {
            if self.owes_positional_expand(var) {
                self.settle_deferred_tensor(var, DeferralAction::Freeze)
                    .map_err(|error| TypeError {
                        kind: error.kind,
                        message: format!(
                            "positional expand defaults could not settle in source order at the \
                             program freeze point; declare the result tensor shape to select a \
                             compatible form. Underlying mismatch: {}",
                            error.message
                        ),
                    })?;
            }
        }
        // A result that only mirrors another's shape settles afterwards. It
        // adds no evidence to its source, so resolving every mirror once every
        // expand has settled cannot change an expand verdict, and it is the
        // only order in which the source is guaranteed concrete: source
        // ordinals are a preorder index, which puts an enclosing comparison
        // ahead of the `expand` nested inside its own argument.
        let mirrors = self
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .settlement_order();
        for var in mirrors {
            self.settle_deferred_tensor(var, DeferralAction::Freeze)?;
        }
        Ok(())
    }

    pub(crate) fn note_deferred_reference(&self, ty: &Type, at: SourceOrdinal) {
        let vars = crate::env::free_tvars(&self.apply(ty));
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .note_reference(&vars, at);
        self.deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .note_reference(&vars, at);
    }

    #[cfg(test)]
    fn deferred_expand_settlement_order_for_test(&self) -> Vec<TypeVar> {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .settlement_order()
    }

    #[cfg(test)]
    fn deferred_expand_ordinals_for_test(&self, var: TypeVar) -> Vec<u64> {
        self.deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .get(var)
            .unwrap_or_default()
            .into_iter()
            .map(|entry| entry.position.observed_ordinal())
            .collect()
    }

    #[cfg(test)]
    fn deferred_reshape_ordinals_for_test(&self, var: TypeVar) -> Vec<u64> {
        self.deferred_reshape_constraints_for_input(var)
            .into_iter()
            .map(|entry| entry.position.observed_ordinal())
            .collect()
    }

    #[cfg(test)]
    fn deferred_reshape_target_dims_for_test(&self, var: TypeVar) -> Vec<Vec<Dim>> {
        self.deferred_reshape_constraints_for_input(var)
            .into_iter()
            .map(|entry| entry.obligation.target_dims)
            .collect()
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
        // Phase 2: compress — point every link in the chain directly at
        // the terminal value.
        if chain.len() > 1 {
            for v in chain {
                map.insert(v, terminal.clone());
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

    /// Apply this substitution to a type, resolving all bound variables.
    /// Path-compresses any chains of length ≥ 2 it encounters so future
    /// lookups land in O(1).
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => self.resolve_tvar(*v),
            Type::Fn(args, ret) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                let ret = Box::new(self.apply(ret));
                Type::Fn(args, ret)
            }
            Type::Ref(inner) => Type::Ref(Box::new(self.apply(inner))),
            Type::Tensor(dims, prec) => {
                // A `Dim::Rank` expands to the whole shape vector it is bound
                // to (or stays as the sole `Dim::Rank` while unbound).
                let mut out: Vec<Dim> = Vec::with_capacity(dims.len());
                for d in dims {
                    match d {
                        Dim::Rank(r) => {
                            for rd in self.resolve_rvar(*r) {
                                out.push(self.apply_dim(&rd));
                            }
                        }
                        _ => out.push(self.apply_dim(d)),
                    }
                }
                Type::Tensor(out, self.apply_tensor_prec(prec))
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
                let mut resolved_dims = Vec::with_capacity(dims.len());
                for dim in dims {
                    match dim {
                        Dim::Var(var) if quantified_dvars.contains(var) => {
                            resolved_dims.push(dim.clone());
                        }
                        Dim::Rank(var) if quantified_rvars.contains(var) => {
                            resolved_dims.push(dim.clone());
                        }
                        Dim::Rank(var) => {
                            for resolved in self.resolve_rvar_excluding(*var, quantified_rvars) {
                                resolved_dims
                                    .push(self.apply_dim_excluding(&resolved, quantified_dvars));
                            }
                        }
                        _ => resolved_dims.push(self.apply_dim_excluding(dim, quantified_dvars)),
                    }
                }
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
            Dim::Var(v) => self.resolve_dvar(*v),
            _ => dim.clone(),
        }
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
            ensure_tvar_restriction(restriction, &resolved)?;
            if let Type::Var(target) = resolved {
                trial.narrow_tvar_restriction(target, restriction)?;
            }
        }

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
                ensure_tvar_restriction(restriction, target)?;
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
                *val = other.apply_dim(val);
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
fn merge_tvar_restrictions(
    existing: TypeVarRestriction,
    incoming: TypeVarRestriction,
) -> Result<TypeVarRestriction, TypeError> {
    existing.intersect(incoming).ok_or_else(|| TypeError {
        kind: TypeErrorKind::PrecisionMismatch,
        message: format!(
            "dtype families `{}` and `{}` share no active dtype, so the type variables they bound cannot be the same type",
            existing.family_name(),
            incoming.family_name()
        ),
    })
}

impl DeferredExpandConstraint {
    fn canonical_for_output(&self, output: &Type) -> Result<Type, TypeError> {
        let Type::Tensor(out_dims, _) = output else {
            return Err(TypeError {
                kind: TypeErrorKind::TypeMismatch,
                message: format!("expand expects tensor output, got {output}"),
            });
        };

        let mut expected = self.input_dims.clone();
        if out_dims.len() == self.input_dims.len() + 1 {
            expected.insert(self.axis, self.size.clone());
        } else if out_dims.len() == self.input_dims.len() {
            if self.axis >= self.input_dims.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: format!(
                        "expand axis {} is out of bounds for rank {} tensor",
                        self.axis,
                        self.input_dims.len(),
                    ),
                });
            }
            expected[self.axis] = self.size.clone();
        } else {
            return Err(TypeError {
                kind: TypeErrorKind::DimensionMismatch,
                message: format!(
                    "expand output rank {} must equal input rank {} or {}",
                    out_dims.len(),
                    self.input_dims.len(),
                    self.input_dims.len() + 1,
                ),
            });
        }
        Ok(Type::Tensor(expected, self.input_prec.clone()))
    }

    fn default_type(&self) -> Result<Type, TypeError> {
        let mut dims = self.input_dims.clone();
        if self.axis == dims.len() {
            dims.insert(self.axis, self.size.clone());
        } else if self.axis < dims.len() {
            dims[self.axis] = self.size.clone();
        } else {
            return Err(TypeError {
                kind: TypeErrorKind::DimensionMismatch,
                message: format!(
                    "expand insert axis {} is out of bounds for rank {} tensor",
                    self.axis,
                    dims.len(),
                ),
            });
        }
        Ok(Type::Tensor(dims, self.input_prec.clone()))
    }

    /// Legal output candidates in context-free preference order. An existing
    /// positional axis defaults to replacement but may be selected as an
    /// insertion by an equal-shape consumer; a trailing axis has only the
    /// insertion form.
    fn candidate_types(&self) -> Result<Vec<Type>, TypeError> {
        let default = self.default_type()?;
        let mut candidates = vec![default];
        if self.axis < self.input_dims.len() {
            let mut inserted = self.input_dims.clone();
            inserted.insert(self.axis, self.size.clone());
            candidates.push(Type::Tensor(inserted, self.input_prec.clone()));
        }
        Ok(candidates)
    }
}

/// Unify two types, producing a substitution or a type error.
pub fn unify(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    let t1 = subst.apply(t1);
    let t2 = subst.apply(t2);

    match (&t1, &t2) {
        // Same type — trivially unified
        (Type::Prim(p1), Type::Prim(p2)) if p1 == p2 => Ok(()),
        (Type::Prim(p1), Type::Prim(p2)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!(
                "precision mismatch: expected {}, got {}",
                p1.name(),
                p2.name()
            ),
        }),

        (Type::Unit, Type::Unit) => Ok(()),

        (Type::Ref(inner1), Type::Ref(inner2)) => unify(inner1, inner2, subst),

        // Type variable binding
        (Type::Var(v), _) => bind_tvar(*v, &t2, subst),
        (_, Type::Var(v)) => bind_tvar(*v, &t1, subst),

        // Function types
        (Type::Fn(args1, ret1), Type::Fn(args2, ret2)) => {
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "function arity mismatch: expected {} args, got {}",
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
                (_, 0) => unify_row_against_ground(&d1, &d2, subst),
                (0, _) => unify_row_against_ground(&d2, &d1, subst),
                // Both carry spreads: only structurally-identical rows unify.
                (_, _) => unify_row_against_row(&d1, &d2, subst),
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
        // still emit a "body has type `<error>`, declared type is `T`"
        // diagnostic so the user sees the unresolved declared shape).
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
    let d1 = subst.apply_dim(d1);
    let d2 = subst.apply_dim(d2);

    match (&d1, &d2) {
        (Dim::Name(n1), Dim::Name(n2)) if n1 == n2 => Ok(()),
        (Dim::Lit(l1), Dim::Lit(l2)) if l1 == l2 => Ok(()),
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

fn bind_tvar(v: TypeVar, ty: &Type, subst: &mut Subst) -> Result<(), TypeError> {
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
        ensure_tvar_restriction(restriction, ty)?;
    }

    if let Type::Var(target) = ty {
        // Identifying two variables is the mechanism behind C3 `Propagate`:
        // the surviving variable carries both obligations, no evidence is
        // added, and neither choice is selected or cloned.
        subst
            .deferred_expand_constraints
            .lock()
            .expect("subst.deferred_expand_constraints poisoned")
            .merge_alias(v, *target);
        subst
            .deferred_reshape_constraints
            .lock()
            .expect("subst.deferred_reshape_constraints poisoned")
            .merge_alias(v, *target);
    } else if subst
        .settle_deferred_tensor(v, DeferralAction::Constrain(ShapeEvidence::Type(ty)))?
        .is_some()
    {
        // C3 `Constrain`: a concrete target is the independent shape
        // equation, and the executor committed the whole transaction.
        return Ok(());
    }
    // An older variable that becomes bound to a younger composite makes all
    // reachable variables part of the older scope. Deferred shape resolution
    // above commits its cloned transaction only after every candidate accepts,
    // so lowering here cannot expose a rejected candidate's level overrides.
    let target_level = subst.level_of_tvar(v);
    subst.lower_type_to(ty, target_level);
    subst.record_validated_type_binding(v, ty.clone());
    subst.remove_tvar_restriction(v);
    if let (Some(restriction), Type::Var(target)) = (merged_restriction, ty) {
        subst.narrow_tvar_restriction(*target, restriction)?;
    }
    Ok(())
}

/// Check one instantiation of a bounded type variable ([04-DTYPE-2]).
///
/// An unresolved variable and a witnessed error both pass: the first is
/// narrowed instead by [`Subst::narrow_tvar_restriction`], and the second
/// already owns a diagnostic.
fn ensure_tvar_restriction(restriction: TypeVarRestriction, ty: &Type) -> Result<(), TypeError> {
    let family = restriction.family_name();
    let gloss = restriction.membership_gloss();
    match ty {
        Type::Var(_) | Type::Error(_) => Ok(()),
        Type::Prim(prim) if restriction.admits(*prim) => Ok(()),
        Type::Prim(prim) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!(
                "type variable bounded by dtype family `{family}` ({gloss}) cannot be instantiated at `{}`",
                prim.name()
            ),
        }),
        other => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!(
                "type variable bounded by dtype family `{family}` ({gloss}) cannot be instantiated at `{other}`"
            ),
        }),
    }
}

fn bind_dvar(v: DimVar, dim: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
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
            if let Dim::Name(n) = &bound
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
                        let name = match &rest[0] {
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
                            .filter(|&j| matches!(&ground[j], Dim::Name(g) if g == name))
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
    use super::*;

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
    /// and that `int32` is consequently rejected.
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

            // And the narrowing is observable: int32 is in Numeric but not in
            // Float, so it must now be rejected.
            let error = unify(&Type::Var(surviving), &Type::Prim(Prim::Int32), &mut subst)
                .expect_err("a narrowed variable must reject a non-float dtype");
            assert!(
                matches!(error.kind, TypeErrorKind::PrecisionMismatch)
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
        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
        assert!(error.message.contains("active float dtype"));
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
        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
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
        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
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

        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
        assert!(error.message.contains("active float dtype"));
        assert!(error.message.contains("int32"));
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
        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
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
        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
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
            "f32 vs int32 precision must fail even with a rank var"
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
        // The current implementation satisfies this because the
        // `(Wildcard, _) | (_, Wildcard) => Ok(())` arm matches before
        // the `(Var(_), _) => bind_dvar(...)` arm (Rust `match` is
        // first-match-wins) and returns `Ok(())` without touching the
        // substitution. This test pins the property in case a future
        // refactor reorders the arms or adds an explicit
        // `(Wildcard, Var)` arm that does bind.
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
    fn deferred_expand_binds_unresolved_consumer_precision() {
        let mut g = var_gen();
        let result = g.fresh_tvar();
        let consumer_prec = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            result,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(2)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );

        let consumer = Type::Tensor(
            vec![Dim::Lit(3), Dim::Lit(2)],
            TensorPrec::Var(consumer_prec),
        );
        unify(&Type::Var(result), &consumer, &mut s)
            .expect("the expand input precision must bind the consumer precision variable");

        assert_eq!(s.apply(&Type::Var(consumer_prec)), Type::Prim(Prim::F32));
    }

    #[test]
    fn deferred_expand_rejects_concrete_consumer_precision_mismatch() {
        let mut g = var_gen();
        let result = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            result,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(2)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );

        let consumer = Type::Tensor(
            vec![Dim::Lit(3), Dim::Lit(2)],
            TensorPrec::Concrete(Prim::F64),
        );
        let error = unify(&Type::Var(result), &consumer, &mut s)
            .expect_err("a concrete f64 consumer must not accept an f32 expand result");
        assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn reshape_numel_selects_rank_increasing_deferred_expand() {
        let mut g = var_gen();
        let result = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            result,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(2)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );

        let output = s
            .settle_deferred_tensor(
                result,
                DeferralAction::Constrain(ShapeEvidence::ElementCount {
                    target_dims: &[Dim::Lit(6)],
                    ordinal: SourceOrdinal::new(1),
                }),
            )
            .expect("a six-element reshape target must select insertion")
            .expect("the result carries a deferred expand constraint");
        assert_eq!(
            output,
            Type::Tensor(vec![Dim::Lit(6)], TensorPrec::Concrete(Prim::F32))
        );
        assert_eq!(
            s.apply(&Type::Var(result)),
            Type::Tensor(
                vec![Dim::Lit(3), Dim::Lit(2)],
                TensorPrec::Concrete(Prim::F32)
            )
        );
    }

    #[test]
    fn reshape_numel_rejects_every_incompatible_deferred_expand_shape() {
        let mut g = var_gen();
        let result = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            result,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(2)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );

        let error = s
            .settle_deferred_tensor(
                result,
                DeferralAction::Constrain(ShapeEvidence::ElementCount {
                    target_dims: &[Dim::Lit(5)],
                    ordinal: SourceOrdinal::new(1),
                }),
            )
            .expect_err("five elements match neither tensor[3] nor tensor[3, 2]");
        assert!(matches!(error.kind, TypeErrorKind::DimensionMismatch));
        assert!(error.message.contains("reshape target has 5 elements"));
    }

    #[test]
    fn reshape_unknown_numel_publishes_output_and_leaves_input_unresolved() {
        let mut g = var_gen();
        let result = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            result,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(2)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );

        let output = s
            .settle_deferred_tensor(
                result,
                DeferralAction::Constrain(ShapeEvidence::ElementCount {
                    target_dims: &[Dim::Wildcard],
                    ordinal: SourceOrdinal::new(1),
                }),
            )
            .expect("an unknown reshape target is compatible with either expand shape")
            .expect("the reshape still has its own output type");
        assert_eq!(
            output,
            Type::Tensor(vec![Dim::Wildcard], TensorPrec::Concrete(Prim::F32))
        );
        assert!(s.has_deferred_expand_constraint(result));
        assert!(s.has_deferred_shape_constraint(result));
    }

    #[test]
    fn reshape_ambiguous_candidates_publish_declared_target_dims() {
        let mut g = var_gen();
        let result = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            result,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(2), Dim::Lit(4)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );

        let output = s
            .settle_deferred_tensor(
                result,
                DeferralAction::Constrain(ShapeEvidence::ElementCount {
                    target_dims: &[Dim::Lit(3), Dim::Wildcard],
                    ordinal: SourceOrdinal::new(1),
                }),
            )
            .expect("both expand candidates remain possible")
            .expect("the reshape still has its own output type");
        assert_eq!(
            output,
            Type::Tensor(
                vec![Dim::Lit(3), Dim::Wildcard],
                TensorPrec::Concrete(Prim::F32)
            )
        );
        assert!(s.has_deferred_expand_constraint(result));
        assert!(s.has_deferred_shape_constraint(result));
    }

    #[test]
    fn rejected_deferred_reshape_requirement_preserves_relation() {
        let mut g = var_gen();
        let input = g.fresh_tvar();
        let mut s = Subst::new();
        s.record_deferred_expand_constraint(
            input,
            SourceOrdinal::new(0),
            DeferredExpandConstraint {
                input_dims: vec![Dim::Lit(1), Dim::Lit(2), Dim::Lit(1)],
                input_prec: tprec(Prim::F32),
                axis: 0,
                size: Dim::Lit(3),
            },
        );
        s.settle_deferred_tensor(
            input,
            DeferralAction::Constrain(ShapeEvidence::ElementCount {
                target_dims: &[Dim::Wildcard, Dim::Wildcard, Dim::Lit(3)],
                ordinal: SourceOrdinal::new(1),
            }),
        )
        .expect("both expand candidates have legal reshape outputs")
        .expect("the input carries a deferred expand relation");

        let inner = s.enter_level(&g);
        let younger_precision = g.fresh_tvar();
        let before_level_state = (
            s.current_level,
            s.level_transitions.clone(),
            s.lowered_tvar_levels.clone(),
            s.lowered_dvar_levels.clone(),
            s.lowered_rvar_levels.clone(),
            s.resume_floors,
        );
        let incompatible = Type::Tensor(
            vec![Dim::Lit(5), Dim::Lit(1), Dim::Lit(1)],
            TensorPrec::Var(younger_precision),
        );
        unify(&Type::Var(input), &incompatible, &mut s)
            .expect_err("five elements violate the retained reshape relation");
        assert_eq!(
            (
                s.current_level,
                s.level_transitions.clone(),
                s.lowered_tvar_levels.clone(),
                s.lowered_dvar_levels.clone(),
                s.lowered_rvar_levels.clone(),
                s.resume_floors,
            ),
            before_level_state,
            "a rejected speculative shape candidate must not mutate level state"
        );
        assert!(
            s.has_deferred_shape_constraint(input),
            "a rejected consumer must not erase the deferred relation"
        );
        s.leave_level(inner, &g);
    }

    #[test]
    fn rejected_deferred_expand_bind_restores_level_state() {
        let mut g = var_gen();
        let output = g.fresh_tvar();
        let older_dim = g.fresh_dvar();
        let mut s = Subst::new();

        for (ordinal, prim) in [Prim::F32, Prim::F64].into_iter().enumerate() {
            s.record_deferred_expand_constraint(
                output,
                SourceOrdinal::new(ordinal as u64),
                DeferredExpandConstraint {
                    input_dims: vec![Dim::Var(older_dim)],
                    input_prec: tprec(prim),
                    axis: 1,
                    size: Dim::Lit(3),
                },
            );
        }

        let inner = s.enter_level(&g);
        let younger_dim = g.fresh_dvar();
        let younger_precision = g.fresh_tvar();
        let before_level_state = (
            s.current_level,
            s.level_transitions.clone(),
            s.lowered_tvar_levels.clone(),
            s.lowered_dvar_levels.clone(),
            s.lowered_rvar_levels.clone(),
            s.resume_floors,
        );
        let candidate = Type::Tensor(
            vec![Dim::Var(younger_dim), Dim::Lit(3)],
            TensorPrec::Var(younger_precision),
        );

        unify(&Type::Var(output), &candidate, &mut s)
            .expect_err("conflicting deferred precisions must reject");

        assert_eq!(
            (
                s.current_level,
                s.level_transitions.clone(),
                s.lowered_tvar_levels.clone(),
                s.lowered_dvar_levels.clone(),
                s.lowered_rvar_levels.clone(),
                s.resume_floors,
            ),
            before_level_state,
            "a rejected deferred expand bind must not leak solver-level lowering"
        );
        assert!(
            s.has_deferred_expand_constraint(output),
            "a rejected binding must preserve the deferred expand relation"
        );
        s.leave_level(inner, &g);
    }

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

    fn hash_order_constraint(size: i64) -> DeferredExpandConstraint {
        DeferredExpandConstraint {
            input_dims: vec![Dim::Lit(2)],
            input_prec: tprec(Prim::F32),
            axis: 0,
            size: Dim::Lit(size),
        }
    }

    #[test]
    fn hash_order_expand_store_uses_source_order_under_insertion_perturbation() {
        let insertion_orders = [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]];
        for insertion_order in insertion_orders {
            let mut g = var_gen();
            let vars = [g.fresh_tvar(), g.fresh_tvar(), g.fresh_tvar()];
            let ordinals = [7_u64, 2, 11];
            let subst = Subst::new();
            for index in insertion_order {
                subst.record_deferred_expand_constraint(
                    vars[index],
                    SourceOrdinal::new(ordinals[index]),
                    hash_order_constraint(3 + index as i64),
                );
            }
            assert_eq!(
                subst.deferred_expand_settlement_order_for_test(),
                vec![vars[1], vars[0], vars[2]],
                "fresh hash state and insertion order must not affect settlement"
            );
        }
    }

    #[test]
    fn hash_order_alias_merge_keeps_expand_obligations_in_source_order() {
        let mut g = var_gen();
        let later = g.fresh_tvar();
        let earlier = g.fresh_tvar();
        let mut subst = Subst::new();
        subst.record_deferred_expand_constraint(
            later,
            SourceOrdinal::new(20),
            hash_order_constraint(3),
        );
        subst.record_deferred_expand_constraint(
            earlier,
            SourceOrdinal::new(4),
            hash_order_constraint(3),
        );

        unify(&Type::Var(later), &Type::Var(earlier), &mut subst)
            .expect("aliasing two compatible deferred results must succeed");
        assert_eq!(
            subst.deferred_expand_ordinals_for_test(earlier),
            vec![4, 20]
        );
    }

    #[test]
    fn hash_order_reshape_obligations_use_reshape_source_order() {
        let mut g = var_gen();
        let input = g.fresh_tvar();
        let subst = Subst::new();
        subst.record_deferred_expand_constraint(
            input,
            SourceOrdinal::new(1),
            hash_order_constraint(3),
        );
        subst.record_deferred_reshape_constraint(
            input,
            SourceOrdinal::new(19),
            vec![Dim::Wildcard],
        );
        subst.record_deferred_reshape_constraint(input, SourceOrdinal::new(8), vec![Dim::Wildcard]);
        assert_eq!(subst.deferred_reshape_ordinals_for_test(input), vec![8, 19]);
    }

    #[test]
    fn hash_order_imported_obligation_rebases_to_first_reference() {
        let mut g = var_gen();
        let unreferenced = g.fresh_tvar();
        let referenced = g.fresh_tvar();
        let mut subst = Subst::new();
        subst.record_deferred_expand_constraint(
            unreferenced,
            SourceOrdinal::new(1),
            hash_order_constraint(3),
        );
        subst.record_deferred_expand_constraint(
            referenced,
            SourceOrdinal::new(9),
            hash_order_constraint(3),
        );
        subst.resume_for_new_check(&g);
        subst.note_deferred_reference(&Type::Var(referenced), SourceOrdinal::new(4));

        assert_eq!(
            subst.deferred_expand_settlement_order_for_test(),
            vec![referenced, unreferenced],
            "a referenced import takes its downstream first-reference position"
        );
    }

    #[test]
    fn hash_order_imported_obligations_keep_earliest_source_reference() {
        let mut g = var_gen();
        let expand = g.fresh_tvar();
        let reshape = g.fresh_tvar();
        let mut subst = Subst::new();
        subst.record_deferred_expand_constraint(
            expand,
            SourceOrdinal::new(1),
            hash_order_constraint(3),
        );
        subst.record_deferred_reshape_constraint(reshape, SourceOrdinal::new(2), vec![Dim::Lit(3)]);
        subst.resume_for_new_check(&g);

        subst.note_deferred_reference(&Type::Var(expand), SourceOrdinal::new(100));
        subst.note_deferred_reference(&Type::Var(expand), SourceOrdinal::new(10));
        subst.note_deferred_reference(&Type::Var(reshape), SourceOrdinal::new(100));
        subst.note_deferred_reference(&Type::Var(reshape), SourceOrdinal::new(10));

        assert_eq!(subst.deferred_expand_ordinals_for_test(expand), vec![10]);
        assert_eq!(subst.deferred_reshape_ordinals_for_test(reshape), vec![10]);
    }

    #[test]
    fn hash_order_unreferenced_imports_follow_program_results_in_library_order() {
        let mut g = var_gen();
        let imported_first = g.fresh_tvar();
        let imported_second = g.fresh_tvar();
        let local = g.fresh_tvar();
        let mut subst = Subst::new();
        subst.record_deferred_expand_constraint(
            imported_first,
            SourceOrdinal::new(2),
            hash_order_constraint(3),
        );
        subst.record_deferred_expand_constraint(
            imported_second,
            SourceOrdinal::new(5),
            hash_order_constraint(3),
        );
        subst.resume_for_new_check(&g);
        subst.record_deferred_expand_constraint(
            local,
            SourceOrdinal::new(100),
            hash_order_constraint(3),
        );

        assert_eq!(
            subst.deferred_expand_settlement_order_for_test(),
            vec![local, imported_first, imported_second]
        );
    }

    #[test]
    fn hash_order_composed_expand_contexts_keep_base_before_later_layer() {
        let mut g = var_gen();
        let base = g.fresh_tvar();
        let later = g.fresh_tvar();
        let mut subst = Subst::new();
        subst.record_deferred_expand_constraint(
            base,
            SourceOrdinal::new(8),
            hash_order_constraint(3),
        );
        subst.resume_for_new_check(&g);
        subst.record_deferred_expand_constraint(
            later,
            SourceOrdinal::new(1),
            hash_order_constraint(5),
        );

        assert_eq!(
            subst.deferred_expand_settlement_order_for_test(),
            vec![later, base],
            "program-local obligations precede unreferenced imports in the current layer"
        );

        subst.resume_for_new_check(&g);
        assert_eq!(
            subst.deferred_expand_settlement_order_for_test(),
            vec![base, later],
            "once composed, the base library must retain its order before the later layer"
        );
    }

    #[test]
    fn hash_order_composed_reshape_contexts_keep_base_before_later_layer() {
        let mut g = var_gen();
        let input = g.fresh_tvar();
        let mut subst = Subst::new();
        subst.record_deferred_reshape_constraint(input, SourceOrdinal::new(8), vec![Dim::Lit(3)]);
        subst.resume_for_new_check(&g);
        subst.record_deferred_reshape_constraint(input, SourceOrdinal::new(1), vec![Dim::Lit(7)]);
        subst.resume_for_new_check(&g);

        assert_eq!(
            subst.deferred_reshape_target_dims_for_test(input),
            vec![vec![Dim::Lit(3)], vec![Dim::Lit(7)]],
            "the base reshape relation must retain its order before the later layer"
        );
    }

    #[test]
    fn hash_order_serialization_preserves_import_order_and_reference_rebasing() {
        let mut g = var_gen();
        let first = g.fresh_tvar();
        let second = g.fresh_tvar();
        let subst = Subst::new();
        subst.record_deferred_expand_constraint(
            first,
            SourceOrdinal::new(3),
            hash_order_constraint(3),
        );
        subst.record_deferred_expand_constraint(
            second,
            SourceOrdinal::new(12),
            hash_order_constraint(3),
        );
        let bytes = bincode::serialize(&subst).expect("serialize substitution");
        let mut restored: Subst = bincode::deserialize(&bytes).expect("deserialize substitution");
        restored.resume_for_new_check(&g);
        restored.note_deferred_reference(&Type::Var(second), SourceOrdinal::new(1));

        assert_eq!(
            restored.deferred_expand_settlement_order_for_test(),
            vec![second, first]
        );
    }
}
