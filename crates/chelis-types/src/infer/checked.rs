//! Checked-program construction, metadata ownership, and totality finalization.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use crate::context::LibraryProofId;

mod deferred_shape;
mod result_origin;

// Counts real constraint examinations for the chelis#2975 scaling oracle.
// A typed-node count misses repeated work over one retained result graph.
// Slots: equation collection, binding producer scan, join replay,
// annotation replay, general producer scan.
#[cfg(test)]
thread_local! {
    static RESULT_REPLAY_WORK: Cell<[usize; 5]> = const { Cell::new([0; 5]) };
}

#[cfg(test)]
pub(super) fn record_result_replay_work(kind: usize) {
    RESULT_REPLAY_WORK.with(|work| {
        let mut counts = work.get();
        counts[kind] += 1;
        work.set(counts);
    });
}

#[cfg(test)]
pub(super) fn take_result_replay_work() -> [usize; 5] {
    RESULT_REPLAY_WORK.with(|work| work.replace([0; 5]))
}

#[cfg(test)]
use deferred_shape::reconcile_replayed_result;
pub(super) use deferred_shape::shape_operand_awaits_binding;

#[derive(Clone)]
pub(super) struct DeclaredSigMetadata {
    pub(super) param_types: Vec<deep::Expr>,
    pub(super) binders: UnordSet<String>,
    /// Exact dtype-family capabilities admitted by the typed annotation boundary.
    pub(super) dtype_bounds: UnordMap<String, chelis_deep::DtypeBound>,
}

/// One checker operation's typed inference result. The product is private,
/// session-local, and never serialized: annotation consumes it immediately
/// after the owning inference traversal completes.
#[derive(Default)]
pub(super) struct InferenceProduct {
    pub(super) typed_nodes: usize,
    pub(super) total_nodes: usize,
    pub(super) next_epoch: u64,
    pub(super) active_epoch: Option<TypeStampEpoch>,
    pub(super) owner_types: UnordMap<usize, FinalOwnerType>,
    pub(super) type_headers: TypeResolutionEnv,
    pub(super) adt_registry: AdtRegistry,
    pub(super) function_inference_plan: FunctionInferencePlan,
    /// The canonical declaration-reference graph built for this inference
    /// run. Scheduling and initialization-cycle diagnostics consume this same
    /// instance so the two policies cannot drift or repeat the lexical walk.
    pub(super) top_level_references: TopLevelReferenceGraph,
    /// chelis#1512: true while `replay_ready_shape_checks` is running. The
    /// `PostApp` replay re-enters `finish_unified_app`, whose own tail calls
    /// this function again; without the flag that recursion is unbounded.
    /// The outer pass runs to a fixpoint, so a nested call has nothing to add.
    replaying_shape_checks: bool,
    /// Constructor arguments form one result-origin region. Replay after
    /// the enclosing aggregate has published its equalities.
    pub(super) aggregate_arg_depth: usize,
    /// chelis#1512: while a `PostApp` entry is being replayed, the ledger's
    /// own CLONE of the call stands in for the live Deep node, and that clone
    /// is dropped at the end of the replay iteration. A route that
    /// re-registers during the replay would otherwise key its new entry by an
    /// address that is freed moments later. This pair carries the translation
    /// `(clone address, original site)`, and it is what makes
    /// [`Self::post_app_key`]'s documented lifetime invariant true rather
    /// than merely asserted.
    replaying_post_app: Option<(usize, usize)>,
    next_deferred_shape_id: u64,
    deferred_shape_checks: Vec<DeferredShapeCheck>,
    /// Result annotations check what inference produces. They are not input
    /// constraints for a suspended type derivation ([04-INF-1]). Keeping them
    /// separate lets readiness follow a producer whose constructor or nested
    /// parameter types are revealed later, rather than snapshotting its holes.
    result_type_constraints: Vec<ResultTypeConstraint>,
    result_inputs_closed: bool,
    closing_group_types: Vec<Type>,
    pub(super) group_result_origins: BTreeMap<String, Scheme>,
    /// [04-PAT-1]: a literal pattern first seen against a flexible type must
    /// be decided after its enclosing declaration has supplied the type.
    deferred_literal_patterns: Vec<DeferredLiteralPattern>,
    /// Newly authored parameter holes and call operand/result requirements,
    /// not contracts transported by function values. These retain their
    /// original variables until the declaration boundary.
    inferred_admission_contracts: Vec<InferredAdmissionContract>,
    pub(super) builtin_selections: Vec<crate::builtin_discovery::BuiltinCaseSelection>,
    pending_builtin_selections: Vec<usize>,
    /// chelis#1801: the fresh dimension variables scheme instantiation minted
    /// during this inference run, in mint order.
    ///
    /// `spec/04-type-system.md` section 3.2 scopes the runtime-extent
    /// absorption to the variables ONE application's instantiation minted,
    /// and the instantiation happens in the Var rule while the decision
    /// happens in the application rule. This log is how the first tells the
    /// second, and the pairing is positional rather than keyed: the
    /// application rule marks the log, infers its callee, and reads back
    /// exactly the entries that inference appended
    /// ([`Self::instantiation_dvar_mark`] /
    /// [`Self::instantiation_dvars_since`]).
    ///
    /// A ledger keyed by the callee's node address was the alternative and is
    /// worse here: the Var rule and the application rule would have to agree
    /// on which address names the callee, a second identity to keep in step
    /// with the positional pairing that already holds. A slot on `Subst` cleared by
    /// convention at each call site was the other, and that is the
    /// flag-by-convention shape chelis#1835 exists to remove.
    ///
    /// Append-only and never truncated. A nested callee (`(f(x))(y)`) is then
    /// covered by the outer mark as well as its own, which is the wanted
    /// reading: the outer call's callee type IS `f`'s result, so a variable of
    /// that result meeting `*` in the outer unification denotes the outer
    /// call's runtime extent.
    instantiation_dvars: Vec<DimVar>,
    /// [04-LIN-9] and spec/04 section 1.1: set by the application rule just
    /// before it infers a `var` callee and taken by the Var rule, which so
    /// tells a builtin called here from a builtin named as a value
    /// (`infer_var`).
    pub(super) callee_reference: bool,
    /// Explicit local tensor ascriptions, recorded independently from the
    /// ordinary inferred `type` metadata that annotation writes on every
    /// checked expression. Lowering consumes this checker-owned carrier at
    /// the matching `let` before aliases can erase the authored boundary.
    local_tensor_ascriptions: Vec<CheckedLocalTensorAscription>,
    /// Exact top-level declaration currently being inferred. This is source
    /// provenance for local ascriptions, not an inferred type fact: composed
    /// checked units may reuse the same byte offsets, so spans alone cannot
    /// identify the declaration that authored a binding.
    active_declaration_name: Option<String>,
    /// The authored-binder contracts not yet decided: the declaration being
    /// inferred, or every member of a recursive group inferred so far.
    /// Decided by the close after the last step that can narrow a binder
    /// (chelis#2537, chelis#2584).
    authored_binder_contracts: Vec<AuthoredBinderContract>,
    /// The declaration that owns the ledger entry being replayed, so an entry
    /// the replay registers again keeps its owner (chelis#2584).
    replaying_owner: Option<String>,
    /// chelis#2590: the in-group references to recursive-group members whose
    /// header omits a type, and each such member's own instance of its
    /// provisional scheme, decided when the component completes
    /// (`group_link::link_group_references`).
    group_references: Vec<GroupReference>,
    group_member_types: Vec<(String, Type)>,
    /// The references to a sibling member of the recursive group being
    /// inferred, each typed at a fresh copy of the member's provisional type
    /// and linked to it when the group completes (`group_link::SiblingLink`).
    sibling_links: Vec<SiblingLink>,
    /// chelis#2626: the name and provisional monomorphic type of each member
    /// of the recursive group being inferred whose declaration writes no
    /// signature. The member's own body and references share it; a sibling's
    /// reference copies it. Dropped when the group completes.
    group_provisional_types: Vec<(String, Type)>,
}

struct InferredAdmissionContract {
    /// The declaration whose inference recorded the contract.
    owner: Option<String>,
    subject: String,
    variable: TypeVar,
    /// The enclosing declaration's authored type binders. A local hole may
    /// resolve to one of these without requiring a concrete instantiation.
    binders: Vec<TypeVar>,
    failed_application: Option<ErrorWitness>,
}

impl InferredAdmissionContract {
    fn unmet_families(&self, subst: &Subst) -> Vec<(TypeVar, TypeVarRestriction)> {
        if self.failed_application.is_some() {
            return Vec::new();
        }
        crate::env::free_tvars(&subst.apply(&Type::Var(self.variable)))
            .into_iter()
            .filter_map(|variable| {
                let required = subst.tvar_restriction(variable)?;
                // A hole that resolved to an authored binder carries that
                // binder's requirement, which `check_declared_dtype_bounds`
                // decides against the declared bound. Deciding it here too
                // reported one defect twice (chelis#2158 round 1).
                let binder = self
                    .binders
                    .iter()
                    .any(|binder| subst.apply(&Type::Var(*binder)) == Type::Var(variable));
                (!binder).then_some((variable, required))
            })
            .collect()
    }
}

#[derive(Clone)]
struct ResultTypeConstraint {
    actual: Type,
    declared: Type,
}

#[derive(Clone)]
pub(super) enum DeferredShapeRule {
    /// Branch results agree after their semantic producers settle. The
    /// published type carries only structure common to both branches.
    ResultJoin {
        last_propagated: Option<Type>,
    },
    Matmul {
        location: Option<TypeDiagnosticLocation>,
    },
    Reduction {
        name: String,
        location: Option<TypeDiagnosticLocation>,
    },
    Expand {
        /// `expand` or `insert`. The replay must reach the same route arm the
        /// original call did, and the two differ in their result forms.
        builtin: &'static str,
        axis_is_dim_name: bool,
        env: Box<Env>,
        location: Option<TypeDiagnosticLocation>,
    },
    LayerNorm {
        location: Option<TypeDiagnosticLocation>,
    },
    Conv {
        location: Option<TypeDiagnosticLocation>,
    },
    ScatterElements {
        node: DeepNode,
    },
    /// chelis#1512: any other checked route that returned early on an
    /// unresolved operand. The replay re-enters `finish_unified_app` itself
    /// rather than a per-route decision function, so the deferred path is the
    /// eager path: there is no second implementation of any route's
    /// validation to disagree with the first.
    /// chelis#1512: an `app_shape` route suspended on an unresolved operand.
    /// The rule, not the inference entry point, is what replays.
    ShapeRoute {
        route: ShapeRouteKind,
        node: DeepNode,
        kids: Vec<deep::Expr>,
        location: Option<TypeDiagnosticLocation>,
    },
    /// A tuple projection or record-field read whose target was still a
    /// type variable; `arg_tys` is `[target]` and `result_ty` is the
    /// projected variable the access published.
    Derivation(TypeDerivation),
    PostApp {
        /// What the replay runs. Both kinds share one entry, one key and one
        /// declaration boundary; they differ only in how much of the call is
        /// re-decided.
        replay: PostAppReplay,
        /// Address of the `app` node this call was registered from. The rule
        /// owns a CLONE of that node, so the address has to be recorded rather
        /// than read back off the copy.
        site: usize,
        node: DeepNode,
        kids: Vec<deep::Expr>,
        func_name: String,
        location: Option<TypeDiagnosticLocation>,
        env: Box<Env>,
    },
}

/// chelis#1512: how much of a suspended call the replay re-decides.
///
/// Both kinds are `PostApp` entries so that one key, one report-once
/// cancellation and one silent declaration boundary cover them. What separates
/// them is scope: a route replay re-enters the whole of `finish_unified_app`,
/// while a dtype replay re-runs only the three dtype-admissibility validators
/// at its head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PostAppReplay {
    /// Re-enter `finish_unified_app`: the route's own arm decides again.
    Route,
    /// Re-run the dtype-admissibility validators and nothing else.
    ///
    /// Re-entering the whole entry point here would re-run the route's SHAPE
    /// rule as well, and `sum`, `mean`, `matmul` and `layer_norm` already carry
    /// their own `Reduction`/`Matmul`/`LayerNorm` entry for the same call, so
    /// the shape diagnostic would be reported twice. The dtype validators
    /// overlap no shape rule, so the two replays are independent.
    DtypeAdmissibility,
}

/// A borrowed view of one `PostApp` entry's replay inputs, so the ready replay
/// and the rigid-binder decision (chelis#2216) hand the same call to
/// [`InferenceProduct::replay_post_app`].
#[derive(Clone, Copy)]
pub(super) struct PostAppCall<'a> {
    pub(super) replay: PostAppReplay,
    pub(super) site: usize,
    pub(super) node: &'a DeepNode,
    pub(super) kids: &'a [deep::Expr],
    pub(super) func_name: &'a str,
    pub(super) location: Option<&'a TypeDiagnosticLocation>,
    pub(super) env: &'a Env,
}

#[derive(Clone)]
pub(super) struct DeferredShapeCheck {
    id: u64,
    /// The declaration whose inference registered the entry. A recursive
    /// group decides its members' entries together, when the group completes
    /// (chelis#2584), and each diagnostic still names its own declaration.
    owner: Option<String>,
    rule: DeferredShapeRule,
    arg_exprs: Vec<deep::Expr>,
    arg_tys: Vec<Type>,
    result_ty: Type,
}

struct DeferredLiteralPattern {
    owner: Option<String>,
    pattern: deep::Expr,
    scrutinee_ty: Type,
    scrutinee_name: Option<String>,
}

pub(super) struct TypeStampEpoch {
    id: u64,
    owners: UnordMap<usize, StampRequirement>,
    writes: UnordMap<usize, Vec<OwnerTypeWrite>>,
}

#[derive(Clone, Copy)]
pub(super) struct StampRequirement {
    role: &'static str,
    stamp_required: bool,
}

pub(super) struct OwnerTypeWrite {
    ty: Type,
    source: &'static str,
}

pub(super) struct FinalOwnerType {
    epoch: u64,
    ty: Type,
}

impl InferenceProduct {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_local_tensor_ascription(
        &mut self,
        origin: LocalTensorAscriptionOrigin,
        binding_name: &str,
        binding_span: Span,
        ascription_span: Span,
        initializer_span: Span,
        authored_type: deep::Expr,
        declared_type: Type,
        rhs_type_before_ascription: &Type,
        errors: &mut DiagnosticSink<'_>,
    ) {
        let Some(outstanding_claims) =
            outstanding_local_ascription_claims(&declared_type, rhs_type_before_ascription)
        else {
            return;
        };
        let Ok(raw_id) = u64::try_from(self.local_tensor_ascriptions.len()) else {
            errors.push(CheckError::new(
                CheckErrorKind::Other,
                "local tensor-ascription identity space exhausted".to_string(),
                vec!["Rejecting rather than aliasing two authored local bindings".to_string()],
            ));
            return;
        };
        self.local_tensor_ascriptions
            .push(CheckedLocalTensorAscription {
                id: LocalAscriptionId(raw_id),
                origin,
                declaration_name: self.active_declaration_name.clone(),
                binding_name: binding_name.to_string(),
                binding_span,
                ascription_span,
                initializer_span,
                authored_type,
                declared_type,
                outstanding_claims,
            });
    }

    /// chelis#1801: record the fresh dimension variables one scheme
    /// instantiation just minted, in quantifier order.
    pub(super) fn record_instantiation_dvars(&mut self, fresh: impl IntoIterator<Item = DimVar>) {
        for var in fresh {
            self.instantiation_dvars.push(var);
        }
    }

    /// chelis#1801: the current end of the instantiation log. Pair it with
    /// [`Self::instantiation_dvars_since`] around the inference of a callee.
    pub(super) fn instantiation_dvar_mark(&self) -> usize {
        self.instantiation_dvars.len()
    }

    /// chelis#1801: the dimension variables instantiation minted since
    /// `mark`, which for a mark taken immediately before a callee's inference
    /// are exactly the variables that application's instantiation owns.
    pub(super) fn instantiation_dvars_since(&self, mark: usize) -> Vec<DimVar> {
        self.instantiation_dvars[mark.min(self.instantiation_dvars.len())..].to_vec()
    }

    pub(super) fn stats(&self) -> InferStats {
        InferStats {
            typed_nodes: self.typed_nodes,
            total_nodes: self.total_nodes,
        }
    }

    pub(super) fn begin_root(&mut self, root: &deep::Expr) {
        assert!(
            self.active_epoch.is_none(),
            "type-stamp epochs must not overlap"
        );
        let id = self.next_epoch;
        self.next_epoch += 1;
        let mut epoch = TypeStampEpoch {
            id,
            owners: UnordMap::new(),
            writes: UnordMap::new(),
        };
        register_annotation_owners(root, &mut epoch);
        self.active_epoch = Some(epoch);
        self.active_declaration_name = stamped_parts(root)
            .and_then(|(tag, _, children)| {
                (tag == DeepTag::Def)
                    .then(|| children.first().and_then(symbol_name))
                    .flatten()
            })
            .map(str::to_string);
    }

    /// Record the declaration's authored-binder contract for
    /// `close_declaration` to decide.
    pub(super) fn record_authored_binder_contract(&mut self, contract: AuthoredBinderContract) {
        self.authored_binder_contracts.push(contract);
    }

    pub(super) fn take_authored_binder_contracts(&mut self) -> Vec<AuthoredBinderContract> {
        std::mem::take(&mut self.authored_binder_contracts)
    }

    /// chelis#2590: an in-group reference to `callee`, a member whose header
    /// omits a type, at the instance `ty` of its provisional scheme.
    pub(super) fn record_group_reference(
        &mut self,
        callee: &str,
        ty: Type,
        span_id: Option<String>,
        span_offset: Option<usize>,
    ) {
        self.group_references.push(GroupReference {
            caller: self.active_declaration_name.clone(),
            callee: callee.to_string(),
            ty,
            span_id,
            span_offset,
        });
    }

    /// Whether a reference to `name` from the declaration being inferred is a
    /// reference to a sibling: a member of its recursive group other than
    /// itself.
    pub(super) fn references_a_sibling(&self, name: &str) -> bool {
        self.active_declaration_name.as_deref() != Some(name)
    }

    /// Whether a reference to `name` that resolved to `scheme` is an in-group
    /// reference to a member that writes no signature, and not to a local
    /// binding that shadows it.
    pub(super) fn is_unsigned_group_reference(&self, name: &str, scheme: &Scheme) -> bool {
        scheme.tvars.is_empty()
            && scheme.dvars.is_empty()
            && scheme.rvars.is_empty()
            && self
                .group_provisional_types
                .iter()
                .any(|(member, provisional)| member == name && *provisional == scheme.body)
    }

    /// A reference to a sibling member, typed at the copy `ty` of its
    /// provisional type `own` (`group_link::sibling_instance`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_sibling_link(
        &mut self,
        callee: &str,
        ty: Type,
        own: Type,
        copies: Vec<(Variable, Variable)>,
        awaits: bool,
        span_id: Option<String>,
        span_offset: Option<usize>,
    ) {
        self.sibling_links.push(SiblingLink {
            caller: self.active_declaration_name.clone(),
            callee: callee.to_string(),
            ty,
            own,
            copies,
            awaits,
            span_id,
            span_offset,
        });
    }

    /// chelis#2590: the instance of `name`'s provisional scheme that its own
    /// body is inferred against.
    pub(super) fn record_group_member_type(&mut self, name: &str, ty: Type) {
        self.group_member_types.push((name.to_string(), ty));
    }

    /// chelis#2626: the names and provisional types of the members of the
    /// recursive group about to be inferred that write no signature.
    pub(super) fn record_group_provisional_types(
        &mut self,
        types: impl IntoIterator<Item = (String, Type)>,
    ) {
        self.group_provisional_types.extend(types);
    }

    /// chelis#2626: whether `ty` is a type variable that the completion of
    /// the recursive group being inferred determines further: a variable of
    /// the provisional type of a member that writes no signature, or of a
    /// sibling reference's copy of one. A rule that cannot be decided on such
    /// a variable waits for it. A body sees neither bound by a sibling, in any
    /// order (`group_link::sibling_instance`), so the wait, and the decision
    /// the group's completion replays, is the same in every declaration order.
    ///
    /// A reference to a member whose signature omits only some types takes a
    /// fresh instance of them (chelis#2590), which the completion decides as
    /// the member's own type or a concrete one. Waiting on it would let the
    /// caller's own use of the reference choose the instance, which is how a
    /// generic function is instantiated, so a rule is decided on it where it
    /// is inferred.
    ///
    /// A variable anywhere inside those types qualifies, not only the whole
    /// type of a parameter or result: a tuple component, a list element or a
    /// tensor's precision is linked just the same.
    pub(super) fn awaits_group_completion(&self, ty: &Type, subst: &Subst) -> bool {
        let Type::Var(var) = subst.apply(ty) else {
            return false;
        };
        self.group_provisional_types
            .iter()
            .map(|(_, provisional)| provisional)
            .chain(
                self.sibling_links
                    .iter()
                    .filter(|link| link.awaits)
                    .map(|link| &link.ty),
            )
            .any(|linked| crate::env::free_tvars(&resolved(linked, subst)).contains(&var))
    }

    pub(super) fn group_links_in_progress(&self) -> bool {
        !self.group_provisional_types.is_empty() || !self.closing_group_types.is_empty()
    }

    pub(super) fn group_links_completed(&mut self) {
        self.closing_group_types.clear();
    }

    /// chelis#2651: the member of the recursive group being inferred whose
    /// types the group has yet to determine and that `ty`, a type variable,
    /// stands for part of: a variable of a signature-less member's
    /// provisional type, of a sibling reference's copy of a member's type, or
    /// of an in-group reference's instance of a member's omitted types. A body
    /// sees each of these unbound by any sibling, in every declaration order,
    /// so the answer is the same in every order.
    pub(super) fn group_variable_owner(&self, ty: &Type, subst: &Subst) -> Option<&str> {
        let Type::Var(var) = subst.apply(ty) else {
            return None;
        };
        let holds = |linked: &Type| crate::env::free_tvars(&resolved(linked, subst)).contains(&var);
        self.group_provisional_types
            .iter()
            .find(|(_, provisional)| holds(provisional))
            .map(|(name, _)| name.as_str())
            .or_else(|| {
                self.sibling_links
                    .iter()
                    .find(|link| holds(&link.ty))
                    .map(|link| link.callee.as_str())
            })
            .or_else(|| {
                self.group_references
                    .iter()
                    .find(|reference| holds(&reference.ty))
                    .map(|reference| reference.callee.as_str())
            })
    }

    /// The component's in-group references, member types and sibling links,
    /// taken for its completion.
    pub(super) fn take_group_links(
        &mut self,
    ) -> (Vec<GroupReference>, Vec<(String, Type)>, Vec<SiblingLink>) {
        self.closing_group_types = self
            .group_provisional_types
            .iter()
            .map(|(_, ty)| ty.clone())
            .chain(self.group_member_types.iter().map(|(_, ty)| ty.clone()))
            .chain(
                self.group_references
                    .iter()
                    .map(|reference| reference.ty.clone()),
            )
            .chain(self.sibling_links.iter().map(|link| link.ty.clone()))
            .collect();
        self.group_provisional_types.clear();
        (
            std::mem::take(&mut self.group_references),
            std::mem::take(&mut self.group_member_types),
            std::mem::take(&mut self.sibling_links),
        )
    }

    /// The authored-binder contracts still to decide, without taking them.
    pub(super) fn pending_authored_binder_contracts(&self) -> &[AuthoredBinderContract] {
        &self.authored_binder_contracts
    }

    /// The declaration that owns a ledger entry registered now: the one whose
    /// entry is being replayed, or else the one being inferred.
    fn entry_owner(&self) -> Option<String> {
        self.replaying_owner
            .clone()
            .or_else(|| self.active_declaration_name.clone())
    }

    pub(super) fn deferred_literal_pattern_checkpoint(&self) -> usize {
        self.deferred_literal_patterns.len()
    }

    pub(super) fn defer_literal_pattern(
        &mut self,
        pattern: &deep::Expr,
        scrutinee_ty: &Type,
        site: PatternSite<'_>,
    ) {
        let scrutinee_name = match site {
            PatternSite::ArmOfVariable(name) => Some(name.to_string()),
            PatternSite::Other => None,
        };
        self.deferred_literal_patterns.push(DeferredLiteralPattern {
            owner: self.entry_owner(),
            pattern: pattern.clone(),
            scrutinee_ty: scrutinee_ty.clone(),
            scrutinee_name,
        });
    }

    /// The same unresolved obligation that must reject at the declaration
    /// boundary must also prevent a local lambda from generalizing first.
    pub(super) fn has_pending_literal_pattern_since(
        &self,
        checkpoint: usize,
        subst: &Subst,
    ) -> bool {
        self.deferred_literal_patterns[checkpoint..]
            .iter()
            .any(|check| matches!(subst.apply(&check.scrutinee_ty), Type::Var(_)))
    }

    pub(super) fn finish_deferred_literal_patterns(
        &mut self,
        env: &Env,
        subst: &Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        for check in self.deferred_literal_patterns.drain(..) {
            super::expr_pattern::validate_deferred_literal_pattern(
                &check.pattern,
                &check.scrutinee_ty,
                check.scrutinee_name.as_deref(),
                check.owner.as_deref(),
                env,
                subst,
                adt_reg,
                errors,
            );
        }
    }

    /// Snapshot before body inference: operation and callee constraints may
    /// narrow these holes, but cannot turn them into implicit generic bounds.
    pub(super) fn record_inferred_contract(
        &mut self,
        subject: &str,
        ty: &Type,
        env: &Env,
        subst: &Subst,
    ) {
        let variables = crate::env::free_tvars(&subst.apply(ty))
            .into_iter()
            .filter(|variable| {
                // Explicit binders have their own rigidity/entailment check.
                subst.tvar_restriction(*variable).is_none()
                    && !env.active_declared_type_names().contains_key(variable)
            });
        self.record_admission_variables(subject, variables, env);
    }

    fn record_admission_variables(
        &mut self,
        subject: &str,
        variables: impl IntoIterator<Item = TypeVar>,
        env: &Env,
    ) {
        let binders: Vec<TypeVar> = env
            .active_declared_type_names()
            .to_sorted()
            .into_iter()
            .map(|(variable, _)| *variable)
            .collect();
        let owner = self.entry_owner();
        for variable in variables {
            self.inferred_admission_contracts
                .push(InferredAdmissionContract {
                    owner: owner.clone(),
                    subject: subject.to_string(),
                    variable,
                    binders: binders.clone(),
                    failed_application: None,
                });
        }
    }

    /// Operand requirements can originate in a temporary value, not only in
    /// a function parameter (for example `sin(to_tensor([]))`). Record them
    /// before the checked callee's requirements enter ordinary unification.
    /// Merely transporting a function value never passes through this edge.
    pub(super) fn record_call_operand_contracts(
        &mut self,
        callee: &Type,
        name: Option<&str>,
        arguments: &[Type],
        env: &Env,
        subst: &Subst,
    ) {
        let Type::Fn(parameters, _) = subst.apply(callee) else {
            return;
        };
        for (index, (parameter, argument)) in parameters.iter().zip(arguments).enumerate() {
            if crate::env::free_tvars(parameter)
                .iter()
                .any(|variable| subst.tvar_restriction(*variable).is_some())
            {
                self.record_inferred_contract(
                    &format!(
                        "operand {} of `{}`",
                        index + 1,
                        name.unwrap_or("<function value>")
                    ),
                    argument,
                    env,
                    subst,
                );
            }
        }
    }

    pub(super) fn record_call_result_contracts(
        &mut self,
        callee: &Type,
        name: Option<&str>,
        env: &Env,
        subst: &Subst,
    ) {
        let Type::Fn(_, result) = subst.apply(callee) else {
            return;
        };
        // Calling a result-polymorphic factory must satisfy its requirement
        // too, even when it has no operands. Inspect only data positions:
        // a returned function (including one nested in an aggregate) carries
        // its already-checked contract, rather than invoking that contract.
        let mut pending = vec![result.as_ref()];
        let mut variables = BTreeSet::new();
        while let Some(ty) = pending.pop() {
            match ty {
                Type::Var(variable) | Type::Tensor(_, TensorPrec::Var(variable)) => {
                    if subst.tvar_restriction(*variable).is_some() {
                        variables.insert(*variable);
                    }
                }
                Type::Ref(inner) => pending.push(inner),
                Type::Adt(_, fields) | Type::Tuple(fields) => pending.extend(fields),
                Type::KindedAdt(_, arguments) => {
                    pending.extend(arguments.iter().filter_map(NominalArg::as_type));
                }
                Type::Fn(..)
                | Type::Prim(_)
                | Type::Tensor(_, TensorPrec::Concrete(_))
                | Type::Unit
                | Type::Error(_) => {}
            }
        }
        self.record_admission_variables(
            &format!("result of `{}`", name.unwrap_or("<function value>")),
            variables,
            env,
        );
    }

    pub(super) fn admission_contract_checkpoint(&self) -> usize {
        self.inferred_admission_contracts.len()
    }

    pub(super) fn has_pending_admission_contract_since(
        &self,
        checkpoint: usize,
        subst: &Subst,
    ) -> bool {
        self.inferred_admission_contracts[checkpoint..]
            .iter()
            .any(|contract| !contract.unmet_families(subst).is_empty())
    }

    fn finish_admission_contracts(&mut self, subst: &Subst, errors: &mut DiagnosticSink<'_>) {
        let mut reported = BTreeSet::new();
        for contract in self.inferred_admission_contracts.drain(..) {
            for (variable, required) in contract.unmet_families(subst) {
                if !reported.insert(variable) {
                    continue;
                }
                errors.push(CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    format!(
                        "{} in `{}` requires `{}` \
                         admission, but its unresolved type `{}` has no sufficient declared \
                         contract at the declaration boundary (spec/04-type-system.md §3.1)",
                        contract.subject,
                        contract.owner.as_deref().unwrap_or("<anonymous>"),
                        required.bound_spelling(),
                        subst.apply(&Type::Var(contract.variable)),
                    ),
                    vec![
                        "Declare the required dtype-family bound or a concrete parameter type; \
                        a local inference hole must bind within its enclosing declaration, \
                        not acquire an implicit generic bound."
                            .to_string(),
                    ],
                ));
            }
        }
    }

    pub(super) fn record_canonical(&mut self, expr: &deep::Expr, ty: Type) {
        self.record(expr, ty, "canonical infer_expr traversal");
    }

    pub(super) fn record_bypass(&mut self, expr: &deep::Expr, ty: Type, source: &'static str) {
        self.record(expr, ty, source);
    }

    pub(super) fn record(&mut self, expr: &deep::Expr, ty: Type, source: &'static str) {
        let Some(epoch) = self.active_epoch.as_mut() else {
            return;
        };
        let key = expr_key(expr);
        if !epoch.owners.contains_key(&key) {
            return;
        }
        epoch
            .writes
            .entry(key)
            .or_default()
            .push(OwnerTypeWrite { ty, source });
    }

    pub(super) fn record_builtin_selection(
        &mut self,
        selection: crate::builtin_discovery::BuiltinCaseSelection,
    ) {
        if matches!(
            &selection,
            crate::builtin_discovery::BuiltinCaseSelection::ByOperand(_)
        ) {
            self.pending_builtin_selections
                .push(self.builtin_selections.len());
        }
        self.builtin_selections.push(selection);
    }

    /// Resolve after enclosing calls have unified their operands. Generic
    /// bodies preserve a type-indexed selection; they do not pick a sibling
    /// until their existing owner type is instantiated.
    fn resolve_builtin_selections(&mut self, subst: &Subst, errors: &mut DiagnosticSink<'_>) {
        for index in std::mem::take(&mut self.pending_builtin_selections) {
            let selection = &mut self.builtin_selections[index];
            match selection.resolve(subst) {
                Ok(resolved) => {
                    if matches!(
                        &resolved,
                        crate::builtin_discovery::BuiltinCaseSelection::ByOperand(_)
                    ) {
                        self.pending_builtin_selections.push(index);
                    }
                    *selection = resolved;
                }
                Err(reason) => errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("builtin operand has no declared semantic case: {reason}"),
                    vec![],
                )),
            }
        }
    }

    pub(super) fn finish_root(&mut self, subst: &Subst, errors: &mut DiagnosticSink<'_>) {
        if errors.is_empty() {
            self.resolve_builtin_selections(subst, errors);
        }
        let Some(epoch) = self.active_epoch.take() else {
            errors.push(internal_owner_stamp_error(
                "attempted to finish a type-stamp epoch that was not active".to_string(),
            ));
            return;
        };
        self.active_declaration_name = None;

        for (key, requirement) in epoch.owners.into_sorted() {
            let Some(writes) = epoch.writes.get(&key) else {
                // Missing owners are diagnosed at the exact annotation lookup,
                // where the original construct and role are still available.
                continue;
            };
            let mut resolved = writes
                .iter()
                .map(|write| (subst.apply(&write.ty), write.source));
            let Some((canonical, canonical_source)) = resolved.next() else {
                continue;
            };
            for (candidate, candidate_source) in resolved {
                if !owner_types_compatible(&canonical, &candidate) {
                    errors.push(internal_owner_stamp_error(format!(
                        "conflicting authoritative type writes for {} in epoch {}: \
                         `{canonical}` from {canonical_source} vs `{candidate}` from \
                         {candidate_source}",
                        requirement.role, epoch.id
                    )));
                }
            }
            if requirement.stamp_required {
                self.owner_types.insert(
                    key,
                    FinalOwnerType {
                        epoch: epoch.id,
                        ty: canonical,
                    },
                );
            }
        }
    }

    /// Apply the complete program substitution to the frozen owner stamps.
    /// Most stamps are already concrete when their root finishes, but a
    /// variable an earlier root left open can be bound by a later one, so
    /// every stamp gets one final resolution before annotation.
    pub(super) fn resolve_owner_types(&mut self, subst: &Subst) {
        let keys = self
            .owner_types
            .to_sorted()
            .into_iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>();
        for key in keys {
            let owner = self
                .owner_types
                .get_mut(&key)
                .expect("collected owner key remains present");
            // All authored-body/result guards have finished. Annotation is
            // the semantic name view; the reusable Env/Subst retains IDs.
            owner.ty = subst.semantic_type(&owner.ty);
        }
    }

    pub(super) fn owner_type(
        &self,
        expr: &deep::Expr,
        role: &'static str,
        errors: &mut DiagnosticSink<'_>,
    ) -> Option<Type> {
        match self.owner_types.get(&expr_key(expr)) {
            Some(owner) => {
                let _owning_epoch = owner.epoch;
                Some(owner.ty.clone())
            }
            None => {
                let construct = match expr.carrier() {
                    deep::ExprCarrier::DecodedNode(tag, _, _) => tag.as_str(),
                    deep::ExprCarrier::UndecodableHead(head, _, _) => head,
                    deep::ExprCarrier::StructuralList(_) => "<bare-list>",
                    deep::ExprCarrier::Atom(_) => "<atom>",
                    deep::ExprCarrier::MetadataMap(_) => "<map>",
                    deep::ExprCarrier::MetadataExpression(_) => "<meta-expr>",
                };
                errors.push(internal_owner_stamp_error(format!(
                    "missing authoritative type stamp for {role} `{construct}`"
                )));
                None
            }
        }
    }

    pub(super) fn current_owner_type(
        &self,
        expr: &deep::Expr,
        subst: &Subst,
        errors: &mut DiagnosticSink<'_>,
    ) -> Option<Type> {
        let Some(epoch) = self.active_epoch.as_ref() else {
            errors.push(internal_owner_stamp_error(
                "canonical inference requested a stamp outside an active epoch".to_string(),
            ));
            return None;
        };
        let key = expr_key(expr);
        let Some(writes) = epoch.writes.get(&key) else {
            errors.push(internal_owner_stamp_error(
                "canonical inference could not find an already-inferred child stamp".to_string(),
            ));
            return None;
        };
        let mut resolved = writes.iter().map(|write| subst.apply(&write.ty));
        let canonical = resolved.next()?;
        if resolved.any(|candidate| !owner_types_compatible(&canonical, &candidate)) {
            errors.push(internal_owner_stamp_error(
                "canonical inference observed conflicting child stamps".to_string(),
            ));
            return None;
        }
        Some(canonical)
    }
}

// chelis#1512 round 2 P2-1: every `PostApp` ledger key, with whether it was
// minted during a replay.
//
// A test cannot read the ledger, and the property at issue is about the KEY
// rather than about any diagnostic, so it has to be observed where it is
// minted. Thread-local because the checker runs single-threaded per program
// and nextest gives each test its own thread. A `///` here would attach to the
// macro invocation and go nowhere, which `unused_doc_comments` rejects.
#[cfg(test)]
thread_local! {
    static POST_APP_KEY_LOG: std::cell::RefCell<Vec<(usize, bool)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn record_post_app_key(site: usize, during_replay: bool) {
    POST_APP_KEY_LOG.with(|log| log.borrow_mut().push((site, during_replay)));
}

/// Drain the log. Returns `(key, minted during a replay)` in mint order.
#[cfg(test)]
pub(crate) fn take_post_app_key_log() -> Vec<(usize, bool)> {
    POST_APP_KEY_LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

pub(super) fn expr_key(expr: &deep::Expr) -> usize {
    std::ptr::from_ref(expr).addr()
}

pub(super) fn owner_types_compatible(left: &Type, right: &Type) -> bool {
    let mut compatibility_subst = Subst::new();
    unify(left, right, &mut compatibility_subst).is_ok()
}

pub(super) fn internal_owner_stamp_error(message: String) -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        format!("internal: annotation owner-stamp invariant violated: {message}"),
        vec![],
    )
}

// Decode-once left this walk with no diagnostic of its own to report: the
// only arm that ever wrote to a sink was the version-skew fallback that
// `child_stamp_role`'s totality made unrepresentable, so the walk no longer
// takes a `DiagnosticSink`.
pub(super) fn register_annotation_owners(expr: &deep::Expr, epoch: &mut TypeStampEpoch) {
    stack_guard!("register_annotation_owners", expr);
    let (tag, kids) = match expr.carrier() {
        deep::ExprCarrier::DecodedNode(tag, _, children) => (Some(tag), children),
        deep::ExprCarrier::MetadataExpression(meta) => {
            register_annotation_owners(&meta.expr, epoch);
            return;
        }
        deep::ExprCarrier::StructuralList(children)
        | deep::ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                register_annotation_owners(child, epoch);
            }
            return;
        }
        deep::ExprCarrier::Atom(_) | deep::ExprCarrier::MetadataMap(_) => return,
    };
    if let Some(tag) = tag {
        let (role, stamp_required) = if tag == DeepTag::Fn {
            ("function node", true)
        } else if should_attach_type_metadata(tag) {
            ("metadata-eligible expression", true)
        } else if matches!(tag, DeepTag::PatVar | DeepTag::PatAs) {
            ("pattern binding", true)
        } else {
            ("semantic runtime node", false)
        };
        register_owner(epoch, expr, role, stamp_required);
    }
    for (index, child) in kids.iter().enumerate() {
        // Decode-once: `child_stamp_role` is total over `DeepTag`, so the
        // old "no child ownership classification" version-skew arm is
        // unrepresentable; only genuinely untagged structural lists (empty
        // guards and malformed nested input) take the recursive fallback,
        // and their owning checker rejects the shape before annotation is
        // returned.
        match tag.map(|tag| child_stamp_role(tag, index, kids.len())) {
            Some(ChildStampRole::RuntimeExpr | ChildStampRole::ExplicitInferenceBypass) | None => {
                register_annotation_owners(child, epoch);
            }
            Some(
                ChildStampRole::Syntax
                | ChildStampRole::Selector
                | ChildStampRole::EffectHandler
                | ChildStampRole::Binder
                | ChildStampRole::Type,
            ) => {}
        }
    }
}

pub(super) fn register_owner(
    epoch: &mut TypeStampEpoch,
    expr: &deep::Expr,
    role: &'static str,
    stamp_required: bool,
) {
    epoch
        .owners
        .entry(expr_key(expr))
        .or_insert(StampRequirement {
            role,
            stamp_required,
        });
}

#[cfg(test)]
pub(crate) enum TypeStampMutationCase {
    Missing,
    CompatibleRepeat,
    IncompatibleRepeat,
    UnregisteredSynthesized,
    RuntimeNonStampOwnerLookup,
}

/// chelis#1512: the two branches of [`reconcile_replayed_result`], driven
/// from `session.rs` where a `DiagnosticSink` can be constructed.
#[cfg(test)]
pub(crate) enum ReconcileMutationCase {
    /// The replayed rule produces a type the suspended call did not publish.
    Disagrees,
    /// The rule produces the type a fresh published variable can take.
    Agrees,
    /// A consumer requires float values but the producer settles to integers.
    FamilyDisagrees,
    /// The same consumer accepts a float result.
    FamilyAgrees,
}

/// Returns the rendered produced type and whether the published type ended up
/// bound to it.
#[cfg(test)]
pub(crate) fn run_reconcile_mutation_case(
    case: ReconcileMutationCase,
    errors: &mut DiagnosticSink<'_>,
) -> (String, bool) {
    let tensor = |dim: i64| Type::Tensor(vec![Dim::Lit(dim)], TensorPrec::Concrete(Prim::F32));
    let mut subst = Subst::new();
    let mut vg = VarGen::default();
    let published = match case {
        ReconcileMutationCase::Disagrees => tensor(3),
        ReconcileMutationCase::Agrees => Type::Var(vg.fresh_tvar()),
        ReconcileMutationCase::FamilyDisagrees | ReconcileMutationCase::FamilyAgrees => {
            let var = vg.fresh_tvar();
            subst
                .narrow_tvar_restriction(var, TypeVarRestriction::FloatValue)
                .unwrap();
            Type::Var(var)
        }
    };
    let produce = match case {
        ReconcileMutationCase::Disagrees => tensor(4),
        ReconcileMutationCase::Agrees => tensor(3),
        ReconcileMutationCase::FamilyDisagrees => {
            Type::Tensor(vec![Dim::Lit(3)], TensorPrec::Concrete(Prim::Int32))
        }
        ReconcileMutationCase::FamilyAgrees => tensor(3),
    };
    let produced = reconcile_replayed_result("permute", &published, produce, &mut subst, errors);
    let bound = subst.apply(&published) == produced;
    (produced.to_string(), bound)
}

/// The `app` owner the mutation cases stamp. Its callee is a `var`, which
/// carries no type stamp of its own, so every case observes only the `app`
/// owner's stamp as it did when the owner was a childless `app`.
#[cfg(test)]
fn mutation_case_app(args: Vec<deep::Expr>) -> deep::Expr {
    let mut children = vec![stamped_node_expr(DeepTag::Var, vec![symbol_expr("f")])];
    children.extend(args);
    stamped_node_expr(DeepTag::App, children)
}

#[cfg(test)]
pub(crate) fn run_type_stamp_mutation_case(
    case: TypeStampMutationCase,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let owner = mutation_case_app(vec![]);
    let mut product = InferenceProduct::default();
    product.begin_root(&owner);
    match case {
        TypeStampMutationCase::Missing => {
            product.finish_root(&Subst::new(), errors);
            product.owner_type(&owner, "test owner", errors).is_none()
        }
        TypeStampMutationCase::CompatibleRepeat => {
            let ty = Type::Prim(Prim::Int64);
            product.record_canonical(&owner, ty.clone());
            product.record_bypass(&owner, ty.clone(), "compatible test repeat");
            product.finish_root(&Subst::new(), errors);
            product.owner_type(&owner, "test owner", errors) == Some(ty)
        }
        TypeStampMutationCase::IncompatibleRepeat => {
            product.record_canonical(&owner, Type::Prim(Prim::Int64));
            product.record_bypass(&owner, Type::Prim(Prim::String), "incompatible test repeat");
            product.finish_root(&Subst::new(), errors);
            true
        }
        TypeStampMutationCase::UnregisteredSynthesized => {
            let synthesized = stamped_node_expr(DeepTag::Var, vec![symbol_expr("temporary")]);
            product.record_bypass(
                &synthesized,
                Type::Prim(Prim::Int64),
                "unregistered synthesized test node",
            );
            product.record_canonical(&owner, Type::Prim(Prim::Int64));
            product.finish_root(&Subst::new(), errors);
            product
                .owner_type(&synthesized, "synthesized test node", errors)
                .is_none()
        }
        TypeStampMutationCase::RuntimeNonStampOwnerLookup => {
            let runtime_child = stamped_node_expr(DeepTag::Var, vec![symbol_expr("x")]);
            let root = stamped_node_expr(DeepTag::App, vec![runtime_child]);
            let mut product = InferenceProduct::default();
            product.begin_root(&root);
            let (_, _, root_children) =
                stamped_parts(&root).expect("`stamped_node_expr` builds a decoded node");
            let runtime_child = &root_children[0];
            product.record_canonical(runtime_child, Type::Prim(Prim::Int64));
            product.current_owner_type(runtime_child, &Subst::new(), errors)
                == Some(Type::Prim(Prim::Int64))
        }
    }
}

#[cfg(test)]
pub(crate) enum FinalizationMutationCase {
    MissingRuntimeStamp,
    SilentErrorOwner,
    SilentErrorSignature,
}

#[cfg(test)]
pub(crate) fn run_finalization_mutation_case(
    case: FinalizationMutationCase,
    errors: &mut DiagnosticSink<'_>,
) {
    let runtime = mutation_case_app(vec![]);
    let mut signature_context = SignatureInferenceMetadata::default();
    let annotated = match case {
        FinalizationMutationCase::MissingRuntimeStamp => vec![runtime],
        FinalizationMutationCase::SilentErrorOwner => {
            let mut product = InferenceProduct::default();
            product.begin_root(&runtime);
            product.record_canonical(&runtime, crate::errors::error_sentinel_for_test());
            product.finish_root(&Subst::new(), errors);
            annotate_ir_program(std::slice::from_ref(&runtime), &product, errors)
        }
        FinalizationMutationCase::SilentErrorSignature => {
            let error_ty = crate::errors::error_sentinel_for_test();
            signature_context.functions.insert(
                "poison".to_string(),
                FunctionSignatureInference {
                    name: "poison".to_string(),
                    authored_signature: false,
                    authored_signature_type: None,
                    recursive_cycle: false,
                    checked_signature: error_ty.clone(),
                    display_signature: error_ty,
                    params: vec![],
                },
            );
            vec![]
        }
    };
    let _ = finalize_checked_program(
        annotated,
        BTreeMap::new(),
        &signature_context,
        &InferenceProduct::default(),
        InferStats::default(),
        errors,
    );
}

/// Collect the declared signature metadata owned by one inference or
/// annotation unit. The returned map is passed explicitly; nested, sequential,
/// and parallel checks cannot observe another unit's declarations.
pub(super) fn collect_declared_sig_metadata<'a>(
    exprs: impl IntoIterator<Item = &'a deep::Expr>,
) -> UnordMap<String, DeclaredSigMetadata> {
    let mut map: UnordMap<String, DeclaredSigMetadata> = UnordMap::new();
    for expr in exprs {
        collect_defsig_param_types(expr, &mut map);
    }
    map
}

/// Recursively collect every valid `(defsig name [(binders...)] type-expr)`
/// entry, descending through `(module ...)` wrappers. Function signatures
/// additionally retain their leading argument type expressions (the trailing
/// return type is dropped); non-function signatures retain an empty parameter
/// list. A re-declared name keeps the first sig seen.
pub(super) fn collect_defsig_param_types(
    expr: &deep::Expr,
    map: &mut UnordMap<String, DeclaredSigMetadata>,
) {
    // Bail before unbounded recursion exhausts the native stack on a
    // deeply-nested input. No error vector here; `stack_guard_tripped`
    // records the bail so the check entry boundary fails hard with a located
    // diagnostic. See `STACK_RED_ZONE_BYTES`. (In practice this walker only
    // descends `module` wrappers, which do not nest deeply, but the guard
    // keeps the "every recursive walker is bounded" invariant uniform and
    // cheap.)
    stack_guard!("collect_defsig_param_types", expr);
    let Some((tag, meta, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            for child in kids.iter().skip(1) {
                collect_defsig_param_types(child, map);
            }
        }
        DeepTag::Defsig => {
            let Some((name_expr, binder_list, type_expr)) = defsig_parts(kids) else {
                return;
            };
            let Some(name) = symbol_name(name_expr) else {
                return;
            };
            let Some(binders) = valid_defsig_binder_names(binder_list) else {
                return;
            };
            let param_type_exprs = match stamped_parts(type_expr) {
                Some((DeepTag::TFn, _, fn_kids)) => {
                    if fn_kids.is_empty() {
                        return;
                    }
                    // All but the trailing return type are parameter types.
                    fn_kids[..fn_kids.len() - 1].to_vec()
                }
                _ => Vec::new(),
            };
            map.entry(name.to_string())
                .or_insert_with(|| DeclaredSigMetadata {
                    param_types: param_type_exprs,
                    binders,
                    dtype_bounds: chelis_deep::decode_dtype_bounds(meta).into_iter().collect(),
                });
        }
        _ => {}
    }
}

/// Artifact-local identity of one explicit local tensor ascription.
///
/// Independently checked programs allocate separate domains.
/// [`CheckedProgram::compose`] explicitly remaps the new-code domain before
/// combining it with a checked library.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct LocalAscriptionId(u64);

impl LocalAscriptionId {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Authored source channel that owns a local tensor ascription.
///
/// Surf stamps inferred metadata explicitly, so unmarked public Deep `type`
/// metadata remains an authored contract rather than being confused with
/// compiler-inferred Surf metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LocalTensorAscriptionOrigin {
    SurfExplicit,
    DeepTypeMetadata,
}

/// One declared tensor axis that checking could not independently discharge.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LocalAscriptionAxisClaim {
    axis: usize,
    required_extent: Dim,
}

impl LocalAscriptionAxisClaim {
    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn required_extent(&self) -> &Dim {
        &self.required_extent
    }
}

/// Exact checker-owned record of an authored local tensor ascription.
///
/// The authored Deep type remains distinct from the checked type so wildcard
/// spelling and source structure cannot be reconstructed from ordinary
/// inferred metadata. `outstanding_claims` contains only declared axes whose
/// agreement was not statically proved from the RHS before ascription
/// unification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CheckedLocalTensorAscription {
    id: LocalAscriptionId,
    origin: LocalTensorAscriptionOrigin,
    declaration_name: Option<String>,
    binding_name: String,
    binding_span: Span,
    ascription_span: Span,
    initializer_span: Span,
    authored_type: deep::Expr,
    declared_type: Type,
    outstanding_claims: Vec<LocalAscriptionAxisClaim>,
}

impl CheckedLocalTensorAscription {
    pub fn id(&self) -> LocalAscriptionId {
        self.id
    }

    pub fn origin(&self) -> LocalTensorAscriptionOrigin {
        self.origin
    }

    pub fn declaration_name(&self) -> Option<&str> {
        self.declaration_name.as_deref()
    }

    pub fn binding_name(&self) -> &str {
        &self.binding_name
    }

    pub fn binding_span(&self) -> Span {
        self.binding_span
    }

    pub fn ascription_span(&self) -> Span {
        self.ascription_span
    }

    pub fn initializer_span(&self) -> Span {
        self.initializer_span
    }

    pub fn authored_type(&self) -> &deep::Expr {
        &self.authored_type
    }

    pub fn declared_type(&self) -> &Type {
        &self.declared_type
    }

    pub fn outstanding_claims(&self) -> &[LocalAscriptionAxisClaim] {
        &self.outstanding_claims
    }

    /// Clone this checked obligation into an identity domain above an earlier
    /// independently checked artifact. The source spans and declaration
    /// identity stay unchanged; only the opaque artifact-local ID moves.
    pub fn with_id_offset(&self, offset: u64) -> Option<Self> {
        let mut remapped = self.clone();
        remapped.id = LocalAscriptionId(remapped.id.0.checked_add(offset)?);
        Some(remapped)
    }
}

/// Compose checker-owned local obligations across independently checked units.
///
/// A replacement definition owns the active body for its declaration name, so
/// obligations attached to the shadowed library body are unreachable and must
/// not participate in source-span selection. Retained records keep their
/// identities; new-code records move above that retained identity domain.
pub fn compose_local_tensor_ascriptions(
    library: &[CheckedLocalTensorAscription],
    new_code: &[CheckedLocalTensorAscription],
    replaced_definitions: &BTreeSet<String>,
) -> Option<Vec<CheckedLocalTensorAscription>> {
    let mut composed = library
        .iter()
        .filter(|ascription| {
            ascription
                .declaration_name()
                .is_none_or(|name| !replaced_definitions.contains(name))
        })
        .cloned()
        .collect::<Vec<_>>();
    let new_code_id_offset = match composed.iter().map(|ascription| ascription.id.0).max() {
        Some(last) => last.checked_add(1)?,
        None => 0,
    };
    for ascription in new_code {
        composed.push(ascription.with_id_offset(new_code_id_offset)?);
    }
    Some(composed)
}

fn tensor_dims(ty: &Type) -> Option<&[Dim]> {
    match ty {
        Type::Tensor(dims, _) => Some(dims),
        Type::Ref(inner) => tensor_dims(inner),
        _ => None,
    }
}

fn outstanding_local_ascription_claims(
    declared_type: &Type,
    rhs_type_before_ascription: &Type,
) -> Option<Vec<LocalAscriptionAxisClaim>> {
    let declared_dims = tensor_dims(declared_type)?;
    let rhs_dims = tensor_dims(rhs_type_before_ascription);
    let mut claims = Vec::new();
    for (axis, required_extent) in declared_dims.iter().enumerate() {
        if matches!(required_extent, Dim::Wildcard | Dim::Rank(_)) {
            continue;
        }
        let independently_proved =
            rhs_dims
                .and_then(|dims| dims.get(axis))
                .is_some_and(|observed| match (required_extent, observed) {
                    (Dim::Lit(required), Dim::Lit(actual)) => required == actual,
                    (Dim::Var(required), Dim::Var(actual)) => required == actual,
                    _ => false,
                });
        if !independently_proved {
            claims.push(LocalAscriptionAxisClaim {
                axis,
                required_extent: required_extent.clone(),
            });
        }
    }
    Some(claims)
}
/// Result of running type inference on a program.
#[derive(Debug)]
pub struct InferResult {
    pub errors: Vec<CheckError>,
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct InferStats {
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckedProgram {
    annotated_exprs: Vec<deep::Expr>,
    type_env: BTreeMap<String, deep::Expr>,
    local_tensor_ascriptions: Vec<CheckedLocalTensorAscription>,
    linearity: LinearityInfo,
    signature_inference: SignatureInferenceMetadata,
    type_headers: TypeResolutionEnv,
    #[serde(default)]
    adt_registry: AdtRegistry,
    /// Honest checker-visit counters for this checked unit. These are
    /// serialized with cached contexts so layered fitness reports can sum the
    /// same inference-product metric as the monolithic path (chelis#973).
    #[serde(default)]
    infer_stats: InferStats,
    #[serde(default)]
    library_proof_id: Option<LibraryProofId>,
    #[serde(default)]
    context_library_proof_id: Option<LibraryProofId>,
}

impl CheckedProgram {
    #[cfg(test)]
    pub(crate) fn unchecked_for_linearity_diagnostic_test(
        annotated_exprs: Vec<deep::Expr>,
        type_env: BTreeMap<String, deep::Expr>,
    ) -> Self {
        Self {
            annotated_exprs,
            type_env,
            local_tensor_ascriptions: Vec::new(),
            linearity: LinearityInfo::default(),
            signature_inference: SignatureInferenceMetadata::default(),
            type_headers: TypeResolutionEnv::default(),
            adt_registry: AdtRegistry::default(),
            infer_stats: InferStats::default(),
            library_proof_id: None,
            context_library_proof_id: None,
        }
    }

    /// Replace only effects-owned metadata on an already-checked program.
    ///
    /// Every span, atom, child, and non-`effects` metadata entry must remain
    /// byte-for-byte identical to `self`. The returned program preserves the
    /// original type/signature/linearity contexts and crosses the same
    /// fallible totality boundary as a fresh checker result.
    pub fn try_with_effect_annotations(
        &self,
        annotated_exprs: Vec<deep::Expr>,
    ) -> Result<Self, InferResult> {
        crate::session::try_checked_program_with_effect_annotations(self, annotated_exprs)
    }

    pub fn exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn annotated_exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn type_env(&self) -> &BTreeMap<String, deep::Expr> {
        &self.type_env
    }

    pub fn local_tensor_ascriptions(&self) -> &[CheckedLocalTensorAscription] {
        &self.local_tensor_ascriptions
    }

    pub fn linearity(&self) -> &LinearityInfo {
        &self.linearity
    }

    pub fn signature_inference(&self) -> &SignatureInferenceMetadata {
        &self.signature_inference
    }

    pub(crate) fn type_headers(&self) -> &TypeResolutionEnv {
        &self.type_headers
    }

    /// Checker-owned, alias-resolved ADT definitions used by lowering.
    ///
    /// Consumers must use this registry instead of reconstructing constructor
    /// layouts or generic-parameter roles from authored `deftype` syntax.
    pub fn adt_registry(&self) -> &AdtRegistry {
        &self.adt_registry
    }

    pub fn infer_stats(&self) -> InferStats {
        self.infer_stats
    }

    pub(crate) fn bind_library_proof(&mut self, proof_id: LibraryProofId) {
        self.library_proof_id = Some(proof_id);
    }

    pub fn library_proof_id(&self) -> Option<LibraryProofId> {
        self.library_proof_id
    }

    pub(crate) fn bind_context_library_proof(&mut self, proof_id: Option<LibraryProofId>) {
        self.context_library_proof_id = proof_id;
    }

    pub fn with_linearity(mut self, linearity: LinearityInfo) -> Self {
        self.linearity = linearity;
        self
    }

    /// Compose a `library` checked program with a `new_code` checked
    /// program into a single whole-program `CheckedProgram`, equivalent
    /// to what `check_ir_program(library_exprs ++ new_code_exprs)` plus
    /// effects + linearity would produce — provided `new_code` was
    /// produced by the `_with_context` variants stacked on `library`.
    ///
    /// This is the seam the cross-process chelis-std typecheck cache's
    /// `chelis build` path uses: `library` is the cached chelis-std
    /// sub-context's `library_checked` and `new_code` is the
    /// `_with_context`-checked non-chelis-std decls + entry. The result
    /// is the one monolithic `CheckedProgram` the `build` lowering
    /// pipeline consumes, without re-inferring chelis-std.
    ///
    /// Composition rule (mirrors the monolithic `library ++ new` shape):
    /// - `annotated_exprs`: `library` exprs followed by `new_code` exprs,
    ///   in that order. Monolithic `check_ir_program` annotates in
    ///   source order, and the linked program places library decls
    ///   before the entry, so this ordering matches.
    /// - `type_env`: union, `new_code` winning on shadow. `new_code`'s
    ///   `type_env` is already unioned with the library's by the
    ///   `_with_context` builder, so this just back-fills any
    ///   library-only entries.
    /// - `linearity`: the two `reusable_inputs_by_offset` maps merged.
    /// - `local_tensor_ascriptions`: library records followed by new-code
    ///   records whose artifact-local identities are remapped above the
    ///   library's identity domain.
    /// - `signature_inference`: the two `functions` maps merged,
    ///   `new_code` winning on a name clash.
    ///
    /// The monolithic-vs-layered acceptance oracle is what proves this
    /// composition is byte-identical to the monolithic path; a
    /// divergence is a compiler-correctness bug, not a tuning knob.
    ///
    /// Returns `None` unless `new_code` retains the exact library proof.
    pub fn compose(library: &CheckedProgram, new_code: &CheckedProgram) -> Option<Self> {
        if library.library_proof_id.is_none()
            || new_code.context_library_proof_id != library.library_proof_id
        {
            return None;
        }
        let mut annotated_exprs =
            Vec::with_capacity(library.annotated_exprs.len() + new_code.annotated_exprs.len());
        annotated_exprs.extend(library.annotated_exprs.iter().cloned());
        annotated_exprs.extend(new_code.annotated_exprs.iter().cloned());

        let mut type_env = new_code.type_env.clone();
        for (name, ty) in &library.type_env {
            type_env.entry(name.clone()).or_insert_with(|| ty.clone());
        }

        let linearity = library.linearity.merged_with(&new_code.linearity);

        let replaced_definitions = top_level_decl_items(&new_code.annotated_exprs)
            .into_iter()
            .filter_map(|expr| {
                let (tag, _, children) = stamped_parts(expr)?;
                (tag == DeepTag::Def)
                    .then(|| children.first().and_then(symbol_name))
                    .flatten()
                    .map(str::to_owned)
            })
            .collect::<BTreeSet<_>>();
        let local_tensor_ascriptions = compose_local_tensor_ascriptions(
            &library.local_tensor_ascriptions,
            &new_code.local_tensor_ascriptions,
            &replaced_definitions,
        )?;

        let mut signature_inference = library.signature_inference.clone();
        for (name, sig) in &new_code.signature_inference.functions {
            signature_inference
                .functions
                .insert(name.clone(), sig.clone());
        }

        let mut type_headers = library.type_headers.clone();
        type_headers.extend_from(&new_code.type_headers);

        // A context-checked new-code program normally already carries the
        // library registry. Back-fill defensively so composition remains
        // total for independently deserialized legacy programs as well.
        let mut adt_registry = new_code.adt_registry.clone();
        for (name, definition) in &library.adt_registry.defs {
            adt_registry
                .defs
                .entry(name.clone())
                .or_insert_with(|| definition.clone());
        }
        for (name, alias) in &library.adt_registry.aliases {
            adt_registry
                .aliases
                .entry(name.clone())
                .or_insert_with(|| alias.clone());
        }

        Some(Self {
            annotated_exprs,
            type_env,
            local_tensor_ascriptions,
            linearity,
            signature_inference,
            type_headers,
            adt_registry,
            infer_stats: InferStats {
                typed_nodes: library.infer_stats.typed_nodes + new_code.infer_stats.typed_nodes,
                total_nodes: library.infer_stats.total_nodes + new_code.infer_stats.total_nodes,
            },
            library_proof_id: new_code.library_proof_id,
            context_library_proof_id: None,
        })
    }
}

/// The sole construction boundary for checked results. It observes the
/// authoritative session sink before adding an invariant diagnostic, so a
/// real root error is never duplicated by the totality backstop.
pub(super) fn finalize_checked_program(
    annotated_exprs: Vec<deep::Expr>,
    type_env: BTreeMap<String, deep::Expr>,
    signature_context: &SignatureInferenceMetadata,
    product: &InferenceProduct,
    infer_stats: InferStats,
    errors: &mut DiagnosticSink<'_>,
) -> CheckedProgram {
    let signature_inference = infer_signature_metadata_with_context_and_headers(
        &annotated_exprs,
        &product.function_inference_plan,
        &type_env,
        signature_context,
        &product.type_headers,
        errors,
    );
    let checked = CheckedProgram {
        annotated_exprs,
        type_env,
        local_tensor_ascriptions: product.local_tensor_ascriptions.clone(),
        linearity: LinearityInfo::default(),
        signature_inference,
        type_headers: product.type_headers.clone(),
        adt_registry: product.adt_registry.clone(),
        infer_stats,
        library_proof_id: None,
        context_library_proof_id: None,
    };

    validate_checked_program_totality(&checked, signature_context, errors);
    checked
}

pub(super) fn validate_checked_program_totality(
    checked: &CheckedProgram,
    signature_context: &SignatureInferenceMetadata,
    errors: &mut DiagnosticSink<'_>,
) {
    if !errors.is_empty() {
        return;
    }
    let mut traces = totality_invariant_traces(signature_context);
    traces.extend(annotated_totality_invariant_traces(
        checked.annotated_exprs(),
    ));
    traces.extend(local_ascription_invariant_traces(
        checked.local_tensor_ascriptions(),
    ));
    traces.extend(totality_invariant_traces(checked.signature_inference()));
    if !traces.is_empty() {
        errors.push(totality_violation_error(&traces));
    }
}

fn local_ascription_invariant_traces(ascriptions: &[CheckedLocalTensorAscription]) -> Vec<String> {
    let mut traces = Vec::new();
    let mut ids = BTreeSet::new();
    for ascription in ascriptions {
        if !ids.insert(ascription.id) {
            traces.push(format!(
                "duplicate local tensor-ascription identity {}",
                ascription.id.get()
            ));
        }
        if ascription.binding_name.is_empty() {
            traces.push(format!(
                "local tensor-ascription {} has an empty binding name",
                ascription.id.get()
            ));
        }
        if ascription
            .declaration_name
            .as_ref()
            .is_some_and(String::is_empty)
        {
            traces.push(format!(
                "local tensor-ascription {} has an empty declaration name",
                ascription.id.get()
            ));
        }
        let Some(declared_dims) = tensor_dims(&ascription.declared_type) else {
            traces.push(format!(
                "local tensor-ascription {} does not carry a tensor type",
                ascription.id.get()
            ));
            continue;
        };
        let mut previous_axis = None;
        for claim in &ascription.outstanding_claims {
            if previous_axis.is_some_and(|axis| claim.axis <= axis) {
                traces.push(format!(
                    "local tensor-ascription {} claims are not in authored axis order",
                    ascription.id.get()
                ));
            }
            previous_axis = Some(claim.axis);
            match declared_dims.get(claim.axis) {
                Some(declared) if declared == &claim.required_extent => {}
                Some(_) => traces.push(format!(
                    "local tensor-ascription {} axis {} claim differs from its declared type",
                    ascription.id.get(),
                    claim.axis
                )),
                None => traces.push(format!(
                    "local tensor-ascription {} claims out-of-rank axis {}",
                    ascription.id.get(),
                    claim.axis
                )),
            }
            if matches!(claim.required_extent, Dim::Wildcard | Dim::Rank(_)) {
                traces.push(format!(
                    "local tensor-ascription {} manufactures a claim from an unclaimed axis",
                    ascription.id.get()
                ));
            }
        }
    }
    traces
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignatureInferenceMetadata {
    pub functions: BTreeMap<String, FunctionSignatureInference>,
}

impl SignatureInferenceMetadata {
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignatureInference {
    pub name: String,
    /// True when this function has an authored or Surf-synthesized `defsig`.
    /// Lowering uses this checker-owned fact to distinguish declared
    /// polymorphism from generalized local-callback inference.
    #[serde(default)]
    pub authored_signature: bool,
    /// The checker-decoded authored signature before body inference sharpens
    /// wildcard dimensions or otherwise specializes the checked function.
    ///
    /// Lowering consumes this record instead of reparsing `defsig` syntax.
    #[serde(default)]
    pub authored_signature_type: Option<Type>,
    pub recursive_cycle: bool,
    pub checked_signature: Type,
    pub display_signature: Type,
    pub params: Vec<ParamSignatureInference>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamSignatureInference {
    pub index: usize,
    pub name: String,
    pub written: bool,
    pub inferred_read_only: bool,
    pub checked_type: Type,
    pub display_type: Type,
}

pub(crate) fn checked_program_with_effect_annotations_in_session(
    original: &CheckedProgram,
    annotated_exprs: Vec<deep::Expr>,
    errors: &mut DiagnosticSink<'_>,
) -> CheckedProgram {
    if !effects_only_rewrite_matches(original.annotated_exprs(), &annotated_exprs) {
        errors.push(CheckError::new(
            CheckErrorKind::Other,
            "checked-program effects-only reannotation changed a span, atom, child, or non-`effects` metadata entry"
                .to_string(),
            vec![
                "Run type checking again for structural, body, type, `eff`, or source-span changes"
                    .to_string(),
            ],
        ));
    }

    let checked = CheckedProgram {
        annotated_exprs,
        type_env: original.type_env.clone(),
        local_tensor_ascriptions: original.local_tensor_ascriptions.clone(),
        linearity: original.linearity.clone(),
        signature_inference: original.signature_inference.clone(),
        type_headers: original.type_headers.clone(),
        adt_registry: original.adt_registry.clone(),
        infer_stats: original.infer_stats,
        library_proof_id: original.library_proof_id,
        context_library_proof_id: original.context_library_proof_id,
    };
    validate_checked_program_totality(&checked, original.signature_inference(), errors);
    checked
}

pub(super) fn effects_only_rewrite_matches(
    original: &[deep::Expr],
    candidate: &[deep::Expr],
) -> bool {
    original.len() == candidate.len()
        && original
            .iter()
            .zip(candidate)
            .all(|(before, after)| effects_only_expr_matches(before, after))
        && chelis_deep::metadata::validate_metadata(candidate).is_ok()
}

pub(super) fn effects_only_expr_matches(before: &deep::Expr, after: &deep::Expr) -> bool {
    stack_guard!("effects_only_expr_matches", before, false);
    // Strict representation equality modulo derived effects. Each carrier is
    // compared only with the same carrier; `_ => false` explicitly rejects a
    // carrier mismatch.
    match (before, after) {
        (deep::Expr::Atom(before_atom, before_span), deep::Expr::Atom(after_atom, after_span)) => {
            before_atom == after_atom && before_span == after_span
        }
        (deep::Expr::Map(before_map, before_span), deep::Expr::Map(after_map, after_span)) => {
            before_span == after_span
                && metadata_matches_except_effects(before_map, after_map, false)
        }
        (
            deep::Expr::MetaExpr(before_meta, before_span),
            deep::Expr::MetaExpr(after_meta, after_span),
        ) => {
            before_span == after_span
                && metadata_matches_except_effects(
                    &before_meta.metadata,
                    &after_meta.metadata,
                    false,
                )
                && effects_only_expr_matches(&before_meta.expr, &after_meta.expr)
        }
        (deep::Expr::Node(before_node, before_span), deep::Expr::Node(after_node, after_span)) => {
            before_span == after_span
                && before_node.tag() == after_node.tag()
                && metadata_matches_except_effects(before_node.meta(), after_node.meta(), true)
                && before_node.children_slice().len() == after_node.children_slice().len()
                && before_node
                    .children_slice()
                    .iter()
                    .zip(after_node.children_slice())
                    .all(|(before_child, after_child)| {
                        effects_only_expr_matches(before_child, after_child)
                    })
        }
        (
            deep::Expr::BareList(before_elements, before_span),
            deep::Expr::BareList(after_elements, after_span),
        ) => {
            before_span == after_span
                && before_elements.len() == after_elements.len()
                && before_elements
                    .iter()
                    .zip(after_elements)
                    .all(|(before_child, after_child)| {
                        effects_only_expr_matches(before_child, after_child)
                    })
        }
        (deep::Expr::UnknownForm(before_data), deep::Expr::UnknownForm(after_data)) => {
            before_data.head == after_data.head
                && before_data.span == after_data.span
                && metadata_matches_except_effects(&before_data.meta, &after_data.meta, true)
                && before_data.children.len() == after_data.children.len()
                && before_data.children.iter().zip(&after_data.children).all(
                    |(before_child, after_child)| {
                        effects_only_expr_matches(before_child, after_child)
                    },
                )
        }
        _ => false,
    }
}

/// Compare the admitted representation after removing only derived effect sets.
/// Scalar choices, structural containers, dtype bounds, and preserved source
/// remain ordinary typed equality checks; only live expression subtrees recurse.
fn metadata_matches_except_effects(
    before: &deep::Metadata,
    after: &deep::Metadata,
    ignore_effects: bool,
) -> bool {
    match (
        metadata_without_derived_effects(before, ignore_effects),
        metadata_without_derived_effects(after, ignore_effects),
    ) {
        (Ok(before), Ok(after)) => before == after,
        _ => false,
    }
}

struct EffectComparisonError;
impl From<chelis_deep::metadata::MetadataError> for EffectComparisonError {
    fn from(_: chelis_deep::metadata::MetadataError) -> Self {
        Self
    }
}
fn metadata_without_derived_effects(
    meta: &deep::Metadata,
    remove_here: bool,
) -> Result<deep::Metadata, EffectComparisonError> {
    let mut result =
        meta.try_map_expressions(&mut |expr, _| expression_without_derived_effects(expr))?;
    if remove_here {
        result.remove(chelis_deep::annotations::MetadataKey::Effects);
    }
    Ok(result)
}
fn expression_without_derived_effects(
    expr: &deep::Expr,
) -> Result<deep::Expr, EffectComparisonError> {
    stack_guard!(
        "expression_without_derived_effects",
        expr,
        Err(EffectComparisonError)
    );
    // The exhaustive transformer rebuilds every expression as the same carrier
    // while removing only derived effects.
    Ok(match expr {
        deep::Expr::Atom(..) => expr.clone(),
        deep::Expr::Map(meta, span) => {
            deep::Expr::Map(metadata_without_derived_effects(meta, false)?, *span)
        }
        deep::Expr::MetaExpr(meta, span) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                metadata: metadata_without_derived_effects(&meta.metadata, false)?,
                expr: Box::new(expression_without_derived_effects(&meta.expr)?),
            },
            *span,
        ),
        deep::Expr::Node(node, span) => deep::Expr::Node(
            Box::new(
                chelis_deep::node::Node::try_new(
                    node.tag(),
                    metadata_without_derived_effects(node.meta(), true)?,
                    node.children_slice()
                        .iter()
                        .map(expression_without_derived_effects)
                        .collect::<Result<_, _>>()?,
                )
                .map_err(|_| EffectComparisonError)?,
            ),
            *span,
        ),
        deep::Expr::BareList(values, span) => deep::Expr::BareList(
            values
                .iter()
                .map(expression_without_derived_effects)
                .collect::<Result<_, _>>()?,
            *span,
        ),
        deep::Expr::UnknownForm(data) => deep::Expr::UnknownForm(Box::new(deep::UnknownFormData {
            head: data.head.clone(),
            meta: metadata_without_derived_effects(&data.meta, true)?,
            children: data
                .children
                .iter()
                .map(expression_without_derived_effects)
                .collect::<Result<_, _>>()?,
            span: data.span,
        })),
    })
}
