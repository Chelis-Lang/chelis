//! Checked-program construction, metadata ownership, and totality finalization.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use crate::context::LibraryProofId;

#[derive(Clone)]
pub(super) struct DeclaredSigMetadata {
    pub(super) param_types: Vec<deep::Expr>,
    pub(super) binders: UnordSet<String>,
    /// Exact dtype-family capabilities admitted by the typed annotation boundary.
    pub(super) dtype_bounds: UnordMap<String, chelis_deep::DtypeFamily>,
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
pub(super) enum DeferredShapeRule {
    Matmul,
    Reduction {
        name: String,
    },
    Expand {
        /// `expand` or `insert`. The replay must reach the same route arm the
        /// original call did, and the two differ in their result forms.
        builtin: &'static str,
        axis_is_dim_name: bool,
        size_class: SizeClass,
        env: Box<Env>,
    },
    LayerNorm,
    Conv,
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

    pub(super) fn deferred_shape_checkpoint(&self) -> u64 {
        self.next_deferred_shape_id
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
                        required.family_name(),
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

    /// chelis#1836, chelis#2523: tie `projected` to element `index` of
    /// `source`, a target still unresolved at the access. The derivation is a
    /// ledger entry like a suspended call, so it is decided when the target
    /// binds, keeps its lambda monomorphic until then ([04-INF-1]), and is
    /// decided at the declaration boundary if the target never binds.
    pub(super) fn defer_tuple_projection(&mut self, source: Type, index: usize, projected: Type) {
        self.defer_shape_check(
            DeferredShapeRule::Derivation(TypeDerivation::TupleProjection { index }),
            Vec::new(),
            vec![source],
            projected,
        );
    }

    /// [`Self::defer_tuple_projection`] for a record field.
    pub(super) fn defer_record_field(&mut self, source: Type, field: String, projected: Type) {
        self.defer_shape_check(
            DeferredShapeRule::Derivation(TypeDerivation::RecordField { field }),
            Vec::new(),
            vec![source],
            projected,
        );
    }

    pub(super) fn has_pending_shape_check_since(&self, checkpoint: u64) -> bool {
        self.deferred_shape_checks
            .iter()
            .any(|check| check.id >= checkpoint)
    }

    /// chelis#1512: the ledger identity of one call.
    ///
    /// Identity is the address of the `app` node in the tree the traversal is
    /// walking. One case is not that tree: during a `PostApp` replay the
    /// caller holds the ledger's own clone, which is dropped when the replay
    /// iteration ends, so this translates that clone back to the original
    /// site. Every key this returns therefore names a node that outlives the
    /// entry, which is the invariant the old spelling documented without
    /// holding: a route that re-registered during a replay stored a freed
    /// address, and a later allocation reusing it would make
    /// [`Self::has_post_app_check_for`] answer for an unrelated live call
    /// (round 2 P2-1, latent: 0 collisions in 2,000 runs, never observed).
    pub(super) fn post_app_key(&self, node: &DeepNode) -> usize {
        let addr = std::ptr::from_ref(node).addr();
        match self.replaying_post_app {
            Some((clone_addr, original_site)) if clone_addr == addr => original_site,
            _ => addr,
        }
    }

    /// The replay kind of every `PostApp` entry for this call, in ledger order.
    ///
    /// chelis#1512: the route and dtype registrations share one key, and which
    /// one survives is a decision no builtin currently exercises, because no
    /// callee reaches both a dtype validator and a `site.register` route arm.
    /// The invariant is still real, so it is asserted here rather than left to
    /// the first callee that does.
    #[cfg(test)]
    pub(super) fn post_app_replays_for(&self, node: &DeepNode) -> Vec<PostAppReplay> {
        let key = self.post_app_key(node);
        self.deferred_shape_checks
            .iter()
            .filter_map(|check| match &check.rule {
                DeferredShapeRule::PostApp { replay, site, .. } if *site == key => Some(*replay),
                _ => None,
            })
            .collect()
    }

    /// Is this call already suspended, under either replay kind?
    pub(super) fn has_post_app_check_for(&self, node: &DeepNode) -> bool {
        let key = self.post_app_key(node);
        self.deferred_shape_checks.iter().any(
            |check| matches!(&check.rule, DeferredShapeRule::PostApp { site, .. } if *site == key),
        )
    }

    /// Whether any call is suspended on this product's ledger. The
    /// rigid-binder decision (chelis#2216) asks it of the scratch product a
    /// replay ran on, to see a call that suspended again.
    pub(super) fn has_post_app_checks(&self) -> bool {
        self.deferred_shape_checks
            .iter()
            .any(|check| matches!(check.rule, DeferredShapeRule::PostApp { .. }))
    }

    /// Is this call already suspended for a FULL route replay?
    ///
    /// A dtype-admissibility entry does not answer yes: the route registration
    /// replaces it rather than standing down for it, because re-entering
    /// `finish_unified_app` runs those same validators on its way to the route.
    pub(super) fn has_route_check_for(&self, node: &DeepNode) -> bool {
        let key = self.post_app_key(node);
        self.deferred_shape_checks.iter().any(|check| {
            matches!(
                &check.rule,
                DeferredShapeRule::PostApp {
                    replay: PostAppReplay::Route,
                    site,
                    ..
                } if *site == key
            )
        })
    }

    /// chelis#1512: drop a suspended `PostApp` decision for this call.
    ///
    /// A call that FAILS on the eager pass has nothing left to decide. It may
    /// still have registered a suspension a moment earlier, for a different
    /// operand that was not the one it rejected on, and replaying it would
    /// re-enter the whole route and re-report the rejection the eager pass
    /// already made. Cancelling the deferral is what fixes that, rather than
    /// filtering the duplicate out afterwards: a filter keyed on the message
    /// cannot tell a re-report from a second call that legitimately fails the
    /// same way, and these diagnostics carry no span to tell them apart.
    pub(super) fn cancel_post_app_check_for(&mut self, node: &DeepNode) {
        let key = self.post_app_key(node);
        self.deferred_shape_checks.retain(
            |check| !matches!(&check.rule, DeferredShapeRule::PostApp { site, .. } if *site == key),
        );
    }

    /// A rejected application supplied an operand, but its checked family
    /// rejected that binding. The same monomorphic parameter must not also be
    /// diagnosed as never applied. Unrelated pending parameters remain live.
    pub(super) fn cancel_shape_checks_for_failed_family_call(
        &mut self,
        callee: &Type,
        subst: &Subst,
        failure: ErrorWitness,
    ) {
        let Type::Fn(params, _) = subst.apply(callee) else {
            return;
        };
        let failed_parameters: Vec<_> = params.iter().flat_map(crate::env::free_tvars).collect();
        for contract in &mut self.inferred_admission_contracts {
            if crate::env::free_tvars(&subst.apply(&Type::Var(contract.variable)))
                .iter()
                .any(|variable| failed_parameters.contains(variable))
            {
                // Retain the slot so lexical checkpoints remain stable.
                // Only an already-reported application can discharge it.
                contract.failed_application = Some(failure);
            }
        }
        self.deferred_shape_checks.retain(|check| {
            !check.arg_tys.iter().any(|argument| {
                crate::env::free_tvars(&subst.apply(argument))
                    .iter()
                    .any(|variable| failed_parameters.contains(variable))
            })
        });
    }

    /// Give an already-suspended specialized shape rule its documented
    /// diagnostic precedence when a later call structurally binds its operand
    /// but fails the checked dtype family.
    ///
    /// Ordinary unification cannot publish that binding: the family rejection
    /// rolls the transaction back before the deferred rule becomes ready.
    /// Build a restriction-free structural overlay instead, apply it only to
    /// the pending rule's arguments, and execute the rule against cloned solver
    /// state. A shape-valid preview emits nothing and leaves the family error
    /// authoritative. A rejecting preview removes the exact deferred entry so
    /// the declaration boundary cannot report it again.
    pub(super) fn report_preceding_shape_error_for_failed_family_call(
        &mut self,
        callee: &Type,
        arguments: &[Type],
        vg: &VarGen,
        subst: &Subst,
        errors: &mut DiagnosticSink<'_>,
    ) -> Option<Type> {
        let Type::Fn(parameters, _) = subst.apply(callee) else {
            return None;
        };
        if parameters.len() != arguments.len() {
            return None;
        }

        // Restrictions live in `subst`, not in `Type`, so a fresh
        // substitution records exactly the structural relation that the
        // failed call established without weakening the real solver.
        let mut structural = Subst::default();
        for (parameter, argument) in parameters.iter().zip(arguments) {
            if unify(
                &subst.apply(parameter),
                &subst.apply(argument),
                &mut structural,
            )
            .is_err()
            {
                return None;
            }
        }

        let failed_parameters: Vec<_> = parameters
            .iter()
            .flat_map(crate::env::free_tvars)
            .filter(|variable| subst.tvar_restriction(*variable).is_some())
            .collect();
        let candidates: Vec<_> = self
            .deferred_shape_checks
            .iter()
            .filter_map(|check| {
                let shares_failed_parameter = check.arg_tys.iter().any(|argument| {
                    crate::env::free_tvars(&subst.apply(argument))
                        .iter()
                        .any(|variable| failed_parameters.contains(variable))
                });
                let DeferredShapeRule::ShapeRoute { route, node, kids } = &check.rule else {
                    return None;
                };
                shares_failed_parameter.then(|| {
                    (
                        check.id,
                        route.clone(),
                        node.clone(),
                        kids.clone(),
                        check.arg_tys.clone(),
                    )
                })
            })
            .collect();
        for (id, route, node, kids, pending_arguments) in candidates {
            let settled: Vec<_> = pending_arguments
                .iter()
                .map(|ty| structural.apply(&subst.apply(ty)))
                .collect();
            if settled
                .iter()
                .any(|ty| shape_operand_awaits_binding(ty, &structural))
            {
                continue;
            }

            let checkpoint = errors.checkpoint();
            let mut trial_vg = vg.clone();
            let mut trial_subst = subst.clone();
            let result = check_shape_route_signature(
                &route,
                &node,
                &kids,
                &settled,
                &mut trial_vg,
                &mut trial_subst,
                errors,
            );
            if errors.iter_since(checkpoint).next().is_none() {
                continue;
            }
            self.deferred_shape_checks.retain(|check| check.id != id);
            return Some(result);
        }
        None
    }

    pub(super) fn defer_shape_check(
        &mut self,
        rule: DeferredShapeRule,
        arg_exprs: Vec<deep::Expr>,
        arg_tys: Vec<Type>,
        result_ty: Type,
    ) {
        #[cfg(test)]
        if let DeferredShapeRule::PostApp { site, .. } = &rule {
            record_post_app_key(*site, self.replaying_post_app.is_some());
        }
        self.push_deferred_shape_check(rule, arg_exprs, arg_tys, result_ty);
    }

    fn push_deferred_shape_check(
        &mut self,
        rule: DeferredShapeRule,
        arg_exprs: Vec<deep::Expr>,
        arg_tys: Vec<Type>,
        result_ty: Type,
    ) {
        let id = self.next_deferred_shape_id;
        self.next_deferred_shape_id += 1;
        let owner = self.entry_owner();
        self.deferred_shape_checks.push(DeferredShapeCheck {
            id,
            owner,
            rule,
            arg_exprs,
            arg_tys,
            result_ty,
        });
    }

    /// Replay shape checks whose previously-free input types have now been
    /// bound by an application. The same checker functions own both the
    /// immediate and deferred paths, so their semantics cannot drift.
    pub(super) fn replay_ready_shape_checks(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        if self.replaying_shape_checks {
            return;
        }
        self.replaying_shape_checks = true;
        // A `PostApp` replay can bind the operand another suspended check was
        // waiting on, and the pass has an order. Repeat until a pass settles
        // nothing new, so a check that became ready mid-pass is not carried
        // to `finish_deferred_shape_checks` and reported as never bound.
        loop {
            let before = self.deferred_shape_checks.len();
            self.replay_ready_shape_checks_once(vg, subst, adt_reg, errors);
            if self.deferred_shape_checks.len() >= before {
                break;
            }
        }
        self.replaying_shape_checks = false;
    }

    fn replay_ready_shape_checks_once(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        let checks = std::mem::take(&mut self.deferred_shape_checks);
        let prior_owner = self.replaying_owner.take();
        for check in checks {
            if check
                .arg_tys
                .iter()
                .any(|ty| shape_operand_awaits_binding(ty, subst))
            {
                self.deferred_shape_checks.push(check);
                continue;
            }
            // An entry the replay registers again belongs to this one's owner.
            self.replaying_owner.clone_from(&check.owner);

            let resolved = match &check.rule {
                DeferredShapeRule::Matmul => {
                    check_matmul_signature(&check.arg_tys, &check.result_ty, subst, errors)
                }
                DeferredShapeRule::Reduction { name } => check_reduction_signature(
                    name,
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    subst,
                    errors,
                ),
                DeferredShapeRule::Expand {
                    builtin,
                    axis_is_dim_name,
                    size_class,
                    env,
                } => check_expand_signature(
                    builtin,
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    *axis_is_dim_name,
                    *size_class,
                    env,
                    subst,
                    errors,
                ),
                DeferredShapeRule::LayerNorm => {
                    check_layer_norm_signature(&check.arg_tys, &check.result_ty, vg, subst, errors)
                }
                DeferredShapeRule::Conv => check_conv_signature(
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    vg,
                    subst,
                    errors,
                ),
                DeferredShapeRule::ScatterElements { node } => {
                    let kids = node.children_slice();
                    check_scatter_elements(
                        node,
                        kids,
                        &check.arg_tys,
                        check.result_ty.clone(),
                        subst,
                        errors,
                    )
                }
                DeferredShapeRule::ShapeRoute { route, node, kids } => {
                    let settled: Vec<Type> =
                        check.arg_tys.iter().map(|ty| subst.apply(ty)).collect();
                    let produced =
                        check_shape_route_signature(route, node, kids, &settled, vg, subst, errors);
                    reconcile_replayed_result(
                        &route.builtin(),
                        &check.result_ty,
                        produced,
                        subst,
                        errors,
                    )
                }
                DeferredShapeRule::Derivation(derivation) => {
                    let decided = resolve_type_derivation(
                        derivation,
                        &check.arg_tys[0],
                        &check.result_ty,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                    );
                    // The readiness test above already waits on a variable
                    // target, so this keeps an undecided entry only in
                    // principle; an entry is never dropped undecided.
                    if !decided {
                        self.deferred_shape_checks.push(check);
                    }
                    continue;
                }
                DeferredShapeRule::PostApp {
                    replay,
                    site,
                    node,
                    kids,
                    func_name,
                    env,
                } => {
                    let settled: Vec<Type> =
                        check.arg_tys.iter().map(|ty| subst.apply(ty)).collect();
                    let call = PostAppCall {
                        replay: *replay,
                        site: *site,
                        node,
                        kids,
                        func_name,
                        env,
                    };
                    self.replay_post_app(
                        call,
                        settled,
                        &check.result_ty,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        false,
                    );
                    continue;
                }
            };
            let _ = resolved;
        }
        self.replaying_owner = prior_owner;
    }

    /// Re-decide one suspended `PostApp` call against the operand types
    /// `settled`.
    ///
    /// The ready replay above and the rigid-binder decision at the declaration
    /// boundary (chelis#2216) both run this one function, so a call cannot be
    /// decided one way when its operand binds and another way at a binder's
    /// instantiation.
    ///
    /// `resuspend` makes a dtype replay record a fresh suspension on this
    /// product when an operand is still unresolved, as a route replay always
    /// does. The rigid-binder decision asks for it, to see a call that cannot
    /// be decided at an arbitrary type; the ready replay does not.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn replay_post_app(
        &mut self,
        call: PostAppCall<'_>,
        settled: Vec<Type>,
        result_ty: &Type,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
        resuspend: bool,
    ) {
        match call.replay {
            PostAppReplay::DtypeAdmissibility => replay_dtype_admissibility(
                call.node,
                call.kids,
                call.func_name,
                call.env,
                &settled,
                vg,
                subst,
                errors,
                self,
                resuspend,
            ),
            PostAppReplay::Route => {
                // chelis#1512 round 2 P2-1: `node` is the ledger's own clone
                // and dies with this replay. Carry the original site so a
                // route that re-registers inside the replay keys its entry by
                // the live node, not by this clone.
                self.replaying_post_app = Some((std::ptr::from_ref(call.node).addr(), call.site));
                let replayed = finish_unified_app(
                    call.node,
                    call.kids,
                    Some(call.func_name.to_string()),
                    settled,
                    result_ty.clone(),
                    call.env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    self,
                    None,
                );
                self.replaying_post_app = None;
                reconcile_replayed_result(call.func_name, result_ty, replayed, subst, errors);
            }
        }
    }

    /// Acceptance boundary for bind-on-first-use shape lambdas, and for every
    /// other obligation still open when the declaration closes. Every ledger
    /// entry left after the ready replay is decided or reported here; none is
    /// dropped (chelis#2518, chelis#2523).
    pub(super) fn finish_deferred_shape_checks(
        &mut self,
        env: &Env,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        self.replay_ready_shape_checks(vg, subst, adt_reg, errors);
        self.finish_admission_contracts(subst, errors);
        // Every entry of a shape rule of its own (`sum`, `matmul`, ...) still
        // unresolved here is rejected below, and the dtype replay of the same
        // call waits on the same operand. Deciding that replay too would report
        // one operand twice, so an operand a shape entry waits on is left to it.
        let checks = std::mem::take(&mut self.deferred_shape_checks);
        let shape_rule_operands: Vec<TypeVar> = checks
            .iter()
            .filter(|check| {
                !matches!(
                    check.rule,
                    DeferredShapeRule::PostApp { .. } | DeferredShapeRule::Derivation(_)
                )
            })
            .flat_map(|check| &check.arg_tys)
            .filter_map(|ty| match type_for_readonly_check(ty, subst) {
                Type::Var(var) => Some(var),
                _ => None,
            })
            .collect();
        for check in checks {
            let operation = match check.rule {
                DeferredShapeRule::Matmul => "matmul".to_string(),
                DeferredShapeRule::Reduction { name } => name,
                DeferredShapeRule::Expand { builtin, .. } => builtin.to_string(),
                DeferredShapeRule::LayerNorm => "layer_norm".to_string(),
                DeferredShapeRule::Conv => "conv".to_string(),
                DeferredShapeRule::ScatterElements { .. } => "scatter_elements".to_string(),
                DeferredShapeRule::ShapeRoute { route, .. } => route.builtin(),
                // chelis#2216, chelis#2518, chelis#2523: a suspended call or a
                // deferred access is decided at the instantiations of the
                // operand types that never bound: an authored binder's
                // ([04-INF-6]), or a flexible variable's that the declaration
                // never determined ([04-INF-1], [04-INF-9]).
                DeferredShapeRule::PostApp {
                    replay,
                    site,
                    ref node,
                    ref kids,
                    ref func_name,
                    env: ref call_env,
                } => {
                    let call = PostAppCall {
                        replay,
                        site,
                        node,
                        kids,
                        func_name,
                        env: call_env,
                    };
                    decide_at_boundary(
                        BoundaryObligation::Call(call),
                        &check.arg_tys,
                        &check.result_ty,
                        &shape_rule_operands,
                        check.owner.as_deref(),
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                    );
                    continue;
                }
                DeferredShapeRule::Derivation(ref derivation) => {
                    decide_at_boundary(
                        BoundaryObligation::Derivation(derivation),
                        &check.arg_tys,
                        &check.result_ty,
                        &shape_rule_operands,
                        check.owner.as_deref(),
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                    );
                    continue;
                }
            };
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "unresolved `{operation}` shape obligation at declaration boundary: \
                     add an outer-constructor parameter annotation or apply the lambda before \
                     the declaration boundary"
                ),
                vec![
                    "A result annotation does not determine an unresolved parameter constructor; top-level declarations do not borrow binding sites from later declarations"
                        .to_string(),
                ],
            ));
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

/// chelis#1512: unify a replayed route's answer with the type the suspended
/// call already published to its consumer, and report loudly when the two
/// disagree.
///
/// A route that builds its result out of band rather than through unification
/// (chelis#1265's class) hands its answer back without ever meeting the
/// declaration the call was accepted against. That is the wrong-answer
/// witness this issue was filed on, so the reconciliation is here, once, for
/// every replayed route rather than per route.
///
/// A replay that REJECTED returns `Type::Error`, which unifies with anything,
/// so a rejected call reports once and not twice.
///
/// `pub(super)` so `ReconcileMutationCase` can drive it directly from
/// `session.rs`: the disagreement branch is the one place where a relocated
/// decision could quietly overwrite a published type, and a test that can only
/// reach it through a whole program cannot show that the returned value is the
/// PRODUCED one.
pub(super) fn reconcile_replayed_result(
    operation: &str,
    published: &Type,
    produced: Type,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if let Err(error) = unify(published, &produced, subst) {
        if matches!(error.kind, crate::unify::TypeErrorKind::DtypeFamilyMismatch) {
            errors.push(error.into());
            return produced;
        }
        let expected = subst.apply(published);
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!(
                "`{operation}` result does not match the type this call produces once its \
                 operand is known: expected {expected}, got {produced}"
            ),
            vec![],
        ));
    }
    produced
}

/// A semantic shape rule can decide symbolic tensor dimensions and precision
/// variables. It must wait only while an operand's *type constructor* is still
/// unknown; treating every free variable as pending would reject legitimate
/// rank/dtype-polymorphic signatures at their declaration boundary.
///
/// chelis#1836: this is the ONE readiness predicate. The `app_post` routes
/// used to suspend on a narrower provenance question instead -- does the
/// unresolved constructor descend from an annotation-free lambda parameter --
/// which answered no for a `pat-tuple` element on an unresolved scrutinee, a
/// field of an unresolved record target, and a chelis#1577 gate's result. Each
/// of those took the route's eager `Type::Var` arm, published the call's own
/// result variable, and let the declaration bind it to any shape at all. A
/// route that suspends on the same condition it resumes on cannot grow that
/// hole again for a fourth provenance, so the provenance set is gone rather
/// than extended.
pub(super) fn shape_operand_awaits_binding(ty: &Type, subst: &Subst) -> bool {
    match subst.apply(ty) {
        Type::Var(_) => true,
        Type::Ref(inner) => shape_operand_awaits_binding(&inner, subst),
        _ => false,
    }
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
