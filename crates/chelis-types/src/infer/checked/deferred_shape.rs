//! The deferred shape ledger.
//!
//! One responsibility: suspend a semantic shape or route rule whose operand
//! type is not yet known, replay it when the operand binds, reconcile the
//! replayed answer with the type the call already published, and decide
//! whatever is still pending at the declaration boundary. The entry types
//! stay beside [`InferenceProduct`] in `checked.rs`.

use super::*;

impl InferenceProduct {
    pub(in crate::infer) fn deferred_shape_checkpoint(&self) -> u64 {
        self.next_deferred_shape_id
    }

    /// chelis#1836, chelis#2523: tie `projected` to element `index` of
    /// `source`, a target still unresolved at the access. The derivation is a
    /// ledger entry like a suspended call, so it is decided when the target
    /// binds, keeps its lambda monomorphic until then ([04-INF-1]), and is
    /// decided at the declaration boundary if the target never binds.
    pub(in crate::infer) fn defer_tuple_projection(
        &mut self,
        source: Type,
        index: usize,
        projected: Type,
    ) {
        self.defer_shape_check(
            DeferredShapeRule::Derivation(TypeDerivation::TupleProjection { index }),
            Vec::new(),
            vec![source],
            projected,
        );
    }

    /// [`Self::defer_tuple_projection`] for a record field.
    pub(in crate::infer) fn defer_record_field(
        &mut self,
        source: Type,
        field: String,
        projected: Type,
    ) {
        self.defer_shape_check(
            DeferredShapeRule::Derivation(TypeDerivation::RecordField { field }),
            Vec::new(),
            vec![source],
            projected,
        );
    }

    pub(in crate::infer) fn has_pending_shape_check_since(&self, checkpoint: u64) -> bool {
        self.deferred_shape_checks.iter().any(|check| {
            check.id >= checkpoint && !matches!(check.rule, DeferredShapeRule::ResultJoin { .. })
        })
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
    pub(in crate::infer) fn post_app_key(&self, node: &DeepNode) -> usize {
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
    pub(in crate::infer) fn post_app_replays_for(&self, node: &DeepNode) -> Vec<PostAppReplay> {
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
    pub(in crate::infer) fn has_post_app_check_for(&self, node: &DeepNode) -> bool {
        let key = self.post_app_key(node);
        self.deferred_shape_checks.iter().any(
            |check| matches!(&check.rule, DeferredShapeRule::PostApp { site, .. } if *site == key),
        )
    }

    /// Whether any call is suspended on this product's ledger. The
    /// rigid-binder decision (chelis#2216) asks it of the scratch product a
    /// replay ran on, to see a call that suspended again.
    pub(in crate::infer) fn has_post_app_checks(&self) -> bool {
        self.deferred_shape_checks
            .iter()
            .any(|check| matches!(check.rule, DeferredShapeRule::PostApp { .. }))
    }

    /// Is this call already suspended for a FULL route replay?
    ///
    /// A dtype-admissibility entry does not answer yes: the route registration
    /// replaces it rather than standing down for it, because re-entering
    /// `finish_unified_app` runs those same validators on its way to the route.
    pub(in crate::infer) fn has_route_check_for(&self, node: &DeepNode) -> bool {
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
    pub(in crate::infer) fn cancel_post_app_check_for(&mut self, node: &DeepNode) {
        let key = self.post_app_key(node);
        self.deferred_shape_checks.retain(
            |check| !matches!(&check.rule, DeferredShapeRule::PostApp { site, .. } if *site == key),
        );
    }

    /// A rejected application supplied an operand, but its checked family
    /// rejected that binding. The same monomorphic parameter must not also be
    /// diagnosed as never applied. Unrelated pending parameters remain live.
    pub(in crate::infer) fn cancel_shape_checks_for_failed_family_call(
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
    pub(in crate::infer) fn report_preceding_shape_error_for_failed_family_call(
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
                let DeferredShapeRule::ShapeRoute {
                    route,
                    node,
                    kids,
                    location,
                } = &check.rule
                else {
                    return None;
                };
                shares_failed_parameter.then(|| {
                    (
                        check.id,
                        route.clone(),
                        node.clone(),
                        kids.clone(),
                        location.clone(),
                        check.arg_tys.clone(),
                    )
                })
            })
            .collect();
        for (id, route, node, kids, location, pending_arguments) in candidates {
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
                CheckSite::Deferred(location.as_ref()),
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

    pub(in crate::infer) fn defer_shape_check(
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

    /// A nested tensor join or `to_tensor` consumes its list before the outer
    /// constructor completes. Flush the pending equations at that boundary.
    pub(in crate::infer) fn replay_ready_shape_checks_at_aggregate_boundary(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        let depth = std::mem::take(&mut self.aggregate_arg_depth);
        self.replay_ready_shape_checks(vg, subst, adt_reg, errors);
        self.aggregate_arg_depth = depth;
    }

    /// Replay shape checks whose previously-free input types have now been
    /// bound by an application. The same checker functions own both the
    /// immediate and deferred paths, so their semantics cannot drift.
    pub(in crate::infer) fn replay_ready_shape_checks(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        if self.replaying_shape_checks || self.aggregate_arg_depth > 0 {
            return;
        }
        self.import_result_constraints(subst);
        self.replaying_shape_checks = true;
        // A `PostApp` replay can bind the operand another suspended check was
        // waiting on, and the pass has an order. Repeat until a pass settles
        // nothing new, so a check that became ready mid-pass is not carried
        // to `finish_deferred_shape_checks` and reported as never bound.
        loop {
            let before = self.deferred_shape_checks.len() + self.result_type_constraints.len();
            let joined = self.replay_result_joins(vg, subst, errors);
            let progressed = self.replay_ready_shape_checks_once(vg, subst, adt_reg, errors);
            let imported = self.import_result_constraints(subst);
            self.replay_result_type_constraints(subst, errors);
            if !joined
                && !imported
                && !progressed
                && self.deferred_shape_checks.len() + self.result_type_constraints.len() >= before
            {
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
    ) -> bool {
        let checks = std::mem::take(&mut self.deferred_shape_checks);
        let prior_owner = self.replaying_owner.take();
        let mut progressed = false;
        for check in checks {
            if matches!(check.rule, DeferredShapeRule::ResultJoin { .. }) {
                self.deferred_shape_checks.push(check);
                continue;
            }
            if !matches!(
                check.rule,
                DeferredShapeRule::Derivation(TypeDerivation::Vmap { .. })
            ) && check
                .arg_tys
                .iter()
                .any(|ty| shape_operand_awaits_binding(ty, subst))
            {
                self.deferred_shape_checks.push(check);
                continue;
            }
            // chelis#2587: a suspended `eq` or `neq` resumes on the condition
            // it suspended on, a reachable field type that is still a variable.
            if let DeferredShapeRule::PostApp {
                replay: PostAppReplay::DtypeAdmissibility,
                func_name,
                ..
            } = &check.rule
                && equality_call_awaits_binding(func_name, &check.arg_tys, subst, adt_reg)
            {
                self.deferred_shape_checks.push(check);
                continue;
            }
            // An entry the replay registers again belongs to this one's owner.
            self.replaying_owner.clone_from(&check.owner);

            let resolved = match &check.rule {
                DeferredShapeRule::ResultJoin { .. } => {
                    unreachable!("joins replay with the complete producer graph")
                }
                DeferredShapeRule::Matmul { location } => check_matmul_signature(
                    CheckSite::Deferred(location.as_ref()),
                    &check.arg_tys,
                    &check.result_ty,
                    subst,
                    errors,
                ),
                DeferredShapeRule::Reduction { name, location } => check_reduction_signature(
                    CheckSite::Deferred(location.as_ref()),
                    name,
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    vg,
                    subst,
                    errors,
                ),
                DeferredShapeRule::Expand {
                    builtin,
                    axis_is_dim_name,
                    env,
                    location,
                } => check_expand_signature(
                    CheckSite::Deferred(location.as_ref()),
                    builtin,
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    *axis_is_dim_name,
                    env,
                    subst,
                    errors,
                ),
                DeferredShapeRule::LayerNorm { location } => check_layer_norm_signature(
                    CheckSite::Deferred(location.as_ref()),
                    &check.arg_tys,
                    &check.result_ty,
                    vg,
                    subst,
                    errors,
                ),
                DeferredShapeRule::Conv { location } => check_conv_signature(
                    CheckSite::Deferred(location.as_ref()),
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
                DeferredShapeRule::ShapeRoute {
                    route,
                    node,
                    kids,
                    location,
                } => {
                    let settled: Vec<Type> =
                        check.arg_tys.iter().map(|ty| subst.apply(ty)).collect();
                    let produced = check_shape_route_signature(
                        CheckSite::Deferred(location.as_ref()),
                        route,
                        node,
                        kids,
                        &settled,
                        vg,
                        subst,
                        errors,
                    );
                    reconcile_replayed_result(
                        &route.builtin(),
                        &check.result_ty,
                        produced,
                        subst,
                        errors,
                    )
                }
                DeferredShapeRule::Derivation(derivation) => {
                    let product: &Self = self;
                    let step = resolve_type_derivation(
                        derivation,
                        &check.arg_tys[0],
                        &check.result_ty,
                        &|ty, subst| product.awaits_group_completion(ty, subst),
                        true,
                        product.group_links_in_progress(),
                        vg,
                        subst,
                        adt_reg,
                        errors,
                    );
                    match step {
                        DerivationStep::Decided => {}
                        DerivationStep::Progress => {
                            progressed = true;
                            self.deferred_shape_checks.push(check);
                        }
                        // The readiness test above already waits on a variable
                        // target, so this keeps an undecided entry only in
                        // principle; an entry is never dropped undecided.
                        DerivationStep::Pending => self.deferred_shape_checks.push(check),
                        DerivationStep::Awaits {
                            derivation,
                            operands,
                        } => {
                            let mut check = check;
                            check.rule = DeferredShapeRule::Derivation(derivation);
                            check.arg_tys = operands;
                            self.deferred_shape_checks.push(check);
                        }
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
                    location,
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
                        location: location.as_ref(),
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
        progressed
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
    pub(in crate::infer) fn replay_post_app(
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
                adt_reg,
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
                    CheckSite::Deferred(call.location),
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
                    false,
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
    pub(in crate::infer) fn finish_deferred_shape_checks(
        &mut self,
        env: &Env,
        vg: &mut VarGen,
        subst: &mut Subst,
        adt_reg: &AdtRegistry,
        errors: &mut DiagnosticSink<'_>,
    ) {
        self.result_inputs_closed = true;
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
                    DeferredShapeRule::PostApp { .. }
                        | DeferredShapeRule::Derivation(_)
                        | DeferredShapeRule::ResultJoin { .. }
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
                DeferredShapeRule::ResultJoin { .. } => {
                    self.deferred_shape_checks.push(check);
                    continue;
                }
                DeferredShapeRule::Matmul { .. } => "matmul".to_string(),
                DeferredShapeRule::Reduction { name, .. } => name,
                DeferredShapeRule::Expand { builtin, .. } => builtin.to_string(),
                DeferredShapeRule::LayerNorm { .. } => "layer_norm".to_string(),
                DeferredShapeRule::Conv { .. } => "conv".to_string(),
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
                    ref location,
                    env: ref call_env,
                } => {
                    let call = PostAppCall {
                        replay,
                        site,
                        node,
                        kids,
                        func_name,
                        env: call_env,
                        location: location.as_ref(),
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
                    // A transformation must either derive its checked type
                    // from settled inputs or reject at this boundary.
                    if matches!(
                        derivation,
                        TypeDerivation::Grad { .. } | TypeDerivation::Vmap { .. }
                    ) && matches!(
                        resolve_type_derivation(
                            derivation,
                            &check.arg_tys[0],
                            &check.result_ty,
                            &|_, _| false,
                            false,
                            false,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                        ),
                        DerivationStep::Decided
                    ) {
                        continue;
                    }
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
        // Every semantic producer was decided above, including the universal
        // decision for authored binders. Result annotations can now be checked
        // without making an unresolved semantic rule appear admissible. The
        // authored-binder checks still run after this last unification step.
        self.replay_ready_shape_checks(vg, subst, adt_reg, errors);
        self.result_inputs_closed = false;
        self.closing_group_types.clear();
    }
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
pub(in crate::infer) fn shape_operand_awaits_binding(ty: &Type, subst: &Subst) -> bool {
    match subst.apply(ty) {
        Type::Var(_) => true,
        Type::Ref(inner) => shape_operand_awaits_binding(&inner, subst),
        _ => false,
    }
}
