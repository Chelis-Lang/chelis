//! Public check entry points and the inference drivers that run the schedule.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

use crate::errors::ErrorWitness;

/// Transactional owner for one cyclic full-reference component's inference
/// level, provisional visibility capability, and temporary bindings. It
/// snapshots every member binding before provisional prebinding so normal and
/// cancelled exits both restore the surrounding environment exactly.
struct ComponentLevelScope {
    level: chelis_type_core::LevelToken,
    members: Vec<(String, Option<Scheme<ErrorWitness>>)>,
    prior_active_component: UnordSet<String>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecursiveAbortObservation {
    current_level: u32,
    group_state_counts: (usize, usize),
    member_count: usize,
    prior_binding_count: usize,
    all_prior_bindings_restored: bool,
    all_temporary_bindings_removed: bool,
}

#[cfg(test)]
std::thread_local! {
    static PRIMARY_RECURSIVE_CANCEL_COUNTDOWN: Cell<Option<usize>> = const { Cell::new(None) };
    static RECURSIVE_ABORT_OBSERVATION: RefCell<Option<RecursiveAbortObservation>> = const { RefCell::new(None) };
}

#[cfg(test)]
struct PrimaryRecursiveCancelGuard;

#[cfg(test)]
impl Drop for PrimaryRecursiveCancelGuard {
    fn drop(&mut self) {
        PRIMARY_RECURSIVE_CANCEL_COUNTDOWN.with(|countdown| countdown.set(None));
    }
}

#[cfg(test)]
fn cancel_primary_recursive_after_members(count: usize) -> PrimaryRecursiveCancelGuard {
    assert!(
        count > 0,
        "the test hook must allow at least one SCC member"
    );
    PRIMARY_RECURSIVE_CANCEL_COUNTDOWN.with(|countdown| {
        assert!(
            countdown.replace(Some(count)).is_none(),
            "the primary recursive cancellation hook is already armed"
        );
    });
    RECURSIVE_ABORT_OBSERVATION.with(|observation| *observation.borrow_mut() = None);
    PrimaryRecursiveCancelGuard
}

#[cfg(test)]
fn primary_recursive_member_finished_for_test() {
    PRIMARY_RECURSIVE_CANCEL_COUNTDOWN.with(|countdown| {
        let Some(remaining) = countdown.get() else {
            return;
        };
        if remaining == 1 {
            countdown.set(None);
            crate::cancel::current_cancel_token()
                .expect("the cancellation test hook requires an installed token")
                .cancel();
        } else {
            countdown.set(Some(remaining - 1));
        }
    });
}

#[cfg(test)]
fn take_recursive_abort_observation() -> Option<RecursiveAbortObservation> {
    RECURSIVE_ABORT_OBSERVATION.with(|observation| observation.borrow_mut().take())
}

#[cfg(test)]
fn schemes_match(left: &Scheme<ErrorWitness>, right: &Scheme<ErrorWitness>) -> bool {
    left.tvars == right.tvars
        && left.tvar_restrictions == right.tvar_restrictions
        && left.dvars == right.dvars
        && left.rvars == right.rvars
        // chelis#1654: an obligation is part of the scheme, so two schemes
        // that differ only there are not the same scheme.
        && left.constraints == right.constraints
        && left.body == right.body
}

impl ComponentLevelScope {
    fn enter(
        indices: &[usize],
        items: &[(Option<String>, &deep::Expr)],
        cycle_precedence_names: &[String],
        env: &mut Env,
        var_gen: &VarGen,
        subst: &mut Subst<ErrorWitness>,
    ) -> Self {
        let mut seen = UnordSet::new();
        let members = indices
            .iter()
            .filter_map(|index| top_level_decl_name(items[*index].1))
            .filter(|name| seen.insert((*name).to_string()))
            .map(|name| (name.to_string(), env.lookup(name).cloned()))
            .collect::<Vec<_>>();
        let mut active_component = members
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<UnordSet<_>>();
        active_component.extend(cycle_precedence_names.iter().cloned());
        let prior_active_component = env.replace_active_top_level_component(active_component);
        Self {
            level: subst.enter_level(var_gen),
            members,
            prior_active_component,
        }
    }

    fn remove_temporary_bindings(&self, env: &mut Env) {
        for (name, _) in &self.members {
            env.remove_binding(name);
        }
    }

    /// Normal completion deliberately does not restore prior defsig/metadata
    /// bindings: completed generalized member schemes replace them below.
    fn complete(mut self, env: &mut Env, var_gen: &VarGen, subst: &mut Subst<ErrorWitness>) {
        self.remove_temporary_bindings(env);
        env.end_holed_group();
        let _finished_component = env
            .replace_active_top_level_component(core::mem::take(&mut self.prior_active_component));
        subst.leave_level(self.level, var_gen);
    }

    fn abort(
        mut self,
        env: &mut Env,
        var_gen: &VarGen,
        subst: &mut Subst<ErrorWitness>,
        operation: &crate::resources::Operation<'_>,
    ) {
        #[cfg(test)]
        let prior_members = self.members.clone();
        super::recursion::abort_group(operation);
        self.remove_temporary_bindings(env);
        env.end_holed_group();
        for (name, prior) in self.members {
            if let Some(scheme) = prior {
                env.bind(name, scheme);
            }
        }
        let _aborted_component = env
            .replace_active_top_level_component(core::mem::take(&mut self.prior_active_component));
        subst.leave_level(self.level, var_gen);
        #[cfg(test)]
        RECURSIVE_ABORT_OBSERVATION.with(|observation| {
            let prior_binding_count = prior_members
                .iter()
                .filter(|(_, prior)| prior.is_some())
                .count();
            let all_prior_bindings_restored = prior_members.iter().all(|(name, prior)| {
                prior.as_ref().is_none_or(|prior| {
                    env.lookup(name)
                        .is_some_and(|restored| schemes_match(restored, prior))
                })
            });
            let all_temporary_bindings_removed = prior_members
                .iter()
                .filter(|(_, prior)| prior.is_none())
                .all(|(name, _)| env.lookup(name).is_none());
            *observation.borrow_mut() = Some(RecursiveAbortObservation {
                current_level: subst.current_level(),
                group_state_counts: super::recursion::group_state_counts(operation),
                member_count: prior_members.len(),
                prior_binding_count,
                all_prior_bindings_restored,
                all_temporary_bindings_removed,
            });
        });
    }
}

/// Re-infer one clean cyclic component against its prior complete schemes so
/// contracts returned across sibling edges travel through the ordinary
/// instantiate/application/generalize lifecycle. Each batch sweep sees only
/// the preceding batch, and a component of N members therefore gets at most N
/// further sweeps.
#[allow(clippy::too_many_arguments)]
fn sweep_recursive_collection_contracts(
    recursive_function_indices: &[usize],
    items: &[(Option<String>, &deep::Expr)],
    cycle_precedence_names: &[String],
    env: &mut Env,
    var_gen: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    declaration_diagnostic_owners: &[Option<DeclarationDiagnosticOwner>],
    operation: &crate::resources::Operation<'_>,
) {
    let members = recursive_function_indices
        .iter()
        .filter_map(|&index| {
            let expr = items[index].1;
            matches!(stamped_parts(expr), Some((DeepTag::Def, _, _)))
                .then(|| top_level_decl_name(expr).map(|name| (index, name.to_string())))
                .flatten()
        })
        .collect::<Vec<_>>();
    // The function-plan projection is authoritative. A cyclic full-reference
    // component with no recursive function component, including eager/mixed
    // value cycles, retains its existing one-pass schedule.
    if members.is_empty() || members.len() != recursive_function_indices.len() {
        return;
    }

    let mut prior_schemes = UnordMap::new();
    for (_, name) in &members {
        let Some(scheme) = env.lookup(name).cloned() else {
            return;
        };
        prior_schemes.insert(name.clone(), scheme);
    }
    // [04-INF-9] forbids manufacturing a checked operation contract from an
    // authored body. The sweep only closes transport edges from an already
    // checked seed, so the overwhelmingly common seedless recursive SCC needs
    // no replay.
    if prior_schemes
        .to_sorted()
        .into_iter()
        .map(|(_, scheme)| scheme)
        .all(|scheme| scheme.constraints.is_empty())
    {
        return;
    }

    for sweep_index in 0..members.len() {
        let final_confirmation = sweep_index + 1 == members.len();
        let sweep_checkpoint = errors.checkpoint();
        let component_scope = ComponentLevelScope::enter(
            recursive_function_indices,
            items,
            cycle_precedence_names,
            env,
            var_gen,
            subst,
        );
        super::recursion::begin_group(
            members.iter().map(|(_, name)| {
                let authored = declared_signatures
                    .get(name)
                    .is_some_and(|metadata| !metadata.binders.is_empty());
                (name.as_str(), authored)
            }),
            env,
            operation,
        );
        let mut scratch = InferenceProduct::default();
        scratch.type_headers = adt_reg.resolution_env().clone();
        scratch.adt_registry = adt_reg.clone();
        let mut deferred_bindings = Vec::with_capacity(members.len());
        let mut diagnosed = false;

        for (declaration_index, name) in &members {
            if operation.cancelled() {
                component_scope.abort(env, var_gen, subst, operation);
                return;
            }
            let (module, expr) = &items[*declaration_index];
            scratch.begin_root(expr, operation);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), Some(name)),
                Some(name.clone()),
                operation,
            );
            let declaration_diagnostic_owner = declaration_diagnostic_owners
                .get(*declaration_index)
                .and_then(Option::as_ref);
            env.set_current_declaration_ordinal(Some(*declaration_index));
            let expected = prior_schemes
                .get(name)
                .expect("recursive sweep snapshots every member scheme");
            if let Some(binding) = infer_top_level(
                expr,
                env,
                var_gen,
                subst,
                adt_reg,
                errors,
                &mut scratch,
                None,
                Some(recursion::RecursiveExpected::Published(expected)),
                true,
                declared_signatures,
                declaration_diagnostic_owner,
                operation,
            ) {
                deferred_bindings.push(binding);
            }
            close_declaration(
                &mut scratch,
                CloseScope::ComponentMember,
                env,
                var_gen,
                subst,
                adt_reg,
                errors,
                operation,
            );
            if errors.iter_since(sweep_checkpoint).next().is_some() {
                diagnosed = true;
                break;
            }
        }

        if diagnosed || deferred_bindings.len() != members.len() {
            component_scope.abort(env, var_gen, subst, operation);
            return;
        }

        close_component(
            &mut scratch,
            env,
            var_gen,
            subst,
            adt_reg,
            errors,
            operation,
        );
        super::recursion::finish_group(subst, errors, operation);
        if errors.iter_since(sweep_checkpoint).next().is_some() {
            component_scope.abort(env, var_gen, subst, operation);
            return;
        }
        component_scope.complete(env, var_gen, subst);
        let schemes = deferred_bindings
            .into_iter()
            .map(|binding| {
                generalize_deferred_recursive_binding(binding, env, subst, &mut scratch, operation)
            })
            .collect::<Vec<_>>();
        debug_assert!(schemes.iter().all(|(name, scheme)| {
            scheme.constraints.len()
                >= prior_schemes
                    .get(name)
                    .expect("recursive sweep retains every member")
                    .constraints
                    .len()
        }));
        let changed = schemes.iter().any(|(name, scheme)| {
            scheme.constraints.len()
                > prior_schemes
                    .get(name)
                    .expect("recursive sweep retains every member")
                    .constraints
                    .len()
        });
        if final_confirmation && changed {
            for (name, scheme) in prior_schemes.to_sorted() {
                env.bind(name.clone(), scheme.clone());
            }
            errors.push(CheckError::new(
                CheckErrorKind::Other,
                format!(
                    "internal: recursive collection-contract closure for [{}] added a relation \
                     on its final member-count confirmation sweep",
                    members
                        .iter()
                        .map(|(_, name)| name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                vec![
                    "The checker rejected this program rather than publishing an incomplete \
                     recursive operation contract closure."
                        .to_string(),
                ],
            ));
            return;
        }
        let mut published_schemes = UnordMap::new();
        for (name, scheme) in schemes {
            env.bind(name.clone(), scheme.clone());
            published_schemes.insert(name, scheme);
        }
        if !changed {
            return;
        }
        prior_schemes = published_schemes;
    }
}

/// Front-end cancellation gate for a check unit (chelis#930).
///
/// Placed between the passes of every public check entry. When the thread's
/// cancellation token has been tripped it records the hard failure and reports
/// `true`, and the caller returns immediately: the passes below it would
/// otherwise spend the rest of the front end's budget walking declarations the
/// abandoned inference never bound, and report their absence as if the program
/// were wrong.
///
/// Same covered-or-rejected discipline as `StackExhaustionScope::drain_into` —
/// a check unit that stopped early must fail, never return a partial `Ok`.
#[must_use = "a tripped cancellation gate must end the check unit"]
fn cancellation_gate(
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> bool {
    if !operation.cancelled() {
        return false;
    }
    if !errors
        .iter()
        .any(|error| crate::cancel::is_cancellation(&error.message))
    {
        errors.push(crate::cancel::cancellation_check_error());
    }
    true
}

/// Run type inference on a list of top-level Deep expressions.
pub fn infer_program<S: chelis_stack_services::StackServices>(
    exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> InferResult {
    resources.run(|operation| infer_program_in_operation(exprs, operation))
}

pub(crate) fn infer_program_in_operation(
    exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> InferResult {
    // WI-1 follow-up: run the whole pipeline on a grown stack so a deeply
    // nested but finite program checks end-to-end instead of tripping a
    // per-site `stack_guard!` partway through one of the recursive passes.
    crate::session::infer_program(exprs, operation)
}

pub(crate) fn infer_program_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> InferStats {
    let normalized = normalize_program_input(exprs);
    infer_program_with_product_in_session(&normalized, errors, operation).stats()
}

pub(super) fn infer_program_with_product_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> InferenceProduct {
    // Reset the stack-exhaustion flag for this check unit; `drain` below
    // turns any walker stack bail into a hard located error so deep input
    // can never produce a silent green / partial result.
    let stack_scope = StackExhaustionScope::enter(operation);
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::<ErrorWitness>::new();
    let mut adt_reg = AdtRegistry::new();
    builtins::register_prelude_adts(&mut env, &mut vg, &mut adt_reg);
    let mut product = InferenceProduct::default();
    product.register_source_binding_occurrences(exprs, errors, operation);

    // RFC v4b (RT-1 F2): reject a named module opened by more than one
    // wrapper in this check unit (module-identity forgery).
    detect_module_reopens(exprs, errors);
    // RFC v5 (RT-1 F2 bypass): reject the reef linker's reserved
    // internal-name format in programs not produced by the linker.
    detect_forged_linker_names(exprs, errors, operation);

    // First pass: collect deftype and defsig declarations. Descend through
    // `(module {} name ...)` wrappers so declarations in every idiomatic
    // Surf source (every .ch starts with `module X`) get collected.
    let items = top_level_decl_items_with_modules(exprs);
    let declaration_diagnostic_owners = declaration_diagnostic_owner_plan(exprs);
    collect_all_declarations(
        &items,
        &declaration_diagnostic_owners,
        &mut env,
        InferenceVars {
            vg: &mut vg,
            subst: &mut subst,
        },
        &mut adt_reg,
        errors,
        operation,
    );
    product.type_headers = adt_reg.resolution_env().clone();
    product.adt_registry = adt_reg.clone();

    // Checker-enforced opacity (RFC D-CHECK): install the per-run
    // context so the inference hooks see module identity, exports,
    // and producer text. Dropped at the end of this function.
    let opacity_meta = build_opacity_meta(&items, &adt_reg, &env);
    let _opacity_guard = crate::opacity::install_opacity_context(
        crate::opacity::OpacityContextData::from_meta(opacity_meta),
        operation,
    );

    // Second pass: infer def bodies. Same module-descent rationale as the
    // declaration pass — without it, the entire HM checker is a no-op on
    // module-wrapped programs.
    let declared_signatures =
        collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr), operation);
    validate_binder_literal_adoption_in_program(&items, &declared_signatures, errors);
    let external_input_types = collect_literal_external_input_types(&items, operation);
    let (metadata_prebound_names, function_metadata_failures) =
        prebind_defsig_less_function_body_types(
            &items,
            &declared_signatures,
            &declaration_diagnostic_owners,
            &mut env,
            &mut vg,
            &mut subst,
            &adt_reg,
            errors,
            operation,
        );
    let top_level_references = TopLevelReferenceGraph::build(&items, operation);
    product.function_inference_plan =
        FunctionInferencePlan::build_from_reference_graph(&top_level_references, operation);
    let inference_groups = primary_inference_groups_with_reference_graph(
        &product.function_inference_plan,
        &items,
        &top_level_references,
        operation,
    );
    // Match the persisted-state driver: cache the TLS token once and poll at
    // declaration granularity. In particular, an incomplete cyclic component
    // must consume its structured scope through `abort` before this schedule
    // returns; finishing or generalizing a partially inferred group would
    // leak its pins, temporary bindings, and child level.
    let cancel = operation.cancellation;
    let cancelled = || cancel.is_some_and(|token| token.is_cancelled());
    super::recursion::reset(operation);
    'schedule: for group in inference_groups {
        if cancelled() {
            break;
        }
        let component_diagnostic_checkpoint = errors.checkpoint();
        let cyclic = group.cyclic;
        let recursion_active = !group.recursive_function_indices.is_empty();
        let cycle_precedence_names = if cyclic {
            top_level_references
                .cycle_precedence_targets(&group.indices)
                .into_iter()
                .map(|target| top_level_references.definition(target).name.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut component_scope = cyclic.then(|| {
            ComponentLevelScope::enter(
                &group.indices,
                &items,
                &cycle_precedence_names,
                &mut env,
                &vg,
                &mut subst,
            )
        });
        let provisional_types = if cyclic {
            prebind_cyclic_component_schemes(
                &group.indices,
                &items,
                &declared_signatures,
                &metadata_prebound_names,
                &mut env,
                &mut vg,
                &subst,
            )
        } else {
            UnordMap::new()
        };
        product.record_group_provisional_types(
            provisional_types
                .to_sorted()
                .into_iter()
                .filter_map(|(index, ty)| {
                    top_level_decl_name(items[*index].1).map(|name| (name.to_string(), ty.clone()))
                }),
        );
        // spec/04 §3.1.1: recursive-instantiation validation remains the
        // function-plan projection. A mixed reference cycle alone must not
        // activate it.
        if recursion_active {
            super::recursion::begin_group(
                group
                    .recursive_function_indices
                    .iter()
                    .filter_map(|&index| {
                        top_level_decl_name(items[index].1).map(|name| {
                            let authored = declared_signatures
                                .get(name)
                                .is_some_and(|metadata| !metadata.binders.is_empty());
                            (name, authored)
                        })
                    }),
                &env,
                operation,
            );
        }
        let mut deferred_bindings = Vec::new();
        for &declaration_index in &group.indices {
            if cancelled() {
                if let Some(scope) = component_scope.take() {
                    scope.abort(&mut env, &vg, &mut subst, operation);
                }
                break 'schedule;
            }
            let (module, expr) = &items[declaration_index];
            product.begin_root(expr, operation);
            let decl_name = top_level_decl_name(expr);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), decl_name),
                decl_name.map(str::to_string),
                operation,
            );
            let declaration_diagnostic_owner = declaration_diagnostic_owners
                .get(declaration_index)
                .and_then(Option::as_ref);
            env.set_current_declaration_ordinal(Some(declaration_index));
            let external_input_failure = prebind_literal_external_input_for_declaration(
                declaration_index,
                expr,
                &external_input_types,
                &declared_signatures,
                declaration_diagnostic_owner,
                &mut env,
                &mut vg,
                &mut subst,
                &adt_reg,
                errors,
                operation,
            );
            let prebound_type_failure = external_input_failure
                .as_ref()
                .or_else(|| function_metadata_failures.get(&declaration_index));
            if let Some(binding) = infer_top_level(
                expr,
                &mut env,
                &mut vg,
                &mut subst,
                &adt_reg,
                errors,
                &mut product,
                prebound_type_failure,
                provisional_types
                    .get(&declaration_index)
                    .map(recursion::RecursiveExpected::Provisional),
                cyclic,
                &declared_signatures,
                declaration_diagnostic_owner,
                operation,
            ) {
                deferred_bindings.push(binding);
            }
            // chelis#1489: one close per declaration. An earlier revision had
            // a second, whole-program phase that three library lanes never
            // reached.
            close_declaration(
                &mut product,
                close_scope(cyclic),
                &env,
                &mut vg,
                &mut subst,
                &adt_reg,
                errors,
                operation,
            );
            #[cfg(test)]
            if cyclic {
                primary_recursive_member_finished_for_test();
            }
        }
        if cancelled() {
            if let Some(scope) = component_scope.take() {
                scope.abort(&mut env, &vg, &mut subst, operation);
            }
            break 'schedule;
        }
        if cyclic {
            close_component(
                &mut product,
                &mut env,
                &mut vg,
                &mut subst,
                &adt_reg,
                errors,
                operation,
            );
            // Function-recursion validation, when independently active, must
            // clear its pins before component-wide generalization.
            if recursion_active {
                super::recursion::finish_group(&subst, errors, operation);
            }
            component_scope
                .take()
                .expect("cyclic component owns an inference-level scope")
                .complete(&mut env, &vg, &mut subst);
            let schemes = deferred_bindings
                .into_iter()
                .map(|binding| {
                    generalize_deferred_recursive_binding(
                        binding,
                        &env,
                        &subst,
                        &mut product,
                        operation,
                    )
                })
                .collect::<Vec<_>>();
            for (name, scheme) in schemes {
                env.bind(name, scheme);
            }
            if recursion_active
                && errors
                    .iter_since(component_diagnostic_checkpoint)
                    .next()
                    .is_none()
            {
                sweep_recursive_collection_contracts(
                    &group.recursive_function_indices,
                    &items,
                    &cycle_precedence_names,
                    &mut env,
                    &mut vg,
                    &mut subst,
                    &adt_reg,
                    errors,
                    &declared_signatures,
                    &declaration_diagnostic_owners,
                    operation,
                );
            }
        }
    }
    crate::opacity::set_current_item(None, None, operation);
    env.set_current_declaration_ordinal(None);

    if cancellation_gate(errors, operation) {
        return product;
    }
    product.resolve_owner_types(&subst);

    // PP9 / [04-TOT-5]: every public entry reaches one ordered semantic pass
    // protocol after inference. Build the same declaration-type view the IR
    // entries supply; stamped input stays stamped and every reader below must
    // handle that carrier directly.
    let semantic_type_env = build_ir_type_env(exprs, operation);
    validate_semantic_program(
        exprs,
        SemanticContext {
            type_env: &semantic_type_env,
            selector_context: &SelectorCallableContext::default(),
            rank_context: &RankBodyContext::default(),
        },
        &top_level_references,
        &product,
        errors,
        operation,
    );
    if cancellation_gate(errors, operation) {
        stack_scope.drain_into(errors);
        product.top_level_references = top_level_references;
        return product;
    }

    // Final shared checks: reject tensor types whose element precision isn't supported
    // by the Phase 0f backend (f16/bf16/f8e4m3). These would silently get
    // downcast to f32 by the current build targets, violating the "no implicit
    // precision promotion" rule. f64 is supported as of v0.2.3.
    validate_tensor_precisions_in_program(exprs, errors, operation);
    crate::invariants::validate_type_invariants_in_program_with_sink(exprs, errors);

    // If any walker bailed on a nearly-exhausted stack during this run,
    // surface it as a hard located failure (covered-or-rejected).
    stack_scope.drain_into(errors);

    product.top_level_references = top_level_references;
    product
}

pub fn check_ir_program<S: chelis_stack_services::StackServices>(
    exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<CompletedCheck, InferResult> {
    resources.run(|operation| check_ir_program_in_operation(exprs, operation))
}

pub(crate) fn check_ir_program_in_operation(
    exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<CompletedCheck, InferResult> {
    // Compose: empty outer scope, then check exprs as new code against it.
    // This keeps a single source of truth for the IR check pipeline.
    check_ir_with_context_in_operation(&ContextSnapshot::empty(), exprs, operation)
}

/// Build an ordinary semantic library snapshot.
/// The snapshot does not grant native library-proof authority.
pub fn build_type_env_from_library<S: chelis_stack_services::StackServices>(
    library_exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<ContextSnapshot, InferResult> {
    resources.run(|operation| build_type_env_from_library_in_operation(library_exprs, operation))
}

pub(crate) fn build_type_env_from_library_in_operation(
    library_exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<ContextSnapshot, InferResult> {
    crate::session::build_type_env_from_library(library_exprs, operation)
}

pub(crate) fn build_type_env_from_library_in_session(
    library_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Result<(ContextSnapshot, Vec<deep::Expr>), InferStats> {
    let normalized = normalize_program_input(library_exprs);
    let library_exprs = &normalized;
    // Reset the stack-exhaustion flag for this check unit; drained below
    // before the empty-errors gate (covered-or-rejected on deep input).
    let stack_scope = StackExhaustionScope::enter(operation);
    operation.phase("build_type_env_sub:start");

    // Start from the empty (builtins + prelude ADTs) state.
    let empty = ContextSnapshot::empty();
    let mut state = empty.clone();

    // Library IR declared-type lookup.
    let library_ir = build_ir_type_env(library_exprs, operation);
    operation.phase("build_type_env_sub:build_ir_type_env_initial");

    let product = infer_ir_program_with_state(library_exprs, &mut state, errors, operation);
    let stats = product.stats();
    operation.phase("build_type_env_sub:infer_ir_program_with_state");
    // chelis#930: inference may have abandoned its schedule. Stop before the
    // validators, which would otherwise report the un-inferred tail as program
    // errors and spend the rest of the phase doing it.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    let mut rank_bodies = validate_semantic_program(
        library_exprs,
        SemanticContext {
            type_env: &library_ir,
            selector_context: &SelectorCallableContext::default(),
            rank_context: &RankBodyContext::default(),
        },
        &product.top_level_references,
        &product,
        errors,
        operation,
    );
    operation.phase("build_type_env_sub:validate_semantic_program");
    validate_tensor_precisions_in_program(library_exprs, errors, operation);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    operation.phase("build_type_env_sub:validate_tensor_precisions");
    // Surface a stack-exhaustion bail from the passes above as a hard
    // located error before the gate (and before the errors drain below).
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Capture library def names — needed by new-code cycle / unbound
    // suppression to distinguish library refs from new-code refs.
    let mut library_def_names = chelis_unord::UnordSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }
    operation.phase("build_type_env_sub:collect_library_def_names");

    // Build a richer `ir_types` by annotating library exprs against
    // the now-populated state and re-extracting type metadata. The raw
    // `library_ir` (built from un-annotated source) only catches defs
    // with explicit type annotations; for downstream callers that read
    // `CheckedProgram::type_env()` to resolve cross-context name refs
    // (Phase D effects, Phase E linearity, Phase F lower) we need every
    // library def's inferred function type, not just the explicitly-typed
    // ones. Mirrors the monolithic `check_ir_program` flow which
    // calls `annotate_ir_program` then `build_ir_type_env` on
    // the annotated result.
    //
    // Pass this unit's declared signatures directly into annotation so
    // separate-`sig` parameter stamps and binder scopes cannot observe a
    // different sequential, nested, or parallel check.
    let declared_signatures = collect_declared_sig_metadata(library_exprs, operation);
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    let library_annotated = annotate_top_levels(
        library_exprs,
        &product,
        annotation_context,
        errors,
        operation,
    );
    operation.phase("build_type_env_sub:annotate_library_exprs_outer_loop");
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    let library_ir_annotated = build_ir_type_env(&library_annotated, operation);
    operation.phase("build_type_env_sub:build_ir_type_env_from_annotated");

    // The annotation loop above recurses (annotate_expr_with_scope); if it
    // bailed on low stack, reject rather than return a partially-annotated
    // library context.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    let selector_callables = extend_selector_callable_context(
        &library_annotated,
        &SelectorCallableContext::default(),
        operation,
    );
    if let Err(message) =
        rank_bodies.bind_checked_declarations(&library_annotated, &library_ir_annotated)
    {
        errors.push(internal_owner_stamp_error(message));
        return Err(stats);
    }

    let type_env = ContextSnapshot {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated,
        library_def_names,
        selector_callables,
        rank_bodies,
        opacity: state.opacity,
    };

    Ok((type_env, library_annotated))
}

/// Return a library snapshot and its opaque successful check.
/// One annotation traversal supplies both products.
pub fn build_compiled_library_context<S: chelis_stack_services::StackServices>(
    library_exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<(ContextSnapshot, CompletedCheck), InferResult> {
    resources.run(|operation| build_compiled_library_context_in_operation(library_exprs, operation))
}

pub(crate) fn build_compiled_library_context_in_operation(
    library_exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<(ContextSnapshot, CompletedCheck), InferResult> {
    crate::session::build_compiled_library_context(library_exprs, operation)
}

pub(crate) fn build_compiled_library_context_in_session(
    library_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Result<(ContextSnapshot, CheckFacts), InferStats> {
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate (and again after the
    // annotation pass) so a deep-input stack bail on the library-compile path
    // always fails the check rather than returning a silent green / partial
    // CheckedProgram. Mirrors `check_ir_with_signature_context`.
    // chelis#1923: fold every pipe into the application it denotes before
    // anything reads this program. This entry annotates `library_exprs`
    // directly rather than through a checked-program call, so without the
    // fold here a library's pipe would reach `annotated_exprs` unfolded and
    // the lowerer would refuse it.
    let stack_scope = StackExhaustionScope::enter(operation);
    // Mirror `build_type_env_from_library` up through the validators so the
    // type_env half stays bit-compatible with the existing public API.
    let empty = ContextSnapshot::empty();
    let mut state = empty.clone();

    let library_ir = build_ir_type_env(library_exprs, operation);

    let product = infer_ir_program_with_state(library_exprs, &mut state, errors, operation);
    let stats = product.stats();
    // chelis#930: see `build_type_env_from_library_in_session` — stop before
    // the validators if inference abandoned its schedule.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    let mut rank_bodies = validate_semantic_program(
        library_exprs,
        SemanticContext {
            type_env: &library_ir,
            selector_context: &SelectorCallableContext::default(),
            rank_context: &RankBodyContext::default(),
        },
        &product.top_level_references,
        &product,
        errors,
        operation,
    );
    validate_tensor_precisions_in_program(library_exprs, errors, operation);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Capture library def names before consuming `state` into `TypeEnv`.
    let mut library_def_names = chelis_unord::UnordSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }

    // SINGLE annotation pass — feeds both the TypeEnv's
    // `ir_types` AND the returned CheckedProgram's `annotated_exprs`.
    // Previously `build_type_env_from_library` did one annotation here
    // (~13.8s on Coral) and `check_ir_with_context(empty, library)`
    // did a separate, redundant inference+annotation pass (~16.8s).
    //
    // The explicit annotation context supplies this unit's separate-`sig`
    // parameter metadata and per-def binder scope to the recursive pass.
    let declared_signatures = collect_declared_sig_metadata(library_exprs, operation);
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    let library_annotated = annotate_top_levels(
        library_exprs,
        &product,
        annotation_context,
        errors,
        operation,
    );
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    let library_ir_annotated = build_ir_type_env(&library_annotated, operation);
    let selector_callables = extend_selector_callable_context(
        &library_annotated,
        &SelectorCallableContext::default(),
        operation,
    );
    if let Err(message) =
        rank_bodies.bind_checked_declarations(&library_annotated, &library_ir_annotated)
    {
        errors.push(internal_owner_stamp_error(message));
        return Err(stats);
    }

    let type_env = ContextSnapshot {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated.clone(),
        library_def_names,
        selector_callables,
        rank_bodies,
        opacity: state.opacity,
    };

    // Build the CheckedProgram with the same `annotated_type_env` shape
    // that `check_ir_with_context(empty, library)` produces. With an
    // empty outer scope, `context.inner().ir_types` is empty, so the
    // union step is a no-op and `annotated_type_env ==
    // library_ir_annotated`.
    let checked = finalize_checked_program(
        library_annotated,
        library_ir_annotated,
        &SignatureInferenceMetadata::default(),
        &product,
        stats,
        errors,
        operation,
    );
    if !errors.is_empty() {
        return Err(stats);
    }
    // chelis#930: see `check_ir_with_signature_context_in_session`.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }

    Ok((type_env, checked))
}

/// Continue library analysis from a borrowed semantic snapshot.
/// Return the combined snapshot and the new layer completion.
pub fn build_compiled_library_context_with_base<S: chelis_stack_services::StackServices>(
    base: &ContextSnapshot,
    library_exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<(ContextSnapshot, CompletedCheck), InferResult> {
    resources.run(|operation| {
        build_compiled_library_context_with_base_in_operation(base, library_exprs, operation)
    })
}

pub(crate) fn build_compiled_library_context_with_base_in_operation(
    base: &ContextSnapshot,
    library_exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<(ContextSnapshot, CompletedCheck), InferResult> {
    crate::session::build_compiled_library_context_with_base(base, library_exprs, operation)
}

pub(crate) fn build_compiled_library_context_with_base_in_session(
    base: &ContextSnapshot,
    library_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Result<(ContextSnapshot, CheckFacts), InferStats> {
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate (and again after the
    // annotation pass) so a deep-input stack bail on the layered
    // library-compile path always fails the check rather than returning a
    // silent green / partial CheckedProgram. Mirrors
    // `check_ir_with_signature_context`.
    // chelis#1923: fold every pipe into the application it denotes before
    // anything reads this program. This entry annotates `library_exprs`
    // directly rather than through a checked-program call, so without the
    // fold here a library's pipe would reach `annotated_exprs` unfolded and
    // the lowerer would refuse it.
    let stack_scope = StackExhaustionScope::enter(operation);
    // Seed from the base context's snapshot rather than the empty state.
    let mut state = base.resume_for_new_check();

    // `library_exprs` declared types (IR), layered on top of the base's.
    let new_ir = build_ir_type_env(library_exprs, operation);
    let combined_ir: BTreeMap<String, deep::Expr> = state
        .ir_types
        .iter()
        .chain(new_ir.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Base is already validated; run inference + validators on
    // `library_exprs` only. Inference's canonical collector binds the
    // `library_exprs`' own declared types (base schemes are already in
    // `state.env`); the combined IR env is supplied to the validators so
    // `(var basefoo)` references resolve to the base's declared type.
    let product = infer_ir_program_with_state(library_exprs, &mut state, errors, operation);
    let stats = product.stats();
    // chelis#930: see `build_type_env_from_library_in_session` — stop before
    // the validators if inference abandoned its schedule.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    let mut rank_bodies = validate_semantic_program(
        library_exprs,
        SemanticContext {
            type_env: &combined_ir,
            selector_context: &base.selector_callables,
            rank_context: &base.rank_bodies,
        },
        &product.top_level_references,
        &product,
        errors,
        operation,
    );
    validate_tensor_precisions_in_program(library_exprs, errors, operation);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Capture `library_exprs` def names, unioned with the base's, so a
    // subsequent `_with_context` check against the returned TypeEnv
    // distinguishes library refs (base + this layer) from new-code refs.
    let mut library_def_names = base.library_def_names.clone();
    for expr in top_level_decl_items(library_exprs) {
        if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }

    // Annotate ONLY the `library_exprs`, starting from the populated
    // `state` so base names resolve during annotation. Install the
    // explicit declared-signature context for this layer's exprs so
    // separate-`sig` defs get borrow-correct `(params ...)` stamps without
    // inheriting binders from the base context.
    let declared_signatures = collect_declared_sig_metadata(library_exprs, operation);
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    let library_annotated = annotate_top_levels(
        library_exprs,
        &product,
        annotation_context,
        errors,
        operation,
    );
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    let new_ir_annotated = build_ir_type_env(&library_annotated, operation);
    let selector_callables =
        extend_selector_callable_context(&library_annotated, &base.selector_callables, operation);

    // The returned TypeEnv's `ir_types` is the union: base declared types
    // plus this layer's, this layer winning on shadow.
    let mut combined_ir_annotated = new_ir_annotated.clone();
    for (name, ty) in &base.ir_types {
        combined_ir_annotated
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }

    if let Err(message) =
        rank_bodies.bind_checked_declarations(&library_annotated, &combined_ir_annotated)
    {
        errors.push(internal_owner_stamp_error(message));
        return Err(stats);
    }

    let type_env = ContextSnapshot {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: combined_ir_annotated,
        library_def_names,
        selector_callables,
        rank_bodies,
        opacity: state.opacity,
    };

    // The CheckedProgram carries this layer's annotated bodies plus a
    // unioned `type_env` so downstream `_with_context` passes resolve
    // both base and this-layer `(var ...)` references. This mirrors the
    // `check_ir_with_context` returned-CheckedProgram contract.
    let mut checked_type_env = new_ir_annotated;
    for (name, ty) in &base.ir_types {
        checked_type_env
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }
    let checked = finalize_checked_program(
        library_annotated,
        checked_type_env,
        &SignatureInferenceMetadata::default(),
        &product,
        stats,
        errors,
        operation,
    );
    if !errors.is_empty() {
        return Err(stats);
    }
    // chelis#930: see `check_ir_with_signature_context_in_session`.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }

    Ok((type_env, checked))
}

/// Check new expressions against a borrowed ordinary snapshot.
/// The result contains only the new expressions.
pub fn check_ir_with_context<S: chelis_stack_services::StackServices>(
    context: &ContextSnapshot,
    new_exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<CompletedCheck, InferResult> {
    resources.run(|operation| check_ir_with_context_in_operation(context, new_exprs, operation))
}

pub(crate) fn check_ir_with_context_in_operation(
    context: &ContextSnapshot,
    new_exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<CompletedCheck, InferResult> {
    check_ir_with_signature_context_in_operation(
        context,
        &SignatureInferenceMetadata::default(),
        new_exprs,
        operation,
    )
}

pub fn check_ir_with_signature_context<S: chelis_stack_services::StackServices>(
    context: &ContextSnapshot,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<CompletedCheck, InferResult> {
    resources.run(|operation| {
        check_ir_with_signature_context_in_operation(
            context,
            signature_context,
            new_exprs,
            operation,
        )
    })
}

pub(crate) fn check_ir_with_signature_context_in_operation(
    context: &ContextSnapshot,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<CompletedCheck, InferResult> {
    crate::session::check_ir_with_signature_context(
        context,
        signature_context,
        new_exprs,
        operation,
    )
}

pub(crate) fn check_ir_with_signature_context_in_session(
    context: &ContextSnapshot,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Result<CheckFacts, InferStats> {
    let normalized = normalize_program_input(new_exprs);
    let new_exprs = &normalized;
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate so a deep-input stack
    // bail always fails the check (never a silent green / partial result).
    let stack_scope = StackExhaustionScope::enter(operation);
    operation.phase("check_ir_sub:start");

    let mut state = context.resume_for_new_check();

    // New-code declared types (IR) layered on top of library's.
    let new_ir = build_ir_type_env(new_exprs, operation);
    let combined_ir: BTreeMap<String, deep::Expr> = state
        .ir_types
        .iter()
        .chain(new_ir.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    operation.phase("check_ir_sub:build_ir_and_combine");

    // Library is already validated; only run validate / inference on
    // new exprs. Inference's canonical collector binds the new-code's
    // own declared types (library schemes are already in state.env).
    let mut product = infer_ir_program_with_state(new_exprs, &mut state, errors, operation);
    let stats = product.stats();
    operation.phase("check_ir_sub:infer_ir_program_with_state");
    // chelis#930: see `build_type_env_from_library_in_session` — stop before
    // the validators if inference abandoned its schedule.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    product.resolve_owner_types(&state.subst);
    // Run cycle / shape / precision validators on new_exprs only. The
    // combined IR env is supplied so `(var libfoo)` references
    // resolve to the library's declared type during shape validation.
    validate_semantic_program(
        new_exprs,
        SemanticContext {
            type_env: &combined_ir,
            selector_context: &context.selector_callables,
            rank_context: &context.rank_bodies,
        },
        &product.top_level_references,
        &product,
        errors,
        operation,
    );
    operation.phase("check_ir_sub:validate_semantic_program");
    validate_tensor_precisions_in_program(new_exprs, errors, operation);
    crate::invariants::validate_type_invariants_in_program_with_sink(new_exprs, errors);
    operation.phase("check_ir_sub:validate_tensor_precisions");
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Annotate ONLY the new-code exprs, starting from the library
    // snapshot state so library names resolve during annotation.
    let annotated_exprs = annotate_ir_program(new_exprs, &product, errors, operation);
    operation.phase("check_ir_sub:annotate_ir_program");
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    // Surface library declared types in the returned type_env so downstream
    // passes (lower, effects, linearity) can resolve `(var libname)` calls
    // from new-code without a separate library lookup. New-code types take
    // precedence on shadow.
    let mut annotated_type_env = build_ir_type_env(&annotated_exprs, operation);
    for (name, ty) in &context.ir_types {
        annotated_type_env
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }
    operation.phase("check_ir_sub:annotated_type_env_build");
    let checked = finalize_checked_program(
        annotated_exprs,
        annotated_type_env,
        signature_context,
        &product,
        stats,
        errors,
        operation,
    );
    if !errors.is_empty() {
        return Err(stats);
    }
    // chelis#930: signature inference (inside `finalize_checked_program`) can
    // abandon its fixed point, which leaves the metadata short rather than
    // wrong-looking, so nothing above would have failed.
    if cancellation_gate(errors, operation) {
        return Err(stats);
    }
    Ok(checked)
}

pub fn check_typed_program<S: chelis_stack_services::StackServices>(
    exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> Result<CompletedCheck, InferResult> {
    resources.run(|operation| check_typed_program_in_operation(exprs, operation))
}

pub(crate) fn check_typed_program_in_operation(
    exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> Result<CompletedCheck, InferResult> {
    crate::session::check_typed_program(exprs, operation)
}

pub(crate) fn check_typed_program_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Result<CheckFacts, InferStats> {
    // Outermost scope covers both inference (which has its own inner scope)
    // and the annotation pass below, so a bail in either surfaces as a hard
    // located failure rather than a partially-annotated `Ok`.
    let stack_scope = StackExhaustionScope::enter(operation);
    // One tree for inference and annotation: the pipe fold, and nothing else
    // (chelis#1023 pins that the checker's output keeps its input's nodes).
    let product = infer_program_with_product_in_session(exprs, errors, operation);
    let stats = product.stats();
    if errors.is_empty() {
        let annotated_exprs = annotate_ir_program(exprs, &product, errors, operation);
        let annotated_type_env = build_ir_type_env(&annotated_exprs, operation);
        // Annotation recurses; reject if it bailed on low stack.
        stack_scope.drain_into(errors);
        if !errors.is_empty() {
            return Err(stats);
        }
        // spec/06 §3.7: an extent derived from vmapped tensor elements is a
        // check-time `batch_varying_extent` type error at every source
        // ingress. IR reaches this validator before annotation; typed Surf
        // supplies the equivalent canonical annotated type environment here.
        validate_vmap_extent_dependencies(&annotated_exprs, &annotated_type_env, errors, operation);
        if !errors.is_empty() {
            return Err(stats);
        }
        // chelis#930: annotation and the signature fixed point below both
        // abandon their walks on cancellation.
        if cancellation_gate(errors, operation) {
            return Err(stats);
        }
        let checked = finalize_checked_program(
            annotated_exprs,
            annotated_type_env,
            &SignatureInferenceMetadata::default(),
            &product,
            stats,
            errors,
            operation,
        );
        if !errors.is_empty() {
            return Err(stats);
        }
        if cancellation_gate(errors, operation) {
            return Err(stats);
        }
        Ok(checked)
    } else {
        Err(stats)
    }
}

pub fn infer_ir_program<S: chelis_stack_services::StackServices>(
    exprs: &[deep::Expr],
    resources: &crate::Resources<'_, S>,
) -> InferResult {
    resources.run(|operation| infer_ir_program_in_operation(exprs, operation))
}

pub(crate) fn infer_ir_program_in_operation(
    exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> InferResult {
    crate::session::infer_ir_program(exprs, operation)
}

pub(crate) fn infer_ir_program_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> InferStats {
    let normalized = normalize_program_input(exprs);
    let exprs = &normalized;
    let stack_scope = StackExhaustionScope::enter(operation);
    let type_env = build_ir_type_env(exprs, operation);
    let product = infer_ir_program_with_env(exprs, &type_env, errors, operation);
    let stats = product.stats();
    // chelis#930 review: this was the one `*_in_session` entry with no gate.
    // Without it, a tripped token let every pass below stop early and the
    // walk's truncated counts flow into a clean-looking report -- compiler::
    // check would return score 1.0 with an empty error list for a program
    // whose tail was never inspected (contradicting section C4.4 and the
    // repo Contract Invariant: perfect success requires an empty error list
    // to be HONEST, and a truncated walk is not).
    if cancellation_gate(errors, operation) {
        stack_scope.drain_into(errors);
        return stats;
    }
    validate_semantic_program(
        exprs,
        SemanticContext {
            type_env: &type_env,
            selector_context: &SelectorCallableContext::default(),
            rank_context: &RankBodyContext::default(),
        },
        &product.top_level_references,
        &product,
        errors,
        operation,
    );
    if cancellation_gate(errors, operation) {
        stack_scope.drain_into(errors);
        return stats;
    }
    validate_tensor_precisions_in_program(exprs, errors, operation);
    crate::invariants::validate_type_invariants_in_program_with_sink(exprs, errors);
    // Surface any walker stack bail as a hard located error.
    stack_scope.drain_into(errors);
    stats
}

pub(super) fn infer_ir_program_with_env(
    exprs: &[deep::Expr],
    _type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> InferenceProduct {
    // Backwards-compat wrapper. Public callers run the shared semantic pass
    // protocol after this driver returns.
    let empty_inner = crate::context::ContextSnapshot::empty();
    let mut state = empty_inner.clone();
    let mut product = infer_ir_program_with_state(exprs, &mut state, errors, operation);
    product.resolve_owner_types(&state.subst);
    product
}

/// Run the inference / IR binding / shape-validation passes against
/// `state`, mutating it as it goes. Library state should be supplied by
/// pre-cloning a snapshot; pass `&[]`-derived state for the monolithic path.
/// Declaration-local external-input ascriptions are prebound while their
/// declaration is checked. A compiler-authored function body stamp may supply
/// a defsig-less callable header, but body stamps never create eager-value
/// scope. Callers supply their complete declaration-type view to
/// `validate_semantic_program` after this driver returns so layered new code
/// can resolve library references without changing inference scope.
pub(super) fn infer_ir_program_with_state(
    exprs: &[deep::Expr],
    state: &mut ContextSnapshot,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> InferenceProduct {
    let mut product = InferenceProduct::default();
    product.register_source_binding_occurrences(exprs, errors, operation);

    // RFC v4b (RT-1 F2): reject a named module opened by more than one
    // wrapper in this check unit (module-identity forgery). Reef-linked
    // decls carry no wrappers, so this only fires on hand-written `.dp`.
    detect_module_reopens(exprs, errors);
    // RFC v5 (RT-1 F2 bypass): reject the reef linker's reserved
    // internal-name format in programs not produced by the linker.
    detect_forged_linker_names(exprs, errors, operation);

    // Descend through `(module {} name ...)` wrappers: every idiomatic
    // Surf source wraps its declarations in `module X`, and without
    // flattening none of the walkers below see any def/defsig/deftype.
    let items = top_level_decl_items_with_modules(exprs);
    let declaration_diagnostic_owners = declaration_diagnostic_owner_plan(exprs);
    collect_all_declarations(
        &items,
        &declaration_diagnostic_owners,
        &mut state.env,
        InferenceVars {
            vg: &mut state.var_gen,
            subst: &mut state.subst,
        },
        &mut state.adt_reg,
        errors,
        operation,
    );
    product.type_headers = state.adt_reg.resolution_env().clone();
    product.adt_registry = state.adt_reg.clone();

    // Checker-enforced opacity (RFC D-CHECK): accumulate this phase's
    // program-shape metadata into the persistent state (so the
    // stacked library/new-code paths keep library exports visible)
    // and install the per-run context for the inference hooks.
    let phase_meta = build_opacity_meta(&items, &state.adt_reg, &state.env);
    state.opacity.merge_from(&phase_meta);
    let _opacity_guard = crate::opacity::install_opacity_context(
        crate::opacity::OpacityContextData::from_meta(state.opacity.clone()),
        operation,
    );

    // Per-decl profile: when CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1, emit
    // one stderr line per top-level decl with its name and inference time.
    // Aggregated by name in caller scripts to attribute cost per module.

    let declared_signatures =
        collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr), operation);
    validate_binder_literal_adoption_in_program(&items, &declared_signatures, errors);
    // PP9 uses the same two bounded prebind capabilities at both entries:
    // [04-INF-4] external-input ascriptions are declaration-local, while a
    // defsig-less function body stamp supplies the callable header that
    // [04-INF-2]/[04-INF-3] make forward-visible. Eager values are never
    // prebound from body metadata, preserving #1134's source-order rule.
    let external_input_types = collect_literal_external_input_types(&items, operation);
    let (metadata_prebound_names, function_metadata_failures) =
        prebind_defsig_less_function_body_types(
            &items,
            &declared_signatures,
            &declaration_diagnostic_owners,
            &mut state.env,
            &mut state.var_gen,
            &mut state.subst,
            &state.adt_reg,
            errors,
            operation,
        );
    let top_level_references = TopLevelReferenceGraph::build(&items, operation);
    product.function_inference_plan =
        FunctionInferencePlan::build_from_reference_graph(&top_level_references, operation);
    let inference_groups = primary_inference_groups_with_reference_graph(
        &product.function_inference_plan,
        &items,
        &top_level_references,
        operation,
    );
    // chelis#930: cooperative cancellation at top-level-declaration
    // granularity. Body inference is one of the two front-end passes whose
    // cost scales with declaration count, so an abandoned compile has to be
    // able to stop inside it rather than only at the end of the phase. The
    // token is read ONCE (a TLS lookup) and polled per declaration as a
    // relaxed load, leaving the type checker's own recursive walk — which runs
    // many orders of magnitude more often — untouched.
    let cancel = operation.cancellation;
    let cancelled = || cancel.is_some_and(|token| token.is_cancelled());
    super::recursion::reset(operation);
    'schedule: for group in inference_groups {
        let component_diagnostic_checkpoint = errors.checkpoint();
        let cyclic = group.cyclic;
        let recursion_active = !group.recursive_function_indices.is_empty();
        let cycle_precedence_names = if cyclic {
            top_level_references
                .cycle_precedence_targets(&group.indices)
                .into_iter()
                .map(|target| top_level_references.definition(target).name.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut component_scope = cyclic.then(|| {
            ComponentLevelScope::enter(
                &group.indices,
                &items,
                &cycle_precedence_names,
                &mut state.env,
                &state.var_gen,
                &mut state.subst,
            )
        });
        let provisional_types = if cyclic {
            prebind_cyclic_component_schemes(
                &group.indices,
                &items,
                &declared_signatures,
                &metadata_prebound_names,
                &mut state.env,
                &mut state.var_gen,
                &state.subst,
            )
        } else {
            UnordMap::new()
        };
        product.record_group_provisional_types(
            provisional_types
                .to_sorted()
                .into_iter()
                .filter_map(|(index, ty)| {
                    top_level_decl_name(items[*index].1).map(|name| (name.to_string(), ty.clone()))
                }),
        );
        // spec/04 §3.1.1: recursive-instantiation validation remains the
        // function-plan projection. A mixed reference cycle alone must not
        // activate it.
        if recursion_active {
            super::recursion::begin_group(
                group
                    .recursive_function_indices
                    .iter()
                    .filter_map(|&index| {
                        top_level_decl_name(items[index].1).map(|name| {
                            let authored = declared_signatures
                                .get(name)
                                .is_some_and(|metadata| !metadata.binders.is_empty());
                            (name, authored)
                        })
                    }),
                &state.env,
                operation,
            );
        }
        let mut deferred_bindings = Vec::new();
        for &declaration_index in &group.indices {
            if cancelled() {
                if let Some(scope) = component_scope.take() {
                    scope.abort(&mut state.env, &state.var_gen, &mut state.subst, operation);
                }
                break 'schedule;
            }
            let (module, expr) = &items[declaration_index];
            product.begin_root(expr, operation);
            operation.declaration_begin(top_level_decl_name(expr).unwrap_or("<anon>"));
            let decl_name = top_level_decl_name(expr);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), decl_name),
                decl_name.map(str::to_string),
                operation,
            );
            let declaration_diagnostic_owner = declaration_diagnostic_owners
                .get(declaration_index)
                .and_then(Option::as_ref);
            state
                .env
                .set_current_declaration_ordinal(Some(declaration_index));
            let external_input_failure = prebind_literal_external_input_for_declaration(
                declaration_index,
                expr,
                &external_input_types,
                &declared_signatures,
                declaration_diagnostic_owner,
                &mut state.env,
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                errors,
                operation,
            );
            let prebound_type_failure = external_input_failure
                .as_ref()
                .or_else(|| function_metadata_failures.get(&declaration_index));
            if let Some(binding) = infer_top_level(
                expr,
                &mut state.env,
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                errors,
                &mut product,
                prebound_type_failure,
                provisional_types
                    .get(&declaration_index)
                    .map(recursion::RecursiveExpected::Provisional),
                cyclic,
                &declared_signatures,
                declaration_diagnostic_owner,
                operation,
            ) {
                deferred_bindings.push(binding);
            }
            close_declaration(
                &mut product,
                close_scope(cyclic),
                &state.env,
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                errors,
                operation,
            );
            operation.declaration_end(top_level_decl_name(expr).unwrap_or("<anon>"));
        }
        if cancelled() {
            if let Some(scope) = component_scope.take() {
                scope.abort(&mut state.env, &state.var_gen, &mut state.subst, operation);
            }
            break 'schedule;
        }
        if cyclic {
            close_component(
                &mut product,
                &mut state.env,
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                errors,
                operation,
            );
            // Function-recursion validation, when independently active, must
            // clear its pins before component-wide generalization.
            if recursion_active {
                super::recursion::finish_group(&state.subst, errors, operation);
            }
            component_scope
                .take()
                .expect("cyclic component owns an inference-level scope")
                .complete(&mut state.env, &state.var_gen, &mut state.subst);
            let schemes = deferred_bindings
                .into_iter()
                .map(|binding| {
                    generalize_deferred_recursive_binding(
                        binding,
                        &state.env,
                        &state.subst,
                        &mut product,
                        operation,
                    )
                })
                .collect::<Vec<_>>();
            for (name, scheme) in schemes {
                state.env.bind(name, scheme);
            }
            if recursion_active
                && errors
                    .iter_since(component_diagnostic_checkpoint)
                    .next()
                    .is_none()
            {
                sweep_recursive_collection_contracts(
                    &group.recursive_function_indices,
                    &items,
                    &cycle_precedence_names,
                    &mut state.env,
                    &mut state.var_gen,
                    &mut state.subst,
                    &state.adt_reg,
                    errors,
                    &declared_signatures,
                    &declaration_diagnostic_owners,
                    operation,
                );
            }
        }
    }
    crate::opacity::set_current_item(None, None, operation);
    state.env.set_current_declaration_ordinal(None);

    if cancelled() {
        // Abandoned mid-schedule. The remaining declarations were never
        // inferred, so the walks below would attribute their absence to the
        // program and spend the phase's remaining budget doing it. The caller's
        // `cancellation_gate` turns this early return into a hard failure.
        return product;
    }

    product.top_level_references = top_level_references;
    product
}

pub(super) type IrTypeEnv = BTreeMap<String, deep::Expr>;

/// Canonical IR type collection result. The type environment retains the
/// historical last-declaration-wins behavior for duplicate names; the paired
/// ordinal identifies the exact flattened declaration that produced each
/// final entry.
pub(super) struct CollectedIrTypes {
    type_env: IrTypeEnv,
    final_origin_by_name: UnordMap<String, usize>,
}

fn collect_literal_external_input_types(
    items: &[(Option<String>, &deep::Expr)],
    operation: &crate::resources::Operation<'_>,
) -> UnordMap<usize, deep::Expr> {
    let collected = collect_ir_types_with_origins(items.iter().map(|(_, expr)| *expr), operation);
    collected
        .type_env
        .iter()
        .filter_map(|(name, ty_expr)| {
            let declaration_index = collected.final_origin_by_name[name];
            let expr = items[declaration_index].1;
            let (DeepTag::Def, _, kids) = stamped_parts(expr)? else {
                return None;
            };
            let body = kids.get(1)?;
            body_is_type_stamped_literal_self_ref(body, name, operation)
                .then(|| (declaration_index, ty_expr.clone()))
        })
        .collect()
}

/// Prebind the compiler-authored type stamp of each defsig-less function at
/// both checker entries.
///
/// [04-INF-2]/[04-INF-3] make function declarations forward-visible, and a
/// defsig-less serialized function has no other callable header. Restricting
/// this prebind to function bodies keeps [04-INF-4]'s eager-value scope
/// source-ordered. An explicit `defsig` remains authoritative and is never
/// overwritten by body metadata (#1124).
#[allow(clippy::too_many_arguments)]
fn prebind_defsig_less_function_body_types(
    items: &[(Option<String>, &deep::Expr)],
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    declaration_diagnostic_owners: &[Option<DeclarationDiagnosticOwner>],
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> (UnordSet<String>, UnordMap<usize, ErrorWitness>) {
    let CollectedIrTypes {
        type_env,
        final_origin_by_name,
    } = collect_ir_types_with_origins(items.iter().map(|(_, expr)| *expr), operation);
    let mut prebound_names = UnordSet::new();
    let mut failures = UnordMap::new();
    for (name, ty_expr) in type_env {
        let Some(&declaration_index) = final_origin_by_name.get(&name) else {
            continue;
        };
        if declared_signatures.contains_key(&name)
            || !definition_owns_function_metadata_prebind(items[declaration_index].1)
        {
            continue;
        }
        prebound_names.insert(name.clone());
        let metadata_level = subst.enter_level(vg);
        let resolved = resolve_deep_type_with_diagnostic_owner(
            &ty_expr,
            vg,
            adt_reg,
            TypeUseSite::CompilerMetadata,
            BinderMode::TrustedCompilerMetadata,
            declaration_diagnostic_owners
                .get(declaration_index)
                .and_then(Option::as_ref),
            errors,
        );
        subst.leave_level(metadata_level, vg);
        match resolved {
            Ok(ty) => {
                let scheme = env.generalize_in_operation(&ty, subst, operation);
                subst.name_generic_parameters(&scheme, &name, &UnordMap::new());
                env.bind(name, scheme);
            }
            Err(witness) => {
                failures.insert(declaration_index, witness);
            }
        }
    }
    (prebound_names, failures)
}

/// [04-INF-4]'s typed literal self-reference declares an external input. Its
/// own type must therefore be visible while that declaration is inferred,
/// but never to an earlier declaration. Bind exactly the current ordinal just
/// before its body check; ordinary values and future external inputs remain
/// source-ordered at both checker ingresses.
#[allow(clippy::too_many_arguments)]
fn prebind_literal_external_input_for_declaration(
    declaration_index: usize,
    expr: &deep::Expr,
    external_input_types: &UnordMap<usize, deep::Expr>,
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Option<ErrorWitness> {
    let ty_expr = external_input_types.get(&declaration_index)?;
    let name = top_level_decl_name(expr)?;
    if declared_signatures.contains_key(name) {
        return None;
    }
    let metadata_level = subst.enter_level(vg);
    let resolved = resolve_deep_type_with_diagnostic_owner(
        ty_expr,
        vg,
        adt_reg,
        TypeUseSite::CompilerMetadata,
        BinderMode::TrustedCompilerMetadata,
        declaration_diagnostic_owner,
        errors,
    );
    subst.leave_level(metadata_level, vg);
    match resolved {
        Ok(ty) => {
            let scheme = env.generalize_in_operation(&ty, subst, operation);
            subst.name_generic_parameters(&scheme, name, &UnordMap::new());
            env.bind(name.to_string(), scheme);
            None
        }
        Err(witness) => Some(witness),
    }
}

pub(super) fn build_ir_type_env(
    exprs: &[deep::Expr],
    operation: &crate::resources::Operation<'_>,
) -> IrTypeEnv {
    collect_ir_types_with_origins(top_level_decl_items(exprs), operation).type_env
}

pub(super) fn collect_ir_types_with_origins<'a>(
    items: impl IntoIterator<Item = &'a deep::Expr>,
    operation: &crate::resources::Operation<'_>,
) -> CollectedIrTypes {
    let mut type_env = BTreeMap::new();
    let mut final_origin_by_name = UnordMap::new();
    for (declaration_index, expr) in items.into_iter().enumerate() {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
            continue;
        };
        if let Some(ty) = expr_type_expr(body, &type_env, operation) {
            type_env.insert(name.to_string(), ty);
            final_origin_by_name.insert(name.to_string(), declaration_index);
        }
    }
    CollectedIrTypes {
        type_env,
        final_origin_by_name,
    }
}

/// The checker's input normalization: pipes folded into the applications they
/// denote. Every checker entry otherwise sees exactly the tree it was given.
///
/// chelis#1923: `spec/02-surf-syntax.md` §0.1 makes a pipe notation for
/// first-argument insertion, and every consumer that reconstructed the
/// application for itself was re-deriving that sentence. The checker folds
/// once, here, so inference and the annotation pass that follows it see one
/// tree. Folding only inside inference would leave the ANNOTATED tree, which
/// is the checker's output and what the lowerer, linearity, the effect pass
/// and the caches all read, still carrying the `Pipe` node.
fn normalize_program_input(exprs: &[deep::Expr]) -> Vec<deep::Expr> {
    exprs.to_vec()
}

#[cfg(test)]
pub(crate) fn builtin_selection_probe(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Vec<crate::builtin_discovery::BuiltinCaseSelection> {
    let normalized = normalize_program_input(exprs);
    infer_program_with_product_in_session(&normalized, errors, operation).builtin_selections
}

#[cfg(test)]
mod component_level_scope_tests {
    use super::*;

    fn mutual_defs() -> Vec<deep::Expr> {
        chelis_deep::parser::parse_str(
            "(def {} left (fn {} (params {} x) (app {} (var {} right) (var {} x))))
         (def {} right (fn {} (params {} x) (app {} (var {} left) (var {} x))))",
            &chelis_syntax_host::NativeStack,
        )
        .expect("mutual fixture parses")
    }

    #[test]
    fn component_scope_mints_provisionals_and_restores_visibility_on_completion() {
        crate::test_support::run(|operation| {
            let exprs = mutual_defs();
            let items = top_level_decl_items_with_modules(&exprs);
            let indices = vec![0, 1];
            let mut env = Env::new();
            env.bind("left".to_string(), Scheme::mono(Type::Prim(Prim::Int32)));
            env.bind(
                "precedence".to_string(),
                Scheme::mono(Type::Prim(Prim::Int64)),
            );
            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();

            env.note_top_level_value_ordinal("right".to_string(), 1, None);
            env.note_top_level_value_ordinal("outside".to_string(), 2, None);
            env.note_top_level_value_ordinal("precedence".to_string(), 3, None);
            env.set_current_declaration_ordinal(Some(0));
            let _empty_component = env
                .replace_active_top_level_component(["outside".to_string()].into_iter().collect());
            let scope = ComponentLevelScope::enter(
                &indices,
                &items,
                &["precedence".to_string()],
                &mut env,
                &var_gen,
                &mut subst,
            );
            assert!(matches!(
                env.top_level_value_visibility("right"),
                TopLevelValueVisibility::Visible
            ));
            assert!(matches!(
                env.top_level_value_visibility("precedence"),
                TopLevelValueVisibility::Visible
            ));
            assert!(matches!(
                env.top_level_value_visibility("outside"),
                TopLevelValueVisibility::NotYetDeclared { .. }
            ));
            let provisional = prebind_cyclic_component_schemes(
                &indices,
                &items,
                &UnordMap::new(),
                &UnordSet::new(),
                &mut env,
                &mut var_gen,
                &subst,
            );
            for (_, ty) in provisional.to_sorted() {
                for var in chelis_type_core::free_tvars(ty) {
                    assert_eq!(subst.level_of_tvar(var), 1);
                }
            }
            scope.complete(&mut env, &var_gen, &mut subst);

            assert_eq!(subst.current_level(), 0);
            assert!(env.lookup("left").is_none());
            assert!(env.lookup("right").is_none());
            assert_eq!(
                env.lookup("precedence").map(|scheme| &scheme.body),
                Some(&Type::Prim(Prim::Int64)),
                "an already-inferred precedence target is visibility-only"
            );
            assert!(matches!(
                env.top_level_value_visibility("right"),
                TopLevelValueVisibility::NotYetDeclared { .. }
            ));
            assert!(matches!(
                env.top_level_value_visibility("precedence"),
                TopLevelValueVisibility::NotYetDeclared { .. }
            ));
            assert!(matches!(
                env.top_level_value_visibility("outside"),
                TopLevelValueVisibility::Visible
            ));
            assert_eq!(
                super::super::recursion::group_state_counts(operation),
                (0, 0)
            );
        })
    }

    #[test]
    fn component_scope_abort_restores_bindings_levels_pins_and_visibility() {
        crate::test_support::run(|operation| {
            let exprs = mutual_defs();
            let items = top_level_decl_items_with_modules(&exprs);
            let indices = vec![0, 1];
            let mut env = Env::new();
            let prior = Scheme::mono(Type::Prim(Prim::Bool));
            env.bind("left".to_string(), prior.clone());
            env.bind(
                "precedence".to_string(),
                Scheme::mono(Type::Prim(Prim::Int64)),
            );
            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();
            env.note_top_level_value_ordinal("right".to_string(), 1, None);
            env.note_top_level_value_ordinal("outside".to_string(), 2, None);
            env.note_top_level_value_ordinal("precedence".to_string(), 3, None);
            env.set_current_declaration_ordinal(Some(0));
            let _empty_component = env
                .replace_active_top_level_component(["outside".to_string()].into_iter().collect());
            let scope = ComponentLevelScope::enter(
                &indices,
                &items,
                &["precedence".to_string()],
                &mut env,
                &var_gen,
                &mut subst,
            );
            assert!(matches!(
                env.top_level_value_visibility("precedence"),
                TopLevelValueVisibility::Visible
            ));
            prebind_cyclic_component_schemes(
                &indices,
                &items,
                &UnordMap::new(),
                &UnordSet::new(),
                &mut env,
                &mut var_gen,
                &subst,
            );
            super::super::recursion::begin_group(
                ["left", "right"].into_iter().map(|name| (name, false)),
                &env,
                operation,
            );
            let pinned = var_gen.fresh_tvar();
            let _caller = super::super::recursion::begin_caller("left", None, &[], operation);
            super::super::recursion::record_occurrence(
                "right",
                &[(pinned, Type::Var(pinned))],
                None,
                None,
                operation,
            );
            assert_eq!(
                super::super::recursion::group_state_counts(operation),
                (2, 1)
            );

            scope.abort(&mut env, &var_gen, &mut subst, operation);
            assert_eq!(subst.current_level(), 0);
            assert_eq!(
                super::super::recursion::group_state_counts(operation),
                (0, 0)
            );
            assert!(!super::super::recursion::tvar_pinned(pinned, operation));
            let restored = env.lookup("left").expect("shadowed prior binding restored");
            assert_eq!(restored.tvars, prior.tvars);
            assert_eq!(restored.tvar_restrictions, prior.tvar_restrictions);
            assert_eq!(restored.dvars, prior.dvars);
            assert_eq!(restored.rvars, prior.rvars);
            assert_eq!(restored.body, prior.body);
            assert!(env.lookup("right").is_none());
            assert_eq!(
                env.lookup("precedence").map(|scheme| &scheme.body),
                Some(&Type::Prim(Prim::Int64)),
                "abort must not roll back an already-inferred precedence target"
            );
            assert!(matches!(
                env.top_level_value_visibility("right"),
                TopLevelValueVisibility::NotYetDeclared { .. }
            ));
            assert!(matches!(
                env.top_level_value_visibility("precedence"),
                TopLevelValueVisibility::NotYetDeclared { .. }
            ));
            assert!(matches!(
                env.top_level_value_visibility("outside"),
                TopLevelValueVisibility::Visible
            ));

            let follow_up = crate::test_support::infer_program(
                &chelis_deep::parser::parse_str(
                    "(def {} clean (fn {} (params {} x) (var {} x)))",
                    &chelis_syntax_host::NativeStack,
                )
                .expect("follow-up parses"),
                operation,
            );
            assert!(follow_up.errors.is_empty());
        })
    }

    #[test]
    fn primary_driver_mid_scc_cancellation_aborts_before_return() {
        crate::test_support::run(|operation| {
            let exprs = chelis_deep::parser::parse_str(
                "(defsig {} left (t-fn {} (t-prim {} i32) (t-prim {} i32)))
         (defsig {} right (t-fn {} (t-prim {} i32) (t-prim {} i32)))
         (def {} left (fn {} (params {} x) (app {} (var {} right) (var {} x))))
         (def {} right (fn {} (params {} x) (app {} (var {} left) (var {} x))))",
                &chelis_syntax_host::NativeStack,
            )
            .expect("authored mutual-recursion fixture parses");

            let observation = {
                let token = crate::cancel::CancelToken::new();
                let _cancel_guard = crate::cancel::install_cancel_token(token);
                let _hook_guard = cancel_primary_recursive_after_members(1);
                let result = crate::test_support::infer_program(&exprs, operation);
                assert!(
                    result
                        .errors
                        .iter()
                        .any(|error| crate::cancel::is_cancellation(&error.message)),
                    "mid-SCC cancellation must be a hard checker error: {:?}",
                    result.errors
                );
                take_recursive_abort_observation()
                    .expect("the primary driver must abort the incomplete recursive scope")
            };

            assert_eq!(observation.current_level, 0);
            assert_eq!(observation.group_state_counts, (0, 0));
            assert_eq!(observation.member_count, 2);
            assert_eq!(observation.prior_binding_count, 2);
            assert!(observation.all_prior_bindings_restored);
            assert!(observation.all_temporary_bindings_removed);

            let follow_up = crate::test_support::infer_program(
                &chelis_deep::parser::parse_str(
                    "(def {} clean (fn {} (params {} x) (var {} x)))",
                    &chelis_syntax_host::NativeStack,
                )
                .expect("follow-up parses"),
                operation,
            );
            assert!(
                follow_up.errors.is_empty(),
                "a cancelled recursive check must not contaminate its successor: {:?}",
                follow_up.errors
            );
        })
    }
}
