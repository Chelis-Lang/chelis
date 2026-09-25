//! Public check entry points and inference schedules.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use crate::context::LibraryProofId;

/// Transactional owner for one cyclic full-reference component's inference
/// level, provisional visibility capability, and temporary bindings. It
/// snapshots every member binding before provisional prebinding so normal and
/// cancelled exits both restore the surrounding environment exactly.
struct ComponentLevelScope {
    level: crate::unify::LevelToken,
    members: Vec<(String, Option<Scheme>)>,
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
fn schemes_match(left: &Scheme, right: &Scheme) -> bool {
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
        subst: &mut Subst,
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
    fn complete(mut self, env: &mut Env, var_gen: &VarGen, subst: &mut Subst) {
        self.remove_temporary_bindings(env);
        let _finished_component = env
            .replace_active_top_level_component(std::mem::take(&mut self.prior_active_component));
        subst.leave_level(self.level, var_gen);
    }

    fn abort(mut self, env: &mut Env, var_gen: &VarGen, subst: &mut Subst) {
        #[cfg(test)]
        let prior_members = self.members.clone();
        super::recursion::abort_group();
        self.remove_temporary_bindings(env);
        for (name, prior) in self.members {
            if let Some(scheme) = prior {
                env.bind(name, scheme);
            }
        }
        let _aborted_component = env
            .replace_active_top_level_component(std::mem::take(&mut self.prior_active_component));
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
                group_state_counts: super::recursion::group_state_counts(),
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
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    user_def_names: &UnordSet<String>,
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    declaration_diagnostic_owners: &[Option<DeclarationDiagnosticOwner>],
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
        );
        let mut scratch = InferenceProduct::default();
        scratch.type_headers = adt_reg.resolution_env().clone();
        scratch.adt_registry = adt_reg.clone();
        let mut deferred_bindings = Vec::with_capacity(members.len());
        let mut diagnosed = false;

        for (declaration_index, name) in &members {
            if crate::cancel::cancellation_requested() {
                component_scope.abort(env, var_gen, subst);
                return;
            }
            let (module, expr) = &items[*declaration_index];
            scratch.begin_root(expr);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), Some(name)),
                Some(name.clone()),
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
                user_def_names,
                declared_signatures,
                declaration_diagnostic_owner,
            ) {
                deferred_bindings.push(binding);
            }
            scratch.finish_deferred_shape_checks(Some(name), var_gen, subst, adt_reg, errors);
            scratch.finish_deferred_literal_patterns(Some(name), env, subst, adt_reg, errors);
            scratch.finish_root(subst, errors);
            validate_deferred_borrow_vars(subst, adt_reg, env.active_declared_type_names(), errors);
            validate_deferred_tensor_operands(subst, env.active_declared_type_names(), errors);
            validate_deferred_opaque_uses(subst, adt_reg, errors);
            if errors.iter_since(sweep_checkpoint).next().is_some() {
                diagnosed = true;
                break;
            }
        }

        if diagnosed || deferred_bindings.len() != members.len() {
            component_scope.abort(env, var_gen, subst);
            return;
        }

        super::recursion::finish_group(subst, errors);
        if errors.iter_since(sweep_checkpoint).next().is_some() {
            component_scope.abort(env, var_gen, subst);
            return;
        }
        component_scope.complete(env, var_gen, subst);
        let schemes = deferred_bindings
            .into_iter()
            .map(|binding| generalize_deferred_recursive_binding(binding, env, subst))
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
fn bind_library_products(
    mut type_env: TypeEnv,
    mut checked: CheckedProgram,
    context_library_proof_id: Option<LibraryProofId>,
) -> (TypeEnv, CheckedProgram) {
    let selector_context_digest =
        selector_callable_context_digest(&type_env.inner().selector_callables);
    let proof_id = LibraryProofId::for_library(
        checked.exprs(),
        context_library_proof_id,
        selector_context_digest,
    );
    type_env.bind_library_proof(proof_id);
    checked.bind_library_proof(proof_id);
    checked.bind_context_library_proof(context_library_proof_id);
    (type_env, checked)
}

fn cancellation_gate(errors: &mut DiagnosticSink<'_>) -> bool {
    if !crate::cancel::cancellation_requested() {
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
pub fn infer_program(exprs: &[deep::Expr]) -> InferResult {
    // WI-1 follow-up: run the whole pipeline on a grown stack so a deeply
    // nested but finite program checks end-to-end instead of tripping a
    // per-site `stack_guard!` partway through one of the recursive passes.
    crate::session::infer_program(exprs)
}

pub(crate) fn infer_program_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> InferStats {
    let normalized = normalize_program_input(exprs);
    infer_program_with_product_in_session(&normalized, errors).stats()
}

pub(super) fn infer_program_with_product_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> InferenceProduct {
    // Reset the stack-exhaustion flag for this check unit; `drain` below
    // turns any walker stack bail into a hard located error so deep input
    // can never produce a silent green / partial result.
    let stack_scope = StackExhaustionScope::enter();
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    builtins::register_prelude_adts(&mut env, &mut vg, &mut adt_reg);
    let mut product = InferenceProduct::default();

    // RFC v4b (RT-1 F2): reject a named module opened by more than one
    // wrapper in this check unit (module-identity forgery).
    detect_module_reopens(exprs, errors);
    // RFC v5 (RT-1 F2 bypass): reject the reef linker's reserved
    // internal-name format in programs not produced by the linker.
    detect_forged_linker_names(exprs, errors);

    // First pass: collect deftype and defsig declarations. Descend through
    // `(module {} name ...)` wrappers so declarations in every idiomatic
    // Surf source (every .ch starts with `module X`) get collected.
    let items = top_level_decl_items_with_modules(exprs);
    let declaration_diagnostic_owners = declaration_diagnostic_owner_plan(exprs);
    collect_all_declarations(
        &items,
        &declaration_diagnostic_owners,
        &mut env,
        &mut vg,
        &mut subst,
        &mut adt_reg,
        errors,
    );
    product.type_headers = adt_reg.resolution_env().clone();
    product.adt_registry = adt_reg.clone();

    // Checker-enforced opacity (RFC D-CHECK): install the per-run
    // context so the inference hooks see module identity, exports,
    // and producer text. Dropped at the end of this function.
    let opacity_meta = build_opacity_meta(&items, &adt_reg, &env);
    let _opacity_guard = crate::opacity::install_opacity_context(
        crate::opacity::OpacityContextData::from_meta(opacity_meta),
    );

    // Second pass: infer def bodies. Same module-descent rationale as the
    // declaration pass — without it, the entire HM checker is a no-op on
    // module-wrapped programs.
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));
    let declared_signatures = collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr));
    validate_binder_literal_adoption_in_program(&items, &declared_signatures, errors);
    let external_input_types = collect_literal_external_input_types(&items);
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
        );
    let top_level_references = TopLevelReferenceGraph::build(&items);
    product.function_inference_plan =
        FunctionInferencePlan::build_from_reference_graph(&top_level_references);
    let inference_groups = primary_inference_groups_with_reference_graph(
        &product.function_inference_plan,
        &items,
        &top_level_references,
    );
    // Match the persisted-state driver: cache the TLS token once and poll at
    // declaration granularity. In particular, an incomplete cyclic component
    // must consume its structured scope through `abort` before this schedule
    // returns; finishing or generalizing a partially inferred group would
    // leak its pins, temporary bindings, and child level.
    let cancel = crate::cancel::current_cancel_token();
    let cancelled = || cancel.as_ref().is_some_and(CancelToken::is_cancelled);
    super::recursion::reset();
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
            )
        } else {
            UnordMap::new()
        };
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
            );
        }
        let mut deferred_bindings = Vec::new();
        for &declaration_index in &group.indices {
            if cancelled() {
                if let Some(scope) = component_scope.take() {
                    scope.abort(&mut env, &vg, &mut subst);
                }
                break 'schedule;
            }
            let (module, expr) = &items[declaration_index];
            product.begin_root(expr);
            let decl_name = top_level_decl_name(expr);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), decl_name),
                decl_name.map(str::to_string),
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
                &user_def_names,
                &declared_signatures,
                declaration_diagnostic_owner,
            ) {
                deferred_bindings.push(binding);
            }
            product.finish_deferred_shape_checks(decl_name, &mut vg, &mut subst, &adt_reg, errors);
            product.finish_deferred_literal_patterns(decl_name, &env, &subst, &adt_reg, errors);
            product.finish_root(&subst, errors);
            // Issue #256 round 2: re-check each deferred borrow against the
            // now-complete substitution (see `validate_deferred_borrow_vars`).
            validate_deferred_borrow_vars(
                &subst,
                &adt_reg,
                env.active_declared_type_names(),
                errors,
            );
            // chelis#1489: decide this def's deferred operands against the
            // final substitution (see `validate_deferred_tensor_operands`).
            // One pass, per def — an earlier revision had a second,
            // whole-program phase that three library lanes never reached.
            validate_deferred_tensor_operands(&mut subst, env.active_declared_type_names(), errors);
            // D-CHECK: drain the per-def deferred-access ledger (see
            // `validate_deferred_opaque_uses`).
            validate_deferred_opaque_uses(&subst, &adt_reg, errors);
            #[cfg(test)]
            if cyclic {
                primary_recursive_member_finished_for_test();
            }
        }
        if cancelled() {
            if let Some(scope) = component_scope.take() {
                scope.abort(&mut env, &vg, &mut subst);
            }
            break 'schedule;
        }
        if cyclic {
            // Function-recursion validation, when independently active, must
            // clear its pins before component-wide generalization.
            if recursion_active {
                super::recursion::finish_group(&subst, errors);
            }
            component_scope
                .take()
                .expect("cyclic component owns an inference-level scope")
                .complete(&mut env, &vg, &mut subst);
            let schemes = deferred_bindings
                .into_iter()
                .map(|binding| generalize_deferred_recursive_binding(binding, &env, &subst))
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
                    &user_def_names,
                    &declared_signatures,
                    &declaration_diagnostic_owners,
                );
            }
        }
    }
    crate::opacity::set_current_item(None, None);
    env.set_current_declaration_ordinal(None);

    if cancellation_gate(errors) {
        return product;
    }
    product.resolve_owner_types(&subst);

    // PP9 / [04-TOT-5]: every public entry reaches one ordered semantic pass
    // protocol after inference. Build the same declaration-type view the IR
    // entries supply; stamped input stays stamped and every reader below must
    // handle that carrier directly.
    let semantic_type_env = build_ir_type_env(exprs);
    validate_semantic_program(
        exprs,
        &semantic_type_env,
        &top_level_references,
        &SelectorCallableContext::default(),
        errors,
    );
    if cancellation_gate(errors) {
        stack_scope.drain_into(errors);
        product.top_level_references = top_level_references;
        return product;
    }

    // Final shared checks: reject tensor types whose element precision isn't supported
    // by the Phase 0f backend (f16/bf16/f8e4m3). These would silently get
    // downcast to f32 by the current build targets, violating the "no implicit
    // precision promotion" rule. f64 is supported as of v0.2.3.
    validate_tensor_precisions_in_program(exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(exprs, errors);

    // If any walker bailed on a nearly-exhausted stack during this run,
    // surface it as a hard located failure (covered-or-rejected).
    stack_scope.drain_into(errors);

    product.top_level_references = top_level_references;
    product
}

pub fn check_ir_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    // Compose: empty outer scope, then check exprs as new code against it.
    // This keeps a single source of truth for the IR check pipeline.
    check_ir_with_context(&TypeEnv::empty(), exprs)
}

/// Build a stacked outer-scope context from a library decl list. The
/// library is run through the full IR pipeline; if any errors are
/// found they are returned to the caller (the context cannot be built
/// from an unchecked library).
///
/// Once built, the returned [`TypeEnv`] can be re-used to type-check
/// many separate "new code" snippets via
/// [`check_ir_with_context`]. The library state is `Arc`-shared and
/// never mutated, so concurrent reads are cheap.
pub fn build_type_env_from_library(library_exprs: &[deep::Expr]) -> Result<TypeEnv, InferResult> {
    crate::session::build_type_env_from_library(library_exprs)
}

pub(crate) fn build_type_env_from_library_in_session(
    library_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Result<TypeEnv, InferStats> {
    let normalized = normalize_program_input(library_exprs);
    let library_exprs = &normalized;
    // Reset the stack-exhaustion flag for this check unit; drained below
    // before the empty-errors gate (covered-or-rejected on deep input).
    let stack_scope = StackExhaustionScope::enter();
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut sub_t = std::time::Instant::now();
    let log_sub = |label: &str, t: &mut std::time::Instant| {
        if detail_profile {
            eprintln!(
                "build_type_env_sub: {:>8.4}s {}",
                t.elapsed().as_secs_f64(),
                label
            );
            *t = std::time::Instant::now();
        }
    };
    // Start from the empty (builtins + prelude ADTs) state.
    let empty = TypeEnv::empty();
    let mut state = empty.inner().clone();

    // Library IR declared-type lookup.
    let library_ir = build_ir_type_env(library_exprs);
    log_sub("build_ir_type_env_initial", &mut sub_t);

    let product = infer_ir_program_with_state(library_exprs, &mut state, errors);
    let stats = product.stats();
    log_sub("infer_ir_program_with_state", &mut sub_t);
    // chelis#930: inference may have abandoned its schedule. Stop before the
    // validators, which would otherwise report the un-inferred tail as program
    // errors and spend the rest of the phase doing it.
    if cancellation_gate(errors) {
        return Err(stats);
    }
    validate_semantic_program(
        library_exprs,
        &library_ir,
        &product.top_level_references,
        &SelectorCallableContext::default(),
        errors,
    );
    log_sub("validate_semantic_program", &mut sub_t);
    validate_tensor_precisions_in_program(library_exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    log_sub("validate_tensor_precisions", &mut sub_t);
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
    log_sub("collect_library_def_names", &mut sub_t);

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
    let declared_signatures = collect_declared_sig_metadata(library_exprs);
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    let library_annotated =
        annotate_top_levels(library_exprs, &product, annotation_context, errors);
    log_sub("annotate_library_exprs_outer_loop", &mut sub_t);
    if cancellation_gate(errors) {
        return Err(stats);
    }
    let library_ir_annotated = build_ir_type_env(&library_annotated);
    log_sub("build_ir_type_env_from_annotated", &mut sub_t);

    // The annotation loop above recurses (annotate_expr_with_scope); if it
    // bailed on low stack, reject rather than return a partially-annotated
    // library context.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    let selector_callables =
        extend_selector_callable_context(&library_annotated, &SelectorCallableContext::default());

    Ok(TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated,
        library_def_names,
        selector_callables,
        opacity: state.opacity,
    }))
}

/// Combined library-build helper: run the IR pipeline ONCE over the
/// library and return both the [`TypeEnv`] (for downstream `_with_context`
/// calls) and a [`CheckedProgram`] equivalent to what
/// `check_ir_with_context(&TypeEnv::empty(), library_exprs)` would
/// return.
///
/// This avoids the duplicated work that occurs when callers run
/// [`build_type_env_from_library`] followed by
/// `check_ir_with_context(empty, library)` — both paths separately
/// run a full HM inference + annotation pass over the same library
/// exprs. Per `docs/archive/perf/perf_baseline_investigation.md`, the unified path
/// saves ~16s of duplicated inference + annotation on Coral.
///
/// Behavior contract:
/// - The returned `TypeEnv` is identical (modulo non-determinism in
///   `UnordMap` iteration) to `build_type_env_from_library(library_exprs)`.
/// - The returned `CheckedProgram` has the same `annotated_exprs()` and
///   `type_env()` shapes that
///   `check_ir_with_context(&TypeEnv::empty(), library_exprs)`
///   produces — namely, annotated library exprs in source order plus a
///   `ir_types` map keyed on every library def.
/// - On any error the same `Err(InferResult)` is returned that the
///   sequential calls would have returned.
///
/// Internal sequencing:
/// 1. Build the per-decl `IrTypeEnv` from un-annotated source.
/// 2. Run `infer_ir_program_with_state` once, populating `state`.
/// 3. Run all validators (`validate_semantic_program`,
///    `validate_tensor_precisions_in_program`).
/// 4. Annotate the library exprs once using the populated `state.env`.
/// 5. Build `library_ir_annotated` from the annotated exprs.
/// 6. Compose the `TypeEnv` from `state` + `library_ir_annotated`.
/// 7. Compose the `CheckedProgram` from the annotated exprs +
///    `library_ir_annotated`.
pub fn build_compiled_library_context(
    library_exprs: &[deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    crate::session::build_compiled_library_context(library_exprs)
}

pub(crate) fn build_compiled_library_context_in_session(
    library_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Result<(TypeEnv, CheckedProgram), InferStats> {
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
    let folded = chelis_deep::pipe::fold_program_pipes(library_exprs);
    let library_exprs = &folded[..];
    let stack_scope = StackExhaustionScope::enter();
    // Mirror `build_type_env_from_library` up through the validators so the
    // type_env half stays bit-compatible with the existing public API.
    let empty = TypeEnv::empty();
    let mut state = empty.inner().clone();

    let library_ir = build_ir_type_env(library_exprs);

    let product = infer_ir_program_with_state(library_exprs, &mut state, errors);
    let stats = product.stats();
    // chelis#930: see `build_type_env_from_library_in_session` — stop before
    // the validators if inference abandoned its schedule.
    if cancellation_gate(errors) {
        return Err(stats);
    }
    validate_semantic_program(
        library_exprs,
        &library_ir,
        &product.top_level_references,
        &SelectorCallableContext::default(),
        errors,
    );
    validate_tensor_precisions_in_program(library_exprs, errors);
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
    let declared_signatures = collect_declared_sig_metadata(library_exprs);
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    let library_annotated =
        annotate_top_levels(library_exprs, &product, annotation_context, errors);
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    if cancellation_gate(errors) {
        return Err(stats);
    }
    let library_ir_annotated = build_ir_type_env(&library_annotated);
    let selector_callables =
        extend_selector_callable_context(&library_annotated, &SelectorCallableContext::default());

    let type_env = TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated.clone(),
        library_def_names,
        selector_callables,
        opacity: state.opacity,
    });

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
    );
    if !errors.is_empty() {
        return Err(stats);
    }
    // chelis#930: see `check_ir_with_signature_context_in_session`.
    if cancellation_gate(errors) {
        return Err(stats);
    }

    Ok(bind_library_products(type_env, checked, None))
}

/// Layered sibling of [`build_compiled_library_context`]: build a library
/// context for `library_exprs` *stacked on top of* an existing `base`
/// context instead of on the empty (builtins + prelude) state.
///
/// This is the seam the cross-process chelis-std typecheck cache uses for
/// its Layer 2 build: `base` is the cached chelis-std sub-context's
/// `TypeEnv`, and `library_exprs` is the non-chelis-std library decls
/// (the user package's own modules + path-deps). The chelis-std library
/// is checked once, cached, and never re-walked here; only the
/// `library_exprs` passed in are inferred + annotated.
///
/// Behavior contract:
/// - `library_exprs` are checked against `base` exactly as
///   [`check_ir_with_context`] would check new code against `base` — base
///   bindings are visible, base ADT constructor sets remain visible to
///   `match` exhaustivity, and `library_exprs` bindings shadow but do not
///   consume base bindings.
/// - The returned `TypeEnv` carries the **union** of base + `library_exprs`
///   declared types and def-name sets, so a subsequent
///   `check_ir_with_context` against it resolves `(var ...)` references
///   into both the base (chelis-std) and the `library_exprs` (package)
///   layers. `library_exprs` types win on shadow.
/// - The returned `CheckedProgram` carries the `library_exprs` annotated
///   bodies (NOT the base bodies — base bodies live in the base context's
///   own `CheckedProgram`). Downstream effects / linearity / lowering must
///   compose this against the base context's `CheckedProgram` /
///   `LoweredLibrary` via the `_with_context` variants, exactly as the
///   monolithic-vs-layered split requires.
/// - On any error the same `Err(InferResult)` is returned that
///   `check_ir_with_context(base, library_exprs)` would return.
pub fn build_compiled_library_context_with_base(
    base: &TypeEnv,
    library_exprs: &[deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    crate::session::build_compiled_library_context_with_base(base, library_exprs)
}

pub(crate) fn build_compiled_library_context_with_base_in_session(
    base: &TypeEnv,
    library_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Result<(TypeEnv, CheckedProgram), InferStats> {
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
    let folded = chelis_deep::pipe::fold_program_pipes(library_exprs);
    let library_exprs = &folded[..];
    let stack_scope = StackExhaustionScope::enter();
    // Seed from the base context's snapshot rather than the empty state.
    let mut state = base.resume_for_new_check();

    // `library_exprs` declared types (IR), layered on top of the base's.
    let new_ir = build_ir_type_env(library_exprs);
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
    let product = infer_ir_program_with_state(library_exprs, &mut state, errors);
    let stats = product.stats();
    // chelis#930: see `build_type_env_from_library_in_session` — stop before
    // the validators if inference abandoned its schedule.
    if cancellation_gate(errors) {
        return Err(stats);
    }
    validate_semantic_program(
        library_exprs,
        &combined_ir,
        &product.top_level_references,
        &base.inner().selector_callables,
        errors,
    );
    validate_tensor_precisions_in_program(library_exprs, errors);
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
    let mut library_def_names = base.inner().library_def_names.clone();
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
    let declared_signatures = collect_declared_sig_metadata(library_exprs);
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    let library_annotated =
        annotate_top_levels(library_exprs, &product, annotation_context, errors);
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    if cancellation_gate(errors) {
        return Err(stats);
    }
    let new_ir_annotated = build_ir_type_env(&library_annotated);
    let selector_callables =
        extend_selector_callable_context(&library_annotated, &base.inner().selector_callables);

    // The returned TypeEnv's `ir_types` is the union: base declared types
    // plus this layer's, this layer winning on shadow.
    let mut combined_ir_annotated = new_ir_annotated.clone();
    for (name, ty) in &base.inner().ir_types {
        combined_ir_annotated
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }

    let type_env = TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: combined_ir_annotated,
        library_def_names,
        selector_callables,
        opacity: state.opacity,
    });

    // The CheckedProgram carries this layer's annotated bodies plus a
    // unioned `type_env` so downstream `_with_context` passes resolve
    // both base and this-layer `(var ...)` references. This mirrors the
    // `check_ir_with_context` returned-CheckedProgram contract.
    let mut checked_type_env = new_ir_annotated;
    for (name, ty) in &base.inner().ir_types {
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
    );
    if !errors.is_empty() {
        return Err(stats);
    }
    // chelis#930: see `check_ir_with_signature_context_in_session`.
    if cancellation_gate(errors) {
        return Err(stats);
    }

    Ok(bind_library_products(
        type_env,
        checked,
        base.library_proof_id(),
    ))
}

/// Type-check `new_exprs` against an outer-scope `context`. New-code
/// bindings shadow but do not consume library bindings; library ADT
/// constructor sets remain visible to new-code `match` exhaustivity
/// checks. The returned [`CheckedProgram`] contains ONLY the new-code's
/// checked decls; library decls are not duplicated.
///
/// The `context` is `Arc`-shared and never mutated — repeated calls
/// against the same context see the same outer scope.
///
/// ## Downstream-caller contract
///
/// The returned `CheckedProgram` is asymmetric on purpose:
/// - `type_env()` is **unioned** — it carries library + new-code declared
///   types so callers like `chelis_ir::lower_program` and
///   `chelis_effects::check_program` can resolve `(var libname)` references
///   from new-code bodies. New-code types win on shadow.
/// - `annotated_exprs()` is **new-code only** — library bodies are NOT
///   present. Phase D / E / F (effects, linearity, lowering) callers MUST
///   use the corresponding `_with_context` variants, not the monolithic
///   `check_program` / `check_linearity` / `lower_program`. The monolithic
///   APIs need to walk library bodies and will silently mis-handle
///   library-effect propagation, library tensor consumption, and library
///   IR roots if fed only the new-code annotated decls.
pub fn check_ir_with_context(
    context: &TypeEnv,
    new_exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    check_ir_with_signature_context(context, &SignatureInferenceMetadata::default(), new_exprs)
}

pub fn check_ir_with_signature_context(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    let mut checked =
        crate::session::check_ir_with_signature_context(context, signature_context, new_exprs)?;
    checked.bind_context_library_proof(context.library_proof_id());
    Ok(checked)
}

pub(crate) fn check_ir_with_signature_context_in_session(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Result<CheckedProgram, InferStats> {
    let normalized = normalize_program_input(new_exprs);
    let new_exprs = &normalized;
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate so a deep-input stack
    // bail always fails the check (never a silent green / partial result).
    let stack_scope = StackExhaustionScope::enter();
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut sub_t = std::time::Instant::now();
    let log_sub = |label: &str, t: &mut std::time::Instant| {
        if detail_profile {
            eprintln!(
                "check_ir_sub: {:>8.4}s {}",
                t.elapsed().as_secs_f64(),
                label
            );
            *t = std::time::Instant::now();
        }
    };
    let mut state = context.resume_for_new_check();

    // New-code declared types (IR) layered on top of library's.
    let new_ir = build_ir_type_env(new_exprs);
    let combined_ir: BTreeMap<String, deep::Expr> = state
        .ir_types
        .iter()
        .chain(new_ir.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    log_sub("build_ir_and_combine", &mut sub_t);

    // Library is already validated; only run validate / inference on
    // new exprs. Inference's canonical collector binds the new-code's
    // own declared types (library schemes are already in state.env).
    let mut product = infer_ir_program_with_state(new_exprs, &mut state, errors);
    let stats = product.stats();
    log_sub("infer_ir_program_with_state", &mut sub_t);
    // chelis#930: see `build_type_env_from_library_in_session` — stop before
    // the validators if inference abandoned its schedule.
    if cancellation_gate(errors) {
        return Err(stats);
    }
    product.resolve_owner_types(&state.subst);
    // Run cycle / shape / precision validators on new_exprs only. The
    // combined IR env is supplied so `(var libfoo)` references
    // resolve to the library's declared type during shape validation.
    validate_semantic_program(
        new_exprs,
        &combined_ir,
        &product.top_level_references,
        &context.inner().selector_callables,
        errors,
    );
    log_sub("validate_semantic_program", &mut sub_t);
    validate_tensor_precisions_in_program(new_exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(new_exprs, errors);
    log_sub("validate_tensor_precisions", &mut sub_t);
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Annotate ONLY the new-code exprs, starting from the library
    // snapshot state so library names resolve during annotation.
    let annotated_exprs = annotate_ir_program(new_exprs, &product, errors);
    log_sub("annotate_ir_program", &mut sub_t);
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    if cancellation_gate(errors) {
        return Err(stats);
    }
    // Surface library declared types in the returned type_env so downstream
    // passes (lower, effects, linearity) can resolve `(var libname)` calls
    // from new-code without a separate library lookup. New-code types take
    // precedence on shadow.
    let mut annotated_type_env = build_ir_type_env(&annotated_exprs);
    for (name, ty) in &context.inner().ir_types {
        annotated_type_env
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }
    log_sub("annotated_type_env_build", &mut sub_t);
    let checked = finalize_checked_program(
        annotated_exprs,
        annotated_type_env,
        signature_context,
        &product,
        stats,
        errors,
    );
    if !errors.is_empty() {
        return Err(stats);
    }
    // chelis#930: signature inference (inside `finalize_checked_program`) can
    // abandon its fixed point, which leaves the metadata short rather than
    // wrong-looking, so nothing above would have failed.
    if cancellation_gate(errors) {
        return Err(stats);
    }
    Ok(checked)
}

pub fn check_typed_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    crate::session::check_typed_program(exprs)
}

pub(crate) fn check_typed_program_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Result<CheckedProgram, InferStats> {
    // Outermost scope covers both inference (which has its own inner scope)
    // and the annotation pass below, so a bail in either surfaces as a hard
    // located failure rather than a partially-annotated `Ok`.
    let stack_scope = StackExhaustionScope::enter();
    // One tree for inference and annotation: the pipe fold, and nothing else
    // (chelis#1023 pins that the checker's output keeps its input's nodes).
    let exprs = &chelis_deep::pipe::fold_program_pipes(exprs)[..];
    let product = infer_program_with_product_in_session(exprs, errors);
    let stats = product.stats();
    if errors.is_empty() {
        let annotated_exprs = annotate_ir_program(exprs, &product, errors);
        let annotated_type_env = build_ir_type_env(&annotated_exprs);
        // Annotation recurses; reject if it bailed on low stack.
        stack_scope.drain_into(errors);
        if !errors.is_empty() {
            return Err(stats);
        }
        // spec/06 §3.7: an extent derived from vmapped tensor elements is a
        // check-time `batch_varying_extent` type error at every source
        // ingress. IR reaches this validator before annotation; typed Surf
        // supplies the equivalent canonical annotated type environment here.
        validate_vmap_extent_dependencies(&annotated_exprs, &annotated_type_env, errors);
        if !errors.is_empty() {
            return Err(stats);
        }
        // chelis#930: annotation and the signature fixed point below both
        // abandon their walks on cancellation.
        if cancellation_gate(errors) {
            return Err(stats);
        }
        let checked = finalize_checked_program(
            annotated_exprs,
            annotated_type_env,
            &SignatureInferenceMetadata::default(),
            &product,
            stats,
            errors,
        );
        if !errors.is_empty() {
            return Err(stats);
        }
        if cancellation_gate(errors) {
            return Err(stats);
        }
        Ok(checked)
    } else {
        Err(stats)
    }
}

pub fn infer_ir_program(exprs: &[deep::Expr]) -> InferResult {
    crate::session::infer_ir_program(exprs)
}

pub(crate) fn infer_ir_program_in_session(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> InferStats {
    let normalized = normalize_program_input(exprs);
    let exprs = &normalized;
    let stack_scope = StackExhaustionScope::enter();
    let type_env = build_ir_type_env(exprs);
    let product = infer_ir_program_with_env(exprs, &type_env, errors);
    let stats = product.stats();
    // chelis#930 review: this was the one `*_in_session` entry with no gate.
    // Without it, a tripped token let every pass below stop early and the
    // walk's truncated counts flow into a clean-looking report -- compiler::
    // check would return score 1.0 with an empty error list for a program
    // whose tail was never inspected (contradicting section C4.4 and the
    // repo Contract Invariant: perfect success requires an empty error list
    // to be HONEST, and a truncated walk is not).
    if cancellation_gate(errors) {
        stack_scope.drain_into(errors);
        return stats;
    }
    validate_semantic_program(
        exprs,
        &type_env,
        &product.top_level_references,
        &SelectorCallableContext::default(),
        errors,
    );
    if cancellation_gate(errors) {
        stack_scope.drain_into(errors);
        return stats;
    }
    validate_tensor_precisions_in_program(exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(exprs, errors);
    // Surface any walker stack bail as a hard located error.
    stack_scope.drain_into(errors);
    stats
}

pub(super) fn infer_ir_program_with_env(
    exprs: &[deep::Expr],
    _type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) -> InferenceProduct {
    // Backwards-compat wrapper. Public callers run the shared semantic pass
    // protocol after this driver returns.
    let empty_inner = crate::context::TypeEnv::empty();
    let mut state = empty_inner.inner().clone();
    let mut product = infer_ir_program_with_state(exprs, &mut state, errors);
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
    state: &mut TypeEnvInner,
    errors: &mut DiagnosticSink<'_>,
) -> InferenceProduct {
    let mut product = InferenceProduct::default();

    // RFC v4b (RT-1 F2): reject a named module opened by more than one
    // wrapper in this check unit (module-identity forgery). Reef-linked
    // decls carry no wrappers, so this only fires on hand-written `.dp`.
    detect_module_reopens(exprs, errors);
    // RFC v5 (RT-1 F2 bypass): reject the reef linker's reserved
    // internal-name format in programs not produced by the linker.
    detect_forged_linker_names(exprs, errors);

    // Descend through `(module {} name ...)` wrappers: every idiomatic
    // Surf source wraps its declarations in `module X`, and without
    // flattening none of the walkers below see any def/defsig/deftype.
    let items = top_level_decl_items_with_modules(exprs);
    let declaration_diagnostic_owners = declaration_diagnostic_owner_plan(exprs);
    collect_all_declarations(
        &items,
        &declaration_diagnostic_owners,
        &mut state.env,
        &mut state.var_gen,
        &mut state.subst,
        &mut state.adt_reg,
        errors,
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
    );

    // Per-decl profile: when CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1, emit
    // one stderr line per top-level decl with its name and inference time.
    // Aggregated by name in caller scripts to attribute cost per module.
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));
    let declared_signatures = collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr));
    validate_binder_literal_adoption_in_program(&items, &declared_signatures, errors);
    // PP9 uses the same two bounded prebind capabilities at both entries:
    // [04-INF-4] external-input ascriptions are declaration-local, while a
    // defsig-less function body stamp supplies the callable header that
    // [04-INF-2]/[04-INF-3] make forward-visible. Eager values are never
    // prebound from body metadata, preserving #1134's source-order rule.
    let external_input_types = collect_literal_external_input_types(&items);
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
        );
    let top_level_references = TopLevelReferenceGraph::build(&items);
    product.function_inference_plan =
        FunctionInferencePlan::build_from_reference_graph(&top_level_references);
    let inference_groups = primary_inference_groups_with_reference_graph(
        &product.function_inference_plan,
        &items,
        &top_level_references,
    );
    // chelis#930: cooperative cancellation at top-level-declaration
    // granularity. Body inference is one of the two front-end passes whose
    // cost scales with declaration count, so an abandoned compile has to be
    // able to stop inside it rather than only at the end of the phase. The
    // token is read ONCE (a TLS lookup) and polled per declaration as a
    // relaxed load, leaving the type checker's own recursive walk — which runs
    // many orders of magnitude more often — untouched.
    let cancel = crate::cancel::current_cancel_token();
    let cancelled = || cancel.as_ref().is_some_and(CancelToken::is_cancelled);
    super::recursion::reset();
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
            )
        } else {
            UnordMap::new()
        };
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
            );
        }
        let mut deferred_bindings = Vec::new();
        for &declaration_index in &group.indices {
            if cancelled() {
                if let Some(scope) = component_scope.take() {
                    scope.abort(&mut state.env, &state.var_gen, &mut state.subst);
                }
                break 'schedule;
            }
            let (module, expr) = &items[declaration_index];
            product.begin_root(expr);
            let t0 = if detail_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let decl_name = top_level_decl_name(expr);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), decl_name),
                decl_name.map(str::to_string),
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
                &user_def_names,
                &declared_signatures,
                declaration_diagnostic_owner,
            ) {
                deferred_bindings.push(binding);
            }
            product.finish_deferred_shape_checks(
                top_level_decl_name(expr),
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                errors,
            );
            product.finish_deferred_literal_patterns(
                top_level_decl_name(expr),
                &state.env,
                &state.subst,
                &state.adt_reg,
                errors,
            );
            product.finish_root(&state.subst, errors);
            if let Some(t0) = t0 {
                let elapsed = t0.elapsed();
                let name = top_level_decl_name(expr).unwrap_or("<anon>");
                eprintln!("infer_ir_decl: {:>8.4}s {}", elapsed.as_secs_f64(), name);
            }
            // Issue #256 round 2: drain the deferred-borrow ledger for this
            // def and re-check each recorded variable against the now-complete
            // substitution. Draining per-def keeps error attribution local and
            // prevents one def's deferrals from leaking into the next.
            validate_deferred_borrow_vars(
                &state.subst,
                &state.adt_reg,
                state.env.active_declared_type_names(),
                errors,
            );
            // chelis#1489: see `validate_deferred_tensor_operands`.
            validate_deferred_tensor_operands(
                &mut state.subst,
                state.env.active_declared_type_names(),
                errors,
            );
            // D-CHECK: drain the per-def deferred-access ledger (see
            // `validate_deferred_opaque_uses`).
            validate_deferred_opaque_uses(&state.subst, &state.adt_reg, errors);
        }
        if cancelled() {
            if let Some(scope) = component_scope.take() {
                scope.abort(&mut state.env, &state.var_gen, &mut state.subst);
            }
            break 'schedule;
        }
        if cyclic {
            // Function-recursion validation, when independently active, must
            // clear its pins before component-wide generalization.
            if recursion_active {
                super::recursion::finish_group(&state.subst, errors);
            }
            component_scope
                .take()
                .expect("cyclic component owns an inference-level scope")
                .complete(&mut state.env, &state.var_gen, &mut state.subst);
            let schemes = deferred_bindings
                .into_iter()
                .map(|binding| {
                    generalize_deferred_recursive_binding(binding, &state.env, &state.subst)
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
                    &user_def_names,
                    &declared_signatures,
                    &declaration_diagnostic_owners,
                );
            }
        }
    }
    crate::opacity::set_current_item(None, None);
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

/// The hoist order: the total order the body-inference schedule used before
/// chelis#1134 and still uses as its priority. Every module function is
/// spliced at the earliest module-function ordinal in the planner's
/// callee-first order, and every other item keeps textual order.
///
/// This order is a linear extension of every precedence edge
/// [`primary_inference_schedule`] builds except a barrier into a hoisted
/// function (an eager value declared before a module function that reads it),
/// which is exactly the [04-INF-4] defect the schedule exists to repair. The
/// schedule therefore reproduces this order on every program that carries no
/// such barrier, up to the contiguity of a contracted recursive component, so
/// the grouped order [`primary_inference_groups`] infers is identical.
fn hoist_order(
    function_plan: &FunctionInferencePlan,
    item_count: usize,
    module_fn_indices: &BTreeSet<usize>,
) -> Vec<usize> {
    let Some(&insertion) = module_fn_indices.first() else {
        return (0..item_count).collect();
    };
    let ordered_module_fns = function_plan
        .ordered_members()
        .map(|member| member.item_index)
        .filter(|index| module_fn_indices.contains(index))
        .collect::<Vec<_>>();
    let mut order = Vec::with_capacity(item_count);
    for index in 0..item_count {
        if index == insertion {
            order.extend(ordered_module_fns.iter().copied());
        }
        if !module_fn_indices.contains(&index) {
            order.push(index);
        }
    }
    order
}

/// Eager (non-function) top-level value `def`s, as name -> flattened ordinal.
/// The first `def` of a duplicated name owns the position, matching
/// `Env::note_top_level_value_ordinal`; the duplicate is already an error.
fn eager_value_definition_ordinals(
    items: &[(Option<String>, &deep::Expr)],
) -> BTreeMap<String, usize> {
    let mut ordinals = BTreeMap::new();
    for (index, (_, expr)) in items.iter().enumerate() {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if definition_owns_function_metadata_prebind(expr) {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        ordinals.entry(name.to_string()).or_insert(index);
    }
    ordinals
}

/// Does this Deep TYPE expression contain an inference hole?
///
/// chelis#1486 / [04-INF-5]: a wildcard slot is `(t-var {} _)` in a type
/// position, `(d-var {} _)` in a dimension position, and `(d-rank {} _)` in a
/// rank position. `DeepTypeResolver::resolve_type_var`, `resolve_dim_var`, and
/// `resolve_rank_var` are the three places that mint a fresh variable for the
/// name `_`, so these are exactly the spellings that can produce a hole.
///
/// The scan is syntactic on purpose: the schedule runs before any signature is
/// resolved, so it cannot ask the resolver. A bail on a nearly-exhausted stack
/// answers `true`, which can only ADD a precedence edge and never drop one;
/// the guard's record still turns the bail into a hard failure at the check
/// boundary, so the over-approximation is never observed on a passing program.
fn deep_type_contains_hole(expr: &deep::Expr) -> bool {
    stack_guard!("deep_type_contains_hole", expr, true);
    if let Some((tag, _, kids)) = stamped_parts(expr) {
        if matches!(tag, DeepTag::TVar | DeepTag::DVar | DeepTag::DRank)
            && kids.first().and_then(symbol_name) == Some("_")
        {
            return true;
        }
        return kids.iter().any(deep_type_contains_hole);
    }
    match expr {
        deep::Expr::BareList(elements, _) => elements.iter().any(deep_type_contains_hole),
        deep::Expr::MetaExpr(meta, _) => deep_type_contains_hole(&meta.expr),
        _ => false,
    }
}

/// The declared-signature facts the schedule needs about each name, from a
/// syntactic scan of the unit's `defsig` items (chelis#1486).
///
/// `signed` is every name that carries a `defsig` at all; `holed` is the
/// subset whose signature contains an inference hole. The two answer the two
/// halves of [04-INF-5] and [04-INF-6]: a `defsig`-less function has no header
/// for a reader to use at all, and a hole header is not honest until the body
/// has filled it, while a complete or authored-binder header is honest before
/// the body and needs no edge.
struct DeclaredSignatureScan {
    signed: UnordSet<String>,
    holed: UnordSet<String>,
}

fn scan_declared_signatures(items: &[(Option<String>, &deep::Expr)]) -> DeclaredSignatureScan {
    let mut signed = UnordSet::new();
    let mut holed = UnordSet::new();
    for (_, expr) in items {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        signed.insert(name.to_string());
        if defsig_parts(kids).is_some_and(|(_, _, type_expr)| deep_type_contains_hole(type_expr)) {
            holed.insert(name.to_string());
        }
    }
    DeclaredSignatureScan { signed, holed }
}

/// Primary body-inference schedule: the order in which top-level declaration
/// bodies are inferred. The returned values are original flattened ordinals,
/// so scheduling never changes diagnostic ownership, collected-type origins,
/// or output order.
///
/// The schedule is the [`hoist_order`]-least linear extension of a
/// precedence graph whose every edge is a real reference in the program:
///
/// - a module function is inferred after every module function it calls,
///   so a forward helper's body-derived scheme is available to its caller
///   (the planner's callee-first order);
/// - an item is inferred after every eager value it reads that is declared
///   before it ([04-INF-4] makes exactly those reads legal; an unannotated
///   value has no header anywhere, so its type exists only once its own
///   `def` has been inferred). For a module function this is the barrier
///   that keeps the hoist from carrying it across the value;
/// - an item at or after the earliest module-function ordinal is inferred
///   after every `defsig`-LESS module function it reads (the mirror edge). A
///   declared signature is in the global header environment from the first
///   pass, and under [04-INF-6] an authored binder is rigid, so a complete or
///   authored-binder header IS the function's scheme before its body runs and
///   a reader needs no edge. A `defsig`-less function has no header at all,
///   which is the availability role this edge keeps; it mirrors what the hoist
///   supplied implicitly, bounded to the same region, so function visibility
///   ([04-INF-2]/[04-INF-3]) is neither narrowed nor widened;
/// - an item is inferred after every declaration it references whose `defsig`
///   contains a wildcard slot (the hole edge, chelis#1486 / [04-INF-5]). A
///   hole is not a binder and is not quantified as one: the slot's type is
///   whatever the body determines, so a reader that instantiates the header
///   early observes a variable the body has not filled and accepts programs
///   the checker rejects when the body is inferred first. Unlike the mirror
///   edge this one is bounded by no region: it holds below the hoist floor and
///   in a bare unit, because the dishonest header is global from the first
///   pass. Kahn's priority keeps the displacement minimal, so the function
///   holds its hoist position and only its readers slide after it;
/// - every strongly connected component of the full syntactic reference graph
///   is one vertex. This includes mixed function/value components as well as
///   ordinary recursive function groups, so an edge one member earns
///   constrains them all.
///
/// Bare functions keep textual availability (no call or mirror edge), and a
/// forward value reference produces no edge because the scope rule leaves it
/// unbound. Nothing else orders the graph: in particular there is no textual
/// chain over non-function items and no chain over the planner order. Such
/// chains are not dependencies, and they close cycles on legal programs.
/// The canonical reference collector follows lambda bodies and both read and
/// application edges. Contracting all of its SCCs therefore makes the
/// remaining precedence graph acyclic by construction; the old
/// hoist-order-least stall release is unnecessary and deliberately absent.
/// A cyclic component containing an eager value is still rejected as
/// `CycleDetected`, but its bodies co-infer under provisional bindings so the
/// checker reaches that one ingress-independent verdict without first leaking
/// an inference-order `UnboundVariable` (chelis#1485).
///
/// This is availability, not ordinary source visibility. Whether a name is in
/// scope is decided by `Env::top_level_value_visibility` from source position.
/// The sole extra capability is owned by the exact active cyclic component:
/// its provisional members see one another, and [04-INF-8] precedence targets
/// that the graph scheduled first remain visible, only while that rejected
/// component is co-inferred. The component scope restores the prior capability
/// on every exit. No schedule position can otherwise widen or narrow
/// [04-INF-4].
/// `crates/chelis-types/src/infer/tests/schedule_invariants.rs` asserts the
/// invariants above directly on the returned order.
#[cfg(test)]
pub(super) fn primary_inference_schedule(
    function_plan: &FunctionInferencePlan,
    items: &[(Option<String>, &deep::Expr)],
) -> Vec<usize> {
    let references = TopLevelReferenceGraph::build(items);
    primary_inference_schedule_with_reference_graph(function_plan, items, &references)
}

fn primary_inference_schedule_with_reference_graph(
    function_plan: &FunctionInferencePlan,
    items: &[(Option<String>, &deep::Expr)],
    references: &TopLevelReferenceGraph,
) -> Vec<usize> {
    let reference_components = references.inference_components();
    if !function_plan.complete || !reference_components.complete {
        return Vec::new();
    }
    let module_fn_indices = function_plan
        .ordered_members()
        .filter(|member| items[member.item_index].0.is_some())
        .map(|member| member.item_index)
        .collect::<BTreeSet<_>>();
    let mut key = vec![0usize; items.len()];
    for (position, index) in hoist_order(function_plan, items.len(), &module_fn_indices)
        .into_iter()
        .enumerate()
    {
        key[index] = position;
    }

    // Contract every SCC of the full reference graph to one vertex, named by
    // its lowest member ordinal and carrying its lowest hoist key. This
    // includes mixed function/value components, not only recursive functions.
    let mut vertex_of = (0..items.len()).collect::<Vec<_>>();
    let mut component_members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for component in &reference_components.components {
        let members = component.members.clone();
        let Some(&representative) = members.iter().min() else {
            continue;
        };
        for &member in &members {
            vertex_of[member] = representative;
        }
        component_members.insert(representative, members);
    }
    let vertex_key = |vertex: usize| -> usize {
        component_members
            .get(&vertex)
            .map_or(key[vertex], |members| {
                members
                    .iter()
                    .map(|member| key[*member])
                    .min()
                    .unwrap_or(key[vertex])
            })
    };

    // Reference edges, `(before, after)`, deduplicated and self-edge free.
    let eager_value_ordinals = eager_value_definition_ordinals(items);
    let mut module_fn_by_name: BTreeMap<String, usize> = BTreeMap::new();
    for &index in &module_fn_indices {
        if let Some(name) = top_level_decl_name(items[index].1) {
            module_fn_by_name.entry(name.to_string()).or_insert(index);
        }
    }
    // chelis#1486: the hole edge is not bounded by the planner's region, so it
    // needs every `def`'s ordinal, not only a module function's.
    let signatures = scan_declared_signatures(items);
    let mut hole_signature_definitions: BTreeMap<String, usize> = BTreeMap::new();
    for (index, (_, expr)) in items.iter().enumerate() {
        let Some((DeepTag::Def, _, _)) = stamped_parts(expr) else {
            continue;
        };
        if let Some(name) = top_level_decl_name(expr)
            && signatures.holed.contains(name)
        {
            hole_signature_definitions
                .entry(name.to_string())
                .or_insert(index);
        }
    }
    let floor = module_fn_indices.first().copied();
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (index, item_references) in references.item_references.iter().enumerate() {
        let reader = vertex_of[index];
        let reader_is_module_fn = module_fn_indices.contains(&index);
        for reference in item_references {
            let name = &references.definition(reference.target).name;
            if let Some(&value) = eager_value_ordinals.get(name)
                && value < index
            {
                edges.insert((vertex_of[value], reader));
            }
            if let Some(&function) = module_fn_by_name.get(name)
                && (reader_is_module_fn
                    || (floor.is_some_and(|floor| index >= floor)
                        && !signatures.signed.contains(name)))
            {
                edges.insert((vertex_of[function], reader));
            }
            // The hole edge, in every region and in a bare unit too: a
            // header with a wildcard slot is honest only after its body.
            if let Some(&declaration) = hole_signature_definitions.get(name) {
                edges.insert((vertex_of[declaration], reader));
            }
        }
    }

    // [04-INF-8] cycle precedence: an eager root that occurs in its own
    // full-reference component is rejected by CycleDetected alone. Infer any
    // later eager value referenced from that exact component first, so the
    // component's narrow visibility capability can resolve the real scheme
    // instead of leaking an earlier [04-INF-4] UnboundVariable. These edges
    // cannot cycle after SCC contraction: a target with a path back into the
    // component would already be one of its members.
    for component in &reference_components.components {
        if !component.cyclic {
            continue;
        }
        let Some(&representative) = component.members.iter().min() else {
            continue;
        };
        for target in references.cycle_precedence_targets(&component.members) {
            edges.insert((
                vertex_of[references.definition(target).item_index],
                vertex_of[representative],
            ));
        }
    }
    edges.retain(|(before, after)| before != after);

    // Kahn's algorithm, releasing the hoist-order-least ready component. Full
    // reference SCC contraction makes this graph a DAG, so there is no
    // user-program stall and no arbitrary release path.
    let mut successors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut remaining_predecessors: BTreeMap<usize, usize> = BTreeMap::new();
    for &(before, after) in &edges {
        successors.entry(before).or_default().push(after);
        *remaining_predecessors.entry(after).or_default() += 1;
    }
    let mut pending = (0..items.len())
        .filter(|index| vertex_of[*index] == *index)
        .map(|vertex| (vertex_key(vertex), vertex))
        .collect::<BTreeSet<_>>();
    let mut ready = pending
        .iter()
        .copied()
        .filter(|(_, vertex)| remaining_predecessors.get(vertex).copied().unwrap_or(0) == 0)
        .collect::<BTreeSet<_>>();
    let mut emitted = vec![false; items.len()];
    let mut schedule = Vec::with_capacity(items.len());
    while let Some(&(key, vertex)) = ready.first() {
        ready.remove(&(key, vertex));
        pending.remove(&(key, vertex));
        if emitted[vertex] {
            continue;
        }
        emitted[vertex] = true;
        match component_members.get(&vertex) {
            Some(members) => schedule.extend(members.iter().copied()),
            None => schedule.push(vertex),
        }
        for &successor in successors.get(&vertex).into_iter().flatten() {
            if emitted[successor] {
                continue;
            }
            let remaining = remaining_predecessors
                .get_mut(&successor)
                .expect("every successor was counted when its edge was added");
            *remaining -= 1;
            if *remaining == 0 {
                ready.insert((vertex_key(successor), successor));
            }
        }
    }
    debug_assert!(
        pending.is_empty(),
        "reference-component DAG must schedule every item"
    );
    schedule
}

#[derive(Debug)]
pub(super) struct PrimaryInferenceGroup {
    indices: Vec<usize>,
    cyclic: bool,
    recursive_function_indices: Vec<usize>,
}

fn primary_inference_groups_with_reference_graph(
    function_plan: &FunctionInferencePlan,
    items: &[(Option<String>, &deep::Expr)],
    references: &TopLevelReferenceGraph,
) -> Vec<PrimaryInferenceGroup> {
    let schedule =
        primary_inference_schedule_with_reference_graph(function_plan, items, references);
    primary_inference_groups_for_schedule(function_plan, references, schedule)
}

/// Group one explicit schedule: every full-reference component is emitted
/// whole in source order. Function recursive-instantiation membership stays a
/// separate projection and is never inferred from a mixed component's cycle.
pub(super) fn primary_inference_groups_for_schedule(
    function_plan: &FunctionInferencePlan,
    references: &TopLevelReferenceGraph,
    schedule: Vec<usize>,
) -> Vec<PrimaryInferenceGroup> {
    let reference_components = references.inference_components();
    if !reference_components.complete {
        return Vec::new();
    }
    let recursive_function_indices = function_plan
        .components
        .iter()
        .filter(|component| component.recursive)
        .flat_map(|component| component.members.iter().map(|member| member.item_index))
        .collect::<UnordSet<_>>();

    let mut emitted_components = UnordSet::new();
    let mut groups = Vec::new();
    for index in schedule {
        let Some(component_index) = reference_components.component_by_item[index] else {
            groups.push(PrimaryInferenceGroup {
                indices: vec![index],
                cyclic: false,
                recursive_function_indices: Vec::new(),
            });
            continue;
        };
        if emitted_components.insert(component_index) {
            let component = &reference_components.components[component_index];
            groups.push(PrimaryInferenceGroup {
                indices: component.members.clone(),
                cyclic: component.cyclic,
                recursive_function_indices: component
                    .members
                    .iter()
                    .copied()
                    .filter(|index| recursive_function_indices.contains(index))
                    .collect(),
            });
        }
    }
    groups
}

/// Install monomorphic types for the un-signed members of one cyclic
/// full-reference component. Functions receive arity-shaped function types;
/// eager values receive one fresh type variable. The component is removed and
/// generalized as a unit after every body has unified with its provisional.
pub(super) fn prebind_cyclic_component_schemes(
    indices: &[usize],
    items: &[(Option<String>, &deep::Expr)],
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    metadata_prebound_names: &UnordSet<String>,
    env: &mut Env,
    vg: &mut VarGen,
) -> UnordMap<usize, Type> {
    let mut provisional = UnordMap::new();
    for index in indices {
        let expr = items[*index].1;
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
            continue;
        };
        if declared_signatures.contains_key(name) || metadata_prebound_names.contains(name) {
            continue;
        }
        let ty = match tagged_children(body, DeepTag::Fn) {
            Some(fn_kids) => {
                let Some(params) = fn_kids.first() else {
                    continue;
                };
                let arity = match params {
                    deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => {
                        node.child_count()
                    }
                    deep::Expr::BareList(elements, _) => elements.len(),
                    _ => continue,
                };
                Type::Fn(
                    (0..arity).map(|_| vg.fresh_type()).collect(),
                    Box::new(vg.fresh_type()),
                )
            }
            None => vg.fresh_type(),
        };
        env.bind(name.to_string(), Scheme::mono(ty.clone()));
        provisional.insert(*index, ty);
    }
    provisional
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
) -> UnordMap<usize, deep::Expr> {
    let collected = collect_ir_types_with_origins(items.iter().map(|(_, expr)| *expr));
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
            body_is_type_stamped_literal_self_ref(body, name)
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
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> (UnordSet<String>, UnordMap<usize, ErrorWitness>) {
    let CollectedIrTypes {
        type_env,
        final_origin_by_name,
    } = collect_ir_types_with_origins(items.iter().map(|(_, expr)| *expr));
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
                let scheme = env.generalize(&ty, subst);
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
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
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
            let scheme = env.generalize(&ty, subst);
            subst.name_generic_parameters(&scheme, name, &UnordMap::new());
            env.bind(name.to_string(), scheme);
            None
        }
        Err(witness) => Some(witness),
    }
}

pub(super) fn build_ir_type_env(exprs: &[deep::Expr]) -> IrTypeEnv {
    collect_ir_types_with_origins(top_level_decl_items(exprs)).type_env
}

pub(super) fn collect_ir_types_with_origins<'a>(
    items: impl IntoIterator<Item = &'a deep::Expr>,
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
        if let Some(ty) = expr_type_expr(body, &type_env) {
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
    chelis_deep::pipe::fold_program_pipes(exprs)
}

#[cfg(test)]
pub(crate) fn builtin_selection_probe(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Vec<crate::builtin_discovery::BuiltinCaseSelection> {
    let normalized = normalize_program_input(exprs);
    infer_program_with_product_in_session(&normalized, errors).builtin_selections
}

#[cfg(test)]
mod component_level_scope_tests {
    use super::*;

    fn mutual_defs() -> Vec<deep::Expr> {
        chelis_deep::parser::parse_str(
            "(def {} left (fn {} (params {} x) (app {} (var {} right) (var {} x))))
             (def {} right (fn {} (params {} x) (app {} (var {} left) (var {} x))))",
        )
        .expect("mutual fixture parses")
    }

    #[test]
    fn component_scope_mints_provisionals_and_restores_visibility_on_completion() {
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
        let mut subst = Subst::new();

        env.note_top_level_value_ordinal("right".to_string(), 1, None);
        env.note_top_level_value_ordinal("outside".to_string(), 2, None);
        env.note_top_level_value_ordinal("precedence".to_string(), 3, None);
        env.set_current_declaration_ordinal(Some(0));
        let _empty_component =
            env.replace_active_top_level_component(["outside".to_string()].into_iter().collect());
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
        );
        for (_, ty) in provisional.to_sorted() {
            for var in crate::env::free_tvars(ty) {
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
        assert_eq!(super::super::recursion::group_state_counts(), (0, 0));
    }

    #[test]
    fn component_scope_abort_restores_bindings_levels_pins_and_visibility() {
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
        let mut subst = Subst::new();
        env.note_top_level_value_ordinal("right".to_string(), 1, None);
        env.note_top_level_value_ordinal("outside".to_string(), 2, None);
        env.note_top_level_value_ordinal("precedence".to_string(), 3, None);
        env.set_current_declaration_ordinal(Some(0));
        let _empty_component =
            env.replace_active_top_level_component(["outside".to_string()].into_iter().collect());
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
        );
        super::super::recursion::begin_group(
            ["left", "right"].into_iter().map(|name| (name, false)),
            &env,
        );
        let pinned = var_gen.fresh_tvar();
        let _caller = super::super::recursion::begin_caller("left", None, &[]);
        super::super::recursion::record_occurrence(
            "right",
            &[(pinned, Type::Var(pinned))],
            None,
            None,
        );
        assert_eq!(super::super::recursion::group_state_counts(), (2, 1));

        scope.abort(&mut env, &var_gen, &mut subst);
        assert_eq!(subst.current_level(), 0);
        assert_eq!(super::super::recursion::group_state_counts(), (0, 0));
        assert!(!super::super::recursion::tvar_pinned(pinned));
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

        let follow_up = infer_program(
            &chelis_deep::parser::parse_str("(def {} clean (fn {} (params {} x) (var {} x)))")
                .expect("follow-up parses"),
        );
        assert!(follow_up.errors.is_empty());
    }

    #[test]
    fn primary_driver_mid_scc_cancellation_aborts_before_return() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} left (t-fn {} (t-prim {} i32) (t-prim {} i32)))
             (defsig {} right (t-fn {} (t-prim {} i32) (t-prim {} i32)))
             (def {} left (fn {} (params {} x) (app {} (var {} right) (var {} x))))
             (def {} right (fn {} (params {} x) (app {} (var {} left) (var {} x))))",
        )
        .expect("authored mutual-recursion fixture parses");

        let observation = {
            let token = crate::cancel::CancelToken::new();
            let _cancel_guard = crate::cancel::install_cancel_token(token);
            let _hook_guard = cancel_primary_recursive_after_members(1);
            let result = infer_program(&exprs);
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

        let follow_up = infer_program(
            &chelis_deep::parser::parse_str("(def {} clean (fn {} (params {} x) (var {} x)))")
                .expect("follow-up parses"),
        );
        assert!(
            follow_up.errors.is_empty(),
            "a cancelled recursive check must not contaminate its successor: {:?}",
            follow_up.errors
        );
    }
}
