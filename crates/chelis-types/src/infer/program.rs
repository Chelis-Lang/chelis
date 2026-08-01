//! Public check entry points and inference schedules.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

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
    let normalized = normalize_nodes_to_lists(exprs);
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
    collect_all_declarations(&items, &mut env, &mut vg, &mut subst, &mut adt_reg, errors);
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
    let metadata_prebound_names = HashSet::new();
    let inference_groups = primary_inference_groups(exprs, &items);
    for group in inference_groups {
        let provisional_types = if group.recursive {
            prebind_recursive_function_schemes(
                &group.indices,
                &items,
                &declared_signatures,
                &metadata_prebound_names,
                &mut env,
                &mut vg,
            )
        } else {
            HashMap::new()
        };
        let mut deferred_bindings = Vec::new();
        for declaration_index in group.indices {
            let (module, expr) = &items[declaration_index];
            product.begin_root(expr);
            let decl_name = top_level_decl_name(expr);
            crate::opacity::set_current_item(
                crate::opacity::module_key_for_item(module.as_deref(), decl_name),
                decl_name.map(str::to_string),
            );
            if let Some(binding) = infer_top_level(
                expr,
                &mut env,
                &mut vg,
                &mut subst,
                &adt_reg,
                errors,
                &mut product,
                None,
                provisional_types.get(&declaration_index),
                group.recursive,
                &user_def_names,
                &declared_signatures,
            ) {
                deferred_bindings.push(binding);
            }
            product.finish_root(&subst, errors);
            // Issue #256 round 2: re-check each deferred borrow against the
            // now-complete substitution (see `validate_deferred_borrow_vars`).
            validate_deferred_borrow_vars(&subst, &adt_reg, errors);
            // D-CHECK: drain the per-def deferred-access ledger (see
            // `validate_deferred_opaque_uses`).
            validate_deferred_opaque_uses(&subst, &adt_reg, errors);
        }
        if group.recursive {
            for (name, _) in &deferred_bindings {
                env.remove_binding(name);
            }
            let schemes = deferred_bindings
                .into_iter()
                .map(|(name, ty)| {
                    let scheme = env.generalize(&ty, &subst);
                    (name, scheme)
                })
                .collect::<Vec<_>>();
            for (name, scheme) in schemes {
                env.bind(name, scheme);
            }
        }
    }
    crate::opacity::set_current_item(None, None);

    // Third pass: reject tensor types whose element precision isn't supported
    // by the Phase 0f backend (f16/bf16/f8e4m3). These would silently get
    // downcast to f32 by the current build targets, violating the "no implicit
    // precision promotion" rule. f64 is supported as of v0.2.3.
    validate_tensor_precisions_in_program(exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(exprs, errors);

    // WS-A8 cross-row enforcement: reject `matmul`/transcendental ops that
    // are reached through a polymorphic-precision sig instantiated at a
    // dtype the spec rules forbid (§5.7.2 / §5.4).
    let local_ir_env = build_ir_type_env(exprs);
    validate_polymorphic_op_constraints(exprs, &local_ir_env, errors);

    // If any walker bailed on a nearly-exhausted stack during this run,
    // surface it as a hard located failure (covered-or-rejected).
    stack_scope.drain_into(errors);

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
    // Normalize Node/BareList → List (#908 producer switch).
    let normalized = normalize_nodes_to_lists(library_exprs);
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

    let product = infer_ir_program_with_state(
        library_exprs,
        &mut state,
        /* combined_ir_for_validate = */ &library_ir,
        /* run_validate_passes_on = */ None,
        errors,
    );
    let stats = product.stats();
    log_sub("infer_ir_program_with_state", &mut sub_t);
    validate_ir_program(library_exprs, &library_ir, errors);
    log_sub("validate_ir_program", &mut sub_t);
    validate_tensor_precisions_in_program(library_exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    log_sub("validate_tensor_precisions", &mut sub_t);
    validate_polymorphic_op_constraints(library_exprs, &library_ir, errors);
    log_sub("validate_polymorphic_op_constraints", &mut sub_t);
    // Surface a stack-exhaustion bail from the passes above as a hard
    // located error before the gate (and before the errors drain below).
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Capture library def names — needed by new-code cycle / unbound
    // suppression to distinguish library refs from new-code refs.
    let mut library_def_names = std::collections::HashSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Def)
            && let Some(name) = children(list).first().and_then(symbol_name)
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
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| annotate_expr_with_scope(e, &product, annotation_context, errors))
        .collect();
    log_sub("annotate_library_exprs_outer_loop", &mut sub_t);
    let library_ir_annotated = build_ir_type_env(&library_annotated);
    log_sub("build_ir_type_env_from_annotated", &mut sub_t);

    // The annotation loop above recurses (annotate_expr_with_scope); if it
    // bailed on low stack, reject rather than return a partially-annotated
    // library context.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    Ok(TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated,
        library_def_names,
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
///   `HashMap` iteration) to `build_type_env_from_library(library_exprs)`.
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
/// 3. Run all validators (`validate_ir_program`,
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
    let stack_scope = StackExhaustionScope::enter();
    // Mirror `build_type_env_from_library` up through the validators so the
    // type_env half stays bit-compatible with the existing public API.
    let empty = TypeEnv::empty();
    let mut state = empty.inner().clone();

    let library_ir = build_ir_type_env(library_exprs);

    let product = infer_ir_program_with_state(
        library_exprs,
        &mut state,
        /* combined_ir_for_validate = */ &library_ir,
        /* run_validate_passes_on = */ None,
        errors,
    );
    let stats = product.stats();
    validate_ir_program(library_exprs, &library_ir, errors);
    validate_tensor_precisions_in_program(library_exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    validate_polymorphic_op_constraints(library_exprs, &library_ir, errors);
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }

    // Capture library def names before consuming `state` into `TypeEnv`.
    let mut library_def_names = std::collections::HashSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Def)
            && let Some(name) = children(list).first().and_then(symbol_name)
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
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| annotate_expr_with_scope(e, &product, annotation_context, errors))
        .collect();
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    let library_ir_annotated = build_ir_type_env(&library_annotated);

    let type_env = TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated.clone(),
        library_def_names,
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
        &product.type_headers,
        &product.adt_registry,
        stats,
        errors,
    );
    if !errors.is_empty() {
        return Err(stats);
    }

    Ok((type_env, checked))
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
    let stack_scope = StackExhaustionScope::enter();
    // Seed from the base context's snapshot rather than the empty state.
    let mut state = base.inner().clone();

    // `library_exprs` declared types (IR), layered on top of the base's.
    let new_ir = build_ir_type_env(library_exprs);
    let combined_ir: HashMap<String, deep::Expr> = state
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
    let product = infer_ir_program_with_state(
        library_exprs,
        &mut state,
        &combined_ir,
        /* run_validate_passes_on = */ None,
        errors,
    );
    let stats = product.stats();
    validate_ir_program(library_exprs, &combined_ir, errors);
    validate_tensor_precisions_in_program(library_exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(library_exprs, errors);
    validate_polymorphic_op_constraints(library_exprs, &combined_ir, errors);
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
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Def)
            && let Some(name) = children(list).first().and_then(symbol_name)
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
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| annotate_expr_with_scope(e, &product, annotation_context, errors))
        .collect();
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(errors);
    if !errors.is_empty() {
        return Err(stats);
    }
    let new_ir_annotated = build_ir_type_env(&library_annotated);

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
        &product.type_headers,
        &product.adt_registry,
        stats,
        errors,
    );
    if !errors.is_empty() {
        return Err(stats);
    }

    Ok((type_env, checked))
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
    crate::session::check_ir_with_signature_context(context, signature_context, new_exprs)
}

pub(crate) fn check_ir_with_signature_context_in_session(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) -> Result<CheckedProgram, InferStats> {
    // Normalize Node/BareList → List (#908 producer switch).
    let normalized = normalize_nodes_to_lists(new_exprs);
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
    let mut state = context.inner().clone();

    // New-code declared types (IR) layered on top of library's.
    let new_ir = build_ir_type_env(new_exprs);
    let combined_ir: HashMap<String, deep::Expr> = state
        .ir_types
        .iter()
        .chain(new_ir.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    log_sub("build_ir_and_combine", &mut sub_t);

    // Library is already validated; only run validate / inference on
    // new exprs. Inference's canonical collector binds the new-code's
    // own declared types (library schemes are already in state.env).
    let product = infer_ir_program_with_state(
        new_exprs,
        &mut state,
        &combined_ir,
        /* run_validate_passes_on = */ None,
        errors,
    );
    let stats = product.stats();
    log_sub("infer_ir_program_with_state", &mut sub_t);
    // Run cycle / shape / precision validators on new_exprs only. The
    // combined IR env is supplied so `(var libfoo)` references
    // resolve to the library's declared type during shape validation.
    validate_ir_program(new_exprs, &combined_ir, errors);
    log_sub("validate_ir_program", &mut sub_t);
    validate_tensor_precisions_in_program(new_exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(new_exprs, errors);
    log_sub("validate_tensor_precisions", &mut sub_t);
    validate_polymorphic_op_constraints(new_exprs, &combined_ir, errors);
    log_sub("validate_polymorphic_op_constraints", &mut sub_t);
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
        &product.type_headers,
        &product.adt_registry,
        stats,
        errors,
    );
    if !errors.is_empty() {
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
    // Normalize Node/BareList → List at the entry boundary so the
    // pointer-based owner-stamp system sees the same addresses throughout
    // both registration, inference, and annotation passes (#908).
    let normalized = normalize_nodes_to_lists(exprs);
    let exprs = &normalized;
    // Outermost scope covers both inference (which has its own inner scope)
    // and the annotation pass below, so a bail in either surfaces as a hard
    // located failure rather than a partially-annotated `Ok`.
    let stack_scope = StackExhaustionScope::enter();
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
        let checked = finalize_checked_program(
            annotated_exprs,
            annotated_type_env,
            &SignatureInferenceMetadata::default(),
            &product.type_headers,
            &product.adt_registry,
            stats,
            errors,
        );
        if !errors.is_empty() {
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
    // Normalize Node/BareList → List (#908 producer switch).
    let normalized = normalize_nodes_to_lists(exprs);
    let exprs = &normalized;
    let stack_scope = StackExhaustionScope::enter();
    let type_env = build_ir_type_env(exprs);
    let stats = infer_ir_program_with_env(exprs, &type_env, errors);
    validate_ir_program(exprs, &type_env, errors);
    validate_tensor_precisions_in_program(exprs, errors);
    crate::invariants::validate_type_invariants_in_program_with_sink(exprs, errors);
    validate_polymorphic_op_constraints(exprs, &type_env, errors);
    // Surface any walker stack bail as a hard located error.
    stack_scope.drain_into(errors);
    stats
}

pub(super) fn infer_ir_program_with_env(
    exprs: &[deep::Expr],
    type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) -> InferStats {
    // Backwards-compat wrapper. Callers (like `infer_ir_program` and
    // `check_typed_program` callers) run `validate_ir_program`
    // separately, so we pass `None` here to skip the embedded validate.
    let empty_inner = crate::context::TypeEnv::empty();
    let mut state = empty_inner.inner().clone();
    infer_ir_program_with_state(
        exprs, &mut state, type_env, /* run_validate_passes_on = */ None, errors,
    )
    .stats()
}

/// Run the inference / IR binding / shape-validation passes against
/// `state`, mutating it as it goes. Library state should be supplied by
/// pre-cloning a snapshot; pass `&[]`-derived state for the monolithic
/// path. Source IR types are collected with their exact final declaration
/// origins and bound into `state.env` here; the
/// `combined_ir` is what `validate_ir_program` consults so
/// new-code shape validation can look up declared types of library
/// references.
pub(super) fn infer_ir_program_with_state(
    exprs: &[deep::Expr],
    state: &mut TypeEnvInner,
    combined_ir: &IrTypeEnv,
    run_validate_passes_on: Option<&[deep::Expr]>,
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
    collect_all_declarations(
        &items,
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

    // The canonical collector returns each final name-keyed IR type together
    // with the exact flattened declaration ordinal that produced it. Duplicate
    // names intentionally retain last-declaration-wins semantics, while the
    // origin keeps an owning witness from leaking into an earlier body.
    let collected_ir_types = collect_ir_types_with_origins(items.iter().map(|(_, expr)| *expr));

    let mut prebound_type_failures = HashMap::new();
    for (name, ty_expr) in &collected_ir_types.type_env {
        match resolve_deep_type(
            ty_expr,
            &mut state.var_gen,
            &state.adt_reg,
            TypeUseSite::CompilerMetadata,
            BinderMode::TrustedCompilerMetadata,
            errors,
        ) {
            Ok(ty) => {
                let scheme = state.env.generalize(&ty, &state.subst);
                state.env.bind(name.clone(), scheme);
            }
            Err(witness) => {
                // The prebinding pass owns this diagnostic. Carry its witness
                // into the matching def-body pass so a literal/ascription
                // consumer propagates the same failure instead of resolving
                // the cloned metadata and reporting it a second time.
                let declaration_index = collected_ir_types.final_origin_by_name[name];
                prebound_type_failures.insert(declaration_index, witness);
            }
        }
    }

    // Per-decl profile: when CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1, emit
    // one stderr line per top-level decl with its name and inference time.
    // Aggregated by name in caller scripts to attribute cost per module.
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));
    let declared_signatures = collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr));
    let metadata_prebound_names = collected_ir_types
        .type_env
        .keys()
        .cloned()
        .collect::<HashSet<_>>();
    let inference_groups = primary_inference_groups(exprs, &items);
    for group in inference_groups {
        let provisional_types = if group.recursive {
            prebind_recursive_function_schemes(
                &group.indices,
                &items,
                &declared_signatures,
                &metadata_prebound_names,
                &mut state.env,
                &mut state.var_gen,
            )
        } else {
            HashMap::new()
        };
        let mut deferred_bindings = Vec::new();
        for declaration_index in group.indices {
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
            if let Some(binding) = infer_top_level(
                expr,
                &mut state.env,
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                errors,
                &mut product,
                prebound_type_failures.get(&declaration_index),
                provisional_types.get(&declaration_index),
                group.recursive,
                &user_def_names,
                &declared_signatures,
            ) {
                deferred_bindings.push(binding);
            }
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
            validate_deferred_borrow_vars(&state.subst, &state.adt_reg, errors);
            // D-CHECK: drain the per-def deferred-access ledger (see
            // `validate_deferred_opaque_uses`).
            validate_deferred_opaque_uses(&state.subst, &state.adt_reg, errors);
        }
        if group.recursive {
            for (name, _) in &deferred_bindings {
                state.env.remove_binding(name);
            }
            let schemes = deferred_bindings
                .into_iter()
                .map(|(name, ty)| {
                    let scheme = state.env.generalize(&ty, &state.subst);
                    (name, scheme)
                })
                .collect::<Vec<_>>();
            for (name, scheme) in schemes {
                state.env.bind(name, scheme);
            }
        }
    }
    crate::opacity::set_current_item(None, None);

    for warning in chelis_deep::validate::validate(exprs) {
        errors.push(
            CheckError::new(
                match warning.kind {
                    chelis_deep::validate::WarningKind::Arity => CheckErrorKind::ArityMismatch,
                    _ => CheckErrorKind::Other,
                },
                warning.message,
                vec!["Use canonical Deep 3-tuple forms from spec/03".to_string()],
            )
            .at_offset(warning.offset),
        );
    }

    if let Some(target_exprs) = run_validate_passes_on {
        validate_ir_program(target_exprs, combined_ir, errors);
    }

    product
}

/// Primary body-inference schedule. Function declarations inside a lexical
/// module use the same dependency/SCC planner as signature inference, so a
/// forward helper's body-derived scheme is available to its caller. Bare defs
/// and every non-function declaration retain textual order. The returned
/// values are original flattened ordinals: scheduling never changes diagnostic
/// ownership, collected-type origins, or output order.
pub(super) fn primary_inference_schedule(
    exprs: &[deep::Expr],
    items: &[(Option<String>, &deep::Expr)],
) -> Vec<usize> {
    let module_fn_by_key = items
        .iter()
        .enumerate()
        .filter_map(|(index, (module, expr))| {
            module.as_ref()?;
            let deep::Expr::List(list, _) = expr else {
                return None;
            };
            if get_tag(list) != Some(DeepTag::Def)
                || children(list)
                    .get(1)
                    .and_then(as_tagged_list_expr(DeepTag::Fn))
                    .is_none()
            {
                return None;
            }
            Some((expr_key(expr), index))
        })
        .collect::<HashMap<_, _>>();
    if module_fn_by_key.is_empty() {
        return (0..items.len()).collect();
    }

    let ordered_module_fns = signature_inference_def_order(exprs)
        .into_iter()
        .filter_map(|expr| module_fn_by_key.get(&expr_key(expr)).copied())
        .collect::<Vec<_>>();
    let module_fn_indices = module_fn_by_key.values().copied().collect::<HashSet<_>>();
    let insertion = module_fn_indices.iter().copied().min().unwrap_or(0);
    let mut schedule = Vec::with_capacity(items.len());
    for index in 0..items.len() {
        if index == insertion {
            schedule.extend(ordered_module_fns.iter().copied());
        }
        if !module_fn_indices.contains(&index) {
            schedule.push(index);
        }
    }
    schedule
}

#[derive(Debug)]
pub(super) struct PrimaryInferenceGroup {
    indices: Vec<usize>,
    recursive: bool,
}

/// Group the flat primary schedule into recursive SCC inference units.
/// Acyclic bare functions stay in textual order; acyclic module functions
/// retain dependency order. Only a genuine recursive component is grouped
/// and prebound, so a bare acyclic forward helper remains unavailable.
pub(super) fn primary_inference_groups(
    exprs: &[deep::Expr],
    items: &[(Option<String>, &deep::Expr)],
) -> Vec<PrimaryInferenceGroup> {
    let schedule = primary_inference_schedule(exprs, items);
    let item_by_key = items
        .iter()
        .enumerate()
        .map(|(index, (_, expr))| (expr_key(expr), index))
        .collect::<HashMap<_, _>>();
    let recursive_components = function_inference_sccs(exprs)
        .into_iter()
        .filter(|component| component.recursive)
        .map(|component| {
            component
                .members
                .into_iter()
                .filter_map(|expr| item_by_key.get(&expr_key(expr)).copied())
                .collect::<Vec<_>>()
        })
        .filter(|indices| !indices.is_empty())
        .collect::<Vec<_>>();
    let mut component_by_index = HashMap::new();
    for (component_index, indices) in recursive_components.iter().enumerate() {
        for index in indices {
            component_by_index.insert(*index, component_index);
        }
    }

    let mut emitted_components = HashSet::new();
    let mut groups = Vec::new();
    for index in schedule {
        let Some(component_index) = component_by_index.get(&index).copied() else {
            groups.push(PrimaryInferenceGroup {
                indices: vec![index],
                recursive: false,
            });
            continue;
        };
        if emitted_components.insert(component_index) {
            groups.push(PrimaryInferenceGroup {
                indices: recursive_components[component_index].clone(),
                recursive: true,
            });
        }
    }
    groups
}

/// Install monomorphic arity-shaped types for the un-signed members of one
/// recursive SCC. The component is removed and generalized as a unit after
/// every body has unified with its provisional type.
pub(super) fn prebind_recursive_function_schemes(
    indices: &[usize],
    items: &[(Option<String>, &deep::Expr)],
    declared_signatures: &HashMap<String, DeclaredSigMetadata>,
    metadata_prebound_names: &HashSet<String>,
    env: &mut Env,
    vg: &mut VarGen,
) -> HashMap<usize, Type> {
    let mut provisional = HashMap::new();
    for index in indices {
        let expr = items[*index].1;
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        let kids = children(list);
        let (Some(name), Some(fn_list)) = (
            kids.first().and_then(symbol_name),
            kids.get(1).and_then(as_tagged_list_expr(DeepTag::Fn)),
        ) else {
            continue;
        };
        if declared_signatures.contains_key(name) || metadata_prebound_names.contains(name) {
            continue;
        }
        let Some(params) = children(fn_list).first() else {
            continue;
        };
        let arity = match params {
            deep::Expr::List(params, _) if get_tag(params) == Some(DeepTag::Params) => {
                children(params).len()
            }
            _ => continue,
        };
        let ty = Type::Fn(
            (0..arity).map(|_| vg.fresh_type()).collect(),
            Box::new(vg.fresh_type()),
        );
        env.bind(name.to_string(), Scheme::mono(ty.clone()));
        provisional.insert(*index, ty);
    }
    provisional
}

pub(super) type IrTypeEnv = HashMap<String, deep::Expr>;

/// Canonical IR type collection result. The type environment retains the
/// historical last-declaration-wins behavior for duplicate names; the paired
/// ordinal identifies the exact flattened declaration that produced each
/// final entry.
pub(super) struct CollectedIrTypes {
    type_env: IrTypeEnv,
    final_origin_by_name: HashMap<String, usize>,
}

pub(super) fn build_ir_type_env(exprs: &[deep::Expr]) -> IrTypeEnv {
    collect_ir_types_with_origins(top_level_decl_items(exprs)).type_env
}

pub(super) fn collect_ir_types_with_origins<'a>(
    items: impl IntoIterator<Item = &'a deep::Expr>,
) -> CollectedIrTypes {
    let mut type_env = HashMap::new();
    let mut final_origin_by_name = HashMap::new();
    for (declaration_index, expr) in items.into_iter().enumerate() {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let kids = children(list);
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

/// Iteratively convert `Expr::Node` → `Expr::List` and `Expr::BareList` →
/// `Expr::List` so the pointer-keyed type-stamp system and existing
/// List-based inference dispatch work on a single representation. The
/// explicit heap worklist is required because this boundary runs before the
/// guarded inference walkers: native recursion here would abort on the same
/// deep input those walkers must reject with a typed diagnostic. This is the
/// transitional normalization boundary for the #908 producer switch; once
/// all inference functions are migrated to accept Node directly, this becomes
/// dead code.
fn normalize_nodes_to_lists(exprs: &[deep::Expr]) -> Vec<deep::Expr> {
    exprs.iter().map(normalize_node_to_list).collect()
}

fn normalize_node_to_list(expr: &deep::Expr) -> deep::Expr {
    enum Action<'a> {
        Visit(&'a deep::Expr),
        FinishList {
            element_count: usize,
            span: Span,
        },
        FinishMap {
            map: &'a deep::MetaMap,
            span: Span,
        },
        FinishMetaExpr {
            meta: &'a deep::MetaExpr,
            span: Span,
        },
        FinishNode {
            tag: DeepTag,
            meta: &'a deep::MetaMap,
            child_count: usize,
            span: Span,
        },
        FinishUnknownForm(&'a deep::UnknownFormData),
    }

    fn split_tail(values: &mut Vec<deep::Expr>, count: usize) -> Vec<deep::Expr> {
        let start = values
            .len()
            .checked_sub(count)
            .expect("normalization action/value stacks remain balanced");
        values.split_off(start)
    }

    let mut actions = vec![Action::Visit(expr)];
    let mut values = Vec::new();

    while let Some(action) = actions.pop() {
        match action {
            Action::Visit(expr) => match expr {
                deep::Expr::Atom(atom, span) => {
                    values.push(deep::Expr::Atom(atom.clone(), *span));
                }
                deep::Expr::List(list, span) => {
                    actions.push(Action::FinishList {
                        element_count: list.elements.len(),
                        span: *span,
                    });
                    actions.extend(list.elements.iter().rev().map(Action::Visit));
                }
                deep::Expr::Map(map, span) => {
                    actions.push(Action::FinishMap { map, span: *span });
                    actions.extend(
                        map.entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
                deep::Expr::MetaExpr(meta, span) => {
                    actions.push(Action::FinishMetaExpr { meta, span: *span });
                    actions.push(Action::Visit(&meta.expr));
                    actions.extend(
                        meta.entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
                deep::Expr::Node(node, span) => {
                    actions.push(Action::FinishNode {
                        tag: node.tag(),
                        meta: node.meta(),
                        child_count: node.child_count(),
                        span: *span,
                    });
                    actions.extend(node.children_slice().iter().rev().map(Action::Visit));
                    actions.extend(
                        node.meta()
                            .entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
                deep::Expr::BareList(elements, span) => {
                    actions.push(Action::FinishList {
                        element_count: elements.len(),
                        span: *span,
                    });
                    actions.extend(elements.iter().rev().map(Action::Visit));
                }
                deep::Expr::UnknownForm(data) => {
                    actions.push(Action::FinishUnknownForm(data));
                    actions.extend(data.children.iter().rev().map(Action::Visit));
                    actions.extend(
                        data.meta
                            .entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
            },
            Action::FinishList {
                element_count,
                span,
            } => {
                let elements = split_tail(&mut values, element_count);
                values.push(deep::Expr::List(deep::List { elements }, span));
            }
            Action::FinishMap { map, span } => {
                let normalized_values = split_tail(&mut values, map.entries.len());
                let entries = map
                    .entries
                    .iter()
                    .zip(normalized_values)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                values.push(deep::Expr::Map(deep::MetaMap { entries }, span));
            }
            Action::FinishMetaExpr { meta, span } => {
                let mut normalized = split_tail(&mut values, meta.entries.len() + 1);
                let normalized_expr = normalized
                    .pop()
                    .expect("MetaExpr normalization visits its expression");
                let entries = meta
                    .entries
                    .iter()
                    .zip(normalized)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                values.push(deep::Expr::MetaExpr(
                    deep::MetaExpr {
                        entries,
                        expr: Box::new(normalized_expr),
                    },
                    span,
                ));
            }
            Action::FinishNode {
                tag,
                meta,
                child_count,
                span,
            } => {
                let mut normalized = split_tail(&mut values, meta.entries.len() + child_count);
                let children = normalized.split_off(meta.entries.len());
                let entries = meta
                    .entries
                    .iter()
                    .zip(normalized)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                let mut elements = Vec::with_capacity(child_count + 2);
                elements.push(deep::Expr::Atom(deep::Atom::Tag(tag), span));
                elements.push(deep::Expr::Map(deep::MetaMap { entries }, span));
                elements.extend(children);
                values.push(deep::Expr::List(deep::List { elements }, span));
            }
            Action::FinishUnknownForm(data) => {
                let mut normalized =
                    split_tail(&mut values, data.meta.entries.len() + data.children.len());
                let children = normalized.split_off(data.meta.entries.len());
                let entries = data
                    .meta
                    .entries
                    .iter()
                    .zip(normalized)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                values.push(deep::Expr::UnknownForm(Box::new(deep::UnknownFormData {
                    head: data.head.clone(),
                    meta: deep::MetaMap { entries },
                    children,
                    span: data.span,
                })));
            }
        }
    }

    assert_eq!(
        values.len(),
        1,
        "one normalization root produces exactly one expression"
    );
    values.pop().expect("normalization produced its root")
}
