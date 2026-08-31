//! Match and pattern inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_match(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return malformed_form(
            list,
            "match",
            "a scrutinee expression and at least one arm",
            errors,
        );
    }

    let scrutinee_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);

    // chelis#710 [04-TOT-3]: a match with a scrutinee but no arms is
    // malformed. Return early so the exhaustiveness check below cannot also
    // fire (which would double-report on an ADT scrutinee); a bare bool/int
    // scrutinee with zero arms used to reach `result_ty.unwrap_or(Type::Error)`
    // as a silent `Type::Error` (census-verified silent-through).
    if kids.len() < 2 {
        return malformed_form(
            list,
            "match",
            "at least one arm after the scrutinee",
            errors,
        );
    }

    let mut result_ty: Option<Type> = None;
    let mut covered_variants: Vec<String> = Vec::new();
    let mut has_wildcard = false;

    for arm_expr in &kids[1..] {
        if let Some((DeepTag::Arm, _, arm_kids)) = stamped_parts(arm_expr) {
            // arm_kids[0] = pattern, arm_kids[1] = guard (usually ()), arm_kids[2] = body
            if arm_kids.len() >= 3 {
                let mut arm_env = env.clone();
                let pat = &arm_kids[0];
                // RFC D-CHECK exhaustiveness fix (RT-0 verified false
                // positives): a TOP-LEVEL irrefutable arm covers the
                // match -- a bare `pat-var`, or a `pat-as` whose
                // inner pattern is irrefutable. Nested `pat-var`
                // keeps not-covering so exhaustiveness is not
                // weakened on ordinary ADTs.
                if top_level_arm_is_irrefutable(pat) {
                    has_wildcard = true;
                }
                pattern_bindings(
                    pat,
                    &scrutinee_ty,
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    &mut covered_variants,
                    &mut has_wildcard,
                );

                let guard = &arm_kids[1];
                let empty_guard = matches!(guard, deep::Expr::List(guard_list, _) if guard_list.elements.is_empty())
                    || matches!(guard, deep::Expr::BareList(elements, _) if elements.is_empty());
                if !empty_guard {
                    let guard_ty =
                        infer_expr(guard, &mut arm_env, vg, subst, adt_reg, errors, product);
                    let resolved_guard = subst.apply(&guard_ty);
                    if !matches!(resolved_guard, Type::Prim(Prim::Bool) | Type::Error(_)) {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!("match arm guard must be bool, got {resolved_guard}"),
                            vec![],
                        ));
                    }
                }

                let body_ty = infer_expr(
                    &arm_kids[2],
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                );

                match &result_ty {
                    None => result_ty = Some(body_ty),
                    Some(prev) => {
                        if let Err(te) = unify(prev, &body_ty, subst) {
                            errors.push(te.into());
                        }
                        result_ty = Some(subst.apply(prev));
                    }
                }
            }
        }
    }

    // Exhaustiveness check (wildcard covers everything)
    if !has_wildcard {
        let resolved_scrutinee = subst.apply(&scrutinee_ty);
        if let Type::Adt(ref adt_name, _) = resolved_scrutinee
            && let Some(all_variants) = adt_reg.variant_names(adt_name)
        {
            let missing: Vec<&String> = all_variants
                .iter()
                .filter(|v| !covered_variants.contains(v))
                .collect();
            if !missing.is_empty() {
                let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                errors.push(CheckError::new(
                    CheckErrorKind::NonExhaustiveMatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("non-exhaustive match: missing variants {:?}", names),
                    ),
                    vec![],
                ));
            }
        }
    }

    // A match with no arms is screened by the `(match {})` malformed guard
    // upstream, so `result_ty` is `Some` for any well-formed match; the
    // fallback is a fresh var (never a silent `Type::Error`, chelis#731 §C3).
    result_ty.unwrap_or_else(|| vg.fresh_type())
}

/// True for arm patterns that match every value of the scrutinee:
/// `pat-var`, `pat-wild`, and `pat-as` wrapping an irrefutable inner
/// pattern (`q @ x`). Applies at the ARM level only.
pub(super) fn top_level_arm_is_irrefutable(pat: &deep::Expr) -> bool {
    let Some((tag, _, kids)) = stamped_parts(pat) else {
        return false;
    };
    match tag {
        DeepTag::PatVar | DeepTag::PatWild => true,
        DeepTag::PatAs => kids.get(1).is_some_and(top_level_arm_is_irrefutable),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pattern_bindings(
    pat: &deep::Expr,
    scrutinee_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    covered_variants: &mut Vec<String>,
    has_wildcard: &mut bool,
) {
    stack_guard!("pattern_bindings", pat);
    if let Some((tag, _, kids)) = stamped_parts(pat) {
        match tag {
            DeepTag::PatVar => {
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    product.record_bypass(pat, resolved.clone(), "pattern binding traversal");
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
            }
            DeepTag::PatWild => {
                // Wildcard covers everything
                *has_wildcard = true;
            }
            DeepTag::PatLit => {
                // No bindings, but value should match scrutinee type
            }
            DeepTag::PatCtor => {
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    // chelis#317: an out-of-scope constructor pattern (a type-
                    // only import that names `| Alpha =>` without importing
                    // `Alpha`) must be rejected at `check` here, the same way
                    // the construction site is in `infer_var`. Without this the
                    // fuzzy terminal fallback below binds the arm to a foreign
                    // module's tag and the mismatch surfaces only as a runtime
                    // non-exhaustive match. `constructor_pattern_out_of_scope`
                    // rejects both the unique-fuzzy case and the non-unique /
                    // unresolvable case; the latter would otherwise push a bare
                    // name with no scheme into `covered_variants` and be silently
                    // accepted under a `_` wildcard arm. Skip coverage/binding so
                    // the bogus arm cannot also mask the real `non-exhaustive`
                    // diagnostic.
                    if constructor_pattern_out_of_scope(ctor_name, env, adt_reg) {
                        errors.push(CheckError::new(
                            CheckErrorKind::UnknownConstructor {
                                identifier: ctor_name.to_string(),
                            },
                            with_macro_provenance(pat, format!("unknown constructor: {ctor_name}")),
                            vec![format!(
                                "Constructor '{ctor_name}' is not in scope. Declare it \
                                 locally or add it to an import (e.g. \
                                 `import Mod ({ctor_name})`)"
                            )],
                        ));
                        return;
                    }
                    // Record the *resolved* variant name for exhaustiveness,
                    // not the bare pattern name. After reef's module-scoped
                    // constructor mangling (chelis#157), the registry keys
                    // variants by their package/module-qualified name, while
                    // a pattern may still be written with the bare terminal
                    // name (e.g. an unqualified `JsonNull` arm). Pushing the
                    // bare name would leave the mangled variant uncovered and
                    // fire a spurious `non-exhaustive match`. Resolve through
                    // the registry's terminal-unique lookup so coverage is
                    // compared on the same (mangled) key `variant_names`
                    // returns.
                    let covered_name = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name))
                        .map(|(_, variant)| variant.name.clone())
                        .unwrap_or_else(|| ctor_name.to_string());
                    covered_variants.push(covered_name);

                    // RFC D-CHECK: constructor pattern match on an
                    // out-of-module opaque type is rejected; binding
                    // inference continues so no error cascades.
                    if let Some((adt_name, _)) = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name))
                    {
                        let adt_name = adt_name.to_string();
                        crate::opacity::check_opaque_use(
                            crate::opacity::OpaqueAction::PatCtor,
                            &adt_name,
                            adt_reg,
                            errors,
                        );
                    }

                    // Look up constructor in env and decompose
                    if let Some(scheme) = env
                        .lookup(ctor_name)
                        .or_else(|| env.lookup_terminal_unique(ctor_name))
                    {
                        let scheme = scheme.clone();
                        let ctor_ty = env.instantiate(&scheme, vg, subst);
                        // Unify the result of the constructor with scrutinee type
                        match &ctor_ty {
                            Type::Fn(arg_types, ret) => {
                                let _ = unify(ret, scrutinee_ty, subst);
                                // Bind sub-patterns to argument types
                                for (i, sub_pat) in kids[1..].iter().enumerate() {
                                    if i < arg_types.len() {
                                        let resolved = subst.apply(&arg_types[i]);
                                        pattern_bindings(
                                            sub_pat,
                                            &resolved,
                                            env,
                                            vg,
                                            subst,
                                            adt_reg,
                                            errors,
                                            product,
                                            covered_variants,
                                            has_wildcard,
                                        );
                                    }
                                }
                            }
                            _ => {
                                // Nullary constructor
                                let _ = unify(&ctor_ty, scrutinee_ty, subst);
                            }
                        }
                    }
                }
            }
            DeepTag::PatAs => {
                // (pat-as {} name inner_pat): bind name to scrutinee type, recurse into inner_pat
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    product.record_bypass(pat, resolved.clone(), "pattern binding traversal");
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
                if kids.len() >= 2 {
                    pattern_bindings(
                        &kids[1],
                        scrutinee_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        product,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            DeepTag::PatRecord => {
                // (pat-record {} TypeName (kv {} k1 p1) ...): validate against ADT registry
                // kids[0] = TypeName, kids[1..] = (kv {} key pat)
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    // chelis#317: same out-of-scope guard as the positional
                    // `pat-ctor` arm — a record-shaped match against a
                    // constructor that was never imported must be an `unknown
                    // constructor` error, not a fuzzy bind to a foreign tag.
                    // `constructor_pattern_out_of_scope` also rejects the
                    // non-unique / unresolvable case a `_` wildcard arm would
                    // otherwise silently accept.
                    if constructor_pattern_out_of_scope(ctor_name, env, adt_reg) {
                        errors.push(CheckError::new(
                            CheckErrorKind::UnknownConstructor {
                                identifier: ctor_name.to_string(),
                            },
                            with_macro_provenance(pat, format!("unknown constructor: {ctor_name}")),
                            vec![format!(
                                "Constructor '{ctor_name}' is not in scope. Declare it \
                                 locally or add it to an import (e.g. \
                                 `import Mod ({ctor_name})`)"
                            )],
                        ));
                        return;
                    }
                    // Look up variant in ADT registry for the canonical
                    // field order and known field-name set used for
                    // validation diagnostics.
                    let variant_info = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name));
                    // RFC D-CHECK: record pattern match on an
                    // out-of-module opaque type is rejected; binding
                    // inference continues so no error cascades.
                    if let Some((adt_name, _)) = variant_info {
                        let adt_name = adt_name.to_string();
                        crate::opacity::check_opaque_use(
                            crate::opacity::OpaqueAction::PatRecord,
                            &adt_name,
                            adt_reg,
                            errors,
                        );
                    }

                    // Record the resolved (mangled) variant name for
                    // exhaustiveness, mirroring `pat-ctor`. See that arm for
                    // why the bare pattern name is not used (chelis#157).
                    let covered_name = variant_info
                        .map(|(_, vi)| vi.name.clone())
                        .unwrap_or_else(|| ctor_name.to_string());
                    covered_variants.push(covered_name);
                    let declared_field_names: Vec<Option<String>> = variant_info
                        .map(|(_, vi)| vi.fields.iter().map(|(n, _)| n.clone()).collect())
                        .unwrap_or_default();
                    let known_field_set: std::collections::HashSet<&str> = declared_field_names
                        .iter()
                        .filter_map(|n| n.as_deref())
                        .collect();

                    // Mirror the `pat-ctor` (positional) path: instantiate
                    // the constructor scheme and unify its return type
                    // with the scrutinee so the ADT's type parameters get
                    // pinned to the scrutinee's concrete instantiation
                    // (e.g. `FooState[a] -> FooState[tensor[n, f32]]`).
                    // The instantiated function's arg types are the
                    // properly substituted per-field types. Without this
                    // step, the declared field types still reference the
                    // ADT's abstract `a`, leaving record-pattern bindings
                    // stuck as fresh type variables and breaking
                    // downstream linearity/borrow checks. (closes #181)
                    let instantiated_arg_types: Vec<Type> = if let Some(scheme) = env
                        .lookup(ctor_name)
                        .or_else(|| env.lookup_terminal_unique(ctor_name))
                    {
                        let scheme = scheme.clone();
                        let ctor_ty = env.instantiate(&scheme, vg, subst);
                        match ctor_ty {
                            Type::Fn(arg_types, ret) => {
                                let _ = unify(&ret, scrutinee_ty, subst);
                                arg_types
                            }
                            // Nullary constructor: the scheme body is the
                            // ADT type itself, no Fn-wrapping. Still unify
                            // with the scrutinee so the ADT's type
                            // parameters are pinned to its concrete
                            // instantiation, mirroring `pat-ctor`'s
                            // positional path. There are no fields to
                            // bind for `Foo {}`, so the empty
                            // `instantiated_arg_types` is the right
                            // return value either way; the unify is the
                            // side-effect that matters.
                            other => {
                                let _ = unify(&other, scrutinee_ty, subst);
                                Vec::new()
                            }
                        }
                    } else {
                        Vec::new()
                    };

                    for kv_expr in kids.iter().skip(1) {
                        if let Some((DeepTag::Kv, _, kv_kids)) = stamped_parts(kv_expr)
                            && kv_kids.len() >= 2
                        {
                            let field_name = symbol_name(&kv_kids[0]);
                            // Look up declared field type — reject unknown fields
                            let field_ty = match field_name {
                                Some(n) => {
                                    if known_field_set.contains(n) {
                                        // Prefer the instantiated arg type from
                                        // the constructor scheme so the
                                        // scrutinee's concrete type arguments
                                        // are reflected in the pattern binding.
                                        let pos = declared_field_names
                                            .iter()
                                            .position(|nm| nm.as_deref() == Some(n));
                                        match pos.and_then(|i| instantiated_arg_types.get(i)) {
                                            Some(ty) => subst.apply(ty),
                                            None => {
                                                // Fallback: un-instantiated declared field
                                                // type when the constructor scheme isn't
                                                // in `env`. This branch SHOULD be
                                                // unreachable in practice: every `deftype`
                                                // registered in `adt_reg` via
                                                // `collect_declarations` also binds its
                                                // constructor scheme in `env` in the same
                                                // call. If that invariant drifts (e.g., a
                                                // future code path populates `adt_reg`
                                                // without binding into `env`), the
                                                // fallback would silently produce
                                                // `Var(T_a)` from the un-instantiated
                                                // VariantInfo — exactly the bug #181 fixed.
                                                // The debug_assert below flags the drift
                                                // in tests; the runtime fallback to
                                                // `vi.fields[i]` preserves pre-fix
                                                // behavior in release builds.
                                                debug_assert!(
                                                    false,
                                                    "env/adt_reg sync invariant violated: \
                                                         field `{n}` of constructor `{ctor_name}` \
                                                         is known to `adt_reg` (variant_info found) \
                                                         but the constructor scheme is missing from \
                                                         `env`. See infer.rs pat-record fallback note."
                                                );
                                                variant_info
                                                    .and_then(|(_, vi)| {
                                                        vi.fields.iter().find_map(|(name, ty)| {
                                                            (name.as_deref() == Some(n))
                                                                .then(|| ty.clone())
                                                        })
                                                    })
                                                    // Per the loop guard `known_field_set
                                                    // .contains(n)` and the fact that
                                                    // `known_field_set` is derived from
                                                    // `declared_field_names` whose
                                                    // `Some(_)` entries are exactly the
                                                    // named fields of `vi.fields`, the
                                                    // find_map above always returns Some
                                                    // here. The expect makes that explicit;
                                                    // if it ever fires, both data sources
                                                    // are themselves out of sync — a bug
                                                    // upstream of this site.
                                                    .expect(
                                                        "known_field_set is derived from \
                                                             vi.fields' named entries; mismatch \
                                                             indicates a corrupted AdtRegistry",
                                                    )
                                            }
                                        }
                                    } else if !known_field_set.is_empty() {
                                        // Unknown field name — error
                                        report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                format!(
                                                    "unknown record field '{}' in pattern for {}",
                                                    n, ctor_name
                                                ),
                                                vec![format!(
                                                    "known fields: {:?}",
                                                    declared_field_names
                                                        .iter()
                                                        .filter_map(|f| f.as_deref())
                                                        .collect::<Vec<_>>()
                                                )],
                                            ),
                                        )
                                    } else {
                                        vg.fresh_type() // no ADT info available
                                    }
                                }
                                None => vg.fresh_type(),
                            };
                            pattern_bindings(
                                &kv_kids[1],
                                &field_ty,
                                env,
                                vg,
                                subst,
                                adt_reg,
                                errors,
                                product,
                                covered_variants,
                                has_wildcard,
                            );
                        }
                    }
                }
            }
            DeepTag::PatTuple => {
                // (pat-tuple {} sub0 sub1 ...): every child is itself a
                // sub-pattern. Recurse into each so a nested
                // `pat-record` / `pat-ctor` reaches the RFC D-CHECK
                // opacity gate (and `pat-var` bindings get the right
                // element type) exactly as a top-level destructure does.
                // Without this recursion the catch-all below silently
                // dropped tuple-nested patterns, bypassing
                // `check_opaque_use` for out-of-module opaque types wrapped
                // in a tuple scrutinee.
                let resolved = subst.apply(scrutinee_ty);
                // Pair each child sub-pattern with the matching tuple
                // element type when the resolved scrutinee is a tuple of
                // equal arity; otherwise hand each child a fresh type
                // variable. The opacity check inside the nested
                // `pat-record` / `pat-ctor` arms keys off the pattern's
                // constructor name, not the scrutinee type, so the gate
                // still fires under a fresh-var element type.
                let elem_tys: Option<&[Type]> = match &resolved {
                    Type::Tuple(ts) if ts.len() == kids.len() => Some(ts.as_slice()),
                    _ => None,
                };
                for (i, sub_pat) in kids.iter().enumerate() {
                    let elem_ty = match elem_tys {
                        Some(ts) => subst.apply(&ts[i]),
                        None => vg.fresh_type(),
                    };
                    pattern_bindings(
                        sub_pat,
                        &elem_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        product,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_pipe(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return malformed_form(list, "pipe", "at least one stage", errors);
    }

    let mut current_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);

    for stage in &kids[1..] {
        // If the stage is the canonical bare-keyword / `cast(type)` pipe-stage
        // shape `(fn (params <single unannotated param>) body)` produced by
        // `crates/chelis-surf/src/parser.rs::parse_pipe_stage` and
        // `desugar_pipe_stage`, infer the lambda with its parameter bound to
        // the upstream pipe value's type. Without this pre-binding, per-builtin
        // inference gates inside the body (e.g. `infer_copy`, `infer_cast`)
        // see a fresh type variable for the parameter and reject before the
        // pipe loop's unification can bind it to `current_ty`. See
        // `docs/investigations/pipe_copy_typecheck_diagnosis.md` for the trace.
        let stage_ty = if let Some(param_name) = synthesized_unary_lambda_param(stage, adt_reg, vg)
        {
            infer_pipe_stage_lambda(
                stage,
                &param_name,
                current_ty.clone(),
                env,
                vg,
                subst,
                adt_reg,
                errors,
                product,
            )
        } else {
            infer_expr(stage, env, vg, subst, adt_reg, errors, product)
        };
        // A bare pipe stage (`x |> recip`) has no `app` node, so the normal
        // post-application policy check cannot see it. Consult the identical
        // chelis#860 operand policy at this application boundary.
        if let Some(fname) = bare_var_stage_name(stage) {
            let resolved = type_for_readonly_check(&current_ty, subst);
            if let Some((kind, message, hints)) = operand_dtype_rejection(fname, &resolved) {
                let mut error = CheckError::new(kind, message, hints);
                if let Some(id) = stage.span_id() {
                    error.span_offset = parse_span_offset(id);
                    error.span_id = Some(id.to_string());
                } else if stage.span().offset > 0 {
                    error.span_offset = Some(stage.span().offset);
                }
                return report(errors, error);
            }
        }
        let ret_tv = vg.fresh_type();
        let stage_arg_tys = auto_borrow_call_arg_types(&stage_ty, vec![current_ty.clone()], subst);
        let expected = Type::Fn(stage_arg_tys, Box::new(ret_tv.clone()));

        match unify(&stage_ty, &expected, subst) {
            Ok(()) => {
                current_ty = subst.apply(&ret_tv);
            }
            Err(te) => {
                let mut e: CheckError = te.into();
                if let Some(id) = stage.span_id() {
                    e.span_offset = parse_span_offset(id);
                    e.span_id = Some(id.to_string());
                } else {
                    let off = stage.span().offset;
                    if off > 0 {
                        e.span_offset = Some(off);
                    }
                }
                return report(errors, e);
            }
        }
    }

    current_ty
}

/// Return the builtin name of a bare-reference pipe stage. Lambda-shaped
/// stages contain ordinary application nodes and are handled by the normal
/// post-application chokepoint.
pub(super) fn bare_var_stage_name(stage: &deep::Expr) -> Option<&str> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(stage)?;
    if tag != DeepTag::Var {
        return None;
    }
    kids.first().and_then(symbol_name)
}

/// If `stage` is a `(fn (params x) body)` Deep node with exactly one
/// unannotated parameter -- the canonical shape produced by the Surf
/// parser's `parse_pipe_stage` and `desugar_pipe_stage` for bare
/// unary-builtin keyword stages (`x |> copy`, `x |> realize`) and the
/// one-arg `cast(type)` form (`x |> cast(f32)`) -- return the
/// parameter's name. Otherwise return `None`.
///
/// Multi-arg lambdas, lambdas with annotated parameters, and any other
/// pipe-stage form (named reference, partial application, etc.) fall
/// through unchanged.
pub(super) fn synthesized_unary_lambda_param(
    stage: &deep::Expr,
    _adt_reg: &AdtRegistry,
    _vg: &mut VarGen,
) -> Option<String> {
    // chelis#1107 amendment: carrier-preserving read (both levels).
    let (tag, _, kids) = stamped_parts(stage)?;
    if tag != DeepTag::Fn {
        return None;
    }
    let params_expr = kids.first()?;
    let (params_tag, _, param_kids) = stamped_parts(params_expr)?;
    if params_tag != DeepTag::Params {
        return None;
    }
    if param_kids.len() != 1 {
        return None;
    }
    // Single param must be a bare symbol; an annotated form would
    // surface as `MetaExpr` or a nested `List`, and the user-written
    // annotation takes precedence over the upstream pipe value's type.
    match &param_kids[0] {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some(name.to_string()),
        _ => None,
    }
}

/// Infer a synthesized unary pipe-stage lambda with its parameter
/// pre-bound to `param_ty`. Mirrors `infer_fn` but seeds the
/// parameter's scheme from `param_ty` instead of allocating a fresh
/// type variable, so per-builtin inference gates inside the body see
/// the upstream pipe value's type.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_pipe_stage_lambda(
    stage: &deep::Expr,
    param_name: &str,
    param_ty: Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    // chelis#1107 amendment: carrier-preserving read -- a stamped pipe-stage
    // `fn` node used to fall straight into the malformed-form rejection.
    let Some((_, _, kids)) = stamped_parts(stage) else {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::MalformedForm,
                "malformed pipe stage: expected a list-form stage node \
                 (spec/03-deep-syntax.md; chelis#731 [04-TOT-3])"
                    .to_string(),
                vec![],
            ),
        );
    };
    let body = match kids.get(1) {
        Some(body) => body,
        None => {
            // A stamped `fn` node satisfies its `Fixed(2)` arity contract at
            // construction, so only a legacy `List` carrier can be short here.
            // chelis#1107 amendment (justified-safe, not routed): see the
            // arity-contract argument in the comment directly above.
            let deep::Expr::List(list, _) = stage else {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::MalformedForm,
                        "malformed pipe stage: expected a body \
                         (spec/03-deep-syntax.md; chelis#731 [04-TOT-3])"
                            .to_string(),
                        vec![],
                    ),
                );
            };
            return malformed_form(list, "pipe stage", "a body", errors);
        }
    };

    let mut fn_env = env.clone();
    fn_env.bind(param_name.to_string(), Scheme::mono(param_ty.clone()));
    // chelis#397/#469: a fresh parameter has no size provenance; clear any
    // entry inherited from an outer name it shadows (BLOCKER C).
    fn_env.clear_size_provenance(param_name);
    // chelis#631: same for a shadowed list-literal length.
    fn_env.clear_list_literal_len(param_name);

    let body_ty = infer_expr(body, &mut fn_env, vg, subst, adt_reg, errors, product);

    let resolved_param = subst.apply(&param_ty);
    let resolved_body = subst.apply(&body_ty);
    let stage_ty = Type::Fn(vec![resolved_param], Box::new(resolved_body));
    product.record_bypass(stage, stage_ty.clone(), "synthesized pipe-stage inference");
    stage_ty
}
