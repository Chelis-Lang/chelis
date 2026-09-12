//! Generic application inference and operation dispatch.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// The §4.5.2 join categories are semantic, not `Dim` representation tags.
/// A private labelled Var can represent a concrete name with an optional
/// known extent; only an active authored binder must retain rigid evidence.
enum ListAxis {
    Variable,
    Wildcard,
    Concrete {
        extent: Option<i64>,
        label: Option<String>,
    },
}

impl ListAxis {
    fn classify(dim: &Dim, subst: &Subst) -> Self {
        let observation = subst.observe_dim(dim);
        if observation.rank().is_some()
            || (observation.variable().is_some()
                && (observation.name().is_none() || observation.is_protected()))
        {
            Self::Variable
        } else if observation.is_wildcard() {
            Self::Wildcard
        } else {
            Self::Concrete {
                extent: observation.known_extent(),
                label: observation.name().map(str::to_owned),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    expected_result: Option<&Type>,
) -> Type {
    // Preserve lexical ownership before inference mutates the environment.
    // This is discovery metadata, not a backend acceptance decision.
    let kids = children(list);
    let builtin = kids.first().and_then(|callee| {
        let (tag, _, parts) = stamped_parts(callee)?;
        if tag != DeepTag::Var {
            return None;
        }
        let name = parts.first().and_then(symbol_name)?;
        if env.is_lexically_bound(name) {
            return None;
        }
        builtins::builtin_decl(name)
    });
    let checkpoint = errors.checkpoint();
    let result = infer_app_inner(
        list,
        env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
        expected_result,
    );
    if errors.iter_since(checkpoint).next().is_none()
        && let Some(builtin) = builtin
    {
        // Only overload selectors need operand stamps. Axis syntax and other
        // non-value children must not be re-inferred to discover an identity.
        let positions: &[usize] = match builtin.name {
            "len" | "to_string" | "eq" | "neq" => &[0],
            "concat" => &[0, 1],
            _ => &[],
        };
        let mut arguments = vec![Type::Unit; kids.len().saturating_sub(1)];
        for &position in positions {
            if let Some(child) = kids.get(position + 1)
                && let Some(ty) = product.current_owner_type(child, subst, errors)
            {
                arguments[position] = ty;
            }
        }
        if errors.iter_since(checkpoint).next().is_none() {
            match builtin.semantic_selection(&arguments, subst) {
                Ok(selection) => product.record_builtin_selection(selection),
                Err(reason) => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "{} operand has no declared semantic case: {reason}",
                                builtin.name
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn infer_app_inner(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    expected_result: Option<&Type>,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return malformed_form(list, "app", "a callee expression", errors);
    }

    // Check if func is a comparison op (for special return type handling)
    let source_func_name = stamped_parts(&kids[0]).and_then(|(tag, _, callee_kids)| {
        (tag == DeepTag::Var)
            .then(|| {
                callee_kids
                    .first()
                    .and_then(symbol_name)
                    .map(str::to_string)
            })
            .flatten()
    });
    // Builtin-specific application rules are selected only after ordinary
    // lexical lookup. A parameter, block binding, or pattern binding with the
    // same builtin spelling owns the call; its inferred function type, rather
    // than the builtin's name-keyed checker route, decides whether the
    // application is valid (spec/04-type-system.md §8.6; chelis#1076).
    //
    // Keep the override exact to the closed builtin vocabulary. In particular,
    // applied uppercase heads retain constructor classification under
    // spec/01-nomenclature.md §3.2 even when a single-letter value binder with
    // the same spelling is in scope.
    let func_name = source_func_name.clone().filter(|name| {
        !builtins::BUILTIN_NAMES.contains(&name.as_str()) || !env.is_lexically_bound(name)
    });

    if matches!(func_name.as_deref(), Some("permute")) {
        return infer_permute_app(list, env, vg, subst, adt_reg, errors, product);
    }

    if matches!(func_name.as_deref(), Some("reshape")) {
        let inferred = infer_reshape_app(list, env, vg, subst, adt_reg, errors, product);
        if let Some(expected) = env.exact_stdlib_expected_result()
            && matches!(expected, Type::Tensor(_, _))
        {
            // #1298 owns the general runtime-axis shape relation. The exact
            // [05-OP-35] stdlib graph is structurally locked and its declared
            // result is authoritative at this package-reserved boundary; the
            // ordinary reshape checker still traverses and stamps every child.
            return expected.clone();
        }
        return inferred;
    }

    if matches!(func_name.as_deref(), Some("shrink")) {
        return infer_shrink_app(list, env, vg, subst, adt_reg, errors, product);
    }

    if matches!(func_name.as_deref(), Some("pad")) {
        return infer_pad_app(list, env, vg, subst, adt_reg, errors, product);
    }

    if matches!(func_name.as_deref(), Some("stride")) {
        return infer_stride_app(list, env, vg, subst, adt_reg, errors, product);
    }

    // chelis#339: the anchored named-axis expand form `expand(x, new, size,
    // anchor)` carries four arguments, but the builtin HM scheme is arity-3
    // (`(&tensor, int32, int32) -> out`), so it would hit the generic arity
    // check before the procedural arm. Dispatch it here (the
    // `infer_permute_app` pattern); 2-/3-arg expand keeps the generic path,
    // which reaches `check_expand_signature` with the scheme intact.
    if let Some(callee @ ("expand" | "insert")) = func_name.as_deref()
        && kids.len() >= 5
    {
        // `&'static str`, not the borrow, so the callee outlives `func_name`.
        let callee = if callee == "insert" {
            "insert"
        } else {
            "expand"
        };
        return infer_expand_app(callee, list, env, vg, subst, adt_reg, errors, product);
    }

    // chelis#339 Part 2: variadic named-axis reduction `sum(x, seq, head)`.
    // The reduction HM schemes are arity-2 (`(input, axis)`), so a 3+-arg
    // call would hit the generic arity check before the reduction arm;
    // dispatch it here. `check_reduction_signature`'s named loop handles N
    // axes (rejecting positional integers, unknown names, and duplicates).
    // The index-returning reductions are routed too, so they get a targeted
    // no-variadic-form rejection instead of a generic arity error.
    if matches!(
        func_name.as_deref(),
        Some(
            "sum"
                | "count"
                | "mean"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    ) && kids.len() >= 4
    {
        return infer_reduction_app(
            list,
            func_name.as_deref().unwrap(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            product,
        );
    }

    if matches!(
        func_name.as_deref(),
        Some(
            "reduce_window_max" | "reduce_window_min" | "reduce_window_sum" | "reduce_window_mean"
        )
    ) {
        return infer_reduce_window_app(
            list,
            func_name.as_deref().unwrap(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            product,
        );
    }

    match prepare_constructor_application(&func_name, env, adt_reg, errors) {
        Ok(_) => {}
        Err(rejected) => return rejected,
    }

    // Applied uppercase heads are constructor syntax, even when a value
    // binder with the same spelling is present (spec/01 §3.2). Do not send a
    // resolved constructor back through the string-keyed value environment:
    // a parameter named `N` would replace the constructor scheme there.
    // Conversely, do not rediscover an owner by scanning the ADT registry:
    // two positional constructors may share a name (`Option::Some` and a
    // local `Wrapper::Some`), and registry order is not scope. The separate
    // constructor authority preserves the active declaration/import owner
    // and scheme across ordinary lexical shadowing.
    let applied_constructor_head = source_func_name.as_deref().is_some_and(is_constructor_name);
    let func_ty = if applied_constructor_head {
        let source_name = source_func_name.as_deref().unwrap();
        let resolved_constructor = if constructor_out_of_scope(source_name, env) {
            None
        } else {
            constructor_for_shape(source_name, CallShape::Positional, env, adt_reg)
                .map(|(_, scheme, _)| env.instantiate(scheme, vg, subst))
        };
        match resolved_constructor {
            Some(constructor_type) => constructor_type,
            None => {
                let name = source_func_name.as_deref().unwrap();
                report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::UnknownConstructor {
                            identifier: name.to_string(),
                        },
                        with_macro_provenance(&kids[0], format!("unknown constructor: {name}")),
                        vec![format!(
                            "Constructor '{name}' is not in scope. Declare it locally or add it \
                             to an import (e.g. `import Mod ({name})`)"
                        )],
                    ),
                )
            }
        }
    } else {
        infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product)
    };
    // The constructor callee no longer passes through `infer_expr`, but it is
    // still a runtime expression owner and must contribute the same fitness
    // and annotation receipt as every other callee.
    if applied_constructor_head {
        product.total_nodes += 1;
        if !matches!(func_ty, Type::Error(_)) {
            product.typed_nodes += 1;
        }
        product.record_canonical(&kids[0], func_ty.clone());
    }
    // A reduction's axis argument may name a *dimension* of the operand
    // (`sum(x, seq)`, Tier-3 named-axis reduction, spec §4.5.3), not a bound
    // *value*. Like `expand`'s symbolic size arg below, such a name is typed as
    // an axis (`int32`) rather than inferred as a value — otherwise the
    // name-resolution pass would report a spurious `unbound variable`. The
    // actual name is read back from the arg expr in `check_reduction_signature`.
    let is_named_reduction = matches!(
        func_name.as_deref(),
        Some(
            "sum"
                | "count"
                | "mean"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // `expand` axis slots that may carry a dim NAME instead of a bound
            // value: the size (index 2, §4.7.2 form 2) and — chelis#339
            // named-axis expand — the inserted-axis name (index 1). The
            // inserted-axis slot is scope-discriminated: a name bound in the
            // value environment is a *runtime value* (the issue #259 class,
            // `expand(&x, ax, 4)` with `ax: int32`), not a dim name, and must
            // keep flowing through ordinary inference into the
            // compile-time-constant rejection. The 4-arg anchored form routes
            // through `infer_expand_app` instead and never reaches this loop.
            let is_expand = matches!(func_name.as_deref(), Some("expand") | Some("insert"));
            let is_expand_size = is_expand && index == 2;
            let is_expand_inserted_name = is_expand
                && index == 1
                && symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none());
            let is_reduction_axis = is_named_reduction && index >= 1;
            if (is_expand_size || is_expand_inserted_name || is_reduction_axis)
                && symbolic_dim_ref_name(arg).is_some()
            {
                // [05-DIM-1]: a dim name in the size slot is an extent
                // (int64); the inserted-axis name and reduction axes are
                // axis-domain (int32).
                if is_expand_size {
                    Type::Prim(Prim::Int64)
                } else {
                    Type::Prim(Prim::Int32)
                }
            } else {
                infer_expr(arg, env, vg, subst, adt_reg, errors, product)
            }
        })
        .collect();

    if matches!(func_name.as_deref(), Some("drop")) && arg_tys.len() == 1 {
        return Type::Unit;
    }

    // The builtin signature structurally shares one precision variable across
    // both tensors and the tolerance. Preserve the operation-specific direct
    // mismatch diagnostics by inspecting concrete operands before generic
    // unification reports its lower-level precision pair. Unresolved generic
    // wrappers pass this precheck and are constrained by the shared variable.
    if matches!(func_name.as_deref(), Some("test_assert_close_tensor"))
        && let Some(rejected) = reject_test_assert_close_tensor_operand_dtypes(
            list,
            &arg_tys,
            subst,
            errors,
            // No suspension: this pass runs before signature unification has
            // constrained anything, and `finish_unified_app` calls the same
            // function again on the types unification produced. That later call
            // is the one that decides, so it is the one that suspends.
            None,
            &Type::Unit,
            product,
        )
    {
        return rejected;
    }

    // [05-DIM-3]: the semantic registry owns axis dtype slots. `concat`
    // is overloaded with ordinary list concatenation and therefore runs
    // the same shared gate only after its tensor-list arm is identified in
    // `postprocess_application`; every unambiguous builtin is screened here.
    if let Some(fname) = func_name.as_deref()
        && fname != "concat"
        && let Err(rejected) = enforce_registered_axis_dtypes(fname, &arg_tys, list, errors)
    {
        return rejected;
    }

    // [05-DIM-1] fix-naming diagnostic for expand's extent slot: a wrong
    // size dtype would otherwise surface as the scheme unification's bare
    // `precision mismatch` pair. Pre-check the resolved size type here so
    // the rejection names the fix, mirroring shrink/pad/stride/reshape.
    if let Some(callee @ ("expand" | "insert")) = func_name.as_deref()
        && arg_tys.len() >= 3
        && let Type::Prim(p) = subst.apply(&arg_tys[2])
        && p != Prim::Int64
    {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "{callee} expects an int64 size (write Ni64 or cast(N, int64)), \
                         got {}",
                        Type::Prim(p)
                    ),
                ),
                vec![],
            ),
        );
    }

    // If the *callee* is Error, propagate. With no resolved callee scheme
    // there is no return type to produce: `subst.apply(&ret_tv)` (see below)
    // would leak a bare `Var`, because the `unify` against `expected_fn` is
    // Error-permissive and never binds `ret_tv`. A callee-`Error` already
    // carries its own diagnostic from the failed `var` lookup, so this
    // early-out suppresses no checking.
    //
    // chelis#773: we do NOT short-circuit on an Error-typed *argument*. The
    // previous arm collapsed the whole call to `Type::Error` the moment any
    // (non-`expand`-size) argument inferred to `Error`, which disabled type
    // checking of every sibling argument in the same call — one reported
    // error in one slot masked genuine mismatches in the others. Error-typed
    // arguments now flow into the per-slot `unify` below, which treats
    // `Type::Error` as permissive (`unify.rs`: `(Error, _) | (_, Error) =>
    // Ok(())`): siblings still check against their scheme slots, and the
    // call returns the callee's resolved return type instead of `Error`.
    //
    // The chelis#530 `expand` SIZE-slot concern is subsumed, not lost: both
    // expand size gates read the size from the raw AST (not from
    // `arg_tys[2]`), and now that no argument short-circuits here, an
    // Error-typed size slot always reaches the per-form located diagnostic.
    // A genuinely sourced size never infers to `Error`.
    if let Some(err) = propagate_if_error([&func_ty]) {
        return err;
    }

    // Issue Chelis-Lang/chelis#218 R3 HIGH-CONCAT: when `Cons` is
    // called with two concrete tensor-typed args (head: tensor,
    // tail: List<tensor>), skip the generic per-dim equality
    // unification and produce a per-axis join (Lit(n) when both
    // dims are Lit(n) and equal; Wildcard otherwise). The generic
    // `unify` recursion walks dims pairwise and rejects
    // `Lit(2) vs Lit(3)`, which breaks `concat([a, b], 0)` after
    // the to_tensor source fix made nested-list literals emit
    // concrete dims.
    //
    // Precondition guards:
    //   * exactly two args (Cons signature)
    //   * head is a concrete tensor type
    //   * tail is `List<tensor[...]>` with a concrete tensor element
    //   * head and tail-element have matching rank (no rank join;
    //     mismatched ranks remain structural errors)
    //   * head and tail-element have matching precision (no
    //     precision join; mismatched precisions would mask real
    //     type errors)
    //
    // When any precondition fails, fall through to the generic
    // unify path so other Cons shapes (e.g. `Cons(scalar, list)` or
    // `Cons(head, Nil)` where `Nil`'s tvar binds the element type)
    // keep their existing semantics.
    if matches!(func_name.as_deref(), Some("Cons")) && arg_tys.len() == 2 {
        let head_resolved = subst.apply(&arg_tys[0]);
        let tail_resolved = subst.apply(&arg_tys[1]);
        if let (Type::Tensor(head_dims, head_prec), Type::Adt(list_name, list_args)) =
            (&head_resolved, &tail_resolved)
            && list_name == "List"
            && list_args.len() == 1
            && let Type::Tensor(tail_dims, tail_prec) = subst.apply(&list_args[0])
        {
            if head_dims.len() != tail_dims.len() {
                // chelis#255: surface the rank-uniform rule and the
                // reshape/flatten remediation in the diagnostic itself,
                // so users (and agents reading JSON output) are not
                // left guessing why a `List[tensor[k, f32]]` rejected
                // a rank-mixed literal. The dim slot `k` is a
                // dimension variable, not a shape-vector variable;
                // see spec/04-type-system.md §4.5.1.
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!(
                                "list element rank mismatch: {} dims vs {} dims; \
                             List[tensor[...]] requires rank-uniform elements \
                             (the dim slot is a dimension variable, not a \
                             shape-vector variable). Reshape or flatten \
                             elements to a common rank before listing \
                             (spec/04-type-system.md §4.5.1).",
                                head_dims.len(),
                                tail_dims.len(),
                            ),
                        ),
                        vec![],
                    ),
                );
            }
            if let Err(te) = unify_tensor_prec(head_prec, &tail_prec, subst) {
                return report(errors, te.into());
            }
            // Per-axis join (chelis#218 concat ergonomics, tightened
            // by chelis#272). Resolve each dim through the current
            // substitution first so already-bound dim variables compare
            // as their concrete value.
            //
            //   * equal concrete literals or equal names  -> keep them;
            //   * genuinely-mismatched concrete literals
            //     (e.g. `Lit(2)` vs `Lit(3)`) or mismatched concrete
            //     names                                    -> widen to
            //     `Wildcard`. This is the deliberate #218 behavior that
            //     lets bare `concat([a, b], axis)` accept ragged
            //     concrete axes; and
            //   * any pair that involves a dimension *variable*
            //     (a declared rigid dim parameter such as `k`/`m`)
            //     -> `unify_dim` the two dims instead of widening.
            //
            // The last arm is the #272 fix: the old `_ => Wildcard`
            // erased named dim variables, so a list body that violated
            // the §4.4 rigid-distinct-dim guarantee
            // (`def make[k, m](a: tensor[k], b: tensor[m])
            //   -> List[tensor[k]] = [a, b]`) collapsed `k`/`m` to a
            // wildcard before `check_declared_dvars_rigid` ran. Unifying
            // them instead keeps the surviving evidence: distinct rigid
            // dims unify with each other (the guard then reports the
            // collapse) and a `(rigid, concrete)` pair pins the rigid
            // dim to a literal (the guard reports the pin). A
            // `unify_dim` failure here (which the permissive
            // Name/Lit/Wildcard arms make rare) surfaces as a structural
            // dimension mismatch rather than being silently widened.
            let mut joined_dims: Vec<Dim> = Vec::with_capacity(head_dims.len());
            for (h, t) in head_dims.iter().zip(tail_dims.iter()) {
                let hr = subst.apply_dim(h);
                let tr = subst.apply_dim(t);
                let joined = match (
                    ListAxis::classify(&hr, subst),
                    ListAxis::classify(&tr, subst),
                ) {
                    (
                        ListAxis::Concrete {
                            extent: a,
                            label: an,
                        },
                        ListAxis::Concrete {
                            extent: b,
                            label: bn,
                        },
                    ) => {
                        // Names and literal extents are separate observations.
                        // Name/Lit and distinct names widen even at equal sizes;
                        // equal names cannot conceal a known extent conflict.
                        if an == bn && !matches!((a, b), (Some(a), Some(b)) if a != b) {
                            hr.clone()
                        } else {
                            Dim::Wildcard
                        }
                    }
                    // Keep genuine variable constraints for the body-rigidity
                    // guards, and preserve the specified wildcard head bias.
                    _ => {
                        if let Err(te) = unify_dim(&hr, &tr, subst) {
                            return report(errors, te.into());
                        }
                        subst.apply_dim(&hr)
                    }
                };
                joined_dims.push(joined);
            }
            let joined_prec = subst.apply_tensor_prec(head_prec);
            let elem = Type::Tensor(joined_dims, joined_prec);
            return Type::Adt("List".to_string(), vec![elem]);
        }
    }

    let ret_tv = vg.fresh_type();

    // [04-DTYPE-2] restricts a bounded type variable to primitive dtypes.
    // It therefore has a scalar surface even before specialization. Reject
    // mixed surfaces before unification can emit an unrelated occurs-check
    // error for p beside tensor[D, p]. Unrestricted variables remain unknown.
    if let Some(ref fname) = func_name
        && (builtins::COMPARISON_OPS.contains(&fname.as_str())
            || matches!(
                fname.as_str(),
                "add" | "sub" | "mul" | "div" | "floor_div" | "trunc_div" | "max_elem" | "min_elem"
            ))
        && arg_tys.len() == 2
    {
        let lhs = type_for_readonly_check(&arg_tys[0], subst);
        let rhs = type_for_readonly_check(&arg_tys[1], subst);
        let scalar = |ty: &Type| {
            matches!(ty, Type::Prim(_))
                || matches!(ty, Type::Var(p) if subst.tvar_restriction(*p).is_some())
        };
        if (matches!(lhs, Type::Tensor(..)) && scalar(&rhs))
            || (scalar(&lhs) && matches!(rhs, Type::Tensor(..)))
        {
            let authority = if builtins::COMPARISON_OPS.contains(&fname.as_str()) {
                "spec/05-risc-primitives.md [05-OP-36] makes a mixed surface a type error, and section 1.2 admits no broadcasting exception"
            } else {
                "spec/05-risc-primitives.md section 1.2 and spec/04-type-system.md section 4.3 require explicit shape construction"
            };
            let mut error = CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("`{fname}` does not admit a scalar beside a tensor, got {lhs} and {rhs}: {authority}"),
                vec![
                    "Give the scalar the tensor's shape explicitly. For a rank-one tensor xs and a scalar c of the same dtype, use `insert(scalar_to_tensor(c), 0i32, shape(xs, 0i32))`; insert each axis for higher ranks.".to_string(),
                    "For concrete f32 values, another explicit spelling is `gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))`.".to_string(),
                ],
            );
            if let Some(id) = list_span_id(list) {
                error.span_offset = parse_span_offset(id);
                error.span_id = Some(id.to_string());
            } else {
                let off = span_of_list(list).offset;
                if off > 0 {
                    error.span_offset = Some(off);
                }
            }
            return report(errors, error);
        }
    }

    let unify_arg_tys = auto_borrow_call_arg_types(&func_ty, arg_tys.clone(), subst);
    let expected_fn = Type::Fn(unify_arg_tys, Box::new(ret_tv.clone()));

    match unify(&func_ty, &expected_fn, subst) {
        Ok(()) => {
            // chelis#1512: watch whether the eager pass rejects this call. A
            // route can suspend on one operand and then reject on another in
            // the same pass, and the replay re-enters the whole route, so the
            // rejection would be reported a second time. A call that has
            // already failed has nothing left to decide, so its suspension is
            // cancelled here.
            let checkpoint = errors.checkpoint();
            let applied = finish_unified_app(
                list,
                kids,
                func_name,
                arg_tys,
                ret_tv,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                product,
                expected_result,
            );
            if errors.iter_since(checkpoint).next().is_some() {
                product.cancel_post_app_check_for(list);
            }
            applied
        }
        Err(te) => {
            let mut e: CheckError = te.into();
            if let Some(id) = list_span_id(list) {
                e.span_offset = parse_span_offset(id);
                e.span_id = Some(id.to_string());
            } else {
                let off = span_of_list(list).offset;
                if off > 0 {
                    e.span_offset = Some(off);
                }
            }
            report(errors, e)
        }
    }
}
