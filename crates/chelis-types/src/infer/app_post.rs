//! Operation-family checks after generic application unification.
//!
//! The checks remain in their original order. Each rejection still returns
//! before the next operation family runs.

use super::app_post_diagonal::{diagonal_result_error, reject_unreachable_diagonal_extent};
use super::*;

/// [05-OP-56]'s key domain: string, bool, or an active signed-integer scalar.
/// Float, aggregate, tensor and random-key types are not dictionary keys.
fn is_dict_key_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => matches!(prim, Prim::String | Prim::Bool) || prim.is_integer(),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_unified_app(
    source_site: CheckSite<'_>,
    node: &DeepNode,
    kids: &[deep::Expr],
    func_name: Option<String>,
    arg_tys: Vec<Type>,
    ret_tv: Type,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    expected_result: Option<&Type>,
) -> Type {
    if let Some(rule) = func_name
        .as_deref()
        .and_then(builtins::builtin_decl)
        .and_then(|decl| {
            decl.capability.sibling_cases.iter().find_map(
                |case| match builtins::case_value_equality(case.case) {
                    builtins::ValueEquality::Aggregate(rule) => Some(rule),
                    builtins::ValueEquality::CallbackApplication
                    | builtins::ValueEquality::NoAggregateEquality => None,
                },
            )
        })
    {
        // Concat has a tensor overload; its selector and call-site shape
        // evidence remain below. Every other registered aggregate uses only
        // the immutable decision API before ordinary dispatch can mutate types.
        if rule != builtins::AggregateRule::Concat {
            let name = func_name
                .as_deref()
                .expect("registered aggregate has a name");
            if arg_tys
                .iter()
                .any(|ty| matches!(subst.apply(ty), Type::Error(_)))
            {
                return subst.apply(&ret_tv);
            }
            let operands = match rule {
                builtins::AggregateRule::Fold | builtins::AggregateRule::Scan => {
                    if arg_tys.len() != 3 {
                        return report_builtin_arity(
                            errors,
                            node,
                            source_site,
                            name,
                            3,
                            arg_tys.len(),
                        );
                    }
                    let element = vg.fresh_type();
                    if let Err(error) = unify(
                        &arg_tys[2],
                        &Type::Adt("List".to_string(), vec![element.clone()]),
                        subst,
                    ) {
                        return report(errors, error.into());
                    }
                    // Invoke the callback with the initial accumulator. Its
                    // result is an independent slot: equality with the next
                    // accumulator belongs to the aggregate rule below.
                    let callback_result = match unify_checked_call_contract(
                        &kids[1],
                        None,
                        &arg_tys[0],
                        &[arg_tys[1].clone(), element],
                        None,
                        vg,
                        subst,
                        errors,
                        product,
                    ) {
                        Ok(result) => result,
                        Err(rejected) => return rejected,
                    };
                    if rule == builtins::AggregateRule::Scan {
                        vec![
                            Type::Adt("List".to_string(), vec![arg_tys[1].clone()]),
                            Type::Adt("List".to_string(), vec![callback_result]),
                        ]
                    } else {
                        vec![arg_tys[1].clone(), callback_result]
                    }
                }
                builtins::AggregateRule::Append
                | builtins::AggregateRule::DictInsert
                | builtins::AggregateRule::DictMerge
                | builtins::AggregateRule::Concat => arg_tys.clone(),
            };
            match rule.decide(&operands, &ret_tv, subst) {
                Ok(Some(equation)) => {
                    subst.record_result_constraint(equation);
                    return subst.apply(&ret_tv);
                }
                Ok(None) => {
                    let name = func_name
                        .as_deref()
                        .expect("registered aggregate has a name");
                    return UnresolvedOperandSite::new(node, kids, name, env).defer(
                        &arg_tys,
                        &ret_tv,
                        product,
                        ret_tv.clone(),
                    );
                }
                Err(mut error) => {
                    error.message = with_node_provenance(node, error.message);
                    return report(errors, *error);
                }
            }
        }
    }
    let mut result_ty = subst.apply(&ret_tv);
    let checked_rule = checked_inference_rule(func_name.as_deref());
    let mut checked_route_observed = false;

    // [04-TENSOR-EXPAND]: `expand` and `insert` each have one result shape,
    // so an expected tensor selects nothing. Both still need the seed, for
    // the same reason. Unseeded, the result is
    // still a variable when `check_expand_signature` runs, so the call takes
    // the `Type::Var(_) if inserts_only` arm, builds its one shape from the
    // operand and axis alone, and any disagreement with the declared result
    // surfaces later as a generic ascription mismatch naming no operation.
    // Seeded, the call reaches the rank and precision arms that name the
    // callee, and for an agreeing result both arms build the same shape.
    // Both spellings and no other operation: everything else retains
    // ordinary bottom-up inference.
    if matches!(func_name.as_deref(), Some("expand") | Some("insert"))
        && let Some(expected) = expected_result
    {
        if let Err(error) = unify(&result_ty, expected, subst) {
            return report_at_check_site(errors, error.into(), source_site);
        }
        result_ty = subst.apply(&result_ty);
    }

    if let Some(rejected) = reject_unregistered_checked_route(func_name.as_deref(), errors) {
        return rejected;
    }

    // chelis#1512: the three dtype-admissibility validators below run ahead of
    // the route dispatch that owns the deferral site at the bottom of this
    // function, so they get the same site built early. Each admitted a
    // `Type::Var` operand and walked away; now each suspends the call instead,
    // and `replay_dtype_admissibility` re-runs these same three functions once
    // the operand settles.
    let dtype_site = func_name
        .as_deref()
        .map(|fname| DtypeAdmissibilitySite::new(node, kids, fname, env).with_site(source_site));

    if let Some(rejected) = validate_numeric_and_reduction_arguments(
        node,
        kids,
        &func_name,
        &arg_tys,
        env,
        subst,
        adt_reg,
        errors,
        &mut checked_route_observed,
        dtype_site.as_ref(),
        &result_ty,
        product,
    ) {
        return rejected;
    }

    if let Some(rejected) = reject_inadmissible_operand_dtypes(
        node,
        func_name.as_deref(),
        &arg_tys,
        env,
        subst,
        errors,
        &mut checked_route_observed,
        dtype_site.as_ref(),
        &result_ty,
        product,
    ) {
        return rejected;
    }

    if let Some(result) = integer_binop_result_type(
        node,
        func_name.as_deref(),
        &arg_tys,
        vg,
        subst,
        errors,
        dtype_site.as_ref(),
        &result_ty,
        product,
    ) {
        return result;
    }

    // chelis#778 follow-up: a shape-computed builtin override derives
    // its result shape from operand shapes and returns
    // `subst.apply(ret_tv)`. When an operand type is `Error`, the
    // Error-permissive per-slot unify above (`(Error, _) => Ok(())`)
    // never bound `ret_tv`, so the override would leak an unbound
    // `Var` as the call's result type. That bare `Var` would be
    // degraded to a rank-0 default `TensorType` during writeback and
    // CLOBBER the node's concrete type annotation, ICEing the IR
    // lowering (`conv output height axis requires a statically known
    // axis`; #778's short-circuit removal exposed this). Returning
    // `Type::Error` instead restores the pre-#778 downstream shape
    // WITHOUT re-adding the arg-`Error` short-circuit: the per-slot
    // unify above already ran, so sibling-argument checking (#773's
    // de-mask win) is preserved. Keep this list in sync with the
    // shape-computed override arms in the match below.
    let shape_override_operand_error = func_name.as_deref().is_some_and(|fname| {
        matches!(
            fname,
            "matmul"
                | "sum"
                | "count"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
                | "mean"
                | "expand"
                | "layer_norm"
                | "conv"
                | "scatter_elements"
        ) && arg_tys
            .iter()
            .any(|ty| matches!(subst.apply(ty), Type::Error(_)))
    });
    if shape_override_operand_error {
        checked_route_observed = true;
        // Cascade: a shape-computed builtin operand already typed as
        // `Type::Error`; propagate its witness rather than mint a fresh
        // error (chelis#731 §C3). The `.any(... Error ...)` guard above
        // guarantees a witness is present, so the fallback is inert.
        result_ty = arg_tys
            .iter()
            .find_map(|ty| match subst.apply(ty) {
                Type::Error(w) => Some(propagate(&w)),
                _ => None,
            })
            .unwrap_or(result_ty);
    } else if let Some(ref fname) = func_name {
        // chelis#1836: ANY unresolved outer constructor suspends the route,
        // not only one an annotation-free lambda parameter owns. The
        // provenance predicate this replaced recognized a variable descending
        // from such a parameter and nothing else, so three provenances took
        // the route's eager `Type::Var` arm instead: a `pat-tuple` element on
        // an unresolved scrutinee, a field of an unresolved record target, and
        // a chelis#1577 gate's result. Each published the call's own result
        // variable, which the declaration was then free to bind to any shape.
        // This is the readiness predicate every other suspension already uses
        // (`defer_or_check_shape_route`, the chelis#1577 dtype gates, and the
        // replay pass itself), so a route now suspends on exactly the
        // condition it resumes on, and provenance stops deciding anything.
        let owes_shape_replay = arg_tys
            .iter()
            .any(|ty| shape_operand_awaits_binding(ty, subst));
        let mut retained_shape_obligation = false;
        match fname.as_str() {
            "matmul" => {
                checked_route_observed = true;
                if owes_shape_replay {
                    product.defer_shape_check(
                        DeferredShapeRule::Matmul {
                            location: source_site.owned_location(),
                        },
                        kids[1..].to_vec(),
                        arg_tys.clone(),
                        result_ty.clone(),
                    );
                    retained_shape_obligation = true;
                }
                result_ty =
                    check_matmul_signature(source_site, &arg_tys, &result_ty, subst, errors);
            }
            "sum" | "count" | "max_reduce" | "min_reduce" | "prod_reduce" | "argmax_reduce"
            | "argmin_reduce" | "mean" => {
                checked_route_observed = true;
                if owes_shape_replay {
                    product.defer_shape_check(
                        DeferredShapeRule::Reduction {
                            name: fname.clone(),
                            location: source_site.owned_location(),
                        },
                        kids[1..].to_vec(),
                        arg_tys.clone(),
                        result_ty.clone(),
                    );
                    retained_shape_obligation = true;
                }
                result_ty = check_reduction_signature(
                    source_site,
                    fname,
                    &kids[1..],
                    &arg_tys,
                    &result_ty,
                    subst,
                    errors,
                );
            }
            name @ ("expand" | "insert") => {
                // `&'static str`, not the borrow: the deferred rule outlives `fname`.
                let callee = if name == "insert" { "insert" } else { "expand" };
                checked_route_observed = true;
                // chelis#339: the axis slot is a dim NAME (the
                // named-axis insert form) only when it is not bound in
                // the value environment — a bound `i32` var is the
                // issue #259 runtime-value class instead.
                let axis_is_dim_name = kids.get(2).is_some_and(|arg| {
                    symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none())
                });
                if owes_shape_replay {
                    product.defer_shape_check(
                        DeferredShapeRule::Expand {
                            builtin: callee,
                            axis_is_dim_name,
                            env: Box::new(env.clone()),
                            location: source_site.owned_location(),
                        },
                        kids[1..].to_vec(),
                        arg_tys.clone(),
                        result_ty.clone(),
                    );
                    retained_shape_obligation = true;
                }
                result_ty = check_expand_signature(
                    source_site,
                    callee,
                    &kids[1..],
                    &arg_tys,
                    &result_ty,
                    axis_is_dim_name,
                    env,
                    subst,
                    errors,
                );
            }
            "layer_norm" => {
                checked_route_observed = true;
                if owes_shape_replay {
                    product.defer_shape_check(
                        DeferredShapeRule::LayerNorm {
                            location: source_site.owned_location(),
                        },
                        kids[1..].to_vec(),
                        arg_tys.clone(),
                        result_ty.clone(),
                    );
                    retained_shape_obligation = true;
                }
                result_ty = check_layer_norm_signature(
                    source_site,
                    &arg_tys,
                    &result_ty,
                    vg,
                    subst,
                    errors,
                );
            }
            "conv" => {
                checked_route_observed = true;
                if owes_shape_replay {
                    product.defer_shape_check(
                        DeferredShapeRule::Conv {
                            location: source_site.owned_location(),
                        },
                        kids[1..].to_vec(),
                        arg_tys.clone(),
                        result_ty.clone(),
                    );
                    retained_shape_obligation = true;
                }
                result_ty = check_conv_signature(
                    source_site,
                    &kids[1..],
                    &arg_tys,
                    &result_ty,
                    vg,
                    subst,
                    errors,
                );
            }
            "split_keys" => {
                checked_route_observed = true;
                result_ty = check_split_keys_signature(
                    source_site,
                    &kids[1..],
                    &result_ty,
                    env,
                    subst,
                    errors,
                );
            }
            "scatter_elements" if owes_shape_replay => {
                checked_route_observed = true;
                product.defer_shape_check(
                    DeferredShapeRule::ScatterElements { node: node.clone() },
                    kids[1..].to_vec(),
                    arg_tys.clone(),
                    result_ty.clone(),
                );
                retained_shape_obligation = true;
            }
            _ => {}
        }
        if owes_shape_replay
            && !retained_shape_obligation
            && builtins::builtin_decl(fname).is_some_and(|decl| {
                decl.inference
                    == builtins::InferenceDisposition::Checked(
                        builtins::BuiltinInferenceRule::ShapeComputed,
                    )
            })
        {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::Other,
                    format!(
                        "internal: shape-computed builtin `{fname}` reached an unbound operand without retaining its semantic obligation"
                    ),
                    vec![
                        "Add the builtin's ordinary checker rule to the deferred shape ledger in the same change as its ShapeComputed disposition"
                            .to_string(),
                    ],
                ),
            );
        }
    }

    // Post-check: logical ops require tensor[D, bool] arguments
    if let Some(ref fname) = func_name
        && LOGICAL_OPS.contains(&fname.as_str())
    {
        checked_route_observed = true;
        for arg_ty in &arg_tys {
            let resolved = type_for_readonly_check(arg_ty, subst);
            match &resolved {
                Type::Tensor(_, TensorPrec::Concrete(Prim::Bool))
                | Type::Prim(Prim::Bool)
                | Type::Var(_)
                | Type::Error(_) => {} // OK
                Type::Tensor(_, prec) => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "{} requires tensor[D, bool] arguments, got tensor[D, {}]",
                                    fname,
                                    prec.name()
                                ),
                            ),
                            vec!["Logical ops only work on bool tensors".to_string()],
                        ),
                    );
                }
                other => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!("{} requires bool arguments, got {}", fname, other),
                            ),
                            vec!["Logical ops only work on bool values".to_string()],
                        ),
                    );
                }
            }
        }
    }

    // Special case: comparison ops return tensor[D, bool] when their
    // arguments resolve to tensors. chelis#1506 removed the scalar/tensor
    // rewrite and rejects a concrete primitive beside a tensor before
    // unification under `[05-OP-36]`. A bounded-binder cast can still enter
    // unification as a type variable and acquire the tensor type (chelis#1621),
    // so resolved types alone do not prove that both source operands were
    // tensors. Keep searching for a tensor shape: one operand can remain a
    // variable while the other is ground, and issue #5 guards the result shape
    // when the first operand is not the one that supplies it.
    if let Some(ref fname) = func_name
        && builtins::COMPARISON_OPS.contains(&fname.as_str())
    {
        checked_route_observed = true;
        let tensor_dims = arg_tys.iter().find_map(|t| match subst.apply(t) {
            Type::Tensor(dims, _) => Some(dims),
            _ => None,
        });
        if let Some(dims) = tensor_dims {
            // chelis#1265: route the result through unification rather than
            // constructing it out of band. A consumer that supplies a shape
            // must reach this call's result variable, or a declared shape
            // simply binds a free variable and selects nothing.
            let result = Type::Tensor(dims, TensorPrec::Concrete(Prim::Bool));
            if let Err(error) = unify(&ret_tv, &result, subst) {
                return report(errors, error.into());
            }
            return subst.apply(&ret_tv);
        }
        // No tensor arg → scalar comparison, returns scalar bool. [05-OP-36]'s
        // recursive equality returns one scalar bool as well.
        if let Some(first_arg) = arg_tys.first() {
            let resolved_arg = type_for_readonly_check(first_arg, subst);
            let recursive_equality = matches!(fname.as_str(), "eq" | "neq")
                && matches!(
                    resolved_arg,
                    Type::Unit | Type::Tuple(_) | Type::Adt(..) | Type::KindedAdt(..)
                );
            if matches!(resolved_arg, Type::Prim(_)) || recursive_equality {
                let result = Type::Prim(Prim::Bool);
                if let Err(error) = unify(&ret_tv, &result, subst) {
                    return report(errors, error.into());
                }
                return subst.apply(&ret_tv);
            }
        }
    }

    if let Some(ref fname) = func_name {
        let site =
            UnresolvedOperandSite::new(node, kids, fname.as_str(), env).with_site(source_site);
        match fname.as_str() {
            "print" => {
                return Type::Unit;
            }
            "fail" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Error(_) => {
                            return Type::Var(vg.fresh_tvar());
                        }
                        Type::Var(_) => {
                            return site.defer(
                                &arg_tys,
                                &result_ty,
                                product,
                                Type::Var(vg.fresh_tvar()),
                            );
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("fail expects string input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "debug" => {
                if let Some(first_arg) = arg_tys.first() {
                    return subst.apply(first_arg);
                }
            }
            name if string_route_owns(name) => {
                // chelis#1512: the string-operand group lives in `app_string.rs`.
                // `None` means it decided nothing, which for a name it owns
                // happens only on an empty argument list, so the generic path
                // below runs exactly as it did before the move.
                if let Some(result) = string_route_result(
                    name, node, &arg_tys, &result_ty, &site, product, subst, errors,
                ) {
                    return result;
                }
            }
            "to_string" => {
                return Type::Prim(Prim::String);
            }
            "rank" => {
                if let Some(first_arg) = arg_tys.first() {
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(_, _) | Type::Error(_) => {
                            return Type::Prim(Prim::Int32);
                        }
                        Type::Var(_) => {
                            return site.defer(
                                &arg_tys,
                                &result_ty,
                                product,
                                Type::Prim(Prim::Int32),
                            );
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("rank expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "shape" => {
                let input_dims = if let Some(first_arg) = arg_tys.first() {
                    // A read-only operand may carry the tensor as `Ref<Var>`.
                    // `type_for_readonly_check` peels that wrapper so borrowed
                    // and unborrowed reads use the same rule before axis
                    // validation.
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(dims, _) => Some(dims),
                        Type::Error(_) => None,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("shape expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                } else {
                    None
                };
                // Issue #216: cast-aware so `shape(x, cast(N, i32))`
                // (the idiomatic form from issue #206 for runtime-dim
                // reshape) surfaces the same diagnostic as the bare-
                // literal form.
                if let Some(axis_expr) = kids.get(2)
                    && let Some(axis) = extract_int_for_dim(axis_expr)
                {
                    if axis < 0 {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::DimensionMismatch,
                                with_node_provenance(
                                    node,
                                    format!("shape requires non-negative axis, got {axis}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                    if let Some(dims) = input_dims.as_ref() {
                        let axis = axis as usize;
                        if axis >= dims.len() {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::DimensionMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "shape axis {axis} is out of bounds for rank {} tensor",
                                            dims.len()
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
                if let Some(axis_arg) = arg_tys.get(1) {
                    match subst.apply(axis_arg) {
                        // [05-DIM-2]: extent-domain out (i64), axis-domain
                        // in (i32).
                        Type::Prim(Prim::Int32) | Type::Error(_) => {
                            return Type::Prim(Prim::Int64);
                        }
                        Type::Var(_) => {
                            return site.defer(
                                &arg_tys,
                                &result_ty,
                                product,
                                Type::Prim(Prim::Int64),
                            );
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("shape expects i32 axis, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "numel" => {
                if let Some(first_arg) = arg_tys.first() {
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(_, _) | Type::Error(_) => {
                            return Type::Prim(Prim::Int64);
                        }
                        Type::Var(_) => {
                            return site.defer(
                                &arg_tys,
                                &result_ty,
                                product,
                                Type::Prim(Prim::Int64),
                            );
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("numel expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "tensor_to_scalar" => {
                if let Some(first_arg) = arg_tys.first() {
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(dims, precision) => {
                            if !dims.is_empty() {
                                return report_at_check_site(
                                    errors,
                                    CheckError::with_types(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "tensor_to_scalar argument 1: expected rank-0 tensor, got rank-{} tensor",
                                                dims.len()
                                            ),
                                        ),
                                        "rank-0 tensor".to_string(),
                                        format!("rank-{} tensor", dims.len()),
                                        vec![],
                                    ),
                                    site.source_site,
                                );
                            }
                            // Conversion changes the surface, not the dtype's
                            // identity or its declared family restriction.
                            return match precision {
                                TensorPrec::Concrete(p) => Type::Prim(p),
                                TensorPrec::Var(p) => Type::Var(p),
                            };
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report_at_check_site(
                                errors,
                                CheckError::with_types(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "tensor_to_scalar argument 1: expected tensor, got {other}"
                                        ),
                                    ),
                                    "tensor".to_string(),
                                    other.to_string(),
                                    vec![],
                                ),
                                site.source_site,
                            );
                        }
                    }
                }
            }
            "scalar_to_tensor" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(precision) if !matches!(precision, Prim::String) => {
                            return Type::Tensor(vec![], TensorPrec::Concrete(precision));
                        }
                        Type::Var(p) if subst.tvar_restriction(p).is_some() => {
                            return Type::Tensor(vec![], TensorPrec::Var(p));
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "scalar_to_tensor expects scalar numeric/bool input, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "einsum" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let Some(equation) = kids.get(1).and_then(extract_string_literal) else {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                "einsum expects a string equation as its first argument"
                                    .to_string(),
                            ),
                            vec![],
                        ),
                    );
                };
                if equation.contains("...") {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                "einsum ellipsis support is deferred in 3h".to_string(),
                            ),
                            vec![],
                        ),
                    );
                }
                return result_ty;
            }
            "gather" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                if let Err(err) = reject_non_int32_axis("gather", &arg_tys[2], node, subst, errors)
                {
                    return err;
                }
                let Some(raw_axis) = kids
                    .get(3)
                    .filter(|axis| gather_axis_has_integer_casts(axis))
                    .and_then(extract_int_for_dim)
                else {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            with_node_provenance(
                                node,
                                "gather axis must be an i32 integer constant (a literal or an integer-cast-wrapped literal); variables and helper calls cannot determine the result shape [05-AXIS-2]".to_string(),
                            ),
                            vec![],
                        ),
                    );
                };
                let route = ShapeRoute::Gather {
                    op: "gather".to_string(),
                    indices: Box::new(type_for_readonly_check(&arg_tys[1], subst)),
                    raw_axis: Some(raw_axis),
                    updates: None,
                    mode: None,
                };
                return decide_shape_route(
                    route,
                    &type_for_readonly_check(&arg_tys[0], subst),
                    node,
                    vg,
                    subst,
                    errors,
                );
            }
            "where" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let cond_ty = type_for_readonly_check(&arg_tys[0], subst);
                let then_ty = type_for_readonly_check(&arg_tys[1], subst);
                let else_ty = type_for_readonly_check(&arg_tys[2], subst);
                match (&cond_ty, &then_ty, &else_ty) {
                    (
                        Type::Tensor(cond_dims, TensorPrec::Concrete(Prim::Bool)),
                        Type::Tensor(then_dims, then_prec),
                        Type::Tensor(else_dims, else_prec),
                    ) => {
                        if then_prec != else_prec
                            || !elementwise_shapes_match(cond_dims, then_dims, subst)
                            || !elementwise_shapes_match(then_dims, else_dims, subst)
                        {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "where expects cond/both branches to have matching tensor shapes and branch precision, got {cond_ty}, {then_ty}, and {else_ty}"
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                        return Type::Tensor(then_dims.clone(), then_prec.clone());
                    }
                    // chelis#1512: an upstream failure keeps the early
                    // return, so the cascade still suppresses. Order matters:
                    // a triple carrying both must suppress, not suspend.
                    (Type::Error(_), _, _) | (_, Type::Error(_), _) | (_, _, Type::Error(_)) => {
                        return result_ty;
                    }
                    (Type::Var(_), _, _) | (_, Type::Var(_), _) | (_, _, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "where expects a bool tensor condition and matching tensor branches, got {cond_ty}, {then_ty}, and {else_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "cumsum" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let cumsum_operand = type_for_readonly_check(&arg_tys[0], subst);
                let _axis = match resolve_builtin_axis(
                    "cumsum",
                    kids.get(2),
                    &subst.apply(&arg_tys[1]),
                    &cumsum_operand,
                    node,
                    subst,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match cumsum_operand {
                    Type::Tensor(dims, precision) => {
                        return Type::Tensor(dims, precision);
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("cumsum expects tensor input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "diagonal" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let diagonal_operand = type_for_readonly_check(&arg_tys[0], subst);
                let axis1 = match resolve_axis_pair_member(
                    "diagonal",
                    kids.get(2),
                    &subst.apply(&arg_tys[1]),
                    &diagonal_operand,
                    0,
                    node,
                    subst,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                let axis2 = match resolve_axis_pair_member(
                    "diagonal",
                    kids.get(3),
                    &subst.apply(&arg_tys[2]),
                    &diagonal_operand,
                    1,
                    node,
                    subst,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match infer_diagonal_result_type(&diagonal_operand, axis1, axis2, subst) {
                    Ok(ty) => {
                        // chelis#1739: the result dim stays a wildcard for a
                        // mixed (symbolic, literal) pair, and a wildcard
                        // unifies with every declared extent. The literal is
                        // still an upper bound on the minimum, so compare the
                        // declared extent against it here, at the one place
                        // that knows both the operand's axes and the
                        // declaration.
                        if let Some(rejection) = reject_unreachable_diagonal_extent(
                            &diagonal_operand,
                            &ty,
                            axis1,
                            axis2,
                            node,
                            source_site,
                            env,
                            vg,
                            adt_reg,
                            subst,
                            errors,
                            expected_result,
                        ) {
                            return rejection;
                        }
                        return ty;
                    }
                    Err(message) => {
                        return report_at_check_site(
                            errors,
                            diagonal_result_error(node, &diagonal_operand, axis1, axis2, message),
                            source_site,
                        );
                    }
                }
            }
            "trace" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                for idx in [1usize, 2] {
                    if let Err(err) =
                        reject_non_int32_axis("trace", &arg_tys[idx], node, subst, errors)
                    {
                        return err;
                    }
                }
                let route = ShapeRoute::Trace {
                    raw_axis1: kids.get(2).and_then(extract_int_for_dim),
                    raw_axis2: kids.get(3).and_then(extract_int_for_dim),
                };
                return decide_shape_route(
                    route,
                    &type_for_readonly_check(&arg_tys[0], subst),
                    node,
                    vg,
                    subst,
                    errors,
                );
            }
            "clamp" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let input_ty = type_for_readonly_check(&arg_tys[0], subst);
                let low_ty = type_for_readonly_check(&arg_tys[1], subst);
                let high_ty = type_for_readonly_check(&arg_tys[2], subst);
                match (&input_ty, &low_ty, &high_ty) {
                    (
                        Type::Tensor(input_dims, input_prec),
                        Type::Tensor(low_dims, low_prec),
                        Type::Tensor(high_dims, high_prec),
                    ) => {
                        let low_ok = low_dims.is_empty()
                            || elementwise_shapes_match(low_dims, input_dims, subst);
                        let high_ok = high_dims.is_empty()
                            || elementwise_shapes_match(high_dims, input_dims, subst);
                        if low_ok && high_ok && low_prec == input_prec && high_prec == input_prec {
                            return input_ty;
                        }
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "clamp expects tensor input plus scalar-tensor or matching-shape tensor bounds of the same precision, got {input_ty}, {low_ty}, and {high_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                    // chelis#1512: an upstream failure keeps the early
                    // return, so the cascade still suppresses. Order matters:
                    // a triple carrying both must suppress, not suspend.
                    (Type::Error(_), _, _) | (_, Type::Error(_), _) | (_, _, Type::Error(_)) => {
                        return result_ty;
                    }
                    (Type::Var(_), _, _) | (_, Type::Var(_), _) | (_, _, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "clamp expects tensor input and tensor bounds, got {input_ty}, {low_ty}, and {high_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "sort" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let sort_operand = type_for_readonly_check(&arg_tys[0], subst);
                let _axis = match resolve_builtin_axis(
                    "sort",
                    kids.get(2),
                    &subst.apply(&arg_tys[1]),
                    &sort_operand,
                    node,
                    subst,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match sort_operand {
                    Type::Tensor(dims, precision) => {
                        return Type::Tuple(vec![
                            Type::Tensor(dims.clone(), precision),
                            Type::Tensor(dims, TensorPrec::Concrete(Prim::Int64)),
                        ]);
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("sort expects tensor input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "scatter" => {
                if arg_tys.len() != 5 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        5,
                        arg_tys.len(),
                    );
                }
                if let Err(err) = reject_non_int32_axis("scatter", &arg_tys[3], node, subst, errors)
                {
                    return err;
                }
                let route = ShapeRoute::Gather {
                    op: "scatter".to_string(),
                    indices: Box::new(subst.apply(&arg_tys[1])),
                    raw_axis: kids.get(4).and_then(extract_int_for_dim),
                    updates: Some(Box::new(subst.apply(&arg_tys[2]))),
                    mode: Some(
                        kids.get(5)
                            .and_then(extract_string_literal)
                            .unwrap_or_default(),
                    ),
                };
                return decide_shape_route(
                    route,
                    &subst.apply(&arg_tys[0]),
                    node,
                    vg,
                    subst,
                    errors,
                );
            }
            "scatter_replace" => {
                if arg_tys.len() != 4 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        4,
                        arg_tys.len(),
                    );
                }
                if let Err(err) =
                    reject_non_int32_axis("scatter_replace", &arg_tys[3], node, subst, errors)
                {
                    return err;
                }
                let route = ShapeRoute::Gather {
                    op: "scatter_replace".to_string(),
                    indices: Box::new(subst.apply(&arg_tys[1])),
                    raw_axis: kids.get(4).and_then(extract_int_for_dim),
                    updates: Some(Box::new(subst.apply(&arg_tys[2]))),
                    mode: None,
                };
                return decide_shape_route(
                    route,
                    &subst.apply(&arg_tys[0]),
                    node,
                    vg,
                    subst,
                    errors,
                );
            }
            "scatter_elements" => {
                return check_scatter_elements(node, kids, &arg_tys, result_ty, subst, errors);
            }
            "len" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, _) if name == "List" || name == "Dict" => {
                            return Type::Prim(Prim::Int64);
                        }
                        Type::Error(_) => return Type::Prim(Prim::Int64),
                        Type::Var(_) => {
                            return site.defer(
                                &arg_tys,
                                &result_ty,
                                product,
                                Type::Prim(Prim::Int64),
                            );
                        }
                        Type::Ref(inner) if matches!(&*inner, Type::Adt(name, _) if name == "List" || name == "Dict") =>
                        {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "len auto-borrows its List/Dict argument, so an explicit `&` is not a \
                                         supported surface form: write `len(xs)`, not `len(&xs)` (got &{inner})"
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("len expects List or Dict input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "index" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let list_arg = subst.apply(&arg_tys[0]);
                let index_arg = subst.apply(&arg_tys[1]);
                match &index_arg {
                    Type::Prim(Prim::Int64) => {}
                    // chelis#1512: not an i64 YET. Suspending the call
                    // re-enters this route once the operand binds, so this
                    // same guard decides against a settled type.
                    Type::Var(_) => site.register(&arg_tys, &result_ty, product),
                    Type::Error(_) => {}
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("index expects i64 index, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
                match list_arg {
                    Type::Adt(name, mut args) if name == "List" && args.len() == 1 => {
                        return args.remove(0);
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    Type::Ref(inner) if matches!(&*inner, Type::Adt(name, _) if name == "List") => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "index auto-borrows its List argument, so an explicit `&` is not a \
                                     supported surface form: write `index(xs, i)`, not `index(&xs, i)` (got &{inner})"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("index expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "concat" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let lhs = subst.apply(&arg_tys[0]);
                let rhs = subst.apply(&arg_tys[1]);
                let lhs_is_list =
                    matches!(&lhs, Type::Adt(name, args) if name == "List" && args.len() == 1);
                let rhs_is_list =
                    matches!(&rhs, Type::Adt(name, args) if name == "List" && args.len() == 1);
                // A right operand that is still a variable may be a list as
                // well as an axis, so it is not constrained to `i32` here; the
                // match below suspends the call on it.
                let rhs_is_unresolved = matches!(rhs, Type::Var(_));
                if lhs_is_list
                    && !rhs_is_list
                    && !rhs_is_unresolved
                    && let Err(err) = reject_non_int32_axis("concat", &rhs, node, subst, errors)
                {
                    return err;
                }
                match (lhs, rhs) {
                    (Type::Adt(lhs_name, lhs_args), Type::Prim(Prim::Int32))
                        if lhs_name == "List" && lhs_args.len() == 1 =>
                    {
                        // chelis#631/#594 (spec §4.5.4): the concat
                        // axis value and the statically-visible
                        // elements decide the concat-axis extent.
                        // kids[1] is the list expr, kids[2] the
                        // axis expr (cast-aware extraction, #216).
                        // A DIRECT literal chain consumes the element
                        // types already produced while inferring the
                        // list argument, so ragged extents can SUM
                        // without a second semantic traversal.
                        let evidence =
                            tensor_concat_call_evidence(kids, env, subst, errors, product);
                        match tensor_concat_result_type(
                            &lhs_args[0],
                            evidence.raw_axis,
                            evidence.list_info,
                            subst,
                        ) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(node, message),
                                        vec![],
                                    ),
                                );
                            }
                        }
                    }
                    (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                        if lhs_name == "List"
                            && rhs_name == "List"
                            && lhs_args.len() == 1
                            && rhs_args.len() == 1 =>
                    {
                        return publish_collection_equation(
                            &CollectionConstraint::Concat {
                                lhs: Type::Adt(lhs_name, lhs_args),
                                rhs: Type::Adt(rhs_name, rhs_args),
                                result: result_ty.clone(),
                            },
                            node,
                            subst,
                            errors,
                        )
                        .unwrap_or_else(|| {
                            site.defer(&arg_tys, &result_ty, product, result_ty.clone())
                        });
                    }
                    // chelis#1512: an upstream failure keeps the early
                    // return, so the cascade still suppresses. Order matters:
                    // an (Error, Var) pair must suppress, not suspend.
                    (Type::Error(_), _) | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (Type::Var(_), _) | (_, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    (lhs, rhs) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "concat expects matching List inputs, got {lhs} and {rhs}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "split" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let tensor_ty = type_for_readonly_check(&arg_tys[0], subst);
                let axis_ty = subst.apply(&arg_tys[1]);
                let sizes_ty = subst.apply(&arg_tys[2]);
                // The pre-guard predicate here was `precision.is_integer()`,
                // the same acceptance hole `concat` carried: it admitted an
                // i64 axis while `sum` rejected one.
                if let Err(err) = reject_non_int32_axis("split", &axis_ty, node, subst, errors) {
                    return err;
                }
                match (tensor_ty, sizes_ty) {
                    (Type::Tensor(dims, precision), Type::Adt(name, args))
                        if name == "List" && args.len() == 1 =>
                    {
                        match &args[0] {
                            Type::Prim(prec) if prec.is_integer() => {}
                            // chelis#1512: not an integer YET. Suspending the
                            // call re-enters this route once the element type
                            // binds, so this same guard decides against it.
                            Type::Var(_) => site.register(&arg_tys, &result_ty, product),
                            Type::Error(_) => {}
                            _ => {
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            "split expects List[int] sizes".to_string(),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                        }
                        // Negative axes index from the end.
                        // Issue #216: cast-aware so a
                        // `cast(N, i32)`-wrapped split axis still
                        // surfaces the bounds diagnostic at infer.
                        let raw_axis = kids.get(2).and_then(extract_int_for_dim);
                        let axis = match raw_axis {
                            Some(raw) => match normalize_static_axis(dims.len(), raw) {
                                Some(axis) => axis,
                                None => {
                                    return report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_node_provenance(
                                                node,
                                                format!(
                                                    "split axis {raw} out of bounds for rank {}",
                                                    dims.len()
                                                ),
                                            ),
                                            vec![],
                                        ),
                                    );
                                }
                            },
                            None => 0,
                        };
                        let mut piece_dims = dims.clone();
                        piece_dims[axis] = Dim::Wildcard;
                        return Type::Adt(
                            "List".to_string(),
                            vec![Type::Tensor(piece_dims, precision)],
                        );
                    }
                    // chelis#1512: an upstream failure keeps the early
                    // return, so the cascade still suppresses. Order matters:
                    // an (Error, Var) pair must suppress, not suspend.
                    (Type::Error(_), _) | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (Type::Var(_), _) | (_, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    (tensor_ty, sizes_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "split expects tensor input and List[int] sizes, got {tensor_ty} and {sizes_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "take" | "skip" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let op_name = func_name.as_deref().unwrap_or("collection helper");
                let list_arg = subst.apply(&arg_tys[0]);
                let count_arg = subst.apply(&arg_tys[1]);
                match &count_arg {
                    Type::Prim(prec) if prec.is_integer() => {}
                    // chelis#1512: not an integer YET. Suspending the call
                    // re-enters this route once the operand binds, so this
                    // same guard decides against a settled type.
                    Type::Var(_) => site.register(&arg_tys, &result_ty, product),
                    Type::Error(_) => {}
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("{op_name} expects integer count, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
                match list_arg {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        return Type::Adt("List".to_string(), vec![args[0].clone()]);
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("{op_name} expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "chunk" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let list_arg = subst.apply(&arg_tys[0]);
                let count_arg = subst.apply(&arg_tys[1]);
                match &count_arg {
                    Type::Prim(prec) if prec.is_integer() => {}
                    // chelis#1512: not an integer YET. Suspending the call
                    // re-enters this route once the operand binds, so this
                    // same guard decides against a settled type.
                    Type::Var(_) => site.register(&arg_tys, &result_ty, product),
                    Type::Error(_) => {}
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("chunk expects integer size, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
                match list_arg {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        return Type::Adt(
                            "List".to_string(),
                            vec![Type::Adt("List".to_string(), vec![args[0].clone()])],
                        );
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("chunk expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "range" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                for arg_ty in &arg_tys {
                    match subst.apply(arg_ty) {
                        Type::Prim(prec) if prec.is_integer() => {}
                        Type::Error(_) => {}
                        Type::Var(_) => site.register(&arg_tys, &result_ty, product),
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("range expects integer arguments, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
                return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
            }
            "map" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let elem_ty = vg.fresh_type();
                let out_ty = vg.fresh_type();
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(vec![elem_ty.clone()], Box::new(out_ty.clone())),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[1]),
                    &Type::Adt("List".to_string(), vec![elem_ty]),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                return Type::Adt("List".to_string(), vec![subst.apply(&out_ty)]);
            }
            "filter" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let elem_ty = vg.fresh_type();
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            node,
                            "filter",
                            "expects a callback that returns bool",
                            te,
                        ),
                    );
                }
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[1]),
                    &Type::Adt("List".to_string(), vec![elem_ty.clone()]),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                return Type::Adt("List".to_string(), vec![subst.apply(&elem_ty)]);
            }
            "tensor_scan" => {
                // `tensor_scan(initial: T, fn: (T, i64) -> T, n: i64) -> tensor[n, T]`.
                //
                // Issue #257: host-runtime scan that produces a tensor
                // directly, sidestepping the right-recursive list build
                // that overflows the worker stack at ~10k elements.
                // Element type `T` must resolve to a concrete scalar
                // Prim before tensor lowering; the runtime arm enforces
                // that at execution time. At type-check time we accept
                // any Type::Prim and let unification do the rest.
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let elem_ty = vg.fresh_type();
                let i64 = Type::Prim(Prim::Int64);
                // arg 0: initial accumulator of type T.
                if let Err(te) = unify(&subst.apply(&arg_tys[0]), &elem_ty.clone(), subst) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            node,
                            "tensor_scan",
                            "expects an initial value whose type matches the callback element type",
                            te,
                        ),
                    );
                }
                // arg 1: callback `(T, i64) -> T`.
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[1]),
                    &Type::Fn(
                        vec![elem_ty.clone(), i64.clone()],
                        Box::new(elem_ty.clone()),
                    ),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            node,
                            "tensor_scan",
                            "expects a callback (T, i64) -> T",
                            te,
                        ),
                    );
                }
                // arg 2: length `n: i64`.
                if let Err(te) = unify(&subst.apply(&arg_tys[2]), &i64, subst) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            node,
                            "tensor_scan",
                            "expects a length `n: i64`",
                            te,
                        ),
                    );
                }
                // Element type must be a concrete scalar Prim once
                // unified. If it's still a Var the call site is
                // under-constrained; if it's a Tensor/Adt/Fn the call
                // is invalid. We only allow primitive scalars so the
                // host-runtime arm can determine precision.
                let resolved_elem = subst.apply(&elem_ty);
                let precision = match &resolved_elem {
                    Type::Prim(p) => TensorPrec::Concrete(*p),
                    Type::Var(tv) => {
                        // Defer: leave the precision as the same type
                        // variable as the element. `Subst::apply` will
                        // resolve it once outer inference pins T.
                        // Using F32 as a placeholder (the previous
                        // behavior) silently lies about the dtype
                        // when T is later pinned to i64 or bool.
                        TensorPrec::Var(*tv)
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "tensor_scan element type must be a scalar primitive, got {other}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                };
                return Type::Tensor(vec![Dim::Wildcard], precision);
            }
            "partition" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let elem_ty = vg.fresh_type();
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            node,
                            "partition",
                            "expects a callback that returns bool",
                            te,
                        ),
                    );
                }
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[1]),
                    &Type::Adt("List".to_string(), vec![elem_ty.clone()]),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                let out_list = Type::Adt("List".to_string(), vec![subst.apply(&elem_ty)]);
                return Type::Tuple(vec![out_list.clone(), out_list]);
            }
            "flat_map" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let elem_ty = vg.fresh_type();
                let out_elem_ty = vg.fresh_type();
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(
                        vec![elem_ty.clone()],
                        Box::new(Type::Adt("List".to_string(), vec![out_elem_ty.clone()])),
                    ),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[1]),
                    &Type::Adt("List".to_string(), vec![elem_ty]),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                return Type::Adt("List".to_string(), vec![subst.apply(&out_elem_ty)]);
            }
            "flatten" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(outer_name, outer_args)
                            if outer_name == "List" && outer_args.len() == 1 =>
                        {
                            match &outer_args[0] {
                                Type::Adt(inner_name, inner_args)
                                    if inner_name == "List" && inner_args.len() == 1 =>
                                {
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![inner_args[0].clone()],
                                    );
                                }
                                Type::Error(_) => return result_ty,
                                Type::Var(_) => {
                                    return site.defer(
                                        &arg_tys,
                                        &result_ty,
                                        product,
                                        result_ty.clone(),
                                    );
                                }
                                other => {
                                    return report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_node_provenance(
                                                node,
                                                format!(
                                                    "flatten expects List[List[T]] input, got List[{other}]"
                                                ),
                                            ),
                                            vec![],
                                        ),
                                    );
                                }
                            }
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("flatten expects List[List[T]] input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "zip" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let lhs = subst.apply(&arg_tys[0]);
                let rhs = subst.apply(&arg_tys[1]);
                match (lhs, rhs) {
                    (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                        if lhs_name == "List"
                            && rhs_name == "List"
                            && lhs_args.len() == 1
                            && rhs_args.len() == 1 =>
                    {
                        return Type::Adt(
                            "List".to_string(),
                            vec![Type::Tuple(vec![lhs_args[0].clone(), rhs_args[0].clone()])],
                        );
                    }
                    // chelis#1512: an upstream failure keeps the early
                    // return, so the cascade still suppresses. Order matters:
                    // an (Error, Var) pair must suppress, not suspend.
                    (Type::Error(_), _) | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (Type::Var(_), _) | (_, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    (lhs, rhs) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("zip expects List inputs, got {lhs} and {rhs}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "enumerate" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                            return Type::Adt(
                                "List".to_string(),
                                vec![Type::Tuple(vec![Type::Prim(Prim::Int64), args[0].clone()])],
                            );
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("enumerate expects List input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "dict_of" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                            // chelis#2523: `dict_of`'s input is a list of
                            // `(key, value)` pairs, so an element whose type is
                            // still a variable is constrained to a pair rather
                            // than admitted. That ties the result's key and
                            // value types to the list's element, so a declared
                            // result determines them (`dict_of([])`), and the
                            // key rule below decides once the key is known.
                            if let Type::Var(_) = subst.apply(&args[0]) {
                                let pair = Type::Tuple(vec![vg.fresh_type(), vg.fresh_type()]);
                                if let Err(te) = unify(&args[0], &pair, subst) {
                                    return report(errors, te.into());
                                }
                            }
                            match subst.apply(&args[0]) {
                                Type::Tuple(items) if items.len() == 2 => {
                                    let dict = Type::Adt(
                                        "Dict".to_string(),
                                        vec![items[0].clone(), items[1].clone()],
                                    );
                                    match &items[0] {
                                        key if is_dict_key_type(key) => {}
                                        Type::Error(_) => return result_ty,
                                        Type::Var(_) => {
                                            return site.defer(&arg_tys, &result_ty, product, dict);
                                        }
                                        other => {
                                            return report(
                                                errors,
                                                CheckError::new(
                                                    CheckErrorKind::TypeMismatch,
                                                    with_node_provenance(
                                                        node,
                                                        format!(
                                                            "dict_of keys must be string, bool, or a signed integer scalar ([05-OP-56]), got {other}"
                                                        ),
                                                    ),
                                                    vec![],
                                                ),
                                            );
                                        }
                                    }
                                    return dict;
                                }
                                Type::Error(_) => return result_ty,
                                other => {
                                    return report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_node_provenance(
                                                node,
                                                format!(
                                                    "dict_of expects List[(K, V)] input, got List[{other}]"
                                                ),
                                            ),
                                            vec![],
                                        ),
                                    );
                                }
                            }
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("dict_of expects List input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "dict_get" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                    (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                        if let Err(te) = unify(&args[0], &key_ty, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt("Option".to_string(), vec![subst.apply(&args[1])]);
                    }
                    (Type::Error(_), _) | (_, Type::Error(_)) => return result_ty,
                    // chelis#2523: not a `Dict` YET. Suspending re-enters this
                    // route once the operand binds, and decides it at its
                    // instantiations if it never does.
                    (Type::Var(_), _) | (_, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    (dict_ty, key_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "dict_get expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "dict_contains" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                    (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                        if let Err(te) = unify(&args[0], &key_ty, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Prim(Prim::Bool);
                    }
                    (Type::Error(_), _) | (_, Type::Error(_)) => return result_ty,
                    // chelis#2523: not a `Dict` YET. Suspending re-enters this
                    // route once the operand binds, and decides it at its
                    // instantiations if it never does.
                    (Type::Var(_), _) | (_, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    (dict_ty, key_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "dict_contains expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "dict_remove" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                    (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                        if let Err(te) = unify(&args[0], &key_ty, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt(
                            "Dict".to_string(),
                            vec![subst.apply(&args[0]), subst.apply(&args[1])],
                        );
                    }
                    (Type::Error(_), _) | (_, Type::Error(_)) => return result_ty,
                    // chelis#2523: not a `Dict` YET. Suspending re-enters this
                    // route once the operand binds, and decides it at its
                    // instantiations if it never does.
                    (Type::Var(_), _) | (_, Type::Var(_)) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    (dict_ty, key_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "dict_remove expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "dict_keys" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                            return Type::Adt("List".to_string(), vec![args[0].clone()]);
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("dict_keys expects Dict input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "dict_values" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                            return Type::Adt("List".to_string(), vec![args[1].clone()]);
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("dict_values expects Dict input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "dict_entries" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                            return Type::Adt(
                                "List".to_string(),
                                vec![Type::Tuple(vec![args[0].clone(), args[1].clone()])],
                            );
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("dict_entries expects Dict input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "to_tensor" => {
                if let Some(first_arg) = arg_tys.first() {
                    // Bucket 4b: support arbitrarily-nested numeric/bool
                    // lists. Each enclosing `List` adds one outer
                    // dimension, and the innermost element type must
                    // be a numeric or bool primitive.
                    //
                    // Issue Chelis-Lang/chelis#218 (R2 HIGH-A from
                    // PR #211): when the argument is a statically-
                    // resolvable Cons-chain literal, emit concrete
                    // `Dim::Lit(n)` per axis instead of wildcards.
                    // The wildcard fallback only fires when the
                    // argument is variable-fed (e.g.
                    // `to_tensor(items)`), where the shape is
                    // genuinely unknown at type-check time. Emitting
                    // concrete dims at this single source point
                    // means every downstream consumer (reductions,
                    // elementwise activations, anything that reads
                    // the to_tensor app's `type:` metadata) sees a
                    // sound shape instead of `Dim::Wildcard`.
                    let resolved = subst.apply(first_arg);
                    if matches!(resolved, Type::Error(_)) {
                        return result_ty;
                    }
                    if matches!(resolved, Type::Var(_)) {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    match peel_to_tensor_argument(&resolved) {
                        ToTensorPeel::Ok { rank, precision } => {
                            if rank == 0 {
                                // Defensive: a bare scalar should never
                                // hit this branch (the typer requires
                                // a `List` head), but guard anyway.
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!("to_tensor expects List input, got {resolved}"),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                            // R2 HIGH-A: try the static-shape walker
                            // on the actual argument expression
                            // first. `kids[0]` is the `(var
                            // to_tensor)` callee; `kids[1]` is the
                            // argument expression. If the walker
                            // can't resolve a uniform shape
                            // (variable-fed argument, ragged
                            // literal, or unrecognized leaf), fall
                            // back to the legacy wildcard rank.
                            let dims = kids
                                .get(1)
                                .and_then(|arg| static_to_tensor_shape(arg, rank))
                                .unwrap_or_else(|| vec![Dim::Wildcard; rank]);
                            return Type::Tensor(dims, TensorPrec::Concrete(precision));
                        }
                        // [05-OP-57]: the leaf of `to_tensor`'s List is one
                        // scalar tensor-element dtype. A leaf variable restricted
                        // to the `Float`, `Int` or `Numeric` family can only be
                        // such a scalar, so the nesting depth already fixes the
                        // rank and the variable is the result precision, exactly
                        // as `to_list` maps `tensor[n, p]` to `List[p]`. Returning
                        // the scheme's opaque result here left every consumer of
                        // a generic `to_tensor` (a `reshape`, say) suspended past
                        // its declaration boundary.
                        // Any dtype bound of spec/04 §5.9 resolves the element,
                        // including the explicit set form: a set restricts WHICH
                        // dtypes are admissible, never which programs type-check
                        // (chelis#2443). Listing only the three families left a
                        // set-bounded `to_tensor` suspended, so `[p: {f32, f64}]`
                        // was rejected where the equivalent `Float` was accepted.
                        ToTensorPeel::Pending { rank, element }
                            if rank > 0
                                && subst.tvar_restriction(element).is_some_and(|restriction| {
                                    !restriction.is_value_constraint()
                                }) =>
                        {
                            let dims = kids
                                .get(1)
                                .and_then(|arg| static_to_tensor_shape(arg, rank))
                                .unwrap_or_else(|| vec![Dim::Wildcard; rank]);
                            return Type::Tensor(dims, TensorPrec::Var(element));
                        }
                        // chelis#2523: `to_tensor` maps `List` nested `r`
                        // deep around a numeric or bool dtype `p` to a rank-`r`
                        // tensor of `p`, so an element type that is still a
                        // variable is determined by a result whose rank and
                        // precision are known, as a declared field does for
                        // `IntCol(to_tensor([]))`. Otherwise the call waits for
                        // either one to bind, and the declaration boundary
                        // decides it if neither does.
                        ToTensorPeel::Pending { rank, element } => {
                            if let Type::Tensor(result_dims, precision) = &result_ty
                                && result_dims.len() >= rank
                            {
                                let scalar = match precision {
                                    TensorPrec::Concrete(prim) => Type::Prim(*prim),
                                    TensorPrec::Var(var) => Type::Var(*var),
                                };
                                let inner = (rank..result_dims.len()).fold(scalar, |inner, _| {
                                    Type::Adt("List".to_string(), vec![inner])
                                });
                                if let Err(te) = unify(&Type::Var(element), &inner, subst) {
                                    return report(errors, te.into());
                                }
                                // A concrete precision settles the call; a
                                // precision variable is decided when it binds,
                                // or at its instantiations.
                                if let TensorPrec::Concrete(prim) = precision
                                    && (prim.is_numeric() || *prim == Prim::Bool)
                                {
                                    return result_ty;
                                }
                            }
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        ToTensorPeel::Poisoned => return result_ty,
                        ToTensorPeel::BadInner(inner) => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "to_tensor expects numeric or bool elements at the innermost level, got {inner}"
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                        ToTensorPeel::NotList => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("to_tensor expects List input, got {resolved}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "to_list" => {
                if let Some(first_arg) = arg_tys.first() {
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(dims, precision) => {
                            if dims.len() != 1 {
                                return report_at_check_site(
                                    errors,
                                    CheckError::with_types(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "to_list argument 1: expected rank-1 tensor, got rank-{} tensor",
                                                dims.len()
                                            ),
                                        ),
                                        "rank-1 tensor".to_string(),
                                        format!("rank-{} tensor", dims.len()),
                                        vec![],
                                    ),
                                    site.source_site,
                                );
                            }
                            let precision = match precision {
                                TensorPrec::Concrete(p) => p,
                                // [05-OP-57] fixes the result constructor and
                                // leaf relation independently of whether the
                                // authored precision has been instantiated:
                                // tensor[n, p] becomes List[p]. Returning the
                                // builtin scheme's opaque result variable here
                                // erased the known List constructor and made a
                                // following index look unresolved at [04-INF-9]'s
                                // declaration boundary (chelis#2126).
                                TensorPrec::Var(variable) => {
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Var(variable)],
                                    );
                                }
                            };
                            if !precision.is_numeric() && !matches!(precision, Prim::Bool) {
                                return report_at_check_site(
                                    errors,
                                    CheckError::with_types(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "to_list argument 1: expected numeric or bool tensor, got tensor[{}, {}]",
                                                dims[0],
                                                precision.name()
                                            ),
                                        ),
                                        "numeric or bool tensor".to_string(),
                                        format!("tensor[{}, {}]", dims[0], precision.name()),
                                        vec![],
                                    ),
                                    site.source_site,
                                );
                            }
                            return Type::Adt("List".to_string(), vec![Type::Prim(precision)]);
                        }
                        Type::Error(_) => return result_ty,
                        Type::Var(_) => {
                            return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                        }
                        other => {
                            return report_at_check_site(
                                errors,
                                CheckError::with_types(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("to_list argument 1: expected tensor, got {other}"),
                                    ),
                                    "tensor".to_string(),
                                    other.to_string(),
                                    vec![],
                                ),
                                site.source_site,
                            );
                        }
                    }
                }
            }
            "pad_sequences" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        2,
                        arg_tys.len(),
                    );
                }
                let seqs_ty = subst.apply(&arg_tys[0]);
                let pad_ty = subst.apply(&arg_tys[1]);
                match seqs_ty {
                    Type::Adt(outer_name, outer_args)
                        if outer_name == "List" && outer_args.len() == 1 =>
                    {
                        match &outer_args[0] {
                            Type::Adt(inner_name, inner_args)
                                if inner_name == "List" && inner_args.len() == 1 =>
                            {
                                if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                    return report(errors, te.into());
                                }
                                match subst.apply(&inner_args[0]) {
                                    Type::Prim(precision) if precision.is_numeric() => {
                                        return Type::Tensor(
                                            vec![Dim::Wildcard, Dim::Wildcard],
                                            TensorPrec::Concrete(precision),
                                        );
                                    }
                                    Type::Error(_) => return result_ty,
                                    Type::Var(_) => {
                                        return site.defer(
                                            &arg_tys,
                                            &result_ty,
                                            product,
                                            result_ty.clone(),
                                        );
                                    }
                                    other => {
                                        return report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_node_provenance(
                                                    node,
                                                    format!(
                                                        "pad_sequences expects numeric nested lists, got {other}"
                                                    ),
                                                ),
                                                vec![],
                                            ),
                                        );
                                    }
                                }
                            }
                            other => {
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "pad_sequences expects List[List[T]], got List[{other}]"
                                            ),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                        }
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "pad_sequences expects List[List[T]] input, got {other}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "pad_sequences_to" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(
                        errors,
                        node,
                        source_site,
                        fname,
                        3,
                        arg_tys.len(),
                    );
                }
                let seqs_ty = subst.apply(&arg_tys[0]);
                let width_ty = subst.apply(&arg_tys[1]);
                let pad_ty = subst.apply(&arg_tys[2]);
                if let Err(te) = unify(&width_ty, &Type::Prim(Prim::Int64), subst) {
                    return report(errors, te.into());
                }
                // The padded (axis-1) dimension equals the `width`
                // argument. When `width` is a literal — including
                // `cast(N, i64)`, the form every caller uses —
                // propagate `Dim::Lit(N)` so the padded width is a
                // concrete dim that participates in shape checking.
                // A non-literal or non-positive width stays
                // `Dim::Wildcard` (the runtime validates the value).
                // `extract_int_for_dim` (not `extract_int_literal`)
                // is the cast-aware extractor used for dim contexts.
                let width_dim = node
                    .children_slice()
                    .get(2)
                    .and_then(extract_int_for_dim)
                    .filter(|width| *width > 0)
                    .map_or(Dim::Wildcard, Dim::Lit);
                match seqs_ty {
                    Type::Adt(outer_name, outer_args)
                        if outer_name == "List" && outer_args.len() == 1 =>
                    {
                        match &outer_args[0] {
                            Type::Adt(inner_name, inner_args)
                                if inner_name == "List" && inner_args.len() == 1 =>
                            {
                                if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                    return report(errors, te.into());
                                }
                                match subst.apply(&inner_args[0]) {
                                    Type::Prim(precision) if precision.is_numeric() => {
                                        return Type::Tensor(
                                            vec![Dim::Wildcard, width_dim],
                                            TensorPrec::Concrete(precision),
                                        );
                                    }
                                    Type::Error(_) => return result_ty,
                                    Type::Var(_) => {
                                        return site.defer(
                                            &arg_tys,
                                            &result_ty,
                                            product,
                                            result_ty.clone(),
                                        );
                                    }
                                    other => {
                                        return report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_node_provenance(
                                                    node,
                                                    format!(
                                                        "pad_sequences_to expects numeric nested lists, got {other}"
                                                    ),
                                                ),
                                                vec![],
                                            ),
                                        );
                                    }
                                }
                            }
                            other => {
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "pad_sequences_to expects List[List[T]], got List[{other}]"
                                            ),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                        }
                    }
                    Type::Error(_) => return result_ty,
                    Type::Var(_) => {
                        return site.defer(&arg_tys, &result_ty, product, result_ty.clone());
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "pad_sequences_to expects List[List[T]] input, got {other}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "read_file" => return Type::Prim(Prim::String),
            "write_file" => return Type::Unit,
            "read_lines" => {
                return Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]);
            }
            "read_bytes" => {
                return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
            }
            "file_exists" => return Type::Prim(Prim::Bool),
            "list_dir" => {
                return Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]);
            }
            "mmap_file" => return Type::Adt("MappedFile".to_string(), Vec::new()),
            "mmap_read" => {
                return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
            }
            "mmap_len" => return Type::Prim(Prim::Int64),
            // Hull Phase 0a: `process_run(cmd, args)` returns
            // `(exit_code, stdout, stderr)`. Eval/test-only; the build
            // backends reject it (see `reject_eval_only_builtins_host`).
            "process_run" => {
                return Type::Tuple(vec![
                    Type::Prim(Prim::Int64),
                    Type::Prim(Prim::String),
                    Type::Prim(Prim::String),
                ]);
            }
            "round_to" => {
                return check_round_to_builtin_signature(node, &arg_tys, subst, errors);
            }
            // Host-lane CSV I/O (chelis#903): parse/serialize plus column
            // accessors over the canonical List[Dict[string,string]] table.
            "parse_csv" | "to_csv" | "csv_f64s" | "csv_ints" | "csv_strs" | "csv_nrows"
            | "csv_cols" | "csv_f64" | "csv_int" | "csv_str" => {
                return check_csv_builtin_signature(fname, node, &arg_tys, subst, errors);
            }
            _ => {}
        }
    }

    if let Some(rule) = checked_rule
        && !checked_route_observed
    {
        return report(
            errors,
            unobserved_checked_route_diagnostic(func_name.as_deref().unwrap_or("<unknown>"), rule),
        );
    }

    product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
    subst.apply(&result_ty)
}
