//! Operation-family checks after generic application unification.
//!
//! The checks remain in their original order. Each rejection still returns
//! before the next operation family runs.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_unified_app(
    list: &deep::List,
    kids: &[deep::Expr],
    func_name: Option<String>,
    arg_tys: Vec<Type>,
    ret_tv: Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let mut result_ty = subst.apply(&ret_tv);

    if let Some(rejected) =
        validate_numeric_and_reduction_arguments(list, kids, &func_name, &arg_tys, subst, errors)
    {
        return rejected;
    }

    if let Some(ref fname) = func_name
        && fname == "uniform_like"
    {
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, prim) if prim.is_float() => {}
                Type::Tensor(_, _) => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "uniform_like expects a float tensor template, got {}",
                                    resolved
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
                Type::Var(_) | Type::Error(_) => {}
                _ => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "uniform_like expects tensor template input, got {}",
                                    resolved
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }

        for (index, arg_ty) in arg_tys.iter().enumerate().skip(1).take(2) {
            let resolved = type_for_readonly_check(arg_ty, subst);
            match &resolved {
                Type::Prim(Prim::F32) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "uniform_like expects f32 bounds for args 2-3, got {}",
                                    resolved
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
            }

            if let Some(expr) = kids.get(index + 1)
                && !is_static_numeric_bound(expr)
            {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            "uniform_like currently requires literal low/high bounds \
                         (a numeric literal, optionally negated or cast to a float \
                         type); a runtime-computed bound is not supported"
                                .to_string(),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }

    if let Some(ref fname) = func_name
        && fname == "dropout"
    {
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("dropout expects tensor input, got {}", resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }

        if let Some(rate_arg) = arg_tys.get(1) {
            let resolved = subst.apply(rate_arg);
            match &resolved {
                Type::Prim(Prim::F32) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("dropout expects f32 rate, got {}", resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }
    }

    if let Some(ref fname) = func_name
        && fname == "conv2d"
    {
        for (index, arg_ty) in arg_tys.iter().enumerate() {
            let resolved = type_for_readonly_check(arg_ty, subst);
            if index < 2 {
                match &resolved {
                    Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {}
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "conv2d expects tensor inputs for args 1-2, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            } else {
                match &resolved {
                    Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error(_) => {}
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "conv2d expects int32 stride/padding, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
        }
    }

    if let Some(ref fname) = func_name
        && INT_BINOPS.contains(&fname.as_str())
    {
        let lhs = arg_tys
            .first()
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        let rhs = arg_tys
            .get(1)
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        match (&lhs, &rhs) {
            (Type::Prim(lhs_prec), Type::Prim(rhs_prec))
                if lhs_prec.is_integer() && rhs_prec.is_integer() && lhs_prec == rhs_prec =>
            {
                return Type::Prim(*lhs_prec);
            }
            (Type::Var(_), Type::Prim(rhs_prec)) if rhs_prec.is_integer() => {
                return lhs;
            }
            (Type::Prim(lhs_prec), Type::Var(_)) if lhs_prec.is_integer() => {
                return lhs;
            }
            (Type::Var(_), Type::Var(_)) | (Type::Error(_), _) | (_, Type::Error(_)) => {
                return lhs;
            }
            _ => {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!(
                                "{} requires matching integer arguments, got {} and {}",
                                fname, lhs, rhs
                            ),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }

    if let Some(ref fname) = func_name
        && INT_SHIFT_OPS.contains(&fname.as_str())
    {
        let lhs = arg_tys
            .first()
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        let rhs = arg_tys
            .get(1)
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        let lhs_ok = matches!(&lhs, Type::Prim(prec) if prec.is_integer())
            || matches!(&lhs, Type::Var(_) | Type::Error(_));
        let rhs_ok = matches!(&rhs, Type::Prim(prec) if prec.is_integer())
            || matches!(&rhs, Type::Var(_) | Type::Error(_));
        if lhs_ok && rhs_ok {
            return lhs;
        }
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "{} requires integer lhs and shift amount, got {} and {}",
                        fname, lhs, rhs
                    ),
                ),
                vec![],
            ),
        );
    }

    // chelis#778 follow-up: a shape-computed builtin override derives
    // its result shape from operand shapes and returns
    // `subst.apply(ret_tv)`. When an operand type is `Error`, the
    // Error-permissive per-slot unify above (`(Error, _) => Ok(())`)
    // never bound `ret_tv`, so the override would leak an unbound
    // `Var` as the call's result type. That bare `Var` would be
    // degraded to a rank-0 default `TensorType` during writeback and
    // CLOBBER the node's concrete type annotation, ICEing the IR
    // lowering (`conv2d output height axis requires a statically known
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
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
                | "mean"
                | "expand"
                | "layer_norm"
                | "conv2d"
        ) && arg_tys
            .iter()
            .any(|ty| matches!(subst.apply(ty), Type::Error(_)))
    });
    if shape_override_operand_error {
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
        match fname.as_str() {
            "matmul" => {
                result_ty = check_matmul_signature(&arg_tys, &result_ty, subst, errors);
            }
            "sum" | "max_reduce" | "min_reduce" | "prod_reduce" | "argmax_reduce"
            | "argmin_reduce" | "mean" => {
                result_ty = check_reduction_signature(
                    fname,
                    &kids[1..],
                    &arg_tys,
                    &result_ty,
                    subst,
                    errors,
                );
            }
            "expand" => {
                // chelis#339: the axis slot is a dim NAME (the
                // named-axis insert form) only when it is not bound in
                // the value environment — a bound `int32` var is the
                // issue #259 runtime-value class instead.
                let axis_is_dim_name = kids.get(2).is_some_and(|arg| {
                    symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none())
                });
                // chelis#397/#469: classify the size by PROVENANCE
                // (static / shape-sourced / sourceless), following
                // `let`/`cast`/arithmetic to a tensor shape source.
                // A truly sourceless runtime scalar is rejected at
                // check so it never reaches the build/eval-only
                // rejection (a check-clean program must build).
                let size_class = kids
                    .get(3)
                    .map(|arg| classify_expand_size(arg, env))
                    .unwrap_or(SizeClass::Unknown);
                result_ty = check_expand_signature(
                    &kids[1..],
                    &arg_tys,
                    &result_ty,
                    axis_is_dim_name,
                    size_class,
                    env,
                    subst,
                    errors,
                );
            }
            "layer_norm" => {
                result_ty = check_layer_norm_signature(&arg_tys, &result_ty, vg, subst, errors);
            }
            "conv2d" => {
                result_ty =
                    check_conv2d_signature(&kids[1..], &arg_tys, &result_ty, vg, subst, errors);
            }
            _ => {}
        }
    }

    // Post-check: logical ops require tensor[D, bool] arguments
    if let Some(ref fname) = func_name
        && LOGICAL_OPS.contains(&fname.as_str())
    {
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
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
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
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("{} requires bool arguments, got {}", fname, other),
                            ),
                            vec!["Logical ops only work on bool values".to_string()],
                        ),
                    );
                }
            }
        }
    }

    // Special case: comparison ops return tensor[D, bool] when any
    // argument is tensor-shaped. Comparison ops broadcast a scalar
    // arg against a tensor arg (see the rewrite block above), so the
    // result shape comes from whichever argument is the tensor —
    // not necessarily the first one (issue #5: `gt(1.5, xs)` was
    // returning `Prim(Bool)` instead of `tensor[D, bool]` because
    // this override only looked at `arg_tys[0]`).
    if let Some(ref fname) = func_name
        && builtins::COMPARISON_OPS.contains(&fname.as_str())
    {
        // Prefer any tensor-shaped arg as the dim source.
        let tensor_dims = arg_tys.iter().find_map(|t| match subst.apply(t) {
            Type::Tensor(dims, _) => Some(dims),
            _ => None,
        });
        if let Some(dims) = tensor_dims {
            return Type::Tensor(dims, TensorPrec::Concrete(Prim::Bool));
        }
        // No tensor arg → scalar comparison, returns scalar bool.
        if let Some(first_arg) = arg_tys.first() {
            let resolved_arg = type_for_readonly_check(first_arg, subst);
            if matches!(resolved_arg, Type::Prim(_)) {
                return Type::Prim(Prim::Bool);
            }
        }
    }

    if let Some(ref fname) = func_name {
        match fname.as_str() {
            "print" => {
                return Type::Unit;
            }
            "fail" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {
                            return Type::Var(vg.fresh_tvar());
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
            "string_len" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {
                            return Type::Prim(Prim::Int64);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("string_len expects string input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "string_concat" => {
                for arg_ty in &arg_tys {
                    match subst.apply(arg_ty) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {}
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "string_concat expects string arguments, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
                return Type::Prim(Prim::String);
            }
            "string_slice" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {}
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("string_slice expects string input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
                for (index, arg_ty) in arg_tys.iter().enumerate().skip(1) {
                    match subst.apply(arg_ty) {
                        Type::Prim(precision) if precision.is_integer() => {}
                        Type::Var(_) | Type::Error(_) => {}
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "string_slice expects integer index arguments; arg {} was {other}",
                                            index + 1
                                        ),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
                return Type::Prim(Prim::String);
            }
            "string_contains" | "string_starts_with" | "string_ends_with" => {
                for arg_ty in &arg_tys {
                    match subst.apply(arg_ty) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {}
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("{} expects string arguments, got {other}", fname),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
                return Type::Prim(Prim::Bool);
            }
            "string_trim" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {
                            return Type::Prim(Prim::String);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("string_trim expects string input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "to_string" => {
                return Type::Prim(Prim::String);
            }
            "to_int" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {
                            return Type::Adt("Option".to_string(), vec![Type::Prim(Prim::Int64)]);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("to_int expects string input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "to_float" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {
                            return Type::Adt("Option".to_string(), vec![Type::Prim(Prim::F64)]);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("to_float expects string input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "rank" => {
                if let Some(first_arg) = arg_tys.first() {
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {
                            return Type::Prim(Prim::Int32);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    match type_for_readonly_check(first_arg, subst) {
                        Type::Tensor(dims, _) => Some(dims),
                        Type::Var(_) | Type::Error(_) => None,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                // Issue #216: cast-aware so `shape(x, cast(N, int32))`
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
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                        Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error(_) => {
                            return Type::Prim(Prim::Int32);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("shape expects int32 axis, got {other}"),
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
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {
                            return Type::Prim(Prim::Int64);
                        }
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            "tensor_to_scalar expects a rank-0 tensor".to_string(),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                            // tensor_to_scalar requires a fully
                            // resolved precision. Polymorphic precision
                            // must be resolved by unification before
                            // this op can name a host scalar type.
                            return match precision {
                                TensorPrec::Concrete(p) => Type::Prim(p),
                                TensorPrec::Var(_) => result_ty,
                            };
                        }
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "tensor_to_scalar expects tensor input, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ),
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let Some(equation) = kids.get(1).and_then(extract_string_literal) else {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
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
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let tensor_ty = type_for_readonly_check(&arg_tys[0], subst);
                let indices_ty = type_for_readonly_check(&arg_tys[1], subst);
                let axis =
                    match resolve_builtin_axis("gather", kids.get(3), &tensor_ty, list, errors) {
                        Ok(axis) => axis,
                        Err(err) => return err,
                    };
                match infer_gather_result_type(&tensor_ty, &indices_ty, axis) {
                    Ok(ty) => return ty,
                    Err(message) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    message,
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "where" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
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
                            || cond_dims != then_dims
                            || then_dims != else_dims
                        {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    (Type::Var(_), _, _)
                    | (_, Type::Var(_), _)
                    | (_, _, Type::Var(_))
                    | (Type::Error(_), _, _)
                    | (_, Type::Error(_), _)
                    | (_, _, Type::Error(_)) => return result_ty,
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let cumsum_operand = type_for_readonly_check(&arg_tys[0], subst);
                let _axis = match resolve_builtin_axis(
                    "cumsum",
                    kids.get(2),
                    &cumsum_operand,
                    list,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match cumsum_operand {
                    Type::Tensor(dims, precision) => {
                        return Type::Tensor(dims, precision);
                    }
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let diagonal_operand = type_for_readonly_check(&arg_tys[0], subst);
                let axis1 = match resolve_axis_pair_member(
                    "diagonal",
                    kids.get(2),
                    &diagonal_operand,
                    0,
                    list,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                let axis2 = match resolve_axis_pair_member(
                    "diagonal",
                    kids.get(3),
                    &diagonal_operand,
                    1,
                    list,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match infer_diagonal_result_type(&diagonal_operand, axis1, axis2) {
                    Ok(ty) => return ty,
                    Err(message) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    message,
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "trace" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let trace_operand = type_for_readonly_check(&arg_tys[0], subst);
                let axis1 = match resolve_axis_pair_member(
                    "trace",
                    kids.get(2),
                    &trace_operand,
                    0,
                    list,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                let axis2 = match resolve_axis_pair_member(
                    "trace",
                    kids.get(3),
                    &trace_operand,
                    1,
                    list,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match infer_trace_result_type(&trace_operand, axis1, axis2) {
                    Ok(ty) => return ty,
                    Err(message) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    message,
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "clamp" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
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
                        let low_ok = low_dims.is_empty() || low_dims == input_dims;
                        let high_ok = high_dims.is_empty() || high_dims == input_dims;
                        if low_ok && high_ok && low_prec == input_prec && high_prec == input_prec {
                            return input_ty;
                        }
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "clamp expects tensor input plus scalar-tensor or matching-shape tensor bounds of the same precision, got {input_ty}, {low_ty}, and {high_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                    (Type::Var(_), _, _)
                    | (_, Type::Var(_), _)
                    | (_, _, Type::Var(_))
                    | (Type::Error(_), _, _)
                    | (_, Type::Error(_), _)
                    | (_, _, Type::Error(_)) => return result_ty,
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let sort_operand = type_for_readonly_check(&arg_tys[0], subst);
                let _axis =
                    match resolve_builtin_axis("sort", kids.get(2), &sort_operand, list, errors) {
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
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 5, arg_tys.len());
                }
                let base_ty = subst.apply(&arg_tys[0]);
                let indices_ty = subst.apply(&arg_tys[1]);
                let updates_ty = subst.apply(&arg_tys[2]);
                let axis =
                    match resolve_builtin_axis("scatter", kids.get(4), &base_ty, list, errors) {
                        Ok(axis) => axis,
                        Err(err) => return err,
                    };
                let mode = kids.get(5).and_then(extract_string_literal);
                match mode.as_deref() {
                    Some("replace") | Some("add") => {}
                    _ => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    "scatter mode must be \"replace\" or \"add\"".to_string(),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
                match infer_gather_result_type(&base_ty, &indices_ty, axis) {
                    Ok(expected_updates) => {
                        if let Err(te) = unify(&expected_updates, &updates_ty, subst) {
                            return report(errors, te.into());
                        }
                        return base_ty;
                    }
                    Err(message) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    message,
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "scatter_replace" => {
                // Tensor-lane replace-scatter (last-write-wins) — lowers
                // to RiscOp::Scatter. Distinct from the host-lane
                // `scatter(..., mode)` pentaop. AD policy: no_grad
                // (rejected via AdError::NotSupported); see
                // spec/05-risc-primitives.md §3.5.
                if arg_tys.len() != 4 {
                    return report_builtin_arity(errors, list, fname, 4, arg_tys.len());
                }
                let base_ty = subst.apply(&arg_tys[0]);
                let indices_ty = subst.apply(&arg_tys[1]);
                let updates_ty = subst.apply(&arg_tys[2]);
                let axis = match resolve_builtin_axis(
                    "scatter_replace",
                    kids.get(4),
                    &base_ty,
                    list,
                    errors,
                ) {
                    Ok(axis) => axis,
                    Err(err) => return err,
                };
                match infer_gather_result_type(&base_ty, &indices_ty, axis) {
                    Ok(expected_updates) => {
                        if let Err(te) = unify(&expected_updates, &updates_ty, subst) {
                            return report(errors, te.into());
                        }
                        return base_ty;
                    }
                    Err(message) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    message,
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "len" => {
                if let Some(first_arg) = arg_tys.first() {
                    match subst.apply(first_arg) {
                        Type::Adt(name, _) if name == "List" || name == "Dict" => {
                            return Type::Prim(Prim::Int64);
                        }
                        Type::Var(_) | Type::Error(_) => return Type::Prim(Prim::Int64),
                        Type::Ref(inner) if matches!(&*inner, Type::Adt(name, _) if name == "List" || name == "Dict") =>
                        {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let list_arg = subst.apply(&arg_tys[0]);
                let index_arg = subst.apply(&arg_tys[1]);
                if !matches!(index_arg, Type::Prim(prec) if prec.is_integer())
                    && !matches!(index_arg, Type::Var(_) | Type::Error(_))
                {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("index expects integer index, got {index_arg}"),
                            ),
                            vec![],
                        ),
                    );
                }
                match list_arg {
                    Type::Adt(name, mut args) if name == "List" && args.len() == 1 => {
                        return args.remove(0);
                    }
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    Type::Ref(inner) if matches!(&*inner, Type::Adt(name, _) if name == "List") => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("index expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "append" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let list_arg = subst.apply(&arg_tys[0]);
                let value_arg = subst.apply(&arg_tys[1]);
                match list_arg {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        if let Err(te) = unify(&args[0], &value_arg, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&args[0])]);
                    }
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("append expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "concat" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let lhs = subst.apply(&arg_tys[0]);
                let rhs = subst.apply(&arg_tys[1]);
                match (lhs, rhs) {
                    (Type::Adt(lhs_name, lhs_args), Type::Prim(precision))
                        if lhs_name == "List" && lhs_args.len() == 1 && precision.is_integer() =>
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
                        let raw_axis = kids.get(2).and_then(extract_int_for_dim);
                        let list_info = match kids.get(1).and_then(collect_cons_chain_for_shape) {
                            Some(elements) => ConcatListInfo::Direct(
                                elements
                                    .iter()
                                    .map(|elem| {
                                        match product.current_owner_type(elem, subst, errors) {
                                            Some(Type::Tensor(dims, _)) => dims,
                                            _ => Vec::new(),
                                        }
                                    })
                                    .collect(),
                            ),
                            None => ConcatListInfo::BindingLen(static_list_len(kids.get(1), env)),
                        };
                        match tensor_concat_result_type(&lhs_args[0], raw_axis, list_info) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            message,
                                        ),
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
                        if let Err(te) = unify(&lhs_args[0], &rhs_args[0], subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&lhs_args[0])]);
                    }
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (lhs, rhs) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let tensor_ty = type_for_readonly_check(&arg_tys[0], subst);
                let axis_ty = subst.apply(&arg_tys[1]);
                let sizes_ty = subst.apply(&arg_tys[2]);
                if !matches!(axis_ty, Type::Prim(prec) if prec.is_integer())
                    && !matches!(axis_ty, Type::Var(_) | Type::Error(_))
                {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("split expects integer axis, got {axis_ty}"),
                            ),
                            vec![],
                        ),
                    );
                }
                match (tensor_ty, sizes_ty) {
                    (Type::Tensor(dims, precision), Type::Adt(name, args))
                        if name == "List" && args.len() == 1 =>
                    {
                        if !matches!(&args[0], Type::Prim(prec) if prec.is_integer())
                            && !matches!(&args[0], Type::Var(_) | Type::Error(_))
                        {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        "split expects List[int] sizes".to_string(),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                        // Negative axes index from the end.
                        // Issue #216: cast-aware so a
                        // `cast(N, int32)`-wrapped split axis still
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
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
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
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (tensor_ty, sizes_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
            "take" | "drop" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let op_name = func_name.as_deref().unwrap_or("collection helper");
                let list_arg = subst.apply(&arg_tys[0]);
                let count_arg = subst.apply(&arg_tys[1]);
                if !matches!(count_arg, Type::Prim(prec) if prec.is_integer())
                    && !matches!(count_arg, Type::Var(_) | Type::Error(_))
                {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("{op_name} expects integer count, got {count_arg}"),
                            ),
                            vec![],
                        ),
                    );
                }
                match list_arg {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        return Type::Adt("List".to_string(), vec![args[0].clone()]);
                    }
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let list_arg = subst.apply(&arg_tys[0]);
                let count_arg = subst.apply(&arg_tys[1]);
                if !matches!(count_arg, Type::Prim(prec) if prec.is_integer())
                    && !matches!(count_arg, Type::Var(_) | Type::Error(_))
                {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("chunk expects integer size, got {count_arg}"),
                            ),
                            vec![],
                        ),
                    );
                }
                match list_arg {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        return Type::Adt(
                            "List".to_string(),
                            vec![Type::Adt("List".to_string(), vec![args[0].clone()])],
                        );
                    }
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                for arg_ty in &arg_tys {
                    match subst.apply(arg_ty) {
                        Type::Prim(prec) if prec.is_integer() => {}
                        Type::Var(_) | Type::Error(_) => {}
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let elem_ty = vg.fresh_type();
                let list_expr = deep::Expr::List(list.clone(), zero_span());
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
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
            "fold" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let acc_ty = vg.fresh_type();
                let elem_ty = vg.fresh_type();
                let list_expr = deep::Expr::List(list.clone(), zero_span());
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(
                        vec![acc_ty.clone(), elem_ty.clone()],
                        Box::new(acc_ty.clone()),
                    ),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "fold",
                            "expects a callback whose accumulator/result type matches the initial accumulator",
                            te,
                        ),
                    );
                }
                if let Err(te) = unify(&subst.apply(&arg_tys[1]), &acc_ty.clone(), subst) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "fold",
                            "expects a callback whose accumulator/result type matches the initial accumulator",
                            te,
                        ),
                    );
                }
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[2]),
                    &Type::Adt("List".to_string(), vec![elem_ty]),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                return subst.apply(&acc_ty);
            }
            "scan" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let acc_ty = vg.fresh_type();
                let elem_ty = vg.fresh_type();
                let list_expr = deep::Expr::List(list.clone(), zero_span());
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(
                        vec![acc_ty.clone(), elem_ty.clone()],
                        Box::new(acc_ty.clone()),
                    ),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "scan",
                            "expects a callback whose accumulator/result type matches the initial accumulator",
                            te,
                        ),
                    );
                }
                if let Err(te) = unify(&subst.apply(&arg_tys[1]), &acc_ty.clone(), subst) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "scan",
                            "expects a callback whose accumulator/result type matches the initial accumulator",
                            te,
                        ),
                    );
                }
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[2]),
                    &Type::Adt("List".to_string(), vec![elem_ty]),
                    subst,
                ) {
                    return report(errors, te.into());
                }
                return Type::Adt("List".to_string(), vec![subst.apply(&acc_ty)]);
            }
            "tensor_scan" => {
                // `tensor_scan(initial: T, fn: (T, int64) -> T, n: int64) -> tensor[n, T]`.
                //
                // Issue #257: host-runtime scan that produces a tensor
                // directly, sidestepping the right-recursive list build
                // that overflows the worker stack at ~10k elements.
                // Element type `T` must resolve to a concrete scalar
                // Prim before tensor lowering; the runtime arm enforces
                // that at execution time. At type-check time we accept
                // any Type::Prim and let unification do the rest.
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let elem_ty = vg.fresh_type();
                let int64 = Type::Prim(Prim::Int64);
                let list_expr = deep::Expr::List(list.clone(), zero_span());
                // arg 0: initial accumulator of type T.
                if let Err(te) = unify(&subst.apply(&arg_tys[0]), &elem_ty.clone(), subst) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "tensor_scan",
                            "expects an initial value whose type matches the callback element type",
                            te,
                        ),
                    );
                }
                // arg 1: callback `(T, int64) -> T`.
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[1]),
                    &Type::Fn(
                        vec![elem_ty.clone(), int64.clone()],
                        Box::new(elem_ty.clone()),
                    ),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "tensor_scan",
                            "expects a callback (T, int64) -> T",
                            te,
                        ),
                    );
                }
                // arg 2: length `n: int64`.
                if let Err(te) = unify(&subst.apply(&arg_tys[2]), &int64, subst) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
                            "tensor_scan",
                            "expects a length `n: int64`",
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
                        // when T is later pinned to int64 or bool.
                        TensorPrec::Var(*tv)
                    }
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &list_expr,
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                let elem_ty = vg.fresh_type();
                let list_expr = deep::Expr::List(list.clone(), zero_span());
                if let Err(te) = unify(
                    &subst.apply(&arg_tys[0]),
                    &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                    subst,
                ) {
                    return report(
                        errors,
                        collection_helper_type_error(
                            &list_expr,
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
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
                                Type::Var(_) | Type::Error(_) => return result_ty,
                                other => {
                                    return report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
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
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (lhs, rhs) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                            match &args[0] {
                                Type::Tuple(items) if items.len() == 2 => {
                                    match &items[0] {
                                        Type::Prim(Prim::Int64) | Type::Prim(Prim::String) => {}
                                        Type::Var(_) | Type::Error(_) => return result_ty,
                                        other => {
                                            return report(
                                                errors,
                                                CheckError::new(
                                                    CheckErrorKind::TypeMismatch,
                                                    with_macro_provenance(
                                                        &deep::Expr::List(
                                                            list.clone(),
                                                            zero_span(),
                                                        ),
                                                        format!(
                                                            "dict_of keys must be int64 or string, got {other}"
                                                        ),
                                                    ),
                                                    vec![],
                                                ),
                                            );
                                        }
                                    }
                                    return Type::Adt(
                                        "Dict".to_string(),
                                        vec![items[0].clone(), items[1].clone()],
                                    );
                                }
                                Type::Var(_) | Type::Error(_) => return result_ty,
                                other => {
                                    return report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                    (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                        if let Err(te) = unify(&args[0], &key_ty, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt("Option".to_string(), vec![subst.apply(&args[1])]);
                    }
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (dict_ty, key_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                    (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                        if let Err(te) = unify(&args[0], &key_ty, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Prim(Prim::Bool);
                    }
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (dict_ty, key_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
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
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (dict_ty, key_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
            "dict_insert" => {
                if arg_tys.len() != 3 {
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                match (
                    subst.apply(&arg_tys[0]),
                    subst.apply(&arg_tys[1]),
                    subst.apply(&arg_tys[2]),
                ) {
                    (Type::Adt(name, args), key_ty, value_ty)
                        if name == "Dict" && args.len() == 2 =>
                    {
                        if let Err(te) = unify(&args[0], &key_ty, subst) {
                            return report(errors, te.into());
                        }
                        if let Err(te) = unify(&args[1], &value_ty, subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt(
                            "Dict".to_string(),
                            vec![subst.apply(&args[0]), subst.apply(&args[1])],
                        );
                    }
                    (Type::Var(_), _, _)
                    | (_, Type::Var(_), _)
                    | (_, _, Type::Var(_))
                    | (Type::Error(_), _, _)
                    | (_, Type::Error(_), _)
                    | (_, _, Type::Error(_)) => {
                        return result_ty;
                    }
                    (dict_ty, key_ty, value_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "dict_insert expects Dict[K, V], K, and V, got {dict_ty}, {key_ty}, and {value_ty}"
                                    ),
                                ),
                                vec![],
                            ),
                        );
                    }
                }
            }
            "dict_merge" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
                }
                match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                    (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                        if lhs_name == "Dict"
                            && rhs_name == "Dict"
                            && lhs_args.len() == 2
                            && rhs_args.len() == 2 =>
                    {
                        if let Err(te) = unify(&lhs_args[0], &rhs_args[0], subst) {
                            return report(errors, te.into());
                        }
                        if let Err(te) = unify(&lhs_args[1], &rhs_args[1], subst) {
                            return report(errors, te.into());
                        }
                        return Type::Adt(
                            "Dict".to_string(),
                            vec![subst.apply(&lhs_args[0]), subst.apply(&lhs_args[1])],
                        );
                    }
                    (Type::Var(_), _)
                    | (_, Type::Var(_))
                    | (Type::Error(_), _)
                    | (_, Type::Error(_)) => {
                        return result_ty;
                    }
                    (lhs_ty, rhs_ty) => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "dict_merge expects matching Dict inputs, got {lhs_ty} and {rhs_ty}"
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                    if matches!(resolved, Type::Var(_) | Type::Error(_)) {
                        return result_ty;
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
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
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
                        ToTensorPeel::Pending => return result_ty,
                        ToTensorPeel::BadInner(inner) => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
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
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "to_list expects a rank-1 tensor, got rank {} tensor",
                                                dims.len()
                                            ),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                            // to_list requires a fully resolved
                            // precision: a polymorphic precision must
                            // be resolved before to_list can name a
                            // concrete element type. Defer if the
                            // precision is still a var.
                            let precision = match precision {
                                TensorPrec::Concrete(p) => p,
                                TensorPrec::Var(_) => return result_ty,
                            };
                            if !precision.is_numeric() && !matches!(precision, Prim::Bool) {
                                return report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "to_list expects numeric or bool tensor input, got {precision:?}"
                                            ),
                                        ),
                                        vec![],
                                    ),
                                );
                            }
                            return Type::Adt("List".to_string(), vec![Type::Prim(precision)]);
                        }
                        Type::Var(_) | Type::Error(_) => return result_ty,
                        other => {
                            return report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("to_list expects Tensor input, got {other}"),
                                    ),
                                    vec![],
                                ),
                            );
                        }
                    }
                }
            }
            "pad_sequences" => {
                if arg_tys.len() != 2 {
                    return report_builtin_arity(errors, list, fname, 2, arg_tys.len());
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
                                    Type::Var(_) | Type::Error(_) => return result_ty,
                                    other => {
                                        return report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(list.clone(), zero_span()),
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
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
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
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    return report_builtin_arity(errors, list, fname, 3, arg_tys.len());
                }
                let seqs_ty = subst.apply(&arg_tys[0]);
                let width_ty = subst.apply(&arg_tys[1]);
                let pad_ty = subst.apply(&arg_tys[2]);
                if let Err(te) = unify(&width_ty, &Type::Prim(Prim::Int64), subst) {
                    return report(errors, te.into());
                }
                // The padded (axis-1) dimension equals the `width`
                // argument. When `width` is a literal — including
                // `cast(N, int64)`, the form every caller uses —
                // propagate `Dim::Lit(N)` so the padded width is a
                // concrete dim that participates in shape checking.
                // A non-literal or non-positive width stays
                // `Dim::Wildcard` (the runtime validates the value).
                // `extract_int_for_dim` (not `extract_int_literal`)
                // is the cast-aware extractor used for dim contexts.
                let width_dim = children(list)
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
                                    Type::Var(_) | Type::Error(_) => return result_ty,
                                    other => {
                                        return report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(list.clone(), zero_span()),
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
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
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
                    Type::Var(_) | Type::Error(_) => return result_ty,
                    other => {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
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
                    // Host-lane JSON I/O (chelis#890): parse/serialize,
                    // dot-path accessors, output constructors over the
                    // prelude `Json` ADT, and `round_to` decimal rounding.
                    // Eval-only; the build backends reject them (see
                    // `reject_eval_only_builtins_host`).
                    "parse_json" | "to_json" | "json_f64" | "json_str" | "json_list"
                    | "json_f64s" | "jnum" | "jstr" | "jlist" | "jdict" | "json_set"
                    | "round_to" => {
                        return check_json_builtin_signature(fname, list, &arg_tys, subst, errors);
                    }
            _ => {}
        }
    }

    result_ty
}

/// Concrete argument/return contracts for the host-lane JSON I/O builtins
/// (chelis#890): `parse_json`, `to_json`, the dot-path accessors
/// (`json_f64`/`json_str`/`json_list`/`json_f64s`), the output
/// constructors (`jnum`/`jstr`/`jlist`/`jdict`/`json_set`), and
/// `round_to`.
///
/// The env schemes (`builtin_env`) only declare arity; this arm pins the
/// real types, and — per the chelis#891 review (finding 6) — it pins them
/// by **unification**, not by permissive matching: an argument whose type
/// is still a `Type::Var` (an un-annotated parameter, say) is unified
/// with the slot's expected type instead of waved through, so
/// `fn (doc) -> json_f64(doc, "a")` genuinely types `Json -> f64` rather
/// than `forall a. a -> f64`.
///
/// Numeric slots: `round_to` accepts ANY float operand / integer `places`
/// precision (unsuffixed literals default to f32/int32 per
/// spec/04-type-system.md §5.3, and its return preserves the operand
/// precision, so accepting f32 stays honest); an unresolved `Var` in
/// either slot unifies with the canonical f64/int64. `jnum` is stricter —
/// exactly f64 (chelis#891 review finding 7): its output feeds the
/// byte-exact `to_json` channel, and silently widening an f32 literal
/// would serialize `0.1f32` as `0.10000000149011612`. The diagnostic
/// names the fix (suffix the literal or cast).
fn check_json_builtin_signature(
    fname: &str,
    list: &deep::List,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    fn json_ty() -> Type {
        Type::Adt("Json".to_string(), Vec::new())
    }

    let expected_arity: usize = match fname {
        "parse_json" | "to_json" | "jnum" | "jstr" | "jlist" | "jdict" => 1,
        "json_set" => 3,
        _ => 2,
    };
    if arg_tys.len() != expected_arity {
        return report_builtin_arity(errors, list, fname, expected_arity, arg_tys.len());
    }

    let mut reject = |slot_description: String, got: &Type| -> Type {
        report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{fname} expects {slot_description}, got {got}"),
                ),
                vec![],
            ),
        )
    };

    // Slot checks: each resolves the argument fresh (earlier unifications
    // may have refined it), then unifies with the expected type. `unify`
    // treats `Type::Error` as success, so already-diagnosed slots do not
    // cascade.
    macro_rules! require_slot {
        ($idx:expr, $expected:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            if unify(&slot, &$expected, subst).is_err() {
                return reject($desc.to_string(), &subst.apply(&slot));
            }
        }};
    }

    // `round_to`'s two slots accept any concrete float/integer precision;
    // only an unresolved Var is pinned (to the canonical f64/int64) so the
    // contract is never vacuous through an un-annotated parameter.
    macro_rules! require_loose_numeric_slot {
        ($idx:expr, $is_kind:ident, $default:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            let ok = match &slot {
                Type::Prim(p) if p.$is_kind() => true,
                Type::Error(_) => true,
                Type::Var(_) => unify(&slot, &Type::Prim($default), subst).is_ok(),
                _ => false,
            };
            if !ok {
                return reject($desc.to_string(), &subst.apply(&slot));
            }
        }};
    }

    match fname {
        "parse_json" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            json_ty()
        }
        "to_json" => {
            require_slot!(
                0,
                json_ty(),
                "a Json argument (from `parse_json` or a J* constructor)"
            );
            Type::Prim(Prim::String)
        }
        "json_f64" | "json_str" | "json_list" | "json_f64s" => {
            require_slot!(
                0,
                json_ty(),
                "a Json first argument (from `parse_json` or a J* constructor)"
            );
            require_slot!(
                1,
                Type::Prim(Prim::String),
                "a dot-separated string path second argument (e.g. \"a.b.c\")"
            );
            match fname {
                "json_f64" => Type::Prim(Prim::F64),
                "json_str" => Type::Prim(Prim::String),
                "json_list" => Type::Adt("List".to_string(), vec![json_ty()]),
                _ => Type::Adt("List".to_string(), vec![Type::Prim(Prim::F64)]),
            }
        }
        "jnum" => {
            require_slot!(
                0,
                Type::Prim(Prim::F64),
                "an f64 argument (suffix the literal, `0.1f64`, or use cast(n, f64); \
                 an f32 value would quantize through the byte-exact serializer)"
            );
            json_ty()
        }
        "jstr" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            json_ty()
        }
        "jlist" => {
            let slot = type_for_readonly_check(&arg_tys[0], subst);
            match &slot {
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    let element = subst.apply(&args[0]);
                    if unify(&element, &json_ty(), subst).is_err() {
                        return reject("List[Json] input".to_string(), &subst.apply(&element));
                    }
                }
                Type::Error(_) => {}
                Type::Var(_) => {
                    let expected = Type::Adt("List".to_string(), vec![json_ty()]);
                    if unify(&slot, &expected, subst).is_err() {
                        return reject("List[Json] input".to_string(), &subst.apply(&slot));
                    }
                }
                other => return reject("List[Json] input".to_string(), other),
            }
            json_ty()
        }
        "jdict" => {
            let entry_ty = || Type::Tuple(vec![Type::Prim(Prim::String), json_ty()]);
            let slot = type_for_readonly_check(&arg_tys[0], subst);
            match &slot {
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    match subst.apply(&args[0]) {
                        Type::Tuple(items) if items.len() == 2 => {
                            let key_ty = subst.apply(&items[0]);
                            if unify(&key_ty, &Type::Prim(Prim::String), subst).is_err() {
                                return reject(
                                    "(string, Json) entry tuples (string keys)".to_string(),
                                    &subst.apply(&key_ty),
                                );
                            }
                            let value_ty = subst.apply(&items[1]);
                            if unify(&value_ty, &json_ty(), subst).is_err() {
                                return reject(
                                    "(string, Json) entry tuples (Json values)".to_string(),
                                    &subst.apply(&value_ty),
                                );
                            }
                        }
                        Type::Error(_) => {}
                        element @ Type::Var(_) => {
                            if unify(&element, &entry_ty(), subst).is_err() {
                                return reject(
                                    "List[(string, Json)] input".to_string(),
                                    &subst.apply(&element),
                                );
                            }
                        }
                        other => {
                            return reject("List[(string, Json)] input".to_string(), &other);
                        }
                    }
                }
                Type::Error(_) => {}
                Type::Var(_) => {
                    let expected = Type::Adt("List".to_string(), vec![entry_ty()]);
                    if unify(&slot, &expected, subst).is_err() {
                        return reject(
                            "List[(string, Json)] input".to_string(),
                            &subst.apply(&slot),
                        );
                    }
                }
                other => return reject("List[(string, Json)] input".to_string(), other),
            }
            json_ty()
        }
        "json_set" => {
            require_slot!(
                0,
                json_ty(),
                "a Json first argument (from `parse_json` or a J* constructor)"
            );
            require_slot!(
                1,
                Type::Prim(Prim::String),
                "a dot-separated string path second argument (e.g. \"a.b.c\")"
            );
            require_slot!(
                2,
                json_ty(),
                "a Json third argument (wrap raw values with jnum/jstr/jlist/jdict)"
            );
            json_ty()
        }
        "round_to" => {
            require_loose_numeric_slot!(
                0,
                is_float,
                Prim::F64,
                "a float first argument (any float precision)"
            );
            require_loose_numeric_slot!(
                1,
                is_integer,
                Prim::Int64,
                "an integer `places` second argument"
            );
            // Precision-preserving: `round_to` returns its operand's float
            // precision; an unresolved operand was just unified with the
            // canonical f64, keeping the checker and the eval lane in
            // agreement on the result dtype (chelis#891 review finding 6).
            match type_for_readonly_check(&arg_tys[0], subst) {
                Type::Prim(p) if p.is_float() => Type::Prim(p),
                _ => Type::Prim(Prim::F64),
            }
        }
        other => unreachable!("check_json_builtin_signature dispatched on `{other}`"),
    }
}
