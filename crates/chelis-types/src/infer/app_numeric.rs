//! Numeric operation rules and precision diagnostics.

use super::*;

pub(super) const TENSOR_OPS: &[&str] = &[
    "add",
    "mul",
    "sub",
    "div",
    "floor_div",
    "trunc_div",
    "neg",
    "recip",
    "exp",
    "log",
    "sin",
    "tan",
    "atan",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "relu",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "matmul",
    "layer_norm",
    "max_elem",
    "min_elem",
    "normalize",
    "cmplt",
    "eq",
    "neq",
    "lt",
    "gt",
    "lte",
    "gte",
    "and",
    "or",
    "not",
];

pub(super) const LOGICAL_OPS: &[&str] = &["and", "or", "not"];
pub(super) const INT_BINOPS: &[&str] = &["mod", "bitand", "bitor", "bitxor"];
pub(super) const INT_SHIFT_OPS: &[&str] = &["shl", "shr"];

/// Arithmetic operations for which `bool` has no authored numeric meaning.
/// Logical `and`/`or`/`not` remain the bool operations.
pub(super) const BOOL_REJECTED_ARITH_OPS: &[&str] = &["add", "sub", "mul", "neg", "floor_div"];

/// The post-desugar operand-dtype policy chokepoint from chelis#860.
///
/// Direct applications, reduction data arguments, and bare pipe stages all
/// consult this function. Unresolved types stay admissible here; the
/// polymorphic-instantiation pass mirrors the decided rows until Phase 4
/// derives both paths from the capability table.
pub(super) fn operand_dtype_rejection(
    fname: &str,
    resolved: &Type,
) -> Option<(CheckErrorKind, String, Vec<String>)> {
    let bool_operand = matches!(
        resolved,
        Type::Tensor(_, TensorPrec::Concrete(Prim::Bool)) | Type::Prim(Prim::Bool)
    );
    if bool_operand && BOOL_REJECTED_ARITH_OPS.contains(&fname) {
        return Some((
            CheckErrorKind::PrecisionMismatch,
            format!(
                "{fname} on bool operands is not admitted per the chelis#726 \
                 capability decision and spec/04-type-system.md section 9 \
                 [04-NUM-4]: bool is exactly {{0, 1}} and not a numeric dtype, \
                 so arithmetic on it has no authored meaning"
            ),
            vec![
                "chelis#726: use first-class `count(x, axes...)` to count true values, \
                 and `and`/`or`/`not` for bool logic."
                    .to_string(),
            ],
        ));
    }

    if fname == "mean" {
        let non_float_elem = match resolved {
            Type::Tensor(_, TensorPrec::Concrete(prim)) if !prim.is_float() => Some(prim.name()),
            Type::Prim(prim) if !prim.is_float() => Some(prim.name()),
            _ => None,
        };
        if let Some(prim_name) = non_float_elem {
            return Some((
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "mean on operand precision `{prim_name}` is not admitted per \
                     the chelis#724 capability decision: mean is float-only \
                     (f32, f64, bf16, f16). An integer mean has no authored \
                     rounding, and a fractional result inside an integer tensor \
                     violates spec/04-type-system.md section 9 [04-NUM-1]"
                ),
                vec![
                    "chelis#724: cast to a float precision first, e.g. \
                     `mean(cast(x, f32), 0)`."
                        .to_string(),
                ],
            ));
        }
        return None;
    }

    if fname == "softmax" {
        if let Type::Tensor(_, TensorPrec::Concrete(prim)) = resolved
            && !prim.is_float()
        {
            return Some((
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "softmax on operand precision `{}` is not admitted per \
                     spec/04-type-system.md §5.4: transcendental operations are \
                     restricted to f32, f64, bf16, f16 (not integer)",
                    prim.name()
                ),
                vec![
                    "spec/04-type-system.md §5.4: cast to a float precision before \
                     applying softmax."
                        .to_string(),
                ],
            ));
        }
        return None;
    }

    if !TENSOR_OPS.contains(&fname) {
        return None;
    }

    let ok = match fname {
        "matmul" | "layer_norm" | "normalize" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
        }
        "add" | "mul" | "sub" | "max_elem" | "min_elem" | "neg" | "floor_div" | "floor"
        | "ceil" | "round" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                || matches!(resolved, Type::Prim(prim) if prim.is_numeric())
        }
        "div" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float())
                || matches!(resolved, Type::Prim(prim) if prim.is_float())
        }
        "trunc_div" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_integer())
                || matches!(resolved, Type::Prim(prim) if prim.is_integer())
        }
        "exp" | "log" | "sin" | "tan" | "atan" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu"
        | "gelu" | "recip" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float())
                || matches!(resolved, Type::Prim(prim) if prim.is_float())
        }
        "cmplt" | "lt" | "gt" | "lte" | "gte" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                || matches!(resolved, Type::Prim(prim) if prim.is_numeric())
        }
        "eq" | "neq" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                || matches!(resolved, Type::Prim(_))
        }
        "and" | "or" | "not" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Concrete(Prim::Bool)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Prim(Prim::Bool))
        }
        _ => matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_)),
    };
    if ok {
        return None;
    }

    let is_transcendental = matches!(
        fname,
        "exp"
            | "log"
            | "sin"
            | "tan"
            | "atan"
            | "sqrt"
            | "relu"
            | "sigmoid"
            | "tanh"
            | "silu"
            | "gelu"
            | "recip"
    );
    let resolved_int_prim = match resolved {
        Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_integer() => Some(prim.name()),
        Type::Prim(prim) if prim.is_integer() => Some(prim.name()),
        _ => None,
    };
    let resolved_float_prim = match resolved {
        Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float() => Some(prim.name()),
        Type::Prim(prim) if prim.is_float() => Some(prim.name()),
        _ => None,
    };

    if is_transcendental
        && let Type::Tensor(_, TensorPrec::Concrete(prim)) = resolved
        && !prim.is_float()
    {
        Some((
            CheckErrorKind::PrecisionMismatch,
            format!(
                "{fname} on operand precision `{}` is not admitted per \
                 spec/04-type-system.md §5.4: transcendental operations are \
                 restricted to f32, f64, bf16, f16 (not integer)",
                prim.name()
            ),
            vec![format!(
                "spec/04-type-system.md §5.4: cast to a float precision before \
                 applying `{fname}`."
            )],
        ))
    } else if fname == "div"
        && let Some(prim_name) = resolved_int_prim
    {
        Some((
            CheckErrorKind::PrecisionMismatch,
            format!(
                "div on integer operand precision `{prim_name}` is not admitted per \
                 spec/05-risc-primitives.md §2.1: `div` is float-only (IEEE-754). \
                 Use `floor_div` (round toward -inf) or `trunc_div` (round toward \
                 zero) for integers."
            ),
            vec![
                "spec/05-risc-primitives.md §2.1: integer division uses `floor_div` \
                 or `trunc_div`; `div` requires float operands."
                    .to_string(),
            ],
        ))
    } else if fname == "trunc_div"
        && let Some(prim_name) = resolved_float_prim
    {
        Some((
            CheckErrorKind::PrecisionMismatch,
            format!(
                "trunc_div on float operand precision `{prim_name}` is not admitted \
                 per spec/05-risc-primitives.md §2.1: `trunc_div` is integer-only. \
                 Use `div` for IEEE-754 float division, or `floor_div` for a floored \
                 float quotient."
            ),
            vec![
                "spec/05-risc-primitives.md §2.1: `trunc_div` requires integer \
                 operands."
                    .to_string(),
            ],
        ))
    } else {
        Some((
            CheckErrorKind::TypeMismatch,
            format!("{fname} does not accept argument type {resolved} in this context"),
            vec![],
        ))
    }
}

/// Check numeric, comparison, softmax, and reduction argument restrictions.
///
/// `None` permits later operation-family checks. `Some` carries the original
/// early rejection type and its diagnostic.
pub(super) fn validate_numeric_and_reduction_arguments(
    list: &deep::List,
    kids: &[deep::Expr],
    func_name: &Option<String>,
    arg_tys: &[Type],
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
    route_observed: &mut bool,
) -> Option<Type> {
    macro_rules! reject {
        ($($arg:tt)*) => {
            return Some(report($($arg)*))
        };
    }

    // Post-check: shared builtins can operate on either tensors or host scalars.
    if let Some(fname) = func_name
        && TENSOR_OPS.contains(&fname.as_str())
    {
        *route_observed = true;
        for arg_ty in arg_tys {
            let resolved = type_for_readonly_check(arg_ty, subst);
            if let Some((kind, message, hints)) = operand_dtype_rejection(fname, &resolved) {
                reject!(
                    errors,
                    CheckError::new(
                        kind,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            message,
                        ),
                        hints,
                    ),
                );
            }
        }
    }

    if let Some(fname) = func_name
        && matches!(
            fname.as_str(),
            "softmax"
                | "mean"
                | "sum"
                | "count"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    {
        *route_observed = true;
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    reject!(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("{} expects tensor input, got {}", fname, resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
            if let Some((kind, message, hints)) = operand_dtype_rejection(fname, &resolved) {
                reject!(
                    errors,
                    CheckError::new(
                        kind,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            message,
                        ),
                        hints,
                    ),
                );
            }
        }

        if let Some(axis_arg) = arg_tys.get(1) {
            let resolved = subst.apply(axis_arg);
            match &resolved {
                Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    reject!(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("{} expects int32 axis, got {}", fname, resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }

        // softmax does not go through `check_reduction_signature`
        // (it is shape-preserving, not shape-reducing), so its
        // axis range is validated here. Negative axes index from
        // the end via `normalize_static_axis`, consistent with
        // the reductions and gather/scatter.
        // Issue #216: cast-aware so `softmax(x, cast(N, int32))`
        // surfaces the bounds-check diagnostic at infer.
        if fname == "softmax"
            && let Some(first_arg) = arg_tys.first()
            && let Type::Tensor(dims, _) = type_for_readonly_check(first_arg, subst)
            && let Some(raw) = kids.get(2).and_then(extract_int_for_dim)
            && normalize_static_axis(dims.len(), raw).is_none()
        {
            reject!(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "softmax axis {raw} is out of bounds for rank {} tensor",
                            dims.len()
                        ),
                    ),
                    vec![],
                ),
            );
        }
    }

    None
}

/// Type an integer binary operation or shift.
///
/// Unlike the operand-admissibility checks, these arms decide the call's
/// result type rather than only rejecting a bad one: `Some(ty)` short-circuits
/// the rest of application checking with `ty`, and `None` means `func_name` is
/// not one of these operations.
pub(super) fn integer_binop_result_type(
    list: &deep::List,
    func_name: Option<&str>,
    arg_tys: &[Type],
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    if let Some(fname) = func_name
        && INT_BINOPS.contains(&fname)
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
                return Some(Type::Prim(*lhs_prec));
            }
            (Type::Var(_), Type::Prim(rhs_prec)) if rhs_prec.is_integer() => {
                return Some(lhs);
            }
            (Type::Prim(lhs_prec), Type::Var(_)) if lhs_prec.is_integer() => {
                return Some(lhs);
            }
            (Type::Var(_), Type::Var(_)) | (Type::Error(_), _) | (_, Type::Error(_)) => {
                return Some(lhs);
            }
            _ => {
                return reject(
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

    if let Some(fname) = func_name
        && INT_SHIFT_OPS.contains(&fname)
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
            return Some(lhs);
        }
        return reject(
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

    None
}
