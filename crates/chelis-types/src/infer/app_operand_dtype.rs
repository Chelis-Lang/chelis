//! Per-callee operand dtype admissibility.
//!
//! A handful of stdlib callees constrain their operands' dtypes beyond what
//! signature unification already enforces: a float-only tensor operand, or a
//! tolerance that must carry the same dtype as the tensor it is compared
//! against. Those checks run after unification and before the builtin
//! dispatch in [`super::app_post::finish_unified_app`], and each one either
//! rejects the call outright or falls through to the next.
//!
//! The checks remain in their original order, and each rejection still
//! returns before the next callee's check runs.

use super::*;

/// Validate `test_assert_close_tensor`'s concrete operand dtypes.
///
/// Generic application inference calls this once before structural signature
/// unification so direct dtype mismatches retain the operation-specific
/// diagnostic. [`reject_inadmissible_operand_dtypes`] calls it again after
/// successful unification so unresolved polymorphic calls still participate
/// in the checked-route accounting and float-only propagation rules.
pub(super) fn reject_test_assert_close_tensor_operand_dtypes(
    list: &deep::List,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let actual = arg_tys.first().map(|ty| type_for_readonly_check(ty, subst));
    let tensor_prim = match actual {
        Some(Type::Tensor(_, TensorPrec::Concrete(prim))) if prim.is_float() => Some(prim),
        Some(Type::Tensor(_, TensorPrec::Concrete(prim))) => {
            return reject(
                errors,
                CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "test_assert_close_tensor expects tensors at one active float dtype, got `{}`",
                            prim.name()
                        ),
                    ),
                    vec![],
                ),
            );
        }
        Some(Type::Tensor(_, TensorPrec::Var(_)))
        | Some(Type::Var(_))
        | Some(Type::Error(_))
        | None => None,
        Some(other) => {
            return reject(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("test_assert_close_tensor expects tensor arguments, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    if let Some(tolerance) = arg_tys.get(2).map(|ty| subst.apply(ty)) {
        match tolerance {
            Type::Prim(tolerance_prim) if tolerance_prim.is_float() => {
                if let Some(tensor_prim) = tensor_prim
                    && tolerance_prim != tensor_prim
                {
                    return reject(
                        errors,
                        CheckError::new(
                            CheckErrorKind::PrecisionMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "test_assert_close_tensor tolerance dtype `{}` must equal tensor dtype `{}`",
                                    tolerance_prim.name(),
                                    tensor_prim.name()
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
            }
            Type::Var(_) | Type::Error(_) => {}
            other => {
                return reject(
                    errors,
                    CheckError::new(
                        CheckErrorKind::PrecisionMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!(
                                "test_assert_close_tensor tolerance must have the tensor's active float dtype, got {other}"
                            ),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }
    None
}

/// Run every per-callee operand dtype check for `func_name`.
///
/// Sets `checked_route_observed` for the callees that carry a checked
/// inference rule, exactly as the inline checks did.
#[allow(clippy::too_many_arguments)]
pub(super) fn reject_inadmissible_operand_dtypes(
    list: &deep::List,
    kids: &[deep::Expr],
    func_name: Option<&str>,
    arg_tys: &[Type],
    env: &Env,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    checked_route_observed: &mut bool,
) -> Option<Type> {
    if let Some(fname) = func_name
        && fname == "test_assert_close_tensor"
    {
        *checked_route_observed = true;
        if let Some(rejected) =
            reject_test_assert_close_tensor_operand_dtypes(list, arg_tys, subst, errors)
        {
            return Some(rejected);
        }
    }

    if let Some(fname) = func_name
        && fname == "uniform_like"
    {
        *checked_route_observed = true;
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, prim)
                    if prim.is_float()
                        || (matches!(prim, TensorPrec::Var(_))
                            && env.exact_stdlib_expected_result().is_some()) => {}
                Type::Tensor(_, _) => {
                    return reject(
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
                    return reject(
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
                    return reject(
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
                return reject(
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

    if let Some(fname) = func_name
        && fname == "dropout"
    {
        *checked_route_observed = true;
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    return reject(
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
                    return reject(
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

    None
}
