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
///
/// Only the second of those two calls carries a suspension. chelis#1512's
/// deferral belongs to the call that DECIDES, and the pre-unification one is a
/// diagnostic-quality pass whose operands unification has not constrained yet;
/// suspending from there would register a call the second one goes on to
/// decide anyway.
#[allow(clippy::too_many_arguments)]
pub(super) fn reject_test_assert_close_tensor_operand_dtypes(
    node: &DeepNode,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    suspension: Option<&DtypeAdmissibilitySite<'_>>,
    result_ty: &Type,
    product: &mut InferenceProduct,
) -> Option<Type> {
    let actual = arg_tys.first().map(|ty| type_for_readonly_check(ty, subst));
    let tensor_prim = match actual {
        Some(Type::Tensor(_, TensorPrec::Concrete(prim))) if prim.is_float() => Some(prim),
        Some(Type::Tensor(_, TensorPrec::Concrete(prim))) => {
            return reject(
                errors,
                CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    with_node_provenance(
                        node,
                        format!(
                            "test_assert_close_tensor expects tensors at one active float dtype, got `{}`",
                            prim.name()
                        ),
                    ),
                    vec![],
                ),
            );
        }
        // chelis#1512: the operand is not a tensor at a known dtype YET.
        Some(Type::Var(_)) => {
            if let Some(site) = suspension {
                site.register(arg_tys, result_ty, subst, product);
            }
            None
        }
        // chelis#1805 measured this arm safe rather than repairing it. The
        // builtin's signature shares ONE `Float`-bounded precision variable
        // across both tensors and the tolerance, so unification propagates that
        // bound onto a caller's own binder and rejects every integer
        // instantiation at the call site, with the [04-DTYPE-2] family
        // diagnostic rather than this one. Admitting the variable here
        // therefore skips no decision;
        // `a_signature_bounded_callee_is_caught_at_the_call_site` is the
        // witness.
        Some(Type::Tensor(_, TensorPrec::Var(_))) | Some(Type::Error(_)) | None => None,
        Some(other) => {
            return reject(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_node_provenance(
                        node,
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
                            with_node_provenance(
                                node,
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
            // chelis#1512: the tolerance has no dtype YET.
            Type::Var(_) => {
                if let Some(site) = suspension {
                    site.register(arg_tys, result_ty, subst, product);
                }
            }
            Type::Error(_) => {}
            other => {
                return reject(
                    errors,
                    CheckError::new(
                        CheckErrorKind::PrecisionMismatch,
                        with_node_provenance(
                            node,
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
    node: &DeepNode,
    func_name: Option<&str>,
    arg_tys: &[Type],
    env: &Env,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    checked_route_observed: &mut bool,
    suspension: Option<&DtypeAdmissibilitySite<'_>>,
    result_ty: &Type,
    product: &mut InferenceProduct,
) -> Option<Type> {
    if let Some(fname) = func_name
        && fname == "test_assert_close_tensor"
    {
        *checked_route_observed = true;
        if let Some(rejected) = reject_test_assert_close_tensor_operand_dtypes(
            node, arg_tys, subst, errors, suspension, result_ty, product,
        ) {
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
                            with_node_provenance(
                                node,
                                format!(
                                    "uniform_like expects a float tensor template, got {}",
                                    resolved
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
                // chelis#1512: `uniform_like`'s template parameter is a bare
                // type variable in the builtin scheme, so signature
                // unification does not bind it and an operand really does
                // reach this arm unresolved.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                Type::Error(_) => {}
                _ => {
                    return reject(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
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

        for arg_ty in arg_tys.iter().skip(1).take(2) {
            let resolved = type_for_readonly_check(arg_ty, subst);
            match &resolved {
                Type::Prim(Prim::F32) => {}
                // chelis#1512: the bound has no dtype YET.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                Type::Error(_) => {}
                _ => {
                    return reject(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
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
        }
    }

    if let Some(fname) = func_name
        && fname == "dropout"
    {
        *checked_route_observed = true;
        let tensor_prim = arg_tys
            .first()
            .and_then(|ty| match type_for_readonly_check(ty, subst) {
                Type::Tensor(_, TensorPrec::Concrete(prim)) => Some(prim),
                _ => None,
            });
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float() => {}
                // chelis#1512: not a tensor at a known dtype YET.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                // chelis#1805: safe for the same reason as
                // `test_assert_close_tensor` above. `dropout`'s signature binds
                // its tensor and its rate to one `Float`-bounded precision
                // variable, so an integer instantiation is rejected where the
                // caller supplies it.
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Error(_) => {}
                _ => {
                    return reject(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "dropout expects a tensor at an active float dtype, got {}",
                                    resolved
                                ),
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
                Type::Prim(prim)
                    if prim.is_float() && tensor_prim.is_none_or(|input| input == *prim) => {}
                // chelis#1512: the rate has no dtype YET.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                Type::Error(_) => {}
                _ => {
                    return reject(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "dropout rate must have the input tensor's active float dtype, got {}",
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

    None
}
