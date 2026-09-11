//! chelis#1512: the `string_*` routes of `finish_unified_app`.
//!
//! One responsibility: every builtin whose operand contract is "this is a
//! string", which is the `string_*` group plus `to_int` and `to_float`.
//! `to_string` stays behind: it accepts anything and reads no operand. They moved here because `app_post.rs` reached the 3,000-line
//! architecture limit when this pull request's arm splits met chelis#1739's
//! `declared_result_literal` on `main`, and that limit's own message asks for
//! a split by responsibility rather than a trim to fit. The string group is
//! the coherent one: it shares an operand contract, it shares a diagnostic
//! shape, and nothing else in `finish_unified_app` reads it.
//!
//! The routing is unchanged. `finish_unified_app` still owns the dispatch and
//! still builds the deferral site; this module holds the arms and nothing
//! else. `None` means the caller's own match runs: either no route here owns
//! the name, or a route owns it and reached its arm with an EMPTY argument
//! list, which is the one path through an owned name that decides nothing.
//! Every owned route with arguments answers `Some`, the `string_contains`
//! family included: it validates each argument and then returns `bool`.

use super::*;

/// Decide a string-operand route, or hand the name back.
///
/// `Some` is the route's answer, already reported if it rejected. `None` means
/// the caller decides: no route here owns the name, or one does and was
/// reached with an empty argument list, which every arm guards with
/// `arg_tys.first()` or a loop that runs zero times.
#[allow(clippy::too_many_arguments)]
pub(super) fn string_route_result(
    fname: &str,
    list: &deep::List,
    arg_tys: &[Type],
    result_ty: &Type,
    site: &UnresolvedOperandSite<'_>,
    product: &mut InferenceProduct,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    match fname {
        "string_len" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Prim(Prim::String) | Type::Error(_) => {
                        return Some(Type::Prim(Prim::Int64));
                    }
                    Type::Var(_) => {
                        return Some(site.defer(
                            arg_tys,
                            result_ty,
                            product,
                            Type::Prim(Prim::Int64),
                        ));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("string_len expects string input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "string_concat" => {
            for arg_ty in arg_tys {
                match subst.apply(arg_ty) {
                    Type::Prim(Prim::String) | Type::Error(_) => {}
                    Type::Var(_) => site.register(arg_tys, result_ty, product),
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("string_concat expects string arguments, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
            return Some(Type::Prim(Prim::String));
        }
        "string_slice" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Prim(Prim::String) | Type::Error(_) => {}
                    Type::Var(_) => site.register(arg_tys, result_ty, product),
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("string_slice expects string input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
            for (index, arg_ty) in arg_tys.iter().enumerate().skip(1) {
                match subst.apply(arg_ty) {
                    Type::Prim(precision) if precision.is_integer() => {}
                    Type::Error(_) => {}
                    Type::Var(_) => site.register(arg_tys, result_ty, product),
                    other => {
                        return Some(report(
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
                        ));
                    }
                }
            }
            return Some(Type::Prim(Prim::String));
        }
        "string_contains" | "string_starts_with" | "string_ends_with" => {
            for arg_ty in arg_tys {
                match subst.apply(arg_ty) {
                    Type::Prim(Prim::String) | Type::Var(_) | Type::Error(_) => {}
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} expects string arguments, got {other}", fname),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
            return Some(Type::Prim(Prim::Bool));
        }
        "string_trim" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Prim(Prim::String) | Type::Error(_) => {
                        return Some(Type::Prim(Prim::String));
                    }
                    Type::Var(_) => {
                        return Some(site.defer(
                            arg_tys,
                            result_ty,
                            product,
                            Type::Prim(Prim::String),
                        ));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("string_trim expects string input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "to_int" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Prim(Prim::String) | Type::Error(_) => {
                        return Some(Type::Adt(
                            "Option".to_string(),
                            vec![Type::Prim(Prim::Int64)],
                        ));
                    }
                    Type::Var(_) => {
                        return Some(site.defer(
                            arg_tys,
                            result_ty,
                            product,
                            Type::Adt("Option".to_string(), vec![Type::Prim(Prim::Int64)]),
                        ));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("to_int expects string input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "to_float" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Prim(Prim::String) | Type::Error(_) => {
                        return Some(Type::Adt("Option".to_string(), vec![Type::Prim(Prim::F64)]));
                    }
                    Type::Var(_) => {
                        return Some(site.defer(
                            arg_tys,
                            result_ty,
                            product,
                            Type::Adt("Option".to_string(), vec![Type::Prim(Prim::F64)]),
                        ));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("to_float expects string input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        _ => {}
    }
    None
}

/// The names this module owns, so the caller's dispatch can hand them over in
/// one arm rather than listing them twice.
pub(super) fn string_route_owns(fname: &str) -> bool {
    fname.starts_with("string_") || matches!(fname, "to_int" | "to_float")
}
