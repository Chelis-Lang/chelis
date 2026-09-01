//! Rank-only type facts used by the post-inference shape validator.

use super::validate::{
    derive_ir_builtin_output_type, expr_type_expr, node_expr, tensor_precision_expr,
    validator_error,
};
use super::*;

/// Bind a Surf function's parameter names to the standalone `defsig`
/// parameter types visible in the same validation unit. Surf deliberately
/// keeps those types on `defsig` rather than duplicating them into `(params
/// ...)`; without this bridge, the post-inference shape validator loses every
/// parameter rank at the function boundary.
pub(super) fn extend_ir_env_with_declared_fn_params(
    fn_expr: &deep::Expr,
    parameter_types: &[deep::Expr],
    type_env: &IrTypeEnv,
) -> IrTypeEnv {
    let mut scoped = type_env.clone();
    let Some((DeepTag::Fn, _, fn_children)) = stamped_parts(fn_expr) else {
        return scoped;
    };
    let Some(params_expr) = fn_children.first() else {
        return scoped;
    };
    let Some((DeepTag::Params, _, params)) = stamped_parts(params_expr) else {
        return scoped;
    };
    for (param, parameter_type) in params.iter().zip(parameter_types) {
        if let Some(name) = param_name_for_refs(param) {
            scoped.insert(name, parameter_type.clone());
        }
    }
    scoped
}

/// Reject a shape-identity call when the stamped argument types prove that
/// two non-scalar tensor operands have different ranks.
///
/// The HM schemes use one dimension row for these builtins, but movement-op
/// wildcards can currently make a rank-divergent call appear unified. The
/// central shape registry is the authority for the family: adding a new
/// identity builtin automatically admits it to this validation boundary.
/// Rank-zero arguments are left to each builtin's ordinary scheme (notably
/// `clamp`'s explicit scalar-bound form); this check does not invent a
/// general scalar-broadcast rule.
pub(super) fn validate_identity_builtin_rank_requirements(
    list: &deep::List,
    func_name: &str,
    type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) {
    let mut expected_rank: Option<usize> = None;
    for argument in list.elements.iter().skip(3) {
        let Some(dims) = arg_tensor_rank_type_expr(argument, type_env)
            .as_ref()
            .and_then(tensor_dims_from_type_expr)
        else {
            continue;
        };
        let rank = dims.len();
        if rank == 0 {
            continue;
        }
        match expected_rank {
            Some(expected) if expected != rank => {
                errors.push(validator_error(
                    CheckErrorKind::DimensionMismatch,
                    list,
                    format!(
                        "IR elementwise builtin `{func_name}` requires matching positive-rank tensor operands, got ranks {expected} and {rank}"
                    ),
                    vec![
                        "Make every tensor operand's dimension list identical; use `expand` explicitly when a rank change is intended"
                            .to_string(),
                    ],
                ));
                return;
            }
            None => expected_rank = Some(rank),
            _ => {}
        }
    }
}

/// Derive a tensor type that preserves only the output rank and precision of
/// a movement operation. Every synthetic dimension is deliberately
/// non-concrete: `stride` changes extents and `expand` inserts an extent, so
/// copying the input dimension expressions would overstate what this
/// validator has proved.
pub(super) fn derive_movement_rank_output_type(
    list: &deep::List,
    type_env: &IrTypeEnv,
    added_axes: usize,
) -> Option<deep::Expr> {
    let input_ty = list
        .elements
        .get(3)
        .and_then(|argument| arg_tensor_rank_type_expr(argument, type_env))?;
    let input_rank = tensor_dims_from_type_expr(&input_ty)?.len();
    let precision = tensor_precision_expr(&input_ty)?;
    let mut children = (0..input_rank.checked_add(added_axes)?)
        .map(|axis| {
            node_expr(
                DeepTag::DName,
                vec![symbol_expr(&format!("__chelis_rank_only_axis_{axis}"))],
            )
        })
        .collect::<Vec<_>>();
    children.push(precision);
    Some(node_expr(DeepTag::TTensor, children))
}

/// Resolve the tensor type expression of a callsite argument, peeking through
/// a `(borrow {} <inner>)` wrapper if present.
pub(super) fn arg_tensor_type_expr(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let inner = peel_borrow(expr);
    let ty = expr_type_expr(inner, type_env)?;
    (!type_expr_is_rank_only(&ty)).then_some(ty)
}

/// Resolve a tensor type for rank comparison, including validator-internal
/// rank-only facts. Exact-shape validators deliberately use
/// `arg_tensor_type_expr` instead, so a synthetic rank never masquerades as a
/// proved symbolic extent.
pub(super) fn arg_tensor_rank_type_expr(
    expr: &deep::Expr,
    type_env: &IrTypeEnv,
) -> Option<deep::Expr> {
    let inner = peel_borrow(expr);
    // An inline identity application can carry a stamped wildcard tensor
    // type even when its operands prove a concrete rank. Prefer structural
    // derivation for recognized builtins before consulting that stamp. This
    // is the same resolver used for let-bound values, so introducing or
    // removing a binding cannot change rank-honesty validation (chelis#668).
    derive_ir_builtin_output_type(inner, type_env).or_else(|| expr_type_expr(inner, type_env))
}

pub(super) fn type_expr_is_rank_only(expr: &deep::Expr) -> bool {
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    if tag == DeepTag::TRef {
        return kids.first().is_some_and(type_expr_is_rank_only);
    }
    if tag != DeepTag::TTensor || kids.is_empty() {
        return false;
    }
    kids[..kids.len() - 1].iter().any(|dim| {
        let Some((DeepTag::DName, _, name_parts)) = stamped_parts(dim) else {
            return false;
        };
        name_parts
            .first()
            .and_then(symbol_name)
            .is_some_and(|name| name.starts_with("__chelis_rank_only_axis_"))
    })
}

pub(super) fn peel_borrow(expr: &deep::Expr) -> &deep::Expr {
    stack_guard!("peel_borrow", expr, expr);
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some(DeepTag::Borrow)
        && let Some(child) = children(list).first()
    {
        return peel_borrow(child);
    }
    expr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeepDimKind {
    Lit(i64),
    NonConcrete,
}

pub(super) fn tensor_dims_from_type_expr(expr: &deep::Expr) -> Option<Vec<DeepDimKind>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) == Some(DeepTag::TRef) {
        return children(list).first().and_then(tensor_dims_from_type_expr);
    }
    if get_tag(list) != Some(DeepTag::TTensor) {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    for kid in &kids[..kids.len().saturating_sub(1)] {
        dims.push(match kid {
            deep::Expr::List(dim_list, _) if get_tag(dim_list) == Some(DeepTag::DLit) => {
                match children(dim_list).first() {
                    Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => DeepDimKind::Lit(*n),
                    _ => DeepDimKind::NonConcrete,
                }
            }
            _ => DeepDimKind::NonConcrete,
        });
    }
    Some(dims)
}

pub(super) fn type_expr_is_ir_concrete(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::TPrim) => true,
        _ => tensor_dims_from_type_expr(expr)
            .map(|dims| dims.iter().all(|d| matches!(d, DeepDimKind::Lit(_))))
            .unwrap_or(false),
    }
}
