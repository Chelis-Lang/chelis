//! Rank-only type facts used by the post-inference shape validator.

use super::validate::{derive_ir_builtin_output_type, validator_error};
use super::*;

/// Validator-private shape knowledge.
///
/// Rank-only facts are deliberately not encoded as Deep dimension syntax.
/// Authored programs can spell every legal `d-name`, so any string sentinel
/// in a `t-tensor` is forgeable and can make an exact-shape consumer mistake
/// source syntax for internal state. This Rust enum is constructed only by
/// the validator and cannot cross the source/Deep boundary.
#[derive(Debug, Clone)]
pub(super) enum ShapeTypeFact {
    Exact(deep::Expr),
    RankOnly { rank: usize },
}

impl ShapeTypeFact {
    fn exact_expr(&self) -> Option<&deep::Expr> {
        match self {
            Self::Exact(expr) => Some(expr),
            Self::RankOnly { .. } => None,
        }
    }

    fn tensor_rank(&self) -> Option<usize> {
        match self {
            Self::Exact(expr) => tensor_dims_from_type_expr(expr).map(|dims| dims.len()),
            Self::RankOnly { rank } => Some(*rank),
        }
    }
}

/// Name-keyed shape evidence for one validation scope.
///
/// Every access is a keyed lookup, insert, or whole-env clone: no caller
/// iterates this environment, and none may start to. The facts here decide a
/// checker verdict, so an iteration order that picked between two of them
/// would be exactly the chelis#1341 class. `UnordMap` has no iteration exit
/// that is not a written `to_sorted` claim of C3.1 authority, which a name
/// keyed by an arbitrary source identifier cannot make; `BTreeMap` would
/// leave `iter` and `keys` open to a later edit.
pub(super) type ShapeTypeEnv = UnordMap<String, ShapeTypeFact>;

pub(super) fn shape_type_env(type_env: &IrTypeEnv) -> ShapeTypeEnv {
    type_env
        .iter()
        .map(|(name, ty)| (name.clone(), ShapeTypeFact::Exact(ty.clone())))
        .collect()
}

/// Bind a Surf function's parameter names to the standalone `defsig`
/// parameter types visible in the same validation unit. Surf deliberately
/// keeps those types on `defsig` rather than duplicating them into `(params
/// ...)`; without this bridge, the post-inference shape validator loses every
/// parameter rank at the function boundary.
pub(super) fn extend_ir_env_with_declared_fn_params(
    fn_expr: &deep::Expr,
    parameter_types: &[deep::Expr],
    type_env: &ShapeTypeEnv,
) -> ShapeTypeEnv {
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
            scoped.insert(name, ShapeTypeFact::Exact(parameter_type.clone()));
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
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
    errors: &mut DiagnosticSink<'_>,
) {
    let mut expected_rank: Option<usize> = None;
    for argument in list.elements.iter().skip(3) {
        let Some(rank) = arg_tensor_rank(argument, type_env, static_env) else {
            continue;
        };
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
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
    added_axes: usize,
) -> Option<ShapeTypeFact> {
    let input_rank = list
        .elements
        .get(3)
        .and_then(|argument| arg_tensor_rank(argument, type_env, static_env))?;
    Some(ShapeTypeFact::RankOnly {
        rank: input_rank.checked_add(added_axes)?,
    })
}

/// Resolve the tensor type expression of a callsite argument, peeking through
/// a `(borrow {} <inner>)` wrapper if present.
pub(super) fn arg_tensor_type_expr(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
) -> Option<deep::Expr> {
    let inner = peel_borrow(expr);
    expr_shape_type_fact(inner, type_env)?.exact_expr().cloned()
}

/// Resolve a tensor rank for comparison, including validator-private
/// rank-only facts. Exact-shape validators deliberately use
/// `arg_tensor_type_expr` instead, so a rank fact never masquerades as a
/// proved symbolic extent.
pub(super) fn arg_tensor_rank(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
) -> Option<usize> {
    let inner = peel_borrow(expr);
    // An inline identity application can carry a stamped wildcard tensor
    // type even when its operands prove a concrete rank. Prefer structural
    // derivation for recognized builtins before consulting that stamp. This
    // is the same resolver used for let-bound values, so introducing or
    // removing a binding cannot change rank-honesty validation (chelis#668).
    derive_ir_builtin_output_type(inner, type_env, static_env)
        .or_else(|| expr_shape_type_fact(inner, type_env))
        .and_then(|fact| fact.tensor_rank())
}

pub(super) fn expr_shape_type_fact(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
) -> Option<ShapeTypeFact> {
    stack_guard!("expr_shape_type_fact", expr, None);
    match expr {
        deep::Expr::Node(node, _) => {
            if node.tag() == DeepTag::Var
                && let Some(name) = node.children_slice().first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::MetaExpr(meta, _) => expr_shape_type_fact(&meta.expr, type_env),
        _ => None,
    }
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
