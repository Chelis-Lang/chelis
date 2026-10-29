//! Exact-shape type facts used by the post-inference shape validator.
//!
//! The environment carries one kind of fact: a Deep type expression the
//! checker itself stamped. There is no rank-only carrier, because rank
//! agreement is unification's (`spec/04-type-system.md` section 4.7.2, and
//! `spec/design/checker_totality.md` section PP5 D6 (d)). A second carrier
//! that held a rank derived here would be a second rank model, and the one
//! that existed disagreed with the checker's own stamped types.

use super::validate::derive_ir_builtin_output_type;
use super::*;

/// Name-keyed shape evidence for the validation scope being walked.
///
/// Every access is a keyed lookup, insert, or remove, and a binder's facts
/// live for one [`ScopedMap::scope`]: no caller iterates this environment,
/// and none may start to. The facts here decide a checker verdict, so an
/// iteration order that picked between two of them would be exactly the
/// chelis#1341 class. `ScopedMap` has no iteration exit at all. It also has
/// no copy, because copying the environment at every binder made a chain of
/// `k` nested `let`s cost `k` squared in time and memory (chelis#3740).
pub(super) type ShapeTypeEnv = ScopedMap<String, deep::Expr>;

pub(super) fn shape_type_env(type_env: &IrTypeEnv) -> ShapeTypeEnv {
    type_env
        .iter()
        .map(|(name, ty)| (name.clone(), ty.clone()))
        .collect()
}

/// Bind a Surf function's parameter names to the standalone `defsig`
/// parameter types visible in the same validation unit. Surf deliberately
/// keeps those types on `defsig` rather than duplicating them into `(params
/// ...)`; without this bridge, the post-inference shape validator loses every
/// parameter rank at the function boundary.
pub(super) fn bind_declared_fn_params(
    fn_expr: &deep::Expr,
    parameter_types: &[deep::Expr],
    type_env: &mut ShapeTypeEnv,
    operation: &crate::resources::Operation<'_>,
) {
    let Some((DeepTag::Fn, _, fn_children)) = stamped_parts(fn_expr) else {
        return;
    };
    let Some(params_expr) = fn_children.first() else {
        return;
    };
    let Some((DeepTag::Params, _, params)) = stamped_parts(params_expr) else {
        return;
    };
    for (param, parameter_type) in params.iter().zip(parameter_types) {
        if let Some(name) = param_name_for_refs(param, operation) {
            type_env.insert(name, parameter_type.clone());
        }
    }
}

/// Resolve the tensor type expression of a callsite argument, peeking through
/// a `(borrow {} <inner>)` wrapper if present.
pub(super) fn arg_tensor_type_expr(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
    operation: &crate::resources::Operation<'_>,
) -> Option<deep::Expr> {
    expr_shape_type_fact(peel_borrow(expr, operation), type_env, operation)
}

pub(super) fn expr_shape_type_fact(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
    operation: &crate::resources::Operation<'_>,
) -> Option<deep::Expr> {
    stack_guard!(operation, "expr_shape_type_fact", expr, None);
    match expr {
        deep::Expr::Node(node, _) => {
            if node.tag() == DeepTag::Var
                && let Some(name) = node.children_slice().first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::MetaExpr(meta, _) => expr_shape_type_fact(&meta.expr, type_env, operation),
        _ => None,
    }
}

pub(super) fn peel_borrow<'source>(
    expr: &'source deep::Expr,
    operation: &crate::resources::Operation<'_>,
) -> &'source deep::Expr {
    stack_guard!(operation, "peel_borrow", expr, expr);
    if let Some((DeepTag::Borrow, _, kids)) = stamped_parts(expr)
        && let Some(child) = kids.first()
    {
        return peel_borrow(child, operation);
    }
    expr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeepDimKind {
    Lit(i64),
    NonConcrete,
}

pub(super) fn tensor_dims_from_type_expr(expr: &deep::Expr) -> Option<Vec<DeepDimKind>> {
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag == DeepTag::TRef {
        return kids.first().and_then(tensor_dims_from_type_expr);
    }
    if tag != DeepTag::TTensor {
        return None;
    }
    if kids.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    for kid in &kids[..kids.len().saturating_sub(1)] {
        dims.push(match stamped_parts(kid) {
            Some((DeepTag::DLit, _, dim_kids)) => match dim_kids.first() {
                Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => DeepDimKind::Lit(*n),
                _ => DeepDimKind::NonConcrete,
            },
            _ => DeepDimKind::NonConcrete,
        });
    }
    Some(dims)
}

/// Record shape facts in the canonical lexical scope.
pub(super) fn record_let_binding_shape_fact(
    name: &str,
    value_expr: &deep::Expr,
    type_env: &mut ShapeTypeEnv,
    static_env: &StaticEnv,
    operation: &crate::resources::Operation<'_>,
) {
    // If the RHS is a shape-sensitive IR builtin
    // whose output type is derivable from its args,
    // register the derived type so downstream uses
    // of `name` resolve correctly.
    let derived = derive_ir_builtin_output_type(value_expr, type_env, static_env, operation);
    match derived {
        Some(ty) => {
            type_env.insert(name.to_string(), ty);
        }
        None => {
            // Drop any fact this name carried from
            // an outer `def` or an earlier binding.
            // `IrTypeEnv` holds only top-level defs,
            // so an entry standing here describes a
            // DIFFERENT binding than the one being
            // introduced; leaving it in place lets a
            // rebinding inherit the previous
            // binding's shape (chelis#668 round-6
            // F1, when the identity-rank validator
            // still read this environment).
            type_env.remove(name);
        }
    }
}
