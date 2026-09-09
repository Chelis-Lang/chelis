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

/// Name-keyed shape evidence for one validation scope.
///
/// Every access is a keyed lookup, insert, or whole-env clone: no caller
/// iterates this environment, and none may start to. The facts here decide a
/// checker verdict, so an iteration order that picked between two of them
/// would be exactly the chelis#1341 class. `UnordMap` has no iteration exit
/// that is not a written `to_sorted` claim of C3.1 authority, which a name
/// keyed by an arbitrary source identifier cannot make; `BTreeMap` would
/// leave `iter` and `keys` open to a later edit.
pub(super) type ShapeTypeEnv = UnordMap<String, deep::Expr>;

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
            scoped.insert(name, parameter_type.clone());
        }
    }
    scoped
}

/// Resolve the tensor type expression of a callsite argument, peeking through
/// a `(borrow {} <inner>)` wrapper if present.
pub(super) fn arg_tensor_type_expr(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
) -> Option<deep::Expr> {
    expr_shape_type_fact(peel_borrow(expr), type_env)
}

pub(super) fn expr_shape_type_fact(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
) -> Option<deep::Expr> {
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

/// Record what the validator can prove about one `let` binding's shape.
///
/// Splitting this out of the `Let` arm keeps shape-fact bookkeeping in the
/// module that owns `ShapeTypeEnv`, and keeps `validate.rs` inside the
/// `source_arch` line budget that told us to divide by responsibility rather
/// than trim.
pub(super) fn record_let_binding_shape_fact(
    name: &str,
    value_expr: &deep::Expr,
    type_env: &mut ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
    failed_let_names: &mut UnordSet<String>,
) {
    // If the RHS is a shape-sensitive IR builtin
    // whose output type is derivable from its args,
    // register the derived type so downstream uses
    // of `name` resolve correctly.
    let derived = derive_ir_builtin_output_type(value_expr, type_env, static_env);
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
            // Mark as failed-derivation when the
            // RHS is structurally a recognized
            // shape-sensitive form (a known
            // shape-sensitive builtin or a
            // unary/binary passthrough wrapper
            // around one, recursively) but its
            // output type could not be derived.
            // This catches `y = conv(bad)`
            // and the R3 F-A passthrough cases
            // like `y = relu(conv(bad))`.
            //
            // RT-205 round-4 / issue #212: the
            // previous guard checked
            // `errors.len() > errs_before` to
            // detect an errored RHS, which fails
            // for chains of length 3+ because
            // cascade suppression already
            // silences the level-2 RHS's
            // diagnostic, so the level-2 name is
            // never marked and the level-3 RHS
            // re-emits a phantom error. The
            // structural check
            // `let_rhs_is_recognized_shape_sensitive`
            // does not depend on diagnostic
            // count and propagates the failed
            // marker unboundedly down the chain.
            //
            // The recognition is intentionally
            // narrow: a clean RHS that is not
            // a recognized shape-sensitive form
            // (e.g. a user-defined fn call) still
            // does NOT cause suppression
            // downstream, so legitimate
            // "really wrong arg" cases still
            // surface their own diagnostic.
            if let deep::Expr::List(_, _) = value_expr
                && let_rhs_is_recognized_shape_sensitive(value_expr, static_env)
            {
                failed_let_names.insert(name.to_string());
            }
        }
    }
}
