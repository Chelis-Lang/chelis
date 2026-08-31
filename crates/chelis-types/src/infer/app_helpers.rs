//! Common application helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn auto_borrow_call_arg_types(
    func_ty: &Type,
    arg_tys: Vec<Type>,
    subst: &Subst,
) -> Vec<Type> {
    let Type::Fn(params, _) = subst.apply(func_ty) else {
        return arg_tys;
    };
    arg_tys
        .into_iter()
        .enumerate()
        .map(
            |(index, actual)| match params.get(index).map(|param| subst.apply(param)) {
                Some(Type::Ref(_)) if !matches!(subst.apply(&actual), Type::Ref(_)) => {
                    Type::Ref(Box::new(actual))
                }
                _ => actual,
            },
        )
        .collect()
}

/// Implicit-copy fan-out v3 Shape A relaxation: when a def's body's
/// tail-position expression is a bare `(var x)` reference (possibly
/// wrapped in `let`, `if`, or `match` structures whose sibling branches
/// all return the same name) and the body's inferred return type is
/// `Ref(R)` while the declared return is owned `R`, return a relaxed
/// declared type `Fn(params, Ref(R))` so the def-body unify can succeed.
/// Returns `None` for any other body shape; the caller surfaces the
/// existing TypeMismatch in that case.
///
/// The PR #91 (W4-A) version of this helper accepted only a bare
/// `(fn (params...) (var x))` body.  0.7.9 broadens the gate to walk
/// `let`/`if`/`match` tail-position structures via
/// `descend_to_tail_var`, closing `Linearity-ShapeABroadReturn-F1`.
pub(super) fn shape_a_relaxed_return(
    body_expr: &deep::Expr,
    body_ty: &Type,
    decl_ty: &Type,
) -> Option<Type> {
    // body is the def's body, which the desugarer wraps as
    // `(fn (params ...) body_inner)` whenever the def has params.  Walk
    // the inner expression's tail position to confirm it resolves to a
    // bare `(var name)` reference across every reachable sibling.
    let body_list = match body_expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(body_list) != Some(DeepTag::Fn) {
        return None;
    }
    let inner = children(body_list).get(1)?;
    descend_to_tail_var(inner)?;

    // The body's inferred type and the declared type both must be
    // `Fn(params, ret)` with matching params and a return-position
    // mismatch of exactly `Ref(R)` (body) vs `R` (decl).
    let (Type::Fn(body_params, body_ret), Type::Fn(decl_params, decl_ret)) = (body_ty, decl_ty)
    else {
        return None;
    };
    if body_params.len() != decl_params.len() {
        return None;
    }
    let Type::Ref(body_inner_ret) = body_ret.as_ref() else {
        return None;
    };
    // The body's unwrapped return must match the declared return
    // structurally before we allow the relaxation; otherwise the
    // relaxed unify would still fail and only the loop would change.
    if !types_structurally_equal(body_inner_ret.as_ref(), decl_ret.as_ref()) {
        return None;
    }
    Some(Type::Fn(
        decl_params.clone(),
        Box::new(Type::Ref(Box::new(decl_ret.as_ref().clone()))),
    ))
}

/// Descend through `let`, `if`, and `match` to a tail-position
/// `(var name)` reference.  Returns `Some(name)` when every sibling
/// branch resolves to the same bare-var name, `None` otherwise.
///
/// This is the broader-Shape-A coverage closure for
/// `Linearity-ShapeABroadReturn-F1`.  The rules:
///
/// - `(var x)` returns `Some("x")` (the leaf case from PR #91).
/// - `(let bind body)` recurses into `body` (the second child).
/// - `(if cond then_e else_e)` recurses into both branches; both must
///   resolve to the same name.
/// - `(match scrutinee arm ...)` recurses into every arm body (the
///   third child of each `(arm pattern guard body)` triple); all arms
///   must resolve to the same name.
/// - Otherwise returns `None`.
///
/// The descent is type-agnostic; the surrounding logic in
/// `shape_a_relaxed_return` already verifies that the body's inferred
/// return type is `Ref(R)` and the declared return is `R` structurally.
///
/// The "same name across siblings" requirement is intentional: the
/// existing relaxation is justified by the caller's borrow lifetime
/// already covering the parameter being returned.  Heterogeneous
/// bare-var returns would extend the relaxation beyond v3 scope and
/// need a richer coercion story.
pub(super) fn descend_to_tail_var(expr: &deep::Expr) -> Option<&str> {
    stack_guard!("descend_to_tail_var", expr, None);
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    match get_tag(list) {
        Some(DeepTag::Var) => var_name_list(list),
        Some(DeepTag::Let) => {
            let body = children(list).get(1)?;
            descend_to_tail_var(body)
        }
        Some(DeepTag::If) => {
            let kids = children(list);
            let then_e = kids.get(1)?;
            let else_e = kids.get(2)?;
            let then_name = descend_to_tail_var(then_e)?;
            let else_name = descend_to_tail_var(else_e)?;
            if then_name == else_name {
                Some(then_name)
            } else {
                None
            }
        }
        Some(DeepTag::Match) => {
            let kids = children(list);
            // Skip the scrutinee (first child); every remaining child is
            // expected to be an `(arm pattern guard body)` triple.
            let arms = kids.get(1..)?;
            if arms.is_empty() {
                return None;
            }
            let mut name: Option<&str> = None;
            for arm in arms {
                let arm_list = match arm {
                    deep::Expr::List(list, _) => list,
                    _ => return None,
                };
                if get_tag(arm_list) != Some(DeepTag::Arm) {
                    return None;
                }
                let arm_body = children(arm_list).get(2)?;
                let arm_name = descend_to_tail_var(arm_body)?;
                match name {
                    None => name = Some(arm_name),
                    Some(prev) if prev == arm_name => {}
                    Some(_) => return None,
                }
            }
            name
        }
        _ => None,
    }
}

/// Two tensor dimensions are *identical* for structural-equality
/// purposes iff they denote the same dimension: same concrete
/// `Dim::Name`, same `Dim::Lit`, or the same `Dim::Var`.  `Dim::Wildcard`
/// matches anything on either side (it is the permissive "unknown"
/// sentinel, consistent with `unify_dim`).
///
/// Two *distinct* symbolic dim variables (`n` vs `m`) are NOT identical
/// even though both contribute rank 1.  This is the soundness fix for
/// `TypeCheck-FreeDimVarUnification-F1` (SR-LEAK-A): the Shape A
/// relaxed-retry must not accept a body whose return dim diverges from
/// the declared return dim.
///
/// Name <-> Lit (issue Chelis-Lang/chelis#219, Option A): mirrors
/// `unify_dim`'s permissive Name <-> Lit arm. The Shape A relaxed-
/// retry's structural check must agree with `unify_dim` so the
/// retry path doesn't silently reject a callee-shape pairing that
/// the call-site unification would accept.
pub(super) fn dims_identical(d1: &Dim, d2: &Dim) -> bool {
    match (d1, d2) {
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => true,
        (Dim::Name(n1), Dim::Name(n2)) => n1 == n2,
        (Dim::Lit(l1), Dim::Lit(l2)) => l1 == l2,
        (Dim::Var(v1), Dim::Var(v2)) => v1 == v2,
        // Issue #219 Option A: Name and Lit count as identical for
        // the Shape A relaxed-retry's structural check.
        (Dim::Name(_), Dim::Lit(_)) | (Dim::Lit(_), Dim::Name(_)) => true,
        _ => false,
    }
}

/// Structural type equality used by `shape_a_relaxed_return` to guard
/// the relaxed retry: the relaxation is only safe when the body and
/// declared return differ exactly by a top-level `Ref` wrapper, so the
/// dim structure underneath must match by *identity*, not just by rank.
///
/// Tensor dims are compared with `dims_identical`: same concrete name,
/// same literal, or the same dim variable.  Distinct dim variables do
/// not match (`TypeCheck-FreeDimVarUnification-F1`).
pub(super) fn types_structurally_equal(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Unit, Type::Unit) => true,
        (Type::Prim(p1), Type::Prim(p2)) => p1 == p2,
        (Type::Ref(i1), Type::Ref(i2)) => types_structurally_equal(i1, i2),
        (Type::Tensor(d1, p1), Type::Tensor(d2, p2)) => {
            p1 == p2
                && d1.len() == d2.len()
                && d1.iter().zip(d2.iter()).all(|(x, y)| dims_identical(x, y))
        }
        (Type::Tuple(es1), Type::Tuple(es2)) => {
            es1.len() == es2.len()
                && es1
                    .iter()
                    .zip(es2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
        }
        (Type::Adt(n1, a1), Type::Adt(n2, a2)) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
        }
        (Type::KindedAdt(n1, a1), Type::KindedAdt(n2, a2)) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1.iter().zip(a2).all(|(left, right)| match (left, right) {
                    (NominalArg::Type(left), NominalArg::Type(right)) => {
                        types_structurally_equal(left, right)
                    }
                    (NominalArg::Dimension(left), NominalArg::Dimension(right)) => {
                        dims_identical(left, right)
                    }
                    _ => false,
                })
        }
        (Type::Fn(p1, r1), Type::Fn(p2, r2)) => {
            p1.len() == p2.len()
                && p1
                    .iter()
                    .zip(p2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
                && types_structurally_equal(r1, r2)
        }
        (Type::Var(_), Type::Var(_)) => true,
        (Type::Error(_), _) | (_, Type::Error(_)) => true,
        _ => false,
    }
}

pub(super) fn type_for_readonly_check(ty: &Type, subst: &Subst) -> Type {
    match subst.apply(ty) {
        Type::Ref(inner) => subst.apply(&inner),
        other => other,
    }
}
