//! Checked folding for compile-time integer extent expressions.

use super::*;

/// Fold the common integer-extent language used by the checker and IR
/// lowering. `resolve_name` supplies values for already-folded lexical
/// bindings; callers that do not have such bindings return `None`.
/// Every arithmetic step is checked, so overflow and division by zero refuse
/// the fold instead of manufacturing a wrapped extent.
pub fn fold_static_int_expr<F>(expr: &deep::Expr, mut resolve_name: F) -> Option<i64>
where
    F: FnMut(&str) -> Option<i64>,
{
    fn walk(expr: &deep::Expr, resolve_name: &mut dyn FnMut(&str) -> Option<i64>) -> Option<i64> {
        if let Some(value) = extract_int_for_dim(expr) {
            return Some(value);
        }
        let (tag, _, kids) = stamped_parts(expr)?;
        match tag {
            DeepTag::Var => resolve_name(symbolic_dim_ref_name(expr)?),
            DeepTag::Cast => walk(kids.first()?, resolve_name),
            DeepTag::App => {
                let callee = kids.first()?;
                let operands = &kids[1..];
                if is_builtin_var(callee, "cast") {
                    return walk(operands.first()?, resolve_name);
                }
                match operands {
                    [value] if is_builtin_var(callee, "neg") => {
                        walk(value, resolve_name)?.checked_neg()
                    }
                    [lhs, rhs] if is_builtin_var(callee, "add") => {
                        walk(lhs, resolve_name)?.checked_add(walk(rhs, resolve_name)?)
                    }
                    [lhs, rhs] if is_builtin_var(callee, "sub") => {
                        walk(lhs, resolve_name)?.checked_sub(walk(rhs, resolve_name)?)
                    }
                    [lhs, rhs] if is_builtin_var(callee, "mul") => {
                        walk(lhs, resolve_name)?.checked_mul(walk(rhs, resolve_name)?)
                    }
                    [lhs, rhs] if is_builtin_var(callee, "floor_div") => {
                        walk(lhs, resolve_name)?.checked_div_euclid(walk(rhs, resolve_name)?)
                    }
                    [lhs, rhs]
                        if is_builtin_var(callee, "trunc_div") || is_builtin_var(callee, "div") =>
                    {
                        walk(lhs, resolve_name)?.checked_div(walk(rhs, resolve_name)?)
                    }
                    [lhs, rhs] if is_builtin_var(callee, "mod") => {
                        walk(lhs, resolve_name)?.checked_rem_euclid(walk(rhs, resolve_name)?)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    walk(expr, &mut resolve_name)
}
