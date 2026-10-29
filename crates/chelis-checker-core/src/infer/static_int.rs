//! Checked folding for compile-time integer extent expressions.

use super::*;

/// Fold the common integer-extent language used by the checker and IR
/// lowering. `resolve_name` supplies values for already-folded lexical
/// bindings; callers that do not have such bindings return `None`.
/// Every arithmetic step is checked, so overflow and division by zero refuse
/// the fold instead of manufacturing a wrapped extent.
pub fn fold_static_int_expr<F, S: chelis_stack_services::StackServices>(
    expr: &deep::Expr,
    resolve_name: F,
    resources: &crate::Resources<'_, S>,
) -> Option<i64>
where
    F: FnMut(&str) -> Option<i64>,
{
    resources
        .run_borrowed(|operation| fold_static_int_expr_in_operation(expr, resolve_name, operation))
}

pub(crate) fn fold_static_int_expr_in_operation<F>(
    expr: &deep::Expr,
    mut resolve_name: F,
    operation: &crate::resources::Operation<'_>,
) -> Option<i64>
where
    F: FnMut(&str) -> Option<i64>,
{
    fn walk(
        expr: &deep::Expr,
        resolve_name: &mut dyn FnMut(&str) -> Option<i64>,
        operation: &crate::resources::Operation<'_>,
    ) -> Option<i64> {
        if let Some(value) = extract_int_for_dim(expr, operation) {
            return Some(value);
        }
        let (tag, _, kids) = stamped_parts(expr)?;
        match tag {
            DeepTag::Var => resolve_name(symbolic_dim_ref_name(expr)?),
            DeepTag::Cast => walk(kids.first()?, resolve_name, operation),
            DeepTag::App => {
                let callee = kids.first()?;
                let operands = &kids[1..];
                if is_builtin_var(callee, "cast") {
                    return walk(operands.first()?, resolve_name, operation);
                }
                match operands {
                    [value] if is_builtin_var(callee, "neg") => {
                        walk(value, resolve_name, operation)?.checked_neg()
                    }
                    [lhs, rhs] if is_builtin_var(callee, "add") => walk(
                        lhs,
                        resolve_name,
                        operation,
                    )?
                    .checked_add(walk(rhs, resolve_name, operation)?),
                    [lhs, rhs] if is_builtin_var(callee, "sub") => walk(
                        lhs,
                        resolve_name,
                        operation,
                    )?
                    .checked_sub(walk(rhs, resolve_name, operation)?),
                    [lhs, rhs] if is_builtin_var(callee, "mul") => walk(
                        lhs,
                        resolve_name,
                        operation,
                    )?
                    .checked_mul(walk(rhs, resolve_name, operation)?),
                    [lhs, rhs] if is_builtin_var(callee, "floor_div") => walk(
                        lhs,
                        resolve_name,
                        operation,
                    )?
                    .checked_div_euclid(walk(rhs, resolve_name, operation)?),
                    [lhs, rhs]
                        if is_builtin_var(callee, "trunc_div") || is_builtin_var(callee, "div") =>
                    {
                        walk(lhs, resolve_name, operation)?.checked_div(walk(
                            rhs,
                            resolve_name,
                            operation,
                        )?)
                    }
                    [lhs, rhs] if is_builtin_var(callee, "mod") => walk(
                        lhs,
                        resolve_name,
                        operation,
                    )?
                    .checked_rem_euclid(walk(rhs, resolve_name, operation)?),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    walk(expr, &mut resolve_name, operation)
}
