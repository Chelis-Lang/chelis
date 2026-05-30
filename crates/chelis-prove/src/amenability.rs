//! SMT amenability classification.
//!
//! Classifies SMT expressions by arithmetic complexity to select the
//! appropriate solver logic (linear, polynomial, transcendental).

use crate::dispatch::SmtAmenability;
use crate::solver::SmtExpr;

/// Classify an SMT expression's amenability level.
pub fn classify(expr: &SmtExpr) -> SmtAmenability {
    if contains_transcendental(expr) {
        SmtAmenability::Transcendental
    } else if contains_nonlinear(expr) {
        SmtAmenability::Polynomial
    } else {
        SmtAmenability::Linear
    }
}

/// Check if an SmtExpr contains transcendental function calls (exp, sin, cos, sqrt, abs).
pub fn contains_transcendental(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Apply(name, args) => {
            matches!(name.as_str(), "exp" | "sin" | "cos" | "sqrt" | "abs")
                || args.iter().any(contains_transcendental)
        }
        SmtExpr::Arith(_, l, r) => contains_transcendental(l) || contains_transcendental(r),
        SmtExpr::Cmp(_, l, r) => contains_transcendental(l) || contains_transcendental(r),
        SmtExpr::Bool(_, children) => children.iter().any(contains_transcendental),
        SmtExpr::Not(inner) => contains_transcendental(inner),
        SmtExpr::Ite(c, t, e) => {
            contains_transcendental(c) || contains_transcendental(t) || contains_transcendental(e)
        }
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_transcendental(body),
        _ => false,
    }
}

/// Check if an SmtExpr contains nonlinear arithmetic (Mul where both sides
/// are non-constant).
fn contains_nonlinear(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Arith(crate::solver::ArithOp::Mul, l, r) => {
            let l_const = is_constant(l);
            let r_const = is_constant(r);
            (!l_const && !r_const) || contains_nonlinear(l) || contains_nonlinear(r)
        }
        SmtExpr::Arith(_, l, r) => contains_nonlinear(l) || contains_nonlinear(r),
        SmtExpr::Cmp(_, l, r) => contains_nonlinear(l) || contains_nonlinear(r),
        SmtExpr::Bool(_, children) => children.iter().any(contains_nonlinear),
        SmtExpr::Not(inner) => contains_nonlinear(inner),
        SmtExpr::Ite(c, t, e) => {
            contains_nonlinear(c) || contains_nonlinear(t) || contains_nonlinear(e)
        }
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_nonlinear(body),
        SmtExpr::Apply(_, args) => args.iter().any(contains_nonlinear),
        _ => false,
    }
}

/// An expression is constant if it's a literal.
fn is_constant(expr: &SmtExpr) -> bool {
    matches!(
        expr,
        SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{ArithOp, SmtExpr};

    #[test]
    fn linear_expression() {
        // x + 1
        let expr = SmtExpr::Arith(
            ArithOp::Add,
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::RealLit(1.0)),
        );
        assert_eq!(classify(&expr), SmtAmenability::Linear);
    }

    #[test]
    fn polynomial_expression() {
        // x * x
        let expr = SmtExpr::Arith(
            ArithOp::Mul,
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Var("x".into())),
        );
        assert_eq!(classify(&expr), SmtAmenability::Polynomial);
    }

    #[test]
    fn constant_mul_is_linear() {
        // 2 * x
        let expr = SmtExpr::Arith(
            ArithOp::Mul,
            Box::new(SmtExpr::RealLit(2.0)),
            Box::new(SmtExpr::Var("x".into())),
        );
        assert_eq!(classify(&expr), SmtAmenability::Linear);
    }

    #[test]
    fn transcendental_expression() {
        // exp(x)
        let expr = SmtExpr::Apply("exp".into(), vec![SmtExpr::Var("x".into())]);
        assert_eq!(classify(&expr), SmtAmenability::Transcendental);
    }
}
