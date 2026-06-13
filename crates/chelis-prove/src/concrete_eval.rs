//! Concrete (native `f64`) evaluation of [`SmtExpr`] terms.
//!
//! This is the predicate evaluator the Tier C fuzzer uses to check a
//! candidate sample against a property's preconditions and postcondition
//! without invoking the solver. It is lifted here (RFC D-STARVE / W4
//! Unit a) so the opaque-binder rejection-sampling and constructor-based
//! generation paths can reuse the exact same evaluation semantics over a
//! flattened invariant predicate: a generation method is only ever a
//! proposal distribution, and every accepted sample is validated against
//! the predicate via [`eval_bool`] here, so the two surfaces agree by
//! construction.
//!
//! Semantics notes (kept identical to the pre-lift Tier C behavior so
//! `tier_c` re-exports are byte-for-byte equivalent):
//!
//! - division by zero yields `NaN` (the candidate is then rejected by
//!   the caller, matching the D-WF partial-eval rule for Tier C);
//! - `==`/`!=` use a `1e-10` tolerance in this *fuzz* evaluator. This is
//!   the legacy user-property comparison semantics; it is deliberately
//!   NOT used for invariant-sample acceptance, which validates strictly
//!   (RFC D-STARVE "exact float equality starves by design"). The
//!   strict-validation path supplies its own predicate built so that the
//!   acceptance test is the predicate's own boolean result, never an
//!   epsilon-relaxed comparison standing in for it.

use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr};
use std::collections::HashMap;

/// Evaluate a boolean-shaped [`SmtExpr`] under a concrete variable
/// environment. Arithmetic expressions in boolean position are truthy
/// when nonzero (legacy Tier C behavior).
pub fn eval_bool(expr: &SmtExpr, env: &HashMap<String, f64>) -> bool {
    match expr {
        SmtExpr::BoolLit(v) => *v,
        SmtExpr::Cmp(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            match op {
                CmpOp::Lt => l < r,
                CmpOp::Le => l <= r,
                CmpOp::Gt => l > r,
                CmpOp::Ge => l >= r,
                CmpOp::Eq => (l - r).abs() < 1e-10,
                CmpOp::Ne => (l - r).abs() >= 1e-10,
            }
        }
        SmtExpr::Bool(BoolOp::And, children) => children.iter().all(|c| eval_bool(c, env)),
        SmtExpr::Bool(BoolOp::Or, children) => children.iter().any(|c| eval_bool(c, env)),
        SmtExpr::Bool(BoolOp::Implies, children) if children.len() == 2 => {
            !eval_bool(&children[0], env) || eval_bool(&children[1], env)
        }
        SmtExpr::Not(inner) => !eval_bool(inner, env),
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool(cond, env) {
                eval_bool(then_e, env)
            } else {
                eval_bool(else_e, env)
            }
        }
        // Arithmetic expressions used in boolean context: nonzero = true
        _ => eval_arith(expr, env) != 0.0,
    }
}

/// Evaluate an arithmetic-shaped [`SmtExpr`] under a concrete variable
/// environment. Unbound variables read as `0.0`; division by zero and
/// out-of-domain transcendentals yield `NaN`.
pub fn eval_arith(expr: &SmtExpr, env: &HashMap<String, f64>) -> f64 {
    match expr {
        SmtExpr::Var(name) => env.get(name).copied().unwrap_or(0.0),
        SmtExpr::RealLit(v) => *v,
        SmtExpr::IntLit(v) => *v as f64,
        SmtExpr::BoolLit(v) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        SmtExpr::Arith(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            match op {
                ArithOp::Add => l + r,
                ArithOp::Sub => l - r,
                ArithOp::Mul => l * r,
                ArithOp::Div => {
                    if r != 0.0 {
                        l / r
                    } else {
                        f64::NAN
                    }
                }
                ArithOp::Neg => -l,
            }
        }
        SmtExpr::Apply(name, args) => {
            let a: Vec<f64> = args.iter().map(|a| eval_arith(a, env)).collect();
            match name.as_str() {
                "exp" => a[0].exp(),
                "log" => a[0].ln(),
                "sqrt" => a[0].sqrt(),
                "sin" => a[0].sin(),
                "cos" => a[0].cos(),
                "abs" => a[0].abs(),
                "min" if a.len() == 2 => a[0].min(a[1]),
                "max" if a.len() == 2 => a[0].max(a[1]),
                _ => f64::NAN,
            }
        }
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool(cond, env) {
                eval_arith(then_e, env)
            } else {
                eval_arith(else_e, env)
            }
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            let result = match op {
                CmpOp::Lt => l < r,
                CmpOp::Le => l <= r,
                CmpOp::Gt => l > r,
                CmpOp::Ge => l >= r,
                CmpOp::Eq => (l - r).abs() < 1e-10,
                CmpOp::Ne => (l - r).abs() >= 1e-10,
            };
            if result { 1.0 } else { 0.0 }
        }
        SmtExpr::Bool(_, _) | SmtExpr::Not(_) => {
            if eval_bool(expr, env) {
                1.0
            } else {
                0.0
            }
        }
        SmtExpr::Forall(_, _) | SmtExpr::Exists(_, _) => f64::NAN, // unreachable after fuzzability check
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SmtExpr;

    fn env(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn eval_bool_comparison_holds() {
        let e = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        assert!(eval_bool(&e, &env(&[("x", 1.0)])));
        assert!(!eval_bool(&e, &env(&[("x", -1.0)])));
    }

    #[test]
    fn eval_arith_div_by_zero_is_nan() {
        let e = SmtExpr::Arith(
            ArithOp::Div,
            Box::new(SmtExpr::RealLit(1.0)),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        assert!(eval_arith(&e, &env(&[])).is_nan());
    }

    #[test]
    fn eval_bool_and_or() {
        let t = SmtExpr::BoolLit(true);
        let f = SmtExpr::BoolLit(false);
        assert!(eval_bool(
            &SmtExpr::Bool(BoolOp::And, vec![t.clone(), t.clone()]),
            &env(&[])
        ));
        assert!(!eval_bool(
            &SmtExpr::Bool(BoolOp::And, vec![t.clone(), f.clone()]),
            &env(&[])
        ));
        assert!(eval_bool(&SmtExpr::Bool(BoolOp::Or, vec![t, f]), &env(&[])));
    }
}
