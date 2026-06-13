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

/// Evaluate a boolean-shaped [`SmtExpr`] with the legacy *fuzz*
/// comparison semantics: `==`/`!=` carry a `1e-10` tolerance. This is the
/// user-property postcondition evaluator and must NOT be used for
/// invariant-sample acceptance (use [`eval_bool_strict`] there).
pub fn eval_bool(expr: &SmtExpr, env: &HashMap<String, f64>) -> bool {
    eval_bool_with(expr, env, false)
}

/// Evaluate a boolean-shaped [`SmtExpr`] with STRICT comparison semantics:
/// `==`/`!=` are exact IEEE comparisons with no tolerance. This is the
/// evaluator invariant-sample validation must use (RFC D-STARVE: "exact
/// float equality starves by design"; epsilon-validated samples would
/// weaken exactly the soundness that validation provides). NaN operands
/// compare false under both `==` and (per IEEE) yield `true` for `!=`, so
/// a NaN representation never spuriously satisfies an equality invariant.
pub fn eval_bool_strict(expr: &SmtExpr, env: &HashMap<String, f64>) -> bool {
    eval_bool_with(expr, env, true)
}

fn eval_bool_with(expr: &SmtExpr, env: &HashMap<String, f64>, strict: bool) -> bool {
    match expr {
        SmtExpr::BoolLit(v) => *v,
        SmtExpr::Cmp(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            eval_cmp(*op, l, r, strict)
        }
        SmtExpr::Bool(BoolOp::And, children) => {
            children.iter().all(|c| eval_bool_with(c, env, strict))
        }
        SmtExpr::Bool(BoolOp::Or, children) => {
            children.iter().any(|c| eval_bool_with(c, env, strict))
        }
        SmtExpr::Bool(BoolOp::Implies, children) if children.len() == 2 => {
            !eval_bool_with(&children[0], env, strict) || eval_bool_with(&children[1], env, strict)
        }
        SmtExpr::Not(inner) => !eval_bool_with(inner, env, strict),
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool_with(cond, env, strict) {
                eval_bool_with(then_e, env, strict)
            } else {
                eval_bool_with(else_e, env, strict)
            }
        }
        // Arithmetic expressions used in boolean context: nonzero = true
        _ => eval_arith(expr, env) != 0.0,
    }
}

/// Evaluate a single comparison. `strict` selects exact IEEE `==`/`!=`
/// (invariant validation) vs the `1e-10`-tolerant fuzz comparison.
fn eval_cmp(op: CmpOp, l: f64, r: f64, strict: bool) -> bool {
    match op {
        CmpOp::Lt => l < r,
        CmpOp::Le => l <= r,
        CmpOp::Gt => l > r,
        CmpOp::Ge => l >= r,
        CmpOp::Eq => {
            if strict {
                l == r
            } else {
                (l - r).abs() < 1e-10
            }
        }
        CmpOp::Ne => {
            if strict {
                l != r
            } else {
                (l - r).abs() >= 1e-10
            }
        }
    }
}

/// Evaluate an arithmetic-shaped [`SmtExpr`] under a concrete variable
/// environment with the fuzz comparison semantics (`1e-10` tolerance for
/// any nested `==`/`!=`). Unbound variables read as `0.0`; division by
/// zero and out-of-domain transcendentals yield `NaN`.
pub fn eval_arith(expr: &SmtExpr, env: &HashMap<String, f64>) -> f64 {
    eval_arith_with(expr, env, false)
}

fn eval_arith_with(expr: &SmtExpr, env: &HashMap<String, f64>, strict: bool) -> f64 {
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
            let l = eval_arith_with(left, env, strict);
            let r = eval_arith_with(right, env, strict);
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
            let a: Vec<f64> = args
                .iter()
                .map(|x| eval_arith_with(x, env, strict))
                .collect();
            apply_intrinsic(name, &a)
        }
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool_with(cond, env, strict) {
                eval_arith_with(then_e, env, strict)
            } else {
                eval_arith_with(else_e, env, strict)
            }
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = eval_arith_with(left, env, strict);
            let r = eval_arith_with(right, env, strict);
            if eval_cmp(*op, l, r, strict) {
                1.0
            } else {
                0.0
            }
        }
        SmtExpr::Bool(_, _) | SmtExpr::Not(_) => {
            if eval_bool_with(expr, env, strict) {
                1.0
            } else {
                0.0
            }
        }
        SmtExpr::Forall(_, _) | SmtExpr::Exists(_, _) => f64::NAN, // unreachable after fuzzability check
    }
}

/// Apply a whitelisted unary/binary intrinsic to its already-evaluated
/// arguments.
fn apply_intrinsic(name: &str, a: &[f64]) -> f64 {
    match name {
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

    fn ne_half() -> SmtExpr {
        // value != 0.5
        SmtExpr::Cmp(
            CmpOp::Ne,
            Box::new(SmtExpr::Var("value".into())),
            Box::new(SmtExpr::RealLit(0.5)),
        )
    }

    #[test]
    fn cr2_strict_ne_rejects_exact_and_accepts_near() {
        // CR-2/CR-5/CR-10: invariant-sample validation must be STRICT. The
        // invariant `value != 0.5` rejects 0.5 exactly and accepts a value
        // a hair off (0.5 + 5e-11) -- the fuzz 1e-10 tolerance would wrongly
        // reject the near value (treating it as == 0.5).
        let e = ne_half();
        assert!(
            !eval_bool_strict(&e, &env(&[("value", 0.5)])),
            "0.5 exactly fails `!= 0.5`"
        );
        assert!(
            eval_bool_strict(&e, &env(&[("value", 0.5 + 5e-11)])),
            "0.5 + 5e-11 satisfies `!= 0.5` under strict semantics"
        );
    }

    #[test]
    fn cr2_fuzz_tolerance_is_preserved_for_eval_bool() {
        // The fuzz evaluator keeps the 1e-10 tolerance (user-property
        // postcondition semantics) -- 0.5 + 5e-11 reads as == 0.5, so
        // `!= 0.5` is false under the tolerant path. This guards that the
        // strict path is a NEW behavior, not a replacement of the fuzz one.
        let e = ne_half();
        assert!(
            !eval_bool(&e, &env(&[("value", 0.5 + 5e-11)])),
            "fuzz path keeps the 1e-10 tolerance"
        );
    }

    #[test]
    fn cr2_strict_eq_is_exact() {
        // value == 0.5 strict: exact only.
        let e = SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::Var("value".into())),
            Box::new(SmtExpr::RealLit(0.5)),
        );
        assert!(eval_bool_strict(&e, &env(&[("value", 0.5)])));
        assert!(!eval_bool_strict(&e, &env(&[("value", 0.5 + 5e-11)])));
    }
}
