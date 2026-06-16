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
//! - `==`/`!=` carry a `1e-10` tolerance ONLY in the fuzz evaluator
//!   [`eval_bool`] (the user-property postcondition semantics).
//!   Invariant-sample acceptance uses [`eval_bool_strict`], which compares
//!   `==`/`!=` exactly (no tolerance), because an epsilon-validated sample
//!   would weaken exactly the soundness that validation provides (RFC
//!   D-STARVE "exact float equality starves by design").

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
            // Thread `strict` into the operands so a nested `==`/`!=` (or an
            // `ite` whose condition compares for equality) used as a Cmp
            // operand keeps the exact-equality semantics under
            // `eval_bool_strict`, instead of silently reverting to the
            // 1e-10 fuzz tolerance via `eval_arith` (strict = false).
            let l = eval_arith_with(left, env, strict);
            let r = eval_arith_with(right, env, strict);
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
        // Non-binary `Implies` (or any other malformed boolean connective)
        // must NOT fall through to the arithmetic catch-all below: that path
        // re-enters `eval_arith_with` -> its `Bool(_, _)` arm -> back here,
        // which would recurse without bound and overflow the stack. Evaluate
        // each connective with a recursion-free, conservative reading:
        //   - `Implies` with != 2 children: treat the last child as the
        //     consequent and the conjunction of the rest as the antecedent;
        //     a 0/1-child `Implies` has no antecedent, so it reduces to its
        //     consequent (true when empty).
        //   - `And`/`Or` are already handled above and are well-defined for
        //     0 or 1 child (And of nothing = true, Or of nothing = false).
        SmtExpr::Bool(BoolOp::Implies, children) => {
            let antecedent = children
                .split_last()
                .map(|(_, rest)| rest.iter().all(|c| eval_bool_with(c, env, strict)))
                .unwrap_or(true);
            let consequent = children
                .last()
                .map(|c| eval_bool_with(c, env, strict))
                .unwrap_or(true);
            !antecedent || consequent
        }
        SmtExpr::Not(inner) => !eval_bool_with(inner, env, strict),
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool_with(cond, env, strict) {
                eval_bool_with(then_e, env, strict)
            } else {
                eval_bool_with(else_e, env, strict)
            }
        }
        // Genuinely arithmetic-shaped expressions used in boolean context:
        // nonzero = true. Thread `strict` so any nested `==`/`!=` inside the
        // arithmetic keeps exact-equality semantics under
        // `eval_bool_strict`. Every boolean-shaped variant (`Bool` in all
        // its forms, `Not`, `Cmp`, `Ite`, `BoolLit`) is handled above, so
        // this arm only ever sees arithmetic variants and cannot re-enter
        // `eval_bool_with` for the same `expr` (no unbounded recursion).
        _ => eval_arith_with(expr, env, strict) != 0.0,
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
/// arguments. Out-of-grammar names and wrong arity yield `NaN` (the
/// candidate is then rejected / the predicate is unsatisfied) rather than
/// panicking on an out-of-bounds index (CR-13): the unary intrinsics now
/// guard `len == 1` exactly as `min`/`max` guard `len == 2`.
fn apply_intrinsic(name: &str, a: &[f64]) -> f64 {
    match name {
        "exp" if a.len() == 1 => a[0].exp(),
        "log" if a.len() == 1 => a[0].ln(),
        "sqrt" if a.len() == 1 => a[0].sqrt(),
        "sin" if a.len() == 1 => a[0].sin(),
        "cos" if a.len() == 1 => a[0].cos(),
        "abs" if a.len() == 1 => a[0].abs(),
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

    #[test]
    fn cr13_zero_arg_intrinsic_does_not_panic() {
        // CR-13: a malformed predicate with a zero-arg intrinsic
        // application (`exp()` with no args) must yield NaN / a clean
        // rejection, never an out-of-bounds index panic on a[0].
        let e = SmtExpr::Apply("exp".into(), vec![]);
        assert!(eval_arith(&e, &env(&[])).is_nan(), "zero-arg exp() is NaN");
        // In boolean position it must also not panic.
        let cmp = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Apply("sqrt".into(), vec![])),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        // NaN >= 0.0 is false; the point is it returns without panicking.
        assert!(!eval_bool(&cmp, &env(&[])));
    }

    #[test]
    fn cr13_wrong_arity_binary_intrinsic_is_nan() {
        // min/max already guarded len==2; a one-arg min() must be NaN, not
        // a panic.
        let e = SmtExpr::Apply("min".into(), vec![SmtExpr::RealLit(1.0)]);
        assert!(eval_arith(&e, &env(&[])).is_nan());
    }

    #[test]
    fn strict_propagates_into_cmp_operand_nested_equality() {
        // FINDING #1: strict `==`/`!=` semantics must propagate into the
        // OPERANDS of an outer comparison, not just the top-level Cmp.
        //
        // Build a nested equality `eq(a, b)` with a = 0.0, b = 1e-11 so that
        // |a - b| = 1e-11 sits JUST UNDER the 1e-10 fuzz tolerance:
        //   - strict: a == b is false  -> the eq operand reads 0.0
        //   - fuzz:   |a - b| < 1e-10  -> the eq operand reads 1.0
        // Wrap it in an `ite(eq(a, b), 1.0, 0.0)` arithmetic operand of an
        // outer comparison `ite(...) >= 0.5`:
        //   - strict outer: 0.0 >= 0.5 -> false (EXACT-equality answer)
        //   - fuzz   outer: 1.0 >= 0.5 -> true
        // Before the fix the Cmp arm called eval_arith (strict = false), so
        // eval_bool_strict wrongly returned the fuzz answer (true).
        let inner_eq = SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::Var("a".into())),
            Box::new(SmtExpr::Var("b".into())),
        );
        let ite_operand = SmtExpr::Ite(
            Box::new(inner_eq),
            Box::new(SmtExpr::RealLit(1.0)),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        let outer = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(ite_operand),
            Box::new(SmtExpr::RealLit(0.5)),
        );
        let e = env(&[("a", 0.0), ("b", 1e-11)]);
        assert!(
            !eval_bool_strict(&outer, &e),
            "strict must propagate into the Cmp operand: a == b is exactly false, so ite -> 0.0 and 0.0 >= 0.5 is false"
        );
        assert!(
            eval_bool(&outer, &e),
            "fuzz path still tolerates |a - b| < 1e-10: a == b reads true, so ite -> 1.0 and 1.0 >= 0.5 is true"
        );
    }

    #[test]
    fn nonbinary_implies_is_determinate_and_recursion_free() {
        // FINDING #6: a 1-child `Implies` previously fell through to the
        // arithmetic catch-all, which re-entered eval_arith_with -> its
        // Bool(_, _) arm -> eval_bool_with -> the same len != 2 mismatch ->
        // unbounded recursion -> stack overflow. It must now evaluate to a
        // determinate value without recursing.
        //
        // A 1-child Implies has no antecedent; it reduces to its consequent.
        let implies_true = SmtExpr::Bool(BoolOp::Implies, vec![SmtExpr::BoolLit(true)]);
        assert!(
            eval_bool(&implies_true, &env(&[])),
            "1-child Implies reduces to its (true) consequent"
        );
        let implies_false = SmtExpr::Bool(BoolOp::Implies, vec![SmtExpr::BoolLit(false)]);
        assert!(
            !eval_bool(&implies_false, &env(&[])),
            "1-child Implies reduces to its (false) consequent"
        );
        // Reaching it via the arithmetic Bool(_, _) bridge must also be
        // recursion-free (this is the exact path that used to overflow).
        assert_eq!(
            eval_arith(&implies_true, &env(&[])),
            1.0,
            "1-child Implies in arithmetic position is 1.0, not a stack overflow"
        );
        // Strict path must be equally determinate.
        assert!(eval_bool_strict(&implies_true, &env(&[])));
        // A zero-child Implies is vacuously true (no antecedent, empty
        // consequent defaults true) and must not panic or recurse.
        let implies_empty = SmtExpr::Bool(BoolOp::Implies, vec![]);
        assert!(eval_bool(&implies_empty, &env(&[])));
    }

    #[test]
    fn cr13_correct_arity_intrinsics_still_evaluate() {
        // Negative-parity: the guard must not break correct calls.
        let e = SmtExpr::Apply("sqrt".into(), vec![SmtExpr::RealLit(4.0)]);
        assert_eq!(eval_arith(&e, &env(&[])), 2.0);
        let m = SmtExpr::Apply(
            "max".into(),
            vec![SmtExpr::RealLit(1.0), SmtExpr::RealLit(2.0)],
        );
        assert_eq!(eval_arith(&m, &env(&[])), 2.0);
    }
}
