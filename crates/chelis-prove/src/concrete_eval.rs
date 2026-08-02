//! Concrete dtype-aware evaluation of [`SmtExpr`] terms.
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
use chelis_types::{
    CompareOp, FloatBinOp, FloatUnOp, IntBinOp, IntUnOp, ScalarValue, cast_scalar, compare_scalars,
    float_binop, float_unop, int_binop, int_unop, scalar_from_f64, scalar_from_i64, types::Prim,
};
use std::collections::HashMap;

/// Exact, dtype-carrying environment for concrete prover evaluation.
///
/// The value type is deliberately sealed by `chelis-types`: callers cannot
/// insert a bare `f64` or erase an integer width before evaluation.
pub type ConcreteEnv = HashMap<String, ScalarValue>;

/// Evaluate a boolean-shaped [`SmtExpr`] with the legacy *fuzz*
/// comparison semantics: `==`/`!=` carry a `1e-10` tolerance. This is the
/// user-property postcondition evaluator and must NOT be used for
/// invariant-sample acceptance (use [`eval_bool_strict`] there).
pub fn eval_bool(expr: &SmtExpr, env: &ConcreteEnv) -> bool {
    eval_bool_with(expr, env, false)
}

/// Evaluate a boolean-shaped [`SmtExpr`] with STRICT comparison semantics:
/// `==`/`!=` are exact IEEE comparisons with no tolerance. This is the
/// evaluator invariant-sample validation must use (RFC D-STARVE: "exact
/// float equality starves by design"; epsilon-validated samples would
/// weaken exactly the soundness that validation provides). NaN operands
/// compare false under both `==` and (per IEEE) yield `true` for `!=`, so
/// a NaN representation never spuriously satisfies an equality invariant.
pub fn eval_bool_strict(expr: &SmtExpr, env: &ConcreteEnv) -> bool {
    eval_bool_with(expr, env, true)
}

fn eval_bool_with(expr: &SmtExpr, env: &ConcreteEnv, strict: bool) -> bool {
    match expr {
        SmtExpr::BoolLit(v) => *v,
        SmtExpr::Cmp(op, left, right) => {
            // Thread `strict` into the operands so a nested `==`/`!=` (or an
            // `ite` whose condition compares for equality) used as a Cmp
            // operand keeps the exact-equality semantics under
            // `eval_bool_strict`, instead of silently reverting to the
            // 1e-10 fuzz tolerance via `eval_arith` (strict = false).
            let Some(l) = eval_scalar_with(left, env, strict) else {
                return false;
            };
            let Some(r) = eval_scalar_with(right, env, strict) else {
                return false;
            };
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
        _ => eval_scalar_with(expr, env, strict).is_some_and(|value| {
            value
                .as_bool_exact()
                .unwrap_or_else(|| value.as_f64_lossy() != 0.0)
        }),
    }
}

/// Evaluate a single comparison. `strict` selects exact IEEE `==`/`!=`
/// (invariant validation) vs the `1e-10`-tolerant fuzz comparison.
fn eval_cmp(op: CmpOp, lhs: ScalarValue, rhs: ScalarValue, strict: bool) -> bool {
    let Some((lhs, rhs)) = align_literal_widths(lhs, rhs) else {
        return false;
    };
    let exact_op = match op {
        CmpOp::Lt => CompareOp::Lt,
        CmpOp::Le => CompareOp::Lte,
        CmpOp::Gt => CompareOp::Gt,
        CmpOp::Ge => CompareOp::Gte,
        CmpOp::Eq => CompareOp::Eq,
        CmpOp::Ne => CompareOp::Ne,
    };
    match op {
        CmpOp::Eq => {
            if strict || !lhs.prim().is_float() || !rhs.prim().is_float() {
                compare_scalars(exact_op, lhs, rhs).unwrap_or(false)
            } else {
                (lhs.as_f64_lossy() - rhs.as_f64_lossy()).abs() < 1e-10
            }
        }
        CmpOp::Ne => {
            if strict || !lhs.prim().is_float() || !rhs.prim().is_float() {
                compare_scalars(exact_op, lhs, rhs).unwrap_or(false)
            } else {
                (lhs.as_f64_lossy() - rhs.as_f64_lossy()).abs() >= 1e-10
            }
        }
        CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
            compare_scalars(exact_op, lhs, rhs).unwrap_or(false)
        }
    }
}

/// `SmtExpr::{RealLit, IntLit}` carry exact values but not their source width.
/// When one operand is a dtype-carrying variable and the other is the canonical
/// F64/Int64 literal carrier, finalize the literal at the variable width before
/// the operation. Checked programs cannot contain a genuine mixed-dtype op.
fn align_literal_widths(lhs: ScalarValue, rhs: ScalarValue) -> Option<(ScalarValue, ScalarValue)> {
    if lhs.prim() == rhs.prim() {
        Some((lhs, rhs))
    } else if lhs.prim() == Prim::F64 && rhs.prim().is_float() {
        Some((
            cast_scalar("prove-real-literal", lhs, rhs.prim()).ok()?,
            rhs,
        ))
    } else if rhs.prim() == Prim::F64 && lhs.prim().is_float() {
        Some((
            lhs,
            cast_scalar("prove-real-literal", rhs, lhs.prim()).ok()?,
        ))
    } else if lhs.prim() == Prim::Int64 && rhs.prim().is_integer() {
        Some((cast_scalar("prove-int-literal", lhs, rhs.prim()).ok()?, rhs))
    } else if rhs.prim() == Prim::Int64 && lhs.prim().is_integer() {
        Some((lhs, cast_scalar("prove-int-literal", rhs, lhs.prim()).ok()?))
    } else {
        None
    }
}

/// Evaluate an arithmetic-shaped [`SmtExpr`] under a concrete variable
/// environment with the fuzz comparison semantics (`1e-10` tolerance for
/// any nested `==`/`!=`). Unbound variables read as `0.0`; division by
/// zero and out-of-domain transcendentals yield `NaN`.
pub fn eval_arith(expr: &SmtExpr, env: &ConcreteEnv) -> f64 {
    eval_scalar_with(expr, env, false)
        .map(|value| value.as_f64_lossy())
        .unwrap_or(f64::NAN)
}

fn eval_scalar_with(expr: &SmtExpr, env: &ConcreteEnv, strict: bool) -> Option<ScalarValue> {
    match expr {
        SmtExpr::Var(name) => env
            .get(name)
            .copied()
            .or_else(|| scalar_from_f64("prove-unbound", Prim::F64, 0.0).ok()),
        SmtExpr::RealLit(v) => scalar_from_f64("prove-real-literal", Prim::F64, *v).ok(),
        SmtExpr::IntLit(v) => scalar_from_i64("prove-int-literal", Prim::Int64, *v).ok(),
        SmtExpr::BoolLit(v) => {
            scalar_from_i64("prove-bool-literal", Prim::Bool, i64::from(*v)).ok()
        }
        SmtExpr::Arith(op, left, right) => {
            let lhs = eval_scalar_with(left, env, strict)?;
            if matches!(op, ArithOp::Neg) {
                return if lhs.prim().is_integer() {
                    int_unop(IntUnOp::Neg, lhs).ok()
                } else if lhs.prim().is_float() {
                    float_unop(FloatUnOp::Neg, lhs).ok()
                } else {
                    None
                };
            }
            let rhs = eval_scalar_with(right, env, strict)?;
            let (lhs, rhs) = align_literal_widths(lhs, rhs)?;
            if matches!(op, ArithOp::Div) && rhs.as_f64_lossy() == 0.0 {
                return None;
            }
            if lhs.prim().is_integer() && rhs.prim().is_integer() {
                if matches!(op, ArithOp::Div) {
                    let lhs = scalar_from_f64(
                        "prove-int-div-cast",
                        Prim::F64,
                        lhs.as_i64_exact()? as f64,
                    )
                    .ok()?;
                    let rhs = scalar_from_f64(
                        "prove-int-div-cast",
                        Prim::F64,
                        rhs.as_i64_exact()? as f64,
                    )
                    .ok()?;
                    return float_binop(FloatBinOp::Div, lhs, rhs).ok();
                }
                let op = match op {
                    ArithOp::Add => IntBinOp::Add,
                    ArithOp::Sub => IntBinOp::Sub,
                    ArithOp::Mul => IntBinOp::Mul,
                    ArithOp::Div | ArithOp::Neg => unreachable!("handled above"),
                };
                int_binop(op, lhs, rhs).ok()
            } else if lhs.prim().is_float() && rhs.prim().is_float() {
                let op = match op {
                    ArithOp::Add => FloatBinOp::Add,
                    ArithOp::Sub => FloatBinOp::Sub,
                    ArithOp::Mul => FloatBinOp::Mul,
                    ArithOp::Div => FloatBinOp::Div,
                    ArithOp::Neg => unreachable!("handled above"),
                };
                float_binop(op, lhs, rhs).ok()
            } else {
                None
            }
        }
        SmtExpr::Apply(name, args) => {
            let values: Vec<ScalarValue> = args
                .iter()
                .map(|expr| eval_scalar_with(expr, env, strict))
                .collect::<Option<_>>()?;
            match (name.as_str(), values.as_slice()) {
                ("abs", [value]) if value.prim().is_integer() => {
                    int_unop(IntUnOp::Abs, *value).ok()
                }
                ("exp" | "log" | "sqrt" | "sin" | "cos" | "abs", [value])
                    if value.prim().is_float() =>
                {
                    let op = match name.as_str() {
                        "exp" => FloatUnOp::Exp,
                        "log" => FloatUnOp::Log,
                        "sqrt" => FloatUnOp::Sqrt,
                        "sin" => FloatUnOp::Sin,
                        "cos" => FloatUnOp::Cos,
                        "abs" => FloatUnOp::Abs,
                        _ => unreachable!("matched closed unary intrinsic"),
                    };
                    float_unop(op, *value).ok()
                }
                ("min" | "max", [lhs, rhs]) => {
                    let (lhs, rhs) = align_literal_widths(*lhs, *rhs)?;
                    if lhs.prim().is_integer() && rhs.prim().is_integer() {
                        int_binop(
                            if name == "min" {
                                IntBinOp::Min
                            } else {
                                IntBinOp::Max
                            },
                            lhs,
                            rhs,
                        )
                        .ok()
                    } else if lhs.prim().is_float() && rhs.prim().is_float() {
                        float_binop(
                            if name == "min" {
                                FloatBinOp::Min
                            } else {
                                FloatBinOp::Max
                            },
                            lhs,
                            rhs,
                        )
                        .ok()
                    } else {
                        None
                    }
                }
                _ if values.iter().all(|value| value.prim().is_float()) => {
                    let values = values
                        .iter()
                        .map(ScalarValue::as_f64_lossy)
                        .collect::<Vec<_>>();
                    scalar_from_f64("prove-intrinsic", Prim::F64, apply_intrinsic(name, &values))
                        .ok()
                }
                _ => None,
            }
        }
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool_with(cond, env, strict) {
                eval_scalar_with(then_e, env, strict)
            } else {
                eval_scalar_with(else_e, env, strict)
            }
        }
        SmtExpr::Cmp(op, left, right) => {
            let lhs = eval_scalar_with(left, env, strict)?;
            let rhs = eval_scalar_with(right, env, strict)?;
            scalar_from_i64(
                "prove-comparison",
                Prim::Bool,
                i64::from(eval_cmp(*op, lhs, rhs, strict)),
            )
            .ok()
        }
        SmtExpr::Bool(_, _) | SmtExpr::Not(_) => scalar_from_i64(
            "prove-bool",
            Prim::Bool,
            i64::from(eval_bool_with(expr, env, strict)),
        )
        .ok(),
        SmtExpr::Forall(_, _) | SmtExpr::Exists(_, _) => None, // unreachable after fuzzability check
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
        "erf" if a.len() == 1 => erf_approx(a[0]),
        "normal_cdf" if a.len() == 1 => normal_cdf_approx(a[0]),
        "min" if a.len() == 2 => a[0].min(a[1]),
        "max" if a.len() == 2 => a[0].max(a[1]),
        // quantile(xs..., q): last argument is the quantile level q ∈ [0,1],
        // preceding arguments are the data values. Requires at least 2 args
        // (1 data value + q).
        "quantile" if a.len() >= 2 => {
            let q = a[a.len() - 1];
            let data = &a[..a.len() - 1];
            quantile_linear(data, q)
        }
        _ => f64::NAN,
    }
}

/// Fast `erf` approximation using Abramowitz & Stegun 7.1.26 (maximum
/// error < 1.5e-7 over the reals). This is the standard rational
/// approximation for concrete f64 evaluation in the fuzz tier — it does
/// NOT need to be sound for proof (that is the certified envelope's job);
/// it only needs to be accurate enough that rejection sampling does not
/// starve on properties involving `normal_cdf` (chelis#659).
fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26: erf(x) ≈ 1 - (a1*t + a2*t² + a3*t³) * exp(-x²)
    // where t = 1 / (1 + 0.3275911 * |x|). Max error: 1.5e-7.
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + P * x_abs);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-x_abs * x_abs).exp())
}

/// Standard normal CDF: Φ(x) = ½·(1 + erf(x / √2)).
/// Uses the same fast erf approximation for concrete evaluation.
fn normal_cdf_approx(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// Quantile with linear interpolation between order statistics (numpy
/// default, method="linear"). Semantics:
///
/// - `q = 0.0` → min(data)
/// - `q = 1.0` → max(data)
/// - Otherwise: linear interpolation at position `q * (n-1)` into the
///   sorted data.
/// - `q` outside [0, 1] → NaN (invalid quantile level)
/// - Empty data → NaN
/// - NaN values in data: NaN elements are sorted to the end; if the
///   interpolation position touches a NaN element, the result is NaN.
///   This differs from numpy (which returns NaN for ANY NaN in data).
///   The choice here is motivated by the fuzz evaluation context where
///   partial NaN data can still yield useful non-NaN quantiles for
///   untouched positions. Document this divergence for consumers.
///
/// This is the concrete evaluator for the `quantile` primitive, enabling
/// the Tier C fuzzer to evaluate properties involving quantiles.
fn quantile_linear(data: &[f64], q: f64) -> f64 {
    quantile_linear_impl(data, q)
}

/// Public accessor for use by the contract fuzz validation (contracts.rs).
pub(crate) fn quantile_linear_pub(data: &[f64], q: f64) -> f64 {
    quantile_linear_impl(data, q)
}

fn quantile_linear_impl(data: &[f64], q: f64) -> f64 {
    if data.is_empty() || q.is_nan() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    let n = data.len();
    if n == 1 {
        return data[0];
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi || hi >= n {
        sorted[lo.min(n - 1)]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SmtExpr;

    fn env(pairs: &[(&str, f64)]) -> ConcreteEnv {
        pairs
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    scalar_from_f64("prove-test", Prim::F64, *v).expect("finite f64 test value"),
                )
            })
            .collect()
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
    fn int64_comparison_is_exact_above_the_f64_mantissa_boundary() {
        let distinct = SmtExpr::Cmp(
            CmpOp::Ne,
            Box::new(SmtExpr::IntLit(9_007_199_254_740_993)),
            Box::new(SmtExpr::IntLit(9_007_199_254_740_992)),
        );
        assert!(
            eval_bool_strict(&distinct, &HashMap::new()),
            "adjacent int64 values above 2^53 must not collapse in the concrete prover"
        );
    }

    #[test]
    fn int64_arithmetic_is_exact_above_the_f64_mantissa_boundary() {
        let plus_one = SmtExpr::Arith(
            ArithOp::Add,
            Box::new(SmtExpr::IntLit(9_007_199_254_740_992)),
            Box::new(SmtExpr::IntLit(1)),
        );
        let exact = SmtExpr::Cmp(
            CmpOp::Ne,
            Box::new(plus_one),
            Box::new(SmtExpr::IntLit(9_007_199_254_740_992)),
        );
        assert!(
            eval_bool_strict(&exact, &HashMap::new()),
            "integer arithmetic must remain exact rather than compute through f64"
        );
    }

    #[test]
    fn integer_intrinsics_do_not_reopen_the_f64_path() {
        let min = SmtExpr::Apply(
            "min".to_string(),
            vec![
                SmtExpr::IntLit(9_007_199_254_740_993),
                SmtExpr::IntLit(9_007_199_254_740_992),
            ],
        );
        let exact = SmtExpr::Cmp(
            CmpOp::Ne,
            Box::new(min),
            Box::new(SmtExpr::IntLit(9_007_199_254_740_993)),
        );
        assert!(
            eval_bool_strict(&exact, &HashMap::new()),
            "integer min must preserve the adjacent int64 distinction"
        );
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

    // --- chelis#659: erf and normal_cdf intrinsic tests ---

    #[test]
    fn erf_zero_is_zero() {
        let e = SmtExpr::Apply("erf".into(), vec![SmtExpr::RealLit(0.0)]);
        let result = eval_arith(&e, &env(&[]));
        assert!(result.abs() < 1e-6, "erf(0) should be ≈0, got {result}");
    }

    #[test]
    fn erf_large_positive_is_near_one() {
        let e = SmtExpr::Apply("erf".into(), vec![SmtExpr::RealLit(3.0)]);
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 1.0).abs() < 1e-4,
            "erf(3) should be ≈1.0, got {result}"
        );
    }

    #[test]
    fn erf_large_negative_is_near_minus_one() {
        let e = SmtExpr::Apply("erf".into(), vec![SmtExpr::RealLit(-3.0)]);
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result + 1.0).abs() < 1e-4,
            "erf(-3) should be ≈-1.0, got {result}"
        );
    }

    #[test]
    fn erf_is_odd_function() {
        // erf(-x) == -erf(x) for all x
        for &x in &[0.5, 1.0, 2.0] {
            let pos = SmtExpr::Apply("erf".into(), vec![SmtExpr::RealLit(x)]);
            let neg = SmtExpr::Apply("erf".into(), vec![SmtExpr::RealLit(-x)]);
            let r_pos = eval_arith(&pos, &env(&[]));
            let r_neg = eval_arith(&neg, &env(&[]));
            assert!(
                (r_pos + r_neg).abs() < 1e-7,
                "erf({x}) + erf(-{x}) should be 0, got {}",
                r_pos + r_neg
            );
        }
    }

    #[test]
    fn normal_cdf_zero_is_half() {
        let e = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(0.0)]);
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 0.5).abs() < 1e-7,
            "normal_cdf(0) should be 0.5, got {result}"
        );
    }

    #[test]
    fn normal_cdf_large_positive_is_near_one() {
        let e = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(5.0)]);
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 1.0).abs() < 1e-6,
            "normal_cdf(5) should be ≈1.0, got {result}"
        );
    }

    #[test]
    fn normal_cdf_large_negative_is_near_zero() {
        let e = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(-5.0)]);
        let result = eval_arith(&e, &env(&[]));
        assert!(
            result.abs() < 1e-6,
            "normal_cdf(-5) should be ≈0.0, got {result}"
        );
    }

    #[test]
    fn normal_cdf_monotonicity() {
        // Φ(x) is strictly increasing
        let vals: Vec<f64> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let results: Vec<f64> = vals
            .iter()
            .map(|&x| {
                let e = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(x)]);
                eval_arith(&e, &env(&[]))
            })
            .collect();
        for i in 1..results.len() {
            assert!(
                results[i] > results[i - 1],
                "normal_cdf must be monotone: Φ({}) = {} should be > Φ({}) = {}",
                vals[i],
                results[i],
                vals[i - 1],
                results[i - 1]
            );
        }
    }

    #[test]
    fn normal_cdf_symmetry() {
        // Φ(x) + Φ(-x) = 1
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let pos = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(x)]);
            let neg = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(-x)]);
            let sum = eval_arith(&pos, &env(&[])) + eval_arith(&neg, &env(&[]));
            assert!(
                (sum - 1.0).abs() < 1e-7,
                "Φ({x}) + Φ(-{x}) should be 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn normal_cdf_known_values() {
        // Check against well-known table values (to 4 decimal places)
        let cases = [
            (-2.0, 0.02275),
            (-1.0, 0.15866),
            (0.0, 0.5),
            (1.0, 0.84134),
            (2.0, 0.97725),
        ];
        for (x, expected) in cases {
            let e = SmtExpr::Apply("normal_cdf".into(), vec![SmtExpr::RealLit(x)]);
            let result = eval_arith(&e, &env(&[]));
            assert!(
                (result - expected).abs() < 1e-4,
                "normal_cdf({x}) should be ≈{expected}, got {result}"
            );
        }
    }

    #[test]
    fn erf_wrong_arity_is_nan() {
        let e = SmtExpr::Apply("erf".into(), vec![]);
        assert!(eval_arith(&e, &env(&[])).is_nan());
        let e2 = SmtExpr::Apply(
            "erf".into(),
            vec![SmtExpr::RealLit(1.0), SmtExpr::RealLit(2.0)],
        );
        assert!(eval_arith(&e2, &env(&[])).is_nan());
    }

    #[test]
    fn normal_cdf_wrong_arity_is_nan() {
        let e = SmtExpr::Apply("normal_cdf".into(), vec![]);
        assert!(eval_arith(&e, &env(&[])).is_nan());
    }

    // --- quantile intrinsic tests ---

    #[test]
    fn quantile_median_of_odd_sequence() {
        // quantile([1, 2, 3, 4, 5], 0.5) = 3.0
        let e = SmtExpr::Apply(
            "quantile".into(),
            vec![
                SmtExpr::RealLit(1.0),
                SmtExpr::RealLit(2.0),
                SmtExpr::RealLit(3.0),
                SmtExpr::RealLit(4.0),
                SmtExpr::RealLit(5.0),
                SmtExpr::RealLit(0.5), // q
            ],
        );
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 3.0).abs() < 1e-10,
            "quantile([1..5], 0.5) should be 3.0, got {result}"
        );
    }

    #[test]
    fn quantile_min_at_q_zero() {
        let e = SmtExpr::Apply(
            "quantile".into(),
            vec![
                SmtExpr::RealLit(5.0),
                SmtExpr::RealLit(1.0),
                SmtExpr::RealLit(3.0),
                SmtExpr::RealLit(0.0), // q = 0 -> min
            ],
        );
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 1.0).abs() < 1e-10,
            "quantile at q=0 should be min=1.0, got {result}"
        );
    }

    #[test]
    fn quantile_max_at_q_one() {
        let e = SmtExpr::Apply(
            "quantile".into(),
            vec![
                SmtExpr::RealLit(5.0),
                SmtExpr::RealLit(1.0),
                SmtExpr::RealLit(3.0),
                SmtExpr::RealLit(1.0), // q = 1 -> max
            ],
        );
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 5.0).abs() < 1e-10,
            "quantile at q=1 should be max=5.0, got {result}"
        );
    }

    #[test]
    fn quantile_interpolation() {
        // quantile([1, 2, 3, 4, 5], 0.25)
        // position = 0.25 * 4 = 1.0 -> exactly sorted[1] = 2.0
        let e = SmtExpr::Apply(
            "quantile".into(),
            vec![
                SmtExpr::RealLit(1.0),
                SmtExpr::RealLit(2.0),
                SmtExpr::RealLit(3.0),
                SmtExpr::RealLit(4.0),
                SmtExpr::RealLit(5.0),
                SmtExpr::RealLit(0.25),
            ],
        );
        let result = eval_arith(&e, &env(&[]));
        assert!(
            (result - 2.0).abs() < 1e-10,
            "quantile([1..5], 0.25) should be 2.0, got {result}"
        );
    }

    #[test]
    fn quantile_monotone_in_q() {
        // Verify quantile is monotone: quantile(data, p) <= quantile(data, q) for p <= q
        let data = vec![
            SmtExpr::RealLit(3.0),
            SmtExpr::RealLit(1.0),
            SmtExpr::RealLit(4.0),
            SmtExpr::RealLit(1.0),
            SmtExpr::RealLit(5.0),
        ];
        let mut prev = f64::NEG_INFINITY;
        for qi in 0..=10 {
            let q = qi as f64 / 10.0;
            let mut args = data.clone();
            args.push(SmtExpr::RealLit(q));
            let e = SmtExpr::Apply("quantile".into(), args);
            let result = eval_arith(&e, &env(&[]));
            assert!(
                result >= prev - 1e-15,
                "quantile must be monotone: q={q}, result={result}, prev={prev}"
            );
            prev = result;
        }
    }

    #[test]
    fn quantile_invalid_q_is_nan() {
        // q < 0 or q > 1 should be NaN
        let e_neg = SmtExpr::Apply(
            "quantile".into(),
            vec![SmtExpr::RealLit(1.0), SmtExpr::RealLit(-0.1)],
        );
        assert!(eval_arith(&e_neg, &env(&[])).is_nan());

        let e_over = SmtExpr::Apply(
            "quantile".into(),
            vec![SmtExpr::RealLit(1.0), SmtExpr::RealLit(1.1)],
        );
        assert!(eval_arith(&e_over, &env(&[])).is_nan());
    }

    #[test]
    fn quantile_single_element() {
        // quantile([x], q) = x for any valid q
        let e = SmtExpr::Apply(
            "quantile".into(),
            vec![SmtExpr::RealLit(42.0), SmtExpr::RealLit(0.5)],
        );
        assert!((eval_arith(&e, &env(&[])) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn quantile_too_few_args_is_nan() {
        // Need at least 2 args (1 data + q)
        let e = SmtExpr::Apply("quantile".into(), vec![SmtExpr::RealLit(0.5)]);
        assert!(eval_arith(&e, &env(&[])).is_nan());
    }
}
