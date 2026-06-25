//! Recognize the univariate-polynomial-nonneg-on-interval subset of
//! [`SmtProperty`] and extract `(p, [a, b])`.
//!
//! This is the Option (i) goal-shape fit (`docs/design/clarabel_sos_engine.md`):
//! the engine fits `GoalShape::Smt` and internally recognizes the subset of
//! `SmtProperty` that IS a univariate polynomial inequality over the reals on a
//! bounded interval, by a pure structural match on the existing `SmtExpr` tree.
//! No frozen-seam change, no new `SmtProperty` fields.
//!
//! The recognizer is FALSE-FIT-SAFE: [`extract_poly_on_interval`] returns `None`
//! for anything outside the recognized subset (multivariate, transcendental
//! `Apply`, a `Div`, a missing interval bound, a non-polynomial body). The
//! engine's `fitness` is exactly "extraction succeeds", so a goal Clarabel
//! cannot honestly attempt is never claimed by Clarabel; it stays with cvc5 /
//! the no-fit path.

use num_rational::BigRational;

use super::poly::RationalPoly;
use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::SmtProperty;

/// A recognized goal: prove `poly >= 0` (or `> 0`; see [`Self::strict`]) for the
/// single real variable `var` ranging over `[lo, hi]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyOnInterval {
    /// The single real variable name the polynomial is in.
    pub var: String,
    /// The polynomial asserted nonnegative on `[lo, hi]` (already normalized so
    /// the assertion is `poly >= 0`, i.e. a `Le`/`Lt` postcondition has had its
    /// sign flipped).
    pub poly: RationalPoly,
    /// The interval lower bound (exact rational from the precondition literal).
    pub lo: BigRational,
    /// The interval upper bound.
    pub hi: BigRational,
    /// Whether the original postcondition was strict (`Gt`/`Lt`). The SoS
    /// machinery proves `poly >= 0`; a strict claim needs `poly` bounded away
    /// from zero, which the certificate does not establish by itself. The engine
    /// uses this to decide whether a `poly >= 0` certificate suffices.
    pub strict: bool,
}

/// Convert the `f64` of a `RealLit` to its EXACT rational value, matching the
/// cvc5 lowering convention (`tier_b.rs`): the literal lowers to the f64's exact
/// value (`BigRational::from_float`), NOT its decimal spelling, so the SoS
/// engine reasons about the same number the runtime f64 evaluator does. A
/// non-finite literal has no exact rational and makes the term non-polynomial.
fn real_lit_to_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    BigRational::from_float(value)
}

/// Convert an `SmtExpr` arithmetic subtree to a [`RationalPoly`] in the single
/// variable `var`, or `None` if the subtree is not a polynomial in exactly that
/// one variable.
///
/// Recognized: `Var(var)`, `RealLit`, `IntLit`, and `Arith(Add | Sub | Mul |
/// Neg, ..)`. Rejected (returns `None`, keeping the recognizer tight and the
/// fit false-fit-safe):
///
/// - a `Var` whose name is not `var` (a second variable -> multivariate),
/// - `Arith(Div, ..)` (a rational function, not a polynomial -- even constant
///   division is declined in this first cut; scaling is expressible as `Mul` by
///   a literal),
/// - `Apply` (a transcendental / uninterpreted function),
/// - `Cmp` / `Bool` / `Not` / `Forall` / `Exists` / `Ite` / `BoolLit` in
///   arithmetic position (not a real-valued polynomial term).
fn smt_to_poly(expr: &SmtExpr, var: &str) -> Option<RationalPoly> {
    match expr {
        SmtExpr::Var(name) => {
            if name == var {
                Some(RationalPoly::x())
            } else {
                // A different variable name -> the term is multivariate.
                None
            }
        }
        SmtExpr::RealLit(v) => Some(RationalPoly::constant(real_lit_to_rational(*v)?)),
        SmtExpr::IntLit(v) => Some(RationalPoly::constant(BigRational::from(
            num_bigint::BigInt::from(*v),
        ))),
        // Unary neg is `Arith(Neg, x, <placeholder>)`; the placeholder is
        // ignored (matching the cvc5 lowering), so negate only the real operand.
        SmtExpr::Arith(ArithOp::Neg, left, _placeholder) => Some(smt_to_poly(left, var)?.neg()),
        SmtExpr::Arith(op, left, right) => {
            let l = smt_to_poly(left, var)?;
            let r = smt_to_poly(right, var)?;
            match op {
                ArithOp::Add => Some(l.add(&r)),
                ArithOp::Sub => Some(l.sub(&r)),
                ArithOp::Mul => Some(l.mul(&r)),
                // Division is not a polynomial operation; decline.
                ArithOp::Div => None,
                ArithOp::Neg => unreachable!("Neg handled in the arm above"),
            }
        }
        // Anything else (Apply, Cmp, Bool, Not, quantifiers, Ite, BoolLit) is
        // not a real-valued polynomial term.
        _ => None,
    }
}

/// Try to read `expr` as an interval bound on `var`, returning
/// `(is_lower_bound, bound_value)`:
///
/// - `var >= c` or `c <= var`  -> lower bound `c`,
/// - `var <= c` or `c >= var`  -> upper bound `c`,
///
/// where `c` is a `RealLit`/`IntLit` constant. Strict bounds (`Gt`/`Lt`) are
/// accepted as the closed bound (a strict-on-the-precondition interval still
/// contains its closure for an SoS-on-`[a,b]` proof; the interval used is the
/// closed `[a, b]`, which is a superset, keeping the proof sound -- proving on a
/// larger interval implies the smaller). Returns `None` for any other shape.
fn read_interval_bound(expr: &SmtExpr, var: &str) -> Option<(bool, BigRational)> {
    let SmtExpr::Cmp(op, left, right) = expr else {
        return None;
    };
    // Identify which side is the variable and which is the constant.
    let const_of = |e: &SmtExpr| -> Option<BigRational> {
        match e {
            SmtExpr::RealLit(v) => real_lit_to_rational(*v),
            SmtExpr::IntLit(v) => Some(BigRational::from(num_bigint::BigInt::from(*v))),
            _ => None,
        }
    };
    let is_var = |e: &SmtExpr| matches!(e, SmtExpr::Var(n) if n == var);

    // var (op) const
    if is_var(left)
        && let Some(c) = const_of(right)
    {
        return match op {
            // var >= c, var > c  -> lower bound c
            CmpOp::Ge | CmpOp::Gt => Some((true, c)),
            // var <= c, var < c  -> upper bound c
            CmpOp::Le | CmpOp::Lt => Some((false, c)),
            CmpOp::Eq | CmpOp::Ne => None,
        };
    }
    // const (op) var
    if is_var(right)
        && let Some(c) = const_of(left)
    {
        return match op {
            // c <= var, c < var  -> lower bound c
            CmpOp::Le | CmpOp::Lt => Some((true, c)),
            // c >= var, c > var  -> upper bound c
            CmpOp::Ge | CmpOp::Gt => Some((false, c)),
            CmpOp::Eq | CmpOp::Ne => None,
        };
    }
    None
}

/// Recognize the univariate-poly-nonneg-on-interval subset of a property and
/// extract `(poly, [lo, hi], strict)`. `None` if the property is not in the
/// subset (the false-fit-safe gate).
///
/// Requirements:
///
/// - exactly one `Real`-sorted variable (the polynomial's variable); no other
///   variables of any sort;
/// - a postcondition `Cmp(Ge | Gt | Le | Lt, lhs, rhs)` whose two sides are both
///   polynomials in that one variable, normalized to `p >= 0` (a `Le`/`Lt`
///   postcondition `lhs <= rhs` becomes `rhs - lhs >= 0`);
/// - preconditions that pin BOTH a lower and an upper interval bound on the
///   variable, with `lo <= hi` (an inverted / empty interval is not a fit).
///   Extra preconditions that are not simple interval bounds make the property
///   fall outside the subset (the engine must not silently ignore an extra
///   constraint it cannot honor).
pub fn extract_poly_on_interval(property: &SmtProperty) -> Option<PolyOnInterval> {
    // Exactly one variable, and it must be Real-sorted.
    if property.variables.len() != 1 {
        return None;
    }
    let (var, sort) = &property.variables[0];
    if *sort != SmtSort::Real {
        return None;
    }

    // Postcondition: a polynomial inequality, normalized to p >= 0.
    let (op, lhs, rhs) = match &property.postcondition {
        SmtExpr::Cmp(op, lhs, rhs) => (op, lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };
    let lp = smt_to_poly(lhs, var)?;
    let rp = smt_to_poly(rhs, var)?;
    let (poly, strict) = match op {
        // lhs >= rhs  <=>  lhs - rhs >= 0
        CmpOp::Ge => (lp.sub(&rp), false),
        CmpOp::Gt => (lp.sub(&rp), true),
        // lhs <= rhs  <=>  rhs - lhs >= 0
        CmpOp::Le => (rp.sub(&lp), false),
        CmpOp::Lt => (rp.sub(&lp), true),
        CmpOp::Eq | CmpOp::Ne => return None,
    };

    // Preconditions: pin both interval bounds; every precondition must be a
    // simple interval bound on the variable (no silently-ignored constraints).
    let mut lo: Option<BigRational> = None;
    let mut hi: Option<BigRational> = None;
    for pre in &property.preconditions {
        let (is_lower, c) = read_interval_bound(pre, var)?;
        if is_lower {
            // Tightest lower bound wins if several are given.
            lo = Some(match lo {
                Some(prev) if prev >= c => prev,
                _ => c,
            });
        } else {
            hi = Some(match hi {
                Some(prev) if prev <= c => prev,
                _ => c,
            });
        }
    }
    let lo = lo?;
    let hi = hi?;
    if lo > hi {
        // Inverted / empty interval: not a fit.
        return None;
    }

    Some(PolyOnInterval {
        var: var.clone(),
        poly,
        lo,
        hi,
        strict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn ri(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    fn var(name: &str) -> SmtExpr {
        SmtExpr::Var(name.to_string())
    }

    fn real(v: f64) -> SmtExpr {
        SmtExpr::RealLit(v)
    }

    fn cmp(op: CmpOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
        SmtExpr::Cmp(op, Box::new(l), Box::new(r))
    }

    fn arith(op: ArithOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(op, Box::new(l), Box::new(r))
    }

    /// `x^2` as an SmtExpr: x * x.
    fn x_squared() -> SmtExpr {
        arith(ArithOp::Mul, var("x"), var("x"))
    }

    /// A property: forall x in [lo, hi], <postcondition>.
    fn interval_property(lo: f64, hi: f64, postcondition: SmtExpr) -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), real(lo)),
                cmp(CmpOp::Le, var("x"), real(hi)),
            ],
            postcondition,
        }
    }

    // --- positive: recognized subset ---

    #[test]
    fn recognizes_x_squared_nonneg_on_interval() {
        // forall x in [-1, 1]: x^2 >= 0.
        let prop = interval_property(-1.0, 1.0, cmp(CmpOp::Ge, x_squared(), real(0.0)));
        let got = extract_poly_on_interval(&prop).expect("recognized");
        assert_eq!(got.var, "x");
        assert_eq!(got.poly, RationalPoly::from_int_coeffs(&[0, 0, 1])); // x^2
        assert_eq!(got.lo, ri(-1));
        assert_eq!(got.hi, ri(1));
        assert!(!got.strict);
    }

    #[test]
    fn normalizes_le_postcondition_to_ge_zero() {
        // forall x in [0, 2]: 0 <= x^2  ==>  x^2 - 0 >= 0, i.e. poly = x^2.
        let prop = interval_property(0.0, 2.0, cmp(CmpOp::Le, real(0.0), x_squared()));
        let got = extract_poly_on_interval(&prop).expect("recognized");
        assert_eq!(got.poly, RationalPoly::from_int_coeffs(&[0, 0, 1]));
        assert!(!got.strict);
    }

    #[test]
    fn subtracts_rhs_for_general_inequality() {
        // forall x in [0, 1]: x^2 >= x  ==>  x^2 - x >= 0, poly = -x + x^2.
        let prop = interval_property(0.0, 1.0, cmp(CmpOp::Ge, x_squared(), var("x")));
        let got = extract_poly_on_interval(&prop).expect("recognized");
        assert_eq!(got.poly, RationalPoly::from_int_coeffs(&[0, -1, 1]));
    }

    #[test]
    fn marks_strict_postcondition() {
        // forall x in [-1, 1]: x^2 + 1 > 0  (strict).
        let body = arith(ArithOp::Add, x_squared(), real(1.0));
        let prop = interval_property(-1.0, 1.0, cmp(CmpOp::Gt, body, real(0.0)));
        let got = extract_poly_on_interval(&prop).expect("recognized");
        assert_eq!(got.poly, RationalPoly::from_int_coeffs(&[1, 0, 1])); // 1 + x^2
        assert!(got.strict);
    }

    #[test]
    fn accepts_const_on_left_interval_bound_forms() {
        // Preconditions written as `lo <= x` and `hi >= x`.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Le, real(-2.0), var("x")), // -2 <= x  -> lower -2
                cmp(CmpOp::Ge, real(3.0), var("x")),  // 3 >= x   -> upper 3
            ],
            postcondition: cmp(CmpOp::Ge, x_squared(), real(0.0)),
        };
        let got = extract_poly_on_interval(&prop).expect("recognized");
        assert_eq!(got.lo, ri(-2));
        assert_eq!(got.hi, ri(3));
    }

    #[test]
    fn real_literal_uses_exact_f64_value_not_decimal() {
        // The constant 0.5 is exactly representable; its rational is 1/2. (This
        // pins the from_float convention shared with the cvc5 path.)
        let half = RationalPoly::constant(real_lit_to_rational(0.5).unwrap());
        assert_eq!(
            half.coeff(0),
            BigRational::new(BigInt::from(1), BigInt::from(2))
        );
    }

    // --- negative: false-fit-safe (each returns None) ---

    #[test]
    fn rejects_two_variables_multivariate() {
        let prop = SmtProperty {
            variables: vec![
                ("x".to_string(), SmtSort::Real),
                ("y".to_string(), SmtSort::Real),
            ],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), real(0.0)),
                cmp(CmpOp::Le, var("x"), real(1.0)),
            ],
            postcondition: cmp(
                CmpOp::Ge,
                arith(ArithOp::Add, var("x"), var("y")),
                real(0.0),
            ),
        };
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_second_variable_in_body_even_with_one_declared() {
        // Only x declared, but the body references y -> smt_to_poly fails.
        let prop = interval_property(
            0.0,
            1.0,
            cmp(
                CmpOp::Ge,
                arith(ArithOp::Add, var("x"), var("y")),
                real(0.0),
            ),
        );
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_transcendental_apply() {
        // forall x in [0,1]: exp(x) >= 0 -- an Apply is not a polynomial.
        let exp_x = SmtExpr::Apply("exp".to_string(), vec![var("x")]);
        let prop = interval_property(0.0, 1.0, cmp(CmpOp::Ge, exp_x, real(0.0)));
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_division() {
        // forall x in [1,2]: 1/x >= 0 -- a rational function, not a polynomial.
        let recip = arith(ArithOp::Div, real(1.0), var("x"));
        let prop = interval_property(1.0, 2.0, cmp(CmpOp::Ge, recip, real(0.0)));
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_missing_upper_bound() {
        // Only a lower bound in the preconditions -> not a bounded interval.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
            postcondition: cmp(CmpOp::Ge, x_squared(), real(0.0)),
        };
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_no_interval_bounds() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: cmp(CmpOp::Ge, x_squared(), real(0.0)),
        };
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_extra_non_interval_precondition() {
        // An extra precondition that is not a simple interval bound (x^2 >= 1)
        // must NOT be silently ignored -> not a fit.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), real(0.0)),
                cmp(CmpOp::Le, var("x"), real(1.0)),
                cmp(CmpOp::Ge, x_squared(), real(1.0)),
            ],
            postcondition: cmp(CmpOp::Ge, x_squared(), real(0.0)),
        };
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_inverted_interval() {
        // lo = 5, hi = 1 -> empty interval, not a fit.
        let prop = interval_property(5.0, 1.0, cmp(CmpOp::Ge, x_squared(), real(0.0)));
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_int_sorted_variable() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Int)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), SmtExpr::IntLit(0)),
                cmp(CmpOp::Le, var("x"), SmtExpr::IntLit(10)),
            ],
            postcondition: cmp(CmpOp::Ge, x_squared(), SmtExpr::IntLit(0)),
        };
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_equality_postcondition() {
        let prop = interval_property(0.0, 1.0, cmp(CmpOp::Eq, x_squared(), real(0.0)));
        assert!(extract_poly_on_interval(&prop).is_none());
    }

    #[test]
    fn rejects_non_finite_literal() {
        let prop = interval_property(
            0.0,
            1.0,
            cmp(
                CmpOp::Ge,
                arith(ArithOp::Add, var("x"), real(f64::INFINITY)),
                real(0.0),
            ),
        );
        assert!(extract_poly_on_interval(&prop).is_none());
    }
}
