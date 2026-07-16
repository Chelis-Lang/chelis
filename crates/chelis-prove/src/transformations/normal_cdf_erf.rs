//! `normal_cdf` → `½·(1 + erf(x/√2))` lowering (chelis#434).
//!
//! This is a **pre-pass** that rewrites a standard-normal CDF application into
//! the exact `erf` identity, so a downstream [`super::abstract_subterm`] pass can
//! discharge it through the **certified `erf` envelope** — a mechanism switch off
//! the existing trusted `normal_cdf` **contract** lane.
//!
//! ## Two lanes, one CDF (the mechanism switch)
//!
//! - **Contract lane** (`property_runner::ContractAbstraction`, unchanged): the
//!   default. `normal_cdf(x)` is abstracted to a fresh `[0,1]`-bounded SMT var
//!   with an optional reflection precondition `N(−x) = 1 − N(x)`. It is a
//!   *trusted* assumption (the qualifier is a fuzz-validated contract), not a
//!   proof of the CDF itself.
//! - **Envelope lane** (this pre-pass, off the certified `erf` envelope): rewrite
//!   `normal_cdf(x)` to `½·(1 + erf(x/√2))`, an EXACT real identity, then let the
//!   abstract-subterm pass bound the `erf` via its Sollya/Gappa/Arb-certified
//!   envelope. The CDF value is then backed by a *certificate*, not a contract.
//!
//! This module implements ONLY the envelope-lane pre-pass. It does NOT remove or
//! alter the contract lane. The property-runner dispatch runs it (then
//! `abstract_subterm`) on a non-inlineable transcendental goal via the
//! certified-envelope lane (chelis#434 milestone 4); a `normal_cdf` site whose
//! argument is soundly boundable and whose `erf` residual proves over reals
//! discharges as `proven_modulo_certified_envelope`.
//!
//! ## Soundness of the identity
//!
//! `Φ(x) = ½·(1 + erf(x/√2))` is exact for the standard normal CDF. The rewrite
//! preserves the goal's meaning; the only approximation introduced downstream is
//! the `erf` envelope's certified `eps`, tagged `SpecialFunctionCertified` by the
//! abstract-subterm pass. `1/√2` is emitted as the nearest `f64`
//! (`0.7071067811865476`), so the argument scale carries a ~1e-17 relative
//! rounding versus the exact `1/√2` — negligible and DOMINATED by the erf
//! envelope's certified `eps` (~5.7e-7), which soundly bounds the resulting
//! `erf` value regardless. (The multiply itself is an exact-real `RealLit` in the
//! SMT lowering; the only inexactness is representing `1/√2` as an `f64`.)

use crate::discharge::{Goal, GoalShape};
use crate::solver::{ArithOp, SmtExpr};
use crate::tier_b::SmtProperty;
use crate::transformation::Transformation;

/// The `f64`-nearest value of `1/√2`, the `erf` argument scale in the identity
/// `Φ(x) = ½·(1 + erf(x/√2))`.
pub const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// The special-function name recognized as a standard-normal CDF for the
/// envelope-lane rewrite.
const NORMAL_CDF: &str = "normal_cdf";

/// Rewrite `x` into `erf(x · (1/√2))`.
fn erf_of_scaled(arg: SmtExpr) -> SmtExpr {
    SmtExpr::Apply(
        "erf".to_string(),
        vec![SmtExpr::Arith(
            ArithOp::Mul,
            Box::new(arg),
            Box::new(SmtExpr::RealLit(INV_SQRT2)),
        )],
    )
}

/// Rewrite every `normal_cdf(x)` in `expr` to `½·(1 + erf(x/√2))`
/// (= `0.5 + 0.5·erf(x/√2)`), recursing into the argument first so a nested
/// `normal_cdf` inside `x` is also rewritten.
pub fn lower_normal_cdf(expr: &SmtExpr) -> SmtExpr {
    match expr {
        SmtExpr::Apply(name, args) if name == NORMAL_CDF && args.len() == 1 => {
            let inner = lower_normal_cdf(&args[0]);
            // 0.5 + 0.5 * erf(inner / √2)
            SmtExpr::Arith(
                ArithOp::Add,
                Box::new(SmtExpr::RealLit(0.5)),
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::RealLit(0.5)),
                    Box::new(erf_of_scaled(inner)),
                )),
            )
        }
        SmtExpr::Arith(op, l, r) => SmtExpr::Arith(
            *op,
            Box::new(lower_normal_cdf(l)),
            Box::new(lower_normal_cdf(r)),
        ),
        SmtExpr::Cmp(op, l, r) => SmtExpr::Cmp(
            *op,
            Box::new(lower_normal_cdf(l)),
            Box::new(lower_normal_cdf(r)),
        ),
        SmtExpr::Bool(op, children) => {
            SmtExpr::Bool(*op, children.iter().map(lower_normal_cdf).collect())
        }
        SmtExpr::Not(inner) => SmtExpr::Not(Box::new(lower_normal_cdf(inner))),
        SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
            Box::new(lower_normal_cdf(c)),
            Box::new(lower_normal_cdf(t)),
            Box::new(lower_normal_cdf(e)),
        ),
        SmtExpr::Apply(name, args) => {
            SmtExpr::Apply(name.clone(), args.iter().map(lower_normal_cdf).collect())
        }
        SmtExpr::Forall(vars, body) => {
            SmtExpr::Forall(vars.clone(), Box::new(lower_normal_cdf(body)))
        }
        SmtExpr::Exists(vars, body) => {
            SmtExpr::Exists(vars.clone(), Box::new(lower_normal_cdf(body)))
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {
            expr.clone()
        }
    }
}

/// The envelope-lane `normal_cdf` → `erf`-identity pre-pass, as a
/// [`Transformation`] to run BEFORE [`super::abstract_subterm::AbstractSubterm`].
/// Sound in the proof direction: the rewrite is an exact identity, so the output
/// goal is equivalent to the input; `abstract_subterm` then supplies the only
/// (certified) approximation.
pub struct NormalCdfToErf;

impl Transformation for NormalCdfToErf {
    fn name(&self) -> &'static str {
        "normal-cdf-to-erf"
    }

    fn apply(&self, goal: &Goal) -> Vec<Goal> {
        let GoalShape::Smt(ref prop) = goal.shape else {
            return vec![goal.clone()];
        };
        let rewritten = lower_normal_cdf(&prop.postcondition);
        if rewritten == prop.postcondition {
            return vec![goal.clone()]; // no normal_cdf to lower
        }
        let new_prop = SmtProperty {
            variables: prop.variables.clone(),
            preconditions: prop.preconditions.clone(),
            postcondition: rewritten,
        };
        vec![Goal::smt(new_prop).with_ir(goal.ir.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{CmpOp, SmtSort};
    use crate::transformations::abstract_subterm::AbstractSubterm;

    fn ncdf(arg: SmtExpr) -> SmtExpr {
        SmtExpr::Apply("normal_cdf".into(), vec![arg])
    }

    fn contains_apply(expr: &SmtExpr, name: &str) -> bool {
        match expr {
            SmtExpr::Apply(n, args) => n == name || args.iter().any(|a| contains_apply(a, name)),
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                contains_apply(l, name) || contains_apply(r, name)
            }
            SmtExpr::Bool(_, cs) => cs.iter().any(|c| contains_apply(c, name)),
            SmtExpr::Not(i) => contains_apply(i, name),
            SmtExpr::Ite(c, t, e) => {
                contains_apply(c, name) || contains_apply(t, name) || contains_apply(e, name)
            }
            SmtExpr::Forall(_, b) | SmtExpr::Exists(_, b) => contains_apply(b, name),
            _ => false,
        }
    }

    #[test]
    fn rewrites_normal_cdf_to_the_erf_identity() {
        let expr = ncdf(SmtExpr::Var("x".into()));
        let out = lower_normal_cdf(&expr);
        // shape: 0.5 + 0.5 * erf(x * INV_SQRT2)
        let SmtExpr::Arith(ArithOp::Add, half, rest) = &out else {
            panic!("expected Add at top, got {out:?}");
        };
        assert_eq!(**half, SmtExpr::RealLit(0.5));
        let SmtExpr::Arith(ArithOp::Mul, halfb, erf) = rest.as_ref() else {
            panic!("expected 0.5 * erf(...)");
        };
        assert_eq!(**halfb, SmtExpr::RealLit(0.5));
        let SmtExpr::Apply(name, args) = erf.as_ref() else {
            panic!("expected erf apply");
        };
        assert_eq!(name, "erf");
        let SmtExpr::Arith(ArithOp::Mul, arg, scale) = &args[0] else {
            panic!("expected x * INV_SQRT2");
        };
        assert_eq!(**arg, SmtExpr::Var("x".into()));
        assert_eq!(**scale, SmtExpr::RealLit(INV_SQRT2));
        assert!(!contains_apply(&out, "normal_cdf"));
        assert!(contains_apply(&out, "erf"));
    }

    #[test]
    fn identity_when_no_normal_cdf() {
        let expr = SmtExpr::Cmp(
            CmpOp::Le,
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::RealLit(1.0)),
        );
        assert_eq!(lower_normal_cdf(&expr), expr);
        // Transformation returns the goal unchanged.
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: expr,
        });
        let out = NormalCdfToErf.apply(&goal);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, goal.shape);
    }

    fn bounded_x_pre() -> Vec<SmtExpr> {
        vec![
            SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::RealLit(0.0)),
                Box::new(SmtExpr::Var("x".into())),
            ),
            SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::RealLit(1.0)),
            ),
        ]
    }

    #[test]
    fn normal_cdf_of_bare_var_abstracts_via_affine_propagation() {
        // normal_cdf(x) with bounded x lowers to 0.5 + 0.5·erf(x/√2). The erf
        // argument x/√2 is AFFINE, so affine-argument propagation (chelis#434)
        // now lets abstract-subterm ABSTRACT it via the certified erf envelope —
        // the case that DECLINED before this slice. No normal_cdf, no erf left.
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: bounded_x_pre(),
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(ncdf(SmtExpr::Var("x".into()))),
                Box::new(SmtExpr::RealLit(1.0)),
            ),
        });

        let lowered = NormalCdfToErf.apply(&goal);
        let GoalShape::Smt(ref lp) = lowered[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            !contains_apply(&lp.postcondition, "normal_cdf"),
            "normal_cdf lowered away"
        );
        assert!(
            contains_apply(&lp.postcondition, "erf"),
            "erf identity present"
        );

        let residual = AbstractSubterm::new().apply(&lowered[0]);
        let GoalShape::Smt(ref rp) = residual[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            rp.variables.iter().any(|(n, _)| n == "__erf_abs_0"),
            "affine x/√2 must now abstract: {:?}",
            rp.variables
        );
        assert!(!contains_apply(&rp.postcondition, "normal_cdf"));
        assert!(!contains_apply(&rp.postcondition, "erf"));
    }

    fn le(l: SmtExpr, r: SmtExpr) -> SmtExpr {
        SmtExpr::Cmp(CmpOp::Le, Box::new(l), Box::new(r))
    }
    fn lit(v: f64) -> SmtExpr {
        SmtExpr::RealLit(v)
    }
    fn var(n: &str) -> SmtExpr {
        SmtExpr::Var(n.into())
    }

    #[test]
    fn normal_cdf_of_boundable_nonlinear_arg_now_abstracts() {
        // chelis#434 milestone 1: a NONLINEAR-but-boundable argument now
        // abstracts. normal_cdf(x·y) → erf((x·y)/√2); x·y over [0,1]² is bounded
        // by interval arithmetic to [0,1], so /√2 ∈ [0,0.707] ⊆ erf coverage and
        // the CDF abstracts via the certified erf envelope. (This DECLINED under
        // the affine-only extractor — the boundary moved.)
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".into(), SmtSort::Real), ("y".into(), SmtSort::Real)],
            preconditions: vec![
                le(lit(0.0), var("x")),
                le(var("x"), lit(1.0)),
                le(lit(0.0), var("y")),
                le(var("y"), lit(1.0)),
            ],
            postcondition: le(
                ncdf(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(var("x")),
                    Box::new(var("y")),
                )),
                lit(1.0),
            ),
        });
        let lowered = NormalCdfToErf.apply(&goal);
        let residual = AbstractSubterm::new().apply(&lowered[0]);
        let GoalShape::Smt(ref rp) = residual[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            rp.variables.iter().any(|(n, _)| n == "__erf_abs_0"),
            "boundable nonlinear arg must now abstract: {:?}",
            rp.variables
        );
        assert!(!contains_apply(&rp.postcondition, "erf"));
        assert!(!contains_apply(&rp.postcondition, "normal_cdf"));
    }

    #[test]
    fn normal_cdf_of_unboundable_arg_still_declines() {
        // The remaining boundary: an argument that is NOT soundly boundable.
        // normal_cdf(x/y) with y spanning 0 → erf((x/y)/√2), and x/y is unbounded
        // (divisor interval contains 0), so the interval extractor fails closed
        // and abstract-subterm DECLINES (never guesses a bound).
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".into(), SmtSort::Real), ("y".into(), SmtSort::Real)],
            preconditions: vec![
                le(lit(1.0), var("x")),
                le(var("x"), lit(2.0)),
                le(lit(-1.0), var("y")), // y ∈ [-1,1] spans 0
                le(var("y"), lit(1.0)),
            ],
            postcondition: le(
                ncdf(SmtExpr::Arith(
                    ArithOp::Div,
                    Box::new(var("x")),
                    Box::new(var("y")),
                )),
                lit(1.0),
            ),
        });
        let lowered = NormalCdfToErf.apply(&goal);
        let residual = AbstractSubterm::new().apply(&lowered[0]);
        assert_eq!(
            residual[0].shape, lowered[0].shape,
            "unboundable x/y (divisor spans 0) argument must decline"
        );
        let GoalShape::Smt(ref rp) = residual[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            contains_apply(&rp.postcondition, "erf"),
            "erf remains (declined)"
        );
        assert!(!rp.variables.iter().any(|(n, _)| n.starts_with("__erf_abs")));
    }
}
