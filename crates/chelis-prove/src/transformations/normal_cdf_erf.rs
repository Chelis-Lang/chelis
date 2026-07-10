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
//! alter the contract lane, and — like every transformation — it has **no
//! production caller** (the pipeline is unwired; see
//! `spec/design/probe_434_transcendental.md` p16). It engages only when the
//! transformation pipeline is run, which is the deferred de-narrowing step. It
//! sits behind the SAME unwired engagement point as `abstract_subterm`.
//!
//! ## Soundness of the identity
//!
//! `Φ(x) = ½·(1 + erf(x/√2))` is exact for the standard normal CDF. The rewrite
//! preserves the goal's meaning; the only approximation introduced downstream is
//! the `erf` envelope's certified `eps`, tagged `SpecialFunctionCertified` by the
//! abstract-subterm pass. `1/√2` is emitted as the nearest `f64`
//! (`0.7071067811865476`); the multiply is exact-real in the SMT lowering
//! (RealLit), so no rounding enters the *model* (any residual is inside the
//! downstream envelope band, which the abstract-subterm certificate covers).

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
    fn name(&self) -> &str {
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

    #[test]
    fn lowering_is_exact_but_discharge_needs_compound_arg_propagation() {
        // The pre-pass rewrite is EXACT: normal_cdf(x) with bounded x becomes
        // 0.5 + 0.5*erf(x/√2). But the erf argument x/√2 is a COMPOUND expression
        // (Mul(x, 1/√2)), not a bare Var — so the abstract-subterm extractor
        // (bare-Var only, by design in this slice) DECLINES. Even a trivial affine
        // argument hits the compound-argument-propagation wall (the research-risk
        // 60%; see spec/design/probe_434_transcendental.md p17). This test pins
        // that honest boundary: the lowering is done, discharge is not, until
        // argument propagation lands.
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![
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
            ],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(ncdf(SmtExpr::Var("x".into()))),
                Box::new(SmtExpr::RealLit(1.0)),
            ),
        });

        // Pre-pass: normal_cdf -> erf identity (exact, done).
        let lowered = NormalCdfToErf.apply(&goal);
        assert_eq!(lowered.len(), 1);
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

        // Abstract-subterm DECLINES: x/√2 is a compound argument. The goal is
        // returned unchanged (erf still present, no fresh __erf_abs var).
        let residual = AbstractSubterm::new().apply(&lowered[0]);
        assert_eq!(residual.len(), 1);
        assert_eq!(
            residual[0].shape, lowered[0].shape,
            "compound erf argument (x/√2) must decline until arg propagation lands"
        );
        let GoalShape::Smt(ref rp) = residual[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(!rp.variables.iter().any(|(n, _)| n.starts_with("__erf_abs")));
    }

    #[test]
    fn bare_arg_erf_identity_would_abstract() {
        // Control: if the lowering DIDN'T scale the argument (i.e. the erf arg
        // were the bare bounded var), abstract-subterm WOULD engage. This isolates
        // that the ONLY blocker above is the compound x/√2 scale, not the erf
        // identity shape. Here we hand-build erf(x) (bare) inside the identity.
        let post = SmtExpr::Cmp(
            CmpOp::Le,
            Box::new(SmtExpr::Arith(
                ArithOp::Add,
                Box::new(SmtExpr::RealLit(0.5)),
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::RealLit(0.5)),
                    Box::new(SmtExpr::Apply("erf".into(), vec![SmtExpr::Var("x".into())])),
                )),
            )),
            Box::new(SmtExpr::RealLit(2.0)),
        );
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![
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
            ],
            postcondition: post,
        });
        let residual = AbstractSubterm::new().apply(&goal);
        let GoalShape::Smt(ref rp) = residual[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            rp.variables.iter().any(|(n, _)| n == "__erf_abs_0"),
            "bare-arg erf must abstract: {:?}",
            rp.variables
        );
        assert!(!contains_apply(&rp.postcondition, "erf"));
    }
}
