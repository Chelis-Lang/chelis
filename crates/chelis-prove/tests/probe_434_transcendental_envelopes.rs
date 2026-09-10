//! chelis#434 PROBE harness (p18, p20): captured evidence for the
//! transcendental-envelope de-narrowing decision.
//!
//! These are PROBES, not acceptance oracles. They pin two facts the #434
//! scoping needs, so the decision to invest in compound-argument interval
//! propagation + hot-path wiring rests on run evidence, not source reading:
//!
//!   p18 — the CompositeVerdict a `SpecialFunctionCertified` discharge projects
//!         to today, END-TO-END through the SAME `base_verdict_from_discharge`
//!         seam the property runner uses. Captures the exact serialized token.
//!
//!   p20 — cvc5 (the release SMT engine, `--features smt`) discharges the
//!         RESIDUAL polynomial goal an `AbstractSubterm`-style transform would
//!         emit: a fresh envelope-bounded variable standing in for a
//!         transcendental subterm, plus the residual (in)equality. Confirms the
//!         residual shape is solvable within timeout, and is honest (an
//!         un-entailed residual is Disproved, not laundered).
//!
//! The p18 projection block is engine-independent; the p20 block needs cvc5.

mod support;

use chelis_prove::composition::{base_verdict_from_discharge, composed_qualifier_strings};
use chelis_prove::discharge::{Qualifier, QualifierSet, Soundness};

/// p18 (updated for the chelis#434 honest tier, milestone 3): a green discharge
/// carrying `SpecialFunctionCertified` — the tag the abstract-subterm transform
/// stamps — now projects to the distinct honest tier
/// `proven_modulo_certified_envelope` (it was the pre-tier `unsupported` floor).
/// It NEVER reads as plain `proven` / `proven_modulo_real_arithmetic`.
#[test]
fn p18_special_function_certified_projects_to_certified_envelope_tier() {
    crate::support::isolate();
    let quals = QualifierSet::from_iter_kinds([Qualifier::SpecialFunctionCertified]);
    let verdict = base_verdict_from_discharge(Soundness::SoundApproximate, &quals);
    let token = serde_json::to_value(verdict).unwrap();
    assert_eq!(
        token,
        serde_json::json!("proven_modulo_certified_envelope"),
        "SpecialFunctionCertified must project to the certified-envelope tier, got {token}"
    );
}

/// p18: the real envelope discharge carries BOTH `SpecialFunctionCertified` (the
/// envelope) AND `RealArith` (the residual's over-reals proof). The
/// certified-envelope tier DOMINATES — the badge discloses the envelope
/// dependency and NEVER launders into `proven_modulo_real_arithmetic` — while
/// BOTH caveats remain in the disclosed qualifier array.
#[test]
fn p18_special_function_certified_plus_real_arith_projects_to_certified_envelope_tier() {
    crate::support::isolate();
    let quals =
        QualifierSet::from_iter_kinds([Qualifier::SpecialFunctionCertified, Qualifier::RealArith]);
    let verdict = base_verdict_from_discharge(Soundness::SoundApproximate, &quals);
    let token = serde_json::to_value(verdict).unwrap();
    assert_eq!(
        token,
        serde_json::json!("proven_modulo_certified_envelope"),
        "SpecialFunctionCertified + RealArith must project to the certified-envelope \
         tier (never proven_modulo_real_arithmetic), got {token}"
    );
    let disclosed = composed_qualifier_strings(Soundness::SoundApproximate, &quals, &[]);
    assert!(
        disclosed.contains(&"special_function_certified") && disclosed.contains(&"real_arithmetic"),
        "both caveats must be disclosed, got {disclosed:?}"
    );

    // Reserved direction: an envelope-FREE over-reals proof stays
    // proven_modulo_real_arithmetic — the new token never leaks onto it.
    let real_only = QualifierSet::from_iter_kinds([Qualifier::RealArith]);
    assert_eq!(
        serde_json::to_value(base_verdict_from_discharge(
            Soundness::SoundApproximate,
            &real_only
        ))
        .unwrap(),
        serde_json::json!("proven_modulo_real_arithmetic"),
    );
}

#[cfg(feature = "smt")]
mod p20 {
    //! p20: the release cvc5 engine discharges the RESIDUAL goal an
    //! abstract-subterm transform emits — a fresh envelope-bounded var in place
    //! of a transcendental subterm — within timeout, and honestly.
    use chelis_prove::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
    use chelis_prove::tier_b::TierBResult;
    use chelis_prove::{Cvc5Engine, DischargeEngine, Goal, SmtProperty};

    const TIMEOUT_MS: u64 = 10_000;

    fn var(n: &str) -> SmtExpr {
        SmtExpr::Var(n.to_string())
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

    fn discharge(prop: SmtProperty) -> TierBResult {
        Cvc5Engine::new()
            .discharge(&Goal::smt(prop), TIMEOUT_MS)
            .into_result()
    }

    /// The exact residual shape `AbstractSubterm::apply` emits for
    /// `for x in [1,2]: log(x) < 1`: `log(x)` is replaced by fresh
    /// `__log_abs_0`, bounded by its certified envelope hull over [1,2]
    /// (`log(2) ≈ 0.6931`, so a sound band ≈ [-eps, 0.70]); the residual is
    /// `__log_abs_0 < 1`. cvc5 must PROVE it (the whole band is < 1).
    #[test]
    fn residual_log_envelope_var_is_proved() {
        crate::support::isolate();
        let v = var("__log_abs_0");
        let prop = SmtProperty {
            variables: vec![("__log_abs_0".into(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Le, real(-1.0e-6), v.clone()),
                cmp(CmpOp::Le, v.clone(), real(0.70)),
            ],
            postcondition: cmp(CmpOp::Lt, v, real(1.0)),
        };
        assert_eq!(
            discharge(prop),
            TierBResult::Proved,
            "cvc5 must discharge the envelope-bounded log residual"
        );
    }

    /// A NONLINEAR residual carrying the fresh var in a product with a live
    /// model variable — the BS-shaped `s * exp_abs` term. For `s in [1,2]` and
    /// `exp_abs in [0.9, 1.1]` (a fresh exp-envelope band), `s * exp_abs > 0`.
    /// cvc5's NRA must PROVE it, showing the residual stays in the solver's
    /// fragment after abstraction.
    #[test]
    fn residual_nonlinear_with_envelope_var_is_proved() {
        crate::support::isolate();
        let s = var("s");
        let e = var("__exp_abs_0");
        let prop = SmtProperty {
            variables: vec![
                ("s".into(), SmtSort::Real),
                ("__exp_abs_0".into(), SmtSort::Real),
            ],
            preconditions: vec![
                cmp(CmpOp::Le, real(1.0), s.clone()),
                cmp(CmpOp::Le, s.clone(), real(2.0)),
                cmp(CmpOp::Le, real(0.9), e.clone()),
                cmp(CmpOp::Le, e.clone(), real(1.1)),
            ],
            postcondition: cmp(CmpOp::Gt, arith(ArithOp::Mul, s, e), real(0.0)),
        };
        assert_eq!(
            discharge(prop),
            TierBResult::Proved,
            "cvc5 must discharge the nonlinear envelope-bounded residual"
        );
    }

    /// Honesty partner: an UN-ENTAILED residual (the envelope band is not
    /// strictly below the bound) must be Disproved, never laundered to Proved.
    /// `exp_abs in [0.9, 1.1]`, residual `exp_abs < 1.0` is FALSE at 1.1.
    #[test]
    fn residual_unentailed_envelope_var_is_disproved() {
        crate::support::isolate();
        let e = var("__exp_abs_0");
        let prop = SmtProperty {
            variables: vec![("__exp_abs_0".into(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Le, real(0.9), e.clone()),
                cmp(CmpOp::Le, e.clone(), real(1.1)),
            ],
            postcondition: cmp(CmpOp::Lt, e, real(1.0)),
        };
        assert!(
            matches!(discharge(prop), TierBResult::Disproved(_)),
            "an un-entailed residual must be Disproved, not laundered"
        );
    }
}
