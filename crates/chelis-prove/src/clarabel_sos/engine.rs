//! The [`ClarabelSosEngine`] discharge engine.
//!
//! Wires the goal-shape fit ([`super::extract`]) and the exact verifier
//! ([`super::exact`]) into the [`DischargeEngine`] seam. The honesty contract,
//! enforced here and locked by tests:
//!
//! - A [`Qualifier::CertificateBearing`] discharge at [`Soundness::Exact`] is
//!   produced ONLY when an exact rational SoS certificate VERIFIES
//!   ([`SosCertificate::verify`] returns `Ok`). This is the single gate to
//!   `Exact`; a float SDP solution alone never reaches it.
//! - A certificate that FAILS exact verification (non-PSD over the rationals, a
//!   polynomial mismatch, an asymmetric Gram -- the boundary-degeneracy failure
//!   modes) yields an honest [`Soundness::Untrusted`] [`TierBResult::Unknown`]
//!   with an EMPTY qualifier set. Never `Exact`, never a fabricated cert.
//! - No certificate found (the proposer returns `None`: SDP infeasible, repair
//!   fails, or -- standalone, before the BLAS-linkage decision -- the proposer
//!   is not yet wired) yields the same honest `Unknown@Untrusted`.
//! - A polynomial NEGATIVE somewhere on the interval has no SoS certificate, so
//!   the proposer returns `None` and the goal is not-proved (never `Disproved`;
//!   the engine only certifies nonnegativity, it does not refute -- cvc5 owns
//!   the disproof). At WS-5 merge time a Clarabel `Unknown` falls through to
//!   cvc5, whose NRA may still decide the goal.
//!
//! The proposer is injected as a [`SosProposer`] so the engine's honesty logic
//! is testable WITHOUT a live Clarabel solve: a mock proposer can hand back a
//! known-good cert (must reach `Exact`), a known-bad cert (must be rejected to
//! `Unknown`), or nothing (must be `Unknown`). The real proposer (Markov-Lukacs
//! SDP, Clarabel, Peyrl-Parrilo repair) plugs in here once the PSD-cone
//! BLAS-linkage decision is pinned.

use serde_json::json;

use super::exact::{CertError, SosCertificate};
use super::extract::{PolyOnInterval, extract_poly_on_interval};
use crate::discharge::{Discharge, DischargeEngine, Goal, Qualifier, QualifierSet, Soundness};
use crate::tier_b::TierBResult;

/// Proposes a candidate SoS certificate for a recognized poly-on-interval goal,
/// or `None` if it cannot (SDP infeasible, repair failed, polynomial negative
/// somewhere on the interval, or the proposer is not yet wired). The candidate
/// is only a PROPOSAL -- the engine re-verifies it exactly before trusting it,
/// so a proposer returning a bad cert cannot launder a false `Exact`.
pub trait SosProposer {
    /// Propose an exact rational SoS certificate for `goal`, or `None`.
    fn propose(&self, goal: &PolyOnInterval) -> Option<SosCertificate>;
}

/// The not-yet-wired proposer: always returns `None`.
///
/// Used until the Markov-Lukacs SDP encoding + Clarabel solve + Peyrl-Parrilo
/// rational repair lands (held on the PSD-cone BLAS-linkage decision). With this
/// proposer the engine is sound but never proves anything: every recognized goal
/// returns the honest `Unknown@Untrusted` (and, integrated with WS-5, falls
/// through to cvc5). The verify-gated honesty path is fully exercised by tests
/// that inject certificates directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnwiredProposer;

impl SosProposer for UnwiredProposer {
    fn propose(&self, _goal: &PolyOnInterval) -> Option<SosCertificate> {
        None
    }
}

/// The Clarabel sum-of-squares certificate discharge engine.
///
/// Generic over the [`SosProposer`] so the real Clarabel proposer and a test
/// mock share the same honesty logic. [`ClarabelSosEngine::unwired`] is the
/// standalone constructor used until the proposer is wired.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClarabelSosEngine<P: SosProposer> {
    proposer: P,
}

impl ClarabelSosEngine<UnwiredProposer> {
    /// The standalone engine with the not-yet-wired proposer: every recognized
    /// goal returns the honest `Unknown@Untrusted` (no false proofs) until the
    /// SDP proposer lands.
    pub fn unwired() -> Self {
        Self {
            proposer: UnwiredProposer,
        }
    }
}

impl<P: SosProposer> ClarabelSosEngine<P> {
    /// Build the engine over a specific proposer.
    pub fn with_proposer(proposer: P) -> Self {
        Self { proposer }
    }

    /// Build the honest non-proof discharge: `Unknown@Untrusted` with an empty
    /// qualifier set and a reason in the evidence. This is the ONLY non-`Exact`
    /// outcome the engine emits; it never fabricates a `Disproved` or a partial
    /// proof.
    fn unknown(reason: &str, extra: serde_json::Value) -> Discharge {
        Discharge::new(
            Soundness::Untrusted,
            QualifierSet::new(),
            TierBResult::Unknown,
            json!({ "engine": ENGINE_NAME, "outcome": "unknown", "reason": reason, "detail": extra }),
        )
        .expect("untrusted discharge with an empty qualifier set is always valid")
    }
}

/// Stable engine identifier.
const ENGINE_NAME: &str = "clarabel_sos";

impl<P: SosProposer> DischargeEngine for ClarabelSosEngine<P> {
    fn name(&self) -> &'static str {
        ENGINE_NAME
    }

    fn fitness(&self, goal: &Goal) -> bool {
        // NARROW, false-fit-safe: the engine fits exactly the recognized
        // univariate-poly-nonneg-on-interval subset of an SMT goal. Anything
        // else (a non-SMT shape, or an SMT property outside the subset) is not a
        // fit and routes to cvc5 / the no-fit path.
        match goal.as_smt() {
            Some(property) => extract_poly_on_interval(property).is_some(),
            None => false,
        }
    }

    fn discharge(&self, goal: &Goal, _timeout_ms: u64) -> Discharge {
        // Re-extract (fitness only checked existence). A goal that does not
        // extract should never reach here via the registry, but if it does, fail
        // honest rather than panic.
        let Some(property) = goal.as_smt() else {
            return Self::unknown(
                "clarabel_sos engine cannot discharge a non-SMT goal shape",
                json!(null),
            );
        };
        let Some(poly_goal) = extract_poly_on_interval(property) else {
            return Self::unknown(
                "goal is not a univariate-polynomial-nonneg-on-interval property",
                json!(null),
            );
        };

        // Propose a candidate certificate. None -> honest Unknown (SDP
        // infeasible / repair failed / polynomial negative somewhere / proposer
        // not yet wired).
        let Some(cert) = self.proposer.propose(&poly_goal) else {
            return Self::unknown(
                "no sum-of-squares certificate found",
                json!({ "strict": poly_goal.strict }),
            );
        };

        // A strict (`>`/`<`) goal needs `poly` bounded away from zero; a
        // `poly >= 0` certificate does not establish strictness by itself. The
        // proposer is responsible for certifying `poly - eps >= 0` for an exact
        // `eps > 0` when `strict`; if it returned a cert for plain `poly` on a
        // strict goal, the engine does NOT upgrade it to a strict proof. Pin
        // this by requiring the verified certificate's polynomial to differ from
        // the goal polynomial on a strict goal (i.e. the proposer subtracted a
        // positive margin). A cert whose polynomial equals the goal polynomial
        // exactly only proves `>= 0`, which is insufficient for `> 0`.
        if poly_goal.strict && cert.poly == poly_goal.poly {
            return Self::unknown(
                "strict goal: certificate proves only `>= 0`, not the strict `> 0`",
                json!({ "strict": true }),
            );
        }

        // VERIFY the candidate EXACTLY. This is the single gate to Exact.
        match cert.verify() {
            Ok(()) => {
                // The exact rational certificate verifies: every Gram is
                // symmetric + exactly PSD over the rationals AND the combined
                // quadratic form equals the goal polynomial exactly. ONLY now do
                // we claim CertificateBearing@Exact.
                let evidence = json!({
                    "engine": ENGINE_NAME,
                    "outcome": "proved",
                    "certificate": cert_evidence(&cert),
                    "strict": poly_goal.strict,
                });
                Discharge::new(
                    Soundness::Exact,
                    QualifierSet::from_iter_kinds([Qualifier::CertificateBearing]),
                    TierBResult::Proved,
                    evidence,
                )
                .unwrap_or_else(|err| {
                    // CertificateBearing's floor is exactly Soundness::Exact, so
                    // this construction cannot fail; degrade to honest Unknown if
                    // it somehow does, never a laundered badge.
                    Self::unknown(
                        "internal: exact certificate discharge failed integrity check",
                        json!({ "error": err.to_string() }),
                    )
                })
            }
            Err(err) => {
                // The candidate FAILED exact verification (the boundary-
                // degeneracy / float-rounding failure mode). NEVER Exact: an
                // unverifiable cert is no proof. Honest Unknown.
                Self::unknown(
                    "candidate certificate failed exact verification",
                    json!({ "cert_error": cert_error_label(&err) }),
                )
            }
        }
    }
}

/// A stable label for a [`CertError`], for the evidence record.
fn cert_error_label(err: &CertError) -> &'static str {
    match err {
        CertError::NotSymmetric { .. } => "not_symmetric",
        CertError::NotPsd { .. } => "not_psd",
        CertError::PolynomialMismatch => "polynomial_mismatch",
    }
}

/// Render a verified certificate as JSON evidence so a downstream auditor can
/// re-check it. Records the interval, the goal polynomial, and each block's
/// Gram matrix + multiplier as exact `n/d` rational strings.
fn cert_evidence(cert: &SosCertificate) -> serde_json::Value {
    let rat = |r: &num_rational::BigRational| format!("{}/{}", r.numer(), r.denom());
    let blocks: Vec<serde_json::Value> = cert
        .blocks
        .iter()
        .map(|b| {
            let n = b.gram.dim();
            let gram: Vec<Vec<String>> = (0..n)
                .map(|i| (0..n).map(|j| rat(b.gram.entry(i, j))).collect())
                .collect();
            let mult: Vec<String> = b.multiplier.coeffs().iter().map(rat).collect();
            json!({ "gram": gram, "multiplier": mult })
        })
        .collect();
    let poly: Vec<String> = cert.poly.coeffs().iter().map(rat).collect();
    json!({
        "lo": rat(&cert.lo),
        "hi": rat(&cert.hi),
        "poly": poly,
        "blocks": blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::super::exact::{RatSymMatrix, SosBlock};
    use super::super::poly::RationalPoly;
    use super::*;
    use crate::discharge::GoalShape;
    use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
    use crate::tier_b::SmtProperty;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn ri(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    fn sym(n: usize, rows: &[i64]) -> RatSymMatrix {
        RatSymMatrix::from_rows(n, rows.iter().map(|&x| ri(x)).collect())
    }

    fn var(name: &str) -> SmtExpr {
        SmtExpr::Var(name.to_string())
    }

    fn cmp(op: CmpOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
        SmtExpr::Cmp(op, Box::new(l), Box::new(r))
    }

    /// `x^2 >= 0 on [lo, hi]` as a recognized property.
    fn x_sq_nonneg(lo: f64, hi: f64) -> Goal {
        let x_squared = SmtExpr::Arith(ArithOp::Mul, Box::new(var("x")), Box::new(var("x")));
        Goal::smt(SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(lo)),
                cmp(CmpOp::Le, var("x"), SmtExpr::RealLit(hi)),
            ],
            postcondition: cmp(CmpOp::Ge, x_squared, SmtExpr::RealLit(0.0)),
        })
    }

    /// A proposer that always hands back a fixed certificate (a test double).
    struct FixedProposer(SosCertificate);
    impl SosProposer for FixedProposer {
        fn propose(&self, _goal: &PolyOnInterval) -> Option<SosCertificate> {
            Some(self.0.clone())
        }
    }

    /// A genuine certificate for `x^2 >= 0` (Gram diag(0,1) over z=[1,x]:
    /// z^T Q z = x^2; PSD).
    fn good_cert_for_x_squared(lo: i64, hi: i64) -> SosCertificate {
        SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[0, 0, 1]), // x^2
            lo: ri(lo),
            hi: ri(hi),
            blocks: vec![SosBlock {
                gram: sym(2, &[0, 0, 0, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        }
    }

    // --- fitness: narrow, false-fit-safe ---

    #[test]
    fn fitness_accepts_recognized_poly_goal() {
        let engine = ClarabelSosEngine::unwired();
        assert!(engine.fitness(&x_sq_nonneg(-1.0, 1.0)));
    }

    #[test]
    fn fitness_rejects_box_range_goal() {
        use crate::discharge::{IntervalBox, OutputRange};
        let engine = ClarabelSosEngine::unwired();
        let box_goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 1.0)],
            },
            OutputRange {
                output: "p".to_string(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .expect("well-formed box goal");
        assert!(!engine.fitness(&box_goal));
    }

    #[test]
    fn fitness_rejects_transcendental_smt_goal() {
        // exp(x) >= 0 on [0,1] -- an Apply is not a polynomial, so not a fit.
        let engine = ClarabelSosEngine::unwired();
        let exp_x = SmtExpr::Apply("exp".to_string(), vec![var("x")]);
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(0.0)),
                cmp(CmpOp::Le, var("x"), SmtExpr::RealLit(1.0)),
            ],
            postcondition: cmp(CmpOp::Ge, exp_x, SmtExpr::RealLit(0.0)),
        });
        assert!(!engine.fitness(&goal));
        assert!(matches!(goal.shape, GoalShape::Smt(_)));
    }

    // --- the honesty contract (soundness-critical) ---

    #[test]
    fn verified_certificate_yields_certificate_bearing_at_exact() {
        let engine =
            ClarabelSosEngine::with_proposer(FixedProposer(good_cert_for_x_squared(-1, 1)));
        let discharge = engine.discharge(&x_sq_nonneg(-1.0, 1.0), 1_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(
            discharge
                .qualifier_set()
                .contains(Qualifier::CertificateBearing)
        );
        // The evidence carries the certificate for downstream re-checking.
        assert_eq!(
            discharge.evidence().get("outcome").and_then(|v| v.as_str()),
            Some("proved")
        );
        assert!(discharge.evidence().get("certificate").is_some());
    }

    #[test]
    fn unwired_proposer_yields_unknown_never_exact() {
        // Standalone (no proposer wired): every recognized goal is honest
        // Unknown@Untrusted, NEVER a proof.
        let engine = ClarabelSosEngine::unwired();
        let discharge = engine.discharge(&x_sq_nonneg(-1.0, 1.0), 1_000);
        assert_eq!(*discharge.result(), TierBResult::Unknown);
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
        assert!(
            !discharge
                .qualifier_set()
                .contains(Qualifier::CertificateBearing)
        );
    }

    #[test]
    fn certificate_failing_psd_check_yields_unknown_never_exact() {
        // The proposer hands back a cert whose polynomial identity HOLDS but
        // whose Gram is NOT PSD (the boundary-degeneracy failure mode). The
        // engine must reject it to Unknown@Untrusted -- NEVER Exact, NEVER a
        // false certificate. This is the soundness-critical negative the RT
        // hammers hardest.
        // Gram [[1,2],[2,1]] (indefinite) has quadratic form 1 + 4x + x^2.
        let bad = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 4, 1]),
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 2, 2, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        // Use a goal whose polynomial is 1 + 4x + x^2 so the cert's polynomial
        // matches the goal (isolating the PSD failure, not a poly mismatch).
        let x = var("x");
        let body = SmtExpr::Arith(
            ArithOp::Add,
            Box::new(SmtExpr::Arith(
                ArithOp::Add,
                Box::new(SmtExpr::RealLit(1.0)),
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::RealLit(4.0)),
                    Box::new(x.clone()),
                )),
            )),
            Box::new(SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(x.clone()),
                Box::new(x),
            )),
        );
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(-1.0)),
                cmp(CmpOp::Le, var("x"), SmtExpr::RealLit(1.0)),
            ],
            postcondition: cmp(CmpOp::Ge, body, SmtExpr::RealLit(0.0)),
        });
        let engine = ClarabelSosEngine::with_proposer(FixedProposer(bad));
        let discharge = engine.discharge(&goal, 1_000);
        assert_eq!(*discharge.result(), TierBResult::Unknown);
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
        assert_eq!(
            discharge
                .evidence()
                .get("detail")
                .and_then(|d| d.get("cert_error"))
                .and_then(|v| v.as_str()),
            Some("not_psd")
        );
    }

    #[test]
    fn certificate_with_polynomial_mismatch_yields_unknown_never_exact() {
        // A PSD Gram (identity, form 1 + x^2) but the cert's claimed poly is x^2
        // -- the verify step's polynomial-identity check fails because the
        // combined quadratic form (1 + x^2) != poly (x^2). Unknown, never Exact.
        let mismatch = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[0, 0, 1]), // x^2
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 0, 0, 1]), // form 1 + x^2 != x^2
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        let engine = ClarabelSosEngine::with_proposer(FixedProposer(mismatch));
        let discharge = engine.discharge(&x_sq_nonneg(-1.0, 1.0), 1_000);
        assert_eq!(*discharge.result(), TierBResult::Unknown);
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
        assert_eq!(
            discharge
                .evidence()
                .get("detail")
                .and_then(|d| d.get("cert_error"))
                .and_then(|v| v.as_str()),
            Some("polynomial_mismatch")
        );
    }

    #[test]
    fn strict_goal_with_nonstrict_certificate_is_not_proved() {
        // A strict goal x^2 + 1 > 0 on [-1,1]; the proposer hands back a cert for
        // exactly poly = 1 + x^2 (proves >= 0, not > 0). The engine must NOT
        // upgrade a >= 0 cert to a strict proof -> Unknown.
        let x = var("x");
        let body = SmtExpr::Arith(
            ArithOp::Add,
            Box::new(SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(x.clone()),
                Box::new(x),
            )),
            Box::new(SmtExpr::RealLit(1.0)),
        );
        let goal = Goal::smt(SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(-1.0)),
                cmp(CmpOp::Le, var("x"), SmtExpr::RealLit(1.0)),
            ],
            postcondition: cmp(CmpOp::Gt, body, SmtExpr::RealLit(0.0)), // strict
        });
        // Cert for poly = 1 + x^2 (equals the goal poly), Gram identity (PSD).
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 0, 1]),
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 0, 0, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        let engine = ClarabelSosEngine::with_proposer(FixedProposer(cert));
        let discharge = engine.discharge(&goal, 1_000);
        assert_eq!(
            *discharge.result(),
            TierBResult::Unknown,
            "a >= 0 certificate must not prove a strict > 0 goal"
        );
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }

    #[test]
    fn interval_multiplier_certificate_verifies_at_exact() {
        // On [0,1]: p(x) = x - x^2 >= 0. Cert: multiplier (x)(1-x) = x - x^2,
        // Gram [1] (sigma1 = 1). Combined form: (x - x^2) * 1 = x - x^2 = p.
        let goal = {
            let x = var("x");
            // x - x^2
            let body = SmtExpr::Arith(
                ArithOp::Sub,
                Box::new(x.clone()),
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(x.clone()),
                    Box::new(x),
                )),
            );
            Goal::smt(SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![
                    cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(0.0)),
                    cmp(CmpOp::Le, var("x"), SmtExpr::RealLit(1.0)),
                ],
                postcondition: cmp(CmpOp::Ge, body, SmtExpr::RealLit(0.0)),
            })
        };
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[0, 1, -1]), // x - x^2
            lo: ri(0),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(1, &[1]),
                multiplier: RationalPoly::from_int_coeffs(&[0, 1, -1]), // x - x^2
            }],
        };
        let engine = ClarabelSosEngine::with_proposer(FixedProposer(cert));
        let discharge = engine.discharge(&goal, 1_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(
            discharge
                .qualifier_set()
                .contains(Qualifier::CertificateBearing)
        );
    }

    #[test]
    fn cert_evidence_round_trips_rationals_as_n_over_d() {
        // The evidence renders Gram entries and the poly as exact n/d strings.
        let cert = SosCertificate {
            poly: RationalPoly::from_coeffs(vec![r(1, 2), ri(0), ri(1)]),
            lo: ri(0),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: RatSymMatrix::from_rows(1, vec![r(3, 4)]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        let ev = cert_evidence(&cert);
        assert_eq!(ev.get("lo").and_then(|v| v.as_str()), Some("0/1"));
        let poly0 = ev
            .get("poly")
            .and_then(|p| p.get(0))
            .and_then(|v| v.as_str());
        assert_eq!(poly0, Some("1/2"));
        let g00 = ev
            .get("blocks")
            .and_then(|b| b.get(0))
            .and_then(|b| b.get("gram"))
            .and_then(|g| g.get(0))
            .and_then(|row| row.get(0))
            .and_then(|v| v.as_str());
        assert_eq!(g00, Some("3/4"));
    }
}
