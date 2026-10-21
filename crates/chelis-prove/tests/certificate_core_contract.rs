use chelis_certificate_core::exact::{RatSymMatrix, SosBlock, SosCertificate};
use chelis_certificate_core::poly::RationalPoly;
use chelis_prove::clarabel_sos::encode::SosLayout;
use chelis_prove::clarabel_sos::extract::PolyOnInterval;
use chelis_prove::clarabel_sos::{ClarabelSosEngine, SosProposer};
use chelis_prove::discharge::{Discharge, DischargeEngine, Goal, Qualifier, Soundness};
use chelis_prove::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
use chelis_prove::tier_b::{SmtProperty, TierBResult};
use num_bigint::BigInt;
use num_rational::BigRational;
use serde_json::json;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

fn goal(coeffs: &[i64], lo: i64, hi: i64, strict: bool) -> Goal {
    let x = || SmtExpr::Var("x".into());
    let cmp = |op, lhs, rhs| SmtExpr::Cmp(op, Box::new(lhs), Box::new(rhs));
    let mut body = SmtExpr::RealLit(0.0);
    for &coefficient in coeffs.iter().rev() {
        body = SmtExpr::Arith(
            ArithOp::Add,
            Box::new(SmtExpr::Arith(ArithOp::Mul, Box::new(body), Box::new(x()))),
            Box::new(SmtExpr::RealLit(coefficient as f64)),
        );
    }
    Goal::smt(SmtProperty {
        variables: vec![("x".into(), SmtSort::Real)],
        preconditions: vec![
            cmp(CmpOp::Ge, x(), SmtExpr::RealLit(lo as f64)),
            cmp(CmpOp::Le, x(), SmtExpr::RealLit(hi as f64)),
        ],
        postcondition: cmp(
            if strict { CmpOp::Gt } else { CmpOp::Ge },
            body,
            SmtExpr::RealLit(0.0),
        ),
    })
}

fn canonical(coeffs: &[i64], lo: i64, hi: i64, vars: &[i64]) -> SosCertificate {
    let poly = RationalPoly::from_int_coeffs(coeffs);
    let layout = SosLayout::for_goal(poly.degree().unwrap_or(0), &q(lo), &q(hi));
    layout.certificate_from_vars(
        &vars.iter().map(|&n| q(n)).collect::<Vec<_>>(),
        &poly,
        &q(lo),
        &q(hi),
    )
}

struct Candidate(SosCertificate);
impl SosProposer for Candidate {
    fn propose(&self, _: &PolyOnInterval) -> Option<SosCertificate> {
        Some(self.0.clone())
    }
}

fn admit(cert: SosCertificate, request: &Goal) -> Discharge {
    ClarabelSosEngine::with_proposer(Candidate(cert)).discharge(request, 5_000)
}

fn nonproof(discharge: &Discharge) {
    assert_eq!(*discharge.result(), TierBResult::Unknown);
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    assert_eq!(discharge.evidence()["engine"], "clarabel_sos");
    assert!(discharge.evidence()["backend"].get("certificate").is_none());
}

#[test]
fn canonical_production_layouts_preserve_exact_admission() {
    for (coeffs, lo, hi, vars) in [
        (vec![1], -1, 1, vec![1]),
        (vec![1, 0, 1], -1, 1, vec![1, 0, 1, 0]),
        (vec![0, 1], 0, 1, vec![1, 0]),
        (vec![0, 0, 0, 1], 0, 1, vec![0, 0, 1, 0, 0, 0]),
        (vec![0], -1, 1, vec![0]),
        (vec![0, 0, 1], 0, 0, vec![0, 0, 1, 0]),
        (vec![1, 2, 1], -1, 1, vec![1, 1, 1, 0]),
    ] {
        let cert = canonical(&coeffs, lo, hi, &vars);
        assert_eq!(cert.verify(), Ok(()));
        let discharge = admit(cert, &goal(&coeffs, lo, hi, false));
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(
            discharge
                .qualifier_set()
                .contains(Qualifier::CertificateBearing)
        );
        assert_eq!(discharge.evidence()["engine"], "clarabel_sos");
    }
}

#[test]
fn evidence_preserves_exact_values_and_block_order() {
    let cert = canonical(&[1, 0, 1], -1, 1, &[1, 0, 1, 0]);
    let discharge = admit(cert, &goal(&[1, 0, 1], -1, 1, false));
    assert_eq!(
        discharge.evidence()["backend"],
        json!({
            "outcome": "proved",
            "strict": false,
            "certificate": {
                "lo": "-1/1", "hi": "1/1", "poly": ["1/1", "0/1", "1/1"],
                "blocks": [
                    {"gram": [["1/1", "0/1"], ["0/1", "1/1"]], "multiplier": ["1/1"]},
                    {"gram": [["0/1"]], "multiplier": ["1/1", "0/1", "-1/1"]}
                ]
            }
        })
    );
}

#[test]
fn goal_domain_and_strictness_cannot_gain_authority() {
    let cert = canonical(&[0, 0, 1], -1, 1, &[0, 0, 1, 0]);
    nonproof(&admit(cert.clone(), &goal(&[-1, 0, 1], -1, 1, false)));
    nonproof(&admit(cert.clone(), &goal(&[0, 0, 1], -1, 2, false)));
    nonproof(&admit(cert.clone(), &goal(&[0, 0, 1], -1, 1, true)));
    nonproof(&admit(cert, &goal(&[1, 0, 1], -1, 1, true)));
    let reversed = goal(&[1], 2, -2, false);
    assert!(!ClarabelSosEngine::unwired().fitness(&reversed));
    nonproof(&ClarabelSosEngine::unwired().discharge(&reversed, 1_000));
}

#[test]
fn negative_altered_and_swapped_multipliers_cannot_gain_authority() {
    let negative = SosCertificate {
        poly: RationalPoly::from_int_coeffs(&[-1]),
        lo: q(0),
        hi: q(1),
        blocks: vec![SosBlock {
            gram: RatSymMatrix::from_rows(1, vec![q(1)]),
            multiplier: RationalPoly::from_int_coeffs(&[-1]),
        }],
    };
    assert_eq!(negative.verify(), Ok(()));
    nonproof(&admit(negative, &goal(&[-1], 0, 1, false)));
    let mut altered = canonical(&[0, 0, 1], -1, 1, &[0, 0, 1, 0]);
    altered.blocks[1].multiplier = RationalPoly::from_int_coeffs(&[-999]);
    assert_eq!(altered.verify(), Ok(()));
    nonproof(&admit(altered, &goal(&[0, 0, 1], -1, 1, false)));
    let mut swapped = canonical(&[0, 1], 0, 1, &[1, 0]);
    swapped.blocks.swap(0, 1);
    assert_eq!(swapped.verify(), Ok(()));
    nonproof(&admit(swapped, &goal(&[0, 1], 0, 1, false)));
}

#[test]
fn missing_extra_and_wrong_size_blocks_cannot_gain_authority() {
    let cert = canonical(&[1], 0, 1, &[1]);
    let mut extra = cert.clone();
    extra.blocks.push(SosBlock {
        gram: RatSymMatrix::from_rows(1, vec![q(0)]),
        multiplier: RationalPoly::from_int_coeffs(&[1]),
    });
    assert_eq!(extra.verify(), Ok(()));
    nonproof(&admit(extra, &goal(&[1], 0, 1, false)));
    let mut wrong_size = cert;
    wrong_size.blocks[0].gram = RatSymMatrix::from_rows(2, vec![q(1), q(0), q(0), q(0)]);
    assert_eq!(wrong_size.verify(), Ok(()));
    nonproof(&admit(wrong_size, &goal(&[1], 0, 1, false)));
    let mut missing = canonical(&[0, 0, 1], -1, 1, &[0, 0, 1, 0]);
    missing.blocks.pop();
    assert_eq!(missing.verify(), Ok(()));
    nonproof(&admit(missing, &goal(&[0, 0, 1], -1, 1, false)));
}

#[test]
fn public_errors_preserve_host_labels() {
    for (entries, coeffs, label) in [
        ([1, 1, 0, 1], vec![1, 1, 1], "not_symmetric"),
        ([1, 2, 2, 1], vec![1, 4, 1], "not_psd"),
        ([1, 0, 0, 1], vec![0, 0, 1], "polynomial_mismatch"),
    ] {
        let mut cert = canonical(&coeffs, -1, 1, &[0, 0, 1, 0]);
        cert.blocks[0].gram = RatSymMatrix::from_rows(2, entries.map(q).to_vec());
        let discharge = admit(cert, &goal(&coeffs, -1, 1, false));
        nonproof(&discharge);
        assert_eq!(
            discharge.evidence()["backend"]["detail"]["cert_error"],
            label
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn native_clarabel_production_parity() {
    let engine = ClarabelSosEngine::clarabel();
    for (coeffs, lo, hi) in [
        (vec![1], -1, 1),
        (vec![1, 0, 1], -1, 1),
        (vec![0, 1], 0, 1),
        (vec![0, 0, 1], 0, 0),
    ] {
        let discharge = engine.discharge(&goal(&coeffs, lo, hi, false), 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert_eq!(discharge.evidence()["engine"], "clarabel_sos");
    }
    nonproof(&engine.discharge(&goal(&[0], -1, 1, false), 5_000));
    nonproof(&engine.discharge(&goal(&[-1], -1, 1, false), 5_000));
    nonproof(&engine.discharge(&goal(&[1, 0, 1], -1, 1, true), 5_000));
}
