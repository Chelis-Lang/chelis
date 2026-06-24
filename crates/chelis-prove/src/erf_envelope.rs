//! WI-13 special-function envelope library: certified `erf` envelopes.
//!
//! A *sound* over-approximation of `erf` over the real line, as committed data
//! the runtime discharge and Beacon's `erf` relaxation (WI-B6) consume. The
//! envelope is a piecewise structure of boxes; each box bounds `erf` on its
//! interval by `approx(x) +- eps`, where `approx` is either a saturating
//! constant (the tails) or a polynomial (the central region) and `eps` is a
//! *rigorously certified* sup-norm error bound for that box.
//!
//! ## Trust anchor
//!
//! `eps` is certified by the WI-14 Arb oracle ([`crate::arb_oracle`]) — the
//! single rigorous trust anchor for the special-function fragment. The
//! generator ([`scripts/generate_erf_envelope.py`]) *proposes* the polynomial
//! coefficients (a float computation: Remez / least-squares), and the Arb
//! certifier *stamps* `eps` as a sound upper bound of
//! `sup_{x in box} |approx(x) - erf(x)|` computed by whole-box ball arithmetic
//! (not sampling). So a wrong proposed polynomial cannot produce an unsound
//! envelope: it only produces a *larger* certified `eps`. This proposer /
//! exact-verifier split mirrors the Clarabel SoS engine's float-proposer with
//! a rational-exact verifier.
//!
//! ## No runtime Arb dependency
//!
//! This module is always compiled and links nothing — the committed envelope
//! is embedded via `include_str!` and evaluated with pure `f64` arithmetic. The
//! Arb oracle is used only *offline* (to generate `eps`) and in *CI* (to
//! re-validate that each committed `eps` still bounds the truth, behind the
//! `arb` feature). A deployed release binary that evaluates the envelope gains
//! no FLINT/Arb link. The CI re-validation harness lives in
//! [`crate::arb_oracle`] tests; the committed-data round-trip and the bound
//! algebra are tested here without Arb.
//!
//! ## Why saturation is mandatory
//!
//! The real `erf`-argument range on the finance desk box reaches the hundreds
//! (master WI-13 / the bake-off). Outside roughly `+-6`, `erf` is within `~1e-17`
//! of `+-1`, so a single central polynomial over `[-300, 300]` would be
//! absurdly loose. The tails are therefore *saturating constant* arms (`erf`
//! is `+-1` to a tiny certified `eps`), and only the central box carries a
//! polynomial. The saturation arm's `eps` is certified the same way (whole-box
//! Arb bound of `|+-1 - erf(x)|` over the tail).

use serde::{Deserialize, Serialize};

/// The approximation used on one envelope box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ErfArm {
    /// A saturating constant: `erf(x) ~ value` over the box (the tails, where
    /// `value` is `+1` or `-1`). The box's `eps` bounds `|value - erf(x)|`.
    Saturation {
        /// The constant value (`+1.0` or `-1.0`).
        value: f64,
    },
    /// A polynomial approximation, coefficients in **descending** degree order
    /// (`coeffs[0]` is the highest-degree term, the last is the constant term),
    /// evaluated by Horner's method. The box's `eps` bounds `|p(x) - erf(x)|`.
    Central {
        /// Polynomial coefficients, highest degree first.
        coeffs: Vec<f64>,
    },
}

impl ErfArm {
    /// Evaluate the arm's approximation at `x` (the *center* value, before the
    /// `+-eps` band is applied). Horner for the polynomial; the constant for
    /// saturation.
    pub fn approx(&self, x: f64) -> f64 {
        match self {
            ErfArm::Saturation { value } => *value,
            ErfArm::Central { coeffs } => {
                // Horner over descending-degree coefficients.
                let mut acc = 0.0_f64;
                for &c in coeffs {
                    acc = acc * x + c;
                }
                acc
            }
        }
    }
}

/// One box of the piecewise envelope: an interval, an approximation arm, and a
/// certified sup-norm error for that arm over that interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErfEnvelopeBox {
    /// Inclusive lower edge of the box.
    pub lo: f64,
    /// Inclusive upper edge of the box.
    pub hi: f64,
    /// The approximation used on this box.
    pub arm: ErfArm,
    /// Certified sup-norm error: `sup_{x in [lo,hi]} |arm.approx(x) - erf(x)| <= eps`,
    /// rigorously certified by the WI-14 Arb oracle (whole-box ball arithmetic).
    pub eps: f64,
}

impl ErfEnvelopeBox {
    /// Whether `x` lies in this box's `[lo, hi]` (inclusive).
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// The sound bound `[approx(x) - eps, approx(x) + eps]` on `erf(x)` from
    /// this box. Because `eps` is a certified sup-norm error over the box, the
    /// true `erf(x)` is guaranteed to lie in this interval for any `x` in the
    /// box.
    pub fn bound(&self, x: f64) -> (f64, f64) {
        let a = self.arm.approx(x);
        (a - self.eps, a + self.eps)
    }
}

/// Provenance for the committed envelope: how it was generated and certified,
/// so a reader can reproduce it and CI can re-validate it. None of these fields
/// affect the soundness of the bound (that rests only on each box's certified
/// `eps`); they are an audit trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErfEnvelopeProvenance {
    /// Generator script version (`scripts/generate_erf_envelope.py`).
    pub generator_version: String,
    /// Arb working precision (bits) used to certify each `eps`.
    pub certify_prec: i64,
    /// Number of sub-boxes the certifier split each box into to beat the ball
    /// arithmetic dependency problem.
    pub certify_subdivisions: usize,
    /// A note on the certification method, for the audit trail.
    pub method: String,
}

/// The full piecewise `erf` envelope: ordered, non-overlapping boxes covering
/// the supported argument range, plus provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErfEnvelope {
    /// Boxes in ascending `lo` order, expected contiguous and non-overlapping
    /// across the covered range.
    pub boxes: Vec<ErfEnvelopeBox>,
    /// How this envelope was generated and certified.
    pub provenance: ErfEnvelopeProvenance,
}

/// The committed, certified envelope, embedded at compile time. The bytes are
/// the source of truth; the generator script reproduces them and CI re-checks
/// each `eps` against the Arb oracle.
const ERF_ENVELOPE_JSON: &str = include_str!("../data/erf_envelope.json");

impl ErfEnvelope {
    /// The canonical committed envelope.
    ///
    /// # Panics
    /// Panics if the embedded data fails to parse, which would mean a corrupt
    /// committed artifact — a build-time invariant, caught by tests.
    pub fn committed() -> Self {
        serde_json::from_str(ERF_ENVELOPE_JSON)
            .expect("committed erf envelope data must be valid JSON")
    }

    /// Find the box containing `x`, if any. Boxes are inclusive on both edges,
    /// so a shared boundary point resolves to the first (lower) box.
    pub fn box_for(&self, x: f64) -> Option<&ErfEnvelopeBox> {
        self.boxes.iter().find(|b| b.contains(x))
    }

    /// The sound bound `[lo, hi]` on `erf(x)` from the envelope, or `None` if
    /// `x` is outside the covered range. This is the consumer entry point: the
    /// runtime discharge and Beacon's relaxation read `erf(x)`'s certified
    /// bound from here with no Arb call.
    pub fn bound(&self, x: f64) -> Option<(f64, f64)> {
        self.box_for(x).map(|b| b.bound(x))
    }

    /// Whether the boxes are well formed: each `lo <= hi`, ascending and
    /// contiguous (each box's `lo` equals the previous box's `hi`), every
    /// `eps >= 0` and finite, and saturation values are exactly `+-1`. This is
    /// a structural invariant of the committed data, locked by a test; it is
    /// not a soundness check (soundness is the per-box certified `eps`).
    pub fn is_well_formed(&self) -> bool {
        if self.boxes.is_empty() {
            return false;
        }
        let mut prev_hi: Option<f64> = None;
        for b in &self.boxes {
            if b.lo > b.hi {
                return false;
            }
            if !b.eps.is_finite() || b.eps < 0.0 {
                return false;
            }
            if let ErfArm::Saturation { value } = b.arm
                && value != 1.0
                && value != -1.0
            {
                return false;
            }
            if let Some(p) = prev_hi
                && b.lo != p
            {
                // Not contiguous: this box does not start where the last ended.
                return false;
            }
            prev_hi = Some(b.hi);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    //! Always-present tests: the committed-data round-trip, the structural
    //! invariant, and the bound algebra. No Arb link (the rigorous
    //! re-certification of each `eps` against the truth is the `arb`-gated CI
    //! harness in `arb_oracle`).
    use super::*;

    #[test]
    fn committed_envelope_parses_and_is_well_formed() {
        let env = ErfEnvelope::committed();
        assert!(
            env.is_well_formed(),
            "committed envelope must be contiguous, ascending, eps>=0, sat=+-1"
        );
        assert!(!env.boxes.is_empty());
    }

    #[test]
    fn committed_envelope_has_saturation_and_central_arms() {
        let env = ErfEnvelope::committed();
        let has_sat = env
            .boxes
            .iter()
            .any(|b| matches!(b.arm, ErfArm::Saturation { .. }));
        let has_central = env
            .boxes
            .iter()
            .any(|b| matches!(b.arm, ErfArm::Central { .. }));
        assert!(
            has_sat,
            "envelope must have saturation tails (range +-100..+-300)"
        );
        assert!(has_central, "envelope must have a central polynomial arm");
    }

    #[test]
    fn committed_envelope_covers_the_finance_range() {
        // The desk box reaches +-300 (master WI-13). The covered range must
        // include it, or the saturation arm is not doing its job.
        let env = ErfEnvelope::committed();
        assert!(env.bound(300.0).is_some(), "+300 must be covered");
        assert!(env.bound(-300.0).is_some(), "-300 must be covered");
        assert!(env.bound(0.0).is_some(), "0 must be covered");
    }

    #[test]
    fn bound_contains_known_erf_values() {
        // The committed eps must make the band contain the true erf at sample
        // points. (This is a sanity check against the committed data using
        // hard-coded high-precision truths; the RIGOROUS whole-box re-check is
        // the arb-gated harness. Here we only confirm the committed band is not
        // obviously too tight at a few points.)
        let env = ErfEnvelope::committed();
        let known: &[(f64, f64)] = &[
            (0.0, 0.0),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
            (-1.0, -0.8427007929497149),
            (100.0, 1.0),
        ];
        for &(x, truth) in known {
            let (lo, hi) = env.bound(x).expect("covered");
            assert!(
                lo <= truth && truth <= hi,
                "erf({x})={truth} must be in committed band [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn out_of_range_is_none() {
        let env = ErfEnvelope::committed();
        // Beyond the covered range, the envelope declines to bound (the caller
        // must widen coverage rather than silently get a wrong answer).
        let max_hi = env.boxes.last().unwrap().hi;
        assert!(env.bound(max_hi + 1.0).is_none());
    }

    #[test]
    fn arm_approx_horner_matches_manual() {
        // p(x) = 2x^2 + 3x + 5, coeffs descending [2,3,5].
        let arm = ErfArm::Central {
            coeffs: vec![2.0, 3.0, 5.0],
        };
        assert_eq!(arm.approx(0.0), 5.0);
        assert_eq!(arm.approx(1.0), 10.0);
        assert_eq!(arm.approx(2.0), 19.0);
    }

    #[test]
    fn saturation_arm_is_constant() {
        let arm = ErfArm::Saturation { value: 1.0 };
        assert_eq!(arm.approx(7.0), 1.0);
        assert_eq!(arm.approx(300.0), 1.0);
    }

    #[test]
    fn bound_band_is_symmetric_around_approx() {
        let b = ErfEnvelopeBox {
            lo: 0.0,
            hi: 1.0,
            arm: ErfArm::Central {
                coeffs: vec![1.0, 0.0],
            },
            eps: 0.01,
        };
        let (lo, hi) = b.bound(0.5);
        assert!((lo - 0.49).abs() < 1e-12);
        assert!((hi - 0.51).abs() < 1e-12);
    }

    #[test]
    fn well_formed_rejects_a_gap() {
        let env = ErfEnvelope {
            boxes: vec![
                ErfEnvelopeBox {
                    lo: 0.0,
                    hi: 1.0,
                    arm: ErfArm::Central { coeffs: vec![0.0] },
                    eps: 0.1,
                },
                // gap: starts at 2.0, not 1.0
                ErfEnvelopeBox {
                    lo: 2.0,
                    hi: 3.0,
                    arm: ErfArm::Central { coeffs: vec![0.0] },
                    eps: 0.1,
                },
            ],
            provenance: dummy_provenance(),
        };
        assert!(!env.is_well_formed(), "a coverage gap is not well formed");
    }

    #[test]
    fn well_formed_rejects_bad_saturation_value() {
        let env = ErfEnvelope {
            boxes: vec![ErfEnvelopeBox {
                lo: 0.0,
                hi: 1.0,
                arm: ErfArm::Saturation { value: 0.9 },
                eps: 0.1,
            }],
            provenance: dummy_provenance(),
        };
        assert!(!env.is_well_formed(), "saturation value must be +-1");
    }

    #[test]
    fn well_formed_rejects_negative_eps() {
        let env = ErfEnvelope {
            boxes: vec![ErfEnvelopeBox {
                lo: 0.0,
                hi: 1.0,
                arm: ErfArm::Central { coeffs: vec![0.0] },
                eps: -0.1,
            }],
            provenance: dummy_provenance(),
        };
        assert!(!env.is_well_formed(), "negative eps is not well formed");
    }

    fn dummy_provenance() -> ErfEnvelopeProvenance {
        ErfEnvelopeProvenance {
            generator_version: "test".to_string(),
            certify_prec: 128,
            certify_subdivisions: 1,
            method: "test".to_string(),
        }
    }
}
