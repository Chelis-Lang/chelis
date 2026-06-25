//! WI-13 special-function envelope library: certified `erf` envelopes.
//!
//! A *sound* over-approximation of `erf` over the real line, as committed data
//! the runtime discharge and Beacon's `erf` relaxation (WI-B6) consume. The
//! envelope is a piecewise structure of boxes; each box bounds `erf` on its
//! interval by `approx(x) +- eps`, where `approx` is either a saturating
//! constant (the tails) or a polynomial (the central region) and `eps` is a
//! *rigorously certified* sup-norm error bound for that box.
//!
//! ## Trust model: Gappa proof term + Arb cross-check
//!
//! Every box's `eps` carries a machine-checkable **Gappa proof term**
//! (`proof_kind = Gappa`). Sollya proposes the approximation (the central remez
//! polynomial; the constant `+-1` for the tails) and a certified local Taylor
//! model of `erf` per sub-interval; Gappa machine-checks `|approx - T| <= bound`
//! (pure polynomial arithmetic — Gappa never sees `erf`/`exp`), which with the
//! certified Taylor remainder gives `|approx - erf| <= eps`. The committed proof
//! bundle (`data/erf_proof/`) is a set of Gappa scripts an auditor re-runs
//! through `gappa` to confirm the bound. See `docs/erf_envelope_regen.md`.
//!
//! Independently, the WI-14 Arb oracle ([`crate::arb_oracle`]) is the every-build
//! **cross-check**: it re-derives each box's `eps` by whole-box ball arithmetic
//! and confirms the committed `eps` bounds it. So each arm has both a Gappa proof
//! term and an Arb confirmation.
//!
//! ## No runtime Arb/Sollya/Gappa dependency
//!
//! This module is always compiled and links nothing — the committed envelope is
//! embedded via `include_str!` and evaluated with pure `f64` arithmetic. Sollya
//! and Gappa are *offline* tools; the Arb oracle is used only offline (to
//! generate the cross-check) and in *CI* (behind the `arb` feature). A deployed
//! release binary that evaluates the envelope gains no FLINT/Arb/Sollya/Gappa
//! link. The committed-data round-trip and the bound algebra are tested here
//! without any of them; the proof re-check and Arb cross-check are CI gates.
//!
//! ## Why saturation is mandatory
//!
//! The real `erf`-argument range on the finance desk box reaches the hundreds
//! (master WI-13 / the bake-off). Outside roughly `+-6`, `erf` is within `~1e-17`
//! of `+-1`, so a single central polynomial over `[-300, 300]` would be
//! absurdly loose. The tails are therefore *saturating constant* arms (`erf`
//! is `+-1` to a tiny certified `eps`), and only the central box carries a
//! polynomial. The tails are proved by the same machinery: Gappa checks
//! `|+-1 - T(x)| <= eps` with `T` the certified Taylor model of `erf`.

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

/// How a box's `eps` bound is certified. Recorded per box so the certification
/// basis of every arm is **machine-visible**, not implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofKind {
    /// A machine-checkable Gappa proof term: `eps` is proved by a committed
    /// bundle of Gappa scripts (under `data/erf_proof/`) that an auditor re-runs
    /// through `gappa`. Every committed arm uses this: the central polynomial
    /// (the remez approximation error bounded per sub-interval via a certified
    /// local Taylor model) AND both saturation tails (the constant `+-1`
    /// approximation, `|+-1 - T(x)|` bounded the same way). Gappa proves the
    /// polynomial bound; it never sees `erf`/`exp`, which it cannot model.
    Gappa,
    /// A rigorous Arb enclosure: `eps` is certified by the WI-14 Arb oracle
    /// (whole-box ball arithmetic), an independently-checkable bound but not a
    /// machine-checkable *proof term*. No committed arm uses this today (all
    /// three are Gappa-proved); it is the honest fallback for any future arm a
    /// tool genuinely cannot prove, so the basis stays machine-visible rather
    /// than a silent gap. The Arb certifier cross-checks every box's `eps` on
    /// every CI build regardless of `proof_kind`.
    ArbEnclosure,
}

/// One box of the piecewise envelope: an interval, an approximation arm, a
/// certified sup-norm error for that arm over that interval, and how that error
/// is certified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErfEnvelopeBox {
    /// Inclusive lower edge of the box.
    pub lo: f64,
    /// Inclusive upper edge of the box.
    pub hi: f64,
    /// The approximation used on this box.
    pub arm: ErfArm,
    /// Certified sup-norm error: `sup_{x in [lo,hi]} |arm.approx(x) - erf(x)| <= eps`.
    /// How it is certified is recorded in `proof_kind`; the Arb oracle
    /// re-validates it on every CI build either way.
    pub eps: f64,
    /// Whether `eps` is backed by a Gappa proof term or an Arb enclosure.
    pub proof_kind: ProofKind,
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
/// the source of truth; the generator scripts reproduce them and CI re-checks
/// the central arm's eps against its committed Gappa proof bundle and every
/// box's eps against the Arb oracle.
const ERF_ENVELOPE_JSON: &str = include_str!("../data/erf_envelope.json");

/// The committed Gappa proof-bundle manifest, embedded so a test can lock the
/// envelope's central eps to the value the Gappa proof actually machine-checks
/// (the proof term and the consumed bound must not drift apart).
#[cfg(test)]
const ERF_PROOF_MANIFEST_JSON: &str = include_str!("../data/erf_proof/manifest.json");

/// The committed f64-Horner rounding proof, embedded so a test can parse the
/// coefficients the proof is LITERALLY over and bind them to the consumer
/// polynomial (consumer == manifest == proven). Without this, a coefficient
/// could drift from the proven polynomial and ship undetected.
#[cfg(test)]
const ERF_CENTRAL_ROUNDING_GAPPA: &str = include_str!("../data/erf_proof/central_rounding.gappa");

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
    fn committed_proof_kind_is_gappa_for_every_arm() {
        // Option B: EVERY arm -- central polynomial and both saturation tails --
        // carries a machine-checkable Gappa proof term. The tails are proved by
        // the same subdivision + certified-Taylor-model + Gappa machinery as the
        // central arm (the constant +-1 approximation needs no erf/exp), so there
        // is no proof-vs-enclosure split: all three boxes are proof_kind = gappa.
        // (ProofKind::ArbEnclosure remains a valid variant for any future arm a
        // tool genuinely cannot prove; the Arb certifier still cross-checks every
        // box's eps on every build regardless of proof_kind.)
        let env = ErfEnvelope::committed();
        for b in &env.boxes {
            assert_eq!(
                b.proof_kind,
                ProofKind::Gappa,
                "every committed arm [{}, {}] must carry the Gappa proof term",
                b.lo,
                b.hi
            );
        }
        let gappa = env
            .boxes
            .iter()
            .filter(|b| b.proof_kind == ProofKind::Gappa)
            .count();
        assert_eq!(
            gappa,
            env.boxes.len(),
            "all committed boxes are Gappa-proved"
        );
        assert!(gappa >= 3, "central + two tails");
    }

    #[test]
    fn tail_eps_matches_the_committed_gappa_proof_bundle() {
        // Each saturation tail's committed eps must equal the value its Gappa
        // proof bundle machine-checks (manifest tail_pos_eps / tail_neg_eps), so
        // the proved tail bound and the consumed tail bound cannot drift.
        let env = ErfEnvelope::committed();
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("proof manifest parses");
        let pos: f64 = manifest["tail_pos_eps"]
            .as_str()
            .unwrap()
            .parse()
            .expect("tail_pos_eps f64");
        let neg: f64 = manifest["tail_neg_eps"]
            .as_str()
            .unwrap()
            .parse()
            .expect("tail_neg_eps f64");
        let upper = env
            .boxes
            .iter()
            .find(|b| matches!(b.arm, ErfArm::Saturation { value } if value == 1.0))
            .expect("a +1 saturation box");
        let lower = env
            .boxes
            .iter()
            .find(|b| matches!(b.arm, ErfArm::Saturation { value } if value == -1.0))
            .expect("a -1 saturation box");
        assert_eq!(
            upper.eps, pos,
            "+1 tail eps must equal manifest tail_pos_eps"
        );
        assert_eq!(
            lower.eps, neg,
            "-1 tail eps must equal manifest tail_neg_eps"
        );
    }

    #[test]
    fn central_eps_matches_the_committed_gappa_proof_bundle() {
        // The central arm's committed eps must be exactly the value the Gappa
        // proof bundle machine-checks (manifest central_eps). If they drift, the
        // committed band is no longer the proved bound -- the proof term would
        // certify a different number than the runtime uses.
        let env = ErfEnvelope::committed();
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("proof manifest parses");
        let manifest_eps: f64 = manifest["central_eps"]
            .as_str()
            .expect("central_eps is a string")
            .parse()
            .expect("central_eps parses as f64");
        let central = env
            .boxes
            .iter()
            .find(|b| matches!(b.arm, ErfArm::Central { .. }))
            .expect("a central box");
        assert_eq!(
            central.eps, manifest_eps,
            "committed central eps {} must equal the Gappa-proved manifest eps {manifest_eps}",
            central.eps
        );
        assert_eq!(
            manifest["proof_kind"].as_str(),
            Some("gappa"),
            "the proof bundle must declare proof_kind gappa"
        );
    }

    #[test]
    fn central_eps_includes_the_f64_evaluation_rounding() {
        // Soundness: the committed central eps must bound the ACTUAL f64-evaluated
        // polynomial, so it is the math approximation error PLUS the Gappa-proved
        // f64-Horner evaluation rounding -- not a math-only bound (which would be
        // unsound by ~1 ULP). Lock central_eps == central_eps_math +
        // central_eps_f64_rounding, both recorded in the manifest.
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("proof manifest parses");
        let eps: f64 = manifest["central_eps"].as_str().unwrap().parse().unwrap();
        let math: f64 = manifest["central_eps_math"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        let rounding: f64 = manifest["central_eps_f64_rounding"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            eps,
            math + rounding,
            "central eps must be math ({math}) + f64-eval rounding ({rounding})"
        );
        assert!(
            rounding > 0.0,
            "the f64-eval rounding term must be positive"
        );
        // The committed envelope's central box carries this rounding-inclusive eps.
        let env = ErfEnvelope::committed();
        let central = env
            .boxes
            .iter()
            .find(|b| matches!(b.arm, ErfArm::Central { .. }))
            .unwrap();
        assert_eq!(central.eps, eps);
    }

    #[test]
    fn central_coeffs_match_the_committed_gappa_proof_bundle() {
        // The committed central polynomial must be the exact polynomial the
        // Gappa proof is about (the manifest coeffs), or the proof certifies a
        // different polynomial than the runtime evaluates.
        let env = ErfEnvelope::committed();
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("proof manifest parses");
        let manifest_coeffs: Vec<f64> = manifest["coeffs"]
            .as_array()
            .expect("coeffs array")
            .iter()
            // Read as JSON numbers (the manifest stores the exact doubles as
            // numbers), through the same serde_json parser the envelope uses, so
            // the comparison is bit-exact.
            .map(|c| c.as_f64().unwrap())
            .collect();
        let central = env
            .boxes
            .iter()
            .find_map(|b| match &b.arm {
                ErfArm::Central { coeffs } => Some(coeffs),
                _ => None,
            })
            .expect("a central box");
        assert_eq!(
            central, &manifest_coeffs,
            "committed central coeffs must equal the Gappa-proved manifest coeffs"
        );
    }

    /// Parse a Gappa numeric literal to the SAME f64 the consumer envelope's
    /// coefficients are read as. The envelope coeffs come through serde_json's
    /// number parser; Rust's `str::parse::<f64>` can disagree with it by one ULP
    /// on some 16-17 digit decimals. To compare bit-exactly we route the Gappa
    /// literal through serde_json too (the literals are plain decimals serde
    /// accepts as JSON numbers).
    #[cfg(test)]
    fn parse_like_consumer(lit: &str) -> f64 {
        serde_json::from_str::<f64>(lit.trim())
            .unwrap_or_else(|_| panic!("gappa numeric literal {lit:?} parses as f64"))
    }

    /// Parse the central coefficients (descending degree) literally embedded in
    /// the committed f64-Horner rounding proof's Horner chain:
    ///   `a{deg} = <coeff[0]>;`
    ///   `a{deg-i} rnd= a{deg-i+1} * x + <coeff[i]>;`  for i = 1..deg
    /// These are the exact doubles the Gappa rounding proof is over.
    #[cfg(test)]
    fn proven_central_coeffs() -> Vec<f64> {
        let text = ERF_CENTRAL_ROUNDING_GAPPA;
        let mut head_deg = None;
        let mut head_c = None;
        // The leading `aN = <c>;` line (no `rnd=`).
        for line in text.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix('a')
                && let Some(eq) = rest.find(" = ")
            {
                let deg: usize = rest[..eq].parse().expect("a{deg} index");
                let c = rest[eq + 3..].trim_end_matches(';').trim();
                head_deg = Some(deg);
                head_c = Some(parse_like_consumer(c));
                break;
            }
        }
        let deg = head_deg.expect("leading a{deg} = c line");
        let mut coeffs = vec![head_c.unwrap()];
        for i in 1..=deg {
            let idx = deg - i;
            // line: `a{idx} rnd= a{idx+1} * x + <c>;`
            let needle = format!("a{idx} rnd=");
            let line = text
                .lines()
                .map(str::trim)
                .find(|l| l.starts_with(&needle))
                .unwrap_or_else(|| panic!("a{idx} rnd= line"));
            let plus = line.rfind('+').expect("+ <coeff>");
            let c = line[plus + 1..].trim_end_matches(';').trim();
            coeffs.push(parse_like_consumer(c));
        }
        coeffs
    }

    #[test]
    fn committed_coeffs_are_bound_to_the_proven_polynomial() {
        // Proof-integrity guard: the consumer's central coefficients must be
        // bit-identical to the polynomial the committed Gappa rounding proof is
        // literally over. This closes the chain consumer == manifest == PROVEN:
        // without it, a coefficient could drift from the proven polynomial (the
        // per-sub-interval proofs embed q = p - T, not p, so they do not pin p;
        // central_rounding.gappa embeds p's coefficients directly).
        let proven = proven_central_coeffs();
        let env = ErfEnvelope::committed();
        let consumer = env
            .boxes
            .iter()
            .find_map(|b| match &b.arm {
                ErfArm::Central { coeffs } => Some(coeffs.clone()),
                _ => None,
            })
            .expect("a central box");
        assert_eq!(
            proven.len(),
            consumer.len(),
            "proven coeff count {} != consumer {}",
            proven.len(),
            consumer.len()
        );
        for (i, (p, c)) in proven.iter().zip(consumer.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "central coeff[{i}] in the consumer ({c:?}) is not the value the \
                 Gappa rounding proof certifies ({p:?})"
            );
        }
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
            proof_kind: ProofKind::Gappa,
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
                    proof_kind: ProofKind::Gappa,
                },
                // gap: starts at 2.0, not 1.0
                ErfEnvelopeBox {
                    lo: 2.0,
                    hi: 3.0,
                    arm: ErfArm::Central { coeffs: vec![0.0] },
                    eps: 0.1,
                    proof_kind: ProofKind::Gappa,
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
                proof_kind: ProofKind::ArbEnclosure,
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
                proof_kind: ProofKind::Gappa,
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
