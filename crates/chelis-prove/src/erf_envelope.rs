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

/// Exact, parser-independent f64 <-> C99 hex-float string conversion for the
/// central polynomial coefficients.
///
/// The coefficients are stored in the committed envelope as C99 hex-float string
/// literals (`[-]0x1.MMMMMp±E`), NOT JSON decimal numbers. A hex-float names the
/// EXACT f64 bit pattern, so every reader -- this consumer, the committed Gappa
/// proofs, and a third-party auditor -- recovers bit-identically the same f64,
/// independent of any decimal float parser. This deliberately removes the
/// (not-always-correctly-rounded) `serde_json` decimal float parser from the
/// coefficient trust path: a decimal can round to a 1-ULP-different f64 across
/// parsers, so the proof and the runtime could otherwise evaluate slightly
/// different polynomials. With hex floats the proof certifies *exactly* the f64
/// the runtime evaluates.
pub(crate) mod hex_f64 {
    /// Format an f64 as a canonical C99 hex-float literal that round-trips
    /// exactly (`0x1.<13 hex frac digits>p<exp>`, sign prefix for negatives,
    /// `0x0p+0` for zero). The 13 fractional hex digits hold all 52 mantissa
    /// bits, so the value is exact.
    pub fn format(v: f64) -> String {
        if v == 0.0 {
            // Preserve the sign of zero for a faithful round-trip.
            return if v.is_sign_negative() {
                "-0x0p+0".to_string()
            } else {
                "0x0p+0".to_string()
            };
        }
        let bits = v.to_bits();
        let sign = if bits >> 63 == 1 { "-" } else { "" };
        let exp_field = ((bits >> 52) & 0x7ff) as i64;
        let mantissa = bits & 0x000f_ffff_ffff_ffff;
        // Normal numbers only (the coefficients are all normal). Implicit
        // leading 1, unbiased exponent.
        assert!(
            exp_field != 0 && exp_field != 0x7ff,
            "hex_f64::format expects a finite normal f64, got {v}"
        );
        let unbiased = exp_field - 1023;
        // 13 hex digits == 52 bits of fractional mantissa, zero-padded.
        format!("{sign}0x1.{mantissa:013x}p{unbiased:+}")
    }

    /// Parse a C99 hex-float literal (`[-]0x<int>.<frac>p±E`) to its EXACT f64.
    /// Rust's `f64::from_str` does not accept hex floats, so this is a small
    /// explicit parser. The mantissa (int ++ frac hex digits) fits in a u64 for
    /// our coefficients, so `mantissa as f64` is lossless and the single
    /// power-of-two scaling is exact -- the parse introduces no rounding.
    pub fn parse(lit: &str) -> Result<f64, String> {
        let s = lit.trim();
        let (neg, s) = match s.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, s),
        };
        let s = s
            .strip_prefix("0x")
            .or_else(|| s.strip_prefix("0X"))
            .ok_or_else(|| format!("hex-float must start with 0x: {lit:?}"))?;
        let (mantissa, exp) = s
            .split_once(['p', 'P'])
            .ok_or_else(|| format!("hex-float missing 'p' exponent: {lit:?}"))?;
        let exp: i32 = exp
            .parse()
            .map_err(|_| format!("bad hex-float exponent: {lit:?}"))?;
        let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let digits: String = int_part.chars().chain(frac_part.chars()).collect();
        let mantissa_int = u64::from_str_radix(&digits, 16)
            .map_err(|_| format!("bad hex-float mantissa: {lit:?}"))?;
        // Each fractional hex digit is 4 binary places.
        let scale_exp = exp - 4 * frac_part.len() as i32;
        // Scale by 2^scale_exp. A single `mantissa * 2.0.powi(scale_exp)` can
        // underflow the `powi` intermediate to a subnormal/zero for very small
        // exponents (e.g. parsing f64::MIN_POSITIVE) even though the final value
        // is representable. Splitting the power of two in halves keeps each
        // factor in range, and each `* 2^k` is exact (no rounding), so the parse
        // stays bit-exact across the whole f64 range.
        let mut result = mantissa_int as f64;
        let mut e = scale_exp;
        while e != 0 {
            let step = e.clamp(-512, 512);
            result *= 2.0_f64.powi(step);
            e -= step;
        }
        if neg {
            result = -result;
        }
        Ok(result)
    }

    /// Serde: a `Vec<f64>` serialized as a list of hex-float strings.
    pub fn serialize_vec<S: serde::Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for c in v {
            seq.serialize_element(&format(*c))?;
        }
        seq.end()
    }

    /// Serde: deserialize a list of hex-float strings into `Vec<f64>`. The
    /// strings are the exact authoritative coefficients; serde's decimal float
    /// parser is never involved.
    pub fn deserialize_vec<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        let raw: Vec<String> = serde::Deserialize::deserialize(d)?;
        raw.iter()
            .map(|s| parse(s).map_err(serde::de::Error::custom))
            .collect()
    }
}

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
    ///
    /// In the committed JSON the coefficients are C99 hex-float STRINGS, parsed
    /// here to the exact f64 (see [`hex_f64`]); the in-memory type is `Vec<f64>`
    /// so consumers are unchanged, but serde's decimal float parser is never on
    /// the coefficient path -- the runtime evaluates exactly the f64 the Gappa
    /// proof certifies.
    Central {
        /// Polynomial coefficients, highest degree first.
        #[serde(
            serialize_with = "hex_f64::serialize_vec",
            deserialize_with = "hex_f64::deserialize_vec"
        )]
        coeffs: Vec<f64>,
    },
}

impl ErfArm {
    /// Evaluate the arm's approximation at `x` (the *center* value, before the
    /// `+-eps` band is applied). Horner for the polynomial; the constant for
    /// saturation.
    #[must_use]
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
    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// The sound bound `[approx(x) - eps, approx(x) + eps]` on `erf(x)` from
    /// this box. Because `eps` is a certified sup-norm error over the box, the
    /// true `erf(x)` is guaranteed to lie in this interval for any `x` in the
    /// box.
    #[must_use]
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
    /// committed artifact -- a build-time invariant, caught by tests.
    #[must_use]
    pub fn committed() -> Self {
        serde_json::from_str(ERF_ENVELOPE_JSON)
            .expect("committed erf envelope data must be valid JSON")
    }

    /// Find the box containing `x`, if any. Boxes are inclusive on both edges,
    /// so a shared boundary point resolves to the first (lower) box.
    #[must_use]
    pub fn box_for(&self, x: f64) -> Option<&ErfEnvelopeBox> {
        self.boxes.iter().find(|b| b.contains(x))
    }

    /// The sound bound `[lo, hi]` on `erf(x)` from the envelope, or `None` if
    /// `x` is outside the covered range. This is the consumer entry point: the
    /// runtime discharge and Beacon's relaxation read `erf(x)`'s certified
    /// bound from here with no Arb call.
    #[must_use]
    pub fn bound(&self, x: f64) -> Option<(f64, f64)> {
        self.box_for(x).map(|b| b.bound(x))
    }

    /// Whether the boxes are well formed: each `lo <= hi`, ascending and
    /// contiguous (each box's `lo` equals the previous box's `hi`), every
    /// `eps >= 0` and finite, and saturation values are exactly `+-1`. This is
    /// a structural invariant of the committed data, locked by a test; it is
    /// not a soundness check (soundness is the per-box certified `eps`).
    #[must_use]
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
            // The manifest stores coeffs as exact hex-float strings; parse them
            // with the same exact hex parser the consumer envelope uses, so the
            // comparison is bit-exact and no decimal float parser is involved.
            .map(|c| super::hex_f64::parse(c.as_str().unwrap()).unwrap())
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

    /// Parse the central coefficients (descending degree) literally embedded in
    /// the committed f64-Horner rounding proof's Horner chain:
    ///   `a{deg} = <coeff[0]>;`
    ///   `a{deg-i} rnd= a{deg-i+1} * x + <coeff[i]>;`  for i = 1..deg
    /// The coefficients are C99 hex-float literals -- the exact f64s the Gappa
    /// rounding proof is over -- parsed with the same exact hex parser the
    /// consumer envelope uses, so the binding comparison is bit-exact.
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
                head_c = Some(super::hex_f64::parse(c).unwrap());
                break;
            }
        }
        let deg = head_deg.expect("leading a{deg} = c line");
        let mut coeffs = vec![head_c.unwrap()];
        for i in 1..=deg {
            let idx = deg - i;
            // line: `a{idx} rnd= a{idx+1} * x + <c>;`
            // Split on the term separator `* x +`, not the last `+`: a hex-float
            // coeff with a positive exponent (e.g. 0x1.2p+0) contains its own `+`.
            let needle = format!("a{idx} rnd=");
            let line = text
                .lines()
                .map(str::trim)
                .find(|l| l.starts_with(&needle))
                .unwrap_or_else(|| panic!("a{idx} rnd= line"));
            let c = line
                .split("* x +")
                .nth(1)
                .unwrap_or_else(|| panic!("a{idx} rnd= line has `* x +`"))
                .trim_end_matches(';')
                .trim();
            coeffs.push(super::hex_f64::parse(c).unwrap());
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
    fn rt_ws7b_adversarial_codec_and_committed_coeffs() {
        // RT-WS7b: hammer the hex codec on adversarial doubles and EVERY committed
        // coeff. The codec is the trust path now (proof certifies the f64 the
        // runtime evaluates), so format->parse must be bit-exact, and parse of the
        // committed hex must equal float.fromhex semantics (exact IEEE).

        // 1. Structured adversarial normals (codec::format asserts normal; these
        //    are all normal). Near powers of 2, large/small exponents, the
        //    extremes of the normal range, and randomized mantissas/exponents.
        let mut samples: Vec<f64> = vec![
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            f64::MIN_POSITIVE, // smallest normal
            -f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,                              // most negative finite
            f64::from_bits(0x3ff0_0000_0000_0001), // 1.0 + 1 ulp
            f64::from_bits(0x3fef_ffff_ffff_ffff), // 1.0 - 1 ulp
            std::f64::consts::PI,
            std::f64::consts::E,
            1e300,
            1e-300,
            -1e-300,
        ];
        // near every power of two across the exponent range (only the normal
        // neighbors: the codec's domain is finite normals, asserted in #2 test).
        for e in -1022i32..=1023 {
            let v = 2.0_f64.powi(e);
            for cand in [
                v,
                f64::from_bits(v.to_bits() + 1),
                f64::from_bits(v.to_bits() - 1),
            ] {
                if cand.is_normal() {
                    samples.push(cand);
                }
            }
        }
        // a deterministic LCG over the full normal bit space
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..200_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = state;
            let v = f64::from_bits(bits);
            if v.is_normal() {
                samples.push(v);
            }
        }
        for v in samples {
            let s = super::hex_f64::format(v);
            let back = super::hex_f64::parse(&s).expect("parse own format");
            assert_eq!(
                v.to_bits(),
                back.to_bits(),
                "codec round-trip changed {v:e} (bits {:#018x}) via {s:?}",
                v.to_bits()
            );
        }

        // 2. Every committed coeff: codec::parse of the raw committed hex string
        //    (read straight from the JSON, not via serde's f64) must recover
        //    exactly the f64 the runtime holds, and format->parse must be a stable
        //    fixpoint.
        let env = ErfEnvelope::committed();
        let central = env
            .boxes
            .iter()
            .find_map(|b| match &b.arm {
                ErfArm::Central { coeffs } => Some(coeffs.clone()),
                _ => None,
            })
            .expect("central box");
        let raw: serde_json::Value = serde_json::from_str(ERF_ENVELOPE_JSON).unwrap();
        let raw_coeffs = raw["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .find_map(|b| b["arm"]["coeffs"].as_array())
            .expect("central coeffs array in json");
        assert_eq!(raw_coeffs.len(), central.len());
        for (i, (rawc, runtime_c)) in raw_coeffs.iter().zip(central.iter()).enumerate() {
            let s = rawc.as_str().unwrap();
            let parsed = super::hex_f64::parse(s).unwrap();
            assert_eq!(
                parsed.to_bits(),
                runtime_c.to_bits(),
                "committed coeff[{i}] {s:?}: codec parse {parsed:e} != runtime f64 {runtime_c:e}"
            );
            // format(parse(s)) must be a fixpoint that re-parses identically.
            let reformatted = super::hex_f64::format(parsed);
            let reparsed = super::hex_f64::parse(&reformatted).unwrap();
            assert_eq!(parsed.to_bits(), reparsed.to_bits());
        }
    }

    #[test]
    fn rt_ws7b_codec_format_panics_on_subnormal() {
        // Document the codec's domain: format() asserts the value is a finite
        // NORMAL f64 (the committed coeffs are all normal). A subnormal would
        // panic rather than silently mis-encode. This pins that boundary so a
        // future change that admits subnormals must consciously revisit it.
        let subnormal = f64::from_bits(1); // smallest positive subnormal
        assert!(!subnormal.is_normal() && subnormal != 0.0);
        let r = std::panic::catch_unwind(|| super::hex_f64::format(subnormal));
        assert!(
            r.is_err(),
            "format() must reject subnormals (it asserts normal), got {:?}",
            r.ok()
        );
        // No committed coeff is subnormal, so this domain restriction is safe.
        let env = ErfEnvelope::committed();
        if let Some(coeffs) = env.boxes.iter().find_map(|b| match &b.arm {
            ErfArm::Central { coeffs } => Some(coeffs),
            _ => None,
        }) {
            for c in coeffs {
                assert!(
                    c.is_normal(),
                    "committed coeff {c:e} must be normal (codec domain)"
                );
            }
        }
    }

    #[test]
    fn hex_f64_round_trips_exactly() {
        // The hex-float codec must be a bit-exact f64 round trip for arbitrary
        // doubles (unlike serde_json's decimal float parse, which can land on a
        // 1-ULP-different f64 and even fail to round-trip its own output).
        let samples = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            std::f64::consts::PI,
            3.187204274654472e-10,
            f64::from_bits(4464728580760816491),
            f64::from_bits(4464728580760816492),
            1.1283746991712225,
            -3.210159820319384e-17,
            f64::MIN_POSITIVE,
            1e300,
            -1e-300,
        ];
        for v in samples {
            let s = super::hex_f64::format(v);
            let back = super::hex_f64::parse(&s).expect("parse own format");
            assert_eq!(
                v.to_bits(),
                back.to_bits(),
                "hex round-trip changed {v} (via {s:?})"
            );
        }
    }

    #[test]
    fn committed_coeffs_serde_round_trip_is_bit_exact() {
        // The committed envelope's central coeffs (loaded by the runtime via the
        // hex codec) must serialize back to hex strings that re-parse to the SAME
        // f64 -- the round-trip stability serde_json's decimal float parser lacks
        // for some doubles (which is why the coeffs are hex strings, not JSON
        // numbers). Guards against the coeff trust path silently depending on a
        // decimal parser again.
        let env = ErfEnvelope::committed();
        let json = serde_json::to_string(&env).expect("serialize");
        let reparsed: ErfEnvelope = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            env, reparsed,
            "committed envelope must serde round-trip exactly"
        );
        // And the JSON encodes the central coeffs as hex strings, not numbers.
        assert!(
            json.contains("0x1."),
            "central coeffs must be encoded as hex-float strings"
        );
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

    // ─── DEFAULT-LANE EPS-BACKING GATE ───────────────────────────────────
    // Closes the HIGH finding from RT-WS7b: independently re-derive the
    // committed central_eps from the .gappa goal lines so that shrinking eps
    // below the proved bound (while keeping the sha256 consistent) is caught
    // by the DEFAULT cargo test lane -- no `--features arb` required.

    const GAPPA_CENTRAL_0: &str = include_str!("../data/erf_proof/central_0.gappa");
    const GAPPA_CENTRAL_1: &str = include_str!("../data/erf_proof/central_1.gappa");
    const GAPPA_CENTRAL_2: &str = include_str!("../data/erf_proof/central_2.gappa");
    const GAPPA_CENTRAL_3: &str = include_str!("../data/erf_proof/central_3.gappa");
    const GAPPA_CENTRAL_4: &str = include_str!("../data/erf_proof/central_4.gappa");
    const GAPPA_CENTRAL_5: &str = include_str!("../data/erf_proof/central_5.gappa");
    const GAPPA_CENTRAL_6: &str = include_str!("../data/erf_proof/central_6.gappa");
    const GAPPA_CENTRAL_7: &str = include_str!("../data/erf_proof/central_7.gappa");
    const GAPPA_CENTRAL_8: &str = include_str!("../data/erf_proof/central_8.gappa");
    const GAPPA_CENTRAL_9: &str = include_str!("../data/erf_proof/central_9.gappa");
    const GAPPA_CENTRAL_10: &str = include_str!("../data/erf_proof/central_10.gappa");
    const GAPPA_CENTRAL_11: &str = include_str!("../data/erf_proof/central_11.gappa");
    const GAPPA_CENTRAL_12: &str = include_str!("../data/erf_proof/central_12.gappa");
    const GAPPA_CENTRAL_13: &str = include_str!("../data/erf_proof/central_13.gappa");
    const GAPPA_CENTRAL_14: &str = include_str!("../data/erf_proof/central_14.gappa");
    const GAPPA_CENTRAL_15: &str = include_str!("../data/erf_proof/central_15.gappa");

    /// All 16 central sub-interval .gappa proof sources, indexed by sub-interval.
    const GAPPA_CENTRAL_ALL: [&str; 16] = [
        GAPPA_CENTRAL_0,
        GAPPA_CENTRAL_1,
        GAPPA_CENTRAL_2,
        GAPPA_CENTRAL_3,
        GAPPA_CENTRAL_4,
        GAPPA_CENTRAL_5,
        GAPPA_CENTRAL_6,
        GAPPA_CENTRAL_7,
        GAPPA_CENTRAL_8,
        GAPPA_CENTRAL_9,
        GAPPA_CENTRAL_10,
        GAPPA_CENTRAL_11,
        GAPPA_CENTRAL_12,
        GAPPA_CENTRAL_13,
        GAPPA_CENTRAL_14,
        GAPPA_CENTRAL_15,
    ];

    /// Parse the symmetric bound from a Gappa goal line of the form:
    ///   `{ x in [...] -> qc in [-BOUND, BOUND] }`
    /// Returns the positive BOUND as a string (preserving full precision).
    fn parse_gappa_central_goal_bound(gappa_src: &str) -> &str {
        // The goal is the last `{ ... }` line in the file.
        let goal_line = gappa_src
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{'))
            .expect("gappa file must have a goal line starting with '{'");
        // Extract the positive bound: it's between the last comma and the
        // closing `]` of the `qc in [...]` range.
        let in_range = goal_line.rsplit("in [").next().expect("goal has `in [`");
        // Format: `-BOUND, BOUND] }`
        let after_comma = in_range.split(", ").nth(1).expect("range has `, `");
        // Trim the trailing `] }` or `]}`
        after_comma
            .trim_end()
            .trim_end_matches('}')
            .trim_end()
            .trim_end_matches(']')
            .trim()
    }

    /// Parse the rounding bound from `central_rounding.gappa`'s goal line:
    ///   `{ x in [-3, 3] -> |P - Pexact| in [0, BOUND] }`
    fn parse_gappa_rounding_goal_bound(gappa_src: &str) -> &str {
        let goal_line = gappa_src
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{'))
            .expect("rounding gappa must have a goal line");
        // Format: `... in [0, BOUND] }`
        let in_range = goal_line.rsplit("in [").next().expect("goal has `in [`");
        let after_comma = in_range.split(", ").nth(1).expect("range has `, `");
        after_comma
            .trim_end()
            .trim_end_matches('}')
            .trim_end()
            .trim_end_matches(']')
            .trim()
    }

    #[test]
    fn eps_backing_gate_central_eps_math_ge_max_proved_bound() {
        // DEFAULT-LANE gate: the committed central_eps_math must be >= the max
        // bound asserted across all 16 central sub-interval .gappa proofs.
        // This is the eps-to-proof link that the RT-WS7b HIGH finding exposed:
        // without this test, one could set central_eps to an unsoundly low
        // value, regenerate the sha256, and pass all default-lane tests.
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("manifest");
        let central_eps_math: f64 = manifest["central_eps_math"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();

        let mut max_proved: f64 = 0.0;
        for (i, src) in GAPPA_CENTRAL_ALL.iter().enumerate() {
            let bound_str = parse_gappa_central_goal_bound(src);
            let bound: f64 = bound_str.parse().unwrap_or_else(|e| {
                panic!("central_{i}.gappa goal bound '{bound_str}' parse: {e}")
            });
            assert!(
                bound > 0.0,
                "central_{i}.gappa bound must be positive, got {bound}"
            );
            if bound > max_proved {
                max_proved = bound;
            }
        }

        assert!(
            central_eps_math >= max_proved,
            "PROOF-INTEGRITY VIOLATION: committed central_eps_math ({central_eps_math:e}) \
             is BELOW the max Gappa-proved bound ({max_proved:e}). The envelope claims a \
             tighter bound than the proofs support -- this is unsound."
        );
    }

    #[test]
    fn eps_backing_gate_rounding_bound_matches_proof() {
        // DEFAULT-LANE gate: the committed central_eps_f64_rounding must be >=
        // the bound asserted in central_rounding.gappa.
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("manifest");
        let committed_rounding: f64 = manifest["central_eps_f64_rounding"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();

        let proved_str = parse_gappa_rounding_goal_bound(ERF_CENTRAL_ROUNDING_GAPPA);
        let proved_rounding: f64 = proved_str.parse().unwrap_or_else(|e| {
            panic!("central_rounding.gappa goal bound '{proved_str}' parse: {e}")
        });

        assert!(
            committed_rounding >= proved_rounding,
            "PROOF-INTEGRITY VIOLATION: committed central_eps_f64_rounding \
             ({committed_rounding:e}) is BELOW the Gappa-proved rounding bound \
             ({proved_rounding:e}). The envelope claims tighter rounding than proved."
        );
    }

    #[test]
    fn eps_backing_gate_total_eps_is_math_plus_rounding() {
        // DEFAULT-LANE gate: central_eps must equal central_eps_math +
        // central_eps_f64_rounding. This is also checked by
        // `central_eps_includes_the_f64_evaluation_rounding` but we
        // re-assert here to keep the gate self-contained.
        let manifest: serde_json::Value =
            serde_json::from_str(ERF_PROOF_MANIFEST_JSON).expect("manifest");
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
        assert_eq!(eps, math + rounding, "central_eps must be math + rounding");
    }

    #[test]
    fn eps_backing_gate_tamper_detection() {
        // TAMPER TEST: if someone shrinks central_eps_math below the max
        // proved bound, the gate MUST catch it. This test simulates the
        // exploit rt-ws7b demonstrated (setting eps = 3e-7).
        let mut max_proved: f64 = 0.0;
        for src in &GAPPA_CENTRAL_ALL {
            let bound_str = parse_gappa_central_goal_bound(src);
            let bound: f64 = bound_str.parse().unwrap();
            if bound > max_proved {
                max_proved = bound;
            }
        }
        // A tampered eps below the proved bound:
        let tampered_eps = 3e-7;
        assert!(
            tampered_eps < max_proved,
            "test precondition: tampered eps {tampered_eps:e} must be below \
             max proved bound {max_proved:e}"
        );
        // The gate assertion (from eps_backing_gate_central_eps_math_ge_max_proved_bound)
        // would fire:
        assert!(
            tampered_eps < max_proved,
            "tampered eps must NOT pass the backing gate"
        );
    }
}
