//! Function-keyed special-function envelope: the generalization of the certified
//! `erf` envelope ([`crate::erf_envelope`]) to a registry of transcendental
//! functions `{erf, exp, log, sqrt}`.
//!
//! The `erf` envelope is a piecewise, per-box `approx(x) ± eps` sound
//! over-approximation whose `eps` is machine-checked (Sollya → Gappa) and
//! cross-checked (Arb). This module keeps that exact structure but drops the
//! erf-specific assumptions so any special function can carry an envelope:
//!
//! - **Saturation is per-function**, not hardcoded `±1`. `erf` saturates at `±1`;
//!   another function's tail constant (if it has one) is whatever its certificate
//!   proves. A function with no saturating tail (e.g. `exp`, `log` over a bounded
//!   box) simply has no `Saturation` arm.
//! - **The output clamp is per-function**, not the hardcoded `[-1, 1]` that lived
//!   in the abstract-subterm consumer. `erf`'s range is `[-1, 1]`; `sqrt`'s is
//!   `[0, ∞)`; `exp`/`log` have no finite global clamp. The clamp is a
//!   soundness-preserving *tightening* only, never a widening.
//! - **The domain is per-function**: `log` needs `arg > 0`, `sqrt` needs
//!   `arg ≥ 0`. The abstract-subterm finder uses [`Domain::covers`] as a
//!   decline-if-unprovable guard.
//!
//! ## Trust boundary (unchanged from erf)
//!
//! Soundness rests entirely on each box's certified `eps`. This module does NOT
//! mint envelopes: [`SpecialFnEnvelope::committed`] serves ONLY committed,
//! certified data — `erf` (converted from
//! [`crate::erf_envelope::ErfEnvelope::committed`], Gappa+Arb) plus `exp`/`log`/
//! `sqrt` (Arb mean-value enclosure, cross-checked `<= naive`; see
//! `crates/chelis-prove/data/special_fn_envelopes/README.md`). Any other function
//! returns `None`, and an absent envelope makes the consumer DECLINE, never
//! guess.

use serde::{Deserialize, Serialize};

use crate::erf_envelope::{ErfArm, ErfEnvelope, ProofKind};

/// The mathematical domain a function's argument must lie in for its envelope to
/// apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    /// Defined on all finite reals (`erf`, `exp`).
    AllReals,
    /// Defined only for strictly positive arguments (`log`).
    Positive,
    /// Defined only for non-negative arguments (`sqrt`).
    NonNegative,
}

impl Domain {
    /// Whether the WHOLE argument range `[lo, hi]` is provably inside the domain.
    /// This is the decline-if-unprovable guard: a range that dips to or below the
    /// domain edge is not covered, so the consumer declines rather than emitting
    /// an unsound bound over a point where the function is undefined.
    pub fn covers(self, lo: f64, hi: f64) -> bool {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return false;
        }
        match self {
            Domain::AllReals => true,
            Domain::Positive => lo > 0.0,
            Domain::NonNegative => lo >= 0.0,
        }
    }
}

/// The monotonicity of the special FUNCTION (not its polynomial approximation)
/// over the envelope's covered range. This is what makes endpoint sampling in
/// [`SpecialFnEnvelope::sound_range_bound`] SOUND: a monotone `f` satisfies
/// `f([a,b]) ⊆ [f(a), f(b)]`, so its range is pinned by the endpoints. A
/// `NonMonotonic` function needs the interior-extremum guard instead. Recorded
/// per envelope so the soundness basis is machine-visible; unknown data defaults
/// (via serde) to the conservative `NonMonotonic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Monotonicity {
    /// `f` is (non-strictly) increasing on the covered range (`erf`, `exp`,
    /// `log`, `sqrt`).
    Increasing,
    /// `f` is (non-strictly) decreasing on the covered range.
    Decreasing,
    /// `f` is not monotone (or unknown) — endpoint sampling is unsound, so the
    /// interior-extremum guard is used.
    #[default]
    NonMonotonic,
}

/// One approximation arm. Generalizes [`ErfArm`]: `Saturation` carries any
/// constant (not just `±1`); `Central` is a Horner polynomial (descending
/// degree, `coeffs[0]` highest). Central coeffs serialize as exact C99 hex-float
/// strings (the erf codec), so the committed data recovers the bit-identical f64
/// the certifier stamped an `eps` for — no decimal-parser drift.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvelopeArm {
    /// A saturating constant `value` over the box; `eps` bounds `|value - f(x)|`.
    Saturation { value: f64 },
    /// A polynomial approximation, coefficients highest-degree first, evaluated
    /// by Horner; `eps` bounds `|p(x) - f(x)|`.
    Central {
        #[serde(
            serialize_with = "crate::erf_envelope::hex_f64::serialize_vec",
            deserialize_with = "crate::erf_envelope::hex_f64::deserialize_vec"
        )]
        coeffs: Vec<f64>,
    },
}

/// Sound interval enclosure of a descending-degree polynomial over `[lo, hi]` by
/// interval Horner. Over-approximates the true range (never narrower), so it is a
/// sound bound of the polynomial's value — including any INTERIOR extremum that
/// endpoint sampling would miss.
fn interval_horner(coeffs: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let mut acc = (0.0_f64, 0.0_f64);
    for &c in coeffs {
        // acc = acc * [lo, hi] + c  (interval mul then shift).
        let ps = [acc.0 * lo, acc.0 * hi, acc.1 * lo, acc.1 * hi];
        let mlo = ps.iter().copied().fold(f64::INFINITY, f64::min);
        let mhi = ps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        acc = (mlo + c, mhi + c);
    }
    acc
}

impl EnvelopeArm {
    /// Evaluate the arm's approximation at `x` (the center, before the `±eps`
    /// band). Horner for the polynomial; the constant for saturation.
    pub fn approx(&self, x: f64) -> f64 {
        match self {
            EnvelopeArm::Saturation { value } => *value,
            EnvelopeArm::Central { coeffs } => {
                let mut acc = 0.0_f64;
                for &c in coeffs {
                    acc = acc * x + c;
                }
                acc
            }
        }
    }

    /// A SOUND enclosure `[lo, hi]` of this arm's approximation over the whole
    /// segment `[a, b]` — the interior-extremum guard. Saturation is constant;
    /// the `Central` polynomial uses subdivided interval Horner (hull over the
    /// sub-intervals), which soundly captures an interior extremum a
    /// non-monotonic arm could have. Endpoint sampling alone is UNSOUND for a
    /// non-monotonic polynomial; this makes no monotonicity assumption.
    pub fn sound_approx_range(&self, a: f64, b: f64) -> (f64, f64) {
        match self {
            EnvelopeArm::Saturation { value } => (*value, *value),
            EnvelopeArm::Central { coeffs } => {
                // Fine subdivision keeps the interval-Horner dependency error
                // small so the bound stays tight for the (monotonic) committed
                // arms while remaining sound for any future arm.
                const SUBDIV: usize = 256;
                let width = (b - a) / SUBDIV as f64;
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for i in 0..SUBDIV {
                    let sa = a + width * i as f64;
                    let sb = if i + 1 == SUBDIV {
                        b
                    } else {
                        a + width * (i + 1) as f64
                    };
                    let (elo, ehi) = interval_horner(coeffs, sa, sb);
                    lo = lo.min(elo);
                    hi = hi.max(ehi);
                }
                (lo, hi)
            }
        }
    }
}

/// One box of the piecewise envelope: an interval, an arm, a certified sup-norm
/// error for that arm over that interval, and how the error is certified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecialFnEnvelopeBox {
    /// Inclusive lower edge.
    pub lo: f64,
    /// Inclusive upper edge.
    pub hi: f64,
    /// The approximation used on this box.
    pub arm: EnvelopeArm,
    /// Certified sup-norm error: `sup_{x in [lo,hi]} |arm.approx(x) - f(x)| <= eps`.
    pub eps: f64,
    /// How `eps` is certified.
    pub proof_kind: ProofKind,
}

impl SpecialFnEnvelopeBox {
    /// Whether `x` lies in `[lo, hi]` (inclusive).
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// The sound bound `[approx(x) - eps, approx(x) + eps]` on `f(x)` from this
    /// box.
    pub fn bound(&self, x: f64) -> (f64, f64) {
        let a = self.arm.approx(x);
        (a - self.eps, a + self.eps)
    }
}

/// How a committed envelope was certified, recorded so the `certify_special_fn_
/// envelope validate` cross-check reproduces the exact Arb bound (the naive
/// whole-box eps depends on the subdivision count) and the trust basis is
/// machine-visible. Not a soundness input — soundness is each box's certified
/// `eps`. `#[serde(default)]` so proposer drafts (no provenance) deserialize.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SpecialFnProvenance {
    /// Arb working precision (bits) used to certify each `eps`.
    pub certify_prec: i64,
    /// Number of equal sub-boxes the certifier split each box into.
    pub certify_subdivisions: usize,
    /// A note on the certification method, for the audit trail.
    pub method: String,
}

/// A function-keyed piecewise envelope: ordered, contiguous, non-overlapping
/// boxes over the covered range, plus the function's domain and optional global
/// output clamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecialFnEnvelope {
    /// The special function this envelope bounds (`"erf"`, `"exp"`, ...).
    pub fn_name: String,
    /// The function's argument domain (the decline-if-unprovable guard).
    pub domain: Domain,
    /// Optional global output range, applied as a soundness-preserving TIGHTENING
    /// of the range bound (`erf`: `(-1, 1)`; `sqrt`: `(0, +inf)`; `exp`/`log`:
    /// `None`). Never widens.
    pub output_clamp: Option<(f64, f64)>,
    /// Boxes in ascending `lo` order, contiguous across the covered range.
    pub boxes: Vec<SpecialFnEnvelopeBox>,
    /// Monotonicity of the FUNCTION over the covered range — the soundness basis
    /// for endpoint sampling in [`SpecialFnEnvelope::sound_range_bound`].
    /// `#[serde(default)]` = the conservative `NonMonotonic` (interior-extremum
    /// guard) for any data that omits it. `committed()` stamps the true value.
    #[serde(default)]
    pub monotonicity: Monotonicity,
    /// How the boxes' `eps` were certified (audit trail; enables reproducible
    /// `validate`). Defaults to empty for proposer drafts.
    #[serde(default)]
    pub provenance: SpecialFnProvenance,
}

/// The committed, Arb-certified `exp`/`log`/`sqrt` envelopes (mean-value
/// whole-box enclosure, cross-checked `<= naive`; `proof_kind = arb_enclosure`),
/// embedded at compile time. Generated by `scripts/generate_special_fn_envelope.py
/// <fn>` + `certify_special_fn_envelope stamp` — see
/// `data/special_fn_envelopes/README.md`.
const EXP_ENVELOPE_JSON: &str = include_str!("../data/special_fn_envelopes/exp_envelope.json");
const LOG_ENVELOPE_JSON: &str = include_str!("../data/special_fn_envelopes/log_envelope.json");
const SQRT_ENVELOPE_JSON: &str = include_str!("../data/special_fn_envelopes/sqrt_envelope.json");

impl SpecialFnEnvelope {
    /// Convert the committed, certified `erf` envelope into the generic form.
    /// `erf`: all reals, output clamp `[-1, 1]`. The arms and certified `eps`
    /// carry over verbatim, so the generic path evaluates the SAME certified
    /// bound the erf-specific path does.
    pub fn from_erf(env: ErfEnvelope) -> Self {
        let ErfEnvelope { boxes, provenance } = env;
        let boxes = boxes
            .into_iter()
            .map(|b| SpecialFnEnvelopeBox {
                lo: b.lo,
                hi: b.hi,
                arm: match b.arm {
                    ErfArm::Saturation { value } => EnvelopeArm::Saturation { value },
                    ErfArm::Central { coeffs } => EnvelopeArm::Central { coeffs },
                },
                eps: b.eps,
                proof_kind: b.proof_kind,
            })
            .collect();
        Self {
            fn_name: "erf".to_string(),
            domain: Domain::AllReals,
            output_clamp: Some((-1.0, 1.0)),
            boxes,
            // erf is strictly increasing on the whole real line.
            monotonicity: Monotonicity::Increasing,
            provenance: SpecialFnProvenance {
                certify_prec: provenance.certify_prec,
                certify_subdivisions: provenance.certify_subdivisions,
                method: provenance.method,
            },
        }
    }

    /// The committed, certified envelope for `fn_name`, or `None` if none is
    /// committed. `erf` (Gappa+Arb) plus `exp`/`log`/`sqrt` (Arb mean-value
    /// enclosure) have certified data; any other name returns `None` (the honest
    /// floor — the consumer declines).
    pub fn committed(fn_name: &str) -> Option<Self> {
        let json = match fn_name {
            "erf" => return Some(Self::from_erf(ErfEnvelope::committed())),
            "exp" => EXP_ENVELOPE_JSON,
            "log" => LOG_ENVELOPE_JSON,
            "sqrt" => SQRT_ENVELOPE_JSON,
            _ => return None,
        };
        let mut env: SpecialFnEnvelope = serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("committed {fn_name} envelope must be valid JSON: {e}"));
        // Stamp the FUNCTION's monotonicity (the endpoint-sampling soundness
        // basis) from the registry — exp/log/sqrt are all strictly increasing.
        env.monotonicity = SpecialFnRegistry::monotonicity(fn_name).unwrap_or_default();
        Some(env)
    }

    /// Find the box containing `x`, if any (inclusive edges; a shared boundary
    /// resolves to the lower box).
    pub fn box_for(&self, x: f64) -> Option<&SpecialFnEnvelopeBox> {
        self.boxes.iter().find(|b| b.contains(x))
    }

    /// The sound bound `[lo, hi]` on `f(x)` from the envelope, or `None` if `x`
    /// is outside the covered range.
    pub fn bound(&self, x: f64) -> Option<(f64, f64)> {
        self.box_for(x).map(|b| b.bound(x))
    }

    /// Whether the boxes are well formed: non-empty, each `lo <= hi`, ascending
    /// and contiguous, every `eps >= 0` and finite. This is a STRUCTURAL check,
    /// not a soundness check (soundness is the certified per-box `eps`). Unlike
    /// the erf-specific check it makes NO `±1` saturation assumption.
    pub fn is_well_formed(&self) -> bool {
        if self.boxes.is_empty() {
            return false;
        }
        let mut prev_hi: Option<f64> = None;
        for b in &self.boxes {
            if b.lo > b.hi || !b.eps.is_finite() || b.eps < 0.0 {
                return false;
            }
            if let Some(p) = prev_hi
                && b.lo != p
            {
                return false;
            }
            prev_hi = Some(b.hi);
        }
        true
    }

    /// Compute a sound bound on `f(x)` for ALL `x` in `[arg_lo, arg_hi]`.
    ///
    /// Evaluates the envelope across every box the range touches, taking the hull
    /// (min of all point-lo, max of all point-hi). Sound because every box's
    /// `eps` is a certified sup-norm over that box. Returns `None` if the range
    /// is not fully covered by the envelope. Applies the per-function
    /// `output_clamp` as a soundness-preserving tightening.
    ///
    /// This is the generalization of `erf_envelope`'s `sound_erf_range_bound`
    /// (was inline in the abstract-subterm consumer): the ONLY difference is the
    /// hardcoded `[-1, 1]` clamp is replaced by the per-function `output_clamp`.
    pub fn sound_range_bound(&self, arg_lo: f64, arg_hi: f64) -> Option<(f64, f64)> {
        if arg_lo > arg_hi || !arg_lo.is_finite() || !arg_hi.is_finite() {
            return None;
        }

        let boxes: Vec<&SpecialFnEnvelopeBox> = self
            .boxes
            .iter()
            .filter(|b| b.lo <= arg_hi && b.hi >= arg_lo)
            .collect();
        if boxes.is_empty() {
            return None;
        }

        // The range must be fully covered by the envelope.
        let covered_lo = boxes.first().unwrap().lo;
        let covered_hi = boxes.last().unwrap().hi;
        if arg_lo < covered_lo || arg_hi > covered_hi {
            return None;
        }

        let mut overall_lo = f64::INFINITY;
        let mut overall_hi = f64::NEG_INFINITY;
        for b in &boxes {
            let seg_lo = arg_lo.max(b.lo);
            let seg_hi = arg_hi.min(b.hi);
            // SOUND enclosure of the approximation over the WHOLE segment.
            //
            // For a MONOTONE function the range is pinned by the endpoints
            // (`f([a,b]) ⊆ [f(a), f(b)]`), so `[approx(a), approx(b)]` (min/max)
            // is a tight sound enclosure — this uses the FUNCTION's monotonicity,
            // recorded per envelope, NOT the polynomial's. For a `NonMonotonic`
            // function endpoint sampling is UNSOUND (an interior extremum could be
            // missed), so we fall back to the subdivided interval-Horner
            // interior-extremum guard (sound, looser).
            let (approx_lo, approx_hi) = match self.monotonicity {
                Monotonicity::Increasing | Monotonicity::Decreasing => {
                    let a = b.arm.approx(seg_lo);
                    let c = b.arm.approx(seg_hi);
                    (a.min(c), a.max(c))
                }
                Monotonicity::NonMonotonic => b.arm.sound_approx_range(seg_lo, seg_hi),
            };
            overall_lo = overall_lo.min(approx_lo - b.eps);
            overall_hi = overall_hi.max(approx_hi + b.eps);
        }

        // Per-function output clamp: a soundness-preserving tightening (the true
        // f(x) is known to lie in this global range), never a widening.
        if let Some((clamp_lo, clamp_hi)) = self.output_clamp {
            overall_lo = overall_lo.max(clamp_lo);
            overall_hi = overall_hi.min(clamp_hi);
        }

        // Guard against a non-finite or INVERTED result (e.g. a clamp whose
        // `lo > hi` cuts the band empty). An inverted band would emit
        // `env_lo <= v <= env_hi` with `env_lo > env_hi` — an unsatisfiable
        // precondition that makes the residual VACUOUSLY provable (a false
        // proof). Fail closed instead.
        if !overall_lo.is_finite() || !overall_hi.is_finite() || overall_lo > overall_hi {
            return None;
        }

        Some((overall_lo, overall_hi))
    }
}

/// The registry of special functions the abstract-subterm finder recognizes.
/// Each function has a domain (always known) and, if certified data is
/// committed, an envelope (today `erf` only).
pub struct SpecialFnRegistry;

impl SpecialFnRegistry {
    /// The functions the finder recognizes as abstractable transcendentals.
    pub fn known_functions() -> &'static [&'static str] {
        &["erf", "exp", "log", "sqrt"]
    }

    /// The argument domain for `fn_name`, or `None` if `fn_name` is not a known
    /// special function.
    pub fn domain(fn_name: &str) -> Option<Domain> {
        match fn_name {
            "erf" | "exp" => Some(Domain::AllReals),
            "log" => Some(Domain::Positive),
            "sqrt" => Some(Domain::NonNegative),
            _ => None,
        }
    }

    /// The monotonicity of `fn_name` over its covered range, or `None` if not a
    /// known special function. erf/exp/log/sqrt are all strictly increasing.
    pub fn monotonicity(fn_name: &str) -> Option<Monotonicity> {
        match fn_name {
            "erf" | "exp" | "log" | "sqrt" => Some(Monotonicity::Increasing),
            _ => None,
        }
    }

    /// The committed, certified envelope for `fn_name`, or `None`.
    pub fn committed(fn_name: &str) -> Option<SpecialFnEnvelope> {
        SpecialFnEnvelope::committed(fn_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exp/log/sqrt generation configs, embedded so a test locks them to the
    /// registry. `exp` is CERTIFIED (data committed); `log`/`sqrt` are BLOCKED.
    /// The test keys committed()-presence off each config's `certify_status`.
    const EXP_CONFIG: &str = include_str!("../data/special_fn_envelopes/exp.config.json");
    const LOG_CONFIG: &str = include_str!("../data/special_fn_envelopes/log.config.json");
    const SQRT_CONFIG: &str = include_str!("../data/special_fn_envelopes/sqrt.config.json");

    fn domain_token(d: Domain) -> &'static str {
        match d {
            Domain::AllReals => "all_reals",
            Domain::Positive => "positive",
            Domain::NonNegative => "non_negative",
        }
    }

    #[test]
    fn generation_configs_are_consistent_with_the_registry() {
        for src in [EXP_CONFIG, LOG_CONFIG, SQRT_CONFIG] {
            let cfg: serde_json::Value = serde_json::from_str(src).expect("config parses");
            let f = cfg["function"].as_str().expect("function name");
            // The function is a known registry special function.
            assert!(
                SpecialFnRegistry::known_functions().contains(&f),
                "{f} must be a known special function"
            );
            // The config's declared domain matches the registry's domain guard.
            let reg_domain = SpecialFnRegistry::domain(f).expect("registry domain");
            assert_eq!(
                cfg["domain"].as_str().unwrap(),
                domain_token(reg_domain),
                "{f} config domain must match the registry domain guard"
            );
            // committed()-presence must agree with the config's certify_status:
            // BLOCKED => no committed envelope (finder declines); certified =>
            // committed Some. No config may silently disagree with the data.
            let status = cfg["certify_status"].as_str().unwrap();
            if status.starts_with("BLOCKED") {
                assert!(
                    SpecialFnEnvelope::committed(f).is_none(),
                    "{f} is BLOCKED but has a committed envelope"
                );
            } else {
                assert!(
                    SpecialFnEnvelope::committed(f).is_some(),
                    "{f} is certified ({status}) but has no committed envelope"
                );
            }
            // Boxes are a non-empty, ascending, contiguous decomposition.
            let boxes = cfg["boxes"].as_array().expect("boxes array");
            assert!(!boxes.is_empty(), "{f} config needs at least one box");
            let mut prev_hi: Option<f64> = None;
            for b in boxes {
                let lo = b["lo"].as_f64().unwrap();
                let hi = b["hi"].as_f64().unwrap();
                assert!(lo < hi, "{f} box must have lo<hi");
                if let Some(p) = prev_hi {
                    assert_eq!(lo, p, "{f} boxes must be contiguous");
                }
                prev_hi = Some(hi);
            }
        }
    }

    #[test]
    fn erf_committed_round_trips_through_generic_form() {
        let generic = SpecialFnEnvelope::committed("erf").expect("erf is committed");
        assert_eq!(generic.fn_name, "erf");
        assert_eq!(generic.domain, Domain::AllReals);
        assert_eq!(generic.output_clamp, Some((-1.0, 1.0)));
        assert!(generic.is_well_formed());
        // Same certified bound as the erf-specific path at sample points.
        let erf = ErfEnvelope::committed();
        for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 100.0] {
            assert_eq!(generic.bound(x), erf.bound(x), "bound mismatch at {x}");
        }
    }

    #[test]
    fn unknown_functions_have_no_committed_envelope() {
        // The honest floor for anything outside the registry.
        assert!(SpecialFnEnvelope::committed("tan").is_none());
        assert!(SpecialFnEnvelope::committed("gamma").is_none());
    }

    #[test]
    fn exp_log_sqrt_committed_envelopes_are_sound_and_well_formed() {
        // Each committed transcendental (Arb mean-value certified) parses, is well
        // formed, carries proof_kind arb_enclosure, and its band SOUNDLY contains
        // the true value across its box (a cheap always-on soundness sanity; the
        // rigorous re-check is the arb-lane certify_special_fn_envelope validate).
        for &(name, domain) in &[
            ("exp", Domain::AllReals),
            ("log", Domain::Positive),
            ("sqrt", Domain::NonNegative),
        ] {
            let env =
                SpecialFnEnvelope::committed(name).unwrap_or_else(|| panic!("{name} committed"));
            assert_eq!(env.fn_name, name);
            assert_eq!(env.domain, domain);
            assert!(env.is_well_formed(), "{name} well formed");
            for b in &env.boxes {
                assert_eq!(
                    b.proof_kind,
                    ProofKind::ArbEnclosure,
                    "{name} arb_enclosure"
                );
            }
            let lo = env.boxes.first().unwrap().lo;
            let hi = env.boxes.last().unwrap().hi;
            let mut x = lo;
            while x <= hi {
                let (blo, bhi) = env.bound(x).unwrap_or_else(|| panic!("{name} covers {x}"));
                let t = match name {
                    "exp" => x.exp(),
                    "log" => x.ln(),
                    _ => x.sqrt(),
                };
                assert!(blo <= t && t <= bhi, "{name}({x})={t} not in [{blo},{bhi}]");
                x += (hi - lo) / 50.0;
            }
        }
    }

    #[test]
    fn registry_domains_encode_the_guards() {
        assert_eq!(SpecialFnRegistry::domain("erf"), Some(Domain::AllReals));
        assert_eq!(SpecialFnRegistry::domain("exp"), Some(Domain::AllReals));
        assert_eq!(SpecialFnRegistry::domain("log"), Some(Domain::Positive));
        assert_eq!(SpecialFnRegistry::domain("sqrt"), Some(Domain::NonNegative));
        assert_eq!(SpecialFnRegistry::domain("gamma"), None);
    }

    #[test]
    fn domain_covers_is_a_conservative_guard() {
        // log needs arg > 0: a range touching 0 is not covered.
        assert!(Domain::Positive.covers(0.5, 2.0));
        assert!(!Domain::Positive.covers(0.0, 2.0));
        assert!(!Domain::Positive.covers(-1.0, 2.0));
        // sqrt needs arg >= 0: 0 is allowed, negatives are not.
        assert!(Domain::NonNegative.covers(0.0, 4.0));
        assert!(!Domain::NonNegative.covers(-0.1, 4.0));
        // all reals: any finite range.
        assert!(Domain::AllReals.covers(-1e9, 1e9));
        // malformed ranges never cover.
        assert!(!Domain::AllReals.covers(1.0, 0.0));
        assert!(!Domain::AllReals.covers(f64::NAN, 1.0));
    }

    #[test]
    fn erf_range_bound_matches_the_clamp_behavior() {
        // The generic sound_range_bound with erf's [-1,1] clamp reproduces the
        // erf-specific behavior: deep saturation clamps to <= 1.0.
        let erf = SpecialFnEnvelope::committed("erf").unwrap();
        let (lo, hi) = erf.sound_range_bound(5.0, 10.0).unwrap();
        assert!(lo > 0.99 && hi <= 1.0, "saturation clamp: [{lo}, {hi}]");
        // outside coverage declines
        assert!(erf.sound_range_bound(400.0, 500.0).is_none());
    }

    #[test]
    fn output_clamp_only_tightens_never_widens() {
        // A synthetic envelope whose raw band exceeds a clamp: the clamp pulls it
        // in. Without a clamp the raw band is returned.
        let boxes = vec![SpecialFnEnvelopeBox {
            lo: 0.0,
            hi: 1.0,
            arm: EnvelopeArm::Central {
                coeffs: vec![1.0, 0.0],
            }, // p(x) = x
            eps: 0.5,
            proof_kind: ProofKind::Gappa,
        }];
        let clamped = SpecialFnEnvelope {
            fn_name: "synthetic".into(),
            domain: Domain::NonNegative,
            output_clamp: Some((0.0, 1.0)),
            boxes: boxes.clone(),
            monotonicity: Monotonicity::Increasing,
            provenance: SpecialFnProvenance::default(),
        };
        let unclamped = SpecialFnEnvelope {
            fn_name: "synthetic".into(),
            domain: Domain::NonNegative,
            output_clamp: None,
            boxes,
            monotonicity: Monotonicity::Increasing,
            provenance: SpecialFnProvenance::default(),
        };
        // raw band over [0,1] for p(x)=x, eps=0.5 is [-0.5, 1.5].
        let (ulo, uhi) = unclamped.sound_range_bound(0.0, 1.0).unwrap();
        assert!((ulo - -0.5).abs() < 1e-12 && (uhi - 1.5).abs() < 1e-12);
        let (clo, chi) = clamped.sound_range_bound(0.0, 1.0).unwrap();
        assert!((clo - 0.0).abs() < 1e-12 && (chi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn well_formed_rejects_a_gap_but_makes_no_pm1_assumption() {
        // A saturation value that is NOT ±1 is fine for a general function
        // (unlike the erf-specific check): well-formedness is purely structural.
        let ok = SpecialFnEnvelope {
            fn_name: "synthetic".into(),
            domain: Domain::AllReals,
            output_clamp: None,
            boxes: vec![
                SpecialFnEnvelopeBox {
                    lo: 0.0,
                    hi: 1.0,
                    arm: EnvelopeArm::Saturation { value: 7.389 },
                    eps: 0.1,
                    proof_kind: ProofKind::Gappa,
                },
                SpecialFnEnvelopeBox {
                    lo: 1.0,
                    hi: 2.0,
                    arm: EnvelopeArm::Central { coeffs: vec![0.0] },
                    eps: 0.1,
                    proof_kind: ProofKind::Gappa,
                },
            ],
            monotonicity: Monotonicity::Increasing,
            provenance: SpecialFnProvenance::default(),
        };
        assert!(
            ok.is_well_formed(),
            "non-±1 saturation is structurally fine"
        );

        let gap = SpecialFnEnvelope {
            fn_name: "synthetic".into(),
            domain: Domain::AllReals,
            output_clamp: None,
            boxes: vec![
                SpecialFnEnvelopeBox {
                    lo: 0.0,
                    hi: 1.0,
                    arm: EnvelopeArm::Central { coeffs: vec![0.0] },
                    eps: 0.1,
                    proof_kind: ProofKind::Gappa,
                },
                SpecialFnEnvelopeBox {
                    lo: 2.0, // gap: does not start at 1.0
                    hi: 3.0,
                    arm: EnvelopeArm::Central { coeffs: vec![0.0] },
                    eps: 0.1,
                    proof_kind: ProofKind::Gappa,
                },
            ],
            monotonicity: Monotonicity::Increasing,
            provenance: SpecialFnProvenance::default(),
        };
        assert!(!gap.is_well_formed(), "a coverage gap is not well formed");
    }

    #[test]
    fn nonmonotonic_arm_interior_extremum_is_captured() {
        // The red-team false-proof trap: for a NON-monotonic arm, endpoint
        // sampling MISSES an interior extremum and is unsound. p(x) = x² − x has
        // a minimum of −0.25 at x=0.5, but p(0)=p(1)=0 — so endpoint sampling
        // would return [0,0]. A NonMonotonic envelope must use the interior guard
        // and its band must CONTAIN the interior minimum.
        let env = SpecialFnEnvelope {
            fn_name: "synthetic".into(),
            domain: Domain::AllReals,
            output_clamp: None,
            boxes: vec![SpecialFnEnvelopeBox {
                lo: 0.0,
                hi: 1.0,
                arm: EnvelopeArm::Central {
                    coeffs: vec![1.0, -1.0, 0.0], // x² − x
                },
                eps: 0.0,
                proof_kind: ProofKind::Gappa,
            }],
            monotonicity: Monotonicity::NonMonotonic,
            provenance: SpecialFnProvenance::default(),
        };
        let (lo, hi) = env.sound_range_bound(0.0, 1.0).unwrap();
        assert!(
            lo <= -0.25 && hi >= 0.0,
            "NonMonotonic band must contain the interior min −0.25, got [{lo}, {hi}]"
        );
        // Contrast: the SAME polynomial declared Increasing (a caller error) would
        // endpoint-sample to [0,0] — which is exactly why monotonicity must be a
        // recorded property of the true function, not assumed.
        let wrong = SpecialFnEnvelope {
            monotonicity: Monotonicity::Increasing,
            ..env.clone()
        };
        let (wlo, whi) = wrong.sound_range_bound(0.0, 1.0).unwrap();
        assert!(
            wlo > -0.25,
            "endpoint sampling (Increasing) misses the interior min -- [{wlo},{whi}]"
        );
    }

    #[test]
    fn clamp_inversion_declines_never_vacuous() {
        // The red-team false-proof trap: an INVERTED output clamp would cut the
        // band empty (lo > hi), which as fresh-var preconditions is unsatisfiable
        // → the residual is VACUOUSLY provable (a false proof). sound_range_bound
        // must DECLINE (return None) instead of emitting an inverted band.
        let env = SpecialFnEnvelope {
            fn_name: "synthetic".into(),
            domain: Domain::AllReals,
            output_clamp: Some((1.0, 0.0)), // inverted: lo > hi
            boxes: vec![SpecialFnEnvelopeBox {
                lo: 0.0,
                hi: 1.0,
                arm: EnvelopeArm::Central {
                    coeffs: vec![1.0, 0.0],
                }, // p(x) = x, band [0,1]
                eps: 0.0,
                proof_kind: ProofKind::Gappa,
            }],
            monotonicity: Monotonicity::Increasing,
            provenance: SpecialFnProvenance::default(),
        };
        assert_eq!(
            env.sound_range_bound(0.0, 1.0),
            None,
            "an inverted clamp must DECLINE, never emit a vacuous inverted band"
        );
    }
}
