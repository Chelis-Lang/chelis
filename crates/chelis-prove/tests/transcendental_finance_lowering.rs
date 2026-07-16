//! chelis#434 oracle: transcendental finance discharge — the certified
//! special-function envelope lane, wired (milestone 4).
//!
//! Invariants locked here:
//!   1. A let-bound intermediate passed into a NESTED CALL argument lowers
//!      correctly (the `d1` substitution path) — it proves at SMT when the body
//!      is otherwise cvc5-lowerable. Regression lock: a leak would re-surface
//!      "variable `<name>` has no declared cvc5 term".
//!   2. A genuinely envelope-provable transcendental goal (`erf(x) <= 0.9` over a
//!      bounded box) FLIPS to `proven_modulo_certified_envelope` — the distinct
//!      honest tier, NEVER plain `proven` / `proven_modulo_real_arithmetic`
//!      (`erf` is non-whitelisted, so it reaches the envelope lane; whitelisted
//!      transcendentals `exp`/`log`/`sqrt` go through cvc5's native support).
//!   3. Black-Scholes positivity is NOT envelope-provable — abstracting the
//!      coupled `normal_cdf(d1)`/`normal_cdf(d2)` is falsifiable in the
//!      over-approximation (chelis#637) — so it stays HONESTLY `unsupported`,
//!      never green and never a false disproof, EVEN with the lane fully wired.
//!   4. An envelope-covered FALSE goal HONESTLY declines (unsupported), never
//!      green — the sound-direction forge guard.
//!
//! cvc5-only oracle (mirrors `cross_engine_oracle.rs`'s file-level gate).
#![cfg(feature = "smt")]

use chelis_prove::composition::CompositeVerdict;
use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus, PropertyTier,
    run_surf_source_properties,
};

fn run_one(source: &str, tier: &str, samples: usize) -> PropertyOutcome {
    let opts = PropertyRunOptions {
        tier: tier.to_string(),
        samples,
        ..Default::default()
    };
    let PropertyRunResult::Ran(mut outcomes) =
        run_surf_source_properties(source, &opts).expect("source parses and runs");
    assert_eq!(outcomes.len(), 1, "exactly one property: {outcomes:?}");
    outcomes.pop().unwrap()
}

fn is_proven_badge(v: CompositeVerdict) -> bool {
    matches!(
        v,
        CompositeVerdict::Proven
            | CompositeVerdict::ProvenModuloRealArithmetic
            | CompositeVerdict::ProvenModuloCertifiedEnvelope
            | CompositeVerdict::ProvenModuloFuzzValidatedContract
            | CompositeVerdict::ProvenModuloAssertedAxiom
    )
}

/// The Black-Scholes call-price positivity property from chelis#434.
const BS_SOURCE: &str = r#"module M
def normal_cdf(x: f32) -> f32 = {
  ax = if (x >= 0.0) then x else -x
  t = 1.0 / (1.0 + 0.2316419 * ax)
  d = 0.3989423 * exp(-(x * x) / 2.0)
  p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
  if (x >= 0.0) then 1.0 - p else p
}
def bs_call(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = {
  sqrt_t = sqrt(t)
  sig_sqrt_t = sigma * sqrt_t
  log_sk = log(s / k)
  drift = (r + 0.5 * sigma * sigma) * t
  d1 = (log_sk + drift) / sig_sqrt_t
  d2 = d1 - sig_sqrt_t
  s * normal_cdf(d1) - k * exp(-r * t) * normal_cdf(d2)
}
@property bs_call_positive forall(s: f32, k: f32, r: f32, sigma: f32, t: f32) where (s > 0.0), (k > 0.0), (sigma > 0.0), (t > 0.0), (r >= 0.0):
  (bs_call(s, k, r, sigma, t) > 0.0)
"#;

/// Pure-polynomial analogue isolating sub-problem 1: a let-bound intermediate
/// `d` is passed into the NESTED CALL `g(d)`. `(a + b)^2 + 1 >= 1` is always
/// true, so a faithful substitution proves it at SMT; a leak would re-surface
/// "variable `d` has no declared cvc5 term".
const LET_NEST_SOURCE: &str = r#"module M
def g(x: f32) -> f32 = x + 1.0
def f(a: f32, b: f32) -> f32 = {
  c = a + b
  d = c * c
  g(d)
}
@property letnest_pos forall(a: f32, b: f32):
  (f(a, b) >= 1.0)
"#;

#[test]
fn let_bound_intermediate_in_nested_call_lowers_to_smt() {
    // Sub-problem-1 regression lock: the `d1`-style substitution must thread a
    // let-bound intermediate through a nested call argument without leaking it
    // as a free cvc5 Var.
    let outcome = run_one(LET_NEST_SOURCE, "smt-only", 0);
    assert_eq!(
        outcome.status,
        PropertyStatus::Passed,
        "let-bound intermediate in a nested call proves at SMT: {outcome:?}"
    );
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        !reason.contains("has no declared cvc5 term"),
        "no internal undeclared-var leak: {reason:?}"
    );
}

#[test]
fn bs_call_positive_smt_only_is_honest_unsupported() {
    // chelis#434 + chelis#637: EVEN WITH the certified-envelope lane fully wired,
    // Black-Scholes positivity stays honestly `unsupported`. Two independent
    // reasons keep it there: (1) the oracle's wide guards leave the transcendental
    // arguments (`log(s/k)`, `sqrt(t)`) unbounded, so the envelope abstraction
    // declines; and (2) even with bounded guards the free-variable abstraction of
    // the coupled `normal_cdf(d1)`/`normal_cdf(d2)` is falsifiable in the
    // over-approximation (chelis#637), so a residual disproof is spurious and the
    // lane declines rather than reporting a false disproof. It NAMES the
    // transcendental, never leaks the internal cvc5-var message, and is NEVER
    // green — the canonical false-under-abstraction case (chelis#637).
    let outcome = run_one(BS_SOURCE, "smt-only", 0);
    assert_eq!(
        outcome.status,
        PropertyStatus::Unsupported,
        "transcendental finance is unsupported under smt-only: {outcome:?}"
    );
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("transcendental") && reason.contains("log"),
        "reason names the transcendental capability boundary: {reason:?}"
    );
    assert!(
        !reason.contains("has no declared cvc5 term"),
        "must NOT leak the internal undeclared-var message (chelis#434): {reason:?}"
    );
    assert!(
        !reason.contains("d1"),
        "must NOT leak the let-bound intermediate name: {reason:?}"
    );
    // Strengthened for the wired envelope lane: never any proven badge, and in
    // particular NEVER the certified-envelope tier (BS is not envelope-provable).
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "transcendental goal must never be laundered into a proof: {outcome:?}"
    );
    assert_ne!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloCertifiedEnvelope,
        "BS positivity is NOT envelope-provable (chelis#637) -- must never read the \
         certified-envelope tier even fully wired: {outcome:?}"
    );
}

#[test]
fn bs_call_positive_is_never_falsely_proven_under_auto() {
    // Fail-closed across the tier: under `auto` the transcendental goal falls
    // through to fuzz; a fuzz pass is empirical, NOT a proof. It must never read
    // as a `proven_*` badge and must never surface a `status: error`.
    let outcome = run_one(BS_SOURCE, "auto", 8);
    assert_ne!(
        outcome.status,
        PropertyStatus::Error,
        "auto must not surface status:error for the transcendental goal: {outcome:?}"
    );
    assert_ne!(
        outcome.proof_tier,
        PropertyTier::Smt,
        "the transcendental goal does not discharge at SMT: {outcome:?}"
    );
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "auto fuzz pass must never read as proven: {outcome:?}"
    );
}

// ─── chelis#434 milestone 4: the certified-envelope lane FLIP oracle ──────────
//
// The flip oracle uses `erf` — a NON-whitelisted transcendental (the whitelist
// `[abs,min,max,sqrt,exp,log,sin,cos]` routes those through cvc5's native
// transcendental support, i.e. the base `proven_modulo_real_arithmetic` path).
// `erf` is not whitelisted, so it reaches the certified-envelope lane.

/// A genuinely envelope-provable goal: `erf(x) <= 0.9` for `x in [0,1]`. `erf`
/// over [0,1] is in `[0, 0.843]`, strictly under 0.9, argument bounded — so the
/// certified `erf` envelope discharges the residual.
const ERF_BOUND_SOURCE: &str = r#"module M
@property erf_bounded forall(x: f32) where (x >= 0.0), (x <= 1.0):
  (erf(x) <= 0.9)
"#;

/// FALSE-under-abstraction forge case: `erf(x) <= 0.5` for `x in [0,1]`. The real
/// goal is FALSE (`erf(0.6) ≈ 0.604 > 0.5`); the certified envelope covers the
/// argument but the residual is falsifiable, so the lane must HONESTLY DECLINE
/// (unsupported under smt-only), NEVER green and NEVER a false disproof.
const ERF_FALSE_SOURCE: &str = r#"module M
@property erf_false forall(x: f32) where (x >= 0.0), (x <= 1.0):
  (erf(x) <= 0.5)
"#;

#[test]
fn envelope_provable_goal_flips_to_proven_modulo_certified_envelope() {
    let outcome = run_one(ERF_BOUND_SOURCE, "smt-only", 0);
    assert_eq!(
        outcome.status,
        PropertyStatus::Passed,
        "envelope-provable erf goal must pass under smt-only: {outcome:?}"
    );
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloCertifiedEnvelope,
        "must project to the distinct certified-envelope tier: {outcome:?}"
    );
    // NEVER plain proven / proven_modulo_real_arithmetic — the envelope
    // dependency is disclosed, not laundered away.
    assert_ne!(
        outcome.composite_verdict,
        CompositeVerdict::Proven,
        "{outcome:?}"
    );
    assert_ne!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
}

#[test]
fn envelope_covered_false_goal_declines_never_green() {
    // smt-only: the residual is falsifiable in the over-approximation, so the
    // lane declines to an HONEST `unsupported` — never green, never a false proof.
    let outcome = run_one(ERF_FALSE_SOURCE, "smt-only", 0);
    assert_ne!(
        outcome.status,
        PropertyStatus::Passed,
        "an envelope-covered FALSE goal must never be green: {outcome:?}"
    );
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "FALSE goal must never read as any proven badge: {outcome:?}"
    );
    assert_ne!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloCertifiedEnvelope,
        "FALSE goal must never read the certified-envelope tier: {outcome:?}"
    );
}
