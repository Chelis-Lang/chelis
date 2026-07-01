//! chelis#434 oracle: transcendental finance properties (Black-Scholes
//! positivity) must NEVER surface the internal "variable `d1` has no declared
//! cvc5 term" leak and must NEVER be falsely `proven`. The minimum acceptable
//! outcome is a clean, honest user-facing `unsupported` that names the
//! transcendental capability boundary; the real discharge of a transcendental
//! finance goal is the WS-7 / Beacon envelope seam (cited in the reason text),
//! NOT in-tree cvc5 NRA.
//!
//! Two invariants are locked here:
//!   1. A let-bound intermediate passed into a NESTED CALL argument lowers
//!      correctly (the `d1` substitution path) -- it proves at SMT when the
//!      body is otherwise cvc5-lowerable. This is the sub-problem-1 regression
//!      lock: a leak would re-surface "variable `<name>` has no declared cvc5
//!      term".
//!   2. The Black-Scholes positivity property (transcendental: `log`, `exp`,
//!      `sqrt`) is HONESTLY `unsupported` under smt-only (naming the
//!      transcendental, never the internal leak) and is NEVER falsely proven
//!      under any tier (fail-closed).
//!
//! cvc5-only oracle (mirrors cross_engine_oracle.rs's file-level gate).
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
    // Sub-problem-2 fail-closed: smt-only is terminal, so the transcendental
    // Black-Scholes goal is an HONEST capability-boundary `unsupported` that
    // NAMES the transcendental, never the internal "no declared cvc5 term"
    // leak, and never a false proof.
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
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "transcendental goal must never be laundered into a proof: {outcome:?}"
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
