//! chelis#463 oracle: boolean connectives (`&&` / `||`) at a property goal
//! site lower to the SMT tier and produce SOUND verdicts, and a conjunction
//! that genuinely cannot lower falls through to Tier C fuzz cleanly under
//! `--tier auto` (never a `status: error`, never a false `proven`).
//!
//! Root cause of the original report: the canonical Surf boolean operators are
//! `&&` / `||` (spec `02-surf-syntax.md` §2; the formatter renders
//! `BinOp::And`/`BinOp::Or` as `&&`/`||`). The `and`/`or` KEYWORD spelling used
//! in the issue repro is not a Surf operator (§1 lists 26 keywords; `and`/`or`
//! are not among them), so `(A) and (B)` parses as a malformed function
//! application and fails type-checking. That is a chelis-surf parser concern,
//! out of scope here; crucially it fails SAFE (unsupported/error, never a false
//! `proven`). These oracles lock the chelis-prove invariants for the canonical
//! `&&`/`||` form and the auto-fallthrough.

use chelis_prove::composition::CompositeVerdict;
use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus, PropertyTier,
    run_surf_source_properties,
};

fn run(source: &str, tier: &str, samples: usize) -> Vec<PropertyOutcome> {
    let opts = PropertyRunOptions {
        tier: tier.to_string(),
        samples,
        ..Default::default()
    };
    let PropertyRunResult::Ran(outcomes) =
        run_surf_source_properties(source, &opts).expect("source parses and runs");
    outcomes
}

fn one(source: &str, tier: &str, samples: usize) -> PropertyOutcome {
    let mut outcomes = run(source, tier, samples);
    assert_eq!(outcomes.len(), 1, "exactly one property: {outcomes:?}");
    outcomes.pop().unwrap()
}

/// The soundness floor: a verdict must never read as any `proven_*` badge
/// unless a deductive engine actually proved it. Used to assert that a FALSE or
/// merely fuzz-validated goal is never laundered into a proof.
fn is_proven_badge(v: CompositeVerdict) -> bool {
    matches!(
        v,
        CompositeVerdict::Proven
            | CompositeVerdict::ProvenModuloRealArithmetic
            | CompositeVerdict::ProvenModuloFuzzValidatedContract
            | CompositeVerdict::ProvenModuloAssertedAxiom
    )
}

#[cfg(feature = "smt")]
const CONJ_TRUE: &str = "module M
@property conj forall(x: f32):
  ((x * x >= 0.0) && (x * x >= 0.0))
";

#[cfg(feature = "smt")]
const DISJ_TRUE: &str = "module M
@property disj forall(x: f32):
  ((x * x >= 0.0) || (x * x >= 1.0))
";

#[cfg(feature = "smt")]
const NEG_TRUE: &str = "module M
@property negg forall(x: f32):
  (not (x * x < 0.0))
";

// A FALSE conjunction goal: `forall x . x*x >= 0 && x >= 0` fails for x < 0.
const CONJ_FALSE: &str = "module M
@property conj_false forall(x: f32):
  ((x * x >= 0.0) && (x >= 0.0))
";

// A FALSE disjunction goal: `forall x . x > 1 || x > 2` fails near 0.
const DISJ_FALSE: &str = "module M
@property disj_false forall(x: f32):
  ((x > 1.0) || (x > 2.0))
";

// A conjunction that type-checks but cannot lower to SMT (the `log`
// transcendental has no cvc5 kind), TRUE on its precondition domain.
const CONJ_NONLOWERABLE_TRUE: &str = "module M
def helper(x: f32) -> f32 = log(x)
@property conj_nl_true forall(x: f32) where (x > 0.0):
  ((helper(x) <= x) && (x > 0.0))
";

// Same non-lowerable shape but FALSE: `log(x) >= 0` fails for x in (0, 1).
const CONJ_NONLOWERABLE_FALSE: &str = "module M
def helper(x: f32) -> f32 = log(x)
@property conj_nl_false forall(x: f32) where (x > 0.0):
  ((helper(x) >= 0.0) && (x > 0.0))
";

// ---------------------------------------------------------------------------
// Soundness invariants (hold in BOTH the default and `--features smt` builds).
// ---------------------------------------------------------------------------

#[test]
fn false_conjunction_goal_is_never_falsely_proven() {
    // smt-only false-conjunction is the laundering-risk path; auto/default may
    // fuzz it. In every build the verdict must NOT be a proof and must not be a
    // clean pass, and must never surface a `status: error`.
    let outcome = one(CONJ_FALSE, "auto", 256);
    assert_ne!(
        outcome.status,
        PropertyStatus::Error,
        "false conjunction must not surface status:error: {outcome:?}"
    );
    assert_ne!(
        outcome.status,
        PropertyStatus::Passed,
        "false conjunction must not pass: {outcome:?}"
    );
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "false conjunction must never be laundered into a proof: {outcome:?}"
    );
}

#[test]
fn false_disjunction_goal_is_never_falsely_proven() {
    let outcome = one(DISJ_FALSE, "auto", 256);
    assert_ne!(outcome.status, PropertyStatus::Error, "{outcome:?}");
    assert_ne!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "false disjunction must never be laundered into a proof: {outcome:?}"
    );
}

#[test]
fn nonlowerable_conjunction_under_auto_falls_to_fuzz_not_error() {
    // The second wart in chelis#463: a conjunction that cannot lower to SMT
    // must, under `--tier auto`, fall through to Tier C fuzz CLEANLY -- never a
    // `status: error`. A true non-lowerable conjunction fuzz-passes, badged as
    // empirical (NOT a proof).
    let outcome = one(CONJ_NONLOWERABLE_TRUE, "auto", 64);
    assert_ne!(
        outcome.status,
        PropertyStatus::Error,
        "non-lowerable conjunction must not error under auto: {outcome:?}"
    );
    assert_eq!(
        outcome.proof_tier,
        PropertyTier::Fuzz,
        "non-lowerable conjunction falls to fuzz under auto: {outcome:?}"
    );
    assert_eq!(
        outcome.status,
        PropertyStatus::Passed,
        "a true non-lowerable conjunction fuzz-passes: {outcome:?}"
    );
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::FuzzValidatedEmpirical,
        "a fuzz pass is empirical, NOT proven: {outcome:?}"
    );
}

#[test]
fn nonlowerable_false_conjunction_under_auto_no_error_no_false_proven() {
    let outcome = one(CONJ_NONLOWERABLE_FALSE, "auto", 256);
    assert_ne!(
        outcome.status,
        PropertyStatus::Error,
        "non-lowerable false conjunction must not error under auto: {outcome:?}"
    );
    assert_eq!(
        outcome.proof_tier,
        PropertyTier::Fuzz,
        "falls to fuzz under auto: {outcome:?}"
    );
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "must never be laundered into a proof: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// SMT-tier capability (only meaningful when the cvc5 engine is linked).
// ---------------------------------------------------------------------------

#[cfg(feature = "smt")]
#[test]
fn conjunction_goal_lowers_to_smt_and_proves() {
    let outcome = one(CONJ_TRUE, "smt-only", 0);
    assert_eq!(
        outcome.status,
        PropertyStatus::Passed,
        "a true conjunction proves at SMT: {outcome:?}"
    );
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "SMT conjunction proof is over the reals: {outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn disjunction_goal_lowers_to_smt_and_proves() {
    let outcome = one(DISJ_TRUE, "smt-only", 0);
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn negation_control_goal_lowers_to_smt_and_proves() {
    let outcome = one(NEG_TRUE, "smt-only", 0);
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn false_conjunction_is_disproved_at_smt_with_counterexample() {
    let outcome = one(CONJ_FALSE, "smt-only", 0);
    assert_eq!(
        outcome.status,
        PropertyStatus::Failed,
        "a false conjunction is disproved at SMT: {outcome:?}"
    );
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::DisprovedModuloRealArithmetic,
        "{outcome:?}"
    );
    assert!(
        outcome.counterexample.is_some(),
        "disproof carries a counterexample: {outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn false_disjunction_is_disproved_at_smt() {
    let outcome = one(DISJ_FALSE, "smt-only", 0);
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::DisprovedModuloRealArithmetic,
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn nonlowerable_conjunction_smt_only_is_honest_unsupported() {
    // smt-only is terminal: a non-lowerable conjunction must be an HONEST
    // capability-boundary `unsupported`, naming the transcendental, NOT an
    // internal lowering-error leak (chelis#434's failure mode).
    let outcome = one(CONJ_NONLOWERABLE_FALSE, "smt-only", 0);
    assert_eq!(
        outcome.status,
        PropertyStatus::Unsupported,
        "non-lowerable conjunction is unsupported under smt-only: {outcome:?}"
    );
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("does not lower to the SMT tier") || reason.contains("transcendental"),
        "reason frames a capability boundary: {reason:?}"
    );
    assert!(
        !reason.contains("no declared cvc5 term"),
        "must not leak the internal 'no declared cvc5 term' message: {reason:?}"
    );
    assert!(
        !is_proven_badge(outcome.composite_verdict),
        "must never be laundered into a proof: {outcome:?}"
    );
}
