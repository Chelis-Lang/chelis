//! Shared property-runner tests (U4 review-3 unification).

use super::*;

fn run_surf(source: &str, tier: &str) -> Vec<PropertyOutcome> {
    let opts = PropertyRunOptions {
        tier: tier.to_string(),
        samples: 32,
        ..Default::default()
    };
    let PropertyRunResult::Ran(o) = run_surf_source_properties(source, &opts).expect("run");
    o
}

const NAMED_PROPERTY: &str = "module M
def double(x: f32) -> f32 = x + x
@property double_is_even forall(x: f32):
  (double(x) == x + x)
";

#[test]
fn u4_discovers_differently_named_property_no_hardcoded_name() {
    // The discovery must find a property whose name is NOT `property`.
    let outcomes = run_surf(NAMED_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property discovered: {outcomes:?}");
    assert_eq!(outcomes[0].name, "double_is_even");
    assert!(
        outcomes[0].is_pass(),
        "a true property passes: {:?}",
        outcomes[0]
    );
}

const MULTIPLE_PROPERTIES: &str = "module M
@property always_true forall(x: f32):
  (x == x)
@property also_true forall(y: f32):
  (y + 0.0 == y)
";

#[test]
fn u4_discovers_multiple_properties() {
    let outcomes = run_surf(MULTIPLE_PROPERTIES, "auto");
    assert_eq!(outcomes.len(), 2, "two properties: {outcomes:?}");
    assert!(outcomes.iter().all(|o| o.is_pass()));
}

const NO_PROPERTY: &str = "module M
def f(x: f32) -> f32 = x + 1.0
";

#[test]
fn u4_no_property_is_empty() {
    let outcomes = run_surf(NO_PROPERTY, "auto");
    assert!(outcomes.is_empty(), "no @property => no outcomes");
}

const FAILING_PROPERTY: &str = "module M
@property false_claim forall(x: f32):
  (x > x)
";

#[test]
fn u4_failing_property_is_not_pass() {
    let outcomes = run_surf(FAILING_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Failed);
    assert!(!outcomes[0].is_pass());
}

const INJECTION_PROPERTY: &str = "module M.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
def prob_value(p: Probability) -> f32 = p.value
@property bounded forall(p: Probability):
  (prob_value(p) <= 1.0)
";

#[test]
fn u4_property_with_opaque_invariant_binder_uses_injection() {
    // A property whose binder is an invariant-carrying opaque type is
    // verified ONLY over invariant-satisfying values (assumption injection);
    // `prob_value(p) <= 1.0` holds for every in-domain Probability, so it
    // passes, and the outcome is marked injected.
    let outcomes = run_surf(INJECTION_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert_eq!(outcomes[0].name, "bounded");
    assert!(
        outcomes[0].is_pass(),
        "injected property passes: {:?}",
        outcomes[0]
    );
    assert!(outcomes[0].injected, "verified through the injection path");
}

const INJECTION_FALSE_PROPERTY: &str = "module M.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
def prob_value(p: Probability) -> f32 = p.value
@property too_strong forall(p: Probability):
  (prob_value(p) <= 0.5)
";

#[test]
fn u4_injection_does_not_hide_a_false_property() {
    // Injection restricts to invariant-valid binders, but a property false
    // ON those binders still fails (a Probability with value 0.9 is valid yet
    // violates `<= 0.5`).
    let outcomes = run_surf(INJECTION_FALSE_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Failed);
    assert!(!outcomes[0].is_pass());
}

#[test]
fn u4_statistically_validated_zero_samples_is_not_pass() {
    // A Passed outcome with zero fuzz samples (the vacuous/timeout sentinel)
    // is NOT a genuine pass.
    let zero = PropertyOutcome {
        name: "p".to_string(),
        status: PropertyStatus::Passed,
        proof_tier: PropertyTier::Fuzz,
        samples: 0,
        seed: 0,
        counterexample: None,
        reason: None,
        injected: false,
    };
    assert!(!zero.is_pass(), "Passed with 0 fuzz samples is not a pass");
    // A Passed SMT proof carries 0 samples but IS a pass.
    let smt = PropertyOutcome {
        proof_tier: PropertyTier::Smt,
        ..zero.clone()
    };
    assert!(
        smt.is_pass(),
        "an SMT-proved property is a pass at 0 samples"
    );
}
