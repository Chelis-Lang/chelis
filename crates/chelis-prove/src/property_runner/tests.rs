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

// --- F7: the deep property path honors the --tier contract ---

fn run_deep(source: &str, tier: &str) -> Vec<PropertyOutcome> {
    let opts = PropertyRunOptions {
        tier: tier.to_string(),
        samples: 16,
        ..Default::default()
    };
    let PropertyRunResult::Ran(o) = run_deep_source_properties(source, &opts).expect("run");
    o
}

/// A canonical Deep user `@property` (the form `chelis deep` emits): the
/// always-true `x >= x`.
const DEEP_TRUE_PROPERTY: &str = r#"(module {}
  m
  (defsig {} always_true (t-fn {} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}),
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_source_kind: "user"
       }
    always_true
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {}
        (var {} gte)
        (var {} x)
        (var {} x)))))
"#;

#[test]
fn f7_deep_fuzz_only_runs_the_fuzz_loop() {
    // `--tier fuzz-only` on a deep property runs the fuzz loop and passes a
    // true property with samples > 0.
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "fuzz-only");
    assert_eq!(outcomes.len(), 1, "one deep property: {outcomes:?}");
    assert_eq!(outcomes[0].name, "always_true");
    assert!(
        outcomes[0].is_pass(),
        "deep fuzz-only passes: {:?}",
        outcomes[0]
    );
    assert!(outcomes[0].samples > 0, "fuzz-only collected samples");
    assert_eq!(outcomes[0].proof_tier, PropertyTier::Fuzz);
}

#[test]
fn f7_deep_smt_only_is_unsupported_not_silently_fuzzed() {
    // `--tier smt-only` on a deep property must NOT silently run the fuzz
    // loop (the tier was previously ignored on the deep path). A deep body
    // has no Surf->SMT lowering path, so smt-only is Unsupported.
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(
        outcomes[0].status,
        PropertyStatus::Unsupported,
        "deep smt-only is unsupported, not a silent fuzz pass: {:?}",
        outcomes[0]
    );
    assert_eq!(outcomes[0].samples, 0, "smt-only ran no fuzz samples");
}

#[test]
fn f7_deep_auto_runs_the_fuzz_loop() {
    // `--tier auto` falls through to fuzz for a deep property (no SMT path).
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert!(
        outcomes[0].is_pass(),
        "deep auto passes via fuzz: {:?}",
        outcomes[0]
    );
    assert!(outcomes[0].samples > 0);
}

// --- F6: the deep discoverer classifies source kind like the CLI ---

/// A `chelis_role: "property"` def with NO `property_source_kind` is a
/// malformed property: the shared discoverer must ERROR (matching the CLI
/// discoverer), not silently skip it (which previously made tide report
/// total:0 / ok:true while the CLI errored).
#[test]
fn f6_deep_chelis_role_property_without_source_kind_is_error() {
    let source = r#"(module {}
  m
  (def {chelis_role: "property",
         property_quantifiers: (params {} (x {type: (t-prim {} f32)}))}
    nameless_kind
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let opts = PropertyRunOptions::default();
    let result = run_deep_source_properties(source, &opts);
    assert!(
        result.is_err(),
        "a chelis_role property with no property_source_kind must error, not be skipped: {result:?}"
    );
    assert!(
        result.unwrap_err().contains("property_source_kind"),
        "the error names the missing metadata"
    );
}

/// A `chelis_role: "property"` def with an INVALID `property_source_kind`
/// is likewise an error (matching the CLI), not silently skipped.
#[test]
fn f6_deep_invalid_source_kind_is_error() {
    let source = r#"(module {}
  m
  (def {chelis_role: "property",
         property_source_kind: "bogus",
         property_quantifiers: (params {} (x {type: (t-prim {} f32)}))}
    bad_kind
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let opts = PropertyRunOptions::default();
    let result = run_deep_source_properties(source, &opts);
    assert!(result.is_err(), "an invalid property_source_kind must error: {result:?}");
}

/// A `bridge:c-earchin` property is SKIPPED by the shared runner (the CLI
/// bridge path owns it) -- not an error, not run. The shared runner returns
/// an empty user-property set for a bridge-only module.
#[test]
fn f6_deep_bridge_source_kind_is_skipped_not_error() {
    let source = r#"(module {}
  m
  (def {chelis_role: "property",
         property_source_kind: "bridge:c-earchin",
         property_quantifiers: (params {} (x {type: (t-prim {} f32)}))}
    bridged
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let opts = PropertyRunOptions::default();
    let PropertyRunResult::Ran(outcomes) =
        run_deep_source_properties(source, &opts).expect("bridge module is not an error");
    assert!(
        outcomes.is_empty(),
        "the shared runner skips bridge:c-earchin properties: {outcomes:?}"
    );
}
