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

#[test]
fn fuzz_counterexample_records_accepted_shrink_steps() {
    let outcomes = run_surf(
        "module M
@property too_strong forall(x: f32):
  (x > 5.0)
",
        "fuzz-only",
    );
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert!(
        outcome.shrink_steps > 0,
        "fuzz failure should record accepted shrink steps: {outcome:?}"
    );
    assert_eq!(
        outcome
            .counterexample
            .as_ref()
            .and_then(|cx| cx["x"].as_f64()),
        Some(0.0),
        "zero is still a failing counterexample for x > 5.0: {outcome:?}"
    );
}

const UNKNOWN_CONTRACT_PROPERTY: &str = r#"module M
@property unknown_contract forall(x: f32):
  x == x
  with contract = "std.normal_cdf.not_a_contract"
"#;

#[test]
fn contract_unknown_id_is_unsupported() {
    let outcomes = run_surf(UNKNOWN_CONTRACT_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("unknown contract"),
        "reason should name unknown contract: {:?}",
        outcomes[0]
    );
}

const NO_CALL_CONTRACT_PROPERTY: &str = r#"module M
@property no_contract_call forall(x: f32):
  x == x
  with contract = "std.normal_cdf.reflection"
"#;

#[test]
fn contract_without_bound_call_is_unsupported() {
    let outcomes = run_surf(NO_CALL_CONTRACT_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("did not bind"),
        "reason should report no binding: {:?}",
        outcomes[0]
    );
}

const MISMATCHED_CONTRACT_PROPERTY: &str = r#"module M
def other_cdf(x: f32) -> f32 = x
@property wrong_binding forall(x: f32):
  other_cdf(0.0 - x) == 1.0 - other_cdf(x)
  with contract = "std.normal_cdf.reflection"
"#;

#[test]
fn contract_bound_to_wrong_call_is_unsupported() {
    let outcomes = run_surf(MISMATCHED_CONTRACT_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("did not bind"),
        "reason should report binding mismatch: {:?}",
        outcomes[0]
    );
}

#[cfg(feature = "smt")]
const REFLECTION_CONTRACT_PROPERTY: &str = r#"module M
def pkg__chelis__std__Std__Contracts__normal_cdf(x: f32) -> f32 = x
@property reflected forall(x: f32):
  pkg__chelis__std__Std__Contracts__normal_cdf(-x) == 1.0 - pkg__chelis__std__Std__Contracts__normal_cdf(x)
  with contract = "std.normal_cdf.reflection"
"#;

#[cfg(feature = "smt")]
#[test]
fn local_linker_shaped_normal_cdf_does_not_receive_std_contract() {
    let outcomes = run_surf(REFLECTION_CONTRACT_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(
        outcome.status,
        PropertyStatus::Unsupported,
        "a local spoofed linker-shaped normal_cdf must not bind std contract assumptions: {outcome:?}"
    );
    assert!(
        outcome
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("did not bind"),
        "reason should report missing trusted std binding: {outcome:?}"
    );
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

#[test]
fn wi8_injected_binder_assumption_carries_prover_stamped_discharge_tier() {
    // WI-8: the binder-matched (injected) assumption carries a prover-stamped
    // discharge tier recording which engine discharged it and with what
    // guarantee, keyed to the binder's source identity. c-earchin emits only
    // source identity; the tier is the prover's stamp.
    let outcomes = run_surf(INJECTION_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert!(outcomes[0].injected);

    let binder_assumption = outcomes[0]
        .assumptions
        .iter()
        .find(|a| a.name == "invariant:Probability:binder:p")
        .unwrap_or_else(|| panic!("injected binder assumption present: {:?}", outcomes[0]));

    let tier = binder_assumption
        .discharge_tier
        .as_ref()
        .expect("injected assumption carries a prover-stamped discharge tier");
    assert_eq!(
        tier.engine, "fuzz-sampler",
        "discharged by the fuzz sampler"
    );
    assert_eq!(tier.guarantee, "fuzz", "with the fuzz guarantee kind");
    // The tier is keyed to the binder's source identity (the same id the
    // assumption carries), so the artifact can join tier -> source.
    assert_eq!(
        tier.source.as_deref(),
        Some("invariant:Probability:binder:p"),
        "the tier is keyed to the binder source identity"
    );
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
    let zero = PropertyOutcome::new(
        "p",
        PropertyStatus::Passed,
        PropertyTier::Fuzz,
        0,
        0,
        None,
        None,
        false,
        Vec::new(),
    );
    assert!(!zero.is_pass(), "Passed with 0 fuzz samples is not a pass");
    // A Passed SMT proof carries 0 samples but IS a pass.
    let smt = PropertyOutcome::new(
        "p",
        PropertyStatus::Passed,
        PropertyTier::Smt,
        0,
        0,
        None,
        None,
        false,
        Vec::new(),
    );
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
    assert!(
        result.is_err(),
        "an invalid property_source_kind must error: {result:?}"
    );
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

// --- WI-8 tier-coverage invariant (WS-5 Part B): every AssumptionRecord
// produced on a green discharge path carries a prover-stamped discharge_tier ---

/// Assert the WI-8 coverage invariant on one outcome: every assumption it
/// carries has a prover-stamped discharge tier. Returns the number of
/// assumptions checked so the caller can assert the path actually produced
/// some (a vacuous zero-assumption pass must not silently satisfy the
/// invariant).
fn assert_assumptions_are_tiered(outcome: &PropertyOutcome) -> usize {
    for assumption in &outcome.assumptions {
        assert!(
            assumption.discharge_tier.is_some(),
            "green-path assumption `{}` on property `{}` is missing a prover-stamped \
             discharge_tier (WI-8): {assumption:?}",
            assumption.name,
            outcome.name,
        );
    }
    outcome.assumptions.len()
}

/// A green fuzz property whose `where` precondition drives the
/// `fuzz_precondition_assumptions` discharge site: the pass carries one
/// `preconditions:*` assumption that must be tiered.
const GREEN_FUZZ_PRECONDITION_PROPERTY: &str = "module M
@property guarded forall(x: f32) where (x > 0.0):
  (x + 1.0 > x)
";

#[test]
fn wi8_green_fuzz_precondition_assumptions_carry_discharge_tier() {
    let outcomes = run_surf(GREEN_FUZZ_PRECONDITION_PROPERTY, "fuzz-only");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert!(
        outcomes[0].is_pass(),
        "the guarded property passes via fuzz: {:?}",
        outcomes[0]
    );
    let checked = assert_assumptions_are_tiered(&outcomes[0]);
    assert_eq!(
        checked, 1,
        "the green fuzz precondition path emits exactly one tiered assumption: {:?}",
        outcomes[0]
    );
}

#[test]
fn wi8_green_injection_binder_assumptions_carry_discharge_tier() {
    // The injection green path (a binder-matched invariant) emits a binder
    // assumption that must also be tiered. INJECTION_PROPERTY is defined above.
    let outcomes = run_surf(INJECTION_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert!(outcomes[0].is_pass(), "injected property passes");
    assert!(outcomes[0].injected, "verified through the injection path");
    let checked = assert_assumptions_are_tiered(&outcomes[0]);
    assert!(
        checked >= 1,
        "the injection green path emits at least one tiered assumption: {:?}",
        outcomes[0]
    );
}

#[test]
fn wi8_smt_precondition_discharge_site_stamps_discharge_tier() {
    // The SMT green discharge site (`property_assumption_records`) is only
    // reached at runtime when Tier B returns Proved, which the default (non-smt)
    // build cannot do (solve_property is a stub). Lock the site directly with a
    // unit call: a discharge with a non-empty precondition set must yield an
    // AssumptionRecord carrying a prover-stamped discharge_tier keyed to the
    // precondition source identity. This is race-free (no process-global env
    // override) and does not require the cvc5 manual gate.
    let smt_prop = crate::tier_b::SmtProperty {
        variables: vec![("x".to_string(), crate::solver::SmtSort::Real)],
        preconditions: vec![crate::solver::SmtExpr::BoolLit(true)],
        postcondition: crate::solver::SmtExpr::BoolLit(true),
    };
    let discharge = AssumptionDischarge::new(
        DischargeMethod::Smt,
        serde_json::json!({"solver": "cvc5", "result": "proved"}),
    );
    let non_vacuity = NonVacuityRecord::established(serde_json::json!({"result": "sat"}));
    let records = property_assumption_records("guarded", &smt_prop, discharge, non_vacuity);
    assert_eq!(
        records.len(),
        1,
        "a non-empty precondition set produces one assumption record"
    );
    let tier = records[0]
        .discharge_tier
        .as_ref()
        .expect("the SMT discharge site stamps a prover-side discharge tier (WI-8)");
    assert_eq!(tier.engine, "cvc5", "discharged by the SMT engine");
    assert_eq!(tier.guarantee, "smt", "with the smt guarantee kind");
    assert_eq!(
        tier.source.as_deref(),
        Some("preconditions:guarded"),
        "keyed to the precondition source identity"
    );

    // Negative companion: an empty precondition set produces no assumption,
    // so there is nothing to tier (the site is vacuous, not silently untiered).
    let empty = crate::tier_b::SmtProperty {
        variables: Vec::new(),
        preconditions: Vec::new(),
        postcondition: crate::solver::SmtExpr::BoolLit(true),
    };
    let none = property_assumption_records(
        "p",
        &empty,
        AssumptionDischarge::new(DischargeMethod::Smt, serde_json::json!({})),
        NonVacuityRecord::established(serde_json::json!({})),
    );
    assert!(
        none.is_empty(),
        "no preconditions => no assumption record to tier"
    );
}
