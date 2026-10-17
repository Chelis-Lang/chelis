//! chelis#3745: a Surf file's declarations are checked and lowered once per
//! file, not once per property or per domain witness.
//!
//! Every property in a file evaluates its samples against the same
//! declarations, so the declaration context they need is the same value. The
//! oracle is the compiler's own count of declaration-context builds, which a
//! per-property build multiplies by the property count whatever the machine.
//!
//! The steps share one test function on purpose: the count is process-wide,
//! and `cargo test` runs the tests of one binary on parallel threads.

mod support;

use chelis_compiler_api::declaration_context_builds;
use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus,
    run_surf_source_properties,
};

fn run(source: &str, tier: &str, samples: usize) -> (Vec<PropertyOutcome>, u64) {
    let options = PropertyRunOptions {
        tier: tier.to_string(),
        samples,
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let before = declaration_context_builds();
    let PropertyRunResult::Ran(outcomes) =
        run_surf_source_properties(source, &options).expect("source parses and runs");
    (outcomes, declaration_context_builds() - before)
}

/// Four fuzzed properties over one helper.
const FOUR_FUZZED: &str = r#"module M
def helper(x: i64) -> i64 = add(x, 1i64)

@property increases_a forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)

@property increases_b forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)

@property increases_c forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)

@property increases_d forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)
"#;

/// Two properties over declarations that desugar but do not check.
const TWO_OVER_A_REJECTED_LIBRARY: &str = r#"module M
def helper(x: i64) -> i64 = add(x, 1i64)
def broken(x: i64) -> f32 = x

@property increases_a forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)

@property increases_b forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)
"#;

#[test]
fn one_declaration_context_per_file() {
    crate::support::isolate();

    // Positive: every property passes, and all four shared one build.
    let (outcomes, builds) = run(FOUR_FUZZED, "fuzz-only", 4);
    assert_eq!(outcomes.len(), 4, "{outcomes:?}");
    for outcome in &outcomes {
        assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    }
    assert_eq!(
        builds, 1,
        "four properties over one file built the declaration context {builds} times; \
         it must be built once per file (chelis#3745)"
    );

    // Negative parity: a library that does not check is attempted once, and
    // every property still reports the rejection rather than only the first.
    let (outcomes, builds) = run(TWO_OVER_A_REJECTED_LIBRARY, "fuzz-only", 4);
    assert_eq!(outcomes.len(), 2, "{outcomes:?}");
    let reasons: Vec<&str> = outcomes
        .iter()
        .map(|outcome| {
            assert_eq!(outcome.status, PropertyStatus::Error, "{outcome:?}");
            outcome
                .reason
                .as_deref()
                .expect("an errored property says why")
        })
        .collect();
    assert!(
        reasons[0].contains("broken") || reasons[0].contains("f32"),
        "the error names the rejected declaration: {reasons:?}"
    );
    assert_eq!(
        reasons[0], reasons[1],
        "both properties report the one rejection: {reasons:?}"
    );
    assert_eq!(
        builds, 1,
        "a rejected library was built {builds} times for two properties"
    );

    #[cfg(feature = "smt")]
    domain_witnesses_share_the_build();
}

/// Two properties whose `sqrt` domain obligations the solver decides false.
/// Each witness is confirmed by concrete evaluation, which needs the same
/// declaration context a fuzz sample does (chelis#3635).
#[cfg(feature = "smt")]
fn domain_witnesses_share_the_build() {
    const TWO_WITNESSES: &str = r#"module M
def k(x: f64) -> f64 = sqrt(x)

@property k_nonneg forall(x: f64) where x > -1.0f64, x < 2.0f64:
  (k(x) >= 0.0f64)

@property k_nonneg_again forall(x: f64) where x > -1.0f64, x < 2.0f64:
  (k(x) >= 0.0f64)
"#;
    let (outcomes, builds) = run(TWO_WITNESSES, "smt-only", 0);
    assert_eq!(outcomes.len(), 2, "{outcomes:?}");
    for outcome in &outcomes {
        assert_eq!(
            outcome.status,
            PropertyStatus::Failed,
            "each witness is confirmed: {outcome:?}"
        );
    }
    assert_eq!(
        builds, 1,
        "two domain witness confirmations built the declaration context {builds} times"
    );
}
