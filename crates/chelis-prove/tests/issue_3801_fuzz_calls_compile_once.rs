//! chelis#3801: a property whose binders are scalars, booleans or strings is
//! compiled once, as functions of its binders, and every sample, guard and
//! shrink trial is one call of them.
//!
//! The oracle is the compiler's own count of programs it checks for
//! evaluation. A fuzz run of such a property compiles exactly one program per
//! property, whatever its sample count and however many shrink steps it takes;
//! an evaluation that errors adds the one program that words its message. The
//! sweep covers the Surf lane (against the file's prepared declarations) and
//! the Deep lane (the whole program), unguarded, guarded, refuted, exhausted
//! and trapping properties, and every scalar binder kind.
//!
//! The steps share one test function on purpose: the count is process-wide,
//! and `cargo test` runs the tests of one binary on parallel threads.

mod support;

use chelis_compiler_api::compiler::eval_compilations;
use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus,
    run_deep_source_properties, run_surf_source_properties,
};

/// Properties whose evaluations never error, with the status each reaches
/// from 16 samples on.
const SOURCE: &str = r#"module M
def helper(x: i64) -> i64 = add(x, 1i64)

@property unguarded_passes forall(x: i64, s: string, b: bool):
  gt(helper(0i64), 0i64)

@property guarded_passes forall(x: i64) where gt(x, 0i64):
  gt(helper(x), x)

@property guarded_refuted forall(a: i64, b: i64) where gt(a, 10i64):
  lt(add(a, b), 100i64)

@property float_refuted forall(x: f64, y: f32) where x > 0.0f64:
  (x * x < 4.0f64)

@property exhausted forall(x: i64) where gt(x, 0i64), lt(x, 0i64):
  true
"#;

const STATUSES: [PropertyStatus; 5] = [
    PropertyStatus::Passed,
    PropertyStatus::Passed,
    PropertyStatus::Failed,
    PropertyStatus::Failed,
    PropertyStatus::Error,
];

/// A property whose evaluation traps on the edge sample `127i8`.
const TRAPPING: &str = r#"module T
@property traps forall(x: i8):
  x + 1i8 > x
"#;

#[derive(Clone, Copy, Debug)]
enum Lane {
    Surf,
    Deep,
}

fn options(samples: usize) -> PropertyRunOptions {
    PropertyRunOptions {
        tier: "fuzz-only".to_string(),
        samples,
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    }
}

fn deep_source(surf: &str) -> String {
    let decls = chelis_surf::parser::parse_str(surf).expect("source parses");
    let program = chelis_surf::desugar::desugar_program(&decls).expect("source desugars");
    chelis_deep::printer::print_canonical(&program)
}

/// Every property's outcome, and the programs the run compiled for
/// evaluation.
fn run(lane: Lane, source: &str, samples: usize) -> (Vec<PropertyOutcome>, u64) {
    let options = options(samples);
    let before = eval_compilations();
    let result = match lane {
        Lane::Surf => run_surf_source_properties(source, &options),
        Lane::Deep => run_deep_source_properties(&deep_source(source), &options),
    };
    let PropertyRunResult::Ran(outcomes) = result.expect("source parses and runs");
    (outcomes, eval_compilations() - before)
}

#[test]
fn fuzz_properties_over_scalar_binders_compile_once() {
    crate::support::isolate();

    for lane in [Lane::Surf, Lane::Deep] {
        for samples in [1, 3, 16, 100, 300] {
            let (outcomes, compiles) = run(lane, SOURCE, samples);
            let statuses = outcomes
                .iter()
                .map(|outcome| outcome.status.clone())
                .collect::<Vec<_>>();
            // Below 16 samples the refutations may not yet be reached.
            if samples >= 16 {
                assert_eq!(statuses, STATUSES, "{lane:?} {samples}: {outcomes:#?}");
            }
            let evaluations = outcomes
                .iter()
                .map(|outcome| outcome.attempted_samples + outcome.accepted_samples)
                .sum::<usize>();
            let shrink_steps = outcomes
                .iter()
                .map(|outcome| outcome.shrink_steps)
                .sum::<usize>();
            assert_eq!(
                compiles,
                outcomes.len() as u64,
                "{lane:?} with {samples} samples compiled {compiles} programs for {} \
                 properties, {evaluations} sample evaluations and {shrink_steps} shrink \
                 steps; each property compiles once (chelis#3801)",
                outcomes.len()
            );
        }

        // The trap is the run's last evaluation: the call program plus the
        // sample's own program, which words the message.
        let (outcomes, compiles) = run(lane, TRAPPING, 100);
        assert_eq!(outcomes[0].status, PropertyStatus::Error, "{outcomes:#?}");
        let reason = outcomes[0].reason.as_deref().unwrap_or_default();
        assert!(reason.contains("overflow"), "{lane:?}: {reason}");
        assert_eq!(
            compiles, 2,
            "{lane:?}: a trapping run compiled {compiles} programs"
        );

        // The same inputs give the same records, seeds and shrink steps.
        assert_eq!(
            run(lane, SOURCE, 40).0,
            run(lane, SOURCE, 40).0,
            "{lane:?} is deterministic"
        );
    }
}
