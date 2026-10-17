//! chelis#3117: the fuzz tier compiles once per chunk of samples or shrink
//! trials, not once per guard, predicate and shrink trial.
//!
//! The oracle is the compiler's own count of programs it checks for
//! evaluation. A per-evaluation compile makes that count grow with every
//! sample and every shrink trial. Chunks of 1, 2, 4, ... up to 64 samples (or
//! trials) per compile keep it logarithmic in the samples a run reaches until
//! chunks reach 64, and one per 64 after that, whatever the machine. The
//! sweep covers the three fuzz lanes (Surf against its prepared declarations,
//! Deep as a whole program, and assumption injection for opaque binders with
//! invariants), unguarded and guarded properties, passing and refuted ones,
//! and sample counts below and above the largest chunk. Each Surf and Deep
//! property here has a tensor binder, which no call argument carries, so it
//! takes the chunked path rather than one call program per property
//! (chelis#3801, `issue_3801_fuzz_calls_compile_once`).
//!
//! The steps share one test function on purpose: the count is process-wide,
//! and `cargo test` runs the tests of one binary on parallel threads.

mod support;

use chelis_compiler_api::compiler::eval_compilations;
use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus,
    run_deep_source_properties, run_surf_source_properties,
};

/// The most samples, or shrink trials, whose probes share one compile.
const MAX_SAMPLES_PER_COMPILE: usize = 64;

/// The most shrink trials of one step: zero, half and the sign for each
/// integer binder, and zeros and halves for a tensor binder.
const fn shrink_trials(int_binders: usize, tensor_binders: usize) -> usize {
    3 * int_binders + 2 * tensor_binders
}

const SOURCE: &str = r#"module M
def helper(x: i64) -> i64 = add(x, 1i64)

@property unguarded_passes forall(t: tensor[2, f64], s: string):
  gt(helper(0i64), 0i64)

@property guarded_passes forall(t: tensor[2, f64], x: i64) where gt(x, 0i64):
  gt(helper(x), x)

@property guarded_refuted forall(t: tensor[2, f64], a: i64, b: i64) where gt(a, 10i64):
  lt(add(a, b), 100i64)
"#;

/// Opaque binders whose invariant the generator establishes, so the
/// assumption-injection lane runs them.
const INJECTED: &str = r#"module Stats.Prob
export (clamp_prob)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }
def prob_value(p: Probability) -> f32 = p.value

@property bounded forall(p: Probability, q: Probability):
  (prob_value(p) <= 1.0)

@property refuted forall(p: Probability):
  (prob_value(p) < 0.0)
"#;

#[derive(Clone, Copy, Debug)]
enum Lane {
    Surf,
    Deep,
    Injection,
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

/// Every selected property's outcome, and the programs the run compiled for
/// evaluation.
fn run(lane: Lane, options: &PropertyRunOptions) -> (Vec<PropertyOutcome>, u64) {
    let before = eval_compilations();
    let result = match lane {
        Lane::Surf => run_surf_source_properties(SOURCE, options),
        Lane::Deep => run_deep_source_properties(&deep_source(SOURCE), options),
        Lane::Injection => run_surf_source_properties(INJECTED, options),
    };
    let PropertyRunResult::Ran(outcomes) = result.expect("source parses and runs");
    (outcomes, eval_compilations() - before)
}

/// How many chunks `items` samples or trials take.
fn chunks(items: usize) -> u64 {
    let (mut next, mut covered, mut count) = (1usize, 0usize, 0u64);
    while covered < items {
        covered += next;
        next = (next * 2).min(MAX_SAMPLES_PER_COMPILE);
        count += 1;
    }
    count
}

/// The compiles a Surf or Deep run of `outcome` may take: one per sampling
/// chunk it drew from, and for a refuted property the chunks of every shrink
/// step's trials, the final step that finds no smaller counterexample
/// included.
fn compile_bound(outcome: &PropertyOutcome, trials_per_step: usize) -> u64 {
    let sampling = chunks(outcome.attempted_samples);
    let shrinking = match outcome.status {
        PropertyStatus::Failed => (outcome.shrink_steps as u64 + 1) * chunks(trials_per_step),
        _ => 0,
    };
    sampling + shrinking
}

#[test]
fn fuzz_evaluations_share_a_compile() {
    crate::support::isolate();

    for lane in [Lane::Surf, Lane::Deep] {
        for samples in [1, 3, 16, 100, 300] {
            let (outcomes, compiles) = run(lane, &options(samples));
            let statuses = outcomes
                .iter()
                .map(|outcome| outcome.status.clone())
                .collect::<Vec<_>>();
            assert_eq!(
                statuses,
                [
                    PropertyStatus::Passed,
                    PropertyStatus::Passed,
                    PropertyStatus::Failed
                ],
                "{lane:?} {samples}: {outcomes:#?}"
            );
            let evaluations = outcomes
                .iter()
                .map(|outcome| outcome.attempted_samples + outcome.accepted_samples)
                .sum::<usize>();
            let bound = outcomes
                .iter()
                .zip([
                    shrink_trials(0, 1),
                    shrink_trials(1, 1),
                    shrink_trials(2, 1),
                ])
                .map(|(outcome, trials)| compile_bound(outcome, trials))
                .sum::<u64>();
            assert!(
                compiles <= bound,
                "{lane:?} with {samples} samples compiled {compiles} programs for \
                 {evaluations} sample evaluations and {} shrink steps; chunked \
                 compiles allow {bound} (chelis#3117)",
                outcomes[2].shrink_steps
            );

            // Every attempt of an unguarded passing property is a sample, so
            // it compiles exactly once per chunk.
            let only = PropertyRunOptions {
                only: Some("unguarded_passes".to_string()),
                ..options(samples)
            };
            let (outcomes, unguarded) = run(lane, &only);
            assert!(outcomes[0].is_pass(), "{outcomes:#?}");
            assert_eq!(
                unguarded,
                chunks(samples),
                "{lane:?}: {samples} unguarded samples compiled {unguarded} programs"
            );
        }
    }

    // Assumption injection has no shrinking, and both of its properties are
    // unguarded, so every attempt is a sample. The refuted one fails at its
    // first sample.
    for samples in [1, 3, 16, 100] {
        let (outcomes, compiles) = run(Lane::Injection, &options(samples));
        assert!(
            outcomes.iter().all(|outcome| outcome.injected),
            "{outcomes:#?}"
        );
        assert_eq!(outcomes[0].status, PropertyStatus::Passed, "{outcomes:#?}");
        assert_eq!(outcomes[1].status, PropertyStatus::Failed, "{outcomes:#?}");
        assert_eq!(
            compiles,
            chunks(samples) + 1,
            "injection with {samples} samples compiled {compiles} programs (chelis#3117)"
        );
    }

    // The same inputs give the same records, seeds and shrink steps included.
    for lane in [Lane::Surf, Lane::Deep, Lane::Injection] {
        assert_eq!(
            run(lane, &options(40)).0,
            run(lane, &options(40)).0,
            "{lane:?} is deterministic"
        );
    }
}
