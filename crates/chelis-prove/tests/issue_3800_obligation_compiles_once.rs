//! chelis#3800: an invariant obligation whose producer takes scalars and
//! literal-shape tensors compiles its producer once, not once per sample and
//! shrink trial.
//!
//! The oracle is the compiler's own count of programs it checks for
//! evaluation. A per-sample probe makes that count grow with the sample count
//! and every shrink trial. One call program per obligation keeps it constant:
//! the same for one sample as for a hundred, refutations and their shrinks
//! included: one program per obligation over a per-run constant. The module
//! mixes passing, refuted (with shrink trials that trap) and trapping
//! producers over scalar, boolean, Option and tensor signatures, plus a
//! nullary producer. A rank-0 tensor input, which the per-sample probe
//! misspelled as rank 1, now checks.
//!
//! A producer whose input is itself an opaque record is evaluated per sample,
//! decided from its signature before anything runs (its argument is a record,
//! chelis#3804); a later step shows that route still scales with samples. The
//! last step runs the opaque-invariant corpus and examples and checks that the
//! call program compiles for every producer they declare.
//!
//! The steps share one test function on purpose: the count is process-wide,
//! and `cargo test` runs the tests of one binary on parallel threads.

mod support;

use std::path::Path;

use chelis_compiler_api::compiler::eval_compilations;
use chelis_prove::obligation_engine::{
    ObligationOutcome, ObligationRunOptions, ObligationRunResult, ObligationStatus,
    run_surf_source_obligations,
};

/// Producers whose inputs are scalars and literal tensors.
const CALLED: &str = r#"module Stats.Prob
export (clamp_prob, raw_prob, from_count, tricky, trapper, from_flag, maybe_prob, from_tensor, half)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }

def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }

def raw_prob(x: f32) -> Probability = Probability { value: x }

def from_count(n: i32) -> Probability = Probability { value: if (n > 0i32) then 1.0 else 0.0 }

def tricky(n: i32) -> Probability = Probability { value: cast(trunc_div(100i32, n), f32) }

def trapper(n: i8) -> Probability = Probability { value: if ((n + 100i8) > 0i8) then 0.5 else 0.25 }

def from_flag(b: bool, x: f64) -> Probability = Probability { value: if b then 1.0 else 0.0 }

def maybe_prob(x: f32) -> Option[Probability] = if (x >= 0.0) then Some(Probability { value: 0.5 }) else None

def from_tensor(t: tensor[2, f32]) -> Probability = Probability { value: tensor_to_scalar(sum(t, 0i32)) }

def half() -> Probability = Probability { value: 0.5 }
"#;

const PRODUCERS: usize = 9;

/// A producer of a rank-0 tensor. The per-sample probe spelled its sample as
/// a rank-1 literal, so every sample errored; the call binds it at shape `[]`
/// and the obligation checks.
const RANK0: &str = r#"module Stats.Rank0
export (rank0)
@opaque
@invariant(p) (p.value >= 0.0)
type Pos =
  | Pos { value: f32 }

def rank0(t: tensor[f32]) -> Pos = Pos { value: abs(tensor_to_scalar(t)) }
"#;

/// A producer whose input is an opaque record.
const PER_SAMPLE: &str = r#"module Stats.Scale
export (clamp_prob, scale)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }

def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }

def scale(p: Probability, k: f32) -> Probability = Probability { value: if (k > 0.0) then p.value else 0.0 }
"#;

fn run(source: &str, samples: usize, only: Option<&str>) -> (Vec<ObligationOutcome>, u64) {
    let options = ObligationRunOptions {
        tier: "fuzz-only".to_string(),
        samples,
        only: only.map(str::to_string),
        ..ObligationRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let before = eval_compilations();
    let result = run_surf_source_obligations(source, &options).expect("source parses");
    let ObligationRunResult::Ran(outcomes) = result else {
        panic!("the module type-checks: {result:?}");
    };
    (outcomes, eval_compilations() - before)
}

fn outcome<'a>(outcomes: &'a [ObligationOutcome], producer: &str) -> &'a ObligationOutcome {
    outcomes
        .iter()
        .find(|outcome| outcome.name == format!("invariant:Probability:{producer}"))
        .unwrap_or_else(|| panic!("no obligation for `{producer}`: {outcomes:#?}"))
}

fn assert_no_internal_error(file: &str, outcomes: &[ObligationOutcome]) {
    for outcome in outcomes {
        let reason = outcome.reason.as_deref().unwrap_or_default();
        assert!(
            !reason.contains("internal error"),
            "{file}: the call program of `{}` does not compile: {reason}",
            outcome.name
        );
    }
}

#[test]
fn scalar_and_tensor_producers_compile_once_per_obligation() {
    crate::support::isolate();

    // The run resolves the module's zero-argument defs (`half`) by evaluation
    // once, whatever the sample count; every obligation adds its one call
    // program. One obligation's run is that constant plus one.
    let (_, single) = run(CALLED, 64, Some("invariant:Probability:clamp_prob"));
    let mut counts = Vec::new();
    for samples in [1, 4, 16, 64] {
        let (outcomes, compiles) = run(CALLED, samples, None);
        assert_eq!(outcomes.len(), PRODUCERS, "{samples}: {outcomes:#?}");
        assert_no_internal_error("CALLED", &outcomes);
        counts.push((samples, compiles));
        if samples < 64 {
            continue;
        }
        for (producer, status) in [
            ("clamp_prob", ObligationStatus::Passed),
            ("half", ObligationStatus::Passed),
            ("maybe_prob", ObligationStatus::Passed),
            ("raw_prob", ObligationStatus::Failed),
            ("from_tensor", ObligationStatus::Failed),
            ("tricky", ObligationStatus::Failed),
            ("trapper", ObligationStatus::Error),
        ] {
            assert_eq!(
                outcome(&outcomes, producer).status,
                status,
                "{producer}: {outcomes:#?}"
            );
        }
        // The trap's verdict is the per-sample probe's, word for word.
        assert_eq!(
            outcome(&outcomes, "trapper").reason.as_deref(),
            Some("numeric trap: overflow in add at i8")
        );
        // `tricky`'s refutation shrinks toward zero, whose trial traps.
        assert!(
            outcome(&outcomes, "tricky").shrink_steps > 0,
            "{outcomes:#?}"
        );
    }
    assert!(
        counts
            .iter()
            .all(|(_, compiles)| *compiles == single + PRODUCERS as u64 - 1),
        "each scalar/tensor producer's obligation compiles once whatever the sample \
         count, shrinks included: one obligation compiles {single}, all {PRODUCERS} \
         compile by sample count {counts:?} (chelis#3800)"
    );

    // A rank-0 tensor input is bound at its own rank, and the obligation
    // reaches its verdict.
    let (outcomes, _) = run(RANK0, 25, None);
    assert_eq!(
        outcomes
            .iter()
            .map(|outcome| (outcome.status, outcome.reason.as_deref()))
            .collect::<Vec<_>>(),
        [(ObligationStatus::Passed, None)],
        "{outcomes:#?}"
    );

    // Opaque-input producers stay on the per-sample route.
    let (_, few) = run(PER_SAMPLE, 2, Some("invariant:Probability:scale"));
    let (_, many) = run(PER_SAMPLE, 16, Some("invariant:Probability:scale"));
    assert!(
        many > few,
        "the opaque-input producer evaluates per sample: {few} compiles at 2 samples, \
         {many} at 16"
    );

    // The call program compiles for every producer of the opaque-invariant
    // corpus and examples.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files: Vec<_> = ["tests/corpus/opaque_invariants/programs", "examples"]
        .iter()
        .flat_map(|dir| std::fs::read_dir(root.join(dir)).expect("corpus directory"))
        .map(|entry| entry.expect("corpus entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "ch"))
        .filter(|path| {
            std::fs::read_to_string(path).is_ok_and(|source| source.contains("@invariant"))
        })
        .collect();
    files.sort();
    let mut ran = 0;
    for path in &files {
        let source = std::fs::read_to_string(path).expect("corpus file");
        let options = ObligationRunOptions {
            tier: "fuzz-only".to_string(),
            samples: 4,
            ..ObligationRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        if let Ok(ObligationRunResult::Ran(outcomes)) =
            run_surf_source_obligations(&source, &options)
        {
            assert_no_internal_error(&path.display().to_string(), &outcomes);
            ran += outcomes.len();
        }
    }
    assert!(ran > 20, "the corpus ran {ran} obligations over {files:?}");
}
