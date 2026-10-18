//! chelis#3697: the multi-binder diagonal limitation, and the two
//! measurements that constrain any fix for it.
//!
//! Tier C's edge schedule (chelis#3636) indexes every binder by the same
//! attempt ordinal, so binders with equal ranges take the same candidate at
//! every scheduled attempt and a multi-binder property is evaluated only on
//! the **diagonal** of the edge grid. chelis#3697 owns that, and **no
//! off-diagonal policy is implemented**.
//!
//! These tests therefore pin *current behaviour*, not a feature. An attempt
//! at a policy broke four shapes at the default budget, each measured, and
//! the two that no existing test could see are pinned here so the next
//! attempt has to confront them rather than rediscover them.
//! `spec/design/chelis_property_spec.md` § "Why the multi-binder diagonal is
//! still open" carries the arithmetic and the full tables.
//!
//! A policy that changes any of these is expected to change these tests
//! deliberately, with the measurement that justifies it.
use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// Run `prove --tier fuzz-only` over `source` and return the exit code with
/// the single property record.
fn prove(source: &str, samples: &str, seed: &str) -> (i32, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hazard.ch");
    std::fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            path.to_str().expect("utf-8 path"),
            "--json",
            "--tier",
            "fuzz-only",
            "--samples",
            samples,
            "--seed",
            seed,
        ])
        .output()
        .expect("run prove");
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| value["kind"] == "property")
        .unwrap_or_else(|| {
            panic!(
                "property record missing: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
    (output.status.code().unwrap_or(-1), record)
}

/// The limitation itself, pinned so it is not rediscovered. `x / y` overflows
/// when a large `x` meets a tiny `y`, and both components satisfy
/// `x > 0, y > 0`, so the pair is a true counterexample the diagonal never
/// proposes.
///
/// This asserts the property is **not** refuted. That is deliberate: the
/// assertion is the open defect, and whoever closes chelis#3697 should have
/// to delete it on purpose after reading the design doc's measurements.
#[test]
fn issue_3697_the_multi_binder_diagonal_is_still_open() {
    let source = "module Hazard.Diagonal
def subject(x: f64, y: f64) -> f64 = (x / y)
@property div_finite forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((subject(x, y) <= 1.7976931348623157e308f64) && (subject(x, y) >= -1.7976931348623157e308f64))
";
    for seed in ["0", "1", "7", "99"] {
        let (code, record) = prove(source, "2000", seed);
        assert_eq!(
            code, 0,
            "seed {seed}: the diagonal limitation is expected to leave this \
             unrefuted; if this now fails, chelis#3697 is closed and this test \
             should be deleted on purpose: {record:#}"
        );
        assert_eq!(record["status"], "passed", "seed {seed}: {record:#}");
    }
}

/// The narrowed witness, which proves the limitation is about *pairing* and
/// not about reach. With the guards moved so the diagonal lands on the
/// refuting pair, the same body refutes immediately -- so each binder does
/// reach the extremes on its own.
#[test]
fn issue_3697_the_diagonal_witness_is_real_when_the_guards_align_it() {
    let source = "module Hazard.Aligned
def subject(x: f64, y: f64) -> f64 = (x / y)
@property div_finite forall(x: f64, y: f64)
where x > 1.0e300f64, y > 0.0f64, y < 1.0e-300f64:
  ((subject(x, y) <= 1.7976931348623157e308f64) && (subject(x, y) >= -1.7976931348623157e308f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let x = record["counterexample"]["x"].as_f64().expect("x");
    let y = record["counterexample"]["y"].as_f64().expect("y");
    assert!(x > 1.0e300 && y > 0.0 && y < 1.0e-300, "{record:#}");
}

/// **The post-diagonal uniform draws are not spare budget.** They are the
/// only thing that reaches a counterexample in the *interior* of the proposal
/// window, where no edge candidate and no magnitude-class representative
/// lands. A policy that funds off-diagonal pairing from these attempts costs
/// exactly this, and at three or more binders a six-class odometer consumes
/// every one of them at the default budget.
///
/// The band `(-7.5, -7.0)` is 2.5% of the `[-10, 10]` window and contains no
/// candidate of any schedule. The refutation is therefore seed-dependent --
/// measured over seeds 0 to 29 at the default 100 samples, 20/30 at one
/// binder, 24/30 at two, 26/30 at three, 23/30 at four -- so this fixes one
/// seed that refutes at every arity rather than asserting a rate. The design
/// doc carries the rates and what a policy that spends these attempts owes.
///
/// Each refutation arrives **after attempt 44**, where the unguarded `f64`
/// schedule is spent. That is the assertion that makes this a measurement of
/// the post-diagonal draws specifically, and not of sampling in general.
#[test]
fn issue_3697_post_diagonal_uniform_draws_carry_refutations() {
    for arity in 1..=4 {
        let binders = (0..arity)
            .map(|index| format!("v{index}: f64"))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "module Hazard.Band
@property outside_band forall({binders}):
  ((v0 <= -7.5f64) || (v0 >= -7.0f64))
"
        );
        let (code, record) = prove(&source, "100", "1");
        assert_eq!(
            code, 1,
            "arity {arity}: only a uniform draw over the window reaches this \
             band, so a policy that spends those attempts loses it: {record:#}"
        );
        let witness = record["counterexample"]["v0"]
            .as_f64()
            .unwrap_or_else(|| panic!("arity {arity}: f64 witness: {record:#}"));
        assert!(
            witness > -7.5 && witness < -7.0,
            "arity {arity}: the witness must come from the band, got {witness}: {record:#}"
        );
        let attempts = record["attempted_samples"]
            .as_u64()
            .unwrap_or_else(|| panic!("arity {arity}: attempts: {record:#}"));
        assert!(
            attempts > 44,
            "arity {arity}: the refutation must arrive after the 44-candidate \
             schedule is spent, or this measures something else; got \
             {attempts}: {record:#}"
        );
    }
}

/// **A successor's candidate list is rebuilt from its predecessor inside the
/// attempt**, so this lane has no stable per-binder sequence: an index into
/// that list names entry `i` of a list whose contents depend on what the
/// predecessor got. A policy proposing new values here re-indexes every
/// successor.
///
/// Pinned by the one consequence a property can observe: the refutation is
/// seed-dependent, because it needs the predecessor's uniform draw to be
/// negative so that `y > x` admits a negative `y` at all. Seeds 1 and 3
/// refute and seeds 0 and 4 do not; a policy that changes the draw sequence
/// at all will move this, which is the point.
#[test]
fn issue_3697_an_order_successors_candidates_depend_on_its_predecessor() {
    let exclusions = (20..40)
        .map(|value| format!("y != {value}.0f64"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "module Hazard.Chain
@property not_bottom forall(x: f64, y: f64)
where y > x, {exclusions}:
  (y != -1.0e-323f64)
"
    );
    for (seed, attempts) in [("1", 77), ("3", 77)] {
        let (code, record) = prove(&source, "100", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        assert_eq!(
            record["attempted_samples"], attempts,
            "seed {seed}: {record:#}"
        );
        let witness = record["counterexample"]["x"]
            .as_f64()
            .unwrap_or_else(|| panic!("seed {seed}: x: {record:#}"));
        assert!(
            witness < 0.0,
            "seed {seed}: the refutation needs a negative predecessor, got \
             {witness}: {record:#}"
        );
    }
    for seed in ["0", "4"] {
        let (code, record) = prove(&source, "100", seed);
        assert_eq!(
            code, 0,
            "seed {seed}: these seeds draw a non-negative predecessor, so \
             `y > x` admits no negative `y`: {record:#}"
        );
    }
}
