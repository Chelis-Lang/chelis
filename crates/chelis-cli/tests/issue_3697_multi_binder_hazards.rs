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
//! **The diagonal limitation itself is not pinned here.**
//! `issue_3636_edge_sampling.rs` already owns it, in
//! `issue_3636_multi_binder_schedule_visits_only_the_diagonal` and
//! `issue_3636_the_diagonal_witness_is_real_when_the_guards_align_it` — that
//! file asserts the `x / y` property is *not* refuted and that its witness is
//! real once the guards align it. Duplicating them here would give whoever
//! closes chelis#3697 two deletion sites instead of one, so this file covers
//! only what they do not.
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
    // 77 is where this shape's schedule reaches `-1e-323`: the 44 base
    // candidates plus the exclusions' surviving neighbours. A different count
    // means the draw sequence moved, which is what this test is for -- but it
    // does not say whether the move is benign, so read the design doc's
    // order-chain figures before adjusting it.
    for (seed, attempts) in [("1", 77), ("3", 77)] {
        let (code, record) = prove(&source, "100", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        assert_eq!(
            record["attempted_samples"], attempts,
            "seed {seed}: expected the refutation at attempt {attempts}; a \
             different count means the draw sequence moved: {record:#}"
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
