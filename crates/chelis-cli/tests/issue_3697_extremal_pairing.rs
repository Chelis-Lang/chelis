//! chelis#3697: Tier C pairs each scheduled binder's two extremal candidates
//! off the diagonal, so a multi-binder counterexample at a *corner* of the
//! edge grid is reached instead of reported `fuzz_validated`.
//!
//! chelis#3636 gave every binder a deterministic 44-candidate edge schedule
//! indexed by the shared attempt ordinal, which evaluated a multi-binder
//! property only on the **diagonal** of that grid. The diagonal still runs
//! first, unchanged and at its original one-for-one rate; the attempts that
//! follow it carry a mixed-radix odometer over two classes per binder -- its
//! largest and smallest admissible candidate -- and then resume the uniform
//! draw.
//!
//! **Three things are load bearing, and each is pinned below by the
//! measurement that chose it: two classes, the placement after the diagonal,
//! and the cap on how much of the post-diagonal budget the cells may take.**
//! The cells and the uniform draw compete for the same attempts, and
//! `2^n` outgrows any fixed allowance — so the allowance is a share of the
//! budget, which bounds the competition at every arity. A six-class set took
//! every post-diagonal attempt at three binders; a fixed cap of 256 took
//! every one at six. Both were total starvation of the generator, and the
//! draws they displaced were carrying most of the refutations.
//! `spec/design/chelis_property_spec.md` § "Pairing the extremal classes"
//! carries the arithmetic and the tables.
//!
//! The deliberate costs are pinned too, as tests that assert a property is
//! *not* refuted: a pairing of an extreme with a moderate value, and any
//! pairing at all for an order-guarded property.

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

/// The pairing itself, on a property false at exactly one corner of the grid
/// and nowhere else, so no shrinking can stand between the draw and the
/// assertion.
///
/// At two binders the odometer has `2^2 = 4` cells, which take attempts 44 to
/// 47. Cell 2 reads digit 0 for the first binder -- its largest candidate --
/// and digit 1 for the second -- its smallest. Under `> 0.0` those are
/// `f64::MAX` and the smallest positive subnormal, so the corner is proposed
/// at the forty-seventh attempt whatever the seed. Before chelis#3697 the two
/// binders took the same index at every attempt and this corner was proposed
/// at none of them.
#[test]
fn issue_3697_the_off_diagonal_corner_is_reached() {
    let source = "module Pairing.Corner
@property not_the_corner forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((x != 1.7976931348623157e308f64) || (y != 5.0e-324f64))
";
    for seed in ["0", "1", "7", "99"] {
        let (code, record) = prove(source, "200", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        assert_eq!(
            record["attempted_samples"], 47,
            "seed {seed}: the corner is cell 2 of 4, so the forty-seventh \
             attempt; a different count means the phase boundary moved: \
             {record:#}"
        );
        assert_eq!(
            record["counterexample"]["x"].as_f64(),
            Some(f64::MAX),
            "seed {seed}: the first binder takes its largest candidate: {record:#}"
        );
        assert_eq!(
            record["counterexample"]["y"].as_f64(),
            Some(f64::from_bits(1)),
            "seed {seed}: the second takes its smallest: {record:#}"
        );
    }
}

/// chelis#3697's reported property, which is chelis#3636's: `x / y` over two
/// positive binders overflows, and the overflow needs a large `x` *with* a
/// small `y`. `issue_3636_edge_sampling.rs` asserted that this was **not**
/// refuted and asked to be deleted when the pairing improved; that test is
/// gone and this one replaces it.
///
/// The reported witness is the shrunk one — `shrink_steps` is 64 here, and
/// `spec/design/chelis_property_spec.md` still says shrinking is deferred,
/// which chelis#3685 owns — so this asserts the property of the witness
/// rather than the drawn values: `y` stays at the subnormal the corner
/// proposed, and `x / y` is not finite. The test above pins the drawn values
/// instead, on a property nothing can shrink.
#[test]
fn issue_3697_the_reported_overflow_is_refuted_at_every_seed() {
    let source = "module Pairing.Div
def subject(x: f64, y: f64) -> f64 = (x / y)
@property div_finite forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((subject(x, y) <= 1.7976931348623157e308f64) && (subject(x, y) >= -1.7976931348623157e308f64))
";
    for seed in ["0", "1", "7", "99", "12345"] {
        let (code, record) = prove(source, "200", seed);
        assert_eq!(
            code, 1,
            "seed {seed}: this is the issue's own property: {record:#}"
        );
        assert_eq!(
            record["attempted_samples"], 47,
            "seed {seed}: reached by the same cell as the corner above: {record:#}"
        );
        let x = record["counterexample"]["x"]
            .as_f64()
            .unwrap_or_else(|| panic!("seed {seed}: x: {record:#}"));
        let y = record["counterexample"]["y"]
            .as_f64()
            .unwrap_or_else(|| panic!("seed {seed}: y: {record:#}"));
        assert_eq!(
            y,
            f64::from_bits(1),
            "seed {seed}: shrinking keeps the proposed subnormal: {record:#}"
        );
        assert!(
            x > 0.0 && !(x / y).is_finite(),
            "seed {seed}: the witness must actually overflow, got {x} / {y}: {record:#}"
        );
    }
}

/// **The post-diagonal uniform draws are not spare budget, and this is the
/// control that says so.** They are the only thing that reaches a
/// counterexample in the *interior* of the proposal window, where no edge
/// candidate and no extremal representative lands. The cell phase is funded
/// from them, so what bounds its width is what keeps the generator alive.
/// Two classes make the grid small enough to fit at the arities below; the
/// budget share is what keeps it reachable above them.
///
/// The band `(-7.5, -7.0)` is 2.5% of the `[-10, 10]` window and contains no
/// candidate of any schedule. The refutation is therefore seed-dependent --
/// measured over seeds 0 to 29 at the default 100 samples, 20/30 at one
/// binder, 23/30 at two, 24/30 at three, 21/30 at four, against 20/24/26/23
/// with no cell phase at all -- so this fixes one seed that refutes at every
/// arity rather than asserting a rate. Arity one is the control within the
/// control: a single binder takes no cell, so its rate is unchanged by
/// construction. `issue_3697_a_high_arity_property_still_draws_uniformly`
/// covers the arities above four, where the share binds instead of the grid.
///
/// Each refutation arrives **after attempt 44**, where the unguarded `f64`
/// schedule is spent. That is the assertion that makes this a measurement of
/// the post-diagonal draws specifically, and not of sampling in general. A
/// policy that widens the cell phase has to re-measure these rates; with six
/// classes they were 20/11/0/0, and with a fixed 256-cell cap instead of a
/// budget share they were 20/23/24/21 here but 9/0/0/0 at arities five to
/// eight.
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

/// **An order-guarded property takes no cell at all**, and this test asserts
/// the exemption by pinning behaviour identical to the policy's absence.
///
/// A successor's candidate list is rebuilt from its predecessor *inside* the
/// attempt, so an index into that list names entry `i` of a list whose
/// contents depend on what the predecessor drew. There is no stable
/// per-binder sequence to pair: proposing a new predecessor value re-indexes
/// every successor, and an earlier attempt at this silently deleted the
/// refuting candidate from `y`'s list by pushing `x` non-negative.
/// `ConstraintSamplingPlan::has_order_edges` therefore gates the phase off
/// for the whole property rather than for the chained binders, because the
/// chain is a property of the plan and not of one seat.
///
/// Pinned by the one consequence a property can observe: the refutation is
/// seed-dependent, because it needs the predecessor's uniform draw to be
/// negative so that `y > x` admits a negative `y` at all. Seeds 1 and 3
/// refute at attempt 77 and seeds 0 and 4 do not -- the same four verdicts
/// and the same attempt count as before chelis#3697, which is the assertion.
#[test]
fn issue_3697_an_order_guarded_property_takes_no_cell() {
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

/// A binder whose guards give it a schedule *longer* than the 44-candidate
/// span keeps the diagonal until that schedule is spent, so the cells never
/// displace a scheduled candidate. Twenty `!=` exclusions on two binders push
/// the length to 52, and the refuting candidate -- a neighbour of an excluded
/// value -- is proposed there.
///
/// The phase boundary is `max(44, schedule_len)` per binder for exactly this
/// reason: a boundary fixed at 44 reported `fuzz_validated` on this shape.
#[test]
fn issue_3697_a_long_schedule_keeps_the_diagonal_until_it_is_spent() {
    let source = "module Pairing.Long
@property not_last forall(x: f64, y: f64)
where x != 5.0f64, x != 6.0f64, x != 7.0f64, x != 8.0f64, y != 5.0f64, y != 6.0f64, y != 7.0f64, y != 8.0f64:
  ((x != -1.0e-323f64) || (y != -1.0e-323f64))
";
    for seed in ["0", "1", "2"] {
        let (code, record) = prove(source, "100", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        assert_eq!(
            record["attempted_samples"], 52,
            "seed {seed}: the diagonal must run past 44 to reach this; a \
             count of 44 or a pass means the boundary was fixed at the span \
             instead of the schedule: {record:#}"
        );
    }
}

/// **The declared cost.** Two classes are the two extremes, so a
/// counterexample needing an extreme paired with a *moderate* value is not
/// reached off the diagonal. `x >= 1e300` with `y == 1.0` is such a pair:
/// `1.0` is a candidate of `y`'s schedule but not one of its two classes, and
/// no uniform draw over the window proposes `1e300`.
///
/// This is asserted rather than left undocumented because the six-class set
/// did reach it -- "nearest 1.0" was one of the six -- and the trade was made
/// deliberately against the rates in
/// `issue_3697_post_diagonal_uniform_draws_carry_refutations`. A future
/// policy that reaches this pair must show it has not moved those.
#[test]
fn issue_3697_an_extreme_paired_with_a_moderate_value_is_not_reached() {
    let source = "module Pairing.Moderate
@property not_huge_over_one forall(x: f64, y: f64):
  ((x < 1.0e300f64) || (y != 1.0f64))
";
    for seed in ["0", "1", "2", "3"] {
        let (code, record) = prove(source, "100", seed);
        assert_eq!(
            code, 0,
            "seed {seed}: two extremal classes do not reach an \
             extreme-with-moderate pair; if this now refutes, the class set \
             grew and the displaced-draw rates need re-measuring: {record:#}"
        );
        assert_eq!(record["status"], "passed", "seed {seed}: {record:#}");
    }
}

/// The cell phase must not starve the rejection lane. A five-binder property
/// under narrow range guards accepts a sample only when every binder's draw
/// satisfies its guards, so it spends thousands of attempts to accept its
/// hundred; a phase that proposed guard-violating corners for every one of
/// those attempts would lower the accepted count rather than the attempted
/// one.
///
/// Measured against the same budget without the phase: attempted samples move
/// within the run-to-run range (means 6746 and 6847 over six seeds, the new
/// figure lower on three of them) and the accepted count is exactly the
/// budget on both. The assertion is therefore on `accepted_samples`, which is
/// the invariant, and not on an attempt count that is noise at this arity.
#[test]
fn issue_3697_the_rejection_lane_still_accepts_its_whole_budget() {
    let source = "module Pairing.Reject
@property all_above_one forall(a: f64, b: f64, c: f64, d: f64, e: f64, z: bool)
where a > 1.1f64, a < 9.7f64, b > 1.1f64, b < 9.7f64, c > 1.1f64, c < 9.7f64,
      d > 1.1f64, d < 9.7f64, e > 1.1f64, e < 9.7f64:
  ((((((a > 1.0f64) && (b > 1.0f64)) && (c > 1.0f64)) && (d > 1.0f64)) && (e > 1.0f64)) || z)
";
    for seed in ["0", "1"] {
        let (code, record) = prove(source, "100", seed);
        assert_eq!(code, 0, "seed {seed}: {record:#}");
        assert_eq!(
            record["accepted_samples"], 100,
            "seed {seed}: the lane must still accept its whole budget: {record:#}"
        );
        assert_eq!(
            record["composite_verdict"], "fuzz_validated",
            "seed {seed}: {record:#}"
        );
    }
}

/// **The cells may never take so much of the budget that the uniform draw
/// stops happening.** This is the arity axis the published four-row table
/// does not cover, and its absence was a default-budget false pass: a
/// contiguous `min(2^n, 256)` block consumed all 56 post-diagonal attempts
/// from six binders, so a six-binder property proposed no uniform draw at
/// all and the window interior became unreachable.
///
/// Measured over seeds 0–29 at the default 100 samples, refutations of the
/// window-interior band: `main` 21/23/23/23 at arities five to eight, a
/// 256-cell cap 9/0/0/0, and the budget share 11/15/11/19. The share never
/// restores `main`'s rate — half the post-diagonal attempts are spent on
/// cells by design — but the loss is a reduced chance rather than the total,
/// seed-independent loss a fixed cap produced.
///
/// Seed 7 refutes at all four arities, so this fixes one seed rather than
/// asserting a rate, exactly as the arity 1-4 control does with seed 1. The
/// rates are in the design doc.
#[test]
fn issue_3697_a_high_arity_property_still_draws_uniformly() {
    for arity in [5usize, 6, 7, 8] {
        let seed = "7";
        let binders = (0..arity)
            .map(|index| format!("v{index}: f64"))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "module Pairing.HighArity
@property outside_band forall({binders}):
  ((v0 <= -7.5f64) || (v0 >= -7.0f64))
"
        );
        let (code, record) = prove(&source, "100", seed);
        assert_eq!(
            code, 1,
            "arity {arity}: the band is reachable only by a uniform draw, so \
             a cell phase that claims every post-diagonal attempt loses it \
             entirely: {record:#}"
        );
        let witness = record["counterexample"]["v0"]
            .as_f64()
            .unwrap_or_else(|| panic!("arity {arity}: witness: {record:#}"));
        assert!(
            witness > -7.5 && witness < -7.0,
            "arity {arity}: got {witness}: {record:#}"
        );
        let attempts = record["attempted_samples"]
            .as_u64()
            .unwrap_or_else(|| panic!("arity {arity}: attempts: {record:#}"));
        assert!(
            attempts > 44,
            "arity {arity}: must arrive after the diagonal: {attempts}: {record:#}"
        );
    }
}

/// The benefit side of the same axis: capping the cells at a share of the
/// budget costs almost no pairing, because **the corners worth proposing sit
/// at low cell indices.** A pair of opposite extremes on the first two
/// binders is cell 2 whatever the arity, so it is reached at attempt 47 at
/// every arity measured, while `main` passes its whole budget.
///
/// That is what makes the share a cheap bound rather than a retreat: the
/// attempts it gives back to the generator buy a prefix of the grid that
/// already contains the pairings the issue is about. Above four binders the
/// grid is covered only to that prefix — `2^5 = 32` exceeds the 28 the
/// default budget allows — so a corner on a high-order binder is not
/// reached, which is the declared cost.
#[test]
fn issue_3697_a_low_order_corner_is_reached_at_every_arity() {
    for arity in [2usize, 4, 5, 6, 8] {
        let binders = (0..arity)
            .map(|index| format!("v{index}: f64"))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "module Pairing.Corner
@property corner forall({binders}):
  ((v0 != 1.7976931348623157e308f64) || (v1 != -1.7976931348623157e308f64))
"
        );
        let (code, record) = prove(&source, "100", "0");
        assert_eq!(
            code, 1,
            "arity {arity}: cell 2 pairs v0's largest with v1's smallest at \
             every arity: {record:#}"
        );
        assert_eq!(
            record["attempted_samples"], 47,
            "arity {arity}: and it is the same attempt at every arity, \
             because the cell index does not depend on the arity: {record:#}"
        );
    }
}

/// **The third declared cost: binders whose schedules differ in length may
/// never be in a cell at the same attempt, and then no pairing happens.**
///
/// Each binder's cell window begins at its own `max(44, L)`. With `L = 52`
/// and `L = 44` those windows are `[52, 56)` and `[44, 48)` — disjoint — so
/// the corner is unreachable at any budget. The two controls below show the
/// witness is real and the pairing works as soon as the lengths agree, which
/// makes this a window-alignment cost and not a magnitude-reach one.
///
/// This is not a regression against `main`, which reaches none of these
/// corners. Aligning every binder's cell window on one shared boundary would
/// fix it and is tracked separately.
#[test]
fn issue_3697_unequal_schedule_lengths_lose_the_pairing() {
    let corner = "((x != 1.7976931348623157e308f64) || (y != -1.7976931348623157e308f64))";
    let four = "x != 11.0f64, x != 12.0f64, x != 13.0f64, x != 14.0f64";
    let unequal = format!(
        "module Pairing.Unequal
@property corner forall(x: f64, y: f64)
where {four}:
  {corner}
"
    );
    for seed in ["0", "1", "7", "99"] {
        let (code, record) = prove(&unequal, "500", seed);
        assert_eq!(
            code, 0,
            "seed {seed}: x's window is [52, 56) and y's is [44, 48), so they \
             are never both in a cell; if this now refutes, the windows were \
             aligned and this cost is retired: {record:#}"
        );
    }
    // Control one: the same corner with both schedules at 52 pairs normally.
    let both = format!(
        "module Pairing.Both
@property corner forall(x: f64, y: f64)
where {four}, y != 11.0f64, y != 12.0f64, y != 13.0f64, y != 14.0f64:
  {corner}
"
    );
    let (code, record) = prove(&both, "500", "0");
    assert_eq!(code, 1, "equal lengths pair: {record:#}");
    // Control two: and with both at 44.
    let neither = format!(
        "module Pairing.Neither
@property corner forall(x: f64, y: f64):
  {corner}
"
    );
    let (code, record) = prove(&neither, "500", "0");
    assert_eq!(code, 1, "no exclusions at all: {record:#}");
    assert_eq!(record["attempted_samples"], 47, "{record:#}");
}
