//! chelis#3697: Tier C spends part of its budget off the diagonal of the
//! edge grid.
//!
//! chelis#3636 gave every binder a deterministic edge schedule indexed by the
//! run's attempt ordinal. Because the ordinal is *shared*, binders with equal
//! ranges took the same candidate at every scheduled attempt, so a
//! multi-binder property was only ever evaluated on the diagonal of the grid
//! and one false at an off-diagonal pair of guard-admitted representable
//! values still reported `fuzz_validated`.
//!
//! The policy is in `spec/design/chelis_property_spec.md`, "The off-diagonal
//! phase": the diagonal runs first at chelis#3636's unchanged one-for-one
//! rate, and once a binder's schedule is spent the attempts that would have
//! been uniform draws carry a mixed-radix odometer over each binder's
//! **extremal class representatives**, selected from that binder's own edge
//! candidates. Each test below names the part of that policy it pins, and
//! every positive test is paired with the control that keeps it from passing
//! for the wrong reason.
//!
//! **On discrimination.** Every element of the policy was removed in turn and
//! the failure observed, rather than asserted. The rows, with the test that
//! catches each:
//!
//! | removed | caught by |
//! |---|---|
//! | the off-diagonal phase | every pairing test here |
//! | the diagonal's priority over the cells | `..._the_whole_diagonal_still_needs_only_forty_four_samples` |
//! | the long-schedule exemption | `..._a_long_schedule_is_not_displaced_past_the_budget` |
//! | the cell cap | `..._a_high_arity_rejection_lane_property_still_collects_its_samples` |
//! | the arity-one carve-out | `..._single_binder_schedule_takes_no_cell` |
//! | the digit's dependence on binder position | every pairing test here |
//! | the largest-candidate role | `..._the_refuting_pair_is_off_diagonal_...` |
//! | the smallest-candidate role | `..._the_largest_negative_role_...` |
//! | the smallest-positive role | `..._the_smallest_positive_role_...` |
//! | the largest-negative role | `..._the_largest_negative_role_...` |
//! | the moderate role | `..._typed_generator_lane_pairs_off_diagonal` |
//! | counting only the scheduled binders | `..._an_unscheduled_binder_...`, `..._a_leading_unscheduled_binder_...` |
//! | the zero role | unit test only -- see below |
//!
//! The **zero role** has no discriminator at this level, and this says so
//! rather than implying one: wherever the guards admit a zero it is also the
//! interval's endpoint or its smallest-magnitude candidate, so another role
//! proposes it anyway and only the exact role list can see the difference. It
//! is pinned in `chelis-prove`'s `property_runner::tests`, under the same
//! `issue_3697_` prefix, along with the slot arithmetic no property observes.
use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// Run `prove` over `source` and return the exit code with the single
/// property record.
fn prove(source: &str, samples: &str, seed: &str) -> (i32, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pairing.ch");
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

/// Read a named `f64` component of the reported counterexample.
fn witness(record: &Value, name: &str) -> f64 {
    record["counterexample"][name]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness `{name}`: {record:#}"))
}

/// The issue's own reproduction. `x / y` overflows when a large `x` meets a
/// tiny `y`, and both components satisfy `x > 0, y > 0`, so the pair is a
/// true counterexample. The issue measured `passed` / `fuzz_validated` at
/// 200, 2000 and 20000 samples under seeds 0 and 7.
///
/// The cells are a fixed function of the attempt ordinal, so this must hold
/// under every seed and at the default budget, not only at a large one.
const DIAGONAL_REPRO: &str = "module Pair.Diag
def subject(x: f64, y: f64) -> f64 = (x / y)
@property div_finite forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((subject(x, y) <= 1.7976931348623157e308f64) && (subject(x, y) >= -1.7976931348623157e308f64))
";

#[test]
fn issue_3697_off_diagonal_pair_is_refuted_at_the_default_budget() {
    for samples in ["100", "200", "2000"] {
        for seed in ["0", "1", "7", "99"] {
            let (code, record) = prove(DIAGONAL_REPRO, samples, seed);
            assert_eq!(code, 1, "samples {samples} seed {seed}: {record:#}");
            assert_eq!(
                record["status"], "failed",
                "samples {samples} seed {seed}: {record:#}"
            );
            assert_ne!(
                record["composite_verdict"], "fuzz_validated",
                "samples {samples} seed {seed}: {record:#}"
            );
        }
    }
}

/// The witness is a *pair*, and each of its components satisfies the guard it
/// was drawn under. Without this the test above would be satisfied by any
/// refutation, including one that reported an inadmissible value.
#[test]
fn issue_3697_the_refuting_pair_is_off_diagonal_and_guard_admitted() {
    let (code, record) = prove(DIAGONAL_REPRO, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let x = witness(&record, "x");
    let y = witness(&record, "y");
    assert!(x > 0.0, "x must satisfy `x > 0.0`, got {x}: {record:#}");
    assert!(y > 0.0, "y must satisfy `y > 0.0`, got {y}: {record:#}");
    assert!(
        x / y > f64::MAX,
        "the pair must actually overflow, got {x} / {y}: {record:#}"
    );
    // Overflow needs the two binders in *different* magnitude classes, which
    // is the whole claim: on the diagonal they share a candidate index and
    // `x / y` is then near 1.
    assert!(
        x > 1.0e150 && y < 1.0e-150,
        "the pair must be off-diagonal, got {x} and {y}: {record:#}"
    );
}

/// The cell, pinned exactly. The reported witness is post-shrink -- the
/// runner shrinks before reporting -- so the test above asserts the witness's
/// *classes* rather than its values, and on its own it cannot say which cell
/// refuted. This property is false at exactly one pair, `(MAX, min
/// subnormal)`, so shrinking cannot move either component and the raw cell is
/// what gets reported.
///
/// `MAX` is each binder's largest-candidate representative and the minimum
/// subnormal its smallest-positive one, which for these guards are digits 0
/// and 1 of a three-representative list: cell 6. The cells begin once the
/// diagonal is spent, at attempt `DIAGONAL_SPAN` = 44, so cell 6 is attempt
/// 50 and the fifty-first attempt refutes, under every seed. That count is
/// the whole mechanism claim.
#[test]
fn issue_3697_the_refuting_cell_is_reached_at_a_fixed_attempt() {
    let source = "module Pair.Exact
@property only_at_the_cell forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((x < 1.7976931348623157e308f64) || (y >= 1.0e-323f64))
";
    for seed in ["0", "1", "7", "99"] {
        let (code, record) = prove(source, "100", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        assert_eq!(witness(&record, "x"), f64::MAX, "seed {seed}: {record:#}");
        assert_eq!(
            witness(&record, "y"),
            f64::from_bits(1),
            "seed {seed}: {record:#}"
        );
        assert_eq!(
            record["attempted_samples"], 51,
            "seed {seed}: the cell is attempt 50, so the fifty-first attempt \
             refutes; a different count means the pairing moved: {record:#}"
        );
    }
}

/// Negative parity for the refutation: the same two binders under the same
/// guards, with a property that is true at every admissible pair, stays green
/// with its accounting unchanged. The off-diagonal cells must not invent a
/// counterexample.
#[test]
fn issue_3697_true_two_binder_property_still_passes_with_unchanged_accounting() {
    let source = "module Pair.True
@property both_positive forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((x > 0.0f64) && (y > 0.0f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["attempted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// The cells do not ration the diagonal. This property is false only when
/// *both* binders take the schedule's last magnitude rung, `1e-323`, at
/// candidate index 42 of the unguarded `f64` schedule, which the diagonal
/// reaches at attempt 42. No extremal class representative is that rung, so
/// only the diagonal refutes this.
#[test]
fn issue_3697_the_diagonal_is_not_rationed_by_the_cells() {
    let source = "module Pair.Diagonal
@property not_both_bottom forall(x: f64, y: f64):
  ((x != 1.0e-323f64) || (y != 1.0e-323f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(witness(&record, "x"), 1.0e-323, "{record:#}");
    assert_eq!(witness(&record, "y"), 1.0e-323, "{record:#}");
}

/// A single-binder property takes no cell at all: an odometer over one
/// binder's own classes is the diagonal, so there would be nothing to gain.
/// `1e-323` is candidate 42, reached at attempt 42.
#[test]
fn issue_3697_single_binder_schedule_takes_no_cell() {
    let source = "module Pair.Single
@property not_bottom forall(x: f64):
  (x != 1.0e-323f64)
";
    let (code, record) = prove(source, "44", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(witness(&record, "x"), 1.0e-323, "{record:#}");
}

/// Negative parity for the arity-one carve-out: a single-binder property true
/// at every candidate still passes with its full accounting, so the test
/// above is not merely "small budgets refute".
#[test]
fn issue_3697_single_binder_true_property_still_passes() {
    let source = "module Pair.SingleTrue
@property reflexive forall(x: f64):
  (x == x)
";
    let (code, record) = prove(source, "44", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 44, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// Three binders, each required to be in a *different* extremal class: a
/// huge `x`, a tiny `y`, and a `z` at the moderate representative. No
/// diagonal attempt satisfies all three, because the diagonal gives them one
/// shared candidate.
///
/// The budget is generous because the cell that refutes this depends on which
/// digit each binder takes, and the digit order is the sampling order rather
/// than anything this test should assert. 500 samples covers every
/// permutation.
#[test]
fn issue_3697_three_binder_pairing_is_refuted() {
    let source = "module Pair.Triple
@property not_all_three forall(x: f64, y: f64, z: f64)
where x > 0.0f64, y > 0.0f64, z > 0.0f64:
  ((x < 1.0e300f64) || (y > 1.0e-300f64) || (z != 1.0f64))
";
    let (code, record) = prove(source, "500", "0");
    assert_eq!(code, 1, "{record:#}");
    let x = witness(&record, "x");
    let y = witness(&record, "y");
    let z = witness(&record, "z");
    assert!(x >= 1.0e300, "{record:#}");
    assert!(y <= 1.0e-300 && y > 0.0, "{record:#}");
    assert_eq!(z, 1.0, "{record:#}");
}

/// Negative parity for three binders: the same guards with a property true at
/// every admissible triple stays green.
#[test]
fn issue_3697_true_three_binder_property_still_passes() {
    let source = "module Pair.TripleTrue
@property all_positive forall(x: f64, y: f64, z: f64)
where x > 0.0f64, y > 0.0f64, z > 0.0f64:
  (((x > 0.0f64) && (y > 0.0f64)) && (z > 0.0f64))
";
    let (code, record) = prove(source, "200", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 200, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// The typed-generator lane pairs off-diagonal too. An unguarded property
/// takes that lane rather than the constraint-directed one, and this is false
/// only when `x` is at its largest candidate while `y` is at the moderate
/// representative `1.0` -- indices 1 and 8 of the same schedule, so never a
/// diagonal pair.
#[test]
fn issue_3697_typed_generator_lane_pairs_off_diagonal() {
    let source = "module Pair.Typed
@property not_huge_over_one forall(x: f64, y: f64):
  ((x < 1.0e300f64) || (y != 1.0f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["sampling_method"], "uniform", "{record:#}");
    assert!(witness(&record, "x") >= 1.0e300, "{record:#}");
    assert_eq!(witness(&record, "y"), 1.0, "{record:#}");
}

/// The integer schedule joins the same odometer. `-999` is the integer
/// schedule's sixth candidate and `x`'s largest is its second, so the
/// diagonal pairs neither with the other.
#[test]
fn issue_3697_integer_binder_joins_the_odometer() {
    let source = "module Pair.Int
@property not_huge_and_minus_999 forall(x: f64, n: i32):
  ((x < 1.0e300f64) || (n != -999))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert!(witness(&record, "x") >= 1.0e300, "{record:#}");
    assert_eq!(record["counterexample"]["n"], -999, "{record:#}");
}

/// Negative parity for the typed lane: an unguarded two-binder property true
/// at every pair of candidates stays green with full accounting.
#[test]
fn issue_3697_true_typed_generator_property_still_passes() {
    let source = "module Pair.TypedTrue
@property reflexive_pair forall(x: f64, y: f64):
  ((x == x) && (y == y))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// The order-guard shapes chelis#3636 fixed are multi-binder, so the
/// odometer runs on exactly the path whose predecessor handling is most
/// delicate: a cell value for an order successor must still be resolved
/// against the predecessor already drawn. These passed before the
/// off-diagonal phase and must still pass.
#[test]
fn issue_3697_order_guarded_pairs_still_sample_cleanly() {
    for source in [
        "module Pair.Order\n@property ord forall(lo: f64, hi: f64)\nwhere lo < hi:\n  (hi > lo)\n",
        "module Pair.Chain\n@property ascending forall(a: f64, b: f64, c: f64, d: f64)\nwhere a < b, b < c, c < d:\n  ((a < d) && (b < c))\n",
        "module Pair.OrderF32\n@property ord forall(x: f32, y: f32)\nwhere x < y:\n  (x < y)\n",
        "module Pair.OrderLower\n@property ord forall(x: f64, y: f64)\nwhere x > 0.0f64, x < y:\n  (y > 0.0f64)\n",
    ] {
        let (code, record) = prove(source, "100", "0");
        assert_eq!(code, 0, "{source}: {record:#}");
        assert_eq!(record["status"], "passed", "{source}: {record:#}");
        assert_eq!(record["accepted_samples"], 100, "{source}: {record:#}");
        assert_eq!(record["rejected_samples"], 0, "{source}: {record:#}");
    }
}

/// The fail-closed boundary is not crossed. Guards that really admit nothing
/// are still decided before any draw, with `attempted_samples: 0`, and a
/// guard shape the sampler cannot enforce is still `unsupported` rather than
/// a green empirical verdict. An off-diagonal cell must not become a route
/// around either.
#[test]
fn issue_3697_unsupportable_multi_binder_shapes_still_fail_closed() {
    for source in [
        // a `!=` between two binders is not an interval constraint
        "module Pair.NotEqualVars\n@property distinct forall(x: f64, y: f64)\nwhere x != y:\n  (x == x)\n",
        // a strict chain with too few representable f32 values to space it
        "module Pair.NoRoom\n@property no_slots forall(x: f32, y: f32)\nwhere x > 0.9999999, x < y, y < 1.0:\n  (x < y)\n",
    ] {
        let (code, record) = prove(source, "32", "0");
        assert_eq!(code, 2, "{source}: {record:#}");
        assert_eq!(record["status"], "unsupported", "{source}: {record:#}");
        assert_eq!(record["accepted_samples"], 0, "{source}: {record:#}");
        assert_ne!(
            record["composite_verdict"], "fuzz_validated",
            "{source}: {record:#}"
        );
    }
}

/// Two binders with schedules of *very* different lengths, and an exclusion
/// on the longer one. `x` is unbounded and excluded from zero, so its
/// schedule runs to twenty-odd candidates, while `y`'s interval `[1, 2]`
/// holds only four. The diagonal therefore exhausts `y` after four attempts
/// and pairs `x`'s `1.0` with a uniform draw rather than with `y`'s `1.0`, so
/// the pole `x == y` is reachable only off the diagonal -- and only because
/// the representatives are read from each binder's own list, whatever its
/// length.
///
/// The exclusion is the control: `x != 0.0f64` must still hold of the
/// witness, so a cell cannot become a route around a `!=` guard.
#[test]
fn issue_3697_exclusions_keep_their_neighbours_beside_the_cells() {
    let source = "module Pair.Excluded
def subject(x: f64, y: f64) -> f64 = (1.0f64 / (x - y))
@property quotient_finite forall(x: f64, y: f64)
where x != 0.0f64, y >= 1.0f64, y <= 2.0f64:
  ((subject(x, y) <= 1.7976931348623157e308f64) && (subject(x, y) >= -1.7976931348623157e308f64))
";
    // `x - y` is zero whenever the two coincide, and `y`'s interval holds
    // `1.0` and `2.0`, both of which `x`'s schedule proposes -- at indices
    // the diagonal never aligns, because `y`'s schedule is spent by then.
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let x = witness(&record, "x");
    let y = witness(&record, "y");
    assert_ne!(x, 0.0, "the `!=` guard must still exclude its value");
    assert!((1.0..=2.0).contains(&y), "{record:#}");
    assert_eq!(x, y, "the pole is `x == y`: {record:#}");
}

/// Determinism: the same seed and inputs produce the same record, cells
/// included.
#[test]
fn issue_3697_off_diagonal_phase_is_seed_deterministic() {
    let (first_code, first) = prove(DIAGONAL_REPRO, "100", "7");
    let (second_code, second) = prove(DIAGONAL_REPRO, "100", "7");
    assert_eq!(first_code, second_code);
    assert_eq!(
        first["counterexample"], second["counterexample"],
        "{first:#}"
    );
}

/// The subnormal underflow role, isolated. This is false only when a huge
/// `x` meets the single double strictly between zero and `1e-323` -- the
/// minimum positive subnormal -- so no other representative reaches it: the
/// minimum role is `-MAX` here, and the moderate role is `1.0`.
#[test]
fn issue_3697_the_smallest_positive_role_pairs_with_a_huge_operand() {
    let source = "module Pair.Subnormal
@property not_huge_over_subnormal forall(x: f64, y: f64):
  (((x < 1.0e300f64) || (y <= 0.0f64)) || (y >= 1.0e-323f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert!(witness(&record, "x") >= 1.0e300, "{record:#}");
    assert_eq!(witness(&record, "y"), f64::from_bits(1), "{record:#}");
}

/// Its mirror, the largest strictly negative role, with the signs reversed
/// throughout. Two roles that differ only in sign are the pair most likely to
/// be collapsed into one by a later simplification, so each needs its own
/// discriminator.
#[test]
fn issue_3697_the_largest_negative_role_pairs_with_a_huge_operand() {
    let source = "module Pair.NegSubnormal
@property not_huge_neg_over_subnormal forall(x: f64, y: f64):
  (((x > -1.0e300f64) || (y >= 0.0f64)) || (y <= -1.0e-323f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert!(witness(&record, "x") <= -1.0e300, "{record:#}");
    assert_eq!(witness(&record, "y"), -f64::from_bits(1), "{record:#}");
}

// --- only the scheduled binders join the odometer ------------------------
//
// Round 1's P1. `arity` and `position` were read off the property's whole
// parameter list, but a `bool`, a `string` and a tensor binder draw at random
// whatever the attempt. Such a binder therefore switched the odometer on
// while contributing nothing to it, and consumed a digit that multiplied
// every useful cell index by the radix.

/// Half (a): a property with one *scheduled* binder beside a `bool` must keep
/// the arity-one carve-out. `1e-323` is candidate 42 of the unguarded `f64`
/// schedule. Under the round-1 defect the unscheduled binder switched the
/// odometer on, and the revision in force at the time also rationed the
/// diagonal, so this passed where `forall(x: f64)` refuted. Both are fixed,
/// and the assertion is equality with the single-binder baseline rather than
/// a hardcoded count, so it holds whatever the rate.
#[test]
fn issue_3697_an_unscheduled_binder_does_not_shift_the_diagonal() {
    let plain = "module Pair.Plain
@property not_bottom forall(x: f64):
  (x != 1.0e-323f64)
";
    let with_bool = "module Pair.WithBool
@property not_bottom forall(x: f64, b: bool):
  ((x != 1.0e-323f64) || (b != b))
";
    let with_string = "module Pair.WithString
@property not_bottom forall(x: f64, s: string):
  ((x != 1.0e-323f64) || (s != s))
";
    let (code, baseline) = prove(plain, "44", "0");
    assert_eq!(code, 1, "{baseline:#}");
    for source in [with_bool, with_string] {
        let (code, record) = prove(source, "44", "0");
        assert_eq!(code, 1, "{source}: {record:#}");
        assert_eq!(witness(&record, "x"), 1.0e-323, "{source}: {record:#}");
        assert_eq!(
            record["attempted_samples"], baseline["attempted_samples"],
            "an unscheduled binder must not change the schedule's rate: {source}: {record:#}"
        );
    }
}

/// Half (b): an unscheduled binder declared *first* must not displace the
/// cells. This is the pull request's own deliverable class -- a two-float
/// property false at exactly one off-diagonal pair -- and under the defect the
/// leading `bool` took digit 0, pushing the refutation from attempt 62 to 362
/// and so past the default budget.
#[test]
fn issue_3697_a_leading_unscheduled_binder_does_not_displace_the_cells() {
    let plain = "module Pair.Floats
@property off_diag forall(x: f64, y: f64):
  ((x != 1.7976931348623157e308f64) || (y != 1.0f64))
";
    let leading_bool = "module Pair.LeadingBool
@property off_diag forall(bb: bool, x: f64, y: f64):
  ((x != 1.7976931348623157e308f64) || (y != 1.0f64))
";
    let (code, baseline) = prove(plain, "100", "0");
    assert_eq!(code, 1, "{baseline:#}");
    let (code, record) = prove(leading_bool, "100", "0");
    assert_eq!(
        code, 1,
        "a leading `bool` must not push the cell past the default budget: {record:#}"
    );
    assert_eq!(witness(&record, "x"), f64::MAX, "{record:#}");
    assert_eq!(witness(&record, "y"), 1.0, "{record:#}");
    assert_eq!(
        record["attempted_samples"], baseline["attempted_samples"],
        "the cell index must not depend on an unscheduled binder: {record:#}"
    );
}

/// Negative parity: an unscheduled binder is still *sampled*, and a property
/// that needs it true still passes. The repair must not stop drawing it.
#[test]
fn issue_3697_unscheduled_binders_are_still_drawn() {
    let source = "module Pair.BoolDrawn
@property either forall(x: f64, b: bool):
  ((b == b) && (x == x))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// An integer binder *is* scheduled, so it must still join the odometer: the
/// repair narrows which binders participate and must not drop a real one.
/// This is the same pairing `issue_3697_integer_binder_joins_the_odometer`
/// pins, asserted here against an unscheduled binder declared beside it.
#[test]
fn issue_3697_an_integer_binder_still_counts_beside_an_unscheduled_one() {
    let source = "module Pair.IntAndBool
@property not_huge_and_minus_999 forall(x: f64, n: i32, b: bool):
  (((x < 1.0e300f64) || (n != -999)) || (b != b))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert!(witness(&record, "x") >= 1.0e300, "{record:#}");
    assert_eq!(record["counterexample"]["n"], -999, "{record:#}");
}

/// The diagonal keeps chelis#3636's one-for-one rate, so the whole
/// 44-candidate schedule is still reached in 44 samples, exactly as before
/// this change.
///
/// An earlier revision of this policy interleaved the diagonal with the cells
/// one attempt each, which made the same schedule need 87 samples and opened
/// a `[44, 87)` band where a property that used to refute no longer did. The
/// diagonal is no longer rationed, so that band does not exist; this test is
/// what would fail if it came back.
///
/// `-1e-323` is candidate 43, the schedule's last entry. No extremal
/// representative is that rung, so only the diagonal reaches it.
#[test]
fn issue_3697_the_whole_diagonal_still_needs_only_forty_four_samples() {
    let source = "module Pair.LastCandidate
@property not_last forall(x: f64, y: f64):
  ((x != -1.0e-323f64) || (y != -1.0e-323f64))
";
    let (code, record) = prove(source, "43", "0");
    assert_eq!(
        code, 0,
        "candidate 43 is not reachable in 43 samples: {record:#}"
    );
    for samples in ["44", "100"] {
        let (code, record) = prove(source, samples, "0");
        assert_eq!(code, 1, "samples {samples}: {record:#}");
        assert_eq!(
            record["attempted_samples"], 44,
            "the diagonal must stay one-for-one: {record:#}"
        );
        assert_eq!(witness(&record, "x"), -1.0e-323, "{record:#}");
        assert_eq!(witness(&record, "y"), -1.0e-323, "{record:#}");
    }
}

/// A schedule longer than the span is not displaced past the budget. Round 2
/// of this pull request's review measured the regression this pins: each `!=`
/// exclusion adds two candidates, so four per binder gives 52, and under the
/// interleaved revision the pair refuted only at 103 samples — reporting
/// `fuzz_validated` at the default 100 where the pre-change schedule had
/// refuted at 52. That is chelis#3697's own defect on a different axis, so it
/// is pinned at the default budget and below it.
#[test]
fn issue_3697_a_long_schedule_is_not_displaced_past_the_budget() {
    let source = "module Pair.LongSchedule
@property not_last forall(x: f64, y: f64)
where x != 5.0f64, x != 6.0f64, x != 7.0f64, x != 8.0f64,
      y != 5.0f64, y != 6.0f64, y != 7.0f64, y != 8.0f64:
  ((x != -1.0e-323f64) || (y != -1.0e-323f64))
";
    for samples in ["60", "100"] {
        for seed in ["0", "7", "99"] {
            let (code, record) = prove(source, samples, seed);
            assert_eq!(code, 1, "samples {samples} seed {seed}: {record:#}");
            assert_eq!(
                record["attempted_samples"], 52,
                "the 52-candidate schedule must still finish at 52: {record:#}"
            );
        }
    }
}

/// The cell phase is capped, so a high-arity property whose precondition
/// rejects every extremal representative still collects its samples.
///
/// Round 2 measured this too: with the cells claiming every other attempt,
/// `6^5 = 7776` of them exceeded half the default attempt budget, and a
/// five-binder rejection-lane property that had passed became an exhausted
/// generator reporting `unsupported`.
#[test]
fn issue_3697_a_high_arity_rejection_lane_property_still_collects_its_samples() {
    let source = "module Pair.Rejected
@property all_above_one forall(a: f64, b: f64, c: f64, d: f64, e: f64, z: bool)
where a > 1.1f64, a < 9.7f64, b > 1.1f64, b < 9.7f64, c > 1.1f64, c < 9.7f64,
      d > 1.1f64, d < 9.7f64, e > 1.1f64, e < 9.7f64:
  ((((((a > 1.0f64) && (b > 1.0f64)) && (c > 1.0f64)) && (d > 1.0f64)) && (e > 1.0f64)) || z)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_ne!(
        record["status"], "error",
        "the capped cell phase must not exhaust the generator: {record:#}"
    );
}
