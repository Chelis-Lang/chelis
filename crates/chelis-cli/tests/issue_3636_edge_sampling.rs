//! chelis#3636: Tier C mixes a deterministic edge set into its proposals.
//!
//! The issue's complaint is not that the proposal box is too narrow
//! (chelis#644) but that no point inside the guards is ever *chosen*
//! deliberately: the sampler drew uniformly and so never visited a guard
//! endpoint, an exact representable singularity, a subnormal, or an extreme
//! magnitude. Each test below names the class of point the issue asks for.
//!
//! Every positive test (the edge point is reached, so the false property
//! fails) is paired with a negative test (a property that is actually true
//! over the same domain still passes, and the sampler still reports its
//! guard-filtered sample accounting).
//!
//! **On discrimination.** Each class has at least one test that fails when
//! that class alone is removed from the schedule, verified by mutation rather
//! than asserted. Two classes cannot be isolated by construction, and the
//! tests say which test covers them instead of implying a dedicated one:
//!
//! - A *strict* guard's endpoint already is its adjacent representable
//!   neighbour, and `x >= 0.0` makes `0.0` the endpoint rather than an
//!   interior zero. So the `..._interior_...` and `..._nonstrict_...` tests
//!   carry the zeros, subnormal and adjacent-neighbour classes; the
//!   endpoint-guard tests cannot.
//! - `±MAX` has no separate candidate. An unbounded side's endpoint *is* the
//!   dtype extremum, and under any narrower guard `±MAX` is inadmissible, so
//!   a dedicated push would be dead code. The endpoint class covers it.
use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// Run `prove --tier fuzz-only` over `source` and return the exit code with
/// the single property record.
fn prove(source: &str, samples: &str, seed: &str) -> (i32, Value) {
    prove_at_tier(source, samples, seed, "fuzz-only")
}

/// As [`prove`], at an explicit tier.
fn prove_at_tier(source: &str, samples: &str, seed: &str, tier: &str) -> (i32, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("edge.ch");
    std::fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            path.to_str().expect("utf-8 path"),
            "--json",
            "--tier",
            tier,
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

/// A finiteness claim over `body`, guarded by `guard` when non-empty.
fn finite_claim(name: &str, body: &str, guard: &str) -> String {
    let where_clause = if guard.is_empty() {
        String::new()
    } else {
        format!("\nwhere {guard}")
    };
    format!(
        "module Edge.Finite\ndef subject(x: f64) -> f64 = {body}\n\
         @property {name} forall(x: f64){where_clause}:\n  \
         ((subject(x) <= 1.7976931348623157e308f64) && (subject(x) >= -1.7976931348623157e308f64))\n"
    )
}

// --- the issue's own two reproductions -----------------------------------

/// `1/(x+1) - 1/x` is infinite at the exactly representable `x = -1.0`,
/// which sits strictly inside the stated guard. The issue measured
/// `100/100 passed` and `10000/10000 passed` here.
#[test]
fn issue_3636_representable_singularity_inside_the_guard_is_refuted() {
    let source = finite_claim(
        "recip_gap_finite_neg",
        "((1.0f64 / (x + 1.0f64)) - (1.0f64 / x))",
        "x > -2.0f64, x < -0.5f64",
    );
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
    assert_ne!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert!(
        witness > -2.0 && witness < -0.5,
        "the witness must satisfy the guard it was drawn under: {record:#}"
    );
}

/// `x * x` overflows at an ordinary representable magnitude. The binder is
/// unguarded, so the admissible range is the whole of `f64` and the issue's
/// "0, ±min subnormal, ±max finite" row is what reaches it.
#[test]
fn issue_3636_unguarded_binder_reaches_extreme_magnitudes() {
    let source = finite_claim("sq_finite", "(x * x)", "");
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
    assert_ne!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert!(
        witness.abs() > 1.0e100,
        "overflow needs an extreme magnitude, got {witness}: {record:#}"
    );
}

// --- one test per class of point the issue's request list names ----------

/// Guard endpoints. `x <= 2.0` admits exactly `2.0`, and only that endpoint
/// refutes `subject(x) < 4.0`.
#[test]
fn issue_3636_guard_endpoint_is_proposed() {
    let source = "module Edge.Endpoint
def subject(x: f64) -> f64 = (x * x)
@property below_four forall(x: f64)
where x >= 0.0f64, x <= 2.0f64:
  (subject(x) < 4.0f64)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["counterexample"]["x"], 2.0, "{record:#}");
}

/// The double adjacent to a guard endpoint. A strict guard admits exactly
/// `next_up(1.0)`, where `x - 1.0` is `2.22e-16`; the property is true from
/// `1.0 + 1e-12` upwards, and the proposal window for a one-sided guard is
/// 20 wide, so a uniform draw refutes this only with probability ~5e-14.
#[test]
fn issue_3636_double_adjacent_to_a_guard_endpoint_is_proposed() {
    let source = "module Edge.Adjacent
@property gap_above_one forall(x: f64)
where x > 1.0f64:
  ((x - 1.0f64) > 1.0e-12f64)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert!(
        witness > 1.0 && witness - 1.0 <= 1.0e-12,
        "the witness must be the adjacent double, got {witness}: {record:#}"
    );
}

/// Negative control for the adjacent double: the same strict guard, with a
/// property that is true at `next_up(1.0)` as well, stays green.
#[test]
fn issue_3636_adjacent_double_does_not_refute_a_true_strict_guard() {
    let source = "module Edge.AdjacentTrue
@property above_one forall(x: f64)
where x > 1.0f64:
  (x > 1.0f64)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// Zero. `x >= 0.0` admits the exact zero at which `1/x` is infinite.
#[test]
fn issue_3636_zero_is_proposed() {
    let source = finite_claim("recip_finite", "(1.0f64 / x)", "x >= 0.0f64, x <= 1.0f64");
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["counterexample"]["x"], 0.0, "{record:#}");
}

/// The minimum positive subnormal. `1/x` overflows there while remaining
/// finite at every normal magnitude the uniform draw can reach.
#[test]
fn issue_3636_minimum_subnormal_is_proposed() {
    let source = finite_claim("recip_finite_pos", "(1.0f64 / x)", "x > 0.0f64, x < 1.0f64");
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert!(
        witness > 0.0 && witness < 1.0e-300,
        "overflowing `1/x` needs a subnormal, got {witness}: {record:#}"
    );
}

/// A small integer. `x != 0` is not stated, and the only representable
/// value at which `subject` divides by zero is the integer `3`.
#[test]
fn issue_3636_small_integers_are_proposed() {
    let source = finite_claim(
        "shifted_pole_finite",
        "(1.0f64 / (x - 3.0f64))",
        "x > 2.0f64, x < 4.0f64",
    );
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["counterexample"]["x"], 3.0, "{record:#}");
}

// --- `!=` guards ---------------------------------------------------------

/// The issue's third reproduction: a `!=` guard reported
/// `unsupported scalar guard: expected a conjunction of <, <=, >, or >=
/// comparisons`. It must now sample, and must reach the neighbours of the
/// excluded value, where `1/x` overflows.
#[test]
fn issue_3636_not_equal_guard_is_sampled_and_its_neighbours_are_proposed() {
    let source = finite_claim("inv_finite", "(1.0f64 / x)", "x != 0.0f64");
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
    assert_eq!(
        record["sampling_method"], "constraint_directed",
        "{record:#}"
    );
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_ne!(witness, 0.0, "the `!=` guard must still exclude its value");
}

/// Negative control for the `!=` guard: a property that holds on the
/// excluded value's neighbourhood passes, and no sample equals the excluded
/// value.
#[test]
fn issue_3636_not_equal_guard_never_proposes_the_excluded_value() {
    let source = "module Edge.NotEqual
@property nonzero forall(x: f64)
where x != 0.0f64, x > -1.0f64, x < 1.0f64:
  ((x * x) >= 0.0f64)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    assert_eq!(
        record["sampling_method"], "constraint_directed",
        "{record:#}"
    );
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// A `!=` between two binders is not an interval constraint and stays
/// unsupported rather than silently sampling a domain it cannot enforce.
#[test]
fn issue_3636_not_equal_between_binders_stays_unsupported() {
    let source = "module Edge.NotEqualVars
@property distinct forall(x: f64, y: f64)
where x != y:
  (x == x)
";
    let (code, record) = prove(source, "32", "0");
    assert_eq!(code, 2, "{record:#}");
    assert_eq!(record["status"], "unsupported", "{record:#}");
    assert_ne!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    assert_eq!(record["accepted_samples"], 0, "{record:#}");
}

/// A `!=` that empties the domain is still caught before any sample.
#[test]
fn issue_3636_not_equal_emptying_a_singleton_domain_fails_closed() {
    let source = "module Edge.NotEqualEmpty
@property impossible forall(x: f64)
where x >= 1.0f64, x <= 1.0f64, x != 1.0f64:
  (x == x)
";
    let (code, record) = prove(source, "32", "0");
    assert_eq!(code, 2, "{record:#}");
    assert_eq!(record["status"], "unsupported", "{record:#}");
    assert_ne!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    assert_eq!(record["accepted_samples"], 0, "{record:#}");
}

// --- negative parity: true properties still pass, with honest accounting --

/// The guarded negative control. Every point the edge set adds to
/// `-2 < x < -0.5` satisfies this, so the verdict stays green and the
/// guard-filtered accounting is unchanged.
#[test]
fn issue_3636_true_guarded_property_still_passes_with_unchanged_accounting() {
    let source = "module Edge.True
@property in_range forall(x: f64)
where x > -2.0f64, x < -0.5f64:
  ((x < -0.5f64) && (x > -2.0f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["attempted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// The unguarded negative control: an edge set that now reaches `±f64::MAX`
/// and the subnormals must not invent a counterexample for a property that
/// is true at every finite double.
#[test]
fn issue_3636_true_unguarded_property_still_passes_over_the_whole_range() {
    let source = "module Edge.TrueUnguarded
@property reflexive forall(x: f64):
  (x == x)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["composite_verdict"], "fuzz_validated", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
}

/// The edge set is part of the sample sequence, so it stays
/// seed-deterministic: the same seed and inputs produce the same record.
#[test]
fn issue_3636_edge_schedule_is_seed_deterministic() {
    let source = finite_claim("sq_finite", "(x * x)", "");
    let (first_code, first) = prove(&source, "100", "7");
    let (second_code, second) = prove(&source, "100", "7");
    assert_eq!(first_code, second_code);
    assert_eq!(
        first["counterexample"], second["counterexample"],
        "{first:#}"
    );
}

/// The edge set does not depend on the seed, so a property refuted only at an
/// edge point is refuted under every seed rather than at one lucky draw.
#[test]
fn issue_3636_edge_schedule_refutes_under_every_seed() {
    let source = finite_claim("sq_finite", "(x * x)", "");
    for seed in ["0", "1", "42", "9999"] {
        let (code, record) = prove(&source, "100", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        assert_eq!(record["status"], "failed", "seed {seed}: {record:#}");
    }
}

/// A sample budget smaller than the edge set still spends it on edge points
/// rather than on uniform draws, so the refutation survives `--samples 4`.
#[test]
fn issue_3636_small_sample_budget_spends_on_edge_points_first() {
    let source = finite_claim("sq_finite", "(x * x)", "");
    let (code, record) = prove(&source, "4", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
}

/// The `f32` absorption `x + 1.0 == x`. It first holds at `x = 2^24`, where
/// the ulp becomes `2.0` and adding exactly half of it ties to even; through
/// `[2^24, 2^25)` it alternates with the mantissa parity, and from `2^25`
/// upward it holds at every representable value. This
/// was the body of `chelis-prove`'s own
/// `GREEN_FUZZ_PRECONDITION_PROPERTY` fixture, asserted to *pass* via Tier C;
/// it passed only because no large magnitude was ever proposed. The fixture
/// now uses a total body and the defect it was hiding is pinned here.
#[test]
fn issue_3636_f32_absorption_at_large_magnitudes_is_refuted() {
    let source = "module Edge.Absorb
@property increments forall(x: f32)
where x > 0.0:
  ((x + 1.0) > x)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f32 witness: {record:#}"));
    assert!(
        witness >= 16_777_216.0,
        "absorption starts at 2^24, got {witness}: {record:#}"
    );
    // Check the witness really absorbs rather than merely being large:
    // absorption alternates through `[2^24, 2^25)`, so clearing the
    // threshold does not by itself make `x + 1.0 == x`.
    let witness = witness as f32;
    assert_eq!(
        witness + 1.0,
        witness,
        "the reported witness must actually absorb: {record:#}"
    );
}

/// Negative parity for absorption: the same guard with a body that survives
/// absorption stays green, so the test above is not just "large magnitudes
/// break everything".
#[test]
fn issue_3636_f32_doubling_at_large_magnitudes_still_passes() {
    let source = "module Edge.Double
@property doubles forall(x: f32)
where x > 0.0:
  ((x * 2.0) > x)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
}

/// Negative zero. `x >= 0.0` admits `-0.0` as well, because IEEE equates the
/// two zeros under comparison but not under division: `1/0.0` is `+inf` and
/// `1/-0.0` is `-inf`, so only the signed zero refutes a positive-result
/// claim.
#[test]
fn issue_3636_negative_zero_is_proposed() {
    let source = "module Edge.NegativeZero
def subject(x: f64) -> f64 = (1.0f64 / x)
@property positive_recip forall(x: f64)
where x >= 0.0f64, x <= 1.0f64:
  (subject(x) > 0.0f64)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_eq!(witness, 0.0, "the witness is a zero: {record:#}");
    assert!(
        witness.is_sign_negative(),
        "only the negatively signed zero refutes this: {record:#}"
    );
}

/// Negative zero, interior to the guard. The shipped
/// `issue_3636_negative_zero_is_proposed` test guards `x >= 0.0`, where
/// `push(-min_positive_subnormal)` clamps up one ulp onto `-0.0` and so
/// reintroduces it without the zeros class. An interior guard has no such
/// route, so only `push(-0.0)` proposes it here.
#[test]
fn issue_3636_interior_negative_zero_is_proposed() {
    let source = "module Edge.InteriorNegativeZero
def subject(x: f64) -> f64 = (1.0f64 / x)
@property positive_recip forall(x: f64)
where x >= -1.0f64, x <= 1.0f64:
  ((x != 0.0f64) || (subject(x) > 0.0f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_eq!(witness, 0.0, "{record:#}");
    assert!(
        witness.is_sign_negative(),
        "only negative zero refutes this: {record:#}"
    );
}

/// A magnitude between the uniform proposal window and `f64::MAX`. The
/// property is true at every edge point *except* the ladder rungs inside
/// `(1e20, 1e40)`: `±f64::MAX`, the zeros, the subnormals and the small
/// integers all satisfy it, and the window only reaches 10. The ladder is a
/// fixed list rather than a random draw, so this holds under every seed.
#[test]
fn issue_3636_magnitude_ladder_reaches_between_the_window_and_max() {
    let source = "module Edge.Ladder
@property outside_band forall(x: f64):
  ((x <= 1.0e20f64) || (x >= 1.0e40f64))
";
    for seed in ["0", "1", "42", "9999"] {
        let (code, record) = prove(source, "100", seed);
        assert_eq!(code, 1, "seed {seed}: {record:#}");
        let witness = record["counterexample"]["x"]
            .as_f64()
            .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
        assert!(
            witness > 1.0e20 && witness < 1.0e40,
            "seed {seed}: the witness must be in the band, got {witness}: {record:#}"
        );
    }
}

/// Negative parity for the ladder: a claim that holds at every rung stays
/// green, so the test above is not just "extreme magnitudes break anything".
#[test]
fn issue_3636_magnitude_ladder_does_not_refute_a_total_claim() {
    let source = "module Edge.LadderTrue
@property finite_or_not forall(x: f64):
  ((x <= 1.7976931348623157e308f64) && (x >= -1.7976931348623157e308f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
}

/// Two `!=` guards that exclude the *top* two representable values of the
/// interval, leaving only its lower endpoint. A draw that lands on the upper
/// endpoint cannot be nudged up, so the nudge must walk down past the second
/// exclusion to reach `1.0`. A nudge that alternated directions would
/// ping-pong between the two exclusions and report the domain as empty — so
/// this is the test that distinguishes a monotone scan from an alternating
/// one, not an interval whose exclusions have room above them.
#[test]
fn issue_3636_exclusions_at_the_interval_top_nudge_downward_not_back_up() {
    let first = f64::from_bits(1.0f64.to_bits() + 1);
    let second = f64::from_bits(1.0f64.to_bits() + 2);
    // `[1.0, second]` holds exactly three doubles; two are excluded.
    let source = format!(
        "module Edge.TopExclusions
@property only_one forall(x: f64)
where x >= 1.0f64, x <= {second:?}f64, x != {second:?}f64, x != {first:?}f64:
  (x == 1.0f64)
"
    );
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// The same two exclusions, with a claim that is false at the next
/// representable value above them: the nudge must land there rather than
/// skip past it.
#[test]
fn issue_3636_nudge_lands_on_the_nearest_admissible_neighbour() {
    let first = f64::from_bits(1.0f64.to_bits() + 1);
    let second = f64::from_bits(1.0f64.to_bits() + 2);
    let source = format!(
        "module Edge.NudgeTarget
@property not_the_third forall(x: f64)
where x >= 1.0f64, x <= 2.0f64, x != 1.0f64, x != {first:?}f64:
  (x != {second:?}f64)
"
    );
    let (code, record) = prove(&source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_eq!(witness, second, "{record:#}");
}

/// `auto` reaches the same schedule. Tier C is `auto`'s fall-through, and in
/// a build without the `smt` feature it is the only lane these properties
/// reach; with the feature the solver refutes them over the reals instead.
/// Either way the verdict must not be green, which is the assertion both
/// builds share.
#[test]
fn issue_3636_auto_does_not_report_green_for_a_refutable_property() {
    for body in ["(x * x)", "((1.0f64 / (x + 1.0f64)) - (1.0f64 / x))"] {
        let source = finite_claim("subject_finite", body, "");
        let (code, record) = prove_at_tier(&source, "100", "0", "auto");
        assert_eq!(code, 1, "{body}: {record:#}");
        assert_eq!(record["status"], "failed", "{body}: {record:#}");
        assert_ne!(
            record["composite_verdict"], "fuzz_validated",
            "{body}: {record:#}"
        );
    }
}

/// A mixed-type property takes the typed-generator lane, where a guard is
/// enforced by rejection rather than by interval construction. Both binders
/// index the same schedule, so a rejected integer candidate discards the
/// float candidate at that ordinal — the limitation recorded in
/// `spec/design/chelis_property_spec.md`. The lane must still propose enough
/// of each schedule to refute: `x * x` overflows at the float binder's
/// `±MAX`, and the surviving attempt is the one whose integer candidate
/// satisfies `n > 0`.
#[test]
fn issue_3636_mixed_binder_lane_proposes_edge_points_despite_rejection() {
    let source = "module Edge.Mixed
def subject(x: f64) -> f64 = (x * x)
@property mixed_finite forall(x: f64, n: i32)
where n > 0:
  ((subject(x) <= 1.7976931348623157e308f64) && (n > 0))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["status"], "failed", "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert!(
        witness.abs() > 1.0e100,
        "overflow needs an extreme magnitude, got {witness}: {record:#}"
    );
    assert!(
        record["counterexample"]["n"].as_i64().unwrap_or_default() > 0,
        "the witness must satisfy the rejected guard: {record:#}"
    );
    assert!(
        record["rejected_samples"].as_u64().unwrap_or_default() >= 1,
        "the integer schedule's non-positive candidates are rejected: {record:#}"
    );
}

// --- binder-order guards -------------------------------------------------
//
// The edge schedule reads each binder's *admissible* range while the random
// draw prefers a narrower proposal window. An order predecessor drawn from
// the admissible range can therefore sit outside its successor's window
// entirely. These properties are the shapes where that happens; each passed
// before the schedule existed, so each is a regression test.

/// The canonical interval shape. The schedule hands `lo` a magnitude far
/// outside `hi`'s proposal window, so `hi`'s window admits nothing and the
/// draw must fall back to its admissible range instead of failing the
/// attempt. Before that fallback this reported
/// `unsupported: inconsistent scalar guards leave `hi` with no representable
/// sample` under `fuzz-only`, `auto` and Tide, under every seed.
#[test]
fn issue_3636_unbounded_binder_order_guard_is_sampled() {
    let source = "module Edge.Order
def width(lo: f64, hi: f64) -> f64 = (hi - lo)
@property width_positive forall(lo: f64, hi: f64)
where lo < hi:
  (width(lo, hi) > 0.0f64)
";
    let (code, record) = prove_at_tier(source, "100", "0", "fuzz-only");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["attempted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");

    // `auto` reaches the same sampler, but its sample counts are not
    // build-portable: without the `smt` feature it falls through to Tier C
    // and accepts 100 samples, while with the feature the solver proves this
    // over the reals and reports zero. Assert only what both builds share --
    // that it is not the `unsupported` this shape used to report.
    let (code, record) = prove_at_tier(source, "100", "0", "auto");
    assert_eq!(code, 0, "auto: {record:#}");
    assert_eq!(record["status"], "passed", "auto: {record:#}");
}

/// The same shape as a four-binder chain, where every successor inherits a
/// predecessor's extreme value. `ord_chain4` regressed identically.
#[test]
fn issue_3636_binder_order_chain_is_sampled() {
    let source = "module Edge.Chain
@property ascending forall(a: f64, b: f64, c: f64, d: f64)
where a < b, b < c, c < d:
  ((a < d) && (b < c))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert_eq!(record["rejected_samples"], 0, "{record:#}");
}

/// A one-sided order guard on an `f32` pair, and a three-binder property
/// where only two binders are ordered: the same fallback, different shapes.
#[test]
fn issue_3636_partially_ordered_and_f32_order_guards_are_sampled() {
    for source in [
        "module Edge.OrderF32\n@property ord forall(x: f32, y: f32)\nwhere x < y:\n  (x < y)\n",
        "module Edge.OrderFree\n@property ord forall(x: f64, y: f64, z: f64)\nwhere x < y:\n  (z == z)\n",
        "module Edge.OrderLower\n@property ord forall(x: f64, y: f64)\nwhere x > 0.0f64, x < y:\n  (y > 0.0f64)\n",
    ] {
        let (code, record) = prove(source, "100", "0");
        assert_eq!(code, 0, "{source}: {record:#}");
        assert_eq!(record["status"], "passed", "{source}: {record:#}");
        assert_eq!(record["accepted_samples"], 100, "{source}: {record:#}");
        assert_eq!(record["rejected_samples"], 0, "{source}: {record:#}");
    }
}

/// An ordered pair must still *reach* the extremes, not merely avoid
/// erroring. `hi - lo` exceeds `1e292` only when both binders are near an
/// IEEE extremum, which the proposal window's width of 20 cannot produce.
///
/// This pins the schedule, not the window fallback: the refuting pair is
/// `lo = -MAX`, `hi = next_up(-MAX)` at attempt 0, whose difference is the
/// ulp at that magnitude, so it is found before any attempt reaches the
/// fallback. The three tests above are what pin the fallback.
#[test]
fn issue_3636_binder_order_guard_reaches_extreme_magnitudes() {
    let source = "module Edge.OrderExtreme
@property narrow_width forall(lo: f64, hi: f64)
where lo < hi:
  ((hi - lo) < 1.0e292f64)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let lo = record["counterexample"]["lo"]
        .as_f64()
        .unwrap_or_else(|| panic!("lo: {record:#}"));
    let hi = record["counterexample"]["hi"]
        .as_f64()
        .unwrap_or_else(|| panic!("hi: {record:#}"));
    assert!(lo < hi, "the witness must satisfy its guard: {record:#}");
    assert!(hi - lo >= 1.0e292, "{record:#}");
}

// --- discriminators for classes the schedule supplies only interiorly ----
//
// A guard whose endpoint *is* the landmark cannot distinguish the landmark's
// own candidate from the endpoint candidate: under `x >= 0.0` the lower
// endpoint is already `0.0`, and under `x > 0.0` it is already the smallest
// subnormal. Each test below places the landmark strictly inside the guard so
// that only its own class can propose it.

/// Both zeros, interior to the guard. `x >= -1.0, x <= 1.0` makes `0.0`
/// neither endpoint, so only the zeros class proposes it.
#[test]
fn issue_3636_interior_zero_is_proposed() {
    let source = "module Edge.InteriorZero
@property nonzero forall(x: f64)
where x >= -1.0f64, x <= 1.0f64:
  ((x > 0.0f64) || (x < 0.0f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["counterexample"]["x"], 0.0, "{record:#}");
}

/// The smallest positive subnormal, interior to the guard. True at `0.0` and
/// from `1e-323` upward, so neither the zeros nor any ladder rung refutes it.
#[test]
fn issue_3636_interior_minimum_subnormal_is_proposed() {
    let source = "module Edge.InteriorSubnormal
@property not_tiny forall(x: f64)
where x >= -1.0f64, x <= 1.0f64:
  ((x == 0.0f64) || (x <= -1.0e-323f64) || (x >= 1.0e-323f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_eq!(
        witness.abs(),
        f64::from_bits(1),
        "only the minimum subnormal refutes this: {record:#}"
    );
}

/// The representable neighbour of a *non-strict* endpoint. Under `x >= 1.0`
/// the endpoint is `1.0` and its neighbour `next_up(1.0)` is a distinct
/// value, unlike a strict guard where the two coincide.
#[test]
fn issue_3636_adjacent_neighbour_of_a_nonstrict_endpoint_is_proposed() {
    let source = "module Edge.AdjacentNonStrict
@property endpoint_or_far forall(x: f64)
where x >= 1.0f64, x <= 2.0f64:
  ((x == 1.0f64) || (x >= 1.0000000000000004f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_eq!(
        witness,
        f64::from_bits(1.0f64.to_bits() + 1),
        "only next_up(1.0) refutes this: {record:#}"
    );
}

/// A ladder rung above `1e81`. True at and below `1e81`, and from `1e300`
/// upward, so the endpoints, `±MAX` and the lower rungs all satisfy it;
/// truncating the ladder restores the gap.
#[test]
fn issue_3636_upper_magnitude_ladder_rungs_are_proposed() {
    let source = "module Edge.UpperLadder
@property outside_band forall(x: f64):
  ((x <= 1.0e81f64) || (x >= 1.0e300f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert!(
        witness > 1.0e81 && witness < 1.0e300,
        "the witness must come from a rung inside the band, got {witness}: {record:#}"
    );
}

/// The ladder's bottom rung is a live subnormal, not zero. `1e-323` is the
/// smallest decade `f64` represents; `1e-324` rounds to zero, and so does
/// `10f64.powi(-323)`, which is why the rungs are written as literals rather
/// than computed. True at the minimum subnormal and from `1e-320` upward, so
/// only the bottom rung refutes.
#[test]
fn issue_3636_bottom_magnitude_ladder_rung_is_a_live_subnormal() {
    let source = "module Edge.BottomLadder
@property not_bottom_rung forall(x: f64)
where x >= 0.0f64:
  ((x <= 4.9e-324f64) || (x >= 1.0e-320f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 1, "{record:#}");
    let witness = record["counterexample"]["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("f64 witness: {record:#}"));
    assert_eq!(witness, 1.0e-323, "{record:#}");
    assert_ne!(witness, 0.0, "the rung must not have rounded to zero");
}

/// The integer schedule. An `i32` binder is offered its fuzz range's lower
/// end at the first attempt, so `--samples 1` refutes deterministically; a
/// uniform draw would propose `-1000` with probability `1/2001`.
#[test]
fn issue_3636_integer_schedule_proposes_its_range_ends() {
    let source = "module Edge.IntEnds
@property above_floor forall(n: i32):
  (n > -1000)
";
    let (code, record) = prove(source, "1", "0");
    assert_eq!(code, 1, "{record:#}");
    assert_eq!(record["counterexample"]["n"], -1000, "{record:#}");
}

/// Negative parity for the integer schedule: a claim true across the whole
/// fuzz range stays green, so the test above is not just "integers break".
#[test]
fn issue_3636_integer_schedule_does_not_refute_a_total_claim() {
    let source = "module Edge.IntTotal
@property in_range forall(n: i32):
  ((n >= -1000) && (n <= 1000))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
}

// --- an unresolvable attempt is rejected, not terminal -------------------
//
// An edge candidate can pin an order successor onto a value its guards then
// refuse: a non-strict order edge reserves no spacing, so a predecessor at
// its interval's top collapses the successor to a singleton that a `!=`
// guard excludes. That attempt yields no sample, but the guards are
// satisfiable and later attempts succeed. Routing it to the same terminal
// verdict as "these guards admit nothing" aborted satisfiable properties.
//
// Emptiness the guards really imply is decided in
// `ConstraintSamplingPlan::derive`, before any draw, so the fail-closed
// tests above still hold.

/// A non-strict order edge plus an exclusion at the successor's upper bound.
/// `x = 1.0, y = 1.5` satisfies every guard, so `unsupported` would be false.
#[test]
fn issue_3636_nonstrict_order_with_an_exclusion_is_sampled() {
    let source = "module Edge.OrderExclusion
@property order_ne_nonstrict forall(x: f64, y: f64)
where x > 0.0f64, y >= x, y <= 2.0f64, y != 2.0f64:
  ((y >= x) && (y <= 2.0f64))
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    // The attempt that could not resolve is accounted for as a rejection,
    // not hidden: attempts exceed accepted samples by exactly that much.
    assert!(
        record["attempted_samples"].as_u64().unwrap_or_default() > 100,
        "the skipped attempt must be counted: {record:#}"
    );
    assert!(
        record["rejected_samples"].as_u64().unwrap_or_default() >= 1,
        "the skipped attempt must be reported as rejected: {record:#}"
    );
}

/// The same class through a three-binder chain, where the exclusion sits on
/// the last link.
#[test]
fn issue_3636_order_chain_with_an_exclusion_is_sampled() {
    let source = "module Edge.ChainExclusion
@property chain_ne forall(x: f64, y: f64, z: f64)
where x <= y, y <= z, z <= 3.0f64, z != 3.0f64:
  (x <= z)
";
    let (code, record) = prove(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert!(
        record["rejected_samples"].as_u64().unwrap_or_default() >= 1,
        "{record:#}"
    );
}

/// Negative parity: a chain whose guards really admit nothing is still
/// decided before any draw, with `unsupported` and zero accepted samples
/// rather than an exhausted generator. This is the boundary the class fix
/// must not cross.
#[test]
fn issue_3636_strict_chain_without_room_still_fails_closed_before_sampling() {
    for source in [
        // too few representable f32 values to space a strict chain
        "module Edge.NoRoom\n@property no_slots forall(x: f32, y: f32)\nwhere x > 0.9999999, x < y, y < 1.0:\n  (x < y)\n",
        // an exclusion that empties a singleton interval
        "module Edge.NoValue\n@property no_value forall(x: f64)\nwhere x >= 1.0f64, x <= 1.0f64, x != 1.0f64:\n  (x == x)\n",
    ] {
        let (code, record) = prove(source, "32", "0");
        assert_eq!(code, 2, "{source}: {record:#}");
        assert_eq!(record["status"], "unsupported", "{source}: {record:#}");
        assert_eq!(record["accepted_samples"], 0, "{source}: {record:#}");
        assert_eq!(record["attempted_samples"], 0, "{source}: {record:#}");
        assert_ne!(
            record["composite_verdict"], "fuzz_validated",
            "{source}: {record:#}"
        );
    }
}

/// The multi-binder diagonal, pinned as a known limitation rather than left
/// to be rediscovered. Binders with equal ranges take the same candidate
/// index at every scheduled attempt, so this property -- false at a pair of
/// guard-admitted representable values -- is not refuted.
///
/// This test asserts the limitation, so it fails if the pairing ever changes.
/// That is deliberate: the change would be an improvement, and whoever makes
/// it should have to read the budget analysis in
/// `spec/design/chelis_property_spec.md` and delete this test on purpose.
#[test]
fn issue_3636_multi_binder_schedule_visits_only_the_diagonal() {
    let source = "module Edge.Diagonal
def subject(x: f64, y: f64) -> f64 = (x / y)
@property div_finite forall(x: f64, y: f64)
where x > 0.0f64, y > 0.0f64:
  ((subject(x, y) <= 1.7976931348623157e308f64) && (subject(x, y) >= -1.7976931348623157e308f64))
";
    for seed in ["0", "1", "7", "99"] {
        let (code, record) = prove(source, "200", seed);
        assert_eq!(
            code, 0,
            "seed {seed}: the diagonal limitation is expected to leave this unrefuted; \
             if this now fails, the pairing improved and this test should be deleted: {record:#}"
        );
        assert_eq!(record["status"], "passed", "seed {seed}: {record:#}");
    }
}

/// The same property with the guards narrowed so the diagonal lands on the
/// refuting pair. This proves the witness exists and that the sampler reaches
/// it when the pairing allows, so the test above records a pairing limit and
/// not a magnitude-reach limit.
#[test]
fn issue_3636_the_diagonal_witness_is_real_when_the_guards_align_it() {
    let source = "module Edge.DiagonalAligned
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

// --- the routing reads the reason ---------------------------------------
//
// A draw that produces no value does not say, by itself, whether the guards
// admit nothing. That is decided after the budget ends: `unsupported` when no
// attempt ever placed every binder, and the exhausted-generator error when
// some draw succeeded. Earlier revisions tried to decide it per attempt --
// first from `derive` predicting every terminal case, then from whether the
// binder had an order predecessor -- and each prediction had a shape it got
// wrong, which is why none is made now.

/// Exclusions covering every value of a two-value interval. `derive`'s
/// singleton check does not fire, because the interval holds two values, not
/// one, so no check before the draws can see it. Every attempt then fails
/// identically, which is what the verdict is read from.
#[test]
fn issue_3636_exclusions_covering_a_two_value_interval_report_unsupported() {
    let source = "module Edge.TwoValueExcluded
@property two_excluded forall(x: f64)
where x >= 1.0f64, x <= 1.0000000000000002f64, x != 1.0f64, x != 1.0000000000000002f64:
  (x == x)
";
    let (code, record) = prove(source, "8", "0");
    assert_eq!(code, 2, "must be unsupported, not exhausted: {record:#}");
    assert_eq!(record["status"], "unsupported", "{record:#}");
    assert!(
        record["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("no representable sample"),
        "the reason must name the guards, not the sample budget: {record:#}"
    );
    assert_ne!(record["composite_verdict"], "fuzz_validated", "{record:#}");
}

/// A binder the typed generator cannot sample at all. `unsupported_property_params`
/// admits any fixed-shape numeric tensor, but `sample_tensor_value` rejects
/// rank 0 and rank > 2, so only the sampler knows. That is a property-level
/// fact reported through the same channel.
#[test]
fn issue_3636_unsamplable_tensor_rank_reports_unsupported() {
    for ty in ["tensor[2, 2, 2, f32]", "tensor[f32]"] {
        let source =
            format!("module Edge.Rank\n@property ranked forall(t: {ty}):\n  (1.0f64 == 1.0f64)\n");
        let (code, record) = prove(&source, "8", "0");
        assert_eq!(code, 2, "{ty} must be unsupported: {record:#}");
        assert_eq!(record["status"], "unsupported", "{ty}: {record:#}");
        assert_eq!(record["accepted_samples"], 0, "{ty}: {record:#}");
        assert!(
            record["reason"]
                .as_str()
                .unwrap_or_default()
                .contains("rank-1 and rank-2"),
            "{ty}: the reason must name the rank limit: {record:#}"
        );
    }
}

/// Negative parity: a rank-2 tensor binder is sampleable, so the same channel
/// must not report it unsupported.
#[test]
fn issue_3636_samplable_tensor_rank_still_samples() {
    let source = "module Edge.Rank2
@property ranked forall(t: tensor[2, 2, f32]):
  (1.0f64 == 1.0f64)
";
    let (code, record) = prove(source, "8", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 8, "{record:#}");
}

// --- the verdict is observed, not predicted -----------------------------
//
// Whether a failed draw is a fact about the property cannot be read off the
// plan's shape. "This binder has an order predecessor" is not "this binder
// varies between attempts": the predecessor can itself be pinned by its own
// constant guards, in which case every attempt hands the successor the same
// bound and the failure recurs forever. So nothing predicts it. The loop
// records whether any attempt in the whole budget placed every binder, and
// decides from that.

/// A binder emptied by a `!=` pair, behind a predecessor that is itself
/// pinned to one value. `derive` cannot see it: the interval holds two
/// values, so the singleton check does not fire. Every attempt fails
/// identically, so the guards admit nothing and the verdict is `unsupported`.
#[test]
fn issue_3636_pinned_predecessor_emptying_a_domain_reports_unsupported() {
    let source = "module Edge.PinnedPred
@property pinned_pred forall(x: f64, y: f64)
where x >= 1.0f64, x <= 1.0f64, y >= x, y <= 1.0000000000000002f64, y != 1.0f64, y != 1.0000000000000002f64:
  (y == y)
";
    for seed in ["0", "1", "99"] {
        let (code, record) = prove(source, "8", seed);
        assert_eq!(code, 2, "seed {seed}: must be unsupported: {record:#}");
        assert_eq!(record["status"], "unsupported", "seed {seed}: {record:#}");
        assert!(
            record["reason"]
                .as_str()
                .unwrap_or_default()
                .contains("no representable sample"),
            "seed {seed}: the reason must name the guards: {record:#}"
        );
        assert_ne!(
            record["composite_verdict"], "fuzz_validated",
            "seed {seed}: {record:#}"
        );
    }
}

/// The control that discriminates the observation itself. This property is
/// satisfiable, most draws succeed, and *one* attempt is unresolvable -- and
/// the budget is capped one attempt short of the sample target, so the run
/// exhausts with 99 accepted samples and one unresolved draw. Reporting
/// `unsupported` here would claim the guards admit nothing about a property
/// that just produced 99 valid samples. Only "did *any* draw succeed", not
/// "was *any* draw unresolved", separates this from the pinned-predecessor
/// case above.
#[test]
fn issue_3636_an_unresolved_attempt_among_successes_is_still_exhaustion() {
    let source = "module Edge.MixedExhaust
@property order_ne_nonstrict forall(x: f64, y: f64)
where x > 0.0f64, y >= x, y <= 2.0f64, y != 2.0f64:
  ((y >= x) && (y <= 2.0f64))
";
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mixed.ch");
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
            "100",
            "--max-attempts",
            "100",
            "--seed",
            "0",
        ])
        .output()
        .expect("run prove");
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| value["kind"] == "property")
        .unwrap_or_else(|| {
            panic!(
                "property record: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
    assert_eq!(output.status.code(), Some(3), "{record:#}");
    assert_eq!(record["status"], "error", "{record:#}");
    assert!(
        record["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("generator exhausted"),
        "draws succeeded, so this is exhaustion: {record:#}"
    );
    assert!(
        record["accepted_samples"].as_u64().unwrap_or_default() >= 1,
        "{record:#}"
    );
    assert!(
        record["rejected_samples"].as_u64().unwrap_or_default() >= 1,
        "an unresolved attempt must be present, or this pins nothing: {record:#}"
    );
}

/// The control that keeps the observation honest in the other direction. This
/// property IS satisfiable and its draws DO succeed; the precondition simply
/// rejects most of them, because the integer proposal range is far wider than
/// the guard (chelis#644). Exhausting the budget here is a genuine
/// exhaustion, and reporting `unsupported` for it would be a false claim that
/// the guards admit nothing.
#[test]
fn issue_3636_a_starved_guard_is_exhaustion_not_unsupported() {
    let source = "module Edge.Starved
@property starved forall(x: f64, n: i32)
where n > 0, n < 3:
  (n > 0)
";
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("starved.ch");
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
            "8",
            "--max-attempts",
            "40",
            "--seed",
            "0",
        ])
        .output()
        .expect("run prove");
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| value["kind"] == "property")
        .unwrap_or_else(|| {
            panic!(
                "property record: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
    assert_eq!(output.status.code(), Some(3), "{record:#}");
    assert_eq!(record["status"], "error", "{record:#}");
    assert!(
        record["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("generator exhausted"),
        "a satisfiable guard must exhaust, not report unsupported: {record:#}"
    );
    assert!(
        record["accepted_samples"].as_u64().unwrap_or_default() > 0,
        "the draws did succeed; only the guard rejected them: {record:#}"
    );
}

/// Run `prove --tier fuzz-only` over the Deep form of `source`, by desugaring
/// it with `chelis deep` first.
fn prove_deep(source: &str, samples: &str, seed: &str) -> (i32, Value) {
    let dir = tempfile::tempdir().expect("tempdir");
    let surf_path = dir.path().join("edge.ch");
    std::fs::write(&surf_path, source).expect("write fixture");
    let desugared = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["deep", surf_path.to_str().expect("utf-8 path")])
        .output()
        .expect("desugar fixture");
    assert!(
        desugared.status.success(),
        "chelis deep failed: {}",
        String::from_utf8_lossy(&desugared.stderr)
    );
    let deep_path = dir.path().join("edge.dp");
    std::fs::write(&deep_path, desugared.stdout).expect("write deep fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            deep_path.to_str().expect("utf-8 path"),
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
                "property record: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
    (output.status.code().unwrap_or(-1), record)
}

/// The Deep lane carries the same observation as the Surf lane, and until this
/// test nothing pinned it: forcing the Deep arm off alone left every suite
/// green, so the Deep lane could silently regress to reporting an exhausted
/// generator for an unsatisfiable property while CI stayed green. The Surf
/// equivalent is `issue_3636_pinned_predecessor_emptying_a_domain_reports_unsupported`.
#[test]
fn issue_3636_deep_lane_reports_unsupported_for_an_emptied_domain() {
    let source = "module Edge.DeepPinned
@property pinned_pred forall(x: f64, y: f64)
where x >= 1.0f64, x <= 1.0f64, y >= x, y <= 1.0000000000000002f64, y != 1.0f64, y != 1.0000000000000002f64:
  (y == y)
";
    let (code, record) = prove_deep(source, "8", "0");
    assert_eq!(code, 2, "{record:#}");
    assert_eq!(record["status"], "unsupported", "{record:#}");
    assert!(
        record["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("no representable sample"),
        "{record:#}"
    );
    assert_ne!(record["composite_verdict"], "fuzz_validated", "{record:#}");
}

/// Deep-lane parity for the other direction: a satisfiable property whose
/// draws succeed must still exhaust rather than be called unsupported.
#[test]
fn issue_3636_deep_lane_exhausts_rather_than_claiming_unsupported() {
    let source = "module Edge.DeepMixed
@property order_ne_nonstrict forall(x: f64, y: f64)
where x > 0.0f64, y >= x, y <= 2.0f64, y != 2.0f64:
  ((y >= x) && (y <= 2.0f64))
";
    let (code, record) = prove_deep(source, "100", "0");
    assert_eq!(code, 0, "{record:#}");
    assert_eq!(record["status"], "passed", "{record:#}");
    assert_eq!(record["accepted_samples"], 100, "{record:#}");
    assert!(
        record["rejected_samples"].as_u64().unwrap_or_default() >= 1,
        "an unresolvable attempt must occur, or this pins nothing: {record:#}"
    );
}
