//! chelis#2316 — a `uniform_like` bound spelled as a chain of casts type-checked,
//! but both compiled lanes folded it BY VALUE and ignored each cast's target
//! dtype. `eval` applied every rounding in the chain, so the two compiled lanes
//! agreed with each other and diverged from `eval`: a shared fold defect rather
//! than a lane inconsistency.
//!
//! ## The false premise
//!
//! `extract_f64_value` (chelis-ir/src/lower.rs) recorded it in its own doc
//! comment: "A float-target cast preserves the numeric value (the f32/f64
//! sampler narrows the same bits in every lane), so it is folded through."
//! That holds for f32 and f64 targets and is false for every NARROWING float
//! target. `chelis-backend-c`'s `static_float_bound` was looser still — it
//! matched `name == "cast"`, recursed on the operand, and never looked at the
//! target at all, not even to make the float/int distinction the IR lane makes.
//!
//! Measured on `main` before the fix, for bounds
//! `cast(cast(0.30000001, f16), f32)` / `cast(cast(0.90000001, f16), f32)`:
//!
//! | | low | high |
//! |---|---|---|
//! | declared | `0.300048828125` (`0x3e99a000`) | `0.89990234375` (`0x3f666000`) |
//! | baked into the C | `0.300000012` (`0x3e99999a`) | `0.900000036` (`0x3f666667`) |
//!
//! ~1600 and ~3200 f32 ULPs — a silent bound substitution, not a rounding
//! artefact. The compiled program sampled an interval its source never declared.
//!
//! ## Spec authority
//!
//! No new atom: the spec was already right and the code was wrong.
//!
//! `spec/04-type-system.md` [04-NUM-14]: "A source cast to a float target is
//! total IEEE-754 round-to-nearest, ties-to-even at the target width per
//! [04-NUM-2], even when that loses integer exactness."
//!
//! That is the whole authority, and it is enough: a compiled bound is a value
//! produced by a cast, so it owes the target width's rounding, and a literal
//! owes its own declared width's rounding before any enclosing cast applies.
//! [04-NUM-5] is adjacent but does NOT govern here — its normative subject is
//! comparisons and condition folds, and a `uniform_like` bound fold is neither.
//!
//! ## Why one shared helper
//!
//! The rounding now lives once, in
//! `chelis_types::dtype_semantics::round_float_bound`, called by both fold
//! sites. The checker's `is_static_numeric_bound` documents itself as
//! MIRRORING the lowering, and that mirroring is the mechanism of this defect:
//! three copies of one rule drift. The lane-parity tests below would still pass
//! with two independent-but-currently-equal implementations, so the exact
//! per-dtype rounding is pinned directly by the unit tests beside
//! `round_float_bound` in `chelis-types/src/dtype_semantics.rs`.
//!
//! ## Test roles
//!
//! * `f16_*`, `bf16_*`, `negative_*` are the NEGATIVE cases: each diverged
//!   before the fix. bf16 diverges furthest (it has the fewest mantissa bits).
//! * `single_widening_cast_*` is a DISPOSITION LOCK: green before and after. A
//!   lone `cast(lit, f32)` bound always agreed across lanes, because "ignore
//!   the cast and bake the literal as f32" and "honour the cast" coincide when
//!   the target cannot narrow. It must keep agreeing.
//! * `integer_target_cast_bound_is_still_rejected` is NEGATIVE PARITY for
//!   chelis#776: an integer-target cast is left unresolved and goes loud. This
//!   fix teaches the fold to honour float targets; it must not open the
//!   integer-target path as a side effect.
//! * `runtime_computed_bound_is_still_rejected` is the other #776 gate.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, link_generated, parse_tensor_data, write_file};

/// A foldable template, so the draw routes to the tensor-DAG lane.
fn dag_lane_program(low: &str, high: &str) -> String {
    format!(
        "template = to_tensor([cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), \
         cast(0.5, f32)])\n\
         sampled = with seed(42i64) \
         {{ uniform_like(copy(template), {low}, {high}) }}\n"
    )
}

/// A runtime-derived template, so the draw routes to the C host lane. This one
/// only became reachable once chelis#2320 added the host-lane arm; before that
/// the host lane refused to build and only the DAG lane could be observed.
fn host_lane_program(low: &str, high: &str) -> String {
    format!(
        "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
         sampled = with seed(42i64) \
         {{ uniform_like(bc(cast(0.5, f32)), {low}, {high}) }}\n"
    )
}

fn eval_sampled(program: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "chelis eval failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    parse_tensor_data(&String::from_utf8_lossy(&out.stdout), "sampled")
}

fn c_sampled(program: &str, name: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link of generated C failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary exited non-zero: {}",
        String::from_utf8_lossy(&run.stderr),
    );
    parse_tensor_data(&String::from_utf8_lossy(&run.stdout), "sampled")
}

fn assert_f32_bit_parity(eval: &[f64], c: &[f64], context: &str) {
    assert_eq!(
        eval.len(),
        c.len(),
        "[{context}] element count mismatch: eval {} vs C {}",
        eval.len(),
        c.len(),
    );
    assert!(!eval.is_empty(), "[{context}] no elements parsed");
    for (i, (e, cv)) in eval.iter().zip(c.iter()).enumerate() {
        assert_eq!(
            (*e as f32).to_bits(),
            (*cv as f32).to_bits(),
            "[{context}] elem[{i}] f32 bits diverge: eval {} ({:#010x}) vs C {} ({:#010x})",
            e,
            (*e as f32).to_bits(),
            cv,
            (*cv as f32).to_bits(),
        );
    }
}

/// Run both lanes for one bound spelling and assert each matches `eval`.
fn assert_both_lanes_match_eval(low: &str, high: &str, tag: &str) {
    if !gcc_available() {
        eprintln!("skipping: no C compiler available");
        return;
    }
    let dag = dag_lane_program(low, high);
    let host = host_lane_program(low, high);
    assert_f32_bit_parity(
        &eval_sampled(&dag),
        &c_sampled(&dag, &format!("{tag}_dag")),
        &format!("{tag}/dag-lane"),
    );
    assert_f32_bit_parity(
        &eval_sampled(&host),
        &c_sampled(&host, &format!("{tag}_host")),
        &format!("{tag}/host-lane"),
    );
}

/// NEGATIVE: the issue's exact spelling. Before the fix, all four elements
/// diverged in both lanes.
#[test]
fn f16_intermediate_cast_bounds_agree_with_eval_in_both_lanes() {
    assert_both_lanes_match_eval(
        "cast(cast(0.30000001, f16), f32)",
        "cast(cast(0.90000001, f16), f32)",
        "f16",
    );
}

/// NEGATIVE: bf16 has fewer mantissa bits than f16, so it diverged furthest.
/// `eval` gave `0.69147855...` where both compiled lanes gave `0.69222945...`.
#[test]
fn bf16_intermediate_cast_bounds_agree_with_eval_in_both_lanes() {
    assert_both_lanes_match_eval(
        "cast(cast(0.30000001, bf16), f32)",
        "cast(cast(0.90000001, bf16), f32)",
        "bf16",
    );
}

/// NEGATIVE: a negative low bound exercises the `neg` arm composed with the
/// `cast` arm, and moves the affine's offset rather than only its span.
#[test]
fn negative_intermediate_cast_bound_agrees_with_eval_in_both_lanes() {
    assert_both_lanes_match_eval(
        "cast(cast(-0.7, f16), f32)",
        "cast(cast(0.9, f16), f32)",
        "neg",
    );
}

/// NEGATIVE: a deeper chain. The admitted shape is "any float-target cast
/// chain over a resolvable value, at ANY depth" — `is_static_numeric_bound`
/// checks only that each target is a float and recurses — so the fold must
/// apply every rounding in the chain, not just the outermost.
#[test]
fn deep_cast_chain_applies_every_rounding_in_the_chain() {
    assert_both_lanes_match_eval(
        "cast(cast(cast(0.30000001, f16), f64), f32)",
        "cast(cast(cast(0.90000001, bf16), f64), f32)",
        "deep",
    );
}

/// DISPOSITION LOCK: a lone widening cast agreed before the fix and must keep
/// agreeing. If this ever fails, the fix broke the path that was already
/// correct.
#[test]
fn single_widening_cast_bound_still_agrees_with_eval() {
    assert_both_lanes_match_eval(
        "cast(0.30000001234567, f32)",
        "cast(0.90000001234567, f32)",
        "widening",
    );
}

/// DISPOSITION LOCK: bare literal bounds, the spelling every earlier probe
/// used. Green before and after.
#[test]
fn bare_literal_bounds_still_agree_with_eval() {
    assert_both_lanes_match_eval("2.0f32", "5.0f32", "bare");
}

/// NEGATIVE PARITY (chelis#776): an integer-target cast bound never reaches
/// the fold, because `uniform_like`'s bounds are typed `f32` and the checker
/// rejects the mismatch first. This asserts the rejection AND its reason.
///
/// An earlier version of this test used `cast(2.7, int32)` and asserted only a
/// non-zero exit. That was vacuous twice over: `int32` is a retired v0.18
/// spelling that dies in the lexer, and the canonical `i32` spelling is
/// rejected by the type checker before `extract_f64_value` ever sees it. It
/// locked nothing. The integer-target `None` path in the fold is covered where
/// it can actually be reached, by `integer_and_bool_targets_are_unresolved`
/// beside `round_float_bound` in `chelis-types`.
#[test]
fn integer_target_cast_bound_is_rejected_by_the_checker_before_the_fold() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("int_target.ch");
    write_file(&path, &dag_lane_program("cast(2.7, i32)", "cast(5.2, i32)"));
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("i32") && (text.contains("f32") || text.contains("recision")),
        "an integer-target bound must be rejected as a dtype mismatch against \
         the f32 bound type, not folded; got:\n{text}",
    );
}

/// NEGATIVE PARITY (chelis#776): a genuinely runtime-computed bound is the
/// checker's reject case and must stay rejected in both lanes.
#[test]
fn runtime_computed_bound_is_still_rejected() {
    let program = "lo = cast(0.3, f32)\n\
                   template = to_tensor([cast(0.5, f32), cast(0.5, f32), \
                   cast(0.5, f32), cast(0.5, f32)])\n\
                   sampled = with seed(42i64) \
                   { uniform_like(copy(template), cast(lo, f32), 0.9f32) }\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("runtime_bound.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        !out.status.success(),
        "a runtime-computed bound must stay rejected (chelis#776)",
    );
}

/// REGRESSION, chelis#2316 round 2: a literal carrying an explicit narrower
/// SUFFIX inside a further-narrowing cast.
///
/// The first revision of this fix taught the `cast` arm to honour its target
/// but left `extract_f64_value`'s `Lit` arm returning the raw `Atom::Float`,
/// ignoring the literal's own declared dtype. `eval` finalizes the literal at
/// `f32` and then rounds to `f16` — two roundings; the fold went `f64 -> f16`
/// in one. The C host lane was unaffected, because `HostExprKind::Float`
/// already holds the f32-finalized value.
///
/// So the first revision turned a defect both compiled lanes shared into a
/// LANE SPLIT: DAG baked `0x3c802000`, host and eval agreed on `0x3c800000`.
/// That is the exact hazard `assign_uniform_like`'s doc comment names —
/// correcting one lane alone makes template foldability observable again,
/// which is what chelis#2120 exists to remove.
///
/// An unsuffixed literal cannot witness this: inside `cast(·, f16)` it is
/// inferred as `f16`, which makes the raw value coincidentally correct. The
/// suffix is required.
#[test]
fn a_suffixed_literal_inside_a_narrowing_cast_agrees_across_lanes() {
    assert_both_lanes_match_eval(
        "cast(cast(0.015632629860192537f32, f16), f32)",
        "cast(cast(0.9, f16), f32)",
        "suffixed",
    );
}

/// A second witness for the same class, so the fix is not pinned to one
/// value's rounding accident.
#[test]
fn a_second_suffixed_literal_witness_agrees_across_lanes() {
    assert_both_lanes_match_eval(
        "cast(cast(0.016761780250817537f32, f16), f32)",
        "cast(cast(0.9, f16), f32)",
        "suffixed2",
    );
}

/// The same shape with the suffix at `f64`, so the literal's declared dtype is
/// WIDER than the chain's first cast rather than narrower. Both orderings must
/// finalize at the literal's own dtype first.
#[test]
fn a_f64_suffixed_literal_inside_a_narrowing_cast_agrees_across_lanes() {
    assert_both_lanes_match_eval(
        "cast(cast(0.015632629860192537f64, f16), f32)",
        "cast(cast(0.9, f16), f32)",
        "suffixed64",
    );
}
