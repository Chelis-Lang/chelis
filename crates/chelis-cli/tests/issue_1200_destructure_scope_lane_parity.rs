//! Chelis-Lang/chelis#1200 — executable acceptance for the three
//! block-poisoning shapes: they must reach the back end and agree across
//! lanes, not merely survive the linearity checker.
//!
//! `crates/chelis-types/tests/issue_1200_destructure_component_scope.rs`
//! pins the checker verdicts (including the true positives that must keep
//! failing). This file is the end-to-end half: each shape is evaluated by
//! the interpreter AND compiled to C, linked, and run, and the two lanes
//! must produce the same tensor. Before the fix each shape failed the
//! front end outright with `variable ... (from a destructured binding) was
//! already consumed`, so neither lane produced a value at all.
//!
//! The shapes, from the issue:
//!
//! * **B** — `_ = eat(v)` then `eat(v)`. The filed shape: a bare `_`
//!   discard binds nothing, yet poisoned the rest of the block.
//! * **C** — a discard on an UNRELATED value between two consuming uses
//!   of an ordinary binding.
//! * **D** — a fully NAMED tuple destructure, no `_` anywhere, poisoning
//!   the destructure's own source.
//!
//! The rest of the file pins the review pass on that fix (the addendum to
//! the plan, and the reviewer's rulings on the issue), in two groups:
//!
//! * shapes stock 0.18.4 rejected and that are ACCEPTED again — a
//!   component double-consumed inside a closure body, a match arm, an
//!   `if` branch, or under a re-bound pattern binder;
//! * shapes that were SILENTLY ACCEPTED and are now compile errors —
//!   consumed in a branch then again after the join, captured then reused,
//!   and an outer double consume hidden by a nested destructure's
//!   colliding `__chelis_tmpN`. Plus a determinism guard on
//!   closure-capture classification, which used to flip run to run.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, parse_tensor_data, write_file};
use tempfile::tempdir;

/// The fixture sources below are written as raw strings that open with a
/// newline for readability; strip it so the `.ch` file is canonically
/// formatted and the style gate stays meaningful.
fn fixture(source: &str) -> String {
    source.trim_start_matches('\n').to_string()
}

/// Evaluate `source` through the interpreter lane and return the `out`
/// tensor's elements. The style gate is deliberately left ENABLED: these
/// fixtures are canonical Surf, so a formatting regression should surface
/// here rather than be masked.
fn eval_out(source: &str, name: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed for `{name}`: {}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    parse_tensor_data(&stdout, "out")
}

/// Run `source` through both lanes and assert they agree with `expected`.
fn assert_lane_parity(source: &str, name: &str, expected: &[f64]) {
    let source = fixture(source);
    let eval = eval_out(&source, name);
    assert_eq!(
        eval.len(),
        expected.len(),
        "{name}: unexpected eval arity: {eval:?}"
    );
    for (i, (a, e)) in eval.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "{name}: eval element {i}: actual={a} expected={e}"
        );
    }

    if !gcc_available() {
        eprintln!("skipping build lane for `{name}`: c compiler not available");
        return;
    }
    let stdout = build_and_run(&source, name);
    let built = parse_tensor_data(&stdout, "out");
    assert_eq!(
        built, eval,
        "{name}: build lane disagrees with eval lane\nfull stdout:\n{stdout}"
    );
}

/// Shape B: `_ = realize(v)` then `realize(v)`. The discard's synthesized
/// temp is destructure-marked; `v` is not a component of anything.
#[test]
fn shape_b_wildcard_discard_then_reuse_runs_in_both_lanes() {
    assert_lane_parity(
        r#"
def run(v: tensor[4, f32]) -> tensor[4, f32] = {
  _ = realize(v)
  realize(v)
}
out = run(to_tensor([1.0, 2.0, 3.0, 4.0], f32))
"#,
        "issue1200_shape_b",
        &[1.0, 2.0, 3.0, 4.0],
    );
}

/// Shape C: the discard names `q`, but `p` — an ordinary binding with no
/// relationship to any destructure — was the variable the error fired on.
#[test]
fn shape_c_unrelated_discard_does_not_poison_the_block_in_both_lanes() {
    assert_lane_parity(
        r#"
def run(p: tensor[4, f32], q: tensor[4, f32]) -> tensor[4, f32] = {
  r1: tensor[4, f32] = realize(p)
  _ = realize(q)
  r2: tensor[4, f32] = realize(p)
  add(r1, r2)
}
out = run(to_tensor([1.0, 2.0, 3.0, 4.0], f32), to_tensor([9.0, 9.0, 9.0, 9.0], f32))
"#,
        "issue1200_shape_c",
        &[2.0, 4.0, 6.0, 8.0],
    );
}

/// Shape D: no wildcard anywhere. A fully named tuple destructure whose
/// RHS consumes `v`, then a later `realize(v)`.
#[test]
fn shape_d_named_destructure_source_reuse_runs_in_both_lanes() {
    assert_lane_parity(
        r#"
def run(v: tensor[4, f32], w: tensor[4, f32]) -> tensor[4, f32] = {
  (a, b) = (realize(v), realize(w))
  r: tensor[4, f32] = realize(v)
  add(add(a, b), r)
}
out = run(to_tensor([1.0, 2.0, 3.0, 4.0], f32), to_tensor([10.0, 20.0, 30.0, 40.0], f32))
"#,
        "issue1200_shape_d",
        &[12.0, 24.0, 36.0, 48.0],
    );
}

// ============================================================
// Restored-0.18.3 deltas, pinned
//
// Every shape below double-consumes a genuine destructured component and
// is ACCEPTED. That is the reviewer's Q1 ruling on chelis#1200: "match
// arms should work like closures. New declarations shadow the destructure
// mark."
//
// A closure body got this for free — `check_fn` `declare`s every capture
// in the closure's own scope, and `declare` pushes a fresh UNMARKED entry
// over the clone, the same shadowing the ordinary-rebinding test in
// `crates/chelis-types/tests/issue_1200_destructure_component_scope.rs`
// pins. Branch scopes clone WITHOUT re-declaring the outer names an arm
// merely mentions, so a component double-consumed in an arm rejected
// while the identical closure body compiled. `check_if` / `check_match`
// now drop the inherited marks on entry
// (`LinearScope::clear_destructured_marks`), which makes the three agree.
// Marks are all that is dropped: a destructure authored INSIDE the arm
// still gates its own components, and `join_branch_states` still carries
// the arm's consume out to the enclosing scope.
//
// Stock 0.18.4 rejected all of these, because `destructure_scope_depth`
// lived on `Checker` rather than on the scope: it survived every scope
// clone and kept gating lexically. That rejection was collateral from the
// over-broad gate, not a contract — 0.18.3 accepted them and produced
// these values. Verified across all three binaries while closing
// chelis#1200.
// ============================================================

/// A destructured component captured by a closure whose body consumes it
/// twice. Accepted, value restored to the 0.18.3 answer.
#[test]
fn closure_captured_component_double_consume_is_accepted_in_both_lanes() {
    assert_lane_parity(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  f = fn () -> add(realize(p), realize(p))
  f()
}
out = run()
"#,
        "issue1200_closure_capture",
        &[2.0, 4.0],
    );
}

/// A match-arm binder consumed twice inside the arm, where the scrutinee
/// was built from a destructured component. Accepted, value restored to
/// the 0.18.3 answer.
#[test]
fn match_arm_binder_double_consume_is_accepted_in_both_lanes() {
    assert_lane_parity(
        r#"
type Box =
  | Wrap(tensor[2, f32])
  | Empty
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  b = Wrap(p)
  match b with {
    | Wrap(w) => add(realize(w), realize(w))
    | Empty => to_tensor([0.0f32, 0.0f32])
  }
}
out = run()
"#,
        "issue1200_match_arm_binder",
        &[2.0, 4.0],
    );
}

/// The true positive still reaches the user as a compile error through the
/// CLI, with the diagnostic unchanged. Pins that the per-name gate did not
/// silently disable Linearity-F2 at the product surface.
#[test]
fn destructured_component_reuse_still_fails_the_cli() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("component_reuse.ch");
    write_file(
        &path,
        &fixture(
            r#"
def run(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  r1: tensor[4, f32] = realize(a)
  realize(a)
}
out = run((to_tensor([1.0, 2.0, 3.0, 4.0], f32), to_tensor([5.0, 6.0, 7.0, 8.0], f32)))
"#,
        ),
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "component reuse without `copy()` must still fail; stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("variable `a` (from a destructured binding)"),
        "expected the unchanged Linearity-F2 diagnostic; got:\n{stderr}"
    );
}

/// Q1, direct form: the component itself — not a re-bound pattern binder
/// — consumed twice inside a match arm. This is the shape that rejected
/// while the closure body above compiled.
#[test]
fn component_double_consume_in_a_match_arm_is_accepted_in_both_lanes() {
    assert_lane_parity(
        r#"
type Flag =
  | On
  | Off
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run(c: Flag) -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  match c with {
    | On => add(realize(p), realize(p))
    | Off => to_tensor([0.0f32, 0.0f32])
  }
}
out = run(On)
"#,
        "issue1200_match_arm_component",
        &[2.0, 4.0],
    );
}

/// Same for an `if` branch: `check_if` clones without re-declaring for
/// the same reason `check_match` does.
#[test]
fn component_double_consume_in_an_if_branch_is_accepted_in_both_lanes() {
    assert_lane_parity(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run(c: bool) -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  if c then add(realize(p), realize(p)) else to_tensor([0.0f32, 0.0f32])
}
out = run(true)
"#,
        "issue1200_if_branch_component",
        &[2.0, 4.0],
    );
}

// ============================================================
// Pass A rejections — shapes that were SILENTLY ACCEPTED
//
// These three reached the back end and produced a value before this pass.
// Each is a real double consume of a destructured component, so each must
// now be a compile error: an accepted program here means F2's own true
// positive is falsifiable.
// ============================================================

/// Assert `source` fails the front end, and that the message names
/// `expect_name` without leaking a desugarer temp.
fn assert_rejects_naming(source: &str, name: &str, expect_name: &str) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, &fixture(source));
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "{name}: expected a compile error; stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains(expect_name),
        "{name}: expected the diagnostic to name {expect_name}; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("__chelis_tmp"),
        "{name}: diagnostic leaks a desugarer temp:\n{stderr}"
    );
}

/// Q2: consumed in one branch, then again after the join. The component's
/// carrier temp is never `Live` — the component's own bind records an
/// `Aliasing` consume on it — so a join gated on `Live` dropped the
/// branch's Structural consume and this compiled.
#[test]
fn component_consumed_in_a_branch_then_after_the_join_fails_the_cli() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run(c: bool) -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  r: tensor[2, f32] = if c then realize(p) else to_tensor([0.0f32, 0.0f32])
  add(r, realize(p))
}
out = run(true)
"#,
        "issue1200_join_consume",
        "variable `p`",
    );
}

/// Capture-then-reuse. The capture consume landed on the component's own
/// entry rather than forwarding to its carrier, so the carrier stayed
/// `Live` and the reuse outside the closure was accepted — while the same
/// reuse without the closure errored.
#[test]
fn component_captured_then_reused_fails_the_cli() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  f = fn () -> realize(p)
  add(f(), realize(p))
}
out = run()
"#,
        "issue1200_capture_reuse",
        "variable `p`",
    );
}

/// Nested-destructure temp collision. The inner block's `__chelis_tmpN`
/// used to shadow the outer's, so the outer component's first consume
/// landed on the inner block's temp and the outer carrier stayed `Live` —
/// the second consume was then silently accepted. Removing the inner
/// destructure (and nothing else) made the identical program error, which
/// is what made this a collision rather than a semantics question.
#[test]
fn nested_destructure_does_not_hide_an_outer_double_consume() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (a, b) = two(v)
  r: tensor[2, f32] = {
    w = to_tensor([3.0f32, 4.0f32])
    (c, d) = two(w)
    realize(a)
  }
  add(r, realize(a))
}
out = run()
"#,
        "issue1200_nested_collision",
        "variable `a`",
    );
}

/// The control for the collision test: the same program without the inner
/// destructure always errored. If this ever stops erroring, the test above
/// is passing for the wrong reason.
#[test]
fn the_nested_collision_control_without_an_inner_destructure_also_fails() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (a, b) = two(v)
  r: tensor[2, f32] = {
    w = to_tensor([3.0f32, 4.0f32])
    realize(a)
  }
  add(r, realize(a))
}
out = run()
"#,
        "issue1200_nested_control",
        "variable `a`",
    );
}

/// Closure-capture classification must be DETERMINISTIC, and the verdict
/// it settles on is REJECTION.
///
/// `check_fn` walks the capture list mutating the outer scope as it goes,
/// so when two captures sit on one alias chain the verdict depends on
/// visit order — and the list came from a `UnordSet`. Measured on stock
/// 0.18.4, this exact program rejected 11 times in 12 and compiled once.
/// `free_vars` now sorts, which is what makes the verdict stable.
///
/// The determinism is the sort's doing ALONE. An earlier version of this
/// comment claimed the capture consume also "forwards through the alias
/// chain so both orders agree anyway" — that is false for the fixture
/// below, and deliberately so: `check_fn` forwards a capture to a carrier
/// only for destructured components, never for an ordinary `y = x` alias
/// (see the carve-out comment on `capture_target` in `linearity.rs`).
/// Here `y = x` is an ordinary alias, so the first capture consumes and
/// the second is a use-after-consume. Asserting "0 or all" let that
/// mistaken reading pass unchallenged; the test now pins the rejection and
/// the name it must blame (chelis#1200 review finding 3).
#[test]
fn closure_capture_verdict_is_stable_across_repeated_runs() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("capture_determinism.ch");
    write_file(
        &path,
        &fixture(
            r#"
def run() -> tensor[2, f32] = {
  x = to_tensor([1.0f32, 2.0f32])
  y: tensor[2, f32] = x
  f = fn () -> add(realize(x), realize(y))
  f()
}
out = run()
"#,
        ),
    );

    let mut accepted = 0usize;
    let mut blamed_y = 0usize;
    const RUNS: usize = 24;
    for _ in 0..RUNS {
        let out = Command::cargo_bin("chelis")
            .expect("binary")
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("chelis eval should run");
        if out.status.success() {
            accepted += 1;
        } else if String::from_utf8_lossy(&out.stderr).contains("variable `y`") {
            blamed_y += 1;
        }
    }
    assert_eq!(
        accepted, 0,
        "the sorted capture order rejects this program; {accepted} of {RUNS} runs accepted"
    );
    assert_eq!(
        blamed_y, RUNS,
        "every run must reject naming `y` (the second capture on the alias chain); \
         {blamed_y} of {RUNS} runs did"
    );
}

/// chelis#1200 review finding 1: a component consumed by a closure capture
/// INSIDE a branch must still be consumed after the join.
///
/// `check_if`/`check_match` clear the region-relative destructured mark on
/// branch entry (Q1). The capture path read that mark to find the carrier,
/// so inside a branch it lost the carrier, consumed the component's own
/// entry, and left the carrier `Consumed(Aliasing)`. The later `realize(p)`
/// then upgraded the carrier to `Structural` with no diagnostic and this
/// program returned `[2.0, 4.0]`. Carrier resolution now asks the permanent
/// `is_component` identity instead.
#[test]
fn component_captured_in_a_branch_then_reused_after_the_join_fails_the_cli() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run(c: bool) -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  r: tensor[2, f32] = if c then {
    f = fn () -> realize(p)
    f()
  } else q
  add(r, realize(p))
}
out = run(true)
"#,
        "issue1200_branch_capture_component",
        "variable `p`",
    );
}

/// chelis#1200 review finding 2, the negative half: the branch join's
/// `Aliasing` -> `Structural` upgrade must NOT reach ordinary aliases.
///
/// `y = x` records the same `Aliasing` shape a component carrier does, for
/// an unrelated reason. Upgrading it made a later BORROW of `y` reject
/// after one branch consumed `x`. Released 0.18.4 accepts this program, so
/// rejecting it would be exactly the ecosystem-breaking tightening
/// chelis#1200 exists to undo.
#[test]
fn ordinary_alias_survives_a_branch_consume_of_its_source_in_both_lanes() {
    assert_lane_parity(
        r#"
def run(c: bool) -> tensor[2, f32] = {
  x = to_tensor([1.0f32, 2.0f32])
  t = to_tensor([3.0f32, 4.0f32])
  y = x
  r: tensor[2, f32] = if c then {
    f = fn () -> realize(x)
    f()
  } else t
  add(r, add(y, y))
}
out = run(false)
"#,
        "issue1200_ordinary_alias_branch_join",
        &[5.0, 8.0],
    );
}

/// chelis#1200 review finding 3: two closures each capturing the SAME
/// component. The first capture consumes the carrier, so the second is a
/// use-after-consume and must be blamed on the component's own name.
#[test]
fn two_closures_capturing_one_component_fails_the_cli() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  f = fn () -> realize(p)
  g = fn () -> realize(p)
  add(add(f(), g()), realize(q))
}
out = run()
"#,
        "issue1200_two_closures_one_component",
        "variable `p`",
    );
}

/// chelis#1200 review finding 3: a component consumed BEFORE a closure
/// captures it. The capture is the second consume and must reject; the
/// ordering is the mirror of `component_captured_then_reused_fails_the_cli`.
#[test]
fn component_consumed_before_a_closure_captures_it_fails_the_cli() {
    assert_rejects_naming(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def run() -> tensor[2, f32] = {
  v = to_tensor([1.0f32, 2.0f32])
  (p, q) = two(v)
  a = realize(p)
  f = fn () -> realize(p)
  add(add(a, f()), realize(q))
}
out = run()
"#,
        "issue1200_component_consumed_before_capture",
        "variable `p`",
    );
}
