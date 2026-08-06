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
//! The last two tests pin the two places where moving from a block-scoped
//! depth counter to a per-name mark restores 0.18.3 behavior that stock
//! 0.18.4 rejected: closure-captured components and match-arm binders.
//! See the comment above them for the mechanism.

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
out = run(to_tensor([1.0, 2.0, 3.0, 4.0]))
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
out = run(to_tensor([1.0, 2.0, 3.0, 4.0]), to_tensor([9.0, 9.0, 9.0, 9.0]))
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
out = run(to_tensor([1.0, 2.0, 3.0, 4.0]), to_tensor([10.0, 20.0, 30.0, 40.0]))
"#,
        "issue1200_shape_d",
        &[12.0, 24.0, 36.0, 48.0],
    );
}

// ============================================================
// Two restored-0.18.3 deltas, pinned
//
// Both shapes below double-consume a genuine destructured component,
// and both are ACCEPTED. That is not the per-name gate leaking: it is
// the pre-existing re-declaration discipline in `check_fn` and
// `check_match`. Each clones the enclosing `LinearScope` and then calls
// `declare` for every captured name / pattern binder, and `declare`
// pushes a fresh UNMARKED entry that shadows whatever the clone carried
// — the same shadowing the ordinary-rebinding test in
// `crates/chelis-types/tests/issue_1200_destructure_component_scope.rs`
// pins. So the component mark does not reach inside a closure body or a
// match arm, and a consume-after-consume there falls through to
// implicit Copy insertion like any other binding.
//
// Stock 0.18.4 rejected both, because `destructure_scope_depth` lived on
// `Checker` rather than on the scope: it survived the clone-and-declare
// that shadows a per-name mark, so it kept gating lexically inside the
// closure/arm. That rejection was collateral from the over-broad gate,
// not a contract — 0.18.3 accepted both and produced these values.
// Verified across all three binaries while closing chelis#1200.
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
out = run((to_tensor([1.0, 2.0, 3.0, 4.0]), to_tensor([5.0, 6.0, 7.0, 8.0])))
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
