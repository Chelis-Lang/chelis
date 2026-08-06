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
