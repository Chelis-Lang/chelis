//! chelis#3340: a generic rank-2 helper that builds a broadcast with two
//! `insert`s sized by `shape(x, axis)`, then multiplies, must evaluate.
//!
//! spec/04 §4.7.2 admits an `insert` size of any provenance, and a declared
//! result extent not proven equal to the size is checked at run time and
//! traps `Domain` on mismatch. The helper's checker dimension reached two
//! nodes of one activation as the parameter's extent and as the runtime
//! `insert` extent; those are equal at run time and must not be refused.
//! The refusal also escaped as a payload the test runner rendered as
//! "test panicked (no message)".

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

fn program(rows: &str, size: &str, expected: &str, count: usize) -> String {
    format!(
        "module Demo.Tests.Probe
import Std.Test (assert_close_tensor)
def full_like_2d[a, b](x: &tensor[a, b, f32], value: f32) -> tensor[a, b, f32] = {{
  d0 = shape(x, 0i32)
  d1 = shape(x, 1i32)
  insert(insert(scalar_to_tensor(value), 0i32, {size}), 1i32, cast(d1, i64))
}}
def scalar_mul_2d[a, b](x: &tensor[a, b, f32], k: f32) -> tensor[a, b, f32] = {{
  k_t = full_like_2d(x, k)
  mul(x, k_t)
}}
def test_scalar_mul_2d() -> unit ! {{ Test }} = {{
  x = to_tensor({rows})
  r = scalar_mul_2d(&x, 0.5f32)
  assert_close_tensor(reshape(r, [{count}i64]), to_tensor({expected}), 1e-6f32, \"mul\")
}}
"
    )
}

fn run(source: &str) -> (bool, String) {
    let (_dir, reef_home, app) = make_app("issue-3340");
    write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\ndef main(x: tensor[2, f32]) -> tensor[2, f32] = relu(x)\n",
    );
    write_file(&app.join("tests/probe.ch"), source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(&app)
        .args(["test", "tests/probe.ch"])
        .output()
        .expect("run chelis test");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

#[test]
fn square_helper_broadcast_evaluates() {
    let (ok, text) = run(&program(
        "[[1.0f32, 2.0f32], [3.0f32, 4.0f32]]",
        "cast(d0, i64)",
        "[0.5f32, 1.0f32, 1.5f32, 2.0f32]",
        4,
    ));
    assert!(ok, "{text}");
    assert!(text.contains("PASS"), "{text}");
}

#[test]
fn rectangular_helper_broadcast_evaluates() {
    let (ok, text) = run(&program(
        "[[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]",
        "cast(d0, i64)",
        "[0.5f32, 1.0f32, 1.5f32, 2.0f32, 2.5f32, 3.0f32]",
        6,
    ));
    assert!(ok, "{text}");
    assert!(text.contains("PASS"), "{text}");
}

#[test]
fn an_unequal_runtime_insert_extent_traps_domain() {
    let (ok, text) = run(&program(
        "[[1.0f32, 2.0f32], [3.0f32, 4.0f32]]",
        "add(cast(d0, i64), 1i64)",
        "[0.5f32, 1.0f32, 1.5f32, 2.0f32]",
        4,
    ));
    assert!(!ok, "a mismatched claimed extent must not pass:\n{text}");
    assert!(text.contains("numeric trap: domain"), "{text}");
    assert!(!text.contains("no message"), "{text}");
}
