//! chelis#3341: a `fail`-guarded function whose body concatenates a
//! runtime-length `List` of tensors must evaluate, and its guards must still
//! fire.
//!
//! A runtime `if` inside a staged host region keeps host control flow, so the
//! definition runs in the host interpreter, which executes the computed
//! concat (spec/04 §4.5.4 rule 3). A `fail` arm is a guarded abort only inside
//! a transform; outside one it must not let the staged region lower past the
//! branch into a concat the static DAG cannot carry.

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

fn program(call: &str, expected: &str) -> String {
    format!(
        "module Demo.Tests.Probe
import Std.Test (assert_close_tensor)
def windows[n, m](x: &tensor[n, f32], kernel: i64, step: i64, m: i64, k: i64) -> List[tensor[1, m, f32]] =
  if gte(k, kernel) then [] else {{
    row = window_row(x, step, m, k)
    rest = windows(x, kernel, step, m, add(k, 1i64))
    concat([row], rest)
  }}
def window_row[n, m](x: &tensor[n, f32], step: i64, m: i64, k: i64) -> tensor[1, m, f32] = {{
  extent = add(add(k, mul(sub(m, 1i64), step)), 1i64)
  sliced = shrink(x, [[k, extent]])
  strided = stride(sliced, step)
  reshape(strided, [1i64, m])
}}
sig avgpool[n, m]: tensor[n, f32] -> i64 -> i64 -> tensor[m, f32]
def avgpool(x, kernel, step) =
  if lte(kernel, 0i64) then fail(\"kernel must be positive\") else {{
    n = cast(shape(x, 0i32), i64)
    if gt(kernel, n) then fail(\"kernel exceeds input length\") else {{
      m = add(floor_div(sub(n, kernel), step), 1i64)
      rows = windows(&x, kernel, step, m, 0i64)
      stacked = concat(rows, 0i32)
      mean(stacked, 0i32)
    }}
  }}
def test_avgpool() -> unit ! {{ Test }} = {{
  out = {call}
  assert_close_tensor(out, to_tensor({expected}), 1e-6f32, \"avgpool\")
}}
"
    )
}

fn run(source: &str) -> (bool, String) {
    let (_dir, reef_home, app) = make_app("issue-3341");
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
fn guarded_computed_concat_evaluates() {
    let (ok, text) = run(&program(
        "avgpool(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), 2i64, 1i64)",
        "[1.5f32, 2.5f32, 3.5f32]",
    ));
    assert!(ok, "{text}");
    assert!(text.contains("PASS"), "{text}");
}

#[test]
fn the_first_guard_still_fires() {
    let (ok, text) = run(&program(
        "avgpool(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), 0i64, 1i64)",
        "[1.5f32, 2.5f32, 3.5f32]",
    ));
    assert!(!ok, "{text}");
    assert!(text.contains("kernel must be positive"), "{text}");
}

#[test]
fn the_second_guard_still_fires() {
    let (ok, text) = run(&program(
        "avgpool(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), 5i64, 1i64)",
        "[1.5f32, 2.5f32, 3.5f32]",
    ));
    assert!(!ok, "{text}");
    assert!(text.contains("kernel exceeds input length"), "{text}");
}
