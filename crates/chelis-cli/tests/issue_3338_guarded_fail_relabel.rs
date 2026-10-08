//! chelis#3338: `grad` through a function that takes an ADT parameter and
//! guards its body with `fail` must differentiate the untaken guard and abort
//! with the authored message when the guard fires.
//!
//! [05-OP-68] gives the guarded abort exactly its fallback's shape and dtype.
//! The ADT field's named dimension `d` reached the guard as a caller-side
//! label, and stamping that label onto the guard alone left it disagreeing
//! with its fallback, so the backward DAG failed verification. A label is a
//! view, not a new extent: the program is valid under spec/06 §2.10.1.

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

fn program(eps: &str) -> String {
    format!(
        "module Demo.Tests.Probe
import Std.Test (assert_close_tensor)
type P =
  | P {{ gamma: tensor[d, f32] }}
def f[n](x: tensor[n, f32], params: P, eps: f32) =
  match params with {{
    | P {{ gamma }} => if lte(eps, 0.0f32) then fail(\"eps must be positive\") else mul(x, gamma)
  }}
def shim(x: tensor[2, f32], params: P, eps: f32) -> f32 = tensor_to_scalar(sum(f(x, params, eps), 0i32))
def test_adt_guard_grad() -> unit ! {{ Test }} = {{
  g = grad(shim)(to_tensor([1.0f32, 3.0f32]), P {{ gamma: to_tensor([2.0f32, 5.0f32]) }}, {eps})
  match g.1 with {{
    | P {{ gamma: gg }} => assert_close_tensor(gg, to_tensor([1.0f32, 3.0f32]), 0.001f32, \"dgamma = x\")
  }}
}}
"
    )
}

fn run(eps: &str) -> (bool, String) {
    let (_dir, reef_home, app) = make_app("issue-3338");
    write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\ndef main(x: tensor[2, f32]) -> tensor[2, f32] = relu(x)\n",
    );
    write_file(&app.join("tests/probe.ch"), &program(eps));
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
fn untaken_guard_over_an_adt_field_differentiates() {
    let (ok, text) = run("0.5f32");
    assert!(ok, "the untaken guard must differentiate:\n{text}");
    assert!(
        text.contains("test_adt_guard_grad") && text.contains("PASS"),
        "{text}"
    );
}

#[test]
fn taken_guard_over_an_adt_field_aborts_with_its_message() {
    let (ok, text) = run("-1.0f32");
    assert!(!ok, "a taken guard must fail the test:\n{text}");
    assert!(text.contains("eps must be positive"), "{text}");
    assert!(
        !text.contains("must have exactly its fallback's type"),
        "{text}"
    );
}
