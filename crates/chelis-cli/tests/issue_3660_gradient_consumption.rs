//! spec/06 §2.1 returns cotangent values: later consumers must read those
//! values, independently of the primal used to compute them ([05-OBS-3]).

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use tempfile::tempdir;

const CONSUMPTION: &str = include_str!("../../../examples/grad_consumption.ch");

fn assert_consumed_values(output: &str) {
    for expected in [
        "z = Model(tensor(shape=[2], data=[0.0, 0.0]))",
        "a = Model(tensor(shape=[2], data=[2.0, 4.0]))",
        "b = Model(tensor(shape=[2], data=[6.0, 10.0]))",
        "zero_sum = 0.0",
        "active_sum = 6.0",
        "second_sum = 16.0",
        "primal_sum = 3.0",
        "zero_again = 0.0",
        "active_again = 6.0",
    ] {
        assert!(
            output.lines().any(|line| line == expected),
            "missing {expected}:\n{output}"
        );
    }
}

#[test]
fn computed_adt_gradients_are_consumed_as_cotangents() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("consumption.ch");
    write_file(&path, CONSUMPTION);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = String::from_utf8(eval.stdout).unwrap();
    assert_consumed_values(&eval);
    if gcc_available() {
        let native = build_and_run(CONSUMPTION, "consumption");
        assert_consumed_values(&native);
        assert_eq!(native, eval);
    }
}

#[test]
fn aggregate_parameters_and_local_aliases_shadow_global_initializers() {
    let source = r#"type Leaf =
  | Leaf { value: tensor[2, f32] }
type Envelope =
  | Envelope { leaf: Leaf }
type Alias = Envelope
def nested(m: &Alias) -> f32 = tensor_to_scalar(sum(m.leaf.value, 0i32))
def tuple_read(pair: (Leaf, Leaf)) -> f32 = tensor_to_scalar(sum(pair.1.value, 0i32))
def local_read(x: &Envelope) -> f32 = {
  m = x
  tensor_to_scalar(sum(m.leaf.value, 0i32))
}
def inner_binder(m: &Envelope) -> f32 = tensor_to_scalar(sum((fn (m: tensor[2, f32]) -> mul(m, m))(to_tensor([2.0f32, 3.0f32])), 0i32))
m = Envelope { leaf: Leaf { value: to_tensor([1.0f32, 2.0f32]) } }
pair = (Leaf { value: to_tensor([1.0f32, 2.0f32]) }, Leaf { value: to_tensor([3.0f32, 4.0f32]) })
other = Envelope { leaf: Leaf { value: to_tensor([5.0f32, 8.0f32]) } }
nested_sum = nested(other)
tuple_sum = tuple_read((pair.1, other.leaf))
local_sum = local_read(other)
global_sum = nested(m)
inner_sum = inner_binder(other)
"#;
    let dir = tempdir().unwrap();
    let path = dir.path().join("aggregate_scope.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = String::from_utf8(eval.stdout).unwrap();
    for output in std::iter::once(eval.clone())
        .chain(gcc_available().then(|| build_and_run(source, "aggregate_scope")))
    {
        for expected in [
            "nested_sum = 13.0",
            "tuple_sum = 13.0",
            "local_sum = 13.0",
            "global_sum = 3.0",
            "inner_sum = 13.0",
        ] {
            assert!(
                output.lines().any(|line| line == expected),
                "missing {expected}:\n{output}"
            );
        }
        assert_eq!(output, eval);
    }
}

#[test]
fn same_spelled_global_cannot_supply_an_invalid_argument_contract() {
    let source = r#"type Model =
  | Model { value: tensor[2, f32] }
def inspect(m: &Model) -> f32 = tensor_to_scalar(sum(m.value, 0i32))
m = Model { value: to_tensor([1.0f32, 2.0f32]) }
out = inspect(Model { value: to_tensor([3.0f32, 4.0f32, 5.0f32]) })
"#;
    let dir = tempdir().unwrap();
    let path = dir.path().join("invalid.ch");
    write_file(&path, source);
    for name in ["check", "eval", "build"] {
        let mut command = Command::cargo_bin("chelis").unwrap();
        command.env("CHELIS_STYLE_GATE_DISABLE", "1");
        // Use each CLI's actual input surface, then require its typed failure.
        // A matching global must never turn the invalid actual into success.
        if name == "eval" {
            command.args([name, "--file", path.to_str().unwrap()]);
        } else {
            command.args([name, path.to_str().unwrap()]);
        }
        if name == "build" {
            command.args([
                "--target",
                "c",
                "--output",
                dir.path().join("out").to_str().unwrap(),
            ]);
        }
        let output = command.output().unwrap();
        assert!(
            !output.status.success(),
            "{name} accepted the invalid argument"
        );
        let diagnostic = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            diagnostic.contains("DimensionMismatch"),
            "{name}: {diagnostic}"
        );
    }
}

#[test]
fn unused_host_actual_keeps_its_eager_effect_once() {
    let source = r#"def square(xs: List[f32], scale: f32) -> f32 = mul(scale, scale)
out = {
  values: List[f32] = {
    _ = print("argument")
    [17.0f32, 19.0f32]
  }
  grad(square, wrt=scale)(values, 2.0f32)
}
"#;
    let dir = tempdir().unwrap();
    let path = dir.path().join("unused_effect.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = String::from_utf8(eval.stdout).unwrap();
    assert_eq!(eval, "argument\nout = 4.0\n");
    if gcc_available() {
        assert_eq!(build_and_run(source, "unused_effect"), eval);
    }
}

#[test]
fn unused_host_actual_still_traps_before_the_numeric_result() {
    let source = r#"def square(xs: List[f32], scale: f32) -> f32 = mul(scale, scale)
out = {
  values: List[f32] = {
    _ = mul(scalar_to_tensor(2i32), scalar_to_tensor(2000000000i32))
    [17.0f32, 19.0f32]
  }
  grad(square, wrt=scale)(values, 2.0f32)
}
"#;
    let dir = tempdir().unwrap();
    let path = dir.path().join("unused_trap.ch");
    let out = dir.path().join("native");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!eval.status.success());
    assert!(String::from_utf8_lossy(&eval.stderr).contains("numeric trap: overflow in mul at i32"));
    if gcc_available() {
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ])
            .assert()
            .success();
        let native = std::process::Command::new(out.join("unused_trap"))
            .output()
            .unwrap();
        assert!(!native.status.success());
        assert!(
            String::from_utf8_lossy(&native.stderr)
                .contains("numeric trap: overflow in mul at i32")
        );
        assert!(!String::from_utf8_lossy(&native.stdout).contains("out ="));
    }
}
