//! C and evaluator parity for gradients of fixed, ADT-valued model parameters.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use tempfile::tempdir;

const MODEL: &str = r#"type Block =
  | Block { weight: tensor[2, f32] }
type Model =
  | Model { first: Block, second: Block }
def loss(p: Model) -> f32 = {
  a = mul(p.first.weight, p.first.weight)
  b = mul(p.second.weight, p.second.weight)
  tensor_to_scalar(sum(add(a, b), 0i32))
}
def value_then_grad(p: Model) -> (f32, Model) = {
  value = loss(p)
  gradient = grad(loss)(p)
  (value, gradient)
}
def grad_then_value(p: Model) -> (f32, Model) = {
  gradient = grad(loss)(p)
  value = loss(p)
  (value, gradient)
}
params = Model { first: Block { weight: to_tensor([1.0f32, 2.0f32]) }, second: Block { weight: to_tensor([3.0f32, 4.0f32]) } }
left = value_then_grad(params)
right = grad_then_value(params)
"#;

const SUM_TYPE: &str = r#"type Choice =
  | A { weight: tensor[2, f32] }
  | B { weight: tensor[2, f32] }
def loss(p: Choice) -> f32 =
  match p with {
    | A { weight } => tensor_to_scalar(sum(mul(weight, weight), 0i32))
    | B { weight } => tensor_to_scalar(sum(mul(weight, weight), 0i32))
  }
def step(p: Choice) -> Choice = grad(loss)(p)
out = step(A { weight: to_tensor([1.0f32, 2.0f32]) })
"#;

const MATCH_BOUND_MODEL: &str = r#"type Block =
  | Block { weight: tensor[2, f32] }
type Model =
  | Model { first: Block, second: Block }
type Batch =
  | Batch { x: tensor[2, f32], y: tensor[2, i64] }
type State =
  | State { p: Model }
def loss(p: Model, b: &Batch) -> f32 = {
  a = mul(p.first.weight, b.x)
  c = mul(p.second.weight, b.x)
  tensor_to_scalar(sum(add(a, c), 0i32))
}
def step(s: State, b: &Batch) -> Model =
  match s with {
    | State { p } => {
    value = loss(p, b)
    gradients = grad(loss)(p, b)
    gradients.0
  }
  }
model = Model { first: Block { weight: to_tensor([1.0f32, 2.0f32]) }, second: Block { weight: to_tensor([3.0f32, 4.0f32]) } }
batch = Batch { x: to_tensor([5.0f32, 6.0f32]), y: to_tensor([1i64, 2i64]) }
out = step(State { p: model }, batch)
"#;

#[test]
fn fixed_adt_blocks_differentiate_on_c_in_both_orders() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.ch");
    write_file(&path, MODEL);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let expected = String::from_utf8(eval.stdout).unwrap();
    assert!(expected.contains("data=[2.0, 4.0]"), "{expected}");
    assert!(expected.contains("data=[6.0, 8.0]"), "{expected}");
    if gcc_available() {
        assert_eq!(build_and_run(MODEL, "model"), expected);
    }
}

#[test]
fn match_bound_model_and_borrowed_batch_differentiate_on_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("match_bound_model.ch");
    write_file(&path, MATCH_BOUND_MODEL);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let expected = String::from_utf8(eval.stdout).unwrap();
    assert!(
        expected.contains("out = Model(Block(tensor(shape=[2], data=[5.0, 6.0]))"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(
            build_and_run(MATCH_BOUND_MODEL, "match_bound_model"),
            expected
        );
    }
}

#[test]
fn runtime_sum_type_grad_needs_compiled_tag_dispatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("choice.ch");
    write_file(&path, SUM_TYPE);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert_eq!(
        String::from_utf8(eval.stdout).unwrap(),
        "out = A(tensor(shape=[2], data=[2.0, 4.0]))\n"
    );
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("choice-out").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!build.status.success());
    let stderr = String::from_utf8(build.stderr).unwrap();
    assert!(stderr.contains("runtime ADT `Choice`"), "{stderr}");
    assert!(stderr.contains("#618"), "{stderr}");
}
