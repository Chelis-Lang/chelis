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

const GENERIC_MODEL: &str = r#"type Block[n] =
  | Block { weight: tensor[n, f32] }
type Model[n] =
  | Model { first: Block[n], second: Block[n] }
def loss[n](p: Model[n]) -> f32 = {
  a = mul(p.first.weight, p.first.weight)
  b = mul(p.second.weight, p.second.weight)
  tensor_to_scalar(sum(add(a, b), 0i32))
}
def step[n](p: Model[n]) -> Model[n] = grad(loss)(p)
params = Model { first: Block { weight: to_tensor([1.0f32, 2.0f32]) }, second: Block { weight: to_tensor([3.0f32, 4.0f32]) } }
out = step(params)
"#;

const RECONSTRUCTED_ARGUMENT: &str = r#"type Block =
  | Block { weight: tensor[2, f32] }
type Model =
  | Model { first: Block, second: Block }
def loss(p: Model) -> f32 = {
  a = mul(p.first.weight, p.first.weight)
  b = mul(p.second.weight, p.second.weight)
  tensor_to_scalar(sum(add(a, b), 0i32))
}
def step(p: Model) -> Model = grad(loss)(Model { first: p.first, second: p.second })
params = Model { first: Block { weight: to_tensor([1.0f32, 2.0f32]) }, second: Block { weight: to_tensor([3.0f32, 4.0f32]) } }
out = step(params)
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
fn generic_fixed_adt_gradient_retains_tensor_result_provenance_on_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("generic_model.ch");
    write_file(&path, GENERIC_MODEL);
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
        assert_eq!(build_and_run(GENERIC_MODEL, "generic_model"), expected);
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

const EFFECTFUL_ACTUALS: &str = r#"type Model =
  | Model { weight: f32 }
def loss(first: Model, second: Model) -> f32 = 1.0f32
def announce(label: string) -> Model ! { IO } = {
  _ = print(label)
  Model { weight: 3.0f32 }
}
def step() -> (Model, Model) ! { IO } = grad(loss)(announce("first"), announce("second"))
out = step()
"#;

#[test]
fn reconstructed_adt_argument_differentiates_on_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("reconstructed.ch");
    write_file(&path, RECONSTRUCTED_ARGUMENT);
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
    assert!(expected.contains("data=[2.0, 4.0]"));
    if gcc_available() {
        assert_eq!(
            build_and_run(RECONSTRUCTED_ARGUMENT, "reconstructed"),
            expected
        );
    }
}

#[test]
fn constructed_adt_actuals_keep_effects_once_in_source_order() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("effectful_actuals.ch");
    write_file(&path, EFFECTFUL_ACTUALS);
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
    assert_eq!(
        expected
            .lines()
            .filter(|line| *line == "first" || *line == "second")
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert!(expected.contains("out.0 = Model(0.0)"), "{expected}");
    assert!(expected.contains("out.1 = Model(0.0)"), "{expected}");
    if gcc_available() {
        assert_eq!(
            build_and_run(EFFECTFUL_ACTUALS, "effectful_actuals"),
            expected
        );
    }
}

#[test]
fn unused_discrete_adt_actual_keeps_its_argument_effect() {
    let source = EFFECTFUL_ACTUALS.replace("def loss(first: Model, second: Model)", "type Tag =\n  | Tag { value: i64 }\ndef loss(first: Model, second: Tag)")
        .replace("def step() -> (Model, Model) ! { IO } = grad(loss)(announce(\"first\"), announce(\"second\"))", "def announce_tag() -> Tag ! { IO } = {\n  _ = print(\"second\")\n  Tag { value: 3i64 }\n}\ndef step() -> Model ! { IO } = grad(loss)(announce(\"first\"), announce_tag())");
    let dir = tempdir().unwrap();
    let path = dir.path().join("effectful_discrete_actual.ch");
    write_file(&path, &source);
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
    assert_eq!(
        expected
            .lines()
            .filter(|line| *line == "first" || *line == "second")
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert!(expected.contains("out = Model(0.0)"), "{expected}");
    if gcc_available() {
        assert_eq!(
            build_and_run(&source, "effectful_discrete_actual"),
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
