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
fn reconstructed_adt_argument_does_not_claim_lexical_host_support() {
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
    assert!(String::from_utf8_lossy(&eval.stdout).contains("data=[2.0, 4.0]"));
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("reconstructed-out").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!build.status.success());
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(stderr.contains("can't lower these defs"), "{stderr}");
    assert!(stderr.contains("Affected defs: step"), "{stderr}");
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

// spec/06 §2.1: disconnected cotangents retain each leaf's actual ordered
// shape; an ADT declaration's formal dimension does not join its instances.
#[test]
fn nested_adt_nominal_dimensions_preserve_each_disconnected_leaf() {
    for source in [
        r#"type Leaf[n] =
  | Leaf { value: tensor[n, f32] }
type Pair =
  | Pair { first: Leaf[2], second: Leaf[3] }
def zero(p: &Pair) -> f32 = 0.0f32
def step(p: Pair) -> Pair = grad(zero)(p)
p = Pair { first: Leaf { value: to_tensor([1.0f32, 2.0f32]) }, second: Leaf { value: to_tensor([3.0f32, 4.0f32, 5.0f32]) } }
out = step(p)
"#,
        r#"type Leaf[n] =
  | Leaf { value: tensor[n, f32] }
type Pair[a, b] =
  | Pair { first: Leaf[a], second: Leaf[b] }
def zero[a, b](p: &Pair[a, b]) -> f32 = 0.0f32
def step[a, b](p: Pair[a, b]) -> Pair[a, b] = grad(zero)(p)
p = Pair { first: Leaf { value: to_tensor([1.0f32, 2.0f32]) }, second: Leaf { value: to_tensor([3.0f32, 4.0f32, 5.0f32]) } }
out = step(p)
"#,
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested_dimensions.ch");
        write_file(&path, source);
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
            expected.contains("shape=[2], data=[0.0, 0.0]"),
            "{expected}"
        );
        assert!(
            expected.contains("shape=[3], data=[0.0, 0.0, 0.0]"),
            "{expected}"
        );
        if gcc_available() {
            assert_eq!(build_and_run(source, "nested_dimensions"), expected);
        }
    }
}

#[test]
fn repeated_nominal_dimension_still_rejects_contradictory_field_shapes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("contradictory_dimensions.ch");
    write_file(
        &path,
        r#"type Leaf[n] =
  | Leaf { value: tensor[n, f32] }
type Pair[n] =
  | Pair { first: Leaf[n], second: Leaf[n] }
def zero[n](p: &Pair[n]) -> f32 = 0.0f32
def step[n](p: Pair[n]) -> Pair[n] = grad(zero)(p)
p = Pair { first: Leaf { value: to_tensor([1.0f32, 2.0f32]) }, second: Leaf { value: to_tensor([3.0f32, 4.0f32, 5.0f32]) } }
out = step(p)
"#,
    );
    for command in ["check", "eval", "build"] {
        let mut invocation = Command::cargo_bin("chelis").unwrap();
        invocation.arg(command);
        if command == "eval" {
            invocation.arg("--file");
        }
        let output = invocation.arg(&path).output().unwrap();
        assert!(!output.status.success());
        let diagnostics = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(diagnostics.contains("dimension mismatch"), "{diagnostics}");
    }
}

#[test]
fn nested_adt_nominal_dimensions_preserve_nonzero_field_cotangents() {
    let source = r#"type Leaf[n] =
  | Leaf { value: tensor[n, f32] }
type Pair =
  | Pair { first: Leaf[2], second: Leaf[3] }
def loss(p: &Pair) -> f32 = {
  a = tensor_to_scalar(sum(mul(p.first.value, p.first.value), 0i32))
  b = tensor_to_scalar(sum(mul(p.second.value, p.second.value), 0i32))
  add(a, b)
}
def step(p: Pair) -> Pair = grad(loss)(p)
p = Pair { first: Leaf { value: to_tensor([1.0f32, 2.0f32]) }, second: Leaf { value: to_tensor([3.0f32, 4.0f32, 5.0f32]) } }
out = step(p)
"#;
    let dir = tempdir().unwrap();
    let path = dir.path().join("nonzero_dimensions.ch");
    write_file(&path, source);
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
        expected.contains("shape=[2], data=[2.0, 4.0]"),
        "{expected}"
    );
    assert!(
        expected.contains("shape=[3], data=[6.0, 8.0, 10.0]"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(build_and_run(source, "nonzero_dimensions"), expected);
    }
}
