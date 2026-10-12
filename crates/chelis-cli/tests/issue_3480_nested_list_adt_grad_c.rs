//! C and evaluator parity for gradients of List fields nested in model ADTs.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use tempfile::tempdir;

const MODEL: &str = r#"module Repro.NestedListAdtGradient
import Std.Index (list_index)
type Block =
  | Block { w: f32 }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = {
  first = list_index(p.blocks, 0i64)
  second = list_index(p.blocks, 1i64)
  add(mul(first.w, first.w), mul(3.0f32, mul(second.w, second.w)))
}
def step(p: Model) -> Model = grad(loss)(p)
two = step(Model { blocks: [Block { w: 3.0f32 }, Block { w: 4.0f32 }] })
three = step(Model { blocks: [Block { w: 5.0f32 }, Block { w: 6.0f32 }, Block { w: 7.0f32 }] })
"#;

const TOO_SHORT: &str = r#"module Repro.NestedListAdtGradientShort
import Std.Index (list_index)
type Block =
  | Block { w: f32 }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = {
  first = list_index(p.blocks, 0i64)
  second = list_index(p.blocks, 1i64)
  add(mul(first.w, first.w), mul(second.w, second.w))
}
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [Block { w: 3.0f32 }] })
"#;

const TENSOR_FIELDS: &str = r#"module Repro.NestedListAdtTensorGradient
import Std.Index (list_index)
type Block =
  | Block { w: tensor[2, f32], bias: f32 }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = {
  first = list_index(p.blocks, 0i64)
  second = list_index(p.blocks, 1i64)
  a = tensor_to_scalar(sum(mul(first.w, first.w), 0i32))
  b = tensor_to_scalar(sum(mul(second.w, second.w), 0i32))
  add(add(a, b), mul(first.bias, first.bias))
}
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [Block { w: to_tensor([1.0f32, 2.0f32]), bias: 3.0f32 }, Block { w: to_tensor([4.0f32, 5.0f32]), bias: 6.0f32 }, Block { w: to_tensor([7.0f32, 8.0f32]), bias: 9.0f32 }] })
"#;

const EMPTY_AND_DISCONNECTED: &str = r#"module Repro.NestedListAdtEmptyGradient
type Block =
  | Block { w: f32 }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = 0.0f32
def step(p: Model) -> Model = grad(loss)(p)
empty = step(Model { blocks: [] })
nonempty = step(Model { blocks: [Block { w: 3.0f32 }, Block { w: 4.0f32 }] })
"#;

const EMPTY_TENSOR_COLUMN: &str = r#"module Repro.NestedListAdtEmptyTensorGradient
type Block =
  | Block { w: tensor[2, f32] }
type Model =
  | Model { blocks: List[Block], scale: f32 }
def loss(p: Model) -> f32 = mul(p.scale, p.scale)
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [], scale: 3.0f32 })
"#;

const EMPTY_TENSOR_COLUMN_INDEX: &str = r#"module Repro.NestedListAdtEmptyTensorIndex
import Std.Index (list_index)
type Block =
  | Block { w: tensor[2, f32] }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = {
  first = list_index(p.blocks, 0i64)
  tensor_to_scalar(sum(first.w, 0i32))
}
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [] })
"#;

const EMPTY_UNWITNESSED_TENSOR_COLUMN: &str = r#"module Repro.NestedListAdtEmptyUnwitnessedTensor
type Block =
  | Block { w: tensor[*, f32] }
type Model =
  | Model { blocks: List[Block], scale: f32 }
def loss(p: Model) -> f32 = mul(p.scale, p.scale)
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [], scale: 3.0f32 })
"#;

const EMPTY_SYMBOLIC_TENSOR_COLUMN: &str = r#"module Repro.NestedListAdtEmptySymbolicTensor
type Block[n] =
  | Block { w: tensor[n, f32] }
type Model[n] =
  | Model { blocks: List[Block[n]], reference: tensor[n, f32] }
def loss[n](p: Model[n]) -> f32 = tensor_to_scalar(sum(mul(p.reference, p.reference), 0i32))
def step[n](p: Model[n]) -> Model[n] = grad(loss)(p)
out = step(Model { blocks: [], reference: to_tensor([3.0f32, 4.0f32]) })
"#;

#[test]
fn nested_list_adt_gradient_preserves_runtime_length_order_and_zero_elements() {
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
    assert!(
        expected.contains("two = Model([Block(6.0), Block(24.0)])"),
        "{expected}"
    );
    assert!(
        expected.contains("three = Model([Block(10.0), Block(36.0), Block(0.0)])"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(build_and_run(MODEL, "model"), expected);
    }
}

#[test]
fn nested_list_adt_gradient_keeps_index_bounds_failure() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("short.ch");
    write_file(&path, TOO_SHORT);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!eval.status.success());
    let eval_error = String::from_utf8_lossy(&eval.stderr);
    assert!(eval_error.contains("index"), "{eval_error}");

    if gcc_available() {
        let output = dir.path().join("short-out");
        let build = Command::cargo_bin("chelis")
            .unwrap()
            .args([
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = std::process::Command::new(output.join("short"))
            .output()
            .unwrap();
        assert!(!run.status.success());
        let run_error = String::from_utf8_lossy(&run.stderr);
        assert!(run_error.contains("index"), "{run_error}");
    }
}

#[test]
fn nested_list_adt_tensor_columns_preserve_each_field_and_unused_rows() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("tensor_model.ch");
    write_file(&path, TENSOR_FIELDS);
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
        expected.contains("Block(tensor(shape=[2], data=[2.0, 4.0]), 6.0)"),
        "{expected}"
    );
    assert!(
        expected.contains("Block(tensor(shape=[2], data=[8.0, 10.0]), 0.0)"),
        "{expected}"
    );
    assert!(
        expected.contains("Block(tensor(shape=[2], data=[0.0, 0.0]), 0.0)"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(build_and_run(TENSOR_FIELDS, "tensor_model"), expected);
    }
}

#[test]
fn nested_list_adt_gradient_preserves_empty_and_disconnected_lengths() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.ch");
    write_file(&path, EMPTY_AND_DISCONNECTED);
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
    assert!(expected.contains("empty = Model([])"), "{expected}");
    assert!(
        expected.contains("nonempty = Model([Block(0.0), Block(0.0)])"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(build_and_run(EMPTY_AND_DISCONNECTED, "empty"), expected);
    }
}

#[test]
fn empty_tensor_column_keeps_declared_rank_and_other_gradients() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_tensor.ch");
    write_file(&path, EMPTY_TENSOR_COLUMN);
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
    assert!(expected.contains("out = Model([], 6.0)"), "{expected}");
    if gcc_available() {
        assert_eq!(build_and_run(EMPTY_TENSOR_COLUMN, "empty_tensor"), expected);
    }
}

#[test]
fn empty_tensor_column_keeps_index_bounds_failure() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_tensor_index.ch");
    write_file(&path, EMPTY_TENSOR_COLUMN_INDEX);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!eval.status.success());
    assert!(String::from_utf8_lossy(&eval.stderr).contains("index"));
    if gcc_available() {
        let output = dir.path().join("empty_tensor_index-out");
        let build = Command::cargo_bin("chelis")
            .unwrap()
            .args([
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = std::process::Command::new(output.join("empty_tensor_index"))
            .output()
            .unwrap();
        assert!(!run.status.success());
        assert!(String::from_utf8_lossy(&run.stderr).contains("index"));
    }
}

#[test]
fn empty_tensor_column_requires_a_trailing_extent_witness() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_unwitnessed.ch");
    write_file(&path, EMPTY_UNWITNESSED_TENSOR_COLUMN);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(eval.status.success());
    if gcc_available() {
        let output = dir.path().join("empty_unwitnessed-out");
        let build = Command::cargo_bin("chelis")
            .unwrap()
            .args([
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = std::process::Command::new(output.join("empty_unwitnessed"))
            .output()
            .unwrap();
        assert!(!run.status.success());
        assert!(
            String::from_utf8_lossy(&run.stderr)
                .contains("cannot infer trailing extents for an empty List tensor column"),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

#[test]
fn empty_tensor_column_reads_checked_symbolic_extent_witness() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_symbolic.ch");
    write_file(&path, EMPTY_SYMBOLIC_TENSOR_COLUMN);
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
        expected.contains("Model([], tensor(shape=[2], data=[6.0, 8.0]))"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(
            build_and_run(EMPTY_SYMBOLIC_TENSOR_COLUMN, "empty_symbolic"),
            expected
        );
    }
}
