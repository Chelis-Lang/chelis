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

const EMPTY_DISCRETE_COLUMN: &str = r#"module Repro.NestedListAdtEmptyDiscreteTensor
type Block =
  | Block { tag: tensor[*, i64] }
type Model =
  | Model { blocks: List[Block], scale: f32 }
def loss(p: Model) -> f32 = mul(p.scale, p.scale)
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [], scale: 3.0f32 })
"#;

const WITNESSED_DISCRETE_COLUMN: &str = r#"module Repro.NestedListAdtWitnessedDiscreteTensor
import Std.Index (list_index)
type Block =
  | Block { tag: tensor[2, i64] }
type Model =
  | Model { blocks: List[Block], scale: f32 }
def loss(p: Model) -> f32 =
  if eq(len(p.blocks), 0i64) then mul(p.scale, p.scale) else {
    first = list_index(p.blocks, 0i64)
    tag_sum = tensor_to_scalar(sum(first.tag, 0i32))
    if eq(tag_sum, 1i64) then mul(p.scale, p.scale) else p.scale
  }
def step(p: Model) -> Model = grad(loss)(p)
empty = step(Model { blocks: [], scale: 3.0f32 })
nonempty = step(Model { blocks: [Block { tag: to_tensor([1i64, 0i64]) }], scale: 3.0f32 })
"#;

const UNWITNESSED_LIVE_DISCRETE_COLUMN: &str = r#"module Repro.NestedListAdtUnwitnessedLiveDiscreteTensor
import Std.Index (list_index)
type Block =
  | Block { tag: tensor[*, i64] }
type Model =
  | Model { blocks: List[Block], scale: f32 }
def loss(p: Model) -> f32 =
  if eq(len(p.blocks), 0i64) then mul(p.scale, p.scale) else {
    first = list_index(p.blocks, 0i64)
    tag_sum = tensor_to_scalar(sum(first.tag, 0i32))
    if eq(tag_sum, 1i64) then mul(p.scale, p.scale) else p.scale
  }
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [], scale: 3.0f32 })
"#;

const NESTED_GUARDED_INDEX: &str = r#"module Repro.NestedGuardGradient
import Std.Index (list_index)
type Block =
  | Block { w: f32 }
type Model =
  | Model { blocks: List[Block], flag: i64, secondary: i64, selected: i64 }
def loss(p: Model) -> f32 =
  if eq(p.flag, 0i64) then 0.0f32 else if eq(p.secondary, 0i64) then {
    bad = list_index(p.blocks, p.selected)
    mul(bad.w, bad.w)
  } else {
    good = list_index(p.blocks, 0i64)
    mul(good.w, good.w)
  }
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [Block { w: 3.0f32 }], flag: 0i64, secondary: 0i64, selected: 3i64 })
"#;

const DYNAMIC_INDEX_OOB: &str = r#"module Repro.DynamicOobGradient
import Std.Index (list_index)
type Block =
  | Block { w: f32 }
type Model =
  | Model { blocks: List[Block], selected: i64 }
def loss(p: Model) -> f32 = {
  item = list_index(p.blocks, p.selected)
  mul(item.w, item.w)
}
def step(p: Model) -> Model = grad(loss)(p)
out = step(Model { blocks: [Block { w: 3.0f32 }], selected: 3i64 })
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
        assert!(!build.status.success());
        assert!(
            String::from_utf8_lossy(&build.stderr)
                .contains("cannot infer trailing extents for a live List tensor column"),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

#[test]
fn empty_discrete_column_without_a_cotangent_needs_no_extent() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_discrete.ch");
    write_file(&path, EMPTY_DISCRETE_COLUMN);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(eval.status.success());
    let expected = String::from_utf8(eval.stdout).unwrap();
    assert!(expected.contains("out = Model([], 6.0)"), "{expected}");
    if gcc_available() {
        assert_eq!(
            build_and_run(EMPTY_DISCRETE_COLUMN, "empty_discrete"),
            expected
        );
    }
}

#[test]
fn live_discrete_column_reads_nonempty_rows_and_preserves_empty_length() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("witnessed_discrete.ch");
    write_file(&path, WITNESSED_DISCRETE_COLUMN);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(eval.status.success());
    let expected = String::from_utf8(eval.stdout).unwrap();
    assert!(expected.contains("empty = Model([], 6.0)"), "{expected}");
    assert!(
        expected.contains("nonempty = Model([Block(())], 6.0)"),
        "{expected}"
    );
    if gcc_available() {
        assert_eq!(
            build_and_run(WITNESSED_DISCRETE_COLUMN, "witnessed_discrete"),
            expected
        );
    }
}

#[test]
fn active_branch_keeps_discrete_column_index_bounds_failure() {
    let dir = tempdir().unwrap();
    let source = WITNESSED_DISCRETE_COLUMN
        .replace("list_index(p.blocks, 0i64)", "list_index(p.blocks, 1i64)")
        .replace("empty = step(Model { blocks: [], scale: 3.0f32 })\n", "");
    let path = dir.path().join("active_index.ch");
    write_file(&path, &source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!eval.status.success());
    assert!(String::from_utf8_lossy(&eval.stderr).contains("index 1 out of bounds"));
    if gcc_available() {
        let output = dir.path().join("active_index-out");
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
        let run = std::process::Command::new(output.join("active_index"))
            .output()
            .unwrap();
        assert!(!run.status.success());
        assert!(String::from_utf8_lossy(&run.stderr).contains("index 1 out of bounds"));
    }
}

#[test]
fn live_discrete_column_without_extent_is_a_typed_c_refusal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("unwitnessed_live_discrete.ch");
    write_file(&path, UNWITNESSED_LIVE_DISCRETE_COLUMN);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(eval.status.success());
    assert!(String::from_utf8_lossy(&eval.stdout).contains("out = Model([], 6.0)"));
    if gcc_available() {
        let output = dir.path().join("unwitnessed_live_discrete-out");
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
        assert!(!build.status.success());
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert!(
            stderr.contains("cannot infer trailing extents for a live List tensor column"),
            "{stderr}"
        );
        assert!(stderr.contains("chelis#3480"), "{stderr}");
    }
}

#[test]
fn nested_inactive_list_index_never_runs_its_private_gather() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested_inactive.ch");
    write_file(&path, NESTED_GUARDED_INDEX);
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
    assert!(expected.contains("out = Model([Block(0.0)]"), "{expected}");
    if gcc_available() {
        assert_eq!(
            build_and_run(NESTED_GUARDED_INDEX, "nested_inactive"),
            expected
        );
    }
}

#[test]
fn nested_active_good_list_index_preserves_the_selected_gradient() {
    let source = NESTED_GUARDED_INDEX.replace(
        "flag: 0i64, secondary: 0i64, selected: 3i64",
        "flag: 1i64, secondary: 1i64, selected: 3i64",
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested_good.ch");
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
    assert!(expected.contains("out = Model([Block(6.0)]"), "{expected}");
    if gcc_available() {
        assert_eq!(build_and_run(&source, "nested_good"), expected);
    }
}

#[test]
fn active_dynamic_list_index_matches_direct_loss_bounds_failure() {
    let direct = DYNAMIC_INDEX_OOB.replace("out = step(", "out = loss(");
    let dir = tempdir().unwrap();
    for (name, source) in [("grad_oob", DYNAMIC_INDEX_OOB), ("direct_oob", &direct)] {
        let path = dir.path().join(format!("{name}.ch"));
        write_file(&path, source);
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            !eval.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&eval.stdout)
        );
        assert!(
            String::from_utf8_lossy(&eval.stderr)
                .contains("index 3 out of bounds for list of len 1"),
            "{name}: {}",
            String::from_utf8_lossy(&eval.stderr)
        );
        if gcc_available() {
            let output = dir.path().join(format!("{name}-out"));
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
                "{name}: {}",
                String::from_utf8_lossy(&build.stderr)
            );
            let run = std::process::Command::new(output.join(name))
                .output()
                .unwrap();
            assert!(
                !run.status.success(),
                "{name}: {}",
                String::from_utf8_lossy(&run.stdout)
            );
            assert!(
                String::from_utf8_lossy(&run.stderr)
                    .contains("index 3 out of bounds for list of len 1"),
                "{name}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
        }
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
