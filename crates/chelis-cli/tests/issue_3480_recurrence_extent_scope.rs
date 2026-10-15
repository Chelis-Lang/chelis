//! spec/04 sections 4.4 and 4.7: independent generic call activations
//! retain their own extents, including runtime recurrence results in AD.

#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use tempfile::tempdir;

const COLUMN_RECURRENCES: &str = include_str!("../../../examples/recurrence_extent_scope.ch");

fn assert_parity(source: &str, name: &str, expected: &str) {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(String::from_utf8(eval).unwrap(), expected);
    if gcc_available() {
        assert_eq!(build_and_run(source, name), expected);
    }
}

#[test]
fn independent_column_recurrence_extents_survive_grad() {
    assert_parity(
        COLUMN_RECURRENCES,
        "columns",
        "main.0 = 40.0\nmain.1 = Model([5.0, 5.0])\n",
    );
}

#[test]
fn empty_recurrences_take_each_seeds_extent() {
    let source = COLUMN_RECURRENCES.replace("[2.0f32, 3.0f32]", "[]");
    assert_parity(
        &source,
        "empty_columns",
        "main.0 = 15.0\nmain.1 = Model([])\n",
    );
}

const TENSOR_RECURRENCES: &str = r#"module Repro.RangeScope
def stage[n](x: tensor[n, f32], count: i64) -> tensor[n, f32] = fold(fn (acc: tensor[n, f32], i: i64) -> add(acc, acc), x, range(0i64, count))
def loss(x: tensor[2, f32], y: tensor[3, f32], count: i64) -> f32 = {
  first = stage(x, count)
  second = stage(y, count)
  add(tensor_to_scalar(sum(first, 0i32)), tensor_to_scalar(sum(second, 0i32)))
}
def derivative(x: tensor[2, f32], y: tensor[3, f32], count: i64) -> (tensor[2, f32], tensor[3, f32]) = grad(loss)(x, y, count)
def main() -> (f32, (tensor[2, f32], tensor[3, f32])) = {
  x = to_tensor([1.0f32, 2.0f32])
  y = to_tensor([3.0f32, 4.0f32, 5.0f32])
  (loss(x, y, 2i64), derivative(x, y, 2i64))
}
"#;

#[test]
fn independent_tensor_recurrence_extents_survive_grad() {
    assert_parity(
        TENSOR_RECURRENCES,
        "ranges",
        "main.0 = 60.0\nmain.1.0 = tensor(shape=[2], data=[4.0, 4.0])\nmain.1.1 = tensor(shape=[3], data=[4.0, 4.0, 4.0])\n",
    );
}

#[test]
fn wrapped_recurrence_extents_survive_grad() {
    let source = COLUMN_RECURRENCES.replace(
        "def loss(p: Model)",
        "def wrapped[n](x: tensor[n, f32], blocks: List[f32]) -> tensor[n, f32] = stage(x, blocks)\ndef loss(p: Model)",
    ).replace("= stage(to_tensor", "= copy(wrapped(to_tensor")
        .replace("p.blocks)\n", "p.blocks))\n");
    assert_parity(
        &source,
        "wrapped_columns",
        "main.0 = 40.0\nmain.1 = Model([5.0, 5.0])\n",
    );
}

#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

#[test]
fn recurrence_callback_and_caller_result_claims_still_trap() {
    let wildcard = COLUMN_RECURRENCES
        .replace("stage[n]", "stage")
        .replace("x: tensor[n, f32]", "x: tensor[*, f32]")
        .replace("-> tensor[n, f32] = fold", "-> tensor[*, f32] = fold")
        .replace(
            "to_tensor([3.0f32, 4.0f32, 5.0f32])",
            "insert(scalar_to_tensor(3.0f32), 0i32, add(len(p.blocks), 1i64))",
        )
        .replace("main() -> (f32, Model)", "main() -> Model")
        .replace("(loss(p), step(p))", "step(p)");
    let entry = wildcard.replace("acc: tensor[n, f32]", "acc: tensor[2, f32]");
    let result = wildcard
        .replace("acc: tensor[n, f32]", "acc: tensor[*, f32]")
        .replace("-> tensor[*, f32] = fold", "-> tensor[2, f32] = fold")
        .replace("main() -> Model", "main() -> f32")
        .replace("  step(p)\n", "  loss(p)\n");
    for (source, message, op) in [
        (entry, "extent `2`: claimed = 2, acc axis 0 = 3", "load"),
        (result, "extent `2`: claimed = 2, fold axis 0 = 3", "fold"),
    ] {
        for native in [false, true] {
            if native && !gcc_available() {
                continue;
            }
            let (success, output) = result_claims::run(&source, native);
            assert!(!success, "{output}");
            let message = if native && op == "load" {
                "input `acc` axis 0 expected 2, got 3"
            } else {
                message
            };
            assert!(output.contains(message), "{output}");
            assert!(
                output.contains(&format!("numeric trap: domain in {op} at i64")),
                "{output}"
            );
            assert!(!output.contains("main ="), "{output}");
        }
    }
}

#[test]
fn adjoint_shape_comes_from_selected_input_not_forward_result() {
    // Each scalar affects all five cells, independent of the four-element
    // parameter List and the two distinct forward result shapes [2] and [3].
    let source =
        COLUMN_RECURRENCES.replace("[2.0f32, 3.0f32]", "[2.0f32, 3.0f32, 7.0f32, 11.0f32]");
    assert_parity(
        &source,
        "adjoint_target",
        "main.0 = 130.0\nmain.1 = Model([5.0, 5.0, 5.0, 5.0])\n",
    );
}

#[test]
fn symbolic_scan_retains_its_rectangular_state_admission_check() {
    let source = COLUMN_RECURRENCES.replace(
        "= fold(fn (acc: tensor[n, f32], v: f32) -> add(acc, insert(scalar_to_tensor(v), 0i32, shape(acc, 0i32))), x, blocks)",
        "= {\n  states = scan(fn (acc: tensor[n, f32], v: f32) -> add(acc, insert(scalar_to_tensor(v), 0i32, shape(acc, 0i32))), x, blocks)\n  fold(fn (acc: tensor[n, f32], value: tensor[n, f32]) -> add(acc, value), x, states)\n}",
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("scan_columns.ch");
    write_file(&path, &source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--emit-c",
            "-o",
        ])
        .arg(dir.path().join("native"))
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "scan states require a per-step List carrier",
        ))
        .stderr(predicates::str::contains("chelis#3366"));
}

#[test]
fn stage_channel_and_repeated_spatial_extents_stay_independent() {
    let source = COLUMN_RECURRENCES
        .replace("stage[n]", "stage[a, c, s]")
        .replace("tensor[n, f32]", "tensor[a, c, s, s, f32]")
        .replace("insert(scalar_to_tensor(v), 0i32, shape(acc, 0i32))",
            "insert(insert(insert(insert(scalar_to_tensor(v), 0i32, shape(acc, 3i32)), 0i32, shape(acc, 2i32)), 0i32, shape(acc, 1i32)), 0i32, shape(acc, 0i32))")
        .replace("to_tensor([1.0f32, 2.0f32])", "reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [1i64, 1i64, 2i64, 2i64])")
        .replace("to_tensor([3.0f32, 4.0f32, 5.0f32])", "reshape(to_tensor([5.0f32, 6.0f32]), [1i64, 2i64, 1i64, 1i64])")
        .replace("sum(first, 0i32)", "sum(sum(sum(sum(first, 0i32), 0i32), 0i32), 0i32)")
        .replace("sum(second, 0i32)", "sum(sum(sum(sum(second, 0i32), 0i32), 0i32), 0i32)");
    assert_parity(
        &source,
        "stage_shapes",
        "main.0 = 51.0\nmain.1 = Model([6.0, 6.0])\n",
    );
}
