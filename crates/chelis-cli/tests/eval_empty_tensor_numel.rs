//! Evaluator coverage for empty tensor shape and element count.
//! A rank-one empty tensor has rank 1 and `numel == 0`; non-empty
//! controls distinguish it from scalar and singleton cases.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn eval_file(path: &Path) -> (bool, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn write_program(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, body).expect("write fixture");
    path
}

#[test]
fn eval_numel_empty_tensor() {
    // [05-OP-57] supplies dtype through the List type, never a default.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "numel_empty.ch",
        "result = {\n  empty_values: List[f32] = []\n  numel(to_tensor(empty_values))\n}\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "numel(empty tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "result = 0",
        "numel of shape=[0] tensor must be 0, not 1 (Runtime-EmptyTensorNumel-F1); stdout={stdout}"
    );
}

#[test]
fn eval_rank_empty_tensor() {
    // Pin that a typed empty List produces a rank-1 tensor, not a scalar. The
    // bug surface for numel(empty) is not in rank inference; this fixture
    // locks the shape distinction so a future "fix" that flips empty
    // to rank-0 trips here.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "rank_empty.ch",
        "result = {\n  empty_values: List[f32] = []\n  rank(to_tensor(empty_values))\n}\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "rank(empty tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "result = 1",
        "rank of an empty 1D tensor must be 1; stdout={stdout}"
    );
}

#[test]
fn eval_numel_three_element_tensor() {
    // Positive control: shape [3] -> numel 3. Guards against a fix that
    // collapses non-empty product paths.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "numel_three.ch",
        "result = numel(to_tensor([1, 2, 3], i32))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "numel(3-element tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "result = 3",
        "numel of shape=[3] must be 3; stdout={stdout}"
    );
}

#[test]
fn eval_numel_single_element_tensor() {
    // Positive control: shape [1] -> numel 1. Distinct from the scalar
    // case (shape []), this fixture pins that a single-element rank-1
    // tensor is also numel 1.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "numel_single.ch",
        "result = numel(to_tensor([1.0], f32))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "numel(single-element tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "result = 1",
        "numel of shape=[1] must be 1; stdout={stdout}"
    );
}
