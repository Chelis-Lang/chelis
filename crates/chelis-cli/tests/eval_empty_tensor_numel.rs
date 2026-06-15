//! Runtime coverage for `numel(to_tensor([]))` returning the wrong value
//! (Runtime-EmptyTensorNumel-F1, pre-0.7.8 release blocker).
//!
//! ## Background
//!
//! `numel(to_tensor([]))` returns `1` instead of `0` via `chelis eval --file`,
//! despite the shape being printed as `tensor(shape=[0], data=[])`. The bug
//! is in two clamp sites that inflate a zero-element product back to 1:
//!
//! - `crates/chelis-compiler-api/src/runtime/eval.rs::eval_builtin "numel"` does
//!   `tensor.value.shape.iter().product::<usize>().max(1)`. For shape `[0]`
//!   the product is 0, then `.max(1)` clamps to 1.
//! - `crates/chelis-runtime/src/lib.rs::chelis_alloc_tensor` and
//!   `chelis_alloc_view` both run `if tensor.size == 0 { tensor.size = 1; }`
//!   after multiplying shape components. `chelis_tensor_numel` then returns
//!   the clamped size. The C backend exhibits the same wrong-value output.
//!
//! Both sites must drop the clamp for the empty rank-1 case. The scalar case
//! (shape `[]`, ndim == 0) already produces 1 via the empty-product identity
//! (`[].iter().product::<usize>() == 1`) and via the runtime initializer
//! `size: 1` plus the skipped multiplication loop; neither path needs the
//! clamp to handle scalars.
//!
//! ## Fixtures
//!
//! All four fixtures are pinned exact-equality. Positive controls cover the
//! non-empty paths to catch regressions where a too-aggressive fix would
//! break the scalar or single-element cases.
//!
//! * `eval_numel_empty_tensor`: shape `[0]`, expect `numel == 0`. Today
//!   returns `1`; this is the primary bug.
//! * `eval_rank_empty_tensor`: shape `[0]`, expect `rank == 1`. Pins that
//!   the empty list literal infers as rank-1, not as a scalar.
//! * `eval_numel_three_element_tensor`: positive control, expect `3`.
//! * `eval_numel_single_element_tensor`: positive control, expect `1`. A
//!   too-aggressive fix that returned shape.iter().product() without
//!   special-casing the scalar would break this fixture if shape `[1]` got
//!   confused with shape `[]`.

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
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "numel_empty.ch",
        "result = numel(to_tensor([]))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "numel(empty tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "0",
        "numel of shape=[0] tensor must be 0, not 1 (Runtime-EmptyTensorNumel-F1); stdout={stdout}"
    );
}

#[test]
fn eval_rank_empty_tensor() {
    // Pin that to_tensor([]) is a rank-1 empty tensor, not a scalar. The
    // bug surface for numel(empty) is not in rank inference; this fixture
    // locks the shape distinction so a future "fix" that flips empty
    // to rank-0 trips here.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "rank_empty.ch",
        "result = rank(to_tensor([]))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "rank(empty tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "1",
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
        "result = numel(to_tensor([1, 2, 3]))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "numel(3-element tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "3",
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
        "result = numel(to_tensor([1.0]))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "numel(single-element tensor) should eval; stderr={stderr} stdout={stdout}"
    );
    assert_eq!(
        stdout.trim(),
        "1",
        "numel of shape=[1] must be 1; stdout={stdout}"
    );
}
