//! Issue #291 (end-to-end): `grad` could not lower the backward pass for
//! the slicing/windowing movement ops `shrink` and `stride`; both failed
//! with "failed to construct backward DAG".
//!
//! Headline reproducers (verbatim from the issue):
//! ```chelis
//! def f(x: tensor[4, f32]) -> f32 =
//!   tensor_to_scalar(sum(shrink(x, [[cast(0, int64), cast(2, int64)]]), cast(0, int32)))
//! def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)
//! ```
//! ```chelis
//! def f(x: tensor[4, f32]) -> f32 =
//!   tensor_to_scalar(sum(stride(x, cast(2, int64)), cast(0, int32)))
//! def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)
//! ```
//!
//! Before the fix:
//!   * `chelis check` -> clean (type-checks) for both.
//!   * `chelis build --target c` -> for `shrink`: "shrink at node ...:
//!     bounds len 0 != input rank 1" (the `[[cast(0,int32),cast(2,
//!     int32)]]` bound literal lost its `cast`-wrapped pair elements in
//!     lowering, producing `RiscOp::Shrink { bounds: [] }`); for
//!     `stride`: "no reverse-mode adjoint is defined for `stride`".
//!
//! Expected gradients:
//!   * `shrink(x, [0,2))` then sum: `f = x0 + x1`, `df = [1, 1, 0, 0]`
//!     (cotangent scattered into the sliced positions).
//!   * `stride(x, 2i64)` then sum: `f = x0 + x2`, `df = [1, 0, 1, 0]`
//!     (cotangent scattered into the strided slots).
//!
//! This file is the end-to-end acceptance oracle: `check` clean, `build
//! --target c` succeeds and emits a kernel, and the host evaluator prints
//! the correct gradient. gcc-compiling the emitted grad kernel is
//! intentionally not asserted here, matching the #288 CLI corpus.

use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

const SHRINK_DF: &str = "module Repro.GradShrink\n\
def f(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(shrink(x, [[cast(0, int64), cast(2, int64)]]), cast(0, int32)))\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)\n";

const SHRINK_EVAL: &str = "module Repro.GradShrink\n\
def f(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(shrink(x, [[cast(0, int64), cast(2, int64)]]), cast(0, int32)))\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)\n\
out = df(to_tensor([10.0, 20.0, 30.0, 40.0]))\n";

const STRIDE_DF: &str = "module Repro.GradStride\n\
def f(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(stride(x, cast(2, int64)), cast(0, int32)))\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)\n";

const STRIDE_EVAL: &str = "module Repro.GradStride\n\
def f(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(stride(x, cast(2, int64)), cast(0, int32)))\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)\n\
out = df(to_tensor([10.0, 20.0, 30.0, 40.0]))\n";

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

fn run_eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

fn run_build_c(source: &str, stem: &str) -> (tempfile::TempDir, PathBuf, std::process::Output) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    let kernel_c = dir.path().join(format!("{stem}.c"));
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            kernel_c.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    (dir, kernel_c, output)
}

// --- check: both reproducers already type-check; pin it. ---

#[test]
fn issue_291_shrink_check_is_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro.ch");
    fs::write(&path, SHRINK_DF).expect("write source");
    let errs = error_messages(&run_check(&path));
    assert!(
        errs.is_empty(),
        "issue #291 shrink reproducer must type-check clean; got {errs:?}",
    );
}

#[test]
fn issue_291_stride_check_is_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro.ch");
    fs::write(&path, STRIDE_DF).expect("write source");
    let errs = error_messages(&run_check(&path));
    assert!(
        errs.is_empty(),
        "issue #291 stride reproducer must type-check clean; got {errs:?}",
    );
}

// --- build --target c: the headline failures. ---

#[test]
fn issue_291_shrink_build_c_succeeds() {
    let (_dir, _kernel, output) = run_build_c(SHRINK_DF, "repro");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #291 shrink: `chelis build --target c` must succeed; stderr={stderr}",
    );
}

#[test]
fn issue_291_stride_build_c_succeeds() {
    let (_dir, _kernel, output) = run_build_c(STRIDE_DF, "repro");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #291 stride: `chelis build --target c` must succeed; stderr={stderr}",
    );
}

// --- eval: host-evaluator numeric correctness. ---

#[test]
fn issue_291_shrink_eval_gradient_is_correct() {
    let output = run_eval(SHRINK_EVAL, "repro");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #291 shrink: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[4]"),
        "gradient must be tensor[4]; got stdout={stdout}",
    );
    // f = x0 + x1, df = [1, 1, 0, 0] (cotangent in the sliced positions).
    assert!(
        stdout.contains("data=[1.0, 1.0, 0.0, 0.0]"),
        "shrink df(x) must equal [1, 1, 0, 0]; got stdout={stdout}",
    );
}

#[test]
fn issue_291_stride_eval_gradient_is_correct() {
    let output = run_eval(STRIDE_EVAL, "repro");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #291 stride: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[4]"),
        "gradient must be tensor[4]; got stdout={stdout}",
    );
    // f = x0 + x2, df = [1, 0, 1, 0] (cotangent in the strided slots).
    assert!(
        stdout.contains("data=[1.0, 0.0, 1.0, 0.0]"),
        "stride df(x) must equal [1, 0, 1, 0]; got stdout={stdout}",
    );
}

// --- build emits a kernel (the grad DAG now lowers and reaches C). ---

#[test]
fn issue_291_shrink_build_c_emits_kernel() {
    let (_dir, kernel_c, output) = run_build_c(SHRINK_DF, "repro");
    assert!(
        output.status.success(),
        "build --target c must succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let emitted = fs::read_to_string(&kernel_c).expect("emitted C kernel must exist");
    assert!(
        emitted.contains("f("),
        "emitted C must define the gradient function `f`; got {} bytes",
        emitted.len(),
    );
}

#[test]
fn issue_291_stride_build_c_emits_kernel() {
    let (_dir, kernel_c, output) = run_build_c(STRIDE_DF, "repro");
    assert!(
        output.status.success(),
        "build --target c must succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let emitted = fs::read_to_string(&kernel_c).expect("emitted C kernel must exist");
    assert!(
        emitted.contains("f("),
        "emitted C must define the gradient function `f`; got {} bytes",
        emitted.len(),
    );
}
