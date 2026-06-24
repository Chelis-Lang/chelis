//! Issue Chelis-Lang/chelis#197: the CLI grad path uses the unchecked
//! `grad_dag` variant, so non-differentiable ops surface as generic
//! errors (argmax/argmin) or silently zero-grad (floor/ceil) instead
//! of the structured `AdError::NotSupported` rejection that
//! `grad_dag_checked` already provides at the IR layer.
//!
//! These tests pin the CLI surface against that gap. Each negative
//! reproducer compiles a `.ch` whose `grad(...)` body contains a
//! non-differentiable op; the post-fix CLI must (a) exit non-zero and
//! (b) emit an `AdError::NotSupported` diagnostic naming the offending
//! op.
//!
//! The four ops covered (each with its own test) are:
//!
//!   - `argmax_reduce` — integer-index output, was generic-error pre-fix
//!   - `argmin_reduce` — integer-index output, was generic-error pre-fix
//!   - `floor`         — piecewise constant, was silent zero-grad pre-fix
//!   - `ceil`          — piecewise constant, was silent zero-grad pre-fix
//!
//! The floor/ceil pair is the load-bearing half: pre-fix code accepted
//! the build cleanly and emitted a zero gradient with no diagnostic, so
//! the assertion that `chelis build` exits non-zero is what catches the
//! silent-corruption mode. Per feedback_ad_reduction_pitfalls: silent
//! zero-grad is the worse half because downstream code receives wrong
//! results without a signal.
//!
//! Positive parity: a normal differentiable body (matmul + sum) still
//! builds cleanly after the switch to `grad_dag_checked`. This is the
//! no-regression anchor for the routing change.

use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_build(path: &Path) -> std::process::Output {
    let out_dir = path.parent().expect("source path has parent");
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    std::process::Command::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("spawn chelis")
}

// =====================================================================
// Negative: argmax_reduce in a grad path emits AdError::NotSupported.
// =====================================================================

#[test]
fn issue_197_argmax_in_grad_path_emits_ad_error_not_supported() {
    // Pre-fix: grad_dag returns None for argmax, the CLI propagates a
    // generic "grad requires a scalar floating forward output" error.
    // Post-fix: grad_dag_checked surfaces AdError::NotSupported with
    // op="argmax" and the IntegerIndexOutput reason.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("argmax_grad.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, 3, f32]) -> f32 =\n\
           tensor_to_scalar(sum(cast(argmax_reduce(copy(theta), 0), f32), 0))\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         out = grad_loss(to_tensor([[1.0, 2.0, 3.0], [0.5, 4.0, 1.0]]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "argmax in a grad path must reject the build; got success with stderr={stderr}"
    );
    assert!(
        stderr.contains("argmax"),
        "argmax rejection must name the op; got stderr={stderr}"
    );
    assert!(
        stderr.contains("non-differentiable"),
        "argmax rejection must include the non-differentiable diagnostic; got stderr={stderr}"
    );
}

// =====================================================================
// Negative: argmin_reduce in a grad path emits AdError::NotSupported.
// =====================================================================

#[test]
fn issue_197_argmin_in_grad_path_emits_ad_error_not_supported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("argmin_grad.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, 3, f32]) -> f32 =\n\
           tensor_to_scalar(sum(cast(argmin_reduce(copy(theta), 0), f32), 0))\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         out = grad_loss(to_tensor([[1.0, 2.0, 3.0], [0.5, 4.0, 1.0]]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "argmin in a grad path must reject the build; got success with stderr={stderr}"
    );
    assert!(
        stderr.contains("argmin"),
        "argmin rejection must name the op; got stderr={stderr}"
    );
    assert!(
        stderr.contains("non-differentiable"),
        "argmin rejection must include the non-differentiable diagnostic; got stderr={stderr}"
    );
}

// =====================================================================
// Negative: floor in a grad path emits AdError::NotSupported.
//
// LOAD-BEARING: pre-fix this silently zero-grads. The assertion that
// the build exits non-zero is what catches the silent-corruption mode.
// =====================================================================

#[test]
fn issue_197_floor_in_grad_path_emits_ad_error_not_supported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("floor_grad.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(floor(copy(theta)), 0))\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         out = grad_loss(to_tensor([1.5, 2.5]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "floor in a grad path must reject the build instead of silently zero-grading; \
         got success with stderr={stderr}"
    );
    assert!(
        stderr.contains("floor"),
        "floor rejection must name the op; got stderr={stderr}"
    );
    assert!(
        stderr.contains("non-differentiable"),
        "floor rejection must include the non-differentiable diagnostic; got stderr={stderr}"
    );
}

// =====================================================================
// Negative: ceil in a grad path emits AdError::NotSupported.
//
// LOAD-BEARING: pre-fix this silently zero-grads.
// =====================================================================

#[test]
fn issue_197_ceil_in_grad_path_emits_ad_error_not_supported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ceil_grad.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(ceil(copy(theta)), 0))\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         out = grad_loss(to_tensor([1.5, 2.5]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "ceil in a grad path must reject the build instead of silently zero-grading; \
         got success with stderr={stderr}"
    );
    assert!(
        stderr.contains("ceil"),
        "ceil rejection must name the op; got stderr={stderr}"
    );
    assert!(
        stderr.contains("non-differentiable"),
        "ceil rejection must include the non-differentiable diagnostic; got stderr={stderr}"
    );
}

// =====================================================================
// Negative: round in a grad path emits AdError::NotSupported.
//
// round is piecewise-constant like floor/ceil; pre-fix this would
// silently zero-grad. The non-zero exit is what catches that mode.
// =====================================================================

#[test]
fn round_in_grad_path_emits_ad_error_not_supported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("round_grad.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(round(copy(theta)), 0))\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         out = grad_loss(to_tensor([1.5, 2.5]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "round in a grad path must reject the build instead of silently zero-grading; \
         got success with stderr={stderr}"
    );
    assert!(
        stderr.contains("round"),
        "round rejection must name the op; got stderr={stderr}"
    );
    assert!(
        stderr.contains("non-differentiable"),
        "round rejection must include the non-differentiable diagnostic; got stderr={stderr}"
    );
}

// =====================================================================
// Positive: a differentiable body (matmul + sum reductions) still
// builds cleanly after the switch to `grad_dag_checked`. No regression
// on the happy path.
// =====================================================================

#[test]
fn issue_197_differentiable_body_still_builds_after_checked_routing() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("diff_body.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(mul(copy(theta), copy(theta)), 0))\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         out = grad_loss(to_tensor([1.0, 2.0]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "a differentiable grad body must still build after routing through \
         grad_dag_checked; got stderr={stderr}"
    );
}
