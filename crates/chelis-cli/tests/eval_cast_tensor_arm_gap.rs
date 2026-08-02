//! Runtime-evaluator coverage for tensor-input `cast` (V2-F2 of the 0.7.6
//! red-team v2, PR #58).
//!
//! ## Background
//!
//! `crates/chelis-compiler-api/src/runtime/eval.rs::eval_cast` matches on the
//! runtime value but has no `RuntimeValue::Tensor` arm. The catch-all
//! `Err(format!("unsupported cast from {other:?}"))` fires for any tensor
//! input, so `chelis eval --file` errors out even though `chelis check` and
//! `chelis build --target c` accept the same program. Spec §2.7 (table at
//! `spec/03-deep-syntax.md` line 319) lists `cast` as `(cast {} expr
//! target-type)` with semantics "Precision cast" -- first-class on tensors.
//!
//! This gap is the same severity class as the runtime/host-lane jit/par gap
//! that PR #56 closed: the IR DAG path is right, the host-runtime dispatch
//! never grew the arm.
//!
//! ## Fixtures
//!
//! All four fixtures failed before the fix commit with `unsupported cast
//! from Tensor(..)`. The diagnosis commit documented the failure mode; the
//! fix commit added the `RuntimeValue::Tensor` arm to `eval_cast` and
//! flipped these fixtures from `#[ignore]` to running.
//!
//! * `eval_cast_tensor_f32_to_f64` -- f32 source widens to f64; element
//!   values are preserved exactly because every f32 representable value is
//!   also representable as f64.
//! * `eval_cast_tensor_f64_to_f32` -- f64 source narrows to f32; we pick
//!   values that round-trip exactly (1.5, 2.5, 3.5 are exactly representable
//!   in both).
//! * `eval_cast_tensor_fractional_f32_to_int32_traps` -- the checked default
//!   refuses to choose a rounding rule for fractional values and traps
//!   `Domain`; callers spell `floor` or `round` before `cast`.
//! * `eval_cast_tensor_int32_to_f32` -- integer source widens to f32;
//!   `cast(cast(to_tensor([..]), int32), f32)` exercises the int-source arm.

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

/// Parse `tensor(shape=[..], data=[..])` from an `eval --file` stdout line.
fn parse_anonymous_tensor_data(stdout: &str) -> Option<Vec<f64>> {
    let line = stdout.lines().find(|l| l.contains("tensor("))?;
    let after_data = line.split("data=[").nth(1)?;
    let inside = after_data.split(']').next()?;
    inside
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "...")
        .map(|s| s.parse::<f64>().ok())
        .collect()
}

#[test]
fn eval_cast_tensor_f32_to_f64() {
    // f32 source widens to f64. Every f32 representable value is also
    // representable as f64, so the data array is element-wise identical.
    // Spec section 2.7: `cast` is a "Precision cast" on tensors.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "cast_f32_to_f64.ch",
        "result = cast(to_tensor([1.5, 2.5, 3.5]), f64)\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "tensor cast f32->f64 should eval (spec section 2.7); stderr={stderr} stdout={stdout}"
    );
    let elements = parse_anonymous_tensor_data(&stdout)
        .unwrap_or_else(|| panic!("expected tensor in stdout: {stdout}"));
    assert_eq!(elements, vec![1.5, 2.5, 3.5], "stdout={stdout}");
}

#[test]
fn eval_cast_tensor_f64_to_f32() {
    // f64 source narrows to f32. 1.5, 2.5, 3.5 are exactly representable in
    // both formats, so the assertion is exact. Source-side `cast(.., f64)`
    // forces the input precision; outer cast exercises narrowing.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "cast_f64_to_f32.ch",
        "result = cast(cast(to_tensor([1.5, 2.5, 3.5]), f64), f32)\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "tensor cast f64->f32 should eval (spec section 2.7); stderr={stderr} stdout={stdout}"
    );
    let elements = parse_anonymous_tensor_data(&stdout)
        .unwrap_or_else(|| panic!("expected tensor in stdout: {stdout}"));
    assert_eq!(elements, vec![1.5, 2.5, 3.5], "stdout={stdout}");
}

#[test]
fn eval_cast_tensor_fractional_f32_to_int32_traps() {
    // The checked default does not choose a rounding policy implicitly.
    // Fractional values require an explicit floor/round/trunc operation.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "cast_f32_to_int32.ch",
        "result = cast(to_tensor([1.5, 2.5, 3.5]), int32)\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(!ok, "fractional f32->int32 must trap; stdout={stdout}");
    assert!(
        stderr.contains("numeric trap: domain in cast at int32"),
        "expected checked-cast Domain trap; stderr={stderr}"
    );
}

#[test]
fn eval_cast_tensor_int32_to_f32() {
    // Integer source widens to f32. `to_tensor([1, 2, 3])` produces int64
    // by default; the inner `cast(.., int32)` forces int32 source, the
    // outer cast exercises int-to-float widening.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "cast_int32_to_f32.ch",
        "result = cast(cast(to_tensor([1, 2, 3]), int32), f32)\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "tensor cast int32->f32 should eval (spec section 2.7); stderr={stderr} stdout={stdout}"
    );
    let elements = parse_anonymous_tensor_data(&stdout)
        .unwrap_or_else(|| panic!("expected tensor in stdout: {stdout}"));
    assert_eq!(elements, vec![1.0, 2.0, 3.0], "stdout={stdout}");
}
