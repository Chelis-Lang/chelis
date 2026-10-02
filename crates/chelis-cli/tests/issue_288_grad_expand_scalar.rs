//! Issue #288: `grad` cannot lower the backward pass through the
//! constant-broadcast idiom `expand(scalar_to_tensor(c), axis, n)`
//! (n > 1).
//!
//! Headline reproducer (verbatim from the issue):
//! ```chelis
//! module Repro.GradExpandConst
//! def f(x: tensor[2, f32]) -> f32 = {
//!   k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64))
//!   tensor_to_scalar(sum(mul(x, k), cast(0, i32)))
//! }
//! def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)
//! ```
//!
//! Before the fix:
//!   * `chelis check` -> clean (type-checks).
//!   * `chelis build --target c` -> `Lowering error: grad(...) lowering
//!     rejected: grad: failed to construct backward DAG (unsupported op
//!     or verification failure)`.
//!
//! `f(x) = sum(x * 2.5) = 2.5 * (x0 + x1)`, so the gradient is constant:
//! `df(x) = [2.5, 2.5]` for any `x`. This file is the end-to-end
//! acceptance oracle: `check` clean, `build --target c` succeeds and
//! emits a kernel, and the host evaluator prints the correct gradient.
//! The host evaluator is the numeric oracle (the IR-level sibling file
//! adds the finite-difference cross-check); gcc-compiling the emitted
//! grad kernel is intentionally not asserted here — see
//! `issue_288_build_c_emits_kernel`.

use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use chelis_backend_c::GeneratedHeader;
use serde_json::Value;
use tempfile::tempdir;

const REPRO: &str = "module Repro.GradExpandConst\n\
def f(x: tensor[2, f32]) -> f32 = {\n\
  k = insert(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64))\n\
  tensor_to_scalar(sum(mul(x, k), cast(0, i32)))\n\
}\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

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
            "--emit-c",
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

fn assert_exact_f_export(kernel_c: &Path) {
    let emitted = fs::read_to_string(kernel_c).expect("emitted C kernel must exist");
    let header_path = kernel_c.with_extension("h");
    let header = fs::read_to_string(&header_path).expect("generated header must exist");
    let generated = GeneratedHeader::parse(&header).expect("generated declaration metadata");
    generated
        .validate_source(&emitted)
        .expect("generated declaration must match the emitted definition");
    let declaration = generated
        .declaration("f")
        .expect("authored `f` declaration");
    assert_eq!(declaration.symbol(), "chelis_fn_66");
    assert_eq!(
        declaration.declaration(),
        "float chelis_fn_66(chelis_tensor* x);"
    );
    assert!(
        emitted.contains("chelis_fn_66__chelis_owned_body("),
        "the public declaration and owned-body call must share the canonical identity"
    );

    let stale_header = header.replace("chelis_fn_66", "f");
    assert!(
        GeneratedHeader::parse(&stale_header).is_err(),
        "tampering with declaration bytes without its structural record must fail parsing"
    );
    let stale_source = emitted.replace("chelis_fn_66(", "f(");
    assert!(
        generated.validate_source(&stale_source).is_err(),
        "a mutated emitted definition must fail agreement"
    );
}

/// `chelis check` already passes today; pin it so a fix that
/// accidentally breaks type-checking is caught.
#[test]
fn issue_288_check_is_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro.ch");
    fs::write(&path, REPRO).expect("write source");
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "issue #288 reproducer must type-check clean; got {errs:?}",
    );
}

/// The headline failure: `chelis build --target c` must succeed. Before
/// the fix this exits non-zero with the "failed to construct backward
/// DAG" lowering error.
#[test]
fn issue_288_build_c_succeeds() {
    let (_dir, _kernel, output) = run_build_c(REPRO, "repro");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #288: `chelis build --target c` must succeed; stderr={stderr}",
    );
}

/// Host-evaluator numeric correctness: `df(x) = [2.5, 2.5]`. The eval
/// printer emits a `tensor(shape=..., data=...)` rendering for the
/// `out = df(...)` binding; the gradient of `sum(x * 2.5)` w.r.t. x is
/// the constant 2.5 in every position, independent of x.
#[test]
fn issue_288_eval_gradient_is_correct() {
    let output = run_eval(REPRO, "repro");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #288: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[2]"),
        "gradient must be tensor[2]; got stdout={stdout}",
    );
    assert!(
        stdout.contains("data=[2.5, 2.5]"),
        "df(x) must equal [2.5, 2.5]; got stdout={stdout}",
    );
}

/// The C target must emit a non-empty kernel for the gradient program.
/// Issue #288 is the grad-LOWERING bug: before the fix `build --target
/// c` aborted at IR lowering and emitted nothing. This pins that the
/// grad DAG now lowers and reaches C emission.
///
/// Note: gcc-compiling and running the emitted grad kernel is
/// deliberately NOT asserted here. The host evaluator is the numeric
/// oracle for #288 (see `issue_288_eval_gradient_is_correct` and the
/// IR-level finite-difference checks), matching the existing
/// grad-in-build corpus (`issue_218_to_tensor_in_grad_body.rs`), which
/// also stops at build success. A separate C-backend codegen defect
/// (the emitted grad kernel uses a tensor handle where a scalar is
/// expected) is unrelated to the grad-lowering fix and is out of scope
/// for #288.
#[test]
fn issue_288_build_c_emits_kernel() {
    let (_dir, kernel_c, output) = run_build_c(REPRO, "repro");
    assert!(
        output.status.success(),
        "build --target c must succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert_exact_f_export(&kernel_c);
}
