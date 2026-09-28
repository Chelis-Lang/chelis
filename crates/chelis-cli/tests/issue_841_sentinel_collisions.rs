//! chelis#841 - the unresolved-callable marker is unspellable.
//!
//! The host lowerer's generic fallback used the in-band string `call`,
//! and ABI projection rejected any call spelled `call` or
//! `__unresolved_*`: a user def legitimately named `call` was
//! uncompilable for C with wrong-reason prose. The marker is now
//! `#chelis-unresolved-callable` (`HOST_UNRESOLVED_CALLABLE_MARKER`),
//! which no Surf/Deep identifier can spell, so the collision class is
//! gone: a def named `call` builds, runs, and prints the same value the
//! evaluator computes, while genuinely-unresolved callables keep the
//! frozen `unsupported:` rejection.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{authored_c_symbol, write_file};

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `chelis build --target c`; Ok((generated .c source, live tempdir,
/// out dir)) or Err(stderr).
fn c_build_source(
    program: &str,
    name: &str,
) -> Result<(String, tempfile::TempDir, std::path::PathBuf), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
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
        .expect("chelis build should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let source = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .expect("the translation unit must be emitted");
    Ok((source, dir, out_dir))
}

fn link_and_run(out_dir: &std::path::Path, name: &str) -> String {
    let status = common::link_generated(out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "cc link failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// The chelis#841 headline: a def legitimately named `call` is an
/// ordinary supported program, end to end.
#[test]
fn a_def_named_call_builds_and_runs() {
    let (source, _dir, out_dir) = c_build_source(
        "def call(x: i32) -> i32 = add(x, 1)\n\
         out = print(call(5))\n",
        "def_named_call",
    )
    .expect("a def named `call` is a legal identifier, not a marker");
    let symbol = authored_c_symbol("call");
    assert!(
        source.contains(&format!("{symbol}(")),
        "the legal def must be declared and referenced through its compiler-owned ABI symbol:\n{source}"
    );
    assert!(
        !source.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("int32_t call(") || line.starts_with("static int32_t call(")
        }),
        "the source spelling must not leak as a bare external C function:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "def_named_call"), "6");
    }
}

/// Negative parity: a genuinely unresolved callable (a used returned
/// function value) keeps the frozen rejection, now with prose that is
/// true - the marker, not a user identifier, was rejected.
#[test]
fn a_used_returned_function_still_rejects_with_the_frozen_diagnostic() {
    let err = c_build_source(
        "def increment(x: i8) -> i8 = add(x, cast(1, i8))\n\
         def choose() -> i8 -> i8 = increment\n\
         chosen = choose()\n\
         out = print(chosen(cast(6, i8)))\n",
        "returned_named_used",
    )
    .expect_err("a used returned function value must not build for C");
    assert!(
        err.contains("unsupported:") && err.contains("function value"),
        "the frozen function-value diagnostic must survive the marker rename:\n{err}"
    );
    assert!(
        !err.contains("grad") && !err.contains("vmap"),
        "no AD-transform misclassification:\n{err}"
    );
    assert!(
        !err.contains("#chelis-unresolved-callable"),
        "the internal marker spelling must not leak into diagnostics:\n{err}"
    );
}

/// The eval and build lanes agree on the value a def named `call`
/// computes.
#[test]
fn eval_agrees_with_the_compiled_def_named_call() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("call_eval.ch");
    write_file(
        &path,
        "def call(x: i32) -> i32 = add(x, 1)\n\
         out = print(call(5))\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval accepts the name: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains('6'),
        "eval must compute 6"
    );
}

/// chelis#841 review, finding 1: a plain callable bug keeps the frozen
/// function-value diagnostic even when an unrelated (and fully
/// supported) `grad` exists elsewhere in the program; the transform
/// marker, not whole-program state, routes the AD workaround text.
#[test]
fn an_unrelated_grad_does_not_reclassify_a_callable_bug() {
    let err = c_build_source(
        "def increment(x: i8) -> i8 = add(x, cast(1, i8))\n\
         def choose() -> i8 -> i8 = increment\n\
         def square(theta: f32) -> f32 = mul(theta, theta)\n\
         def gradient(theta: f32) -> f32 = grad(square)(theta)\n\
         chosen = choose()\n\
         out = print(chosen(cast(6, i8)))\n",
        "mixed_grad_callable",
    )
    .expect_err("the returned callable still rejects");
    assert!(
        err.contains("unsupported:") && err.contains("function value"),
        "the callable bug keeps the frozen diagnostic:\n{err}"
    );
    assert!(
        !err.contains("applies/binds"),
        "an unrelated grad must not reclassify the callable bug as an AD failure:\n{err}"
    );
}
