//! Issue #288: `grad` cannot lower the backward pass through the
//! constant-broadcast idiom `expand(scalar_to_tensor(c), axis, n)`
//! (n > 1).
//!
//! Headline reproducer (verbatim from the issue):
//! ```chelis
//! module Repro.GradExpandConst
//! def f(x: tensor[2, f32]) -> f32 = {
//!   k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32))
//!   tensor_to_scalar(sum(mul(x, k), cast(0, int32)))
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
//! `df(x) = [2.5, 2.5]` for any `x`. This file is the acceptance oracle
//! for the end-to-end fix: `check` clean, `build --target c` succeeds,
//! and the host evaluator prints the correct gradient. The
//! evaluator-vs-C-backend agreement is asserted by gcc-compiling and
//! running the emitted kernel.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

const REPRO: &str = "module Repro.GradExpandConst\n\
def f(x: tensor[2, f32]) -> f32 = {\n\
  k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32))\n\
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
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
        .env("CHELIS_DEBUG_GRAD_DAG", "1")
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
/// printer emits one `name = tensor(shape=..., data=...)` line per
/// top-level binding; the `out` line must carry the constant gradient.
#[test]
fn issue_288_eval_gradient_is_correct() {
    let output = run_eval(REPRO, "repro");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #288: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    let out_line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("out ="))
        .unwrap_or_else(|| panic!("eval output missing `out` binding; stdout={stdout}"));
    assert!(
        out_line.contains("shape=[2]"),
        "gradient must be tensor[2]; got {out_line}",
    );
    // The gradient of `sum(x * 2.5)` w.r.t. x is the constant 2.5 in
    // every position, independent of x.
    assert!(
        out_line.contains("2.5") && out_line.contains("data=[2.5, 2.5]"),
        "df(x) must equal [2.5, 2.5]; got {out_line}",
    );
}

/// Backend-numerics: the C backend must agree with the host evaluator.
/// Build the kernel, gcc-compile it against the runtime static lib, run
/// it, and assert its printed output matches `chelis eval`.
#[test]
fn issue_288_c_backend_agrees_with_eval() {
    let eval_output = run_eval(REPRO, "repro");
    assert!(
        eval_output.status.success(),
        "eval must succeed before backend agreement check",
    );
    let eval_out = String::from_utf8_lossy(&eval_output.stdout)
        .trim_end()
        .to_string();

    let (build_dir, kernel_c, build_output) = run_build_c(REPRO, "repro");
    assert!(
        build_output.status.success(),
        "build --target c must succeed; stderr={}",
        String::from_utf8_lossy(&build_output.stderr),
    );

    let canonical = target_debug_dir().join("libchelis_runtime.a");
    ensure_runtime_static_lib(&canonical).expect("materialize libchelis_runtime.a");

    let bin = build_dir.path().join("repro_bin");
    let compile = StdCommand::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.path().to_str().unwrap(),
            kernel_c.to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            canonical.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc compile failed: stderr={}",
        String::from_utf8_lossy(&compile.stderr),
    );
    let run = StdCommand::new(&bin).output().expect("run kernel binary");
    assert!(
        run.status.success(),
        "kernel binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    let c_out = String::from_utf8_lossy(&run.stdout).trim_end().to_string();
    assert_eq!(
        c_out, eval_out,
        "C backend output must agree with host evaluator for issue #288",
    );
}

// --- runtime static-lib materialization (mirrors
// cbackend_print_tensor_f64.rs) ---

fn target_debug_dir() -> PathBuf {
    // The test binary lives at target/<profile>/deps/<bin>; the runtime
    // static lib is built into target/<profile>/deps.
    let exe = std::env::current_exe().expect("current_exe");
    let deps = exe.parent().expect("deps dir").to_path_buf();
    deps.parent().expect("profile dir").to_path_buf()
}

fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    let exe = std::env::current_exe().expect("current_exe");
    let deps_dir = exe.parent().expect("deps dir");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(deps_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let mtime = entry.metadata()?.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    let Some((_, hashed)) = newest else {
        return Err(std::io::Error::other(format!(
            "no libchelis_runtime-*.a found in {}",
            deps_dir.display()
        )));
    };
    let tmp = canonical.with_extension("a.tmp");
    fs::copy(&hashed, &tmp)?;
    fs::rename(&tmp, canonical)?;
    Ok(())
}
