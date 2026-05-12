//! Runtime-evaluator and C-backend coverage for `jit` and `par` (Findings 1+2
//! of the 0.7.6 toolchain hygiene red-team, PR #51).
//!
//! ## Background
//!
//! PR #40 ("fix: jit pass-through and par sequential semantics per spec") wired
//! `jit` and `par` into IR lowering (`crates/chelis-ir/src/lower.rs::lower_jit`
//! pass-through; `lower_par` sequential-yielding-last) and the checker. The IR
//! DAG path now lowers both forms correctly, which is enough for scalar bodies
//! whose top-level `def` classifies as "lowered" (DAG-evaluable) by
//! `top_level_lowering_map`.
//!
//! What PR #40 did NOT do was extend the **runtime evaluator's** dispatch in
//! `crates/chelis-compiler-api/src/runtime.rs::eval_list` and the C-backend's
//! `lower_host_expr_kind` in `crates/chelis-ir/src/host.rs`. When a def's body
//! reaches the host lane (typical for tensor-returning forms via `to_tensor`),
//! the evaluator falls through to a default error
//! `host runtime does not support 'jit'/'par'`, and the C-backend's
//! `lower_host_expr_kind` falls through to its `_ => HostExpr::Unit` catch-all
//! so the emitted binding silently drops the value (prints `()`).
//!
//! Spec semantics (`spec/03-deep-syntax.md`):
//!
//! * §2.7 `jit` — compilation trigger, semantically a no-op at evaluation.
//!   The runtime arm must return the value of the inner expression.
//! * §2.3 `par` v1 — sequential composition. The runtime arm must evaluate
//!   each child in order and return the value of the last child.
//!
//! ## Fixtures
//!
//! These tests reproduce the gaps empirically. They are gated `#[ignore]`
//! pending the fix commit on this branch (`fix/jit-par-runtime-arms`).
//!
//! * `eval_jit_scalar_returns_inner_value` — `result: f32 = jit(1.5)` should
//!   eval to `1.5`. Today: passes (scalar `jit` routes through IR DAG).
//!   Kept as a positive baseline so a regression that breaks the DAG path
//!   is caught at the same surface.
//! * `eval_jit_tensor_returns_inner_value` — `result = jit(to_tensor([..]))`
//!   should eval to the tensor. Today: `host runtime does not support 'jit'`.
//! * `eval_par_scalar_returns_last_value` — `result: f32 = par {1.0;2.0;3.0}`
//!   should eval to `3.0`. Today: passes via DAG.
//! * `eval_par_tensor_returns_last_value` — `result = par {to_tensor(a);
//!   to_tensor(b)}` should eval to the second tensor. Today:
//!   `host runtime does not support 'par'`.
//! * `build_c_jit_tensor_runs_and_prints_value` — generated C must print
//!   the inner-tensor value, not `()`. Today: emits `result = ()`.
//! * `build_c_par_tensor_runs_and_prints_last_value` — generated C must
//!   print the last tensor's value, not `()`. Today: emits `result = ()`.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
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

/// Parse `binding = tensor(shape=[..], data=[..])` lines into a Vec<f32>.
/// Same shape as the helper in `phase3t_build_runtime_gaps.rs`. Used for
/// compiled-binary stdout where each binding is printed `name = value`.
fn parse_named_tensor_data(stdout: &str, binding: &str) -> Option<Vec<f32>> {
    let prefix = format!("{binding} = tensor(");
    let line = stdout.lines().find(|l| l.contains(&prefix))?;
    parse_tensor_data_after_prefix(line)
}

/// Parse `tensor(shape=[..], data=[..])` from a line that may lack a
/// `binding =` prefix. `chelis eval --file` prints each root's value on its
/// own line without the binding name.
fn parse_anonymous_tensor_data(stdout: &str) -> Option<Vec<f32>> {
    let line = stdout.lines().find(|l| l.contains("tensor("))?;
    parse_tensor_data_after_prefix(line)
}

fn parse_tensor_data_after_prefix(line: &str) -> Option<Vec<f32>> {
    let after_data = line.split("data=[").nth(1)?;
    let inside = after_data.split(']').next()?;
    inside
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "...")
        .map(|s| s.parse::<f32>().ok())
        .collect()
}

fn parse_anonymous_scalar(stdout: &str) -> Option<f32> {
    let elements = parse_anonymous_tensor_data(stdout)?;
    if elements.len() != 1 {
        return None;
    }
    Some(elements[0])
}

// ── eval (runtime evaluator) ─────────────────────────────────────────────────

#[test]
fn eval_jit_scalar_returns_inner_value() {
    // Scalar `jit`: spec §2.7 says jit is a compilation trigger and a no-op
    // at evaluation. The scalar path routes through IR DAG lowering, so this
    // passes today. Kept as a positive baseline so a DAG-side regression is
    // caught at the same surface as the host-side gap.
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "scalar_jit.ch", "result: f32 = jit(1.5)\n");
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(ok, "scalar jit should eval; stderr={stderr}");
    let value = parse_anonymous_scalar(&stdout)
        .unwrap_or_else(|| panic!("expected scalar tensor in stdout: {stdout}"));
    assert!(
        (value - 1.5).abs() < 1e-6,
        "result expected 1.5, got {value}; stdout={stdout}"
    );
}

#[test]
#[ignore = "Finding 1 (jit): runtime evaluator missing `jit` arm; fix in fix/jit-par-runtime-arms"]
fn eval_jit_tensor_returns_inner_value() {
    // Tensor `jit`: body uses `to_tensor`, which forces host-runtime
    // classification (see `expr_requires_host_runtime` in lower.rs). The
    // runtime evaluator's `eval_list` falls through to the catch-all
    // `host runtime does not support 'jit'`. Spec §2.7: jit is a no-op at
    // eval, so the inner tensor must be returned.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "tensor_jit.ch",
        "result = jit(to_tensor([1.5, 2.5, 3.5]))\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "tensor jit should eval (spec section 2.7: no-op at eval); stderr={stderr} stdout={stdout}"
    );
    let elements = parse_anonymous_tensor_data(&stdout)
        .unwrap_or_else(|| panic!("expected tensor in stdout: {stdout}"));
    assert_eq!(elements, vec![1.5, 2.5, 3.5], "stdout={stdout}");
}

#[test]
fn eval_par_scalar_returns_last_value() {
    // Scalar `par`: spec §2.3 v1 sequential, last-yields. Scalar path routes
    // through IR DAG, so this passes today (positive baseline).
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "scalar_par.ch",
        "result: f32 = par {\n  1.0;\n  2.0;\n  3.0\n}\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(ok, "scalar par should eval; stderr={stderr}");
    let value = parse_anonymous_scalar(&stdout)
        .unwrap_or_else(|| panic!("expected scalar tensor in stdout: {stdout}"));
    assert!(
        (value - 3.0).abs() < 1e-6,
        "result expected 3.0, got {value}; stdout={stdout}"
    );
}

#[test]
#[ignore = "Finding 2 (par): runtime evaluator missing `par` arm; fix in fix/jit-par-runtime-arms"]
fn eval_par_tensor_returns_last_value() {
    // Tensor `par`: hits host-runtime classification via `to_tensor`. Today
    // errors with `host runtime does not support 'par'`. Spec §2.3 (v1
    // sequential): the par's value is the value of the last child.
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "tensor_par.ch",
        "result = par {\n  to_tensor([1.0, 2.0]);\n  to_tensor([3.0, 4.0, 5.0])\n}\n",
    );
    let (ok, stdout, stderr) = eval_file(&path);
    assert!(
        ok,
        "tensor par should eval (spec section 2.3 v1: sequential, last-yields); stderr={stderr} stdout={stdout}"
    );
    let elements = parse_anonymous_tensor_data(&stdout)
        .unwrap_or_else(|| panic!("expected tensor in stdout: {stdout}"));
    assert_eq!(elements, vec![3.0, 4.0, 5.0], "stdout={stdout}");
}

// ── chelis build --target c (C backend) ──────────────────────────────────────

/// Run `chelis build --target c` on `src_path`, compile the emitted C with
/// gcc, run the binary, and return its stdout.
fn build_c_and_run(out_dir: &Path, src_path: &Path) -> String {
    fs::create_dir_all(out_dir).expect("mkdir out_dir");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let stem = src_path.file_stem().expect("file_stem").to_str().unwrap();
    let c_source = format!("{stem}.c");
    let needs_blas = fs::read_to_string(out_dir.join(&c_source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(&c_source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", stem]);
    let status = cmd.status().expect("gcc should run");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {} (stderr: {})",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
    String::from_utf8(run_output.stdout).expect("compiled stdout should be utf-8")
}

#[test]
#[ignore = "Finding 1 (jit): C backend lower_host_expr_kind drops jit; fix in fix/jit-par-runtime-arms"]
fn build_c_jit_tensor_runs_and_prints_value() {
    // The host-lane lowerer (`crates/chelis-ir/src/host.rs::lower_host_expr_kind`)
    // has no `jit` arm, so it falls through to `_ => HostExpr::Unit` and the
    // binding's value is silently dropped (`result = ()`). Spec §2.7: jit
    // is a no-op at eval; the compiled binary must print the inner value.
    let dir = tempdir().expect("tempdir");
    let src = write_program(
        dir.path(),
        "tensor_jit.ch",
        "result = jit(to_tensor([1.5, 2.5, 3.5]))\n",
    );
    let out = dir.path().join("out");
    let stdout = build_c_and_run(&out, &src);
    let elements = parse_named_tensor_data(&stdout, "result")
        .unwrap_or_else(|| panic!("expected tensor 'result' in compiled stdout: {stdout}"));
    assert_eq!(elements, vec![1.5, 2.5, 3.5], "stdout={stdout}");
}

#[test]
#[ignore = "Finding 2 (par): C backend lower_host_expr_kind drops par; fix in fix/jit-par-runtime-arms"]
fn build_c_par_tensor_runs_and_prints_last_value() {
    // Same host-lane drop as `jit`: par falls through to `HostExpr::Unit`
    // and the emitted C prints `result = ()`. Spec §2.3 v1 sequential:
    // the value of `par` is the value of the last child.
    let dir = tempdir().expect("tempdir");
    let src = write_program(
        dir.path(),
        "tensor_par.ch",
        "result = par {\n  to_tensor([1.0, 2.0]);\n  to_tensor([3.0, 4.0, 5.0])\n}\n",
    );
    let out = dir.path().join("out");
    let stdout = build_c_and_run(&out, &src);
    let elements = parse_named_tensor_data(&stdout, "result")
        .unwrap_or_else(|| panic!("expected tensor 'result' in compiled stdout: {stdout}"));
    assert_eq!(elements, vec![3.0, 4.0, 5.0], "stdout={stdout}");
}
