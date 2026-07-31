//! CLI surface coverage for the Hull `process_run` subprocess-exec builtin.
//!
//! `process_run(cmd, args) -> (exit_code, stdout, stderr)` carries the `Io`
//! effect and is eval/test-only: it runs under `chelis eval` / `chelis test`
//! but the C/HIP/Metal build backends reject it with a clean diagnostic
//! rather than the silent `/* unsupported builtin */ 0` fallthrough.
//!
//! Negative-test parity is intentional here:
//!   * positive eval (exit 0 tuple, stdout capture) <-> negative eval
//!     (spawn failure is a clean error, non-zero exit captured in the tuple)
//!   * positive build rejection (message names the eval/test-only gap) is
//!     the failure-side mirror of the eval path working.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

/// PATH with the just-built `chelis` binary's directory prepended, so an inner
/// `process_run("chelis", ...)` resolves the build under test. CI runners have
/// no `chelis` on PATH (the binary lives under `target/`), and a developer box
/// may have a stale `chelis` installed; both cases would otherwise make these
/// tests non-hermetic.
fn path_with_built_chelis() -> String {
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    let dir = bin.parent().expect("chelis binary has a parent directory");
    match std::env::var("PATH") {
        Ok(existing) => format!("{}:{}", dir.display(), existing),
        Err(_) => dir.display().to_string(),
    }
}

/// `process_run("chelis", ["--version"])` exits 0; the tuple renders with
/// `eval_result.0 = 0` and the stdout slot carries the version banner.
#[test]
fn eval_process_run_chelis_version_yields_exit_zero_tuple() {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("PATH", path_with_built_chelis())
        .arg("eval")
        .arg(r#"process_run("chelis", ["--version"])"#)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8_lossy(&output);
    assert!(
        stdout.contains("eval_result.0 = 0"),
        "expected exit code 0 in tuple slot 0, got stdout={stdout}"
    );
    assert!(
        stdout.contains("eval_result.1 = chelis "),
        "expected the version banner in the stdout tuple slot, got stdout={stdout}"
    );
}

/// `process_run("echo", ["hi"])` captures the program's stdout into tuple
/// slot 1 and exits 0.
#[test]
fn eval_process_run_echo_captures_stdout() {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .arg("eval")
        .arg(r#"process_run("echo", ["hi"])"#)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8_lossy(&output);
    assert!(
        stdout.contains("eval_result.0 = 0"),
        "expected echo to exit 0, got stdout={stdout}"
    );
    assert!(
        stdout.contains("eval_result.1 = hi"),
        "expected echo stdout `hi` in tuple slot 1, got stdout={stdout}"
    );
}

/// Spawning a binary that does not exist is a clean evaluation error, not a
/// panic or crash. The error names the failed command and the spawn failure.
#[test]
fn eval_process_run_missing_binary_fails_cleanly() {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .arg("eval")
        .arg(r#"process_run("definitely_not_a_binary_xyz", [])"#)
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("process_run failed to spawn"),
        "expected a clean spawn-failure error, got stderr={stderr}"
    );
    assert!(
        stderr.contains("definitely_not_a_binary_xyz"),
        "spawn-failure error must name the failed command, got stderr={stderr}"
    );
    // A clean error, never a Rust panic.
    assert!(
        !stderr.contains("panicked"),
        "process_run spawn failure must not panic, got stderr={stderr}"
    );
}

/// Passing a non-list value where `args: List[String]` is expected is a clean
/// evaluation error, not a panic. Mirrors the missing-binary failure on the
/// argument-shape axis.
#[test]
fn eval_process_run_rejects_non_list_args() {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .arg("eval")
        .arg(r#"process_run("echo", "hi")"#)
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("expected list arg"),
        "expected a clean list-arg error, got stderr={stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "process_run arg-shape error must not panic, got stderr={stderr}"
    );
}

/// A subprocess that exits non-zero is NOT an evaluation error: the non-zero
/// exit code is captured in tuple slot 0. `chelis check` on a path that does
/// not exist exits non-zero, so the outer eval still succeeds and the inner
/// exit code is a non-zero integer.
#[test]
fn eval_process_run_captures_nonzero_exit_code() {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("PATH", path_with_built_chelis())
        .arg("eval")
        .arg(r#"process_run("chelis", ["check", "no_such_file_zzz.ch"])"#)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8_lossy(&output);
    assert!(
        !stdout.contains("eval_result.0 = 0"),
        "expected a non-zero captured exit code, got stdout={stdout}"
    );
    assert!(
        stdout.contains("eval_result.0 = "),
        "expected the exit code in tuple slot 0, got stdout={stdout}"
    );
}

/// `chelis build` on a program that uses `process_run` must fail with the
/// eval/test-only diagnostic. This is the anti-regression that matters most:
/// without it the silent fallthrough would emit C returning 0 for the tuple,
/// a wrong value rather than a diagnostic.
#[test]
fn build_rejects_process_run_program_with_clear_message() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("runner.ch");
    let out = dir.path().join("out");
    fs::write(&src, "result = process_run(\"echo\", [\"hi\"])\n").expect("write source");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported: builtin `process_run`"),
        "build must reject process_run with the eval/test-only message, got stderr={stderr}"
    );
    assert!(
        stderr.contains("on compiled targets") && stderr.contains("chelis eval"),
        "build rejection must explain process_run is unavailable in compiled targets \
         and name the eval-lane remediation, got stderr={stderr}"
    );
    // The rejection must fire BEFORE any C artifact is written.
    assert!(
        !out.join("runner.c").exists(),
        "build must not emit a C artifact for a rejected process_run program"
    );
}

/// `chelis check` (front-end type/effect pass) ACCEPTS a `process_run`
/// program: the rejection is build-only, not a type error. This is the
/// pass-side mirror of `build_rejects_process_run_program_with_clear_message`.
#[test]
fn check_accepts_process_run_program() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("runner.ch");
    fs::write(&src, "result = process_run(\"echo\", [\"hi\"])\n").expect("write source");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", src.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value =
        serde_json::from_slice(&output).expect("check output should be json");
    assert_eq!(
        json["score"].as_f64(),
        Some(1.0),
        "process_run program should type-check clean, got {json}"
    );
    assert!(
        json["errors"]
            .as_array()
            .map(|e| e.is_empty())
            .unwrap_or(false),
        "process_run program should have no check errors, got {json}"
    );
}
