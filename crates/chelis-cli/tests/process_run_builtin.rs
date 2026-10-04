//! CLI surface coverage for the `process_run` subprocess-exec builtin.
//!
//! `process_run(cmd, args) -> (exit_code, stdout, stderr)` carries the `IO`
//! effect and runs identically under `chelis eval` and compiled C: both lanes
//! call the runtime's one definition (chelis#1297).
//!
//! Negative-test parity is intentional here: every accepted capture has a
//! failing twin (spawn failure, captures that are not UTF-8), and each
//! failure is the same language trap on both lanes.

#[path = "common/host_effect_parity.rs"]
mod parity;

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

/// Passing a non-list value where `args: List[String]` is expected is a type
/// error the checker reports before evaluation (chelis#2524: [05-OP-38] makes
/// the signature exact), never a panic. Mirrors the missing-binary failure on
/// the argument-shape axis.
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
        stderr.contains("type mismatch: List string vs string"),
        "expected the checker's argument-type error, got stderr={stderr}"
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

/// Compiled C runs `process_run` exactly as eval does: the captured exit
/// code, stdout and stderr, a signal's `-1`, and every call in program order,
/// including one whose result is discarded.
#[test]
fn compiled_process_run_matches_eval_in_value_and_effect_order() {
    let source = "def log_line(text: string) -> unit ! {IO} = {\n\
                  _ = process_run(\"sh\", [\"-c\", string_concat(\"printf \", string_concat(text, \" >> log\"))])\n\
                  ()\n\
                  }\n\
                  echo = process_run(\"echo\", [\"hi\", \"$HOME\"])\n\
                  failing = process_run(\"sh\", [\"-c\", \"printf err >&2; exit 3\"])\n\
                  signalled = process_run(\"sh\", [\"-c\", \"kill -9 $$\"])\n\
                  first = log_line(\"1\")\n\
                  second = log_line(\"2\")\n\
                  contents = process_run(\"cat\", [\"log\"])\n";
    let run = parity::assert_lanes_agree(source, "runner");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "echo.0 = 0\n",
        "echo.1 = hi $HOME\n",
        "failing.0 = 3\n",
        "failing.2 = err\n",
        "signalled.0 = -1\n",
        "contents.1 = 12\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
}

/// The failing twins: a missing program and a capture that is not UTF-8
/// fail the whole call with the same trap on both lanes. stdout is checked
/// before stderr, and no replacement character reaches a result.
#[test]
fn compiled_process_run_failures_match_eval() {
    for (name, source, failure) in [
        (
            "missing",
            "out = process_run(\"definitely_not_a_binary_xyz\", [])\n",
            "process_run failed to spawn `definitely_not_a_binary_xyz`",
        ),
        (
            "badout",
            "out = process_run(\"printf\", [\"\\\\377\"])\n",
            "IO trap in process_run: program `printf`, stdout: output is not valid UTF-8",
        ),
        (
            "baderr",
            "out = process_run(\"sh\", [\"-c\", \"printf ok; printf '\\\\377' >&2\"])\n",
            "IO trap in process_run: program `sh`, stderr: output is not valid UTF-8",
        ),
        (
            "badboth",
            "out = process_run(\"sh\", [\"-c\", \"printf '\\\\377'; printf '\\\\377' >&2\"])\n",
            "IO trap in process_run: program `sh`, stdout: output is not valid UTF-8",
        ),
    ] {
        let run = parity::assert_lanes_agree(source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert!(
            run.failure.starts_with(failure),
            "{name}: expected `{failure}`, got {run:?}"
        );
        assert!(!run.stdout.contains('\u{fffd}'), "{name}: {run:?}");
    }
}

/// `chelis check` (front-end type/effect pass) accepts a `process_run`
/// program.
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
