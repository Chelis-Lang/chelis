//! A timed-out `chelis test` row must not keep working during later rows.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::tempdir;

fn package_fixture(dir: &Path, sleep_seconds: &str) -> (PathBuf, PathBuf, PathBuf) {
    let pkg = dir.join("timeout-workers");
    fs::create_dir_all(pkg.join("src")).expect("src directory");
    fs::create_dir_all(pkg.join("tests")).expect("tests directory");
    fs::write(
        pkg.join("reef.toml"),
        format!(
            "schema = \"1\"\n[package]\nname = \"timeout-workers\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"TimeoutWorkers\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    )
    .expect("reef manifest");
    fs::write(pkg.join("src/main.ch"), "module TimeoutWorkers.Main\n").expect("module");

    let late = dir.join("late.txt");
    let fast = dir.join("fast.txt");
    let late_literal = serde_json::to_string(&late.to_string_lossy()).expect("late path literal");
    let fast_literal = serde_json::to_string(&fast.to_string_lossy()).expect("fast path literal");
    fs::write(
        pkg.join("tests/timeout.ch"),
        format!(
            "module TimeoutWorkers.Tests.Timeout\n\
             def test_a_slow() -> unit ! {{ Test, IO }} = {{\n\
               _ = process_run(\"sleep\", [\"{sleep_seconds}\"])\n\
               write_file({late_literal}, \"continued\")\n\
             }}\n\
             def test_b_fast() -> unit ! {{ Test, IO }} = {{\n\
               _ = process_run(\"sleep\", [\"1\"])\n\
               write_file({fast_literal}, \"ran\")\n\
             }}\n"
        ),
    )
    .expect("test source");
    (pkg, late, fast)
}

fn run_case(pkg: &Path, mode: &str, json: bool) -> std::process::Output {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.timeout(Duration::from_secs(20))
        .current_dir(pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "test",
            "tests/timeout.ch",
            "--timeout",
            "2",
            "--suite-timeout",
            "30",
            "--jobs",
            "1",
            "--batch-mode",
            mode,
        ]);
    if json {
        cmd.arg("--json");
    }
    cmd.output().expect("run test case")
}

fn json_rows(output: &std::process::Output) -> Vec<Value> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("each stdout line is JSON"))
        .collect()
}

#[test]
fn cooperative_timeout_stops_old_eval_before_next_row_in_file_text_mode() {
    let dir = tempdir().expect("tempdir");
    let (pkg, late, fast) = package_fixture(dir.path(), "2.5");
    let output = run_case(&pkg, "file", false);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.lines()
            .any(|line| line.contains("test_a_slow") && line.contains("FAIL (timeout after 2s)")),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|line| line.contains("test_b_fast") && line.contains("PASS")),
        "{text}"
    );
    assert!(text.contains("1 passed, 1 failed"), "{text}");
    assert!(
        !late.exists(),
        "timed-out evaluator wrote after its verdict"
    );
    assert_eq!(fs::read_to_string(fast).expect("fast marker"), "ran");
}

#[test]
fn cooperative_timeout_stops_old_eval_before_next_row_in_batch_json_mode() {
    let dir = tempdir().expect("tempdir");
    let (pkg, late, fast) = package_fixture(dir.path(), "2.5");
    let output = run_case(&pkg, "auto", true);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let rows = json_rows(&output);
    assert!(
        rows.iter().all(|row| row.get("batch_fallback").is_none()),
        "expected the batch worker: {rows:?}"
    );
    assert!(
        rows.iter().any(|row| {
            row["test"] == "test_a_slow"
                && row["status"] == "fail"
                && row["message"] == "timeout after 2s"
        }),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row["test"] == "test_b_fast" && row["status"] == "pass"),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row["summary"]["passed"] == 1 && row["summary"]["failed"] == 1),
        "{rows:?}"
    );
    assert!(
        !late.exists(),
        "timed-out evaluator wrote after its verdict"
    );
    assert_eq!(fs::read_to_string(fast).expect("fast marker"), "ran");
}

#[test]
fn blocking_timeout_marks_remaining_rows_unrun_in_file_json_mode() {
    let dir = tempdir().expect("tempdir");
    let (pkg, late, fast) = package_fixture(dir.path(), "10");
    let output = run_case(&pkg, "file", true);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let rows = json_rows(&output);
    assert!(
        rows.iter().any(|row| {
            row["test"] == "test_a_slow"
                && row["status"] == "fail"
                && row["message"] == "timeout after 2s"
        }),
        "{rows:?}"
    );
    assert!(
        rows.iter().any(|row| {
            row["test"] == "test_b_fast"
                && row["status"] == "fail"
                && row["message"] == "unrun after timed-out test did not stop"
        }),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row["summary"]["passed"] == 0 && row["summary"]["failed"] == 2),
        "{rows:?}"
    );
    assert!(!late.exists(), "blocking evaluator wrote after worker exit");
    assert!(!fast.exists(), "unrun row executed");
}

#[test]
fn blocking_timeout_marks_remaining_rows_unrun_in_batch_text_mode() {
    let dir = tempdir().expect("tempdir");
    let (pkg, late, fast) = package_fixture(dir.path(), "10");
    let output = run_case(&pkg, "auto", false);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        !text.contains("batch abandoned"),
        "expected the batch worker: {text}"
    );
    assert!(
        text.lines()
            .any(|line| line.contains("test_a_slow") && line.contains("FAIL (timeout after 2s)")),
        "{text}"
    );
    assert!(
        text.lines().any(|line| {
            line.contains("test_b_fast")
                && line.contains("FAIL (unrun after timed-out test did not stop)")
        }),
        "{text}"
    );
    assert!(text.contains("0 passed, 2 failed"), "{text}");
    assert!(!late.exists(), "blocking evaluator wrote after worker exit");
    assert!(!fast.exists(), "unrun row executed");
}
