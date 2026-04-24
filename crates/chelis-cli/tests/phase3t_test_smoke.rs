//! Phase 3t.4 — smoke tests for the `chelis test` CLI subcommand.
//!
//! These are sanity tests proving the pipe works: discovery, per-test
//! isolation, `--filter`, `--json`, and exit-code shape for the runner-error
//! paths. Exhaustive coverage (timeouts, typecheck failures, mixed
//! modules, full pseudo-nautilus fixture) is 3t.5's job.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

/// Create a bare reef package with a `tests/` directory and return
/// `(tempdir, package_root)`. No chelis-std dep — tests call
/// `test_assert_*` as direct runtime builtins.
fn make_reef_package(dir_name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(dir_name);
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "=0.2.3"
module_prefix = "Smoke"
"#
        ),
    );
    // Reef requires at least one source file to resolve a module graph.
    write_file(
        &pkg.join("src/main.ch"),
        "module Smoke.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    (dir, pkg)
}

#[test]
fn chelis_test_passing_file_exits_zero() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-pass");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.Pass

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_ok"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

#[test]
fn chelis_test_failing_file_exits_one_with_message() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-fail");
    write_file(
        &pkg.join("tests/fail.ch"),
        r#"module Smoke.Tests.Fail

def test_bad() -> unit = test_assert(false, "boom")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("test_bad"))
        .stdout(predicate::str::contains("FAIL"))
        .stdout(predicate::str::contains("assert failed: boom"))
        .stdout(predicate::str::contains("0 passed, 1 failed"));
}

#[test]
fn chelis_test_filter_narrows_selection() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-filter");
    write_file(
        &pkg.join("tests/many.ch"),
        r#"module Smoke.Tests.Many

def test_alpha() -> unit = test_assert(true, "a")
def test_beta() -> unit = test_assert(false, "b failed")
def test_gamma() -> unit = test_assert(true, "c")
"#,
    );

    // Filter picks a single test → suite passes.
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "--filter", "alpha", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_alpha"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("1 passed, 0 failed"))
        // Filter must drop test_beta and test_gamma entirely.
        .stdout(predicate::str::contains("test_beta").not())
        .stdout(predicate::str::contains("test_gamma").not());

    // Filter that matches nothing → no counts, runner exits 0 with "0 passed, 0 failed".
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "--filter", "nonexistent_xyz", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("0 passed, 0 failed"));
}

#[test]
fn chelis_test_json_emits_ndjson_records() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-json");
    write_file(
        &pkg.join("tests/mixed.ch"),
        r#"module Smoke.Tests.Mixed

def test_one() -> unit = test_assert(true, "one")
def test_two() -> unit = test_assert(false, "two broken")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "--json", "tests/"])
        .assert()
        .failure()
        .code(1)
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        3,
        "expected 2 test rows + 1 summary row, got {lines:?}"
    );
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("line not valid JSON: {line:?} ({e})"));
    }
    // First row: passing test with no `message` field.
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("first record parses");
    assert_eq!(first["test"], "test_one");
    assert_eq!(first["status"], "pass");
    assert!(first.get("message").is_none());
    // Second row: failing test carries its error string.
    let second: serde_json::Value = serde_json::from_str(lines[1]).expect("second record parses");
    assert_eq!(second["test"], "test_two");
    assert_eq!(second["status"], "fail");
    assert_eq!(second["message"], "assert failed: two broken");
    // Final row: summary.
    let summary: serde_json::Value = serde_json::from_str(lines[2]).expect("summary parses");
    assert_eq!(summary["summary"]["passed"], 1);
    assert_eq!(summary["summary"]["failed"], 1);
}

#[test]
fn chelis_test_missing_tests_dir_exits_two() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-nodir");
    // Remove the tests dir so the default path does not resolve.
    fs::remove_dir_all(pkg.join("tests")).expect("remove tests dir");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("does not exist"));
}

#[test]
fn chelis_test_non_reef_context_exits_two() {
    // Fresh tempdir with NO reef.toml.
    let dir = tempdir().expect("tempdir");
    let tests = dir.path().join("tests");
    fs::create_dir_all(&tests).expect("mkdir tests");
    write_file(
        &tests.join("empty.ch"),
        "def test_nothing() -> unit = test_assert(true, \"noop\")\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .args(["test", "tests/"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("reef"));
}
