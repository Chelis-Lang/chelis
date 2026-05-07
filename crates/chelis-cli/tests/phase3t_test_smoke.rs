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
compiler = "={ver}"
module_prefix = "Smoke"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
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

// === RT3 regression tests ===

#[test]
fn chelis_test_empty_tests_dir_exits_zero_with_zero_zero_summary() {
    // RT3 H5: empty `tests/` must exit 0 with "0 passed, 0 failed", not
    // exit 2. Matches `cargo test` and `pytest` ergonomics — a fresh
    // package with no tests yet is a legitimate state, not a runner error.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-empty");
    // tests/ exists but has no .ch files. Ensure it is truly empty.
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .code(0)
        .stdout(predicate::str::contains("0 passed, 0 failed"));
}

#[test]
fn chelis_test_hidden_dotfile_is_skipped() {
    // RT3 M2: editor temp files and other dot-prefixed files must not be
    // enumerated as tests — avoids accidentally running an editor's backup
    // or a hidden fixture.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-dotfile");
    write_file(
        &pkg.join("tests/.secret.ch"),
        r#"module Smoke.Tests.Hidden

def test_should_not_run() -> unit = test_assert(false, "hidden should be skipped")
"#,
    );
    // A real test file alongside the dotfile so we still have something to run.
    write_file(
        &pkg.join("tests/visible.ch"),
        r#"module Smoke.Tests.Visible

def test_visible() -> unit = test_assert(true, "ok")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_visible"))
        .stdout(predicate::str::contains("test_should_not_run").not())
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

#[test]
fn chelis_test_non_unit_returning_def_is_not_enumerated() {
    // RT3 H4: `def test_x : bool = true` parses as a zero-param FunDef but
    // is not a test — it's a typed value binding with non-unit type. The
    // enumerator must skip it so the real `def test_real() -> unit`
    // alongside it runs and passes.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-non-unit");
    write_file(
        &pkg.join("tests/mixed.ch"),
        r#"module Smoke.Tests.Mixed

def test_x : bool = true

def test_real() -> unit = test_assert(true, "real test runs")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The real test must PASS. The value-binding `test_x` must not appear
    // as its own row — enumerator must have skipped it.
    assert!(
        stdout.contains("test_real"),
        "test_real missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("PASS"),
        "PASS missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    // `test_x` must not appear as an enumerated test row — a cascade row
    // (e.g., `test_x ... FAIL (module-init failed)`) would imply it was
    // enumerated. Check that the only lines mentioning `test_x` (if any)
    // are not test-result rows.
    for line in stdout.lines() {
        if line.contains("test_x") {
            assert!(
                !line.contains("PASS") && !line.contains("FAIL"),
                "test_x appeared as a test row; got:\nstdout={stdout}\nstderr={stderr}"
            );
        }
    }
    assert!(
        stdout.contains("1 passed, 0 failed"),
        "summary missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(output.status.success(), "expected exit 0");
}

#[test]
fn chelis_test_duplicate_test_name_is_reported_as_file_level_error() {
    // RT3 H3: two `def test_foo()` in the same file is Chelis-level
    // shadowing and will execute the second body silently. Test runner
    // must surface this as a fatal enumeration error attributed to the
    // file, not pretend both ran.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-dup");
    write_file(
        &pkg.join("tests/dup.ch"),
        r#"module Smoke.Tests.Dup

def test_foo() -> unit = test_assert(true, "first")

def test_foo() -> unit = test_assert(false, "second")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("duplicate test definition"));
}

#[test]
fn chelis_test_module_level_bad_binding_cascades_briefly() {
    // RT3 H2 variant: a failing module-level binding (a `let _bad = ...`
    // that throws during eval) surfaces as a single module-init row with
    // the full error detail, and per-test rows only carry the brief
    // "module-init failed" cascade marker — the full error text is NOT
    // repeated per test. This exercises the module-init pre-check path.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-mod-init-cascade");
    write_file(
        &pkg.join("tests/bad.ch"),
        r#"module Smoke.Tests.Bad

_init_failure = test_assert(false, "module init broken")

def test_one() -> unit = test_assert(true, "would have passed 1")

def test_two() -> unit = test_assert(true, "would have passed 2")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The full module-init error detail must appear exactly once across
    // all rows — not repeated per test (that was the cascade bug).
    let full_err_count = stdout.matches("module init broken").count();
    assert_eq!(
        full_err_count, 1,
        "full module-init error should appear once, not per-test; got:\n{stdout}"
    );
    // Per-test rows carry the brief marker.
    let cascade_count = stdout.matches("module-init failed").count();
    assert_eq!(
        cascade_count, 2,
        "two tests should each carry the brief cascade marker; got:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn chelis_test_module_init_failure_surfaces_as_module_init_row_and_cascades() {
    // RT3 H1: a top-level `let _ = assert_*(...)` that fails must surface
    // as its own `module-init` row and cascade every discovered test to
    // FAIL with a "module-init failed" message. Matches the spec's
    // worked output example.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-moduleinit");
    write_file(
        &pkg.join("tests/init.ch"),
        r#"module Smoke.Tests.Init

_sanity = test_assert(false, "module init fails here")

def test_one() -> unit = test_assert(true, "would have passed one")

def test_two() -> unit = test_assert(true, "would have passed two")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("module-init"),
        "module-init row missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("test_one") && stdout.contains("test_two"),
        "cascade-failed tests missing from:\n{stdout}"
    );
    // All three rows are FAIL (module-init plus two cascades).
    let fail_count = stdout.matches("FAIL").count();
    assert!(
        fail_count >= 3,
        "expected module-init + 2 cascades = 3+ FAIL rows; got {fail_count} in:\n{stdout}"
    );
    assert!(
        stdout.contains("module-init failed"),
        "cascade message missing from:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

// === Phase 3t.5 extras: richer failure-mode coverage ===

#[test]
fn chelis_test_infinite_recursion_times_out_and_suite_continues() {
    // 3t.5 coverage: an infinite loop in one test should surface as a FAIL
    // carrying the timeout message, not hang the suite. A sibling passing
    // test must still run and pass so the runner is proven to proceed past
    // the timed-out worker. The timeout is the `--timeout` override so we do
    // not have to wait the default 30s.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-timeout");
    write_file(
        &pkg.join("tests/loopy.ch"),
        r#"module Smoke.Tests.Loopy

-- A fold over a multi-million-element range is a stack-safe way to trip
-- the per-test timeout without blowing the worker stack (native
-- recursion with no base case overflows the 32 MB test-worker stack
-- before the timeout deadline fires). 10_000_000 iterations consistently
-- exceeds the --timeout 2 budget used below.
def test_infinite() -> unit = test_assert(eq(fold(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), range(cast(0, int64), cast(10000000, int64))), cast(0, int64)), "never")

def test_quick() -> unit = test_assert(true, "quick")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "--timeout", "2", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The looping test must appear as FAIL with a timeout-flavored message.
    assert!(
        stdout.contains("test_infinite"),
        "test_infinite missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("timeout"),
        "timeout message missing from:\n{stdout}"
    );
    // The sibling test must have PASSed — the runner must not bail after one
    // timed-out test.
    assert!(
        stdout.contains("test_quick"),
        "test_quick missing from:\n{stdout}"
    );
    let pass_count = stdout.matches("PASS").count();
    assert!(
        pass_count >= 1,
        "expected at least one PASS (test_quick); got {pass_count} in:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn chelis_test_missing_import_is_reported_as_file_level_error() {
    // 3t.5 coverage: a test file that imports a module that does not exist
    // in the reef graph must report a file-level error row and continue to
    // the next file. Specifically: the compile error must surface, and the
    // runner must exit 1 (test failure) — not exit 2 (runner error), because
    // the file was enumerated successfully; it is one test file's content
    // that is broken, not the runner state.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-missing-import");
    write_file(
        &pkg.join("tests/ghost.ch"),
        r#"module Smoke.Tests.Ghost

import Nonexistent.Module (imaginary_helper)

def test_uses_ghost() -> unit = test_assert(true, "would never compile")
"#,
    );
    // Sibling file that is fine — runner must keep going past the broken
    // file, exercise this one, and still exit 1 because of the first.
    write_file(
        &pkg.join("tests/ok.ch"),
        r#"module Smoke.Tests.Ok

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The broken file must produce a FAIL row that references either the
    // compile error or the unresolved module name.
    assert!(
        stdout.contains("ghost.ch"),
        "ghost.ch missing from output:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("FAIL"),
        "FAIL row missing from:\nstdout={stdout}"
    );
    // The healthy sibling file must still report its PASS row.
    assert!(
        stdout.contains("test_ok"),
        "sibling test_ok missing from:\nstdout={stdout}"
    );
    assert!(
        stdout.contains("PASS"),
        "PASS row missing from:\nstdout={stdout}"
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1 (failing tests), got {:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );
}

/// Bucket 6c: `chelis test path/to/file.ch` must succeed when invoked from
/// any cwd, by walking up the target file's ancestry to find `reef.toml`.
/// The pre-fix behavior required the cwd to be inside the reef package,
/// which forced workflows like `cd packages/foo && chelis test tests/x.ch`.
#[test]
fn chelis_test_resolves_reef_root_from_target_file_path() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-cwd-anywhere");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.Pass

def test_from_anywhere() -> unit = test_assert(true, "ok")
"#,
    );
    let test_file = pkg.join("tests/pass.ch");

    // Use the OS temp dir as the cwd. The temp-dir boundary in
    // `find_package_root_from_dir` ensures we don't accidentally pick up a
    // stray reef.toml from /tmp; we want the resolution to come from the
    // *target file's* directory walk, not from the cwd.
    let foreign_cwd = std::env::temp_dir();
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&foreign_cwd)
        .args(["test", test_file.to_str().expect("utf-8 path")])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_from_anywhere"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

/// Bucket 6c negative parity: `chelis test some_file.ch` from a foreign
/// cwd, where the target also has no reef.toml in its ancestry, must
/// produce the "no reef.toml found" error rather than silently succeeding
/// or panicking.
#[test]
fn chelis_test_errors_clearly_when_no_reef_anywhere() {
    let dir = tempdir().expect("tempdir");
    let test_file = dir.path().join("orphan.ch");
    write_file(
        &test_file,
        "module Orphan\n\ndef test_x() -> unit = test_assert(true, \"x\")\n",
    );
    let foreign_cwd = std::env::temp_dir();
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&foreign_cwd)
        .args(["test", test_file.to_str().expect("utf-8 path")])
        .output()
        .expect("run");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "expected failure when no reef found; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("reef.toml") || stdout.contains("reef.toml"),
        "expected reef.toml mention in error; stdout={stdout}\nstderr={stderr}"
    );
}
