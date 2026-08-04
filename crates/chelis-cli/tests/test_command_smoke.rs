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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
fn chelis_test_pipe_stage_host_runtime_fallback_does_not_panic() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-pipe-stage");
    write_file(
        &pkg.join("tests/pipe.ch"),
        r#"module Smoke.Tests.Pipe

type Sign =
  | Neg
  | Pos

def choose(s: Sign) -> f32 = match s with {
  | Neg => cast(0.0, f32)
  | Pos => cast(1.0, f32)
}

def pipe_choose(s: Sign) -> f32 = s |> choose

def test_pipe_choose() -> unit = test_assert(pipe_choose(Pos) == cast(1.0, f32), "pipe choose")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/pipe.ch"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_pipe_choose"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("worker exited").not())
        .stdout(predicate::str::contains("panic").not())
        .stderr(predicate::str::contains("panic").not());
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
fn chelis_test_jobs_emits_json_in_discovery_order() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-jobs");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.First

def test_first() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.Second

def test_second() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--json", "--jobs", "2", "tests/"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(lines[0]["file"], "tests/a_first.ch");
    assert_eq!(lines[0]["test"], "test_first");
    assert_eq!(lines[1]["file"], "tests/b_second.ch");
    assert_eq!(lines[1]["test"], "test_second");
    assert_eq!(lines[2]["summary"]["passed"], 2);
    assert_eq!(lines[2]["summary"]["failed"], 0);
}

#[test]
fn chelis_test_auto_batch_preserves_two_file_ndjson_order() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-order");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.BatchFirst

def test_first() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.BatchSecond

def test_second() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "--json", "tests/"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(lines[0]["file"], "tests/a_first.ch");
    assert_eq!(lines[0]["test"], "test_first");
    assert_eq!(lines[0]["status"], "pass");
    assert_eq!(lines[1]["file"], "tests/b_second.ch");
    assert_eq!(lines[1]["test"], "test_second");
    assert_eq!(lines[1]["status"], "pass");
    assert_eq!(lines[2]["summary"]["passed"], 2);
    assert_eq!(lines[2]["summary"]["failed"], 0);
}

#[test]
fn chelis_test_batch_mode_file_keeps_per_file_output_shape() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-file-mode");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.BatchFileMode

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "file", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tests/pass.ch"))
        .stdout(predicate::str::contains("test_ok"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

#[test]
fn chelis_test_auto_batch_compile_error_falls_back_to_file_rows() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-compile-fallback");
    write_file(
        &pkg.join("tests/a_broken.ch"),
        r#"module Smoke.Tests.BatchBroken

import Missing.Module (ghost)

def test_broken() -> unit = test_assert(true, "unreachable")
"#,
    );
    write_file(
        &pkg.join("tests/b_ok.ch"),
        r#"module Smoke.Tests.BatchOk

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected test failure exit; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("tests/a_broken.ch")
            && stdout.contains("<file>")
            && stdout.contains("FAIL"),
        "broken file-level row missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("tests/b_ok.ch") && stdout.contains("test_ok") && stdout.contains("PASS"),
        "healthy sibling PASS missing from:\nstdout={stdout}\nstderr={stderr}"
    );
}

#[test]
fn chelis_test_auto_batch_name_collision_uses_file_isolation() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-collision");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.CollisionFirst

def test_same() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.CollisionSecond

def test_same() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "--json", "tests/"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let rows: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(rows[0]["file"], "tests/a_first.ch");
    assert_eq!(rows[0]["test"], "test_same");
    assert_eq!(rows[0]["status"], "pass");
    assert_eq!(rows[1]["file"], "tests/b_second.ch");
    assert_eq!(rows[1]["test"], "test_same");
    assert_eq!(rows[1]["status"], "pass");
    assert_eq!(rows[2]["summary"]["passed"], 2);
}

#[test]
fn chelis_test_auto_batch_skips_module_init_files() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-module-init");
    write_file(
        &pkg.join("tests/init.ch"),
        r#"module Smoke.Tests.BatchInit

_init_failure = test_assert(false, "module init failed in fallback")

def test_one() -> unit = test_assert(true, "would pass")
"#,
    );
    write_file(
        &pkg.join("tests/ok.ch"),
        r#"module Smoke.Tests.BatchInitOk

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("module-init"),
        "module-init row missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("module init failed in fallback"),
        "module-init error missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("test_ok") && stdout.contains("PASS"),
        "batchable sibling did not pass:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn chelis_test_auto_batch_worker_abort_falls_back_to_file_workers() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-abort-fallback");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.BatchAbortFirst

def test_first() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.BatchAbortSecond

def test_second() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_FORCE_BATCH_ABORT", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "batch abort should fall back to per-file workers; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("test_first") && stdout.contains("test_second"),
        "fallback did not run both files:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("2 passed, 0 failed"),
        "fallback summary missing:\nstdout={stdout}\nstderr={stderr}"
    );
}

#[test]
fn chelis_test_rejects_zero_jobs() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-zero-jobs");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.Pass

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--jobs", "0", "tests/"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("jobs"));
}

#[test]
fn chelis_test_missing_tests_dir_exits_two() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-nodir");
    // Remove the tests dir so the default path does not resolve.
    fs::remove_dir_all(pkg.join("tests")).expect("remove tests dir");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    // RT3 H4: `def test_x() -> bool = true` is a nullary function but is not
    // a test because its result is non-unit. The
    // enumerator must skip it so the real `def test_real() -> unit`
    // alongside it runs and passes.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-non-unit");
    write_file(
        &pkg.join("tests/mixed.ch"),
        r#"module Smoke.Tests.Mixed

def test_x() -> bool = true

def test_real() -> unit = test_assert(true, "real test runs")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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

/// RT-1 F2-bypass (RFC v6): a reef package test file may NOT use a
/// hand-authored reef-internal mangled name (`pkg__<pkg>__<Module>__<Name>`)
/// that self-keys to a victim module and forges the real opaque type.
/// The `chelis test` entry path re-asserts the linked flag without
/// re-mangling user test/entry decl names, so the forge def self-keys to
/// `Demo.Types` and constructs the opaque type as if in-module. eval /
/// check / build / validate all reject the same forge; this was test-only.
fn make_opaque_reef_package(dir_name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(dir_name);
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            "[package]\nname = \"{dir_name}\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"Smoke\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &pkg.join("src/types.ch"),
        "module Smoke.Types\n\
         export (probability, prob_value)\n\
         @opaque\n\
         type Probability =\n  | Probability { value: f32 }\n\
         def probability(x: f32) -> Probability = Probability { value: x }\n\
         def prob_value(p: Probability) -> f32 = p.value\n",
    );
    (dir, pkg)
}

#[test]
fn chelis_test_rejects_reef_mangled_name_forge() {
    // The package name is `forgepkg`, so a def in module Smoke.Types
    // links to `pkg__forgepkg__Smoke__Types__<name>`. A test file that
    // hand-authors `def pkg__forgepkg__Smoke__Types__forge` self-keys to
    // the victim module `forgepkg.Smoke.Types`, so `Probability { ... }`
    // would be in-module. `chelis test` must REJECT this forge.
    let (_dir, pkg) = make_opaque_reef_package("forgepkg");
    write_file(
        &pkg.join("tests/forge.ch"),
        "module Smoke.Tests.Forge\n\
         import Smoke.Types (prob_value)\n\
         def pkg__forgepkg__Smoke__Types__forge(x: f32) -> f32 = {\n\
         \x20 p = Probability { value: x }\n\
         \x20 prob_value(p)\n\
         }\n\
         def test_forge() -> unit = \
         test_assert(pkg__forgepkg__Smoke__Types__forge(0.5) >= 0.0, \"forge\")\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("1 passed, 0 failed"),
        "the forge must NOT pass `chelis test`; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("reserved internal-name format")
            || stdout.contains("ReservedLinkerName")
            || stdout.contains("outside its defining module"),
        "the forge must be rejected with a module-identity / reserved-name error; \
         stdout={stdout}\nstderr={stderr}"
    );
}

#[test]
fn chelis_test_legit_opaque_package_test_still_passes() {
    // No-regression: a LEGITIMATE test file (exported producers, normal
    // helper names) in the same @opaque package must still pass.
    let (_dir, pkg) = make_opaque_reef_package("legitpkg");
    write_file(
        &pkg.join("tests/use.ch"),
        "module Smoke.Tests.Use\n\
         import Smoke.Types (probability, prob_value)\n\
         def round_trip(x: f32) -> f32 = prob_value(probability(x))\n\
         def test_round_trip() -> unit = test_assert(round_trip(0.5) >= 0.0, \"ok\")\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_round_trip"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}
