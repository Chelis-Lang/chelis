//! Phase 3t.A3 — regression test for the per-file subprocess isolation in
//! `chelis test`.
//!
//! Phase 3t shipped subprocess-per-file isolation in `cmd_test`: a worker
//! crash in one test file (stack overflow, SIGABRT, panic in the evaluator)
//! must NOT kill the parent or the sibling test files. This test locks that
//! protection in so it cannot silently rot.
//!
//! Two approaches:
//!
//! 1. **Env-var escape hatch (default):** `cmd_internal_test_file` honors
//!    `CHELIS_TEST_FORCE_ABORT=<substring>` *only when*
//!    `CHELIS_TEST_INTERNAL_TESTING=1` is also set. The substring is
//!    matched against the worker's `--rel-display`; matching workers
//!    abort via `std::process::abort()` before running anything.
//!    Substring-matching (rather than a global `=1`) is necessary because
//!    the worker subprocess inherits the parent's env, so a global flag
//!    would crash *every* worker. We spawn `chelis test tests/` with
//!    both env vars set against a tempdir reef package containing a
//!    `crash.ch` file and a `fine.ch` file, then verify the parent
//!    attributes the crash to the right file, the sibling still PASSES,
//!    and the parent exits 1.
//!
//! 2. **Pathological recursion (`#[ignore]`'d, manual gate):** induce a real
//!    stack overflow inside Chelis evaluation and verify the same isolation
//!    properties without the test hook. Default-`#[ignore]`'d because the
//!    induction depends on evaluator depth and may not reliably overflow in
//!    every build configuration.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

/// Create a bare reef package with a `tests/` directory and return
/// `(tempdir, package_root)`.
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
compiler = "=0.2.4"
module_prefix = "Iso"
"#
        ),
    );
    write_file(
        &pkg.join("src/main.ch"),
        "module Iso.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    (dir, pkg)
}

/// Look up the row for `target_test` (e.g. `test_ok` or `<file>`) appearing
/// after the file header `rel_display` in `chelis test` plain stdout. The
/// formatter prints `<rel_display>` on its own line followed by indented
/// `  test_name ......... STATUS (msg)` rows for every test in that file.
fn row_for_test_in_file<'a>(
    stdout: &'a str,
    rel_display: &str,
    target_test: &str,
) -> Option<&'a str> {
    let mut in_file = false;
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if !line.starts_with(' ') && trimmed == rel_display {
            in_file = true;
            continue;
        }
        // Once we hit another file header (a non-indented, non-empty,
        // non-summary line) we leave the current file's section.
        if !line.starts_with(' ') && !trimmed.is_empty() {
            // Could be the summary line ("N passed, M failed") or another
            // file header. Either way, we are no longer in `rel_display`'s
            // section.
            in_file = false;
            // If this turns out to be another file header equal to
            // rel_display we'd re-enter on the next iteration's check above;
            // do not `break` here so we keep scanning if this line is the
            // summary.
            continue;
        }
        if in_file && trimmed.starts_with(target_test) {
            return Some(trimmed);
        }
    }
    None
}

#[test]
fn worker_crash_in_one_file_does_not_kill_sibling_file() {
    // Approach 1: the env-var escape hatch. `cmd_internal_test_file`
    // aborts immediately when CHELIS_TEST_INTERNAL_TESTING=1 is set AND
    // the worker's `--rel-display` contains the value of
    // CHELIS_TEST_FORCE_ABORT, simulating a worker that died (stack
    // overflow, SIGABRT, panic) before writing any NDJSON. The
    // substring scope is what makes the sibling worker survive even
    // though it inherits the same env.
    let (_dir, pkg) = make_reef_package("phase3t-iso-abort");

    // The "crashing" file. Even though the body would normally pass, the
    // env-var hatch will make the worker abort before evaluating it.
    write_file(
        &pkg.join("tests/crash.ch"),
        r#"module Iso.Tests.Crash

def test_unreachable() -> unit = test_assert(true, "would have passed if worker did not crash")
"#,
    );

    // The "fine" sibling file. Must still run to completion in its OWN
    // worker subprocess and report PASS — that is the whole point of
    // per-file subprocess isolation.
    write_file(
        &pkg.join("tests/fine.ch"),
        r#"module Iso.Tests.Fine

def test_ok() -> unit = test_assert(true, "sibling still runs")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        // Crash only the worker whose --rel-display contains "crash.ch".
        // The sibling fine.ch worker inherits the same env vars but its
        // rel-display does not contain this substring, so it runs normally.
        .env("CHELIS_TEST_FORCE_ABORT", "crash.ch")
        .args(["test", "tests/"])
        .output()
        .expect("run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // (c) Parent exits 1 — failures present.
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1, got {:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );

    // (a) Crashing file is reported as a worker-exit failure attributed to
    // tests/crash.ch with the synthetic `<file>` test name and a message
    // mentioning the worker exited or was killed by signal.
    let crash_row = row_for_test_in_file(&stdout, "tests/crash.ch", "<file>")
        .unwrap_or_else(|| panic!("crash.ch <file> row missing from:\nstdout={stdout}"));
    assert!(
        crash_row.contains("FAIL"),
        "crash.ch <file> row should be FAIL; got: {crash_row}\nfull stdout=\n{stdout}"
    );
    assert!(
        crash_row.contains("worker exited") || crash_row.contains("worker killed by signal"),
        "crash.ch row should mention worker exit/signal; got: {crash_row}\nfull stdout=\n{stdout}"
    );
    // RT-A3 MEDIUM B2: when killed by a signal, the row must name the signal
    // (e.g. "SIGABRT (6)") so SIGABRT vs SIGKILL are distinguishable. The
    // env-hatch uses std::process::abort, which delivers SIGABRT(6).
    if crash_row.contains("worker killed by signal") {
        assert!(
            crash_row.contains("SIGABRT (6)"),
            "signal-killed crash.ch row should name the signal as 'SIGABRT (6)' \
             (RT-A3 MEDIUM B2 — drop signal number was the symptom); \
             got: {crash_row}\nfull stdout=\n{stdout}"
        );
    }

    // (b) Sibling file still ran to completion: tests/fine.ch::test_ok PASS.
    let fine_row = row_for_test_in_file(&stdout, "tests/fine.ch", "test_ok")
        .unwrap_or_else(|| panic!("fine.ch test_ok row missing from:\nstdout={stdout}"));
    assert!(
        fine_row.contains("PASS"),
        "fine.ch test_ok row should be PASS; got: {fine_row}\nfull stdout=\n{stdout}"
    );

    // The unreachable test inside the crashing worker must NOT be reported
    // as PASS — the worker died before running it.
    assert!(
        !stdout.contains("test_unreachable") || !stdout.contains("test_unreachable .... PASS"),
        "test_unreachable must not appear as a PASS row; full stdout=\n{stdout}"
    );

    // Summary line proves the runner counted exactly one passing sibling
    // and one failure (the worker-exit synthetic row).
    assert!(
        stdout.contains("1 passed, 1 failed"),
        "expected '1 passed, 1 failed' summary; got:\n{stdout}"
    );
}

#[test]
fn force_abort_env_requires_both_variables() {
    // Belt-and-suspenders check: setting only CHELIS_TEST_FORCE_ABORT=1
    // (without CHELIS_TEST_INTERNAL_TESTING=1) must NOT trigger the abort.
    // This guards against a production user who happens to set the
    // FORCE_ABORT variable in their environment and would otherwise see
    // their test runs blow up. With the gate intact, the test file runs
    // normally and the suite PASSes.
    let (_dir, pkg) = make_reef_package("phase3t-iso-abort-gate");
    write_file(
        &pkg.join("tests/fine.ch"),
        r#"module Iso.Tests.GateFine

def test_ok() -> unit = test_assert(true, "should still run")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        // Only one of the two — must NOT abort. Use a substring that
        // would otherwise match (so the failure is the gate's behavior,
        // not a missing match).
        .env("CHELIS_TEST_FORCE_ABORT", "fine.ch")
        .env_remove("CHELIS_TEST_INTERNAL_TESTING")
        .args(["test", "tests/"])
        .output()
        .expect("run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(0),
        "single-var gate should not trigger abort; expected exit 0, got {:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("test_ok") && stdout.contains("PASS"),
        "test_ok PASS missing — gate may be broken:\nstdout={stdout}"
    );
    assert!(
        stdout.contains("1 passed, 0 failed"),
        "expected '1 passed, 0 failed' summary; got:\n{stdout}"
    );
}

#[test]
#[ignore = "manual gate: depends on evaluator stack depth; see docs/manual_gates.md"]
fn worker_stack_overflow_in_one_file_does_not_kill_sibling_file() {
    // Approach 2 (manual gate): induce a real stack overflow in the
    // worker via deep Chelis recursion and verify the same isolation
    // properties without using the env-var test hook. Marked `#[ignore]`
    // because the depth needed to overflow the 32 MB worker stack is
    // implementation-dependent and may not reliably overflow in every
    // build/optimization configuration.
    let (_dir, pkg) = make_reef_package("phase3t-iso-overflow");

    write_file(
        &pkg.join("tests/crash.ch"),
        r#"module Iso.Tests.Overflow

def deep(n: int64) -> int64 = if eq(n, cast(0, int64)) then cast(0, int64) else add(cast(1, int64), deep(sub(n, cast(1, int64))))

def test_overflow() -> unit = test_assert(eq(deep(cast(10000000, int64)), cast(10000000, int64)), "would survive")
"#,
    );
    write_file(
        &pkg.join("tests/fine.ch"),
        r#"module Iso.Tests.OverflowFine

def test_ok() -> unit = test_assert(true, "sibling still runs")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        // Use a generous timeout so we are clearly catching a crash, not a
        // timeout-induced FAIL.
        .args(["test", "--timeout", "30", "tests/"])
        .output()
        .expect("run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1; stdout={stdout}\nstderr={stderr}"
    );

    let crash_row = row_for_test_in_file(&stdout, "tests/crash.ch", "<file>")
        .unwrap_or_else(|| panic!("crash.ch <file> row missing from:\nstdout={stdout}"));
    assert!(
        crash_row.contains("FAIL")
            && (crash_row.contains("worker exited")
                || crash_row.contains("worker killed by signal")),
        "crash.ch <file> row should be a worker-exit FAIL; got: {crash_row}\nfull stdout=\n{stdout}"
    );

    let fine_row = row_for_test_in_file(&stdout, "tests/fine.ch", "test_ok")
        .unwrap_or_else(|| panic!("fine.ch test_ok row missing from:\nstdout={stdout}"));
    assert!(
        fine_row.contains("PASS"),
        "fine.ch test_ok row should be PASS; got: {fine_row}\nfull stdout=\n{stdout}"
    );
}
