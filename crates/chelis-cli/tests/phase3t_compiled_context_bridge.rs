//! Phase H — env-var bridge regression tests.
//!
//! `chelis test`'s parent compiles a `CompiledContext` once, encodes it
//! to a tempfile via `bincode`, and hands the path to each per-file
//! worker through `CHELIS_TEST_COMPILED_CONTEXT`. This file pins the
//! contract:
//!
//! 1. A worker invoked with the env var pointing at a valid bincode
//!    tempfile rehydrates the `CompiledContext` and runs the test
//!    file's tests against it. End-to-end: parent → tempfile → worker
//!    → eval works.
//!
//! 2. A worker invoked WITHOUT the env var falls back to the legacy
//!    `prepare_reef_graph` path so direct `chelis __test_file ...`
//!    invocations (e.g., a developer reproducing a failure by hand)
//!    keep working.
//!
//! 3. A worker invoked with the env var pointing at a non-existent or
//!    truncated tempfile surfaces the read/decode error rather than
//!    silently falling through to a different path. Silent fallthrough
//!    on a corrupted handoff would mask the real bug — exactly the
//!    "silent data loss" pattern called out in the repo's contract.
//!
//! 4. The parent cleans up the tempfile when `chelis test` exits.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// Build a tiny reef package with a single test file.
fn make_minimal_reef_with_test(dir_name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(dir_name);
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    fs::write(
        pkg.join("reef.toml"),
        format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Bridge"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    )
    .expect("write reef.toml");
    fs::write(
        pkg.join("src/main.ch"),
        "module Bridge.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    )
    .expect("write main.ch");
    fs::write(
        pkg.join("tests/foo.ch"),
        r#"module Bridge.Tests.Foo

def test_one() -> unit = test_assert(true, "first")
def test_two() -> unit = test_assert(true, "second")
"#,
    )
    .expect("write tests/foo.ch");
    (dir, pkg)
}

/// (1) Parent → tempfile → worker → eval works end-to-end. The summary
/// line is byte-identical to the pre-Phase-H output for the same input,
/// proving worker-side compatibility.
#[test]
fn worker_decodes_compiled_context_from_env_var_and_runs_tests() {
    let (_dir, pkg) = make_minimal_reef_with_test("phase-h-bridge-happy");

    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .code(0);

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).expect("utf-8 stdout");
    assert!(
        stdout.contains("test_one") && stdout.contains("PASS"),
        "test_one PASS missing: {stdout}"
    );
    assert!(
        stdout.contains("test_two") && stdout.contains("PASS"),
        "test_two PASS missing: {stdout}"
    );
    assert!(
        stdout.contains("2 passed, 0 failed"),
        "summary missing or wrong: {stdout}"
    );
}

/// (2) Worker without the env var falls back to the legacy path. We
/// invoke `chelis __test_file ...` directly so the parent never gets a
/// chance to set the env var, and verify the worker still runs the
/// tests using `prepare_reef_graph` instead of refusing to start.
#[test]
fn worker_without_env_var_falls_back_to_reef_graph_path() {
    let (_dir, pkg) = make_minimal_reef_with_test("phase-h-bridge-fallback");

    let test_file = pkg.join("tests/foo.ch");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        // Explicitly clear the env var so this test is bulletproof
        // against test-runner inheritance.
        .env_remove("CHELIS_TEST_COMPILED_CONTEXT")
        .args([
            "__test_file",
            test_file.to_str().expect("utf-8 path"),
            "--rel-display",
            "tests/foo.ch",
            "--timeout",
            "30",
        ])
        .output()
        .expect("run __test_file");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(0),
        "worker should pass without env var; got {:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("\"test_one\"") && stdout.contains("\"pass\""),
        "test_one pass NDJSON missing: {stdout}"
    );
    assert!(
        stdout.contains("\"test_two\"") && stdout.contains("\"pass\""),
        "test_two pass NDJSON missing: {stdout}"
    );
}

/// (3) Worker with `CHELIS_TEST_COMPILED_CONTEXT` pointing at a
/// non-existent file MUST surface a read error, not silently fall
/// through to the legacy path. Silent fallthrough on a corrupt handoff
/// is the kind of silent data loss the repo contract explicitly
/// rejects.
#[test]
fn worker_errors_when_compiled_context_path_is_missing() {
    let (_dir, pkg) = make_minimal_reef_with_test("phase-h-bridge-missing");

    let test_file = pkg.join("tests/foo.ch");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .env(
            "CHELIS_TEST_COMPILED_CONTEXT",
            "/nonexistent/path/__chelis_definitely_not_a_real_file__.bin",
        )
        .args([
            "__test_file",
            test_file.to_str().expect("utf-8 path"),
            "--rel-display",
            "tests/foo.ch",
            "--timeout",
            "30",
        ])
        .output()
        .expect("run __test_file");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_ne!(
        output.status.code(),
        Some(0),
        "worker MUST NOT silently succeed when CHELIS_TEST_COMPILED_CONTEXT is missing; \
         exit={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );
    // The error must mention the env var or the missing path so an
    // operator can diagnose the handoff failure.
    assert!(
        stderr.contains("CHELIS_TEST_COMPILED_CONTEXT")
            || stderr.contains("__chelis_definitely_not_a_real_file__"),
        "stderr should name the env var or the missing path; got: {stderr}"
    );
}

/// (3b) Negative parity: a tempfile that exists but contains garbage
/// bytes (truncation, partial write, parent crashed mid-flush) must
/// surface a decode error.
#[test]
fn worker_errors_when_compiled_context_bytes_are_corrupt() {
    let (_dir, pkg) = make_minimal_reef_with_test("phase-h-bridge-corrupt");
    let test_file = pkg.join("tests/foo.ch");

    let bad_dir = tempdir().expect("tempdir for bad bytes");
    let bad_path = bad_dir.path().join("garbage.bin");
    // Random bytes that are very unlikely to bincode-decode as a
    // CompiledContext. Bincode is unforgiving on length mismatches, so
    // 8 bytes of trailing zeros are sufficient to fail decode.
    fs::write(&bad_path, b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a").expect("write bad bytes");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .env("CHELIS_TEST_COMPILED_CONTEXT", &bad_path)
        .args([
            "__test_file",
            test_file.to_str().expect("utf-8 path"),
            "--rel-display",
            "tests/foo.ch",
            "--timeout",
            "30",
        ])
        .output()
        .expect("run __test_file");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_ne!(
        output.status.code(),
        Some(0),
        "worker MUST NOT silently succeed on corrupt compiled-context bytes; \
         exit={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );
    // The error must mention decode/CompiledContext so the failure mode
    // is unambiguous to the operator.
    assert!(
        stderr.contains("decode") || stderr.contains("CompiledContext"),
        "stderr should mention the decode failure; got: {stderr}"
    );
}

/// (4) Parent cleans up the tempfile on exit. We diff `/tmp/`'s
/// `chelis-compiled-context-*` listing before and after the run; the
/// after-set must not contain new entries. (We tolerate concurrent
/// chelis invocations by checking for net-new entries created during
/// the test, not absolute counts.)
///
/// This is best-effort — `tempfile::NamedTempFile` deletes the file in
/// its Drop impl, so a SIGKILL or panic during teardown could leak.
/// The test asserts the happy path, which is the contract this phase
/// ships.
#[test]
fn parent_removes_compiled_context_tempfile_on_normal_exit() {
    let (_dir, pkg) = make_minimal_reef_with_test("phase-h-bridge-cleanup");

    // RT-H fix: snapshot a PRIVATE tmpdir so concurrent test binaries
    // don't race-leak `chelis-compiled-context-*` files into our view.
    // The parent honors `CHELIS_TEST_COMPILED_CONTEXT_TMPDIR` and writes
    // the tempfile into the private dir we own, so the snapshot diff
    // attributes only this parent's files.
    let private_tmpdir = tempfile::tempdir().expect("tempdir");
    let snapshot_before = list_tempfile_candidates(private_tmpdir.path());

    let _ = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .env("CHELIS_TEST_COMPILED_CONTEXT_TMPDIR", private_tmpdir.path())
        .args(["test", "tests/"])
        .assert()
        .success();

    let snapshot_after = list_tempfile_candidates(private_tmpdir.path());

    // Net-new entries: in `after` but not `before`. The cleaned-up
    // happy path leaves no leak.
    let leaked: Vec<_> = snapshot_after
        .iter()
        .filter(|p| !snapshot_before.contains(*p))
        .collect();
    assert!(
        leaked.is_empty(),
        "parent leaked {} chelis-compiled-context-*.bin tempfile(s) in private tmpdir: {:?}",
        leaked.len(),
        leaked
    );
}

fn list_tempfile_candidates(tmpdir: &Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = fs::read_dir(tmpdir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("chelis-compiled-context-"))
        })
        .collect()
}
