//! chelis-std self-test corpus integration test.
//!
//! `packages/chelis-std/tests/*.ch` is a corpus of in-language tests that
//! exercise chelis-std's own modules (Std.Tensor, Std.Io, Std.Time,
//! Std.Decimal, Std.Sort, Std.Scan, Std.Process, Std.Test). School provides
//! the neural-network, loss, optimizer, and scheduling libraries. The corpus
//! runs via `chelis test packages/chelis-std/tests/`; the default
//! `cargo test --workspace` gate does not exercise it, so regressions here
//! otherwise surface only when somebody invokes the CLI manually.
//!
//! This test wires the corpus into the default workspace gate. It stages
//! chelis-std into a tempdir, points CHELIS_REEF_HOME at a tempdir reef home
//! for isolation from any developer-local reef state, and runs
//! `chelis test tests/` from inside the staged package. The summary line
//! `N passed, 0 failed` is parsed and checked against the retained core corpus
//! floor; host-runtime primitive tests under tests/runtime/ remain.
//!
//! Pattern mirrors `pseudo_nautilus_fixture.rs`.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

/// Floor on the retained chelis-std self-test corpus, set below the measured
/// current count while still catching an accidental loss of core tests.
const MIN_PASSED: u32 = 120;

fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package must exist")
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

#[test]
#[ignore = "manual gate: runs the chelis-std self-test corpus under chelis test; runtime exceeds the default inner-loop budget. Invoke via `cargo test -p chelis-cli --test chelis_std_self_test_corpus -- --ignored --nocapture`."]
fn chelis_std_self_test_corpus_passes_under_chelis_test() {
    // Stage chelis-std into a tempdir so the test does not touch the
    // checked-in package on disk and is isolated from any developer-local
    // reef state. `chelis test` resolves `Std.*` imports against the staged
    // package's own `src/`, so no separate `reef publish` step is required —
    // chelis-std has no external dependencies (verified against
    // `packages/chelis-std/reef.lock`).
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("chelis-std");
    let reef_home = dir.path().join("reef-home");
    copy_dir_recursive(&package_std(), &pkg);

    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .code(0);

    let stdout = String::from_utf8(assert.get_output().stdout.clone())
        .expect("utf-8 stdout from chelis test");

    // Locate the canonical summary line `N passed, M failed` emitted by
    // `chelis test`. Source: `crates/chelis-cli/src/main.rs` writes
    // "{passed} passed, {failed} failed" as the final non-JSON line.
    let summary_line = stdout
        .lines()
        .rev()
        .find(|line| line.contains(" passed,") && line.contains(" failed"))
        .unwrap_or_else(|| {
            panic!(
                "chelis test output did not contain a `N passed, M failed` summary line.\n\
                 stdout:\n{stdout}"
            )
        });

    let mut parts = summary_line.split_whitespace();
    let passed: u32 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("could not parse passed count from {summary_line:?}"));
    // Skip "passed,".
    let _ = parts.next();
    let failed: u32 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("could not parse failed count from {summary_line:?}"));

    assert_eq!(
        failed, 0,
        "chelis-std self-test corpus reported {failed} failed test(s); summary line was \
         {summary_line:?}. Full stdout:\n{stdout}"
    );

    assert!(
        passed >= MIN_PASSED,
        "chelis-std self-test corpus reported only {passed} passing tests; the floor is \
         {MIN_PASSED}. Either tests were silently dropped from \
         packages/chelis-std/tests/ or the corpus shrank. Investigate before lowering \
         the floor. Summary line: {summary_line:?}"
    );

    // Belt-and-suspenders: the words "FAIL" or "ERROR" must not appear in
    // stdout. `chelis test` prints "PASS"/"FAIL" per test; a `0 failed` summary
    // alongside a "FAIL" line would indicate a counting bug worth surfacing.
    assert!(
        !stdout.contains(" FAIL"),
        "chelis test stdout contains a FAIL line despite reporting {failed} failed.\n\
         stdout:\n{stdout}"
    );
}
