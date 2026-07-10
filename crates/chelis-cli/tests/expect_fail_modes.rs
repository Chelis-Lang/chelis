//! Phase 1 oracle for `chelis test --expect <neg|blocked>`.
//!
//! The classification logic is unit-tested exhaustively in
//! `chelis_conformance::expect`; this drives the whole pipe end-to-end through
//! the compiled binary — discovery, per-file isolated execution, sidecar
//! correlation, verdict, and exit code — asserting every verdict in both
//! directions plus the fail-closed config-error paths.
//!
//! Each probe drives a controlled diagnostic via `test_assert(false, "<msg>")`
//! (a runtime failure whose message is `assert failed: <msg>`), so the required
//! `.expect` substring is exact and does not depend on compiler wording.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, contents).expect("write file");
}

/// A bare reef package with a `tests/` dir. Probes call `test_assert` as a
/// direct runtime builtin, so no chelis-std dep is needed.
fn make_probe_package(dir_name: &str) -> (tempfile::TempDir, PathBuf) {
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
module_prefix = "Probe"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &pkg.join("src/main.ch"),
        "module Probe.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    (dir, pkg)
}

/// Write a probe `.ch` + its `.expect` sidecar under `tests/`.
fn write_probe(pkg: &Path, name: &str, module: &str, body: &str, sidecar: Option<&str>) {
    write_file(
        &pkg.join(format!("tests/{name}.ch")),
        &format!("module Probe.Tests.{module}\n\n{body}\n"),
    );
    if let Some(s) = sidecar {
        write_file(&pkg.join(format!("tests/{name}.expect")), s);
    }
}

fn run_expect(pkg: &Path, mode: &str) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(pkg)
        .args(["test", "tests/", "--expect", mode, "--json"])
        .assert()
}

fn verdict_is(tag: &str) -> predicates::str::ContainsPredicate {
    predicate::str::contains(format!("\"verdict\":\"{tag}\""))
}

// ---------------------------------------------------------------- neg mode

#[test]
fn neg_fail_with_substring_is_ok() {
    let (_d, pkg) = make_probe_package("neg-ok");
    write_probe(
        &pkg,
        "case",
        "Case",
        r#"def test_neg_rejects() -> unit = test_assert(false, "expected int32, got f32")"#,
        Some("expected int32\n"),
    );
    run_expect(&pkg, "neg").success().stdout(verdict_is("ok"));
}

#[test]
fn neg_pass_is_should_have_failed() {
    let (_d, pkg) = make_probe_package("neg-regress");
    write_probe(
        &pkg,
        "case",
        "Case",
        r#"def test_neg_rejects() -> unit = test_assert(true, "unexpectedly accepted")"#,
        Some("expected int32\n"),
    );
    run_expect(&pkg, "neg")
        .failure()
        .code(1)
        .stdout(verdict_is("should-have-failed"));
}

#[test]
fn neg_fail_wrong_diagnostic() {
    let (_d, pkg) = make_probe_package("neg-wrong");
    write_probe(
        &pkg,
        "case",
        "Case",
        r#"def test_neg_rejects() -> unit = test_assert(false, "some unrelated failure")"#,
        Some("expected int32\n"),
    );
    run_expect(&pkg, "neg")
        .failure()
        .code(1)
        .stdout(verdict_is("wrong-diagnostic"));
}

// ------------------------------------------------------------ blocked mode

#[test]
fn blocked_fail_with_substring_is_ok() {
    let (_d, pkg) = make_probe_package("blk-ok");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> unit = test_assert(false, "rank mismatch in expand")"#,
        Some("rank mismatch\nchelis#345: promote to tests/ on fix\n"),
    );
    run_expect(&pkg, "blocked")
        .success()
        .stdout(verdict_is("ok"));
}

#[test]
fn blocked_pass_is_fix_detected() {
    let (_d, pkg) = make_probe_package("blk-fix");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> unit = test_assert(true, "now works")"#,
        Some("rank mismatch\nchelis#345: promote to tests/ on fix\n"),
    );
    run_expect(&pkg, "blocked")
        .failure()
        .code(1)
        .stdout(verdict_is("fix-detected"));
}

#[test]
fn blocked_fail_wrong_diagnostic_is_drifted() {
    let (_d, pkg) = make_probe_package("blk-drift");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> unit = test_assert(false, "a completely different error")"#,
        Some("rank mismatch\nchelis#345\n"),
    );
    run_expect(&pkg, "blocked")
        .failure()
        .code(1)
        .stdout(verdict_is("drifted"));
}

// ---------------------------------------------------------- fail-closed

#[test]
fn missing_sidecar_is_config_error() {
    let (_d, pkg) = make_probe_package("no-sidecar");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> unit = test_assert(false, "rank mismatch")"#,
        None, // no .expect
    );
    run_expect(&pkg, "blocked")
        .failure()
        .code(1)
        .stdout(verdict_is("config-error"));
}

#[test]
fn blocked_without_citation_is_config_error() {
    let (_d, pkg) = make_probe_package("blk-uncited");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> unit = test_assert(false, "rank mismatch")"#,
        Some("rank mismatch\nimplementation convenience\n"), // no auditable citation
    );
    run_expect(&pkg, "blocked")
        .failure()
        .code(1)
        .stdout(verdict_is("config-error"));
}

// ------------------------------------------------ empty suite / filter guards

#[test]
fn empty_expect_suite_is_rejected() {
    // A `tests/` dir with zero `.ch` probes must not report green under
    // --expect: a guard that runs nothing is silently disabled.
    let (_d, pkg) = make_probe_package("empty-suite");
    run_expect(&pkg, "neg")
        .failure()
        .stderr(predicate::str::contains("non-empty suite"));
}

#[test]
fn filter_with_expect_is_rejected() {
    let (_d, pkg) = make_probe_package("filter-expect");
    write_probe(
        &pkg,
        "case",
        "Case",
        r#"def test_x() -> unit = test_assert(false, "boom")"#,
        Some("boom\n"),
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/", "--expect", "neg", "--filter", "nomatch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be combined"));
}
