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
        "module Probe.Main\n\ndef noop() -> () = test_assert(true, \"noop\")\n",
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

/// Write a file-level expected-failure probe with no `test_*` declaration.
///
/// The expected-failure adapter must still compile/check the file: the
/// diagnostic itself is the probe outcome. A genuinely clean file remains
/// recordless and therefore a config error.
fn write_file_probe(pkg: &Path, name: &str, body: &str, sidecar: &str) {
    write_probe(pkg, name, "FileProbe", body, Some(sidecar));
}

// ---------------------------------------------------------------- neg mode

#[test]
fn neg_fail_with_substring_is_ok() {
    let (_d, pkg) = make_probe_package("neg-ok");
    write_probe(
        &pkg,
        "case",
        "Case",
        r#"def test_neg_rejects() -> () = test_assert(false, "expected int32, got f32")"#,
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
        r#"def test_neg_rejects() -> () = test_assert(true, "unexpectedly accepted")"#,
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
        r#"def test_neg_rejects() -> () = test_assert(false, "some unrelated failure")"#,
        Some("expected int32\n"),
    );
    run_expect(&pkg, "neg")
        .failure()
        .code(1)
        .stdout(verdict_is("wrong-diagnostic"));
}

#[test]
fn neg_bare_file_check_failure_matches_sidecar() {
    let (_d, pkg) = make_probe_package("neg-file-check");
    write_file_probe(
        &pkg,
        "case",
        r#"def helper() -> int64 = true"#,
        "body doesn't match declared signature\n",
    );
    run_expect(&pkg, "neg").success().stdout(predicate::eq(
        "{\"detail\":\"\",\"expect\":\"neg\",\"file\":\"tests/case.ch\",\"verdict\":\"ok\"}\n\
             {\"summary\":{\"ok\":1,\"failed\":0,\"mode\":\"neg\"}}\n",
    ));
}

#[test]
fn ordinary_testless_compile_mismatch_keeps_legacy_zero_record_behavior() {
    let (_d, pkg) = make_probe_package("ordinary-testless");
    write_file_probe(
        &pkg,
        "case",
        r#"def helper() -> int64 = true"#,
        "body doesn't match declared signature\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args([
            "test",
            "tests/case.ch",
            "--json",
            "--batch-mode",
            "file",
            "--jobs",
            "1",
        ])
        .assert()
        .success()
        .stdout(predicate::eq("{\"summary\":{\"passed\":0,\"failed\":0}}\n"));
}

#[test]
fn neg_bare_file_mismatch_ndjson_preserves_actual_diagnostic() {
    let (_d, pkg) = make_probe_package("neg-file-drift");
    write_file_probe(
        &pkg,
        "case",
        r#"def helper() -> int64 = true"#,
        "diagnostic that must not match\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/", "--expect", "neg", "--json"])
        .output()
        .expect("run neg expected-failure adapter");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let record: serde_json::Value =
        serde_json::from_str(stdout.lines().next().expect("verdict row")).expect("valid NDJSON");
    assert_eq!(record["verdict"], "wrong-diagnostic");
    assert_eq!(
        record["got"],
        serde_json::json!([
            "compile: def 'helper' body doesn't match declared signature: body has type `() -> bool`, declared type is `() -> int64`"
        ])
    );
}

// ------------------------------------------------------------ blocked mode

#[test]
fn blocked_fail_with_substring_is_ok() {
    let (_d, pkg) = make_probe_package("blk-ok");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> () = test_assert(false, "rank mismatch in expand")"#,
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
        r#"def test_blocked_repro() -> () = test_assert(true, "now works")"#,
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
        r#"def test_blocked_repro() -> () = test_assert(false, "a completely different error")"#,
        Some("rank mismatch\nchelis#345\n"),
    );
    run_expect(&pkg, "blocked")
        .failure()
        .code(1)
        .stdout(verdict_is("drifted"));
}

#[test]
fn blocked_bare_file_compile_failure_matches_sidecar_in_plain_output() {
    let (_d, pkg) = make_probe_package("blocked-file-compile");
    write_file_probe(
        &pkg,
        "probe",
        r#"def helper() -> () = missing_file_level_symbol()"#,
        "unbound variable\nchelis#967: preserve bare file diagnostics\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/", "--expect", "blocked"])
        .assert()
        .success()
        .stdout(predicate::eq(
            "OK                 tests/probe.ch\n\n\
             1 ok, 0 failing (blocked mode)\n",
        ));
}

#[test]
fn blocked_bare_file_diagnostic_mismatch_is_drifted_and_preserved() {
    let (_d, pkg) = make_probe_package("blocked-file-mismatch");
    write_file_probe(
        &pkg,
        "probe",
        r#"def helper() -> () = missing_file_level_symbol()"#,
        "some other diagnostic\nchelis#967: preserve bare file diagnostics\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/", "--expect", "blocked"])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::eq(
            "DRIFTED            tests/probe.ch\n\
             \x20   expected diagnostic substring: \"some other diagnostic\"\n\
             \x20   got: compile: unbound variable: missing_file_level_symbol; def 'helper' body doesn't match declared signature: body has type `() -> <error>`, declared type is `() -> ()`\n\n\
             0 ok, 1 failing (blocked mode)\n",
        ));
}

#[test]
fn blocked_bare_file_mismatch_ndjson_preserves_actual_diagnostic() {
    let (_d, pkg) = make_probe_package("blocked-file-drift-json");
    write_file_probe(
        &pkg,
        "probe",
        r#"def helper() -> () = missing_file_level_symbol()"#,
        "some other diagnostic\nchelis#967: preserve bare file diagnostics\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/", "--expect", "blocked", "--json"])
        .output()
        .expect("run blocked expected-failure adapter");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let record: serde_json::Value =
        serde_json::from_str(stdout.lines().next().expect("verdict row")).expect("valid NDJSON");
    assert_eq!(record["verdict"], "drifted");
    assert_eq!(
        record["got"],
        serde_json::json!([
            "compile: unbound variable: missing_file_level_symbol; def 'helper' body doesn't match declared signature: body has type `() -> <error>`, declared type is `() -> ()`"
        ])
    );
}

// ---------------------------------------------------------- fail-closed

#[test]
fn missing_sidecar_is_config_error() {
    let (_d, pkg) = make_probe_package("no-sidecar");
    write_probe(
        &pkg,
        "probe",
        "Probe",
        r#"def test_blocked_repro() -> () = test_assert(false, "rank mismatch")"#,
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
        r#"def test_blocked_repro() -> () = test_assert(false, "rank mismatch")"#,
        Some("rank mismatch\nimplementation convenience\n"), // no auditable citation
    );
    run_expect(&pkg, "blocked")
        .failure()
        .code(1)
        .stdout(verdict_is("config-error"));
}

#[test]
fn clean_bare_file_is_config_error_with_exact_ndjson() {
    let (_d, pkg) = make_probe_package("clean-file");
    write_file_probe(&pkg, "clean", r#"def helper() -> () = ()"#, "must fail\n");
    run_expect(&pkg, "neg")
        .failure()
        .code(1)
        .stdout(predicate::eq(
            "{\"detail\":\"no test records produced (nothing compiled-failed and no test_* functions)\",\"expect\":\"neg\",\"file\":\"tests/clean.ch\",\"verdict\":\"config-error\"}\n\
             {\"summary\":{\"ok\":0,\"failed\":1,\"mode\":\"neg\"}}\n",
        ));
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
        r#"def test_x() -> () = test_assert(false, "boom")"#,
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
