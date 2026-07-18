//! chelis#706: bare non-tail expression statements in a block used to
//! silently cross-newline juxtapose into an application (e.g. two
//! `assert_close_tensor(...)` calls collapsed to `f(...)(g(...))`). The
//! parser now rejects them with a targeted diagnostic. These CLI-level
//! tests lock the user-facing surface: `chelis fmt` fails loudly instead
//! of collapsing, and `chelis check` surfaces the diagnostic in its JSON
//! errors array with the errors-exit code.
//!
//! Owning code: `parse_block` in `crates/chelis-surf/src/parser.rs`.

use assert_cmd::prelude::*;
use std::io::Write;
use std::process::Command;

/// Match the constant in `crates/chelis-cli/src/main.rs`.
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

/// The #706 reproducer: two newline-separated calls in a block, the
/// first of which is a bare non-tail statement.
const REPRO: &str = "def f(a: f32, b: f32, c: f32, d: f32) -> f32 = {\n  g(a, b)\n  h(c, d)\n}\n";

fn write_tempfile(prefix: &str, src: &str) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    tmp.write_all(src.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");
    tmp
}

#[test]
fn fmt_rejects_bare_statement_instead_of_collapsing() {
    let tmp = write_tempfile("issue706-fmt-", REPRO);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis fmt");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(
        !output.status.success(),
        "fmt should fail on a bare statement; stdout={stdout} stderr={stderr}"
    );
    // It must not silently emit the collapsed application.
    assert!(
        !stdout.contains("g(a, b)(h(c, d))"),
        "fmt must not collapse the two statements into an application; stdout={stdout}"
    );
    assert!(
        stderr.contains("expression statement must be bound") && stderr.contains("_ ="),
        "fmt should print the targeted #706 diagnostic; stderr={stderr}"
    );
}

#[test]
fn check_surfaces_bare_statement_diagnostic_with_errors_code() {
    let tmp = write_tempfile("issue706-check-", REPRO);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert_eq!(
        output.status.code(),
        Some(CHECK_ERRORS_EXIT_CODE),
        "check should exit {CHECK_ERRORS_EXIT_CODE} on the parse error; stderr={stderr}"
    );
    assert!(
        stdout.contains("expression statement must be bound"),
        "check JSON should carry the #706 diagnostic; stdout={stdout}"
    );
}

#[test]
fn bound_wildcard_fix_parses_and_checks_clean() {
    // The suggested fix — bind the non-tail value with `_ =` — parses and
    // checks cleanly (proves the diagnostic points at a real remedy).
    let fixed = "def f(a: f32, b: f32) -> f32 = {\n  _ = drop(a)\n  b\n}\n";
    let tmp = write_tempfile("issue706-fixed-", fixed);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert_eq!(
        output.status.code(),
        Some(0),
        "the `_ =` fix should check clean; stdout={stdout} stderr={stderr}"
    );
}
