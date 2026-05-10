//! Integration tests for the style gate wired into `chelis build`,
//! `chelis check`, `chelis validate`, and `chelis eval --file`.
//!
//! Each test writes a synthetic source file to a tempdir, runs the
//! relevant CLI subcommand, and asserts pass/fail behavior. Real
//! committed fixtures stay untouched — the gate runs in isolation.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;

/// `chelis check` rejects a source whose formatting does not match the
/// canonical re-print.
#[test]
fn check_fails_on_non_canonical_surf_source() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// `chelis check` rejects a source whose formatting is canonical but
/// which violates a registered lint rule.
#[test]
fn check_fails_on_lint_violation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("leading_underscore.ch");
    // `_internal` parses cleanly but trips `surf-value-snake-case`
    // (§3.2 forbids leading underscore).
    let src = "def _internal() -> i32 = 1\n";
    let decls = chelis_surf::parser::parse_str(src).expect("parses");
    let canonical = chelis_surf::format::format_program(&decls);
    fs::write(&path, &canonical).unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("surf-value-snake-case"));
}

/// Blocking lint rules added to the registry also participate in the
/// built-in style gate.
#[test]
fn check_fails_on_no_em_dash_public_string() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("message.ch");
    let dash = '\u{2014}';
    let src = format!("def message() -> string = \"one {dash} two\"\n");
    let decls = chelis_surf::parser::parse_str(&src).expect("parses");
    let canonical = chelis_surf::format::format_program(&decls);
    fs::write(&path, &canonical).unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("no-em-dash-in-public-strings"));
}

/// Relative bare filenames must still match violations returned by
/// the lint walker, which reports paths rooted at `./`.
#[test]
fn check_relative_bare_filename_fails_on_lint_violation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("message.ch");
    let dash = '\u{2014}';
    let src = format!("def message() -> string = \"one {dash} two\"\n");
    let decls = chelis_surf::parser::parse_str(&src).expect("parses");
    let canonical = chelis_surf::format::format_program(&decls);
    fs::write(&path, &canonical).unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .args(["check", "message.ch"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("no-em-dash-in-public-strings"));
}

/// `--allow-style-violations` lets a non-canonical build through with a
/// stderr warning.
#[test]
fn check_passes_with_allow_style_violations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap(), "--allow-style-violations"])
        .assert()
        // The check command may still fail on type/effect grounds, but
        // the gate-bypass message must appear on stderr regardless.
        .stderr(predicates::str::contains("--allow-style-violations"));
}

/// `chelis build` rejects a non-canonical Surf source before it hits
/// the front-end pipeline.
#[test]
fn build_fails_on_non_canonical_surf_source() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// `chelis build --allow-style-violations` lets the gate-bypass through.
#[test]
fn build_bypass_emits_warning_on_stderr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", path.to_str().unwrap(), "--allow-style-violations"])
        .assert()
        .stderr(predicates::str::contains("style gate bypassed"));
}

/// `chelis eval --file` runs the gate; `chelis eval EXPR` does not.
#[test]
fn eval_file_runs_gate_but_eval_expr_does_not() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));

    // `eval EXPR` is a one-liner with no on-disk source; the gate must
    // not block it. The expression itself is trivial; we just check
    // that the gate-failure message is *absent*.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "1 + 1"])
        .assert()
        .stderr(predicates::str::contains("not canonically formatted").not());
}
