//! chelis#1267: canonical Surf v0.19 rejects `;` as a binding-block
//! separator, but every diagnostic that fired on one described something
//! else. Between bindings it read `expected separator (`;` or newline),
//! found Semicolon`, naming the token it had just refused; after the tail it
//! claimed a bare unbound statement; and after a newline it degraded to a
//! bare `unexpected end of input` with no offset at all.
//!
//! The issue was reported through `chelis eval --file`, and generated-Surf
//! shells (which never `fmt` their temporaries) are the population that hits
//! it, so these CLI-level tests lock the user-facing surface rather than the
//! parser's internal error type.
//!
//! Owning code: `parse_block_inner` in `crates/chelis-surf/src/parser.rs`.

use assert_cmd::prelude::*;
use std::io::Write;
use std::process::Command;

/// Match the constant in `crates/chelis-cli/src/main.rs`.
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

/// The issue's reproducer, with `;` between block bindings.
const REPRO: &str = "def main() -> int64 = { a = 1i64; b = 2i64; add(a, b) }\n";

/// The remedy the diagnostic names: the same program, newline separated.
const REMEDY: &str = "def main() -> int64 = {\n  a = 1i64\n  b = 2i64\n  add(a, b)\n}\n";

/// A `;` after the tail expression. spec/02-surf-syntax.md §P5 rejects a
/// trailing `;` by name, but this used to report a bare statement.
const TRAILING: &str = "def main() -> int64 = {\n  a = 1i64\n  add(a, 1i64);\n}\n";

/// A `;` on its own line between bindings, which used to degrade to
/// `unexpected end of input`.
const AFTER_NEWLINE: &str =
    "def main() -> int64 = {\n  a = 1i64\n  ;\n  b = 2i64\n  add(a, b)\n}\n";

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

fn fmt_stderr(prefix: &str, src: &str) -> String {
    let tmp = write_tempfile(prefix, src);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis fmt");
    assert!(
        !output.status.success(),
        "a `;` block separator must fail canonical fmt"
    );
    String::from_utf8(output.stderr).expect("utf8 stderr")
}

#[test]
fn fmt_names_the_newline_rule_and_never_offers_the_semicolon() {
    let stderr = fmt_stderr("issue1267-fmt-", REPRO);
    assert!(
        stderr.contains("expected a newline separator, found `;`"),
        "the newline must be named as the expectation; stderr={stderr}"
    );
    assert!(
        stderr.contains("not a block separator in canonical Surf v0.19"),
        "the rule must be stated; stderr={stderr}"
    );
    // The reported defect: the old wording listed `;` as one of two
    // acceptable separators while refusing that exact token.
    assert!(
        !stderr.contains("separator (`;`"),
        "`;` must not be offered as acceptable; stderr={stderr}"
    );
    // The byte offset lands on the first `;`, as the issue reported.
    let offset = REPRO.find(';').expect("reproducer has a `;`");
    assert!(
        stderr.contains(&format!("(byte {offset})")),
        "diagnostic should point at the first `;` (byte {offset}); stderr={stderr}"
    );
}

#[test]
fn fmt_reports_the_semicolon_rule_after_the_tail_expression() {
    // Previously "expression statement must be bound ... move it to tail
    // position", which is false for this input twice over: the expression IS
    // the tail, and nothing is unbound.
    let stderr = fmt_stderr("issue1267-trailing-", TRAILING);
    assert!(
        stderr.contains("not a block separator in canonical Surf v0.19"),
        "a trailing `;` must name the rule; stderr={stderr}"
    );
    assert!(
        !stderr.contains("expression statement must be bound"),
        "a trailing `;` is not a bare statement; stderr={stderr}"
    );
}

#[test]
fn fmt_reports_the_semicolon_rule_when_a_newline_precedes_it() {
    // Previously a bare "unexpected end of input" carrying no offset, which
    // editors rendered past the end of the file.
    let stderr = fmt_stderr("issue1267-newline-", AFTER_NEWLINE);
    assert!(
        stderr.contains("not a block separator in canonical Surf v0.19"),
        "a newline-preceded `;` must name the rule; stderr={stderr}"
    );
    assert!(
        !stderr.contains("unexpected end of input"),
        "the input does not end here; stderr={stderr}"
    );
    let offset = AFTER_NEWLINE.find(';').expect("fixture has a `;`");
    assert!(
        stderr.contains(&format!("(byte {offset})")),
        "diagnostic should point at the `;` (byte {offset}); stderr={stderr}"
    );
}

#[test]
fn check_surfaces_the_semicolon_diagnostic_with_the_errors_exit_code() {
    let tmp = write_tempfile("issue1267-check-", REPRO);
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
        stdout.contains("not a block separator in canonical Surf v0.19"),
        "check JSON should carry the diagnostic; stdout={stdout}"
    );
}

#[test]
fn the_newline_remedy_the_diagnostic_names_checks_clean() {
    // The diagnostic tells the reader to replace the `;` with a newline.
    // Prove that remedy reaches a clean check, so the advice is not a
    // dead end the way the bare-statement advice was.
    let tmp = write_tempfile("issue1267-remedy-", REMEDY);
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
        "the newline form should check clean; stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn the_migrator_the_diagnostic_names_rewrites_the_reproducer() {
    // The other half of the advice: `chelis migrate surf --from 0.18`. If the
    // migrator stops handling this shape, the diagnostic starts lying.
    let tmp = write_tempfile("issue1267-migrate-", REPRO);
    let path = tmp.path().to_str().expect("path utf8").to_string();
    let migrate = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace", &path])
        .output()
        .expect("run chelis migrate");
    let stderr = String::from_utf8(migrate.stderr).expect("utf8 stderr");
    assert!(
        migrate.status.success(),
        "the migrator should rewrite the `;` block; stderr={stderr}"
    );
    let migrated = std::fs::read_to_string(&path).expect("read migrated file");
    assert!(
        !migrated.contains(';'),
        "migration should remove the block separators: {migrated}"
    );
    // And the rewrite must satisfy the gate that rejected the original.
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check", &path])
        .output()
        .expect("run chelis fmt");
    assert!(
        output.status.success(),
        "migrator output must pass canonical fmt --check; migrated={migrated}"
    );
    // fmt --check only proves it parses. Take it all the way to a clean
    // check, so both halves of the advice are held to the same bar.
    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", &path])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(checked.stdout).expect("utf8 stdout");
    assert_eq!(
        checked.status.code(),
        Some(0),
        "migrator output should check clean; migrated={migrated} stdout={stdout}"
    );
}
