//! chelis#706 red-team keepers — adversarial coverage found by a fresh
//! red-team pass on PR #769 that the shipped tests do not lock:
//!   * bare statements inside a *nested* block and inside a `with seed`
//!     block are rejected too (both silently collapsed before #706);
//!   * the diagnostic offset lands on the stray statement even when a
//!     `--` line comment sits between the tail and the bare statement;
//!   * `validate --surf` and `validate --desugar` AGREE on the #706
//!     reproducer — both reject it. The grammar path used to accept it
//!     (grammar too lenient) while the parser path rejected; `validate_surf`
//!     now follows the compiler parser as the acceptance authority.
//!
//! Owning code: `parse_block` in `crates/chelis-surf/src/parser.rs`;
//! `validate_surf` in `crates/chelis-validate/src/lib.rs`.

use assert_cmd::prelude::*;
use std::io::Write;
use std::process::Command;

const CHECK_ERRORS_EXIT_CODE: i32 = 2;

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

fn run(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(args)
        .output()
        .expect("run chelis")
}

/// A bare non-tail statement inside a *nested* block is rejected with the
/// #706 diagnostic pointing at the stray statement (was a silent collapse).
#[test]
fn nested_block_bare_statement_rejected() {
    let src = "def f(x: int32) -> int32 = {\n  y = 1\n  {\n    a(x)\n    b(x)\n  }\n}\n";
    let out = run(&[
        "check",
        write_tempfile("rt706-nested-", src)
            .path()
            .to_str()
            .unwrap(),
    ]);
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(CHECK_ERRORS_EXIT_CODE));
    assert!(
        stdout.contains("expression statement must be bound"),
        "nested-block bare statement should surface #706 diagnostic; stdout={stdout}"
    );
    // Offset must land on the *inner* stray statement `b(x)`.
    let byte = src.find("b(x)").unwrap();
    assert!(
        stdout.contains(&format!("byte {byte}")),
        "offset should point at the inner stray statement (byte {byte}); stdout={stdout}"
    );
}

/// The bounded tail also applies inside a `with seed(..)` handler block.
#[test]
fn with_seed_block_bare_statement_rejected() {
    let src = "def f(x: tensor[batch, f32]) -> tensor[batch, f32] = with seed(42i64) {\n  a(x)\n  b(x)\n}\n";
    let out = run(&[
        "check",
        write_tempfile("rt706-seed-", src).path().to_str().unwrap(),
    ]);
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(CHECK_ERRORS_EXIT_CODE));
    assert!(
        stdout.contains("expression statement must be bound"),
        "with-seed bare statement should surface #706 diagnostic; stdout={stdout}"
    );
}

/// A `--` comment between the tail and the stray statement must not skew
/// the offset: it lands on the stray statement, not the comment.
#[test]
fn comment_between_statements_offset_on_stray_stmt() {
    let src =
        "def f(a: int32, b: int32) -> int32 = {\n  g(a)\n  -- explanatory comment\n  h(b)\n}\n";
    let out = run(&[
        "check",
        write_tempfile("rt706-comment-", src)
            .path()
            .to_str()
            .unwrap(),
    ]);
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(CHECK_ERRORS_EXIT_CODE));
    let byte = src.find("h(b)").unwrap();
    assert!(
        stdout.contains(&format!("byte {byte}")),
        "offset should skip the comment and point at `h(b)` (byte {byte}); stdout={stdout}"
    );
}

/// chelis#706: `validate --surf` used to run the pest grammar first and
/// only cross-check the hand-written parser when the grammar *failed*. The
/// grammar admits bare-statement juxtaposition, so `validate --surf`
/// green-lit the reproducer (exit 0) while `validate --desugar` (parser
/// path) and `check` rejected it. `validate_surf` now follows the compiler
/// parser as the acceptance authority in both directions, so the two
/// validate modes agree: both reject the bare-statement reproducer.
#[test]
fn validate_surf_and_desugar_agree_on_bare_statement() {
    let src = "def f(a: f32, b: f32, c: f32, d: f32) -> f32 = {\n  g(a, b)\n  h(c, d)\n}\n";
    let tmp = write_tempfile("rt706-validate-", src);
    let path = tmp.path().to_str().unwrap();
    let surf = run(&["validate", "--surf", path]);
    let desugar = run(&["validate", "--desugar", path]);
    assert_eq!(
        surf.status.success(),
        desugar.status.success(),
        "validate --surf and --desugar must agree on the #706 reproducer; \
         surf_ok={} desugar_ok={}",
        surf.status.success(),
        desugar.status.success()
    );
    // Both should reject (the parser rejects, so the grammar should too).
    assert!(
        !surf.status.success(),
        "validate --surf should reject the bare-statement reproducer"
    );
}
