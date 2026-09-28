//! Wave-1 red-team finding M1 (issue #207 follow-up):
//!
//! `chelis check` documents the contract `exit_code != 0 iff
//! json.errors.len() > 0`, with the non-zero exit defined as
//! `CHECK_ERRORS_EXIT_CODE` (literal 2). The current implementation
//! routes parse errors through the `Err(err)` arm in `main`, which
//! exits 1 and emits no JSON at all (stdout empty, diagnostic on
//! stderr).
//!
//! The iff invariant is therefore violated by parse-error fixtures:
//! the program has a real failure (non-zero exit) but the JSON
//! `errors` array does not exist (the JSON itself does not exist).
//!
//! These tests are currently EXPECTED TO FAIL. They lock the spec
//! reading. If we instead decide the brief language was imprecise and
//! parse-error-via-exit-1 is acceptable, delete or invert these tests
//! and tighten the brief language to "exit non-zero" rather than
//! "exit 2".
//!
//! Owning code: `cmd_check` in `crates/chelis-cli/src/main.rs`.

use assert_cmd::prelude::*;
use std::io::Write;
use std::process::Command;

/// Match the constant in `crates/chelis-cli/src/main.rs`.
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

fn run_check_capture(path: &std::path::Path) -> (Option<i32>, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    (output.status.code(), stdout, stderr)
}

/// Parse error: a `@` in expression position is rejected by the
/// parser. Per the #207 brief language ("exit 2 matching chelis
/// test"), this MUST exit 2, not 1.
#[test]
fn rt_wave1_207_parse_error_exits_with_check_errors_code() {
    let src = "def main[n]() -> tensor[n, f32] = { @invalid }\n";
    let tmp = write_tempfile("rt207-parse-", src);
    let (code, _stdout, stderr) = run_check_capture(tmp.path());
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "parse errors should also exit {CHECK_ERRORS_EXIT_CODE} per #207 invariant; stderr={stderr}"
    );
}

/// Parse error: the iff invariant (`exit != 0 iff errors[] non-empty`)
/// requires JSON to be emitted whenever exit is non-zero. The current
/// implementation prints no JSON at all on parse failure (stdout is
/// empty). This breaks the literal iff.
#[test]
fn rt_wave1_207_parse_error_emits_errors_array_in_json() {
    let src = "def main[n]() -> tensor[n, f32] = { @invalid }\n";
    let tmp = write_tempfile("rt207-parse-json-", src);
    let (_code, stdout, stderr) = run_check_capture(tmp.path());
    assert!(
        !stdout.is_empty(),
        "parse-error path must still emit JSON with a populated errors[] (iff invariant); \
         stdout was empty; stderr={stderr}"
    );
    // Parse the JSON and assert errors[] is non-empty.
    let json: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout must be JSON: {e}\nstdout={stdout}"));
    let errors = json
        .get("errors")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("missing errors array; stdout={stdout}"));
    assert!(
        !errors.is_empty(),
        "errors[] must be non-empty for a parse error; stdout={stdout}"
    );
}
