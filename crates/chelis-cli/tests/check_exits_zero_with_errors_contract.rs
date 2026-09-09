//! Issue #207 inversion of the original RT-205 F7 contract.
//!
//! The original RT-205 F7 contract pinned `chelis check` to exit `0`
//! even when the JSON `errors` array was non-empty. Issue #207
//! inverted that decision: the exit code now mirrors the errors
//! array (0 iff empty, non-zero otherwise) so shell-script
//! consumers no longer have to parse the JSON to detect type errors.
//!
//! This file's pre-#207 name is preserved so the git history of the
//! contract flip is searchable. The validator-rejection fixture is
//! the same shape RT-205 F7 used; the assertion is now `exit 2 +
//! non-empty errors` instead of `exit 0 + non-empty errors`.
//!
//! See `crates/chelis-cli/tests/issue_207_check_exit_code_invariant.rs`
//! for the full iff sweep across multiple error categories. The owning
//! code comment lives at the head of `cmd_check` in
//! `crates/chelis-cli/src/main.rs`.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;

/// `chelis check` exit code on a non-empty errors array. Matches
/// `chelis test`'s exit `2` for "compile test context" failures.
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

#[test]
fn issue_207_check_exits_nonzero_when_validator_rejects_conv() {
    // Concrete-shape conv with stride 0; the RT-205 F1 fix rejects
    // this at validator time. Pre-#207 this exited 0 with a non-empty
    // errors array. Post-#207 the exit code matches the errors array
    // so downstream CI shell scripts can detect the rejection without
    // parsing JSON.
    let mut tmp = tempfile::Builder::new()
        .prefix("issue207-validator-")
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    let src = "module Probe\n\
               def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])\n";
    tmp.write_all(src.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert_eq!(
        output.status.code(),
        Some(CHECK_ERRORS_EXIT_CODE),
        "issue #207: validator rejection must exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
    let json: Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("chelis check stdout must be valid JSON: {err}\n{stdout}"));
    let errors = json
        .get("errors")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("missing errors array; stdout={stdout}"));
    assert!(
        !errors.is_empty(),
        "validator rejection must produce non-empty errors array; stdout={stdout}"
    );
    let has_positive_stride = errors.iter().any(|e| {
        e.get("message")
            .and_then(Value::as_str)
            .is_some_and(|m| m.contains("positive stride"))
    });
    assert!(
        has_positive_stride,
        "expected `positive stride` validator message in errors; stdout={stdout}"
    );
}
