//! Red Team #205 finding F7: `chelis check` exit-code contract.
//!
//! `chelis check` intentionally exits 0 when it produced a parseable
//! fitness report, regardless of whether the JSON `errors` array is
//! empty or full. The JSON shape is the machine-facing contract;
//! consumer tooling reads stdout and inspects `report.errors[]`. A
//! non-zero exit on "type errors found" would break that contract.
//!
//! This file pins the contract directly so future refactors do not
//! silently flip the behavior. Two existing tests in
//! `check_in_reef_context.rs` cover the same surface but are gated
//! behind a reef-context setup; this one runs in the default inner
//! loop and uses only a tempfile.
//!
//! The owning code comment lives at the head of `cmd_check` in
//! `crates/chelis-cli/src/main.rs`.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;

#[test]
fn red_team_205_f7_check_exits_zero_when_validator_rejects_conv2d() {
    // Concrete-shape conv2d with stride 0; the RT-205 F1 fix rejects
    // this at validator time. Without the F7 contract this would
    // exit non-zero, but the JSON-tooling contract requires exit 0
    // with a non-empty errors array.
    let mut tmp = tempfile::Builder::new()
        .prefix("rt205-f7-")
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    let src = "module Probe\n\
               def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv2d(&x, &k, 0, 0)\n";
    tmp.write_all(src.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", tmp.path().to_str().expect("path utf8")])
        .assert()
        .success() // F7 contract: exit 0 even when validator rejects
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("utf8");
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
