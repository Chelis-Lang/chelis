//! Phase H acceptance for `cmd_check`.
//!
//! `cmd_check` now routes reef-in-scope inputs through
//! `compile_reef_context` so the library compile becomes the
//! architectural seed Phase I's disk cache plugs into. Diagnostic JSON
//! still travels through the legacy fitness scorer so the byte-identical
//! output guarantee holds.
//!
//! The fixtures here cover the three failure-mode classes the plan
//! requires the new path to handle:
//!
//! 1. Clean program: `score = 1`, empty `errors` array, exit 0.
//! 2. Program with a type error: `score < 1`, non-empty `errors`, exit 2.
//! 3. Program with an unhandled effect: `score < 1`, non-empty `errors`,
//!    exit 2 (issue #207 ties the exit code to the errors array).
//!
//! Each test asserts both the exit code and the JSON diagnostic shape;
//! together they lock the parity contract chelis-tide and other
//! downstream consumers depend on.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

/// Parse `chelis check` stdout and assert the keys downstream consumers
/// rely on are present with the expected shape.
fn assert_check_json_shape(stdout: &str) -> Value {
    let v: Value = serde_json::from_str(stdout)
        .unwrap_or_else(|err| panic!("`chelis check` output must be valid JSON: {err}\n{stdout}"));
    let obj = v
        .as_object()
        .unwrap_or_else(|| panic!("`chelis check` output must be a JSON object: {stdout}"));
    for key in [
        "score",
        "components",
        "typed_nodes",
        "untyped_nodes",
        "total_nodes",
        "unresolved_names",
        "errors",
    ] {
        assert!(
            obj.contains_key(key),
            "missing `{key}` in chelis check output: {stdout}"
        );
    }
    let components = obj
        .get("components")
        .and_then(Value::as_object)
        .expect("components");
    for key in ["parse", "structure", "names", "types"] {
        assert!(
            components.contains_key(key),
            "missing `components.{key}` in chelis check output: {stdout}"
        );
    }
    assert!(
        obj.get("errors").and_then(Value::as_array).is_some(),
        "errors not array"
    );
    assert!(
        obj.get("unresolved_names")
            .and_then(Value::as_array)
            .is_some(),
        "unresolved_names not array"
    );
    v
}

#[test]
#[ignore = "manual gate: Reef-context CLI check acceptance exceeds the default inner-loop budget"]
fn check_clean_reef_program_yields_score_one_and_empty_errors() {
    let (_dir, reef_home, app_pkg) = make_app("phase-h-clean");

    write_file(
        &app_pkg.join("src/main.ch"),
        r"module Demo.Main

def answer -> int32 = cast(7, int32)
",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("utf8");
    let json = assert_check_json_shape(&stdout);
    assert_eq!(
        json.get("score").and_then(Value::as_f64).expect("score"),
        1.0,
        "clean program must report score 1.0; stdout={stdout}"
    );
    let errors = json
        .get("errors")
        .and_then(Value::as_array)
        .expect("errors");
    assert!(
        errors.is_empty(),
        "clean program must have empty errors array; stdout={stdout}"
    );
}

#[test]
#[ignore = "manual gate: Reef-context CLI check acceptance exceeds the default inner-loop budget"]
fn check_type_error_reef_program_yields_lower_score_and_kept_shape() {
    let (_dir, reef_home, app_pkg) = make_app("phase-h-type-error");

    // `add` (chelis-std builtin) takes two int32s; passing a bool surfaces
    // a type-mismatch at the type-checker stage.
    write_file(
        &app_pkg.join("src/main.ch"),
        r"module Demo.Main

def broken -> int32 = add(1, true)
",
    );

    // Issue #207: `chelis check` now exits with code 2 when the JSON
    // `errors` array is non-empty. The fixture below produces a
    // TypeMismatch, so we drive the command via `.output()` and
    // assert on the exit code explicitly.
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .output()
        .expect("run chelis check");
    assert_eq!(
        output.status.code(),
        Some(2),
        "issue #207: type errors must produce exit 2; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let json = assert_check_json_shape(&stdout);
    let score = json.get("score").and_then(Value::as_f64).expect("score");
    assert!(
        score < 1.0,
        "type error must reduce score below 1.0; got {score}; stdout={stdout}"
    );
    let errors = json
        .get("errors")
        .and_then(Value::as_array)
        .expect("errors");
    assert!(
        !errors.is_empty(),
        "type error must produce non-empty errors array; stdout={stdout}"
    );
}

#[test]
#[ignore = "manual gate: Reef-context CLI check acceptance exceeds the default inner-loop budget"]
fn check_effect_error_reef_program_yields_lower_score_and_kept_shape() {
    let (_dir, reef_home, app_pkg) = make_app("phase-h-effect-error");

    // `broken` declares an empty effect row (`! {}`) but its body calls
    // a `! { Test }` library function. `validate_declared_vs_inferred`
    // surfaces this as an `UnhandledEffect` diagnostic in the JSON
    // `errors` array. Issue #207 ties the exit code to the errors
    // array, so an unhandled-effect diagnostic now produces exit 2.
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_true)

def broken(c: bool) -> unit ! {} = assert_true(c, "expect ok")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .output()
        .expect("run chelis check");
    assert_eq!(
        output.status.code(),
        Some(2),
        "issue #207: effect errors render to stdout JSON and produce exit 2; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let json = assert_check_json_shape(&stdout);
    let errors = json
        .get("errors")
        .and_then(Value::as_array)
        .expect("errors");
    assert!(
        !errors.is_empty(),
        "effect error must produce non-empty errors array; stdout={stdout}"
    );
    // The legacy fitness path is the diagnostic source of truth, so the
    // unhandled-effect error must surface with its message intact for
    // downstream parsers (chelis-tide consumes the `errors[].message`
    // verbatim).
    let any_effect_mentioned = errors.iter().any(|e| {
        e.get("message")
            .and_then(Value::as_str)
            .is_some_and(|m| m.contains("effect") || m.contains("Test"))
    });
    assert!(
        any_effect_mentioned,
        "effect error message must mention `effect` or the missing `Test` row; stdout={stdout}"
    );
}

#[test]
#[ignore = "manual gate: Reef-context CLI check acceptance exceeds the default inner-loop budget"]
fn check_non_reef_file_uses_legacy_path_and_returns_score_one() {
    // A file outside any reef package: the new path must short-circuit
    // to the legacy fitness pipeline. This covers the `Phase H legacy
    // fallback` requirement (`chelis check <raw_program.ch>` still works
    // via legacy path).
    let dir = tempfile::tempdir().expect("tempdir");
    let raw = dir.path().join("raw_program.ch");
    std::fs::write(
        &raw,
        r"def answer -> int32 = cast(7, int32)
",
    )
    .expect("write raw");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", raw.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("utf8");
    let json = assert_check_json_shape(&stdout);
    assert_eq!(
        json.get("score").and_then(Value::as_f64).expect("score"),
        1.0,
        "raw clean program must report score 1.0; stdout={stdout}"
    );
}

#[test]
#[ignore = "manual gate: Reef-context CLI check acceptance exceeds the default inner-loop budget"]
fn check_reef_does_not_export_propagates_to_stderr_unchanged() {
    // The historical "missing import" failure mode must continue to
    // surface on stderr (legacy cmd_check propagates this as `Err`,
    // main exits non-zero). Phase H's compile_reef_context short-
    // circuits the same wording before the legacy path's duplicate
    // reef walk, so the stderr message and exit code are unchanged.
    let (_dir, reef_home, app_pkg) = make_app("phase-h-missing-import");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io (missing_symbol)

x = missing_symbol("foo")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing_symbol"))
        .stderr(predicate::str::contains("does not export"));
}
