//! Regression: a malformed-yet-parseable Deep `@property` must surface as a
//! prove ERROR, never a silent pass (opaque-types Finding #3).
//!
//! `chelis_prove::property_runner::run_deep_source_properties` returns `Err`
//! when it discovers a malformed `@property` (here: a property classified as
//! `user` whose metadata is missing the required `property_quantifiers`). The
//! Deep prove driver (`prove::property_run::run_deep_properties_shared`)
//! previously mapped EVERY `Err` from that runner to `Status::Passed`, so a
//! malformed property was reported PASSED with exit 0 -- re-introducing the
//! silent skip the shared discoverer's `Err` exists to prevent. The fix
//! distinguishes a benign parse error (which already surfaced upstream) from
//! a genuine malformed-property discovery error and surfaces the latter as a
//! prove error (exit 3), the way the obligation path surfaces its errors.
//!
//! The malformed fixture below uses the legacy `c_earchin_role:
//! "property_witness"` marker with `property_source_kind: "user"` and NO
//! `property_quantifiers`. The CLI-local Deep discoverer tolerates that form
//! (it falls back to the fn parameters when `chelis_role` is absent), so the
//! ONLY component that rejects it is the shared runner -- which means the
//! whole error path runs THROUGH the swallow at `run_deep_properties_shared`.
//! Without the fix this prove run exits 0 "passed"; with the fix it exits 3.
//!
//! These tests are gated on `chelis-prove` (the feature that compiles the
//! shared runner and `property_run.rs`; the optional dep is enabled by
//! `--features smt`, matching the gating of the other shared-runner prove
//! tests). In a default no-`chelis-prove` build, `chelis prove` runs the
//! CLI-local Deep property path instead, which tolerates the legacy fixture by
//! design, so this regression only holds when the shared runner is compiled.

#![cfg(feature = "chelis-prove")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// A malformed user `@property` (missing `property_quantifiers`) must NOT be
/// reported as passed. It must surface as a prove error with a non-zero exit.
#[test]
fn malformed_deep_property_missing_quantifiers_is_error_not_pass() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("malformed.dp");
    // Classified `user` (so the shared runner owns it) but the metadata omits
    // `property_quantifiers`. Parses and validate_deep-passes; only the shared
    // discoverer rejects it.
    std::fs::write(
        &path,
        r#"
(def {c_earchin_role: "property_witness",
      property_source_kind: "user"}
  malformed_missing_quantifiers
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write malformed deep");

    // Non-zero exit (Error is exit code 3), never the passing exit 0.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap()])
        .assert()
        .failure()
        .code(3);
}

/// JSON twin: the malformed property must NOT emit a passing property record;
/// it must emit an error record and a summary with at least one error and
/// zero passed.
#[test]
fn malformed_deep_property_json_reports_error_not_passed() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("malformed.dp");
    std::fs::write(
        &path,
        r#"
(def {c_earchin_role: "property_witness",
      property_source_kind: "user"}
  malformed_missing_quantifiers
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write malformed deep");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");

    assert!(
        !output.status.success(),
        "malformed property must not exit 0; stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();

    // No property record may claim the malformed property passed.
    let passed_property = records
        .iter()
        .any(|record| record["kind"] == "property" && record["status"] == "passed");
    assert!(
        !passed_property,
        "malformed property must not be reported passed; records={records:?}"
    );

    // The summary must fold at least one error and report zero passed.
    let summary = records
        .iter()
        .find(|record| record["kind"] == "summary")
        .expect("summary record present");
    assert_eq!(
        summary["passed"], 0,
        "malformed property must not increment passed; summary={summary}"
    );
    assert!(
        summary["errors"].as_u64().unwrap_or(0) >= 1,
        "malformed property must fold an error; summary={summary}"
    );
}

/// Positive companion: a WELL-FORMED user `@property` (with
/// `property_quantifiers`) still passes cleanly, proving the fix did not break
/// the legitimate pass path.
#[test]
fn wellformed_deep_property_still_passes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wellformed.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  wellformed_trivial
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write wellformed deep");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");

    assert!(
        output.status.success(),
        "well-formed property must pass; stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();

    let property_record = records
        .iter()
        .find(|record| record["kind"] == "property")
        .expect("a property record is emitted");
    assert_eq!(property_record["name"], "wellformed_trivial");
    assert_eq!(property_record["status"], "passed");

    let summary = records
        .iter()
        .find(|record| record["kind"] == "summary")
        .expect("summary record present");
    assert_eq!(
        summary["errors"], 0,
        "no errors expected; summary={summary}"
    );
}
