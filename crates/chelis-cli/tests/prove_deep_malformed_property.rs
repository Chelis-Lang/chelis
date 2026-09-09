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
//! Canonical property metadata is validated at Deep ingress. The same
//! missing-field defect must still reject before discovery, in default and
//! SMT builds, while a complete property runs normally.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// A malformed user `@property` (missing `property_quantifiers`) must NOT be
/// reported as passed. It must surface as a prove error with a non-zero exit.
#[test]
fn malformed_deep_property_missing_quantifiers_is_error_not_pass() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("malformed.dp");
    // Canonical property role with missing required quantifiers fails at ingress.
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
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
/// stamped ingress identifies the missing field before property discovery.
#[test]
fn malformed_deep_property_json_rejects_at_ingress() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("malformed.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
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

    assert!(
        String::from_utf8_lossy(&output.stderr).contains("property_quantifiers"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
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
