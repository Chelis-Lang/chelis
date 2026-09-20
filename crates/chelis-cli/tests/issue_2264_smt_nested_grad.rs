//! chelis#2264: CLI precheck and property filtering must preserve the typed
//! SMT boundary for valid nested gradients without hiding malformed siblings.

#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn prove_json(source: &str, extra: &[&str]) -> (i32, Vec<Value>, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_grad.ch");
    std::fs::write(&path, source).expect("write nested-gradient fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .arg("prove")
        .arg(&path)
        .arg("--json")
        .args(extra)
        .output()
        .expect("run chelis prove");
    let records = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("stdout must be NDJSON"))
        .collect();
    (
        output.status.code().unwrap_or(-1),
        records,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn nested_grad_pair() -> &'static str {
    "module M
@property nested_grad_left forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
@property nested_grad_right forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx + xx, wrt=xx), wrt=xx)(x) >= 0.0)
"
}

#[test]
fn filtered_valid_nested_grad_reaches_typed_smt_boundary_without_check_errors() {
    let (code, records, stderr) = prove_json(
        nested_grad_pair(),
        &["--tier", "smt-only", "--only", "nested_grad_left"],
    );

    assert_eq!(code, 2, "records={records:#?}\nstderr={stderr}");
    assert!(
        records
            .iter()
            .all(|record| record["kind"] != "error" || record["stage"] != "check"),
        "valid nested gradients must survive the CLI precheck: {records:#?}"
    );
    let properties = records
        .iter()
        .filter(|record| record["kind"] == "property")
        .collect::<Vec<_>>();
    assert_eq!(properties.len(), 1, "{records:#?}");
    assert_eq!(properties[0]["name"], "nested_grad_left");
    assert_eq!(properties[0]["status"], "unsupported");
    assert_eq!(properties[0]["proof_tier"], "smt");
    assert_eq!(
        properties[0]["reason"],
        "scalar grad SMT lowering does not support nested gradients"
    );
    let summary = records
        .iter()
        .find(|record| record["kind"] == "summary")
        .expect("summary record");
    assert_eq!(summary["unsupported"], 1, "{summary}");
    assert_eq!(summary["errors"], 0, "{summary}");
}

#[test]
fn malformed_nested_grad_sibling_still_fails_closed_before_filtering() {
    let source = "module M
@property malformed_sibling forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=missing)(x) >= 0.0)
@property nested_grad_left forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
";
    let (code, records, stderr) = prove_json(
        source,
        &["--tier", "smt-only", "--only", "nested_grad_left"],
    );

    assert_eq!(code, 3, "records={records:#?}\nstderr={stderr}");
    let check_errors = records
        .iter()
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .collect::<Vec<_>>();
    assert_eq!(check_errors.len(), 1, "{records:#?}");
    assert!(
        check_errors[0].to_string().contains("missing"),
        "{check_errors:#?}"
    );
    assert!(
        records.iter().all(|record| record["kind"] != "property"),
        "a malformed sibling must prevent property verdicts: {records:#?}"
    );
}
