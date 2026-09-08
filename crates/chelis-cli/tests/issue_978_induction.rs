//! End-to-end acceptance surface for chelis#978.

#![cfg(all(feature = "chelis-prove", feature = "smt"))]

use assert_cmd::Command;
use serde_json::Value;

const BOND: &str = include_str!("../../../examples/induction_bond.ch");

fn run_with_args(source: &str, extra_args: &[&str]) -> (std::process::ExitStatus, Value) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bond.ch");
    std::fs::write(&path, source).expect("write fixture");
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.args(["prove", path.to_str().expect("utf8 path"), "--json"]);
    command.args(extra_args);
    let output = command.output().expect("run prove");
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|record| record["kind"] == "property")
        .unwrap_or_else(|| {
            panic!(
                "missing property record: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
    (output.status, record)
}

fn run(source: &str) -> (std::process::ExitStatus, Value) {
    run_with_args(source, &["--tier", "induction-only"])
}

#[test]
fn general_bond_emits_separate_base_and_step_evidence() {
    let (status, record) = run(BOND);
    assert!(status.success(), "{record}");
    assert_eq!(record["status"], "passed", "{record}");
    assert_eq!(record["proof_tier"], "induction", "{record}");
    assert_eq!(record["arith_model"], "real", "{record}");
    assert_eq!(record["induction"]["base"]["status"], "proved", "{record}");
    assert_eq!(record["induction"]["step"]["status"], "proved", "{record}");
    assert!(!record.to_string().contains("ASSUMED"), "{record}");
}

#[test]
fn false_general_bond_exits_red_with_disproved_step() {
    let false_bond = BOND.replace(
        "coupon + (discount * bond_value((n - 1), coupon, discount))",
        "(coupon - cast(1.0, f64)) + (discount * bond_value((n - 1), coupon, discount))",
    );
    assert_ne!(false_bond, BOND, "negative fixture rewrite must apply");
    let (status, record) = run(&false_bond);
    assert_eq!(status.code(), Some(1), "{record}");
    assert_eq!(record["status"], "failed", "{record}");
    assert_eq!(record["induction"]["base"]["status"], "proved", "{record}");
    assert_eq!(
        record["induction"]["step"]["status"], "disproved",
        "{record}"
    );
}

#[test]
fn default_auto_dispatches_true_induction_without_sampling_or_assumptions() {
    let (status, record) = run_with_args(BOND, &[]);
    assert!(status.success(), "{record}");
    assert_eq!(record["status"], "passed", "{record}");
    assert_eq!(record["proof_tier"], "induction", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(record.get("sampling_method").is_none(), "{record}");
    assert!(!record.to_string().contains("ASSUMED"), "{record}");
}

#[test]
fn default_auto_rejects_false_induction_without_fuzz_fallthrough() {
    let false_bond = BOND.replace(
        "coupon + (discount * bond_value((n - 1), coupon, discount))",
        "(coupon - cast(1.0, f64)) + (discount * bond_value((n - 1), coupon, discount))",
    );
    let (status, record) = run_with_args(&false_bond, &[]);
    assert_eq!(status.code(), Some(1), "{record}");
    assert_eq!(record["status"], "failed", "{record}");
    assert_eq!(record["proof_tier"], "induction", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(record.get("sampling_method").is_none(), "{record}");
    assert!(!record.to_string().contains("ASSUMED"), "{record}");
}

#[test]
fn default_auto_fails_closed_on_unsupported_recursive_structure() {
    let non_decreasing = BOND.replace("n - 1", "n + 1");
    let (status, record) = run_with_args(&non_decreasing, &[]);
    assert_eq!(status.code(), Some(2), "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(record["proof_tier"], "induction", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(record.get("sampling_method").is_none(), "{record}");
    assert!(!record.to_string().contains("ASSUMED"), "{record}");
}

#[test]
fn deep_induction_only_is_unsupported_and_never_fuzzes() {
    let source = r#"(module {}
  m
  (defsig {} always_true (t-fn {} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}),
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_source_kind: "user"}
    always_true
    (fn {} (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("property.dp");
    std::fs::write(&path, source).expect("write deep fixture");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            path.to_str().expect("utf8 path"),
            "--json",
            "--tier",
            "induction-only",
        ])
        .output()
        .expect("run prove");
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|record| record["kind"] == "property")
        .expect("property record");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(record["proof_tier"], "induction", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(record.get("sampling_method").is_none(), "{record}");
    assert!(record.get("induction").is_none(), "{record}");
}
