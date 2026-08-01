//! End-to-end acceptance surface for chelis#978.

#![cfg(all(feature = "chelis-prove", feature = "smt"))]

use assert_cmd::Command;
use serde_json::Value;

const BOND: &str = include_str!("../../../examples/induction_bond.ch");

fn run(source: &str) -> (std::process::ExitStatus, Value) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bond.ch");
    std::fs::write(&path, source).expect("write fixture");
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
        "coupon + discount * bond_value(n - 1, coupon, discount)",
        "coupon - cast(1.0, f64) + discount * bond_value(n - 1, coupon, discount)",
    );
    let (status, record) = run(&false_bond);
    assert_eq!(status.code(), Some(1), "{record}");
    assert_eq!(record["status"], "failed", "{record}");
    assert_eq!(record["induction"]["base"]["status"], "proved", "{record}");
    assert_eq!(
        record["induction"]["step"]["status"], "disproved",
        "{record}"
    );
}
