//! Red package witnesses for the proof-only scalar Beacon route in
//! spec/design/prove_beacon_shell_integration.md, Authored Beacon goals.
#![cfg(feature = "chelis-prove")]

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::{TempDir, tempdir};

fn package(model: &str) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("beacon-pkg");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            "beacon-pkg",
            "--module-prefix",
            "BeaconPkg",
            "--output",
        ])
        .arg(&root)
        .assert()
        .success();
    fs::write(root.join("src/model.ch"), model).expect("model");
    (dir, root)
}

fn prove(root: &Path, property: &str) -> Value {
    fs::write(root.join("src/main.ch"), property).expect("property");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env_remove("CHELIS_BEACON_BIN")
        .args(["prove", "src/main.ch", "--tier", "beacon-only", "--json"])
        .output()
        .expect("prove");
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("NDJSON"))
        .collect::<Vec<_>>();
    records
        .into_iter()
        .find(|record| record["kind"] == "property")
        .expect("property record")
}

#[test]
fn imported_f64_scalar_function_reaches_beacon_after_proof_only_extraction() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f64) -> f64 = x + 1.0f64\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f64) where x >= -1.0f64, x <= 1.0f64:\n  affine(x) <= 2.0f64\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(
        record["reason"], "CHELIS_BEACON_BIN is not configured",
        "{record}"
    );
}

#[test]
fn imported_f32_scalar_function_uses_exact_typed_box_without_sampling() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f32) -> f32 = x + 1.0f32\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f32) where x >= -1.0f32, x <= 1.0f32:\n  affine(x) <= 2.0f32\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(
        record["reason"], "CHELIS_BEACON_BIN is not configured",
        "{record}"
    );
}

#[test]
fn scalar_goal_with_missing_bound_stays_unsupported_without_fallback() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f64) -> f64 = x + 1.0f64\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property unbounded forall(x: f64) where x >= -1.0f64:\n  affine(x) <= 2.0f64\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(
        record["reason"]
            .as_str()
            .is_some_and(|s| s.contains("missing upper bound")),
        "{record}"
    );
}

#[test]
fn strict_scalar_box_is_rejected_before_engine_dispatch() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f64) -> f64 = x + 1.0f64\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property strict_box forall(x: f64) where x > -1.0f64, x <= 1.0f64:\n  affine(x) <= 2.0f64\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(
        record["reason"]
            .as_str()
            .is_some_and(|s| s.contains("closed")),
        "{record}"
    );
}
