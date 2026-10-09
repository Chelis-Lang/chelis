//! Package witnesses for the proof-only scalar Beacon route in
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
fn imported_f64_lower_and_two_sided_goals_reach_beacon() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f64) -> f64 = x + 1.0f64\n",
    );
    for body in [
        "affine(x) >= -1.0f64",
        "(affine(x) >= -1.0f64) && (affine(x) <= 3.0f64)",
    ] {
        let record = prove(
            &root,
            &format!(
                "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f64) where x >= -1.0f64, x <= 1.0f64:\n  {body}\n"
            ),
        );
        assert_eq!(record["proof_tier"], "beacon", "{record}");
        assert_eq!(record["samples"], 0, "{record}");
        assert_eq!(
            record["reason"], "CHELIS_BEACON_BIN is not configured",
            "{record}"
        );
    }
}

#[test]
fn imported_f32_scalar_goal_preserves_typed_source_at_dispatch() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f32) -> f32 = x + 0.1f32\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f32) where x >= 0.1f32, x <= 0.2f32:\n  affine(x) <= 1.0f32\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert_eq!(
        record["reason"], "CHELIS_BEACON_BIN is not configured",
        "{record}"
    );
}

#[test]
fn mixed_float_dtypes_do_not_enter_beacon_proof_graph() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f32) -> f32 = x + 0.1f32\n",
    );
    fs::write(root.join("src/main.ch"), "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property mixed forall(x: f32) where x >= 0.1f32, x <= 0.2f32:\n  affine(x) <= 1.0f64\n")
        .expect("property");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env_remove("CHELIS_BEACON_BIN")
        .args(["prove", "src/main.ch", "--tier", "beacon-only", "--json"])
        .output()
        .expect("prove");
    assert!(
        !output.status.success(),
        "mixed dtype property unexpectedly succeeded"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("\"status\":\"passed\""));
}

#[cfg(unix)]
#[test]
fn imported_f32_request_uses_exact_stored_values_in_its_real_graph() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f32) -> f32 = x + 0.1f32\n",
    );
    fs::write(
        root.join("src/main.ch"),
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f32) where x >= 0.1f32, x <= 0.2f32:\n  affine(x) <= 1.0f32\n",
    ).expect("property");
    let beacon = root.join("capture-beacon");
    fs::write(
        &beacon,
        "#!/bin/sh\ncp \"$3\" \"$CAPTURE_REQUEST\"\nexit 3\n",
    )
    .expect("capture binary");
    fs::set_permissions(&beacon, fs::Permissions::from_mode(0o755)).expect("executable");
    let capture = root.join("captured-request.json");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_BEACON_BIN", &beacon)
        .env("CAPTURE_REQUEST", &capture)
        .args(["prove", "src/main.ch", "--tier", "beacon-only", "--json"])
        .output()
        .expect("prove");
    let record: Value = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("NDJSON"))
        .find(|record| record["kind"] == "property")
        .expect("property record");
    assert_ne!(record["status"], "passed", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert_eq!(record["engine_evidence"]["source_dtype"], "f32");
    assert_eq!(record["engine_evidence"]["proof_graph_dtype"], "f64");
    let request: Value =
        serde_json::from_slice(&fs::read(capture).expect("request")).expect("request JSON");
    assert_eq!(request["inputs"]["x"]["lo"], f64::from(0.1_f32));
    assert_eq!(request["inputs"]["x"]["hi"], f64::from(0.2_f32));
    let exact_bits = format!("{:016x}", f64::from(0.1_f32).to_bits());
    assert!(
        request["dag"]["nodes"]
            .as_array()
            .expect("nodes")
            .iter()
            .any(|node| {
                node["op"]["kind"] == "const"
                    && node["op"]["value"]["dtype"] == "f64"
                    && node["op"]["value"]["bits"] == exact_bits
            }),
        "{request}"
    );
}

#[cfg(unix)]
#[test]
fn two_sided_property_cannot_hide_an_unknown_upper_side() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f64) -> f64 = x + 1.0f64\n",
    );
    fs::write(
        root.join("src/main.ch"),
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f64) where x >= -1.0f64, x <= 1.0f64:\n  (affine(x) >= -1.0f64) && (affine(x) <= 3.0f64)\n",
    ).expect("property");
    let beacon = root.join("mock-beacon");
    fs::write(&beacon, concat!(
        "#!/bin/sh\n",
        "if [ -f \"$MOCK_SECOND_CALL\" ]; then\n",
        "  printf '%s' '{\"schema_version\":\"beacon.relu_search.v1\",\"semantics_note\":\"real-valued semantics; no floating-point roundoff soundness claim\",\"verdict\":\"unknown\",\"tree\":[],\"reason\":\"forced unknown\"}'\n",
        "else\n",
        "  touch \"$MOCK_SECOND_CALL\"\n",
        "  printf '%s' '{\"schema_version\":\"beacon.relu_search.v1\",\"semantics_note\":\"real-valued semantics; no floating-point roundoff soundness claim\",\"verdict\":\"certified\",\"tree\":[],\"final_bound\":[{\"lo\":0.0,\"hi\":1.0},{\"lo\":-2.0,\"hi\":-1.0}]}'\n",
        "fi\n",
    )).expect("mock binary");
    fs::set_permissions(&beacon, fs::Permissions::from_mode(0o755)).expect("executable");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_BEACON_BIN", &beacon)
        .env("MOCK_SECOND_CALL", root.join("second-call"))
        .args(["prove", "src/main.ch", "--tier", "beacon-only", "--json"])
        .output()
        .expect("prove");
    let record: Value = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("NDJSON"))
        .find(|record| record["kind"] == "property")
        .expect("property record");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(record["samples"], 0);
    assert_eq!(
        record["engine_evidence"]["bounds"]["lower"]["result"],
        "proved"
    );
    assert_eq!(
        record["engine_evidence"]["bounds"]["upper"]["result"],
        "unknown"
    );
    assert_eq!(
        record["engine_evidence"]["bounds"]["lower"]["engine_evidence"]["output_root"],
        record["engine_evidence"]["bounds"]["upper"]["engine_evidence"]["output_root"]
    );
}

#[test]
fn explicit_tensor_bridge_control_for_imported_scalar_call() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef affine(x: f64) -> f64 = x + 1.0f64\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property tensor_bridge forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:\n  tensor_to_scalar(scalar_to_tensor(affine(tensor_to_scalar(x)))) <= 2.0f64\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(
        record["reason"], "lowered entry has no named root",
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

#[test]
fn unsupported_scalar_operation_stays_on_beacon_with_no_samples() {
    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (rounded)\ndef rounded(x: f64) -> f64 = round(x)\n",
    );
    let record = prove(
        &root,
        "module BeaconPkg.Main\nimport BeaconPkg.Model (rounded)\n@property rounded_bound forall(x: f64) where x >= -1.0f64, x <= 1.0f64:\n  rounded(x) <= 2.0f64\n",
    );
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    assert!(
        record["reason"]
            .as_str()
            .is_some_and(|s| s.contains("round")),
        "{record}"
    );
}

#[cfg(unix)]
#[test]
fn scalar_graph_uses_selected_private_helper_not_same_named_decoy() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, root) = package(
        "module BeaconPkg.Model\nexport (affine)\ndef helper(x: f64) -> f64 = x - 1.0f64\ndef affine(x: f64) -> f64 = helper(x)\n",
    );
    fs::write(
        root.join("src/other.ch"),
        "module BeaconPkg.Other\nexport (helper)\ndef helper(x: f64) -> f64 = x * x\n",
    )
    .expect("decoy module");
    let source = "module BeaconPkg.Main\nimport BeaconPkg.Model (affine)\n@property bounded forall(x: f64) where x >= -1.0f64, x <= 1.0f64:\n  affine(x) <= 2.0f64\n";
    fs::write(root.join("src/main.ch"), source).expect("property");
    let beacon = root.join("capture-beacon");
    fs::write(
        &beacon,
        "#!/bin/sh\ncp \"$3\" \"$CAPTURE_REQUEST\"\nexit 3\n",
    )
    .expect("capture binary");
    fs::set_permissions(&beacon, fs::Permissions::from_mode(0o755)).expect("executable");
    let capture = root.join("captured-request.json");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_BEACON_BIN", &beacon)
        .env("CAPTURE_REQUEST", &capture)
        .args(["prove", "src/main.ch", "--tier", "beacon-only", "--json"])
        .output()
        .expect("prove");
    let records = String::from_utf8(output.stdout).expect("utf8");
    let record: Value = records
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("NDJSON"))
        .find(|record| record["kind"] == "property")
        .expect("property record");
    assert_ne!(record["status"], "passed", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    let request: Value = serde_json::from_slice(&fs::read(capture).expect("captured request"))
        .expect("request JSON");
    let nodes = request["dag"]["nodes"].as_array().expect("graph nodes");
    assert!(
        nodes.iter().any(|node| node["op"]["kind"] == "add"),
        "{request}"
    );
    assert!(
        nodes.iter().any(|node| node["op"]["kind"] == "neg"),
        "{request}"
    );
    assert!(
        !nodes.iter().any(|node| node["op"]["kind"] == "sub"),
        "{request}"
    );
    assert!(
        !nodes.iter().any(|node| node["op"]["kind"] == "mul"),
        "{request}"
    );
    let declarations = request["dag"]["declarations"]
        .as_array()
        .expect("declarations");
    assert!(
        declarations.iter().any(|name| name
            .as_str()
            .is_some_and(|s| s.contains("Model") && s.contains("helper"))),
        "{request}"
    );
    assert!(
        !declarations.iter().any(|name| name
            .as_str()
            .is_some_and(|s| s.contains("Other") && s.contains("helper"))),
        "{request}"
    );
    assert_eq!(request["inputs"]["x"]["lo"], -1.0);
    assert_eq!(request["inputs"]["x"]["hi"], 1.0);
    let source = record["engine_evidence"]["source_binding"]["declarations"]
        .as_array()
        .expect("selected linked source declarations");
    assert_eq!(source.len(), 2, "{record}");
    assert!(
        source
            .iter()
            .any(|decl| decl["name"] == "BeaconPkg.Model.affine")
    );
    assert!(
        source
            .iter()
            .any(|decl| decl["name"] == "BeaconPkg.Model.helper")
    );
    assert!(source.iter().all(|decl| {
        decl["name"]
            .as_str()
            .is_some_and(|name| !name.contains("pkg__") && !name.contains("Other"))
    }));
}
