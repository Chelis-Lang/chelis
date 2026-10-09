//! An explicit Beacon property resolves an imported numerical definition
//! through the checked Reef graph before dispatch.
#![cfg(feature = "chelis-prove")]

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::{TempDir, tempdir};

const LIBRARY: &str = "module BeaconPkg.Model\nexport (neuron)\ndef neuron(x: tensor[f64]) -> tensor[f64] = relu(x)\n";
const PROPERTY: &str = "module BeaconPkg.Main\nimport BeaconPkg.Model (neuron)\n@property bounded forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:\n  tensor_to_scalar(neuron(x)) <= 2.0f64\n";

fn package() -> (TempDir, std::path::PathBuf) {
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
    fs::write(root.join("src/model.ch"), LIBRARY).expect("model");
    fs::write(root.join("src/main.ch"), PROPERTY).expect("property");
    (dir, root)
}

fn prove(root: &Path, source: &str, beacon: Option<&Path>, capture: Option<&Path>) -> Vec<Value> {
    fs::write(root.join("src/main.ch"), source).expect("property");
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1");
    match beacon {
        Some(binary) => {
            command.env("CHELIS_BEACON_BIN", binary);
        }
        None => {
            command.env_remove("CHELIS_BEACON_BIN");
        }
    }
    if let Some(path) = capture {
        command.env("CAPTURE_REQUEST", path);
    }
    let output = command
        .args(["prove", "src/main.ch", "--tier", "beacon-only", "--json"])
        .output()
        .expect("prove");
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    stdout
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("NDJSON"))
        .collect()
}

fn property(records: &[Value]) -> &Value {
    records
        .iter()
        .find(|record| record["kind"] == "property")
        .unwrap_or_else(|| panic!("no property record in {records:#?}"))
}

#[cfg(feature = "chelis-prove")]
#[test]
fn imported_rank_zero_function_reaches_the_explicit_beacon_boundary() {
    let (_dir, root) = package();
    let records = prove(&root, PROPERTY, None, None);
    let record = property(&records);
    assert_eq!(record["name"], "bounded", "{record}");
    assert_eq!(record["proof_tier"], "beacon", "{record}");
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(
        record["reason"], "CHELIS_BEACON_BIN is not configured",
        "{record}"
    );
    assert_eq!(record["samples"], 0, "{record}");
}

#[cfg(feature = "chelis-prove")]
#[test]
fn unresolved_import_never_reaches_beacon() {
    let (_dir, root) = package();
    let source = PROPERTY.replace(
        "import BeaconPkg.Model (neuron)",
        "import BeaconPkg.Model (missing)",
    );
    let records = prove(&root, &source, None, None);
    assert!(
        records.iter().any(|record| record["kind"] == "error"),
        "{records:#?}"
    );
    assert!(
        !records.iter().any(|record| record["status"] == "passed"),
        "{records:#?}"
    );
}

#[cfg(all(feature = "chelis-prove", unix))]
#[test]
fn imported_call_uses_the_selected_linked_declaration_in_the_exact_graph() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, root) = package();
    // Both modules export a `neuron`; the source import resolves Model's
    // declaration, whose ReLU is the operation Beacon must see.
    fs::write(
        root.join("src/other.ch"),
        "module BeaconPkg.Other\nexport (neuron)\ndef neuron(x: tensor[f64]) -> tensor[f64] = -x\n",
    )
    .expect("decoy module");
    let beacon = root.join("capture-beacon");
    fs::write(
        &beacon,
        "#!/bin/sh\ncp \"$3\" \"$CAPTURE_REQUEST\"\nexit 3\n",
    )
    .expect("capture binary");
    fs::set_permissions(&beacon, fs::Permissions::from_mode(0o755)).expect("executable");
    let request_path = root.join("captured-request.json");

    let records = prove(&root, PROPERTY, Some(&beacon), Some(&request_path));
    let record = property(&records);
    assert_ne!(record["status"], "passed", "{record}");
    assert_eq!(record["samples"], 0, "{record}");
    let request: Value = serde_json::from_slice(&fs::read(&request_path).expect("Beacon request"))
        .expect("request JSON");
    assert_eq!(
        record["engine_evidence"]["output_root"], request["dag"]["roots"][0],
        "{record}"
    );
    assert_eq!(
        record["engine_evidence"]["graph_sha256"]
            .as_str()
            .expect("graph hash")
            .len(),
        64,
        "{record}"
    );
    let nodes = request["dag"]["nodes"].as_array().expect("graph nodes");
    assert!(
        nodes.iter().any(|node| node["op"]["kind"] == "relu"),
        "{request}"
    );
    assert!(
        !nodes.iter().any(|node| node["op"]["kind"] == "neg"),
        "{request}"
    );
    // Inlining leaves only generated graph declarations. The outcome binds
    // the selected linked source declaration alongside the graph evidence.
    let declarations = record["engine_evidence"]["source_binding"]["declarations"]
        .as_array()
        .expect("source declarations");
    assert_eq!(declarations.len(), 1, "{record}");
    assert_eq!(
        declarations[0]["name"], "BeaconPkg.Model.neuron",
        "{record}"
    );
    assert!(
        declarations.iter().all(|decl| decl["name"]
            .as_str()
            .is_some_and(|name| !name.contains("pkg__"))),
        "{record}"
    );
    assert!(
        !declarations.iter().any(|decl| decl["name"]
            .as_str()
            .is_some_and(|name| name.contains("Other") && name.contains("neuron"))),
        "{record}"
    );
    assert!(
        record["engine_evidence"]["source_binding"]["expression_sha256"]
            .as_str()
            .is_some(),
        "{record}"
    );

    let chosen = record["engine_evidence"]["source_binding"]["declarations"][0]["sha256"]
        .as_str()
        .expect("selected declaration digest")
        .to_owned();
    // Replacing the selected implementation changes the source digest and
    // selected graph operation. The unrelated same-named export remains inert.
    fs::write(
        root.join("src/model.ch"),
        "module BeaconPkg.Model\nexport (neuron)\ndef neuron(x: tensor[f64]) -> tensor[f64] = -x\n",
    )
    .expect("change selected declaration");
    let records = prove(&root, PROPERTY, Some(&beacon), Some(&request_path));
    let changed = property(&records);
    assert_ne!(
        changed["engine_evidence"]["source_binding"]["declarations"][0]["sha256"], chosen,
        "{changed}"
    );
    let request: Value = serde_json::from_slice(&fs::read(&request_path).expect("Beacon request"))
        .expect("request JSON");
    let nodes = request["dag"]["nodes"].as_array().expect("graph nodes");
    assert!(
        nodes.iter().any(|node| node["op"]["kind"] == "neg"),
        "{request}"
    );
    assert!(
        !nodes.iter().any(|node| node["op"]["kind"] == "relu"),
        "{request}"
    );
}
