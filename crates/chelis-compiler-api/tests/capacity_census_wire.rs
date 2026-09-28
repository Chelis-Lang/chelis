//! The C6 wire acceptance oracle: current graph, exact codec and admission
//! execution, publication ownership, paired mutations, and zero exceptions.
//! Baseline regeneration records an executed classification; it cannot grant one.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/capacity_census_authority.rs"]
#[allow(dead_code)]
mod capacity_census_authority;
#[path = "../../../tests/support/capacity_census_wire_verifier.rs"]
mod capacity_census_wire_verifier;
#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler-api crate lives under <workspace>/crates")
        .to_path_buf()
}

#[test]
fn wire_schema_numeric_fields_match_the_reviewed_baseline() {
    let witness = capacity_census_wire_verifier::discover()
        .unwrap_or_else(|problem| panic!("wire authority verification failed: {problem}"));
    let root = workspace_root();
    let baseline = std::fs::read(root.join("spec/design/capacity_census_wire.json"))
        .expect("read reviewed final wire baseline");
    witness
        .compare_baseline(&baseline)
        .unwrap_or_else(|problem| panic!("{problem}"));
    let classifications = witness.classifications();
    assert!(
        !classifications.is_empty(),
        "the wire oracle cannot select zero leaves"
    );
    assert!(
        classifications.iter().all(|(_, authority)| matches!(
            authority,
            capacity_census_authority::FinalAuthority::TaggedTransport
                | capacity_census_authority::FinalAuthority::NumericOperation { .. }
        )),
        "a numeric leaf cannot inherit a nonnumeric or legacy disposition"
    );
    let serialized = witness
        .baseline_json()
        .expect("serialize executed final rows");
    witness
        .compare_baseline(&serialized)
        .expect("canonical baseline preserves executed authority");
    let receipt = root.join("target/capacity-census-wire-execution.json");
    std::fs::write(
        &receipt,
        serde_json::to_vec_pretty(&witness.evidence_json()).unwrap(),
    )
    .expect("write actual wire acceptance evidence");
    println!(
        "CAPACITY CENSUS WIRE: PASS ({} final numeric leaves; receipt {})",
        classifications.len(),
        receipt.display()
    );
}

fn controls(selection: &str) {
    let root = workspace_root();
    let python = managed_python::managed_python(&root).unwrap_or_else(|error| panic!("{error}"));
    let output = Command::new(python)
        .arg(root.join("scripts/capacity_census_wire_acceptance.py"))
        .arg(selection)
        .current_dir(&root)
        .output()
        .expect("run framework-supervised wire controls");
    assert!(
        output.status.success(),
        "wire controls failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("framework execution receipt");
    let selected = receipt["selected"]
        .as_array()
        .expect("selected test identities");
    assert!(!selected.is_empty(), "wire controls selected zero cases");
    assert_eq!(
        receipt["selected"], receipt["executed"],
        "every selected control executed"
    );
}

#[test]
fn adding_or_removing_a_public_serialized_f64_field_changes_the_census() {
    controls("descriptor-controls");
}

#[test]
fn verified_wire_authority_cannot_be_replaced_by_a_descriptor_or_baseline() {
    controls("authority-controls");
}
