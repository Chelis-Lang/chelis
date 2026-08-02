//! Phase-1 entry hard edge for chelis#729's typed wire-schema census.
//!
//! Owning contract: `spec/design/dtype_semantics.md` section C6. This test
//! inventories carrier shape only. Root identity, manifest ordering,
//! dotted-root expansion, `requires_main`, artifact routing, and `HostReason`
//! remain chelis#912's authority.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const FROZEN_BASELINE_SHA256: &str =
    "11eca7c5350703bc05d8f77a3fd7a3c9d2963469c583405c3387be86ad0f620e";

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct SurfaceRow {
    kind: String,
    id: String,
    flags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Baseline {
    version: u32,
    citation: String,
    rows: Vec<SurfaceRow>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler-api crate lives under <workspace>/crates")
        .to_path_buf()
}

fn run_typed_enumerator() -> Output {
    let root = workspace_root();
    Command::new(root.join(".venv/bin/python"))
        .arg(root.join("scripts/capacity_census_typed.py"))
        .args(["wire", "--target-dir"])
        .arg(root.join("target/agents/729-capacity-wire-rustdoc"))
        .current_dir(&root)
        .output()
        .expect("run typed wire census enumerator")
}

fn baseline_bytes() -> Vec<u8> {
    std::fs::read(workspace_root().join("spec/design/capacity_census_wire.json"))
        .expect("read reviewed wire census baseline")
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[test]
fn wire_schema_numeric_fields_match_the_reviewed_baseline() {
    let baseline_bytes = baseline_bytes();
    assert_eq!(
        sha256(&baseline_bytes),
        FROZEN_BASELINE_SHA256,
        "wire census guard changed: regenerating or editing the baseline is not the fix; \
         review the carrier under dtype_semantics.md section C6 and update the frozen \
         fingerprint only with that disposition"
    );
    let baseline: Baseline = serde_json::from_slice(&baseline_bytes).expect("wire baseline JSON");
    assert_eq!(baseline.version, 1, "unknown wire census baseline version");
    assert!(
        baseline.citation.contains("chelis#729"),
        "the frozen wire baseline must remain liveness-bound to chelis#729"
    );

    let output = run_typed_enumerator();
    assert!(
        output.status.success(),
        "typed wire census failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let current: Vec<SurfaceRow> =
        serde_json::from_slice(&output.stdout).expect("typed wire census JSON");
    assert_eq!(
        current, baseline.rows,
        "public serialized numeric wire carrier shape changed. Do not decide root/manifest \
         semantics here. For the carrier itself, use the tagged payload, remove the new \
         numeric channel, or obtain the explicit C6 review disposition; then update the \
         reviewed baseline and frozen fingerprint together"
    );
}

#[test]
fn adding_or_removing_a_public_serialized_f64_field_changes_the_census() {
    let root = workspace_root();
    let output = Command::new(root.join(".venv/bin/python"))
        .arg(root.join("scripts/test_capacity_census_typed.py"))
        .arg("WireEnumerator")
        .current_dir(&root)
        .output()
        .expect("run typed wire mutation tests");
    assert!(
        output.status.success(),
        "wire add/remove mutation controls failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
