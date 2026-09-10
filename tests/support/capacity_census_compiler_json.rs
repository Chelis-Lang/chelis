//! Numeric Python bindings require their actual current execution factories.
//! Descriptors and baseline rows cannot construct this witness.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::capacity_census_authority::SurfaceDescriptor;
use super::managed_python;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DiscoveredRow {
    pub kind: String,
    pub id: String,
    pub flags: Vec<String>,
    pub legacy_flags: Vec<String>,
    pub identity: Option<String>,
    pub problem: Option<String>,
    pub implementation: String,
    pub authority: Option<String>,
    pub contract: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExecutionReport {
    version: u32,
    rows: Vec<DiscoveredRow>,
    compiler_json: serde_json::Value,
    native: serde_json::Value,
}

/// Only `discover` creates this value by executing the fixed private verifier.
#[derive(Debug)]
pub struct VerifiedBindingCensus {
    rows: Vec<DiscoveredRow>,
    evidence: serde_json::Value,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("binding census crate is under <workspace>/crates")
        .to_path_buf()
}

fn digest(value: &serde_json::Value) -> bool {
    value.as_str().is_some_and(|text| {
        text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn report_shape(report: &ExecutionReport) -> Result<(), String> {
    if report.version != 2
        || !digest(&report.compiler_json["source_sha256"])
        || !digest(&report.compiler_json["wire_graph_identity"])
        || report.compiler_json["ownership"]
            .as_array()
            .is_none_or(|items| items.len() != 5)
        || !digest(&report.native["source_sha256"])
        || !digest(&report.native["registration_packet_sha256"])
        || !digest(&report.native["compiler_identity"])
        || report.native["ownership"]
            .as_array()
            .is_none_or(|items| items.len() != 4)
        || report.native["execution"]["selected"] != 37
        || report.native["execution"]["captures"] != 49
        || report.native["execution"]["binaries"] != 6
        || !digest(&report.native["execution"]["packet_sha256"])
        || !digest(&report.native["execution"]["identity_sha256"])
    {
        return Err("binding execution omitted its current codec/native evidence".into());
    }
    let mut found = std::collections::BTreeSet::new();
    for row in &report.rows {
        let Some(authority) = row.authority.as_deref() else {
            continue;
        };
        if authority == "nonnumeric" {
            continue;
        }
        let Some(contract) = &row.contract else {
            return Err("binding authority lacks its exact contract".into());
        };
        let key = if let Some(owner) = contract.strip_prefix("compiler-json/chelis_python::") {
            if !["check_json", "compile_json", "desugar_json", "eval_json"].contains(&owner)
                || authority != "TaggedTransport"
                || row.kind != "binding-pyfunction"
                || !row.id.starts_with(&format!("chelis_python::{owner}("))
                || row.implementation != format!("chelis_python::{owner}#function")
            {
                return Err("invalid executed compiler JSON transport".into());
            }
            format!("compiler-json/{owner}")
        } else {
            let (public, implementation, expected_authority) = match contract.as_str() {
                "native/compiled-tensor-call" => (
                    "chelis_python::CompiledModel::__call__",
                    "chelis_python::NativeCompiledModel::__call__#method",
                    "TaggedTransport",
                ),
                "native/dlpack-capsule" => (
                    "chelis_python::NativeTensor::__dlpack__",
                    "chelis_python::NativeTensor::__dlpack__#method",
                    "TaggedTransport",
                ),
                "native/dlpack-device" => (
                    "chelis_python::NativeTensor::__dlpack_device__",
                    "chelis_python::NativeTensor::__dlpack_device__#method",
                    "TaggedTransport",
                ),
                "[05-OP-45]" => (
                    "chelis_python::NativeTensor::shape",
                    "chelis_python::NativeTensor::shape#getter",
                    "NumericOperation",
                ),
                _ => return Err("unowned native binding authority".into()),
            };
            if authority != expected_authority
                || row.kind != "binding-pymethod"
                || !row.id.starts_with(&format!("{public}("))
                || row.implementation != implementation
            {
                return Err("invalid executed native binding authority".into());
            }
            contract.clone()
        };
        if !found.insert(key)
            || row.flags.is_empty()
            || row.identity.is_none()
            || row.problem.is_some()
        {
            return Err("invalid or duplicated executed binding authority".into());
        }
    }
    if found.len() != 8 {
        return Err("execution omitted a final numeric binding".into());
    }
    Ok(())
}

/// Inputs identify live registrations only. The command accepts no alternate
/// verifier, descriptor, graph, codec receipt, saved output or baseline.
pub fn discover(
    functions: &[String],
    methods: &[String],
    classes: &[(String, String)],
    provenance: &impl Serialize,
) -> Result<VerifiedBindingCensus, String> {
    let root = root();
    let mut command = Command::new(managed_python::managed_python(&root)?);
    let file = tempfile::NamedTempFile::new().map_err(|error| error.to_string())?;
    serde_json::to_writer(file.as_file(), provenance).map_err(|error| error.to_string())?;
    command
        .arg(root.join("scripts/capacity_census_typed.py"))
        .arg("bindings-discovery")
        .arg("--registered-provenance")
        .arg(file.path())
        .current_dir(&root);
    for name in functions {
        command.args(["--registered", name]);
    }
    for name in methods {
        command.args(["--registered-method", name]);
    }
    for (name, identity) in classes {
        command.args(["--registered-class", &format!("{name}={identity}")]);
    }
    let output = command.output().map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "actual binding verifier failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let report: ExecutionReport =
        serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())?;
    report_shape(&report)?;
    let evidence = serde_json::to_value(&report).map_err(|error| error.to_string())?;
    Ok(VerifiedBindingCensus {
        rows: report.rows,
        evidence,
    })
}

impl VerifiedBindingCensus {
    pub fn rows(&self) -> &[DiscoveredRow] {
        &self.rows
    }

    pub fn permits(&self, surface: &SurfaceDescriptor) -> bool {
        surface.family == "pyo3-binding"
            && self.rows.iter().any(|row| {
                row.kind == surface.kind
                    && row.id == surface.id
                    && row.flags == surface.flags
                    && matches!(
                        row.authority.as_deref(),
                        Some("TaggedTransport" | "NumericOperation")
                    )
                    && row.problem.is_none()
            })
    }

    pub fn evidence(&self) -> &serde_json::Value {
        &self.evidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_report_shape_cannot_omit_or_duplicate_a_conversion() {
        // Shape validation is deliberately separate from witness construction.
        let rows =
            ["check_json", "compile_json", "desugar_json", "eval_json"].map(|name| DiscoveredRow {
                kind: "binding-pyfunction".into(),
                id: format!("chelis_python::{name}(typed)"),
                flags: vec!["numeric-return".into()],
                legacy_flags: vec![],
                identity: Some("a".repeat(64)),
                problem: None,
                implementation: format!("chelis_python::{name}#function"),
                authority: Some("TaggedTransport".into()),
                contract: Some(format!("compiler-json/chelis_python::{name}")),
            });
        let native_rows = [
            (
                "chelis_python::CompiledModel::__call__",
                "chelis_python::NativeCompiledModel::__call__#method",
                "TaggedTransport",
                "native/compiled-tensor-call",
            ),
            (
                "chelis_python::NativeTensor::__dlpack__",
                "chelis_python::NativeTensor::__dlpack__#method",
                "TaggedTransport",
                "native/dlpack-capsule",
            ),
            (
                "chelis_python::NativeTensor::__dlpack_device__",
                "chelis_python::NativeTensor::__dlpack_device__#method",
                "TaggedTransport",
                "native/dlpack-device",
            ),
            (
                "chelis_python::NativeTensor::shape",
                "chelis_python::NativeTensor::shape#getter",
                "NumericOperation",
                "[05-OP-45]",
            ),
        ]
        .map(
            |(public, implementation, authority, contract)| DiscoveredRow {
                kind: "binding-pymethod".into(),
                id: format!("{public}(typed)"),
                flags: vec!["numeric-return".into()],
                legacy_flags: vec![],
                identity: Some("c".repeat(64)),
                problem: None,
                implementation: implementation.into(),
                authority: Some(authority.into()),
                contract: Some(contract.into()),
            },
        );
        let mut report = ExecutionReport {
            version: 2,
            rows: rows.to_vec(),
            compiler_json: serde_json::json!({
                "source_sha256":"a".repeat(64), "wire_graph_identity":"b".repeat(64), "ownership":[1,2,3,4,5]
            }),
            native: serde_json::json!({
                "source_sha256":"a".repeat(64),
                "registration_packet_sha256":"b".repeat(64),
                "compiler_identity":"c".repeat(64),
                "ownership":[1,2,3,4],
                "execution":{
                    "packet_sha256":"d".repeat(64),
                    "identity_sha256":"e".repeat(64),
                    "selected":37,
                    "captures":49,
                    "binaries":6
                }
            }),
        };
        report.rows.extend(native_rows.clone());
        assert!(report_shape(&report).is_ok());
        report.rows.pop();
        assert!(report_shape(&report).is_err());
        report.rows.push(report.rows[0].clone());
        assert!(report_shape(&report).is_err());
        report.rows = rows.into_iter().chain(native_rows).collect();
        report.compiler_json["ownership"] = serde_json::json!([]);
        assert!(report_shape(&report).is_err());
    }
}
