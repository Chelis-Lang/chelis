//! The four CompilerJson bindings require the actual current execution factory.
//! Descriptors and baseline rows cannot construct this witness.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::capacity_census_authority::SurfaceDescriptor;
use super::managed_python;

#[derive(Clone, Debug, Deserialize)]
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionReport {
    version: u32,
    rows: Vec<DiscoveredRow>,
    compiler_json: serde_json::Value,
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
    if report.version != 1
        || !digest(&report.compiler_json["source_sha256"])
        || !digest(&report.compiler_json["wire_graph_identity"])
        || report.compiler_json["ownership"]
            .as_array()
            .is_none_or(|items| items.len() != 5)
    {
        return Err("binding execution omitted its current codec/graph evidence".into());
    }
    let mut found = std::collections::BTreeSet::new();
    for row in &report.rows {
        if row.authority.as_deref() != Some("TaggedTransport") {
            continue;
        }
        let Some(contract) = &row.contract else {
            return Err("binding transport lacks its exact contract".into());
        };
        let Some(owner) = contract.strip_prefix("compiler-json/chelis_python::") else {
            return Err("unowned binding transport".into());
        };
        if !["check_json", "compile_json", "desugar_json", "eval_json"].contains(&owner)
            || !found.insert(owner)
            || row.kind != "binding-pyfunction"
            || !row.id.starts_with(&format!("chelis_python::{owner}("))
            || row.implementation != format!("chelis_python::{owner}#function")
            || row.flags.is_empty()
            || row.identity.is_none()
            || row.problem.is_some()
        {
            return Err("invalid or duplicated executed binding transport".into());
        }
    }
    if found.len() != 4 {
        return Err("execution omitted a CompilerJson binding".into());
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
    Ok(VerifiedBindingCensus {
        rows: report.rows,
        evidence: report.compiler_json,
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
                    && row.authority.as_deref() == Some("TaggedTransport")
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
    fn compiler_json_report_shape_cannot_omit_or_duplicate_a_conversion() {
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
        let mut report = ExecutionReport {
            version: 1,
            rows: rows.to_vec(),
            compiler_json: serde_json::json!({
                "source_sha256":"a".repeat(64), "wire_graph_identity":"b".repeat(64), "ownership":[1,2,3,4,5]
            }),
        };
        assert!(report_shape(&report).is_ok());
        report.rows.pop();
        assert!(report_shape(&report).is_err());
        report.rows.push(rows[0].clone());
        assert!(report_shape(&report).is_err());
        report.rows = rows.to_vec();
        report.compiler_json["ownership"] = serde_json::json!([]);
        assert!(report_shape(&report).is_err());
    }
}
