//! The wire leg obtains authority only by running the current verifier.
//! A baseline or StaticSurfaceDescriptor cannot construct these witnesses.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::capacity_census_authority::FinalAuthority;
use super::managed_python;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
enum AuthorityClass {
    TaggedTransport,
    NumericOperation,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WireRow {
    kind: String,
    id: String,
    flags: Vec<String>,
    authority: AuthorityClass,
    contract: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Baseline {
    version: u32,
    rows: Vec<WireRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifierOutput {
    version: u32,
    rows: Vec<WireRow>,
    source_sha256: String,
    graph_identity: String,
    evidence: serde_json::Value,
}

/// Its fields and constructors are private. Only `discover` runs the artifact,
/// codec, reconstruction and publication verifier that can establish authority.
#[derive(Debug)]
pub struct VerifiedWireCensus {
    baseline: Baseline,
    source_sha256: String,
    graph_identity: String,
    evidence: serde_json::Value,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("census crate is under <workspace>/crates")
        .to_path_buf()
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_rows(rows: &[WireRow]) -> Result<(), String> {
    if rows.is_empty() {
        return Err("wire verifier selected zero numeric leaves".into());
    }
    let mut identities = BTreeSet::new();
    let mut previous = None;
    for row in rows {
        if row.kind != "wire-schema-numeric-field"
            || row.id.is_empty()
            || row.contract.is_empty()
            || !(row.flags == ["numeric-field"] || row.flags == ["float-carrier"])
        {
            return Err("invalid wire leaf descriptor or missing final contract".into());
        }
        if !identities.insert(&row.id) || previous.is_some_and(|id| id >= &row.id) {
            return Err("wire identities are duplicated or not in canonical order".into());
        }
        previous = Some(&row.id);
    }
    Ok(())
}

/// No saved artifact, descriptor, baseline or alternate program is accepted as
/// input. The fixed enumerator performs the expensive proof in its own target.
pub fn discover() -> Result<VerifiedWireCensus, String> {
    let root = root();
    let python = managed_python::managed_python(&root)?;
    let output = Command::new(python)
        .arg(root.join("scripts/capacity_census_typed.py"))
        .arg("wire")
        .current_dir(&root)
        .output()
        .map_err(|error| format!("cannot execute actual wire verifier: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "actual wire verifier failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    let report: VerifierOutput = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("invalid actual wire verifier result: {error}"))?;
    if report.version != 2 || !digest(&report.source_sha256) || !digest(&report.graph_identity) {
        return Err("wire verifier omitted its current artifact identity".into());
    }
    validate_rows(&report.rows)?;
    if !report.evidence.is_object() || report.evidence.as_object().is_none_or(|map| map.is_empty())
    {
        return Err("wire verifier omitted its execution evidence".into());
    }
    Ok(VerifiedWireCensus {
        baseline: Baseline {
            version: 2,
            rows: report.rows,
        },
        source_sha256: report.source_sha256,
        graph_identity: report.graph_identity,
        evidence: report.evidence,
    })
}

impl VerifiedWireCensus {
    pub fn compare_baseline(&self, bytes: &[u8]) -> Result<(), String> {
        let baseline: Baseline = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid final wire baseline: {error}"))?;
        if baseline.version != 2 {
            return Err("wire baseline is not the zero-exception format".into());
        }
        validate_rows(&baseline.rows)?;
        if baseline != self.baseline {
            return Err(
                "wire baseline differs from executed final authority; a baseline edit supplies no admission"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn baseline_json(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec_pretty(&self.baseline).map_err(|error| error.to_string())
    }

    pub fn classifications(&self) -> Vec<(&str, FinalAuthority<'_>)> {
        self.baseline
            .rows
            .iter()
            .map(|row| {
                let authority = match row.authority {
                    AuthorityClass::TaggedTransport => FinalAuthority::TaggedTransport,
                    AuthorityClass::NumericOperation => FinalAuthority::NumericOperation {
                        atom: &row.contract,
                    },
                };
                (row.id.as_str(), authority)
            })
            .collect()
    }

    pub fn evidence_json(&self) -> serde_json::Value {
        serde_json::json!({
            "source_sha256": self.source_sha256,
            "graph_identity": self.graph_identity,
            "evidence": self.evidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> WireRow {
        WireRow {
            kind: "wire-schema-numeric-field".into(),
            id: "source::Span.offset: u64".into(),
            flags: vec!["numeric-field".into()],
            authority: AuthorityClass::TaggedTransport,
            contract: "source-byte-coordinate".into(),
        }
    }

    #[test]
    fn wire_baseline_rejects_exception_fields_and_nonfinal_classes() {
        let value = serde_json::to_value(Baseline {
            version: 2,
            rows: vec![row()],
        })
        .unwrap();
        assert!(serde_json::from_value::<Baseline>(value.clone()).is_ok());
        for key in [
            "citation",
            "grandfather",
            "successor_overrides",
            "integer_plumbing",
        ] {
            let mut changed = value.clone();
            changed[key] = serde_json::json!("reviewed");
            assert!(serde_json::from_value::<Baseline>(changed).is_err());
        }
        for class in [
            "Nonnumeric",
            "grandfather",
            "permanent-disposition",
            "integer-plumbing",
        ] {
            let mut changed = value.clone();
            changed["rows"][0]["authority"] = serde_json::json!(class);
            assert!(serde_json::from_value::<Baseline>(changed).is_err());
        }
    }

    #[test]
    fn wire_rows_cannot_erase_capacity_or_duplicate_an_identity() {
        assert!(validate_rows(&[row()]).is_ok());
        assert!(validate_rows(&[]).is_err());
        assert!(validate_rows(&[row(), row()]).is_err());
        let mut unflagged = row();
        unflagged.flags.clear();
        assert!(validate_rows(&[unflagged]).is_err());
        let mut unregistered = row();
        unregistered.contract.clear();
        assert!(validate_rows(&[unregistered]).is_err());
    }
}
