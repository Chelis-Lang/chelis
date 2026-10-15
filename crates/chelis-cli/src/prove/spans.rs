//! Verified source attribution for executable properties.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub(crate) struct VerifiedSpans {
    manifest: PathBuf,
    artifact_hash: String,
    properties: BTreeMap<String, RequirementLocation>,
}

#[derive(Debug, Clone)]
pub(super) struct RequirementLocation {
    pub id: String,
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub text: String,
    source_hash: String,
    span: ByteSpan,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteSpan {
    start_byte: usize,
    end_byte: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Clause {
    kind: String,
    span: ByteSpan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Requirement {
    id: String,
    source: String,
    span: ByteSpan,
    clauses: Vec<Clause>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PropertyMapping {
    name: String,
    requirement_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: usize,
    artifact: FileIdentity,
    sources: Vec<FileIdentity>,
    requirements: Vec<Requirement>,
    properties: Vec<PropertyMapping>,
}

#[derive(Deserialize)]
struct LegacyManifest {
    source: String,
    source_hash: String,
    spans: Vec<LegacyEntry>,
}

#[derive(Deserialize)]
struct LegacyEntry {
    deep_node_id: String,
    ears_id: String,
    ears_file: String,
    ears: LegacySpan,
}

#[derive(Deserialize)]
struct LegacySpan {
    start_byte: usize,
    end_byte: usize,
}

impl VerifiedSpans {
    pub(super) fn load(path: &Path, artifact: &Path) -> Result<Self, String> {
        let path = fs::canonicalize(path)
            .map_err(|err| format!("read spans {}: {err}", path.display()))?;
        let text = read(&path)?;
        let header: Value = serde_json::from_str(&text)
            .map_err(|err| format!("parse spans {}: {err}", path.display()))?;
        let artifact_source = read(artifact)?;
        let properties = if header.get("schema_version").is_some() {
            let manifest: Manifest = serde_json::from_str(&text)
                .map_err(|err| format!("parse spans {}: {err}", path.display()))?;
            if manifest.schema_version != 1 {
                return Err(format!(
                    "unsupported spans schema_version {}",
                    manifest.schema_version
                ));
            }
            let base = path.parent().ok_or("spans path has no parent")?;
            if canonical(&base.join(&manifest.artifact.path))? != canonical(artifact)? {
                return Err("spans artifact path does not match the selected input".into());
            }
            verify_hash(
                &artifact_source,
                &manifest.artifact.sha256,
                "generated artifact",
            )?;
            let mut sources = BTreeMap::new();
            let mut source_paths = BTreeSet::new();
            for source in manifest.sources {
                let resolved = canonical(&base.join(&source.path))?;
                if !source_paths.insert(resolved.clone()) || sources.contains_key(&source.path) {
                    return Err(format!("duplicate spans source {}", source.path));
                }
                let bytes = read(&resolved)?;
                verify_hash(&bytes, &source.sha256, &source.path)?;
                sources.insert(source.path, (bytes, source.sha256));
            }
            let mut requirements = BTreeMap::new();
            for requirement in manifest.requirements {
                if requirement.id.is_empty() || requirements.contains_key(&requirement.id) {
                    return Err(format!(
                        "empty or duplicate requirement ID {}",
                        requirement.id
                    ));
                }
                let (source, source_hash) = sources.get(&requirement.source).ok_or_else(|| {
                    format!("unverified requirement source {}", requirement.source)
                })?;
                validate_span(source, &requirement.span)?;
                let mut clause_kinds = BTreeSet::new();
                for clause in requirement.clauses {
                    validate_span(source, &clause.span)?;
                    if clause.kind.is_empty()
                        || !clause_kinds.insert(clause.kind)
                        || clause.span.start_byte < requirement.span.start_byte
                        || clause.span.end_byte > requirement.span.end_byte
                    {
                        return Err("invalid, duplicate, or out-of-requirement clause span".into());
                    }
                }
                let location = location(
                    &requirement.id,
                    &requirement.source,
                    source_hash,
                    source,
                    requirement.span,
                )?;
                requirements.insert(requirement.id, location);
            }
            let mut properties = BTreeMap::new();
            for property in manifest.properties {
                let requirement = requirements
                    .get(&property.requirement_id)
                    .ok_or_else(|| format!("unknown requirement ID {}", property.requirement_id))?;
                if property.name.is_empty()
                    || properties
                        .insert(property.name.clone(), requirement.clone())
                        .is_some()
                {
                    return Err(format!(
                        "empty or duplicate property mapping {}",
                        property.name
                    ));
                }
            }
            properties
        } else {
            if artifact.extension().and_then(|ext| ext.to_str()) != Some("dp") {
                return Err("Surf spans require schema_version 1".into());
            }
            let legacy: LegacyManifest = serde_json::from_str(&text)
                .map_err(|err| format!("parse legacy spans {}: {err}", path.display()))?;
            // Legacy c-earchin paths are relative to the invocation directory.
            let source_path = canonical(Path::new(&legacy.source))?;
            let source = read(&source_path)?;
            verify_hash(&source, &legacy.source_hash, &legacy.source)?;
            let mut properties = BTreeMap::new();
            let mut ids = BTreeSet::new();
            for entry in legacy.spans {
                if canonical(Path::new(&entry.ears_file))? != source_path {
                    return Err("legacy requirement source differs from the hashed source".into());
                }
                if entry.ears_id.is_empty() || !ids.insert(entry.ears_id.clone()) {
                    return Err("empty or duplicate legacy requirement ID".into());
                }
                let span = ByteSpan {
                    start_byte: entry.ears.start_byte,
                    end_byte: entry.ears.end_byte,
                };
                let location = location(
                    &entry.ears_id,
                    &entry.ears_file,
                    &legacy.source_hash,
                    &source,
                    span,
                )?;
                if properties.insert(entry.deep_node_id, location).is_some() {
                    return Err("duplicate legacy property mapping".into());
                }
            }
            properties
        };
        Ok(Self {
            manifest: path,
            artifact_hash: hash(&artifact_source),
            properties,
        })
    }

    pub(super) fn verify_artifact(&self, source: &str) -> Result<(), String> {
        verify_hash(
            source,
            &self.artifact_hash,
            "generated artifact changed during discovery",
        )
    }

    pub(super) fn verify_properties(
        &self,
        names: &[(String, Option<String>)],
    ) -> Result<(), String> {
        let discovered = names.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
        let mapped = self.properties.keys().collect::<BTreeSet<_>>();
        if discovered != mapped {
            return Err("spans property mappings do not match the discovered properties".into());
        }
        for (name, source_id) in names {
            if let Some(source_id) = source_id
                && self.properties[name].id != *source_id
            {
                return Err(format!(
                    "spans requirement ID does not match property {name}"
                ));
            }
        }
        Ok(())
    }

    pub(super) fn get(&self, name: &str) -> Option<&RequirementLocation> {
        self.properties.get(name)
    }

    pub(super) fn source_json(&self, file: &Path, kind: &str, name: &str) -> Value {
        let mut value = json!({"kind":kind,"file":file.display().to_string(),"spans":self.manifest.display().to_string(),"artifact_hash":self.artifact_hash});
        if let Some(source) = self.get(name) {
            value["requirement"] = json!({"id":source.id,"file":source.file,"line":source.line,"column":source.column,"text":source.text,"source_hash":source.source_hash,"span":{"start_byte":source.span.start_byte,"end_byte":source.span.end_byte}});
        }
        value
    }
}

fn location(
    id: &str,
    file: &str,
    source_hash: &str,
    source: &str,
    span: ByteSpan,
) -> Result<RequirementLocation, String> {
    validate_span(source, &span)?;
    let before = &source[..span.start_byte];
    let line = before.bytes().filter(|b| *b == b'\n').count() + 1;
    let column = before
        .rfind('\n')
        .map_or(before.len() + 1, |at| before.len() - at);
    Ok(RequirementLocation {
        id: id.into(),
        file: file.into(),
        line,
        column,
        text: source[span.start_byte..span.end_byte].into(),
        source_hash: source_hash.into(),
        span,
    })
}

fn validate_span(source: &str, span: &ByteSpan) -> Result<(), String> {
    if span.start_byte >= span.end_byte || source.get(span.start_byte..span.end_byte).is_none() {
        return Err(
            "spans byte range must be nonempty and lie on UTF-8 boundaries within its source"
                .into(),
        );
    }
    Ok(())
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))
}

fn canonical(path: &Path) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|err| format!("resolve spans path {}: {err}", path.display()))
}

fn hash(source: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(source.as_bytes()))
}

fn verify_hash(source: &str, expected: &str, subject: &str) -> Result<(), String> {
    if hash(source) != expected {
        return Err(format!("spans hash mismatch for {subject}"));
    }
    Ok(())
}
