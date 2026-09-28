//! Shared compiled-library JSON metadata; spec/11 §1.4 owns the protocol.

use super::CompileTarget;
use crate::compiler::ExecutionTensorSpec;
use serde::{Deserialize, Serialize};

/// A closed protocol discriminant, not a numeric payload or dtype selector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub enum ArtifactAbiVersion {
    V2,
}

impl TryFrom<u32> for ArtifactAbiVersion {
    type Error = String;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::V2),
            _ => Err(format!(
                "unsupported artifact ABI version {value}; expected 2"
            )),
        }
    }
}

impl From<ArtifactAbiVersion> for u32 {
    fn from(version: ArtifactAbiVersion) -> Self {
        match version {
            ArtifactAbiVersion::V2 => 2,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompiledArtifactManifest {
    pub abi_version: ArtifactAbiVersion,
    pub target: CompileTarget,
    pub host_entry_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_entry_name: Option<String>,
    pub inputs: Vec<ExecutionTensorSpec>,
    pub outputs: Vec<ExecutionTensorSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbolic_dims: Vec<String>,
    pub source_path: String,
    pub source_hash: String,
    /// Lowercase hexadecimal SHA-256 of the runtime archive the writing
    /// extension carried and linked (spec/08 §2.1, spec/11 §1.4).
    pub runtime_sha256: String,
    /// Lowercase hexadecimal SHA-256 of the compiled library's bytes.
    pub library_sha256: String,
}

#[derive(Deserialize)]
struct ArtifactAbiHeader {
    abi_version: ArtifactAbiVersion,
}

#[derive(Deserialize)]
struct CompiledArtifactManifestFields {
    abi_version: ArtifactAbiVersion,
    target: CompileTarget,
    host_entry_name: String,
    device_entry_name: Option<String>,
    inputs: Vec<ExecutionTensorSpec>,
    outputs: Vec<ExecutionTensorSpec>,
    #[serde(default)]
    symbolic_dims: Vec<String>,
    source_path: String,
    source_hash: String,
    runtime_sha256: String,
    library_sha256: String,
}

impl<'de> Deserialize<'de> for CompiledArtifactManifest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Box::<serde_json::value::RawValue>::deserialize(deserializer)?;
        let ArtifactAbiHeader {
            abi_version: ArtifactAbiVersion::V2,
        } = serde_json::from_str(raw.get()).map_err(|error| {
            serde::de::Error::custom(format!("invalid artifact ABI version: {error}"))
        })?;
        let fields: CompiledArtifactManifestFields =
            serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        Ok(Self {
            abi_version: fields.abi_version,
            target: fields.target,
            host_entry_name: fields.host_entry_name,
            device_entry_name: fields.device_entry_name,
            inputs: fields.inputs,
            outputs: fields.outputs,
            symbolic_dims: fields.symbolic_dims,
            source_path: fields.source_path,
            source_hash: fields.source_hash,
            runtime_sha256: fields.runtime_sha256,
            library_sha256: fields.library_sha256,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ArtifactAbiVersion, CompiledArtifactManifest};
    use serde_json::json;

    #[test]
    fn artifact_manifest_shared_codec_preserves_version_and_exact_extents() {
        for size in [0, 1, 9_007_199_254_740_993, i64::MAX] {
            let expected = json!({
                "abi_version": 2, "target": "c", "host_entry_name": "chelis_main",
                "inputs": [{"name": "x", "dtype": "float32", "dims": [{"name": "n", "size": size}]}],
                "outputs": [], "source_path": "model.chelis", "source_hash": "digest",
                "runtime_sha256": "runtime-digest", "library_sha256": "library-digest"
            });
            let manifest: CompiledArtifactManifest =
                serde_json::from_value(expected.clone()).unwrap();
            assert!(matches!(manifest.abi_version, ArtifactAbiVersion::V2));
            assert_eq!(manifest.inputs[0].dims[0].size.unwrap().get(), size);
            assert_eq!(serde_json::to_value(manifest).unwrap(), expected);
        }
    }

    #[test]
    fn artifact_manifest_version_admission_precedes_metadata() {
        for header in [
            "",
            ",\"abi_version\":0",
            ",\"abi_version\":1",
            ",\"abi_version\":3",
            ",\"abi_version\":4294967295",
            ",\"abi_version\":-1",
            ",\"abi_version\":2.0",
            ",\"abi_version\":true",
            ",\"abi_version\":\"2\"",
            ",\"abi_version\":2,\"abi_version\":2",
        ] {
            let error = serde_json::from_str::<CompiledArtifactManifest>(&format!(
                "{{\"inputs\":\"invalid\"{header}}}"
            ))
            .unwrap_err();
            assert!(
                error.to_string().contains("artifact ABI version"),
                "{header}: {error}"
            );
        }
        let error = serde_json::from_str::<CompiledArtifactManifest>(
            r#"{"abi_version":2,"target":"c","host_entry_name":"main","inputs":[{"name":"x","dtype":"float32","dims":[{"name":"n","size":-1}]}],"outputs":[],"source_path":"","source_hash":"","runtime_sha256":"","library_sha256":""}"#
        ).unwrap_err();
        assert!(error.to_string().contains("nonnegative"), "{error}");
    }

    /// spec/11 §1.4: the runtime and library digests have no missing-field
    /// default. A version-2 manifest without either is refused after the ABI
    /// version is admitted.
    #[test]
    fn artifact_manifest_requires_runtime_and_library_digests() {
        let complete = json!({
            "abi_version": 2, "target": "c", "host_entry_name": "chelis_main",
            "inputs": [], "outputs": [], "source_path": "model.chelis", "source_hash": "digest",
            "runtime_sha256": "runtime-digest", "library_sha256": "library-digest"
        });
        for field in ["runtime_sha256", "library_sha256"] {
            let mut incomplete = complete.clone();
            incomplete.as_object_mut().unwrap().remove(field);
            let error = serde_json::from_value::<CompiledArtifactManifest>(incomplete).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("missing field `{field}`")),
                "{field}: {error}"
            );
        }
        let manifest: CompiledArtifactManifest = serde_json::from_value(complete.clone()).unwrap();
        assert_eq!(serde_json::to_value(manifest).unwrap(), complete);
    }
}
