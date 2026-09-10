//! Check envelope versions before materializing numeric payloads. Raw JSON
//! preserves duplicate keys for the actual carrier decoder to reject.

use super::{
    EXECUTION_VALUE_SCHEMA_VERSION, EvalResult, EvaluatedRoot, RootManifestResult, WireApiEnvelope,
    WireBatchResult,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

#[derive(Deserialize)]
struct VersionHeader {
    schema_version: Option<u32>,
}

pub(super) fn version(json: &str) -> Result<Option<u32>, serde_json::Error> {
    serde_json::from_str::<VersionHeader>(json).map(|header| header.schema_version)
}

fn validate_execution_version(found: Option<u32>) -> Result<(), String> {
    if found == Some(EXECUTION_VALUE_SCHEMA_VERSION) {
        Ok(())
    } else {
        Err(format!(
            "execution schema_version must be exactly {EXECUTION_VALUE_SCHEMA_VERSION}, found {found:?}"
        ))
    }
}

#[derive(Deserialize)]
struct ExecutionFields {
    schema_version: u32,
    roots: Vec<EvaluatedRoot>,
    #[serde(default)]
    manifest: RootManifestResult,
    #[serde(default)]
    transcript: Vec<String>,
}

#[derive(Serialize)]
struct ExecutionFieldsRef<'a> {
    schema_version: u32,
    roots: &'a [EvaluatedRoot],
    manifest: &'a RootManifestResult,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    transcript: &'a [String],
}

impl Serialize for EvalResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        validate_execution_version(Some(self.schema_version)).map_err(serde::ser::Error::custom)?;
        ExecutionFieldsRef {
            schema_version: self.schema_version,
            roots: &self.roots,
            manifest: &self.manifest,
            transcript: &self.transcript,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EvalResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let found = version(raw.get()).map_err(serde::de::Error::custom)?;
        validate_execution_version(found).map_err(serde::de::Error::custom)?;
        let fields: ExecutionFields =
            serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        Ok(Self {
            schema_version: fields.schema_version,
            roots: fields.roots,
            manifest: fields.manifest,
            transcript: fields.transcript,
        })
    }
}

// Serde's untagged and internally tagged enum visitors buffer into Content.
// Explicitly dispatch these enclosing response variants from raw bytes too,
// so they cannot discard the raw JSON needed by a nested numeric envelope.
#[derive(Deserialize)]
struct ResponseHeader {
    ok: bool,
}

impl<'de, T: serde::de::DeserializeOwned> Deserialize<'de> for WireApiEnvelope<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let header: ResponseHeader =
            serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        if header.ok {
            serde_json::from_str(raw.get())
                .map(Self::Success)
                .map_err(serde::de::Error::custom)
        } else {
            serde_json::from_str(raw.get())
                .map(Self::Failure)
                .map_err(serde::de::Error::custom)
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum BatchKind {
    Parse,
    Desugar,
    Check,
    Lower,
    Compile,
    Eval,
    Grad,
    Validate,
    Decompile,
}

#[derive(Deserialize)]
struct BatchHeader {
    kind: BatchKind,
}

impl<'de> Deserialize<'de> for WireBatchResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let header: BatchHeader =
            serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        match header.kind {
            BatchKind::Parse => serde_json::from_str(raw.get()).map(Self::Parse),
            BatchKind::Desugar => serde_json::from_str(raw.get()).map(Self::Desugar),
            BatchKind::Check => serde_json::from_str(raw.get()).map(Self::Check),
            BatchKind::Lower => serde_json::from_str(raw.get()).map(Self::Lower),
            BatchKind::Compile => serde_json::from_str(raw.get()).map(Self::Compile),
            BatchKind::Eval => serde_json::from_str(raw.get()).map(Self::Eval),
            BatchKind::Grad => serde_json::from_str(raw.get()).map(Self::Grad),
            BatchKind::Validate => serde_json::from_str(raw.get()).map(Self::Validate),
            BatchKind::Decompile => serde_json::from_str(raw.get()).map(Self::Decompile),
        }
        .map_err(serde::de::Error::custom)
    }
}

fn result_references(
    dag: &super::WireDag,
    ids: impl IntoIterator<Item = u64>,
) -> Result<(), String> {
    let node_count = super::host_index(dag.nodes.len());
    if ids.into_iter().any(|id| id >= node_count) {
        return Err("result reference is outside the owning DAG".to_string());
    }
    Ok(())
}

#[derive(Deserialize)]
struct LowerFields {
    dag: super::WireDag,
    named_roots: std::collections::BTreeMap<String, u64>,
}
#[derive(Serialize)]
struct LowerFieldsRef<'a> {
    dag: &'a super::WireDag,
    named_roots: &'a std::collections::BTreeMap<String, u64>,
}
impl Serialize for super::LowerResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        result_references(&self.dag, self.named_roots.values().copied())
            .map_err(serde::ser::Error::custom)?;
        LowerFieldsRef {
            dag: &self.dag,
            named_roots: &self.named_roots,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for super::LowerResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = LowerFields::deserialize(deserializer)?;
        result_references(&fields.dag, fields.named_roots.values().copied())
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            dag: fields.dag,
            named_roots: fields.named_roots,
        })
    }
}

#[derive(Deserialize)]
struct GradFields {
    dag: super::WireDag,
    output_node: u64,
    grad_nodes_by_name: std::collections::BTreeMap<String, u64>,
    forward_nodes_by_name: std::collections::BTreeMap<String, u64>,
}
#[derive(Serialize)]
struct GradFieldsRef<'a> {
    dag: &'a super::WireDag,
    output_node: u64,
    grad_nodes_by_name: &'a std::collections::BTreeMap<String, u64>,
    forward_nodes_by_name: &'a std::collections::BTreeMap<String, u64>,
}
impl Serialize for super::GradResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        result_references(
            &self.dag,
            std::iter::once(self.output_node)
                .chain(self.grad_nodes_by_name.values().copied())
                .chain(self.forward_nodes_by_name.values().copied()),
        )
        .map_err(serde::ser::Error::custom)?;
        GradFieldsRef {
            dag: &self.dag,
            output_node: self.output_node,
            grad_nodes_by_name: &self.grad_nodes_by_name,
            forward_nodes_by_name: &self.forward_nodes_by_name,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for super::GradResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = GradFields::deserialize(deserializer)?;
        result_references(
            &fields.dag,
            std::iter::once(fields.output_node)
                .chain(fields.grad_nodes_by_name.values().copied())
                .chain(fields.forward_nodes_by_name.values().copied()),
        )
        .map_err(serde::de::Error::custom)?;
        Ok(Self {
            dag: fields.dag,
            output_node: fields.output_node,
            grad_nodes_by_name: fields.grad_nodes_by_name,
            forward_nodes_by_name: fields.forward_nodes_by_name,
        })
    }
}
