use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentDigest([u8; 32]);

impl ContentDigest {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ContentDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for ContentDigest {
    type Err = InputError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(InputError::InvalidDigest);
        }
        let mut bytes = [0; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            fn nibble(byte: u8) -> u8 {
                if byte <= b'9' {
                    byte - b'0'
                } else {
                    byte - b'a' + 10
                }
            }
            bytes[index] = nibble(pair[0]) * 16 + nibble(pair[1]);
        }
        Ok(Self(bytes))
    }
}

impl Serialize for ContentDigest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
impl<'de> Deserialize<'de> for ContentDigest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone)]
pub struct ContentHasher(Sha256);
impl ContentHasher {
    pub fn new() -> Self {
        Self(Sha256::new())
    }
    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
    pub fn finish(self) -> ContentDigest {
        ContentDigest(self.0.finalize().into())
    }
}
impl Default for ContentHasher {
    fn default() -> Self {
        Self::new()
    }
}
pub fn hash_bytes(bytes: &[u8]) -> ContentDigest {
    let mut hasher = ContentHasher::new();
    hasher.update(bytes);
    hasher.finish()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputClass {
    Source,
    Build,
    Header,
    Toolchain,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredInput {
    pub logical_path: String,
    pub class: InputClass,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedInput {
    pub logical_path: String,
    pub digest: ContentDigest,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryRoot {
    pub logical_prefix: String,
    pub files: Vec<String>,
    pub explicitly_required: Vec<String>,
    pub class: InputClass,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathMapping {
    pub physical: String,
    pub logical: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageIdentity {
    pub name: String,
    pub version: String,
    pub source: String,
    pub checksum: Option<ContentDigest>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitKind {
    Library,
    ProcMacro,
    BuildScript,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetObservation {
    pub triple: String,
    pub llvm_triple: String,
    pub data_layout: String,
    pub cpu: String,
    pub features: Vec<String>,
    pub specification: Option<ContentDigest>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileConfiguration {
    pub opt_level: String,
    pub debuginfo: String,
    pub debug_assertions: bool,
    pub panic: String,
    pub rustflags: Vec<String>,
    pub cfg: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolObservation {
    pub name: String,
    pub identity: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerObservation {
    pub verbose_version: String,
    pub tools: Vec<ToolObservation>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentObservation {
    pub name: String,
    pub value: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyBinding {
    pub name: String,
    pub unit: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileUnit {
    pub package: PackageIdentity,
    pub target_name: String,
    pub kind: UnitKind,
    pub target: TargetObservation,
    pub features: Vec<String>,
    pub configuration: CompileConfiguration,
    pub compiler: CompilerObservation,
    pub inputs: Vec<String>,
    pub dependencies: Vec<DependencyBinding>,
    pub build_script: Option<usize>,
    pub build_environment: Vec<EnvironmentObservation>,
    pub compiler_environment: Vec<EnvironmentObservation>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileClass {
    Debug,
    Release,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRecipe {
    pub schema_version: u32,
    pub runtime_unit: usize,
    pub units: Vec<CompileUnit>,
    pub required_inputs: Vec<RequiredInput>,
    pub cargo_profile: ProfileClass,
    pub public_abi: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub schema_version: u16,
    pub recipe: ContentDigest,
    pub source: ContentDigest,
    pub interface: ContentDigest,
    pub target: ContentDigest,
    pub features: ContentDigest,
    pub compile: ContentDigest,
    pub toolchain: ContentDigest,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    Runtime,
    Cli,
    Python,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum BuildProvenance {
    SourceWorktree {
        source_root: String,
        recipe: RuntimeRecipe,
        roots: Vec<PathMapping>,
    },
    SealedDistribution {
        source_closure: ContentDigest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputError {
    #[error("invalid lowercase SHA-256 digest")]
    InvalidDigest,
    #[error("invalid logical path: {path}")]
    InvalidPath { path: String },
    #[error("duplicate input: {path}")]
    DuplicateInput { path: String },
    #[error("required input is missing: {path}")]
    MissingInput { path: String },
    #[error("captured input was not declared: {path}")]
    UnexpectedInput { path: String },
    #[error("unsupported {field}: {value}")]
    Unsupported { field: String, value: String },
    #[error("missing or invalid observation: {field}")]
    InvalidObservation { field: String },
    #[error("missing compilation unit {unit}")]
    MissingUnit { unit: usize },
    #[error("cycle through compilation unit {unit}")]
    Cycle { unit: usize },
    #[error("invalid path mapping: {physical}")]
    InvalidMapping { physical: String },
    #[error("invalid provenance JSON: {message}")]
    Serialization { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecordError {
    #[error("identity record is missing")]
    Missing,
    #[error("identity record occurs more than once")]
    Duplicate,
    #[error("malformed identity record: {reason}")]
    Malformed { reason: String },
    #[error("truncated identity record: {reason}")]
    Truncated { reason: String },
    #[error("unsupported identity record: {reason}")]
    Unsupported { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityDimension {
    Schema,
    Recipe,
    Source,
    Interface,
    Target,
    Features,
    Compile,
    Toolchain,
}
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("runtime identity mismatch: {dimensions:?}")]
pub struct IdentityMismatch {
    pub dimensions: Vec<IdentityDimension>,
}
