use super::{
    ArtifactPlatform, ArtifactSpec, ChelisSrcSpec, ConformSpec, DependencySpec, LockSource,
    LockedDependency, ManifestPackage, PackageId, ReefLock, ReefManifest,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use toml_edit::DocumentMut;

const MANIFEST_SCHEMA_CURRENT: u32 = 1;
const LOCK_SCHEMA_CURRENT: u32 = 1;
pub(crate) const PROJECT_WRITE_LOCK_FILE: &str = ".reef-write.lock";
const PROJECT_WRITE_IGNORE_RULES: &[&str] =
    &[PROJECT_WRITE_LOCK_FILE, ".*.reef-tmp-*", ".*.reef-backup-*"];

static WARNED_LEGACY_MANIFESTS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
static WARNED_LEGACY_LOCKS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const MANIFEST_ROOT_KEYS: &[&str] = &[
    "schema",
    "package",
    "dependencies",
    "chelis-src",
    "conform",
    "artifacts",
];
const MANIFEST_PACKAGE_KEYS: &[&str] = &[
    "name",
    "version",
    "compiler",
    "module_prefix",
    "additional_sources",
];
const DEPENDENCY_KEYS: &[&str] = &["version", "path"];
const CHELIS_SRC_KEYS: &[&str] = &["crates", "pin_commit"];
const CONFORM_KEYS: &[&str] = &["local_skills"];
const ARTIFACT_KEYS: &[&str] = &["repo", "tag", "platforms"];
const ARTIFACT_PLATFORM_KEYS: &[&str] = &["asset", "sha256"];
const LOCK_ROOT_KEYS: &[&str] = &["schema", "package", "dependencies"];
const LOCK_PACKAGE_KEYS: &[&str] = &["name", "version"];
const LOCK_DEPENDENCY_KEYS: &[&str] = &[
    "name",
    "version",
    "source",
    "compiler",
    "archive_sha256",
    "shell_sha256",
];
const PATH_SOURCE_KEYS: &[&str] = &["kind", "path"];
const REGISTRY_SOURCE_KEYS: &[&str] = &["kind", "remote_origin"];
const BUNDLED_SOURCE_KEYS: &[&str] = &["kind", "compiler_version"];
const BINARY_SOURCE_KEYS: &[&str] = &["kind", "remote_origin", "platform", "asset", "sha256"];
const UNKNOWN_SOURCE_KEYS: &[&str] = &["kind"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ManifestSchemaVersion(u32);

impl ManifestSchemaVersion {
    pub const CURRENT: Self = Self(MANIFEST_SCHEMA_CURRENT);

    pub fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ManifestSchemaVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for ManifestSchemaVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_explicit_schema(value, "manifest").map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LockSchemaVersion(u32);

impl LockSchemaVersion {
    pub const CURRENT: Self = Self(LOCK_SCHEMA_CURRENT);

    pub fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for LockSchemaVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for LockSchemaVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_explicit_schema(value, "lock").map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpgradeMode {
    Check,
    InPlace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeReport {
    pub steps: Vec<String>,
}

#[derive(Debug)]
pub enum DocumentUpgradeError {
    Io {
        document: &'static str,
        path: PathBuf,
        operation: &'static str,
        message: String,
    },
    Schema {
        document: &'static str,
        path: PathBuf,
        schema: Option<u32>,
        message: String,
    },
    Migration {
        document: &'static str,
        path: PathBuf,
        from: u32,
        to: u32,
        operation: &'static str,
        message: String,
    },
    ProjectLock {
        path: PathBuf,
        operation: &'static str,
        message: String,
    },
}

impl fmt::Display for DocumentUpgradeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                document,
                path,
                operation,
                message,
            } => write!(
                formatter,
                "{document} `{}`: {operation}: {message}",
                path.display()
            ),
            Self::Schema {
                document,
                path,
                schema,
                message,
            } => {
                if let Some(schema) = schema {
                    write!(
                        formatter,
                        "{document} `{}` schema {schema}: {message}",
                        path.display()
                    )
                } else {
                    write!(formatter, "{document} `{}`: {message}", path.display())
                }
            }
            Self::Migration {
                document,
                path,
                from,
                to,
                operation,
                message,
            } => write!(
                formatter,
                "{document} `{}` schema {from} -> {to}: {operation}: {message}",
                path.display()
            ),
            Self::ProjectLock {
                path,
                operation,
                message,
            } => write!(
                formatter,
                "project write lock `{}`: {operation}: {message}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for DocumentUpgradeError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
enum SchemaOne {
    #[serde(rename = "1")]
    One,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ManifestWireV1 {
    schema: SchemaOne,
    package: ManifestPackageWireV1,
    #[serde(default)]
    dependencies: BTreeMap<String, DependencySpecWireV1>,
    #[serde(default, rename = "chelis-src")]
    chelis_src: Option<ChelisSrcWireV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    conform: Option<ConformWireV1>,
    #[serde(default)]
    artifacts: BTreeMap<String, ArtifactSpecWireV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ManifestPackageWireV1 {
    name: String,
    version: String,
    compiler: String,
    module_prefix: String,
    #[serde(default)]
    additional_sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DependencySpecWireV1 {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ChelisSrcWireV1 {
    #[serde(default)]
    crates: Vec<String>,
    #[serde(default)]
    pin_commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ConformWireV1 {
    #[serde(default)]
    local_skills: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ArtifactSpecWireV1 {
    repo: String,
    tag: String,
    platforms: BTreeMap<String, ArtifactPlatformWireV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ArtifactPlatformWireV1 {
    asset: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct LockWireV1 {
    schema: SchemaOne,
    package: PackageIdWireV1,
    dependencies: Vec<LockedDependencyWireV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PackageIdWireV1 {
    name: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct LockedDependencyWireV1 {
    name: String,
    version: String,
    source: LockSourceWireV1,
    compiler: String,
    archive_sha256: String,
    shell_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum LockSourceWireV1 {
    Path {
        path: String,
    },
    LocalRegistry {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remote_origin: Option<String>,
    },
    Bundled {
        compiler_version: String,
    },
    Binary {
        remote_origin: String,
        platform: String,
        asset: String,
        sha256: String,
    },
}

impl From<ManifestWireV1> for ReefManifest {
    fn from(wire: ManifestWireV1) -> Self {
        Self {
            package: ManifestPackage {
                name: wire.package.name,
                version: wire.package.version,
                compiler: wire.package.compiler,
                module_prefix: wire.package.module_prefix,
                additional_sources: wire.package.additional_sources,
            },
            dependencies: wire
                .dependencies
                .into_iter()
                .map(|(name, dependency)| {
                    (
                        name,
                        DependencySpec {
                            version: dependency.version,
                            path: dependency.path,
                        },
                    )
                })
                .collect(),
            chelis_src: wire.chelis_src.map(|source| ChelisSrcSpec {
                crates: source.crates,
                pin_commit: source.pin_commit,
            }),
            conform: wire.conform.map(|conform| ConformSpec {
                local_skills: conform.local_skills,
            }),
            artifacts: wire
                .artifacts
                .into_iter()
                .map(|(name, artifact)| {
                    (
                        name,
                        ArtifactSpec {
                            repo: artifact.repo,
                            tag: artifact.tag,
                            platforms: artifact
                                .platforms
                                .into_iter()
                                .map(|(platform, entry)| {
                                    (
                                        platform,
                                        ArtifactPlatform {
                                            asset: entry.asset,
                                            sha256: entry.sha256,
                                        },
                                    )
                                })
                                .collect(),
                        },
                    )
                })
                .collect(),
        }
    }
}

impl From<&ReefManifest> for ManifestWireV1 {
    fn from(manifest: &ReefManifest) -> Self {
        Self {
            schema: SchemaOne::One,
            package: ManifestPackageWireV1 {
                name: manifest.package.name.clone(),
                version: manifest.package.version.clone(),
                compiler: manifest.package.compiler.clone(),
                module_prefix: manifest.package.module_prefix.clone(),
                additional_sources: manifest.package.additional_sources.clone(),
            },
            dependencies: manifest
                .dependencies
                .iter()
                .map(|(name, dependency)| {
                    (
                        name.clone(),
                        DependencySpecWireV1 {
                            version: dependency.version.clone(),
                            path: dependency.path.clone(),
                        },
                    )
                })
                .collect(),
            chelis_src: manifest.chelis_src.as_ref().map(|source| ChelisSrcWireV1 {
                crates: source.crates.clone(),
                pin_commit: source.pin_commit.clone(),
            }),
            conform: manifest.conform.as_ref().map(|conform| ConformWireV1 {
                local_skills: conform.local_skills.clone(),
            }),
            artifacts: manifest
                .artifacts
                .iter()
                .map(|(name, artifact)| {
                    (
                        name.clone(),
                        ArtifactSpecWireV1 {
                            repo: artifact.repo.clone(),
                            tag: artifact.tag.clone(),
                            platforms: artifact
                                .platforms
                                .iter()
                                .map(|(platform, entry)| {
                                    (
                                        platform.clone(),
                                        ArtifactPlatformWireV1 {
                                            asset: entry.asset.clone(),
                                            sha256: entry.sha256.clone(),
                                        },
                                    )
                                })
                                .collect(),
                        },
                    )
                })
                .collect(),
        }
    }
}

impl From<LockWireV1> for ReefLock {
    fn from(wire: LockWireV1) -> Self {
        Self {
            package: PackageId {
                name: wire.package.name,
                version: wire.package.version,
            },
            dependencies: wire
                .dependencies
                .into_iter()
                .map(|dependency| LockedDependency {
                    name: dependency.name,
                    version: dependency.version,
                    source: dependency.source.into(),
                    compiler: dependency.compiler,
                    archive_sha256: dependency.archive_sha256,
                    shell_sha256: dependency.shell_sha256,
                })
                .collect(),
        }
    }
}

impl From<LockSourceWireV1> for LockSource {
    fn from(source: LockSourceWireV1) -> Self {
        match source {
            LockSourceWireV1::Path { path } => Self::Path { path },
            LockSourceWireV1::LocalRegistry { remote_origin } => {
                Self::LocalRegistry { remote_origin }
            }
            LockSourceWireV1::Bundled { compiler_version } => Self::Bundled { compiler_version },
            LockSourceWireV1::Binary {
                remote_origin,
                platform,
                asset,
                sha256,
            } => Self::Binary {
                remote_origin,
                platform,
                asset,
                sha256,
            },
        }
    }
}

impl From<&ReefLock> for LockWireV1 {
    fn from(lock: &ReefLock) -> Self {
        Self {
            schema: SchemaOne::One,
            package: PackageIdWireV1 {
                name: lock.package.name.clone(),
                version: lock.package.version.clone(),
            },
            dependencies: lock
                .dependencies
                .iter()
                .map(|dependency| LockedDependencyWireV1 {
                    name: dependency.name.clone(),
                    version: dependency.version.clone(),
                    source: LockSourceWireV1::from(&dependency.source),
                    compiler: dependency.compiler.clone(),
                    archive_sha256: dependency.archive_sha256.clone(),
                    shell_sha256: dependency.shell_sha256.clone(),
                })
                .collect(),
        }
    }
}

impl From<&LockSource> for LockSourceWireV1 {
    fn from(source: &LockSource) -> Self {
        match source {
            LockSource::Path { path } => Self::Path { path: path.clone() },
            LockSource::LocalRegistry { remote_origin } => Self::LocalRegistry {
                remote_origin: remote_origin.clone(),
            },
            LockSource::Bundled { compiler_version } => Self::Bundled {
                compiler_version: compiler_version.clone(),
            },
            LockSource::Binary {
                remote_origin,
                platform,
                asset,
                sha256,
            } => Self::Binary {
                remote_origin: remote_origin.clone(),
                platform: platform.clone(),
                asset: asset.clone(),
                sha256: sha256.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DocumentKind {
    Manifest,
    Lock,
}

impl DocumentKind {
    fn name(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Lock => "lock",
        }
    }

    fn current(self) -> u32 {
        match self {
            Self::Manifest => MANIFEST_SCHEMA_CURRENT,
            Self::Lock => LOCK_SCHEMA_CURRENT,
        }
    }
}

type MigrationResult = (Option<String>, Vec<(u32, u32)>);

struct MigrationStep {
    from: u32,
    to: u32,
    apply: fn(&str, &Path) -> Result<String, DocumentUpgradeError>,
}

const MANIFEST_MIGRATIONS: &[MigrationStep] = &[MigrationStep {
    from: 0,
    to: 1,
    apply: migrate_manifest_0_to_1,
}];

const LOCK_MIGRATIONS: &[MigrationStep] = &[MigrationStep {
    from: 0,
    to: 1,
    apply: migrate_lock_0_to_1,
}];

fn parse_explicit_schema(value: &str, document: &str) -> Result<u32, String> {
    if value.is_empty()
        || value == "0"
        || value.starts_with('0')
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!(
            "{document} schema must be a positive decimal ASCII integer without signs, whitespace, or leading zeroes"
        ));
    }
    value
        .parse::<u32>()
        .map_err(|_| format!("{document} schema `{value}` is too large"))
}

fn parse_schema_header(
    text: &str,
    path: &Path,
    kind: DocumentKind,
) -> Result<u32, DocumentUpgradeError> {
    let value =
        toml::from_str::<toml::Value>(text).map_err(|error| DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: None,
            message: format!("failed to parse TOML schema header: {error}"),
        })?;
    let table = value
        .as_table()
        .ok_or_else(|| DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: None,
            message: "document root must be a TOML table".to_string(),
        })?;
    let Some(raw_schema) = table.get("schema") else {
        return Ok(0);
    };
    let Some(raw_schema) = raw_schema.as_str() else {
        return Err(DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: None,
            message: "schema must use the string form, such as `schema = \"1\"`".to_string(),
        });
    };
    let schema = parse_explicit_schema(raw_schema, kind.name()).map_err(|message| {
        DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: None,
            message: format!("invalid schema `{raw_schema}`: {message}"),
        }
    })?;
    if schema > kind.current() {
        return Err(DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: Some(schema),
            message: format!(
                "unsupported {} schema {schema}; newest supported schema is {}. Run `chelis reef upgrade` with a compatible toolchain",
                kind.name(),
                kind.current()
            ),
        });
    }
    Ok(schema)
}

fn reject_unknown_keys(
    table: &toml::map::Map<String, toml::Value>,
    allowed: &[&str],
    table_name: &str,
) -> Result<(), String> {
    if let Some(key) = table.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unknown field `{key}` in {table_name} table"));
    }
    Ok(())
}

fn strict_manifest_key_preflight(text: &str) -> Result<(), String> {
    let value = toml::from_str::<toml::Value>(text).map_err(|error| error.to_string())?;
    let root = value
        .as_table()
        .ok_or_else(|| "document root must be a TOML table".to_string())?;
    reject_unknown_keys(root, MANIFEST_ROOT_KEYS, "manifest root")?;
    if let Some(package) = root.get("package").and_then(toml::Value::as_table) {
        reject_unknown_keys(package, MANIFEST_PACKAGE_KEYS, "[package]")?;
    }
    if let Some(dependencies) = root.get("dependencies").and_then(toml::Value::as_table) {
        for (name, dependency) in dependencies {
            if let Some(dependency) = dependency.as_table() {
                reject_unknown_keys(
                    dependency,
                    DEPENDENCY_KEYS,
                    &format!("[dependencies.{name}]"),
                )?;
            }
        }
    }
    if let Some(source) = root.get("chelis-src").and_then(toml::Value::as_table) {
        reject_unknown_keys(source, CHELIS_SRC_KEYS, "[chelis-src]")?;
    }
    if let Some(conform) = root.get("conform").and_then(toml::Value::as_table) {
        reject_unknown_keys(conform, CONFORM_KEYS, "[conform]")?;
    }
    if let Some(artifacts) = root.get("artifacts").and_then(toml::Value::as_table) {
        for (name, artifact) in artifacts {
            let Some(artifact) = artifact.as_table() else {
                continue;
            };
            reject_unknown_keys(artifact, ARTIFACT_KEYS, &format!("[artifacts.{name}]"))?;
            if let Some(platforms) = artifact.get("platforms").and_then(toml::Value::as_table) {
                for (platform, entry) in platforms {
                    if let Some(entry) = entry.as_table() {
                        reject_unknown_keys(
                            entry,
                            ARTIFACT_PLATFORM_KEYS,
                            &format!("[artifacts.{name}.platforms.{platform}]"),
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn strict_lock_key_preflight(text: &str) -> Result<(), String> {
    let value = toml::from_str::<toml::Value>(text).map_err(|error| error.to_string())?;
    let root = value
        .as_table()
        .ok_or_else(|| "document root must be a TOML table".to_string())?;
    reject_unknown_keys(root, LOCK_ROOT_KEYS, "lock root")?;
    if let Some(package) = root.get("package").and_then(toml::Value::as_table) {
        reject_unknown_keys(package, LOCK_PACKAGE_KEYS, "[package]")?;
    }
    if let Some(dependencies) = root.get("dependencies").and_then(toml::Value::as_array) {
        for (index, dependency) in dependencies.iter().enumerate() {
            let Some(dependency) = dependency.as_table() else {
                continue;
            };
            reject_unknown_keys(
                dependency,
                LOCK_DEPENDENCY_KEYS,
                &format!("[[dependencies]] entry {index}"),
            )?;
            let Some(source) = dependency.get("source").and_then(toml::Value::as_table) else {
                continue;
            };
            let allowed = match source.get("kind").and_then(toml::Value::as_str) {
                Some("path") => PATH_SOURCE_KEYS,
                Some("local_registry") => REGISTRY_SOURCE_KEYS,
                Some("bundled") => BUNDLED_SOURCE_KEYS,
                Some("binary") => BINARY_SOURCE_KEYS,
                Some(kind) => return Err(format!("unsupported source kind `{kind}`")),
                None => UNKNOWN_SOURCE_KEYS,
            };
            reject_unknown_keys(
                source,
                allowed,
                &format!("[[dependencies]] entry {index} source"),
            )?;
        }
    }
    Ok(())
}

fn first_warning_for_path(warned: &'static OnceLock<Mutex<HashSet<PathBuf>>>, path: &Path) -> bool {
    let identity = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    warned
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut paths| paths.insert(identity))
        .unwrap_or(true)
}

pub(crate) fn parse_manifest_text(
    text: &str,
    path: &Path,
    warn_for_legacy: bool,
) -> Result<ReefManifest, DocumentUpgradeError> {
    match parse_schema_header(text, path, DocumentKind::Manifest)? {
        0 => {
            if warn_for_legacy && first_warning_for_path(&WARNED_LEGACY_MANIFESTS, path) {
                eprintln!(
                    "chelis reef: warning: legacy manifest `{}` uses schema 0; run `chelis reef upgrade --inplace --path {}`",
                    path.display(),
                    path.parent().unwrap_or_else(|| Path::new(".")).display()
                );
            }
            toml::from_str(text).map_err(|error| DocumentUpgradeError::Schema {
                document: "manifest",
                path: path.to_path_buf(),
                schema: Some(0),
                message: format!("failed to parse legacy document: {error}"),
            })
        }
        1 => {
            strict_manifest_key_preflight(text).map_err(|message| {
                DocumentUpgradeError::Schema {
                    document: "manifest",
                    path: path.to_path_buf(),
                    schema: Some(1),
                    message,
                }
            })?;
            toml::from_str::<ManifestWireV1>(text)
                .map(ReefManifest::from)
                .map_err(|error| DocumentUpgradeError::Schema {
                    document: "manifest",
                    path: path.to_path_buf(),
                    schema: Some(1),
                    message: format!("failed strict field parsing: {error}"),
                })
        }
        schema => Err(DocumentUpgradeError::Schema {
            document: "manifest",
            path: path.to_path_buf(),
            schema: Some(schema),
            message: "no parser is registered for this schema".to_string(),
        }),
    }
}

pub(crate) fn parse_lock_text(
    text: &str,
    path: &Path,
    warn_for_legacy: bool,
) -> Result<ReefLock, DocumentUpgradeError> {
    match parse_schema_header(text, path, DocumentKind::Lock)? {
        0 => {
            if warn_for_legacy && first_warning_for_path(&WARNED_LEGACY_LOCKS, path) {
                eprintln!(
                    "chelis reef: warning: legacy lock `{}` uses schema 0; run `chelis reef upgrade --inplace --path {}`",
                    path.display(),
                    path.parent().unwrap_or_else(|| Path::new(".")).display()
                );
            }
            toml::from_str(text).map_err(|error| DocumentUpgradeError::Schema {
                document: "lock",
                path: path.to_path_buf(),
                schema: Some(0),
                message: format!("failed to parse legacy document: {error}"),
            })
        }
        1 => {
            strict_lock_key_preflight(text).map_err(|message| DocumentUpgradeError::Schema {
                document: "lock",
                path: path.to_path_buf(),
                schema: Some(1),
                message,
            })?;
            toml::from_str::<LockWireV1>(text)
                .map(ReefLock::from)
                .map_err(|error| DocumentUpgradeError::Schema {
                    document: "lock",
                    path: path.to_path_buf(),
                    schema: Some(1),
                    message: format!("failed strict field parsing: {error}"),
                })
        }
        schema => Err(DocumentUpgradeError::Schema {
            document: "lock",
            path: path.to_path_buf(),
            schema: Some(schema),
            message: "no parser is registered for this schema".to_string(),
        }),
    }
}

pub(crate) fn serialize_manifest_v1(manifest: &ReefManifest) -> Result<Vec<u8>, String> {
    let text = toml::to_string_pretty(&ManifestWireV1::from(manifest))
        .map_err(|error| format!("serialize manifest schema 1: {error}"))?;
    Ok(format!("{text}\n").into_bytes())
}

pub(crate) fn serialize_lock_v1(lock: &ReefLock) -> Result<Vec<u8>, String> {
    let text = toml::to_string_pretty(&LockWireV1::from(lock))
        .map_err(|error| format!("serialize lock schema 1: {error}"))?;
    Ok(format!("{text}\n").into_bytes())
}

fn migrate_schema_0_to_1(
    text: &str,
    path: &Path,
    kind: DocumentKind,
) -> Result<String, DocumentUpgradeError> {
    let document =
        text.parse::<DocumentMut>()
            .map_err(|error| DocumentUpgradeError::Migration {
                document: kind.name(),
                path: path.to_path_buf(),
                from: 0,
                to: 1,
                operation: "parse editable TOML",
                message: error.to_string(),
            })?;
    let legacy_text = document.to_string();
    let migrated = if legacy_text
        .lines()
        .next()
        .is_some_and(|line| line.trim_start().starts_with("#:schema"))
    {
        let split = legacy_text
            .find('\n')
            .map_or(legacy_text.len(), |index| index + 1);
        format!(
            "{}schema = \"1\"\n{}",
            &legacy_text[..split],
            &legacy_text[split..]
        )
    } else {
        format!("schema = \"1\"\n{legacy_text}")
    };
    match kind {
        DocumentKind::Manifest => {
            let manifest = parse_manifest_text(&migrated, path, false)?;
            super::validate_manifest_with(&manifest, true).map_err(|message| {
                DocumentUpgradeError::Migration {
                    document: "manifest",
                    path: path.to_path_buf(),
                    from: 0,
                    to: 1,
                    operation: "validate migrated manifest",
                    message,
                }
            })?;
        }
        DocumentKind::Lock => {
            parse_lock_text(&migrated, path, false)?;
        }
    }
    Ok(migrated)
}

fn migrate_manifest_0_to_1(text: &str, path: &Path) -> Result<String, DocumentUpgradeError> {
    migrate_schema_0_to_1(text, path, DocumentKind::Manifest)
}

fn migrate_lock_0_to_1(text: &str, path: &Path) -> Result<String, DocumentUpgradeError> {
    migrate_schema_0_to_1(text, path, DocumentKind::Lock)
}

fn apply_registered_migrations(
    text: &str,
    path: &Path,
    kind: DocumentKind,
    current: u32,
    target: u32,
    registry: &[MigrationStep],
) -> Result<MigrationResult, DocumentUpgradeError> {
    if current == target {
        match kind {
            DocumentKind::Manifest => {
                let manifest = parse_manifest_text(text, path, false)?;
                super::validate_manifest_with(&manifest, true).map_err(|message| {
                    DocumentUpgradeError::Migration {
                        document: "manifest",
                        path: path.to_path_buf(),
                        from: current,
                        to: target,
                        operation: "validate current manifest",
                        message,
                    }
                })?;
            }
            DocumentKind::Lock => {
                parse_lock_text(text, path, false)?;
            }
        }
        return Ok((None, Vec::new()));
    }

    let mut schema = current;
    let mut migrated = text.to_string();
    let mut steps = Vec::new();
    while schema < target {
        let step = registry
            .iter()
            .find(|step| step.from == schema && step.to <= target)
            .ok_or_else(|| DocumentUpgradeError::Migration {
                document: kind.name(),
                path: path.to_path_buf(),
                from: schema,
                to: target,
                operation: "select registered migration",
                message: "no ordered migration step is registered".to_string(),
            })?;
        if step.to <= schema {
            return Err(DocumentUpgradeError::Migration {
                document: kind.name(),
                path: path.to_path_buf(),
                from: schema,
                to: step.to,
                operation: "select registered migration",
                message: "migration registry is not strictly increasing".to_string(),
            });
        }
        migrated = (step.apply)(&migrated, path)?;
        steps.push((step.from, step.to));
        schema = step.to;
    }
    if schema != target {
        return Err(DocumentUpgradeError::Migration {
            document: kind.name(),
            path: path.to_path_buf(),
            from: schema,
            to: target,
            operation: "complete registered migration",
            message: "migration registry did not reach the selected target".to_string(),
        });
    }
    Ok((Some(migrated), steps))
}

fn selected_target(
    requested: Option<u32>,
    current: u32,
    kind: DocumentKind,
    path: &Path,
) -> Result<u32, DocumentUpgradeError> {
    let target = requested.unwrap_or_else(|| kind.current());
    if target > kind.current() {
        return Err(DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: Some(target),
            message: format!(
                "unsupported {} schema {target}; newest supported schema is {}",
                kind.name(),
                kind.current()
            ),
        });
    }
    if target < current {
        return Err(DocumentUpgradeError::Schema {
            document: kind.name(),
            path: path.to_path_buf(),
            schema: Some(target),
            message: format!("target schema {target} is less than current schema {current}"),
        });
    }
    Ok(target)
}

type PlannedReplacement<'a> = (&'static str, &'a Path, &'a str, u32, u32);

fn preflight_replacement_target(path: &Path) -> Result<(), String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("inspect target {}: {error}", path.display())),
    };
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "refuse to replace symbolic-link target {}",
            path.display()
        ));
    }
    if !metadata.file_type().is_file() {
        return Err(format!(
            "replacement target {} is not a regular file",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() > 1 {
            return Err(format!(
                "refuse to replace multiply-hard-linked target {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn apply_preflighted_replacements(
    manifest: Option<PlannedReplacement<'_>>,
    lock: Option<PlannedReplacement<'_>>,
    mut replace: impl FnMut(&Path, &[u8]) -> Result<(), String>,
) -> Result<(), DocumentUpgradeError> {
    for (document, path, _, from, to) in manifest.as_ref().into_iter().chain(lock.as_ref()) {
        preflight_replacement_target(path).map_err(|message| DocumentUpgradeError::Migration {
            document,
            path: path.to_path_buf(),
            from: *from,
            to: *to,
            operation: "preflight replacement target",
            message,
        })?;
    }
    for (document, path, text, from, to) in manifest.into_iter().chain(lock) {
        replace(path, text.as_bytes()).map_err(|message| DocumentUpgradeError::Migration {
            document,
            path: path.to_path_buf(),
            from,
            to,
            operation: "atomic replacement",
            message,
        })?;
    }
    Ok(())
}

pub fn upgrade_documents(
    root: &Path,
    mode: UpgradeMode,
    manifest_to: Option<ManifestSchemaVersion>,
    lock_to: Option<LockSchemaVersion>,
) -> Result<UpgradeReport, DocumentUpgradeError> {
    let root = root
        .canonicalize()
        .map_err(|error| DocumentUpgradeError::Io {
            document: "package",
            path: root.to_path_buf(),
            operation: "canonicalize package root",
            message: error.to_string(),
        })?;
    let _project_lock = if mode == UpgradeMode::InPlace {
        Some(acquire_project_write_lock(&root)?)
    } else {
        None
    };
    let manifest_path = root.join("reef.toml");
    let lock_path = root.join("reef.lock");
    let manifest_text =
        fs::read_to_string(&manifest_path).map_err(|error| DocumentUpgradeError::Io {
            document: "manifest",
            path: manifest_path.clone(),
            operation: "read",
            message: error.to_string(),
        })?;
    let manifest_schema =
        parse_schema_header(&manifest_text, &manifest_path, DocumentKind::Manifest)?;
    let manifest_target = selected_target(
        manifest_to.map(ManifestSchemaVersion::get),
        manifest_schema,
        DocumentKind::Manifest,
        &manifest_path,
    )?;

    let lock_text = if lock_path.exists() {
        Some(
            fs::read_to_string(&lock_path).map_err(|error| DocumentUpgradeError::Io {
                document: "lock",
                path: lock_path.clone(),
                operation: "read",
                message: error.to_string(),
            })?,
        )
    } else {
        None
    };
    if lock_text.is_none() && lock_to.is_some() {
        return Err(DocumentUpgradeError::Io {
            document: "lock",
            path: lock_path,
            operation: "select target",
            message: "cannot select a lock target because reef.lock does not exist".to_string(),
        });
    }
    let (lock_schema, lock_target) = if let Some(text) = lock_text.as_ref() {
        let schema = parse_schema_header(text, &lock_path, DocumentKind::Lock)?;
        let target = selected_target(
            lock_to.map(LockSchemaVersion::get),
            schema,
            DocumentKind::Lock,
            &lock_path,
        )?;
        (Some(schema), Some(target))
    } else {
        (None, None)
    };

    let (manifest_result, manifest_steps) = apply_registered_migrations(
        &manifest_text,
        &manifest_path,
        DocumentKind::Manifest,
        manifest_schema,
        manifest_target,
        MANIFEST_MIGRATIONS,
    )?;
    let (lock_result, lock_steps) = match (lock_text.as_ref(), lock_schema, lock_target) {
        (Some(text), Some(schema), Some(target)) => apply_registered_migrations(
            text,
            &lock_path,
            DocumentKind::Lock,
            schema,
            target,
            LOCK_MIGRATIONS,
        )?,
        _ => (None, Vec::new()),
    };

    let mut steps = manifest_steps
        .into_iter()
        .map(|(from, to)| format!("reef.toml: schema {from} -> {to}"))
        .collect::<Vec<_>>();
    steps.extend(
        lock_steps
            .into_iter()
            .map(|(from, to)| format!("reef.lock: schema {from} -> {to}")),
    );

    if mode == UpgradeMode::InPlace && (!steps.is_empty()) {
        let manifest_replacement = manifest_result.as_deref().map(|text| {
            (
                "manifest",
                manifest_path.as_path(),
                text,
                manifest_schema,
                manifest_target,
            )
        });
        let lock_replacement = lock_result.as_deref().map(|text| {
            (
                "lock",
                lock_path.as_path(),
                text,
                lock_schema.unwrap_or(0),
                lock_target.unwrap_or(LOCK_SCHEMA_CURRENT),
            )
        });
        for (document, path, _, from, to) in manifest_replacement
            .as_ref()
            .into_iter()
            .chain(lock_replacement.as_ref())
        {
            preflight_replacement_target(path).map_err(|message| {
                DocumentUpgradeError::Migration {
                    document,
                    path: path.to_path_buf(),
                    from: *from,
                    to: *to,
                    operation: "preflight replacement target",
                    message,
                }
            })?;
        }
        ensure_project_lock_ignore(&root).map_err(|message| DocumentUpgradeError::Io {
            document: "project",
            path: root.join(".gitignore"),
            operation: "write project ignore rules",
            message,
        })?;
        apply_preflighted_replacements(manifest_replacement, lock_replacement, atomic_replace)?;
    }

    Ok(UpgradeReport { steps })
}

#[must_use = "the project lock is released when the guard drops"]
pub(crate) struct ProjectWriteLock {
    _file: fs::File,
}

pub(crate) fn acquire_project_write_lock(
    root: &Path,
) -> Result<ProjectWriteLock, DocumentUpgradeError> {
    use rustix::fs::{FlockOperation, flock};
    use std::os::fd::AsFd;

    let path = root.join(PROJECT_WRITE_LOCK_FILE);
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|error| DocumentUpgradeError::ProjectLock {
            path: path.clone(),
            operation: "open",
            message: error.to_string(),
        })?;
    flock(file.as_fd(), FlockOperation::LockExclusive).map_err(|error| {
        DocumentUpgradeError::ProjectLock {
            path,
            operation: "acquire exclusive lock",
            message: error.to_string(),
        }
    })?;
    Ok(ProjectWriteLock { _file: file })
}

fn has_project_write_ignore_rules(text: &str) -> bool {
    PROJECT_WRITE_IGNORE_RULES
        .iter()
        .all(|rule| text.lines().any(|line| line == *rule))
}

pub(crate) fn ensure_project_lock_ignore(root: &Path) -> Result<(), String> {
    for ancestor in root.ancestors().skip(1) {
        let path = ancestor.join(".gitignore");
        if let Ok(text) = fs::read_to_string(path)
            && has_project_write_ignore_rules(&text)
        {
            return Ok(());
        }
    }

    let path = root.join(".gitignore");
    let mut text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    if has_project_write_ignore_rules(&text) {
        return Ok(());
    }
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    for rule in PROJECT_WRITE_IGNORE_RULES {
        if !text.lines().any(|line| line == *rule) {
            text.push_str(rule);
            text.push('\n');
        }
    }
    atomic_replace(&path, text.as_bytes())
}

pub(crate) fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), String> {
    atomic_replace_with(path, bytes, |_| Ok(()), sync_parent)
}

fn atomic_replace_with(
    path: &Path,
    bytes: &[u8],
    before_rename: impl FnOnce(&Path) -> Result<(), String>,
    sync_directory: impl FnMut(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    atomic_replace_with_sequence(path, bytes, sequence, before_rename, sync_directory)
}

fn atomic_replace_with_sequence(
    path: &Path,
    bytes: &[u8],
    sequence: u64,
    before_rename: impl FnOnce(&Path) -> Result<(), String>,
    mut sync_directory: impl FnMut(&Path) -> Result<(), String>,
) -> Result<(), String> {
    preflight_replacement_target(path)?;
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no UTF-8 file name", path.display()))?;
    let temporary = parent.join(format!(
        ".{name}.reef-tmp-{}-{sequence}",
        std::process::id()
    ));
    let backup = parent.join(format!(
        ".{name}.reef-backup-{}-{sequence}",
        std::process::id()
    ));

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("create temporary sibling {}: {error}", temporary.display()))?;
    let prepare_result = file
        .write_all(bytes)
        .map_err(|error| format!("write temporary sibling {}: {error}", temporary.display()))
        .and_then(|()| {
            file.sync_all().map_err(|error| {
                format!("flush temporary sibling {}: {error}", temporary.display())
            })
        });
    drop(file);
    if let Err(error) = prepare_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }

    let had_target = path.exists();
    if had_target && let Err(error) = fs::hard_link(path, &backup) {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "preserve prior target {} at {}: {error}",
            path.display(),
            backup.display()
        ));
    }

    if let Err(error) = before_rename(&temporary) {
        let _ = fs::remove_file(&temporary);
        let _ = fs::remove_file(&backup);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        let _ = fs::remove_file(&backup);
        return Err(format!(
            "rename temporary sibling {} over {}: {error}",
            temporary.display(),
            path.display()
        ));
    }

    if let Err(sync_error) = sync_directory(parent) {
        let restore_result = if had_target {
            fs::rename(&backup, path)
        } else {
            fs::remove_file(path)
        };
        if let Err(restore_error) = restore_result {
            return Err(format!(
                "{sync_error}; restoration failed: {restore_error}; prior bytes remain at {}",
                backup.display()
            ));
        }
        let _ = sync_directory(parent);
        return Err(sync_error);
    }

    if had_target {
        let _ = fs::remove_file(&backup);
    }
    Ok(())
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> Result<(), String> {
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync parent directory {}: {error}", parent.display()))
}

#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> Result<(), String> {
    Ok(())
}

pub fn manifest_schema_v1_json() -> String {
    let schema = schema_for!(ManifestWireV1);
    format!(
        "{}\n",
        serde_json::to_string_pretty(&schema).expect("manifest schema serialization")
    )
}

pub fn lock_schema_v1_json() -> String {
    let schema = schema_for!(LockWireV1);
    format!(
        "{}\n",
        serde_json::to_string_pretty(&schema).expect("lock schema serialization")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use tempfile::tempdir;

    #[test]
    fn schema_versions_reject_noncanonical_values() {
        for value in ["", "0", "01", "+1", " 1", "1 ", "-1", "١"] {
            assert!(value.parse::<ManifestSchemaVersion>().is_err(), "{value}");
            assert!(value.parse::<LockSchemaVersion>().is_err(), "{value}");
        }
        assert_eq!("1".parse::<ManifestSchemaVersion>().unwrap().get(), 1);
    }

    fn json_schema_property_keys(value: &serde_json::Value) -> BTreeSet<String> {
        value["properties"]
            .as_object()
            .expect("schema object properties")
            .keys()
            .cloned()
            .collect()
    }

    fn expected_keys(keys: &[&str]) -> BTreeSet<String> {
        keys.iter().map(|key| (*key).to_string()).collect()
    }

    #[test]
    fn strict_key_preflight_lists_match_wire_schema_properties() {
        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_schema_v1_json()).expect("manifest schema JSON");
        assert_eq!(
            json_schema_property_keys(&manifest),
            expected_keys(MANIFEST_ROOT_KEYS)
        );
        for (definition, keys) in [
            ("ManifestPackageWireV1", MANIFEST_PACKAGE_KEYS),
            ("DependencySpecWireV1", DEPENDENCY_KEYS),
            ("ChelisSrcWireV1", CHELIS_SRC_KEYS),
            ("ConformWireV1", CONFORM_KEYS),
            ("ArtifactSpecWireV1", ARTIFACT_KEYS),
            ("ArtifactPlatformWireV1", ARTIFACT_PLATFORM_KEYS),
        ] {
            assert_eq!(
                json_schema_property_keys(&manifest["definitions"][definition]),
                expected_keys(keys),
                "manifest schema key drift in {definition}"
            );
        }

        let lock: serde_json::Value =
            serde_json::from_str(&lock_schema_v1_json()).expect("lock schema JSON");
        assert_eq!(
            json_schema_property_keys(&lock),
            expected_keys(LOCK_ROOT_KEYS)
        );
        for (definition, keys) in [
            ("PackageIdWireV1", LOCK_PACKAGE_KEYS),
            ("LockedDependencyWireV1", LOCK_DEPENDENCY_KEYS),
        ] {
            assert_eq!(
                json_schema_property_keys(&lock["definitions"][definition]),
                expected_keys(keys),
                "lock schema key drift in {definition}"
            );
        }
        let source_variants = lock["definitions"]["LockSourceWireV1"]["oneOf"]
            .as_array()
            .expect("lock source variants");
        let expected_sources = BTreeMap::from([
            ("path", PATH_SOURCE_KEYS),
            ("local_registry", REGISTRY_SOURCE_KEYS),
            ("bundled", BUNDLED_SOURCE_KEYS),
            ("binary", BINARY_SOURCE_KEYS),
        ]);
        for variant in source_variants {
            let kind = variant["properties"]["kind"]["enum"][0]
                .as_str()
                .expect("source kind");
            assert_eq!(
                json_schema_property_keys(variant),
                expected_keys(expected_sources[kind]),
                "lock source schema key drift for {kind}"
            );
        }
    }

    #[test]
    fn missing_registered_migration_fails_closed() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("reef.toml");

        let error = apply_registered_migrations(
            "schema = \"1\"\n",
            &path,
            DocumentKind::Manifest,
            1,
            2,
            &[],
        )
        .expect_err("missing migration must fail");

        assert!(error.to_string().contains("no ordered migration step"));
    }

    #[test]
    fn atomic_replace_ignores_unowned_siblings() {
        let directory = tempdir().expect("tempdir");
        let target = directory.path().join("reef.lock");
        let unowned = directory.path().join(".reef.lock.reef-tmp-unowned");
        fs::write(&target, b"old").unwrap();
        fs::write(&unowned, b"other").unwrap();

        atomic_replace(&target, b"new").unwrap();

        assert_eq!(fs::read(target).unwrap(), b"new");
        assert_eq!(fs::read(unowned).unwrap(), b"other");
    }

    #[test]
    fn colliding_temporary_sibling_is_not_removed() {
        let directory = tempdir().expect("tempdir");
        let target = directory.path().join("reef.lock");
        let stale = directory
            .path()
            .join(format!(".reef.lock.reef-tmp-{}-42", std::process::id()));
        fs::write(&target, b"old").unwrap();
        fs::write(&stale, b"another command").unwrap();

        let result = atomic_replace_with_sequence(&target, b"new", 42, |_| Ok(()), sync_parent);

        assert!(result.is_err());
        assert_eq!(fs::read(target).unwrap(), b"old");
        assert_eq!(fs::read(stale).unwrap(), b"another command");
    }

    #[test]
    fn colliding_backup_sibling_is_not_overwritten_or_removed() {
        let directory = tempdir().expect("tempdir");
        let target = directory.path().join("reef.lock");
        let stale = directory
            .path()
            .join(format!(".reef.lock.reef-backup-{}-42", std::process::id()));
        fs::write(&target, b"old").unwrap();
        fs::write(&stale, b"another command").unwrap();

        let result = atomic_replace_with_sequence(&target, b"new", 42, |_| Ok(()), sync_parent);

        assert!(result.is_err());
        assert_eq!(fs::read(target).unwrap(), b"old");
        assert_eq!(fs::read(stale).unwrap(), b"another command");
    }

    #[test]
    fn failed_rename_preserves_the_prior_target() {
        let directory = tempdir().expect("tempdir");
        let target = directory.path().join("reef.lock");
        fs::write(&target, b"old").unwrap();

        let result = atomic_replace_with(
            &target,
            b"new",
            |temporary| fs::remove_file(temporary).map_err(|error| error.to_string()),
            sync_parent,
        );

        assert!(result.is_err());
        assert_eq!(fs::read(target).unwrap(), b"old");
    }

    #[test]
    fn failed_parent_sync_restores_the_prior_target() {
        let directory = tempdir().expect("tempdir");
        let target = directory.path().join("reef.lock");
        fs::write(&target, b"old").unwrap();
        let mut sync_count = 0;

        let result = atomic_replace_with(
            &target,
            b"new",
            |_| Ok(()),
            |_| {
                sync_count += 1;
                if sync_count == 1 {
                    Err("injected parent sync failure".to_string())
                } else {
                    Ok(())
                }
            },
        );

        assert!(result.is_err());
        assert_eq!(fs::read(target).unwrap(), b"old");
    }

    #[test]
    fn failed_lock_replacement_keeps_the_prior_readable_lock() {
        let directory = tempdir().expect("tempdir");
        let root = directory.path();
        let manifest_path = root.join("reef.toml");
        let lock_path = root.join("reef.lock");
        let old_manifest = "[package]\nname = \"demo\"\n";
        let old_lock = "dependencies = []\n";
        let new_manifest = "schema = \"1\"\n[package]\nname = \"demo\"\n";
        let new_lock = "schema = \"1\"\ndependencies = []\n";
        fs::write(&manifest_path, old_manifest).unwrap();
        fs::write(&lock_path, old_lock).unwrap();

        let _project_lock = acquire_project_write_lock(root).expect("project lock");
        let result = apply_preflighted_replacements(
            Some(("manifest", &manifest_path, new_manifest, 0, 1)),
            Some(("lock", &lock_path, new_lock, 0, 1)),
            |path, bytes| {
                if path == lock_path {
                    Err("injected lock replacement failure".to_string())
                } else {
                    atomic_replace(path, bytes)
                }
            },
        );

        assert!(result.is_err());
        assert_eq!(fs::read_to_string(manifest_path).unwrap(), new_manifest);
        assert_eq!(fs::read_to_string(lock_path).unwrap(), old_lock);
    }

    #[test]
    fn concurrent_upgrade_waits_for_the_project_lock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let directory = tempdir().expect("tempdir");
        let root = directory.path().to_path_buf();
        let manifest_path = root.join("reef.toml");
        let lock_path = root.join("reef.lock");
        fs::write(
            &manifest_path,
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.4\"\nmodule_prefix = \"Demo\"\n",
        )
        .unwrap();
        fs::write(
            &lock_path,
            "dependencies = []\n[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        let first = acquire_project_write_lock(&root).expect("first lock");
        let worker_root = root.clone();
        let (sender, receiver) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let result = upgrade_documents(&worker_root, UpgradeMode::InPlace, None, None);
            sender.send(result).expect("send upgrade result");
        });

        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        assert!(
            !fs::read_to_string(&manifest_path)
                .unwrap()
                .contains("schema")
        );
        assert!(!fs::read_to_string(&lock_path).unwrap().contains("schema"));
        drop(first);
        receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("upgrade completes after release")
            .expect("upgrade succeeds");
        writer.join().expect("writer thread");
        assert!(
            fs::read_to_string(manifest_path)
                .unwrap()
                .starts_with("schema = \"1\"")
        );
        assert!(
            fs::read_to_string(lock_path)
                .unwrap()
                .starts_with("schema = \"1\"")
        );
    }

    #[test]
    fn project_write_lock_serializes_concurrent_writers() {
        use std::sync::mpsc;
        use std::time::Duration;

        let directory = tempdir().expect("tempdir");
        let first = acquire_project_write_lock(directory.path()).expect("first lock");
        let root = directory.path().to_path_buf();
        let (sender, receiver) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let _second = acquire_project_write_lock(&root).expect("second lock");
            sender.send(()).expect("signal acquisition");
        });

        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        drop(first);
        receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("second writer acquires after release");
        writer.join().expect("writer thread");
    }
}
