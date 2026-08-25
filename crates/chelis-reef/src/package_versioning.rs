//! Typed package identities and the pure, bounded Reef local resolver.
//!
//! This module performs no file or network access. Callers parse provider data
//! into these types before resolution, then materialize only the selected graph.

use semver::{BuildMetadata, Version, VersionReq};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

pub const MAX_CANDIDATES_PER_PACKAGE: u64 = 256;
pub const MAX_RESOLVED_PACKAGE_NAMES: u64 = 256;
pub const MAX_DEPENDENCIES_PER_MANIFEST: u64 = 256;
pub const MAX_DEPENDENCY_DEPTH: u64 = 128;
pub const MAX_RESOLVER_STATES: u64 = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersioningError {
    InvalidPackageName {
        value: String,
        reason: String,
    },
    InvalidPackageVersion {
        value: String,
        reason: String,
    },
    InvalidRequirement {
        value: String,
        reason: String,
    },
    InvalidCompilerVersion {
        value: String,
        reason: String,
    },
    UnsupportedResolver {
        value: String,
    },
    InvalidDependency {
        reason: String,
    },
    PathPackageMismatch {
        dependency: PackageName,
        actual: ResolvedPackageId,
        requirement: Option<String>,
    },
    PathSourceConflict {
        package: PackageName,
        paths: Vec<String>,
    },
    SourceConflict {
        package: PackageName,
        version: PackageVersion,
        sources: Vec<String>,
    },
    LimitExceeded {
        dimension: &'static str,
        package: Option<PackageName>,
        limit: u64,
        attempted: u64,
    },
    IncompatibleRequirements {
        package: PackageName,
        requirements: Vec<RequestedPackage>,
        candidates: Vec<PackageVersion>,
    },
    InvalidLock {
        reason: String,
    },
    UnavailableLockedOrigin {
        package: PackageName,
        version: PackageVersion,
        origin: Option<String>,
    },
}

impl fmt::Display for VersioningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPackageName { value, reason } => {
                write!(formatter, "invalid package name `{value}`: {reason}")
            }
            Self::InvalidPackageVersion { value, reason } => {
                write!(formatter, "invalid package version `{value}`: {reason}")
            }
            Self::InvalidRequirement { value, reason } => {
                write!(formatter, "invalid package requirement `{value}`: {reason}")
            }
            Self::InvalidCompilerVersion { value, reason } => {
                write!(
                    formatter,
                    "invalid exact compiler version `{value}`: {reason}"
                )
            }
            Self::UnsupportedResolver { value } => {
                write!(formatter, "unsupported Reef resolver `{value}`")
            }
            Self::InvalidDependency { reason } => write!(formatter, "invalid dependency: {reason}"),
            Self::PathPackageMismatch {
                dependency,
                actual,
                requirement,
            } => write!(
                formatter,
                "path dependency `{dependency}` resolved to `{actual}` but required {}",
                requirement.as_deref().unwrap_or("that package name")
            ),
            Self::PathSourceConflict { package, paths } => write!(
                formatter,
                "path-source conflict for `{package}`: {}",
                paths.join(", ")
            ),
            Self::SourceConflict {
                package,
                version,
                sources,
            } => write!(
                formatter,
                "source conflict for `{package}` `{version}`: {}",
                sources.join(", ")
            ),
            Self::LimitExceeded {
                dimension,
                package,
                limit,
                attempted,
            } => {
                write!(
                    formatter,
                    "Reef local resolver {dimension} limit {limit} reached before operation {attempted}"
                )?;
                if let Some(package) = package {
                    write!(formatter, " for `{package}`")?;
                }
                Ok(())
            }
            Self::IncompatibleRequirements {
                package,
                requirements,
                candidates,
            } => {
                write!(formatter, "no local version of `{package}` satisfies")?;
                for item in requirements {
                    write!(
                        formatter,
                        " `{}` from `{}`",
                        item.requirement, item.requester
                    )?;
                }
                if !candidates.is_empty() {
                    write!(
                        formatter,
                        "; considered {}",
                        candidates
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                }
                Ok(())
            }
            Self::InvalidLock { reason } => write!(formatter, "invalid Reef lock: {reason}"),
            Self::UnavailableLockedOrigin {
                package,
                version,
                origin,
            } => write!(
                formatter,
                "locked origin for `{package}` is unavailable{}. Restore the exact origin with \
                 `chelis reef install --from-github chelis-lang/{package}@v{version}` or regenerate \
                 the lock from verified local bytes",
                origin
                    .as_deref()
                    .map(|origin| format!(" (`{origin}`)"))
                    .unwrap_or_default()
            ),
        }
    }
}

impl std::error::Error for VersioningError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PackageName(String);

impl PackageName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for PackageName {
    type Err = VersioningError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = value.as_bytes();
        let invalid = |reason: &str| VersioningError::InvalidPackageName {
            value: value.to_string(),
            reason: reason.to_string(),
        };
        if bytes.is_empty() || bytes.len() > 64 {
            return Err(invalid("length must be 1 through 64 ASCII bytes"));
        }
        if !bytes[0].is_ascii_lowercase() {
            return Err(invalid("the first byte must be an ASCII lowercase letter"));
        }
        if !bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
        {
            return Err(invalid(
                "bytes must be ASCII lowercase letters, digits, or hyphens",
            ));
        }
        for (index, byte) in bytes.iter().enumerate() {
            if *byte == b'-'
                && (index == 0
                    || index + 1 == bytes.len()
                    || !bytes[index - 1].is_ascii_alphanumeric()
                    || !bytes[index + 1].is_ascii_alphanumeric())
            {
                return Err(invalid("each hyphen must occur between alphanumeric bytes"));
            }
        }
        let reserved = matches!(value, "con" | "prn" | "aux" | "nul")
            || value
                .strip_prefix("com")
                .or_else(|| value.strip_prefix("lpt"))
                .is_some_and(|suffix| {
                    suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9')
                });
        if reserved {
            return Err(invalid("the name is reserved on a supported platform"));
        }
        Ok(Self(value.to_string()))
    }
}

impl<'de> Deserialize<'de> for PackageName {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_str(&value).map_err(DeserializerType::Error::custom)
    }
}

impl Borrow<str> for PackageName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for PackageName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for PackageName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PackageVersion(Version);

impl PackageVersion {
    pub fn as_semver(&self) -> &Version {
        &self.0
    }
}

impl FromStr for PackageVersion {
    type Err = VersioningError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let version =
            Version::parse(value).map_err(|error| VersioningError::InvalidPackageVersion {
                value: value.to_string(),
                reason: error.to_string(),
            })?;
        if version.build != BuildMetadata::EMPTY {
            return Err(VersioningError::InvalidPackageVersion {
                value: value.to_string(),
                reason: "build metadata is not part of a selectable package identity".to_string(),
            });
        }
        Ok(Self(version))
    }
}

impl<'de> Deserialize<'de> for PackageVersion {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_str(&value).map_err(DeserializerType::Error::custom)
    }
}

impl fmt::Display for PackageVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolverVersion {
    One,
    Two,
}

impl FromStr for ResolverVersion {
    type Err = VersioningError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            _ => Err(VersioningError::UnsupportedResolver {
                value: value.to_string(),
            }),
        }
    }
}

impl fmt::Display for ResolverVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::One => "1",
            Self::Two => "2",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageRequirement {
    text: String,
    requirement: VersionReq,
}

impl PackageRequirement {
    pub fn exact(version: &PackageVersion) -> Self {
        let text = format!("={version}");
        Self {
            requirement: VersionReq::parse(&text).expect("exact parsed package version"),
            text,
        }
    }

    pub fn matches(&self, version: &PackageVersion) -> bool {
        self.requirement.matches(version.as_semver())
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl FromStr for PackageRequirement {
    type Err = VersioningError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.contains('+') {
            return Err(VersioningError::InvalidRequirement {
                value: value.to_string(),
                reason: "build metadata is not selectable by package requirements".to_string(),
            });
        }
        let requirement =
            VersionReq::parse(value).map_err(|error| VersioningError::InvalidRequirement {
                value: value.to_string(),
                reason: error.to_string(),
            })?;
        Ok(Self {
            text: value.to_string(),
            requirement,
        })
    }
}

impl Serialize for PackageRequirement {
    fn serialize<SerializerType>(
        &self,
        serializer: SerializerType,
    ) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_str(&self.text)
    }
}

impl<'de> Deserialize<'de> for PackageRequirement {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_str(&value).map_err(DeserializerType::Error::custom)
    }
}

impl fmt::Display for PackageRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyRequirement {
    Exact(PackageVersion),
    Compatible(PackageRequirement),
}

impl DependencyRequirement {
    pub fn parse(resolver: ResolverVersion, value: &str) -> Result<Self, VersioningError> {
        match resolver {
            ResolverVersion::One => PackageVersion::from_str(value).map(Self::Exact),
            ResolverVersion::Two => PackageRequirement::from_str(value).map(Self::Compatible),
        }
    }

    pub fn matches(&self, version: &PackageVersion) -> bool {
        match self {
            Self::Exact(expected) => expected == version,
            Self::Compatible(requirement) => requirement.matches(version),
        }
    }

    pub fn as_text(&self) -> String {
        match self {
            Self::Exact(version) => version.to_string(),
            Self::Compatible(requirement) => requirement.to_string(),
        }
    }

    pub fn as_package_requirement(&self) -> PackageRequirement {
        match self {
            Self::Exact(version) => PackageRequirement::exact(version),
            Self::Compatible(requirement) => requirement.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExactCompilerVersion(PackageVersion);

impl ExactCompilerVersion {
    pub fn package_version(&self) -> &PackageVersion {
        &self.0
    }
}

impl FromStr for ExactCompilerVersion {
    type Err = VersioningError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let raw =
            value
                .strip_prefix('=')
                .ok_or_else(|| VersioningError::InvalidCompilerVersion {
                    value: value.to_string(),
                    reason: "the value must start with one exact `=` operator".to_string(),
                })?;
        if raw.starts_with('=') {
            return Err(VersioningError::InvalidCompilerVersion {
                value: value.to_string(),
                reason: "the value must contain one exact comparator".to_string(),
            });
        }
        let version = PackageVersion::from_str(raw).map_err(|error| {
            VersioningError::InvalidCompilerVersion {
                value: value.to_string(),
                reason: error.to_string(),
            }
        })?;
        if !version.as_semver().pre.is_empty() {
            return Err(VersioningError::InvalidCompilerVersion {
                value: value.to_string(),
                reason: "compiler pins must select stable releases".to_string(),
            });
        }
        Ok(Self(version))
    }
}

impl Serialize for ExactCompilerVersion {
    fn serialize<SerializerType>(
        &self,
        serializer: SerializerType,
    ) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ExactCompilerVersion {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_str(&value).map_err(DeserializerType::Error::custom)
    }
}

impl fmt::Display for ExactCompilerVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "={}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResolvedPackageId {
    pub name: PackageName,
    pub version: PackageVersion,
}

impl ResolvedPackageId {
    pub fn new(name: PackageName, version: PackageVersion) -> Self {
        Self { name, version }
    }
}

impl fmt::Display for ResolvedPackageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} {}", self.name, self.version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypedDependency {
    Registry {
        requirement: DependencyRequirement,
    },
    Path {
        path: String,
        requirement: Option<DependencyRequirement>,
    },
}

impl TypedDependency {
    pub fn registry(resolver: ResolverVersion, value: &str) -> Result<Self, VersioningError> {
        Ok(Self::Registry {
            requirement: DependencyRequirement::parse(resolver, value)?,
        })
    }

    pub fn inline(
        resolver: ResolverVersion,
        version: Option<&str>,
        path: Option<&str>,
    ) -> Result<Self, VersioningError> {
        let requirement = version
            .map(|value| DependencyRequirement::parse(resolver, value))
            .transpose()?;
        match (requirement, path) {
            (Some(requirement), None) => Ok(Self::Registry { requirement }),
            (requirement, Some(path)) if !path.is_empty() => Ok(Self::Path {
                path: path.to_string(),
                requirement,
            }),
            _ => Err(VersioningError::InvalidDependency {
                reason: "a dependency must contain a version, a path, or both".to_string(),
            }),
        }
    }

    pub fn requirement(&self) -> Option<&DependencyRequirement> {
        match self {
            Self::Registry { requirement } => Some(requirement),
            Self::Path { requirement, .. } => requirement.as_ref(),
        }
    }

    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Registry { .. } => None,
            Self::Path { path, .. } => Some(path),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedManifest {
    pub package: ResolvedPackageId,
    pub compiler: ExactCompilerVersion,
    pub dependencies: BTreeMap<PackageName, TypedDependency>,
    pub resolver: ResolverVersion,
}

impl TypedManifest {
    pub fn schema_one(
        package_name: &str,
        package_version: &str,
        compiler: &str,
        dependencies: impl IntoIterator<Item = (String, Option<String>, Option<String>)>,
    ) -> Result<Self, VersioningError> {
        let dependencies = dependencies.into_iter().collect::<Vec<_>>();
        if dependencies.len() as u64 > MAX_DEPENDENCIES_PER_MANIFEST {
            return Err(limit_error(
                "dependencies per manifest",
                PackageName::from_str(package_name).ok(),
                MAX_DEPENDENCIES_PER_MANIFEST,
                dependencies.len() as u64,
            ));
        }
        let mut parsed_dependencies = BTreeMap::new();
        for (raw_name, version, path) in dependencies {
            let dependency_name = PackageName::from_str(&raw_name)?;
            let dependency =
                TypedDependency::inline(ResolverVersion::One, version.as_deref(), path.as_deref())?;
            parsed_dependencies.insert(dependency_name, dependency);
        }
        Ok(Self {
            package: ResolvedPackageId::new(
                PackageName::from_str(package_name)?,
                PackageVersion::from_str(package_version)?,
            ),
            compiler: ExactCompilerVersion::from_str(compiler)?,
            dependencies: parsed_dependencies,
            resolver: ResolverVersion::One,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CandidateSource {
    Locked {
        source_identity: String,
    },
    LocalRegistry {
        source_identity: String,
    },
    BundledRuntime {
        compiler_version: ExactCompilerVersion,
    },
    Path {
        canonical_path: String,
    },
}

impl CandidateSource {
    fn source_identity(&self) -> String {
        match self {
            Self::Locked { source_identity } | Self::LocalRegistry { source_identity } => {
                source_identity.clone()
            }
            Self::BundledRuntime { compiler_version } => {
                format!("bundled://{compiler_version}")
            }
            Self::Path { canonical_path } => format!("path://{canonical_path}"),
        }
    }

    fn is_locked(&self) -> bool {
        matches!(self, Self::Locked { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestedPackage {
    pub requester: PackageName,
    pub package: PackageName,
    pub requirement: PackageRequirement,
    #[serde(default = "root_depth")]
    depth: u64,
}

const fn root_depth() -> u64 {
    1
}

impl RequestedPackage {
    pub fn new(
        requester: PackageName,
        package: PackageName,
        requirement: PackageRequirement,
    ) -> Self {
        Self {
            requester,
            package,
            requirement,
            depth: root_depth(),
        }
    }

    fn at_depth(mut self, depth: u64) -> Self {
        self.depth = depth;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalCandidate {
    pub id: ResolvedPackageId,
    pub source: CandidateSource,
    pub content_identity: String,
    pub dependencies: Vec<RequestedPackage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolverLimits {
    pub candidates_per_name: u64,
    pub resolved_names: u64,
    pub dependencies_per_manifest: u64,
    pub depth: u64,
    pub states: u64,
}

impl Default for ResolverLimits {
    fn default() -> Self {
        Self {
            candidates_per_name: MAX_CANDIDATES_PER_PACKAGE,
            resolved_names: MAX_RESOLVED_PACKAGE_NAMES,
            dependencies_per_manifest: MAX_DEPENDENCIES_PER_MANIFEST,
            depth: MAX_DEPENDENCY_DEPTH,
            states: MAX_RESOLVER_STATES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalResolution {
    pub selected: BTreeMap<PackageName, LocalCandidate>,
    pub explored_states: u64,
    pub memoized_failures: u64,
    pub memo_hits: u64,
}

impl LocalResolution {
    pub fn canonical_lock_order(&self) -> Vec<ResolvedPackageId> {
        self.selected
            .values()
            .map(|candidate| candidate.id.clone())
            .collect()
    }
}

pub fn resolve_local(
    root_requirements: &[RequestedPackage],
    candidates: &BTreeMap<PackageName, Vec<LocalCandidate>>,
    limits: ResolverLimits,
) -> Result<LocalResolution, VersioningError> {
    let mut requirements = BTreeMap::<PackageName, Vec<RequestedPackage>>::new();
    for item in root_requirements {
        if item.depth > limits.depth {
            return Err(limit_error(
                "dependency depth",
                Some(item.package.clone()),
                limits.depth,
                item.depth,
            ));
        }
        requirements
            .entry(item.package.clone())
            .or_default()
            .push(item.clone());
    }
    sort_requirements(&mut requirements);
    let mut selected = BTreeMap::new();
    let mut failed_states = BTreeSet::new();
    let mut explored_states = 0_u64;
    let mut memo_hits = 0_u64;
    search(
        candidates,
        limits,
        &mut requirements,
        &mut selected,
        &mut failed_states,
        &mut explored_states,
        &mut memo_hits,
    )?;
    Ok(LocalResolution {
        selected,
        explored_states,
        memoized_failures: failed_states.len() as u64,
        memo_hits,
    })
}

fn search(
    providers: &BTreeMap<PackageName, Vec<LocalCandidate>>,
    limits: ResolverLimits,
    requirements: &mut BTreeMap<PackageName, Vec<RequestedPackage>>,
    selected: &mut BTreeMap<PackageName, LocalCandidate>,
    failed_states: &mut BTreeSet<String>,
    explored_states: &mut u64,
    memo_hits: &mut u64,
) -> Result<(), VersioningError> {
    for (package, active) in requirements.iter() {
        if let Some(candidate) = selected.get(package)
            && !active
                .iter()
                .all(|item| item.requirement.matches(&candidate.id.version))
        {
            return Err(incompatible_error(package, active, providers.get(package)));
        }
    }
    let Some(package) = requirements
        .keys()
        .find(|name| !selected.contains_key(*name))
        .cloned()
    else {
        return Ok(());
    };

    let state_key = failed_requirement_key(&package, &requirements[&package]);
    if failed_states.contains(&state_key) {
        *memo_hits = memo_hits.saturating_add(1);
        return Err(incompatible_error(
            &package,
            &requirements[&package],
            providers.get(&package),
        ));
    }

    let attempted_names = selected.len().checked_add(1).ok_or_else(|| {
        limit_error(
            "resolved package names",
            None,
            limits.resolved_names,
            u64::MAX,
        )
    })? as u64;
    if attempted_names > limits.resolved_names {
        return Err(limit_error(
            "resolved package names",
            Some(package.clone()),
            limits.resolved_names,
            attempted_names,
        ));
    }

    let raw_candidates = providers.get(&package).cloned().unwrap_or_default();
    if raw_candidates.len() as u64 > limits.candidates_per_name {
        return Err(limit_error(
            "candidates per package name",
            Some(package.clone()),
            limits.candidates_per_name,
            raw_candidates.len() as u64,
        ));
    }
    let active = requirements[&package].clone();
    let candidates = ordered_candidates(&package, raw_candidates, &active)?;
    let mut last_error = None;
    let mut matched = false;

    for candidate in candidates {
        *explored_states = explored_states.checked_add(1).ok_or_else(|| {
            limit_error(
                "explored resolver states",
                Some(package.clone()),
                limits.states,
                u64::MAX,
            )
        })?;
        if *explored_states > limits.states {
            return Err(limit_error(
                "explored resolver states",
                Some(package.clone()),
                limits.states,
                *explored_states,
            ));
        }
        if !active
            .iter()
            .all(|item| item.requirement.matches(&candidate.id.version))
        {
            continue;
        }
        matched = true;
        if candidate.dependencies.len() as u64 > limits.dependencies_per_manifest {
            return Err(limit_error(
                "dependencies per manifest",
                Some(package.clone()),
                limits.dependencies_per_manifest,
                candidate.dependencies.len() as u64,
            ));
        }
        let depth = active.iter().map(|item| item.depth).max().unwrap_or(1);
        let child_depth = depth.checked_add(1).ok_or_else(|| {
            limit_error(
                "dependency depth",
                Some(package.clone()),
                limits.depth,
                u64::MAX,
            )
        })?;
        if !candidate.dependencies.is_empty() && child_depth > limits.depth {
            let child = candidate.dependencies[0].package.clone();
            return Err(limit_error(
                "dependency depth",
                Some(child),
                limits.depth,
                child_depth,
            ));
        }

        let old_requirements = requirements.clone();
        selected.insert(package.clone(), candidate.clone());
        for dependency in &candidate.dependencies {
            requirements
                .entry(dependency.package.clone())
                .or_default()
                .push(dependency.clone().at_depth(child_depth));
        }
        sort_requirements(requirements);
        match search(
            providers,
            limits,
            requirements,
            selected,
            failed_states,
            explored_states,
            memo_hits,
        ) {
            Ok(()) => return Ok(()),
            Err(error @ VersioningError::LimitExceeded { .. }) => return Err(error),
            Err(error @ VersioningError::PathSourceConflict { .. }) => return Err(error),
            Err(error @ VersioningError::SourceConflict { .. }) => return Err(error),
            Err(error) => last_error = Some(error),
        }
        selected.remove(&package);
        *requirements = old_requirements;
    }

    if matched {
        Err(last_error
            .unwrap_or_else(|| incompatible_error(&package, &active, providers.get(&package))))
    } else {
        failed_states.insert(state_key);
        Err(incompatible_error(
            &package,
            &active,
            providers.get(&package),
        ))
    }
}

fn sort_requirements(requirements: &mut BTreeMap<PackageName, Vec<RequestedPackage>>) {
    for entries in requirements.values_mut() {
        entries.sort_by(|left, right| {
            left.requester
                .cmp(&right.requester)
                .then_with(|| left.requirement.as_str().cmp(right.requirement.as_str()))
                .then_with(|| left.depth.cmp(&right.depth))
        });
        entries.dedup();
    }
}

fn failed_requirement_key(package: &PackageName, active: &[RequestedPackage]) -> String {
    let mut key = format!("R:{package}=");
    for item in active {
        key.push_str(&format!("{}@{},", item.requester, item.requirement));
    }
    key
}

fn ordered_candidates(
    package: &PackageName,
    candidates: Vec<LocalCandidate>,
    active: &[RequestedPackage],
) -> Result<Vec<LocalCandidate>, VersioningError> {
    let has_path_source = candidates
        .iter()
        .any(|candidate| matches!(candidate.source, CandidateSource::Path { .. }));
    let matching = candidates
        .into_iter()
        .filter(|candidate| {
            active
                .iter()
                .all(|item| item.requirement.matches(&candidate.id.version))
        })
        .collect::<Vec<_>>();
    let mut paths = matching
        .iter()
        .filter_map(|candidate| match &candidate.source {
            CandidateSource::Path { canonical_path } => Some(canonical_path.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    if paths.len() > 1 {
        return Err(VersioningError::PathSourceConflict {
            package: package.clone(),
            paths,
        });
    }

    let mut selected = if has_path_source {
        matching
            .into_iter()
            .filter(|candidate| matches!(candidate.source, CandidateSource::Path { .. }))
            .collect()
    } else {
        reject_conflicting_non_path_sources(package, matching)?
    };
    selected.sort_by(|left, right| {
        right
            .source
            .is_locked()
            .cmp(&left.source.is_locked())
            .then_with(|| right.id.version.cmp(&left.id.version))
            .then_with(|| {
                left.source
                    .source_identity()
                    .cmp(&right.source.source_identity())
            })
            .then_with(|| left.content_identity.cmp(&right.content_identity))
            .then_with(|| left.source.cmp(&right.source))
            .then_with(|| {
                dependency_fingerprint(&left.dependencies)
                    .cmp(&dependency_fingerprint(&right.dependencies))
            })
    });
    selected.dedup();
    Ok(selected)
}

fn reject_conflicting_non_path_sources(
    package: &PackageName,
    candidates: Vec<LocalCandidate>,
) -> Result<Vec<LocalCandidate>, VersioningError> {
    let mut by_version = BTreeMap::<PackageVersion, Vec<&LocalCandidate>>::new();
    for candidate in &candidates {
        by_version
            .entry(candidate.id.version.clone())
            .or_default()
            .push(candidate);
    }
    for (version, same_version) in by_version {
        let Some(first) = same_version.first() else {
            continue;
        };
        if same_version.iter().any(|candidate| {
            candidate.source.source_identity() != first.source.source_identity()
                || candidate.content_identity != first.content_identity
                || dependency_fingerprint(&candidate.dependencies)
                    != dependency_fingerprint(&first.dependencies)
        }) {
            return Err(VersioningError::SourceConflict {
                package: package.clone(),
                version,
                sources: same_version
                    .iter()
                    .map(|candidate| {
                        format!(
                            "{}#{}",
                            candidate.source.source_identity(),
                            candidate.content_identity
                        )
                    })
                    .collect(),
            });
        }
    }
    Ok(candidates)
}

fn dependency_fingerprint(dependencies: &[RequestedPackage]) -> String {
    let mut entries = dependencies
        .iter()
        .map(|dependency| {
            format!(
                "{}>{}@{}#{}",
                dependency.requester, dependency.package, dependency.requirement, dependency.depth
            )
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries.join(";")
}

fn incompatible_error(
    package: &PackageName,
    requirements: &[RequestedPackage],
    candidates: Option<&Vec<LocalCandidate>>,
) -> VersioningError {
    let mut versions = candidates
        .into_iter()
        .flatten()
        .map(|candidate| candidate.id.version.clone())
        .collect::<Vec<_>>();
    versions.sort_by(|left, right| right.cmp(left));
    versions.dedup();
    VersioningError::IncompatibleRequirements {
        package: package.clone(),
        requirements: requirements.to_vec(),
        candidates: versions,
    }
}

fn limit_error(
    dimension: &'static str,
    package: Option<PackageName>,
    limit: u64,
    attempted: u64,
) -> VersioningError {
    VersioningError::LimitExceeded {
        dimension,
        package,
        limit,
        attempted,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockDependency {
    pub id: ResolvedPackageId,
    pub source: CandidateSource,
    pub archive_sha256: String,
    pub shell_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockSnapshot {
    pub root: ResolvedPackageId,
    pub dependencies: Vec<LockDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockAssessment {
    Reusable,
    Stale { reasons: Vec<String> },
}

pub fn assess_lock(
    root: &ResolvedPackageId,
    declarations: &BTreeMap<PackageName, TypedDependency>,
    lock: &LockSnapshot,
) -> Result<LockAssessment, VersioningError> {
    let mut reasons = Vec::new();
    if &lock.root != root {
        reasons.push(format!(
            "lock root `{}` does not match manifest root `{root}`",
            lock.root
        ));
    }
    let mut locked = BTreeMap::new();
    for dependency in &lock.dependencies {
        if locked
            .insert(dependency.id.name.clone(), dependency)
            .is_some()
        {
            return Err(VersioningError::InvalidLock {
                reason: format!(
                    "duplicate package `{}` in lock dependencies",
                    dependency.id.name
                ),
            });
        }
        let missing_hash = match dependency.source {
            CandidateSource::Path { .. } => false,
            _ => dependency.archive_sha256.is_empty() || dependency.shell_sha256.is_empty(),
        };
        if missing_hash {
            return Err(VersioningError::InvalidLock {
                reason: format!(
                    "locked package `{}` lacks required content hashes",
                    dependency.id.name
                ),
            });
        }
    }

    for (package, declaration) in declarations {
        let Some(dependency) = locked.get(package) else {
            reasons.push(format!("lock lacks direct dependency `{package}`"));
            continue;
        };
        if let Some(requirement) = declaration.requirement()
            && !requirement.matches(&dependency.id.version)
        {
            reasons.push(format!(
                "locked `{package}` version `{}` does not satisfy `{}`",
                dependency.id.version,
                requirement.as_text()
            ));
        }
        match (declaration, &dependency.source) {
            (TypedDependency::Registry { .. }, CandidateSource::Path { .. }) => reasons.push(
                format!("locked source for `{package}` is path but the manifest uses registry"),
            ),
            (
                TypedDependency::Path { path, .. },
                CandidateSource::Path { canonical_path },
            ) if path != canonical_path => reasons.push(format!(
                "locked path for `{package}` is `{canonical_path}` but the manifest declares `{path}`"
            )),
            (TypedDependency::Path { .. }, CandidateSource::Path { .. }) => {}
            (TypedDependency::Path { .. }, _) => reasons.push(format!(
                "locked source for `{package}` is not the declared path source"
            )),
            _ => {}
        }
    }

    if reasons.is_empty() {
        Ok(LockAssessment::Reusable)
    } else {
        Ok(LockAssessment::Stale { reasons })
    }
}

pub fn sort_registry_versions(
    versions: &mut [crate::RegistryVersion],
) -> Result<(), VersioningError> {
    let mut parsed = versions
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            PackageVersion::from_str(&entry.version).map(|version| (index, version))
        })
        .collect::<Result<Vec<_>, _>>()?;
    parsed.sort_by(|left, right| left.1.cmp(&right.1));
    let reordered = parsed
        .into_iter()
        .map(|(index, _)| versions[index].clone())
        .collect::<Vec<_>>();
    versions.clone_from_slice(&reordered);
    Ok(())
}
