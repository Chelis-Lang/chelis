//! Bounded remote candidate discovery for Reef resolver 2.

use crate::package_versioning::{
    self, CandidateSource, LocalCandidate, PackageName, PackageRequirement, PackageVersion,
    RequestedPackage, TypedDependency, TypedManifest,
};
use crate::{
    CANONICAL_REEF_ORG, EmbeddedRuntime, LockSource, LockedDependency, PackageId, ReefLock,
    ReefManifest, RegistryVersion, acquire_reef_home_lock, document_schema, parse_remote_origin,
    read_lockfile, read_registry_index, registry_root, sha256_file, try_github_token,
    validate_manifest_schema_with, verify_artifact_pair, write_lockfile_unlocked,
};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::time::{Duration, Instant};
use tar::Archive;

pub const MAX_GITHUB_RELEASE_PAGES_PER_PACKAGE: u64 = 10;
pub const MAX_HTTP_REQUESTS_PER_COMMAND: u64 = 2_048;
pub const MAX_ACCEPTED_RELEASE_TAGS_PER_PACKAGE: u64 = 256;
/// Every candidate of a package can be downloaded once, so the download
/// budget equals the resolver's candidates-per-name limit.
pub const MAX_CANDIDATE_MANIFESTS_PER_PACKAGE: u64 = package_versioning::MAX_CANDIDATES_PER_PACKAGE;
/// Resolution and refresh take this many of the highest listed versions
/// satisfying each known requirement.
pub const MAX_CANDIDATES_PER_REQUIREMENT: usize = 64;
/// Inspection downloads this many listed releases, matching ones first.
pub const MAX_INSPECTED_CANDIDATES_PER_PACKAGE: usize = 64;
pub const MAX_COMPRESSED_CANDIDATE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_SCANNED_CANDIDATE_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
pub const MAX_TOTAL_CANDIDATE_DOWNLOAD_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_HTTP_REQUEST_TIME: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMode {
    Locked,
    Resolve,
    Refresh,
    Inspect,
}

impl FromStr for DiscoveryMode {
    type Err = DiscoveryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "locked" => Ok(Self::Locked),
            "resolve" => Ok(Self::Resolve),
            "refresh" => Ok(Self::Refresh),
            "inspect" => Ok(Self::Inspect),
            _ => Err(DiscoveryError::UnsupportedMode {
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SourceLocator {
    GitHub { org: String, repo: String },
}

impl SourceLocator {
    pub fn github_parts(&self) -> (&str, &str) {
        match self {
            Self::GitHub { org, repo } => (org, repo),
        }
    }
}

impl FromStr for SourceLocator {
    type Err = DiscoveryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let rest =
            value
                .strip_prefix("github://")
                .ok_or_else(|| DiscoveryError::UnsupportedSource {
                    value: value.to_string(),
                })?;
        let (org, repo) = rest
            .split_once('/')
            .ok_or_else(|| DiscoveryError::InvalidSource {
                value: value.to_string(),
                reason: "expected github://<org>/<repo>".to_string(),
            })?;
        if org.is_empty() || org.len() > 39 {
            return Err(DiscoveryError::InvalidSource {
                value: value.to_string(),
                reason: "invalid GitHub organization".to_string(),
            });
        }
        let org_bytes = org.as_bytes();
        if !org_bytes.first().is_some_and(u8::is_ascii_alphanumeric)
            || !org_bytes.last().is_some_and(u8::is_ascii_alphanumeric)
            || !org_bytes
                .iter()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
        {
            return Err(DiscoveryError::InvalidSource {
                value: value.to_string(),
                reason: "GitHub organization must use canonical lowercase ASCII".to_string(),
            });
        }
        let parsed_repo =
            crate::package_versioning::PackageName::from_str(repo).map_err(|error| {
                DiscoveryError::InvalidSource {
                    value: value.to_string(),
                    reason: error.to_string(),
                }
            })?;
        Ok(Self::GitHub {
            org: org.to_string(),
            repo: parsed_repo.to_string(),
        })
    }
}

impl fmt::Display for SourceLocator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GitHub { org, repo } => write!(formatter, "github://{org}/{repo}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetDimension {
    ReleasePages,
    HttpRequests,
    AcceptedTags,
    CandidateManifests,
    CompressedCandidateBytes,
    ScannedCandidateBytes,
    ManifestBytes,
    TotalDownloadBytes,
    RequestElapsedTime,
}

impl BudgetDimension {
    fn production_limit(self) -> u64 {
        match self {
            Self::ReleasePages => MAX_GITHUB_RELEASE_PAGES_PER_PACKAGE,
            Self::HttpRequests => MAX_HTTP_REQUESTS_PER_COMMAND,
            Self::AcceptedTags => MAX_ACCEPTED_RELEASE_TAGS_PER_PACKAGE,
            Self::CandidateManifests => MAX_CANDIDATE_MANIFESTS_PER_PACKAGE,
            Self::CompressedCandidateBytes => MAX_COMPRESSED_CANDIDATE_BYTES,
            Self::ScannedCandidateBytes => MAX_SCANNED_CANDIDATE_BYTES,
            Self::ManifestBytes => MAX_MANIFEST_BYTES,
            Self::TotalDownloadBytes => MAX_TOTAL_CANDIDATE_DOWNLOAD_BYTES,
            Self::RequestElapsedTime => MAX_HTTP_REQUEST_TIME.as_secs(),
        }
    }
}

impl fmt::Display for BudgetDimension {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::ReleasePages => "GitHub release pages per package",
            Self::HttpRequests => "HTTP requests per command",
            Self::AcceptedTags => "accepted release tags per package",
            Self::CandidateManifests => "fetched candidate manifests per package",
            Self::CompressedCandidateBytes => "compressed candidate archive bytes",
            Self::ScannedCandidateBytes => "scanned decompressed candidate bytes",
            Self::ManifestBytes => "parsed reef.toml bytes",
            Self::TotalDownloadBytes => "total candidate download bytes",
            Self::RequestElapsedTime => "elapsed HTTP request seconds",
        };
        formatter.write_str(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetError {
    dimension: BudgetDimension,
    limit: u64,
    observed: u64,
    package: String,
    operation: String,
}

impl BudgetError {
    pub fn dimension(&self) -> BudgetDimension {
        self.dimension
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn observed(&self) -> u64 {
        self.observed
    }

    pub fn package(&self) -> &str {
        &self.package
    }

    pub fn operation(&self) -> &str {
        &self.operation
    }
}

impl fmt::Display for BudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Reef remote discovery {} limit {} reached at {} for package `{}` during {}",
            self.dimension, self.limit, self.observed, self.package, self.operation
        )
    }
}

impl std::error::Error for BudgetError {}

#[derive(Debug)]
pub enum DiscoveryError {
    UnsupportedMode {
        value: String,
    },
    UnsupportedSource {
        value: String,
    },
    InvalidSource {
        value: String,
        reason: String,
    },
    RemoteUnavailable {
        package: String,
        reason: String,
    },
    Budget(BudgetError),
    CandidateArchive {
        package: String,
        message: String,
    },
    CandidateManifest {
        package: String,
        message: String,
    },
    Resolution {
        error: Box<package_versioning::VersioningError>,
        exclusions: Box<BTreeMap<String, Vec<String>>>,
    },
    Io {
        operation: String,
        message: String,
    },
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMode { value } => {
                write!(formatter, "unsupported Reef discovery mode `{value}`")
            }
            Self::UnsupportedSource { value } => {
                write!(formatter, "unsupported Reef source locator `{value}`")
            }
            Self::InvalidSource { value, reason } => {
                write!(formatter, "invalid Reef source locator `{value}`: {reason}")
            }
            Self::RemoteUnavailable { package, reason } => {
                write!(
                    formatter,
                    "remote discovery is unavailable for `{package}`: {reason}"
                )
            }
            Self::Budget(error) => error.fmt(formatter),
            Self::CandidateArchive { package, message } => {
                write!(
                    formatter,
                    "candidate archive for `{package}` is invalid: {message}"
                )
            }
            Self::CandidateManifest { package, message } => {
                write!(
                    formatter,
                    "candidate manifest for `{package}` is invalid: {message}"
                )
            }
            Self::Resolution { error, exclusions } => {
                write!(formatter, "{error}")?;
                if !exclusions.is_empty() {
                    formatter.write_str("; excluded provider results:")?;
                    for (package, entries) in exclusions.iter() {
                        write!(formatter, " {package}=[{}]", entries.join("; "))?;
                    }
                }
                if crate::offline_mode()
                    && matches!(
                        error.as_ref(),
                        package_versioning::VersioningError::IncompatibleRequirements { .. }
                    )
                {
                    formatter.write_str(crate::offline_import_hint())?;
                }
                Ok(())
            }
            Self::Io { operation, message } => write!(formatter, "{operation}: {message}"),
        }
    }
}

impl std::error::Error for DiscoveryError {}

impl From<BudgetError> for DiscoveryError {
    fn from(error: BudgetError) -> Self {
        Self::Budget(error)
    }
}

#[derive(Debug, Clone)]
pub struct ResolutionBudget {
    counters: BTreeMap<(BudgetDimension, String), u64>,
    limits: BTreeMap<BudgetDimension, u64>,
}

impl ResolutionBudget {
    pub fn production() -> Self {
        let limits = [
            BudgetDimension::ReleasePages,
            BudgetDimension::HttpRequests,
            BudgetDimension::AcceptedTags,
            BudgetDimension::CandidateManifests,
            BudgetDimension::CompressedCandidateBytes,
            BudgetDimension::ScannedCandidateBytes,
            BudgetDimension::ManifestBytes,
            BudgetDimension::TotalDownloadBytes,
            BudgetDimension::RequestElapsedTime,
        ]
        .into_iter()
        .map(|dimension| (dimension, dimension.production_limit()))
        .collect();
        Self {
            counters: BTreeMap::new(),
            limits,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_limits(overrides: &[(BudgetDimension, u64)]) -> Self {
        let mut budget = Self::production();
        for (dimension, limit) in overrides {
            budget.limits.insert(*dimension, *limit);
        }
        budget
    }

    fn limit(&self, dimension: BudgetDimension) -> u64 {
        self.limits
            .get(&dimension)
            .copied()
            .expect("every budget dimension has a limit")
    }

    pub fn charge(
        &mut self,
        dimension: BudgetDimension,
        amount: u64,
        package: &str,
        operation: &str,
    ) -> Result<(), BudgetError> {
        let key_scope = match dimension {
            BudgetDimension::ReleasePages
            | BudgetDimension::AcceptedTags
            | BudgetDimension::CandidateManifests => package.to_string(),
            BudgetDimension::CompressedCandidateBytes
            | BudgetDimension::ScannedCandidateBytes
            | BudgetDimension::ManifestBytes => format!("{package}:{operation}"),
            BudgetDimension::HttpRequests
            | BudgetDimension::TotalDownloadBytes
            | BudgetDimension::RequestElapsedTime => "<command>".to_string(),
        };
        let key = (dimension, key_scope);
        let current = self.counters.get(&key).copied().unwrap_or(0);
        let limit = self.limit(dimension);
        let observed = match current.checked_add(amount) {
            Some(observed) => observed,
            None => {
                return Err(BudgetError {
                    dimension,
                    limit,
                    observed: u64::MAX,
                    package: package.to_string(),
                    operation: operation.to_string(),
                });
            }
        };
        if observed > limit {
            return Err(BudgetError {
                dimension,
                limit,
                observed,
                package: package.to_string(),
                operation: operation.to_string(),
            });
        }
        self.counters.insert(key, observed);
        Ok(())
    }

    pub fn check_request_elapsed(
        &self,
        elapsed: Duration,
        package: &str,
        operation: &str,
    ) -> Result<(), BudgetError> {
        let observed = elapsed.as_secs();
        let limit = self.limit(BudgetDimension::RequestElapsedTime);
        if elapsed > Duration::from_secs(limit) {
            return Err(BudgetError {
                dimension: BudgetDimension::RequestElapsedTime,
                limit,
                observed,
                package: package.to_string(),
                operation: operation.to_string(),
            });
        }
        Ok(())
    }
}

struct LimitedCounter<R> {
    inner: R,
    observed: Rc<Cell<u64>>,
    limit: u64,
}

impl<R: Read> Read for LimitedCounter<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        let Some(observed) = self.observed.get().checked_add(read as u64) else {
            self.observed.set(u64::MAX);
            return Err(io::Error::other("candidate scan counter overflow"));
        };
        self.observed.set(observed);
        if observed > self.limit {
            return Err(io::Error::other(
                "candidate decompression scan limit reached",
            ));
        }
        Ok(read)
    }
}

fn archive_path_is_safe(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.to_string_lossy().contains('\\')
        && path.components().all(|component| {
            matches!(component, Component::Normal(_))
                && component.as_os_str() != "."
                && component.as_os_str() != ".."
        })
}

fn inspect_candidate_archive_typed(
    archive_path: &Path,
    package: &str,
    budget: &mut ResolutionBudget,
) -> Result<(ReefManifest, TypedManifest), DiscoveryError> {
    let candidate_operation = format!("inspect candidate archive {}", archive_path.display());
    let compressed = fs::metadata(archive_path)
        .map_err(|error| DiscoveryError::Io {
            operation: format!("inspect {}", archive_path.display()),
            message: error.to_string(),
        })?
        .len();
    budget.charge(
        BudgetDimension::CompressedCandidateBytes,
        compressed,
        package,
        &candidate_operation,
    )?;

    let file = fs::File::open(archive_path).map_err(|error| DiscoveryError::Io {
        operation: format!("open {}", archive_path.display()),
        message: error.to_string(),
    })?;
    let decoder = zstd::stream::read::Decoder::new(file).map_err(|error| {
        DiscoveryError::CandidateArchive {
            package: package.to_string(),
            message: format!("open zstd stream: {error}"),
        }
    })?;
    let scanned = Rc::new(Cell::new(0_u64));
    let reader = LimitedCounter {
        inner: decoder,
        observed: Rc::clone(&scanned),
        limit: budget.limit(BudgetDimension::ScannedCandidateBytes),
    };
    let mut archive = Archive::new(reader);
    let mut manifest = None;
    let result = (|| -> Result<(), DiscoveryError> {
        let entries = archive
            .entries()
            .map_err(|error| DiscoveryError::CandidateArchive {
                package: package.to_string(),
                message: format!("read tar entries: {error}"),
            })?;
        for entry in entries {
            let mut entry = entry.map_err(|error| DiscoveryError::CandidateArchive {
                package: package.to_string(),
                message: format!("read tar record: {error}"),
            })?;
            let path = entry
                .path()
                .map_err(|error| DiscoveryError::CandidateArchive {
                    package: package.to_string(),
                    message: format!("read tar path: {error}"),
                })?;
            if !archive_path_is_safe(&path) {
                return Err(DiscoveryError::CandidateArchive {
                    package: package.to_string(),
                    message: format!("unsafe tar path `{}`", path.display()),
                });
            }
            let kind = entry.header().entry_type();
            if !(kind.is_file() || kind.is_dir()) {
                return Err(DiscoveryError::CandidateArchive {
                    package: package.to_string(),
                    message: format!(
                        "tar entry `{}` is not a regular file or directory",
                        path.display()
                    ),
                });
            }
            if path == Path::new("reef.toml") {
                if !kind.is_file() {
                    return Err(DiscoveryError::CandidateArchive {
                        package: package.to_string(),
                        message: "root reef.toml must be a regular file".to_string(),
                    });
                }
                if manifest.is_some() {
                    return Err(DiscoveryError::CandidateArchive {
                        package: package.to_string(),
                        message: "archive must contain exactly one root reef.toml".to_string(),
                    });
                }
                let mut bytes = Vec::new();
                entry
                    .by_ref()
                    .take(budget.limit(BudgetDimension::ManifestBytes) + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|error| DiscoveryError::CandidateArchive {
                        package: package.to_string(),
                        message: format!("read root reef.toml: {error}"),
                    })?;
                budget.charge(
                    BudgetDimension::ManifestBytes,
                    bytes.len() as u64,
                    package,
                    &candidate_operation,
                )?;
                manifest = Some(bytes);
            }
        }
        Ok(())
    })();
    let observed = scanned.get();
    if observed > budget.limit(BudgetDimension::ScannedCandidateBytes) {
        let error = budget
            .charge(
                BudgetDimension::ScannedCandidateBytes,
                observed,
                package,
                &candidate_operation,
            )
            .expect_err("scan over the production limit");
        return Err(DiscoveryError::Budget(error));
    }
    result?;
    budget.charge(
        BudgetDimension::ScannedCandidateBytes,
        observed,
        package,
        &candidate_operation,
    )?;
    let bytes = manifest.ok_or_else(|| DiscoveryError::CandidateArchive {
        package: package.to_string(),
        message: "archive must contain exactly one root reef.toml".to_string(),
    })?;
    let text = std::str::from_utf8(&bytes).map_err(|error| DiscoveryError::CandidateManifest {
        package: package.to_string(),
        message: format!("reef.toml is not UTF-8: {error}"),
    })?;
    let virtual_path = Path::new("candidate/reef.toml");
    let schema = document_schema::manifest_schema_version(text, virtual_path).map_err(|error| {
        DiscoveryError::CandidateManifest {
            package: package.to_string(),
            message: error.to_string(),
        }
    })?;
    let manifest =
        document_schema::parse_manifest_text(text, virtual_path, false).map_err(|error| {
            DiscoveryError::CandidateManifest {
                package: package.to_string(),
                message: error.to_string(),
            }
        })?;
    let typed = validate_manifest_schema_with(&manifest, schema, crate::allow_dep_compiler_drift())
        .map_err(|message| DiscoveryError::CandidateManifest {
            package: package.to_string(),
            message,
        })?;
    Ok((manifest, typed))
}

pub fn inspect_candidate_manifest_archive(
    archive_path: &Path,
    package: &str,
    budget: &mut ResolutionBudget,
) -> Result<ReefManifest, DiscoveryError> {
    inspect_candidate_archive_typed(archive_path, package, budget).map(|(manifest, _)| manifest)
}

#[derive(Debug, Clone, Deserialize)]
struct GitHubAsset {
    id: u64,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<GitHubAsset>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProviderRelease {
    version: PackageVersion,
    tag: String,
    archive_asset: u64,
    shell_asset: u64,
}

/// A listed release whose archive discovery has not downloaded. Resolver 2
/// orders candidates by lock preference and version, both known from the
/// listing, so a release waits until the search selects it; see
/// [`DiscoverySession::resolve`].
#[derive(Debug, Clone)]
struct DeferredRelease {
    locator: SourceLocator,
    release: ProviderRelease,
}

impl DeferredRelease {
    fn source_identity(&self) -> String {
        format!("{}@{}", self.locator, self.release.tag)
    }

    /// Never equal to an archive hash, so a deferred stand-in cannot be
    /// mistaken for fetched material.
    fn content_identity(&self) -> String {
        format!("deferred:{}", self.source_identity())
    }
}

#[derive(Debug)]
struct ProviderListing {
    releases: Vec<ProviderRelease>,
    exclusions: Vec<String>,
}

#[derive(Debug, Clone)]
struct RemoteMaterial {
    typed: TypedManifest,
    artifacts: BTreeMap<String, crate::ArtifactSpec>,
    dependencies: Vec<RequestedPackage>,
    locator: SourceLocator,
    tag: String,
    shell_asset: u64,
    archive_path: PathBuf,
    archive_sha256: String,
    verified_shell_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
enum CandidateMaterial {
    Path {
        root: PathBuf,
        lock_path: String,
        typed: TypedManifest,
        artifacts: BTreeMap<String, crate::ArtifactSpec>,
        dependencies: Vec<RequestedPackage>,
    },
    Local {
        entry: RegistryVersion,
        typed: TypedManifest,
        artifacts: BTreeMap<String, crate::ArtifactSpec>,
        dependencies: Vec<RequestedPackage>,
    },
    BundledRuntime {
        runtime: &'static EmbeddedRuntime,
        typed: TypedManifest,
        dependencies: Vec<RequestedPackage>,
    },
    Remote(RemoteMaterial),
}

impl CandidateMaterial {
    fn typed(&self) -> &TypedManifest {
        match self {
            Self::Path { typed, .. }
            | Self::Local { typed, .. }
            | Self::BundledRuntime { typed, .. } => typed,
            Self::Remote(material) => &material.typed,
        }
    }

    fn artifacts(&self) -> Option<&BTreeMap<String, crate::ArtifactSpec>> {
        match self {
            Self::Path { artifacts, .. } | Self::Local { artifacts, .. } => Some(artifacts),
            Self::Remote(material) => Some(&material.artifacts),
            Self::BundledRuntime { .. } => None,
        }
    }

    fn dependencies(&self) -> &[RequestedPackage] {
        match self {
            Self::Path { dependencies, .. }
            | Self::Local { dependencies, .. }
            | Self::BundledRuntime { dependencies, .. } => dependencies,
            Self::Remote(material) => &material.dependencies,
        }
    }

    fn source_identity(&self) -> String {
        match self {
            Self::Path { root, .. } => format!("path://{}", root.display()),
            Self::Local { entry, .. } => entry
                .remote_origin
                .clone()
                .unwrap_or_else(|| format!("local://{}", entry.archive_sha256)),
            Self::BundledRuntime { typed, .. } => format!("bundled://{}", typed.compiler),
            Self::Remote(material) => {
                format!("{}@{}", material.locator, material.tag)
            }
        }
    }

    fn content_identity(&self) -> String {
        match self {
            Self::Path { root, .. } => format!("editable:{}", root.display()),
            Self::Local { entry, .. } => {
                format!("{}#{}", entry.archive_sha256, entry.shell_sha256)
            }
            Self::BundledRuntime { runtime, .. } => {
                format!("{}#{}", runtime.archive_sha256(), runtime.shell_sha256())
            }
            Self::Remote(material) => material.archive_sha256.clone(),
        }
    }
}

trait CandidateProvider {
    fn list_releases(
        &self,
        locator: &SourceLocator,
        package: &PackageName,
        budget: &mut ResolutionBudget,
    ) -> Result<ProviderListing, DiscoveryError>;

    fn fetch_manifest(
        &self,
        locator: &SourceLocator,
        package: &PackageName,
        release: &ProviderRelease,
        temp_root: &Path,
        budget: &mut ResolutionBudget,
    ) -> Result<RemoteMaterial, DiscoveryError>;

    fn fetch_selected_shell(
        &self,
        material: &RemoteMaterial,
        temp_root: &Path,
        budget: &mut ResolutionBudget,
    ) -> Result<PathBuf, DiscoveryError>;
}

/// Render Unix seconds as `YYYY-MM-DD HH:MM:SS UTC` (proleptic Gregorian).
fn utc_timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let second_of_day = seconds % 86_400;
    // Civil date from days since 1970-01-01, after Howard Hinnant's
    // `civil_from_days`.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} UTC",
        second_of_day / 3_600,
        second_of_day % 3_600 / 60,
        second_of_day % 60
    )
}

struct GitHubReleaseProvider {
    client: reqwest::blocking::Client,
    api_base: String,
    token: Option<String>,
}

struct AssetDownload<'a> {
    locator: &'a SourceLocator,
    package: &'a PackageName,
    asset: u64,
    destination: &'a Path,
    operation: &'a str,
    compressed_candidate: bool,
}

impl GitHubReleaseProvider {
    fn new() -> Result<Self, DiscoveryError> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(MAX_HTTP_REQUEST_TIME)
            .user_agent("chelis-reef")
            .build()
            .map_err(|error| DiscoveryError::Io {
                operation: "construct GitHub discovery client".to_string(),
                message: crate::http_client_error(&error),
            })?;
        Ok(Self {
            client,
            api_base: std::env::var("CHELIS_REEF_GITHUB_BASE_API")
                .unwrap_or_else(|_| "https://api.github.com".to_string()),
            token: try_github_token(),
        })
    }

    fn get(
        &self,
        url: &str,
        accept: &str,
        locator: &SourceLocator,
        package: &PackageName,
        operation: &str,
        budget: &mut ResolutionBudget,
    ) -> Result<(reqwest::blocking::Response, Instant), DiscoveryError> {
        budget.charge(
            BudgetDimension::HttpRequests,
            1,
            package.as_str(),
            operation,
        )?;
        let mut request = self.client.get(url).header("Accept", accept);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("token {token}"));
        }
        let started = Instant::now();
        let response = match request.send() {
            Ok(response) => response,
            Err(error) => {
                if let Err(budget_error) =
                    budget.check_request_elapsed(started.elapsed(), package.as_str(), operation)
                {
                    return Err(DiscoveryError::Budget(budget_error));
                }
                return Err(DiscoveryError::RemoteUnavailable {
                    package: package.to_string(),
                    reason: format!("{operation} at {url}: {error}"),
                });
            }
        };
        budget.check_request_elapsed(started.elapsed(), package.as_str(), operation)?;
        if let Some(reason) = self.rate_limit_reason(&response, locator) {
            return Err(DiscoveryError::RemoteUnavailable {
                package: package.to_string(),
                reason: format!(
                    "{operation} at {url} returned HTTP {}: {reason}",
                    response.status()
                ),
            });
        }
        if !response.status().is_success() {
            return Err(DiscoveryError::RemoteUnavailable {
                package: package.to_string(),
                reason: format!(
                    "{operation} at {url} returned HTTP {}{}",
                    response.status(),
                    if matches!(response.status().as_u16(), 401 | 403 | 404) {
                        "; for a restricted repository, set GITHUB_TOKEN with contents: read access or sign in with gh auth login"
                    } else {
                        ""
                    }
                ),
            });
        }
        Ok((response, started))
    }

    /// GitHub reports an exhausted primary limit as HTTP 403 or 429 with
    /// `x-ratelimit-remaining: 0`. A secondary limit is HTTP 429, or HTTP
    /// 403 with `retry-after`, while primary quota remains.
    fn rate_limit_reason(
        &self,
        response: &reqwest::blocking::Response,
        locator: &SourceLocator,
    ) -> Option<String> {
        let status = response.status().as_u16();
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        };
        let (org, repo) = locator.github_parts();
        let install = format!(
            "install a release without discovery with `chelis reef install --from-github {org}/{repo}@<tag>`"
        );
        if matches!(status, 403 | 429) && header("x-ratelimit-remaining").as_deref() == Some("0") {
            let reset = header("x-ratelimit-reset")
                .and_then(|reset| reset.parse::<u64>().ok())
                .map(|reset| format!(" until {}", utc_timestamp(reset)))
                .unwrap_or_default();
            return Some(if self.token.is_some() {
                format!(
                    "GitHub's API rate limit for the configured token is exhausted{reset}; wait for the reset, or {install}"
                )
            } else {
                format!(
                    "GitHub's API rate limit for anonymous requests is exhausted{reset}; set GITHUB_TOKEN or sign in with gh auth login to use the authenticated limit, or {install}"
                )
            });
        }
        let retry_after = header("retry-after");
        if status == 429 || (status == 403 && retry_after.is_some()) {
            let wait = retry_after
                .map(|seconds| format!("retry after {seconds} seconds"))
                .unwrap_or_else(|| "retry later".to_string());
            return Some(format!(
                "GitHub's secondary rate limit refused the request; {wait}, or {install}"
            ));
        }
        None
    }

    fn asset_url(&self, locator: &SourceLocator, asset: u64) -> String {
        let (org, repo) = locator.github_parts();
        format!(
            "{}/repos/{org}/{repo}/releases/assets/{asset}",
            self.api_base.trim_end_matches('/')
        )
    }

    fn download_asset(
        &self,
        download: AssetDownload<'_>,
        budget: &mut ResolutionBudget,
    ) -> Result<(), DiscoveryError> {
        let AssetDownload {
            locator,
            package,
            asset,
            destination,
            operation,
            compressed_candidate,
        } = download;
        let url = self.asset_url(locator, asset);
        let (mut response, started) = self.get(
            &url,
            "application/octet-stream",
            locator,
            package,
            operation,
            budget,
        )?;
        let mut output = fs::File::create(destination).map_err(|error| DiscoveryError::Io {
            operation: format!("create {}", destination.display()),
            message: error.to_string(),
        })?;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = match response.read(&mut buffer) {
                Ok(read) => read,
                Err(error) => {
                    if let Err(budget_error) =
                        budget.check_request_elapsed(started.elapsed(), package.as_str(), operation)
                    {
                        return Err(DiscoveryError::Budget(budget_error));
                    }
                    return Err(DiscoveryError::RemoteUnavailable {
                        package: package.to_string(),
                        reason: format!("read {operation}: {error}"),
                    });
                }
            };
            budget.check_request_elapsed(started.elapsed(), package.as_str(), operation)?;
            if read == 0 {
                break;
            }
            budget.charge(
                BudgetDimension::TotalDownloadBytes,
                read as u64,
                package.as_str(),
                operation,
            )?;
            if compressed_candidate {
                budget.charge(
                    BudgetDimension::CompressedCandidateBytes,
                    read as u64,
                    package.as_str(),
                    operation,
                )?;
            }
            output
                .write_all(&buffer[..read])
                .map_err(|error| DiscoveryError::Io {
                    operation: format!("write {}", destination.display()),
                    message: error.to_string(),
                })?;
        }
        output.sync_all().map_err(|error| DiscoveryError::Io {
            operation: format!("sync {}", destination.display()),
            message: error.to_string(),
        })?;
        Ok(())
    }
}

impl CandidateProvider for GitHubReleaseProvider {
    fn list_releases(
        &self,
        locator: &SourceLocator,
        package: &PackageName,
        budget: &mut ResolutionBudget,
    ) -> Result<ProviderListing, DiscoveryError> {
        let (org, repo) = locator.github_parts();
        if repo != package.as_str() {
            return Err(DiscoveryError::InvalidSource {
                value: locator.to_string(),
                reason: format!("repository `{repo}` does not match package `{package}`"),
            });
        }
        let mut releases = Vec::new();
        let mut exclusions = Vec::new();
        for page in 1..=MAX_GITHUB_RELEASE_PAGES_PER_PACKAGE + 1 {
            budget.charge(
                BudgetDimension::ReleasePages,
                1,
                package.as_str(),
                "list GitHub releases",
            )?;
            let url = format!(
                "{}/repos/{org}/{repo}/releases?per_page=100&page={page}",
                self.api_base.trim_end_matches('/')
            );
            let (mut response, started) = self.get(
                &url,
                "application/vnd.github+json",
                locator,
                package,
                "list GitHub releases",
                budget,
            )?;
            let has_next = response
                .headers()
                .get("link")
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.contains("rel=\"next\""));
            let mut body = Vec::new();
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let read = match response.read(&mut buffer) {
                    Ok(read) => read,
                    Err(error) => {
                        if let Err(budget_error) = budget.check_request_elapsed(
                            started.elapsed(),
                            package.as_str(),
                            "list GitHub releases",
                        ) {
                            return Err(DiscoveryError::Budget(budget_error));
                        }
                        return Err(DiscoveryError::RemoteUnavailable {
                            package: package.to_string(),
                            reason: format!("read GitHub release page {page}: {error}"),
                        });
                    }
                };
                budget.check_request_elapsed(
                    started.elapsed(),
                    package.as_str(),
                    "list GitHub releases",
                )?;
                if read == 0 {
                    break;
                }
                budget.charge(
                    BudgetDimension::TotalDownloadBytes,
                    read as u64,
                    package.as_str(),
                    "list GitHub releases",
                )?;
                body.extend_from_slice(&buffer[..read]);
            }
            let page_releases =
                serde_json::from_slice::<Vec<GitHubRelease>>(&body).map_err(|error| {
                    DiscoveryError::RemoteUnavailable {
                        package: package.to_string(),
                        reason: format!("parse GitHub release page {page}: {error}"),
                    }
                })?;
            let page_len = page_releases.len();
            for release in page_releases {
                if release.draft {
                    exclusions.push(format!("{}: draft release", release.tag_name));
                    continue;
                }
                let raw_version = release
                    .tag_name
                    .strip_prefix('v')
                    .unwrap_or(&release.tag_name);
                let version = match PackageVersion::from_str(raw_version) {
                    Ok(version) => version,
                    Err(error) => {
                        exclusions.push(format!("{}: {error}", release.tag_name));
                        continue;
                    }
                };
                if release.prerelease && version.as_semver().pre.is_empty() {
                    exclusions.push(format!(
                        "{}: stable version marked as a GitHub prerelease",
                        release.tag_name
                    ));
                    continue;
                }
                let archive_name = format!("{repo}-{version}.tar.zst");
                let shell_name = format!("{repo}-{version}.chb");
                let Some(archive_asset) = release
                    .assets
                    .iter()
                    .find(|asset| asset.name == archive_name)
                    .map(|asset| asset.id)
                else {
                    exclusions.push(format!("{}: missing archive asset", release.tag_name));
                    continue;
                };
                let Some(shell_asset) = release
                    .assets
                    .iter()
                    .find(|asset| asset.name == shell_name)
                    .map(|asset| asset.id)
                else {
                    exclusions.push(format!("{}: missing shell asset", release.tag_name));
                    continue;
                };
                releases.push(ProviderRelease {
                    version,
                    tag: release.tag_name,
                    archive_asset,
                    shell_asset,
                });
            }
            // The whole listing is read: a requirement learned later must
            // see the same releases as one known now.
            if !has_next || page_len < 100 {
                break;
            }
        }
        releases.sort_by(|left, right| {
            right
                .version
                .cmp(&left.version)
                .then_with(|| left.tag.cmp(&right.tag))
        });
        for pair in releases.windows(2) {
            if pair[0].version == pair[1].version && pair[0].tag != pair[1].tag {
                return Err(DiscoveryError::InvalidSource {
                    value: locator.to_string(),
                    reason: format!(
                        "release tags `{}` and `{}` select the same package version `{}`",
                        pair[0].tag, pair[1].tag, pair[0].version
                    ),
                });
            }
        }
        releases.dedup_by(|left, right| left.version == right.version && left.tag == right.tag);
        Ok(ProviderListing {
            releases,
            exclusions,
        })
    }

    fn fetch_manifest(
        &self,
        locator: &SourceLocator,
        package: &PackageName,
        release: &ProviderRelease,
        temp_root: &Path,
        budget: &mut ResolutionBudget,
    ) -> Result<RemoteMaterial, DiscoveryError> {
        budget.charge(
            BudgetDimension::CandidateManifests,
            1,
            package.as_str(),
            "fetch candidate manifest",
        )?;
        let archive_path = temp_root.join(format!(
            "{}-{}-{}.tar.zst",
            package, release.version, release.archive_asset
        ));
        let operation = format!("fetch candidate archive {}", release.tag);
        self.download_asset(
            AssetDownload {
                locator,
                package,
                asset: release.archive_asset,
                destination: &archive_path,
                operation: &operation,
                compressed_candidate: true,
            },
            budget,
        )?;
        let (manifest, typed) =
            inspect_candidate_archive_typed(&archive_path, package.as_str(), budget)?;
        if typed.package.name != *package || typed.package.version != release.version {
            return Err(DiscoveryError::CandidateManifest {
                package: package.to_string(),
                message: format!(
                    "release `{}` advertises `{}` but its manifest names `{}`",
                    release.tag, release.version, typed.package
                ),
            });
        }
        let archive_sha256 = sha256_file(&archive_path).map_err(|message| DiscoveryError::Io {
            operation: format!("hash {}", archive_path.display()),
            message,
        })?;
        Ok(RemoteMaterial {
            typed,
            artifacts: manifest.artifacts,
            dependencies: Vec::new(),
            locator: locator.clone(),
            tag: release.tag.clone(),
            shell_asset: release.shell_asset,
            archive_path,
            archive_sha256,
            verified_shell_path: None,
        })
    }

    fn fetch_selected_shell(
        &self,
        material: &RemoteMaterial,
        temp_root: &Path,
        budget: &mut ResolutionBudget,
    ) -> Result<PathBuf, DiscoveryError> {
        let package = &material.typed.package.name;
        let path = temp_root.join(format!(
            "{}-{}-{}.chb",
            package, material.typed.package.version, material.shell_asset
        ));
        let operation = format!("fetch selected shell {}", material.tag);
        self.download_asset(
            AssetDownload {
                locator: &material.locator,
                package,
                asset: material.shell_asset,
                destination: &path,
                operation: &operation,
                compressed_candidate: false,
            },
            budget,
        )?;
        Ok(path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionChange {
    pub package: String,
    pub previous: Option<String>,
    pub selected: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateReport {
    pub mode: DiscoveryMode,
    pub changes: Vec<VersionChange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutdatedPackage {
    pub package: String,
    pub current: Option<String>,
    pub newest_compatible: Option<String>,
    pub newest_incompatible: Option<String>,
    pub blocked: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutdatedReport {
    pub packages: Vec<OutdatedPackage>,
}

struct ProjectSeed {
    root: PathBuf,
    manifest: crate::ParsedManifest,
    initial_lock: Option<ReefLock>,
}

/// Project preferences and canonical default discovery are distinct inputs.
/// Bootstrap has no fabricated project path, manifest, or project lock.
enum DiscoverySeed {
    Project(Box<ProjectSeed>),
    Bootstrap {
        compiler: package_versioning::ExactCompilerVersion,
    },
}

impl DiscoverySeed {
    fn canonical_remote_only(&self) -> bool {
        matches!(self, Self::Bootstrap { .. })
    }

    fn project(&self) -> Result<&ProjectSeed, DiscoveryError> {
        match self {
            Self::Project(project) => Ok(project),
            Self::Bootstrap { .. } => Err(DiscoveryError::UnsupportedMode {
                value: "default bootstrap has no project state".to_string(),
            }),
        }
    }

    fn compiler(&self) -> &package_versioning::ExactCompilerVersion {
        match self {
            Self::Project(project) => &project.manifest.typed.compiler,
            Self::Bootstrap { compiler } => compiler,
        }
    }
}

struct DiscoverySession {
    seed: DiscoverySeed,
    mode: DiscoveryMode,
    target: Option<PackageName>,
    network_enabled: bool,
    root_requests: Vec<RequestedPackage>,
    candidates: BTreeMap<PackageName, Vec<CandidateMaterial>>,
    path_stack: Vec<PackageName>,
    pending_local: BTreeSet<PackageName>,
    loaded_local: BTreeMap<PackageName, Vec<PackageRequirement>>,
    queried_remote: BTreeSet<PackageName>,
    deferred: BTreeMap<PackageName, Vec<DeferredRelease>>,
    /// Each queried package's accepted releases, newest first.
    listings: BTreeMap<PackageName, (SourceLocator, Vec<ProviderRelease>)>,
    /// The known requirements each package's window was last derived for.
    windowed: BTreeMap<PackageName, Vec<PackageRequirement>>,
    /// Listed versions already fetched or deferred.
    placed: BTreeMap<PackageName, BTreeSet<PackageVersion>>,
    /// Packages a targeted refresh queries with their remote dependencies.
    refresh_closure: BTreeSet<PackageName>,
    refresh_dependencies: BTreeSet<PackageName>,
    exclusions: BTreeMap<PackageName, Vec<String>>,
    locked: BTreeMap<PackageName, LockedDependency>,
    budget: ResolutionBudget,
    explored_states: u64,
    temp: tempfile::TempDir,
    provider: Option<GitHubReleaseProvider>,
    runtime: &'static EmbeddedRuntime,
    /// Test oracle: query and search as eager discovery did, for the
    /// differential test against deferred discovery.
    #[cfg(test)]
    eager_reference: bool,
}

impl DiscoverySession {
    fn new(
        root: &Path,
        mode: DiscoveryMode,
        target: Option<&str>,
        network_enabled: bool,
        runtime: &'static EmbeddedRuntime,
    ) -> Result<Self, DiscoveryError> {
        if mode == DiscoveryMode::Locked {
            return Err(DiscoveryError::UnsupportedMode {
                value: "locked mode cannot create a discovery session".to_string(),
            });
        }
        let root = root.canonicalize().map_err(|error| DiscoveryError::Io {
            operation: format!("canonicalize {}", root.display()),
            message: error.to_string(),
        })?;
        let root_manifest = crate::read_manifest(&root.join("reef.toml")).map_err(|message| {
            DiscoveryError::CandidateManifest {
                package: "<root>".to_string(),
                message,
            }
        })?;
        if root_manifest.typed.resolver != package_versioning::ResolverVersion::Two {
            return Err(DiscoveryError::CandidateManifest {
                package: root_manifest.typed.package.name.to_string(),
                message: "remote discovery requires manifest schema 2 and resolver 2".to_string(),
            });
        }
        let target = target
            .map(PackageName::from_str)
            .transpose()
            .map_err(|error| DiscoveryError::CandidateManifest {
                package: "<target>".to_string(),
                message: error.to_string(),
            })?;
        let initial_lock = if root.join("reef.lock").exists() {
            Some(
                read_lockfile(&root.join("reef.lock"))
                    .map_err(|message| DiscoveryError::CandidateManifest {
                        package: root_manifest.typed.package.name.to_string(),
                        message,
                    })?
                    .raw,
            )
        } else {
            None
        };
        let root_name = root_manifest.typed.package.name.clone();
        let root_typed = root_manifest.typed.clone();
        let root_path = root.clone();
        let mut session = Self::from_seed(
            DiscoverySeed::Project(Box::new(ProjectSeed {
                root,
                manifest: root_manifest,
                initial_lock,
            })),
            mode,
            target,
            network_enabled,
            runtime,
        )?;
        session.root_requests =
            session.requests_for_manifest(&root_name, Some(&root_path), &root_typed)?;
        if let Some(target) = &session.target
            && !session
                .root_requests
                .iter()
                .any(|request| &request.package == target)
            && !session.locked.contains_key(target)
        {
            return Err(DiscoveryError::CandidateManifest {
                package: target.to_string(),
                message: "target package is not present in the manifest or lock".to_string(),
            });
        }
        Ok(session)
    }

    fn from_seed(
        seed: DiscoverySeed,
        mode: DiscoveryMode,
        target: Option<PackageName>,
        network_enabled: bool,
        runtime: &'static EmbeddedRuntime,
    ) -> Result<Self, DiscoveryError> {
        let mut locked = BTreeMap::new();
        if let DiscoverySeed::Project(project) = &seed
            && let Some(lock) = &project.initial_lock
        {
            for dependency in &lock.dependencies {
                if matches!(dependency.source, LockSource::Binary { .. }) {
                    continue;
                }
                let name = PackageName::from_str(&dependency.name).map_err(|error| {
                    DiscoveryError::CandidateManifest {
                        package: dependency.name.clone(),
                        message: error.to_string(),
                    }
                })?;
                locked.insert(name, dependency.clone());
            }
        }
        let temp = tempfile::tempdir().map_err(|error| DiscoveryError::Io {
            operation: "create remote discovery temporary directory".to_string(),
            message: error.to_string(),
        })?;
        Ok(Self {
            seed,
            mode,
            target,
            network_enabled,
            root_requests: Vec::new(),
            candidates: BTreeMap::new(),
            path_stack: Vec::new(),
            pending_local: BTreeSet::new(),
            loaded_local: BTreeMap::new(),
            queried_remote: BTreeSet::new(),
            deferred: BTreeMap::new(),
            listings: BTreeMap::new(),
            windowed: BTreeMap::new(),
            placed: BTreeMap::new(),
            refresh_closure: BTreeSet::new(),
            refresh_dependencies: BTreeSet::new(),
            exclusions: BTreeMap::new(),
            locked,
            budget: ResolutionBudget::production(),
            explored_states: 0,
            temp,
            provider: None,
            runtime,
            #[cfg(test)]
            eager_reference: false,
        })
    }

    fn for_bootstrap(runtime: &'static EmbeddedRuntime) -> Result<Self, DiscoveryError> {
        let compiler =
            package_versioning::ExactCompilerVersion::from_str(crate::CURRENT_COMPILER_VERSION)
                .map_err(|error| DiscoveryError::CandidateManifest {
                    package: "<compiler>".to_string(),
                    message: error.to_string(),
                })?;
        let mut session = Self::from_seed(
            DiscoverySeed::Bootstrap { compiler },
            DiscoveryMode::Refresh,
            None,
            true,
            runtime,
        )?;
        let requirement = PackageRequirement::from_str("*").map_err(|error| {
            DiscoveryError::CandidateManifest {
                package: "<default-roots>".to_string(),
                message: error.to_string(),
            }
        })?;
        for root in crate::DEFAULT_BOOTSTRAP_ROOTS {
            let name =
                PackageName::from_str(root).map_err(|error| DiscoveryError::CandidateManifest {
                    package: root.to_string(),
                    message: error.to_string(),
                })?;
            session.pending_local.insert(name.clone());
            session.root_requests.push(RequestedPackage::new(
                name.clone(),
                name,
                requirement.clone(),
            ));
        }
        let name = PackageName::from_str(crate::CHELIS_STD_PACKAGE_NAME).map_err(|error| {
            DiscoveryError::CandidateManifest {
                package: crate::CHELIS_STD_PACKAGE_NAME.to_string(),
                message: error.to_string(),
            }
        })?;
        let version = PackageVersion::from_str(runtime.version()).map_err(|error| {
            DiscoveryError::CandidateManifest {
                package: name.to_string(),
                message: error.to_string(),
            }
        })?;
        session.pending_local.insert(name.clone());
        session.root_requests.push(RequestedPackage::new(
            name.clone(),
            name,
            PackageRequirement::exact(&version),
        ));
        Ok(session)
    }

    fn requests_for_manifest(
        &mut self,
        requester: &PackageName,
        package_root: Option<&Path>,
        manifest: &TypedManifest,
    ) -> Result<Vec<RequestedPackage>, DiscoveryError> {
        let mut requests = Vec::new();
        for (name, dependency) in &manifest.dependencies {
            match dependency {
                TypedDependency::Registry { requirement } => {
                    self.pending_local.insert(name.clone());
                    requests.push(RequestedPackage::new(
                        requester.clone(),
                        name.clone(),
                        requirement.as_package_requirement(),
                    ));
                }
                TypedDependency::Path { path, requirement } => {
                    if let Some(position) = self.path_stack.iter().position(|entry| entry == name) {
                        let mut cycle = self.path_stack[position..]
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>();
                        cycle.push(name.to_string());
                        return Err(DiscoveryError::CandidateManifest {
                            package: name.to_string(),
                            message: format!(
                                "path dependency cycle detected: {}",
                                cycle.join(" -> ")
                            ),
                        });
                    }
                    let Some(package_root) = package_root else {
                        return Err(DiscoveryError::CandidateManifest {
                            package: requester.to_string(),
                            message: format!(
                                "path dependency `{name}` has no package root to resolve `{path}` against"
                            ),
                        });
                    };
                    let dependency_root =
                        package_root.join(path).canonicalize().map_err(|error| {
                            DiscoveryError::CandidateManifest {
                                package: name.to_string(),
                                message: format!("resolve path dependency `{path}`: {error}"),
                            }
                        })?;
                    let parsed = crate::read_manifest(&dependency_root.join("reef.toml")).map_err(
                        |message| DiscoveryError::CandidateManifest {
                            package: name.to_string(),
                            message,
                        },
                    )?;
                    if parsed.typed.package.name != *name {
                        return Err(DiscoveryError::CandidateManifest {
                            package: name.to_string(),
                            message: format!("path manifest names `{}`", parsed.typed.package.name),
                        });
                    }
                    let exact = requirement.clone().unwrap_or_else(|| {
                        package_versioning::DependencyRequirement::Compatible(
                            PackageRequirement::exact(&parsed.typed.package.version),
                        )
                    });
                    requests.push(RequestedPackage::new(
                        requester.clone(),
                        name.clone(),
                        exact.as_package_requirement(),
                    ));
                    let lock_path = dependency_root
                        .strip_prefix(&self.seed.project()?.root)
                        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                        .unwrap_or_else(|_| path.clone());
                    if self.candidates.get(name).into_iter().flatten().any(
                        |material| matches!(material, CandidateMaterial::Path { root, .. } if root == &dependency_root),
                    ) {
                        continue;
                    }
                    self.path_stack.push(name.clone());
                    let child_requests =
                        self.requests_for_manifest(name, Some(&dependency_root), &parsed.typed);
                    self.path_stack.pop();
                    let child_requests = child_requests?;
                    self.insert_material(
                        name.clone(),
                        CandidateMaterial::Path {
                            root: dependency_root,
                            lock_path,
                            typed: parsed.typed,
                            artifacts: parsed.raw.artifacts,
                            dependencies: child_requests,
                        },
                    )?;
                }
            }
        }
        Ok(requests)
    }

    fn insert_material(
        &mut self,
        package: PackageName,
        material: CandidateMaterial,
    ) -> Result<(), DiscoveryError> {
        let entries = self.candidates.entry(package.clone()).or_default();
        let version = &material.typed().package.version;
        for existing in entries.iter() {
            if &existing.typed().package.version != version {
                continue;
            }
            if matches!(existing, CandidateMaterial::Path { .. })
                || matches!(material, CandidateMaterial::Path { .. })
            {
                if existing.source_identity() == material.source_identity() {
                    return Ok(());
                }
                continue;
            }
            let same_archive = match (existing, &material) {
                (CandidateMaterial::Local { entry, .. }, CandidateMaterial::Remote(remote))
                | (CandidateMaterial::Remote(remote), CandidateMaterial::Local { entry, .. }) => {
                    entry.archive_sha256 == remote.archive_sha256
                }
                _ => false,
            };
            if same_archive || existing.content_identity() == material.content_identity() {
                return Ok(());
            }
            return Err(DiscoveryError::CandidateManifest {
                package: package.to_string(),
                message: format!(
                    "source conflict for `{package}` `{version}` between {} and {}",
                    existing.source_identity(),
                    material.source_identity()
                ),
            });
        }
        if entries.len() as u64 >= package_versioning::MAX_CANDIDATES_PER_PACKAGE {
            return Err(DiscoveryError::Resolution {
                error: Box::new(package_versioning::VersioningError::LimitExceeded {
                    dimension: "candidates per package name",
                    package: Some(package),
                    limit: package_versioning::MAX_CANDIDATES_PER_PACKAGE,
                    attempted: entries.len() as u64 + 1,
                }),
                exclusions: Box::new(BTreeMap::new()),
            });
        }
        entries.push(material);
        Ok(())
    }

    /// Load local candidates and derive remote windows until both are
    /// current for every known requirement.
    fn load_local_closure(&mut self) -> Result<(), DiscoveryError> {
        loop {
            self.load_local_candidates()?;
            let Some(package) = self
                .listings
                .keys()
                .find(|package| {
                    self.windowed.get(*package) != Some(&self.known_requirements(package))
                })
                .cloned()
            else {
                return Ok(());
            };
            self.derive_window(&package)?;
        }
    }

    /// A queried package's window is, for each known requirement on it, the
    /// highest listed versions satisfying that requirement. It depends only
    /// on the listing and the known requirements, and only grows as they
    /// do. Each release enters once: as a deferred stand-in, or fetched now
    /// when a registry candidate already has its version.
    fn derive_window(&mut self, package: &PackageName) -> Result<(), DiscoveryError> {
        let requirements = self.known_requirements(package);
        self.windowed.insert(package.clone(), requirements.clone());
        let Some((locator, releases)) = self.listings.get(package).cloned() else {
            return Ok(());
        };
        let mut window = BTreeSet::new();
        for requirement in &requirements {
            window.extend(
                releases
                    .iter()
                    .enumerate()
                    .filter(|(_, release)| requirement.matches(&release.version))
                    .take(MAX_CANDIDATES_PER_REQUIREMENT)
                    .map(|(position, _)| position),
            );
        }
        for position in window {
            let release = releases[position].clone();
            if !self
                .placed
                .entry(package.clone())
                .or_default()
                .insert(release.version.clone())
            {
                continue;
            }
            self.accept_tag(package, &release)?;
            let deferred = DeferredRelease {
                locator: locator.clone(),
                release,
            };
            if self.has_registry_material(package, &deferred.release.version) {
                self.fetch_listed(package, &deferred)?;
            } else {
                self.deferred
                    .entry(package.clone())
                    .or_default()
                    .push(deferred);
            }
        }
        Ok(())
    }

    fn accept_tag(
        &mut self,
        package: &PackageName,
        release: &ProviderRelease,
    ) -> Result<(), DiscoveryError> {
        self.budget.charge(
            BudgetDimension::AcceptedTags,
            1,
            package.as_str(),
            &format!("accept GitHub release tag {}", release.tag),
        )?;
        Ok(())
    }

    fn load_local_candidates(&mut self) -> Result<(), DiscoveryError> {
        while let Some(package) = self
            .pending_local
            .iter()
            .find(|package| {
                self.loaded_local.get(*package) != Some(&self.known_requirements(package))
            })
            .cloned()
        {
            let requirements = self.known_requirements(&package);
            self.loaded_local
                .insert(package.clone(), requirements.clone());
            if package.as_str() == crate::CHELIS_STD_PACKAGE_NAME {
                let runtime_package = self.runtime.package().map_err(|message| {
                    DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message: format!("load compiler-bundled runtime: {message}"),
                    }
                })?;
                let typed =
                    crate::typed_manifest_for_loaded(&runtime_package).map_err(|message| {
                        DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message,
                        }
                    })?;
                if typed.package.name != package
                    || typed.package.version.to_string() != self.runtime.version()
                {
                    return Err(DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message: format!(
                            "bundled runtime manifest names `{}` but the compiler advertises `{}@{}`",
                            typed.package,
                            crate::CHELIS_STD_PACKAGE_NAME,
                            self.runtime.version()
                        ),
                    });
                }
                // The embedded runtime has no filesystem root to resolve a
                // path dependency against.
                let dependencies = self.requests_for_manifest(&package, None, &typed)?;
                self.insert_material(
                    package,
                    CandidateMaterial::BundledRuntime {
                        runtime: self.runtime,
                        typed,
                        dependencies,
                    },
                )?;
                continue;
            }
            if self.seed.canonical_remote_only() {
                continue;
            }
            let registry = registry_root().map_err(|message| DiscoveryError::Io {
                operation: "resolve Reef registry root".to_string(),
                message,
            })?;
            let index = read_registry_index(&registry).map_err(|message| DiscoveryError::Io {
                operation: "read Reef registry index".to_string(),
                message,
            })?;
            let entries = index
                .packages
                .get(package.as_str())
                .cloned()
                .unwrap_or_default();
            for entry in entries {
                let version = PackageVersion::from_str(&entry.version).map_err(|error| {
                    DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message: format!("invalid registry candidate version: {error}"),
                    }
                })?;
                if let Some(locked) = self.locked.get(&package)
                    && locked.version == entry.version
                    && matches!(&locked.source, LockSource::LocalRegistry { .. })
                {
                    // Compare index hashes before loading: missing cache
                    // directories must not bypass an existing lock pin.
                    crate::verify_locked_hashes(
                        locked,
                        Some(&entry.archive_sha256),
                        Some(&entry.shell_sha256),
                    )
                    .map_err(|message| DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message,
                    })?;
                }
                // A root requirement applies in every solution. Requirements
                // from other candidates are alternatives until those
                // candidates are selected, so they only widen discovery when
                // the root does not exclude the version.
                if self.mode == DiscoveryMode::Resolve
                    && self
                        .root_requests
                        .iter()
                        .filter(|request| request.package == package)
                        .any(|request| !request.requirement.matches(&version))
                {
                    continue;
                }
                let matching = requirements
                    .iter()
                    .filter(|requirement| requirement.matches(&version))
                    .map(PackageRequirement::as_str)
                    .collect::<Vec<_>>();
                if self.mode == DiscoveryMode::Resolve && matching.is_empty() {
                    continue;
                }
                if self.candidates.get(&package).into_iter().flatten().any(
                    |candidate| matches!(candidate, CandidateMaterial::Local { entry: loaded, .. } if loaded.version == entry.version),
                ) {
                    continue;
                }
                let context = if matching.is_empty() {
                    format!("local registry candidate {}", entry.version)
                } else {
                    format!(
                        "local registry candidate {} matching {}",
                        entry.version,
                        matching.join(", ")
                    )
                };
                let installed = match crate::load_registry_package(package.as_str(), &entry.version)
                {
                    Ok(installed) => installed,
                    Err(crate::LoadRegistryError::MissingPackageDir)
                    | Err(crate::LoadRegistryError::MissingFromIndex) => continue,
                    Err(crate::LoadRegistryError::Other(message)) => {
                        return Err(DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message: format!("load {context}: {message}"),
                        });
                    }
                };
                let parsed =
                    crate::read_manifest(&installed.root.join("reef.toml")).map_err(|message| {
                        DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message: format!("read {context}: {message}"),
                        }
                    })?;
                if parsed.typed.package.name != package {
                    return Err(DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message: format!(
                            "registry candidate names `{}`",
                            parsed.typed.package.name
                        ),
                    });
                }
                if parsed.typed.package.version.to_string() != entry.version
                    || parsed.typed.compiler.to_string() != entry.compiler
                {
                    return Err(DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message: "registry index and candidate manifest identities disagree"
                            .to_string(),
                    });
                }
                // A fetched release of this version must precede the
                // registry copy, as it would have without deferral.
                self.fetch_deferred(&package, &version)?;
                let dependencies =
                    self.requests_for_manifest(&package, Some(&installed.root), &parsed.typed)?;
                self.insert_material(
                    package.clone(),
                    CandidateMaterial::Local {
                        entry,
                        typed: parsed.typed,
                        artifacts: parsed.raw.artifacts,
                        dependencies,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn source_locator(&self, package: &PackageName) -> Result<SourceLocator, DiscoveryError> {
        if self.seed.canonical_remote_only() {
            return Ok(SourceLocator::GitHub {
                org: CANONICAL_REEF_ORG.to_string(),
                repo: package.to_string(),
            });
        }
        let mut locators = BTreeSet::new();
        let mut add_origin = |origin: &str| -> Result<(), DiscoveryError> {
            let exact =
                parse_remote_origin(origin).map_err(|error| DiscoveryError::InvalidSource {
                    value: origin.to_string(),
                    reason: error.to_string(),
                })?;
            locators.insert(SourceLocator::from_str(&format!(
                "github://{}/{}",
                exact.org, exact.repo
            ))?);
            Ok(())
        };
        if let Some(LockedDependency {
            source:
                LockSource::LocalRegistry {
                    remote_origin: Some(origin),
                },
            ..
        }) = self.locked.get(package)
        {
            add_origin(origin)?;
        }
        for material in self.candidates.get(package).into_iter().flatten() {
            let CandidateMaterial::Local { entry, .. } = material else {
                continue;
            };
            if let Some(origin) = &entry.remote_origin {
                add_origin(origin)?;
            }
        }
        if locators.len() > 1 {
            return Err(DiscoveryError::InvalidSource {
                value: package.to_string(),
                reason: "lock and local candidates record inconsistent remote repositories"
                    .to_string(),
            });
        }
        Ok(locators
            .into_iter()
            .next()
            .unwrap_or(SourceLocator::GitHub {
                org: CANONICAL_REEF_ORG.to_string(),
                repo: package.to_string(),
            }))
    }

    fn known_requirements(&self, package: &PackageName) -> Vec<PackageRequirement> {
        let mut requirements = self
            .root_requests
            .iter()
            .chain(
                self.candidates
                    .values()
                    .flatten()
                    .flat_map(CandidateMaterial::dependencies),
            )
            .filter(|request| &request.package == package)
            .map(|request| request.requirement.clone())
            .collect::<Vec<_>>();
        requirements.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        requirements.dedup();
        requirements
    }

    fn query_remote(&mut self, package: &PackageName) -> Result<(), DiscoveryError> {
        if self.queried_remote.contains(package) {
            return Ok(());
        }
        #[cfg(test)]
        if self.eager_reference {
            return self.query_remote_eagerly(package);
        }
        if !self.network_enabled {
            return Err(DiscoveryError::RemoteUnavailable {
                package: package.to_string(),
                reason: "network access is disabled".to_string(),
            });
        }
        let locator = self.source_locator(package)?;
        let known_requirements = self.known_requirements(package);
        let temp_root = self.temp.path().to_path_buf();
        let provider = match self.provider.take() {
            Some(provider) => provider,
            None => GitHubReleaseProvider::new()?,
        };
        let result: Result<(), DiscoveryError> = (|| {
            // An uncached registry pin must be verified before a refresh or
            // inspection can replace or report it. Reserve a manifest slot
            // even when newer releases fill the ordinary candidate window.
            let missing_pin = self
                .locked
                .get(package)
                .filter(|locked| matches!(&locked.source, LockSource::LocalRegistry { .. }))
                .filter(|locked| {
                    !self
                        .candidates
                        .get(package)
                        .into_iter()
                        .flatten()
                        .any(|candidate| match candidate {
                            CandidateMaterial::Local { entry, .. } => {
                                entry.version == locked.version
                            }
                            CandidateMaterial::Path { .. } => true,
                            _ => false,
                        })
                })
                .map(|locked| {
                    PackageVersion::from_str(&locked.version).map_err(|error| {
                        DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message: format!("invalid locked version: {error}"),
                        }
                    })
                })
                .transpose()?;
            let listing = provider.list_releases(&locator, package, &mut self.budget)?;
            self.exclusions
                .entry(package.clone())
                .or_default()
                .extend(listing.exclusions);
            let mut releases = listing.releases;
            let pinned_release = if let Some(version) = &missing_pin {
                let position = releases
                    .iter()
                    .position(|release| &release.version == version)
                    .ok_or_else(|| DiscoveryError::RemoteUnavailable {
                        package: package.to_string(),
                        reason: format!(
                            "locked release `{version}` is unavailable within the bounded GitHub release listing"
                        ),
                    })?;
                Some(releases.remove(position))
            } else {
                None
            };
            if self.mode == DiscoveryMode::Inspect {
                // Inspection reports versions the selection does not use,
                // so it downloads its whole window now.
                releases.sort_by(|left, right| {
                    let left_matches = known_requirements
                        .iter()
                        .any(|requirement| requirement.matches(&left.version));
                    let right_matches = known_requirements
                        .iter()
                        .any(|requirement| requirement.matches(&right.version));
                    right_matches
                        .cmp(&left_matches)
                        .then_with(|| right.version.cmp(&left.version))
                        .then_with(|| left.tag.cmp(&right.tag))
                });
                for release in pinned_release
                    .into_iter()
                    .chain(releases)
                    .take(MAX_INSPECTED_CANDIDATES_PER_PACKAGE)
                {
                    self.accept_tag(package, &release)?;
                    self.fetch_release(
                        &provider,
                        &locator,
                        package,
                        &release,
                        &temp_root,
                        missing_pin.as_ref(),
                    )?;
                }
            } else {
                if let Some(release) = &pinned_release {
                    self.placed
                        .entry(package.clone())
                        .or_default()
                        .insert(release.version.clone());
                    self.accept_tag(package, release)?;
                    self.fetch_release(
                        &provider,
                        &locator,
                        package,
                        release,
                        &temp_root,
                        missing_pin.as_ref(),
                    )?;
                }
                // The window is derived from this listing in
                // `load_local_closure`, and again whenever the package's
                // known requirements change.
                self.listings
                    .insert(package.clone(), (locator.clone(), releases));
            }
            if let Some(version) = &missing_pin
                && !self
                    .candidates
                    .get(package)
                    .into_iter()
                    .flatten()
                    .any(|candidate| {
                        matches!(candidate, CandidateMaterial::Remote(material) if &material.typed.package.version == version)
                    })
            {
                return Err(DiscoveryError::RemoteUnavailable {
                    package: package.to_string(),
                    reason: format!("locked release `{version}` could not be verified"),
                });
            }
            Ok(())
        })();
        self.provider = Some(provider);
        result?;
        self.queried_remote.insert(package.clone());
        self.load_local_closure()
    }

    /// The eager query step that deferred discovery replaced, kept as the
    /// differential oracle's reference. Listing stopped at the first page
    /// on which enough releases matched the requirements known at query
    /// time, charging every tag read; the window was those releases ranked
    /// matching first, then newest first, and its first 64 were all
    /// downloaded. The oracle publishes releases newest first, so the
    /// pages read are a prefix of the sorted listing. Lock pins are not
    /// modelled.
    #[cfg(test)]
    fn query_remote_eagerly(&mut self, package: &PackageName) -> Result<(), DiscoveryError> {
        const EAGER_WINDOW: usize = 64;
        let locator = self.source_locator(package)?;
        let known_requirements = self.known_requirements(package);
        let temp_root = self.temp.path().to_path_buf();
        let provider = match self.provider.take() {
            Some(provider) => provider,
            None => GitHubReleaseProvider::new()?,
        };
        let result: Result<(), DiscoveryError> = (|| {
            let required = if !known_requirements.is_empty()
                && known_requirements
                    .iter()
                    .all(|requirement| requirement.as_str().starts_with('='))
            {
                1
            } else {
                EAGER_WINDOW
            };
            // The full listing is read under a scratch budget; only the
            // pages eager discovery read are charged below.
            let mut scratch =
                ResolutionBudget::with_test_limits(&[(BudgetDimension::AcceptedTags, u64::MAX)]);
            let listing = provider.list_releases(&locator, package, &mut scratch)?;
            self.exclusions
                .entry(package.clone())
                .or_default()
                .extend(listing.exclusions);
            let mut releases = listing.releases;
            let mut read = releases.len();
            let mut matched = 0;
            for (position, release) in releases.iter().enumerate() {
                if known_requirements
                    .iter()
                    .any(|requirement| requirement.matches(&release.version))
                {
                    matched += 1;
                    if matched >= required {
                        read = read.min((position / 100 + 1) * 100);
                        break;
                    }
                }
            }
            releases.truncate(read);
            self.budget.charge(
                BudgetDimension::AcceptedTags,
                releases.len() as u64,
                package.as_str(),
                "accept GitHub release tags",
            )?;
            releases.sort_by(|left, right| {
                let left_matches = known_requirements
                    .iter()
                    .any(|requirement| requirement.matches(&left.version));
                let right_matches = known_requirements
                    .iter()
                    .any(|requirement| requirement.matches(&right.version));
                right_matches
                    .cmp(&left_matches)
                    .then_with(|| right.version.cmp(&left.version))
                    .then_with(|| left.tag.cmp(&right.tag))
            });
            for release in releases.into_iter().take(EAGER_WINDOW) {
                self.fetch_release(&provider, &locator, package, &release, &temp_root, None)?;
            }
            Ok(())
        })();
        self.provider = Some(provider);
        result?;
        self.queried_remote.insert(package.clone());
        self.load_local_closure()
    }

    /// Download one release's archive, verify it against its tag and any
    /// lock pin, and add it as a candidate or record why it is excluded.
    fn fetch_release(
        &mut self,
        provider: &GitHubReleaseProvider,
        locator: &SourceLocator,
        package: &PackageName,
        release: &ProviderRelease,
        temp_root: &Path,
        missing_pin: Option<&PackageVersion>,
    ) -> Result<(), DiscoveryError> {
        let mut material =
            match provider.fetch_manifest(locator, package, release, temp_root, &mut self.budget) {
                Ok(material) => material,
                Err(error @ DiscoveryError::CandidateArchive { .. })
                | Err(error @ DiscoveryError::CandidateManifest { .. }) => {
                    if self.locked.get(package).is_some_and(|locked| {
                        locked.version == release.version.to_string()
                            && matches!(&locked.source, LockSource::LocalRegistry { .. })
                    }) {
                        return Err(error);
                    }
                    self.exclusions
                        .entry(package.clone())
                        .or_default()
                        .push(format!("{}: {error}", release.tag));
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
        if let Some(locked) = self.locked.get(package)
            && locked.version == release.version.to_string()
            && matches!(&locked.source, LockSource::LocalRegistry { .. })
        {
            crate::verify_locked_hashes(locked, Some(&material.archive_sha256), None).map_err(
                |message| DiscoveryError::CandidateManifest {
                    package: package.to_string(),
                    message,
                },
            )?;
            // A verified installed pin already supplies its shell.
            // An uncached pin must verify both downloaded assets.
            if missing_pin == Some(&release.version) {
                let shell_path =
                    provider.fetch_selected_shell(&material, temp_root, &mut self.budget)?;
                let verified = verify_artifact_pair(&material.archive_path, &shell_path).map_err(
                    |message| DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message,
                    },
                )?;
                crate::verify_locked_hashes(
                    locked,
                    Some(&verified.archive_sha256),
                    Some(&verified.shell_sha256),
                )
                .map_err(|message| DiscoveryError::CandidateManifest {
                    package: package.to_string(),
                    message,
                })?;
                if verified.package.name != package.as_str()
                    || verified.package.version != release.version.to_string()
                    || verified.compiler != material.typed.compiler.to_string()
                {
                    return Err(DiscoveryError::CandidateManifest {
                        package: package.to_string(),
                        message: "locked remote archive, shell, and manifest identities disagree"
                            .to_string(),
                    });
                }
                material.verified_shell_path = Some(shell_path);
            }
        }
        let typed = material.typed.clone();
        if self.seed.canonical_remote_only() && typed.compiler != *self.seed.compiler() {
            self.exclusions
                .entry(package.clone())
                .or_default()
                .push(format!(
                    "{}: compiler pin `{}` does not match `{}`",
                    release.tag,
                    typed.compiler,
                    self.seed.compiler()
                ));
            return Ok(());
        }
        if let Some((dependency, path)) =
            typed
                .dependencies
                .iter()
                .find_map(|(dependency, declaration)| match declaration {
                    TypedDependency::Path { path, .. } => Some((dependency, path)),
                    TypedDependency::Registry { .. } => None,
                })
        {
            self.exclusions
                .entry(package.clone())
                .or_default()
                .push(format!(
                    "{}: remote candidate `{}` contains path dependency `{dependency}` at `{path}`",
                    release.tag, typed.package
                ));
            return Ok(());
        }
        material.dependencies = self.requests_for_manifest(package, Some(temp_root), &typed)?;
        if self.refresh_closure.contains(package) {
            self.refresh_dependencies.extend(
                material
                    .dependencies
                    .iter()
                    .map(|request| request.package.clone()),
            );
        }
        self.insert_material(package.clone(), CandidateMaterial::Remote(material))?;
        Ok(())
    }

    /// Whether a fetched or registry candidate already has `version`, so a
    /// release of that version must be fetched for the source-conflict
    /// check. A path source overrides every registry candidate.
    fn has_registry_material(&self, package: &PackageName, version: &PackageVersion) -> bool {
        self.candidates
            .get(package)
            .into_iter()
            .flatten()
            .any(|material| {
                !matches!(material, CandidateMaterial::Path { .. })
                    && &material.typed().package.version == version
            })
    }

    fn is_deferred(&self, candidate: &LocalCandidate) -> bool {
        self.deferred
            .get(&candidate.id.name)
            .into_iter()
            .flatten()
            .any(|deferred| {
                deferred.release.version == candidate.id.version
                    && deferred.content_identity() == candidate.content_identity
            })
    }

    /// Download a deferred release of `package` at `version`, if one is
    /// waiting.
    fn fetch_deferred(
        &mut self,
        package: &PackageName,
        version: &PackageVersion,
    ) -> Result<(), DiscoveryError> {
        let Some(releases) = self.deferred.get_mut(package) else {
            return Ok(());
        };
        let Some(position) = releases
            .iter()
            .position(|deferred| &deferred.release.version == version)
        else {
            return Ok(());
        };
        let deferred = releases.remove(position);
        if releases.is_empty() {
            self.deferred.remove(package);
        }
        self.fetch_listed(package, &deferred)
    }

    fn fetch_listed(
        &mut self,
        package: &PackageName,
        deferred: &DeferredRelease,
    ) -> Result<(), DiscoveryError> {
        let provider = match self.provider.take() {
            Some(provider) => provider,
            None => GitHubReleaseProvider::new()?,
        };
        let temp_root = self.temp.path().to_path_buf();
        let result = self.fetch_release(
            &provider,
            &deferred.locator,
            package,
            &deferred.release,
            &temp_root,
            None,
        );
        self.provider = Some(provider);
        result
    }

    fn lock_preferred(&self, package: &PackageName, version: &PackageVersion) -> bool {
        let permitted = match self.mode {
            DiscoveryMode::Resolve => true,
            DiscoveryMode::Refresh => self.target.as_ref().is_some_and(|target| target != package),
            DiscoveryMode::Inspect | DiscoveryMode::Locked => false,
        };
        permitted
            && self
                .locked
                .get(package)
                .is_some_and(|dependency| dependency.version == version.to_string())
    }

    fn resolver_candidates(&self) -> BTreeMap<PackageName, Vec<LocalCandidate>> {
        let mut candidates = self.material_candidates();
        // A deferred release stands in with no dependencies. Its subtree is
        // then at least as satisfiable as the real one, so a search that
        // rejects it rejects the release, and a search that selects it
        // fetches it and searches again.
        for (package, releases) in &self.deferred {
            let entries = candidates.entry(package.clone()).or_default();
            for deferred in releases {
                let source_identity = deferred.source_identity();
                entries.push(LocalCandidate {
                    id: package_versioning::ResolvedPackageId {
                        name: package.clone(),
                        version: deferred.release.version.clone(),
                    },
                    source: if self.lock_preferred(package, &deferred.release.version) {
                        CandidateSource::Locked { source_identity }
                    } else {
                        CandidateSource::Remote { source_identity }
                    },
                    content_identity: deferred.content_identity(),
                    dependencies: Vec::new(),
                });
            }
        }
        candidates
    }

    fn material_candidates(&self) -> BTreeMap<PackageName, Vec<LocalCandidate>> {
        self.candidates
            .iter()
            .map(|(package, materials)| {
                let candidates = materials
                    .iter()
                    .map(|material| {
                        let typed = material.typed();
                        let source_identity = material.source_identity();
                        let source = if matches!(material, CandidateMaterial::Path { .. }) {
                            CandidateSource::Path {
                                canonical_path: source_identity
                                    .strip_prefix("path://")
                                    .unwrap_or(&source_identity)
                                    .to_string(),
                            }
                        } else if self.lock_preferred(package, &typed.package.version) {
                            CandidateSource::Locked { source_identity }
                        } else {
                            match material {
                                CandidateMaterial::Local { .. } => {
                                    CandidateSource::LocalRegistry { source_identity }
                                }
                                CandidateMaterial::BundledRuntime { typed, .. } => {
                                    CandidateSource::BundledRuntime {
                                        compiler_version: typed.compiler.clone(),
                                    }
                                }
                                CandidateMaterial::Remote(_) => {
                                    CandidateSource::Remote { source_identity }
                                }
                                CandidateMaterial::Path { .. } => unreachable!(),
                            }
                        };
                        LocalCandidate {
                            id: typed.package.clone(),
                            source,
                            content_identity: material.content_identity(),
                            dependencies: material.dependencies().to_vec(),
                        }
                    })
                    .collect();
                (package.clone(), candidates)
            })
            .collect()
    }

    fn resolve_once(
        &mut self,
    ) -> Result<package_versioning::LocalResolution, package_versioning::VersioningError> {
        let candidates = self.resolver_candidates();
        // Each search has the whole state limit. Fetching a selected
        // stand-in reruns the search, so a session-wide count would make
        // the limit depend on how many releases were deferred.
        #[cfg(test)]
        let cumulative = self.eager_reference;
        #[cfg(not(test))]
        let cumulative = false;
        if !cumulative {
            self.explored_states = 0;
        }
        package_versioning::resolve_local_with_state_counter(
            &self.root_requests,
            &candidates,
            package_versioning::ResolverLimits::default(),
            &mut self.explored_states,
        )
    }

    fn refresh_all_known(&mut self) -> Result<(), DiscoveryError> {
        loop {
            self.load_local_closure()?;
            let next = self
                .pending_local
                .iter()
                .find(|package| {
                    package.as_str() != crate::CHELIS_STD_PACKAGE_NAME
                        && !self.queried_remote.contains(*package)
                })
                .cloned();
            let Some(package) = next else {
                return Ok(());
            };
            self.query_remote(&package)?;
        }
    }

    /// Query the remote dependencies of a targeted refresh's remote
    /// candidates, transitively.
    fn query_refresh_closure(&mut self) -> Result<(), DiscoveryError> {
        while let Some(dependency) = self
            .refresh_dependencies
            .iter()
            .find(|package| {
                package.as_str() != crate::CHELIS_STD_PACKAGE_NAME
                    && !self.queried_remote.contains(*package)
            })
            .cloned()
        {
            self.refresh_closure.insert(dependency.clone());
            self.query_remote(&dependency)?;
        }
        Ok(())
    }

    fn resolve(&mut self) -> Result<package_versioning::LocalResolution, DiscoveryError> {
        self.load_local_closure()?;
        match self.mode {
            DiscoveryMode::Refresh | DiscoveryMode::Inspect if self.target.is_none() => {
                if self.network_enabled {
                    self.refresh_all_known()?;
                }
            }
            DiscoveryMode::Refresh | DiscoveryMode::Inspect => {
                let target = self.target.clone().expect("target exists");
                if self.network_enabled && target.as_str() != crate::CHELIS_STD_PACKAGE_NAME {
                    self.refresh_closure.insert(target.clone());
                    self.query_remote(&target)?;
                    self.query_refresh_closure()?;
                }
            }
            DiscoveryMode::Resolve | DiscoveryMode::Locked => {}
        }
        loop {
            match self.resolve_once() {
                Ok(resolution) => {
                    // Fetch the first deferred release in the selection, then
                    // query remote dependencies as eager fetching would have.
                    let Some(candidate) = resolution
                        .selected
                        .values()
                        .find(|candidate| self.is_deferred(candidate))
                        .cloned()
                    else {
                        return Ok(resolution);
                    };
                    self.fetch_deferred(&candidate.id.name, &candidate.id.version)?;
                    self.load_local_closure()?;
                    match self.mode {
                        DiscoveryMode::Refresh | DiscoveryMode::Inspect
                            if self.target.is_some() =>
                        {
                            self.query_refresh_closure()?;
                        }
                        DiscoveryMode::Locked => {}
                        DiscoveryMode::Resolve
                        | DiscoveryMode::Refresh
                        | DiscoveryMode::Inspect => self.refresh_all_known()?,
                    }
                }
                Err(package_versioning::VersioningError::IncompatibleRequirements { .. })
                    if self.network_enabled
                        && self.mode == DiscoveryMode::Resolve
                        && self.pending_local.iter().any(|package| {
                            package.as_str() != crate::CHELIS_STD_PACKAGE_NAME
                                && !self.queried_remote.contains(package)
                        }) =>
                {
                    self.refresh_all_known()?;
                }
                Err(package_versioning::VersioningError::IncompatibleRequirements {
                    package,
                    ..
                }) if self.network_enabled
                    && package.as_str() != crate::CHELIS_STD_PACKAGE_NAME
                    && !self.queried_remote.contains(&package) =>
                {
                    self.query_remote(&package)?;
                }
                Err(error) => {
                    return Err(DiscoveryError::Resolution {
                        error: Box::new(error),
                        exclusions: Box::new(
                            self.exclusions
                                .iter()
                                .map(|(package, entries)| (package.to_string(), entries.clone()))
                                .collect(),
                        ),
                    });
                }
            }
        }
    }

    fn material_for(
        &self,
        candidate: &LocalCandidate,
    ) -> Result<&CandidateMaterial, DiscoveryError> {
        self.candidates
            .get(&candidate.id.name)
            .into_iter()
            .flatten()
            .find(|material| {
                material.typed().package.version == candidate.id.version
                    && (matches!(candidate.source, CandidateSource::Path { .. })
                        == matches!(material, CandidateMaterial::Path { .. }))
                    && material.content_identity() == candidate.content_identity
            })
            .ok_or_else(|| DiscoveryError::CandidateManifest {
                package: candidate.id.name.to_string(),
                message: format!("selected material `{}` is unavailable", candidate.id),
            })
    }
}

#[derive(Debug, Clone)]
struct StagedRemote {
    package: PackageName,
    version: PackageVersion,
    compiler: String,
    origin: String,
    archive_path: PathBuf,
    shell_path: PathBuf,
    archive_sha256: String,
    shell_sha256: String,
}

fn stage_selected_remote(
    session: &mut DiscoverySession,
    resolution: &package_versioning::LocalResolution,
) -> Result<Vec<StagedRemote>, DiscoveryError> {
    let mut staged = Vec::new();
    let provider = session.provider.take();
    for candidate in resolution.selected.values() {
        let material = session.material_for(candidate)?.clone();
        let CandidateMaterial::Remote(material) = material else {
            continue;
        };
        let provider_ref = provider
            .as_ref()
            .ok_or_else(|| DiscoveryError::RemoteUnavailable {
                package: candidate.id.name.to_string(),
                reason: "the selected provider session is unavailable".to_string(),
            })?;
        let shell_path = match &material.verified_shell_path {
            Some(path) => path.clone(),
            None => provider_ref.fetch_selected_shell(
                &material,
                session.temp.path(),
                &mut session.budget,
            )?,
        };
        let verified =
            verify_artifact_pair(&material.archive_path, &shell_path).map_err(|message| {
                DiscoveryError::CandidateManifest {
                    package: candidate.id.name.to_string(),
                    message,
                }
            })?;
        if verified.package.name != candidate.id.name.as_str()
            || verified.package.version != candidate.id.version.to_string()
            || verified.compiler != material.typed.compiler.to_string()
        {
            return Err(DiscoveryError::CandidateManifest {
                package: candidate.id.name.to_string(),
                message: "selected archive, shell, manifest, and compiler identities disagree"
                    .to_string(),
            });
        }
        if verified.archive_sha256 != material.archive_sha256 {
            return Err(DiscoveryError::CandidateManifest {
                package: candidate.id.name.to_string(),
                message: "selected archive changed after candidate inspection".to_string(),
            });
        }
        if let Some(locked) = session.locked.get(&candidate.id.name)
            && locked.version == candidate.id.version.to_string()
            && matches!(&locked.source, LockSource::LocalRegistry { .. })
        {
            crate::verify_locked_hashes(
                locked,
                Some(&verified.archive_sha256),
                Some(&verified.shell_sha256),
            )
            .map_err(|message| DiscoveryError::CandidateManifest {
                package: candidate.id.name.to_string(),
                message,
            })?;
        }
        staged.push(StagedRemote {
            package: candidate.id.name.clone(),
            version: candidate.id.version.clone(),
            compiler: verified.compiler,
            origin: crate::format_github_origin(
                material.locator.github_parts().0,
                material.locator.github_parts().1,
                &material.tag,
            ),
            archive_path: material.archive_path,
            shell_path,
            archive_sha256: verified.archive_sha256,
            shell_sha256: verified.shell_sha256,
        });
    }
    session.provider = provider;
    Ok(staged)
}

fn lock_for_resolution(
    session: &DiscoverySession,
    resolution: &package_versioning::LocalResolution,
    staged: &[StagedRemote],
) -> Result<ReefLock, DiscoveryError> {
    let project = session.seed.project()?;
    let mut dependencies = Vec::new();
    for candidate in resolution.selected.values() {
        let material = session.material_for(candidate)?;
        let locked = match material {
            CandidateMaterial::Path {
                lock_path, typed, ..
            } => LockedDependency {
                name: candidate.id.name.to_string(),
                version: candidate.id.version.to_string(),
                source: LockSource::Path {
                    path: lock_path.clone(),
                },
                compiler: typed.compiler.to_string(),
                archive_sha256: String::new(),
                shell_sha256: String::new(),
            },
            CandidateMaterial::Local { entry, .. } => LockedDependency {
                name: candidate.id.name.to_string(),
                version: candidate.id.version.to_string(),
                source: LockSource::LocalRegistry {
                    remote_origin: entry.remote_origin.clone(),
                },
                compiler: entry.compiler.clone(),
                archive_sha256: entry.archive_sha256.clone(),
                shell_sha256: entry.shell_sha256.clone(),
            },
            CandidateMaterial::BundledRuntime { runtime, typed, .. } => LockedDependency {
                name: candidate.id.name.to_string(),
                version: candidate.id.version.to_string(),
                source: LockSource::bundled_for_current_compiler(),
                compiler: typed.compiler.to_string(),
                archive_sha256: runtime.archive_sha256().to_string(),
                shell_sha256: runtime.shell_sha256().to_string(),
            },
            CandidateMaterial::Remote(_) => {
                let item = staged
                    .iter()
                    .find(|item| {
                        item.package == candidate.id.name && item.version == candidate.id.version
                    })
                    .ok_or_else(|| DiscoveryError::CandidateManifest {
                        package: candidate.id.name.to_string(),
                        message: "selected remote artifact was not staged".to_string(),
                    })?;
                LockedDependency {
                    name: candidate.id.name.to_string(),
                    version: candidate.id.version.to_string(),
                    source: LockSource::LocalRegistry {
                        remote_origin: Some(item.origin.clone()),
                    },
                    compiler: item.compiler.clone(),
                    archive_sha256: item.archive_sha256.clone(),
                    shell_sha256: item.shell_sha256.clone(),
                }
            }
        };
        dependencies.push(locked);
    }
    // Every lock records the runtime, the runtime's own included, exactly as
    // `build_lockfile` does for a resolver-1 graph.
    if !dependencies
        .iter()
        .any(|dependency| dependency.name == crate::CHELIS_STD_PACKAGE_NAME)
    {
        dependencies.push(LockedDependency {
            name: crate::CHELIS_STD_PACKAGE_NAME.to_string(),
            version: session.runtime.version().to_string(),
            source: LockSource::bundled_for_current_compiler(),
            compiler: project.manifest.typed.compiler.to_string(),
            archive_sha256: session.runtime.archive_sha256().to_string(),
            shell_sha256: session.runtime.shell_sha256().to_string(),
        });
    }
    let mut declarations = vec![(
        &project.manifest.raw.artifacts,
        project.manifest.raw.package.compiler.as_str(),
    )];
    for (candidate, dependency) in resolution.selected.values().zip(&dependencies) {
        let material = session.material_for(candidate)?;
        if let Some(artifacts) = material.artifacts() {
            declarations.push((artifacts, dependency.compiler.as_str()));
        }
    }
    let binaries = crate::binary_lock_entries(declarations).map_err(|message| {
        DiscoveryError::CandidateManifest {
            package: project.manifest.typed.package.name.to_string(),
            message,
        }
    })?;
    dependencies.extend(binaries);
    crate::sort_lock_dependencies(&mut dependencies);
    Ok(ReefLock {
        package: PackageId {
            name: project.manifest.typed.package.name.to_string(),
            version: project.manifest.typed.package.version.to_string(),
        },
        dependencies,
    })
}

fn verify_existing_package(directory: &Path, staged: &StagedRemote) -> Result<(), DiscoveryError> {
    let archive = directory.join(format!("{}-{}.tar.zst", staged.package, staged.version));
    let shell = directory.join(format!("{}-{}.chb", staged.package, staged.version));
    let mut names = fs::read_dir(directory)
        .map_err(|error| DiscoveryError::Io {
            operation: format!("read {}", directory.display()),
            message: error.to_string(),
        })?
        .map(|entry| {
            entry
                .map(|entry| entry.file_name())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|message| DiscoveryError::Io {
            operation: format!("read {}", directory.display()),
            message,
        })?;
    names.sort();
    let mut expected = vec![
        archive
            .file_name()
            .expect("archive file name")
            .to_os_string(),
        shell.file_name().expect("shell file name").to_os_string(),
    ];
    expected.sort();
    if names != expected {
        return Err(DiscoveryError::CandidateManifest {
            package: staged.package.to_string(),
            message: format!(
                "existing cache entry {} is incomplete or has extra bytes",
                directory.display()
            ),
        });
    }
    let verified = verify_artifact_pair(&archive, &shell).map_err(|message| {
        DiscoveryError::CandidateManifest {
            package: staged.package.to_string(),
            message,
        }
    })?;
    if verified.archive_sha256 != staged.archive_sha256
        || verified.shell_sha256 != staged.shell_sha256
        || verified.package.name != staged.package.as_str()
        || verified.package.version != staged.version.to_string()
    {
        return Err(DiscoveryError::CandidateManifest {
            package: staged.package.to_string(),
            message: "existing same-name and same-version cache entry differs by bytes".to_string(),
        });
    }
    Ok(())
}

/// Where a resolved lock goes. A read-only command resolves in memory and
/// leaves the package directory untouched (chelis#1520).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LockPublication {
    /// Replace the project's `reef.lock` last. `project_lock_held` says
    /// whether the caller already holds the package-root write lock.
    Project { project_lock_held: bool },
    /// Publish verified registry entries only; the caller keeps the lock in
    /// memory.
    InMemory,
}

fn publish_lock_last(
    session: &DiscoverySession,
    lock: &ReefLock,
    staged: &[StagedRemote],
    publication: LockPublication,
) -> Result<(), DiscoveryError> {
    let LockPublication::Project { project_lock_held } = publication else {
        return publish_registry_entries(staged);
    };
    let project = session.seed.project()?;
    let _project_lock = if project_lock_held {
        None
    } else {
        Some(
            document_schema::acquire_project_write_lock(&project.root).map_err(|error| {
                DiscoveryError::Io {
                    operation: "acquire package-root project lock".to_string(),
                    message: error.to_string(),
                }
            })?,
        )
    };
    let current_manifest =
        crate::read_manifest(&project.root.join("reef.toml")).map_err(|message| {
            DiscoveryError::CandidateManifest {
                package: project.manifest.typed.package.name.to_string(),
                message,
            }
        })?;
    if current_manifest.raw != project.manifest.raw {
        return Err(DiscoveryError::Io {
            operation: "recheck project manifest under the project lock".to_string(),
            message: "reef.toml changed during remote discovery".to_string(),
        });
    }
    let current_lock = if project.root.join("reef.lock").exists() {
        Some(
            read_lockfile(&project.root.join("reef.lock"))
                .map_err(|message| DiscoveryError::Io {
                    operation: "recheck project lock under the project lock".to_string(),
                    message,
                })?
                .raw,
        )
    } else {
        None
    };
    if current_lock != project.initial_lock {
        return Err(DiscoveryError::Io {
            operation: "recheck project lock under the project lock".to_string(),
            message: "reef.lock changed during remote discovery".to_string(),
        });
    }
    document_schema::ensure_project_lock_ignore(&project.root).map_err(|message| {
        DiscoveryError::Io {
            operation: "update project ignore rules".to_string(),
            message,
        }
    })?;
    publish_registry_entries(staged)?;
    write_lockfile_unlocked(&project.root.join("reef.lock"), lock).map_err(|message| {
        DiscoveryError::Io {
            operation: "replace project reef.lock last".to_string(),
            message,
        }
    })
}

/// Publish every staged remote package and its index entry under the
/// registry lock.
fn publish_registry_entries(staged: &[StagedRemote]) -> Result<(), DiscoveryError> {
    let registry = registry_root().map_err(|message| DiscoveryError::Io {
        operation: "resolve Reef registry root".to_string(),
        message,
    })?;
    publish_registry_entries_at(staged, &registry, RegistryOrigins::PreserveExisting)
}

#[derive(Clone, Copy)]
enum RegistryOrigins {
    PreserveExisting,
    SelectedRemote,
}

fn publish_registry_entries_at(
    staged: &[StagedRemote],
    registry: &Path,
    origins: RegistryOrigins,
) -> Result<(), DiscoveryError> {
    let _registry_lock =
        acquire_reef_home_lock(registry, MAX_HTTP_REQUEST_TIME).map_err(|error| {
            DiscoveryError::Io {
                operation: "acquire Reef registry lock".to_string(),
                message: error.to_string(),
            }
        })?;
    let mut index = read_registry_index(registry).map_err(|message| DiscoveryError::Io {
        operation: "recheck Reef registry index".to_string(),
        message,
    })?;

    for item in staged {
        let verified =
            verify_artifact_pair(&item.archive_path, &item.shell_path).map_err(|message| {
                DiscoveryError::CandidateManifest {
                    package: item.package.to_string(),
                    message,
                }
            })?;
        if verified.archive_sha256 != item.archive_sha256
            || verified.shell_sha256 != item.shell_sha256
            || verified.package.name != item.package.as_str()
            || verified.package.version != item.version.to_string()
        {
            return Err(DiscoveryError::CandidateManifest {
                package: item.package.to_string(),
                message: "staged identity changed before publication".to_string(),
            });
        }
        let package_parent = registry.join("packages").join(item.package.as_str());
        fs::create_dir_all(&package_parent).map_err(|error| DiscoveryError::Io {
            operation: format!("create {}", package_parent.display()),
            message: error.to_string(),
        })?;
        let final_directory = package_parent.join(item.version.to_string());
        if final_directory.exists() {
            verify_existing_package(&final_directory, item)?;
        } else {
            let temporary = tempfile::Builder::new()
                .prefix(".reef-package-")
                .tempdir_in(&package_parent)
                .map_err(|error| DiscoveryError::Io {
                    operation: format!("create package sibling in {}", package_parent.display()),
                    message: error.to_string(),
                })?;
            let archive_name = format!("{}-{}.tar.zst", item.package, item.version);
            let shell_name = format!("{}-{}.chb", item.package, item.version);
            let archive_destination = temporary.path().join(archive_name);
            let shell_destination = temporary.path().join(shell_name);
            fs::copy(&item.archive_path, &archive_destination).map_err(|error| {
                DiscoveryError::Io {
                    operation: format!("stage {}", archive_destination.display()),
                    message: error.to_string(),
                }
            })?;
            fs::copy(&item.shell_path, &shell_destination).map_err(|error| DiscoveryError::Io {
                operation: format!("stage {}", shell_destination.display()),
                message: error.to_string(),
            })?;
            fs::File::open(&archive_destination)
                .and_then(|file| file.sync_all())
                .and_then(|_| fs::File::open(&shell_destination))
                .and_then(|file| file.sync_all())
                .map_err(|error| DiscoveryError::Io {
                    operation: format!("sync staged package `{}`", item.package),
                    message: error.to_string(),
                })?;
            let temporary_path = temporary.keep();
            if let Err(error) = fs::rename(&temporary_path, &final_directory) {
                let _ = fs::remove_dir_all(&temporary_path);
                return Err(DiscoveryError::Io {
                    operation: format!(
                        "publish {} as {}",
                        temporary_path.display(),
                        final_directory.display()
                    ),
                    message: error.to_string(),
                });
            }
            document_schema::sync_parent(&package_parent).map_err(|message| {
                DiscoveryError::Io {
                    operation: format!("sync package parent {}", package_parent.display()),
                    message,
                }
            })?;
        }
        let versions = index.packages.entry(item.package.to_string()).or_default();
        if let Some(existing) = versions
            .iter_mut()
            .find(|entry| entry.version == item.version.to_string())
        {
            if existing.archive_sha256 != item.archive_sha256
                || existing.shell_sha256 != item.shell_sha256
            {
                return Err(DiscoveryError::CandidateManifest {
                    package: item.package.to_string(),
                    message: "registry index conflicts with selected package bytes".to_string(),
                });
            }
            if matches!(origins, RegistryOrigins::SelectedRemote) {
                // Remote observations, not prior cache provenance, declare
                // default bootstrap origins. Artifact hashes remain immutable.
                existing.remote_origin = Some(item.origin.clone());
            }
        } else {
            versions.push(RegistryVersion {
                version: item.version.to_string(),
                compiler: item.compiler.clone(),
                archive_sha256: item.archive_sha256.clone(),
                shell_sha256: item.shell_sha256.clone(),
                remote_origin: Some(item.origin.clone()),
            });
        }
    }

    crate::canonicalize_local_registry_index(&mut index).map_err(|message| DiscoveryError::Io {
        operation: "canonicalize Reef registry index".to_string(),
        message,
    })?;
    if !staged.is_empty() {
        let index_bytes =
            serde_json::to_vec_pretty(&index).map_err(|error| DiscoveryError::Io {
                operation: "serialize Reef registry index".to_string(),
                message: error.to_string(),
            })?;
        crate::atomic_write(&registry.join("index.json"), &index_bytes).map_err(|message| {
            DiscoveryError::Io {
                operation: "replace Reef registry index".to_string(),
                message,
            }
        })?;
    }
    Ok(())
}

fn selected_changes(
    session: &DiscoverySession,
    resolution: &package_versioning::LocalResolution,
) -> Vec<VersionChange> {
    resolution
        .selected
        .values()
        .filter_map(|candidate| {
            let previous = session
                .locked
                .get(&candidate.id.name)
                .map(|dependency| dependency.version.clone());
            let selected = candidate.id.version.to_string();
            (previous.as_deref() != Some(selected.as_str())).then(|| VersionChange {
                package: candidate.id.name.to_string(),
                previous,
                selected,
            })
        })
        .collect()
}

pub(super) fn install_default_bootstrap(
    registry: &Path,
    runtime: &'static EmbeddedRuntime,
) -> Result<Vec<crate::InstalledArtifact>, crate::BootstrapError> {
    use chelis_reef_bootstrap_core::{BootstrapIdentity, BootstrapNode, validate_and_order};

    let mut session = DiscoverySession::for_bootstrap(runtime)?;
    let resolution = session.resolve()?;
    let packages = resolution
        .selected
        .values()
        .filter(|candidate| candidate.id.name.as_str() != crate::CHELIS_STD_PACKAGE_NAME)
        .collect::<Vec<_>>();
    let nodes = packages
        .iter()
        .map(|candidate| BootstrapNode {
            identity: BootstrapIdentity {
                name: candidate.id.name.to_string(),
                version: candidate.id.version.to_string(),
            },
            dependencies: candidate
                .dependencies
                .iter()
                .filter(|request| request.package.as_str() != crate::CHELIS_STD_PACKAGE_NAME)
                .map(|request| BootstrapIdentity {
                    name: request.package.to_string(),
                    version: resolution.selected.get(&request.package).map_or_else(
                        || request.requirement.to_string(),
                        |selected| selected.id.version.to_string(),
                    ),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let order = validate_and_order(&nodes)?;
    let mut staged = stage_selected_remote(&mut session, &resolution)?
        .into_iter()
        .map(|item| (item.package.clone(), item))
        .collect::<BTreeMap<_, _>>();
    let staged = order
        .into_iter()
        .map(|index| {
            let name = &packages[index].id.name;
            staged
                .remove(name)
                .ok_or_else(|| DiscoveryError::CandidateManifest {
                    package: name.to_string(),
                    message: "selected canonical remote artifact was not staged".to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    publish_registry_entries_at(&staged, registry, RegistryOrigins::SelectedRemote)?;
    Ok(staged
        .into_iter()
        .map(|item| {
            let root = registry
                .join("packages")
                .join(item.package.as_str())
                .join(item.version.to_string());
            crate::InstalledArtifact {
                package: PackageId {
                    name: item.package.to_string(),
                    version: item.version.to_string(),
                },
                shell_path: root.join(format!("{}-{}.chb", item.package, item.version)),
                archive_path: root.join(format!("{}-{}.tar.zst", item.package, item.version)),
                archive_sha256: item.archive_sha256,
                shell_sha256: item.shell_sha256,
            }
        })
        .collect())
}

pub fn update_project(
    root: &Path,
    target: Option<&str>,
    network_enabled: bool,
    runtime: &'static EmbeddedRuntime,
) -> Result<UpdateReport, DiscoveryError> {
    let mut session = DiscoverySession::new(
        root,
        DiscoveryMode::Refresh,
        target,
        network_enabled,
        runtime,
    )?;
    let resolution = session.resolve()?;
    let staged = stage_selected_remote(&mut session, &resolution)?;
    let lock = lock_for_resolution(&session, &resolution, &staged)?;
    let changes = selected_changes(&session, &resolution);
    publish_lock_last(
        &session,
        &lock,
        &staged,
        LockPublication::Project {
            project_lock_held: false,
        },
    )?;
    Ok(UpdateReport {
        mode: DiscoveryMode::Refresh,
        changes,
    })
}

/// Resolve the project and return its lock, published as `publication`
/// directs.
pub(crate) fn resolve_project(
    root: &Path,
    network_enabled: bool,
    publication: LockPublication,
    runtime: &'static EmbeddedRuntime,
) -> Result<ReefLock, DiscoveryError> {
    let mut session =
        DiscoverySession::new(root, DiscoveryMode::Resolve, None, network_enabled, runtime)?;
    let resolution = session.resolve()?;
    let staged = stage_selected_remote(&mut session, &resolution)?;
    let lock = lock_for_resolution(&session, &resolution, &staged)?;
    publish_lock_last(&session, &lock, &staged, publication)?;
    Ok(lock)
}

fn direct_requirement(
    session: &DiscoverySession,
    package: &PackageName,
) -> Option<PackageRequirement> {
    session
        .root_requests
        .iter()
        .find(|request| &request.package == package)
        .map(|request| request.requirement.clone())
}

pub fn outdated_project(
    root: &Path,
    target: Option<&str>,
    network_enabled: bool,
    runtime: &'static EmbeddedRuntime,
) -> Result<OutdatedReport, DiscoveryError> {
    let mut session = DiscoverySession::new(
        root,
        DiscoveryMode::Inspect,
        target,
        network_enabled,
        runtime,
    )?;
    let resolution = session.resolve()?;
    let names = if let Some(target) = &session.target {
        vec![target.clone()]
    } else {
        session
            .root_requests
            .iter()
            .map(|request| request.package.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    let mut packages = Vec::new();
    for package in names {
        let current = session
            .locked
            .get(&package)
            .map(|dependency| dependency.version.clone());
        let selected = resolution
            .selected
            .get(&package)
            .map(|candidate| candidate.id.version.clone());
        let requirement = direct_requirement(&session, &package)
            .or_else(|| selected.as_ref().map(PackageRequirement::exact));
        let mut versions = session
            .candidates
            .get(&package)
            .into_iter()
            .flatten()
            .map(|material| material.typed().package.version.clone())
            .collect::<Vec<_>>();
        versions.sort_by(|left, right| right.cmp(left));
        versions.dedup();
        let newest_requirement_match = requirement.as_ref().and_then(|requirement| {
            versions
                .iter()
                .find(|version| requirement.matches(version))
                .cloned()
        });
        let newest_compatible = selected.as_ref().map(ToString::to_string);
        let newest_incompatible = requirement.as_ref().and_then(|requirement| {
            versions
                .iter()
                .find(|version| !requirement.matches(version))
                .map(ToString::to_string)
        });
        let blocked = if let (Some(requirement_match), Some(selected)) =
            (&newest_requirement_match, &selected)
            && requirement_match != selected
        {
            Some(format!(
                "candidate {requirement_match} is blocked by transitive requirements; newest complete compatible version is {selected}"
            ))
        } else if newest_compatible.is_none() {
            let exclusions = session
                .exclusions
                .get(&package)
                .map(|entries| entries.join("; "))
                .unwrap_or_default();
            Some(if exclusions.is_empty() {
                "no compatible candidate".to_string()
            } else {
                format!("no compatible candidate; excluded provider results: {exclusions}")
            })
        } else if requirement
            .as_ref()
            .is_some_and(|requirement| requirement.as_str().starts_with('='))
        {
            Some(format!(
                "exact requirement {} permits no compatible update",
                requirement.expect("checked requirement")
            ))
        } else {
            None
        };
        packages.push(OutdatedPackage {
            package: package.to_string(),
            current,
            newest_compatible,
            newest_incompatible,
            blocked,
        });
    }
    Ok(OutdatedReport { packages })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tar::{Builder, Header};

    fn compressed_manifest(body: &[u8]) -> Vec<u8> {
        let mut tar = Builder::new(Vec::new());
        let mut header = Header::new_gnu();
        header.set_path("reef.toml").unwrap();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append(&header, Cursor::new(body)).unwrap();
        zstd::stream::encode_all(Cursor::new(tar.into_inner().unwrap()), 3).unwrap()
    }

    #[test]
    fn smaller_test_budget_stops_the_decompression_stream() {
        let directory = tempfile::tempdir().unwrap();
        let mut tar = Builder::new(Vec::new());
        let body = vec![b'x'; 1024];
        let mut header = Header::new_gnu();
        header.set_path("reef.toml").unwrap();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append(&header, Cursor::new(body)).unwrap();
        let compressed =
            zstd::stream::encode_all(Cursor::new(tar.into_inner().unwrap()), 3).unwrap();
        let path = directory.path().join("candidate.tar.zst");
        fs::write(&path, compressed).unwrap();
        let mut budget = ResolutionBudget::with_test_limits(&[
            (BudgetDimension::ScannedCandidateBytes, 64),
            (BudgetDimension::ManifestBytes, 32),
        ]);
        let error = inspect_candidate_manifest_archive(&path, "demo", &mut budget).unwrap_err();
        assert!(matches!(
            error,
            DiscoveryError::Budget(BudgetError {
                dimension: BudgetDimension::ScannedCandidateBytes,
                ..
            })
        ));
    }

    #[test]
    fn candidate_byte_limits_reset_for_each_candidate() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = format!(
            "schema = \"1\"\n[package]\nname = \"demo\"\nversion = \"1.0.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Demo\"\n",
            env!("CARGO_PKG_VERSION")
        );
        let compressed = compressed_manifest(manifest.as_bytes());
        let first = directory.path().join("first.tar.zst");
        let second = directory.path().join("second.tar.zst");
        fs::write(&first, &compressed).unwrap();
        fs::write(&second, &compressed).unwrap();
        let mut budget = ResolutionBudget::with_test_limits(&[
            (
                BudgetDimension::CompressedCandidateBytes,
                compressed.len() as u64,
            ),
            (BudgetDimension::ScannedCandidateBytes, 4096),
            (BudgetDimension::ManifestBytes, manifest.len() as u64),
        ]);
        inspect_candidate_manifest_archive(&first, "demo", &mut budget).unwrap();
        inspect_candidate_manifest_archive(&second, "demo", &mut budget).unwrap();
    }

    #[test]
    fn publication_rechecks_project_state_under_the_project_lock() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("state-recheck");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("reef.toml"),
            format!(
                "schema = \"2\"\n[package]\nname = \"state-recheck\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"State\"\nresolver = \"2\"\n",
                env!("CARGO_PKG_VERSION")
            ),
        )
        .unwrap();
        let session = DiscoverySession::new(
            &root,
            DiscoveryMode::Resolve,
            None,
            false,
            crate::tests::test_runtime(),
        )
        .unwrap();
        let replacement = ReefLock {
            package: PackageId {
                name: "state-recheck".to_string(),
                version: "0.1.0".to_string(),
            },
            dependencies: Vec::new(),
        };
        write_lockfile_unlocked(&root.join("reef.lock"), &replacement).unwrap();
        let error = publish_lock_last(
            &session,
            &replacement,
            &[],
            LockPublication::Project {
                project_lock_held: false,
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("reef.lock changed during remote discovery")
        );
    }

    #[test]
    fn provider_io_charges_request_tag_manifest_and_download_dimensions() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/repos/chelis-lang/demo/releases/assets/1"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0_u8; 10]))
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path("/repos/chelis-lang/demo/releases"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {
                        "tag_name": "v2.0.0",
                        "assets": [
                            {"id": 1, "name": "demo-2.0.0.tar.zst"},
                            {"id": 2, "name": "demo-2.0.0.chb"}
                        ]
                    },
                    {
                        "tag_name": "v1.0.0",
                        "assets": [
                            {"id": 3, "name": "demo-1.0.0.tar.zst"},
                            {"id": 4, "name": "demo-1.0.0.chb"}
                        ]
                    }
                ])))
                .mount(&server)
                .await;
            let provider = GitHubReleaseProvider {
                client: client.clone(),
                api_base: server.uri(),
                token: Some("test-token".to_string()),
            };
            let locator = SourceLocator::from_str("github://chelis-lang/demo").unwrap();
            let package = PackageName::from_str("demo").unwrap();
            let directory = tempfile::tempdir().unwrap();

            let mut compressed = ResolutionBudget::with_test_limits(&[
                (BudgetDimension::CompressedCandidateBytes, 5),
                (BudgetDimension::TotalDownloadBytes, 100),
            ]);
            let error = tokio::task::block_in_place(|| {
                provider.download_asset(
                    AssetDownload {
                        locator: &locator,
                        package: &package,
                        asset: 1,
                        destination: &directory.path().join("compressed"),
                        operation: "compressed candidate",
                        compressed_candidate: true,
                    },
                    &mut compressed,
                )
            })
            .unwrap_err();
            assert!(matches!(
                error,
                DiscoveryError::Budget(BudgetError {
                    dimension: BudgetDimension::CompressedCandidateBytes,
                    ..
                })
            ));

            let mut total =
                ResolutionBudget::with_test_limits(&[(BudgetDimension::TotalDownloadBytes, 5)]);
            let error = tokio::task::block_in_place(|| {
                provider.download_asset(
                    AssetDownload {
                        locator: &locator,
                        package: &package,
                        asset: 1,
                        destination: &directory.path().join("total"),
                        operation: "total bytes",
                        compressed_candidate: false,
                    },
                    &mut total,
                )
            })
            .unwrap_err();
            assert!(matches!(
                error,
                DiscoveryError::Budget(BudgetError {
                    dimension: BudgetDimension::TotalDownloadBytes,
                    ..
                })
            ));

            let mut requests =
                ResolutionBudget::with_test_limits(&[(BudgetDimension::HttpRequests, 0)]);
            let error = tokio::task::block_in_place(|| {
                provider.download_asset(
                    AssetDownload {
                        locator: &locator,
                        package: &package,
                        asset: 1,
                        destination: &directory.path().join("request"),
                        operation: "request limit",
                        compressed_candidate: false,
                    },
                    &mut requests,
                )
            })
            .unwrap_err();
            assert!(matches!(
                error,
                DiscoveryError::Budget(BudgetError {
                    dimension: BudgetDimension::HttpRequests,
                    ..
                })
            ));

            // Listing accepts tags without charging them; a tag is charged
            // when its release enters a candidate window.
            let mut tags =
                ResolutionBudget::with_test_limits(&[(BudgetDimension::AcceptedTags, 0)]);
            let listed = tokio::task::block_in_place(|| {
                provider.list_releases(&locator, &package, &mut tags)
            })
            .unwrap();
            assert_eq!(listed.releases.len(), 2);

            let first_order = tokio::task::block_in_place(|| {
                provider.list_releases(&locator, &package, &mut ResolutionBudget::production())
            })
            .unwrap()
            .releases;
            let reversed_server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/repos/chelis-lang/demo/releases"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {
                        "tag_name": "v1.0.0",
                        "assets": [
                            {"id": 3, "name": "demo-1.0.0.tar.zst"},
                            {"id": 4, "name": "demo-1.0.0.chb"}
                        ]
                    },
                    {
                        "tag_name": "v2.0.0",
                        "assets": [
                            {"id": 1, "name": "demo-2.0.0.tar.zst"},
                            {"id": 2, "name": "demo-2.0.0.chb"}
                        ]
                    }
                ])))
                .mount(&reversed_server)
                .await;
            let reversed_provider = GitHubReleaseProvider {
                client: client.clone(),
                api_base: reversed_server.uri(),
                token: Some("test-token".to_string()),
            };
            let second_order = tokio::task::block_in_place(|| {
                reversed_provider.list_releases(
                    &locator,
                    &package,
                    &mut ResolutionBudget::production(),
                )
            })
            .unwrap()
            .releases;
            assert_eq!(first_order, second_order);

            let mut manifests =
                ResolutionBudget::with_test_limits(&[(BudgetDimension::CandidateManifests, 0)]);
            let error = tokio::task::block_in_place(|| {
                provider.fetch_manifest(
                    &locator,
                    &package,
                    &ProviderRelease {
                        version: PackageVersion::from_str("1.0.0").unwrap(),
                        tag: "v1.0.0".to_string(),
                        archive_asset: 1,
                        shell_asset: 2,
                    },
                    directory.path(),
                    &mut manifests,
                )
            })
            .unwrap_err();
            assert!(matches!(
                error,
                DiscoveryError::Budget(BudgetError {
                    dimension: BudgetDimension::CandidateManifests,
                    ..
                })
            ));
            std::mem::forget(provider);
            std::mem::forget(reversed_provider);
            std::mem::forget(server);
            std::mem::forget(reversed_server);
        });
    }

    /// One published release of a fixture package: its tag version, the
    /// version its archive's manifest names, and that manifest's
    /// `[dependencies]` body.
    struct FixtureRelease {
        tag_version: &'static str,
        manifest_version: &'static str,
        dependencies: &'static str,
    }

    fn release(version: &'static str) -> FixtureRelease {
        FixtureRelease {
            tag_version: version,
            manifest_version: version,
            dependencies: "",
        }
    }

    fn release_with_dependencies(
        version: &'static str,
        dependencies: &'static str,
    ) -> FixtureRelease {
        FixtureRelease {
            tag_version: version,
            manifest_version: version,
            dependencies,
        }
    }

    fn fixture_manifest(name: &str, version: &str, dependencies: &str) -> String {
        format!(
            "schema = \"2\"\n[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"={}\"\nmodule_prefix = \"Fixture\"\nresolver = \"2\"\n\n[dependencies]\n{dependencies}",
            env!("CARGO_PKG_VERSION")
        )
    }

    /// A mock GitHub API with fixture packages. Each archive asset id maps
    /// back to the tag version it publishes, so a test can name exactly
    /// which candidate archives discovery downloaded.
    struct FixtureProvider {
        runtime: tokio::runtime::Runtime,
        server: wiremock::MockServer,
        archives: BTreeMap<u64, (String, String)>,
        next_asset: u64,
        token: Option<String>,
    }

    impl FixtureProvider {
        fn new() -> Self {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let server = runtime.block_on(wiremock::MockServer::start());
            Self {
                runtime,
                server,
                archives: BTreeMap::new(),
                next_asset: 1,
                token: None,
            }
        }

        /// Publish `releases` for `name`, listed in the given order and
        /// paginated `page_size` releases per page with GitHub's `link`
        /// header.
        fn publish(&mut self, name: &str, releases: &[FixtureRelease], page_size: usize) {
            use wiremock::matchers::{method, path, query_param};
            use wiremock::{Mock, ResponseTemplate};

            let mut listed = Vec::new();
            for release in releases {
                let archive = self.next_asset;
                let shell = archive + 1;
                self.next_asset += 2;
                self.archives
                    .insert(archive, (name.to_string(), release.tag_version.to_string()));
                let body = compressed_manifest(
                    fixture_manifest(name, release.manifest_version, release.dependencies)
                        .as_bytes(),
                );
                self.runtime.block_on(
                    Mock::given(method("GET"))
                        .and(path(format!(
                            "/repos/chelis-lang/{name}/releases/assets/{archive}"
                        )))
                        .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
                        .mount(&self.server),
                );
                listed.push(serde_json::json!({
                    "tag_name": format!("v{}", release.tag_version),
                    "assets": [
                        {"id": archive, "name": format!("{name}-{}.tar.zst", release.tag_version)},
                        {"id": shell, "name": format!("{name}-{}.chb", release.tag_version)}
                    ]
                }));
            }
            let pages = listed.chunks(page_size.max(1)).collect::<Vec<_>>();
            let page_count = pages.len().max(1);
            for page in 1..=page_count {
                let entries = pages.get(page - 1).map(|entries| entries.to_vec());
                let mut response = ResponseTemplate::new(200)
                    .set_body_json(serde_json::Value::Array(entries.unwrap_or_default()));
                if page < page_count {
                    response = response.insert_header(
                        "link",
                        format!(
                            "<{}/repos/chelis-lang/{name}/releases?per_page=100&page={}>; rel=\"next\"",
                            self.server.uri(),
                            page + 1
                        ),
                    );
                }
                self.runtime.block_on(
                    Mock::given(method("GET"))
                        .and(path(format!("/repos/chelis-lang/{name}/releases")))
                        .and(query_param("page", page.to_string()))
                        .respond_with(response)
                        .mount(&self.server),
                );
            }
        }

        fn provider(&self) -> GitHubReleaseProvider {
            GitHubReleaseProvider {
                client: reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(10))
                    .build()
                    .unwrap(),
                api_base: self.server.uri(),
                token: self.token.clone(),
            }
        }

        /// Answer every listing of `name` with `status` and `headers`.
        fn refuse_listing(&self, name: &str, status: u16, headers: &[(&str, &str)]) {
            use wiremock::matchers::{method, path};
            use wiremock::{Mock, ResponseTemplate};

            let mut response = ResponseTemplate::new(status);
            for (header, value) in headers {
                response = response.insert_header(*header, *value);
            }
            self.runtime.block_on(
                Mock::given(method("GET"))
                    .and(path(format!("/repos/chelis-lang/{name}/releases")))
                    .respond_with(response)
                    .mount(&self.server),
            );
        }

        fn request_paths(&self) -> Vec<String> {
            self.runtime
                .block_on(self.server.received_requests())
                .unwrap()
                .into_iter()
                .map(|request| request.url.path().to_string())
                .collect()
        }

        /// The candidate archives downloaded, as `name@version`, in request
        /// order.
        fn fetched_archives(&self) -> Vec<String> {
            self.request_paths()
                .into_iter()
                .filter_map(|path| {
                    let asset = path
                        .rsplit_once("/releases/assets/")?
                        .1
                        .parse::<u64>()
                        .ok()?;
                    let (name, version) = self.archives.get(&asset)?;
                    Some(format!("{name}@{version}"))
                })
                .collect()
        }

        fn listing_requests(&self, name: &str) -> usize {
            let listing = format!("/repos/chelis-lang/{name}/releases");
            self.request_paths()
                .into_iter()
                .filter(|path| path == &listing)
                .count()
        }
    }

    /// Resolve a fresh project whose `[dependencies]` body is
    /// `dependencies` against `fixture`, with an empty Reef home.
    fn resolve_fixture_project(
        fixture: &FixtureProvider,
        mode: DiscoveryMode,
        dependencies: &str,
    ) -> Result<BTreeMap<String, String>, DiscoveryError> {
        let _guard = crate::tests::lock_reef_home_env();
        let directory = tempfile::tempdir().unwrap();
        let previous_home = std::env::var_os("CHELIS_REEF_HOME");
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", directory.path().join("reef-home"));
        }
        let root = directory.path().join("app");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("reef.toml"),
            fixture_manifest("app", "0.1.0", dependencies),
        )
        .unwrap();
        let result = (|| {
            let mut session =
                DiscoverySession::new(&root, mode, None, true, crate::tests::test_runtime())?;
            session.provider = Some(fixture.provider());
            let resolution = session.resolve()?;
            Ok(resolution
                .selected
                .iter()
                .map(|(name, candidate)| (name.to_string(), candidate.id.version.to_string()))
                .collect())
        })();
        unsafe {
            match previous_home {
                Some(value) => std::env::set_var("CHELIS_REEF_HOME", value),
                None => std::env::remove_var("CHELIS_REEF_HOME"),
            }
        }
        result
    }

    fn selection(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(name, version)| (name.to_string(), version.to_string()))
            .collect()
    }

    fn many_releases(versions: &[&'static str]) -> Vec<FixtureRelease> {
        versions.iter().map(|version| release(version)).collect()
    }

    const DEMO_VERSIONS: [&str; 10] = [
        "1.5.0", "1.4.0", "1.3.0", "1.2.0", "1.1.0", "1.0.0", "0.3.2", "0.3.1", "0.2.0", "0.1.0",
    ];

    #[test]
    fn exact_pin_downloads_only_the_pinned_release_archive() {
        let mut fixture = FixtureProvider::new();
        fixture.publish("demo", &many_releases(&DEMO_VERSIONS), 100);
        let selected =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"=1.2.0\"\n")
                .unwrap();
        assert_eq!(selected, selection(&[("demo", "1.2.0")]));
        assert_eq!(fixture.fetched_archives(), ["demo@1.2.0"]);
    }

    #[test]
    fn exact_pin_without_a_matching_tag_downloads_no_archive() {
        let mut fixture = FixtureProvider::new();
        fixture.publish("demo", &many_releases(&DEMO_VERSIONS), 100);
        let error =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"=2.0.0\"\n")
                .unwrap_err();
        assert!(
            matches!(error, DiscoveryError::Resolution { .. }),
            "unexpected error: {error}"
        );
        assert_eq!(fixture.fetched_archives(), Vec::<String>::new());
    }

    #[test]
    fn range_downloads_only_the_newest_matching_release_archive() {
        let mut fixture = FixtureProvider::new();
        fixture.publish("demo", &many_releases(&DEMO_VERSIONS), 100);
        let selected =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^0.3\"\n").unwrap();
        assert_eq!(selected, selection(&[("demo", "0.3.2")]));
        assert_eq!(fixture.fetched_archives(), ["demo@0.3.2"]);
    }

    #[test]
    fn range_without_a_matching_tag_downloads_no_archive() {
        let mut fixture = FixtureProvider::new();
        fixture.publish("demo", &many_releases(&DEMO_VERSIONS), 100);
        let error = resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^0.4\"\n")
            .unwrap_err();
        assert!(
            matches!(error, DiscoveryError::Resolution { .. }),
            "unexpected error: {error}"
        );
        assert_eq!(fixture.fetched_archives(), Vec::<String>::new());
    }

    #[test]
    fn update_refresh_downloads_only_the_newest_matching_release_archive() {
        let mut fixture = FixtureProvider::new();
        fixture.publish("demo", &many_releases(&DEMO_VERSIONS), 100);
        let selected =
            resolve_fixture_project(&fixture, DiscoveryMode::Refresh, "demo = \"^1\"\n").unwrap();
        assert_eq!(selected, selection(&[("demo", "1.5.0")]));
        assert_eq!(fixture.fetched_archives(), ["demo@1.5.0"]);
    }

    #[test]
    fn newest_first_walks_back_past_an_unsatisfiable_release_and_stops() {
        // demo 1.5.0 needs helper ^2, which no helper tag satisfies, so the
        // resolver walks back to demo 1.4.0. Only the newest helper that
        // satisfies ^1 is downloaded, and nothing older than demo 1.4.0.
        let mut fixture = FixtureProvider::new();
        let mut releases = many_releases(&DEMO_VERSIONS);
        releases[0] = release_with_dependencies("1.5.0", "helper = \"^2\"\n");
        releases[1] = release_with_dependencies("1.4.0", "helper = \"^1\"\n");
        fixture.publish("demo", &releases, 100);
        fixture.publish("helper", &many_releases(&["1.1.0", "1.0.0", "0.9.0"]), 100);
        let selected =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n").unwrap();
        assert_eq!(
            selected,
            selection(&[("demo", "1.4.0"), ("helper", "1.1.0")])
        );
        let mut fetched = fixture.fetched_archives();
        fetched.sort();
        assert_eq!(fetched, ["demo@1.4.0", "demo@1.5.0", "helper@1.1.0"]);
    }

    #[test]
    fn walking_back_fails_closed_when_no_release_is_compatible() {
        let mut fixture = FixtureProvider::new();
        fixture.publish(
            "demo",
            &[
                release_with_dependencies("1.1.0", "helper = \"^2\"\n"),
                release_with_dependencies("1.0.0", "helper = \"^3\"\n"),
                release("0.9.0"),
            ],
            100,
        );
        fixture.publish("helper", &many_releases(&["1.0.0"]), 100);
        let error = resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
            .unwrap_err();
        assert!(
            matches!(error, DiscoveryError::Resolution { .. }),
            "unexpected error: {error}"
        );
        let mut fetched = fixture.fetched_archives();
        fetched.sort();
        assert_eq!(fetched, ["demo@1.0.0", "demo@1.1.0"]);
    }

    #[test]
    fn deferred_downloads_keep_the_backtracking_resolver_selection() {
        // The newest demo pins an older helper. Resolver 2 orders packages
        // bytewise and versions descending, so the complete-candidate
        // answer is demo 1.1.0 with helper 1.0.0, not the per-package
        // newest pair demo 1.0.0 with helper 1.5.0.
        let mut fixture = FixtureProvider::new();
        fixture.publish(
            "demo",
            &[
                release_with_dependencies("1.1.0", "helper = \"=1.0.0\"\n"),
                release_with_dependencies("1.0.0", "helper = \"^1\"\n"),
            ],
            100,
        );
        fixture.publish(
            "helper",
            &many_releases(&["1.5.0", "1.4.0", "1.0.0", "0.1.0"]),
            100,
        );
        let selected = resolve_fixture_project(
            &fixture,
            DiscoveryMode::Resolve,
            "demo = \"^1\"\nhelper = \"^1\"\n",
        )
        .unwrap();
        assert_eq!(
            selected,
            selection(&[("demo", "1.1.0"), ("helper", "1.0.0")])
        );
        let mut fetched = fixture.fetched_archives();
        fetched.sort();
        assert_eq!(fetched, ["demo@1.1.0", "helper@1.0.0"]);
    }

    #[test]
    fn shuffled_release_listings_give_the_same_selection_and_downloads() {
        let order = [
            "0.3.1", "1.2.0", "0.1.0", "1.5.0", "1.0.0", "0.3.2", "1.4.0", "0.2.0", "1.1.0",
            "1.3.0",
        ];
        let mut outcomes = Vec::new();
        for versions in [&DEMO_VERSIONS[..], &order[..]] {
            let mut fixture = FixtureProvider::new();
            let mut releases = many_releases(versions);
            for release in &mut releases {
                if release.tag_version == "1.5.0" {
                    release.dependencies = "helper = \"^2\"\n";
                }
            }
            fixture.publish("demo", &releases, 100);
            fixture.publish("helper", &many_releases(&["1.0.0"]), 100);
            let selected =
                resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
                    .unwrap();
            let mut fetched = fixture.fetched_archives();
            fetched.sort();
            outcomes.push((selected, fetched));
        }
        assert_eq!(outcomes[0], outcomes[1]);
        assert_eq!(outcomes[0].0, selection(&[("demo", "1.4.0")]));
        assert_eq!(outcomes[0].1, ["demo@1.4.0", "demo@1.5.0"]);
    }

    #[test]
    fn tag_and_manifest_version_mismatch_is_rejected_loudly() {
        // The newest tag publishes a manifest naming another version. The
        // mismatch is excluded with its reason, discovery walks back, and
        // the mismatched bytes are never selected.
        let mut fixture = FixtureProvider::new();
        let mut releases = many_releases(&DEMO_VERSIONS);
        releases[0].manifest_version = "1.4.9";
        fixture.publish("demo", &releases, 100);
        let selected =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n").unwrap();
        assert_eq!(selected, selection(&[("demo", "1.4.0")]));
        let mut fetched = fixture.fetched_archives();
        fetched.sort();
        assert_eq!(fetched, ["demo@1.4.0", "demo@1.5.0"]);
    }

    #[test]
    fn an_exact_pin_whose_manifest_disagrees_with_its_tag_fails() {
        let mut fixture = FixtureProvider::new();
        let mut releases = many_releases(&DEMO_VERSIONS);
        releases[3].manifest_version = "1.2.1";
        fixture.publish("demo", &releases, 100);
        let error =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"=1.2.0\"\n")
                .unwrap_err();
        let rendered = error.to_string();
        assert!(
            rendered.contains("release `v1.2.0` advertises `1.2.0` but its manifest names"),
            "unexpected error: {rendered}"
        );
        assert_eq!(fixture.fetched_archives(), ["demo@1.2.0"]);
    }

    #[test]
    fn paginated_listings_download_only_the_selected_release_archive() {
        // 120 non-matching releases precede the matching ones across three
        // pages of 100 entries each in GitHub order.
        let mut fixture = FixtureProvider::new();
        let old = (0..120)
            .map(|patch| -> &'static str { Box::leak(format!("0.1.{patch}").into_boxed_str()) })
            .collect::<Vec<_>>();
        let mut versions = old;
        versions.extend(["1.0.0", "1.1.0", "1.2.0"]);
        fixture.publish("demo", &many_releases(&versions), 100);
        let selected =
            resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n").unwrap();
        assert_eq!(selected, selection(&[("demo", "1.2.0")]));
        assert_eq!(fixture.listing_requests("demo"), 2);
        assert_eq!(fixture.fetched_archives(), ["demo@1.2.0"]);
    }

    #[test]
    fn paginated_listing_without_a_match_downloads_no_archive() {
        let mut fixture = FixtureProvider::new();
        let versions = (0..150)
            .map(|patch| -> &'static str { Box::leak(format!("0.1.{patch}").into_boxed_str()) })
            .collect::<Vec<_>>();
        fixture.publish("demo", &many_releases(&versions), 100);
        let error = resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
            .unwrap_err();
        assert!(
            matches!(error, DiscoveryError::Resolution { .. }),
            "unexpected error: {error}"
        );
        assert_eq!(fixture.listing_requests("demo"), 2);
        assert_eq!(fixture.fetched_archives(), Vec::<String>::new());
    }

    const EXHAUSTED: [(&str, &str); 2] = [
        ("x-ratelimit-remaining", "0"),
        ("x-ratelimit-reset", "1798761600"),
    ];

    #[test]
    fn anonymous_rate_limit_names_the_token_and_explicit_install_remedies() {
        let fixture = FixtureProvider::new();
        fixture.refuse_listing("demo", 403, &EXHAUSTED);
        let rendered = resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
            .unwrap_err()
            .to_string();
        for expected in [
            "GitHub's API rate limit for anonymous requests is exhausted until 2027-01-01 00:00:00 UTC",
            "set GITHUB_TOKEN or sign in with gh auth login",
            "`chelis reef install --from-github chelis-lang/demo@<tag>`",
        ] {
            assert!(
                rendered.contains(expected),
                "missing `{expected}` in: {rendered}"
            );
        }
        assert!(!rendered.contains("restricted repository"), "{rendered}");
    }

    #[test]
    fn secondary_rate_limits_are_not_reported_as_exhausted_quota() {
        for (status, headers, wait) in [
            (429, &[][..], "retry later"),
            (
                403,
                &[("retry-after", "30"), ("x-ratelimit-remaining", "41")][..],
                "retry after 30 seconds",
            ),
        ] {
            let fixture = FixtureProvider::new();
            fixture.refuse_listing("demo", status, headers);
            let rendered =
                resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
                    .unwrap_err()
                    .to_string();
            let expected = format!(
                "GitHub's secondary rate limit refused the request; {wait}, or install a release without discovery with `chelis reef install --from-github chelis-lang/demo@<tag>`"
            );
            assert!(rendered.contains(&expected), "{rendered}");
            for absent in [
                "anonymous",
                "GITHUB_TOKEN",
                "exhausted",
                "restricted repository",
            ] {
                assert!(!rendered.contains(absent), "`{absent}` in: {rendered}");
            }
        }
    }

    #[test]
    fn rate_limit_reset_times_render_as_utc_dates() {
        assert_eq!(utc_timestamp(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29 00:00:00 UTC");
        assert_eq!(
            utc_timestamp(1_798_761_600 + 3_661),
            "2027-01-01 01:01:01 UTC"
        );
        assert_eq!(utc_timestamp(4_107_542_399), "2100-02-28 23:59:59 UTC");
    }

    /// Resolve like [`resolve_fixture_project`] and also return the states
    /// the final search explored.
    fn resolve_fixture_with_states(
        fixture: &FixtureProvider,
        dependencies: &str,
    ) -> (Result<BTreeMap<String, String>, String>, u64) {
        let _guard = crate::tests::lock_reef_home_env();
        let directory = tempfile::tempdir().unwrap();
        let previous_home = std::env::var_os("CHELIS_REEF_HOME");
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", directory.path().join("reef-home"));
        }
        let root = directory.path().join("app");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("reef.toml"),
            fixture_manifest("app", "0.1.0", dependencies),
        )
        .unwrap();
        let mut session = DiscoverySession::new(
            &root,
            DiscoveryMode::Resolve,
            None,
            true,
            crate::tests::test_runtime(),
        )
        .unwrap();
        session.provider = Some(fixture.provider());
        let result = session
            .resolve()
            .map(|resolution| {
                resolution
                    .selected
                    .iter()
                    .map(|(name, candidate)| (name.to_string(), candidate.id.version.to_string()))
                    .collect()
            })
            .map_err(|error| error.to_string());
        let states = session.explored_states;
        unsafe {
            match previous_home {
                Some(value) => std::env::set_var("CHELIS_REEF_HOME", value),
                None => std::env::remove_var("CHELIS_REEF_HOME"),
            }
        }
        (result, states)
    }

    fn leaked_versions(count: usize, format: impl Fn(usize) -> String) -> Vec<&'static str> {
        (0..count)
            .map(|index| -> &'static str { Box::leak(format(index).into_boxed_str()) })
            .collect()
    }

    #[test]
    fn a_requirement_learned_after_a_query_widens_that_package_window() {
        // beta has 70 releases. Only the older alpha, fetched after beta is
        // queried, needs beta 1.0.0, which the newest 64 releases omit.
        let mut fixture = FixtureProvider::new();
        fixture.publish(
            "alpha",
            &[
                release_with_dependencies("1.1.0", "beta = \"=5.0.0\"\n"),
                release_with_dependencies("1.0.0", "beta = \"=1.0.0\"\n"),
            ],
            100,
        );
        let mut versions = leaked_versions(69, |minor| format!("2.{minor}.0"));
        versions.push("1.0.0");
        fixture.publish("beta", &many_releases(&versions), 100);
        let (result, _) = resolve_fixture_with_states(&fixture, "alpha = \"^1\"\n");
        assert_eq!(
            result,
            Ok(selection(&[("alpha", "1.0.0"), ("beta", "1.0.0")]))
        );
        assert_eq!(fixture.listing_requests("beta"), 1);
        let mut fetched = fixture.fetched_archives();
        fetched.sort();
        assert_eq!(fetched, ["alpha@1.0.0", "alpha@1.1.0", "beta@1.0.0"]);
    }

    #[test]
    fn a_learned_range_requirement_widens_the_window_to_its_newest_match() {
        let mut fixture = FixtureProvider::new();
        fixture.publish(
            "alpha",
            &[
                release_with_dependencies("1.1.0", "beta = \"^3\"\n"),
                release_with_dependencies("1.0.0", "beta = \"^1\"\n"),
            ],
            100,
        );
        let mut versions = leaked_versions(66, |minor| format!("2.{minor}.0"));
        versions.extend(["1.2.0", "1.1.0", "1.0.0"]);
        fixture.publish("beta", &many_releases(&versions), 100);
        let (result, _) = resolve_fixture_with_states(&fixture, "alpha = \"^1\"\n");
        assert_eq!(
            result,
            Ok(selection(&[("alpha", "1.0.0"), ("beta", "1.2.0")]))
        );
        let mut fetched = fixture.fetched_archives();
        fetched.sort();
        assert_eq!(fetched, ["alpha@1.0.0", "alpha@1.1.0", "beta@1.2.0"]);
    }

    #[test]
    fn a_learned_requirement_outside_every_listed_release_still_fails() {
        let mut fixture = FixtureProvider::new();
        fixture.publish(
            "alpha",
            &[
                release_with_dependencies("1.1.0", "beta = \"=5.0.0\"\n"),
                release_with_dependencies("1.0.0", "beta = \"=0.9.0\"\n"),
            ],
            100,
        );
        let mut versions = leaked_versions(69, |minor| format!("2.{minor}.0"));
        versions.push("1.0.0");
        fixture.publish("beta", &many_releases(&versions), 100);
        let (result, _) = resolve_fixture_with_states(&fixture, "alpha = \"^1\"\n");
        assert!(result.is_err(), "{result:?}");
        assert!(
            !fixture
                .fetched_archives()
                .iter()
                .any(|archive| archive.starts_with("beta@"))
        );
    }

    #[test]
    fn a_requirement_learned_after_a_query_sees_every_listing_page() {
        // beta's first page holds 100 2.x releases and 1.0.0 is on the
        // second. When beta is queried only alpha 1.1.0's exact `=2.5.0` is
        // known; alpha 1.1.0 then fails on delta, and alpha 1.0.0 needs
        // beta 1.0.0.
        let mut fixture = FixtureProvider::new();
        fixture.publish(
            "alpha",
            &[
                release_with_dependencies("1.1.0", "beta = \"=2.5.0\"\ndelta = \"=9.0.0\"\n"),
                release_with_dependencies("1.0.0", "beta = \"=1.0.0\"\n"),
            ],
            100,
        );
        let mut versions = leaked_versions(100, |minor| format!("2.{minor}.0"));
        versions.extend(leaked_versions(30, |minor| format!("1.{minor}.0")));
        fixture.publish("beta", &many_releases(&versions), 100);
        fixture.publish("delta", &many_releases(&["1.0.0"]), 100);
        let (result, _) = resolve_fixture_with_states(&fixture, "alpha = \"^1\"\n");
        assert_eq!(
            result,
            Ok(selection(&[("alpha", "1.0.0"), ("beta", "1.0.0")]))
        );
        assert_eq!(fixture.listing_requests("beta"), 2);
    }

    #[test]
    fn each_search_after_a_deferred_fetch_has_the_whole_state_limit() {
        // Walking back through 64 alpha releases reruns the search once per
        // fetched release. The count is per search, so the final search is
        // the size of one search over the 64 releases.
        let mut fixture = FixtureProvider::new();
        let mut releases = Vec::new();
        for minor in (0..64).rev() {
            let version: &'static str = Box::leak(format!("1.{minor}.0").into_boxed_str());
            releases.push(if minor == 0 {
                release(version)
            } else {
                release_with_dependencies(version, "beta = \"^2\"\n")
            });
        }
        fixture.publish("alpha", &releases, 100);
        fixture.publish("beta", &many_releases(&["1.0.0"]), 100);
        let (result, states) = resolve_fixture_with_states(&fixture, "alpha = \"^1\"\n");
        assert_eq!(result, Ok(selection(&[("alpha", "1.0.0")])));
        assert_eq!(fixture.fetched_archives().len(), 64);
        assert!(
            states <= 2 * 64 + 2,
            "final search explored {states} states"
        );
    }

    #[test]
    fn authenticated_rate_limit_does_not_claim_the_request_was_anonymous() {
        let mut fixture = FixtureProvider::new();
        fixture.token = Some("test-token".to_string());
        fixture.refuse_listing("demo", 403, &EXHAUSTED);
        let rendered = resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
            .unwrap_err()
            .to_string();
        assert!(
            rendered.contains("GitHub's API rate limit for the configured token is exhausted"),
            "{rendered}"
        );
        assert!(
            rendered.contains("`chelis reef install --from-github chelis-lang/demo@<tag>`"),
            "{rendered}"
        );
        assert!(!rendered.contains("anonymous"), "{rendered}");
    }

    #[test]
    fn a_forbidden_response_with_remaining_quota_is_not_a_rate_limit() {
        let fixture = FixtureProvider::new();
        fixture.refuse_listing("demo", 403, &[("x-ratelimit-remaining", "41")]);
        let rendered = resolve_fixture_project(&fixture, DiscoveryMode::Resolve, "demo = \"^1\"\n")
            .unwrap_err()
            .to_string();
        assert!(
            rendered.contains("for a restricted repository"),
            "{rendered}"
        );
        assert!(!rendered.contains("rate limit"), "{rendered}");
    }

    #[test]
    fn a_range_over_hundreds_of_releases_resolves_to_its_newest_match() {
        let mut fixture = FixtureProvider::new();
        let mut versions = leaked_versions(300, |minor| format!("1.{minor}.0"));
        versions.reverse();
        fixture.publish("beta", &many_releases(&versions), 100);
        let (result, _) = resolve_fixture_with_states(&fixture, "beta = \"^1\"\n");
        assert_eq!(result, Ok(selection(&[("beta", "1.299.0")])));
        assert_eq!(fixture.listing_requests("beta"), 3);
        assert_eq!(fixture.fetched_archives(), ["beta@1.299.0"]);
    }

    #[test]
    fn a_window_over_the_candidate_limit_fails_with_the_accepted_tag_budget() {
        // Five compatible requirements on beta whose newest 64 matches are
        // disjoint, so the window would hold 5 x 64 = 320 candidates, more
        // than the 256 limit.
        let mut fixture = FixtureProvider::new();
        let mut versions = Vec::new();
        for major in (1..=5).rev() {
            versions.extend(leaked_versions(66, |minor| {
                format!("{major}.{}.0", 65 - minor)
            }));
        }
        fixture.publish("beta", &many_releases(&versions), 100);
        let mut root = String::from("beta = \"<2.0.0\"\n");
        for (name, requirement) in [
            ("alpha", "<3.0.0"),
            ("gamma", "<4.0.0"),
            ("delta", "<5.0.0"),
            ("omega", ">=1.0.0"),
        ] {
            let dependency: &'static str =
                Box::leak(format!("beta = \"{requirement}\"\n").into_boxed_str());
            fixture.publish(name, &[release_with_dependencies("1.0.0", dependency)], 100);
            root.push_str(&format!("{name} = \"=1.0.0\"\n"));
        }
        let (result, _) = resolve_fixture_with_states(&fixture, &root);
        let error = result.unwrap_err();
        assert!(
            error.contains(
                "accepted release tags per package limit 256 reached at 257 for package `beta`"
            ),
            "{error}"
        );
    }

    /// A generated registry served by one wiremock responder: each
    /// package's releases newest first, 100 per listing page, and archives
    /// built on request. Every request path is logged.
    struct GeneratedRegistry {
        packages: BTreeMap<String, Vec<(String, String)>>,
        log: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        archives: std::sync::Mutex<BTreeMap<u64, Vec<u8>>>,
    }

    impl GeneratedRegistry {
        fn asset_id(&self, name: &str, index: usize) -> u64 {
            let package = self
                .packages
                .keys()
                .position(|candidate| candidate == name)
                .unwrap() as u64;
            (package + 1) * 1_000_000 + index as u64 * 2
        }
    }

    impl wiremock::Respond for GeneratedRegistry {
        fn respond(&self, request: &wiremock::Request) -> wiremock::ResponseTemplate {
            let path = request.url.path().to_string();
            self.log.lock().unwrap().push(path.clone());
            let parts = path.trim_start_matches('/').split('/').collect::<Vec<_>>();
            let Some(releases) = parts.get(2).and_then(|name| self.packages.get(*name)) else {
                return wiremock::ResponseTemplate::new(404);
            };
            let name = parts[2];
            match parts.as_slice() {
                ["repos", _, _, "releases"] => {
                    let page = request
                        .url
                        .query_pairs()
                        .find(|(key, _)| key == "page")
                        .and_then(|(_, value)| value.parse::<usize>().ok())
                        .unwrap_or(1);
                    let start = (page - 1) * 100;
                    let entries = releases
                        .iter()
                        .enumerate()
                        .skip(start)
                        .take(100)
                        .map(|(index, (version, _))| {
                            let archive = self.asset_id(name, index);
                            serde_json::json!({
                                "tag_name": format!("v{version}"),
                                "assets": [
                                    {"id": archive, "name": format!("{name}-{version}.tar.zst")},
                                    {"id": archive + 1, "name": format!("{name}-{version}.chb")}
                                ]
                            })
                        })
                        .collect::<Vec<_>>();
                    let mut response = wiremock::ResponseTemplate::new(200)
                        .set_body_json(serde_json::Value::Array(entries));
                    if start + 100 < releases.len() {
                        response = response.insert_header("link", "<next>; rel=\"next\"");
                    }
                    response
                }
                ["repos", _, _, "releases", "assets", asset] => {
                    let asset = asset.parse::<u64>().unwrap_or(0);
                    let Some(index) =
                        (0..releases.len()).find(|index| self.asset_id(name, *index) == asset)
                    else {
                        return wiremock::ResponseTemplate::new(404);
                    };
                    let (version, dependencies) = &releases[index];
                    let body = self
                        .archives
                        .lock()
                        .unwrap()
                        .entry(asset)
                        .or_insert_with(|| {
                            compressed_manifest(
                                fixture_manifest(name, version, dependencies).as_bytes(),
                            )
                        })
                        .clone();
                    wiremock::ResponseTemplate::new(200).set_body_bytes(body)
                }
                _ => wiremock::ResponseTemplate::new(404),
            }
        }
    }

    /// SplitMix64, so each case is a fixed function of its seed.
    struct CaseRng(u64);

    impl CaseRng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^ (value >> 31)
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }

        fn chance(&mut self, percent: usize) -> bool {
            self.below(100) < percent
        }
    }

    const ORACLE_SEED: u64 = 0x4063_0000_D15C_0FE7;
    /// Cases compared with the eager reference, which downloads every
    /// window release and so dominates the test's time.
    const ORACLE_CASES: u64 = 64;
    /// Cases given only the final-window check.
    const FINAL_WINDOW_CASES: u64 = 512;
    const RELEASE_COUNTS: [usize; 18] = [
        1, 2, 3, 10, 40, 63, 64, 65, 99, 100, 101, 120, 200, 255, 256, 257, 300, 400,
    ];

    /// Release `index` (0 is newest) of a package with `count` releases.
    /// Majors 1 to 3 split the listing into thirds, so ranges cover part of
    /// it.
    fn generated_version(count: usize, index: usize) -> String {
        let ascending = count - 1 - index;
        format!("{}.{ascending}.0", ascending * 3 / count + 1)
    }

    fn generated_requirement(rng: &mut CaseRng, count: usize) -> String {
        match rng.below(6) {
            0 | 1 => format!("={}", generated_version(count, rng.below(count))),
            2 => "=9.9.9".to_string(),
            3 => format!("^{}", rng.below(4) + 1),
            4 => {
                let low = generated_version(count, rng.below(count));
                format!(">={low}, <{}.0.0", rng.below(3) + 2)
            }
            _ => format!("^{}", rng.below(3) + 1),
        }
    }

    /// One generated case: the root's `[dependencies]` body and each
    /// package's releases, newest first, with their dependency bodies.
    fn generated_case(seed: u64) -> (String, BTreeMap<String, Vec<(String, String)>>) {
        let mut rng = CaseRng(seed);
        let names = ["pa", "pb", "pc", "pd"];
        let package_count = rng.below(4) + 1;
        let counts = (0..package_count)
            .map(|_| RELEASE_COUNTS[rng.below(RELEASE_COUNTS.len())])
            .collect::<Vec<_>>();
        let mut packages = BTreeMap::new();
        for (package, &count) in counts.iter().enumerate() {
            let mut releases = Vec::new();
            for index in 0..count {
                let mut dependencies = String::new();
                for later in package + 1..package_count {
                    if rng.chance(50) {
                        let requirement = if index < 3 && rng.chance(30) {
                            "=9.9.9".to_string()
                        } else {
                            generated_requirement(&mut rng, counts[later])
                        };
                        dependencies.push_str(&format!("{} = \"{requirement}\"\n", names[later]));
                    }
                }
                releases.push((generated_version(count, index), dependencies));
            }
            packages.insert(names[package].to_string(), releases);
        }
        let mut root = format!(
            "pa = \"{}\"\n",
            match rng.below(3) {
                0 => format!("={}", generated_version(counts[0], rng.below(counts[0]))),
                _ => format!("^{}", rng.below(3) + 1),
            }
        );
        if package_count > 1 && rng.chance(40) {
            root.push_str(&format!(
                "pb = \"{}\"\n",
                generated_requirement(&mut rng, counts[1])
            ));
        }
        (root, packages)
    }

    struct OracleOutcome {
        result: Result<BTreeMap<String, String>, String>,
        candidates: BTreeSet<(String, String)>,
        known: BTreeMap<String, Vec<PackageRequirement>>,
        downloaded: Vec<(String, u64)>,
        archives: usize,
        listings: usize,
    }

    /// The requirements in the root and in every archive a run downloaded,
    /// per required package, recomputed from the generated manifests.
    fn requirements_read(
        root: &str,
        packages: &BTreeMap<String, Vec<(String, String)>>,
        downloaded: &[(String, u64)],
    ) -> BTreeMap<String, BTreeSet<String>> {
        let directory = tempfile::tempdir().unwrap();
        let mut manifests = vec![("app".to_string(), "0.1.0".to_string(), root.to_string())];
        for (name, asset) in downloaded {
            let position = packages.keys().position(|key| key == name).unwrap() as u64;
            let index = ((asset - (position + 1) * 1_000_000) / 2) as usize;
            let (version, dependencies) = &packages[name][index];
            manifests.push((name.clone(), version.clone(), dependencies.clone()));
        }
        let mut read = BTreeMap::<String, BTreeSet<String>>::new();
        for (name, version, dependencies) in manifests {
            let path = directory.path().join(format!("{name}-{version}.toml"));
            fs::write(&path, fixture_manifest(&name, &version, &dependencies)).unwrap();
            let typed = crate::read_manifest(&path).unwrap().typed;
            for request in registry_requests(&PackageName::from_str(&name).unwrap(), &typed) {
                read.entry(request.package.to_string())
                    .or_default()
                    .insert(request.requirement.as_str().to_string());
            }
        }
        read
    }

    fn registry_requests(requester: &PackageName, typed: &TypedManifest) -> Vec<RequestedPackage> {
        typed
            .dependencies
            .iter()
            .filter_map(|(name, dependency)| match dependency {
                TypedDependency::Registry { requirement } => Some(RequestedPackage::new(
                    requester.clone(),
                    name.clone(),
                    requirement.as_package_requirement(),
                )),
                TypedDependency::Path { .. } => None,
            })
            .collect()
    }

    /// The spec's statement of the selection, computed directly: the
    /// window for the final known requirements, every candidate in it with
    /// its real manifest, and one resolver-2 search. It shares no window
    /// or deferral code with discovery.
    fn resolve_over_final_window(
        root: &str,
        packages: &BTreeMap<String, Vec<(String, String)>>,
        known: &BTreeMap<String, Vec<PackageRequirement>>,
    ) -> Result<BTreeMap<String, String>, String> {
        let directory = tempfile::tempdir().unwrap();
        let read = |name: &str, version: &str, dependencies: &str| {
            let path = directory.path().join(format!("{name}-{version}.toml"));
            fs::write(&path, fixture_manifest(name, version, dependencies)).unwrap();
            crate::read_manifest(&path).unwrap().typed
        };
        let app = PackageName::from_str("app").unwrap();
        let root_requests = registry_requests(&app, &read("app", "0.1.0", root));
        let mut candidates = BTreeMap::new();
        for (name, releases) in packages {
            let package = PackageName::from_str(name).unwrap();
            let mut window = BTreeSet::new();
            for requirement in known.get(name).into_iter().flatten() {
                window.extend(
                    releases
                        .iter()
                        .enumerate()
                        .filter(|(_, (version, _))| {
                            requirement.matches(&PackageVersion::from_str(version).unwrap())
                        })
                        .take(MAX_CANDIDATES_PER_REQUIREMENT)
                        .map(|(position, _)| position),
                );
            }
            let entries = window
                .into_iter()
                .map(|position| {
                    let (version, dependencies) = &releases[position];
                    let typed = read(name, version, dependencies);
                    package_versioning::LocalCandidate {
                        id: typed.package.clone(),
                        source: CandidateSource::Remote {
                            source_identity: format!("github://chelis-lang/{name}@v{version}"),
                        },
                        content_identity: format!("{name}@{version}"),
                        dependencies: registry_requests(&package, &typed),
                    }
                })
                .collect::<Vec<_>>();
            if !entries.is_empty() {
                candidates.insert(package, entries);
            }
        }
        let mut states = 0;
        package_versioning::resolve_local_with_state_counter(
            &root_requests,
            &candidates,
            package_versioning::ResolverLimits::default(),
            &mut states,
        )
        .map(|resolution| {
            resolution
                .selected
                .iter()
                .map(|(name, candidate)| (name.to_string(), candidate.id.version.to_string()))
                .collect()
        })
        .map_err(|error| error.to_string())
    }

    fn run_generated(
        client: &reqwest::blocking::Client,
        server: &wiremock::MockServer,
        log: &std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        root: &str,
        eager_reference: bool,
    ) -> OracleOutcome {
        let directory = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", directory.path().join("reef-home"));
        }
        let project = directory.path().join("app");
        fs::create_dir_all(&project).unwrap();
        fs::write(
            project.join("reef.toml"),
            fixture_manifest("app", "0.1.0", root),
        )
        .unwrap();
        log.lock().unwrap().clear();
        let mut session = DiscoverySession::new(
            &project,
            DiscoveryMode::Resolve,
            None,
            true,
            crate::tests::test_runtime(),
        )
        .unwrap();
        session.eager_reference = eager_reference;
        session.provider = Some(GitHubReleaseProvider {
            client: client.clone(),
            api_base: server.uri(),
            token: None,
        });
        let result = session
            .resolve()
            .map(|resolution| {
                resolution
                    .selected
                    .iter()
                    .map(|(name, candidate)| (name.to_string(), candidate.id.version.to_string()))
                    .collect()
            })
            .map_err(|error| error.to_string());
        let candidates = session
            .candidates
            .iter()
            .flat_map(|(name, materials)| {
                materials.iter().map(move |material| {
                    (
                        name.to_string(),
                        material.typed().package.version.to_string(),
                    )
                })
            })
            .collect();
        let requests = log.lock().unwrap().clone();
        let downloaded = requests
            .iter()
            .filter_map(|path| {
                let (prefix, asset) = path.rsplit_once("/releases/assets/")?;
                Some((
                    prefix.rsplit('/').next()?.to_string(),
                    asset.parse::<u64>().ok()?,
                ))
            })
            .collect();
        let known = session
            .listings
            .keys()
            .map(|name| (name.to_string(), session.known_requirements(name)))
            .collect();
        OracleOutcome {
            result,
            candidates,
            known,
            downloaded,
            archives: requests
                .iter()
                .filter(|path| path.contains("/releases/assets/"))
                .count(),
            listings: requests
                .iter()
                .filter(|path| path.ends_with("/releases"))
                .count(),
        }
    }

    /// Differential oracle on generated graphs, with two checks. First,
    /// deferred discovery must select exactly what one resolver-2 search
    /// selects over the window for its final known requirements with every
    /// candidate's manifest read, or both must fail. Those requirements are
    /// the deferred run's, as the window rule defines them over the
    /// manifests a run reads, and must equal the requirements recomputed
    /// from the root and the archives the run downloaded. Second, as
    /// regression and download-count evidence against the eager reference,
    /// rather than a proof of equivalence, a case passes when the reference
    /// fails, when
    /// both select the same graph, or when deferred discovery selects, for
    /// some package, a release the reference never had as a candidate for
    /// that package: its window dropped the release, so it reached a later
    /// resolver-2 solution. Deferred discovery failing where the reference
    /// succeeds, or selecting differently while the reference had every
    /// selected release, fails.
    #[test]
    fn deferred_discovery_agrees_with_eager_discovery_on_generated_graphs() {
        let _guard = crate::tests::lock_reef_home_env();
        let previous_home = std::env::var_os("CHELIS_REEF_HOME");
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(wiremock::MockServer::start());
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let mut tally = BTreeMap::<&str, u64>::new();
        let mut spec_tally = BTreeMap::<&str, u64>::new();
        let (mut deferred_archives, mut eager_archives) = (0, 0);
        let (mut deferred_time, mut eager_time) = (Duration::ZERO, Duration::ZERO);
        let mut failures = Vec::new();
        for case in 0..FINAL_WINDOW_CASES {
            let seed = ORACLE_SEED.wrapping_add(case);
            let (root, packages) = generated_case(seed);
            let shape = packages
                .iter()
                .map(|(name, releases)| format!("{name}:{}", releases.len()))
                .collect::<Vec<_>>()
                .join(",");
            runtime.block_on(server.reset());
            runtime.block_on(
                wiremock::Mock::given(wiremock::matchers::method("GET"))
                    .respond_with(GeneratedRegistry {
                        packages: packages.clone(),
                        log: log.clone(),
                        archives: std::sync::Mutex::new(BTreeMap::new()),
                    })
                    .mount(&server),
            );
            let started = Instant::now();
            let deferred = run_generated(&client, &server, &log, &root, false);
            deferred_time += started.elapsed();
            deferred_archives += deferred.archives;
            // Check 1: the selection over the final window with every
            // candidate's manifest read. This needs no reference
            // implementation, so it cannot share a reference's defect. The
            // known requirements are the deferred run's, and are checked
            // against the root and the archives that run downloaded.
            let read = requirements_read(&root, &packages, &deferred.downloaded);
            let known_matches = deferred.known.iter().all(|(name, requirements)| {
                requirements
                    .iter()
                    .map(|requirement| requirement.as_str().to_string())
                    .collect::<BTreeSet<_>>()
                    == read.get(name).cloned().unwrap_or_default()
            });
            let direct = resolve_over_final_window(&root, &packages, &deferred.known);
            let agrees = known_matches
                && match (&direct, &deferred.result) {
                    (Ok(left), Ok(right)) => left == right,
                    (Err(_), Err(_)) => true,
                    _ => false,
                };
            *spec_tally
                .entry(if agrees { "agrees" } else { "FAIL" })
                .or_default() += 1;
            if !agrees {
                failures.push(format!(
                    "case {case} seed {seed:#x} packages [{shape}] root {root:?}: final-window search {direct:?}; deferred {:?} ({} archives); known requirements match the manifests read: {known_matches}",
                    deferred.result, deferred.archives
                ));
            }
            if case >= ORACLE_CASES {
                continue;
            }
            let started = Instant::now();
            let eager = run_generated(&client, &server, &log, &root, true);
            eager_time += started.elapsed();
            eager_archives += eager.archives;
            let class = match (&eager.result, &deferred.result) {
                (Err(_), Err(_)) => "both fail",
                (Err(_), Ok(_)) => "eager fails, deferred resolves",
                (Ok(left), Ok(right)) if left == right => "same selection",
                (Ok(_), Ok(right))
                    if right.iter().any(|(name, version)| {
                        !eager.candidates.contains(&(name.clone(), version.clone()))
                    }) =>
                {
                    "deferred selects a release outside the eager window"
                }
                _ => {
                    failures.push(format!(
                        "case {case} seed {seed:#x} packages [{shape}] root {root:?}: eager {:?} ({} archives, {} listings); deferred {:?} ({} archives, {} listings)",
                        eager.result,
                        eager.archives,
                        eager.listings,
                        deferred.result,
                        deferred.archives,
                        deferred.listings
                    ));
                    "FAIL"
                }
            };
            *tally.entry(class).or_default() += 1;
        }
        unsafe {
            match previous_home {
                Some(value) => std::env::set_var("CHELIS_REEF_HOME", value),
                None => std::env::remove_var("CHELIS_REEF_HOME"),
            }
        }
        eprintln!(
            "oracle seed {ORACLE_SEED:#x}: final window over {FINAL_WINDOW_CASES} cases {spec_tally:?}; against eager over {ORACLE_CASES} cases {tally:?}; archives deferred {deferred_archives}, eager {eager_archives}; time deferred {deferred_time:?}, eager {eager_time:?}"
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
