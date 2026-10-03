//! Bounded remote candidate discovery for Reef resolver 2.

use crate::package_versioning::{
    self, CandidateSource, LocalCandidate, PackageName, PackageRequirement, PackageVersion,
    RequestedPackage, TypedDependency, TypedManifest,
};
use crate::{
    CANONICAL_REEF_ORG, EmbeddedRuntime, LockSource, LockedDependency, PackageId, ReefLock,
    ReefManifest, RegistryVersion, acquire_reef_home_lock, document_schema, parse_remote_origin,
    read_lockfile, read_registry_index, registry_root, resolve_github_token, sha256_file,
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
pub const MAX_CANDIDATE_MANIFESTS_PER_PACKAGE: u64 = 64;
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

#[derive(Debug)]
struct ProviderListing {
    releases: Vec<ProviderRelease>,
    exclusions: Vec<String>,
}

#[derive(Debug, Clone)]
struct RemoteMaterial {
    typed: TypedManifest,
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
        dependencies: Vec<RequestedPackage>,
    },
    Local {
        entry: RegistryVersion,
        typed: TypedManifest,
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
        requirements: &[PackageRequirement],
        stop_after_matches: Option<u64>,
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

struct GitHubReleaseProvider {
    client: reqwest::blocking::Client,
    api_base: String,
    token: String,
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
                message: error.to_string(),
            })?;
        Ok(Self {
            client,
            api_base: std::env::var("CHELIS_REEF_GITHUB_BASE_API")
                .unwrap_or_else(|_| "https://api.github.com".to_string()),
            token: resolve_github_token().map_err(|error| DiscoveryError::RemoteUnavailable {
                package: "<provider>".to_string(),
                reason: error.to_string(),
            })?,
        })
    }

    fn get(
        &self,
        url: &str,
        accept: &str,
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
        let request = self
            .client
            .get(url)
            .header("Accept", accept)
            .header("Authorization", format!("token {}", self.token));
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
        if !response.status().is_success() {
            return Err(DiscoveryError::RemoteUnavailable {
                package: package.to_string(),
                reason: format!("{operation} at {url} returned HTTP {}", response.status()),
            });
        }
        Ok((response, started))
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
        let (mut response, started) =
            self.get(&url, "application/octet-stream", package, operation, budget)?;
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
        requirements: &[PackageRequirement],
        stop_after_matches: Option<u64>,
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
                budget.charge(
                    BudgetDimension::AcceptedTags,
                    1,
                    package.as_str(),
                    "accept GitHub release tag",
                )?;
                releases.push(ProviderRelease {
                    version,
                    tag: release.tag_name,
                    archive_asset,
                    shell_asset,
                });
            }
            let enough_matches = stop_after_matches.is_some_and(|required| {
                releases
                    .iter()
                    .filter(|release| {
                        requirements.is_empty()
                            || requirements
                                .iter()
                                .any(|requirement| requirement.matches(&release.version))
                    })
                    .count() as u64
                    >= required
            });
            if enough_matches || !has_next || page_len < 100 {
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
        let (_manifest, typed) =
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

struct DiscoverySession {
    root: PathBuf,
    mode: DiscoveryMode,
    target: Option<PackageName>,
    network_enabled: bool,
    root_manifest: crate::ParsedManifest,
    root_requests: Vec<RequestedPackage>,
    candidates: BTreeMap<PackageName, Vec<CandidateMaterial>>,
    path_stack: Vec<PackageName>,
    pending_local: BTreeSet<PackageName>,
    loaded_local: BTreeSet<PackageName>,
    queried_remote: BTreeSet<PackageName>,
    refresh_dependencies: BTreeSet<PackageName>,
    exclusions: BTreeMap<PackageName, Vec<String>>,
    initial_lock: Option<ReefLock>,
    locked: BTreeMap<PackageName, LockedDependency>,
    budget: ResolutionBudget,
    explored_states: u64,
    temp: tempfile::TempDir,
    provider: Option<GitHubReleaseProvider>,
    runtime: &'static EmbeddedRuntime,
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
        let locked = initial_lock
            .as_ref()
            .map(|lock| {
                lock.dependencies
                    .iter()
                    .cloned()
                    .filter_map(|dependency| {
                        PackageName::from_str(&dependency.name)
                            .ok()
                            .map(|name| (name, dependency))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let temp = tempfile::tempdir().map_err(|error| DiscoveryError::Io {
            operation: "create remote discovery temporary directory".to_string(),
            message: error.to_string(),
        })?;
        let mut session = Self {
            root,
            mode,
            target,
            network_enabled,
            root_manifest,
            root_requests: Vec::new(),
            candidates: BTreeMap::new(),
            path_stack: Vec::new(),
            pending_local: BTreeSet::new(),
            loaded_local: BTreeSet::new(),
            queried_remote: BTreeSet::new(),
            refresh_dependencies: BTreeSet::new(),
            exclusions: BTreeMap::new(),
            initial_lock,
            locked,
            budget: ResolutionBudget::production(),
            explored_states: 0,
            temp,
            provider: None,
            runtime,
        };
        let root_name = session.root_manifest.typed.package.name.clone();
        let root_path = session.root.clone();
        let root_typed = session.root_manifest.typed.clone();
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
                        .strip_prefix(&self.root)
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

    fn load_local_closure(&mut self) -> Result<(), DiscoveryError> {
        while let Some(package) = self
            .pending_local
            .iter()
            .find(|package| !self.loaded_local.contains(*package))
            .cloned()
        {
            self.loaded_local.insert(package.clone());
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
                let installed = match crate::load_registry_package(package.as_str(), &entry.version)
                {
                    Ok(installed) => installed,
                    Err(crate::LoadRegistryError::MissingPackageDir)
                    | Err(crate::LoadRegistryError::MissingFromIndex) => continue,
                    Err(crate::LoadRegistryError::Other(message)) => {
                        return Err(DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message: format!("load local registry candidate: {message}"),
                        });
                    }
                };
                let parsed =
                    crate::read_manifest(&installed.root.join("reef.toml")).map_err(|message| {
                        DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message,
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
                let dependencies =
                    self.requests_for_manifest(&package, Some(&installed.root), &parsed.typed)?;
                self.insert_material(
                    package.clone(),
                    CandidateMaterial::Local {
                        entry,
                        typed: parsed.typed,
                        dependencies,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn source_locator(&self, package: &PackageName) -> Result<SourceLocator, DiscoveryError> {
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
        if !self.network_enabled {
            return Err(DiscoveryError::RemoteUnavailable {
                package: package.to_string(),
                reason: "network access is disabled".to_string(),
            });
        }
        let locator = self.source_locator(package)?;
        let known_requirements = self.known_requirements(package);
        let temp_root = self.temp.path().to_path_buf();
        let provider = self
            .provider
            .take()
            .unwrap_or(GitHubReleaseProvider::new()?);
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
            let stop_after_matches = (self.mode == DiscoveryMode::Resolve && missing_pin.is_none())
                .then(|| {
                    if !known_requirements.is_empty()
                        && known_requirements
                            .iter()
                            .all(|requirement| requirement.as_str().starts_with('='))
                    {
                        1
                    } else {
                        self.budget.limit(BudgetDimension::CandidateManifests)
                    }
                });
            let mut listing = provider.list_releases(
                &locator,
                package,
                &known_requirements,
                stop_after_matches,
                &mut self.budget,
            )?;
            listing.releases.sort_by(|left, right| {
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
            let pinned_release = if let Some(version) = &missing_pin {
                let position = listing
                    .releases
                    .iter()
                    .position(|release| &release.version == version)
                    .ok_or_else(|| DiscoveryError::RemoteUnavailable {
                        package: package.to_string(),
                        reason: format!(
                            "locked release `{version}` is unavailable within the bounded GitHub release listing"
                        ),
                    })?;
                Some(listing.releases.remove(position))
            } else {
                None
            };
            self.exclusions
                .entry(package.clone())
                .or_default()
                .extend(listing.exclusions);
            for release in pinned_release
                .into_iter()
                .chain(listing.releases)
                .take(self.budget.limit(BudgetDimension::CandidateManifests) as usize)
            {
                let mut material = match provider.fetch_manifest(
                    &locator,
                    package,
                    &release,
                    &temp_root,
                    &mut self.budget,
                ) {
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
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                if let Some(locked) = self.locked.get(package)
                    && locked.version == release.version.to_string()
                    && matches!(&locked.source, LockSource::LocalRegistry { .. })
                {
                    crate::verify_locked_hashes(locked, Some(&material.archive_sha256), None)
                        .map_err(|message| DiscoveryError::CandidateManifest {
                            package: package.to_string(),
                            message,
                        })?;
                    // A verified installed pin already supplies its shell.
                    // An uncached pin must verify both downloaded assets.
                    if missing_pin.as_ref() == Some(&release.version) {
                        let shell_path = provider.fetch_selected_shell(
                            &material,
                            &temp_root,
                            &mut self.budget,
                        )?;
                        let verified = verify_artifact_pair(&material.archive_path, &shell_path)
                            .map_err(|message| DiscoveryError::CandidateManifest {
                                package: package.to_string(),
                                message,
                            })?;
                        crate::verify_locked_hashes(
                            locked,
                            Some(&verified.archive_sha256),
                            Some(&verified.shell_sha256),
                        )
                        .map_err(|message| {
                            DiscoveryError::CandidateManifest {
                                package: package.to_string(),
                                message,
                            }
                        })?;
                        if verified.package.name != package.as_str()
                            || verified.package.version != release.version.to_string()
                            || verified.compiler != material.typed.compiler.to_string()
                        {
                            return Err(DiscoveryError::CandidateManifest {
                                package: package.to_string(),
                                message:
                                    "locked remote archive, shell, and manifest identities disagree"
                                        .to_string(),
                            });
                        }
                        material.verified_shell_path = Some(shell_path);
                    }
                }
                let typed = material.typed.clone();
                if let Some((dependency, path)) = typed.dependencies.iter().find_map(
                    |(dependency, declaration)| match declaration {
                        TypedDependency::Path { path, .. } => Some((dependency, path)),
                        TypedDependency::Registry { .. } => None,
                    },
                ) {
                    self.exclusions
                        .entry(package.clone())
                        .or_default()
                        .push(format!(
                            "{}: remote candidate `{}` contains path dependency `{dependency}` at `{path}`",
                            release.tag, typed.package
                        ));
                    continue;
                }
                material.dependencies =
                    self.requests_for_manifest(package, Some(&temp_root), &typed)?;
                self.refresh_dependencies.extend(
                    material
                        .dependencies
                        .iter()
                        .map(|request| request.package.clone()),
                );
                self.insert_material(package.clone(), CandidateMaterial::Remote(material))?;
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
                    self.query_remote(&target)?;
                    while let Some(dependency) = self
                        .refresh_dependencies
                        .iter()
                        .find(|package| {
                            package.as_str() != crate::CHELIS_STD_PACKAGE_NAME
                                && !self.queried_remote.contains(*package)
                        })
                        .cloned()
                    {
                        self.query_remote(&dependency)?;
                    }
                }
            }
            DiscoveryMode::Resolve | DiscoveryMode::Locked => {}
        }
        loop {
            match self.resolve_once() {
                Ok(resolution) => return Ok(resolution),
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
            compiler: session.root_manifest.typed.compiler.to_string(),
            archive_sha256: session.runtime.archive_sha256().to_string(),
            shell_sha256: session.runtime.shell_sha256().to_string(),
        });
    }
    for dependency in session.locked.values() {
        if matches!(dependency.source, LockSource::Binary { .. })
            && !dependencies.iter().any(|item| item.name == dependency.name)
        {
            dependencies.push(dependency.clone());
        }
    }
    dependencies.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(ReefLock {
        package: PackageId {
            name: session.root_manifest.typed.package.name.to_string(),
            version: session.root_manifest.typed.package.version.to_string(),
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
    let _project_lock = if project_lock_held {
        None
    } else {
        Some(
            document_schema::acquire_project_write_lock(&session.root).map_err(|error| {
                DiscoveryError::Io {
                    operation: "acquire package-root project lock".to_string(),
                    message: error.to_string(),
                }
            })?,
        )
    };
    let current_manifest =
        crate::read_manifest(&session.root.join("reef.toml")).map_err(|message| {
            DiscoveryError::CandidateManifest {
                package: session.root_manifest.typed.package.name.to_string(),
                message,
            }
        })?;
    if current_manifest.raw != session.root_manifest.raw {
        return Err(DiscoveryError::Io {
            operation: "recheck project manifest under the project lock".to_string(),
            message: "reef.toml changed during remote discovery".to_string(),
        });
    }
    let current_lock = if session.root.join("reef.lock").exists() {
        Some(
            read_lockfile(&session.root.join("reef.lock"))
                .map_err(|message| DiscoveryError::Io {
                    operation: "recheck project lock under the project lock".to_string(),
                    message,
                })?
                .raw,
        )
    } else {
        None
    };
    if current_lock != session.initial_lock {
        return Err(DiscoveryError::Io {
            operation: "recheck project lock under the project lock".to_string(),
            message: "reef.lock changed during remote discovery".to_string(),
        });
    }
    document_schema::ensure_project_lock_ignore(&session.root).map_err(|message| {
        DiscoveryError::Io {
            operation: "update project ignore rules".to_string(),
            message,
        }
    })?;
    publish_registry_entries(staged)?;
    write_lockfile_unlocked(&session.root.join("reef.lock"), lock).map_err(|message| {
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
    let _registry_lock =
        acquire_reef_home_lock(&registry, MAX_HTTP_REQUEST_TIME).map_err(|error| {
            DiscoveryError::Io {
                operation: "acquire Reef registry lock".to_string(),
                message: error.to_string(),
            }
        })?;
    let mut index = read_registry_index(&registry).map_err(|message| DiscoveryError::Io {
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
            .iter()
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
                token: "test-token".to_string(),
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

            let mut tags =
                ResolutionBudget::with_test_limits(&[(BudgetDimension::AcceptedTags, 1)]);
            let error = tokio::task::block_in_place(|| {
                provider.list_releases(&locator, &package, &[], None, &mut tags)
            })
            .unwrap_err();
            assert!(matches!(
                error,
                DiscoveryError::Budget(BudgetError {
                    dimension: BudgetDimension::AcceptedTags,
                    ..
                })
            ));

            let first_order = tokio::task::block_in_place(|| {
                provider.list_releases(
                    &locator,
                    &package,
                    &[],
                    None,
                    &mut ResolutionBudget::production(),
                )
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
                token: "test-token".to_string(),
            };
            let second_order = tokio::task::block_in_place(|| {
                reversed_provider.list_releases(
                    &locator,
                    &package,
                    &[],
                    None,
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
}
