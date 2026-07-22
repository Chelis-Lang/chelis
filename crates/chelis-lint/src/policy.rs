//! Deterministic traversal policy for `chelis lint`.
//!
//! Policy comes only from the shipped baseline and the nearest ancestor
//! `chelis-lint.toml`. Git, hidden-file, parent, and machine-local ignore
//! sources are deliberately not part of this contract.

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use serde::Deserialize;
use std::fmt;
use std::path::{Component, Path, PathBuf};

const POLICY_FILE: &str = "chelis-lint.toml";
const SHIPPED_POLICY_PATH: &str = "crates/chelis-lint/default_policy.toml";
const SHIPPED_POLICY: &str = include_str!("../default_policy.toml");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraversalClass {
    Infrastructure,
    Build,
    Dependency,
    Generated,
    Immutable,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraversalExclusion {
    pub pattern: String,
    pub class: TraversalClass,
    pub cross_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyDocument {
    version: u32,
    spec: Option<PathBuf>,
    #[serde(default, rename = "exclude")]
    exclusions: Vec<TraversalExclusion>,
}

#[derive(Debug)]
struct CompiledExclusion {
    exclusion: TraversalExclusion,
    matcher: Gitignore,
}

#[derive(Debug)]
pub struct TraversalPolicy {
    repository_root: Option<PathBuf>,
    exclusions: Vec<CompiledExclusion>,
}

impl TraversalPolicy {
    pub fn load_for(target: &Path) -> Result<Self, TraversalPolicyError> {
        let start = policy_search_start(target);
        let repository_policy = start
            .ancestors()
            .map(|ancestor| ancestor.join(POLICY_FILE))
            .find(|candidate| candidate.is_file());
        let repository_root = repository_policy
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let baseline_root = repository_root.as_deref().unwrap_or(start);

        let shipped = parse_policy(SHIPPED_POLICY, Path::new(SHIPPED_POLICY_PATH), false)?;
        let mut exclusions = compile_exclusions(
            baseline_root,
            Path::new(SHIPPED_POLICY_PATH),
            shipped.exclusions,
        )?;

        if let (Some(policy_path), Some(policy_root)) =
            (repository_policy.as_deref(), repository_root.as_deref())
        {
            let source = std::fs::read_to_string(policy_path).map_err(|source| {
                TraversalPolicyError::Io {
                    path: policy_path.to_path_buf(),
                    source,
                }
            })?;
            let document = parse_policy(&source, policy_path, true)?;
            let spec = document.spec.as_ref().expect("required by parse_policy");
            let spec_path = policy_root.join(spec);
            let spec_source =
                std::fs::read_to_string(&spec_path).map_err(|source| TraversalPolicyError::Io {
                    path: spec_path.clone(),
                    source,
                })?;
            verify_cross_refs(policy_path, &document.exclusions, &spec_source)?;
            exclusions.extend(compile_exclusions(
                policy_root,
                policy_path,
                document.exclusions,
            )?);
        }

        Ok(Self {
            repository_root,
            exclusions,
        })
    }

    pub fn repository_root(&self) -> Option<&Path> {
        self.repository_root.as_deref()
    }

    pub fn exclusion_for(&self, path: &Path, is_dir: bool) -> Option<&TraversalExclusion> {
        self.exclusions
            .iter()
            .find(|compiled| compiled.matcher.matched(path, is_dir).is_ignore())
            .map(|compiled| &compiled.exclusion)
    }

    pub fn shipped_exclusions() -> Result<Vec<TraversalExclusion>, TraversalPolicyError> {
        Ok(parse_policy(SHIPPED_POLICY, Path::new(SHIPPED_POLICY_PATH), false)?.exclusions)
    }

    pub fn verify_shipped_cross_refs(spec_source: &str) -> Result<(), TraversalPolicyError> {
        let exclusions = Self::shipped_exclusions()?;
        verify_cross_refs(Path::new(SHIPPED_POLICY_PATH), &exclusions, spec_source)
    }
}

fn policy_search_start(target: &Path) -> &Path {
    if target.is_dir() {
        target
    } else {
        target.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn parse_policy(
    source: &str,
    path: &Path,
    require_spec: bool,
) -> Result<PolicyDocument, TraversalPolicyError> {
    let document: PolicyDocument =
        toml::from_str(source).map_err(|source| TraversalPolicyError::Toml {
            path: path.to_path_buf(),
            source,
        })?;
    if document.version != 1 {
        return Err(TraversalPolicyError::UnsupportedVersion {
            path: path.to_path_buf(),
            version: document.version,
        });
    }
    if require_spec && document.spec.is_none() {
        return Err(TraversalPolicyError::MissingSpec {
            path: path.to_path_buf(),
        });
    }
    if let Some(spec) = &document.spec
        && spec.components().any(|component| {
            matches!(
                component,
                Component::Prefix(_) | Component::RootDir | Component::ParentDir
            )
        })
    {
        return Err(TraversalPolicyError::InvalidSpecPath {
            path: path.to_path_buf(),
            spec: spec.clone(),
        });
    }
    Ok(document)
}

fn compile_exclusions(
    root: &Path,
    policy_path: &Path,
    exclusions: Vec<TraversalExclusion>,
) -> Result<Vec<CompiledExclusion>, TraversalPolicyError> {
    exclusions
        .into_iter()
        .map(|exclusion| {
            let trimmed = exclusion.pattern.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('!') {
                return Err(TraversalPolicyError::PatternSyntax {
                    path: policy_path.to_path_buf(),
                    pattern: exclusion.pattern.clone(),
                    reason: "exclusions must be non-empty ignore patterns, not comments or negations",
                });
            }
            let mut builder = GitignoreBuilder::new(root);
            builder.allow_unclosed_class(false);
            builder
                .add_line(Some(policy_path.to_path_buf()), &exclusion.pattern)
                .map_err(|source| TraversalPolicyError::Pattern {
                    path: policy_path.to_path_buf(),
                    pattern: exclusion.pattern.clone(),
                    source,
                })?;
            let matcher = builder
                .build()
                .map_err(|source| TraversalPolicyError::Pattern {
                    path: policy_path.to_path_buf(),
                    pattern: exclusion.pattern.clone(),
                    source,
                })?;
            Ok(CompiledExclusion { exclusion, matcher })
        })
        .collect()
}

fn verify_cross_refs(
    policy_path: &Path,
    exclusions: &[TraversalExclusion],
    spec_source: &str,
) -> Result<(), TraversalPolicyError> {
    let known = crate::exceptions::section_ids_from_spec(spec_source);
    for exclusion in exclusions {
        if !known.contains(&exclusion.cross_ref) {
            return Err(TraversalPolicyError::UnresolvedCrossRef {
                path: policy_path.to_path_buf(),
                pattern: exclusion.pattern.clone(),
                cross_ref: exclusion.cross_ref.clone(),
            });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum TraversalPolicyError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Toml {
        path: PathBuf,
        source: toml::de::Error,
    },
    UnsupportedVersion {
        path: PathBuf,
        version: u32,
    },
    MissingSpec {
        path: PathBuf,
    },
    InvalidSpecPath {
        path: PathBuf,
        spec: PathBuf,
    },
    Pattern {
        path: PathBuf,
        pattern: String,
        source: ignore::Error,
    },
    PatternSyntax {
        path: PathBuf,
        pattern: String,
        reason: &'static str,
    },
    UnresolvedCrossRef {
        path: PathBuf,
        pattern: String,
        cross_ref: String,
    },
}

impl fmt::Display for TraversalPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read traversal policy input {}: {source}",
                    path.display()
                )
            }
            Self::Toml { path, source } => {
                write!(
                    f,
                    "invalid traversal policy TOML {}: {source}",
                    path.display()
                )
            }
            Self::UnsupportedVersion { path, version } => write!(
                f,
                "unsupported traversal policy version {version} in {}; expected version 1",
                path.display()
            ),
            Self::MissingSpec { path } => write!(
                f,
                "repository traversal policy {} is missing required `spec`",
                path.display()
            ),
            Self::InvalidSpecPath { path, spec } => write!(
                f,
                "repository traversal policy {} has spec path `{}` outside its policy root",
                path.display(),
                spec.display()
            ),
            Self::Pattern {
                path,
                pattern,
                source,
            } => write!(
                f,
                "invalid traversal exclusion pattern `{pattern}` in {}: {source}",
                path.display()
            ),
            Self::PatternSyntax {
                path,
                pattern,
                reason,
            } => write!(
                f,
                "invalid traversal exclusion pattern `{pattern}` in {}: {reason}",
                path.display()
            ),
            Self::UnresolvedCrossRef {
                path,
                pattern,
                cross_ref,
            } => write!(
                f,
                "traversal exclusion `{pattern}` in {} has unresolved cross-reference `{cross_ref}`",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TraversalPolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Toml { source, .. } => Some(source),
            Self::Pattern { source, .. } => Some(source),
            _ => None,
        }
    }
}
