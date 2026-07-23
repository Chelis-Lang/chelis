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
    scope_root: PathBuf,
    canonical_scope_root: Option<PathBuf>,
    canonical_explicit_root: Option<PathBuf>,
    matcher: Gitignore,
    may_exclude_files: bool,
    exclusions: Vec<CompiledExclusion>,
}

impl TraversalPolicy {
    pub fn load_for(target: &Path) -> Result<Self, TraversalPolicyError> {
        let start = policy_search_start(target);
        let repository_policy = find_repository_policy(start)?;
        let repository_root = repository_policy
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let baseline_root = repository_root.as_deref().unwrap_or(start);
        // Existing lint roots always have a canonical scope. The optional form
        // preserves direct Rule::check callers that use a not-yet-written
        // synthetic path; those callers have no discovered entries to admit.
        let canonical_scope_root = (baseline_root.as_os_str().is_empty() || baseline_root.exists())
            .then(|| canonicalize_policy_root(baseline_root))
            .transpose()?;
        let canonical_explicit_root = target
            .is_dir()
            .then(|| canonicalize_input(target))
            .transpose()?;

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
            let canonical_policy_root = canonicalize_policy_root(policy_root)?;
            let canonical_spec_path = canonicalize_input(&spec_path)?;
            if !canonical_spec_path.starts_with(&canonical_policy_root) {
                return Err(TraversalPolicyError::InvalidSpecPath {
                    path: policy_path.to_path_buf(),
                    spec: spec.clone(),
                });
            }
            let spec_source = std::fs::read_to_string(&canonical_spec_path).map_err(|source| {
                TraversalPolicyError::Io {
                    path: spec_path.clone(),
                    source,
                }
            })?;
            verify_cross_refs(policy_path, &document.exclusions, &spec_source)?;
            exclusions.extend(compile_exclusions(
                policy_root,
                policy_path,
                document.exclusions,
            )?);
        }

        let combined_policy_path = repository_policy
            .as_deref()
            .unwrap_or_else(|| Path::new(SHIPPED_POLICY_PATH));
        let matcher = compile_combined_matcher(baseline_root, combined_policy_path, &exclusions)?;
        let may_exclude_files = exclusions
            .iter()
            .any(|compiled| !is_directory_only_pattern(&compiled.exclusion.pattern));
        let scope_root = baseline_root.to_path_buf();

        Ok(Self {
            repository_root,
            scope_root,
            canonical_scope_root,
            canonical_explicit_root,
            matcher,
            may_exclude_files,
            exclusions,
        })
    }

    pub fn repository_root(&self) -> Option<&Path> {
        self.repository_root.as_deref()
    }

    /// Return whether `path` is excluded using the combined hot-path matcher.
    pub fn is_excluded(&self, path: &Path, is_dir: bool) -> bool {
        (is_dir || self.may_exclude_files) && self.matcher.matched(path, is_dir).is_ignore()
    }

    /// Admit one recursively discovered entry.
    ///
    /// Lexical policy matching handles ordinary entries. Existing inputs are
    /// also resolved before use so a symlink alias cannot import content from
    /// an excluded directory or from outside the policy root. Internal links
    /// whose targets remain policy-admitted preserve their existing behavior.
    pub(crate) fn is_admitted_entry(
        &self,
        path: &Path,
        is_dir: bool,
        resolve_target: bool,
    ) -> bool {
        !self.is_excluded(path, is_dir)
            && (!resolve_target || self.resolved_input_is_admitted(path, is_dir, true))
    }

    /// Admit an explicit walk root while overriding only exclusion matching.
    ///
    /// Explicit targets still have to be regular files/directories (or links
    /// resolving to one) inside the repository policy boundary.
    pub(crate) fn is_admitted_explicit_entry(&self, path: &Path, is_dir: bool) -> bool {
        let Some(canonical_scope_root) = self.canonical_scope_root.as_deref() else {
            return false;
        };
        let Ok(resolved) = std::fs::canonicalize(path) else {
            return false;
        };
        let Ok(metadata) = std::fs::metadata(&resolved) else {
            return false;
        };
        let resolved_kind_matches = if is_dir {
            metadata.is_dir()
        } else {
            metadata.is_file()
        };
        resolved_kind_matches && resolved.starts_with(canonical_scope_root)
    }

    /// Admit an ancillary input, including its governed parents.
    pub fn is_admitted_ancillary(&self, path: &Path, is_dir: bool) -> bool {
        !self.is_excluded_or_parent(path, is_dir)
            && self.resolved_input_is_admitted(path, is_dir, false)
    }

    /// Return policy-admitted `Cargo.toml` files one level below
    /// `<root>/crates` for the documented package-name compatibility path.
    ///
    /// This intentionally exposes a purpose-specific bounded query, not a
    /// generic directory API that rules could compose into recursive traversal.
    /// Recursive discovery remains owned by `walker::walk`.
    pub(crate) fn admitted_workspace_cargo_manifests(&self, root: &Path) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(root.join("crates")) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter_map(|entry| {
                let crate_path = entry.path();
                // Workspace layouts may expose an internal crate directory
                // through a symlink. Resolve its kind before policy admission;
                // `DirEntry::file_type` reports only the lexical link kind.
                if !std::fs::metadata(&crate_path).ok()?.is_dir()
                    || !self.is_admitted_ancillary(&crate_path, true)
                {
                    return None;
                }
                let manifest = crate_path.join("Cargo.toml");
                self.is_admitted_ancillary(&manifest, false)
                    .then_some(manifest)
            })
            .collect()
    }

    /// Return whether `path` or one of its governed parents is excluded.
    ///
    /// Recursive traversal reaches this parent-aware path only when a symlink
    /// resolves to a different governed location; ordinary directories are
    /// pruned on their direct match before descendants are visited. Rules that
    /// consult an ancillary file outside the entry vector must use
    /// [`Self::is_admitted_ancillary`] so an excluded directory cannot
    /// influence an admitted entry indirectly.
    pub fn is_excluded_or_parent(&self, path: &Path, is_dir: bool) -> bool {
        let Some(mut relative) = self.path_relative_to_scope(path) else {
            return false;
        };
        if self.matcher.matched(&relative, is_dir).is_ignore() {
            return true;
        }
        while let Some(parent) = relative.parent() {
            if self.matcher.matched(parent, true).is_ignore() {
                return true;
            }
            relative = parent.to_path_buf();
        }
        false
    }

    fn path_relative_to_scope(&self, path: &Path) -> Option<PathBuf> {
        let scope = if self.scope_root.as_os_str().is_empty() {
            Path::new(".")
        } else {
            &self.scope_root
        };
        if path.is_absolute() {
            let absolute_scope = if scope.is_absolute() {
                scope.to_path_buf()
            } else {
                std::env::current_dir().ok()?.join(scope)
            };
            path.strip_prefix(&absolute_scope)
                .ok()
                .or_else(|| {
                    self.canonical_scope_root
                        .as_deref()
                        .and_then(|root| path.strip_prefix(root).ok())
                })
                .map(Path::to_path_buf)
        } else if scope == Path::new(".") {
            Some(path.to_path_buf())
        } else {
            path.strip_prefix(scope).ok().map(Path::to_path_buf)
        }
    }

    fn resolved_input_is_admitted(
        &self,
        path: &Path,
        is_dir: bool,
        honor_explicit_root: bool,
    ) -> bool {
        let Ok(link_metadata) = std::fs::symlink_metadata(path) else {
            return false;
        };
        let lexical_relative = self.path_relative_to_scope(path);
        let governed = lexical_relative.is_some();
        let is_symlink = link_metadata.file_type().is_symlink();
        if !governed {
            // A discovered repository policy governs lexical paths, not only
            // their resolved targets. Machine-local ancestors above that
            // boundary cannot contribute ancillary metadata, even when a link
            // points back into the repository. Loose targets without a policy
            // retain the historical ancestor-manifest compatibility path.
            if self.repository_root.is_some() {
                return false;
            }
            if !is_symlink {
                return true;
            }
        }
        let Some(canonical_scope_root) = self.canonical_scope_root.as_deref() else {
            return false;
        };
        let Ok(resolved) = std::fs::canonicalize(path) else {
            return false;
        };
        let Ok(metadata) = std::fs::metadata(&resolved) else {
            return false;
        };
        let resolved_is_dir = metadata.is_dir();
        let resolved_kind_matches = if is_dir {
            resolved_is_dir
        } else {
            metadata.is_file()
        };
        let Ok(resolved_relative) = resolved.strip_prefix(canonical_scope_root) else {
            return false;
        };
        if !resolved_kind_matches {
            return false;
        }
        let traverses_alias = lexical_relative.as_deref() != Some(resolved_relative);
        if !traverses_alias {
            return !self.is_excluded(resolved_relative, resolved_is_dir);
        }
        if honor_explicit_root {
            !self.is_excluded_or_parent_below_explicit_root(resolved_relative, resolved_is_dir)
        } else {
            !self.is_excluded_or_parent(&resolved, resolved_is_dir)
        }
    }

    fn is_excluded_or_parent_below_explicit_root(&self, path: &Path, is_dir: bool) -> bool {
        if self.matcher.matched(path, is_dir).is_ignore() {
            return true;
        }
        let explicit_root = self
            .canonical_explicit_root
            .as_deref()
            .zip(self.canonical_scope_root.as_deref())
            .and_then(|(explicit, scope)| explicit.strip_prefix(scope).ok());
        let mut relative = path.to_path_buf();
        while let Some(parent) = relative.parent() {
            if explicit_root == Some(parent) {
                break;
            }
            if self.matcher.matched(parent, true).is_ignore() {
                return true;
            }
            relative = parent.to_path_buf();
        }
        false
    }

    /// Explain the first policy entry excluding `path`.
    ///
    /// The walker uses [`Self::is_excluded`] so normal traversal evaluates
    /// one combined glob set. Per-entry matchers are retained only for this
    /// lower-frequency explainability path.
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

fn find_repository_policy(start: &Path) -> Result<Option<PathBuf>, TraversalPolicyError> {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join(POLICY_FILE);
        let metadata = match std::fs::symlink_metadata(&candidate) {
            Ok(metadata) => metadata,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(TraversalPolicyError::Io {
                    path: candidate,
                    source,
                });
            }
        };
        if !metadata.is_file() && !metadata.file_type().is_symlink() {
            return Err(TraversalPolicyError::InvalidPolicyPath {
                path: candidate,
                reason: "must be a regular file",
            });
        }
        let canonical_root = canonicalize_policy_root(ancestor)?;
        let canonical_candidate = canonicalize_input(&candidate)?;
        if !canonical_candidate.is_file() || !canonical_candidate.starts_with(&canonical_root) {
            return Err(TraversalPolicyError::InvalidPolicyPath {
                path: candidate,
                reason: "resolves outside its policy root",
            });
        }
        return Ok(Some(candidate));
    }
    Ok(None)
}

fn canonicalize_policy_root(root: &Path) -> Result<PathBuf, TraversalPolicyError> {
    let root = if root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        root
    };
    canonicalize_input(root)
}

fn canonicalize_input(path: &Path) -> Result<PathBuf, TraversalPolicyError> {
    std::fs::canonicalize(path).map_err(|source| TraversalPolicyError::Io {
        path: path.to_path_buf(),
        source,
    })
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

fn is_directory_only_pattern(pattern: &str) -> bool {
    let normalized = if pattern.ends_with("\\ ") {
        pattern
    } else {
        pattern.trim_end()
    };
    normalized.ends_with('/')
}

fn compile_combined_matcher(
    root: &Path,
    policy_path: &Path,
    exclusions: &[CompiledExclusion],
) -> Result<Gitignore, TraversalPolicyError> {
    let mut builder = GitignoreBuilder::new(root);
    builder.allow_unclosed_class(false);
    for compiled in exclusions {
        builder
            .add_line(None::<PathBuf>, &compiled.exclusion.pattern)
            .map_err(|source| TraversalPolicyError::Pattern {
                path: policy_path.to_path_buf(),
                pattern: compiled.exclusion.pattern.clone(),
                source,
            })?;
    }
    builder
        .build()
        .map_err(|source| TraversalPolicyError::Pattern {
            path: policy_path.to_path_buf(),
            pattern: "<combined policy>".to_string(),
            source,
        })
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
    InvalidPolicyPath {
        path: PathBuf,
        reason: &'static str,
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
            Self::InvalidPolicyPath { path, reason } => {
                write!(f, "repository traversal policy {} {reason}", path.display())
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_only_fast_path_matches_ignore_parser_edges() {
        let cases = [
            ("archive/", "archive"),
            ("archive/   ", "archive"),
            ("archive/\\ ", "archive/ "),
            ("archive\\/", "archive"),
            ("archive", "archive"),
            ("/blocked.ch", "blocked.ch"),
            ("\\!literal", "!literal"),
        ];

        for (pattern, candidate) in cases {
            let mut builder = GitignoreBuilder::new(".");
            builder.allow_unclosed_class(false);
            builder.add_line(None::<PathBuf>, pattern).unwrap();
            let matcher = builder.build().unwrap();
            let parser_directory_only = matcher.matched(candidate, true).is_ignore()
                && !matcher.matched(candidate, false).is_ignore();
            assert_eq!(
                is_directory_only_pattern(pattern),
                parser_directory_only,
                "fast-path classification drifted for {pattern:?}"
            );
        }
    }
}
