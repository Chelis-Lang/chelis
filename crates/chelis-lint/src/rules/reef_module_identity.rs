//! Rule `reef-module-identity` — a `.ch` file inside a reef package declares
//! exactly one module, rooted at the package's `module_prefix` and derived
//! from the file's path beneath its source root (§6.5).
//!
//! This is not a style preference. A file whose module identity disagrees
//! with its path cannot be placed in the package's module graph, so the
//! reef loader rejects it and the whole package becomes uncompilable. Before
//! this rule the style surface could not see that: `chelis lint --check` and
//! `chelis fmt --check` both exited 0 on a package `chelis reef build`
//! rejected at load time, so a green style gate was not evidence the package
//! loads and any workflow using it as a pre-build check got a false pass
//! (chelis#2116).
//!
//! The verdict comes from [`chelis_surf::module_identity::validate_module_path`],
//! the same function the loader calls, so the two cannot drift. What the rule
//! adds is only the package context the loader gets from the manifest: which
//! `module_prefix` applies, and which directories are source roots.
//!
//! Scope. The rule reports what is decidable from one file plus its
//! manifest. Loader rejections that need the whole module graph — a duplicate
//! module across two source roots, a cross-package macro export, an
//! unresolvable dependency — remain invisible here, because reaching them
//! would make the linter a partial build.

use crate::policy::TraversalPolicy;
use crate::walker::Entry;
use crate::{Context, LintError, PreparedRuleState, Rule, Surface, Violation};
use chelis_surf::ast::Decl;
use chelis_surf::module_identity::validate_module_path;
use serde::Deserialize;
use std::any::Any;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One reef package's contribution to the rule's verdict.
#[derive(Debug, Clone)]
struct PackageContext {
    module_prefix: String,
    /// Source roots, package-root-relative, `src` first.
    source_roots: Vec<String>,
}

/// Every reef package the invocation may consult, keyed by package root.
#[derive(Debug, Default)]
pub struct ReefPackages(BTreeMap<PathBuf, PackageContext>);

#[derive(Deserialize)]
struct ManifestFile {
    package: Option<ManifestPackage>,
}

#[derive(Deserialize)]
struct ManifestPackage {
    module_prefix: Option<String>,
    #[serde(default)]
    additional_sources: Vec<String>,
}

/// Read `module_prefix` and the declared source roots from a `reef.toml`.
///
/// A manifest that does not parse, or that declares no `module_prefix`, yields
/// `None`: reef itself rejects both, and this rule is not the surface that
/// reports a malformed manifest.
fn read_package_context(manifest: &Path) -> Option<PackageContext> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let parsed: ManifestFile = toml::from_str(&text).ok()?;
    let package = parsed.package?;
    let module_prefix = package.module_prefix?;
    if module_prefix.trim().is_empty() {
        return None;
    }
    let mut source_roots = vec!["src".to_string()];
    source_roots.extend(package.additional_sources);
    Some(PackageContext {
        module_prefix,
        source_roots,
    })
}

/// Index every `reef.toml` this invocation may consult.
///
/// Walked entries cover `chelis lint --check .` at or above a package root.
/// Ancestor discovery covers the other direction — `chelis lint --check
/// src/` or a single file inside a package, where the manifest is above the
/// walk root and never appears as an entry. Ancestors are admitted only when
/// traversal policy admits them, and the search is a bounded ancestor climb:
/// no rule-local recursive filesystem discovery, per the same contract
/// `doc-filename-convention` follows.
fn prepare_reef_packages(root: &Path, entries: &[Entry], policy: &TraversalPolicy) -> ReefPackages {
    let mut packages: BTreeMap<PathBuf, PackageContext> = BTreeMap::new();
    let mut record = |manifest: &Path| {
        let Some(package_root) = manifest.parent() else {
            return;
        };
        if packages.contains_key(package_root) {
            return;
        }
        if let Some(context) = read_package_context(manifest) {
            packages.insert(package_root.to_path_buf(), context);
        }
    };

    for entry in entries {
        if entry.surface != Some(Surface::ManifestToml)
            || entry.path.file_name().and_then(|name| name.to_str()) != Some("reef.toml")
        {
            continue;
        }
        record(&entry.path);
    }

    let mut cursor = Some(root);
    while let Some(directory) = cursor {
        let manifest = directory.join("reef.toml");
        if manifest.is_file() && policy.is_admitted_ancillary(&manifest, false) {
            record(&manifest);
        }
        cursor = directory.parent();
    }

    ReefPackages(packages)
}

impl ReefPackages {
    /// The innermost package containing `path`, with `path` expressed relative
    /// to that package's root.
    fn enclosing(&self, path: &Path) -> Option<(&PackageContext, PathBuf)> {
        self.0
            .iter()
            .filter_map(|(root, context)| {
                path.strip_prefix(root)
                    .ok()
                    .map(|relative| (root, context, relative.to_path_buf()))
            })
            .max_by_key(|(root, _, _)| root.components().count())
            .map(|(_, context, relative)| (context, relative))
    }
}

/// Split a package-relative path into the source root that owns it and the
/// path the loader validates against.
///
/// `src/nn/linear.ch` validates as `nn/linear.ch`; a file under an additional
/// source root keeps that root as its first component, so `properties/laws.ch`
/// validates as itself. A file under no declared source root is not a package
/// module at all — the loader never reads it — so the rule declines to reach a
/// verdict about it.
fn path_for_validation(context: &PackageContext, relative: &Path) -> Option<PathBuf> {
    let first = relative.components().next()?.as_os_str().to_str()?;
    let root = context
        .source_roots
        .iter()
        .find(|candidate| candidate.as_str() == first)?;
    if root == "src" {
        Some(relative.strip_prefix("src").ok()?.to_path_buf())
    } else {
        Some(relative.to_path_buf())
    }
}

pub struct ReefModuleIdentity;

impl Rule for ReefModuleIdentity {
    fn id(&self) -> &str {
        "reef-module-identity"
    }

    fn spec_ref(&self) -> &str {
        "§6.5"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "a .ch file in a reef package declares exactly one module, rooted at module_prefix and derived from its path (§6.5)"
    }

    fn prepare_run(
        &self,
        root: &Path,
        entries: &[Entry],
        policy: &TraversalPolicy,
    ) -> Result<PreparedRuleState, LintError> {
        Ok(Box::new(prepare_reef_packages(root, entries, policy)))
    }

    fn check(&self, _ctx: &Context<'_>) -> Vec<Violation> {
        // Without the invocation's manifest index there is no package context,
        // and a verdict reached without one would be a guess. The driver
        // always calls `check_prepared`.
        Vec::new()
    }

    fn check_prepared(
        &self,
        ctx: &Context<'_>,
        prepared: &(dyn Any + Send + Sync),
    ) -> Vec<Violation> {
        let Some(packages) = prepared.downcast_ref::<ReefPackages>() else {
            return Vec::new();
        };
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let Some((context, relative)) = packages.enclosing(ctx.path) else {
            return Vec::new();
        };
        let Some(rel_for_validation) = path_for_validation(context, &relative) else {
            return Vec::new();
        };
        // A file that does not parse is reported by the formatter check and by
        // `chelis check`; reaching a module-identity verdict from a broken
        // parse would invent one. The same skip the other parsing rules take.
        let Ok(decls) = chelis_surf::parser::parse_str(source) else {
            return Vec::new();
        };

        let modules: Vec<(&String, usize)> = decls
            .iter()
            .filter_map(|decl| match decl {
                Decl::Module { name, span, .. } => Some((name, span.offset)),
                _ => None,
            })
            .collect();

        let violation = |line: usize, col: usize, message: String| Violation {
            rule_id: self.id().to_string(),
            spec_ref: self.spec_ref().to_string(),
            path: ctx.path.to_path_buf(),
            line: Some(line),
            col: Some(col),
            message,
        };

        let [(name, offset)] = modules.as_slice() else {
            // §6.5 violation 3, worded as the loader words it: per file, not
            // per declaration. Only the zero case reaches here — a second
            // `module` line is a parse error, which the branch above defers
            // on and `chelis fmt --check` already rejects.
            return vec![violation(
                1,
                1,
                format!(
                    "{} must contain exactly one top-level module declaration",
                    relative.display()
                ),
            )];
        };

        match validate_module_path(&context.module_prefix, name, &rel_for_validation) {
            Ok(()) => Vec::new(),
            Err(message) => {
                let (line, col) = line_col(source, *offset);
                vec![violation(line, col, message)]
            }
        }
    }
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}
