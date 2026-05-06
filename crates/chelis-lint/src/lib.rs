//! Naming-convention lint for the Chelis ecosystem.
//!
//! The authoritative rule set lives in `chelis/spec/01-nomenclature.md`. This
//! crate translates each rule into an executable check, walks a target tree,
//! and reports violations with `file:line:col: rule_id: message` references.
//!
//! Architecture:
//!
//! - [`Rule`] is the trait every check implements. A rule names its `id`,
//!   the [`Surface`]s it applies to (Surf source, Rust source, manifest, doc,
//!   etc.), the spec section that documents it, and a `check` function that
//!   produces zero or more [`Violation`]s for a given [`Context`].
//! - The driver in [`lint`] walks a root directory, classifies each entry by
//!   surface, and dispatches to every matching rule.
//! - [`Exception`]s carry a mandatory `cross_ref` to a spec section that
//!   explains why a particular path/identifier is exempt — never free-form
//!   prose. Unresolvable cross-refs are build-time errors.
//!
//! Rules are added under [`rules`]; the [`registry`] module wires them up.

use std::fmt;
use std::path::{Path, PathBuf};

pub mod exceptions;
pub mod registry;
pub mod rules;
pub mod surface;
pub mod walker;

pub use surface::Surface;

/// A single lint violation, ready to print as `file:line:col: rule: message`.
#[derive(Debug, Clone)]
pub struct Violation {
    pub rule_id: String,
    pub spec_ref: String,
    pub path: PathBuf,
    pub line: Option<usize>,
    pub col: Option<usize>,
    pub message: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.display())?;
        if let Some(line) = self.line {
            write!(f, ":{line}")?;
            if let Some(col) = self.col {
                write!(f, ":{col}")?;
            }
        }
        write!(
            f,
            ": {} ({}): {}",
            self.rule_id, self.spec_ref, self.message
        )
    }
}

/// Inputs a [`Rule`] sees when checking one file or directory entry.
pub struct Context<'a> {
    /// Repository root the lint was invoked against. Used to compute paths
    /// relative to the repo for human-readable output and to resolve globs.
    pub root: &'a Path,
    /// Absolute path of the entry being checked.
    pub path: &'a Path,
    /// File contents for source/text surfaces. `None` for directory-entry
    /// surfaces and for files the rule doesn't need to read.
    pub source: Option<&'a str>,
    /// Pre-classified surface kind.
    pub surface: Surface,
}

/// The interface every lint rule implements.
pub trait Rule: Send + Sync {
    /// Stable rule identifier (kebab-case), e.g. `"module-compound-titlecase"`.
    /// Used in violation output, exception cross-refs, and `--rule <id>` and
    /// `--explain <id>` invocations.
    fn id(&self) -> &str;

    /// Section of `spec/01-nomenclature.md` that documents this rule, e.g.
    /// `"§6.2"`. Printed alongside violations so readers can find the rule.
    fn spec_ref(&self) -> &str;

    /// Surfaces this rule cares about. The driver only invokes `check` when
    /// the entry's surface is in this list.
    fn applies_to(&self) -> &[Surface];

    /// One-line summary, shown by `chelis lint --explain <id>`.
    fn summary(&self) -> &str;

    /// Run the check. Return zero or more violations.
    fn check(&self, ctx: &Context<'_>) -> Vec<Violation>;
}

/// A lint exception. Every entry must cross-reference a section of
/// `spec/01-nomenclature.md` that explains why the case is exempt; free-form
/// `description` strings are deliberately not part of the schema. Entries with
/// unresolvable `cross_ref`s are rejected at registry-load time.
#[derive(Debug, Clone)]
pub struct Exception {
    /// Glob pattern applied to the entry's path-relative-to-root.
    pub pattern: String,
    /// Rule being waived.
    pub rule_id: String,
    /// Section in `spec/01-nomenclature.md` that explains why, e.g. `"§8.5"`.
    pub cross_ref: String,
}

/// Walk `root`, classify each entry, dispatch to every matching rule, and
/// collect violations. Returns violations sorted by path (stable across runs).
pub fn lint(root: &Path, rules: &[Box<dyn Rule>]) -> Result<Vec<Violation>, LintError> {
    let mut violations = Vec::new();
    for entry in walker::walk(root)? {
        let entry = entry?;
        let surface = match entry.surface {
            Some(s) => s,
            None => continue,
        };
        let source = if surface.needs_source() {
            match std::fs::read_to_string(&entry.path) {
                Ok(s) => Some(s),
                Err(_) => continue, // binary file or unreadable; skip
            }
        } else {
            None
        };
        let ctx = Context {
            root,
            path: &entry.path,
            source: source.as_deref(),
            surface,
        };
        for rule in rules {
            if !rule.applies_to().contains(&surface) {
                continue;
            }
            violations.extend(rule.check(&ctx));
        }
    }
    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then(a.line.cmp(&b.line))
            .then(a.rule_id.cmp(&b.rule_id))
    });
    Ok(violations)
}

/// Errors that can occur during a lint run.
#[derive(Debug)]
pub enum LintError {
    Io(std::io::Error),
    Walk(walkdir::Error),
}

impl fmt::Display for LintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LintError::Io(e) => write!(f, "io error: {e}"),
            LintError::Walk(e) => write!(f, "walk error: {e}"),
        }
    }
}

impl std::error::Error for LintError {}

impl From<std::io::Error> for LintError {
    fn from(e: std::io::Error) -> Self {
        LintError::Io(e)
    }
}

impl From<walkdir::Error> for LintError {
    fn from(e: walkdir::Error) -> Self {
        LintError::Walk(e)
    }
}
