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

/// How strongly a rule participates in user-facing and CI surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Advisory,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Advisory => "advisory",
        }
    }

    pub fn blocks_check(self) -> bool {
        matches!(self, Severity::Error)
    }
}

/// A byte-range source replacement produced by a fixable rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replacement {
    pub path: PathBuf,
    pub start: usize,
    pub end: usize,
    pub text: String,
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

    /// Severity for CLI reporting and build-gate behavior.
    fn severity(&self) -> Severity {
        Severity::Error
    }

    /// Run the check. Return zero or more violations.
    fn check(&self, ctx: &Context<'_>) -> Vec<Violation>;

    /// Return an auto-fix for `violation`, when this occurrence is safely
    /// fixable. The CLI fix driver suppresses fixes when a `keep` annotation
    /// applies.
    fn fix(&self, _ctx: &Context<'_>, _violation: &Violation) -> Option<Replacement> {
        None
    }

    /// Whether the CLI fix driver must verify the post-fix source against
    /// the typed/linearity pipeline before writing the replacement to disk.
    ///
    /// Default `false` for rules whose fix is purely structural (case
    /// rename, allowlist update, identifier normalization, etc.).
    ///
    /// Rules whose rewrites cross the semantic safety bar named in
    /// `spec/01-nomenclature.md` §12 — currently `redundant-linearity-call`
    /// — override to `true`. The CLI's `apply_lint_fixes` driver then
    /// applies the replacement to a candidate `String`, runs the typed
    /// pipeline (parse, desugar, expand, type-check, effect-check,
    /// linearity-check), and only writes the replacement to disk if every
    /// stage accepts. If the pipeline rejects the candidate, the
    /// replacement is silently dropped.
    ///
    /// Architectural choice documented in
    /// `docs/investigations/redundant_linearity_autofix_architecture.md`
    /// (Path 1B): the gate lives in the CLI driver so `chelis-lint` stays
    /// dep-pure.
    fn fix_requires_typed_pipeline_check(&self) -> bool {
        false
    }
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
            violations.extend(
                rule.check(&ctx)
                    .into_iter()
                    .filter(|v| !inline_allows(source.as_deref(), surface, v)),
            );
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

fn inline_allows(source: Option<&str>, surface: Surface, violation: &Violation) -> bool {
    let Some(source) = source else {
        return false;
    };
    let rule = violation.rule_id.as_str();
    if file_level_allows(source, rule) {
        return true;
    }
    let Some(line_no) = violation.line else {
        return false;
    };
    let lines: Vec<&str> = source.lines().collect();
    let current = lines.get(line_no.saturating_sub(1)).copied().unwrap_or("");
    let previous = line_no
        .checked_sub(2)
        .and_then(|idx| lines.get(idx))
        .copied()
        .unwrap_or("");
    inline_line_allows(current, surface, rule) || inline_line_allows(previous, surface, rule)
}

pub fn inline_keeps(source: &str, surface: Surface, line_no: usize, rule: &str) -> bool {
    let lines: Vec<&str> = source.lines().collect();
    let current = lines.get(line_no.saturating_sub(1)).copied().unwrap_or("");
    let previous = line_no
        .checked_sub(2)
        .and_then(|idx| lines.get(idx))
        .copied()
        .unwrap_or("");
    inline_line_keeps(current, surface, rule) || inline_line_keeps(previous, surface, rule)
}

fn inline_line_allows(line: &str, surface: Surface, rule: &str) -> bool {
    let Some(directive) = lint_directive(line, surface) else {
        return false;
    };
    let directive = directive.trim();
    directive
        .strip_prefix("allow")
        .map(|rest| rest.split_whitespace().any(|name| name == rule))
        .unwrap_or(false)
}

fn inline_line_keeps(line: &str, surface: Surface, rule: &str) -> bool {
    let Some(directive) = lint_directive(line, surface) else {
        return false;
    };
    let directive = directive.trim();
    directive
        .strip_prefix("keep")
        .map(|rest| rest.split_whitespace().any(|name| name == rule))
        .unwrap_or(false)
}

fn file_level_allows(source: &str, rule: &str) -> bool {
    let allow = format!("#[allow({rule})]");
    source.lines().any(|line| line.trim() == allow)
}

pub(crate) fn lint_directive(line: &str, surface: Surface) -> Option<&str> {
    let mut in_string = false;
    let mut escaped = false;
    let mut cursor = 0usize;
    while cursor < line.len() {
        let ch = line[cursor..].chars().next()?;
        if in_string {
            if escaped {
                escaped = false;
            } else {
                match ch {
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
            }
            cursor += ch.len_utf8();
            continue;
        }
        if ch == '"' {
            in_string = true;
            cursor += ch.len_utf8();
            continue;
        }
        let starts_comment = match surface {
            Surface::DeepSource => line[cursor..].starts_with(';'),
            Surface::PythonSource => line[cursor..].starts_with('#'),
            Surface::SurfSource | Surface::RustSource => line[cursor..].starts_with("//"),
            _ => {
                line[cursor..].starts_with("//")
                    || line[cursor..].starts_with('#')
                    || line[cursor..].starts_with(';')
            }
        };
        if starts_comment {
            return line[cursor..]
                .split_once("chelis-lint:")
                .map(|(_, directive)| directive);
        }
        cursor += ch.len_utf8();
    }
    None
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
