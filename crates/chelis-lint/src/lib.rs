//! Naming-convention lint for the Chelis ecosystem.
//!
//! Language naming and style rules live in `chelis/spec/01-nomenclature.md`.
//! Repository workflow conventions may instead be owned by `CONTRIBUTING.md`.
//! This crate translates each rule into an executable check, walks a target
//! tree, and reports violations with `file:line:col: rule_id: message`
//! references.
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

use std::any::Any;
use std::fmt;
use std::path::{Path, PathBuf};

pub mod exceptions;
pub mod policy;
pub mod reef_package;
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

/// Immutable state prepared by one rule for one [`lint`] invocation.
pub type PreparedRuleState = Box<dyn Any + Send + Sync>;

/// The interface every lint rule implements.
pub trait Rule: Send + Sync {
    /// Stable rule identifier (kebab-case), e.g. `"module-compound-titlecase"`.
    /// Used in violation output, exception cross-refs, and `--rule <id>` and
    /// `--explain <id>` invocations.
    fn id(&self) -> &str;

    /// Owning documentation reference for this rule, for example `"§6.2"` in
    /// `spec/01-nomenclature.md` or `"CONTRIBUTING.md § Declarative Naming"`.
    /// Printed alongside violations so readers can find the rule.
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

    /// Prepare immutable state once for a [`lint`] invocation.
    ///
    /// The default state is empty. Rules that need corpus-wide context can
    /// override this hook and consume their state in [`Rule::check_prepared`].
    /// The shared `policy` is the invocation's single loaded traversal
    /// policy; rules must not reload it.
    fn prepare_run(
        &self,
        _root: &Path,
        _entries: &[walker::Entry],
        _policy: &policy::TraversalPolicy,
    ) -> Result<PreparedRuleState, LintError> {
        Ok(Box::new(()))
    }

    /// Run the check directly, without invocation-prepared state.
    fn check(&self, ctx: &Context<'_>) -> Vec<Violation>;

    /// Run the check with state returned by [`Rule::prepare_run`].
    ///
    /// The default preserves existing rules by delegating to [`Rule::check`].
    fn check_prepared(
        &self,
        ctx: &Context<'_>,
        _prepared: &(dyn Any + Send + Sync),
    ) -> Vec<Violation> {
        self.check(ctx)
    }

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
    /// `docs/archive/investigations/redundant_linearity_autofix_architecture.md`
    /// (Path 1B): the gate lives in the CLI driver so `chelis-lint` stays
    /// dep-pure.
    fn fix_requires_typed_pipeline_check(&self) -> bool {
        false
    }

    /// Whether the CLI lint driver should suppress this rule's warnings
    /// when the autofix is unavailable for the violation (no fix
    /// proposed, fix proposed but bailed out, or fix rejected by the
    /// typed-pipeline gate).
    ///
    /// Default `false`: warnings fire even when no fix is offered, so
    /// the user can see the diagnostic and rewrite manually. This is
    /// the right behavior for purely informational advisories whose
    /// lack-of-fix carries its own signal (no such rule is currently
    /// registered; the closest historical example was the pre-0.7.9
    /// `redundant-linearity-call`).
    ///
    /// Rules whose warning is only meaningful when paired with a safe
    /// rewrite (currently `prefer-pipe-operator` and
    /// `redundant-linearity-call`) opt in to `true`.
    /// `chelis lint --fix` then converges for those rules: either the
    /// rewrite is applied (and the warning disappears with the
    /// rewrite), or the warning is suppressed (because the rule
    /// declines to propose an unsafe transformation).
    ///
    /// Architectural rationale in
    /// `docs/archive/investigations/prefer_pipe_trigger_emit_diagnosis.md`.
    fn check_mirrors_fix(&self) -> bool {
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
/// collect violations. Entries are checked in parallel and their results
/// reassembled in walk order, so the returned list, sorted by path, is
/// identical to a serial pass.
pub fn lint(root: &Path, rules: &[Box<dyn Rule>]) -> Result<Vec<Violation>, LintError> {
    let policy = std::sync::Arc::new(policy::TraversalPolicy::load_for(root)?);
    let entries: Vec<walker::Entry> = walker::walk_with_policy(root, &policy)?
        .into_iter()
        .collect::<Result<_, _>>()?;
    let prepared: Vec<PreparedRuleState> = rules
        .iter()
        .map(|rule| rule.prepare_run(root, &entries, &policy))
        .collect::<Result<_, _>>()?;

    let mut violations: Vec<Violation> =
        check_entries_in_parallel(&entries, |entry| check_entry(root, rules, &prepared, entry))
            .into_iter()
            .flatten()
            .collect();
    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then(a.line.cmp(&b.line))
            .then(a.rule_id.cmp(&b.rule_id))
    });
    Ok(violations)
}

/// Stack size for each file-checking worker. Rules parse Surf recursively, so
/// workers get the same 8 MiB the main thread has rather than the 2 MiB spawned
/// threads default to.
const ENTRY_WORKER_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Run `check` over every entry on up to `available_parallelism` workers and
/// return the results in entry order. Worker `w` takes entries `w`, `w + n`,
/// `w + 2n`, ...; results are placed back by index, so the output is identical
/// to a serial pass whatever the worker count or scheduling.
fn check_entries_in_parallel<F>(entries: &[walker::Entry], check: F) -> Vec<Vec<Violation>>
where
    F: Fn(&walker::Entry) -> Vec<Violation> + Sync,
{
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(entries.len());
    if workers <= 1 {
        return entries.iter().map(&check).collect();
    }
    let mut results: Vec<Vec<Violation>> = vec![Vec::new(); entries.len()];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let check = &check;
                std::thread::Builder::new()
                    .stack_size(ENTRY_WORKER_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        (worker..entries.len())
                            .step_by(workers)
                            .map(|index| (index, check(&entries[index])))
                            .collect::<Vec<_>>()
                    })
                    .expect("spawn lint worker")
            })
            .collect();
        for handle in handles {
            let checked = handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            for (index, violations) in checked {
                results[index] = violations;
            }
        }
    });
    results
}

/// Every violation the applicable rules raise on one walked entry.
fn check_entry(
    root: &Path,
    rules: &[Box<dyn Rule>],
    prepared: &[PreparedRuleState],
    entry: &walker::Entry,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    let surface = match entry.surface {
        Some(s) => s,
        None => return violations,
    };
    let source = if surface.needs_source() {
        match std::fs::read_to_string(&entry.path) {
            Ok(s) => Some(s),
            Err(_) => return violations, // binary file or unreadable; skip
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
    for (rule, prepared) in rules.iter().zip(prepared) {
        if !rule.applies_to().contains(&surface) {
            continue;
        }
        violations.extend(
            rule.check_prepared(&ctx, prepared.as_ref())
                .into_iter()
                .filter(|v| !inline_allows(source.as_deref(), surface, v)),
        );
    }
    violations
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
    Policy(policy::TraversalPolicyError),
    Walk(ignore::Error),
}

impl fmt::Display for LintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LintError::Io(e) => write!(f, "io error: {e}"),
            LintError::Policy(e) => write!(f, "traversal policy error: {e}"),
            LintError::Walk(e) => write!(f, "walk error: {e}"),
        }
    }
}

impl std::error::Error for LintError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Policy(source) => Some(source),
            Self::Walk(source) => Some(source),
        }
    }
}

impl From<std::io::Error> for LintError {
    fn from(e: std::io::Error) -> Self {
        LintError::Io(e)
    }
}

impl From<policy::TraversalPolicyError> for LintError {
    fn from(e: policy::TraversalPolicyError) -> Self {
        LintError::Policy(e)
    }
}

impl From<ignore::Error> for LintError {
    fn from(e: ignore::Error) -> Self {
        LintError::Walk(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    /// chelis#3108: parallel entry checking returns exactly the serial result,
    /// in walk order, even when later entries finish first. The worker count
    /// comes from `available_parallelism`; on a one-core runner the helper
    /// takes its serial path and this test does not exercise parallel order.
    #[test]
    fn parallel_entry_checks_match_serial_order() {
        let entries: Vec<walker::Entry> = (0..64)
            .map(|index| walker::Entry {
                path: PathBuf::from(format!("entry_{index:02}.ch")),
                surface: Some(Surface::SurfSource),
            })
            .collect();
        let check = |entry: &walker::Entry| {
            let index: u64 = entry.path.to_string_lossy()[6..8].parse().unwrap();
            // Early entries sleep longest so workers finish out of order.
            std::thread::sleep(std::time::Duration::from_micros((64 - index) * 100));
            (0..index % 3)
                .map(|ordinal| Violation {
                    rule_id: "order-test".to_string(),
                    spec_ref: "§12.1".to_string(),
                    path: entry.path.clone(),
                    line: Some(ordinal as usize + 1),
                    col: None,
                    message: format!("{index}:{ordinal}"),
                })
                .collect::<Vec<_>>()
        };
        let render = |results: Vec<Vec<Violation>>| -> Vec<String> {
            results
                .into_iter()
                .flatten()
                .map(|v| v.to_string())
                .collect()
        };
        let serial = render(entries.iter().map(check).collect());
        let parallel = render(check_entries_in_parallel(&entries, check));
        assert!(!serial.is_empty());
        assert_eq!(parallel, serial);
    }

    struct RunLifecycleRule {
        prepare_calls: Arc<AtomicUsize>,
        checked_generations: Arc<Mutex<Vec<usize>>>,
    }

    impl Rule for RunLifecycleRule {
        fn id(&self) -> &str {
            "run-lifecycle-test"
        }

        fn spec_ref(&self) -> &str {
            "§12.1"
        }

        fn applies_to(&self) -> &[Surface] {
            &[Surface::SurfSource]
        }

        fn summary(&self) -> &str {
            "test-only lint-run lifecycle probe"
        }

        fn prepare_run(
            &self,
            _root: &Path,
            _entries: &[walker::Entry],
            _policy: &policy::TraversalPolicy,
        ) -> Result<PreparedRuleState, LintError> {
            let generation = self.prepare_calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Box::new(generation))
        }

        fn check_prepared(
            &self,
            _ctx: &Context<'_>,
            prepared: &(dyn std::any::Any + Send + Sync),
        ) -> Vec<Violation> {
            let generation = *prepared
                .downcast_ref::<usize>()
                .expect("driver paired prepared state with its owning rule");
            self.checked_generations
                .lock()
                .expect("generation lock")
                .push(generation);
            Vec::new()
        }

        fn check(&self, _ctx: &Context<'_>) -> Vec<Violation> {
            panic!("lint driver must use the prepared-check path")
        }
    }

    #[test]
    fn prepares_each_rule_once_per_lint_invocation_and_reprepares_next_run() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(temp.path().join("first.ch"), "def first() = 1\n")
            .expect("write first source");
        std::fs::write(temp.path().join("second.ch"), "def second() = 2\n")
            .expect("write second source");

        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let checked_generations = Arc::new(Mutex::new(Vec::new()));
        let rules: Vec<Box<dyn Rule>> = vec![Box::new(RunLifecycleRule {
            prepare_calls: Arc::clone(&prepare_calls),
            checked_generations: Arc::clone(&checked_generations),
        })];

        lint(temp.path(), &rules).expect("first lint invocation");
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *checked_generations.lock().expect("generation lock"),
            vec![1, 1],
            "both checked files must share the first invocation's state"
        );

        lint(temp.path(), &rules).expect("second lint invocation");
        assert_eq!(
            prepare_calls.load(Ordering::SeqCst),
            2,
            "the same rule objects must prepare fresh state for a later invocation"
        );
        assert_eq!(
            *checked_generations.lock().expect("generation lock"),
            vec![1, 1, 2, 2],
            "prepared state must not leak across lint invocations"
        );
    }

    struct DefaultLifecycleRule {
        check_calls: Arc<AtomicUsize>,
    }

    impl Rule for DefaultLifecycleRule {
        fn id(&self) -> &str {
            "default-lifecycle-test"
        }

        fn spec_ref(&self) -> &str {
            "§12.1"
        }

        fn applies_to(&self) -> &[Surface] {
            &[Surface::SurfSource]
        }

        fn summary(&self) -> &str {
            "test-only default lifecycle probe"
        }

        fn check(&self, _ctx: &Context<'_>) -> Vec<Violation> {
            self.check_calls.fetch_add(1, Ordering::SeqCst);
            Vec::new()
        }
    }

    struct UsizePreparedRule {
        check_calls: Arc<AtomicUsize>,
    }

    impl Rule for UsizePreparedRule {
        fn id(&self) -> &str {
            "usize-prepared-test"
        }

        fn spec_ref(&self) -> &str {
            "§12.1"
        }

        fn applies_to(&self) -> &[Surface] {
            &[Surface::SurfSource]
        }

        fn summary(&self) -> &str {
            "test-only usize prepared-state probe"
        }

        fn prepare_run(
            &self,
            _root: &Path,
            _entries: &[walker::Entry],
            _policy: &policy::TraversalPolicy,
        ) -> Result<PreparedRuleState, LintError> {
            Ok(Box::new(603usize))
        }

        fn check_prepared(
            &self,
            _ctx: &Context<'_>,
            prepared: &(dyn std::any::Any + Send + Sync),
        ) -> Vec<Violation> {
            assert_eq!(prepared.downcast_ref::<usize>(), Some(&603));
            self.check_calls.fetch_add(1, Ordering::SeqCst);
            Vec::new()
        }

        fn check(&self, _ctx: &Context<'_>) -> Vec<Violation> {
            panic!("lint driver must use the prepared-check path")
        }
    }

    struct StringPreparedRule {
        check_calls: Arc<AtomicUsize>,
    }

    impl Rule for StringPreparedRule {
        fn id(&self) -> &str {
            "string-prepared-test"
        }

        fn spec_ref(&self) -> &str {
            "§12.1"
        }

        fn applies_to(&self) -> &[Surface] {
            &[Surface::SurfSource]
        }

        fn summary(&self) -> &str {
            "test-only string prepared-state probe"
        }

        fn prepare_run(
            &self,
            _root: &Path,
            _entries: &[walker::Entry],
            _policy: &policy::TraversalPolicy,
        ) -> Result<PreparedRuleState, LintError> {
            Ok(Box::new(String::from("opaque-catalog")))
        }

        fn check_prepared(
            &self,
            _ctx: &Context<'_>,
            prepared: &(dyn std::any::Any + Send + Sync),
        ) -> Vec<Violation> {
            assert_eq!(
                prepared.downcast_ref::<String>().map(String::as_str),
                Some("opaque-catalog")
            );
            self.check_calls.fetch_add(1, Ordering::SeqCst);
            Vec::new()
        }

        fn check(&self, _ctx: &Context<'_>) -> Vec<Violation> {
            panic!("lint driver must use the prepared-check path")
        }
    }

    #[test]
    fn default_rule_hooks_delegate_and_type_erased_state_stays_with_its_rule() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(temp.path().join("first.ch"), "def first() = 1\n")
            .expect("write first source");
        std::fs::write(temp.path().join("second.ch"), "def second() = 2\n")
            .expect("write second source");

        let default_checks = Arc::new(AtomicUsize::new(0));
        let usize_checks = Arc::new(AtomicUsize::new(0));
        let string_checks = Arc::new(AtomicUsize::new(0));
        let rules: Vec<Box<dyn Rule>> = vec![
            Box::new(DefaultLifecycleRule {
                check_calls: Arc::clone(&default_checks),
            }),
            Box::new(UsizePreparedRule {
                check_calls: Arc::clone(&usize_checks),
            }),
            Box::new(StringPreparedRule {
                check_calls: Arc::clone(&string_checks),
            }),
        ];

        lint(temp.path(), &rules).expect("lint with heterogeneous prepared states");
        assert_eq!(default_checks.load(Ordering::SeqCst), 2);
        assert_eq!(usize_checks.load(Ordering::SeqCst), 2);
        assert_eq!(string_checks.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn lint_invocation_loads_traversal_policy_exactly_once() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(temp.path().join("agent.ch"), "def agent() = 1\n").expect("write source");
        std::fs::create_dir_all(temp.path().join("docs")).expect("create docs");
        std::fs::write(temp.path().join("docs/overview.md"), "# Overview\n").expect("write doc");
        let rules: Vec<Box<dyn Rule>> = vec![
            Box::new(crate::rules::doc_filename_convention::DocFilenameConvention),
            Box::new(crate::rules::opaque_domain_construction::OpaqueDomainConstruction),
        ];

        policy::reset_policy_load_count();
        lint(temp.path(), &rules).expect("first lint invocation");
        assert_eq!(
            policy::policy_load_count(),
            1,
            "one lint invocation must load the traversal policy exactly once, shared by the walker and every prepare_run hook"
        );

        lint(temp.path(), &rules).expect("second lint invocation");
        assert_eq!(
            policy::policy_load_count(),
            2,
            "each invocation loads fresh policy; no cross-invocation caching"
        );
    }
}
