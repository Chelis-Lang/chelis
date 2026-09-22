//! The conformance audit engine: `chelis reef conform audit`.
//!
//! Walks a shell repo and evaluates each [`crate::manifest::MANIFEST`] row,
//! producing an [`AuditReport`]. It is **offline and hermetic** (never opens a
//! socket) so it is safe as a required CI gate and safe for the HEAD-binary
//! canary. Every check is a filesystem read, a TOML/line parse, or a
//! managed-block hash — no compilation.
//!
//! Honesty rule (contract discipline): rows that are review judgement, not
//! mechanically decidable (≥2-config acceptance; installer/uv correctness in
//! part), report [`Verdict::Manual`] — never a false `Pass`. Conditional rows
//! whose trigger is absent report [`Verdict::Na`]. `since_version` gating
//! downgrades a row to `Na` when the shell's pin predates the row, so the HEAD
//! canary does not fail a stale shell for a requirement that postdates its pin.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::manifest::{CONTRACT_BASELINE_VERSION, ContractRow, MANIFEST, Tier};
use crate::{canonical, conform, managed_block, registry, skills};

/// The version of the toolchain performing the audit — the crate's own build
/// version, which equals `chelis_compiler_api::COMPILER_VERSION` (both are the
/// workspace version). It is the only version whose canonical bodies this
/// binary embeds, so a managed block's body can only be compared to canonical
/// when the block is stamped for this same version (see `check_managed_block`).
const AUDITOR_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The outcome of one row check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Requirement satisfied.
    Pass,
    /// Requirement violated.
    Fail,
    /// Cannot be decided mechanically; needs human review.
    Manual,
    /// Not applicable (conditional trigger absent, or pin predates the row).
    Na,
}

impl Verdict {
    pub fn tag(&self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Manual => "manual",
            Verdict::Na => "na",
        }
    }
}

/// One row's result.
#[derive(Debug, Clone)]
pub struct RowResult {
    pub row: u8,
    pub key: &'static str,
    pub section: &'static str,
    pub tier: Tier,
    pub verdict: Verdict,
    /// Empty when `Pass`/`Na`; otherwise the specific problem.
    pub diagnostic: String,
    /// Suggested remedy (may be empty).
    pub fix: String,
    /// Per-finding evidence surfaced under `conform audit --explain` (chelis#654):
    /// the exact site(s) and the per-candidate reasoning behind the verdict.
    /// Empty for rows that carry no site-level detail.
    pub evidence: Vec<String>,
}

/// The full audit result for a shell root.
#[derive(Debug, Clone)]
pub struct AuditReport {
    pub root: PathBuf,
    /// The shell's resolved `compiler = "=X.Y.Z"` pin, if readable.
    pub reef_pin: Option<String>,
    pub rows: Vec<RowResult>,
}

impl AuditReport {
    /// Count of hard failures: `Fail` verdicts on applicable MUST-tier rows.
    /// SHOULD misses and MANUAL/NA never count.
    pub fn must_failures(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| r.verdict == Verdict::Fail && tier_is_must(r.tier))
            .count()
    }

    /// True iff there are no hard failures **and** the audit actually
    /// evaluated something. An all-`Na`/`Manual` report (e.g. a pin below the
    /// contract baseline that would otherwise gate every row out) evaluated
    /// nothing and must not read as conformant.
    pub fn ok(&self) -> bool {
        self.must_failures() == 0 && self.evaluated_any()
    }

    /// Whether at least one row reached a mechanical `Pass`/`Fail` verdict.
    pub fn evaluated_any(&self) -> bool {
        self.rows
            .iter()
            .any(|r| matches!(r.verdict, Verdict::Pass | Verdict::Fail))
    }
}

/// MUST-class tiers (unconditional or triggered-conditional). SHOULD is the only
/// non-gating tier; conditional tiers are gating once their trigger fires, and
/// the row check emits `Na` when the trigger is absent, so treating them as MUST
/// here is correct.
fn tier_is_must(tier: Tier) -> bool {
    tier.gates()
}

// ---------------------------------------------------------------- entry point

/// Audit the shell rooted at `root`.
pub fn audit(root: &Path) -> AuditReport {
    let ctx = Ctx::load(root);
    let rows = MANIFEST.iter().map(|row| check_row(row, &ctx)).collect();
    AuditReport {
        root: root.to_path_buf(),
        reef_pin: ctx.reef_pin.clone(),
        rows,
    }
}

/// Precomputed reads shared across row checks.
struct Ctx {
    root: PathBuf,
    reef_pin: Option<String>,
    /// The shell's package name from `reef.toml`, used to look the shell up in
    /// the registry (its authoritative classification).
    shell_name: Option<String>,
    agents_md: Option<String>,
    claude_symlink_ok: bool,
    reef_toml: Option<String>,
    cargo_toml: Option<String>,
    /// `(filename, contents)` for every `.github/workflows/*.yml`.
    workflows: Vec<(String, String)>,
    /// The `conform` control surface parsed out of `reef.toml` (chelis#1262).
    /// `None` when there is no `reef.toml` at all (row 2 owns that); `Some(Err)`
    /// when the manifest does not parse, which §8 must report rather than read
    /// as an empty declaration.
    conform: Option<Result<conform::ConformDecl, String>>,
}

impl Ctx {
    fn load(root: &Path) -> Ctx {
        let reef_toml = read_opt(&root.join("reef.toml"));
        let reef_pin = reef_toml.as_deref().and_then(parse_compiler_pin);
        let shell_name = reef_toml.as_deref().and_then(parse_package_name);
        let conform = reef_toml.as_deref().map(conform::parse);
        let agents_md = read_opt(&root.join("AGENTS.md"));
        let claude_symlink_ok = claude_is_symlink_to_agents(root);
        let cargo_toml = read_opt(&root.join("Cargo.toml"));
        let workflows = read_workflows(root);
        Ctx {
            root: root.to_path_buf(),
            reef_pin,
            shell_name,
            agents_md,
            claude_symlink_ok,
            reef_toml,
            cargo_toml,
            workflows,
            conform,
        }
    }

    /// The recognized `conform.local_skills` declaration, or empty when there is
    /// no manifest, no declaration, or a manifest that does not parse. §8's own
    /// row reports the unparseable case; the drift scan must not additionally
    /// treat an unreadable file as an allowlist.
    fn local_skills(&self) -> &[String] {
        match &self.conform {
            Some(Ok(decl)) => &decl.local_skills,
            _ => &[],
        }
    }

    /// The recognized `conform.excluded_skills` declaration, or empty when the
    /// manifest or declaration is absent. The vendored-skills row reports parse
    /// failures and invalid names before using this as an omission list.
    fn excluded_skills(&self) -> &[String] {
        match &self.conform {
            Some(Ok(decl)) => &decl.excluded_skills,
            _ => &[],
        }
    }

    fn read(&self, rel: &str) -> Option<String> {
        read_opt(&self.root.join(rel))
    }

    fn exists(&self, rel: &str) -> bool {
        self.root.join(rel).exists()
    }
}

/// Sentinel diagnostic prefix emitted by the `check_row` catch-all for a
/// MANIFEST key with no dispatch arm. `tests/audit_negative_parity.rs` audits a
/// scaffolded shell and asserts no row ever carries it, so a MANIFEST row added
/// without a real check fails the build instead of silently reporting an
/// unimplemented `Manual` that reads as "reviewed, nothing to flag" (chelis#739).
pub const NO_CHECK_IMPLEMENTED_PREFIX: &str = "no check implemented for row key";

fn check_row(row: &ContractRow, ctx: &Ctx) -> RowResult {
    // since_version gating: a stale shell is spared rows added *after* its pin
    // (canary-safe), but never the baseline rows. The pin is floored at the
    // contract baseline, so a pre-contract pin (e.g. `=0.1.0`) cannot gate the
    // whole table out to `Na` and report a bare shell as conformant.
    if let Some(pin) = &ctx.reef_pin
        && !version_ge(&gate_version(pin), row.since_version)
    {
        return result(
            row,
            Verdict::Na,
            format!(
                "not applicable: shell pins {pin}, row introduced in {}",
                row.since_version
            ),
            "",
            Vec::new(),
        );
    }

    let (verdict, diagnostic, fix, evidence) = match row.key {
        "agents-md" => check_agents_md(ctx),
        "reef-pin" => check_reef_pin(ctx),
        "workflow-env-pins" => check_workflow_env_pins(ctx),
        "pin-consistency-guard" => check_pin_consistency_guard(ctx),
        "toolchain-installer" => check_toolchain_installer(ctx),
        "uv-python" => check_uv_python(ctx),
        "chelis-surface" => check_chelis_surface(ctx),
        "upstream-bugs" => check_upstream_bugs(ctx),
        "staleness-audit" => check_narrowing_coverage(ctx),
        "issue-drafts" => check_issue_drafts(ctx),
        "tests-neg" => check_tests_neg(ctx),
        "tests-blocked" => check_tests_blocked(ctx),
        "pin-bump-checklist" => check_agents_heading(ctx, "Pin Bump Checklist"),
        "vendored-skills" => check_vendored_skills(ctx),
        "parity-harness" => check_parity_harness(ctx),
        "two-config-acceptance" => (
            Verdict::Manual,
            "≥2-config acceptance is review discipline, not mechanically decidable".to_string(),
            "reviewer must confirm each new public verb has ≥2 distinct shape/config cases"
                .to_string(),
            Vec::new(),
        ),
        "scaffolding-drift-rule" => check_agents_heading(ctx, "Scaffolding Drift Rule"),
        "chelis-src" => check_chelis_src(ctx),
        other => (
            Verdict::Manual,
            format!("{NO_CHECK_IMPLEMENTED_PREFIX} {other:?}"),
            String::new(),
            Vec::new(),
        ),
    };
    result(row, verdict, diagnostic, fix, evidence)
}

fn result(
    row: &ContractRow,
    verdict: Verdict,
    diagnostic: impl Into<String>,
    fix: impl Into<String>,
    evidence: Vec<String>,
) -> RowResult {
    RowResult {
        row: row.row,
        key: row.key,
        section: row.section,
        tier: row.tier,
        verdict,
        diagnostic: diagnostic.into(),
        fix: fix.into(),
        evidence,
    }
}

type Check = (Verdict, String, String, Vec<String>);

fn pass() -> Check {
    (Verdict::Pass, String::new(), String::new(), Vec::new())
}

fn fail(diag: impl Into<String>, fix: impl Into<String>) -> Check {
    (Verdict::Fail, diag.into(), fix.into(), Vec::new())
}

/// Like [`fail`], but attaches per-finding `evidence` lines surfaced under
/// `conform audit --explain` (chelis#654).
fn fail_ex(diag: impl Into<String>, fix: impl Into<String>, evidence: Vec<String>) -> Check {
    (Verdict::Fail, diag.into(), fix.into(), evidence)
}

/// A `Manual` verdict with a non-empty diagnostic (and optional `--explain`
/// evidence): the honest "cannot be decided mechanically, a human must look"
/// outcome. The diagnostic MUST be non-empty — `print_audit_report` renders a
/// row's detail only when its diagnostic is non-empty, so an empty-diagnostic
/// `Manual` prints nothing and reads as a silent pass (chelis#739).
fn manual_ex(diag: impl Into<String>, fix: impl Into<String>, evidence: Vec<String>) -> Check {
    (Verdict::Manual, diag.into(), fix.into(), evidence)
}

/// Whether `text` has a markdown heading line (any level) containing `needle`.
/// Stricter than a bare `contains`, so the section name merely appearing in
/// prose or a code span does not satisfy a "has this section" check.
fn has_heading(text: &str, needle: &str) -> bool {
    text.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with('#') && t.contains(needle)
    })
}

// ---------------------------------------------------------------- row checks

fn check_agents_md(ctx: &Ctx) -> Check {
    let Some(agents) = &ctx.agents_md else {
        return fail("AGENTS.md is missing", "run `chelis reef conform init`");
    };
    if !ctx.claude_symlink_ok {
        return fail(
            "CLAUDE.md must be a symlink to AGENTS.md",
            "ln -sf AGENTS.md CLAUDE.md",
        );
    }
    if !has_heading(agents, "Repo Identity") {
        return fail(
            "AGENTS.md lacks a Repo Identity section",
            "add a Repo Identity section stating the shell's intent",
        );
    }
    // The managed inheritance block must be present, stamped to the pin, and
    // untampered.
    let canonical = canonical::body("agents-inheritance").expect("embedded AGENTS contract");
    let expected = match crate::scaffold::apply_agents_exclusions(canonical, agents) {
        Ok(expected) => expected,
        Err(why) => {
            return fail(
                format!("AGENTS.md has an invalid shell-local exclusion block: {why}"),
                "fix or remove the shell-owned exclusion block, then run `chelis reef conform sync`",
            );
        }
    };
    check_managed_block(
        ctx,
        agents,
        "agents-inheritance",
        "AGENTS.md",
        Some(&expected),
    )
}

fn check_reef_pin(ctx: &Ctx) -> Check {
    match &ctx.reef_pin {
        Some(_) => pass(),
        None => match ctx.reef_toml {
            Some(_) => fail(
                "reef.toml has no well-formed `compiler = \"=X.Y.Z\"` pin",
                "add an exact pin, e.g. compiler = \"=0.14.0\"",
            ),
            None => fail("reef.toml is missing", "run `chelis reef conform init`"),
        },
    }
}

fn check_workflow_env_pins(ctx: &Ctx) -> Check {
    let Some(pin) = &ctx.reef_pin else {
        return fail(
            "cannot check workflow pins without a reef pin",
            "fix row 2 first",
        );
    };
    let bare = pin.trim_start_matches('=');
    let mut mismatches = Vec::new();
    let mut checked = 0;
    for (name, body) in &ctx.workflows {
        if !workflow_installs_toolchain(body) {
            continue;
        }
        checked += 1;
        if let Some(v) = extract_env(body, "CHELIS_VERSION")
            && v.trim_start_matches('v') != bare
        {
            mismatches.push(format!("{name}: CHELIS_VERSION={v} != {bare}"));
        }
        if let Some(t) = extract_env(body, "CHELIS_TAG")
            && t.trim_start_matches('v') != bare
        {
            mismatches.push(format!("{name}: CHELIS_TAG={t} != v{bare}"));
        }
        // A literal `chelisup install <ver>` is a pin location too — an env var
        // is not the only way a workflow installs a version.
        for v in extract_chelisup_install_versions(body) {
            if v.trim_start_matches('v') != bare {
                mismatches.push(format!("{name}: chelisup install {v} != {bare}"));
            }
        }
    }
    if !mismatches.is_empty() {
        return fail(
            format!(
                "workflow pin(s) disagree with reef.toml (={bare}): {}",
                mismatches.join("; ")
            ),
            "update every workflow CHELIS_TAG/CHELIS_VERSION to match the reef pin, or run `chelis reef conform bump`",
        );
    }
    if checked == 0 {
        return (
            Verdict::Manual,
            "no toolchain-installing workflow detected to cross-check".to_string(),
            "confirm the shell installs the toolchain in CI".to_string(),
            Vec::new(),
        );
    }
    pass()
}

fn check_pin_consistency_guard(ctx: &Ctx) -> Check {
    // The guard is wired iff CI runs the offline conformance/pin check.
    let ci = ctx
        .workflows
        .iter()
        .find(|(n, _)| n == "ci.yml")
        .map(|(_, b)| b);
    let Some(ci) = ci else {
        return fail(
            ".github/workflows/ci.yml is missing",
            "add a CI workflow wiring the conformance guard",
        );
    };
    if ci.contains("conform bump-check") || ci.contains("conform audit") {
        pass()
    } else {
        fail(
            "ci.yml does not run the offline pin/conformance guard",
            "add a step running `chelis reef conform bump-check --base origin/main` (and/or `conform audit`)",
        )
    }
}

fn check_toolchain_installer(ctx: &Ctx) -> Check {
    let uses_chelisup = ctx
        .workflows
        .iter()
        .any(|(_, b)| b.contains("chelisup") || b.contains("install-chelis"));
    if uses_chelisup {
        pass()
    } else {
        (
            Verdict::Manual,
            "could not confirm a pin-resolving installer (chelisup / install-chelis action)"
                .to_string(),
            "install the toolchain via chelisup; never hand-symlink a machine-global default"
                .to_string(),
            Vec::new(),
        )
    }
}

fn check_uv_python(ctx: &Ctx) -> Check {
    if ctx.exists("requirements.txt") || ctx.exists("poetry.lock") || ctx.exists("Pipfile") {
        return fail(
            "non-uv Python dependency manifest present (requirements.txt/poetry.lock/Pipfile)",
            "make any dep-bearing harness its own uv project (pyproject.toml + uv.lock)",
        );
    }
    pass()
}

fn check_chelis_surface(ctx: &Ctx) -> Check {
    let Some(surface) = ctx.read("docs/CHELIS_SURFACE.md") else {
        return fail(
            "docs/CHELIS_SURFACE.md is missing",
            "run `chelis reef conform init` / `sync`",
        );
    };
    if !surface.contains("@pin") || !surface.contains("@upstream") {
        return fail(
            "docs/CHELIS_SURFACE.md lacks @pin/@upstream capability markers",
            "mark each capability row @pin (usable now) or @upstream (next bump)",
        );
    }
    check_managed_block(
        ctx,
        &surface,
        "chelis-surface-header",
        "docs/CHELIS_SURFACE.md",
        None,
    )
}

/// Row 8 (§4): `docs/UPSTREAM_BUGS.md` exists, carries the four required
/// sections, and — the cite-by-number machine check added for chelis#739 —
/// every confidently parsed entry under §Actively blocking / §Tracking /
/// §Parked cites its bug as `chelis#NNN`, as a registry sibling's `<repo>#NNN`
/// (chelis#1270), or as a `docs/issue_drafts/<name>` draft path, never a prose
/// name. §4 makes this a MUST ("cite by number … never by a
/// prose name"): a prose-name citation is invisible to every mechanical audit,
/// which is the exact failure the contract's own §4 rationale cites (School
/// carried a "generic-callback-unification limit" through three docs and a
/// shipped PR while chelis#293 was already fixed in the pin it was built on).
///
/// **Entry granularity** is a documented heuristic: within a section, an *entry*
/// is a top-level markdown list item (`-`/`*`/`+`, or `N.`/`N)`, indented ≤3
/// spaces) or a sub-heading (any heading deeper than the section heading, i.e.
/// `###`+ under a `##` section). Nested/indented lines and prose paragraphs
/// belong to the entry above them; the `(none yet)` placeholder and blank lines
/// are not entries. §Archived is exempt entirely — its entries are closed
/// history, not live narrowings.
///
/// A section that holds non-placeholder content but no parseable entry at all is
/// reported `Manual` (its shape is not machine-decidable), never a mechanical
/// `Pass` — a mechanical `Pass` there would launder unreviewed prose as
/// conformant.
fn check_upstream_bugs(ctx: &Ctx) -> Check {
    let Some(bugs) = ctx.read("docs/UPSTREAM_BUGS.md") else {
        return fail(
            "docs/UPSTREAM_BUGS.md is missing",
            "add docs/UPSTREAM_BUGS.md with the four required sections",
        );
    };
    // Structural gate: each required section must appear *exactly once* as a
    // well-formed, section-level ATX heading. Both directions fail closed, and
    // both close a confirmed false-green (chelis#739 red team):
    //   - MISSING/malformed: a lenient `has_heading` (bare `starts_with('#')`)
    //     used to accept a malformed `##Actively blocking` (no space) that the
    //     strict `section_body` locator could not resolve, and the loop silently
    //     skipped that section's entries. The gate now uses the same
    //     `is_section_heading` predicate as the locator, so they cannot disagree.
    //   - DUPLICATED: `section_body` reads only the *first* matching heading's
    //     body (up to the next same-level heading — the duplicate), so entries
    //     under a second `## Tracking` were never citation-checked and audited
    //     green while rendering as a normal section to a human. A duplicated
    //     required section is a malformed doc; fail it here.
    let lines: Vec<&str> = bugs.lines().collect();
    let required = ["Actively blocking", "Tracking", "Parked", "Archived"];
    let mut missing: Vec<&str> = Vec::new();
    let mut duplicated: Vec<&str> = Vec::new();
    for s in required {
        match section_heading_indices(&lines, s).len() {
            0 => missing.push(s),
            1 => {}
            _ => duplicated.push(s),
        }
    }
    if !missing.is_empty() {
        return fail(
            format!(
                "docs/UPSTREAM_BUGS.md missing or malformed section heading(s): {}",
                missing.join(", ")
            ),
            "add each as a well-formed ATX heading (e.g. `## Actively blocking`, with a space after the `#`)",
        );
    }
    if !duplicated.is_empty() {
        return fail(
            format!(
                "docs/UPSTREAM_BUGS.md duplicated section heading(s): {}",
                duplicated.join(", ")
            ),
            "each required section must appear exactly once; a duplicated heading orphans the second body from the citation check, so merge them",
        );
    }

    // §4 cite-by-number check over the three *live* sections (§Archived exempt).
    let mut uncited: Vec<String> = Vec::new();
    let mut manual_sections: Vec<&str> = Vec::new();
    let mut manual_evidence: Vec<String> = Vec::new();
    for section in ["Actively blocking", "Tracking", "Parked"] {
        // Unreachable in practice — each live section passed the strict gate
        // above, so `section_body` resolves it here too. Kept as a fail-closed
        // guard: an unlocatable section is skipped, never silently passed.
        let Some(body) = section_body(&lines, section) else {
            continue;
        };
        let entries = parse_bug_entries(&body);
        if entries.is_empty() {
            if let Some((line, first)) = first_content_line(&body) {
                manual_sections.push(section);
                manual_evidence.push(format!(
                    "docs/UPSTREAM_BUGS.md:{line}: §{section} content {:?} is not a top-level list item or sub-heading; cannot machine-check its citation",
                    snippet(first),
                ));
            }
            continue;
        }
        for entry in &entries {
            if scan_citations(&entry.text).is_empty() && !cites_issue_draft(&entry.text) {
                uncited.push(format!(
                    "docs/UPSTREAM_BUGS.md:{}: §{section} entry {:?} cites no chelis#NNN, no registry sibling's <repo>#NNN, and no docs/issue_drafts/ path",
                    entry.line,
                    snippet(entry.head()),
                ));
            }
        }
    }

    // A Fail (a confidently-parsed uncited entry) dominates a Manual: it is the
    // actionable §4 violation. Surface the un-machine-checkable sections in the
    // same evidence so `--explain` still names them.
    if !uncited.is_empty() {
        // Counted from the uncited entries themselves, never by re-matching a
        // substring of the rendered evidence: the evidence wording is a
        // diagnostic, not a data channel, and a reworded message must not be
        // able to change the count.
        let n = uncited.len();
        let mut evidence = uncited;
        evidence.extend(manual_evidence);
        // Name the accepted repo set once, under `--explain`. An author whose
        // citation was rejected needs to know which repos resolve, and deriving
        // it from the registry keeps the message correct as the ecosystem grows.
        evidence.push(format!(
            "  accepted citation repos: {}",
            registry::citable_repos().collect::<Vec<_>>().join(", ")
        ));
        return fail_ex(
            format!(
                "docs/UPSTREAM_BUGS.md: {n} entr{} with a prose-name citation, not chelis#NNN / <sibling>#NNN / a docs/issue_drafts/ path (§4)",
                if n == 1 { "y" } else { "ies" },
            ),
            "cite every entry at the entry by `chelis#NNN`, by a registry sibling's `<repo>#NNN` (e.g. `nautilus#43`), or by a `docs/issue_drafts/<name>` draft path, never by a prose name (contract §4)",
            evidence,
        );
    }
    if !manual_sections.is_empty() {
        return manual_ex(
            format!(
                "docs/UPSTREAM_BUGS.md §{} ha{} content but no parseable entry (top-level list item or sub-heading); §4 cite-by-number cannot be machine-checked there",
                manual_sections.join(", §"),
                if manual_sections.len() == 1 {
                    "s"
                } else {
                    "ve"
                },
            ),
            "structure each bug as a top-level list item or sub-heading citing chelis#NNN / a registry sibling's <repo>#NNN / a draft path so the §4 cite-by-number rule is machine-checkable",
            manual_evidence,
        );
    }
    pass()
}

/// One parsed `docs/UPSTREAM_BUGS.md` entry: its 1-based start line (for
/// evidence) and its full text (start line plus its continuation lines, joined),
/// which is what the citation scan runs over so a `chelis#NNN` on a nested line
/// still covers its entry.
struct BugEntry {
    line: usize,
    text: String,
}

impl BugEntry {
    /// The entry's first line (the list-item or sub-heading line itself).
    fn head(&self) -> &str {
        self.text.lines().next().unwrap_or("")
    }
}

/// Required-section headings are document-level: the scaffold and every shell
/// (School included) write them as `##` (level 2). A *section-level* heading is
/// thus any ATX heading at level ≤ 2; deeper (`###`+) headings are entry
/// sub-headings, never section headings. Restricting section matching to this
/// ceiling keeps a `### Tracking …` entry sub-heading from being mistaken for
/// the §Tracking section (in body location or duplicate detection).
const SECTION_HEADING_MAX_LEVEL: usize = 2;

/// Whether `line` is a section-level heading (level ≤ 2) naming `needle`. Uses a
/// substring match, not equality, because real section headings carry trailing
/// context (School: `## Tracking (filed upstream, not blocking)`).
fn is_section_heading(line: &str, needle: &str) -> bool {
    heading_level(line).is_some_and(|lvl| lvl <= SECTION_HEADING_MAX_LEVEL) && line.contains(needle)
}

/// Indices of every section-level heading naming `needle`. A length > 1 means
/// the section is duplicated (a malformed doc that would orphan a body from the
/// citation check).
fn section_heading_indices(lines: &[&str], needle: &str) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| is_section_heading(l, needle))
        .map(|(i, _)| i)
        .collect()
}

/// The body of the markdown section whose section-level heading names `needle`,
/// from just after that heading to just before the next heading of
/// equal-or-shallower level (or EOF) — so a `###`+ sub-heading stays *inside* a
/// `##` section rather than closing it. Uses the *first* section-level match;
/// the structural gate rejects a duplicated section before this is reached. Each
/// body line is paired with its 1-based file line number. `None` if no
/// section-level heading names `needle`.
fn section_body<'a>(lines: &[&'a str], needle: &str) -> Option<Vec<(usize, &'a str)>> {
    let start = lines.iter().position(|l| is_section_heading(l, needle))?;
    let level = heading_level(lines[start]).unwrap_or(usize::MAX);
    let mut body = Vec::new();
    for (i, l) in lines.iter().enumerate().skip(start + 1) {
        if heading_level(l).is_some_and(|lvl| lvl <= level) {
            break;
        }
        body.push((i + 1, *l));
    }
    Some(body)
}

/// Partition a section body into entries. An entry starts at each entry-start
/// line and runs until the next entry-start line (or the body end), so its
/// continuation/nested lines travel with it.
fn parse_bug_entries(body: &[(usize, &str)]) -> Vec<BugEntry> {
    let starts: Vec<usize> = body
        .iter()
        .enumerate()
        .filter(|(_, (_, l))| is_bug_entry_start(l))
        .map(|(pos, _)| pos)
        .collect();
    let mut entries = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(body.len());
        let text = body[s..end]
            .iter()
            .map(|(_, l)| *l)
            .collect::<Vec<_>>()
            .join("\n");
        entries.push(BugEntry {
            line: body[s].0,
            text,
        });
    }
    entries
}

/// Whether `line` begins a bug entry: a sub-heading (any ATX heading — the body
/// already excludes headings at or above the section level, so every heading it
/// contains is a sub-heading) or a top-level list item.
fn is_bug_entry_start(line: &str) -> bool {
    heading_level(line).is_some() || is_top_list_item(line)
}

/// The first non-blank, non-`(none yet)` line of a section body and its 1-based
/// line number, or `None` when the section is empty/placeholder-only.
fn first_content_line<'a>(body: &[(usize, &'a str)]) -> Option<(usize, &'a str)> {
    body.iter().copied().find(|(_, l)| {
        let t = l.trim();
        !t.is_empty() && t != "(none yet)"
    })
}

/// The ATX heading level of `line` (count of leading `#`), or `None` if it is
/// not a heading. A real ATX heading has a space/tab (or nothing) after the run
/// of `#`, so a bare `#316` is text, not a level-1 heading.
fn heading_level(line: &str) -> Option<usize> {
    let t = line.trim_start();
    let hashes = t.chars().take_while(|&c| c == '#').count();
    if hashes == 0 {
        return None;
    }
    let after = &t[hashes..];
    (after.is_empty() || after.starts_with(' ') || after.starts_with('\t')).then_some(hashes)
}

/// Whether `line` starts a top-level markdown list item — unordered
/// (`-`/`*`/`+`) or ordered (`N.`/`N)`) — indented no more than 3 spaces. A
/// more-indented item is a nested continuation of the entry above it, so it is
/// not itself an entry.
fn is_top_list_item(line: &str) -> bool {
    let indent = line.len() - line.trim_start().len();
    if indent > 3 {
        return false;
    }
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix(['-', '*', '+']) {
        return rest.starts_with(' ') || rest.starts_with('\t') || rest.is_empty();
    }
    let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 {
        let after = &t[digits..];
        return after.starts_with('.') || after.starts_with(')');
    }
    false
}

/// Whether `text` cites a parked issue draft by path — a reference to a file
/// under `docs/issue_drafts/`, the §4-sanctioned alternative to a `chelis#NNN`
/// number for a not-yet-filed bug. The directory prefix must be followed by a
/// filename character, so a bare mention of the directory itself does not count.
fn cites_issue_draft(text: &str) -> bool {
    const PREFIX: &str = "docs/issue_drafts/";
    text.match_indices(PREFIX).any(|(i, _)| {
        text[i + PREFIX.len()..]
            .chars()
            .next()
            .is_some_and(|c| !c.is_whitespace() && !matches!(c, ')' | ']' | '/'))
    })
}

/// A one-line, length-capped snippet of `line` for an evidence message.
fn snippet(line: &str) -> String {
    let t = line.trim();
    let capped: String = t.chars().take(60).collect();
    if t.chars().count() > 60 {
        format!("{capped}…")
    } else {
        capped
    }
}

/// Row 9: narrowing-coverage (the offline half of the §4 staleness audit). Every
/// citation at a live `src/**/*.ch` narrowing — `chelis#NNN` or a registry
/// sibling's `<repo>#NNN` (chelis#1270) — must be covered by a `tests_blocked/`
/// probe, a `docs/UPSTREAM_BUGS.md` entry, or the `tests_blocked/README.md`
/// can't-be-probed list. Both sides run the same scanner, so the widened grammar
/// widens cite and coverage together: a `nautilus#43` narrowing is now visible to
/// the audit instead of reading as uncited prose, and it owes the same coverage
/// an upstream cite owes.
fn check_narrowing_coverage(ctx: &Ctx) -> Check {
    let cited = collect_citations_in_dir(&ctx.root.join("src"));
    if cited.is_empty() {
        return pass();
    }
    let blocked_text = concat_dir_texts(&ctx.root.join("tests_blocked"));
    let upstream = ctx.read("docs/UPSTREAM_BUGS.md").unwrap_or_default();
    let readme = ctx.read("tests_blocked/README.md").unwrap_or_default();
    let corpus = format!("{blocked_text}\n{upstream}\n{readme}");

    // Group every uncovered cite by token, preserving each site so `--explain`
    // can name where the orphan is cited (chelis#654: the diagnostic used to
    // list only the tokens, forcing an empirical bisect to localize them).
    let mut orphans: BTreeMap<String, Vec<(PathBuf, usize)>> = BTreeMap::new();
    for c in cited {
        if !corpus_covers(&corpus, &c.token) {
            orphans.entry(c.token).or_default().push((c.file, c.line));
        }
    }
    if orphans.is_empty() {
        return pass();
    }

    let mut evidence = Vec::new();
    for (token, sites) in &orphans {
        for (file, line) in sites {
            let rel = file.strip_prefix(&ctx.root).unwrap_or(file);
            evidence.push(format!("{token} cited at {}:{}", rel.display(), line));
        }
        evidence.push(coverage_evidence(token, &blocked_text, &upstream, &readme));
    }
    let tokens: Vec<&str> = orphans.keys().map(String::as_str).collect();
    fail_ex(
        format!("uncovered narrowing citation(s): {}", tokens.join(", ")),
        "add a tests_blocked/ probe, a docs/UPSTREAM_BUGS.md entry, or a tests_blocked/README.md can't-be-probed note for each",
        evidence,
    )
}

/// One `--explain` line naming which coverage sources were checked for an orphan
/// `token` and why each failed, with a near-miss hint when the issue number
/// appears in a non-canonical (bare-`#NNN`) form — the exact trap in
/// chelis#654, where a space-form `chelis #316` (now matched, chelis#652) or a
/// bare `#316` left the fix message ("add an UPSTREAM_BUGS entry") misleading.
/// The number is split off the token rather than trimmed against a literal
/// `chelis#` prefix, so the hint stays correct for a registry sibling's token
/// (chelis#1270).
fn coverage_evidence(token: &str, blocked: &str, upstream: &str, readme: &str) -> String {
    let num = token.split_once('#').map_or(token, |(_, n)| n);
    let bare = format!("#{num}");
    let upstream_note = if corpus_covers(upstream, token) {
        "covered".to_string()
    } else if upstream.contains(&bare) {
        format!("mentions {bare} but not as a `{token}` token")
    } else {
        "no entry".to_string()
    };
    format!(
        "  coverage checked: tests_blocked/ ({}), docs/UPSTREAM_BUGS.md ({}), tests_blocked/README.md ({})",
        if corpus_covers(blocked, token) {
            "covered"
        } else {
            "no probe"
        },
        upstream_note,
        if corpus_covers(readme, token) {
            "covered"
        } else {
            "no note"
        },
    )
}

fn check_issue_drafts(ctx: &Ctx) -> Check {
    if ctx.exists("docs/issue_drafts") {
        pass()
    } else {
        // SHOULD: a Fail here does not gate (see must_failures).
        fail(
            "docs/issue_drafts/ convention not present (SHOULD)",
            "add docs/issue_drafts/README.md for parked filings",
        )
    }
}

fn check_tests_neg(ctx: &Ctx) -> Check {
    if !dir_has_ch(&ctx.root.join("tests_neg")) {
        return fail(
            "tests_neg/ is missing or has no .ch cases",
            "add tests_neg/<area>/<name>.ch + .expect sidecars",
        );
    }
    if !ctx
        .workflows
        .iter()
        .any(|(_, b)| b.contains("--expect neg") || b.contains("--expect=neg"))
    {
        return fail(
            "no CI step runs `chelis test tests_neg --expect neg`",
            "wire the negative suite into ci.yml",
        );
    }
    pass()
}

fn check_tests_blocked(ctx: &Ctx) -> Check {
    let has_blocker = dir_has_ch(&ctx.root.join("tests_blocked"))
        || !collect_citations_in_dir(&ctx.root.join("src")).is_empty();
    if !has_blocker {
        return (
            Verdict::Na,
            "no open upstream blocker with an expressible reproducer".to_string(),
            String::new(),
            Vec::new(),
        );
    }
    if !dir_has_ch(&ctx.root.join("tests_blocked")) {
        return fail(
            "an upstream blocker is cited but tests_blocked/ has no probe",
            "add tests_blocked/<area>/<name>.ch + .expect for each expressible blocker",
        );
    }
    if !ctx
        .workflows
        .iter()
        .any(|(_, b)| b.contains("--expect blocked") || b.contains("--expect=blocked"))
    {
        return fail(
            "no CI step runs `chelis test tests_blocked --expect blocked`",
            "wire the blocked-probe suite into ci.yml",
        );
    }
    pass()
}

fn check_agents_heading(ctx: &Ctx, heading: &str) -> Check {
    match &ctx.agents_md {
        Some(a) if has_heading(a, heading) => pass(),
        Some(_) => fail(
            format!("AGENTS.md lacks a {heading} section"),
            format!("add a {heading} section (may be a managed block)"),
        ),
        None => fail("AGENTS.md is missing", "run `chelis reef conform init`"),
    }
}

fn check_vendored_skills(ctx: &Ctx) -> Check {
    let skills_dir = ctx.root.join("agent-skills");
    let mut problems = Vec::new();
    // The control surface is read from the PARSED manifest, so the answer does not
    // depend on how it was spelled. An unparseable manifest fails here rather
    // than reading as "declares nothing": §8 cannot be checked against a file
    // this tool cannot read, and a silent pass on a MUST row is the failure this
    // whole row exists to prevent.
    match &ctx.conform {
        None => {}
        Some(Err(parse_error)) => {
            return fail(
                format!(
                    "reef.toml does not parse as TOML, so the `conform` control surface cannot be \
                     read: {parse_error}"
                ),
                "fix the manifest. Until it parses, no tool can tell whether this shell declares \
                 a conformance control, so §8 cannot be audited and this row fails closed.",
            );
        }
        Some(Ok(decl)) if !decl.is_clean() => {
            return fail(
                format!(
                    "reef.toml declares conformance control(s) contract §8 does not define: {}. \
                     The recognized declarations are `conform.local_skills` and \
                     `conform.excluded_skills`, both arrays of strings.",
                    decl.findings().join(", ")
                ),
                "use `[conform] local_skills = [...]` for shell-owned additions and \
                 `excluded_skills = [...]` for exact embedded shared-skill removals. Remove or \
                 correct every other declaration before syncing.",
            );
        }
        Some(Ok(_)) => {}
    }
    // A `local_skills` entry may not shadow a shared skill — that would let a
    // shell "own" (and silently fork) toolchain-managed content (chelis#651).
    for local in ctx.local_skills() {
        if skills::SHARED_SKILLS.contains(&local.as_str()) {
            problems.push(format!(
                "{local}: [conform] local_skills may not name a shared skill"
            ));
        }
    }
    // Exclusions are exact names from this pinned toolchain. Failing unknown
    // names catches both typos and a skill removed or renamed upstream.
    for excluded in ctx.excluded_skills() {
        if !skills::SHARED_SKILLS.contains(&excluded.as_str()) {
            problems.push(format!(
                "{excluded}: [conform] excluded_skills does not name a shared skill in this toolchain"
            ));
        }
    }
    for (name, body) in skills::EMBEDDED_SKILLS {
        let skill_dir = skills_dir.join(name);
        let path = skill_dir.join("SKILL.md");
        if ctx.excluded_skills().iter().any(|s| s == name) {
            if skill_dir.exists() {
                problems.push(format!(
                    "{name}: present but declared in [conform] excluded_skills"
                ));
            }
            continue;
        }
        match read_opt(&path) {
            None => problems.push(format!("{name}: missing")),
            Some(live) => {
                // A trailing shell-local block (chelis#653) is shell-owned. §8
                // derives the expected managed span by applying its validated
                // exclusions to the embedded body, then byte-checks that span.
                let (managed, block) = crate::scaffold::split_shell_local(&live);
                match crate::scaffold::apply_shell_local_exclusions(body, block) {
                    Ok(expected) if managed.trim_end() != expected.trim_end() => {
                        problems.push(format!("{name}: forked/stale"));
                    }
                    Err(why) => problems.push(format!("{name}: {why}")),
                    Ok(_) => {}
                }
            }
        }
    }
    // Reverse direction: the shell owns *zero* extra shared-skill content, so an
    // addition is drift too. Enumerate the on-disk tree and flag any skill dir
    // outside the pinned set (unless declared in `[conform] local_skills`), or
    // any file beyond `SKILL.md` inside a pinned dir (materialize never creates
    // these, so their presence is a fork/leftover).
    if let Ok(entries) = std::fs::read_dir(&skills_dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                if skills::SHARED_SKILLS.contains(&name.as_str()) {
                    if ctx.excluded_skills().iter().any(|s| s == &name) {
                        // Presence was already reported above; do not inspect a
                        // skill whose configured state is absence.
                        continue;
                    }
                    if let Ok(inner) = std::fs::read_dir(e.path()) {
                        for f in inner.flatten() {
                            let fname = f.file_name().to_string_lossy().into_owned();
                            if fname != "SKILL.md" {
                                problems.push(format!("{name}/{fname}: unexpected skill file"));
                            }
                        }
                    }
                } else if ctx.local_skills().iter().any(|s| s == &name) {
                    // Repo-local domain skill (chelis#651): shell-owned, exempt.
                } else {
                    problems.push(format!(
                        "{name}: not a pinned skill (remove, or declare in [conform] local_skills)"
                    ));
                }
            } else if name != "UPSTREAM.toml" {
                problems.push(format!("{name}: unexpected file in agent-skills/"));
            }
        }
    }
    for mirror in [".claude/skills", ".codex/skills"] {
        if let Err(why) = identical_trees(&skills_dir, &ctx.root.join(mirror)) {
            problems.push(format!("{mirror}: {why}"));
        }
    }
    if !problems.is_empty() {
        return fail(
            format!(
                "vendored skills drifted from the pinned set: {}",
                problems.join(", ")
            ),
            "run `chelis reef conform sync` to re-materialize agent-skills/, .claude/skills/, \
             and .codex/skills/ from the toolchain. \
             Configure shell-owned additions with `[conform] local_skills` and embedded removals \
             with `[conform] excluded_skills`; use a trailing `<!-- shell-local:begin -->` block \
             to retain and amend a shared skill, with `shell-local:exclude` selectors for \
             inherited sections that should be omitted.",
        );
    }
    pass()
}

fn check_parity_harness(ctx: &Ctx) -> Check {
    if !ctx.exists("parity") {
        return (
            Verdict::Na,
            "no parity/ harness (shell does not validate against external oracles)".to_string(),
            String::new(),
            Vec::new(),
        );
    }
    if !ctx.exists("parity/pyproject.toml") {
        return fail(
            "parity/ exists but is not its own uv project (no parity/pyproject.toml)",
            "make the parity harness a uv project with checked-in goldens",
        );
    }
    (
        Verdict::Manual,
        "parity harness present; oracle-guard correctness needs review".to_string(),
        "confirm CI guards keep oracle libs out of shell code".to_string(),
        Vec::new(),
    )
}

fn check_chelis_src(ctx: &Ctx) -> Check {
    // Prefer the registry's authoritative `links_chelis_crates` flag when the
    // shell is known (its name matches a registry entry). The registry is the
    // source of truth for whether row 18 applies; the Cargo.toml substring scan
    // is only a fallback for shells not yet registered (freshly scaffolded, new,
    // or under test).
    let links = match ctx.shell_name.as_deref().and_then(registry::shell) {
        Some(entry) => entry.links_chelis_crates,
        None => ctx.cargo_toml.as_deref().is_some_and(links_chelis_crates),
    };
    if !links {
        return (
            Verdict::Na,
            "shell does not link chelis crates as Cargo path deps".to_string(),
            String::new(),
            Vec::new(),
        );
    }
    match &ctx.reef_toml {
        Some(t) if t.contains("[chelis-src]") => pass(),
        _ => fail(
            "shell links chelis crates but reef.toml has no [chelis-src] section",
            "add [chelis-src] with crates + pin_commit and wire `chelis reef src check` into the local gate",
        ),
    }
}

/// Heuristic: does this `Cargo.toml` link chelis crates from a local source (a
/// path/git dep on a `chelis-*` crate, or a `[patch]` onto chelis)? The audit
/// receives only a path, not the shell name, so it cannot consult the
/// registry's authoritative `links_chelis_crates` flag — this is a best-effort
/// trigger, broadened beyond the single `../chelis/crates` spelling to catch
/// absolute paths and patch sections. A plain registry dep like
/// `chelis-std = "=0.14.0"` (no `path`/`git`) is deliberately not matched.
fn links_chelis_crates(cargo_toml: &str) -> bool {
    if cargo_toml.contains("chelis/crates") || cargo_toml.contains("../chelis\"") {
        return true;
    }
    if cargo_toml.contains("[patch") && cargo_toml.contains("chelis") {
        return true;
    }
    cargo_toml.lines().any(|line| {
        let l = line.trim_start();
        (l.starts_with("chelis-") || l.starts_with("\"chelis-"))
            && (l.contains("path") || l.contains("git"))
    })
}

/// Shared managed-block freshness check: present, stamped to the reef pin,
/// untampered (integrity vs its own fence hash), and — when stamped for the
/// auditing version — byte-equal to the embedded canonical body.
fn check_managed_block(
    ctx: &Ctx,
    doc: &str,
    id: &str,
    file: &str,
    expected_body: Option<&str>,
) -> Check {
    let Some(block) = managed_block::find(doc, id) else {
        return fail(
            format!("{file} has no managed block `{id}`"),
            "run `chelis reef conform sync`",
        );
    };
    if !block.integrity_ok() {
        return fail(
            format!("{file} managed block `{id}` was hand-edited (hash mismatch)"),
            "run `chelis reef conform sync` (edit the upstream canonical text, not the block)",
        );
    }
    if let Some(pin) = &ctx.reef_pin {
        let bare = pin.trim_start_matches('=');
        if block.version != bare {
            return fail(
                format!(
                    "{file} managed block `{id}` is stamped chelis@{} but reef.toml pins ={bare}",
                    block.version
                ),
                "run `chelis reef conform sync` to restamp to the pin",
            );
        }
    }
    // Body must equal the embedded canonical text for this block id. This is
    // the check that catches a *stale-but-self-consistent* block: `integrity_ok`
    // only proves the body matches its own fence hash, so a hand-edit that also
    // recomputes the stamp (or a genuine within-version drift) passes it. We can
    // only compare against canonical when the block is stamped for the version
    // this binary embeds; a stale shell audited by a newer HEAD canary
    // (`block.version != AUDITOR_VERSION`) is spared here — no historical bodies
    // are embedded — but the version-stamp and integrity checks still apply.
    if block.version == AUDITOR_VERSION
        && let Some(canon) = expected_body.or_else(|| canonical::body(id))
        && !block.matches_canonical(canon)
    {
        return fail(
            format!(
                "{file} managed block `{id}` body differs from the canonical upstream text for chelis@{AUDITOR_VERSION}"
            ),
            "run `chelis reef conform sync` (do not hand-edit inside the fences)",
        );
    }
    pass()
}

// ---------------------------------------------------------------- helpers

fn read_opt(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn identical_trees(expected: &Path, actual: &Path) -> Result<(), String> {
    let expected_entries = tree_inventory(expected)?;
    let actual_entries = tree_inventory(actual)?;
    if expected_entries == actual_entries {
        Ok(())
    } else {
        Err("is not a real, byte-identical copy of agent-skills/".to_string())
    }
}

fn tree_inventory(root: &Path) -> Result<BTreeMap<PathBuf, Option<Vec<u8>>>, String> {
    let metadata =
        std::fs::symlink_metadata(root).map_err(|e| format!("missing or unreadable ({e})"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("must be a real directory, not a symlink or file".to_string());
    }
    let mut inventory = BTreeMap::new();
    collect_tree(root, root, &mut inventory)?;
    Ok(inventory)
}

fn collect_tree(
    root: &Path,
    directory: &Path,
    inventory: &mut BTreeMap<PathBuf, Option<Vec<u8>>>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|e| format!("cannot read {} ({e})", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {} ({e})", directory.display()))?;
        let path = entry.path();
        let rel = path
            .strip_prefix(root)
            .expect("walk remains below root")
            .to_path_buf();
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|e| format!("cannot inspect {} ({e})", path.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("contains symlink {}", rel.display()));
        }
        if metadata.is_dir() {
            inventory.insert(rel, None);
            collect_tree(root, &path, inventory)?;
        } else if metadata.is_file() {
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("cannot read {} ({e})", path.display()))?;
            inventory.insert(rel, Some(bytes));
        } else {
            return Err(format!("contains unsupported entry {}", rel.display()));
        }
    }
    Ok(())
}

/// `CLAUDE.md` must be a symlink whose target is (or resolves to) `AGENTS.md`.
fn claude_is_symlink_to_agents(root: &Path) -> bool {
    let claude = root.join("CLAUDE.md");
    let Ok(meta) = std::fs::symlink_metadata(&claude) else {
        return false;
    };
    if !meta.file_type().is_symlink() {
        return false;
    }
    match std::fs::read_link(&claude) {
        Ok(target) => target.file_name().and_then(|s| s.to_str()) == Some("AGENTS.md"),
        Err(_) => false,
    }
}

/// Parse the exact `compiler = "=X.Y.Z"` pin from `reef.toml`, returning the
/// value including the leading `=` (e.g. `"=0.14.0"`).
///
/// This is kept in lockstep with the chelisup shim's resolver
/// (`chelisup::resolve::compiler_pin`) so conformance and the version manager
/// agree on *which* value is the pin and *whether it is usable*:
/// - only the top-level or `[package]` `compiler` key is read — a `compiler`
///   under some other table is ignored, exactly as the shim reads
///   `package.compiler` with a root-table fallback;
/// - the version must be a strict `X.Y.Z` the shim can actually install
///   ([`is_installable_version`], mirroring `validate_install_version`).
///
/// Conformance is *stricter* than the shim on one point — it requires the
/// exact-pin leading `=`, because a conformant shell must pin exactly — but it
/// is never *looser*: a value the shim would reject (unsafe path component,
/// non-`X.Y.Z`, pre-release) must never audit green, or conform would bless a
/// toolchain the shim cannot resolve.
pub fn parse_compiler_pin(toml: &str) -> Option<String> {
    let raw = toml_string_field(toml, "compiler")?;
    // Require the exact-pin leading `=` (conform is stricter than the shim
    // here) and a version the installer can actually fetch.
    let ver = raw.strip_prefix('=')?;
    is_installable_version(ver).then(|| format!("={ver}"))
}

/// Parse the shell's package `name` from `reef.toml` (root or `[package]`),
/// used to look the shell up in the registry.
fn parse_package_name(toml: &str) -> Option<String> {
    toml_string_field(toml, "name")
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Read a string field `key` from the root or `[package]` table of `toml`,
/// returning the raw (unquoted) value. A same-named key under any other table
/// is ignored — this reproduces the section scoping the chelisup shim's
/// TOML-based resolver applies (`package.compiler` with a root-table fallback),
/// without pulling in a TOML dependency.
fn toml_string_field<'a>(toml: &'a str, key: &str) -> Option<&'a str> {
    let mut in_scope = true; // root table, before the first `[header]`
    for line in toml.lines() {
        let t = line.trim();
        if let Some(header) = t.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
            in_scope = header.trim() == "package";
            continue;
        }
        if !in_scope {
            continue;
        }
        let Some(rest) = t.strip_prefix(key) else {
            continue;
        };
        // The key itself, not `<key>_extra`/`<key>-x`: next non-space is `=`.
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('"') else {
            continue;
        };
        let end = rest.find('"')?;
        return Some(&rest[..end]);
    }
    None
}

/// Whether `v` is a strict `X.Y.Z` version the chelisup installer accepts.
///
/// Delegates to the shared [`chelis_version::is_strict_semver`] — the single
/// source of truth also used by `chelisup::version::validate_install_version`,
/// so conform can never bless a pin the shim would reject.
pub fn is_installable_version(v: &str) -> bool {
    chelis_version::is_strict_semver(v)
}

fn read_workflows(root: &Path) -> Vec<(String, String)> {
    let dir = root.join(".github/workflows");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            let is_yaml = p
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e == "yml" || e == "yaml");
            if is_yaml
                && let (Some(name), Some(body)) =
                    (p.file_name().and_then(|s| s.to_str()), read_opt(&p))
            {
                out.push((name.to_string(), body));
            }
        }
    }
    out.sort();
    out
}

fn workflow_installs_toolchain(body: &str) -> bool {
    body.contains("CHELIS_VERSION")
        || body.contains("CHELIS_TAG")
        || body.contains("chelisup install")
        || body.contains("install-chelis")
}

/// Extract a `KEY: value` env value (first occurrence), tolerating quotes.
fn extract_env(body: &str, key: &str) -> Option<String> {
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(key)
            && let Some(v) = rest.trim_start().strip_prefix(':')
        {
            let v = v.trim().trim_matches('"').trim_matches('\'');
            if !v.is_empty() && !v.starts_with("${{") {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Extract the version argument of each `chelisup install <ver>` occurrence,
/// tolerating surrounding quotes. Only tokens that look like a version (start
/// with a digit or `v`) are returned, so flags like `--force` are ignored.
fn extract_chelisup_install_versions(body: &str) -> Vec<String> {
    let needle = "chelisup install ";
    body.match_indices(needle)
        .filter_map(|(i, _)| {
            let rest = body[i + needle.len()..].trim_start_matches(['"', '\'']);
            let ver: String = rest
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '"' && *c != '\'')
                .collect();
            (ver.chars().next()).and_then(|c| (c.is_ascii_digit() || c == 'v').then_some(ver))
        })
        .collect()
}

fn dir_has_ch(dir: &Path) -> bool {
    walk_files(dir).any(|p| p.extension().and_then(|s| s.to_str()) == Some("ch"))
}

/// One `chelis#NNN` citation and the source site it appears at (for `--explain`).
struct Citation {
    token: String,
    file: PathBuf,
    line: usize,
}

/// Collect `chelis#NNN` citations appearing in any `.ch` file under `dir`, each
/// paired with its source site (file + 1-based line).
fn collect_citations_in_dir(dir: &Path) -> Vec<Citation> {
    let mut out = Vec::new();
    for p in walk_files(dir) {
        if p.extension().and_then(|s| s.to_str()) != Some("ch") {
            continue;
        }
        if let Some(text) = read_opt(&p) {
            for (token, offset) in scan_citations(&text) {
                out.push(Citation {
                    token,
                    file: p.clone(),
                    line: line_of(&text, offset),
                });
            }
        }
    }
    out
}

/// 1-based line number of the byte `offset` within `text`.
fn line_of(text: &str, offset: usize) -> usize {
    text[..offset].bytes().filter(|&b| b == b'\n').count() + 1
}

/// Whether `corpus` covers `citation` as a whole `chelis#NNN` token. The corpus
/// is scanned with the same whitespace-tolerant scanner as the `src/` side, so a
/// coverage entry written `chelis #316` (with a space) still covers a `chelis#316`
/// cite (chelis#652). Whole-token by construction: [`scan_citations`] consumes all
/// contiguous digits, so `chelis#293` never spuriously covers an orphan `chelis#29`.
fn corpus_covers(corpus: &str, citation: &str) -> bool {
    scan_citations(corpus)
        .iter()
        .any(|(tok, _)| tok == citation)
}

/// Scan `text` for narrowing citations, tolerating inline whitespace between the
/// repo name, `#`, and the number, so `chelis#316`, `chelis #316`, and
/// `chelis # 316` all normalize to the canonical token `chelis#316`
/// (chelis#652 — a space-form cite/coverage entry must not read as uncovered).
/// Returns each canonical `<repo>#NNN` token paired with the byte offset where
/// the match starts, in source order (the offset feeds `--explain` site
/// reporting, chelis#654). Newlines are NOT tolerated between the parts, so a
/// sentence-final repo name followed by an unrelated `#heading` on the next line
/// is not a false match.
///
/// **The repo may be the upstream monorepo or any registry shell** (chelis#1270):
/// `nautilus#43` is a citation, `torch#43` is not, and a bare `#43` is not. A
/// cascade wave makes a sibling's issue the literal blocking artifact for most of
/// the ecosystem, and registry membership gives that reference exactly the
/// liveness the cite-by-number rule is buying.
///
/// The scan is anchored on the `#` and reads the repo name **leftward** as the
/// maximal run of `[A-Za-z0-9_-]`. That anchoring is what keeps the widened
/// grammar honest in both directions:
///   - `hello-chelis#43` yields `hello-chelis#43` and never *also* a phantom
///     `chelis#43`. A left-unanchored scan for the substring `chelis` would emit
///     both, silently manufacturing an upstream citation out of a sibling one.
///   - `chelischelis#5` yields nothing, because that maximal run matches no
///     registered repo. (The previous scanner matched it as `chelis#5`.)
///
/// **The whitespace tolerance is upstream-only, and that asymmetry is
/// deliberate.** It exists for chelis#652, where shells had written upstream
/// cites as `chelis #316`. Extending it to the registry made ordinary English
/// prose scan as citations, because four shells are also common nouns: a `.ch`
/// comment reading `the school #1 priority is the hull #3 mesh` produced two
/// "uncovered narrowing citations" and turned a conformant shell red. A sibling
/// therefore requires strict `<repo>#NNN` adjacency, with no whitespace on
/// either side of the `#`; only `chelis` keeps the spaced forms.
///
/// A `Chelis-Lang/` prefix, or a repo URL whose fragment is the issue number
/// (`https://github.com/Chelis-Lang/chelis#1270`), still resolves: `/` is not a
/// repo character, so the leftward run stops at it and leaves the bare name. A
/// path-style issue or PR URL (`.../chelis/issues/43`, `.../pull/43`) is NOT a
/// citation: it carries no `#`, so there is nothing for this scan to anchor on.
fn scan_citations(text: &str) -> Vec<(String, usize)> {
    fn is_inline_ws(b: u8) -> bool {
        b == b' ' || b == b'\t'
    }
    fn is_repo_char(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
    }
    /// The maximal repo-char run ending at `end`, as `(name, start)`.
    fn run_ending_at(text: &str, end: usize) -> Option<(&str, usize)> {
        let bytes = text.as_bytes();
        let mut start = end;
        while start > 0 && is_repo_char(bytes[start - 1]) {
            start -= 1;
        }
        (start < end).then(|| (&text[start..end], start))
    }

    let mut out = Vec::new();
    let bytes = text.as_bytes();
    for (hash, _) in text.match_indices('#') {
        // The repo name, to the left. Strict adjacency first; that is the only
        // form a registry sibling may take.
        let mut repo = run_ending_at(text, hash).filter(|(n, _)| registry::is_citable_repo(n));
        // Upstream-only whitespace tolerance (chelis#652): `chelis #316`.
        if repo.is_none() {
            let mut w = hash;
            while w > 0 && is_inline_ws(bytes[w - 1]) {
                w -= 1;
            }
            if w < hash {
                repo = run_ending_at(text, w).filter(|(n, _)| *n == registry::UPSTREAM_REPO);
            }
        }
        let Some((name, start)) = repo else {
            continue;
        };
        // The issue number, to the right. The same upstream-only rule applies:
        // `chelis # 316` normalizes, `coral # 27` does not.
        let mut j = hash + 1;
        if name == registry::UPSTREAM_REPO {
            while j < bytes.len() && is_inline_ws(bytes[j]) {
                j += 1;
            }
        }
        let num_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j == num_start {
            continue;
        }
        out.push((format!("{name}#{}", &text[num_start..j]), start));
    }
    out
}

fn concat_dir_texts(dir: &Path) -> String {
    let mut out = String::new();
    for p in walk_files(dir) {
        if let Some(text) = read_opt(&p) {
            out.push_str(&text);
            out.push('\n');
        }
    }
    out
}

/// Recursively yield files under `dir` (empty iterator if `dir` is absent).
fn walk_files(dir: &Path) -> impl Iterator<Item = PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    let mut files = Vec::new();
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                // Use the entry's own type, which does *not* traverse symlinks,
                // so a symlinked directory is never descended into. An offline,
                // hermetic audit must not be loopable by a symlink cycle.
                match e.file_type() {
                    Ok(ft) if ft.is_symlink() => {} // neither descend nor read
                    Ok(ft) if ft.is_dir() => stack.push(p),
                    _ => files.push(p),
                }
            }
        }
    }
    files.into_iter()
}

/// The pin used for `since_version` gating: the shell's pin, floored at the
/// contract baseline. Flooring is what stops a pre-baseline pin (e.g. `=0.1.0`)
/// from gating every baseline row out to `Na` — the whole-audit bypass.
fn gate_version(pin: &str) -> String {
    if version_ge(pin, CONTRACT_BASELINE_VERSION) {
        pin.to_string()
    } else {
        CONTRACT_BASELINE_VERSION.to_string()
    }
}

/// `a >= b` on `X.Y.Z[-pre]` (leading `=`/`v` tolerated). A pre-release ranks
/// immediately below its release (`0.14.0-rc1 < 0.14.0`).
fn version_ge(a: &str, b: &str) -> bool {
    match (parse_semver(a), parse_semver(b)) {
        (Some(a), Some(b)) => a >= b,
        _ => true, // if either is unparseable, do not gate the row out
    }
}

/// Parse to `(major, minor, patch, release_rank)` where `release_rank` is `1`
/// for a final release and `0` for a pre-release (`-rc1`, `-alpha`, …), so a
/// pre-release orders just below its release. Pre-releases are not ordered
/// among themselves (all rank `0`); that resolution is unneeded for gating.
fn parse_semver(v: &str) -> Option<(u64, u64, u64, u8)> {
    let v = v.trim().trim_start_matches('=').trim_start_matches('v');
    let mut it = v.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    let patch_raw = it.next().unwrap_or("0");
    let digits: String = patch_raw
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let patch = digits.parse().unwrap_or(0);
    let is_release = digits.len() == patch_raw.len();
    Some((major, minor, patch, u8::from(is_release)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn citation_extraction() {
        let cites = |t: &str| {
            scan_citations(t)
                .into_iter()
                .map(|(c, _)| c)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            cites("fail(...) // blocked on chelis#293 and chelis#345x"),
            vec!["chelis#293", "chelis#345"]
        );
        assert!(cites("no citation, chelis# alone").is_empty());
        // A space between `chelis`, `#`, and the number canonicalizes to the
        // no-space token (chelis#652). A newline in between does NOT match.
        assert_eq!(cites("blocked on chelis #316"), vec!["chelis#316"]);
        assert_eq!(cites("see chelis # 42 here"), vec!["chelis#42"]);
        assert!(cites("chelis\n#316").is_empty());
    }

    /// The grammar widened to registry siblings (chelis#1270): the literal
    /// blocking artifact in a cascade wave is another shell's issue, and a
    /// registry entry is exactly as live and checkable as an upstream one.
    #[test]
    fn registry_sibling_citations_are_scanned() {
        let cites = |t: &str| {
            scan_citations(t)
                .into_iter()
                .map(|(c, _)| c)
                .collect::<Vec<_>>()
        };
        assert_eq!(cites("blocked on nautilus#43"), vec!["nautilus#43"]);
        assert_eq!(cites("waiting for coral#27"), vec!["coral#27"]);
        // A sibling requires strict adjacency; see
        // `prose_containing_a_shell_name_before_a_number_is_not_a_citation`.
        assert!(cites("waiting for coral #27").is_empty());
        // Every registered shell, so adding a shell to REGISTRY widens the
        // grammar with it and this test proves the coupling rather than
        // spot-checking two names.
        for s in registry::REGISTRY {
            let text = format!("blocked on {}#7", s.name);
            assert_eq!(
                cites(&text),
                vec![format!("{}#7", s.name)],
                "{} is registered and must be citable",
                s.name
            );
        }
        // An org prefix or a full GitHub URL still resolves to the bare repo:
        // `/` is not a repo character, so the leftward run stops at it.
        assert_eq!(
            cites("see Chelis-Lang/nautilus#43"),
            vec!["nautilus#43"],
            "an org-qualified reference is the same citation"
        );
        assert_eq!(
            cites("https://github.com/Chelis-Lang/chelis#1270 is the issue"),
            vec!["chelis#1270"]
        );
        // A path-style issue/PR URL carries no `#`, so it is not a citation.
        assert!(cites("see https://github.com/Chelis-Lang/chelis/pull/43").is_empty());
        assert!(cites("see https://github.com/Chelis-Lang/nautilus/issues/43").is_empty());
    }

    /// Four registry shells are also ordinary English nouns, so tolerating
    /// whitespace before the `#` for siblings made prose scan as citations: a
    /// `.ch` comment could turn a conformant shell red with two "uncovered
    /// narrowing citations" it never wrote. Siblings require strict adjacency;
    /// only `chelis` keeps the chelis#652 spaced forms.
    #[test]
    fn prose_containing_a_shell_name_before_a_number_is_not_a_citation() {
        let cites = |t: &str| {
            scan_citations(t)
                .into_iter()
                .map(|(c, _)| c)
                .collect::<Vec<_>>()
        };
        assert!(
            cites("# the school #1 priority is the hull #3 mesh").is_empty(),
            "English prose must not scan as citations"
        );
        for prose in [
            "coral #2 in the list",
            "whale # 7 of the fixtures",
            "the hull\t#4 case",
        ] {
            assert!(cites(prose).is_empty(), "{prose:?} must not scan");
        }
        // Strict adjacency still scans, for every shell.
        assert_eq!(cites("blocked on coral#27"), vec!["coral#27"]);
        assert_eq!(cites("blocked on school#12"), vec!["school#12"]);
        // And the upstream spaced forms the tolerance was added for survive.
        assert_eq!(cites("blocked on chelis #43"), vec!["chelis#43"]);
        assert_eq!(cites("blocked on chelis # 43"), vec!["chelis#43"]);
        assert_eq!(cites("blocked on chelis#43"), vec!["chelis#43"]);
    }

    /// The negative half of the widening. A grammar that accepts any
    /// `word#NNN` would accept everything and check nothing.
    #[test]
    fn an_unregistered_repo_is_not_a_citation() {
        let cites = |t: &str| {
            scan_citations(t)
                .into_iter()
                .map(|(c, _)| c)
                .collect::<Vec<_>>()
        };
        // A bare number names no tracker.
        assert!(cites("see #43 for details").is_empty());
        // Third-party repos are not in the registry and do not resolve in the org.
        assert!(cites("blocked on torch#43").is_empty());
        assert!(cites("numpy #7 has the answer").is_empty());
        // A package that is not a shell repo.
        assert!(cites("chelis-std#12").is_empty());
        // The org itself is not a repo.
        assert!(cites("Chelis-Lang#5").is_empty());
    }

    /// The trap the leftward anchoring exists for. `hello-chelis` ends in
    /// `chelis`, so a substring scan for the upstream name would read
    /// `hello-chelis#43` as BOTH a sibling citation and an upstream `chelis#43`,
    /// manufacturing an upstream reference nobody wrote and letting a coverage
    /// entry for one silently satisfy the other.
    #[test]
    fn a_sibling_name_ending_in_the_upstream_name_is_not_also_an_upstream_citation() {
        let cites = |t: &str| {
            scan_citations(t)
                .into_iter()
                .map(|(c, _)| c)
                .collect::<Vec<_>>()
        };
        assert_eq!(cites("blocked on hello-chelis#43"), vec!["hello-chelis#43"]);
        assert!(
            !cites("blocked on hello-chelis#43").contains(&"chelis#43".to_string()),
            "a hello-chelis cite must never also read as an upstream cite"
        );
        // A run that matches no registered repo is not a citation at all.
        assert!(cites("chelischelis#5").is_empty());
        assert!(cites("mychelis#5").is_empty());
    }

    #[test]
    fn heading_level_is_atx_strict() {
        assert_eq!(heading_level("# Title"), Some(1));
        assert_eq!(heading_level("## Tracking"), Some(2));
        assert_eq!(heading_level("### chelis#316 foo"), Some(3));
        assert_eq!(heading_level("   ## indented heading"), Some(2));
        // A bare `#316` (no space after the hashes) is text, not a heading.
        assert_eq!(heading_level("#316 is the bug"), None);
        assert_eq!(heading_level("no heading here"), None);
        assert_eq!(heading_level("- a list item"), None);
    }

    #[test]
    fn top_list_item_detection() {
        assert!(is_top_list_item("- bug"));
        assert!(is_top_list_item("* bug"));
        assert!(is_top_list_item("+ bug"));
        assert!(is_top_list_item("  - indented ≤3"));
        assert!(is_top_list_item("1. ordered"));
        assert!(is_top_list_item("12) ordered paren"));
        // Nested items (>3 spaces) belong to their parent entry, not new entries.
        assert!(!is_top_list_item("    - deeply nested"));
        // A horizontal rule / plain dash is not a list item.
        assert!(!is_top_list_item("---"));
        assert!(!is_top_list_item("prose line"));
        assert!(!is_top_list_item("(none yet)"));
    }

    #[test]
    fn issue_draft_path_is_cited() {
        assert!(cites_issue_draft(
            "parked as docs/issue_drafts/foo.md until filed"
        ));
        assert!(cites_issue_draft(
            "see (docs/issue_drafts/callback-unif.md)"
        ));
        // A bare directory mention (no filename after the slash) is not a cite.
        assert!(!cites_issue_draft(
            "filed under docs/issue_drafts/ somewhere"
        ));
        assert!(!cites_issue_draft("no draft reference at all"));
    }

    #[test]
    fn section_body_stops_at_next_same_level_heading() {
        let doc = "# T\n\n## Tracking\n\n- a\n### sub of a\ntext\n\n## Parked\n\n(none yet)\n";
        let lines: Vec<&str> = doc.lines().collect();
        let body = section_body(&lines, "Tracking").unwrap();
        // The `### sub of a` is *inside* Tracking (deeper than `##`); the body
        // ends at `## Parked`.
        let joined = body.iter().map(|(_, l)| *l).collect::<Vec<_>>().join("|");
        assert_eq!(joined, "|- a|### sub of a|text|");
    }

    #[test]
    fn bug_entries_partition_by_start_line() {
        let doc =
            "## Tracking\n\n- first entry\n  continuation\n- second (chelis#5)\n### third\nbody\n";
        let lines: Vec<&str> = doc.lines().collect();
        let body = section_body(&lines, "Tracking").unwrap();
        let entries = parse_bug_entries(&body);
        assert_eq!(entries.len(), 3);
        // A nested continuation line travels with its entry.
        assert!(entries[0].text.contains("continuation"));
        // The citation on the second entry is found by the whole-entry scan.
        let toks: Vec<String> = scan_citations(&entries[1].text)
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(toks, vec!["chelis#5".to_string()]);
        assert_eq!(entries[2].head().trim(), "### third");
    }

    #[test]
    fn pin_parse() {
        assert_eq!(
            parse_compiler_pin("[package]\ncompiler = \"=0.14.0\"\n").as_deref(),
            Some("=0.14.0")
        );
        assert_eq!(parse_compiler_pin("compiler = \"0.14.0\"\n"), None); // missing leading =
    }

    #[test]
    fn semver_ordering() {
        assert!(version_ge("=0.14.0", "0.7.0"));
        assert!(!version_ge("0.7.0", "0.14.0"));
        assert!(version_ge("v0.8.1", "0.8.1"));
    }

    #[test]
    fn prerelease_orders_below_release() {
        assert!(version_ge("0.14.0", "0.14.0-rc1"));
        assert!(!version_ge("0.14.0-rc1", "0.14.0"));
        assert!(version_ge("0.14.0-rc1", "0.14.0-rc1")); // equal ranks
    }

    #[test]
    fn gate_version_floors_at_baseline() {
        assert_eq!(gate_version("=0.1.0"), CONTRACT_BASELINE_VERSION);
        assert_eq!(gate_version("=0.14.0"), "=0.14.0");
        assert_eq!(gate_version("=0.7.0"), "=0.7.0");
    }

    #[test]
    fn installable_version_matches_installer_rules() {
        for good in ["0.12.0", "10.0.255", "0.14.0"] {
            assert!(is_installable_version(good), "{good} should be installable");
        }
        for bad in ["0.13", "1.2.3.4", "v0.1.0", "0.1.0-rc1", "", "abc", "0..0"] {
            assert!(!is_installable_version(bad), "{bad} must be rejected");
        }
    }

    #[test]
    fn pin_parse_is_section_aware_and_strict() {
        // A `compiler` under an unrelated table is not the pin.
        assert_eq!(
            parse_compiler_pin("[tool.other]\ncompiler = \"=9.9.9\"\n"),
            None
        );
        // Package table is in scope.
        assert_eq!(
            parse_compiler_pin("[package]\ncompiler = \"=0.14.0\"\n").as_deref(),
            Some("=0.14.0")
        );
        // A pin the shim cannot install must not parse (no free pass).
        assert_eq!(parse_compiler_pin("compiler = \"=garbage\"\n"), None);
        assert_eq!(parse_compiler_pin("compiler = \"=0.14.0-rc1\"\n"), None);
    }

    #[test]
    fn package_name_is_section_scoped() {
        assert_eq!(
            parse_package_name("[package]\nname = \"octant\"\n").as_deref(),
            Some("octant")
        );
        // A `name` under another table is not the package name.
        assert_eq!(
            parse_package_name("[dependencies]\nname = \"nope\"\n"),
            None
        );
        // Root-table name is accepted.
        assert_eq!(
            parse_package_name("name = \"school\"\n").as_deref(),
            Some("school")
        );
    }

    #[test]
    fn corpus_covers_is_whole_token() {
        assert!(corpus_covers("see chelis#293 here", "chelis#293"));
        assert!(!corpus_covers("only chelis#293 here", "chelis#29"));
        assert!(corpus_covers("chelis#29\n", "chelis#29"));
        // A space-form coverage entry still covers the canonical cite
        // (chelis#652: the whitespace variant must not read as uncovered).
        assert!(corpus_covers(
            "tracked upstream as chelis #316 (open)",
            "chelis#316"
        ));
        assert!(corpus_covers("[chelis # 316] is the issue", "chelis#316"));
        assert!(!corpus_covers("unrelated chelis #317 note", "chelis#316"));
    }

    /// The `local_skills` corpus the hand-rolled parser was carrying, re-pointed
    /// at the structural one (chelis#1262 review round 2). Every property here
    /// still holds; the difference is that they now hold because the document is
    /// parsed rather than because each spelling was anticipated.
    #[test]
    fn local_skills_parse() {
        let names = |t: &str| conform::parse(t).expect("valid toml").local_skills;
        // single-line
        assert_eq!(
            names("[package]\nname = \"s\"\n[conform]\nlocal_skills = [\"chelis-std\"]\n"),
            vec!["chelis-std".to_string()]
        );
        // multi-line, with a trailing comma
        let toml = "[conform]\nlocal_skills = [\n  \"a\",\n  \"b\",\n]\n[dependencies]\n";
        assert_eq!(names(toml), vec!["a".to_string(), "b".to_string()]);
        // absent section / key
        assert!(names("[package]\nname = \"s\"\n").is_empty());
        assert!(names("[conform]\nother = 1\n").is_empty());
        // a `local_skills` outside [conform] is ignored
        assert!(names("[other]\nlocal_skills = [\"x\"]\n").is_empty());
        // single-quoted (TOML literal) names are accepted
        assert_eq!(
            names("[conform]\nlocal_skills = ['chelis-std']\n"),
            vec!["chelis-std".to_string()]
        );
        // inline comments do not corrupt the following entry
        let commented = "[conform]\nlocal_skills = [ # keep these\n  \"a\", # first\n  \"b\",\n]\n";
        assert_eq!(names(commented), vec!["a".to_string(), "b".to_string()]);
        // a single-line array with a trailing comment
        assert_eq!(
            names("[conform]\nlocal_skills = [\"a\"] # note\n"),
            vec!["a".to_string()]
        );
        // a key that merely has `local_skills` as a prefix is not the key
        assert!(names("[conform]\nlocal_skills_extra = [\"x\"]\n").is_empty());
        // an empty array yields no names
        assert!(names("[conform]\nlocal_skills = []\n").is_empty());
        // a malformed non-array value yields no names (and is reported by §8)
        assert!(names("[conform]\nlocal_skills = \"x\"\n[dependencies]\nfoo = 1\n").is_empty());
        // NEW: the quoted spelling is now HONORED, not merely un-reported. The
        // two hand-rolled parsers disagreed here, so an author who wrote
        // `'local_skills'` got a row-14 failure telling them to do what they had
        // just done.
        assert_eq!(
            names("[conform]\n'local_skills' = [\"chelis-std\"]\n"),
            vec!["chelis-std".to_string()]
        );
        assert_eq!(
            names("conform.local_skills = [\"chelis-std\"]\n"),
            vec!["chelis-std".to_string()]
        );
    }

    #[test]
    fn chelisup_install_versions_extracted() {
        let body = "run: chelisup install 0.13.0 --force\n  and chelisup install \"v0.14.0\"\n";
        assert_eq!(
            extract_chelisup_install_versions(body),
            vec!["0.13.0".to_string(), "v0.14.0".to_string()]
        );
        assert!(extract_chelisup_install_versions("chelisup install --help").is_empty());
    }

    #[test]
    fn env_extraction() {
        assert_eq!(
            extract_env("  CHELIS_VERSION: 0.14.0\n", "CHELIS_VERSION").as_deref(),
            Some("0.14.0")
        );
        assert_eq!(
            extract_env("  CHELIS_TAG: \"v0.14.0\"\n", "CHELIS_TAG").as_deref(),
            Some("v0.14.0")
        );
        assert_eq!(extract_env("  X: ${{ env.Y }}\n", "X"), None);
    }
}
