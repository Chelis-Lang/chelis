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

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::manifest::{CONTRACT_BASELINE_VERSION, ContractRow, MANIFEST, Tier};
use crate::{canonical, managed_block, registry, skills};

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
    /// The `[conform] local_skills` allowlist from `reef.toml` (chelis#651):
    /// repo-local domain skills the shell owns, which `sync` preserves and §8
    /// exempts from the "not a pinned skill" drift check.
    local_skills: Vec<String>,
}

impl Ctx {
    fn load(root: &Path) -> Ctx {
        let reef_toml = read_opt(&root.join("reef.toml"));
        let reef_pin = reef_toml.as_deref().and_then(parse_compiler_pin);
        let shell_name = reef_toml.as_deref().and_then(parse_package_name);
        let local_skills = reef_toml
            .as_deref()
            .map(parse_local_skills)
            .unwrap_or_default();
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
            local_skills,
        }
    }

    fn read(&self, rel: &str) -> Option<String> {
        read_opt(&self.root.join(rel))
    }

    fn exists(&self, rel: &str) -> bool {
        self.root.join(rel).exists()
    }
}

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
            format!("no check implemented for row key {other:?}"),
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
    check_managed_block(ctx, agents, "agents-inheritance", "AGENTS.md")
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
    let expected_pins = [
        ("CHELIS_TAG", format!("v{bare}")),
        ("CHELIS_VERSION", bare.to_string()),
    ];
    let mut mismatches = Vec::new();
    let mut checked = 0;
    for (name, body) in &ctx.workflows {
        if !workflow_installs_toolchain(body) {
            continue;
        }
        checked += 1;
        for (key, expected) in &expected_pins {
            let found = extract_static_env_values(body, key);
            if found.len() != 1 || !found.contains(expected) {
                let actual = if found.is_empty() {
                    "missing".to_string()
                } else {
                    found.into_iter().collect::<Vec<_>>().join(", ")
                };
                mismatches.push(format!("{name}: {key}={actual}; expected {expected}"));
            }
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
    )
}

fn check_upstream_bugs(ctx: &Ctx) -> Check {
    let Some(bugs) = ctx.read("docs/UPSTREAM_BUGS.md") else {
        return fail(
            "docs/UPSTREAM_BUGS.md is missing",
            "add docs/UPSTREAM_BUGS.md with the four required sections",
        );
    };
    let required = ["Actively blocking", "Tracking", "Parked", "Archived"];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|s| !has_heading(&bugs, s))
        .collect();
    if missing.is_empty() {
        pass()
    } else {
        fail(
            format!(
                "docs/UPSTREAM_BUGS.md missing section(s): {}",
                missing.join(", ")
            ),
            "add the §Actively blocking / §Tracking / §Parked / §Archived sections",
        )
    }
}

/// Row 9: narrowing-coverage (the offline half of the §4 staleness audit). Every
/// `chelis#NNN` cited from a live `src/**/*.ch` narrowing must be covered by a
/// `tests_blocked/` probe, a `docs/UPSTREAM_BUGS.md` entry, or the
/// `tests_blocked/README.md` can't-be-probed list.
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
/// appears in a non-canonical (non-`chelis#NNN`) form — the exact trap in
/// chelis#654, where a space-form `chelis #316` (now matched, chelis#652) or a
/// bare `#316` left the fix message ("add an UPSTREAM_BUGS entry") misleading.
fn coverage_evidence(token: &str, blocked: &str, upstream: &str, readme: &str) -> String {
    let num = token.trim_start_matches("chelis#");
    let bare = format!("#{num}");
    let upstream_note = if corpus_covers(upstream, token) {
        "covered".to_string()
    } else if upstream.contains(&bare) {
        format!("mentions {bare} but not as a `chelis#{num}` token")
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
    // A `local_skills` entry may not shadow a shared skill — that would let a
    // shell "own" (and silently fork) toolchain-managed content (chelis#651).
    for local in &ctx.local_skills {
        if skills::SHARED_SKILLS.contains(&local.as_str()) {
            problems.push(format!(
                "{local}: [conform] local_skills may not name a shared skill"
            ));
        }
    }
    for (name, body) in skills::EMBEDDED_SKILLS {
        let path = skills_dir.join(name).join("SKILL.md");
        match read_opt(&path) {
            None => problems.push(format!("{name}: missing")),
            Some(live) => {
                // A trailing shell-local block (chelis#653) is shell-owned; §8
                // byte-checks the toolchain-owned managed span above it, then
                // treats the complete validated file as the effective value both
                // same-name command wrappers must project byte-for-byte.
                let (managed, block) = crate::scaffold::split_shell_local(&live);
                let mut effective_is_valid = true;
                if managed.trim_end() != body.trim_end() {
                    problems.push(format!("{name}: forked/stale"));
                    effective_is_valid = false;
                }
                if let Some(block) = block
                    && let Err(why) = validate_shell_local_block(block)
                {
                    problems.push(format!("{name}: {why}"));
                    effective_is_valid = false;
                }
                if effective_is_valid {
                    for tool in [".claude", ".codex"] {
                        let rel = format!("{tool}/commands/{name}.md");
                        let command_path = ctx.root.join(&rel);
                        match std::fs::read(&command_path) {
                            Ok(command) if command != live.as_bytes() => problems.push(format!(
                                "{name}: {rel} differs from agent-skills/{name}/SKILL.md"
                            )),
                            Ok(_) => {}
                            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                                problems.push(format!("{name}: missing {rel}"));
                            }
                            Err(error) => {
                                problems.push(format!("{name}: cannot read {rel}: {error}"));
                            }
                        }
                    }
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
                    if let Ok(inner) = std::fs::read_dir(e.path()) {
                        for f in inner.flatten() {
                            let fname = f.file_name().to_string_lossy().into_owned();
                            if fname != "SKILL.md" {
                                problems.push(format!("{name}/{fname}: unexpected skill file"));
                            }
                        }
                    }
                } else if ctx.local_skills.iter().any(|s| s == &name) {
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
    if !problems.is_empty() {
        return fail(
            format!(
                "shared agent surface drifted from the pinned set: {}",
                problems.join(", ")
            ),
            "run `chelis reef conform sync` to re-materialize agent-skills/ and same-name .claude/commands/ + .codex/commands/ from the toolchain",
        );
    }
    // Both tool-surface skill dirs must be symlinks that actually resolve to
    // `agent-skills/` — a symlink to somewhere else (or a real dir) is not the
    // materialized-pointer model.
    for link in [".claude/skills", ".codex/skills"] {
        if !symlink_targets_agent_skills(&ctx.root.join(link)) {
            return fail(
                format!("{link} is not a symlink to agent-skills/"),
                format!("ln -s ../agent-skills {link}"),
            );
        }
    }
    pass()
}

/// Parse the `local_skills` allowlist from a `[conform]` table in `reef.toml`
/// (chelis#651). Hand-parsed — the crate has no `toml` dependency — accepting a
/// single- or multi-line array of double- or single-quoted names, tolerating
/// inline `#` comments. `chelis-reef` ignores the `[conform]` table (no
/// `deny_unknown_fields`), so this is its only reader.
pub(crate) fn parse_local_skills(reef_toml: &str) -> Vec<String> {
    // Skill names and TOML table headers never contain `#`, and the values are
    // quoted names, so a bare `#` starts a comment. Cutting each physical line
    // there keeps an inline comment from corrupting the entry that follows it in
    // a multi-line array. (Narrow but sufficient; the crate has no TOML parser.)
    fn strip_comment(line: &str) -> &str {
        match line.find('#') {
            Some(i) => &line[..i],
            None => line,
        }
    }
    let mut in_conform = false;
    let mut collecting = false;
    let mut buf = String::new();
    for raw in reef_toml.lines() {
        let line = strip_comment(raw);
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            // A new table header while still collecting means the array was
            // never closed (malformed) — stop rather than swallow the header's
            // name as a phantom skill.
            if collecting {
                break;
            }
            in_conform = t == "[conform]";
            continue;
        }
        if !in_conform {
            continue;
        }
        if collecting {
            buf.push_str(line);
            buf.push('\n');
            if line.contains(']') {
                break;
            }
        } else if let Some(rest) = t.strip_prefix("local_skills")
            && let Some(rest) = rest.trim_start().strip_prefix('=')
        {
            buf.push_str(rest);
            buf.push('\n');
            if rest.contains(']') {
                break;
            }
            collecting = true;
        }
    }
    let (Some(open), Some(close)) = (buf.find('['), buf.rfind(']')) else {
        return Vec::new();
    };
    if close < open {
        return Vec::new();
    }
    buf[open + 1..close]
        .split(',')
        .map(|s| s.trim().trim_matches(['"', '\'']).trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// A trailing shell-local block (chelis#653) must be well-formed: exactly one
/// begin marker, an end marker after it, and nothing but whitespace past the end
/// marker (the block is strictly the file suffix, so `sync` can regenerate the
/// managed span above it without touching author content).
fn validate_shell_local_block(block: &str) -> Result<(), String> {
    use crate::scaffold::{SHELL_LOCAL_BEGIN, SHELL_LOCAL_END};
    if block.matches(SHELL_LOCAL_BEGIN).count() != 1 {
        return Err(format!(
            "malformed shell-local block (expected exactly one `{SHELL_LOCAL_BEGIN}`)"
        ));
    }
    let Some(end) = block.find(SHELL_LOCAL_END) else {
        return Err(format!("shell-local block missing `{SHELL_LOCAL_END}`"));
    };
    if !block[end + SHELL_LOCAL_END.len()..].trim().is_empty() {
        return Err(format!(
            "content after `{SHELL_LOCAL_END}` (the shell-local block must be the file suffix)"
        ));
    }
    Ok(())
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
fn check_managed_block(ctx: &Ctx, doc: &str, id: &str, file: &str) -> Check {
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
        && let Some(canon) = canonical::body(id)
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

/// Whether `path` is a symlink whose target's final component is
/// `agent-skills` (i.e. it resolves to the shell's `agent-skills/` dir).
fn symlink_targets_agent_skills(path: &Path) -> bool {
    let is_symlink = std::fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    if !is_symlink {
        return false;
    }
    std::fs::read_link(path)
        .ok()
        .and_then(|t| t.file_name().map(|s| s.to_os_string()))
        .is_some_and(|name| name == "agent-skills")
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
    body.contains("chelisup install")
        || body.contains("install-chelis")
        || (body.contains("gh release download") && body.contains("Chelis-Lang/chelis"))
}

/// Collect every static literal assigned to `key` in a workflow.
///
/// GitHub Actions expressions are deliberately excluded: the shell contract
/// requires the workflow-level pair to remain an auditable literal mirror of
/// `reef.toml`. A set catches conflicting duplicate declarations while allowing
/// harmless repeated declarations of the same expected value.
fn extract_static_env_values(body: &str, key: &str) -> BTreeSet<String> {
    body.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix(key)?.trim_start();
            let raw = rest.strip_prefix(':')?.trim();
            parse_static_yaml_scalar(raw)
        })
        .collect()
}

fn parse_static_yaml_scalar(raw: &str) -> Option<String> {
    if raw.is_empty() || raw.starts_with("${{") {
        return None;
    }

    let first = raw.chars().next()?;
    if first == '\'' || first == '"' {
        let tail = &raw[first.len_utf8()..];
        let end = tail.find(first)?;
        let value = &tail[..end];
        let trailing = tail[end + first.len_utf8()..].trim();
        if !trailing.is_empty() && !trailing.starts_with('#') {
            return None;
        }
        return (!value.is_empty()).then(|| value.to_string());
    }

    let value = raw.split_once('#').map_or(raw, |(value, _)| value).trim();
    (!value.is_empty()).then(|| value.to_string())
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

/// Scan `text` for narrowing citations, tolerating inline whitespace between
/// `chelis`, `#`, and the number, so `chelis#316`, `chelis #316`, and
/// `chelis # 316` all normalize to the canonical token `chelis#316`
/// (chelis#652 — a space-form cite/coverage entry must not read as uncovered).
/// Returns each canonical `chelis#NNN` token paired with the byte offset where
/// the match starts, in source order (the offset feeds `--explain` site
/// reporting, chelis#654). Newlines are NOT tolerated between the parts, so a
/// sentence-final `chelis` followed by an unrelated `#heading` on the next line
/// is not a false match.
fn scan_citations(text: &str) -> Vec<(String, usize)> {
    fn is_inline_ws(b: u8) -> bool {
        b == b' ' || b == b'\t'
    }
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(rel) = text[i..].find("chelis") {
        let start = i + rel;
        let mut j = start + "chelis".len();
        while j < bytes.len() && is_inline_ws(bytes[j]) {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'#' {
            j += 1;
            while j < bytes.len() && is_inline_ws(bytes[j]) {
                j += 1;
            }
            let num_start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > num_start {
                out.push((format!("chelis#{}", &text[num_start..j]), start));
            }
        }
        // Advance just past this `chelis` occurrence so overlapping tokens
        // (`chelischelis#5`) are still found.
        i = start + "chelis".len();
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

    #[test]
    fn local_skills_parse() {
        // single-line
        assert_eq!(
            parse_local_skills(
                "[package]\nname = \"s\"\n[conform]\nlocal_skills = [\"chelis-std\"]\n"
            ),
            vec!["chelis-std".to_string()]
        );
        // multi-line, with a trailing comma
        let toml = "[conform]\nlocal_skills = [\n  \"a\",\n  \"b\",\n]\n[dependencies]\n";
        assert_eq!(
            parse_local_skills(toml),
            vec!["a".to_string(), "b".to_string()]
        );
        // absent section / key
        assert!(parse_local_skills("[package]\nname = \"s\"\n").is_empty());
        assert!(parse_local_skills("[conform]\nother = 1\n").is_empty());
        // a `local_skills` outside [conform] is ignored
        assert!(parse_local_skills("[other]\nlocal_skills = [\"x\"]\n").is_empty());
        // single-quoted (TOML literal) names are accepted
        assert_eq!(
            parse_local_skills("[conform]\nlocal_skills = ['chelis-std']\n"),
            vec!["chelis-std".to_string()]
        );
        // inline comments do not corrupt the following entry
        let commented = "[conform]\nlocal_skills = [ # keep these\n  \"a\", # first\n  \"b\",\n]\n";
        assert_eq!(
            parse_local_skills(commented),
            vec!["a".to_string(), "b".to_string()]
        );
        // a single-line array with a trailing comment
        assert_eq!(
            parse_local_skills("[conform]\nlocal_skills = [\"a\"] # note\n"),
            vec!["a".to_string()]
        );
        // a key that merely has `local_skills` as a prefix is not the key
        assert!(parse_local_skills("[conform]\nlocal_skills_extra = [\"x\"]\n").is_empty());
        // an empty array yields no names
        assert!(parse_local_skills("[conform]\nlocal_skills = []\n").is_empty());
        // a malformed non-array value must not swallow the next table header as
        // a phantom skill name
        assert!(
            parse_local_skills("[conform]\nlocal_skills = \"x\"\n[dependencies]\nfoo = 1\n")
                .is_empty()
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
    fn static_env_values_collect_literals_and_reject_expressions() {
        let body = concat!(
            "  CHELIS_VERSION: 0.14.0\n",
            "  CHELIS_VERSION: '0.13.0' # stale duplicate\n",
            "  CHELIS_TAG: \"v0.14.0\" # current\n",
            "  DYNAMIC: ${{ env.CHELIS_VERSION }}\n",
            "  CHELIS_VERSION_EXTRA: 9.9.9\n",
        );
        assert_eq!(
            extract_static_env_values(body, "CHELIS_VERSION"),
            ["0.13.0".to_string(), "0.14.0".to_string()]
                .into_iter()
                .collect()
        );
        assert_eq!(
            extract_static_env_values(body, "CHELIS_TAG"),
            ["v0.14.0".to_string()].into_iter().collect()
        );
        assert!(extract_static_env_values(body, "DYNAMIC").is_empty());
    }
}
