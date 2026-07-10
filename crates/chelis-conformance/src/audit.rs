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
}

impl Ctx {
    fn load(root: &Path) -> Ctx {
        let reef_toml = read_opt(&root.join("reef.toml"));
        let reef_pin = reef_toml.as_deref().and_then(parse_compiler_pin);
        let shell_name = reef_toml.as_deref().and_then(parse_package_name);
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
        );
    }

    let (verdict, diagnostic, fix) = match row.key {
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
        ),
        "scaffolding-drift-rule" => check_agents_heading(ctx, "Scaffolding Drift Rule"),
        "chelis-src" => check_chelis_src(ctx),
        other => (
            Verdict::Manual,
            format!("no check implemented for row key {other:?}"),
            String::new(),
        ),
    };
    result(row, verdict, diagnostic, fix)
}

fn result(
    row: &ContractRow,
    verdict: Verdict,
    diagnostic: impl Into<String>,
    fix: impl Into<String>,
) -> RowResult {
    RowResult {
        row: row.row,
        key: row.key,
        section: row.section,
        tier: row.tier,
        verdict,
        diagnostic: diagnostic.into(),
        fix: fix.into(),
    }
}

type Check = (Verdict, String, String);

fn pass() -> Check {
    (Verdict::Pass, String::new(), String::new())
}

fn fail(diag: impl Into<String>, fix: impl Into<String>) -> Check {
    (Verdict::Fail, diag.into(), fix.into())
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
    let mut orphans: Vec<String> = cited
        .into_iter()
        .filter(|c| !corpus_covers(&corpus, c))
        .collect();
    orphans.sort();
    orphans.dedup();
    if orphans.is_empty() {
        pass()
    } else {
        fail(
            format!("uncovered narrowing citation(s): {}", orphans.join(", ")),
            "add a tests_blocked/ probe, a docs/UPSTREAM_BUGS.md entry, or a tests_blocked/README.md can't-be-probed note for each",
        )
    }
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
    for (name, body) in skills::EMBEDDED_SKILLS {
        let path = skills_dir.join(name).join("SKILL.md");
        match read_opt(&path) {
            None => problems.push(format!("{name}: missing")),
            Some(live) if live != *body => problems.push(format!("{name}: forked/stale")),
            Some(_) => {}
        }
    }
    // Reverse direction: the shell owns *zero* extra skill content, so an
    // addition is drift too. Enumerate the on-disk tree and flag any skill dir
    // outside the pinned set, or any file beyond `SKILL.md` inside a pinned dir
    // (materialize never creates these, so their presence is a fork/leftover).
    if let Ok(entries) = std::fs::read_dir(&skills_dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                if !skills::SHARED_SKILLS.contains(&name.as_str()) {
                    problems.push(format!("{name}: not a pinned skill (remove)"));
                } else if let Ok(inner) = std::fs::read_dir(e.path()) {
                    for f in inner.flatten() {
                        let fname = f.file_name().to_string_lossy().into_owned();
                        if fname != "SKILL.md" {
                            problems.push(format!("{name}/{fname}: unexpected skill file"));
                        }
                    }
                }
            } else if name != "UPSTREAM.toml" {
                problems.push(format!("{name}: unexpected file in agent-skills/"));
            }
        }
    }
    if !problems.is_empty() {
        return fail(
            format!(
                "vendored skills drifted from the pinned set: {}",
                problems.join(", ")
            ),
            "run `chelis reef conform sync` to re-materialize agent-skills/ from the toolchain",
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

fn check_parity_harness(ctx: &Ctx) -> Check {
    if !ctx.exists("parity") {
        return (
            Verdict::Na,
            "no parity/ harness (shell does not validate against external oracles)".to_string(),
            String::new(),
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

/// Collect `chelis#NNN` citations appearing in any `.ch` file under `dir`.
fn collect_citations_in_dir(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for p in walk_files(dir) {
        if p.extension().and_then(|s| s.to_str()) != Some("ch") {
            continue;
        }
        if let Some(text) = read_opt(&p) {
            out.extend(extract_citations(&text));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Whether `corpus` covers `citation` as a whole `chelis#NNN` token — the
/// digits must not continue past the citation. A bare `corpus.contains` would
/// let `chelis#293` spuriously "cover" an orphaned `chelis#29`.
fn corpus_covers(corpus: &str, citation: &str) -> bool {
    corpus.match_indices(citation).any(|(i, _)| {
        corpus[i + citation.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_digit())
    })
}

fn extract_citations(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let needle = b"chelis#";
    let mut i = 0;
    while let Some(rel) = text[i..].find("chelis#") {
        let start = i + rel;
        let mut j = start + needle.len();
        let num_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > num_start {
            out.push(text[start..j].to_string());
        }
        i = j.max(start + 1);
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
        let cites = extract_citations("fail(...) // blocked on chelis#293 and chelis#345x");
        assert_eq!(cites, vec!["chelis#293", "chelis#345"]);
        assert!(extract_citations("no citation, chelis# alone").is_empty());
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
