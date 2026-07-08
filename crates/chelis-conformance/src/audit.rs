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

use crate::manifest::{ContractRow, MANIFEST, Tier};
use crate::{managed_block, skills};

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

    /// True iff there are no hard failures.
    pub fn ok(&self) -> bool {
        self.must_failures() == 0
    }
}

/// MUST-class tiers (unconditional or triggered-conditional). SHOULD is the only
/// non-gating tier; conditional tiers are gating once their trigger fires, and
/// the row check emits `Na` when the trigger is absent, so treating them as MUST
/// here is correct.
fn tier_is_must(tier: Tier) -> bool {
    !matches!(tier, Tier::Should)
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
        let agents_md = read_opt(&root.join("AGENTS.md"));
        let claude_symlink_ok = claude_is_symlink_to_agents(root);
        let cargo_toml = read_opt(&root.join("Cargo.toml"));
        let workflows = read_workflows(root);
        Ctx {
            root: root.to_path_buf(),
            reef_pin,
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
    // since_version gating: if the shell's pin predates the row, it does not yet
    // apply (canary-safe).
    if let Some(pin) = &ctx.reef_pin
        && !version_ge(pin, row.since_version)
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
    if !agents.contains("Repo Identity") {
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
        .filter(|s| !bugs.contains(s))
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
    let mut orphans: Vec<String> = cited.into_iter().filter(|c| !corpus.contains(c)).collect();
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
        Some(a) if a.contains(heading) => pass(),
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
    if !problems.is_empty() {
        return fail(
            format!(
                "vendored skills drifted from the pinned set: {}",
                problems.join(", ")
            ),
            "run `chelis reef conform sync` to re-materialize agent-skills/ from the toolchain",
        );
    }
    // Skill-dir symlinks are a strong signal but optional across tool surfaces;
    // require at least the .claude/skills symlink into agent-skills/.
    if !path_is_symlink(&ctx.root.join(".claude/skills")) {
        return fail(
            ".claude/skills is not a symlink to agent-skills/",
            "ln -s ../agent-skills .claude/skills",
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
    let links = ctx
        .cargo_toml
        .as_deref()
        .is_some_and(|c| c.contains("../chelis/crates") || c.contains("../chelis\""));
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

/// Shared managed-block freshness check: present, stamped to the reef pin, and
/// untampered + matching nothing-stale (integrity vs its own fence hash).
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
    pass()
}

// ---------------------------------------------------------------- helpers

fn read_opt(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn path_is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
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

/// Parse a `compiler = "=X.Y.Z"` pin (returns the value including the leading
/// `=`, e.g. `"=0.14.0"`).
pub fn parse_compiler_pin(toml: &str) -> Option<String> {
    for line in toml.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("compiler") {
            let rest = rest.trim_start().strip_prefix('=')?.trim_start();
            let rest = rest.strip_prefix('"')?;
            let end = rest.find('"')?;
            let val = &rest[..end];
            if val.starts_with('=') {
                return Some(val.to_string());
            }
        }
    }
    None
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
                if p.is_dir() {
                    stack.push(p);
                } else {
                    files.push(p);
                }
            }
        }
    }
    files.into_iter()
}

/// `a >= b` on `X.Y.Z` (leading `=`/`v` tolerated; non-digit patch suffixes
/// truncated).
fn version_ge(a: &str, b: &str) -> bool {
    match (parse_semver(a), parse_semver(b)) {
        (Some(a), Some(b)) => a >= b,
        _ => true, // if either is unparseable, do not gate the row out
    }
}

fn parse_semver(v: &str) -> Option<(u64, u64, u64)> {
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
    Some((major, minor, patch))
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
