//! Phase 2 oracle: spec-first negative parity for the conformance audit.
//!
//! Stamp a temp shell with `scaffold` (the core of `conform init`), confirm the
//! audit is green, then mutate it four ways and assert each mutation fails the
//! audit **on the correct row** with a non-empty diagnostic. This is the
//! authoritative Phase 2 completion oracle; the CLI `conform audit` verb is a
//! thin wrapper over `audit::audit`, smoke-tested separately.

use std::path::{Path, PathBuf};

use chelis_conformance::audit::{self, Verdict};
use chelis_conformance::scaffold;

// Must track the crate version: the audit's canonical-body comparison only
// applies to blocks stamped at AUDITOR_VERSION, so a hardcoded scaffold
// version silently skips the forged-block oracle after every release bump.
const VER: &str = env!("CARGO_PKG_VERSION");

fn stamp(dir: &Path, name: &str) -> PathBuf {
    let root = dir.join(name);
    scaffold::scaffold(&root, name, "Myshell", VER).expect("scaffold");
    root
}

fn verdict_of(report: &audit::AuditReport, key: &str) -> Verdict {
    report
        .rows
        .iter()
        .find(|r| r.key == key)
        .unwrap_or_else(|| panic!("no row with key {key:?}"))
        .verdict
}

fn diagnostic_of(report: &audit::AuditReport, key: &str) -> String {
    report
        .rows
        .iter()
        .find(|r| r.key == key)
        .map(|r| r.diagnostic.clone())
        .unwrap_or_default()
}

#[test]
fn scaffolded_shell_audits_green() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "greenshell");
    let report = audit::audit(&root);

    // No MUST-tier failures.
    let must_fails: Vec<_> = report
        .rows
        .iter()
        .filter(|r| {
            r.verdict == Verdict::Fail
                && !matches!(r.tier, chelis_conformance::manifest::Tier::Should)
        })
        .map(|r| format!("row {} ({}): {}", r.row, r.key, r.diagnostic))
        .collect();
    assert!(
        report.ok() && must_fails.is_empty(),
        "freshly scaffolded shell must audit green, but MUST rows failed:\n  {}",
        must_fails.join("\n  ")
    );
}

#[test]
fn deleting_chelis_surface_fails_row_7() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s7");
    std::fs::remove_file(root.join("docs/CHELIS_SURFACE.md")).unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "chelis-surface"), Verdict::Fail);
    assert!(diagnostic_of(&report, "chelis-surface").contains("CHELIS_SURFACE.md"));
}

#[test]
fn editing_workflow_pin_fails_row_3() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s3");
    let ci = root.join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&ci).unwrap();
    std::fs::write(
        &ci,
        text.replace(&format!("CHELIS_VERSION: {VER}"), "CHELIS_VERSION: 0.13.0"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert!(diagnostic_of(&report, "workflow-env-pins").contains("0.13.0"));
}

#[test]
fn matching_quoted_workflow_pin_pair_passes_row_3() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s3quoted");
    let ci = root.join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&ci).unwrap();
    let text = text
        .replace(
            &format!("CHELIS_TAG: v{VER}"),
            &format!("CHELIS_TAG: 'v{VER}' # audit mirror"),
        )
        .replace(
            &format!("CHELIS_VERSION: {VER}"),
            &format!("CHELIS_VERSION: \"{VER}\" # audit mirror"),
        );
    std::fs::write(&ci, text).unwrap();

    assert_eq!(
        verdict_of(&audit::audit(&root), "workflow-env-pins"),
        Verdict::Pass
    );
}

#[test]
fn missing_workflow_pin_pair_member_fails_row_3() {
    let tmp = tempfile::tempdir().unwrap();
    for (name, declaration, key) in [
        (
            "s3missingtag",
            format!("  CHELIS_TAG: v{VER}\n"),
            "CHELIS_TAG",
        ),
        (
            "s3missingversion",
            format!("  CHELIS_VERSION: {VER}\n"),
            "CHELIS_VERSION",
        ),
    ] {
        let root = stamp(tmp.path(), name);
        let ci = root.join(".github/workflows/ci.yml");
        let text = std::fs::read_to_string(&ci).unwrap();
        std::fs::write(&ci, text.replace(&declaration, "")).unwrap();

        let report = audit::audit(&root);
        assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
        let diagnostic = diagnostic_of(&report, "workflow-env-pins");
        assert!(
            diagnostic.contains(key) && diagnostic.contains("missing"),
            "diagnostic for {name} was: {diagnostic}"
        );
    }
}

#[test]
fn conflicting_duplicate_workflow_pin_fails_row_3() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s3duplicate");
    let ci = root.join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&ci).unwrap();
    let declaration = format!("  CHELIS_VERSION: {VER}");
    std::fs::write(
        &ci,
        text.replace(
            &declaration,
            &format!("{declaration}\n  CHELIS_VERSION: 0.13.0"),
        ),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    let diagnostic = diagnostic_of(&report, "workflow-env-pins");
    assert!(diagnostic.contains("CHELIS_VERSION") && diagnostic.contains("0.13.0"));
}

#[test]
fn direct_chelis_release_download_requires_pin_pair() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s3download");
    let ci = root.join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&ci).unwrap();
    let direct_download = text.replace(
        &format!("chelisup install {VER}"),
        "gh release download --repo Chelis-Lang/chelis",
    );
    std::fs::write(&ci, &direct_download).unwrap();
    assert_eq!(
        verdict_of(&audit::audit(&root), "workflow-env-pins"),
        Verdict::Pass,
        "a direct release download with the exact audit pair must pass"
    );

    let missing_pair = direct_download
        .replace(&format!("  CHELIS_TAG: v{VER}\n"), "")
        .replace(&format!("  CHELIS_VERSION: {VER}\n"), "");
    std::fs::write(&ci, missing_pair).unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    let diagnostic = diagnostic_of(&report, "workflow-env-pins");
    assert!(diagnostic.contains("CHELIS_TAG") && diagnostic.contains("CHELIS_VERSION"));
}

#[test]
fn noninstalling_workflow_pin_text_is_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s3noninstaller");
    std::fs::write(
        root.join(".github/workflows/release.yml"),
        "name: docs\nenv:\n  CHELIS_VERSION: 0.13.0\njobs:\n  docs:\n    steps:\n      - run: echo CHELIS_VERSION\n",
    )
    .unwrap();

    assert_eq!(
        verdict_of(&audit::audit(&root), "workflow-env-pins"),
        Verdict::Pass
    );
}

#[test]
fn forking_a_skill_fails_row_14() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s14");
    let skill = root.join("agent-skills/spec-sync/SKILL.md");
    let mut text = std::fs::read_to_string(&skill).unwrap();
    text.push_str("\nlocally forked line\n");
    std::fs::write(&skill, text).unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "vendored-skills"), Verdict::Fail);
    assert!(diagnostic_of(&report, "vendored-skills").contains("spec-sync"));
}

#[test]
fn stale_managed_block_stamp_fails_row_1() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s1");
    let agents = root.join("AGENTS.md");
    let text = std::fs::read_to_string(&agents).unwrap();
    // Restamp the managed block to an older version without touching the body,
    // simulating a pin bump that skipped `conform sync`.
    std::fs::write(
        &agents,
        text.replace(&format!("chelis@{VER}"), "chelis@0.13.0"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "agents-md"), Verdict::Fail);
    let diag = diagnostic_of(&report, "agents-md");
    assert!(
        diag.contains("stamped") && diag.contains("0.13.0"),
        "diag was: {diag}"
    );
}

/// B2: a managed block whose body is forged (edited *and* re-stamped so the
/// fence hash matches) at the current version passes integrity + version but
/// must be caught by the canonical-body comparison.
#[test]
fn self_consistent_forked_managed_block_fails() {
    use chelis_conformance::managed_block;

    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "forge");
    let agents = root.join("AGENTS.md");
    let text = std::fs::read_to_string(&agents).unwrap();

    let block = managed_block::find(&text, "agents-inheritance").unwrap();
    let (start, end) = block.span;
    let forged = managed_block::render(
        "agents-inheritance",
        VER,
        "Forged inheritance text that is not the canonical upstream.",
    );
    // Sanity: the forgery is self-consistent (would fool integrity_ok alone).
    let refound = managed_block::find(&forged, "agents-inheritance").unwrap();
    assert!(refound.integrity_ok() && refound.version == VER);

    std::fs::write(
        &agents,
        format!("{}{forged}{}", &text[..start], &text[end..]),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok(), "a self-consistent fork must fail the audit");
    assert_eq!(verdict_of(&report, "agents-md"), Verdict::Fail);
    assert!(diagnostic_of(&report, "agents-md").contains("canonical"));
}

/// B1: a shell pinning below the contract baseline must not gate every row out
/// to `Na` and read as conformant.
#[test]
fn below_baseline_pin_does_not_pass_empty_shell() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("ancient");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("reef.toml"), "compiler = \"=0.1.0\"\n").unwrap();

    let report = audit::audit(&root);
    assert!(
        !report.ok(),
        "a bare shell pinning below baseline must not audit conformant"
    );
    assert!(
        report.evaluated_any(),
        "baseline rows must still apply under a floored pin"
    );
}

/// M1: an extra skill dir outside the pinned set is drift (additions, not just
/// edits/deletions, must fail).
#[test]
fn extra_skill_dir_fails_row_14() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "extra");
    std::fs::create_dir_all(root.join("agent-skills/rogue")).unwrap();
    std::fs::write(root.join("agent-skills/rogue/SKILL.md"), "forked\n").unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "vendored-skills"), Verdict::Fail);
    assert!(diagnostic_of(&report, "vendored-skills").contains("rogue"));
}

/// M1: an extra file inside a pinned skill dir is drift too.
#[test]
fn extra_file_in_skill_dir_fails_row_14() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "extraf");
    std::fs::write(root.join("agent-skills/spec-sync/EXTRA.md"), "x\n").unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "vendored-skills"), Verdict::Fail);
    assert!(diagnostic_of(&report, "vendored-skills").contains("EXTRA.md"));
}

/// M1: `sync` (materialize) prunes drift so the tree converges back to green.
#[test]
fn sync_prunes_extra_skill_content() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "prune");
    std::fs::create_dir_all(root.join("agent-skills/rogue")).unwrap();
    std::fs::write(root.join("agent-skills/rogue/SKILL.md"), "forked\n").unwrap();
    std::fs::write(root.join("agent-skills/spec-sync/EXTRA.md"), "x\n").unwrap();
    assert!(!audit::audit(&root).ok(), "drift must be present first");

    scaffold::materialize_skills(&root).unwrap();
    assert!(
        !root.join("agent-skills/rogue").exists(),
        "rogue skill dir must be pruned"
    );
    assert!(
        !root.join("agent-skills/spec-sync/EXTRA.md").exists(),
        "extra skill file must be pruned"
    );
    assert!(audit::audit(&root).ok(), "audit green after prune");
}

/// M3: row 18 applicability is driven by the registry's authoritative
/// `links_chelis_crates` flag, not a Cargo.toml guess.
#[test]
fn chelis_src_trigger_comes_from_registry() {
    let tmp = tempfile::tempdir().unwrap();

    // `octant` links chelis crates (registry flag true), so [chelis-src] is
    // required even though the scaffolded tree has no Cargo path deps.
    let octant = stamp(tmp.path(), "octant");
    assert_eq!(
        verdict_of(&audit::audit(&octant), "chelis-src"),
        Verdict::Fail,
        "a registry crate-linking shell without [chelis-src] must fail row 18"
    );

    // `school` is a Reef shell (registry flag false), so it is NA even when its
    // Cargo.toml looks like it links crates — the registry overrides the guess.
    let school = stamp(tmp.path(), "school");
    std::fs::write(
        school.join("Cargo.toml"),
        "[dependencies]\nchelis-ir = { path = \"../chelis/crates/chelis-ir\" }\n",
    )
    .unwrap();
    assert_eq!(
        verdict_of(&audit::audit(&school), "chelis-src"),
        Verdict::Na,
        "the registry's Reef classification must override the Cargo.toml heuristic"
    );
}
