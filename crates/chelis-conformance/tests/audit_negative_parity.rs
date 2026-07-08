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

const VER: &str = "0.14.0";

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
        text.replace("CHELIS_VERSION: 0.14.0", "CHELIS_VERSION: 0.13.0"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert!(diagnostic_of(&report, "workflow-env-pins").contains("0.13.0"));
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
    std::fs::write(&agents, text.replace("chelis@0.14.0", "chelis@0.13.0")).unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "agents-md"), Verdict::Fail);
    let diag = diagnostic_of(&report, "agents-md");
    assert!(
        diag.contains("stamped") && diag.contains("0.13.0"),
        "diag was: {diag}"
    );
}
