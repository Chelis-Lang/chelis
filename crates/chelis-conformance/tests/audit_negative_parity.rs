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
const CENTRAL_SHA: &str = "1111111111111111111111111111111111111111";
const ARCHIVE_DIGEST: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

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

fn central_profile_wrapper(profile: &str, inputs: &[String]) -> String {
    let mut lines = vec![
        "name: central".to_string(),
        "on: workflow_dispatch".to_string(),
        "permissions:".to_string(),
        "  contents: read".to_string(),
        "jobs:".to_string(),
        "  call-central:".to_string(),
        "    name: Call central profile".to_string(),
        format!("    uses: Chelis-Lang/ci/.github/workflows/consumer.yml@{CENTRAL_SHA}"),
        "    with:".to_string(),
        format!("      profile: {profile}"),
    ];
    lines.extend(inputs.iter().map(|input| format!("      {input}")));
    lines.extend([
        "    secrets:".to_string(),
        "      CHELIS_RELEASE_TOKEN: ${{ secrets.CHELIS_RELEASE_TOKEN }}".to_string(),
    ]);
    format!("{}\n", lines.join("\n"))
}

fn central_ci_wrapper(package: &str, version: &str) -> String {
    let profile = match package {
        "coral" => "coral-ci",
        "nautilus" => "nautilus-ci",
        other => panic!("unsupported central wrapper fixture {other}"),
    };
    let mut inputs = vec![
        format!("chelis-tag: v{version}"),
        format!("chelis-version: {version}"),
        format!("chelis-linux-sha256: {ARCHIVE_DIGEST}"),
        format!("chelis-darwin-sha256: {ARCHIVE_DIGEST}"),
    ];
    if package == "coral" {
        inputs.push("nautilus-tag: v4.5.6".to_string());
        inputs.push("package-version: 7.8.9".to_string());
    }
    central_profile_wrapper(profile, &inputs)
}

#[test]
fn immutable_central_ci_wrappers_satisfy_executed_contract_rows() {
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            central_ci_wrapper(package, VER),
        )
        .unwrap();

        let report = audit::audit(&root);
        for row in [
            "workflow-env-pins",
            "pin-consistency-guard",
            "toolchain-installer",
            "tests-neg",
        ] {
            assert_eq!(
                verdict_of(&report, row),
                Verdict::Pass,
                "{package} central wrapper did not satisfy {row}: {}",
                diagnostic_of(&report, row)
            );
        }
        assert!(report.ok(), "{package}: central wrapper must audit green");
    }
}

#[test]
fn central_nightly_and_release_profiles_are_real_install_contracts() {
    let cases = [
        (
            "coral",
            "coral-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        ),
        (
            "nautilus",
            "nautilus-nightly",
            vec![
                format!("chelis-tag: v{VER}"),
                format!("chelis-version: {VER}"),
                format!("chelis-linux-sha256: {ARCHIVE_DIGEST}"),
            ],
        ),
        (
            "nautilus",
            "nautilus-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        ),
    ];
    for (package, profile, inputs) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            "name: inert\njobs:\n  no-install:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo no\n",
        )
        .unwrap();
        std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();
        std::fs::write(
            root.join(".github/workflows/central.yml"),
            central_profile_wrapper(profile, &inputs),
        )
        .unwrap();

        let report = audit::audit(&root);
        assert_eq!(
            verdict_of(&report, "workflow-env-pins"),
            Verdict::Pass,
            "{profile} was not recognized: {}",
            diagnostic_of(&report, "workflow-env-pins")
        );
        assert_eq!(
            verdict_of(&report, "toolchain-installer"),
            Verdict::Pass,
            "{profile} was not recognized as an installer"
        );
    }
}

#[test]
fn central_wrapper_pin_drift_fails_workflow_pin_row() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    std::fs::write(
        root.join(".github/workflows/ci.yml"),
        central_ci_wrapper("nautilus", "0.0.1"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert!(diagnostic_of(&report, "workflow-env-pins").contains("0.0.1"));
}

#[test]
fn central_nautilus_profile_binds_blocked_suite() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    std::fs::write(
        root.join(".github/workflows/ci.yml"),
        central_ci_wrapper("nautilus", VER),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("tests_blocked/example")).unwrap();
    std::fs::write(root.join("tests_blocked/example/case.ch"), "1\n").unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "tests-blocked"), Verdict::Pass);
}

#[test]
fn inert_or_skippable_central_markers_do_not_satisfy_audit() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    let inert = [
        "name: inert".to_string(),
        "jobs:".to_string(),
        "  test:".to_string(),
        "    runs-on: ubuntu-latest".to_string(),
        "    steps:".to_string(),
        "      - run: |".to_string(),
        format!(
            "          echo 'uses: Chelis-Lang/ci/.github/workflows/consumer.yml@{CENTRAL_SHA}'"
        ),
        "          echo 'chelis reef conform audit'".to_string(),
        "          echo 'uses: ./.github/actions/install-chelis'".to_string(),
        "          echo 'chelis test tests_neg --expect neg'".to_string(),
    ]
    .join("\n");
    std::fs::write(root.join(".github/workflows/ci.yml"), inert).unwrap();
    std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "pin-consistency-guard"), Verdict::Fail);
    assert_ne!(verdict_of(&report, "toolchain-installer"), Verdict::Pass);
    assert_eq!(verdict_of(&report, "tests-neg"), Verdict::Fail);
}

#[test]
fn central_call_rejects_mutable_skipped_and_wrong_secret_shapes() {
    let cases = [
        central_ci_wrapper("nautilus", VER).replace(CENTRAL_SHA, "main"),
        central_ci_wrapper("nautilus", VER).replace("    with:\n", "    if: false\n    with:\n"),
        central_ci_wrapper("nautilus", VER)
            .replace("${{ secrets.CHELIS_RELEASE_TOKEN }}", "${{ github.token }}"),
        central_ci_wrapper("nautilus", VER).replace("profile: nautilus-ci", "profile: coral-ci"),
    ];
    for (index, wrapper) in cases.into_iter().enumerate() {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "nautilus");
        std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
        let report = audit::audit(&root);
        assert_eq!(
            verdict_of(&report, "pin-consistency-guard"),
            Verdict::Fail,
            "case {index} must not become central authority"
        );
    }
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

// Row 8 (§4) cite-by-number has its own adversarial matrix in
// `tests/row8_citation_matrix.rs` (prose-name Fail, chelis#NNN / draft-path
// Pass, per-entry partition, sub-heading + ordered-list entries, nested-item
// handling, the honest-Manual fallback, §Archived exemption, and the
// malformed-heading fail-closed case). Kept there rather than duplicated here.

/// Tripwire: a MANIFEST row added without a `check_row` dispatch arm falls
/// through to the catch-all `Manual`, silently reporting "no check implemented"
/// instead of a real verdict. Auditing a scaffolded shell exercises every row
/// (the audit iterates all of MANIFEST); assert none carries the catch-all
/// sentinel, so adding a row without a check fails the build (chelis#739).
///
/// Boundary (chelis#739 red team, LOW): this is pin-scoped. `check_row` applies
/// `since_version` gating *before* the dispatch match, so a future row whose
/// `since_version` is above the scaffold's pin returns `Na` and never reaches
/// its arm — the tripwire would not exercise it. Today every row's
/// `since_version` is the contract baseline, so all 18 are exercised; a
/// future-dated row would need its own coverage.
#[test]
fn every_manifest_key_hits_a_real_arm() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "arms");
    let report = audit::audit(&root);
    let orphans: Vec<String> = report
        .rows
        .iter()
        .filter(|r| r.diagnostic.starts_with(audit::NO_CHECK_IMPLEMENTED_PREFIX))
        .map(|r| format!("row {} ({})", r.row, r.key))
        .collect();
    assert!(
        orphans.is_empty(),
        "MANIFEST rows with no check_row arm (fell through to the catch-all): {orphans:?}"
    );
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
