//! Escape-hatch + diagnostic seams added from the first downstream `conform`
//! run (School #171): the `[conform] local_skills` allowlist (chelis#651), the
//! trailing shell-local block (chelis#653), and the §4 whitespace-tolerant
//! citation match + `--explain` evidence (chelis#652 / #654).

use std::path::Path;

use chelis_conformance::audit;
use chelis_conformance::scaffold;

const VER: &str = env!("CARGO_PKG_VERSION");

fn green_shell() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", VER).expect("scaffold");
    assert!(audit::audit(&root).ok(), "baseline must be green");
    (tmp, root)
}

fn row<'a>(report: &'a audit::AuditReport, key: &str) -> &'a audit::RowResult {
    report
        .rows
        .iter()
        .find(|r| r.key == key)
        .unwrap_or_else(|| panic!("no row {key}"))
}

fn append(path: &Path, extra: &str) {
    let mut t = std::fs::read_to_string(path).unwrap();
    t.push_str(extra);
    std::fs::write(path, t).unwrap();
}

// ---------------------------------------------------------------- #651 allowlist

#[test]
fn declared_local_skill_survives_sync_and_audit() {
    let (_tmp, root) = green_shell();
    // A repo-local domain skill the shared set does not include.
    std::fs::create_dir_all(root.join("agent-skills/chelis-std")).unwrap();
    std::fs::write(
        root.join("agent-skills/chelis-std/SKILL.md"),
        "# chelis-std domain skill\nShell-authored.\n",
    )
    .unwrap();
    // Undeclared, it is drift.
    assert_eq!(
        row(&audit::audit(&root), "vendored-skills").verdict,
        audit::Verdict::Fail
    );

    // Declare it; now §8 exempts it and sync preserves it.
    append(
        &root.join("reef.toml"),
        "\n[conform]\nlocal_skills = [\"chelis-std\"]\n",
    );
    assert!(
        audit::audit(&root).ok(),
        "declared local skill must not fail §8"
    );
    let notices = scaffold::materialize_skills(&root).unwrap();
    assert!(
        root.join("agent-skills/chelis-std/SKILL.md").exists(),
        "sync must preserve a declared local skill"
    );
    assert!(
        !notices.iter().any(|n| n.contains("chelis-std")),
        "a declared local skill must not warn as pruned: {notices:?}"
    );
    assert!(audit::audit(&root).ok());
}

#[test]
fn undeclared_extra_skill_is_pruned_with_a_warning() {
    let (_tmp, root) = green_shell();
    std::fs::create_dir_all(root.join("agent-skills/rogue")).unwrap();
    std::fs::write(root.join("agent-skills/rogue/SKILL.md"), "rogue\n").unwrap();

    let notices = scaffold::materialize_skills(&root).unwrap();
    assert!(
        !root.join("agent-skills/rogue").exists(),
        "an undeclared skill is still pruned"
    );
    assert!(
        notices.iter().any(|n| n.contains("rogue")),
        "prune must warn before deleting an author-added skill: {notices:?}"
    );
    assert!(audit::audit(&root).ok());
}

#[test]
fn local_skills_may_not_shadow_a_shared_skill() {
    let (_tmp, root) = green_shell();
    append(
        &root.join("reef.toml"),
        "\n[conform]\nlocal_skills = [\"example-corpus\"]\n",
    );
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, audit::Verdict::Fail);
    assert!(r.diagnostic.contains("example-corpus"), "{}", r.diagnostic);
}

// ------------------------------------------------------------- #653 shell-local

const BLOCK: &str = "<!-- shell-local:begin -->\n## School override\nexamples/ is a reef source root.\n<!-- shell-local:end -->\n";

#[test]
fn shell_local_block_passes_audit_and_survives_sync() {
    let (_tmp, root) = green_shell();
    let skill = root.join("agent-skills/example-corpus/SKILL.md");
    append(&skill, &format!("\n{BLOCK}"));

    assert!(
        audit::audit(&root).ok(),
        "a well-formed shell-local block is exempt from §8"
    );

    // Sync regenerates the managed span but keeps the block verbatim.
    let notices = scaffold::materialize_skills(&root).unwrap();
    assert!(
        notices.is_empty(),
        "no change-flag when upstream is unchanged: {notices:?}"
    );
    let after = std::fs::read_to_string(&skill).unwrap();
    assert!(
        after.contains("## School override"),
        "sync dropped the block"
    );
    assert!(audit::audit(&root).ok());
}

#[test]
fn shell_local_change_flag_fires_when_upstream_span_diverges() {
    let (_tmp, root) = green_shell();
    let skill = root.join("agent-skills/example-corpus/SKILL.md");
    // Simulate the previous toolchain's (now-divergent) body under a block by
    // prepending a stale line to the managed span.
    let body = std::fs::read_to_string(&skill).unwrap();
    std::fs::write(&skill, format!("STALE UPSTREAM LINE\n{body}\n{BLOCK}")).unwrap();

    let notices = scaffold::materialize_skills(&root).unwrap();
    assert!(
        notices
            .iter()
            .any(|n| n.contains("example-corpus") && n.contains("re-check")),
        "a diverged managed span under a block must flag for re-check: {notices:?}"
    );
    let after = std::fs::read_to_string(&skill).unwrap();
    assert!(
        !after.contains("STALE UPSTREAM LINE"),
        "managed span must regenerate"
    );
    assert!(after.contains("## School override"), "block must survive");
    assert!(audit::audit(&root).ok());
}

#[test]
fn malformed_shell_local_block_fails_audit() {
    let (_tmp, root) = green_shell();
    append(
        &root.join("agent-skills/spec-sync/SKILL.md"),
        "\n<!-- shell-local:begin -->\nno end marker\n",
    );
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, audit::Verdict::Fail);
    assert!(r.diagnostic.contains("shell-local:end"), "{}", r.diagnostic);
}

#[test]
fn content_after_end_marker_fails_audit() {
    let (_tmp, root) = green_shell();
    append(
        &root.join("agent-skills/spec-sync/SKILL.md"),
        "\n<!-- shell-local:begin -->\n## override\n<!-- shell-local:end -->\ntrailing prose\n",
    );
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, audit::Verdict::Fail);
    assert!(
        r.diagnostic.contains("content after") && r.diagnostic.contains("file suffix"),
        "a block that is not the file suffix must be rejected: {}",
        r.diagnostic
    );
}

#[test]
fn two_begin_markers_fail_audit() {
    let (_tmp, root) = green_shell();
    append(
        &root.join("agent-skills/spec-sync/SKILL.md"),
        "\n<!-- shell-local:begin -->\n## a\n<!-- shell-local:end -->\n\
         <!-- shell-local:begin -->\n## b\n<!-- shell-local:end -->\n",
    );
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, audit::Verdict::Fail);
    assert!(
        r.diagnostic.contains("exactly one"),
        "two begin markers must be rejected: {}",
        r.diagnostic
    );
}

#[test]
fn sync_with_a_block_is_byte_stable_across_repeated_runs() {
    let (_tmp, root) = green_shell();
    let skill = root.join("agent-skills/example-corpus/SKILL.md");
    append(&skill, &format!("\n{BLOCK}"));

    // First materialize normalizes the layout; every subsequent materialize must
    // be a byte-for-byte no-op (no drift, no re-flagging, no block duplication).
    scaffold::materialize_skills(&root).unwrap();
    let once = std::fs::read_to_string(&skill).unwrap();
    let notices = scaffold::materialize_skills(&root).unwrap();
    let twice = std::fs::read_to_string(&skill).unwrap();
    assert_eq!(
        once, twice,
        "re-materialize must be byte-stable under a block"
    );
    assert!(
        notices.is_empty(),
        "an unchanged upstream body must not re-flag: {notices:?}"
    );
    assert_eq!(
        twice.matches("<!-- shell-local:begin -->").count(),
        1,
        "the block must not be duplicated"
    );
    assert!(audit::audit(&root).ok());
}

// ------------------------------------------------- #652 / #654 §4 citation seam

#[test]
fn spaced_coverage_entry_covers_a_cite_and_explain_names_the_site() {
    let (_tmp, root) = green_shell();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/thing.ch"),
        "def x() -> I32 = fail(\"blocked on chelis#316\")\n",
    )
    .unwrap();

    // Uncovered: row 9 fails and --explain names the citation site.
    let report = audit::audit(&root);
    let r = row(&report, "staleness-audit");
    assert_eq!(r.verdict, audit::Verdict::Fail);
    assert!(
        r.evidence.iter().any(|e| e.contains("src/thing.ch:1")),
        "explain evidence must name the site: {:?}",
        r.evidence
    );

    // A space-form coverage entry (`chelis #316`) now counts (chelis#652).
    append(
        &root.join("docs/UPSTREAM_BUGS.md"),
        "\n## Tracking\n- tracked as chelis #316 upstream (re-probe next release)\n",
    );
    // The §4 staleness row specifically must now pass (a `tests_blocked/` probe
    // for the cited blocker is a separate §5 requirement, orthogonal to #652).
    let after = audit::audit(&root);
    let staleness = row(&after, "staleness-audit");
    assert_eq!(
        staleness.verdict,
        audit::Verdict::Pass,
        "a whitespace-variant coverage entry must cover the cite (chelis#652); got: {}",
        staleness.diagnostic
    );
    // The now-passing §4 row carries no residual evidence (chelis#654: evidence
    // is per-finding, so a pass is empty — the negative parity for the populated
    // failing-row assertion above).
    assert!(
        staleness.evidence.is_empty(),
        "a passing row must carry no explain evidence: {:?}",
        staleness.evidence
    );
}

// --------------------------------------------- #1270 sibling-repo citation seam

#[test]
fn a_sibling_narrowing_citation_is_visible_to_row_9_and_owes_coverage() {
    // Row 8 and row 9 read the same grammar, so widening it widens cite AND
    // coverage together: a `nautilus#43` narrowing stops reading as uncited
    // prose, and in exchange it owes the same coverage an upstream cite owes.
    let (_tmp, root) = green_shell();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/thing.ch"),
        "def x() -> I32 = fail(\"blocked on nautilus#43\")\n",
    )
    .unwrap();

    let before = audit::audit(&root);
    let r = row(&before, "staleness-audit");
    assert_eq!(
        r.verdict,
        audit::Verdict::Fail,
        "an uncovered sibling narrowing must fail row 9"
    );
    assert!(
        r.diagnostic.contains("nautilus#43"),
        "diag must name the sibling token: {}",
        r.diagnostic
    );

    append(
        &root.join("docs/UPSTREAM_BUGS.md"),
        "\n## Tracking\n- blocked on nautilus#43 until the sibling cuts its release\n",
    );
    let after = audit::audit(&root);
    assert_eq!(
        row(&after, "staleness-audit").verdict,
        audit::Verdict::Pass,
        "an entry citing the same sibling token covers it"
    );
}

#[test]
fn a_hello_chelis_citation_is_not_covered_by_an_upstream_entry() {
    // The behavioral form of the leftward-anchoring control. `hello-chelis` ends
    // in `chelis`; a substring scan would read `hello-chelis#43` as an upstream
    // `chelis#43` too, and then a `chelis#43` coverage entry would silently
    // satisfy a sibling narrowing that nobody actually tracked.
    let (_tmp, root) = green_shell();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/thing.ch"),
        "def x() -> I32 = fail(\"blocked on hello-chelis#43\")\n",
    )
    .unwrap();
    append(
        &root.join("docs/UPSTREAM_BUGS.md"),
        "\n## Tracking\n- tracked as chelis#43 upstream\n",
    );

    let before = audit::audit(&root);
    let r = row(&before, "staleness-audit");
    assert_eq!(
        r.verdict,
        audit::Verdict::Fail,
        "an upstream entry must not cover a same-numbered sibling cite"
    );
    assert!(
        r.diagnostic.contains("hello-chelis#43"),
        "the orphan is the sibling token, not the upstream one: {}",
        r.diagnostic
    );

    // Its own entry does cover it (negative parity for the assertion above).
    append(
        &root.join("docs/UPSTREAM_BUGS.md"),
        "- and separately hello-chelis#43 in the shell\n",
    );
    let after = audit::audit(&root);
    assert_eq!(row(&after, "staleness-audit").verdict, audit::Verdict::Pass);
}

#[test]
fn an_unregistered_repo_narrowing_is_not_a_citation_at_all() {
    // Negative control: widening to the registry must not turn every
    // `word#NNN` into a tracked citation. `torch#43` is not scanned, so row 9
    // sees no narrowing to cover and stays green on that basis alone. (The
    // narrowing itself is still uncited prose under §4, which row 8 owns for
    // UPSTREAM_BUGS entries.)
    let (_tmp, root) = green_shell();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/thing.ch"),
        "def x() -> I32 = fail(\"blocked on torch#43\")\n",
    )
    .unwrap();
    let report = audit::audit(&root);
    assert_eq!(
        row(&report, "staleness-audit").verdict,
        audit::Verdict::Pass,
        "an unregistered repo reference is not a citation the audit tracks"
    );
}
