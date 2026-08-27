//! Row-8 (§4) cite-by-number adversarial matrix.
//!
//! Landed from the chelis#739 red-team probe on PR #765: it stamps a scaffolded
//! shell, overwrites `docs/UPSTREAM_BUGS.md` with an adversarial body, and
//! asserts the exact row-8 verdict + diagnostic. It exercises the laundering
//! vectors a headings-only check missed — prose-name entries, per-entry citation
//! partitioning (a cite must not bleed across entries), citation spacing
//! variants, draft-path vs bare-`#NNN`, nested vs top-level list items,
//! sub-heading entries, the honest-`Manual` fallback, §Archived exemption, and
//! the malformed-heading fail-closed case (the MEDIUM the red team found).

use std::path::{Path, PathBuf};

use chelis_conformance::audit::{self, Verdict};
use chelis_conformance::scaffold;

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

fn row_of<'a>(report: &'a audit::AuditReport, key: &str) -> &'a audit::RowResult {
    report.rows.iter().find(|r| r.key == key).unwrap()
}

/// Overwrite UPSTREAM_BUGS.md with a fully caller-controlled body and audit.
fn audit_with_bugs(root: &Path, doc: &str) -> audit::AuditReport {
    std::fs::write(root.join("docs/UPSTREAM_BUGS.md"), doc).unwrap();
    audit::audit(root)
}

/// A canonical 4-section doc with caller-supplied section bodies (Actively
/// blocking / Tracking / Parked / Archived), all headings well-formed.
fn doc(active: &str, tracking: &str, parked: &str, archived: &str) -> String {
    format!(
        "# Upstream Bugs\n\nintro line\n\n\
         ## Actively blocking\n\n{active}\n\n\
         ## Tracking\n\n{tracking}\n\n\
         ## Parked\n\n{parked}\n\n\
         ## Archived\n\n{archived}\n"
    )
}

// A citation in the section PREAMBLE must not rescue the entry below it.
#[test]
fn preamble_citation_does_not_rescue_uncited_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "preamble");
    let tracking = "Filed under chelis#293 (this is preamble prose, not an entry).\n\n\
                    - the actual bug, prose name only";
    let report = audit_with_bugs(
        &root,
        &doc("(none yet)", tracking, "(none yet)", "(none yet)"),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Fail,
        "a chelis#NNN in the preamble must NOT attribute to the entry below it"
    );
    let r = row_of(&report, "upstream-bugs");
    assert!(
        r.diagnostic.contains("1 entry with a prose-name"),
        "diag: {}",
        r.diagnostic
    );
    // Evidence must name the uncited entry, not the preamble.
    assert!(
        r.evidence.iter().any(|e| e.contains("the actual bug")),
        "evidence: {:?}",
        r.evidence
    );
}

// A citation on entry A must not cover an uncited entry B: count == 1, evidence
// names B and not A.
#[test]
fn per_entry_partition_a_cited_b_uncited() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "partition");
    let tracking = "- entry A is fine (chelis#293)\n- entry B has only a prose name";
    let report = audit_with_bugs(
        &root,
        &doc("(none yet)", tracking, "(none yet)", "(none yet)"),
    );
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("1 entry with a prose-name"),
        "exactly one uncited entry expected; diag: {}",
        r.diagnostic
    );
    assert!(
        r.evidence.iter().any(|e| e.contains("entry B")),
        "evidence must cite entry B: {:?}",
        r.evidence
    );
    assert!(
        !r.evidence.iter().any(|e| e.contains("entry A")),
        "entry A is cited and must NOT appear as uncited: {:?}",
        r.evidence
    );
}

// `chelis #316` (space before the number) normalizes and counts.
#[test]
fn spacing_variant_hash_space() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "sp1");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "- spaced cite chelis #316",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Pass,
        "chelis #316 should count"
    );
}

// `chelis # 316` (space around the hash) normalizes and counts.
#[test]
fn spacing_variant_hash_space_number() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "sp2");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "- widely spaced chelis # 316",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Pass,
        "chelis # 316 (space around #) should normalize and count"
    );
}

// A bare `#316` with no `chelis` prefix is not a citation.
#[test]
fn bare_hash_number_is_not_a_citation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "bare");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "- see #316 for details (no chelis prefix)",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Fail,
        "a bare #316 with no chelis prefix must not count as a citation"
    );
}

// A `docs/issue_drafts/<name>` path is the sanctioned alternative and passes.
#[test]
fn draft_path_plural_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "draftok");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "- parked as docs/issue_drafts/foo.md until filed",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Pass);
}

// The singular `docs/issue_draft/` directory is not the sanctioned path.
#[test]
fn draft_path_singular_does_not_count() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "draftbad");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "- parked as docs/issue_draft/foo.md (singular dir, wrong)",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Fail,
        "docs/issue_draft/ (singular) is not the sanctioned docs/issue_drafts/ path"
    );
}

// A 4-space-indented item is nested (part of its parent entry), not its own.
#[test]
fn four_space_indented_item_travels_with_parent() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nested");
    let tracking = "- parent bug (chelis#293)\n    - nested detail, no cite of its own";
    let report = audit_with_bugs(
        &root,
        &doc("(none yet)", tracking, "(none yet)", "(none yet)"),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Pass,
        "a 4-space-indented child is part of its parent entry, not a new uncited entry"
    );
}

// A section whose only content is a deeply-nested item has no top-level entry:
// honest Manual, never a silent Pass.
#[test]
fn four_space_indented_item_alone_is_manual_not_fail() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nestedonly");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "    - deeply nested only, prose name",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Manual);
}

// A `###` sub-heading under `## Tracking` is an entry; uncited -> Fail.
#[test]
fn subheading_entry_is_checked_and_can_fail() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "subhead");
    let tracking = "### generic-callback-unification limit\nblocks the training loop";
    let report = audit_with_bugs(
        &root,
        &doc("(none yet)", tracking, "(none yet)", "(none yet)"),
    );
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(
        r.verdict,
        Verdict::Fail,
        "an uncited ### sub-heading entry must fail"
    );
    assert!(
        r.evidence.iter().any(|e| e.contains("generic-callback")),
        "evidence: {:?}",
        r.evidence
    );
}

#[test]
fn subheading_entry_cited_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "subheadok");
    let tracking = "### generic-callback limit (chelis#293)\nblocks the training loop";
    let report = audit_with_bugs(
        &root,
        &doc("(none yet)", tracking, "(none yet)", "(none yet)"),
    );
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Pass);
    // A clean Pass carries no diagnostic (locks the cited-pass path).
    assert!(row_of(&report, "upstream-bugs").diagnostic.is_empty());
}

// A DUPLICATED section heading orphans the second occurrence's body: with only
// the first `## Tracking` body checked, an uncited entry under a second
// `## Tracking` used to audit a silent Pass. A duplicated required section is a
// malformed doc and now fails closed (chelis#739 RT delta residual).
#[test]
fn duplicated_section_heading_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "dup");
    let body = "# Upstream Bugs\n\nintro\n\n\
        ## Actively blocking\n\n(none yet)\n\n\
        ## Tracking\n\n- cited entry (chelis#1)\n\n\
        ## Tracking\n\n- generic-callback-unification limit blocks training (prose, uncited)\n\n\
        ## Parked\n\n(none yet)\n\n\
        ## Archived\n\n(none yet)\n";
    let report = audit_with_bugs(&root, body);
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(
        r.verdict,
        Verdict::Fail,
        "an uncited entry under a duplicated `## Tracking` must not silently pass"
    );
    assert!(
        r.diagnostic.contains("duplicated") && r.diagnostic.contains("Tracking"),
        "diag should name the duplicated section: {}",
        r.diagnostic
    );
    assert!(!report.ok());
}

// §Archived uncited entries are exempt (closed history) -> Pass.
#[test]
fn archived_uncited_is_exempt() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "arch");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "(none yet)",
            "(none yet)",
            "- long-closed prose-name bug, no number at all",
        ),
    );
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Pass);
}

// A section with content but no parseable entry -> honest Manual with a
// non-empty diagnostic, non-gating.
#[test]
fn prose_only_section_is_manual_with_diagnostic() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "prose");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "We are tracking a suspected miscompile but have not filed it yet.",
            "(none yet)",
            "(none yet)",
        ),
    );
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(r.verdict, Verdict::Manual);
    assert!(!r.diagnostic.is_empty(), "Manual must render a diagnostic");
    assert!(r.diagnostic.contains("Tracking"), "diag: {}", r.diagnostic);
    assert!(report.ok(), "an honest Manual must not gate the audit");
}

// FLIPPED from the red-team residual (was asserting the Pass false-green): a
// malformed no-space heading (`##Actively blocking`) is now a fail-closed
// missing/malformed-heading Fail. Previously the lenient structural gate
// accepted it while the strict `section_body` locator returned None, silently
// skipping the uncited entry beneath it — the MEDIUM the red team confirmed via
// the CLI (row 8 PASS, exit 0). The gate now uses the same strict locator.
#[test]
fn malformed_no_space_heading_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "malformed");
    let body = "# Upstream Bugs\n\nintro\n\n\
        ##Actively blocking\n\n- generic-callback-unification limit blocks training (prose name)\n\n\
        ## Tracking\n\n(none yet)\n\n\
        ## Parked\n\n(none yet)\n\n\
        ## Archived\n\n(none yet)\n";
    let report = audit_with_bugs(&root, body);
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(
        r.verdict,
        Verdict::Fail,
        "a malformed no-space heading must fail closed, not orphan its entry into a Pass"
    );
    assert!(
        r.diagnostic.contains("malformed") && r.diagnostic.contains("Actively blocking"),
        "diag should name the malformed section: {}",
        r.diagnostic
    );
    assert!(!report.ok());
}

// A prose-paragraph bug (no list marker / heading) downgrades the MUST from a
// gating Fail to a non-gating Manual. Sanctioned by chelis#739 as the honest
// fallback: the cite-by-number MUST is enforced only for list/heading shapes,
// and prose is surfaced for human review rather than silently passed.
#[test]
fn prose_paragraph_bug_is_non_gating_manual() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "prosedodge");
    let report = audit_with_bugs(
        &root,
        &doc(
            "The generic-callback-unification limit blocks the training loop; no number filed.",
            "(none yet)",
            "(none yet)",
            "(none yet)",
        ),
    );
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(
        r.verdict,
        Verdict::Manual,
        "a prose-paragraph bug is Manual (non-gating), not the hard Fail a list item gets"
    );
    assert!(
        report.ok(),
        "Manual does not gate -> the shell still audits ok()"
    );
}

// A trailing blanket citation after N uncited entries covers only the LAST entry
// (it travels with the entry above it); earlier entries still Fail.
#[test]
fn trailing_blanket_citation_only_covers_last_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "blanket");
    let tracking =
        "- bug one, prose name\n- bug two, prose name\n\nAll of the above filed as chelis#999.";
    let report = audit_with_bugs(
        &root,
        &doc("(none yet)", tracking, "(none yet)", "(none yet)"),
    );
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(r.verdict, Verdict::Fail);
    // Only bug one remains uncited (bug two absorbs the trailing citation line).
    assert!(
        r.diagnostic.contains("1 entry with a prose-name"),
        "trailing blanket cite should rescue only the last entry; diag: {}",
        r.diagnostic
    );
}

// Ordered list markers (`N.` / `N)`) are entries too.
#[test]
fn ordered_list_uncited_entry_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "ordered");
    let report = audit_with_bugs(
        &root,
        &doc(
            "(none yet)",
            "1. prose-named ordered bug",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Fail);
}

// ------------------------------------------------------------------ chelis#1270
//
// The §4 grammar accepts a registry sibling's `<repo>#NNN`. Before this, the
// literal blocking artifact of the 0.18.5 cascade (`nautilus#43`) failed row 8
// as a prose-name citation, and the shell that hit it had to manufacture a
// `docs/issue_drafts/` file whose only content was a pointer at that PR: a
// citation handle wearing a draft's costume, which satisfies the grammar while
// defeating its purpose. What the rule is actually buying is a live, checkable,
// dedupable reference, and registry membership supplies exactly that.

#[test]
fn sibling_repo_citation_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "sibling");
    let report = audit_with_bugs(
        &root,
        &doc(
            "- blocked on the sibling release: nautilus#43",
            "(none yet)",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(
        verdict_of(&report, "upstream-bugs"),
        Verdict::Pass,
        "a registry sibling's issue is a citation"
    );
    assert!(report.ok());
}

#[test]
fn every_registry_shell_is_an_accepted_citation() {
    // Locks the coupling rather than spot-checking: a shell added to REGISTRY
    // becomes citable by that fact alone.
    for shell in chelis_conformance::registry::REGISTRY {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "regshell");
        let entry = format!("- blocked on {}#7", shell.name);
        let report = audit_with_bugs(
            &root,
            &doc(&entry, "(none yet)", "(none yet)", "(none yet)"),
        );
        assert_eq!(
            verdict_of(&report, "upstream-bugs"),
            Verdict::Pass,
            "{}#7 must be an accepted citation",
            shell.name
        );
    }
}

#[test]
fn org_qualified_sibling_citation_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "orgqual");
    let report = audit_with_bugs(
        &root,
        &doc(
            "- blocked on Chelis-Lang/coral#27",
            "(none yet)",
            "(none yet)",
            "(none yet)",
        ),
    );
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Pass);
}

#[test]
fn an_arbitrary_repo_reference_is_still_rejected() {
    // The negative control the widening owes. Registry membership is the whole
    // property; a grammar that took any `word#NNN` would check nothing.
    for entry in [
        "- blocked on torch#43",
        "- blocked on numpy #7",
        "- blocked on chelis-std#12",
        "- blocked on Chelis-Lang#5",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "arbitrary");
        let report = audit_with_bugs(&root, &doc(entry, "(none yet)", "(none yet)", "(none yet)"));
        let r = row_of(&report, "upstream-bugs");
        assert_eq!(
            r.verdict,
            Verdict::Fail,
            "{entry:?} names no repo the audit can resolve"
        );
        assert!(!report.ok());
    }
}

#[test]
fn the_uncited_diagnostic_names_the_sibling_form() {
    // A shell hitting this failure should learn the accepted forms from the
    // message, not from the source.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "teach");
    let report = audit_with_bugs(
        &root,
        &doc(
            "- blocked on the nautilus release, no number",
            "(none yet)",
            "(none yet)",
            "(none yet)",
        ),
    );
    let r = row_of(&report, "upstream-bugs");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("1 entry with a prose-name"),
        "diag: {}",
        r.diagnostic
    );
    assert!(
        r.fix.contains("nautilus#43") || r.fix.contains("<repo>#NNN"),
        "the fix must name the sibling form: {}",
        r.fix
    );
}

// A fresh scaffold's `(none yet)` placeholder body is end-to-end green.
#[test]
fn fresh_scaffold_row8_pass_and_ok() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "fresh");
    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "upstream-bugs"), Verdict::Pass);
    assert!(report.ok());
}
