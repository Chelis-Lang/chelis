//! Phase 4 oracle: the version-propagation arm's core invariant.
//!
//! A pin bump that rewrites the pins but skips the managed-block restamp leaves
//! the pointer stamp behind the pin — and the audit MUST catch that. This is the
//! exact failure mode that raw direct-to-`main` cascades used to slip past; here
//! `conform bump-check` (which requires a green audit when the pin changed) would
//! block the PR until `conform bump` (which follows the rewrite with `sync`)
//! restores conformance.

use chelis_conformance::{audit, bump, scaffold};

#[test]
fn pin_rewrite_without_sync_is_caught_then_fixed() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", "0.14.0").expect("scaffold");
    assert!(audit::audit(&root).ok(), "baseline green");

    // Mechanical pin rewrite (reef.toml + workflow envs) to a new version.
    let changed = bump::rewrite_pins(&root, "0.15.0").expect("rewrite");
    assert!(
        changed.iter().any(|p| p.ends_with("reef.toml")),
        "reef.toml must be repinned"
    );
    assert!(
        changed
            .iter()
            .any(|p| p.to_string_lossy().contains("ci.yml")),
        "ci.yml env pins must be repinned in lockstep"
    );

    // Audit now FAILS: the managed-block stamps still read chelis@0.14.0 while
    // reef.toml pins =0.15.0. This is the guard that a raw cascade trips.
    let after_rewrite = audit::audit(&root);
    assert!(
        !after_rewrite.ok(),
        "a pin bump without a restamp must fail the audit"
    );
    let stale = after_rewrite
        .rows
        .iter()
        .find(|r| r.key == "agents-md")
        .unwrap();
    assert_eq!(stale.verdict, audit::Verdict::Fail);
    assert!(
        stale.diagnostic.contains("0.14.0") && stale.diagnostic.contains("0.15.0"),
        "diagnostic should name both the stale stamp and the new pin: {}",
        stale.diagnostic
    );

    // The bump completes by re-materializing the skills (their links are
    // pinned to the release) and restamping, which is what `conform bump`
    // does after the rewrite.
    scaffold::materialize_skills(&root, "0.15.0").expect("materialize");
    scaffold::sync_managed_blocks(&root, "0.15.0").expect("sync");
    assert!(
        audit::audit(&root).ok(),
        "after restamp, the bumped shell is conformant again"
    );
}

#[test]
fn rewrite_is_noop_when_already_at_version() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", "0.14.0").expect("scaffold");
    let changed = bump::rewrite_pins(&root, "0.14.0").expect("rewrite");
    assert!(changed.is_empty(), "no files change when already pinned");
}
