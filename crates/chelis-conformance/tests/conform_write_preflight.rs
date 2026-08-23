//! chelis#1263: `conform sync` / `conform bump` must fail closed on a repo that
//! has not been conformed, before writing anything.
//!
//! The measured 0.18.5 behavior on two never-conformed repos was a *partial
//! application*: the edit sequence (repin -> materialize skills -> restamp
//! blocks) ran until it reached the first missing artifact and left everything
//! before that point on disk. A tree missing `AGENTS.md` came back repinned with
//! `agent-skills/` materialized; a tree missing `docs/CHELIS_SURFACE.md` came
//! back with all of that plus a restamped `AGENTS.md`. The wave also observed
//! the same shapes reported as success by a scripted caller.
//!
//! These tests fix the *precondition*, which is the part a library test can own:
//! `preflight_restamp_targets` names every artifact the write path restamps in
//! place, so the CLI can refuse before its first write. The end-to-end refusal
//! (exit code, message, and the on-disk proof that nothing moved) is locked in
//! `chelis-cli/tests/conform_bump_smoke.rs`.

use std::path::Path;

use chelis_conformance::scaffold;

const VER: &str = env!("CARGO_PKG_VERSION");

fn stamp(dir: &Path) -> std::path::PathBuf {
    let root = dir.join("shell");
    scaffold::scaffold(&root, "shell", "Shell", VER).expect("scaffold");
    root
}

fn missing_names(root: &Path) -> Vec<&'static str> {
    match scaffold::preflight_restamp_targets(root) {
        Ok(()) => Vec::new(),
        Err(missing) => missing.into_iter().map(|(rel, _)| rel).collect(),
    }
}

#[test]
fn a_conformed_shell_passes_preflight() {
    // Negative parity for every refusal below: the sanctioned state must not be
    // refused. A preflight that blocked a normal bump would be worse than the
    // partial write it replaces.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    assert!(
        scaffold::preflight_restamp_targets(&root).is_ok(),
        "a freshly stamped shell must be bumpable"
    );
}

#[test]
fn missing_agents_md_is_refused() {
    // The hello-chelis shape.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::remove_file(root.join("AGENTS.md")).unwrap();
    assert_eq!(missing_names(&root), vec!["AGENTS.md"]);
}

#[test]
fn missing_chelis_surface_is_refused() {
    // The c-earchin / calcify shape.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::remove_file(root.join("docs/CHELIS_SURFACE.md")).unwrap();
    assert_eq!(missing_names(&root), vec!["docs/CHELIS_SURFACE.md"]);
}

#[test]
fn missing_reef_toml_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::remove_file(root.join("reef.toml")).unwrap();
    assert_eq!(missing_names(&root), vec!["reef.toml"]);
}

#[test]
fn a_never_conformed_repo_reports_every_missing_artifact_at_once() {
    // The point of preflighting rather than failing at the first bad read: an
    // agent retrofitting a repo should learn the whole gap in one run, not
    // discover the next missing file only after fixing the previous one.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("bare");
    std::fs::create_dir_all(root.join("docs")).unwrap();
    let missing = missing_names(&root);
    assert_eq!(
        missing,
        vec!["reef.toml", "AGENTS.md", "docs/CHELIS_SURFACE.md"],
        "all three restamp targets are reported together"
    );
}

#[test]
fn a_directory_in_place_of_a_managed_document_is_refused() {
    // `is_file`, not `exists`: a directory named AGENTS.md would pass an
    // existence check and then fail the read the preflight exists to prevent.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::remove_file(root.join("AGENTS.md")).unwrap();
    std::fs::create_dir(root.join("AGENTS.md")).unwrap();
    assert_eq!(missing_names(&root), vec!["AGENTS.md"]);
}

#[test]
fn the_refusal_message_names_the_gap_and_the_repair_verb() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::remove_file(root.join("docs/CHELIS_SURFACE.md")).unwrap();
    let missing = scaffold::preflight_restamp_targets(&root).unwrap_err();
    let msg = scaffold::preflight_failure_message("bump", &root, &missing);
    assert!(msg.contains("docs/CHELIS_SURFACE.md"), "{msg}");
    assert!(msg.contains("Nothing was written"), "{msg}");
    assert!(
        msg.contains("chelis reef conform init"),
        "the message must name the repair verb: {msg}"
    );
}
