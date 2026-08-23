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
        Err(gaps) => gaps.into_iter().map(|g| g.rel).collect(),
    }
}

fn gap_reason(root: &Path, rel: &str) -> String {
    scaffold::preflight_restamp_targets(root)
        .expect_err("expected a preflight gap")
        .into_iter()
        .find(|g| g.rel == rel)
        .unwrap_or_else(|| panic!("no gap for {rel}"))
        .reason
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

/// Existence is not enough for `reef.toml`. A shell whose pin is unreadable has
/// no version for `sync` to stamp, and `sync` used to fall back to the
/// TOOLCHAIN's version, stamp the managed blocks with it, and exit 0. That is
/// the same class of defect chelis#1263 names, in a quieter register: the shell
/// ends up carrying blocks claiming a version it never adopted.
#[test]
fn an_unparseable_compiler_pin_is_refused() {
    for (label, body) in [
        ("no compiler key", "[package]\nname = \"s\"\n"),
        ("range pin", "[package]\ncompiler = \">=0.18.0\"\n"),
        ("no exact marker", "[package]\ncompiler = \"0.18.5\"\n"),
        ("truncated", "[package]\ncompiler = \"=0.18\"\n"),
        ("empty", ""),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path());
        std::fs::write(root.join("reef.toml"), body).unwrap();
        assert_eq!(
            missing_names(&root),
            vec!["reef.toml"],
            "{label}: an unreadable pin must be refused"
        );
        assert!(
            gap_reason(&root, "reef.toml").contains("no readable"),
            "{label}: the reason must say the pin is unreadable, not that the file is missing"
        );
    }
}

/// A manifest that does not parse is refused BEFORE the pin check, because the
/// write path reads more than the pin out of it: `materialize_skills` takes the
/// `[conform] local_skills` allowlist from the same file, so proceeding would
/// prune a repo-local skill the shell did declare.
#[test]
fn an_unparseable_manifest_is_refused() {
    for (label, body) in [
        ("garbage", "not a toml file at all\n"),
        (
            "unterminated array",
            "[package]\ncompiler = \"=0.18.5\"\nx = [\n",
        ),
        ("unterminated header", "[package\ncompiler = \"=0.18.5\"\n"),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path());
        std::fs::write(root.join("reef.toml"), body).unwrap();
        assert_eq!(missing_names(&root), vec!["reef.toml"], "{label}");
        assert!(
            gap_reason(&root, "reef.toml").contains("does not parse as TOML"),
            "{label}: the reason must name the parse failure, got {:?}",
            gap_reason(&root, "reef.toml")
        );
    }
}

#[test]
fn a_readable_pin_that_is_not_the_toolchain_version_still_passes() {
    // Negative parity for the pin check: the preflight requires a pin it can
    // READ, not a pin equal to anything. A shell mid-cascade sits on an older
    // pin and must still be bumpable.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::write(
        root.join("reef.toml"),
        "[package]\nname = \"shell\"\ncompiler = \"=0.9.1\"\nmodule_prefix = \"Shell\"\n",
    )
    .unwrap();
    assert!(scaffold::preflight_restamp_targets(&root).is_ok());
}

#[test]
fn the_refusal_message_names_the_gap_and_the_repair_verb() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path());
    std::fs::remove_file(root.join("docs/CHELIS_SURFACE.md")).unwrap();
    let gaps = scaffold::preflight_restamp_targets(&root).unwrap_err();
    let msg = scaffold::preflight_failure_message("bump", &root, &gaps);
    assert!(msg.contains("docs/CHELIS_SURFACE.md"), "{msg}");
    assert!(msg.contains("Nothing was written"), "{msg}");
    assert!(
        msg.contains("chelis reef conform init"),
        "the message must name the repair verb: {msg}"
    );
    // And must not steer a retrofit agent into losing real source: `init`
    // rewrites the whole scaffold surface, source and tests included.
    for owned in ["src/main.ch", "tests_neg/", "tests_blocked/"] {
        assert!(
            msg.contains(owned),
            "the message must warn that init overwrites {owned}: {msg}"
        );
    }
}
