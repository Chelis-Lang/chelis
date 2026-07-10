//! Binary-level smoke for `chelis reef conform bump-check` — the pin-change
//! guard's git path.
//!
//! The guard's substance (pin change ⇒ audit must be green) is proven fast and
//! in-process by `chelis-conformance`'s `conform_bump` test. This exercises the
//! real git-diff wiring: a fresh commit passes; a raw pin edit that skips the
//! checklist (the direct-to-`main` cascade failure mode) fails.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;
use tempfile::tempdir;

fn git(root: &Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .status()
        .expect("run git");
    assert!(status.success(), "git {args:?} failed");
}

fn init_committed_shell(root: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "init",
            "myshell",
            "--module-prefix",
            "Myshell",
            "--output",
        ])
        .arg(root)
        .assert()
        .success();

    git(root, &["init", "-q"]);
    git(root, &["add", "-A"]);
    git(
        root,
        &[
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "init",
        ],
    );
}

#[test]
fn bump_check_passes_when_pin_unchanged() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    init_committed_shell(&root);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "bump-check", "--base", "HEAD", "--path"])
        .arg(&root)
        .assert()
        .success()
        .stdout(predicate::str::contains("unchanged"));
}

#[test]
fn bump_check_fails_on_raw_pin_edit() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    init_committed_shell(&root);

    // Simulate a lazy cascade: bump only reef.toml's pin, skipping the
    // lockstep workflow pins and the managed-block restamp.
    let reef = root.join("reef.toml");
    let text = std::fs::read_to_string(&reef).unwrap();
    let old = format!("compiler = \"={}\"", chelis_compiler_api::COMPILER_VERSION);
    let new = "compiler = \"=99.0.0\"";
    assert!(
        text.contains(&old),
        "reef.toml should contain the current pin"
    );
    std::fs::write(&reef, text.replace(&old, new)).unwrap();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "bump-check", "--base", "HEAD", "--path"])
        .arg(&root)
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("checklist did not run"));
}

#[test]
fn bump_check_fails_closed_on_unresolvable_base() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    init_committed_shell(&root);

    // A base ref that cannot be resolved (the shallow-clone / bad-ref case that
    // used to make the guard silently no-op). It must fail closed, not pass.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "bump-check",
            "--base",
            "origin/definitely-not-a-real-ref",
            "--path",
        ])
        .arg(&root)
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("fails closed"));
}
