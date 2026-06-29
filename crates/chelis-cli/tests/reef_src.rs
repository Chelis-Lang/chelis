//! Phase 4 — `chelis reef src` / `chelis reef doctor` integration tests.
//!
//! Exercises the source-crate dependency manager end to end through the real
//! `chelis` binary, fully offline:
//!   * `CHELIS_SRC_REMOTE` points the canonical fetch at a LOCAL fixture git
//!     repo (the test-injection seam paralleling `CHELIS_REEF_GITHUB_BASE_API`),
//!   * `CHELIS_SRC_HOME` isolates the version-keyed store inside a tempdir.
//!
//! Coverage:
//!   * `src sync` builds the mirror + worktree and links `../chelis`; a
//!     follow-up `src check` passes.
//!   * `src check` fails loudly when `Cargo.lock` drifts off the pin (the
//!     Phase 1.5 `.cargo`-override failure mode).
//!   * `src sync` errors cleanly on a shell with no `[chelis-src]` section.
//!   * `doctor` discovers a crate-linking shell and reports its `src` status.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use std::process::Command as Proc;
use tempfile::tempdir;

fn git(dir: &Path, args: &[&str]) {
    let status = Proc::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .expect("spawn git");
    assert!(status.success(), "git {args:?} failed in {}", dir.display());
}

fn git_out(dir: &Path, args: &[&str]) -> String {
    let out = Proc::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Build a canonical-style chelis repo with one commit tagged `tag`.
/// Returns the commit SHA.
fn init_canonical(dir: &Path, tag: &str) -> String {
    fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q"]);
    git(dir, &["config", "user.email", "fixture@example.invalid"]);
    git(dir, &["config", "user.name", "Fixture"]);
    git(dir, &["config", "gc.auto", "0"]);
    fs::write(dir.join("VERSION"), "fixture\n").unwrap();
    git(dir, &["add", "."]);
    git(dir, &["commit", "-q", "-m", "init"]);
    git(dir, &["tag", tag]);
    git_out(dir, &["rev-parse", "HEAD"])
}

/// Write a crate-linking shell package at `dir` pinned to `pin`.
fn write_shell(dir: &Path, pin: &str, crates: &[&str]) {
    fs::create_dir_all(dir.join("src")).unwrap();
    let list = crates
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        dir.join("reef.toml"),
        format!(
            "[package]\nname = \"shelly\"\nversion = \"0.1.0\"\n\
             compiler = \"={pin}\"\nmodule_prefix = \"Shelly\"\n\n\
             [chelis-src]\ncrates = [{list}]\n"
        ),
    )
    .unwrap();
}

/// A `chelis` invocation with the store env wired to the fixtures.
fn chelis(store: &Path, remote: &Path) -> Command {
    let mut c = Command::cargo_bin("chelis").expect("chelis binary built");
    c.env("CHELIS_SRC_HOME", store)
        .env("CHELIS_SRC_REMOTE", remote)
        // Keep token resolution from reaching out; a local remote needs none.
        .env_remove("GITHUB_TOKEN");
    c
}

#[test]
fn reef_src_sync_links_slot_and_check_passes_at_pin() {
    let tmp = tempdir().unwrap();
    let canonical = tmp.path().join("canonical");
    init_canonical(&canonical, "v0.0.1");
    let store = tmp.path().join("src-home");
    let ws = tmp.path().join("ws");
    let shell = ws.join("shelly");
    write_shell(&shell, "0.0.1", &["chelis-types"]);

    chelis(&store, &canonical)
        .args(["reef", "src", "sync", "--path"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("synced chelis source crates"));

    // The store worktree exists and `../chelis` is a symlink into it.
    assert!(
        store.join("0.0.1").join(".git").exists(),
        "worktree created"
    );
    let slot = ws.join("chelis");
    let target = fs::read_link(&slot).expect("../chelis is a symlink");
    assert_eq!(target, store.join("0.0.1"));

    // A lockfile at the pin makes `check` clean.
    fs::write(
        shell.join("Cargo.lock"),
        "[[package]]\nname = \"chelis-types\"\nversion = \"0.0.1\"\n",
    )
    .unwrap();

    chelis(&store, &canonical)
        .args(["reef", "src", "check", "--path"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "ok: chelis source crates pinned at 0.0.1",
        ));
}

#[test]
fn reef_src_check_fails_when_cargo_lock_is_off_pin() {
    let tmp = tempdir().unwrap();
    let canonical = tmp.path().join("canonical");
    init_canonical(&canonical, "v0.0.1");
    let store = tmp.path().join("src-home");
    let ws = tmp.path().join("ws");
    let shell = ws.join("shelly");
    write_shell(&shell, "0.0.1", &["chelis-types"]);

    chelis(&store, &canonical)
        .args(["reef", "src", "sync", "--path"])
        .arg(&shell)
        .assert()
        .success();

    // Lockfile records a non-pin version — the exact drift the guard exists for.
    fs::write(
        shell.join("Cargo.lock"),
        "[[package]]\nname = \"chelis-types\"\nversion = \"9.9.9\"\n",
    )
    .unwrap();

    chelis(&store, &canonical)
        .args(["reef", "src", "check", "--path"])
        .arg(&shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("non-pin"));
}

#[test]
fn reef_src_sync_errors_without_chelis_src_section() {
    let tmp = tempdir().unwrap();
    let store = tmp.path().join("src-home");
    let shell = tmp.path().join("plain");
    fs::create_dir_all(shell.join("src")).unwrap();
    fs::write(
        shell.join("reef.toml"),
        "[package]\nname = \"plain\"\nversion = \"0.1.0\"\n\
         compiler = \"=0.0.1\"\nmodule_prefix = \"Plain\"\n",
    )
    .unwrap();

    chelis(&store, &store) // remote unused on the error path
        .args(["reef", "src", "sync", "--path"])
        .arg(&shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("no [chelis-src]"));
}

#[test]
fn reef_doctor_reports_a_crate_linking_shell() {
    let tmp = tempdir().unwrap();
    let canonical = tmp.path().join("canonical");
    init_canonical(&canonical, "v0.0.1");
    let store = tmp.path().join("src-home");
    let ws = tmp.path().join("ws");
    let shell = ws.join("shelly");
    write_shell(&shell, "0.0.1", &["chelis-types"]);

    // Not synced yet: doctor should report the shell and a src gap.
    chelis(&store, &canonical)
        .args(["reef", "doctor", "--root"])
        .arg(&ws)
        .assert()
        .success()
        .stdout(predicate::str::contains("shelly (pin 0.0.1)"))
        .stdout(predicate::str::contains("src:"));
}
