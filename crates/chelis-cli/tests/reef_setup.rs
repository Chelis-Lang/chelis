//! WS-C (§7) — `chelis reef setup` orchestrator integration tests.
//!
//! Exercises the one-command "clone -> setup -> build" verb end to end
//! through the real `chelis` binary, fully offline. The chelisup store, the
//! toolchain check, and any binary-artifact dir are all isolated behind the
//! single `CHELIS_HOME` seam (chelis-reef and chelisup resolve it
//! identically). The toolchain auto-install path runs the real `chelisup`
//! binary against a synthesized release tarball via `CHELISUP_RELEASE_BASE`.
//!
//! Coverage (positive + negative parity):
//!   * toolchain already installed, no reef.lock, no [chelis-src] -> clean.
//!   * reef.lock present (bundled skip-class entry) -> install section runs.
//!   * [chelis-src] present -> source-crate sync section runs.
//!   * toolchain missing + chelisup available -> auto-installs; the shim at
//!     `<home>/bin/chelis` is chelisup's, NOT the compiler (trap guard).
//!   * toolchain missing + chelisup absent -> loud, actionable error naming
//!     `chelisup install`, never the retired `install_chelis_toolchain.py`.
//!   * missing reef.toml -> clear parse/read error.
//!   * malformed reef.lock -> the install step's error stops setup before
//!     the doctor summary.
//!   * unreachable [chelis-src] remote -> the sync step's error stops setup
//!     before the doctor summary.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command as Proc;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::{stub_toolchain, write_pinned_reef_toml};

/// A `chelis` invocation with the chelisup store isolated and the chelisup
/// delegation seams cleared unless a test sets them.
fn chelis(home: &Path) -> Command {
    let mut c = Command::cargo_bin("chelis").expect("chelis binary built");
    c.env("CHELIS_HOME", home)
        .env_remove("CHELISUP_BIN")
        .env_remove("CHELISUP_RELEASE_BASE")
        .env_remove("GITHUB_TOKEN");
    c
}

// ---- toolchain present ----------------------------------------------------

#[test]
fn setup_succeeds_when_toolchain_present_no_lock_no_src() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.9.9", "");
    stub_toolchain(&home, "0.9.9");

    chelis(&home)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("toolchain: ok"))
        .stdout(predicate::str::contains("no reef.lock"))
        .stdout(predicate::str::contains("no [chelis-src]"))
        .stdout(predicate::str::contains("--- doctor ---"));
}

#[test]
fn setup_runs_install_section_when_reef_lock_present() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.9.9", "");
    stub_toolchain(&home, "0.9.9");
    // A bundled skip-class dependency: install_from_lockfile reports it
    // without any network access.
    fs::write(
        shell.join("reef.lock"),
        "[package]\nname = \"shelly\"\nversion = \"0.1.0\"\n\n\
         [[dependencies]]\nname = \"chelis-std\"\nversion = \"0.4.0\"\n\
         compiler = \"=0.9.9\"\n\
         archive_sha256 = \"ca722bcff18b7bc10c703f7acb01f929c6e0bf816e17eb83f2491cf5695c8407\"\n\
         shell_sha256 = \"93bede1b92af7e61a7a76922d733909b9974e026b96cc820c7739fd6c3162310\"\n\n\
         [dependencies.source]\nkind = \"bundled\"\ncompiler_version = \"0.9.9\"\n",
    )
    .unwrap();

    chelis(&home)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("toolchain: ok"))
        .stdout(predicate::str::contains(
            "Skipped bundled runtime chelis-std",
        ));
}

// ---- source crates --------------------------------------------------------

fn git(dir: &Path, args: &[&str]) {
    let status = Proc::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .expect("spawn git");
    assert!(status.success(), "git {args:?} failed in {}", dir.display());
}

/// A canonical-style chelis repo with one commit tagged `tag`.
fn init_canonical(dir: &Path, tag: &str) {
    fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q"]);
    git(dir, &["config", "user.email", "fixture@example.invalid"]);
    git(dir, &["config", "user.name", "Fixture"]);
    git(dir, &["config", "gc.auto", "0"]);
    fs::write(dir.join("VERSION"), "fixture\n").unwrap();
    git(dir, &["add", "."]);
    git(dir, &["commit", "-q", "-m", "init"]);
    git(dir, &["tag", tag]);
}

#[test]
fn setup_syncs_source_crates_when_chelis_src_present() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let canonical = tmp.path().join("canonical");
    init_canonical(&canonical, "v0.0.1");
    let ws = tmp.path().join("ws");
    let shell = ws.join("shelly");
    write_pinned_reef_toml(
        &shell,
        "0.0.1",
        "\n[chelis-src]\ncrates = [\"chelis-types\"]\n",
    );
    stub_toolchain(&home, "0.0.1");

    chelis(&home)
        // Point the source-crate fetch at the local fixture repo.
        .env("CHELIS_SRC_REMOTE", &canonical)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("toolchain: ok"))
        .stdout(predicate::str::contains("synced chelis source crates"));

    // The `../chelis` slot is now a symlink into the store worktree.
    let slot = ws.join("chelis");
    assert!(fs::read_link(&slot).is_ok(), "../chelis is a symlink");
}

// ---- toolchain missing: auto-install via chelisup -------------------------

/// The host release-asset slug, matching `chelisup::install::detect_slug`.
fn host_slug() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "darwin-arm64",
        ("macos", "x86_64") => "darwin-x86_64",
        ("linux", "x86_64") => "linux-x86_64",
        other => panic!("unsupported host for this test: {other:?}"),
    }
}

/// Write a gzip toolchain release tarball `chelis-v<ver>-<slug>.tar.gz` into
/// `base` containing `chelis-v<ver>-<slug>/bin/chelis`, the shape
/// `chelisup install` extracts.
fn write_release_tarball(base: &Path, ver: &str, slug: &str) {
    let inner = format!("chelis-v{ver}-{slug}");
    let mut tar_buf: Vec<u8> = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut tar_buf);
        let payload = b"#!/bin/true\n";
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, format!("{inner}/bin/chelis"), &payload[..])
            .unwrap();
        builder.finish().unwrap();
    }
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(&tar_buf).unwrap();
    let gz = enc.finish().unwrap();
    fs::write(base.join(format!("chelis-v{ver}-{slug}.tar.gz")), gz).unwrap();
}

/// Build and return the path to the real `chelisup` binary that
/// `reef setup` delegates installs to. Nextest exposes `CARGO_BIN_EXE_*`
/// only for the package under test, so this cross-package helper resolves
/// Cargo's target directory after building instead of asking assert_cmd for
/// an unset `CARGO_BIN_EXE_chelisup`. The build runs from the workspace root so
/// a relative `CARGO_TARGET_DIR` cannot create an untracked crate-local target.
/// A fresh isolated target starts without chelisup; when it is already built
/// this is a fast no-op.
fn chelisup_bin() -> PathBuf {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root");
    let target_dir = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(value) => {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        }
        None => workspace_root.join("target"),
    };
    // A managed build puts its observed Cargo launcher first on PATH; $CARGO
    // names the real binary, which would inherit the enclosing session.
    let status = Proc::new("cargo")
        .current_dir(&workspace_root)
        .args(["build", "-p", "chelisup", "--bin", "chelisup"])
        .status()
        .expect("spawn cargo build for chelisup");
    assert!(status.success(), "building chelisup failed");
    let path = target_dir
        .join("debug")
        .join(format!("chelisup{}", std::env::consts::EXE_SUFFIX));
    assert!(path.exists(), "chelisup binary at {}", path.display());
    path
}

#[test]
fn setup_auto_installs_missing_toolchain_and_keeps_shim_intact() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.9.9", "");
    // No toolchain in `home` yet -> setup must delegate to chelisup.
    let releases = tmp.path().join("releases");
    fs::create_dir_all(&releases).unwrap();
    write_release_tarball(&releases, "0.9.9", host_slug());
    let chelisup = chelisup_bin();

    chelis(&home)
        .env("CHELISUP_BIN", &chelisup)
        // Offline install seam: chelisup reads the tarball from this dir.
        .env("CHELISUP_RELEASE_BASE", &releases)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("running `chelisup install 0.9.9`"))
        .stdout(predicate::str::contains("toolchain: ok"));

    // The toolchain landed in the chelisup store.
    assert!(
        home.join("toolchains/0.9.9/bin/chelis").is_file(),
        "toolchain installed under the chelisup store"
    );
    // Trap guard: `<home>/bin/chelis` is the chelisup shim (a copy of the
    // chelisup binary), NOT the chelis compiler. If setup had called
    // `chelisup::install::install` in-process it would be the compiler.
    let shim = home.join("bin/chelis");
    assert!(shim.is_file(), "shim installed at {}", shim.display());
    assert_eq!(
        fs::read(&shim).unwrap(),
        fs::read(&chelisup).unwrap(),
        "the shim is chelisup, not the chelis compiler"
    );
}

// ---- negatives ------------------------------------------------------------

#[test]
fn setup_errors_when_toolchain_missing_and_chelisup_absent() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.9.9", "");
    // Empty home (no toolchain) and a CHELISUP_BIN that does not exist, so
    // the delegation spawn fails deterministically regardless of PATH.
    let missing = tmp.path().join("no-such-chelisup");

    chelis(&home)
        .env("CHELISUP_BIN", &missing)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("not installed"))
        .stderr(predicate::str::contains("chelisup install 0.9.9"))
        .stderr(predicate::str::contains("install_chelis_toolchain.py").not());
}

#[test]
fn setup_errors_without_reef_toml() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("empty");
    fs::create_dir_all(&shell).unwrap();

    chelis(&home)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("reef.toml"));
}

#[test]
fn setup_stops_before_doctor_when_install_step_fails() {
    // Failure parity for the install section: a malformed reef.lock must
    // fail the run *and* stop the orchestrator before the doctor summary,
    // not degrade into a green setup with a broken install.
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.9.9", "");
    stub_toolchain(&home, "0.9.9");
    fs::write(shell.join("reef.lock"), "this is not toml [").unwrap();

    chelis(&home)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("reef.lock"))
        // Step 1 passed; the failure is the install step, not a precondition.
        .stdout(predicate::str::contains("toolchain: ok"))
        .stdout(predicate::str::contains("--- doctor ---").not());
}

#[test]
fn setup_stops_before_doctor_when_src_sync_fails() {
    // Failure parity for the source-crate section: an unreachable
    // [chelis-src] remote must fail the run and stop the orchestrator
    // before the doctor summary.
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("ws").join("shelly");
    write_pinned_reef_toml(
        &shell,
        "0.9.9",
        "\n[chelis-src]\ncrates = [\"chelis-types\"]\n",
    );
    stub_toolchain(&home, "0.9.9");
    let missing_remote = tmp.path().join("no-such-remote");

    chelis(&home)
        .env("CHELIS_SRC_REMOTE", &missing_remote)
        .args(["reef", "setup", "--path"])
        .arg(&shell)
        .assert()
        .failure()
        // Steps 1-2 passed; the failure is the sync step.
        .stdout(predicate::str::contains("toolchain: ok"))
        .stdout(predicate::str::contains("no reef.lock"))
        .stdout(predicate::str::contains("--- doctor ---").not());
}
