//! WS-C (§7) — unified `chelis reef doctor` integration tests.
//!
//! Doctor now reports against the consolidated chelisup store
//! (`$CHELIS_HOME` / `~/.chelis`): a machine-wide header (chelis home, shim,
//! recorded default), the pinned toolchain under `toolchains/<ver>` (fixed
//! with `chelisup install`, replacing the retired `install_chelis_toolchain.py`
//! path), and binary artifacts declared in `[artifacts]` (Item 11 / WS-A).
//! Everything is isolated behind the single `CHELIS_HOME` seam.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::{stub_toolchain, write_pinned_reef_toml};

fn chelis(home: &Path) -> Command {
    let mut c = Command::cargo_bin("chelis").expect("chelis binary built");
    c.env("CHELIS_HOME", home).env_remove("GITHUB_TOKEN");
    c
}

const ARTIFACT_SECTION: &str = "\n[artifacts.foo]\nrepo = \"Chelis-Lang/octant\"\n\
     tag = \"v0.4.2\"\n\
     platforms.linux-x86_64 = { asset = \"foo-linux-x86_64.tar.gz\", \
     sha256 = \"0000000000000000000000000000000000000000000000000000000000000000\" }\n";

// ---- toolchain ------------------------------------------------------------

#[test]
fn doctor_reports_header_and_installed_toolchain() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.0.1", "");
    stub_toolchain(&home, "0.0.1");
    // Seed the shim and recorded default for the header.
    fs::create_dir_all(home.join("bin")).unwrap();
    fs::write(home.join("bin/chelis"), b"shim\n").unwrap();
    fs::write(home.join("default"), "0.0.1\n").unwrap();

    chelis(&home)
        .args(["reef", "doctor", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("chelis home:"))
        .stdout(predicate::str::contains("shim:").and(predicate::str::contains("present")))
        .stdout(predicate::str::contains("default:").and(predicate::str::contains("0.0.1")))
        .stdout(predicate::str::contains("toolchain: ok"));
}

#[test]
fn doctor_reports_missing_toolchain_with_chelisup_fix() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.0.1", "");
    // No toolchain dir in the store.

    chelis(&home)
        .args(["reef", "doctor", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("toolchain: MISSING"))
        .stdout(predicate::str::contains("chelisup install 0.0.1"))
        // Negative parity: the retired installer must not be named.
        .stdout(predicate::str::contains("install_chelis_toolchain.py").not());
}

// ---- binary artifacts -----------------------------------------------------

#[test]
fn doctor_reports_installed_binary_artifact() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.0.1", ARTIFACT_SECTION);
    stub_toolchain(&home, "0.0.1");
    // The artifact is installed at <home>/bin/foo.
    fs::create_dir_all(home.join("bin")).unwrap();
    fs::write(home.join("bin/foo"), b"binary\n").unwrap();

    chelis(&home)
        .args(["reef", "doctor", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("artifacts: ok (foo ->"));
}

#[test]
fn doctor_reports_missing_binary_artifact() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.0.1", ARTIFACT_SECTION);
    stub_toolchain(&home, "0.0.1");
    // No <home>/bin/foo.

    chelis(&home)
        .args(["reef", "doctor", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("artifacts: MISSING: foo"));
}

// ---- home resolution ------------------------------------------------------

#[test]
fn doctor_treats_empty_chelis_home_as_unset_for_all_classes() {
    // Regression: with `CHELIS_HOME=""`, the toolchain class (chelisup's
    // `Store::from_env`, which treats empty as unset) fell back to
    // `$HOME/.chelis` while the artifact class (`chelis_reef::chelis_home`)
    // took the empty value verbatim, so one doctor run reported against two
    // different homes. Both classes must agree on `$HOME/.chelis`.
    let tmp = tempdir().unwrap();
    let user_home = tmp.path().join("user-home");
    let store = user_home.join(".chelis");
    let shell = tmp.path().join("shell");
    write_pinned_reef_toml(&shell, "0.0.1", ARTIFACT_SECTION);
    stub_toolchain(&store, "0.0.1");
    fs::create_dir_all(store.join("bin")).unwrap();
    fs::write(store.join("bin/foo"), b"binary\n").unwrap();

    let mut c = Command::cargo_bin("chelis").expect("chelis binary built");
    c.env("CHELIS_HOME", "")
        .env("HOME", &user_home)
        .env_remove("GITHUB_TOKEN")
        .args(["reef", "doctor", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("toolchain: ok"))
        .stdout(predicate::str::contains("artifacts: ok (foo ->"));
}
