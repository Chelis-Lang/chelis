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

fn write_shell(dir: &Path, pin: &str, extra: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("reef.toml"),
        format!(
            "[package]\nname = \"shelly\"\nversion = \"0.1.0\"\n\
             compiler = \"={pin}\"\nmodule_prefix = \"Shelly\"\n{extra}"
        ),
    )
    .unwrap();
}

fn stub_toolchain(home: &Path, ver: &str) {
    let bin = home.join("toolchains").join(ver).join("bin");
    fs::create_dir_all(&bin).unwrap();
    fs::write(bin.join("chelis"), b"#!/bin/true\n").unwrap();
}

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
    write_shell(&shell, "0.0.1", "");
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
    write_shell(&shell, "0.0.1", "");
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
    write_shell(&shell, "0.0.1", ARTIFACT_SECTION);
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
    write_shell(&shell, "0.0.1", ARTIFACT_SECTION);
    stub_toolchain(&home, "0.0.1");
    // No <home>/bin/foo.

    chelis(&home)
        .args(["reef", "doctor", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("artifacts: MISSING: foo"));
}
