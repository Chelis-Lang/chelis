//! Validation for the bootstrap installer `bootstrap/chelisup.sh`, the
//! single shell carve-out in this repo (AGENTS.md Scripting Language
//! Policy). The carve-out requires the script to be covered by a test:
//! a `sh -n` parse check always, and `shellcheck` when it is on PATH.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bootstrap")
        .join("chelisup.sh")
}

fn tool_available(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn bootstrap_script_exists_and_is_executable() {
    use std::os::unix::fs::PermissionsExt;
    let p = script_path();
    assert!(p.is_file(), "missing bootstrap script at {}", p.display());
    let mode = std::fs::metadata(&p).unwrap().permissions().mode();
    assert!(
        mode & 0o111 != 0,
        "bootstrap script should be executable (mode {mode:o})"
    );
}

#[test]
fn bootstrap_script_parses_with_sh_n() {
    let p = script_path();
    let out = Command::new("sh")
        .arg("-n")
        .arg(&p)
        .output()
        .expect("failed to run `sh -n`");
    assert!(
        out.status.success(),
        "sh -n rejected the script:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn bootstrap_script_is_shellcheck_clean_when_available() {
    if !tool_available("shellcheck") {
        eprintln!("skipping shellcheck: not on PATH (sh -n still gates the script)");
        return;
    }
    let p = script_path();
    let out = Command::new("shellcheck")
        .arg("--shell=sh")
        .arg(&p)
        .output()
        .expect("failed to run shellcheck");
    assert!(
        out.status.success(),
        "shellcheck reported issues:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn bootstrap_uses_public_url_when_gh_is_signed_out() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let fake_bin = dir.path().join("fake-bin");
    std::fs::create_dir(&fake_bin).unwrap();
    let asset = dir.path().join("asset");
    std::fs::write(&asset, b"public chelisup").unwrap();
    let gh = fake_bin.join("gh");
    let curl = fake_bin.join("curl");
    std::fs::write(&gh, b"#!/bin/sh\nexit 1\n").unwrap();
    std::fs::write(&curl, b"#!/bin/sh\ncp \"$CHELISUP_TEST_ASSET\" \"$3\"\n").unwrap();
    for path in [&gh, &curl] {
        let mut permissions = std::fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions).unwrap();
    }
    let path = format!("{}:{}", fake_bin.display(), std::env::var("PATH").unwrap());
    let home = dir.path().join("home");
    let output = Command::new("sh")
        .arg(script_path())
        .env("PATH", path)
        .env("CHELIS_HOME", &home)
        .env("CHELISUP_TEST_ASSET", &asset)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(home.join("bin/chelisup")).unwrap(),
        b"public chelisup"
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("https://github.com/"));
}
