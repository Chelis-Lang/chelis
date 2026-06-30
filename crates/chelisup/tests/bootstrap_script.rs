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
