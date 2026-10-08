//! Deep but finite Surf source must not abort the standalone lint command.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn nested_adds(depth: usize) -> String {
    let mut expression = "1i64".to_string();
    for _ in 0..depth {
        expression = format!("add(1i64, {expression})");
    }
    format!("module Probe.Nest\nresult = {expression}\n")
}

#[test]
fn lint_accepts_deep_valid_surf_without_aborting() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested.ch");
    fs::write(&path, nested_adds(350)).expect("write nested Surf");

    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--check"])
        .arg(&path)
        .output()
        .expect("run lint");
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
}

#[test]
fn lint_reports_deep_invalid_surf_as_a_violation() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("invalid.ch");
    fs::write(&path, format!("{}extra\n", nested_adds(350))).expect("write invalid nested Surf");

    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--check", "--rule", "surf-parses"])
        .arg(&path)
        .output()
        .expect("run lint");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("surf-parses"),
        "{output:?}"
    );
}
