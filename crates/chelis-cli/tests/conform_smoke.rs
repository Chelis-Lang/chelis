//! Binary-level smoke for `chelis reef conform init|audit`.
//!
//! The audit engine's row-by-row negative parity is proven fast and in-process
//! by `chelis-conformance`'s `audit_negative_parity` test. This drives the same
//! path through the compiled binary end-to-end: `init` scaffolds a conformant
//! shell, `audit` reports it green (exit 0), and a broken variant fails (exit 1)
//! naming the offending row.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn conform_init_then_audit_roundtrips() {
    let dir = tempdir().expect("tempdir");
    let shell = dir.path().join("myshell");

    // init
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
        .arg(&shell)
        .assert()
        .success();

    // audit (green)
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "audit", "--root"])
        .arg(&shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("no MUST failures"));

    // break it: drop the capability surface doc, then audit must fail on row 7.
    std::fs::remove_file(shell.join("docs/CHELIS_SURFACE.md")).expect("rm surface");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "audit", "--json", "--root"])
        .arg(&shell)
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("\"key\":\"chelis-surface\""))
        .stdout(predicate::str::contains("\"verdict\":\"fail\""));
}
