//! Binary-level smoke for `chelis reef conform init|audit`.
//!
//! The audit engine's row-by-row negative parity is proven fast and in-process
//! by `chelis-conformance`'s `audit_negative_parity` test. This drives the same
//! path through the compiled binary end-to-end: `init` scaffolds a conformant
//! shell, `audit` reports it green (exit 0), and a broken variant fails (exit 1)
//! naming the offending row.

use assert_cmd::Command;
use predicates::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            if metadata.file_type().is_symlink() {
                let target = std::fs::read_link(&path).unwrap();
                out.insert(
                    relative,
                    format!("symlink:{}", target.display()).into_bytes(),
                );
            } else if metadata.is_dir() {
                visit(root, &path, out);
            } else {
                out.insert(relative, std::fs::read(path).unwrap());
            }
        }
    }

    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

#[test]
fn conform_init_and_two_syncs_are_byte_idempotent() {
    let dir = tempdir().expect("tempdir");
    let shell = dir.path().join("idempotent-shell");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "init",
            "idempotent-shell",
            "--module-prefix",
            "IdempotentShell",
            "--output",
        ])
        .arg(&shell)
        .assert()
        .success();
    let initialized = snapshot(&shell);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "sync", "--path"])
        .arg(&shell)
        .assert()
        .success();
    let synced_once = snapshot(&shell);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "sync", "--path"])
        .arg(&shell)
        .assert()
        .success();
    let synced_twice = snapshot(&shell);

    assert_eq!(initialized, synced_once, "init and first sync must agree");
    assert_eq!(synced_once, synced_twice, "second sync must be a no-op");
}

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
