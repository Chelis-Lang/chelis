use std::fs;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn migrate_surf_prints_canonical_v019_without_writing() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("legacy.ch");
    let legacy = "def identity(x: f32): f32 = { f x; }\n";
    fs::write(&path, legacy).expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18"])
        .arg(&path)
        .assert()
        .success()
        .stdout("def identity(x: f32) -> f32 = f(x)\n");

    assert_eq!(fs::read_to_string(path).unwrap(), legacy);
}

#[test]
fn migrate_surf_check_and_inplace_form_a_fixed_point() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("legacy.ch");
    fs::write(&path, "def f = value\n").expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("migration required"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&path)
        .assert()
        .success();
    assert_eq!(fs::read_to_string(&path).unwrap(), "def f() = value\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&path)
        .assert()
        .success();
}

#[test]
fn migrate_surf_batch_is_all_or_nothing_on_ambiguous_comments() {
    let dir = tempdir().expect("tempdir");
    let valid_path = dir.path().join("valid.ch");
    let ambiguous_path = dir.path().join("ambiguous.ch");
    let valid = "def f = value\n";
    let ambiguous = "def g(x) = f({- attachment is ambiguous -} x)\n";
    fs::write(&valid_path, valid).expect("write valid fixture");
    fs::write(&ambiguous_path, ambiguous).expect("write ambiguous fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&valid_path)
        .arg(&ambiguous_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("inside a declaration"));

    assert_eq!(fs::read_to_string(valid_path).unwrap(), valid);
    assert_eq!(fs::read_to_string(ambiguous_path).unwrap(), ambiguous);
}

#[test]
fn migrate_surf_rejects_unknown_source_versions() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("source.ch");
    fs::write(&path, "value = x\n").expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.17"])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected `--from 0.18`"));
}
