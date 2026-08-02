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

#[cfg(unix)]
#[test]
fn migrate_surf_batch_is_all_or_nothing_on_write_failure() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempdir().expect("tempdir");
    let writable_path = dir.path().join("first.ch");
    let read_only_path = dir.path().join("second.ch");
    let writable_source = "def first = value\n";
    let read_only_source = "def second = value\n";
    fs::write(&writable_path, writable_source).expect("write first fixture");
    fs::write(&read_only_path, read_only_source).expect("write second fixture");
    fs::set_permissions(&read_only_path, fs::Permissions::from_mode(0o444))
        .expect("make second fixture read-only");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&writable_path)
        .arg(&read_only_path)
        .assert()
        .failure();

    assert_eq!(
        fs::read_to_string(&writable_path).unwrap(),
        writable_source,
        "a later write failure must not leave the first file migrated",
    );
    assert_eq!(
        fs::read_to_string(&read_only_path).unwrap(),
        read_only_source,
    );

    fs::set_permissions(&read_only_path, fs::Permissions::from_mode(0o644))
        .expect("restore fixture permissions for cleanup");
}

#[cfg(unix)]
#[test]
fn migrate_surf_rejects_symlink_paths_before_writing_any_file() {
    use std::os::unix::fs::symlink;

    let dir = tempdir().expect("tempdir");
    let first_path = dir.path().join("first.ch");
    let target_path = dir.path().join("target.ch");
    let symlink_path = dir.path().join("linked.ch");
    let first_source = "def first = value\n";
    let target_source = "def target = value\n";
    fs::write(&first_path, first_source).expect("write first fixture");
    fs::write(&target_path, target_source).expect("write symlink target");
    symlink(&target_path, &symlink_path).expect("create symlink fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&first_path)
        .arg(&symlink_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("symbolic link"));

    assert_eq!(fs::read_to_string(&first_path).unwrap(), first_source);
    assert_eq!(fs::read_to_string(&target_path).unwrap(), target_source);
    assert!(
        fs::symlink_metadata(&symlink_path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

#[cfg(unix)]
#[test]
fn migrate_surf_rejects_multiply_linked_files_before_writing() {
    let dir = tempdir().expect("tempdir");
    let first_path = dir.path().join("first.ch");
    let linked_path = dir.path().join("linked.ch");
    let linked_alias = dir.path().join("linked-alias.ch");
    let first_source = "def first = value\n";
    let linked_source = "def linked = value\n";
    fs::write(&first_path, first_source).expect("write first fixture");
    fs::write(&linked_path, linked_source).expect("write linked fixture");
    fs::hard_link(&linked_path, &linked_alias).expect("create hard-link fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&first_path)
        .arg(&linked_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("multiple hard links"));

    assert_eq!(fs::read_to_string(&first_path).unwrap(), first_source);
    assert_eq!(fs::read_to_string(&linked_path).unwrap(), linked_source);
    assert_eq!(fs::read_to_string(&linked_alias).unwrap(), linked_source);
}

#[test]
fn migrate_surf_rolls_back_a_committed_file_on_late_persist_failure() {
    let dir = tempdir().expect("tempdir");
    let first_path = dir.path().join("first.ch");
    let second_path = dir.path().join("second.ch");
    let first_source = "def first = value\n";
    let second_source = "def second = value\n";
    fs::write(&first_path, first_source).expect("write first fixture");
    fs::write(&second_path, second_source).expect("write second fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_TEST_MIGRATION_FAIL_PERSIST_INDEX", "1")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&first_path)
        .arg(&second_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "injected migration persist failure",
        ));

    assert_eq!(fs::read_to_string(&first_path).unwrap(), first_source);
    assert_eq!(fs::read_to_string(&second_path).unwrap(), second_source);
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

#[test]
fn migrate_surf_roundtrips_legacy_unit_through_deep_preflight() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unit.ch");
    fs::write(&path, "def unit_value(): unit = ()\n").expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18"])
        .arg(&path)
        .assert()
        .success()
        .stdout("def unit_value() -> () = ()\n");
}
