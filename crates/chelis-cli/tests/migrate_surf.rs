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
fn migrate_surf_requires_manual_renaming_for_newly_reserved_identifiers() {
    let dir = tempdir().expect("tempdir");
    for (name, source) in [
        ("future", "def resume(x: int32) -> int32 = x\n"),
        ("keyword", "def quote(x: int32) -> int32 = x\n"),
    ] {
        let path = dir.path().join(format!("{name}.ch"));
        fs::write(&path, source).expect("write reserved-identifier fixture");

        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["migrate", "surf", "--from", "0.18"])
            .arg(&path)
            .assert()
            .failure()
            .stderr(predicate::str::contains("rename"))
            .stderr(predicate::str::contains("manually"));
    }
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
fn migrate_surf_accepts_operator_named_pipe_stages() {
    // chelis#1197: `chelis fmt --check` already called these files canonical,
    // but the migration preflight's Deep resugaring rejected every stage whose
    // callee is an operator-named primitive.
    let dir = tempdir().expect("tempdir");
    let canonical_path = dir.path().join("canonical.ch");
    let canonical = "def b() -> f32 = 0.0 |> fn (p) -> cast(p, f32) |> mul(cast(2.0, f32))\n";
    fs::write(&canonical_path, canonical).expect("write canonical fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&canonical_path)
        .assert()
        .success();

    let legacy_path = dir.path().join("legacy.ch");
    fs::write(
        &legacy_path,
        "def scale(x: f32, y: f32): f32 = x |> mul(y)\n",
    )
    .expect("write legacy fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&legacy_path)
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(&legacy_path).unwrap(),
        "def scale(x: f32, y: f32) -> f32 = x |> mul(y)\n",
    );
}

#[test]
fn migrate_surf_reports_every_preflight_failure_in_one_run() {
    // chelis#1197: the preflight stays all-or-nothing, but a batch that cannot
    // be migrated must name every file that blocked it, not only the first.
    let dir = tempdir().expect("tempdir");
    let valid_path = dir.path().join("valid.ch");
    let ambiguous_path = dir.path().join("ambiguous.ch");
    let reserved_path = dir.path().join("reserved.ch");
    let valid = "def f = value\n";
    fs::write(&valid_path, valid).expect("write valid fixture");
    fs::write(
        &ambiguous_path,
        "def g(x) = f({- attachment is ambiguous -} x)\n",
    )
    .expect("write ambiguous fixture");
    fs::write(&reserved_path, "def resume(x: int32) -> int32 = x\n")
        .expect("write reserved-identifier fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&valid_path)
        .arg(&ambiguous_path)
        .arg(&reserved_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("ambiguous.ch"))
        .stderr(predicate::str::contains("reserved.ch"))
        .stderr(predicate::str::contains("2 of 3"));

    assert_eq!(fs::read_to_string(valid_path).unwrap(), valid);
}

#[test]
fn migrate_surf_keeps_a_single_failure_diagnostic_unaggregated() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ambiguous.ch");
    fs::write(&path, "def g(x) = f({- attachment is ambiguous -} x)\n").expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("inside a declaration"))
        .stderr(predicate::str::contains("of 1").not());
}

#[test]
fn migrate_surf_keeps_the_bare_diagnostic_when_one_file_of_many_blocks() {
    // The aggregate header earns its place by counting more than one blocked
    // file. One blocked file out of a batch still reads as that file's problem.
    let dir = tempdir().expect("tempdir");
    let first_path = dir.path().join("first.ch");
    let ambiguous_path = dir.path().join("ambiguous.ch");
    let last_path = dir.path().join("last.ch");
    let unmigrated = "def f = value\n";
    fs::write(&first_path, unmigrated).expect("write first fixture");
    fs::write(
        &ambiguous_path,
        "def g(x) = f({- attachment is ambiguous -} x)\n",
    )
    .expect("write ambiguous fixture");
    fs::write(&last_path, unmigrated).expect("write last fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&first_path)
        .arg(&ambiguous_path)
        .arg(&last_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("inside a declaration"))
        .stderr(predicate::str::contains("of 3").not());

    assert_eq!(fs::read_to_string(first_path).unwrap(), unmigrated);
    assert_eq!(fs::read_to_string(last_path).unwrap(), unmigrated);
}

#[test]
fn migrate_surf_reports_an_untaken_write_only_when_writing_was_asked_for() {
    let dir = tempdir().expect("tempdir");
    let ambiguous_path = dir.path().join("ambiguous.ch");
    let reserved_path = dir.path().join("reserved.ch");
    fs::write(
        &ambiguous_path,
        "def g(x) = f({- attachment is ambiguous -} x)\n",
    )
    .expect("write ambiguous fixture");
    fs::write(&reserved_path, "def resume(x: int32) -> int32 = x\n")
        .expect("write reserved-identifier fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&ambiguous_path)
        .arg(&reserved_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("no file was modified"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&ambiguous_path)
        .arg(&reserved_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("2 of 2"))
        .stderr(predicate::str::contains("no file was modified").not());
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
fn migrate_surf_preflights_an_unchanged_symlink_before_writing_changed_files() {
    use std::os::unix::fs::symlink;

    let dir = tempdir().expect("tempdir");
    let changed_path = dir.path().join("changed.ch");
    let target_path = dir.path().join("target.ch");
    let symlink_path = dir.path().join("already-canonical.ch");
    let changed_source = "def changed = value\n";
    let canonical_target = "def target() = value\n";
    fs::write(&changed_path, changed_source).expect("write changed fixture");
    fs::write(&target_path, canonical_target).expect("write canonical target");
    symlink(&target_path, &symlink_path).expect("create canonical symlink fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&changed_path)
        .arg(&symlink_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("symbolic link"));

    assert_eq!(fs::read_to_string(changed_path).unwrap(), changed_source);
    assert_eq!(fs::read_to_string(target_path).unwrap(), canonical_target);
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

#[cfg(unix)]
#[test]
fn migrate_surf_preflights_an_unchanged_hard_link_before_writing_changed_files() {
    let dir = tempdir().expect("tempdir");
    let changed_path = dir.path().join("changed.ch");
    let linked_path = dir.path().join("already-canonical.ch");
    let linked_alias = dir.path().join("already-canonical-alias.ch");
    let changed_source = "def changed = value\n";
    let canonical_linked_source = "def linked() = value\n";
    fs::write(&changed_path, changed_source).expect("write changed fixture");
    fs::write(&linked_path, canonical_linked_source).expect("write canonical linked fixture");
    fs::hard_link(&linked_path, &linked_alias).expect("create canonical hard-link fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18", "--inplace"])
        .arg(&changed_path)
        .arg(&linked_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("multiple hard links"));

    assert_eq!(fs::read_to_string(changed_path).unwrap(), changed_source);
    assert_eq!(
        fs::read_to_string(linked_path).unwrap(),
        canonical_linked_source
    );
    assert_eq!(
        fs::read_to_string(linked_alias).unwrap(),
        canonical_linked_source
    );
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
        .stdout("def unit_value() -> unit = ()\n");
}
