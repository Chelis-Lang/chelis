//! Authoritative acceptance oracle for versioned Reef documents.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));

fn legacy_manifest(name: &str) -> String {
    format!(
        r#"# retained package comment
[package]
name = "{name}"
version = "0.1.0"
compiler = "{COMPILER_PIN}"
module_prefix = "Schema"
"#,
    )
}

fn legacy_lock(name: &str) -> String {
    format!(
        r#"dependencies = []

[package]
name = "{name}"
version = "0.1.0"
"#,
    )
}

fn stage_legacy_package(parent: &Path, name: &str) -> PathBuf {
    let root = parent.join(name);
    fs::create_dir_all(root.join("src")).expect("create package");
    fs::write(root.join("reef.toml"), legacy_manifest(name)).expect("write manifest");
    fs::write(root.join("reef.lock"), legacy_lock(name)).expect("write lock");
    fs::write(
        root.join("src/main.ch"),
        "module Schema.Main\n\ndef main() -> int32 = 1\n",
    )
    .expect("write source");
    root
}

fn chelis(root: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.current_dir(root);
    command
}

#[test]
fn reef_init_writes_manifest_schema_3_resolver_2_and_project_lock_ignore_rule() {
    let directory = tempdir().expect("tempdir");
    let root = directory.path().join("new-package");

    chelis(directory.path())
        .args([
            "reef",
            "init",
            "new-package",
            "--module-prefix",
            "NewPackage",
            "--output",
            root.to_str().expect("UTF-8 path"),
        ])
        .assert()
        .success();

    let manifest = fs::read_to_string(root.join("reef.toml")).expect("manifest");
    assert!(manifest.starts_with("schema = \"3\"\n"), "{manifest}");
    assert!(manifest.contains("resolver = \"2\""), "{manifest}");
    let ignore = fs::read_to_string(root.join(".gitignore")).expect("generated ignore");
    assert!(ignore.lines().any(|line| line == ".reef-write.lock"));
    assert!(ignore.lines().any(|line| line == ".*.reef-tmp-*"));
    assert!(ignore.lines().any(|line| line == ".*.reef-backup-*"));
}

#[test]
fn ordinary_legacy_read_reports_one_upgrade_warning() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "legacy-warning");

    chelis(&root)
        .args(["reef", "schema"])
        .assert()
        .success()
        .stderr(
            predicate::str::contains("legacy manifest")
                .count(1)
                .and(predicate::str::contains("chelis reef upgrade --inplace")),
        );
}

#[test]
fn ordinary_legacy_lock_read_reports_one_upgrade_warning() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "legacy-lock-warning");

    chelis(&root)
        .env("CHELIS_REEF_HOME", directory.path().join("registry"))
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            root.to_str().expect("UTF-8 path"),
        ])
        .assert()
        .success()
        .stderr(
            predicate::str::contains("legacy lock")
                .count(1)
                .and(predicate::str::contains("chelis reef upgrade --inplace")),
        );
}

#[test]
fn check_reports_both_legacy_steps_without_writes() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "legacy-check");
    let manifest_before = fs::read(root.join("reef.toml")).expect("manifest bytes");
    let lock_before = fs::read(root.join("reef.lock")).expect("lock bytes");

    chelis(directory.path())
        .args([
            "reef",
            "upgrade",
            "--check",
            "--path",
            root.to_str().expect("UTF-8 path"),
        ])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("reef.toml: schema 0 -> 1")
                .and(predicate::str::contains("reef.toml: schema 2 -> 3"))
                .and(predicate::str::contains("reef.lock: schema 0 -> 1")),
        );

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert_eq!(fs::read(root.join("reef.lock")).unwrap(), lock_before);
}

#[test]
fn upgrade_accepts_a_manifest_pinned_to_an_older_compiler() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "older-compiler");
    let manifest = fs::read_to_string(root.join("reef.toml"))
        .unwrap()
        .replace(COMPILER_PIN, "=0.0.1");
    fs::write(root.join("reef.toml"), manifest).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .success()
        .stdout(predicate::str::contains("reef.toml: schema 0 -> 1"));
    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();

    let upgraded = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(upgraded.starts_with("schema = \"3\"\n"));
    assert!(upgraded.contains("compiler = \"=0.0.1\""));
    assert!(upgraded.contains("resolver = \"2\""));
}

#[test]
fn inplace_upgrade_preserves_manifest_text_and_exact_lock_values() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "legacy-inplace");
    fs::write(root.join(".gitignore"), "dist/\n").unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();

    let manifest = fs::read_to_string(root.join("reef.toml")).expect("manifest");
    assert_eq!(
        manifest,
        format!(
            "schema = \"3\"\n# retained package comment\n[package]\nname = \"legacy-inplace\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Schema\"\nresolver = \"2\"\n"
        )
    );
    let lock = fs::read_to_string(root.join("reef.lock")).expect("lock");
    assert!(lock.starts_with("schema = \"1\"\n"), "{lock}");
    assert!(lock.contains("name = \"legacy-inplace\""), "{lock}");
    assert!(lock.contains("version = \"0.1.0\""), "{lock}");
    let ignore = fs::read_to_string(root.join(".gitignore")).unwrap();
    assert!(ignore.starts_with("dist/\n"), "{ignore}");
    assert!(ignore.lines().any(|line| line == ".reef-write.lock"));
    assert!(ignore.lines().any(|line| line == ".*.reef-tmp-*"));
    assert!(ignore.lines().any(|line| line == ".*.reef-backup-*"));
}

#[test]
fn upgrade_keeps_a_taplo_schema_directive_on_the_first_line() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "taplo-directive");
    let manifest = format!(
        "#:schema ../../docs/schemas/reef/manifest-v1.schema.json\n[package]\nname = \"taplo-directive\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Schema\"\n"
    );
    fs::write(root.join("reef.toml"), manifest).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();

    let upgraded = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(
        upgraded.starts_with(
            "#:schema ../../docs/schemas/reef/manifest-v3.schema.json\nschema = \"3\"\n"
        ),
        "{upgraded}"
    );
}

#[test]
fn current_documents_are_an_unchanged_success() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "current-noop");
    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();
    let lock_before = fs::read(root.join("reef.lock")).unwrap();

    chelis(&root)
        .args([
            "reef",
            "upgrade",
            "--inplace",
            "--manifest-to",
            "3",
            "--lock-to",
            "1",
        ])
        .assert()
        .success();

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert_eq!(fs::read(root.join("reef.lock")).unwrap(), lock_before);
}

#[test]
fn invalid_mode_and_unsupported_target_fail_without_writes() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "invalid-options");
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();

    chelis(&root)
        .args(["reef", "upgrade"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--check"));
    chelis(&root)
        .args(["reef", "upgrade", "--check", "--inplace"])
        .assert()
        .failure();
    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unsupported manifest schema 4"));

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
}

#[test]
fn malformed_and_future_schemas_fail_before_document_fields() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "bad-schema");

    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"01\"\n[package]\nname = \"bad-schema\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Schema\"\n"
        ),
    )
    .unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("schema").and(predicate::str::contains("01")));

    fs::write(root.join("reef.toml"), "schema = 1\n").unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("schema must use the string form"));

    fs::write(
        root.join("reef.toml"),
        "schema = \"99\"\nfuture_key = true\n",
    )
    .unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unsupported manifest schema 99")
                .and(predicate::str::contains("chelis reef upgrade"))
                .and(predicate::str::contains("unknown field").not()),
        );
}

#[test]
fn conform_local_skills_survives_upgrade_and_current_reads() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "conform-shell");
    let manifest = format!(
        "[package]\nname = \"conform-shell\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Schema\"\n\n[conform]\nlocal_skills = [\"chelis-std\"]\n"
    );
    fs::write(root.join("reef.toml"), manifest).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();
    chelis(&root).args(["reef", "schema"]).assert().success();

    let upgraded = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(upgraded.contains("[conform]"));
    assert!(upgraded.contains("local_skills = [\"chelis-std\"]"));
}

#[test]
fn schema_1_unknown_keys_report_the_containing_table() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "strict-keys");
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"1\"\n[package]\nname = \"strict-keys\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Schema\"\nadditional_source = \"properties\"\n"
        ),
    )
    .unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("additional_source").and(predicate::str::contains("package")),
        );
}

#[test]
fn strict_legacy_preflight_rejects_unknown_keys_without_writes() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "legacy-unknown");
    let manifest = format!(
        "[package]\nname = \"legacy-unknown\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Schema\"\nunknown_setting = true\n"
    );
    fs::write(root.join("reef.toml"), &manifest).unwrap();
    let lock_before = fs::read(root.join("reef.lock")).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unknown_setting").and(predicate::str::contains("[package]")),
        );

    assert_eq!(
        fs::read_to_string(root.join("reef.toml")).unwrap(),
        manifest
    );
    assert_eq!(fs::read(root.join("reef.lock")).unwrap(), lock_before);
    assert!(
        !root.join(".gitignore").exists(),
        "failed preflight must not write project ignore rules"
    );
}

#[test]
fn lock_schema_errors_precede_lock_field_errors_and_preserve_files() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "future-lock");
    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "1"])
        .assert()
        .success();
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();
    let future_lock = "schema = \"99\"\nfuture_lock_key = true\n";
    fs::write(root.join("reef.lock"), future_lock).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unsupported lock schema 99")
                .and(predicate::str::contains("future_lock_key").not()),
        );

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert_eq!(
        fs::read_to_string(root.join("reef.lock")).unwrap(),
        future_lock
    );
}

#[test]
fn schema_1_lock_rejects_unknown_source_keys() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "strict-lock");
    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();
    let lock = format!(
        "schema = \"1\"\n\n[package]\nname = \"strict-lock\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"dep\"\nversion = \"1.0.0\"\ncompiler = \"{COMPILER_PIN}\"\narchive_sha256 = \"a\"\nshell_sha256 = \"b\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"../dep\"\nunexpected = true\n"
    );
    fs::write(root.join("reef.lock"), &lock).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unexpected").and(predicate::str::contains("source")));

    let unknown_kind = lock
        .replace("kind = \"path\"", "kind = \"totally-bogus\"")
        .replace("unexpected = true\n", "");
    fs::write(root.join("reef.lock"), unknown_kind).unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unsupported source kind `totally-bogus`",
        ));
}

#[cfg(unix)]
#[test]
fn inplace_upgrade_rejects_symbolic_link_documents() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "linked-manifest");
    let real_manifest = root.join("real-reef.toml");
    fs::rename(root.join("reef.toml"), &real_manifest).unwrap();
    symlink(&real_manifest, root.join("reef.toml")).unwrap();
    let real_before = fs::read(&real_manifest).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("symbolic-link target"));

    assert!(
        fs::symlink_metadata(root.join("reef.toml"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read(real_manifest).unwrap(), real_before);
}

#[cfg(unix)]
#[test]
fn linked_lock_is_rejected_before_manifest_replacement() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "linked-lock");
    let real_lock = root.join("real-reef.lock");
    fs::rename(root.join("reef.lock"), &real_lock).unwrap();
    symlink(&real_lock, root.join("reef.lock")).unwrap();
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("symbolic-link target"));

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert!(
        fs::symlink_metadata(root.join("reef.lock"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!root.join(".gitignore").exists());
}

#[cfg(unix)]
#[test]
fn inplace_upgrade_rejects_multiply_hard_linked_documents() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "hard-linked-manifest");
    let second_name = root.join("reef-copy.toml");
    fs::hard_link(root.join("reef.toml"), &second_name).unwrap();
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("multiply-hard-linked target"));

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert_eq!(fs::read(second_name).unwrap(), manifest_before);
}

#[cfg(unix)]
#[test]
fn multiply_hard_linked_lock_is_rejected_before_manifest_replacement() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "hard-linked-lock");
    let second_name = root.join("reef-copy.lock");
    fs::hard_link(root.join("reef.lock"), &second_name).unwrap();
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("multiply-hard-linked target"));

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert_eq!(
        fs::read(root.join("reef.lock")).unwrap(),
        fs::read(second_name).unwrap()
    );
    assert!(!root.join(".gitignore").exists());
}

#[test]
fn interrupted_upgrade_resumes_only_the_legacy_lock_step() {
    let directory = tempdir().expect("tempdir");
    let root = stage_legacy_package(directory.path(), "resume-upgrade");
    let legacy_lock = fs::read(root.join("reef.lock")).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();
    fs::write(root.join("reef.lock"), legacy_lock).unwrap();
    let manifest_before = fs::read(root.join("reef.toml")).unwrap();
    fs::write(root.join(".reef-write.lock.unowned"), b"leave me").unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace"])
        .assert()
        .success();

    assert_eq!(fs::read(root.join("reef.toml")).unwrap(), manifest_before);
    assert!(
        fs::read_to_string(root.join("reef.lock"))
            .unwrap()
            .starts_with("schema = \"1\"\n")
    );
    assert_eq!(
        fs::read(root.join(".reef-write.lock.unowned")).unwrap(),
        b"leave me"
    );
}

#[test]
fn committed_editor_schemas_match_the_wire_models() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = fs::read_to_string(repository.join("docs/schemas/reef/manifest-v1.schema.json"))
        .expect("manifest schema");
    let manifest_v2 =
        fs::read_to_string(repository.join("docs/schemas/reef/manifest-v2.schema.json"))
            .expect("manifest schema 2");
    let manifest_v3 =
        fs::read_to_string(repository.join("docs/schemas/reef/manifest-v3.schema.json"))
            .expect("manifest schema 3");
    let lock = fs::read_to_string(repository.join("docs/schemas/reef/lock-v1.schema.json"))
        .expect("lock schema");

    assert_eq!(
        manifest,
        chelis_reef::manifest_schema_v1_json(),
        "manifest schema 1 drifted; run `cargo run -p chelis-reef --example generate_document_schemas`"
    );
    assert_eq!(
        manifest_v2,
        chelis_reef::manifest_schema_v2_json(),
        "manifest schema 2 drifted; run `cargo run -p chelis-reef --example generate_document_schemas`"
    );
    assert_eq!(
        manifest_v3,
        chelis_reef::manifest_schema_v3_json(),
        "manifest schema 3 drifted; run `cargo run -p chelis-reef --example generate_document_schemas`"
    );
    assert_eq!(
        lock,
        chelis_reef::lock_schema_v1_json(),
        "lock schema 1 drifted; run `cargo run -p chelis-reef --example generate_document_schemas`"
    );
}
