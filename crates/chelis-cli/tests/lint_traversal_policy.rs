use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_spec(root: &Path) {
    let spec = root.join("spec/01-nomenclature.md");
    fs::create_dir_all(spec.parent().unwrap()).unwrap();
    fs::write(
        spec,
        "# Nomenclature\n\n### 12.2 Lint traversal exclusions\n",
    )
    .unwrap();
}

fn chelis_lint(root: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.args(["lint", "--check", "--rule", "surf-value-snake-case"]);
    command.arg(root);
    command
}

#[test]
fn configured_exclusion_suppresses_violation_through_standalone_cli() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(root.join("generated/bad.ch"), "def addOne() = 1\n").unwrap();
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    chelis_lint(root).assert().success().stdout("");
}

#[test]
fn unexcluded_violation_still_fails_through_standalone_cli() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    fs::write(root.join("bad.ch"), "def addOne() = 1\n").unwrap();

    chelis_lint(root)
        .assert()
        .failure()
        .stdout(predicate::str::contains("surf-value-snake-case"))
        .stdout(predicate::str::contains("addOne"));
}

#[cfg(unix)]
#[test]
fn external_policy_symlink_fails_standalone_cli() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let external = tempdir().unwrap();
    write_spec(repository.path());
    let external_policy = external.path().join("policy.toml");
    fs::write(
        &external_policy,
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    symlink(&external_policy, repository.path().join("chelis-lint.toml")).unwrap();
    fs::write(repository.path().join("visible.ch"), "def visible() = 1\n").unwrap();

    chelis_lint(repository.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("chelis-lint.toml"))
        .stderr(predicate::str::contains("outside its policy root"));
}

#[test]
fn malformed_policy_fails_standalone_cli_with_path_and_reason() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 2\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    chelis_lint(root)
        .assert()
        .failure()
        .stderr(predicate::str::contains("chelis-lint.toml"))
        .stderr(predicate::str::contains(
            "unsupported traversal policy version 2",
        ));
}
