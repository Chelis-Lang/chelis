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

fn chelis_lint_rule(root: &Path, rule: &str) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.args(["lint", "--check", "--rule", rule]);
    command.arg(root);
    command
}

fn chelis_lint(root: &Path) -> Command {
    chelis_lint_rule(root, "surf-value-snake-case")
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

#[test]
fn excluded_manifest_cannot_grant_doc_package_exception_through_cli() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"crates/generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates/generated")).unwrap();
    fs::write(
        root.join("crates/generated/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    chelis_lint_rule(root, "doc-filename-convention")
        .assert()
        .failure()
        .stdout(predicate::str::contains("doc-filename-convention"))
        .stdout(predicate::str::contains("foo-bar.md"));
}

#[cfg(unix)]
#[test]
fn excluded_surf_symlink_target_cannot_influence_cli_opaque_catalog() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(
        root.join("generated/opaque.ch"),
        "module Hidden.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
    )
    .unwrap();
    symlink("generated/opaque.ch", root.join("linked.ch")).unwrap();
    fs::write(
        root.join("agent.ch"),
        "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
    )
    .unwrap();

    chelis_lint_rule(root, "opaque-domain-construction")
        .assert()
        .success()
        .stdout("");
}

#[cfg(unix)]
#[test]
fn internal_surf_symlink_target_retains_cli_opaque_catalog_behavior() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("shared")).unwrap();
    fs::write(
        root.join("shared/opaque.txt"),
        "module Shared.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
    )
    .unwrap();
    symlink("shared/opaque.txt", root.join("linked.ch")).unwrap();
    fs::write(
        root.join("agent.ch"),
        "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
    )
    .unwrap();

    chelis_lint_rule(root, "opaque-domain-construction")
        .assert()
        .failure()
        .stdout(predicate::str::contains("opaque-domain-construction"));
}

#[cfg(unix)]
#[test]
fn explicit_excluded_directory_preserves_internal_symlink_override_through_cli() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n\n[[exclude]]\npattern = \"nested/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    let generated = root.join("generated");
    fs::create_dir_all(generated.join("shared")).unwrap();
    fs::create_dir_all(generated.join("nested")).unwrap();
    fs::write(
        generated.join("shared/opaque.txt"),
        "module Shared.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
    )
    .unwrap();
    fs::write(
        generated.join("nested/opaque.txt"),
        "module Nested.Types\n@opaque\ntype NestedSecret = | NestedSecret { value: f32 }\n",
    )
    .unwrap();
    symlink("shared/opaque.txt", generated.join("admitted.ch")).unwrap();
    symlink("nested/opaque.txt", generated.join("excluded.ch")).unwrap();
    fs::write(
        generated.join("agent.ch"),
        "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\ndef forge_nested(x: f32) -> NestedSecret = NestedSecret { value: x }\n",
    )
    .unwrap();

    chelis_lint_rule(&generated, "opaque-domain-construction")
        .assert()
        .failure()
        .stdout(predicate::str::contains("opaque domain type `Secret`"))
        .stdout(predicate::str::contains("NestedSecret").not());
}

#[cfg(unix)]
#[test]
fn excluded_manifest_symlink_target_cannot_grant_doc_exception_through_cli() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(
        root.join("generated/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    symlink(
        "../../generated/Cargo.toml",
        root.join("crates/foo-bar/Cargo.toml"),
    )
    .unwrap();

    chelis_lint_rule(root, "doc-filename-convention")
        .assert()
        .failure()
        .stdout(predicate::str::contains("doc-filename-convention"));
}

#[cfg(unix)]
#[test]
fn internal_manifest_symlink_target_retains_doc_exception_through_cli() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("config")).unwrap();
    fs::write(
        root.join("config/package.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    symlink(
        "../../config/package.toml",
        root.join("crates/foo-bar/Cargo.toml"),
    )
    .unwrap();

    chelis_lint_rule(root, "doc-filename-convention")
        .assert()
        .success()
        .stdout("");
}

#[cfg(unix)]
#[test]
fn internal_symlinked_workspace_crate_retains_doc_exception_through_cli() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    let doc = docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates")).unwrap();
    fs::create_dir_all(root.join("real/foo-bar")).unwrap();
    fs::write(
        root.join("real/foo-bar/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    symlink("../real/foo-bar", root.join("crates/foo-bar")).unwrap();

    for target in [&docs, &doc] {
        chelis_lint_rule(target, "doc-filename-convention")
            .assert()
            .success()
            .stdout("");
    }
}

#[cfg(unix)]
#[test]
fn external_symlinked_workspace_crate_cannot_grant_doc_exception_through_cli() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let external = tempdir().unwrap();
    let root = repository.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    let doc = docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates")).unwrap();
    fs::create_dir_all(external.path().join("foo-bar")).unwrap();
    fs::write(
        external.path().join("foo-bar/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    symlink(external.path().join("foo-bar"), root.join("crates/foo-bar")).unwrap();

    for target in [&docs, &doc] {
        chelis_lint_rule(target, "doc-filename-convention")
            .assert()
            .failure()
            .stdout(predicate::str::contains("doc-filename-convention"))
            .stdout(predicate::str::contains("foo-bar.md"));
    }
}

#[test]
fn admitted_manifest_retains_doc_package_exception_through_cli() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    fs::write(
        root.join("crates/foo-bar/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    chelis_lint_rule(root, "doc-filename-convention")
        .assert()
        .success()
        .stdout("");
}

#[test]
fn admitted_sibling_manifest_retains_doc_exception_for_subdirectory_cli_target() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    let doc = docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    fs::write(
        root.join("crates/foo-bar/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    for target in [&docs, &doc] {
        chelis_lint_rule(target, "doc-filename-convention")
            .assert()
            .success()
            .stdout("");
    }
}

#[test]
fn external_parent_manifest_cannot_grant_doc_exception_through_cli() {
    let temp = tempdir().unwrap();
    fs::write(
        temp.path().join("Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let root = temp.path().join("repository");
    write_spec(&root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    let doc = docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();

    for target in [&docs, &doc] {
        chelis_lint_rule(target, "doc-filename-convention")
            .assert()
            .failure()
            .stdout(predicate::str::contains("doc-filename-convention"))
            .stdout(predicate::str::contains("foo-bar.md"));
    }
}

#[cfg(unix)]
#[test]
fn external_parent_manifest_symlink_pointing_inward_cannot_grant_doc_exception() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path().join("repository");
    write_spec(&root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    fs::write(docs.join("foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("config")).unwrap();
    fs::write(
        root.join("config/package.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    symlink(
        root.join("config/package.toml"),
        temp.path().join("Cargo.toml"),
    )
    .unwrap();

    chelis_lint_rule(&docs, "doc-filename-convention")
        .assert()
        .failure()
        .stdout(predicate::str::contains("doc-filename-convention"))
        .stdout(predicate::str::contains("foo-bar.md"));
}

#[cfg(unix)]
#[test]
fn inadmissible_explicit_special_root_fails_loudly_through_cli() {
    use std::os::unix::net::UnixListener;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let socket_path = root.join("explicit.ch");
    let _socket = UnixListener::bind(&socket_path).unwrap();

    chelis_lint(&socket_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("explicit.ch"))
        .stderr(predicate::str::contains("not a regular file or directory"));
}

#[cfg(unix)]
#[test]
fn escaping_explicit_symlink_root_fails_loudly_through_cli() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let external = tempdir().unwrap();
    let root = repository.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    fs::write(external.path().join("outside.ch"), "def outside() = 1\n").unwrap();
    let external_link = root.join("external-link");
    symlink(external.path(), &external_link).unwrap();

    chelis_lint(&external_link)
        .assert()
        .failure()
        .stderr(predicate::str::contains("external-link"))
        .stderr(predicate::str::contains(
            "outside the repository policy boundary",
        ));
}

#[test]
fn subdirectory_cwd_lint_applies_repository_policy_through_cli() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"sub/generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("sub/generated")).unwrap();
    fs::write(root.join("sub/keep.ch"), "def keep() = 1\n").unwrap();
    fs::write(root.join("sub/generated/bad.ch"), "def addOne() = 1\n").unwrap();

    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .args(["lint", "--check", "--rule", "surf-value-snake-case", "."])
        .current_dir(root.join("sub"));
    command.assert().success().stdout("");
}

#[test]
fn subdirectory_cwd_lint_still_reports_admitted_violations_through_cli() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n\n[[exclude]]\npattern = \"sub/generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("sub/generated")).unwrap();
    fs::write(root.join("sub/bad.ch"), "def addOne() = 1\n").unwrap();
    fs::write(root.join("sub/generated/bad.ch"), "def addTwo() = 2\n").unwrap();

    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .args(["lint", "--check", "--rule", "surf-value-snake-case", "."])
        .current_dir(root.join("sub"));
    command
        .assert()
        .failure()
        .stdout(predicate::str::contains("addOne"))
        .stdout(predicate::str::contains("addTwo").not());
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
