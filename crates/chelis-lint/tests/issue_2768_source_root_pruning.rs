//! Regression tests for chelis#2768.
//!
//! The shipped traversal policy excludes `target/`, `node_modules/`,
//! `__pycache__/` and `.venv*/` as unanchored gitignore-style patterns, so
//! they match at any depth. The reef loader's walk applies no such filter.
//! A `.ch` file under `src/target/` is therefore part of the package as far
//! as `chelis reef build` is concerned, and invisible to `chelis lint
//! --check` — a false negative, measured on both commands.
//!
//! §12.2 permits whole-tree exclusions only for material "that should not be
//! part of the editable lint corpus". A file the loader compiles is part of
//! it, whatever the directory happens to be called. Inside a declared source
//! root the shipped baseline's names are a collision, not a statement about
//! this package, so they stop pruning there.
//!
//! A repository `chelis-lint.toml` entry is different: it is a deliberate
//! local declaration carrying its own cross-reference, and it keeps pruning
//! inside a source root. Generated `.ch` under `src/` is exactly the case
//! that needs it.

use chelis_lint::Violation;
use std::fs;
use std::path::Path;

const RULE: &str = "reef-module-identity";

fn write(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, body).expect("write");
}

fn package(root: &Path) {
    write(
        &root.join("reef.toml"),
        "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"Ub10\"\n",
    );
    write(
        &root.join("src/data.ch"),
        "module Ub10.Data\ndef value(x: i32) -> i32 = x\n",
    );
}

/// Violations of the identity rule over `root`.
fn identity_violations(root: &Path) -> Vec<Violation> {
    let rules = chelis_lint::registry::all_rules();
    chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect()
}

/// A bad `.ch` at `rel` inside the package must be seen.
fn assert_seen_at(rel: &str) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    package(root);
    write(
        &root.join(rel),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );
    let found = identity_violations(root);
    assert_eq!(
        found.len(),
        1,
        "{rel} is inside a source root and the loader reads it, so the lint must too; got {found:?}"
    );
}

#[test]
fn sees_a_source_file_under_a_directory_named_target() {
    assert_seen_at("src/target/x.ch");
}

#[test]
fn sees_a_source_file_under_a_directory_named_node_modules() {
    assert_seen_at("src/node_modules/x.ch");
}

#[test]
fn sees_a_source_file_under_a_directory_named_pycache() {
    assert_seen_at("src/__pycache__/x.ch");
}

#[test]
fn sees_a_source_file_under_a_directory_named_dot_venv() {
    assert_seen_at("src/.venv/x.ch");
}

#[test]
fn sees_a_source_file_nested_deeper_under_an_excluded_name() {
    assert_seen_at("src/target/deep/nested/x.ch");
}

// --- the exemption must not leak outside a source root ---

#[test]
fn still_prunes_an_excluded_directory_outside_every_source_root() {
    // `<pkg>/target/` is build output: the loader never reads it, and the
    // exemption must not reach it just because a reef.toml is nearby.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    package(root);
    write(
        &root.join("target/x.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );
    assert!(
        identity_violations(root).is_empty(),
        "build output outside a source root stays pruned"
    );
}

#[test]
fn still_prunes_an_excluded_directory_where_no_package_exists() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(
        &root.join("target/src/x.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );
    assert!(
        identity_violations(root).is_empty(),
        "no reef.toml is reachable, so nothing is exempt"
    );
}

// --- a bad source root must not delete the package ---

#[test]
fn an_invalid_additional_source_does_not_silence_the_rule_in_src() {
    // Round-3 P1, and the second instance of one class: a guard that declines
    // a manifest too broadly makes the lint silent on a package `reef build`
    // rejects. `additional_sources = [".git"]` is rejected by reef, so it must
    // grant no exemption -- but it must not delete `module_prefix` and the
    // `src` root with it, which is what returning None for the whole manifest
    // did. The violating file is in an ORDINARY `src/`, so pruning is not
    // involved.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(
        &root.join("reef.toml"),
        "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"Ub10\"\nadditional_sources = [\".git\"]\n",
    );
    write(
        &root.join("src/bad.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );
    assert_eq!(
        identity_violations(root).len(),
        1,
        "a rejected source root must not stop the rule judging src/"
    );
}

#[test]
fn an_invalid_additional_source_still_grants_no_exemption() {
    // The other half of the same property: dropping the entry must not quietly
    // turn it into a source root.
    //
    // Every spelling reef rejects gets a row. An earlier version used `.git`
    // alone, which trips only the character-set check, so the reserved-name,
    // empty and separator checks each had no killing test. `tests` is the one
    // that matters: accepting it would un-prune and judge `tests/` in a
    // package reef rejects.
    for (entry, dir) in [
        (".git", ".git/objects"),
        ("tests", "tests"),
        ("a/b", "a/b"),
        ("", "sub"),
        ("   ", "sub"),
        ("..", "sub"),
    ] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        write(
            &root.join("reef.toml"),
            &format!(
                "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"Ub10\"\nadditional_sources = [\"{entry}\"]\n"
            ),
        );
        write(
            &root.join("src/data.ch"),
            "module Ub10.Data\ndef value(x: i32) -> i32 = x\n",
        );
        write(
            &root.join(dir).join("sneaky.ch"),
            "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
        );
        assert!(
            identity_violations(root).is_empty(),
            "reef rejects {entry:?}, so {dir} is not a source root and stays pruned"
        );
    }
}

#[test]
fn a_manifest_whose_content_lives_under_an_excluded_directory_grants_nothing() {
    // §12.2: content under an excluded directory must not change an admitted
    // entry's verdict indirectly. The policy passes its own
    // `is_admitted_ancillary` as the manifest admission; without that call
    // this tree would un-prune `src/target/`. `surf-parses` is the probe
    // because `reef-module-identity` re-checks admission itself and would mask
    // the difference.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(
        &root.join("target/m.toml"),
        "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"Ub10\"\n",
    );
    std::os::unix::fs::symlink(root.join("target/m.toml"), root.join("reef.toml"))
        .expect("symlink");
    write(&root.join("src/target/broken.ch"), "@@@ not Surf (((\n");

    let rules = chelis_lint::registry::all_rules();
    let found: Vec<Violation> = chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == "surf-parses")
        .collect();
    assert!(
        found.is_empty(),
        "a manifest under an excluded directory must not un-prune: {found:?}"
    );
}

// --- a manifest's admission, not its file kind, decides whether it speaks ---

#[test]
fn an_internally_symlinked_manifest_still_governs_its_package() {
    // Round-2 P1. A first attempt at closing the admission asymmetry declined
    // every symlinked `reef.toml`. That refused the internal links §12.2
    // admits and that `chelis reef build` follows, so `chelis lint --check`
    // exited 0 on a package the loader rejects -- the false green this whole
    // change exists to remove, reintroduced by the repair.
    //
    // The violating file sits in an ORDINARY `src/`, so pruning is not
    // involved: this is purely about whether the manifest is read.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    fs::create_dir_all(root.join("shared")).expect("mkdir");
    write(
        &root.join("shared/pkg.toml"),
        "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"Ub10\"\n",
    );
    std::os::unix::fs::symlink(root.join("shared/pkg.toml"), root.join("reef.toml"))
        .expect("symlink");
    write(
        &root.join("src/bad.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );

    let found = identity_violations(root);
    assert_eq!(
        found.len(),
        1,
        "an internal symlinked manifest governs its package; got {found:?}"
    );
}

#[test]
fn an_internally_symlinked_manifest_also_grants_its_source_roots() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    fs::create_dir_all(root.join("shared")).expect("mkdir");
    write(
        &root.join("shared/pkg.toml"),
        "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"Ub10\"\n",
    );
    std::os::unix::fs::symlink(root.join("shared/pkg.toml"), root.join("reef.toml"))
        .expect("symlink");
    write(
        &root.join("src/data.ch"),
        "module Ub10.Data\ndef value(x: i32) -> i32 = x\n",
    );
    write(
        &root.join("src/target/x.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );

    assert_eq!(
        identity_violations(root).len(),
        1,
        "the exemption applies from a symlinked manifest too"
    );
}

// --- repository policy still prunes inside a source root ---

#[test]
fn a_repository_exclusion_still_prunes_inside_a_source_root() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    package(root);
    write(
        &root.join("spec/lint.md"),
        "# Lint\n\n## 4.2 Generated sources\n",
    );
    write(
        &root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/lint.md\"\n\n[[exclude]]\npattern = \"src/generated/\"\nclass = \"generated\"\ncross_ref = \"§4.2\"\n",
    );
    write(
        &root.join("src/generated/x.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );
    assert!(
        identity_violations(root).is_empty(),
        "a deliberate repository exclusion keeps pruning inside a source root"
    );
}

#[test]
fn a_repository_policy_does_not_resurrect_baseline_pruning_in_a_source_root() {
    // The presence of a repository policy must not change the baseline's
    // behaviour for names it does not mention.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    package(root);
    write(
        &root.join("spec/lint.md"),
        "# Lint\n\n## 4.2 Generated sources\n",
    );
    write(
        &root.join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/lint.md\"\n\n[[exclude]]\npattern = \"src/generated/\"\nclass = \"generated\"\ncross_ref = \"§4.2\"\n",
    );
    write(
        &root.join("src/target/x.ch"),
        "module Totally.Wrong\ndef v(x: i32) -> i32 = x\n",
    );
    assert_eq!(
        identity_violations(root).len(),
        1,
        "src/target/ is still source, repository policy or not"
    );
}
