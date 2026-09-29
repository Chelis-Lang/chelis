//! Regression tests for chelis#2116.
//!
//! `chelis lint --check` exited 0 on a package that `chelis reef build`
//! rejects at load time, because the style surface never looked at the
//! package's `module_prefix`. A green style gate was therefore not evidence
//! that the package loads, and an agent or CI loop that used it as a
//! pre-build check got a false pass and discovered the failure one stage
//! later.
//!
//! `reef-module-identity` closes that: §6.5's three violation bullets are
//! reported from the lint, with the same verdict the loader reaches.
//!
//! These tests pin:
//!   * each of §6.5's three violations is reported, at the module line;
//!   * every shape §6.5 says must load is accepted, including nested paths,
//!     additional source roots, and free casing;
//!   * a `.ch` file with no enclosing reef package is left alone, so linting
//!     loose files and non-package trees is unaffected;
//!   * the rule is blocking, so `lint --check` fails on it.

use chelis_lint::{Severity, Violation};
use std::fs;
use std::path::Path;

const RULE: &str = "reef-module-identity";

fn manifest(prefix: &str, additional: &[&str]) -> String {
    let extra = if additional.is_empty() {
        String::new()
    } else {
        let list = additional
            .iter()
            .map(|root| format!("\"{root}\""))
            .collect::<Vec<_>>()
            .join(", ");
        format!("additional_sources = [{list}]\n")
    };
    format!(
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\ncompiler = \"=0.18.11\"\nmodule_prefix = \"{prefix}\"\n{extra}"
    )
}

fn source(module: &str) -> String {
    format!("module {module}\nexport (value)\ndef value() -> i64 = cast(1, i64)\n")
}

fn write(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, body).expect("write");
}

/// Lint a temp tree containing one package and return this rule's violations.
fn lint_package(prefix: &str, additional: &[&str], files: &[(&str, &str)]) -> Vec<Violation> {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(&root.join("reef.toml"), &manifest(prefix, additional));
    for (rel, module) in files {
        write(&root.join(rel), &source(module));
    }
    let rules = chelis_lint::registry::all_rules();
    chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect()
}

// --- §6.5 violation 1: not rooted at module_prefix ---

#[test]
fn reports_module_not_rooted_at_prefix() {
    let found = lint_package("Ub10", &[], &[("src/data.ch", "Data")]);
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
    assert!(
        found[0]
            .message
            .contains("module `Data` does not belong to module_prefix `Ub10`"),
        "{}",
        found[0].message
    );
    assert_eq!(found[0].line, Some(1), "should point at the module line");
}

// --- §6.5 violation 2: rooted but not derived from the path ---

#[test]
fn reports_module_that_does_not_match_its_path() {
    let found = lint_package("Ub10", &[], &[("src/data.ch", "Ub10.Other")]);
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
    assert!(
        found[0].message.contains("does not match file path"),
        "{}",
        found[0].message
    );
}

#[test]
fn reports_dropped_directory_component() {
    let found = lint_package("Ub10", &[], &[("src/nn/linear.ch", "Ub10.Linear")]);
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
}

// --- §6.5 violation 3: no module, or more than one ---

#[test]
fn reports_file_with_no_module_declaration() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(&root.join("reef.toml"), &manifest("Ub10", &[]));
    write(
        &root.join("src/data.ch"),
        "export (value)\ndef value() -> i64 = cast(1, i64)\n",
    );
    let rules = chelis_lint::registry::all_rules();
    let found: Vec<Violation> = chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect();
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
    assert!(
        found[0].message.contains("exactly one"),
        "{}",
        found[0].message
    );
}

#[test]
fn defers_on_two_module_declarations_because_the_parser_rejects_them() {
    // Measured on the loader: two `module` lines are a PARSE error
    // ("expected module declaration only as the first declaration in a
    // file"), not a module-identity error, and `chelis fmt --check` already
    // exits 1 on it. So this shape is not part of chelis#2116's false-green
    // class, and the rule deliberately reaches no verdict rather than
    // inventing a second diagnosis for a file nothing can parse.
    let body = "module Ub10.Data\nmodule Ub10.Other\ndef value() -> i64 = cast(1, i64)\n";
    assert!(
        chelis_surf::parser::parse_str(body).is_err(),
        "premise: the parser rejects a second module declaration"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(&root.join("reef.toml"), &manifest("Ub10", &[]));
    write(&root.join("src/data.ch"), body);
    let rules = chelis_lint::registry::all_rules();
    let found: Vec<Violation> = chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect();
    assert!(found.is_empty(), "expected deferral, got {found:?}");
}

// --- positive parity: everything §6.5 says must load ---

#[test]
fn accepts_every_shape_the_loader_accepts() {
    let found = lint_package(
        "Ub10",
        &["properties"],
        &[
            ("src/data.ch", "Ub10.Data"),
            ("src/nn/linear.ch", "Ub10.Nn.Linear"),
            ("src/io/json.ch", "Ub10.Io.Json"),
            ("src/cross_entropy.ch", "Ub10.Cross_entropy"),
            ("properties/laws.ch", "Ub10.Properties.Laws"),
        ],
    );
    assert!(found.is_empty(), "expected no violations, got {found:?}");
}

#[test]
fn accepts_free_casing_on_both_sides() {
    let found = lint_package("Ub10", &[], &[("src/data.ch", "UB10.DATA")]);
    assert!(found.is_empty(), "expected no violations, got {found:?}");
}

// --- scope: the rule is about reef packages, not about `.ch` files ---

#[test]
fn ignores_ch_file_with_no_enclosing_reef_package() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(&root.join("src/data.ch"), &source("Data"));
    let rules = chelis_lint::registry::all_rules();
    let found: Vec<Violation> = chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect();
    assert!(
        found.is_empty(),
        "a loose .ch file is not a package member: {found:?}"
    );
}

#[test]
fn ignores_ch_file_outside_any_source_root() {
    // `examples/` is not a declared source root, so the loader never reads
    // the file and the lint must not invent a verdict about it.
    let found = lint_package("Ub10", &[], &[("examples/demo.ch", "Whatever")]);
    assert!(found.is_empty(), "expected no violations, got {found:?}");
}

// --- agreement at the walker boundary ---

#[test]
fn ignores_a_symlinked_ch_file_because_the_loader_never_reads_it() {
    // Measured: reef build exits 0 on this package. Its walk does not follow
    // a symlink below the root, so the linked file is not a package module.
    // The lint walker does follow it, so without the guard the rule reports a
    // violation in a package that loads -- a false positive.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(&root.join("reef.toml"), &manifest("Ub10", &[]));
    write(&root.join("src/data.ch"), &source("Ub10.Data"));
    write(&root.join("outside.ch"), &source("Totally.Wrong"));
    std::os::unix::fs::symlink(root.join("outside.ch"), root.join("src/linked.ch"))
        .expect("symlink");

    let rules = chelis_lint::registry::all_rules();
    let found: Vec<Violation> = chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect();
    assert!(
        found.is_empty(),
        "a symlinked file is not a package module: {found:?}"
    );
}

#[test]
fn still_reports_a_real_file_beside_a_symlinked_one() {
    // The guard must not become a blanket escape: a genuine violation in the
    // same source root is still reported.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    write(&root.join("reef.toml"), &manifest("Ub10", &[]));
    write(&root.join("src/data.ch"), &source("Data"));
    write(&root.join("outside.ch"), &source("Totally.Wrong"));
    std::os::unix::fs::symlink(root.join("outside.ch"), root.join("src/linked.ch"))
        .expect("symlink");

    let rules = chelis_lint::registry::all_rules();
    let found: Vec<Violation> = chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "expected the real violation only, got {found:?}"
    );
    assert!(found[0].path.ends_with("data.ch"), "{:?}", found[0].path);
}

// --- the rule blocks, which is what makes `lint --check` honest ---

#[test]
fn rule_is_blocking_and_registered() {
    let rule = chelis_lint::registry::all_rules()
        .into_iter()
        .find(|rule| rule.id() == RULE)
        .expect("reef-module-identity must be in all_rules()");
    assert_eq!(rule.severity(), Severity::Error);
    assert!(rule.severity().blocks_check());
    assert_eq!(rule.spec_ref(), "§6.5");
}
