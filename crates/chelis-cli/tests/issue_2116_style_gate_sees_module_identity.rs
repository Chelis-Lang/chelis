//! Acceptance for chelis#2116.
//!
//! `chelis lint --check` exited 0 on a package that `chelis reef build`
//! rejects at load time. The style tools read each file in isolation and
//! never saw the manifest, so a green style gate was not evidence that the
//! package loads — and a CI or agent loop that used it as a pre-build check
//! got a false pass, then discovered the failure one stage later. That was
//! reported by the QFBench/Voyage agent-authoring benchmark, where a
//! regenerated source file passed the style gate and broke the build.
//!
//! What this suite pins is the invariant behind the report, not just the one
//! reproducer: **for the module-identity class, `chelis lint --check` and
//! `chelis reef build` agree.** A lint that says yes to a package the loader
//! says no to is the defect; so is a lint that rejects a package the loader
//! accepts, which would be worse, because it would block valid work.
//!
//! `chelis fmt --check` is deliberately NOT widened. It claims that one file
//! is canonically formatted and nothing more, and the contract invariant is
//! that a command's success claim be true, not that every command check
//! everything. Widening the formatter into a partial build would trade this
//! defect for a larger one.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// Every module-identity shape measured against the loader, with the verdict
/// both surfaces must reach.
const CASES: &[(&str, &str, &str, bool)] = &[
    // (case name, file path under the package root, module declared, loads?)
    ("control", "src/data.ch", "Ub10.Data", true),
    ("nested path", "src/nn/linear.ch", "Ub10.Nn.Linear", true),
    ("free casing", "src/data.ch", "UB10.DATA", true),
    ("prefix mismatch", "src/data.ch", "Data", false),
    ("foreign prefix", "src/data.ch", "Other.Data", false),
    ("path mismatch", "src/data.ch", "Ub10.Other", false),
    (
        "dropped directory",
        "src/nn/linear.ch",
        "Ub10.Linear",
        false,
    ),
];

fn write_package(root: &Path, file: &str, module: &str) {
    let path = root.join(file);
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir src");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"ub10\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Ub10\"\n\n[dependencies]\n",
            env!("CARGO_PKG_VERSION")
        ),
    )
    .expect("write reef.toml");
    fs::write(
        &path,
        format!("module {module}\ndef value(x: i32) -> i32 = x\n"),
    )
    .expect("write source");
}

/// Whether `lint --check` reports a module-identity violation for `root`.
///
/// Deliberately not "did `lint --check` exit 0": a package can load and still
/// break an unrelated style rule. `module UB10.DATA` is exactly that — the
/// loader accepts it, and §6.2 rejects the ALL-CAPS components. Reading the
/// overall exit code here would make this suite fail for a reason that has
/// nothing to do with chelis#2116.
fn lint_reports_identity_violation(root: &Path) -> bool {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--check", root.to_str().expect("utf-8 path")])
        .output()
        .expect("run lint");
    String::from_utf8_lossy(&output.stdout).contains("reef-module-identity")
        || String::from_utf8_lossy(&output.stderr).contains("reef-module-identity")
}

fn build_succeeds(root: &Path, reef_home: &Path) -> bool {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", reef_home)
        .args(["reef", "build", root.to_str().expect("utf-8 path")])
        .output()
        .expect("run reef build")
        .status
        .success()
}

/// The reproducer exactly as chelis#2116 filed it: `lint --check` must no
/// longer report success on a package the loader rejects, and must say why in
/// the loader's own words.
#[test]
fn lint_check_rejects_the_reported_package() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("ub10");
    write_package(&root, "src/data.ch", "Data");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--check", root.to_str().expect("utf-8 path")])
        .assert()
        .failure()
        .stdout(predicates::str::contains("reef-module-identity"))
        .stdout(predicates::str::contains("(§6.5)"))
        .stdout(predicates::str::contains(
            "module `Data` does not belong to module_prefix `Ub10`",
        ));
}

/// `lint --check` reaches the same verdict whether it is pointed at the
/// package root, a source subdirectory, or the single file — the manifest is
/// an ancestor in the last two cases, not a walked entry.
#[test]
fn lint_check_rejects_from_every_invocation_shape() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("ub10");
    write_package(&root, "src/data.ch", "Data");

    for target in [root.clone(), root.join("src"), root.join("src/data.ch")] {
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .args(["lint", "--check", target.to_str().expect("utf-8 path")])
            .assert()
            .failure()
            .stdout(predicates::str::contains("reef-module-identity"));
    }
}

/// The invariant. Both directions, over every measured shape: a package the
/// loader accepts must lint clean, and a package it rejects must not.
#[test]
fn lint_check_and_reef_build_agree_on_module_identity() {
    for (name, file, module, loads) in CASES {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("ub10");
        let reef_home = dir.path().join("reef-home");
        write_package(&root, file, module);

        let built = build_succeeds(&root, &reef_home);
        assert_eq!(
            built, *loads,
            "case {name:?}: premise wrong; `reef build` disagrees with the table"
        );
        let flagged = lint_reports_identity_violation(&root);
        assert_eq!(
            flagged, !built,
            "case {name:?}: lint flagged identity = {flagged}, `reef build` loaded = {built}"
        );
    }
}

/// `fmt --check` is unchanged, and that is deliberate. Pinning it keeps a
/// later well-meaning widening from happening silently.
#[test]
fn fmt_check_still_only_judges_formatting() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("ub10");
    write_package(&root, "src/data.ch", "Data");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "fmt",
            "--check",
            root.join("src/data.ch").to_str().expect("utf-8 path"),
        ])
        .assert()
        .success();
}
