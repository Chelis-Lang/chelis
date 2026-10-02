//! Phase 3t — `chelis reef install --from-monorepo` integration test.
//!
//! Closes Nautilus's UPSTREAM_BUGS N1: a fresh runner with no
//! `~/.chelis/reef/` cache could not install chelis-std declaratively
//! because no `chelis reef install / add / fetch` subcommand existed.
//!
//! This file exercises the `chelis reef install --from-monorepo
//! <chelis-repo-path> chelis-std=<version>` form against a monorepo layout
//! staged in a tempdir: the real `packages/chelis-std/reef.toml` and, in
//! its `dist/`, the archive and shell this binary embeds (the pair
//! `chelis reef build packages/chelis-std` writes there). `CHELIS_REEF_HOME`
//! points inside a tempdir for total isolation from the developer's real
//! `~/.chelis/reef/`.
//!
//! Coverage:
//!   * Positive: install populates the registry and a downstream
//!     `chelis check` against the bundled chelis-std version returns
//!     score 1.
//!   * Positive (no args): install with no name selectors installs
//!     every package in the monorepo's `packages/`.
//!   * Negative: requesting a package that doesn't exist in the
//!     monorepo errors cleanly without writing a partial index.
//!   * Negative: invocation with no `--from-monorepo` and no positional
//!     args prints a usage-style error and exits non-zero.

use assert_cmd::Command;
use chelis_std_bundle::{BUNDLED_CHELIS_STD_VERSION, CHELIS_STD_ARCHIVE, CHELIS_STD_SHELL};
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

/// A monorepo layout under `dir` whose `packages/chelis-std` carries the real
/// manifest and, in `dist/`, the archive and shell this binary embeds.
fn staged_monorepo(dir: &Path) -> PathBuf {
    let monorepo = dir.join("monorepo");
    let package = monorepo.join("packages/chelis-std");
    fs::create_dir_all(package.join("dist")).expect("mkdir staged dist");
    fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std/reef.toml"),
        package.join("reef.toml"),
    )
    .expect("copy chelis-std manifest");
    let stem = format!("chelis-std-{BUNDLED_CHELIS_STD_VERSION}");
    fs::write(
        package.join(format!("dist/{stem}.tar.zst")),
        CHELIS_STD_ARCHIVE,
    )
    .expect("write staged archive");
    fs::write(package.join(format!("dist/{stem}.chb")), CHELIS_STD_SHELL)
        .expect("write staged shell");
    monorepo
}

fn std_selector() -> String {
    format!("chelis-std={BUNDLED_CHELIS_STD_VERSION}")
}

fn installed_message() -> String {
    format!("Installed chelis-std {BUNDLED_CHELIS_STD_VERSION}")
}

#[test]
fn reef_install_from_monorepo_populates_registry_and_unblocks_check() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let monorepo = staged_monorepo(dir.path());

    // Pre-condition: the reef home does not exist (or at least has no
    // index). This is the critical "fresh runner" state that N1
    // describes.
    assert!(
        !reef_home.join("index.json").exists(),
        "reef home must start empty"
    );

    // Run the install.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            monorepo.to_str().unwrap(),
            &std_selector(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(installed_message()));

    // Post-condition: the registry layout matches what
    // load_registry_package expects.
    let pkg_dir = reef_home.join(format!("packages/chelis-std/{BUNDLED_CHELIS_STD_VERSION}"));
    let archive = pkg_dir.join(format!("chelis-std-{BUNDLED_CHELIS_STD_VERSION}.tar.zst"));
    let shell = pkg_dir.join(format!("chelis-std-{BUNDLED_CHELIS_STD_VERSION}.chb"));
    assert!(
        archive.exists(),
        "archive must exist at {}",
        archive.display()
    );
    assert!(shell.exists(), "shell must exist at {}", shell.display());

    let index_path = reef_home.join("index.json");
    assert!(
        index_path.exists(),
        "index.json must exist at {}",
        index_path.display()
    );
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&index_path).expect("read index"))
            .expect("parse index");
    let chelis_std = index
        .get("packages")
        .and_then(|p| p.get("chelis-std"))
        .and_then(|p| p.as_array())
        .expect("index.packages.chelis-std must be an array");
    assert_eq!(chelis_std.len(), 1);
    let entry = &chelis_std[0];
    assert_eq!(entry["version"], BUNDLED_CHELIS_STD_VERSION);
    // `compiler` is whatever the prebuilt shell was built against; it
    // must be a non-empty string.
    let compiler = entry["compiler"].as_str().expect("compiler string");
    assert!(!compiler.is_empty());
    assert!(
        entry["archive_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64),
        "archive_sha256 must be a 64-char hex digest"
    );
    assert!(
        entry["shell_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64),
        "shell_sha256 must be a 64-char hex digest"
    );

    // Stage a tiny downstream package that depends on chelis-std and
    // run `chelis check` against it. With no other workaround in place,
    // this is the end-to-end proof that install unblocked Nautilus's
    // bootstrap path.
    let app = dir.path().join("downstream");
    fs::create_dir_all(app.join("src")).expect("mkdir downstream/src");
    fs::write(
        app.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.4.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "{BUNDLED_CHELIS_STD_VERSION}" }}
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    )
    .expect("write reef.toml");
    fs::write(
        app.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_true)

def test_case() -> unit ! { Test } = assert_true(true, "ok")

ran = test_case()
"#,
    )
    .expect("write main.ch");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app)
        .args(["check", app.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
}

#[test]
fn reef_install_from_monorepo_no_args_installs_every_package() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let monorepo = staged_monorepo(dir.path());

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            monorepo.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(installed_message()));

    // chelis-std is the only package in the staged `packages/`, so no-args
    // install must have populated it.
    assert!(
        reef_home
            .join(format!(
                "packages/chelis-std/{v}/chelis-std-{v}.chb",
                v = BUNDLED_CHELIS_STD_VERSION
            ))
            .exists()
    );
}

#[test]
fn reef_install_from_monorepo_unknown_package_errors_cleanly() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let monorepo = staged_monorepo(dir.path());

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            monorepo.to_str().unwrap(),
            "nonexistent-pkg=0.2.0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("nonexistent-pkg"));

    // Negative: the index must NOT be created on a failed install.
    // (We accept the possibility that the directory itself was created
    // before the failure — what matters is no half-written index.)
    assert!(
        !reef_home.join("index.json").exists(),
        "index.json must not exist after a failed install"
    );
}

#[test]
fn reef_install_without_source_emits_usage_error() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "install"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--from-monorepo"));
}

#[test]
fn reef_install_help_lists_subcommand() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("install"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "install", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--from-monorepo"));
}

// Sanity — the install path must populate the registry strictly under
// CHELIS_REEF_HOME and never touch the developer's real ~/.chelis/reef/.
#[test]
fn reef_install_respects_chelis_reef_home_isolation() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");

    // Sentinel: capture mtime of any pre-existing real reef index, if
    // any. We don't assert on its contents (the developer might have
    // one) — just that we never wrote to it.
    let real_reef_home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".chelis/reef/index.json"));
    let pre_state = real_reef_home
        .as_ref()
        .and_then(|p| fs::metadata(p).ok().map(|m| (p.clone(), m.modified().ok())));

    let monorepo = staged_monorepo(dir.path());
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            monorepo.to_str().unwrap(),
            &std_selector(),
        ])
        .assert()
        .success();

    if let Some((path, before)) = pre_state {
        let after = fs::metadata(&path).ok().and_then(|m| m.modified().ok());
        assert_eq!(
            before,
            after,
            "real reef index at {} must not have been touched",
            path.display()
        );
    }
}
