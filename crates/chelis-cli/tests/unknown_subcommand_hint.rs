//! WS-C (§5.4) — the cross-version unknown-subcommand hint.
//!
//! When the shim routes to a chelis lacking a requested verb, chelis's
//! unrecognized-subcommand error is augmented with: its own version, the
//! pin source that routed here, and the `chelis +<ver> ...` override. Every
//! other clap outcome (help, version, unknown flag) is left untouched, so
//! the hint fires only for `ErrorKind::InvalidSubcommand`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::write_pinned_reef_toml;

const HINT: &str = "You are running chelis";

/// A `chelis` invocation whose cwd and chelisup home are isolated so pin
/// resolution is deterministic (no ambient `CHELIS_TOOLCHAIN`, no leaked
/// home).
fn chelis(cwd: &Path, home: &Path) -> Command {
    let mut c = Command::cargo_bin("chelis").expect("chelis binary built");
    c.current_dir(cwd)
        .env("CHELIS_HOME", home)
        .env_remove("CHELIS_TOOLCHAIN")
        .env_remove("GITHUB_TOKEN");
    c
}

#[test]
fn hint_on_nested_unknown_reef_subcommand_names_pin() {
    let tmp = tempdir().unwrap();
    let shell = tmp.path().join("shell");
    let home = tmp.path().join("home");
    write_pinned_reef_toml(&shell, "0.9.9", "");

    chelis(&shell, &home)
        .args(["reef", "definitely-not-a-verb"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unrecognized subcommand"))
        .stderr(predicate::str::contains(HINT))
        // The pin source is named from the walked-up reef.toml.
        .stderr(predicate::str::contains("reef.toml"))
        .stderr(predicate::str::contains("pinned by"))
        .stderr(predicate::str::contains("chelis +<ver>"));
}

#[test]
fn hint_on_top_level_unknown_subcommand() {
    let tmp = tempdir().unwrap();
    let shell = tmp.path().join("shell");
    let home = tmp.path().join("home");
    write_pinned_reef_toml(&shell, "0.9.9", "");

    chelis(&shell, &home)
        .arg("totally-bogus")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unrecognized subcommand"))
        .stderr(predicate::str::contains(HINT))
        .stderr(predicate::str::contains("chelis +<ver>"));
}

#[test]
fn hint_without_a_pin_still_names_version_and_omits_pinned_by() {
    // A bare cwd with no reef.toml anywhere above and an empty store: the
    // hint prints the version + `+<ver>` guidance but no "pinned by" clause.
    let tmp = tempdir().unwrap();
    let bare = tmp.path().join("bare");
    let home = tmp.path().join("home");
    fs::create_dir_all(&bare).unwrap();

    chelis(&bare, &home)
        .arg("totally-bogus")
        .assert()
        .code(2)
        .stderr(predicate::str::contains(HINT))
        .stderr(predicate::str::contains("chelis +<ver>"))
        .stderr(predicate::str::contains("pinned by").not());
}

// ---- negative parity: these must NOT fire the hint ------------------------

#[test]
fn help_has_no_hint() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    chelis(tmp.path(), &home)
        .arg("--help")
        .assert()
        .success()
        .stderr(predicate::str::contains(HINT).not())
        .stdout(predicate::str::contains(HINT).not());
}

#[test]
fn version_has_no_hint() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    chelis(tmp.path(), &home)
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("chelis "))
        .stderr(predicate::str::contains(HINT).not());
}

#[test]
fn unknown_flag_is_left_to_clap_without_hint() {
    let tmp = tempdir().unwrap();
    let home = tmp.path().join("home");
    chelis(tmp.path(), &home)
        .arg("--definitely-not-a-flag")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unexpected argument"))
        // Not an InvalidSubcommand -> no cross-version hint.
        .stderr(predicate::str::contains(HINT).not());
}
