//! chelis#2414: a `Std.*` import in a package from `chelis reef init`
//! resolves on the first run of every command, with or without a
//! `reef.lock`. The bundled runtime is an implicit dependency of every
//! package, so the resolved package set cannot depend on whether a lock
//! exists. `chelis eval --file` must not write the lock (chelis#1520).

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const JSON_PROGRAM: &str = "module Plock.Main
import Std.Io.Json (try_parse_json)
export (main)
def main() -> i64 =
  match try_parse_json(\"{}\") with {
    | Some(_) => cast(1, i64)
    | None => cast(0, i64)
  }
";

const TEST_PROGRAM: &str = "module Plock.Main
import Std.Test (assert_true)
export (main)
def main() -> i64 = cast(1, i64)
";

const TEST_FILE: &str = "module Plock.Truth
import Std.Test (assert_true)
export (test_truth)
def test_truth() -> unit ! { Test } = assert_true(true, \"truth\")
";

const MISSING_PROGRAM: &str = "module Plock.Main
import Std.Nope (anything)
export (main)
def main() -> i64 = cast(1, i64)
";

fn chelis(package: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .current_dir(package)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env(
            "CHELIS_REEF_HOME",
            package.parent().expect("package parent").join("reef-home"),
        );
    command
}

/// A fresh `chelis reef init` package whose `src/main.ch` is `program`.
/// With `with_lock`, the scaffold is checked first, which writes the
/// package's `reef.lock` before `program` replaces the scaffold source.
fn fresh_package(program: &str, with_lock: bool) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let package = dir.path().join("plock");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            "plock",
            "--module-prefix",
            "Plock",
            "--output",
        ])
        .arg(&package)
        .assert()
        .success();
    assert!(
        !package.join("reef.lock").exists(),
        "reef init must not write a lock"
    );
    if with_lock {
        chelis(&package)
            .args(["check", "src/main.ch"])
            .assert()
            .success();
        assert!(
            package.join("reef.lock").is_file(),
            "the lock-seeding check must write reef.lock"
        );
    }
    fs::write(package.join("src/main.ch"), program).expect("write program");
    (dir, package)
}

fn assert_first_runs_succeed(program: &str, with_lock: bool) {
    let label = if with_lock {
        "with a lock"
    } else {
        "without a lock"
    };

    let (_dir, package) = fresh_package(program, with_lock);
    for _ in 0..2 {
        chelis(&package)
            .args(["eval", "--file", "src/main.ch"])
            .assert()
            .success()
            .stdout("main = 1\n");
    }
    assert_eq!(
        package.join("reef.lock").exists(),
        with_lock,
        "eval must leave the lock state unchanged ({label}; chelis#1520)"
    );

    let (_dir, package) = fresh_package(program, with_lock);
    let output = chelis(&package)
        .args(["check", "src/main.ch"])
        .output()
        .expect("run check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("\"score\": 1,"),
        "first check {label} must score 1: {stdout}"
    );

    let (_dir, package) = fresh_package(program, with_lock);
    chelis(&package)
        .args(["build", "--target", "c", "src/main.ch", "-o"])
        .arg(package.join("out"))
        .assert()
        .success();
    assert!(package.join("out/main.c").is_file(), "build {label}");

    let (_dir, package) = fresh_package(program, with_lock);
    chelis(&package)
        .args(["prove", "src/main.ch"])
        .assert()
        .success();

    let (_dir, package) = fresh_package(program, with_lock);
    chelis(&package).args(["reef", "build"]).assert().success();
}

#[test]
fn std_io_json_import_resolves_on_first_run_without_lock() {
    assert_first_runs_succeed(JSON_PROGRAM, false);
}

#[test]
fn std_io_json_import_resolves_on_first_run_with_lock() {
    assert_first_runs_succeed(JSON_PROGRAM, true);
}

#[test]
fn std_test_import_resolves_on_first_run_without_lock() {
    assert_first_runs_succeed(TEST_PROGRAM, false);
}

#[test]
fn std_test_import_resolves_on_first_run_with_lock() {
    assert_first_runs_succeed(TEST_PROGRAM, true);
}

#[test]
fn chelis_test_resolves_std_test_on_first_run() {
    for with_lock in [false, true] {
        let (_dir, package) = fresh_package(TEST_PROGRAM, with_lock);
        fs::create_dir_all(package.join("tests")).expect("mkdir tests");
        fs::write(package.join("tests/truth.ch"), TEST_FILE).expect("write test");
        chelis(&package)
            .arg("test")
            .assert()
            .success()
            .stdout(predicates::str::contains("1 passed, 0 failed"));
    }
}

/// Negative parity: a module no package provides stays unresolved on every
/// run, with or without a lock, so the implicit runtime does not turn an
/// unknown import into a silent success.
#[test]
fn unknown_std_module_stays_unresolved_with_and_without_lock() {
    for with_lock in [false, true] {
        let (_dir, package) = fresh_package(MISSING_PROGRAM, with_lock);
        for _ in 0..2 {
            chelis(&package)
                .args(["eval", "--file", "src/main.ch"])
                .assert()
                .failure()
                .stderr(predicates::str::contains("unresolved import `Std.Nope`"));
        }
        let output = chelis(&package)
            .args(["check", "src/main.ch"])
            .output()
            .expect("run check");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !output.status.success() && stdout.contains("unresolved import `Std.Nope`"),
            "check must reject the unknown module (lock: {with_lock}): {stdout}"
        );
    }
}
