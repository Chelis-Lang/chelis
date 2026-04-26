//! Phase 3t.A1 follow-up — Std.Decimal panic-path coverage (RT-A1W1 MEDIUM).
//!
//! `Std.Decimal` exposes two panic paths that cannot be covered from inside
//! `chelis test` because `fail` aborts the worker before any subsequent
//! assertion can run:
//!
//!   * `decimal("garbage")` — `decimal/1` calls `fail` for malformed input
//!     because the Option-returning variant is `try_decimal/1`. The non-try
//!     entry is the panicking convenience.
//!   * `decimal_div(_, decimal_from_int(0), _, _)` — `decimal_div/4` calls
//!     `fail("decimal_div: division by zero")` for a zero denominator.
//!     There is intentionally no `try_decimal_div`.
//!
//! Both paths are exercised here through `chelis eval --file`. Each test
//! stages chelis-std into a tempdir reef home, writes a `main.ch` whose
//! module-load triggers the panic path, and asserts:
//!
//!   * exit code != 0
//!   * stderr contains the branded fail message
//!
//! The pattern mirrors `phase3t_test_std.rs::assert_eval_fails_with`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package must exist")
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

fn make_app(dir_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join(dir_name);
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();
    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "=0.2.6"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

fn assert_eval_fails_with(reef_home: &Path, app_pkg: &Path, stderr_contains: &[&str]) {
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ]);
    let mut assertion = cmd.assert().failure();
    for fragment in stderr_contains {
        assertion = assertion.stderr(predicate::str::contains(*fragment));
    }
    let _ = assertion;
}

#[test]
fn decimal_of_garbage_string_calls_fail_with_branded_message() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-decimal-panic-garbage");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Decimal (decimal)

bad = decimal("not a number")
"#,
    );
    // The fail in Std.Decimal.decimal/1 reads:
    //   string_concat("decimal: invalid literal ", text)
    // so the branded prefix and the offending text both appear on stderr.
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["decimal: invalid literal", "not a number"],
    );
}

#[test]
fn decimal_div_by_zero_calls_fail_with_branded_message() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-decimal-panic-divzero");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Decimal (decimal_div, decimal_from_int, round_half_even)

quotient = decimal_div(decimal_from_int(cast(1, int64)), decimal_from_int(cast(0, int64)), cast(0, int64), round_half_even())
"#,
    );
    // Std.Decimal.decimal_div/4 fails with the literal:
    //   "decimal_div: division by zero"
    assert_eval_fails_with(&reef_home, &app_pkg, &["decimal_div: division by zero"]);
}
