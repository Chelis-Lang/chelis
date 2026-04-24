//! Phase 3t.3 — Std.Test module integration tests.
//!
//! Verifies that:
//! 1. A program that imports every `Std.Test` assertion with passing values
//!    checks and evaluates cleanly (Test effect flows through the wrappers).
//! 2. Calling an `Std.Test` assertion from a `! {}`-declared function is
//!    rejected with `UnhandledEffect` (the Test effect carried by the wrapper
//!    propagates up to the call site).
//! 3. The `Std.Test.fail` wrapper specifically carries Test (the runtime
//!    builtin `fail` is tagless; `Std.Test.fail` must not be).
//! 4. `assert_close` rejects negative/NaN tolerance with a clear error at
//!    runtime.
//! 5. `assert_shape` on a mismatched tensor fails with the caller's label.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn example_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .canonicalize()
        .expect("path should exist")
}

fn package_std() -> PathBuf {
    example_path("../../packages/chelis-std")
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
compiler = "=0.2.4"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

#[test]
fn std_test_all_assertions_pass_under_test_effect() {
    // Every Std.Test assertion called with a passing value from a
    // `def test_*() -> unit ! { Test }` wrapper should check + eval clean.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-test-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (
  assert_close,
  assert_close_tensor,
  assert_eq,
  assert_eq_bool,
  assert_eq_int,
  assert_eq_string,
  assert_false,
  assert_shape,
  assert_true
)

def test_all() -> unit ! { Test } = {
  _ = assert_true(true, "t");
  _ = assert_false(false, "f");
  _ = assert_eq(1.5, 1.5, "eq-f32");
  _ = assert_eq_int(cast(3, int64), cast(3, int64), "eq-int");
  _ = assert_eq_bool(true, true, "eq-bool");
  _ = assert_eq_string("hi", "hi", "eq-str");
  _ = assert_close(1.0, 1.0, 0.0, "close-exact");
  _ = assert_close(1.0, 1.01, 0.05, "close-tol");
  _ = assert_close_tensor(to_tensor([1.0, 2.0]), to_tensor([1.0, 2.0]), 0.001, "close-tensor");
  assert_shape(to_tensor([1.0, 2.0, 3.0]), cast(3, int64), "shape-3")
}

ran = test_all()
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success();
}

#[test]
fn std_test_assertion_from_empty_effect_fn_is_rejected() {
    // A function that declares `! {}` but calls an `Std.Test` assertion must
    // be rejected by the checker with an UnhandledEffect diagnostic.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-test-leak");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_true)

def g() -> unit ! {} = assert_true(true, "x")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("UnhandledEffect"))
        .stdout(predicate::str::contains("Test"));
}

#[test]
fn std_test_fail_wrapper_carries_test_effect() {
    // Std.Test.fail is a Chelis wrapper over test_assert(false, _) so it
    // carries Test, unlike the runtime builtin `fail` which is tagless.
    // Calling Std.Test.fail from a `! {}` function must be rejected.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-test-fail-wrapper");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (fail)

def h() -> unit ! {} = fail("msg")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("UnhandledEffect"))
        .stdout(predicate::str::contains("Test"));
}

#[test]
fn std_test_assert_close_rejects_negative_tolerance() {
    // `assert_close` must fail with an "invalid tolerance" diagnostic when
    // given a negative tol. The failure is routed through test_assert(false,
    // ...) so the Test effect propagates and the eval surfaces the message.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-test-neg-tol");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_close)

def test_bad_tol() -> unit ! { Test } = assert_close(1.0, 1.0, sub(cast(0.0, f32), cast(1.0, f32)), "neg")

ran = test_bad_tol()
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid tolerance"))
        .stderr(predicate::str::contains("neg"));
}

#[test]
fn std_test_assert_shape_reports_label_on_mismatch() {
    // `assert_shape` should propagate the caller's label when the tensor's
    // length does not match `expected_n`.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-test-shape-mismatch");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_shape)

def test_shape_bad() -> unit ! { Test } = assert_shape(to_tensor([1.0, 2.0]), cast(7, int64), "shape-bad")

ran = test_shape_bad()
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        // assert_shape must brand its own failures with `assert_shape (...)`
        // rather than leaking the internal test_assert_eq_int prefix (RT2 C.2).
        .stderr(predicate::str::contains("assert_shape"))
        .stderr(predicate::str::contains("shape-bad"))
        .stderr(predicate::str::contains("expected 7"));
}
