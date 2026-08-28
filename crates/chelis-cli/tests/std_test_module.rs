//! Phase 3t.3 — Std.Test module integration tests.
//!
//! Per-assertion positive + negative coverage. For each assertion exported by
//! `Std.Test` we verify:
//!   * positive: a passing invocation through a `! { Test }` wrapper checks
//!     and evaluates cleanly (Test effect flows through the wrapper).
//!   * negative: a mismatched invocation surfaces the assertion's branded
//!     error message on stderr at eval time, with the caller's label.
//!
//! Plus structural tests:
//!   * Calling an `Std.Test` assertion from a `! {}` function is rejected
//!     with `UnhandledEffect` (the Test effect propagates).
//!   * `Std.Test.fail` carries Test (unlike the runtime builtin `fail`).
//!   * `assert_close` rejects negative tolerance.
//!   * `assert_shape` mismatches report the caller's label.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

/// Run `chelis check` on the given app's `src/main.ch` and assert it returns
/// a perfect score. Returns the assertion handle so callers can chain extra
/// stdout predicates if needed.
fn assert_check_clean(reef_home: &Path, app_pkg: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
}

/// Run `chelis eval --file src/main.ch` and assert it succeeds (status 0).
fn assert_eval_succeeds(reef_home: &Path, app_pkg: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success();
}

/// Run `chelis eval --file src/main.ch` and assert it fails. Then assert each
/// substring in `stderr_contains` appears on stderr.
fn assert_eval_fails_with(reef_home: &Path, app_pkg: &Path, stderr_contains: &[&str]) {
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
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
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_true_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-true-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_true)

def test_case() -> unit ! { Test } = assert_true(true, "t-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_true_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-true-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_true)

def test_case() -> unit ! { Test } = assert_true(false, "t-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    // `assert_true` routes through `test_assert(cond, label)`; the runtime
    // brands the failure as `assert failed: <label>`.
    assert_eval_fails_with(&reef_home, &app_pkg, &["assert failed: t-fail"]);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_false_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-false-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_false)

def test_case() -> unit ! { Test } = assert_false(false, "f-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_false_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-false-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_false)

def test_case() -> unit ! { Test } = assert_false(true, "f-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    // `assert_false(cond, label)` => `test_assert(not(cond), label)`; the
    // runtime brands the failure as `assert failed: <label>`.
    assert_eval_fails_with(&reef_home, &app_pkg, &["assert failed: f-fail"]);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq(1.5, 1.5, "eq-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq(1.5, 2.5, "eq-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    // `assert_eq` routes through the dtype-generic `test_assert_eq` builtin.
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert_eq (eq-fail): expected 2.5, got 1.5"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_int_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-int-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq(cast(3, int64), cast(3, int64), "eq-int-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_int_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-int-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq(cast(3, int64), cast(5, int64), "eq-int-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert_eq (eq-int-fail): expected 5, got 3"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_bool_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-bool-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq(true, true, "eq-bool-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_bool_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-bool-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq(true, false, "eq-bool-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert_eq (eq-bool-fail): expected false, got true"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_string_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-str-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq("hi", "hi", "eq-str-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_string_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-str-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq)

def test_case() -> unit ! { Test } = assert_eq("foo", "bar", "eq-str-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    // The generic runtime equality diagnostic renders string values directly.
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert_eq (eq-str-fail): expected bar, got foo"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_close_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-close-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_close)

def test_case() -> unit ! { Test } = {
  _ = assert_close(1.0, 1.0, 0.0, "close-exact")
  assert_close(1.0, 1.01, 0.05, "close-tol")
}

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_close_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-close-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_close)

def test_case() -> unit ! { Test } = assert_close(1.0, 2.0, 0.001, "close-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    // `assert_close` builds its own diagnostic via string_concat:
    // `assert_close (<label>): expected <expected>, got <actual>, tol <tol>`.
    // Float rendering preserves the decimal point for integral float values.
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert failed: assert_close (close-fail): expected 2.0, got 1.0, tol 0.001"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_close_tensor_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-close-tensor-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_close_tensor)

def test_case() -> unit ! { Test } =
  assert_close_tensor(to_tensor([1.0, 2.0]), to_tensor([1.0, 2.0]), 0.001, "close-tensor-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_close_tensor_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-close-tensor-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_close_tensor)

def test_case() -> unit ! { Test } =
  assert_close_tensor(to_tensor([1.0, 2.0]), to_tensor([1.0, 9.0]), 0.001, "close-tensor-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    // `assert_close_tensor` is a runtime builtin that emits the index of the
    // first mismatch: `at index <i> expected <e>, got <a>, tol <tol>`.
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert_close_tensor (close-tensor-fail): at index 1 expected 9.0, got 2.0, tol 0.001"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_tensor_int64_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-tensor-i64-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq_tensor)

def test_case() -> unit ! { Test } =
  assert_eq_tensor(to_tensor([cast(0, int64), cast(1, int64), cast(2, int64)]), to_tensor([cast(0, int64), cast(1, int64), cast(2, int64)]), "eq-tensor-i64-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_eq_tensor_int64_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-eq-tensor-i64-fail");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_eq_tensor)

def test_case() -> unit ! { Test } =
  assert_eq_tensor(to_tensor([cast(0, int64), cast(1, int64), cast(2, int64)]), to_tensor([cast(0, int64), cast(7, int64), cast(2, int64)]), "eq-tensor-i64-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["assert_eq_tensor (eq-tensor-i64-fail): first mismatch at row-major index 1"],
    );
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_shape_pass() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-shape-pass");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_shape)

def test_case() -> unit ! { Test } =
  assert_shape(to_tensor([1.0, 2.0, 3.0]), [cast(3, int64)], "shape-3-pass")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_succeeds(&reef_home, &app_pkg);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_shape_fail_reports_label() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-assert-shape-fail-perassert");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_shape)

def test_case() -> unit ! { Test } =
  assert_shape(to_tensor([1.0, 2.0]), [cast(7, int64)], "shape-fail")

ran = test_case()
"#,
    );
    assert_check_clean(&reef_home, &app_pkg);
    assert_eval_fails_with(&reef_home, &app_pkg, &["assert failed: shape-fail"]);
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
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

    // Issue #207: `chelis check` exits non-zero (exit 2) when the JSON
    // `errors` array is non-empty. Assert on stdout content; the
    // dedicated invariant test covers the exit code separately.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(predicate::str::contains("UnhandledEffect"))
        .stdout(predicate::str::contains("Test"));
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
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

    // Issue #207: see note above.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(predicate::str::contains("UnhandledEffect"))
        .stdout(predicate::str::contains("Test"));
}

#[test]
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
#[ignore = "manual gate: exhaustive Std.Test assertion matrix exceeds the default inner-loop budget"]
fn std_test_assert_shape_reports_label_on_mismatch() {
    // `assert_shape` should propagate the caller's label when the tensor's
    // length does not match `expected_n`.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-std-test-shape-mismatch");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Test (assert_shape)

def test_shape_bad() -> unit ! { Test } = assert_shape(to_tensor([1.0, 2.0]), [cast(7, int64)], "shape-bad")

ran = test_shape_bad()
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("assert failed: shape-bad"));
}
