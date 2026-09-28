//! CLI evaluator fixtures for nested zero-argument function calls
//! and calls inside subexpressions. Each assertion checks exact
//! output or a specific failure.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn eval_file(path: &Path) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
}

// ============================================================
// Nested zero-argument calls
// ============================================================

/// Two-level nested zero-argument call: `outer()` calls `inner()`.
#[test]
fn host_eval_two_level_nested_zero_arg_i32() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("two_level_nested.ch");
    write_file(
        &fixture,
        "def inner() -> i32 = 42\ndef outer() -> i32 = inner()\nresult = outer()\n",
    );

    // chelis#732 P1 ([05-OBS-4]/[05-OBS-2]): scalar roots render bare,
    // integers as integers. [05-OBS-7] also makes each pure nullary
    // declaration an owed root, in source order.
    eval_file(&fixture)
        .success()
        .stdout("inner = 42\nouter = 42\nresult = 42\n");
}

/// Three-level zero-arg chain. Each level returns f32 via the bare-def
/// form (no parens).
#[test]
fn host_eval_three_level_nested_zero_arg_f32() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("three_level_nested.ch");
    write_file(
        &fixture,
        "def deepest() -> f32 = 3.14\n\
         def middle() -> f32 = deepest()\n\
         def outer() -> f32 = middle()\n\
         result = outer()\n",
    );

    // Bare scalar root; the exact digits are the stored f32 value's
    // rendering (pinned value-level by the observation harness).
    eval_file(&fixture)
        .success()
        .stdout("deepest = 3.14\nmiddle = 3.14\nouter = 3.14\nresult = 3.14\n");
}

/// Zero-arg fn-call used as a subexpression inside a non-trivial app.
/// `result = add(go(), 1)` exercises the `lower_app` arity guard on
/// a nested `(app ... (var go))` form embedded in another app's
/// arglist.
#[test]
fn host_eval_zero_arg_in_subexpression() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_in_subexpr.ch");
    write_file(&fixture, "def go() -> i32 = 7\nresult = add(go(), 1)\n");

    eval_file(&fixture).success().stdout("go = 7\nresult = 8\n");
}

/// Zero-arg fn-call inside another zero-arg fn body. `def go() = add(helper(), 3)`
/// nests the call through `lower_app` twice in a single decl.
#[test]
fn host_eval_zero_arg_inside_zero_arg_body() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_in_body.ch");
    write_file(
        &fixture,
        "def helper() -> i32 = 5\n\
         def go() -> i32 = add(helper(), 3)\n\
         result = go()\n",
    );

    eval_file(&fixture)
        .success()
        .stdout("helper = 5\ngo = 8\nresult = 8\n");
}

/// Zero-arg fn returning bool.
#[test]
fn host_eval_zero_arg_bool() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_bool.ch");
    write_file(&fixture, "def go() -> bool = true\nresult = go()\n");

    eval_file(&fixture)
        .success()
        .stdout("go = true\nresult = true\n");
}

/// Zero-arg fn returning large i64 (above f32 representable-int range).
/// If the bug were latent on i64, the round-trip through f32 storage
/// would corrupt large integers; the existing fixture only tests `7`.
#[test]
fn host_eval_zero_arg_i64_large_value() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_i64_large.ch");
    write_file(&fixture, "def go() -> i64 = 9999999999i64\nresult = go()\n");

    eval_file(&fixture)
        .success()
        .stdout("go = 9999999999\nresult = 9999999999\n");
}
