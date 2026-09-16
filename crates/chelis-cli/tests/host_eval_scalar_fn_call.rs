// Regression tests for HostEval-ScalarFn-F1.
//
// `def go() -> f32 = 7.5; result = go()` silently evaluates `result`
// as `0.0` through `chelis eval --file` instead of `7.5`. The bug
// reproduces across every scalar return type (f32, f64, i64, bool) for
// zero-arg user-def calls. A one-arg control passes, confirming the bug
// is zero-arg-specific.
//
// Output shape (chelis#732 Phase 1, [05-OBS-4]): `chelis eval --file`
// renders scalar roots BARE - the old `tensor(shape=[], data=[N.N])`
// wrapper was the interpreter's rank-0 realization leaking into the
// observation channel and is no longer an exit form. bool prints
// true/false, integers print as integers ([05-OBS-2]). [05-OBS-7] makes the
// pure nullary declaration and the value binding roots in source order.

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
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
}

#[test]
fn host_eval_scalar_f32_zero_arg_returns_body_literal() {
    // Primary reproduction. With the bug, this prints `0.0`; expected
    // is the body literal 7.5.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("scalar_f32_zero_arg.ch");
    write_file(&fixture, "def go() -> f32 = 7.5\nresult = go()\n");

    eval_file(&fixture)
        .success()
        .stdout("go = 7.5\nresult = 7.5\n");
}

#[test]
fn host_eval_scalar_f64_zero_arg_returns_body_literal() {
    // f64 path. Surf float literals default to f32 so the body uses
    // an explicit `cast` to f64 to match the declared return type.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("scalar_f64_zero_arg.ch");
    write_file(
        &fixture,
        "def go() -> f64 = cast(7.5, f64)\nresult = go()\n",
    );

    eval_file(&fixture)
        .success()
        .stdout("go = 7.5\nresult = 7.5\n");
}

#[test]
fn host_eval_scalar_i64_zero_arg_returns_body_literal() {
    // i64 path. Integers print as integers ([05-OBS-2]).
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("scalar_i64_zero_arg.ch");
    write_file(&fixture, "def go() -> i64 = 7i64\nresult = go()\n");

    eval_file(&fixture).success().stdout("go = 7\nresult = 7\n");
}

#[test]
fn host_eval_scalar_bool_zero_arg_returns_body_literal() {
    // bool path. `true` renders as `true` ([05-OBS-2]).
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("scalar_bool_zero_arg.ch");
    write_file(&fixture, "def go() -> bool = true\nresult = go()\n");

    eval_file(&fixture)
        .success()
        .stdout("go = true\nresult = true\n");
}

#[test]
fn host_eval_scalar_one_arg_returns_arg_value() {
    // Negative control. A one-arg scalar fn call works correctly
    // through `chelis eval --file` before and after the fix. This
    // pins the bug as zero-arg-specific.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("scalar_one_arg.ch");
    write_file(&fixture, "def go(x: f32) -> f32 = x\nresult = go(7.5)\n");

    eval_file(&fixture).success().stdout("result = 7.5\n");
}
