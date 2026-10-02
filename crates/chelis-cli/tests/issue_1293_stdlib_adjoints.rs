//! Executable acceptance for chelis#1293's recursive stdlib List adjoints.
//!
//! The three wrappers must preserve the primal List length and positional
//! order in their cotangents. These are package-linked calls, not copies of
//! the wrapper bodies, so the test covers the published stdlib surface.

use assert_cmd::Command;
use predicates::prelude::*;
use std::process::Command as StdCommand;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, link_generated, make_app, write_file};

fn eval_app_stdout(reef_home: &std::path::Path, app_pkg: &std::path::Path) -> String {
    let assert = Command::cargo_bin("chelis")
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
    String::from_utf8(assert.get_output().stdout.clone()).expect("utf-8 stdout")
}

#[test]
fn stdlib_list_selection_adjoints_compile_and_run_with_eval_parity() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-adjoints-compiled");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index, take_list, skip_list)

def index_loss(xs: List[f32], idx: i64) -> f32 = list_index(xs, idx)
def fixed_index_loss(xs: List[f32], ignored: i64) -> f32 = list_index(xs, cast(0, i64))
def take_loss(xs: List[f32], count: i64) -> f32 = {
  ys = take_list(xs, count)
  add(list_index(ys, cast(0, i64)), list_index(ys, cast(1, i64)))
}

def drop_loss(xs: List[f32], count: i64) -> f32 = {
  ys = skip_list(xs, count)
  add(list_index(ys, cast(0, i64)), list_index(ys, cast(1, i64)))
}
def take_dynamic_loss(xs: List[f32], count: i64) -> f32 = {
  selected = take_list(xs, count)
  if eq(len(selected), cast(0, i64)) then cast(0.0, f32) else list_index(selected, cast(0, i64))
}
def drop_dynamic_loss(xs: List[f32], count: i64) -> f32 = {
  selected = skip_list(xs, count)
  if eq(len(selected), cast(0, i64)) then cast(0.0, f32) else list_index(selected, cast(0, i64))
}

values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
index_mask: tensor[3, bool] = [true, false, false]
runtime_index: i64 = tensor_to_scalar(count(&index_mask, 0))
runtime_count: i64 = runtime_index
ignored_count: i64 = add(runtime_count, cast(99, i64))
index_static = grad(index_loss, wrt=xs)(values, cast(1, i64))
index_runtime = grad(index_loss, wrt=xs)(values, runtime_index)
index_unrelated_count = grad(fixed_index_loss, wrt=xs)(values, ignored_count)
take_static = grad(take_loss, wrt=xs)(values, cast(2, i64))
take_truncated = grad(take_loss, wrt=xs)(values, cast(99, i64))
drop_static = grad(drop_loss, wrt=xs)(values, cast(1, i64))
take_runtime = grad(take_dynamic_loss, wrt=xs)(values, runtime_count)
drop_runtime = grad(drop_dynamic_loss, wrt=xs)(values, runtime_count)
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "index_static = [0.0, 1.0, 0.0]",
        "index_runtime = [0.0, 1.0, 0.0]",
        "index_unrelated_count = [1.0, 0.0, 0.0]",
        "take_static = [1.0, 1.0, 0.0]",
        "take_truncated = [1.0, 1.0, 0.0]",
        "drop_static = [0.0, 1.0, 1.0]",
        "take_runtime = [1.0, 0.0, 0.0]",
        "drop_runtime = [0.0, 1.0, 0.0]",
    ] {
        assert!(
            eval.contains(expected),
            "eval missing `{expected}`:\n{eval}"
        );
        assert!(
            compiled.contains(expected),
            "compiled C missing `{expected}`:\n{compiled}"
        );
    }
}

/// List selection adjoints preserve the source positions in eval and generated
/// C for static and runtime indices, truncation, and skipped prefixes.
#[test]
fn stdlib_list_selection_adjoints_preserve_runtime_positions() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-adjoints");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index, take_list, skip_list)

def index_loss(xs: List[f32], idx: i64) -> f32 = list_index(xs, idx)
def take_loss(xs: List[f32], count: i64) -> f32 = {
  ys = take_list(xs, count)
  add(list_index(ys, cast(0, i64)), list_index(ys, cast(1, i64)))
}

def drop_loss(xs: List[f32], count: i64) -> f32 = {
  ys = skip_list(xs, count)
  add(list_index(ys, cast(0, i64)), list_index(ys, cast(1, i64)))
}
def take_past_loss(xs: List[f32], count: i64) -> f32 = {
  ys = take_list(xs, count)
  add(
    add(list_index(ys, cast(0, i64)), list_index(ys, cast(1, i64))),
    list_index(ys, cast(2, i64))
  )
}
def drop_zero_loss(xs: List[f32], count: i64) -> f32 = {
  ys = skip_list(xs, count)
  add(
    add(list_index(ys, cast(0, i64)), list_index(ys, cast(1, i64))),
    list_index(ys, cast(2, i64))
  )
}
def nested_loss(xss: List[List[f32]]) -> f32 = {
  xs = list_index(xss, cast(0, i64))
  list_index(xs, cast(1, i64))
}

index_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
take_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
drop_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
take_past_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
drop_zero_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
nested_values: List[List[f32]] = [
  [cast(2.0, f32), cast(3.0, f32)],
  [cast(5.0, f32)]
]
index_grad = grad(index_loss, wrt=xs)(index_values, cast(1, i64))
take_grad = grad(take_loss, wrt=xs)(take_values, cast(2, i64))
drop_grad = grad(drop_loss, wrt=xs)(drop_values, cast(1, i64))
take_past_grad = grad(take_past_loss, wrt=xs)(take_past_values, cast(99, i64))
drop_zero_grad = grad(drop_zero_loss, wrt=xs)(drop_zero_values, cast(0, i64))
nested_grad = grad(nested_loss, wrt=xss)(nested_values)
"#,
    );

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
        .success()
        .stdout(predicate::str::contains("index_grad = [0.0, 1.0, 0.0]"))
        .stdout(predicate::str::contains("take_grad = [1.0, 1.0, 0.0]"))
        .stdout(predicate::str::contains("drop_grad = [0.0, 1.0, 1.0]"))
        .stdout(predicate::str::contains("take_past_grad = [1.0, 1.0, 1.0]"))
        .stdout(predicate::str::contains("drop_zero_grad = [1.0, 1.0, 1.0]"))
        .stdout(predicate::str::contains(
            "nested_grad = [[0.0, 1.0], [0.0]]",
        ));
}

fn assert_list_selection_grad_rejects(
    source_name: &str,
    import_name: &str,
    body: &str,
    message: &str,
) {
    let (_dir, reef_home, app_pkg) = make_app(source_name);
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!("module Demo.Main\n\nimport Std.Index ({import_name})\n\n{body}\n"),
    );

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
        .stderr(predicate::str::contains(message));
}

fn assert_runtime_index_grad_error_parity(source_name: &str, selector: &str, message: &str) {
    let (_dir, reef_home, app_pkg) = make_app(source_name);
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Index (list_index)

def loss(xs: List[f32], index: i64) -> f32 = list_index(xs, index)
values: List[f32] = [cast(2.0, f32), cast(3.0, f32)]
mask: tensor[2, bool] = [true, false]
one: i64 = tensor_to_scalar(count(&mask, 0))
runtime_index: i64 = {selector}
bad = grad(loss, wrt=xs)(values, runtime_index)
"#
        ),
    );

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
        .stderr(predicate::str::contains(message));

    let out_dir = app_pkg.join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            "--emit-c",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "link failed: {status}");
    let output = StdCommand::new(out_dir.join("main"))
        .output()
        .expect("compiled binary should run");
    assert!(
        !output.status.success(),
        "invalid index unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(message),
        "compiled error missing `{message}`:\n{stderr}"
    );
}

fn assert_runtime_list_count_grad_error_parity(source_name: &str, selection: &str, message: &str) {
    let (_dir, reef_home, app_pkg) = make_app(source_name);
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Index ({selection}, list_index)

def loss(xs: List[f32], count: i64) -> f32 = {{
  selected = {selection}(xs, count)
  if eq(len(selected), cast(0, i64)) then cast(0.0, f32) else list_index(selected, cast(0, i64))
}}
values: List[f32] = [cast(2.0, f32), cast(3.0, f32)]
mask: tensor[2, bool] = [true, false]
one: i64 = tensor_to_scalar(count(&mask, 0))
runtime_count: i64 = sub(cast(0, i64), one)
bad = grad(loss, wrt=xs)(values, runtime_count)
"#
        ),
    );

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
        .stderr(predicate::str::contains(message));

    let out_dir = app_pkg.join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            "--emit-c",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "link failed: {status}");
    let output = StdCommand::new(out_dir.join("main"))
        .output()
        .expect("compiled binary should run");
    assert!(
        !output.status.success(),
        "negative runtime count unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(message),
        "compiled error missing `{message}`:\n{stderr}"
    );
}

#[test]
fn compiled_runtime_list_index_adjoint_matches_eval_errors() {
    assert_runtime_index_grad_error_parity(
        "issue-1293-list-adjoints-runtime-negative-index",
        "sub(cast(0, i64), one)",
        "index requires non-negative index",
    );
    assert_runtime_index_grad_error_parity(
        "issue-1293-list-adjoints-runtime-out-of-bounds-index",
        "add(one, one)",
        "index 2 out of bounds for list of len 2",
    );
}

#[test]
fn compiled_runtime_list_take_and_drop_adjoints_match_eval_errors() {
    assert_runtime_list_count_grad_error_parity(
        "issue-1293-list-adjoints-runtime-negative-take",
        "take_list",
        "take requires non-negative count",
    );
    assert_runtime_list_count_grad_error_parity(
        "issue-1293-list-adjoints-runtime-negative-skip",
        "skip_list",
        "skip requires non-negative count",
    );
}

#[test]
fn stdlib_list_selection_adjoints_reject_invalid_runtime_counts() {
    assert_list_selection_grad_rejects(
        "issue-1293-list-adjoints-negative-index",
        "list_index",
        r#"def loss(xs: List[f32], count: i64) -> f32 = list_index(xs, count)
values: List[f32] = [cast(2.0, f32), cast(3.0, f32)]
bad = grad(loss, wrt=xs)(values, cast(-1, i64))"#,
        "index requires non-negative index",
    );
    assert_list_selection_grad_rejects(
        "issue-1293-list-adjoints-out-of-bounds-index",
        "list_index",
        r#"def loss(xs: List[f32], count: i64) -> f32 = list_index(xs, count)
values: List[f32] = [cast(2.0, f32), cast(3.0, f32)]
bad = grad(loss, wrt=xs)(values, cast(2, i64))"#,
        "index 2 out of bounds for list of len 2",
    );
    assert_list_selection_grad_rejects(
        "issue-1293-list-adjoints-negative-take",
        "take_list, list_index",
        r#"def loss(xs: List[f32], count: i64) -> f32 = {
  ys = take_list(xs, count)
  list_index(ys, cast(0, i64))
}
values: List[f32] = [cast(2.0, f32), cast(3.0, f32)]
bad = grad(loss, wrt=xs)(values, cast(-1, i64))"#,
        "take requires non-negative count",
    );
    assert_list_selection_grad_rejects(
        "issue-1293-list-adjoints-negative-skip",
        "skip_list, list_index",
        r#"def loss(xs: List[f32], count: i64) -> f32 = {
  ys = skip_list(xs, count)
  list_index(ys, cast(0, i64))
}
values: List[f32] = [cast(2.0, f32), cast(3.0, f32)]
bad = grad(loss, wrt=xs)(values, cast(-1, i64))"#,
        "skip requires non-negative count",
    );
}

#[test]
fn stdlib_list_selection_adjoints_reject_discrete_elements() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-adjoints-discrete");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index)

def discrete_loss(xs: List[i64]) -> f32 =
  cast(list_index(xs, cast(0, i64)), f32)

values: List[i64] = [cast(2, i64)]
bad = grad(discrete_loss, wrt=xs)(values)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("not differentiable"));
}
