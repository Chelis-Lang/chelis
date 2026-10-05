//! Permanent regressions for the third executable review of chelis#1293.
//!
//! These programs exercise the public evaluator and generated-C doors. They
//! deliberately avoid copying stdlib wrapper bodies into the test corpus.

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, make_app, write_file};

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
fn compiled_list_adjoints_are_compositional_and_preserve_recursive_shapes() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-adjoints-compositional");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (skip_list, list_index, take_list)

def sum_pair(value: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(value, cast(0, i32)))

def wrapped_select(xs: List[f32], index: i64) -> f32 = list_index(xs, index)
def wrapped_values(xs: List[f32]) -> List[f32] = xs

def second_list_loss(
  left: List[f32],
  right: List[f32],
  index: i64
) -> f32 = list_index(right, index)

def reused_selector_loss(xs: List[f32], index: i64) -> f32 =
  add(list_index(xs, index), cast(index, f32))

def wrapped_loss(xs: List[f32], index: i64) -> f32 =
  wrapped_select(xs, index)

def repeated_loss(xs: List[f32], index: i64) -> f32 =
  add(list_index(xs, index), list_index(xs, index))

def tensor_list_loss(xs: List[tensor[2, f32]], index: i64) -> f32 =
  sum_pair(list_index(xs, index))

def empty_take_loss(xs: List[f32], count: i64) -> f32 = {
  selected = take_list(xs, count)
  if eq(len(selected), cast(0, i64)) then cast(0.0, f32) else list_index(selected, cast(0, i64))
}

def reused_count_loss(xs: List[f32], count: i64) -> f32 = {
  selected = skip_list(take_list(xs, count), sub(count, cast(1, i64)))
  add(list_index(selected, cast(0, i64)), mul(cast(count, f32), cast(0.0, f32)))
}

def both_lists_loss(
  left: List[f32],
  right: List[f32],
  index: i64
) -> f32 = add(list_index(left, index), list_index(right, index))

def nested_runtime_loss(
  values: List[List[f32]],
  outer: i64,
  inner: i64
) -> f32 = list_index(list_index(values, outer), inner)

left: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
right: List[f32] = [cast(7.0, f32), cast(11.0, f32), cast(13.0, f32)]
tensor_values: List[tensor[2, f32]] = [
  to_tensor([cast(2.0, f32), cast(3.0, f32)]),
  to_tensor([cast(5.0, f32), cast(7.0, f32)])
]
nested_values: List[List[f32]] = [
  [cast(19.0, f32), cast(23.0, f32)],
  [cast(29.0, f32)]
]
mask: tensor[2, bool] = [true, false]
runtime_one: i64 = tensor_to_scalar(count(&mask, 0))
runtime_zero: i64 = sub(runtime_one, runtime_one)

second = grad(second_list_loss, wrt=right)(left, right, runtime_one)
reused = grad(reused_selector_loss, wrt=xs)(left, runtime_one)
wrapped = grad(wrapped_loss, wrt=xs)(left, runtime_one)
wrapped_actual = grad(wrapped_loss, wrt=xs)(wrapped_values(left), runtime_one)
repeated = grad(repeated_loss, wrt=xs)(left, runtime_one)
shaped = grad(tensor_list_loss, wrt=xs)(tensor_values, runtime_one)
empty_take = grad(empty_take_loss, wrt=xs)([], runtime_zero)
singleton = grad(wrapped_loss, wrt=xs)([cast(17.0, f32)], runtime_zero)
reused_count = grad(reused_count_loss, wrt=xs)(left, add(runtime_one, runtime_one))
both = grad(both_lists_loss, wrt=(left, right))(left, right, runtime_one)
nested_runtime = grad(nested_runtime_loss, wrt=values)(
  nested_values,
  runtime_one,
  runtime_zero
)
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "second = [0.0, 1.0, 0.0]",
        "reused = [0.0, 1.0, 0.0]",
        "wrapped = [0.0, 1.0, 0.0]",
        "wrapped_actual = [0.0, 1.0, 0.0]",
        "repeated = [0.0, 2.0, 0.0]",
        "shaped = [tensor(shape=[2], data=[0.0, 0.0]), tensor(shape=[2], data=[1.0, 1.0])]",
        "empty_take = []",
        "singleton = [1.0]",
        "reused_count = [0.0, 1.0, 0.0]",
        "both.0 = [0.0, 1.0, 0.0]",
        "both.1 = [0.0, 1.0, 0.0]",
        "nested_runtime = [[0.0, 0.0], [1.0]]",
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

/// The runtime selector is computed once and shared by the forward and
/// adjoint paths in both evaluation and generated C.
#[test]
fn compiled_list_grad_evaluates_runtime_selector_argument_once() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-grad-argument-once");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index)

def loss(xs: List[f32], index: i64) -> f32 =
  add(list_index(xs, index), mul(cast(index, f32), cast(0.0, f32)))

values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
mask: tensor[2, bool] = [true, false]
out = grad(loss, wrt=xs)(values, tensor_to_scalar(count(&mask, 0)))
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    assert!(eval.contains("out = [0.0, 1.0, 0.0]"), "{eval}");

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
    let source = std::fs::read_to_string(out_dir.join("main.c")).expect("generated main.c");
    assert_eq!(
        source.matches("int64_t __count_n_").count(),
        1,
        "the runtime selector expression must be lowered exactly once"
    );
    let status = common::link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "link failed: {status}");
    let compiled = std::process::Command::new(out_dir.join("main"))
        .output()
        .expect("compiled binary should run");
    assert!(
        compiled.status.success(),
        "compiled binary failed: {compiled:?}"
    );
    let stdout = String::from_utf8(compiled.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("out = [0.0, 1.0, 0.0]"), "{stdout}");
}

#[test]
fn empty_runtime_list_index_adjoint_reports_the_same_bounds_error_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-empty-list-index-adjoint-error");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index)

def loss(xs: List[f32], index: i64) -> f32 = list_index(xs, index)

false_mask: tensor[1, bool] = [false]
runtime_zero: i64 = tensor_to_scalar(count(&false_mask, 0))
bad = grad(loss, wrt=xs)([], runtime_zero)
"#,
    );

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("eval should run");
    assert!(
        !eval.status.success(),
        "empty List index unexpectedly evaluated"
    );
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        eval_stderr.contains("index 0 out of bounds for list of len 0"),
        "unexpected eval error: {eval_stderr}"
    );

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
    let status = common::link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "link failed: {status}");
    let compiled = std::process::Command::new(out_dir.join("main"))
        .output()
        .expect("compiled binary should run");
    assert!(
        !compiled.status.success(),
        "empty List index unexpectedly compiled and ran"
    );
    let compiled_stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(
        compiled_stderr.contains("index 0 out of bounds for list of len 0"),
        "unexpected compiled error: {compiled_stderr}"
    );
}

fn runtime_list_index_grad_source(len: usize) -> String {
    let values = (0..len)
        .map(|value| format!("cast({}.0, f32)", value + 1))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"module Demo.Main

import Std.Index (list_index)

def loss(xs: List[f32], index: i64) -> f32 = list_index(xs, index)

values: List[f32] = [{values}]
mask: tensor[2, bool] = [true, false]
runtime_one: i64 = tensor_to_scalar(count(&mask, 0))
out = grad(loss, wrt=xs)(values, runtime_one)
"#
    )
}

fn generate_runtime_list_index_grad_source(len: usize) -> String {
    let (_dir, reef_home, app_pkg) = make_app(&format!("issue-1293-list-growth-{len}"));
    write_file(
        &app_pkg.join("src/main.ch"),
        &runtime_list_index_grad_source(len),
    );
    let out_dir = app_pkg.join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
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
    std::fs::read_to_string(out_dir.join("main.c")).expect("generated main.c")
}

#[test]
fn runtime_list_adjoint_generated_source_growth_is_linear() {
    // This is a structural source-growth oracle, so generate each source but
    // do not redundantly evaluate, compile, link, and execute all three large
    // fixtures in one test.  The surrounding round3/round4 and stdlib suites
    // retain evaluator/native-C execution parity for runtime List adjoints;
    // keeping this test single-purpose leaves the N=128 ratchet intact while
    // respecting the per-test CI ceiling.
    let small = generate_runtime_list_index_grad_source(8);
    let medium = generate_runtime_list_index_grad_source(64);
    let large = generate_runtime_list_index_grad_source(128);
    assert!(
        medium.len() <= small.len() * 12,
        "runtime List adjoint source must scale linearly: N=8 {} bytes, N=64 {} bytes",
        small.len(),
        medium.len()
    );
    assert!(
        large.len() <= medium.len() * 3,
        "runtime List adjoint source must remain near-linear when the list doubles: N=64 {} bytes, N=128 {} bytes",
        medium.len(),
        large.len()
    );
    let helper_count = |source: &str| {
        source
            .lines()
            .filter(|line| {
                line.starts_with("static void ")
                    && line.contains("__tensor_")
                    && line.contains("chelis_tensor **inputs")
            })
            .count()
    };
    let small_helpers = helper_count(&small);
    let medium_helpers = helper_count(&medium);
    let large_helpers = helper_count(&large);
    assert!(
        small_helpers > 0,
        "the structural helper census must be live"
    );
    eprintln!(
        "runtime List codegen ratchet: N=8 {} bytes/{small_helpers} helpers; N=64 {} bytes/{medium_helpers} helpers; N=128 {} bytes/{large_helpers} helpers",
        small.len(),
        medium.len(),
        large.len()
    );
    assert!(
        medium_helpers <= small_helpers + 2 && large_helpers <= small_helpers + 2,
        "runtime List adjoint helper count must stay bounded: N=8 {small_helpers}, N=64 {medium_helpers}, N=128 {large_helpers}"
    );
}
