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

import Std.Index (drop_list, list_index, take_list)

def sum_pair(value: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(value, cast(0, int32)))

def wrapped_select(xs: List[f32], index: int64) -> f32 = list_index(xs, index)
def wrapped_values(xs: List[f32]) -> List[f32] = xs

def second_list_loss(
  left: List[f32],
  right: List[f32],
  index: int64
) -> f32 = list_index(right, index)

def reused_selector_loss(xs: List[f32], index: int64) -> f32 =
  add(list_index(xs, index), cast(index, f32))

def wrapped_loss(xs: List[f32], index: int64) -> f32 =
  wrapped_select(xs, index)

def repeated_loss(xs: List[f32], index: int64) -> f32 =
  add(list_index(xs, index), list_index(xs, index))

def tensor_list_loss(xs: List[tensor[2, f32]], index: int64) -> f32 =
  sum_pair(list_index(xs, index))

def empty_take_loss(xs: List[f32], count: int64) -> f32 = {
  selected = take_list(xs, count)
  if eq(len(selected), cast(0, int64)) then cast(0.0, f32) else list_index(selected, cast(0, int64))
}

def reused_count_loss(xs: List[f32], count: int64) -> f32 = {
  selected = drop_list(take_list(xs, count), sub(count, cast(1, int64)))
  add(list_index(selected, cast(0, int64)), mul(cast(count, f32), cast(0.0, f32)))
}

def both_lists_loss(
  left: List[f32],
  right: List[f32],
  index: int64
) -> f32 = add(list_index(left, index), list_index(right, index))

def nested_runtime_loss(
  values: List[List[f32]],
  outer: int64,
  inner: int64
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
runtime_one: int64 = tensor_to_scalar(count(&mask, 0))
runtime_zero: int64 = sub(runtime_one, runtime_one)

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

#[test]
fn handled_random_grad_merges_computed_runtime_predicate_paths_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-random-computed-predicates");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Init.Random (normal_like)
import Std.Init.XavierExt (xavier_uniform)

def fresh_template() -> tensor[2, f32] =
  to_tensor([cast(0.0, f32), cast(0.0, f32)])

def sum_all(value: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(value, cast(0, int32)))

def float_condition(template: tensor[2, f32], scale: f32) -> f32 ! { Random } =
  if gt(scale, cast(0.0, f32))
  then sum_all(normal_like(template, cast(0.0, f32), scale))
  else sum_all(template)

def int_condition(template: tensor[2, f32], gate: int64, scale: f32) -> f32 ! { Random } =
  if eq(gate, cast(1, int64))
  then add(
    sum_all(normal_like(template, cast(0.0, f32), scale)),
    sum_all(xavier_uniform(template, scale, scale))
  )
  else sum_all(template)

def inner_condition(template: tensor[2, f32], gate: int64, scale: f32) -> f32 ! { Random } =
  if eq(gate, cast(1, int64))
  then add(
    sum_all(normal_like(template, cast(0.0, f32), scale)),
    sum_all(xavier_uniform(template, scale, scale))
  )
  else sum_all(normal_like(template, cast(0.0, f32), scale))

def nested_condition(template: tensor[2, f32], gate: int64, scale: f32) -> f32 ! { Random } = {
  positive = gt(scale, cast(0.0, f32))
  selected = if positive then inner_condition(template, gate, scale) else sum_all(template)
  reused = if positive then cast(0.0, f32) else cast(0.0, f32)
  add(selected, reused)
}

def guarded_error(template: tensor[2, f32], scale: f32) -> f32 ! { Random } =
  if gt(scale, cast(0.0, f32))
  then sum_all(normal_like(template, cast(0.0, f32), scale))
  else fail("untaken error branch")

false_mask: tensor[1, bool] = [false]
runtime_zero: int64 = tensor_to_scalar(count(&false_mask, 0))

after_float_grad = with seed(241i64) {
  skipped = grad(float_condition, wrt=scale)(fresh_template(), cast(-1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_float_forward = with seed(241i64) {
  skipped = float_condition(fresh_template(), cast(-1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_int_grad = with seed(251i64) {
  skipped = grad(int_condition, wrt=scale)(
    fresh_template(),
    runtime_zero,
    cast(-1.0, f32)
  )
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_int_forward = with seed(251i64) {
  skipped = int_condition(fresh_template(), runtime_zero, cast(-1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_nested_two_grad = with seed(257i64) {
  used = grad(nested_condition, wrt=(template, scale))(
    fresh_template(),
    cast(1, int64),
    cast(1.0, f32)
  )
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_nested_two_forward = with seed(257i64) {
  used = nested_condition(fresh_template(), cast(1, int64), cast(1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_nested_one_grad = with seed(259i64) {
  used = grad(nested_condition, wrt=scale)(
    fresh_template(),
    runtime_zero,
    cast(1.0, f32)
  )
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_nested_one_forward = with seed(259i64) {
  used = nested_condition(fresh_template(), runtime_zero, cast(1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_guarded_grad = with seed(269i64) {
  used = grad(guarded_error, wrt=scale)(fresh_template(), cast(1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}
after_guarded_forward = with seed(269i64) {
  used = guarded_error(fresh_template(), cast(1.0, f32))
  sum_all(normal_like(fresh_template(), cast(0.0, f32), cast(1.0, f32)))
}

float_path_parity = eq(after_float_grad, after_float_forward)
int_path_parity = eq(after_int_grad, after_int_forward)
nested_two_word_parity = eq(after_nested_two_grad, after_nested_two_forward)
nested_one_word_parity = eq(after_nested_one_grad, after_nested_one_forward)
guarded_error_parity = eq(after_guarded_grad, after_guarded_forward)
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "float_path_parity = true",
        "int_path_parity = true",
        "nested_two_word_parity = true",
        "nested_one_word_parity = true",
        "guarded_error_parity = true",
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

#[test]
fn compiled_list_grad_evaluates_runtime_selector_argument_once() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-grad-argument-once");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index)

def loss(xs: List[f32], index: int64) -> f32 =
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

def loss(xs: List[f32], index: int64) -> f32 = list_index(xs, index)

false_mask: tensor[1, bool] = [false]
runtime_zero: int64 = tensor_to_scalar(count(&false_mask, 0))
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

def loss(xs: List[f32], index: int64) -> f32 = list_index(xs, index)

values: List[f32] = [{values}]
mask: tensor[2, bool] = [true, false]
runtime_one: int64 = tensor_to_scalar(count(&mask, 0))
out = grad(loss, wrt=xs)(values, runtime_one)
"#
    )
}

fn build_runtime_list_index_grad_source(len: usize) -> String {
    let (_dir, reef_home, app_pkg) = make_app(&format!("issue-1293-list-growth-{len}"));
    write_file(
        &app_pkg.join("src/main.ch"),
        &runtime_list_index_grad_source(len),
    );
    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let out_dir = app_pkg.join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let source = std::fs::read_to_string(out_dir.join("main.c")).expect("generated main.c");
    let status = common::link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "N={len} link failed: {status}");
    let compiled = std::process::Command::new(out_dir.join("main"))
        .output()
        .unwrap_or_else(|error| panic!("N={len} compiled binary should run: {error}"));
    assert!(
        compiled.status.success(),
        "N={len} compiled binary failed: {compiled:?}"
    );
    let compiled_stdout = String::from_utf8(compiled.stdout).expect("utf-8 stdout");
    assert_eq!(
        compiled_stdout, eval,
        "N={len} evaluator and generated C must remain byte-exact"
    );
    source
}

#[test]
fn runtime_list_adjoint_generated_source_growth_is_linear() {
    let small = build_runtime_list_index_grad_source(8);
    let large = build_runtime_list_index_grad_source(64);
    assert!(
        large.len() <= small.len() * 12,
        "runtime List adjoint source must scale linearly: N=8 {} bytes, N=64 {} bytes",
        small.len(),
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
    let large_helpers = helper_count(&large);
    assert!(
        small_helpers > 0,
        "the structural helper census must be live"
    );
    eprintln!(
        "runtime List codegen ratchet: N=8 {} bytes/{small_helpers} helpers; N=64 {} bytes/{large_helpers} helpers",
        small.len(),
        large.len()
    );
    assert!(
        large_helpers <= small_helpers + 2,
        "runtime List adjoint helper count must stay bounded: N=8 {small_helpers}, N=64 {large_helpers}"
    );
}
