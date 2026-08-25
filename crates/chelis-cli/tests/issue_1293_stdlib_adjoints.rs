//! Executable acceptance for chelis#1293's recursive stdlib List adjoints.
//!
//! The three wrappers must preserve the primal List length and positional
//! order in their cotangents. These are package-linked calls, not copies of
//! the wrapper bodies, so the test covers the published stdlib surface.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

fn assert_random_wrapper_rejects(source_name: &str, body: &str, message: &str) {
    let (_dir, reef_home, app_pkg) = make_app(source_name);
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!("module Demo.Main\n\n{body}\n"),
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

#[test]
fn stdlib_list_selection_adjoints_preserve_runtime_positions() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-adjoints");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index, take_list, drop_list)

def index_loss(xs: List[f32]) -> f32 = list_index(xs, cast(1, int64))
def take_loss(xs: List[f32]) -> f32 = {
  ys = take_list(xs, cast(2, int64))
  add(list_index(ys, cast(0, int64)), list_index(ys, cast(1, int64)))
}

def drop_loss(xs: List[f32]) -> f32 = {
  ys = drop_list(xs, cast(1, int64))
  add(list_index(ys, cast(0, int64)), list_index(ys, cast(1, int64)))
}
def nested_loss(xss: List[List[f32]]) -> f32 = {
  xs = list_index(xss, cast(0, int64))
  list_index(xs, cast(1, int64))
}

index_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
take_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
drop_values: List[f32] = [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)]
nested_values: List[List[f32]] = [
  [cast(2.0, f32), cast(3.0, f32)],
  [cast(5.0, f32)]
]
index_grad = grad(index_loss, wrt=xs)(index_values)
take_grad = grad(take_loss, wrt=xs)(take_values)
drop_grad = grad(drop_loss, wrt=xs)(drop_values)
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
        .stdout(predicate::str::contains(
            "nested_grad = [[0.0, 1.0], [0.0]]",
        ));
}

#[test]
fn stdlib_list_selection_adjoints_reject_discrete_elements() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-adjoints-discrete");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index)

def discrete_loss(xs: List[int64]) -> f32 =
  cast(list_index(xs, cast(0, int64)), f32)

values: List[int64] = [cast(2, int64)]
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

#[test]
fn stdlib_normal_like_has_seeded_pathwise_scalar_parameter_adjoints() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-normal-like-adjoints");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Init.Random (normal_like)

def total_normal(template: tensor[2, 2, f32], mean: f32, std: f32) -> f32 ! { Random } =
  normal_like(template, mean, std)
  |> sum(cast(0, int32))
  |> sum(cast(0, int32))
  |> tensor_to_scalar

template = to_tensor([
  [cast(0.0, f32), cast(0.0, f32)],
  [cast(0.0, f32), cast(0.0, f32)]
])
sample_sum = with seed(42i64) {
  total_normal(copy(template), cast(0.0, f32), cast(1.0, f32))
}
normal_grads = with seed(42i64) {
  grad(total_normal, wrt=(template, mean, std))(
    template,
    cast(0.0, f32),
    cast(1.0, f32)
  )
}
"#,
    );

    let assert = Command::cargo_bin("chelis")
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
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).expect("utf-8 stdout");

    assert!(
        stdout.contains("normal_grads.0 = tensor(shape=[2, 2], data=[0.0, 0.0, 0.0, 0.0])"),
        "Random draws are non-differentiable with respect to the template values:\n{stdout}"
    );
    let scalar = |name: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name} = ")))
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("missing scalar `{name}` in:\n{stdout}"))
    };
    assert_eq!(scalar("normal_grads.1"), 4.0);
    assert!(
        (scalar("normal_grads.2") - scalar("sample_sum")).abs() < 1e-5,
        "the std cotangent must reuse the fixed-seed standard-normal sample:\n{stdout}"
    );
}

#[test]
fn stdlib_random_wrappers_validate_parameters_before_sampling() {
    assert_random_wrapper_rejects(
        "issue-1293-normal-domain",
        r#"import Std.Init.Random (normal_like)

template = to_tensor([cast(0.0, f32)])
bad = with seed(42i64) {
  normal_like(template, cast(0.0, f32), cast(-1.0, f32))
}"#,
        "numeric trap: domain in cast at bool",
    );
    assert_random_wrapper_rejects(
        "issue-1293-normal-finite-domain",
        r#"import Std.Init.Random (normal_like)

template = to_tensor([cast(0.0, f32)])
nan = sqrt(cast(-1.0, f32))
bad = with seed(42i64) {
  normal_like(template, nan, cast(1.0, f32))
}"#,
        "numeric trap: domain in cast at bool",
    );
    assert_random_wrapper_rejects(
        "issue-1293-kaiming-domain",
        r#"import Std.Init.Kaiming (kaiming_uniform)

template = to_tensor([cast(0.0, f32)])
bad = with seed(42i64) {
  kaiming_uniform(template, cast(0.0, f32))
}"#,
        "numeric trap: domain in cast at bool",
    );
    assert_random_wrapper_rejects(
        "issue-1293-xavier-domain",
        r#"import Std.Init.XavierExt (xavier_normal)

template = to_tensor([cast(0.0, f32)])
bad = with seed(42i64) {
  xavier_normal(template, cast(1.0, f32), cast(-1.0, f32))
}"#,
        "numeric trap: domain in cast at bool",
    );
    assert_random_wrapper_rejects(
        "issue-1293-xavier-overflow-domain",
        r#"import Std.Init.XavierExt (xavier_uniform)

template = to_tensor([cast(0.0, f32)])
large = cast(3.4e38, f32)
bad = with seed(42i64) {
  xavier_uniform(template, large, large)
}"#,
        "numeric trap: domain in cast at bool",
    );
    assert_random_wrapper_rejects(
        "issue-1293-trunc-domain",
        r#"import Std.Init.XavierExt (trunc_normal)

template = to_tensor([cast(0.0, f32)])
bad = with seed(42i64) {
  trunc_normal(
    template,
    cast(0.0, f32),
    cast(1.0, f32),
    cast(2.0, f32),
    cast(1.0, f32)
  )
}"#,
        "numeric trap: domain in cast at bool",
    );
}
