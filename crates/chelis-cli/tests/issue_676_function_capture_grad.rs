//! Issue chelis#676 — `grad` of a typed lambda that captures a function
//! value used to lose that callable's checked metadata while entering the
//! gradient sub-context. The model application then lowered with a rank-zero
//! fallback and the backward DAG failed verification with a rank mismatch.
//!
//! This is the inline, dependency-free equivalent of Nautilus's
//! `lm_jacobian_model_wrapper`: all tensor values are direct target arguments;
//! only the model function is captured. Both evaluator and generated-C lanes
//! must return the exact Jacobian row `[1, 1]`. A capture-free objective is the
//! discriminator, and the chelis#1952 local-alias transform fence remains a
//! negative control.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::{build_and_run, parse_tensor_data, write_file};

const FUNCTION_CAPTURE: &str = r#"
def jacobian_row(
  model: &tensor[2, f32] -> &tensor[6, f32] -> tensor[6, f32],
  theta: tensor[2, f32],
  x_data: tensor[6, f32],
  output_seed: tensor[6, f32]
) -> tensor[2, f32] = {
  target = fn (
    theta_local: tensor[2, f32],
    x_local: tensor[6, f32],
    seed_local: tensor[6, f32]
  ) -> {
    prediction = model(theta_local, x_local)
    tensor_to_scalar(sum(mul(prediction, seed_local), 0i32))
  }
  grad(target, wrt=theta_local)(theta, x_data, output_seed)
}

def linear_model(
  theta: &tensor[2, f32],
  x_data: &tensor[6, f32]
) -> tensor[6, f32] = insert(sum(copy(theta), 0i32), 0i32, 6i64)

out = jacobian_row(
  linear_model,
  to_tensor([1.0f32, 2.0f32]),
  to_tensor([0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]),
  to_tensor([1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32])
)
"#;

const CAPTURE_FREE_CONTROL: &str = r#"
def objective(
  theta: tensor[2, f32],
  x_data: tensor[6, f32],
  output_seed: tensor[6, f32]
) -> f32 = {
  prediction = insert(sum(copy(theta), 0i32), 0i32, 6i64)
  tensor_to_scalar(sum(mul(prediction, output_seed), 0i32))
}

out = grad(objective, wrt=theta)(
  to_tensor([1.0f32, 2.0f32]),
  to_tensor([0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]),
  to_tensor([1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32])
)
"#;

const LOCAL_CALLABLE_SHADOW_CONTROL: &str = r#"
def model(
  theta: &tensor[2, f32],
  x_data: &tensor[6, f32]
) -> tensor[6, f32] = insert(sum(copy(theta), 0i32), 0i32, 6i64)

def jacobian_row(
  callback: &tensor[2, f32] -> &tensor[6, f32] -> tensor[6, f32],
  theta: tensor[2, f32],
  x_data: tensor[6, f32],
  output_seed: tensor[6, f32]
) -> tensor[2, f32] = {
  target = fn (
    theta_local: tensor[2, f32],
    x_local: tensor[6, f32],
    seed_local: tensor[6, f32]
  ) -> {
    prediction = callback(theta_local, x_local)
    tensor_to_scalar(sum(mul(prediction, seed_local), 0i32))
  }
  grad(target, wrt=theta_local)(theta, x_data, output_seed)
}

out = {
  model = fn (
    theta: &tensor[2, f32],
    x_data: &tensor[6, f32]
  ) -> insert(
    sum(mul(copy(theta), to_tensor([2.0f32, 3.0f32])), 0i32),
    0i32,
    6i64
  )
  jacobian_row(
    model,
    to_tensor([1.0f32, 2.0f32]),
    to_tensor([0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]),
    to_tensor([1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32])
  )
}
"#;

const DIRECT_LOCAL_CALLABLE_SHADOW_CONTROL: &str = r#"
def model(
  theta: &tensor[2, f32],
  x_data: &tensor[6, f32]
) -> tensor[6, f32] = insert(sum(copy(theta), 0i32), 0i32, 6i64)

out = {
  model = fn (
    theta: &tensor[2, f32],
    x_data: &tensor[6, f32]
  ) -> insert(
    sum(mul(copy(theta), to_tensor([2.0f32, 3.0f32])), 0i32),
    0i32,
    6i64
  )
  target = fn (
    theta_local: tensor[2, f32],
    x_local: tensor[6, f32],
    seed_local: tensor[6, f32]
  ) -> {
    prediction = model(theta_local, x_local)
    tensor_to_scalar(sum(mul(prediction, seed_local), 0i32))
  }
  grad(target, wrt=theta_local)(
    to_tensor([1.0f32, 2.0f32]),
    to_tensor([0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]),
    to_tensor([1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32])
  )
}
"#;

fn eval_tensor(source: &str, stem: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "{stem}: eval failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    parse_tensor_data(
        &String::from_utf8(output.stdout).expect("utf-8 stdout"),
        "out",
    )
}

fn assert_exact_row(actual: &[f64], lane: &str) {
    assert_eq!(actual, [1.0, 1.0], "wrong Jacobian row on {lane}");
}

#[test]
fn function_valued_capture_has_exact_eval_and_native_c_jacobian() {
    let eval = eval_tensor(FUNCTION_CAPTURE, "issue676_function_capture_eval");
    let native_stdout = build_and_run(FUNCTION_CAPTURE, "issue676_function_capture_c");
    let native = parse_tensor_data(&native_stdout, "out");
    assert_exact_row(&eval, "eval");
    assert_exact_row(&native, "native C");
    assert_eq!(eval, native, "eval and generated C must agree exactly");
}

#[test]
fn capture_free_direct_objective_remains_the_numerical_control() {
    let eval = eval_tensor(CAPTURE_FREE_CONTROL, "issue676_capture_free_eval");
    let native_stdout = build_and_run(CAPTURE_FREE_CONTROL, "issue676_capture_free_c");
    let native = parse_tensor_data(&native_stdout, "out");
    assert_exact_row(&eval, "capture-free eval");
    assert_exact_row(&native, "capture-free native C");
}

#[test]
fn local_callable_shadow_is_not_rebound_to_same_named_top_level_def() {
    let eval = eval_tensor(
        LOCAL_CALLABLE_SHADOW_CONTROL,
        "issue676_local_callable_shadow_eval",
    );
    let native_stdout = build_and_run(
        LOCAL_CALLABLE_SHADOW_CONTROL,
        "issue676_local_callable_shadow_c",
    );
    let native = parse_tensor_data(&native_stdout, "out");
    assert_eq!(eval, [2.0, 3.0], "eval rebound the lexical model shadow");
    assert_eq!(
        native,
        [2.0, 3.0],
        "native C rebound the lexical model shadow"
    );
}

#[test]
fn directly_captured_local_callable_shadow_is_not_rebound_to_top_level() {
    let eval = eval_tensor(
        DIRECT_LOCAL_CALLABLE_SHADOW_CONTROL,
        "issue676_direct_local_callable_shadow_eval",
    );
    let native_stdout = build_and_run(
        DIRECT_LOCAL_CALLABLE_SHADOW_CONTROL,
        "issue676_direct_local_callable_shadow_c",
    );
    let native = parse_tensor_data(&native_stdout, "out");
    assert_eq!(eval, [2.0, 3.0], "eval rebound the directly captured model");
    assert_eq!(
        native,
        [2.0, 3.0],
        "native C rebound the directly captured model"
    );
    assert_eq!(
        eval, native,
        "direct capture disagreed across execution lanes"
    );
}

#[test]
fn local_alias_transform_target_remains_fenced() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("issue676_local_alias_negative.ch");
    write_file(
        &path,
        "def loss(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0i32))\n\
         out = {\n\
           alias = loss\n\
           grad(alias)(to_tensor([1.0f32, 2.0f32]))\n\
         }\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let json: Value = serde_json::from_slice(&output.stdout).expect("check output json");
    assert!(
        json["score"].as_f64().is_some_and(|score| score < 1.0),
        "the chelis#1952 local-alias target must remain rejected: {json}",
    );
    assert!(
        json["errors"]
            .as_array()
            .is_some_and(|errors| errors.iter().any(|error| {
                error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("core transform fragment"))
            })),
        "expected the chelis#1952 transform-fence diagnostic: {json}",
    );
}
