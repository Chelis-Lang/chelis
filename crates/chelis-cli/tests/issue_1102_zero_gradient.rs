//! Executable acceptance tests for chelis#1102.
//!
//! A complete constant function has an exact zero cotangent with the shape
//! and dtype of the differentiated input. Both the evaluator and generated C
//! must execute that value; a successful build without execution would not
//! prove that the root survived lowering.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

const SHAPED_ZERO_PROGRAM: &str = "\
def constish(x: tensor[3, f32]) -> f32 = cast(1.0, f32)\n\
def g(theta: tensor[3, f32]) -> tensor[3, f32] = grad(constish, wrt=theta)(theta)\n\
out = g(to_tensor([cast(3.0, f32), cast(-2.0, f32), cast(7.0, f32)]))\n";

const RANK_ZERO_TENSOR_PROGRAM: &str = "\
def const_rank_zero(x: tensor[f32]) -> f32 = cast(1.0, f32)\n\
def rank_zero_grad(x: tensor[f32]) -> tensor[f32] = grad(const_rank_zero, wrt=x)(x)\n\
seed = sum(to_tensor([cast(7.0, f32)]), 0)\n\
out = rank_zero_grad(seed)\n";

const SYMBOLIC_ZERO_PROGRAM: &str = "\
def constant[n](x: tensor[n, f32]) -> f32 = cast(1.0, f32)\n\
def zero_grad[n](x: tensor[n, f32]) -> tensor[n, f32] = grad(constant, wrt=x)(x)\n\
out = zero_grad(to_tensor([1.0, 2.0, 3.0, 4.0]))\n";

const MULTIDIMENSIONAL_SYMBOLIC_ZERO_PROGRAM: &str = "\
def constant_matrix[rows, cols](x: tensor[rows, cols, f32]) -> f32 = cast(1.0, f32)\n\
def zero_matrix_grad[rows, cols](x: tensor[rows, cols, f32]) -> tensor[rows, cols, f32] = grad(constant_matrix, wrt=x)(x)\n\
out = zero_matrix_grad(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";

const INDIRECT_CALLABLE_PROGRAM: &str = "\
def apply(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> f32 = model(x)\n\
def apply_pipe(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> f32 = x |> model\n\
def sumsq(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))\n\
def indirect(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = {\n\
  target = fn (v: tensor[3, f32]) -> apply(model, v)\n\
  grad(target, wrt=v)(x)\n\
}\n\
def indirect_pipe(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = {\n\
  target = fn (v: tensor[3, f32]) -> apply_pipe(model, v)\n\
  grad(target, wrt=v)(x)\n\
}\n\
x = to_tensor([1.0, 2.0, 3.0])\n\
direct_out = indirect(sumsq, x)\n\
pipe_out = indirect_pipe(sumsq, x)\n";

fn eval_program(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    common::write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval must run");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 eval output")
}

#[test]
fn eval_materializes_the_exact_shaped_zero() {
    let stdout = eval_program(SHAPED_ZERO_PROGRAM, "zero_grad_eval");
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[0.0, 0.0, 0.0])"),
        "eval must realize the shaped f32 zero exactly; got {stdout:?}"
    );
}

#[test]
fn eval_materializes_the_exact_rank_zero() {
    let stdout = eval_program(RANK_ZERO_TENSOR_PROGRAM, "zero_rank_zero_eval");
    assert!(
        stdout.contains("out = 0.0"),
        "eval must realize the rank-zero f32 cotangent exactly; got {stdout:?}"
    );
}

#[test]
#[cfg(unix)]
fn generated_c_materializes_the_exact_shaped_zero() {
    let stdout = common::build_and_run(SHAPED_ZERO_PROGRAM, "zero_grad_c");
    assert_eq!(
        stdout.trim_end(),
        "out = tensor(shape=[3], data=[0.0, 0.0, 0.0])",
        "generated C must compile, link, and execute the exact shaped zero"
    );
}

#[test]
#[cfg(unix)]
fn generated_c_materializes_the_exact_rank_zero() {
    let stdout = common::build_and_run(RANK_ZERO_TENSOR_PROGRAM, "zero_rank_zero_c");
    assert_eq!(
        stdout.trim_end(),
        "seed = 7.0\nout = 0.0",
        "generated C must compile, link, and execute the exact rank-zero cotangent"
    );
}

#[test]
#[cfg(unix)]
fn symbolic_zero_gradients_preserve_concrete_extents_in_both_lanes() {
    for (name, source, expected) in [
        (
            "zero_symbolic_rank_one",
            SYMBOLIC_ZERO_PROGRAM,
            "out = tensor(shape=[4], data=[0.0, 0.0, 0.0, 0.0])",
        ),
        (
            "zero_symbolic_matrix",
            MULTIDIMENSIONAL_SYMBOLIC_ZERO_PROGRAM,
            "out = tensor(shape=[2, 3], data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])",
        ),
    ] {
        let eval = eval_program(source, &format!("{name}_eval"));
        assert_eq!(
            eval.trim_end(),
            expected,
            "eval must preserve runtime extents"
        );
        let compiled = common::build_and_run(source, &format!("{name}_c"));
        assert_eq!(
            compiled.trim_end(),
            expected,
            "generated C must preserve runtime extents"
        );
    }
}

#[test]
#[cfg(unix)]
fn generated_c_specializes_indirect_callable_dependencies() {
    let stdout = common::build_and_run(INDIRECT_CALLABLE_PROGRAM, "indirect_callable_grad");
    assert_eq!(
        stdout.trim_end(),
        "x = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n\
         direct_out = tensor(shape=[3], data=[2.0, 4.0, 6.0])\n\
         pipe_out = tensor(shape=[3], data=[2.0, 4.0, 6.0])",
        "concrete call-site specialization must recover both helper-mediated gradients"
    );
}

#[test]
fn rank_zero_and_shaped_sources_still_check_cleanly() {
    for (name, source) in [
        ("zero_grad_check", SHAPED_ZERO_PROGRAM),
        ("zero_rank_zero_check", RANK_ZERO_TENSOR_PROGRAM),
    ] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("{name}.ch"));
        common::write_file(&path, source);
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .assert()
            .success()
            .stderr(predicate::str::is_empty());
    }
}
