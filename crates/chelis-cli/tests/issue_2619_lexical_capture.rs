//! Names keep the meaning they have where they are written, on every route
//! that lowers or evaluates an expression somewhere else.
//!
//! - chelis#2619: the interpreter's `grad`/`vmap` of a function literal that
//!   calls a declaration served the caller's locals to the declaration.
//! - chelis#2380: that route lost a captured scalar.
//! - chelis#2603: a let-bound list or `shape` binding was lowered again at its
//!   use, after a rebinding of a name it reads.
//! - chelis#1949: a tensor formal named like a top-level function was taken
//!   for the function.
//! - chelis#1964: a local function named like a builtin was typed as the
//!   builtin.
//! - chelis#2547: a block-defined top-level value had no recorded type, so a
//!   named-axis call reading it saw rank 0.
//!
//! Each capture test failed at the pre-fix base `7807ca4ff`; the controls
//! rename the shadowing binding and were always right.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn eval(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("capture.ch");
    fs::write(
        &path,
        chelis_surf::format::format_source(source).expect("format"),
    )
    .expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(output.status.success(), "{output:?}\n{source}");
    String::from_utf8(output.stdout).expect("utf-8")
}

fn printed(stdout: &str, root: &str) -> String {
    let prefix = format!("{root} = ");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{root}` in {stdout}"))
        .to_string()
}

fn assert_eval(source: &str, root: &str, expected: &str) {
    assert_eq!(printed(&eval(source), root), expected, "eval\n{source}");
}

/// The value on the evaluator and on compiled, linked and executed C.
fn assert_both_lanes(source: &str, root: &str, expected: &str) {
    assert!(common::gcc_available(), "native execution is required");
    assert_eval(source, root, expected);
    assert_eq!(
        printed(&common::build_and_run(source, "capture"), root),
        expected,
        "compiled C\n{source}"
    );
}

const LOSS: &str = "w = to_tensor([7.0f32, 11.0f32])\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, w), 0i32)\n";

// chelis#2619. A `grad` or `vmap` in a top-level value's block is applied by
// the interpreter; the compiled lane has no lowering for these positions
// (chelis#2604), so they pin the evaluator.

#[test]
fn lambda_grad_target_does_not_serve_the_caller_local_to_a_declaration() {
    let source = format!(
        "{LOSS}out = {{\n  w = to_tensor([100.0f32, 100.0f32])\n  grad(fn (x: tensor[2, f32]) -> loss(x))(to_tensor([1.0f32, 2.0f32]))\n}}\n"
    );
    assert_eval(&source, "out", "tensor(shape=[2], data=[7.0, 11.0])");
}

#[test]
fn let_bound_lambda_grad_target_does_not_serve_the_caller_local_to_a_declaration() {
    let source = format!(
        "{LOSS}out = {{\n  w = to_tensor([100.0f32, 100.0f32])\n  g = fn (x: tensor[2, f32]) -> loss(x)\n  grad(g)(to_tensor([1.0f32, 2.0f32]))\n}}\n"
    );
    assert_eval(&source, "out", "tensor(shape=[2], data=[7.0, 11.0])");
}

#[test]
fn lambda_vmap_target_does_not_serve_the_caller_local_to_a_declaration() {
    let source = "w = to_tensor([7.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, w)\n\
out = {\n  w = to_tensor([100.0f32])\n  vmap(fn (x: tensor[1, f32]) -> f(x))(to_tensor([[1.0f32], [2.0f32]]))\n}\n";
    assert_eval(source, "out", "tensor(shape=[2, 1], data=[7.0, 14.0])");
}

#[test]
fn renamed_caller_local_is_the_twin() {
    let source = format!(
        "{LOSS}out = {{\n  q = to_tensor([100.0f32, 100.0f32])\n  grad(fn (x: tensor[2, f32]) -> loss(x))(add(to_tensor([1.0f32, 2.0f32]), sub(q, q)))\n}}\n"
    );
    assert_both_lanes(&source, "out", "tensor(shape=[2], data=[7.0, 11.0])");
}

#[test]
fn a_caller_closure_does_not_replace_a_function_a_declaration_calls() {
    let source = "def scale(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, to_tensor([7.0f32, 11.0f32]))\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(scale(x), 0i32)\n\
out = {\n  scale = fn (x: tensor[2, f32]) -> mul(x, to_tensor([100.0f32, 100.0f32]))\n  grad(fn (x: tensor[2, f32]) -> loss(x))(to_tensor([1.0f32, 2.0f32]))\n}\n";
    assert_both_lanes(source, "out", "tensor(shape=[2], data=[7.0, 11.0])");
}

#[test]
fn a_closure_reads_what_it_closed_over_not_a_later_caller_binding() {
    let source = "def scale(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, to_tensor([7.0f32, 11.0f32]))\n\
out = {\n  g = fn (x: tensor[2, f32]) -> sum(scale(x), 0i32)\n  scale = fn (x: tensor[2, f32]) -> mul(x, to_tensor([100.0f32, 100.0f32]))\n  grad(g)(to_tensor([1.0f32, 2.0f32]))\n}\n";
    assert_both_lanes(source, "out", "tensor(shape=[2], data=[7.0, 11.0])");
}

/// chelis#2380: the target captures a function-valued parameter and two
/// scalars; d/dtheta of y - (sum(theta) + x) is -1 per element.
#[test]
fn a_local_grad_wrapper_keeps_its_captured_scalars() {
    let source = "def jac_row[n](model: tensor[n, f32] -> f32 -> f32 -> f32, theta: tensor[n, f32], x: f32, y: f32) -> tensor[n, f32] = {\n\
  target = fn (theta_local: tensor[n, f32]) -> model(theta_local, x, y)\n\
  grad(target, wrt=theta_local)(theta)\n}\n\
def lm_model(theta: tensor[2, f32], x: f32, y: f32) -> f32 = {\n\
  y_hat = if lt(x, cast(0.0, f32)) then tensor_to_scalar(sum(copy(theta), 0)) else add(tensor_to_scalar(sum(copy(theta), 0)), x)\n\
  sub(y, y_hat)\n}\n\
out = jac_row(lm_model, to_tensor([1.0, 2.0]), cast(1.0, f32), cast(3.0, f32))\n";
    assert_eval(source, "out", "tensor(shape=[2], data=[-1.0, -1.0])");
}

// chelis#2603.

#[test]
fn a_let_bound_list_keeps_the_values_it_was_built_from() {
    let source = "def main() -> tensor[2, f32] = {\n  y = to_tensor([1.0f32])\n  rows = [y, y]\n  y = to_tensor([5.0f32])\n  concat(rows, 0i32)\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[2], data=[1.0, 1.0])");
}

#[test]
fn a_let_bound_shape_keeps_the_operand_it_was_read_from() {
    let source = "def main() -> tensor[*, f32] = {\n  x = to_tensor([1.0f32, 2.0f32, 3.0f32])\n  n = shape(&x, 0)\n  x = to_tensor([1.0f32, 2.0f32])\n  insert(scalar_to_tensor(7.0f32), 0, cast(n, i64))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[3], data=[7.0, 7.0, 7.0])");
}

#[test]
fn a_let_bound_list_in_a_grad_body_keeps_its_values() {
    let source = "def main() -> tensor[1, f32] =\n  grad(fn (v: tensor[1, f32]) -> {\n    rows = [copy(v), copy(v)]\n    v = mul(v, to_tensor([10.0f32]))\n    sum(concat(rows, 0i32), 0i32)\n  })(to_tensor([1.0f32]))\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[2.0])");
}

// chelis#1949.

#[test]
fn a_tensor_formal_shadows_a_same_named_top_level_function() {
    let source = "def scale(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)\n\
def relay(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\n\
def wrapper(scale: tensor[2, f32]) -> tensor[2, f32] = relay(scale)\n\
out = sum(wrapper(to_tensor([1.0f32, 2.0f32])), 0i32)\n";
    assert_both_lanes(source, "out", "6.0");
}

// chelis#1964.

#[test]
fn a_local_function_shadows_a_same_named_builtin() {
    let source = "def const_col[n](spots: tensor[n, f32]) -> tensor[n, 1, f64] = {\n\
  nn = len(to_list(copy(spots)))\n\
  to_tensor = fn (x: tensor[n, f64]) -> x\n\
  reshape(to_tensor(cast(spots, f64)), [nn, cast(1, i64)])\n}\n\
def total[n](spots: tensor[n, f32]) -> tensor[f64] = {\n  kc = const_col(spots)\n  sum(sum(kc, 0i32), 0i32)\n}\n\
out = total(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_both_lanes(source, "out", "6.0");
}

// chelis#2547.

#[test]
fn a_named_axis_call_reads_a_block_defined_top_level_value_at_its_rank() {
    let source = "weights = {\n  w = to_tensor([3.0f32, 5.0f32])\n  w\n}\n\
def total(x: &tensor[seq, f32]) -> (tensor[f32], tensor[f32]) = (sum(x, seq), sum(weights, 0i32))\n\
def main() = total(to_tensor([7.0f32, 11.0f32]))\n";
    let stdout = eval(source);
    assert_eq!(printed(&stdout, "main.0"), "18.0", "{stdout}");
    assert_eq!(printed(&stdout, "main.1"), "8.0", "{stdout}");
    assert!(common::gcc_available(), "native execution is required");
    let native = common::build_and_run(source, "capture");
    assert_eq!(printed(&native, "main.0"), "18.0", "{native}");
    assert_eq!(printed(&native, "main.1"), "8.0", "{native}");
}

/// A local transform value is itself staged, never left to be read as a
/// same-named top-level function.
#[test]
fn a_local_transform_value_is_not_read_as_a_same_named_top_level_function() {
    let source = "def loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
def g(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, to_tensor([100.0f32, 100.0f32]))\n\
out = {\n  g = grad(loss)\n  vmap(fn (x: tensor[2, f32]) -> g(x))(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n}\n";
    assert_eval(
        source,
        "out",
        "tensor(shape=[2, 2], data=[2.0, 4.0, 6.0, 8.0])",
    );
}

/// A local with no lowered form that the target passes along but never uses
/// stays harmless.
#[test]
fn an_unused_string_local_the_target_reads_is_harmless() {
    let source = "def loss(x: tensor[2, f32], label: string) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
out = {\n  s = \"run-1\"\n  grad(fn (x: tensor[2, f32]) -> loss(x, s))(to_tensor([1.0f32, 2.0f32]))\n}\n";
    assert_eval(source, "out", "tensor(shape=[2], data=[2.0, 4.0])");
}

/// A recorded list that calls a top-level function keeps calling it after a
/// later local takes the function's name.
#[test]
fn a_let_bound_list_keeps_calling_the_top_level_function_it_named() {
    let source = "def scale(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, to_tensor([3.0f32]))\n\
def main() -> tensor[2, f32] = {\n  y = to_tensor([1.0f32])\n  rows = [scale(y), y]\n  scale = to_tensor([100.0f32])\n  concat(rows, 0i32)\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[2], data=[3.0, 1.0])");
}

/// The same for a recorded `shape` of a call.
#[test]
fn a_let_bound_shape_keeps_calling_the_top_level_function_it_named() {
    let source = "def twice(x: tensor[2, f32]) -> tensor[4, f32] = concat([x, x], 0i32)\n\
def main() -> tensor[2, 4, f32] = {\n  y = to_tensor([1.0f32, 2.0f32])\n  n = shape(twice(y), 0i32)\n  twice = to_tensor([100.0f32])\n  reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32]), [cast(2, i64), n])\n}\n";
    assert_both_lanes(
        source,
        "main",
        "tensor(shape=[2, 4], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])",
    );
}
