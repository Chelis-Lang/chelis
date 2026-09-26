//! chelis#2588: an inlined function body reads its free names in the scope it
//! was written in, never in the caller's.
//!
//! Every tensor call is inlined when it is lowered, on both the evaluator's
//! kernels and the compiled C lane. The lowerer used to lower the callee's
//! body in the caller's lexical tables, so a caller local, parameter or local
//! function alias spelled like a top-level name the callee reads replaced it.
//! Each capture case below returned the caller's value on both lanes before
//! the fix; its twin renames the caller's binding and was always right.
//!
//! Lexical scoping fixes every expected value: the callee's free names are the
//! top-level declarations, and the caller's locals reach it only as
//! arguments. Positive controls keep that second half honest: a local passed
//! as an argument, and a function literal that closes over a local.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn eval(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scope.ch");
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

/// The value on the evaluator and on compiled, linked and executed C.
fn assert_both_lanes(source: &str, root: &str, expected: &str) {
    assert!(common::gcc_available(), "native execution is required");
    assert_eq!(printed(&eval(source), root), expected, "eval\n{source}");
    assert_eq!(
        printed(&common::build_and_run(source, "scope"), root),
        expected,
        "compiled C\n{source}"
    );
}

/// The issue's witness at rank 1, the rank the C lane can compile today.
const VALUE_WITNESS: &str = "y = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, y)\n\
def main() -> tensor[1, f32] = {\n  y = to_tensor([10.0f32])\n  f(add(to_tensor([1.0f32]), y))\n}\n";

#[test]
fn caller_local_does_not_capture_a_callee_top_level_value() {
    assert_both_lanes(VALUE_WITNESS, "main", "tensor(shape=[1], data=[13.0])");
}

#[test]
fn renamed_caller_local_is_the_twin() {
    let source = VALUE_WITNESS
        .replace("  y = to_tensor([10.0f32])", "  w = to_tensor([10.0f32])")
        .replace("1.0f32]), y))", "1.0f32]), w))");
    assert_both_lanes(&source, "main", "tensor(shape=[1], data=[13.0])");
}

/// The issue's own rank-0 witness. Its C build fails to compile with or
/// without the shadowing local, a separate defect, so this pins the evaluator.
#[test]
fn issue_witness_evaluates_lexically() {
    let source = "y = scalar_to_tensor(2.0f32)\n\
def f(x: tensor[f32]) -> tensor[f32] = add(x, copy(y))\n\
def main() -> tensor[f32] = {\n  y = scalar_to_tensor(10.0f32)\n  f(add(scalar_to_tensor(1.0f32), y))\n}\n";
    assert_eq!(printed(&eval(source), "main"), "13.0");
}

#[test]
fn nested_inlining_keeps_the_innermost_callee_scope() {
    let source = "y = to_tensor([2.0f32])\n\
def h(x: tensor[1, f32]) -> tensor[1, f32] = add(x, y)\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = h(x)\n\
def main() -> tensor[1, f32] = {\n  y = to_tensor([10.0f32])\n  f(add(to_tensor([1.0f32]), y))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[13.0])");
}

#[test]
fn a_top_level_value_read_through_another_declaration_is_not_captured() {
    let source = "y = to_tensor([2.0f32])\nz = add(y, to_tensor([1.0f32]))\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, z)\n\
def main() -> tensor[1, f32] = {\n  y = to_tensor([10.0f32])\n  z = to_tensor([100.0f32])\n  f(add(to_tensor([1.0f32]), y))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[14.0])");
}

#[test]
fn a_local_function_alias_does_not_capture_a_callee_top_level_function() {
    let source = "def g(x: tensor[1, f32]) -> tensor[1, f32] = add(x, to_tensor([2.0f32]))\n\
def h(x: tensor[1, f32]) -> tensor[1, f32] = add(x, to_tensor([10.0f32]))\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = g(x)\n\
def main() -> tensor[1, f32] = {\n  g = h\n  add(f(to_tensor([1.0f32])), g(to_tensor([0.0f32])))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[13.0])");
}

#[test]
fn a_caller_parameter_does_not_capture_a_callee_top_level_value() {
    let source = "y = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, y)\n\
def main(y: tensor[1, f32]) -> tensor[1, f32] = f(add(to_tensor([1.0f32]), y))\n\
out = main(to_tensor([10.0f32]))\n";
    assert_both_lanes(source, "out", "tensor(shape=[1], data=[13.0])");
}

#[test]
fn an_inlined_intermediate_local_does_not_capture_a_callee_top_level_value() {
    let source = "y = to_tensor([2.0f32])\n\
def apply(h: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = {\n  y = to_tensor([10.0f32])\n  h(add(x, y))\n}\n\
def f(v: tensor[1, f32]) -> tensor[1, f32] = add(v, y)\n\
def main() -> tensor[1, f32] = apply(f, to_tensor([1.0f32]))\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[13.0])");
}

#[test]
fn vmap_of_a_declaration_inside_a_def_is_not_captured() {
    let source = "y = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, y)\n\
def main() -> tensor[2, 1, f32] = {\n  y = to_tensor([10.0f32])\n  vmap(f)(to_tensor([[1.0f32], [3.0f32]]))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[2, 1], data=[3.0, 5.0])");
}

#[test]
fn grad_of_a_declaration_inside_a_def_is_not_captured() {
    let source = "y = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[f32] = sum(mul(x, y), 0i32)\n\
def main() -> tensor[1, f32] = {\n  y = to_tensor([10.0f32])\n  grad(f)(add(to_tensor([1.0f32]), y))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[2.0])");
}

/// A transform in a top-level value's block runs through the host
/// interpreter's transform route. The compiled lane has no lowering for these
/// positions, and now says so instead of building the caller's value.
fn assert_evaluator_only(source: &str, expected: &str) {
    assert_eq!(printed(&eval(source), "out"), expected, "eval\n{source}");
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scope.ch");
    fs::write(&path, source).expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .expect("build");
    assert!(!output.status.success(), "{output:?}\n{source}");
}

#[test]
fn transform_route_vmap_of_a_declaration_is_not_captured() {
    assert_evaluator_only(
        "y = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, y)\n\
out = {\n  y = to_tensor([10.0f32])\n  vmap(f)(to_tensor([[1.0f32], [3.0f32]]))\n}\n",
        "tensor(shape=[2, 1], data=[3.0, 5.0])",
    );
}

#[test]
fn transform_route_grad_through_a_callee_is_not_captured() {
    assert_evaluator_only(
        "w = to_tensor([3.0f32, 5.0f32])\n\
def inner(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, w)\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(inner(x), 0i32)\n\
out = {\n  w = to_tensor([7.0f32, 11.0f32])\n  grad(loss)(to_tensor([1.0f32, 2.0f32]))\n}\n",
        "tensor(shape=[2], data=[3.0, 5.0])",
    );
}

#[test]
fn transform_route_local_function_does_not_replace_a_callee_function() {
    assert_evaluator_only(
        "def scale(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, to_tensor([3.0f32, 5.0f32]))\n\
def triple(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, to_tensor([7.0f32, 11.0f32]))\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(scale(x), 0i32)\n\
out = {\n  scale = triple\n  grad(loss)(to_tensor([1.0f32, 2.0f32]))\n}\n",
        "tensor(shape=[2], data=[3.0, 5.0])",
    );
}

#[test]
fn a_function_literal_still_closes_over_the_local_it_names() {
    let source = "a = to_tensor([3.0f32])\n\
out = {\n  a = to_tensor([7.0f32])\n  f = fn (x: tensor[1, f32]) -> add(x, a)\n  f(to_tensor([1.0f32]))\n}\n";
    assert_both_lanes(source, "out", "tensor(shape=[1], data=[8.0])");
}

#[test]
fn a_function_literal_argument_keeps_its_own_scope_inside_the_callee() {
    let source = "def apply(h: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = {\n  y = to_tensor([10.0f32])\n  h(add(x, y))\n}\n\
def main() -> tensor[1, f32] = {\n  y = to_tensor([2.0f32])\n  apply(fn (v: tensor[1, f32]) -> add(v, y), to_tensor([1.0f32]))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[13.0])");
}

#[test]
fn a_function_literal_keeps_the_binding_it_closed_over_after_a_rebinding() {
    let source = "def main() -> tensor[1, f32] = {\n  y = to_tensor([2.0f32])\n  g = fn (x: tensor[1, f32]) -> add(x, y)\n  y = to_tensor([10.0f32])\n  g(add(to_tensor([1.0f32]), y))\n}\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[13.0])");
}

/// A binder that is not in scope at the call site cannot capture there: a
/// later local and a function literal's parameter each spell the callee's
/// top-level `n`, and the host lane must still substitute the callee.
const HIGHER_ORDER_CALLEE: &str = "n = to_tensor([2.0f32])\n\
def apply(h: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = h(mul(x, n))\n\
def dbl(v: tensor[1, f32]) -> tensor[1, f32] = add(v, v)\n";

#[test]
fn a_later_local_does_not_stop_a_callee_from_being_substituted() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}def main() -> tensor[1, f32] = {{\n  a = apply(dbl, to_tensor([1.0f32]))\n  n = to_tensor([10.0f32])\n  add(a, n)\n}}\n"
    );
    assert_both_lanes(&source, "main", "tensor(shape=[1], data=[14.0])");
}

#[test]
fn a_function_literal_parameter_does_not_stop_a_callee_from_being_substituted() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}def main() -> tensor[1, f32] = {{\n  a = apply(dbl, to_tensor([1.0f32]))\n  apply(fn (n: tensor[1, f32]) -> add(n, a), to_tensor([1.0f32]))\n}}\n"
    );
    assert_both_lanes(&source, "main", "tensor(shape=[1], data=[6.0])");
}

/// In a top-level value's block the host scope also carries the globals, so
/// whether a name there is the global depends on the binders in scope at the
/// call site, not on the block's binders as a whole.
#[test]
fn a_global_block_local_before_the_call_does_not_capture() {
    let source = "n = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, n)\n\
out = {\n  n = to_tensor([10.0f32])\n  add(f(to_tensor([1.0f32])), n)\n}\n";
    assert_both_lanes(source, "out", "tensor(shape=[1], data=[12.0])");
}

#[test]
fn a_global_block_local_after_the_call_does_not_stop_substitution() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}out = {{\n  a = apply(dbl, to_tensor([1.0f32]))\n  n = to_tensor([10.0f32])\n  add(a, n)\n}}\n"
    );
    assert_both_lanes(&source, "out", "tensor(shape=[1], data=[14.0])");
}

#[test]
fn a_global_block_function_literal_parameter_does_not_stop_substitution() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}out = {{\n  a = apply(dbl, to_tensor([1.0f32]))\n  apply(fn (n: tensor[1, f32]) -> add(n, a), to_tensor([1.0f32]))\n}}\n"
    );
    assert_both_lanes(&source, "out", "tensor(shape=[1], data=[6.0])");
}

#[test]
fn a_global_block_local_after_a_grad_does_not_stop_its_kernel() {
    let source = "c = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[f32] = sum(mul(x, c), 0i32)\n\
out = {\n  a = grad(f)(to_tensor([1.0f32]))\n  c = to_tensor([10.0f32])\n  add(a, c)\n}\n";
    assert_both_lanes(source, "out", "tensor(shape=[1], data=[12.0])");
}
