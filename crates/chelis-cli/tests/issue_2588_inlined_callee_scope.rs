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

/// A formal and a callee's top-level read may have the same source spelling.
/// The two values have distinct lexical origins through AD and native code.
#[test]
fn grad_formal_does_not_replace_a_callee_global_with_the_same_name() {
    let source = "y = to_tensor([2.0f32, 3.0f32])\n\
def h(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, y)\n\
def loss(y: tensor[2, f32]) -> tensor[f32] = sum(h(y), 0i32)\n\
def main() -> tensor[2, f32] = grad(loss)(to_tensor([1.0f32, 1.0f32]))\n";
    assert_both_lanes(source, "main", "tensor(shape=[2], data=[2.0, 3.0])");
}

#[test]
fn ordinary_formal_does_not_replace_a_callee_global_with_the_same_name() {
    let source = "y = to_tensor([2.0f32, 3.0f32])\n\
def h(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, y)\n\
def loss(y: tensor[2, f32]) -> tensor[f32] = sum(h(y), 0i32)\n\
out = loss(to_tensor([1.0f32, 1.0f32]))\n";
    assert_both_lanes(source, "out", "5.0");
}

#[test]
fn transform_in_global_block_reads_global_beside_a_same_named_local() {
    let source = "w = to_tensor([3.0f32, 5.0f32])\n\
def inner(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, w)\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(inner(x), 0i32)\n\
out = {\n  w = to_tensor([7.0f32, 11.0f32])\n  grad(loss)(to_tensor([1.0f32, 2.0f32]))\n}\n";
    assert_both_lanes(source, "out", "tensor(shape=[2], data=[3.0, 5.0])");
}

/// A local in an argument's function literal is not in scope at the outer
/// substitution site. Host scope must carry binding origin explicitly.
#[test]
fn unrelated_literal_local_does_not_block_outer_callee_substitution() {
    let source = "n = to_tensor([2.0f32])\n\
def apply(h: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = h(mul(x, n))\n\
def outer(k: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = apply(k, x)\n\
out = outer(fn (v: tensor[1, f32]) -> {\n  n = to_tensor([10.0f32])\n  add(v, n)\n}, to_tensor([1.0f32]))\n";
    assert_both_lanes(source, "out", "tensor(shape=[1], data=[12.0])");
}

/// Callable target identity has a separate residual (#2062). The evaluator
/// resolves this local function name; the C lane still rejects it explicitly.
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
    assert_both_lanes(
        "y = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, y)\n\
out = {\n  y = to_tensor([10.0f32])\n  vmap(f)(to_tensor([[1.0f32], [3.0f32]]))\n}\n",
        "out",
        "tensor(shape=[2, 1], data=[3.0, 5.0])",
    );
}

#[test]
fn transform_route_grad_through_a_callee_is_not_captured() {
    assert_both_lanes(
        "w = to_tensor([3.0f32, 5.0f32])\n\
def inner(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, w)\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(inner(x), 0i32)\n\
out = {\n  w = to_tensor([7.0f32, 11.0f32])\n  grad(loss)(to_tensor([1.0f32, 2.0f32]))\n}\n",
        "out",
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

/// A top-level function name in a global block's scope is that function: a
/// higher-order callee substituted one level down still reaches `apply`.
#[test]
fn a_global_block_nested_substitution_keeps_its_callees() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}def outer(k: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = apply(k, x)\n\
out = outer(dbl, to_tensor([1.0f32]))\n"
    );
    assert_both_lanes(&source, "out", "tensor(shape=[1], data=[4.0])");
}

/// A match arm that binds another top-level name leaves the rest of the
/// block's names unambiguous.
#[test]
fn a_global_block_match_binder_on_another_name_does_not_stop_substitution() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}x = to_tensor([7.0f32])\n\
def pick(v: tensor[1, f32]) -> Option[tensor[1, f32]] = Some(v)\n\
out = {{\n  a = apply(dbl, to_tensor([1.0f32]))\n  b = match pick(to_tensor([10.0f32])) with {{\n    | Some(x) => x\n    | None => to_tensor([0.0f32])\n  }}\n  add(a, b)\n}}\n"
    );
    assert_both_lanes(&source, "out", "tensor(shape=[1], data=[14.0])");
}

/// A local ascription on a block local spelled like a top-level value keeps
/// its runtime claim on the compiled lane.
#[test]
fn a_global_block_local_ascription_spelled_like_a_top_level_value_is_still_checked() {
    let source = "y = to_tensor([5.0f32])\n\
def g(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
out = {\n  v = g(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n  y: tensor[2, f32] = pad(v, [[0i64, 0i64]], 0.0f32)\n  y\n}\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("claim.ch");
    fs::write(&path, source).expect("source");
    let out = dir.path().join("out");
    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out)
        .output()
        .expect("build");
    assert!(build.status.success(), "{build:?}");
    assert!(
        common::link_generated(&out, "claim.c", "claim").success(),
        "link failed"
    );
    let run = std::process::Command::new(out.join("claim"))
        .output()
        .expect("run");
    assert!(
        !run.status.success(),
        "the claimed extent 2 must trap: {run:?}"
    );
}

/// Inside a substituted body, the block's binders that count are those in
/// scope where the substitution happened: a later local does not.
#[test]
fn a_global_block_nested_substitution_ignores_a_later_local() {
    let source = format!(
        "{HIGHER_ORDER_CALLEE}def outer(k: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = apply(k, x)\n\
out = {{\n  a = outer(dbl, to_tensor([1.0f32]))\n  n = to_tensor([10.0f32])\n  add(a, n)\n}}\n"
    );
    assert_both_lanes(&source, "out", "tensor(shape=[1], data=[14.0])");
}

/// A function literal passed as an actual is substituted with the callee's
/// body, bringing its own locals: a local it binds must not capture a name a
/// callee it calls reads.
#[test]
fn a_function_literal_actual_local_does_not_capture_inside_a_substitution() {
    let source = "c = to_tensor([2.0f32])\n\
def f(x: tensor[1, f32]) -> tensor[1, f32] = add(x, c)\n\
def apply(h: (tensor[1, f32]) -> tensor[1, f32], x: tensor[1, f32]) -> tensor[1, f32] = h(x)\n\
out = apply(fn (v: tensor[1, f32]) -> {\n  c = to_tensor([10.0f32])\n  add(f(v), c)\n}, to_tensor([1.0f32]))\n";
    assert_both_lanes(source, "out", "tensor(shape=[1], data=[13.0])");
}

/// A function literal applied inside a `grad` body keeps the scope it was
/// written in, even when the callee it is passed to binds the same name.
#[test]
fn a_function_literal_in_a_grad_body_keeps_its_scope_inside_the_callee() {
    let source = "def apply(h: (tensor[1, f32]) -> tensor[1, f32], v: tensor[1, f32]) -> tensor[1, f32] = {\n  a = to_tensor([100.0f32])\n  add(h(v), a)\n}\n\
def main() -> tensor[1, f32] =\n  grad(fn (x: tensor[1, f32]) -> {\n    a = mul(x, x)\n    sum(apply(fn (v: tensor[1, f32]) -> mul(v, a), x), 0i32)\n  })(to_tensor([5.0f32]))\n";
    assert_both_lanes(source, "main", "tensor(shape=[1], data=[75.0])");
}

/// The same through `vmap` inside the `grad` body. The evaluator has no
/// lowering for this `vmap`, so only the compiled value is pinned.
#[test]
fn a_function_literal_vmapped_in_a_grad_body_keeps_its_scope_inside_the_callee() {
    let source = "def apply(h: (tensor[1, f32]) -> tensor[1, f32], v: tensor[2, 1, f32]) -> tensor[2, 1, f32] = {\n  a = to_tensor([100.0f32])\n  vmap(h)(v)\n}\n\
def main() -> tensor[1, f32] =\n  grad(fn (x: tensor[1, f32]) -> {\n    a = mul(x, x)\n    sum(sum(apply(fn (v: tensor[1, f32]) -> mul(v, a), insert(x, 0i32, 2i64)), 0i32), 0i32)\n  })(to_tensor([5.0f32]))\n";
    assert!(common::gcc_available(), "native execution is required");
    assert_eq!(
        printed(&common::build_and_run(source, "scope"), "main"),
        "tensor(shape=[1], data=[150.0])"
    );
}
