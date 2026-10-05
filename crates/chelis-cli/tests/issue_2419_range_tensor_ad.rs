//! [05-OP-54/55/57], spec/06 §2.7.1: runtime integer builders retain their
//! source extent and zero cotangent through isolated and composed Grad.
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;

fn assert_both(source: &str, expected: &[f64], stem: &str) {
    let source = chelis_surf::format::format_source(source).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    common::write_file(&path, &source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(eval.status.success(), "{eval:?}");
    for output in [
        String::from_utf8(eval.stdout).unwrap(),
        common::build_and_run(&source, stem),
    ] {
        let line = output
            .lines()
            .find(|line| line.starts_with("out = tensor("))
            .unwrap();
        if expected.is_empty() {
            assert!(line.contains("data=[]"), "{line}");
        } else {
            assert_eq!(common::parse_tensor_data(&output, "out"), expected);
        }
        assert!(
            line.contains(&format!("shape=[{}]", expected.len())),
            "{line}"
        );
    }
}

const BASIS: &str = r#"
def basis[n](k: i64, s: f32, template: &tensor[n, f32]) -> tensor[n, f32] = {
  items_len = len(to_list(template))
  idxs = range(0i64, items_len)
  to_tensor(map(fn (i: i64) -> if eq(i, k) then s else 0.0f32, idxs))
}
"#;

#[test]
fn range_map_basis_has_the_exact_isolated_gradient() {
    let source = format!(
        "{BASIS}def loss(theta: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, basis(0i64, 1.0f32, theta)), 0i32))\nout = grad(loss)(to_tensor([1.0f32, 2.0f32]))\n"
    );
    assert_both(&source, &[1.0, 0.0], "basis_gradient");
}

#[test]
fn literal_basis_remains_the_exact_gradient_control() {
    assert_both(
        "def loss(theta: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, to_tensor([1.0f32, 0.0f32])), 0i32))\nout = grad(loss)(to_tensor([1.0f32, 2.0f32]))\n",
        &[1.0, 0.0],
        "literal_basis",
    );
}

#[test]
fn composed_generic_basis_has_the_exact_jacobian_row() {
    assert_both(
        r#"
def basis[n](k: i64, template: &tensor[n, f32]) -> tensor[n, f32] = {
  count = len(to_list(template))
  to_tensor(map(fn (i: i64) -> if eq(i, k) then 1.0f32 else 0.0f32, range(0i64, count)))
}
def model[n, m](theta: &tensor[n, f32], x: &tensor[m, f32]) -> tensor[m, f32] = {
  coefficient = tensor_to_scalar(sum(mul(theta, basis(0i64, theta)), 0i32))
  add(insert(scalar_to_tensor(coefficient), 0i32, shape(x, 0i32)), insert(sum(theta, 0i32), 0i32, shape(x, 0i32)))
}
def jacobian_row[n, m](f: &tensor[n, f32] -> &tensor[m, f32] -> tensor[m, f32], x: &tensor[m, f32], theta: tensor[n, f32], seed: tensor[m, f32]) -> tensor[n, f32] = {
  target = fn (theta_local: tensor[n, f32], x_local: tensor[m, f32], seed_local: tensor[m, f32]) -> {
    prediction = f(theta_local, x_local)
    tensor_to_scalar(sum(mul(prediction, seed_local), 0i32))
  }
  grad(target, wrt=theta_local)(theta, copy(x), seed)
}
def step[n, m](theta: tensor[n, f32], x: tensor[m, f32], seed: tensor[m, f32]) -> tensor[n, f32] = jacobian_row(model, x, theta, seed)
out = step(to_tensor([1.0f32, 2.0f32]), to_tensor([0.0f32, 1.0f32, 2.0f32]), to_tensor([1.0f32, 0.0f32, 0.0f32]))
"#,
        &[2.0, 1.0],
        "composed_basis",
    );
}

#[test]
fn runtime_builder_captures_receive_their_exact_cotangent() {
    let source = format!(
        "{BASIS}def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, basis(0i64, tensor_to_scalar(sum(scale, 0i32)), theta)), 0i32))\nout = grad(loss, wrt=scale)(to_tensor([3.0f32, 5.0f32, 7.0f32, 9.0f32]), to_tensor([2.0f32]))\n"
    );
    assert_both(&source, &[3.0], "capture_cotangent");
}

// spec/06 §2.4: List callbacks expose invocation-ordered consumer edges,
// including repeated input slots, to the shared capture's single +0 tree.
#[test]
fn runtime_capture_accumulation_preserves_the_executed_consumer_tree() {
    for (callback, coefficients, expected, stem) in [
        (
            "mul(s, 1.0f32)",
            "100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32",
            0.0,
            "capture_cancel",
        ),
        (
            "s",
            "100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32",
            0.0,
            "capture_identity",
        ),
        (
            "add(s, s)",
            "100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32",
            0.0,
            "capture_repeated_slot",
        ),
        (
            "mul(s, 1.0f32)",
            "100000000000000000000.0f32, 3.0f32, -100000000000000000000.0f32",
            0.0,
            "capture_permuted",
        ),
        ("add(s, s)", "2.0f32, 3.0f32, 5.0f32", 20.0, "capture_small"),
    ] {
        let source = format!(
            r#"
def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = {{
  s = tensor_to_scalar(sum(scale, 0i32))
  weights = to_tensor(map(fn (i: i64) -> {callback}, range(0i64, shape(theta, 0i32))))
  tensor_to_scalar(sum(mul(theta, weights), 0i32))
}}
out = grad(loss, wrt=scale)(to_tensor([{coefficients}]), to_tensor([2.0f32]))
"#
        );
        assert_both(&source, &[expected], stem);
    }
}

#[test]
fn runtime_capture_tree_interleaves_the_map_with_outside_consumers() {
    assert_both(
        r#"
def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = {
  s = tensor_to_scalar(sum(scale, 0i32))
  before = mul(s, 3.0f32)
  weights = to_tensor(map(fn (i: i64) -> mul(s, 1.0f32), range(0i64, shape(theta, 0i32))))
  after = mul(s, 0.0f32)
  add(add(before, tensor_to_scalar(sum(mul(theta, weights), 0i32))), after)
}
out = grad(loss, wrt=scale)(to_tensor([100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32]), to_tensor([2.0f32]))
"#,
        &[6.0],
        "capture_outside",
    );
}

#[test]
fn runtime_capture_tree_rounds_each_pair_at_the_capture_dtype() {
    assert_both(
        r#"
def loss[n](theta: tensor[n, f16], scale: tensor[1, f16]) -> f16 = {
  s = tensor_to_scalar(cast(sum(scale, 0i32), f16))
  weights = to_tensor(map(fn (i: i64) -> mul(s, 1.0f16), range(0i64, shape(theta, 0i32))))
  tensor_to_scalar(sum(mul(theta, weights), 0i32))
}
out = grad(loss, wrt=scale)(to_tensor([2048.0f16, -2048.0f16, 0.5f16]), to_tensor([2.0f16]))
"#,
        &[0.0],
        "capture_half_width",
    );
}

#[test]
fn runtime_capture_tree_preserves_inactive_and_empty_rows() {
    for (values, expected, stem) in [
        (
            "to_tensor([3.0f32, 5.0f32, 7.0f32])",
            10.0,
            "capture_inactive",
        ),
        (
            "insert(scalar_to_tensor(0.0f32), 0i32, 0i64)",
            0.0,
            "capture_empty",
        ),
    ] {
        let source = format!(
            r#"
def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = {{
  s = tensor_to_scalar(sum(scale, 0i32))
  weights = to_tensor(map(fn (i: i64) -> if eq(i, 1i64) then 0.0f32 else mul(s, 1.0f32), range(0i64, shape(theta, 0i32))))
  tensor_to_scalar(sum(mul(theta, weights), 0i32))
}}
out = grad(loss, wrt=scale)({values}, to_tensor([2.0f32]))
"#
        );
        assert_both(&source, &[expected], stem);
    }
}
#[test]
fn runtime_builder_handles_a_different_extent_and_nonzero_selected_index() {
    let source = format!(
        "{BASIS}def loss[n](theta: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, basis(2i64, 4.0f32, theta)), 0i32))\nout = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n"
    );
    assert_both(&source, &[0.0, 0.0, 4.0, 0.0], "runtime_extent");
}

#[test]
fn callbacks_with_observable_checks_require_ordered_execution() {
    let source = chelis_surf::format::format_source(r#"
def loss[n](theta: tensor[n, f32]) -> f32 = {
    values = to_tensor(map(fn (i: i64) -> cast(trunc_div(1i64, sub(i, 1i64)), f32), range(0i64, shape(theta, 0i32))))
    tensor_to_scalar(sum(mul(theta, values), 0i32))
}
out = grad(loss)(to_tensor([1.0f32, 2.0f32]))
"#).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("ordered.ch");
    common::write_file(&path, &source);
    for lane in ["eval", "build"] {
        let mut command = Command::cargo_bin("chelis").unwrap();
        command.env("CHELIS_STYLE_GATE_DISABLE", "1");
        if lane == "eval" {
            command.args(["eval", "--file"]).arg(&path);
        } else {
            command
                .arg("build")
                .arg("--emit-c")
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(dir.path().join("out"));
        }
        let output = command.output().unwrap();
        assert!(!output.status.success(), "{output:?}");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            text.contains("runtime map callback requires ordered execution"),
            "{text}"
        );
        assert!(!text.contains("missing required input"), "{text}");
    }
}

#[test]
fn represented_list_aliases_and_nested_maps_keep_their_source() {
    assert_both(
        r#"
def loss[n](theta: tensor[n, f32]) -> f32 = {
    indices = range(0i64, len(to_list(theta)))
    weights = map(fn (i: i64) -> if eq(i, 1i64) then 2.0f32 else 0.0f32, indices)
    alias = weights
    scaled = map(fn (value: f32) -> mul(value, 3.0f32), alias)
    tensor_to_scalar(sum(mul(theta, to_tensor(scaled)), 0i32))
}
out = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#,
        &[0.0, 6.0, 0.0],
        "represented_alias",
    );
}

#[test]
fn independent_builder_invocations_keep_distinct_runtime_extents() {
    let source = format!(
        "{BASIS}def loss[n](theta: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, basis(0i64, 1.0f32, theta)), 0i32))\na = grad(loss)(to_tensor([1.0f32, 2.0f32]))\nb = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\nout = concat([a, b], 0i32)\n"
    );
    assert_both(
        &source,
        &[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "independent_extents",
    );
}

#[test]
fn empty_runtime_basis_has_an_empty_cotangent() {
    let source = format!(
        "{BASIS}def loss[n](theta: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, basis(0i64, 1.0f32, theta)), 0i32))\nout = grad(loss)(insert(scalar_to_tensor(0.0f32), 0i32, 0i64))\n"
    );
    assert_both(&source, &[], "empty_basis");
}

// The false `n` claim still fails on both lanes. The trap names `where`, the
// operation the mapped `if` lowers to, which owns the declared-result guard as
// a same-shape producer (chelis#2642); it named `range` before. Neither is the
// name spec/04 §4.7 requires, the returned builtin `to_tensor`: the pin moves
// from one non-conforming producer to another, and chelis#2922 tracks naming
// `to_tensor`. The exact line is pinned so that any further change fails here.
#[test]
fn captured_cotangent_keeps_the_false_forward_range_claim() {
    let source = format!(
        "{}def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, basis(0i64, tensor_to_scalar(sum(scale, 0i32)), theta)), 0i32))\nout = grad(loss, wrt=scale)(to_tensor([3.0f32, 5.0f32]), to_tensor([2.0f32]))\n",
        BASIS.replace(
            "range(0i64, items_len)",
            "range(0i64, add(items_len, 1i64))"
        )
    );
    let source = chelis_surf::format::format_source(&source).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("false_claim.ch");
    common::write_file(&path, &source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    let out = dir.path().join("out");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg("--emit-c")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .assert()
        .success();
    assert!(common::link_generated(&out, "false_claim.c", "false_claim").success());
    let native = std::process::Command::new(out.join("false_claim"))
        .output()
        .unwrap();
    for result in [eval, native] {
        assert!(!result.status.success(), "{result:?}");
        let text = String::from_utf8_lossy(&result.stderr);
        assert!(
            text.contains("claimed = 2, where axis 0 = 3\nnumeric trap: domain in where at i64"),
            "{text}"
        );
    }
}

#[test]
fn distinct_maps_of_the_same_carrier_keep_invocation_order() {
    assert_both(
        r#"
def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = {
  s = tensor_to_scalar(sum(scale, 0i32))
  indices = range(0i64, shape(theta, 0i32))
  first = to_tensor(map(fn (i: i64) -> mul(s, 1.0f32), indices))
  second = to_tensor(map(fn (i: i64) -> mul(s, 1.0f32), indices))
  add(tensor_to_scalar(sum(mul(theta, first), 0i32)), tensor_to_scalar(sum(mul(theta, second), 0i32)))
}
out = grad(loss, wrt=scale)(to_tensor([100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32]), to_tensor([2.0f32]))
"#,
        &[3.0],
        "capture_two_maps",
    );
}

#[test]
fn recursive_parameter_cotangent_keeps_its_ordered_capture_leaf() {
    assert_both(
        r#"
def loss[n](theta: tensor[n, f32], params: (tensor[1, f32], tensor[1, f32])) -> f32 = {
  s = tensor_to_scalar(sum(params.0, 0i32))
  alias = s
  weights = to_tensor(map(fn (i: i64) -> add(s, alias), range(0i64, shape(theta, 0i32))))
  tensor_to_scalar(sum(mul(theta, weights), 0i32))
}
out = (grad(loss, wrt=params)(to_tensor([100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32]), (to_tensor([2.0f32]), to_tensor([4.0f32])))).0
"#,
        &[0.0],
        "capture_recursive_leaf",
    );
}

#[test]
fn authored_expand_keeps_its_existing_sum_adjoint() {
    assert_both(
        r#"
def loss[n](theta: tensor[n, f32], scale: tensor[1, f32]) -> f32 = {
  weights = insert(sum(scale, 0i32), 0i32, shape(theta, 0i32))
  tensor_to_scalar(sum(mul(theta, weights), 0i32))
}
out = grad(loss, wrt=scale)(to_tensor([100000000000000000000.0f32, -100000000000000000000.0f32, 3.0f32]), to_tensor([2.0f32]))
"#,
        &[3.0],
        "ordinary_expand_control",
    );
}
