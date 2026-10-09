//! chelis#3382 and chelis#3391: index math under `grad`.
//!
//! spec/05 §2.4: bound scalars are a zero-cotangent boundary and "do not pull
//! their producers (for example a window-count `floor_div`) into a structural
//! differentiability rejection"; [05-MOV-1] puts `expand` and `insert` sizes
//! under that rule. So an integer `floor_div` that only sizes an `insert` is
//! not rejected (#3382), while a `floor_div` whose value reaches the loss
//! still is ([05-OP-64]). spec/05 §5 gives every comparison a zero cotangent,
//! and spec/04 §4.7.2 forbids rejecting an extent because of its provenance,
//! so a comparison over a `range`-sized or wildcard extent differentiates
//! (#3391), while operands whose extents disagree still trap. The masks an
//! adjoint builds (`abs`, `pow`) are comparisons too. Each gradient is
//! checked against its closed form on the evaluator and on the C lane.
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;
use common::{build_and_run, parse_tensor_data};

fn write_file(path: &std::path::Path, source: &str) {
    common::write_file(
        path,
        &chelis_surf::format::format_source(source).expect("canonical Surf"),
    );
}

fn eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap()
}

/// The gradient printed as `out`, exactly, on the evaluator and on the C lane.
fn assert_gradient(source: &str, expected: &[f64], stem: &str) {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let eval_text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(parse_tensor_data(&eval_text, "out"), expected, "eval");
    let c_text = build_and_run(source, stem);
    assert_eq!(parse_tensor_data(&c_text, "out"), expected, "C");
}

/// As [`assert_gradient`], to a relative `tolerance` for a transcendental
/// closed form.
fn assert_gradient_close(source: &str, expected: &[f64], tolerance: f64, stem: &str) {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let eval_text = String::from_utf8(output.stdout).unwrap();
    let c_text = build_and_run(source, stem);
    for (lane, text) in [("eval", eval_text), ("C", c_text)] {
        let actual = parse_tensor_data(&text, "out");
        assert_eq!(actual.len(), expected.len(), "{lane}: {actual:?}");
        for (got, want) in actual.iter().zip(expected) {
            assert!(
                (got - want).abs() <= tolerance * want.abs().max(1.0),
                "{lane}: got {actual:?}, want {expected:?}"
            );
        }
    }
}

fn assert_refused(source: &str, needle: &str, stem: &str) {
    let output = eval(source, stem);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "must refuse: {stderr}");
    assert!(stderr.contains(needle), "{stderr}");
}

/// #3382's reproducer: `nrest = floor_div(c, groups)` only sizes a reshape and
/// an `insert`, so `d/dx sum((x + 1) * x) = 2x + 1`.
#[test]
fn integer_floor_div_that_only_sizes_an_insert_differentiates() {
    assert_gradient(
        r#"
def x0() -> tensor[2, 4, f32] = to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]])
def shift[a, c](x: tensor[a, c, f32], groups: i64, e: f32) -> tensor[a, c, f32] = {
  nb = shape(&x, 0i32)
  nrest = floor_div(shape(&x, 1i32), groups)
  g = reshape(x, [nb, groups, nrest])
  t = insert(insert(insert(scalar_to_tensor(e), 0i32, nb), 1i32, groups), 2i32, nrest)
  reshape(add(g, t), [nb, shape(&x, 1i32)])
}
def loss(x: tensor[2, 4, f32]) -> f32 = tensor_to_scalar(sum(sum(mul(shift(copy(x), 2i64, 1.0f32), x), 1i32), 0i32))
out = reshape(grad(loss)(x0()), [8i64])
"#,
        &[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0],
        "floor_div_extent",
    );
}

/// Negative twin: the same integer `floor_div` also reaches the loss as data
/// through a cast, so [05-OP-64]'s structural rejection still applies.
#[test]
fn integer_floor_div_whose_value_reaches_the_loss_is_still_rejected() {
    assert_refused(
        r#"
def loss(x: tensor[2, 4, f32], groups: i64) -> f32 = {
  nrest = floor_div(shape(&x, 1i32), groups)
  t = insert(insert(scalar_to_tensor(cast(nrest, f32)), 0i32, shape(&x, 0i32)), 1i32, mul(groups, nrest))
  tensor_to_scalar(sum(sum(mul(x, t), 1i32), 0i32))
}
out = grad(loss, wrt=x)(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]), 2i64)
"#,
        "floor_div is non-differentiable (piecewise constant)",
        "floor_div_data",
    );
}

/// Negative twin: a float `floor_div` on the loss path is piecewise constant.
#[test]
fn float_floor_div_on_the_loss_path_is_still_rejected() {
    assert_refused(
        r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(floor_div(x, insert(scalar_to_tensor(2.0f32), 0i32, 3i64)), 0i32))
out = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#,
        "floor_div is non-differentiable (piecewise constant)",
        "float_floor_div",
    );
}

/// #3391's reproducer: a causal mask whose extents come from
/// `to_tensor(range(n))` routes the cotangent to the lower triangle.
#[test]
fn comparison_over_range_extents_differentiates() {
    assert_gradient(
        r#"
def tri[s](seq_len: i64) -> tensor[s, s, bool] = {
  idx = to_tensor(range(0i64, seq_len))
  width = shape(&idx, 0i32)
  gte(insert(copy(idx), 1i32, width), insert(idx, 0i32, width))
}
def zeros3() -> tensor[3, 3, f32] = insert(insert(scalar_to_tensor(0.0f32), 0i32, 3i64), 0i32, 3i64)
def pick[s](m: tensor[s, s, bool], x: tensor[s, s, f32], z: tensor[s, s, f32]) -> tensor[s, s, f32] = where(m, x, z)
def loss(x: tensor[3, 3, f32]) -> f32 = tensor_to_scalar(sum(reshape(pick(tri(3i64), x, zeros3()), [9i64]), 0i32))
out = reshape(grad(loss)(zeros3()), [9i64])
"#,
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "range_mask",
    );
}

/// The same class over a wildcard parameter extent: `where(x > x*x, x, 0)`
/// passes the cotangent exactly where `0 < x < 1`.
#[test]
fn comparison_over_wildcard_extents_differentiates() {
    assert_gradient(
        r#"
def zero_like(x: &tensor[*, f32]) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, shape(x, 0i32))
def loss(x: tensor[*, f32]) -> f32 = tensor_to_scalar(sum(where(gt(&x, mul(&x, &x)), x, zero_like(&x)), 0i32))
out = grad(loss)(to_tensor([0.5f32, 2.0f32, 0.25f32]))
"#,
        &[1.0, 0.0, 1.0],
        "wildcard_mask",
    );
}

/// Negative twin: comparison operands whose runtime extents disagree trap on
/// both lanes rather than producing a gradient.
#[test]
fn comparison_with_disagreeing_extents_still_traps_under_grad() {
    let source = r#"
def loss(x: tensor[*, f32], y: tensor[*, f32]) -> f32 = tensor_to_scalar(sum(where(gt(&x, y), copy(x), mul(&x, &x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
"#;
    let needle = "gt operands disagree at axis 0";
    assert_refused(source, needle, "disagreeing_compare");
    let dir = tempdir().unwrap();
    let path = dir.path().join("disagreeing_compare.ch");
    let out = dir.path().join("out");
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .assert()
        .success();
    let run = std::process::Command::new(out.join("disagreeing_compare"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "C must trap: {stderr}");
    assert!(stderr.contains(needle), "{stderr}");
    assert!(
        parse_out_absent(&String::from_utf8_lossy(&run.stdout)),
        "no gradient may be printed"
    );
}

fn parse_out_absent(stdout: &str) -> bool {
    !stdout.lines().any(|line| line.starts_with("out = "))
}

/// The comparisons an adjoint builds share the class: `abs`'s sign masks over
/// a wildcard extent. `d/dx sum(abs(x)) = sign(x)`.
#[test]
fn abs_adjoint_masks_over_a_wildcard_extent_differentiate() {
    assert_gradient(
        r#"
def loss(x: tensor[*, f32]) -> f32 = tensor_to_scalar(sum(abs(x), 0i32))
out = grad(loss)(to_tensor([-1.0f32, 2.0f32, 3.0f32]))
"#,
        &[-1.0, 1.0, 1.0],
        "abs_wildcard",
    );
}

/// `pow`'s adjoint ([05-OP-79]) builds comparison and logical masks and
/// constants beside its operands, all over the wildcard extent:
/// `d/dx sum(pow(x, 2)) = 2x` exactly, and `d/dx sum(pow(x, x)) =
/// x^x (ln x + 1)`, evaluated in float64.
#[test]
fn pow_adjoint_over_a_wildcard_extent_differentiates() {
    assert_gradient(
        r#"
def loss(x: tensor[*, f32]) -> f32 = tensor_to_scalar(sum(pow(x, insert(scalar_to_tensor(2.0f32), 0i32, shape(&x, 0i32))), 0i32))
out = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#,
        &[2.0, 4.0, 6.0],
        "pow_square_wildcard",
    );
    assert_gradient_close(
        r#"
def loss(x: tensor[*, f32]) -> f32 = tensor_to_scalar(sum(pow(&x, &x), 0i32))
out = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#,
        &[1.0, 6.772_588_722_239_781, 56.662_531_794_038_97],
        1e-6,
        "pow_self_wildcard",
    );
}
