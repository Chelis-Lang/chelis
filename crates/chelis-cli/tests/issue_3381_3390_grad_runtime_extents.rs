//! chelis#3381 and chelis#3390: `grad` over runtime extents.
//!
//! spec/06 §2.10: `grad` reverses the recurrence the forward program actually
//! executed, and spec/04 §4.7.2 forbids rejecting an extent because of its
//! provenance. So every backward node takes its extents from a forward value
//! that carries them: a synthesized adjoint constant is sized from the value it
//! stands beside (#3381), and a concat over parts whose concat-axis extent is a
//! signature name or a wildcard splits its cotangent at the runtime source
//! boundaries ([05-OP-62], #3390). Each gradient is checked against its
//! closed-form derivative on the evaluator and on the C lane.
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

fn assert_close(lane: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{lane}: {actual:?}");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "{lane}: element {index}: got {got}, want {want}; {actual:?}"
        );
    }
}

/// The gradient printed as `out`, on the evaluator and on the C lane.
fn assert_gradient(source: &str, expected: &[f64], tolerance: f64, stem: &str) {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let eval_text = String::from_utf8(output.stdout).unwrap();
    let c_text = build_and_run(source, stem);
    let eval_data = parse_tensor_data(&eval_text, "out");
    assert_close("eval", &eval_data, expected, tolerance);
    let c_data = parse_tensor_data(&c_text, "out");
    assert_close("C", &c_data, expected, tolerance);
}

fn assert_refused(source: &str, needle: &str, stem: &str) {
    let output = eval(source, stem);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "must refuse: {stderr}");
    assert!(stderr.contains(needle), "{stderr}");
}

/// #3381's reproducer: School's `group_norm` RMS step, with the divisor
/// broadcast to extents computed from shape reads. The references are
/// `torch.autograd` of `y = x / sqrt(mean(x*x)); loss = y[0]` in float64.
#[test]
fn group_rms_over_computed_extents_differentiates() {
    assert_gradient(
        r#"
def rms_groups[a, c, h, w](x: tensor[a, c, h, w, f32], groups: i64, group_size: i64) -> tensor[a, c, h, w, f32] = {
  nb = shape(&x, 0i32)
  ch = shape(&x, 1i32)
  hh = shape(&x, 2i32)
  ww = shape(&x, 3i32)
  nrest = mul(group_size, mul(hh, ww))
  g = reshape(x, [nb, groups, nrest])
  count = insert(insert(scalar_to_tensor(cast(nrest, f32)), 0i32, nb), 1i32, groups)
  ms = insert(div(sum(mul(&g, &g), 2i32), count), 2i32, nrest)
  reshape(div(g, sqrt(ms)), [nb, ch, hh, ww])
}
def x0() -> tensor[1, 2, 1, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [1i64, 2i64, 1i64, 2i64])
def loss(x: tensor[1, 2, 1, 2, f32]) -> f32 = tensor_to_scalar(sum(reshape(mul(rms_groups(x, 1i64, 2i64), reshape(to_tensor([1.0f32, 0.0f32, 0.0f32, 0.0f32]), [1i64, 2i64, 1i64, 2i64])), [4i64]), 0i32))
out = reshape(grad(loss)(x0()), [4i64])
"#,
        &[
            0.352_976_759_281_107_06,
            -0.024_343_224_778_007_38,
            -0.036_514_837_167_011_07,
            -0.048_686_449_556_014_76,
        ],
        1e-6,
        "group_rms",
    );
}

/// The class behind #3381: every adjoint that synthesizes a constant
/// (`tanh`, `atan`, `erf`, `sqrt`) beside a value whose axis is an inserted
/// runtime extent. Each `x[i, j]` reaches `k = 3` copies of `f(s_i)`, so
/// its derivative is `k * f'(s_i)` with `s = [0.75, 1.75]`.
#[test]
fn adjoint_constants_take_runtime_extents_from_their_operand() {
    // `3 * (1 - tanh(s)^2 + 1/(1 + s^2) + 2/sqrt(pi) * exp(-s^2) + 1/(2 sqrt(s)))`
    // evaluated in float64.
    let (first, second) = (7.370_601_439_998_493, 2.372_116_231_836_498);
    assert_gradient(
        r#"
def act_sum(x: tensor[2, 2, f32]) -> f32 = {
  k = add(shape(&x, 1i32), 1i64)
  t = insert(sum(x, 1i32), 1i32, k)
  tensor_to_scalar(sum(sum(add(add(tanh(&t), atan(&t)), add(erf(&t), sqrt(t))), 1i32), 0i32))
}
out = reshape(grad(act_sum)(reshape(to_tensor([0.25f32, 0.5f32, 0.75f32, 1.0f32]), [2i64, 2i64])), [4i64])
"#,
        &[first, first, second, second],
        1e-6,
        "adjoint_constants",
    );
}

/// #3390's reproducer: both parts carry the signature name `n`.
#[test]
fn concat_of_symbolic_parts_differentiates() {
    assert_gradient(
        r#"
def loss[n](x: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(concat([x, x], 0i32), 0i32))
out = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#,
        &[2.0, 2.0, 2.0],
        0.0,
        "symbolic_concat",
    );
}

/// Wildcard parts around a literal one, under a position-dependent weight,
/// so a cotangent split at the wrong boundary gives the wrong numbers:
/// `d/dx_i sum(w * z * z) = 2 * w_i * z_i` read back at each part's offset.
#[test]
fn concat_splits_its_cotangent_at_runtime_boundaries() {
    let source = |part: &str| {
        format!(
            r#"
def loss3(a: tensor[*, f32], b: tensor[2, f32], c: tensor[*, f32], w: tensor[7, f32]) -> f32 = {{
  joined = concat([a, b, c], 0i32)
  tensor_to_scalar(sum(mul(mul(&joined, &joined), w), 0i32))
}}
def a0() -> tensor[3, f32] = to_tensor([1.0f32, 2.0f32, 3.0f32])
def b0() -> tensor[2, f32] = to_tensor([4.0f32, 5.0f32])
def c0() -> tensor[2, f32] = to_tensor([6.0f32, 7.0f32])
def w0() -> tensor[7, f32] = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32])
out = grad(loss3, wrt={part})(a0(), b0(), c0(), w0())
"#
        )
    };
    assert_gradient(&source("a"), &[2.0, 8.0, 18.0], 0.0, "concat_part_a");
    assert_gradient(&source("b"), &[32.0, 50.0], 0.0, "concat_part_b");
    assert_gradient(&source("c"), &[72.0, 98.0], 0.0, "concat_part_c");
}

/// A runtime-width concat on a trailing axis of a rank-2 tensor.
#[test]
fn concat_on_a_runtime_trailing_axis_differentiates() {
    assert_gradient(
        r#"
def loss2(a: tensor[2, *, f32], b: tensor[2, 1, f32], w: tensor[2, 4, f32]) -> f32 = {
  joined = concat([a, b], 1i32)
  tensor_to_scalar(sum(sum(mul(mul(&joined, &joined), w), 1i32), 0i32))
}
def a0() -> tensor[2, 3, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]), [2i64, 3i64])
def b0() -> tensor[2, 1, f32] = reshape(to_tensor([7.0f32, 8.0f32]), [2i64, 1i64])
def w0() -> tensor[2, 4, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32]), [2i64, 4i64])
out = reshape(grad(loss2, wrt=a)(a0(), b0(), w0()), [6i64])
"#,
        &[2.0, 8.0, 18.0, 40.0, 60.0, 84.0],
        0.0,
        "concat_trailing_axis",
    );
}

/// Negative twin: a concat axis that is not a static literal (spec/04
/// §4.5.4 rule 5) still has no gradient DAG, and `grad` refuses it.
#[test]
fn concat_over_a_runtime_axis_still_refuses_under_grad() {
    assert_refused(
        r#"
def loss(x: tensor[2, f32], axis: i32) -> f32 = tensor_to_scalar(sum(concat([x, x], axis), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32]), 0i32)
"#,
        "tensor concat cannot be represented by the static tensor DAG",
        "dynamic_axis_concat",
    );
}

/// Negative twin: parts whose bystander axes disagree are a loud failure
/// under `grad`, never a silently mis-sized gradient. The evaluator traps when
/// the parts meet; the C lane, which sees the inlined literal extents, refuses
/// at build time.
#[test]
fn concat_with_disagreeing_runtime_bystanders_fails_loudly_under_grad() {
    let source = r#"
def loss(a: tensor[*, *, f32], b: tensor[*, *, f32]) -> f32 = tensor_to_scalar(sum(sum(concat([a, b], 0i32), 1i32), 0i32))
def a0() -> tensor[1, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32]), [1i64, 2i64])
def b0() -> tensor[1, 3, f32] = reshape(to_tensor([3.0f32, 4.0f32, 5.0f32]), [1i64, 3i64])
out = reshape(grad(loss, wrt=a)(a0(), b0()), [2i64])
"#;
    assert_refused(source, "disagree at axis 1", "disagreeing_bystanders");
    let dir = tempdir().unwrap();
    let path = dir.path().join("disagreeing_bystanders.ch");
    write_file(&path, source);
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(!build.status.success(), "C must refuse: {stderr}");
    assert!(stderr.contains("expected 2, got 3"), "{stderr}");
}
