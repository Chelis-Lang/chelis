//! Issue #364 (grad acceptance arm): `grad` through a `cast(N, int32)`-axis
//! reduce and through `softmax(_, -1)` must differentiate the CORRECT axis.
//!
//! ## Why this file exists
//!
//! #364's headline is "the DAG lane silently lowers `cast(N, int32)`
//! reduction axes as axis 0". The FORWARD half of that is already fixed on
//! `main`: `extract_axis_raw` (chelis-ir/src/lower.rs) now routes through
//! `extract_int_axis`, which unwraps `cast`/`lit` (#473) and resolves the
//! negative-axis desugar `-1` -> `(app (var neg) (lit 1))` (#479), and
//! FATAL-errors on a non-constant axis instead of defaulting to 0. The
//! forward eval-vs-backend parity is pinned in `rank_poly_tier3.rs`
//! (`cast_axis_reduction_lowers_to_named_axis_not_zero`,
//! `cast_axis_reduction_axis_two_rank_three`,
//! `negative_axis_softmax_resolves_to_last_axis`,
//! `negative_axis_reduce_lowers_to_last_axis_in_backend`).
//!
//! What had NO coverage is the issue's MOST DANGEROUS symptom and the
//! explicit acceptance bullet "grad through a cast-axis reduce matches
//! finite differences": under `grad`, the traced forward AND backward
//! treat the axis as 0, so `grad` silently returns the gradient of a
//! DIFFERENT function (e.g. column-softmax instead of row-softmax) with NO
//! error — forward, eval-vs-backend, and finite-difference of the forward
//! all stay correct while `grad` disagrees. Since the lowering fix is what
//! the grad lane reads its normalized axis from, the fix transitively
//! corrects `grad` too; this file is the executable proof of that.
//!
//! ## Discriminating fixtures (non-vacuous by construction)
//!
//! A reduce-twice-to-scalar loss like `sum(sum(x*x, a), b)` collapses the
//! whole tensor and gives `2*x` for ANY axes — it does NOT discriminate
//! the reduced axis and would be a vacuous grad test. Every loss here
//! routes the reduction output through a non-symmetric `square` so a wrong
//! axis produces a DIFFERENT gradient, and every expected vector is
//! cross-checked against central finite differences (Python, h=1e-4) AND
//! against the explicit wrong-axis control evaluated next to it:
//!
//! * cast(1)-axis rank-2 `sum(square(sum(x, 1)))`:
//!   row-sum gradient `[[12,12,12],[30,30,30]]` (correct) vs the axis-0
//!   column-sum gradient `[[10,14,18],[10,14,18]]` (the pre-fix value).
//! * cast(2)-axis rank-3 `sum(square(sum(x, 2)))`:
//!   `[6,6,14,14,22,22]` (correct) vs the axis-1 value
//!   `[18,24,18,24,18,24]`.
//! * `softmax(scores, -1)` SDPA q-grad (the downstream
//!   `School.Nn.Attention` scenario, `school
//!   tests_blocked/grad/attention_sdpa_qgrad.ch`): uniform-weight rows give
//!   the true `dL/dq = 0.5*(r_j - 2) = [-0.5, 0.5]` per row; the mis-axed
//!   (column-softmax) lane returns EXACTLY `[0,0,0,0]` on this fixture.
//!
//! That last pair is the issue's vacuousness trap: both `[-0.5,0.5,...]`
//! and `[0,0,...]` have q-grad row-sum 0 (softmax backward rows sum to 0),
//! so the #319 "sum-of-q-grad approx 0" assertion passed against the WRONG
//! gradient. Asserting EXACT per-element values is what makes these tests
//! detect the regression.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

/// `grad` of `sum(square(sum(x, cast(1, int32))))` on a non-square rank-2
/// operand. Reduces axis 1 (per-row sums), squares, sums. The wrong-axis
/// (0) lowering reduces COLUMNS and gives a different gradient.
const CAST_AXIS1_RANK2: &str = "module Repro.GradCastAxis1\n\
def f(x: tensor[2, 3, f32]) -> f32 = {\n\
  r = sum(x, cast(1, int32))\n\
  sum(mul(r, r), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]]))\n";

/// Wrong-axis control for the rank-2 cast case: the SAME loss with the
/// reduce axis written as `cast(0, int32)` instead of `cast(1, int32)`.
/// Its gradient is what a silent default-to-0 lowering of the `cast(1)`
/// form would have produced; pinning it proves `CAST_AXIS1_RANK2` is not
/// vacuously equal to the wrong answer.
const CAST_AXIS0_RANK2_CONTROL: &str = "module Repro.GradCastAxis0\n\
def f(x: tensor[2, 3, f32]) -> f32 = {\n\
  r = sum(x, cast(0, int32))\n\
  sum(mul(r, r), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]]))\n";

/// `grad` of `sum(square(sum(x, cast(2, int32))))` on a rank-3 operand.
/// Reduces the LAST axis; the cross-rank control reduces axis 1.
const CAST_AXIS2_RANK3: &str = "module Repro.GradCastAxis2\n\
def f(x: tensor[1, 3, 2, f32]) -> f32 = {\n\
  r = sum(x, cast(2, int32))\n\
  sum(sum(mul(r, r), cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)], [cast(5.0, f32), cast(6.0, f32)]]]))\n";

/// Wrong-axis control for the rank-3 cast case: reduce axis 1 instead of
/// axis 2.
const CAST_AXIS1_RANK3_CONTROL: &str = "module Repro.GradCastAxis1R3\n\
def f(x: tensor[1, 3, 2, f32]) -> f32 = {\n\
  r = sum(x, cast(1, int32))\n\
  sum(sum(mul(r, r), cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)], [cast(5.0, f32), cast(6.0, f32)]]]))\n";

/// SDPA-shaped q-grad through `softmax(scores, -1)`. `k = I` so
/// `matmul(q, k) = q` and `scores = q` (scale = ones); `v = [[1,0],[0,3]]`.
/// Uniform rows (`[1,1]`, `[2,2]`) give row-softmax weights 0.5, so the
/// true `dL/dq` row is `0.5*(r_j - 2) = [-0.5, 0.5]`.
const SDPA_NEG_AXIS_QGRAD: &str = "module Repro.GradSdpaNegAxis\n\
def f(q: tensor[2, 2, f32], k: tensor[2, 2, f32], v: tensor[2, 2, f32], scale: tensor[2, 2, f32]) -> f32 = {\n\
  scores = mul(matmul(q, k), scale)\n\
  w = softmax(scores, -1)\n\
  o = matmul(w, v)\n\
  sum(sum(o, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(1.0, f32)], [cast(2.0, f32), cast(2.0, f32)]]), to_tensor([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]]), to_tensor([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(3.0, f32)]]), to_tensor([[cast(1.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]])).0\n";

/// Positive control: the SAME SDPA loss with `softmax(scores, 1)`. At
/// rank 2, axis `1` IS the last axis, so this must equal the `-1` form
/// element-for-element. This is the row-softmax oracle the negative form
/// has to reproduce.
const SDPA_POS_AXIS_QGRAD: &str = "module Repro.GradSdpaPosAxis\n\
def f(q: tensor[2, 2, f32], k: tensor[2, 2, f32], v: tensor[2, 2, f32], scale: tensor[2, 2, f32]) -> f32 = {\n\
  scores = mul(matmul(q, k), scale)\n\
  w = softmax(scores, 1)\n\
  o = matmul(w, v)\n\
  sum(sum(o, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(1.0, f32)], [cast(2.0, f32), cast(2.0, f32)]]), to_tensor([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]]), to_tensor([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(3.0, f32)]]), to_tensor([[cast(1.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]])).0\n";

/// Wrong-axis SDPA control: `softmax(scores, 0)` (column softmax). On this
/// uniform-row fixture the resulting q-grad is EXACTLY `[0,0,0,0]` — the
/// "q-grads come back ~0" downstream symptom and the value the mis-axed
/// `-1` lane produced before the fix. Asserting `SDPA_NEG_AXIS_QGRAD` is
/// NOT this proves the negative-parity arm and guards the sum-of-q-grad
/// approx 0 vacuousness trap (this all-zero vector also has row-sum 0).
const SDPA_COL_AXIS_QGRAD: &str = "module Repro.GradSdpaColAxis\n\
def f(q: tensor[2, 2, f32], k: tensor[2, 2, f32], v: tensor[2, 2, f32], scale: tensor[2, 2, f32]) -> f32 = {\n\
  scores = mul(matmul(q, k), scale)\n\
  w = softmax(scores, 0)\n\
  o = matmul(w, v)\n\
  sum(sum(o, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(1.0, f32)], [cast(2.0, f32), cast(2.0, f32)]]), to_tensor([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]]), to_tensor([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(3.0, f32)]]), to_tensor([[cast(1.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]])).0\n";

/// Run `chelis eval --file` on `source` and return stdout. Uses
/// `CHELIS_STYLE_GATE_DISABLE=1` per the repo's ad-hoc-Surf test
/// convention; the eval pipeline (parse/type/effect/lower/grad) still runs
/// in full.
fn eval_stdout(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path: &Path = dir.path();
    let file = path.join(format!("{stem}.ch"));
    fs::write(&file, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(path)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", file.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{stem}: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    stdout
}

/// Parse the `tensor(shape=[..], data=[..])` line `eval` prints into
/// `(shape, data)`. A single root prints the bare `tensor(...)` value with
/// no `name =` label; multiple roots print `name = tensor(...)`. These
/// fixtures each have exactly one root, so match the line carrying
/// `tensor(`. Tolerates `1` and `1.0` float rendering.
fn parse_out_tensor(stdout: &str) -> (Vec<usize>, Vec<f64>) {
    let line = stdout
        .lines()
        .find(|l| l.contains("tensor(shape="))
        .unwrap_or_else(|| panic!("no `tensor(shape=...)` line in eval stdout: {stdout}"));
    let shape_str = line
        .split("shape=[")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .unwrap_or_else(|| panic!("no shape in: {line}"));
    let shape: Vec<usize> = if shape_str.trim().is_empty() {
        vec![]
    } else {
        shape_str
            .split(',')
            .map(|t| t.trim().parse::<usize>().expect("shape int"))
            .collect()
    };
    let data_str = line
        .split("data=[")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .unwrap_or_else(|| panic!("no data in: {line}"));
    let data: Vec<f64> = if data_str.trim().is_empty() {
        vec![]
    } else {
        data_str
            .split(',')
            .map(|t| t.trim().parse::<f64>().expect("data float"))
            .collect()
    };
    (shape, data)
}

/// Eval `source`, assert the `out` gradient has `shape` and matches
/// `expected` element-for-element to 1e-3.
fn assert_grad(source: &str, stem: &str, shape: &[usize], expected: &[f64]) {
    let stdout = eval_stdout(source, stem);
    let (got_shape, got_data) = parse_out_tensor(&stdout);
    assert_eq!(
        got_shape, shape,
        "{stem}: gradient shape mismatch; got {got_shape:?}, stdout={stdout}",
    );
    assert_eq!(
        got_data.len(),
        expected.len(),
        "{stem}: gradient length mismatch; stdout={stdout}",
    );
    for (i, (g, e)) in got_data.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "{stem}: grad[{i}] = {g} != {e} (full {got_data:?}); stdout={stdout}",
        );
    }
}

/// Acceptance: `grad` through a `cast(1, int32)`-axis reduce on a rank-2
/// operand matches the row-sum (finite-difference) gradient
/// `[[12,12,12],[30,30,30]]` — NOT the column gradient a default-to-0
/// lowering would yield. `loss = (x0+x1+x2)^2 + (x3+x4+x5)^2`, row sums
/// `r = [6, 15]`, so `dL/dx_ij = 2*r_i`. Cross-checked against central
/// finite differences (h=1e-4).
#[test]
fn issue_364_grad_cast_axis1_rank2_matches_fd() {
    assert_grad(
        CAST_AXIS1_RANK2,
        "cast_ax1_r2",
        &[2, 3],
        &[12.0, 12.0, 12.0, 30.0, 30.0, 30.0],
    );
}

/// Negative parity for the rank-2 cast case: the wrong-axis control
/// (`cast(0)`) gives the DIFFERENT column gradient
/// `[[10,14,18],[10,14,18]]` (`2*c_j`, c = [5,7,9]). If the `cast(1)`
/// lowering silently defaulted to axis 0, `issue_364_..._rank2_matches_fd`
/// would equal THIS; pinning both proves the discriminator is live.
#[test]
fn issue_364_grad_cast_axis0_rank2_is_distinct_control() {
    assert_grad(
        CAST_AXIS0_RANK2_CONTROL,
        "cast_ax0_r2",
        &[2, 3],
        &[10.0, 14.0, 18.0, 10.0, 14.0, 18.0],
    );
    // The two rank-2 gradients must NOT be equal, or the cast(1) test is
    // vacuous.
    let ax1 = parse_out_tensor(&eval_stdout(CAST_AXIS1_RANK2, "cast_ax1_r2_cmp")).1;
    let ax0 = parse_out_tensor(&eval_stdout(CAST_AXIS0_RANK2_CONTROL, "cast_ax0_r2_cmp")).1;
    assert_ne!(
        ax1, ax0,
        "cast(1) and cast(0) reduce grads must differ; equal means the axis is ignored",
    );
}

/// Acceptance: `grad` through a `cast(2, int32)`-axis reduce on a rank-3
/// operand matches the last-axis (finite-difference) gradient
/// `[6,6,14,14,22,22]`. Reduce axis 2 -> r = [[3,7,11]], `loss = sum(r^2)`,
/// `dL/dx = 2*r_j` shared across the size-2 last axis.
#[test]
fn issue_364_grad_cast_axis2_rank3_matches_fd() {
    assert_grad(
        CAST_AXIS2_RANK3,
        "cast_ax2_r3",
        &[1, 3, 2],
        &[6.0, 6.0, 14.0, 14.0, 22.0, 22.0],
    );
}

/// Negative parity for the rank-3 cast case: the cross-rank wrong-axis
/// control (`cast(1)`) gives `[18,24,18,24,18,24]` (column sums `[9,12]`,
/// `2*[9,12]`). Distinct from the `cast(2)` gradient.
#[test]
fn issue_364_grad_cast_axis1_rank3_is_distinct_control() {
    assert_grad(
        CAST_AXIS1_RANK3_CONTROL,
        "cast_ax1_r3",
        &[1, 3, 2],
        &[18.0, 24.0, 18.0, 24.0, 18.0, 24.0],
    );
    let ax2 = parse_out_tensor(&eval_stdout(CAST_AXIS2_RANK3, "cast_ax2_r3_cmp")).1;
    let ax1 = parse_out_tensor(&eval_stdout(CAST_AXIS1_RANK3_CONTROL, "cast_ax1_r3_cmp")).1;
    assert_ne!(
        ax2, ax1,
        "cast(2) and cast(1) reduce grads must differ; equal means the axis is ignored",
    );
}

/// Acceptance (the dangerous scenario): `grad` through `softmax(scores,
/// -1)` differentiates the LAST axis (row softmax), so the SDPA q-grad is
/// the true `[-0.5, 0.5, -0.5, 0.5]` (cross-checked against central finite
/// differences). Before #473/#479, the `-1` desugar `(app (var neg) (lit
/// 1))` was not recognized and `extract_axis_raw` defaulted it to 0
/// (column softmax), and the grad lane silently returned `[0,0,0,0]`.
#[test]
fn issue_364_grad_softmax_neg_axis_qgrad_matches_fd() {
    assert_grad(
        SDPA_NEG_AXIS_QGRAD,
        "sdpa_neg_qgrad",
        &[2, 2],
        &[-0.5, 0.5, -0.5, 0.5],
    );
}

/// Positive control: `softmax(scores, 1)` (the explicit last axis at rank
/// 2) gives the SAME q-grad as the `-1` form. This is the row-softmax
/// oracle; the negative form must reproduce it exactly.
#[test]
fn issue_364_grad_softmax_pos_axis_qgrad_equals_neg_form() {
    assert_grad(
        SDPA_POS_AXIS_QGRAD,
        "sdpa_pos_qgrad",
        &[2, 2],
        &[-0.5, 0.5, -0.5, 0.5],
    );
    let neg = parse_out_tensor(&eval_stdout(SDPA_NEG_AXIS_QGRAD, "sdpa_neg_cmp")).1;
    let pos = parse_out_tensor(&eval_stdout(SDPA_POS_AXIS_QGRAD, "sdpa_pos_cmp")).1;
    for (i, (n, p)) in neg.iter().zip(&pos).enumerate() {
        assert!(
            (n - p).abs() < 1e-3,
            "softmax(-1) q-grad[{i}]={n} must equal softmax(1) q-grad[{i}]={p}",
        );
    }
}

/// Negative parity + vacuousness guard: the wrong-axis `softmax(scores, 0)`
/// (column softmax) q-grad is EXACTLY `[0,0,0,0]` on this uniform-row
/// fixture. That is what the mis-axed `-1` lane returned before the fix.
/// Both `[-0.5,0.5,-0.5,0.5]` and `[0,0,0,0]` have q-grad row-sum 0, so a
/// sum-of-q-grad approx 0 assertion would pass against BOTH; the exact
/// per-element assertion in `..._neg_axis_qgrad_matches_fd` is what
/// distinguishes them. This test pins the wrong value and asserts the
/// correct form is NOT it.
#[test]
fn issue_364_grad_softmax_col_axis_qgrad_is_zero_wrong_control() {
    assert_grad(
        SDPA_COL_AXIS_QGRAD,
        "sdpa_col_qgrad",
        &[2, 2],
        &[0.0, 0.0, 0.0, 0.0],
    );
    let neg = parse_out_tensor(&eval_stdout(SDPA_NEG_AXIS_QGRAD, "sdpa_neg_vac")).1;
    // Sum-of-q-grad is 0 for BOTH the correct and the wrong gradient: the
    // trap the #319 downstream test fell into. Prove it here so the trap is
    // documented and the exact-value assertion is justified.
    let neg_sum: f64 = neg.iter().sum();
    assert!(
        neg_sum.abs() < 1e-3,
        "softmax(-1) q-grad row-sum should be ~0 (softmax backward), got {neg_sum}",
    );
    let col = parse_out_tensor(&eval_stdout(SDPA_COL_AXIS_QGRAD, "sdpa_col_vac")).1;
    let col_sum: f64 = col.iter().sum();
    assert!(
        col_sum.abs() < 1e-3,
        "the WRONG (column) q-grad also sums to ~0 ({col:?}), exactly why sum-of-q-grad is vacuous",
    );
    // The correct gradient must NOT be the all-zero wrong one.
    assert!(
        neg.iter().any(|g| g.abs() > 1e-2),
        "softmax(-1) q-grad must be the non-zero row gradient, not the all-zero column control: {neg:?}",
    );
}
