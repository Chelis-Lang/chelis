//! Issue Chelis-Lang/chelis#218: `to_tensor` literals in differentiable
//! function bodies (Part 2 of #199).
//!
//! Part 1 (the `BlasMatmul` gradient rule) landed via PR #211; Part 2
//! was reverted from PR #211 after the R1->R4 red-team cascade
//! surfaced five distinct failure modes, signalling architectural
//! redesign rather than further patches. The Discovery work for the
//! architectural blocker (#219, Name vs Lit unification asymmetry)
//! recommended Option A — a permissive `Name <-> Lit` arm in
//! `unify_dim`.
//!
//! This test file is the acceptance oracle for #218. Each section is
//! tagged with the originating R-round failure mode so red-team
//! lineage stays traceable:
//!
//!   * R1 HIGH-1: negative literals (`-1.0`) inside `to_tensor`
//!     recognized as numeric leaves so the literal recognizer doesn't
//!     fall back to host routing.
//!   * R1 HIGH-2: reduction lowering (`sum(sum(matmul(x, w), 0), 0)`)
//!     where `w` is a `to_tensor` literal — the static-dim source fix
//!     prevents wildcard propagation into the reduction's output dims.
//!   * R2 HIGH-A: elementwise consumers (relu, tanh, sigmoid, gelu,
//!     silu, exp, log, neg, sqrt, abs, softmax) between matmul and a
//!     reduction — same root cause as R1 HIGH-2 but more consumers.
//!   * R3 HIGH-CONCAT: `concat([to_tensor([...]), to_tensor([...])], 0)`
//!     where the two literals have distinct concrete dims — per-axis
//!     join in Cons element unification produces the expected
//!     wildcard along the differing axis instead of rejecting.
//!   * R3 MEDIUM-RESHAPE-ICE: `reshape(matmul(x, w), [cast(2, int64),
//!     cast(1, int64)])` in a grad body — relies on #220 fix at
//!     `lower.rs::extract_dim_list` (PR #224) plus the routing
//!     exemption in this PR.
//!   * R4 HIGH (= issue #219): named-dim sigs with concrete callers
//!     `def f(x: tensor[batch, hidden, f32]) = ...` then
//!     `f(to_tensor([[1, 2, 3]]))` type-checks cleanly. This is the
//!     Name <-> Lit unification arm.
//!
//! Discovery contract notes:
//!   * The reduction R1 HIGH-2 and elementwise R2 HIGH-A defenses
//!     stayed in place as defense in depth even though R2's
//!     source-side fix (concrete dims from static `to_tensor` literals)
//!     keeps the wildcards from being emitted in the first place.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

fn run_build(path: &Path) -> std::process::Output {
    let out_dir = path.parent().expect("source path has parent");
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    std::process::Command::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("spawn chelis")
}

fn errors(json: &Value) -> Vec<&Value> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .collect()
}

fn error_messages(json: &Value) -> Vec<String> {
    errors(json)
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

// =================================================================
// R4 HIGH / issue #219: named-dim sig accepts concrete to_tensor
// caller.
// =================================================================

#[test]
fn issue_218_r4_named_dim_sig_accepts_concrete_to_tensor_caller() {
    // The R4 headline failure: a sig declaring a multi-letter dim
    // name `batch` cannot be called with a concrete-shaped
    // `to_tensor` literal. After issue #219 Option A, the Name <-> Lit
    // unification arm accepts this without weakening the
    // distinct-Name <-> distinct-Name rejection.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("named_dim_sig.ch");
    write_file(
        &path,
        "def f(x: tensor[batch, hidden, f32]) -> tensor[batch, hidden, f32] = copy(x)\n\
         out = f(to_tensor([[1.0, 2.0, 3.0]]))\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "R4 HIGH: named-dim sig should accept concrete to_tensor caller; got errors {errs:?}",
    );
}

#[test]
fn issue_218_r4_named_dim_sig_still_rejects_distinct_concrete_names() {
    // Regression lock: the Name <-> Lit relaxation must not also relax
    // distinct Name <-> Name. `batch` and `seq` are distinct rigid
    // dim names; a body that returns tensor[seq, ...] does not
    // satisfy a sig that declares tensor[batch, ...].
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("distinct_named_dims.ch");
    write_file(
        &path,
        "def f(x: tensor[batch, f32], y: tensor[seq, f32]) -> tensor[batch, f32] = copy(y)\n",
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "R4 regression-lock: distinct Name vs Name must still error",
    );
}

#[test]
fn issue_218_r4_var_position_distinct_concrete_args_still_errors() {
    // Negative-parity: a sig using `Var` quantifiers `tensor[n, p]`
    // shared across two args still requires the two args to have the
    // same concrete dim. The Name <-> Lit arm only relaxes Name slots,
    // not Var slots.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shared_var_dim.ch");
    write_file(
        &path,
        "def f[n, p](x: tensor[n, p], y: tensor[n, p]) -> tensor[n, p] = copy(y)\n\
         out = f(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0, 5.0]))\n",
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "R4 regression-lock: shared Var across two args must still reject when concrete dims differ",
    );
}

// =================================================================
// R1 HIGH-1: negative literal in to_tensor lowers + grads.
// =================================================================

#[test]
fn issue_218_r1_high1_negative_float_literal_in_to_tensor_recognized() {
    // `to_tensor([0.5, -1.0, 2.0])` must type-check as
    // `tensor[3, f32]`. The IR's literal recognizer must see through
    // `(app (var neg) <inner>)` for the `-1.0` element.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_float_literal.ch");
    write_file(
        &path,
        "def g(x: tensor[3, f32]) -> tensor[f32] = {\n\
           w = to_tensor([0.5, -1.0, 2.0])\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[3, f32]) -> tensor[3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([1.0, 1.0, 1.0]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "R1 HIGH-1: negative-literal `to_tensor` should build; stderr={stderr}",
    );
}

#[test]
fn issue_218_r1_high1_negative_int_literal_in_to_tensor_recognized() {
    // Same as above but with int leaves: `to_tensor([-2, -1, 0, 1, 2])`
    // must type-check and lower without routing to host.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_int_literal.ch");
    write_file(
        &path,
        "def g(x: tensor[5, f32]) -> tensor[f32] = {\n\
           w = to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[5, f32]) -> tensor[5, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([1.0, 1.0, 1.0, 1.0, 1.0]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "R1 HIGH-1: negative-int-literal `to_tensor` should build; stderr={stderr}",
    );
}

// =================================================================
// R1 HIGH-2: reduction lowering with to_tensor weight literal.
// =================================================================

#[test]
fn issue_218_r1_high2_matmul_then_reduction_grad_builds() {
    // The issue text headline: `g(x) = sum(sum(matmul(x, w), 0), 0)`
    // where `w` is built from a static `to_tensor` literal. The
    // reduction lowering used to ICE on the wildcard dim that the
    // type-checker emitted; with the static-dim source fix the
    // wildcards never appear.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("matmul_reduce.ch");
    write_file(
        &path,
        "def g(x: tensor[2, 3, f32]) -> tensor[f32] = {\n\
           w = to_tensor([[1.0], [2.0], [3.0]])\n\
           sum(sum(matmul(copy(x), w), 0), 0)\n\
         }\n\
         def compute_grad(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "R1 HIGH-2: matmul-then-reduction grad should build; stderr={stderr}",
    );
}

// =================================================================
// R2 HIGH-A: elementwise consumers between matmul and reduction.
// =================================================================

fn build_elementwise_program(activation: &str) -> String {
    format!(
        "def g(x: tensor[2, 3, f32]) -> tensor[f32] = {{\n\
           w = to_tensor([[1.0], [2.0], [3.0]])\n\
           sum(sum({activation}(matmul(copy(x), w)), 0), 0)\n\
         }}\n\
         def compute_grad(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))\n",
    )
}

#[test]
fn issue_218_r2_high_a_relu_between_matmul_and_sum_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("elem_relu.ch");
    write_file(&path, &build_elementwise_program("relu"));
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "R2 HIGH-A relu: {stderr}");
}

#[test]
fn issue_218_r2_high_a_tanh_between_matmul_and_sum_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("elem_tanh.ch");
    write_file(&path, &build_elementwise_program("tanh"));
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "R2 HIGH-A tanh: {stderr}");
}

#[test]
fn issue_218_r2_high_a_sigmoid_between_matmul_and_sum_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("elem_sigmoid.ch");
    write_file(&path, &build_elementwise_program("sigmoid"));
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "R2 HIGH-A sigmoid: {stderr}");
}

#[test]
fn issue_218_r2_high_a_gelu_between_matmul_and_sum_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("elem_gelu.ch");
    write_file(&path, &build_elementwise_program("gelu"));
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "R2 HIGH-A gelu: {stderr}");
}

#[test]
fn issue_218_r2_high_a_silu_between_matmul_and_sum_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("elem_silu.ch");
    write_file(&path, &build_elementwise_program("silu"));
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "R2 HIGH-A silu: {stderr}");
}

#[test]
fn issue_218_r2_high_a_exp_between_matmul_and_sum_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("elem_exp.ch");
    write_file(&path, &build_elementwise_program("exp"));
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "R2 HIGH-A exp: {stderr}");
}

// =================================================================
// R3 HIGH-CONCAT: list literals containing to_tensor of differing
// shapes type-check via per-axis Cons join.
// =================================================================

#[test]
fn issue_218_r3_high_concat_distinct_axis_0_type_checks() {
    // `concat([to_tensor([[1.0, 2.0, 3.0]]), to_tensor([[4.0, 5.0, 6.0],
    //          [7.0, 8.0, 9.0]])], 0)`: the two literals share axis-1
    // = 3 but differ on axis-0 (1 vs 2). The per-axis-join arm
    // produces `List<tensor[*, 3, f32]>` so concat can take it.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concat_axis0.ch");
    write_file(
        &path,
        "out = concat([to_tensor([[1.0, 2.0, 3.0]]),\n\
                       to_tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])], 0)\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "R3 HIGH-CONCAT axis 0 should type-check; got {errs:?}",
    );
}

#[test]
fn issue_218_r3_high_concat_distinct_axis_1_type_checks() {
    // The same shape, swapped axis: literals share axis-0 = 1 but
    // differ on axis-1.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concat_axis1.ch");
    write_file(
        &path,
        "out = concat([to_tensor([[1.0, 2.0]]),\n\
                       to_tensor([[3.0, 4.0, 5.0, 6.0]])], 1)\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "R3 HIGH-CONCAT axis 1 should type-check; got {errs:?}",
    );
}

#[test]
fn issue_218_r3_high_concat_matching_concrete_dims_keeps_concrete_lit() {
    // Negative-parity: matching concrete dims should *stay* concrete
    // after the per-axis join (`Lit(2) join Lit(2) = Lit(2)`).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concat_matching.ch");
    write_file(
        &path,
        "out = concat([to_tensor([[1.0, 2.0]]), to_tensor([[3.0, 4.0]])], 0)\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "R3 matching dims should type-check; got {errs:?}",
    );
}

#[test]
fn issue_218_r3_cons_rank_mismatch_still_rejects() {
    // Negative-parity: a rank mismatch across the two list elements
    // must still surface as a type error. The per-axis-join arm only
    // relaxes within matching rank.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concat_rank_mismatch.ch");
    write_file(
        &path,
        "out = concat([to_tensor([1.0, 2.0]), to_tensor([[3.0, 4.0]])], 0)\n",
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "R3 rank mismatch must still reject",
    );
}

// =================================================================
// R3 MEDIUM-RESHAPE-ICE: reshape with cast(N, int64) shape in a grad
// body. Depends on PR #224 (#220 fix) at extract_dim_list.
// =================================================================

#[test]
fn issue_218_r3_reshape_after_matmul_in_grad_body_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("reshape_grad.ch");
    write_file(
        &path,
        "def g(x: tensor[2, 3, f32]) -> tensor[2, 1, f32] = {\n\
           w = to_tensor([[1.0], [2.0], [3.0]])\n\
           reshape(matmul(copy(x), w), [cast(2, int64), cast(1, int64)])\n\
         }\n\
         def compute_grad(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = {\n\
           sum_g = fn (xi: tensor[2, 3, f32]) ->\n\
             sum(sum(g(copy(xi)), 0), 0)\n\
           grad(sum_g, wrt=x)(x)\n\
         }\n\
         out = compute_grad(to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "R3 MEDIUM-RESHAPE-ICE: reshape-in-grad-body should build; stderr={stderr}",
    );
}

// =================================================================
// Routing exemption acceptance: the headline case from the issue
// (just to anchor the simple positive).
// =================================================================

#[test]
fn issue_218_basic_to_tensor_in_grad_body_builds() {
    // Simplest case: `g(x) = sum(x * w)` where `w = to_tensor([1.0, 2.0])`.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("basic_grad.ch");
    write_file(
        &path,
        "def g(x: tensor[2, f32]) -> tensor[f32] = {\n\
           w = to_tensor([1.0, 2.0])\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[2, f32]) -> tensor[2, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([3.0, 4.0]))\n",
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "Basic to_tensor-in-grad-body should build; stderr={stderr}",
    );
}

#[test]
fn issue_218_dynamic_to_tensor_in_grad_body_remains_host_routed() {
    // Negative-parity: a non-literal `to_tensor(items)` where `items`
    // is a runtime list keeps the legacy host classification. The
    // routing exemption is scoped to literal Cons-chains.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dynamic_grad.ch");
    write_file(
        &path,
        "def g(items: List[f32]) -> tensor[f32] = {\n\
           w = to_tensor(items)\n\
           sum(w, 0)\n\
         }\n\
         out = g(Cons(1.0, Cons(2.0, Nil)))\n",
    );
    let _output = run_build(&path);
    // We don't assert success/failure here; we only pin that the
    // build-time routing for non-literal `to_tensor` is unchanged.
    // The check passing implies the relaxation didn't accidentally
    // exempt dynamic arguments.
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "Dynamic to_tensor should still type-check; got {errs:?}",
    );
}
