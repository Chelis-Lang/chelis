//! Issue chelis#345 — 0.7.24 regression: `grad` through a symbolic-dim
//! wrapper ICEd in the IR-to-backend handoff with
//! `internal compiler error: symbolic dim `dN` is referenced by a
//! non-Load node (id 1, op Const { value: 0.0 } ...)`.
//!
//! Mechanism (verified by the bisect in the issue thread): PR #326's
//! `annotate_fn_children` seeds separate-`sig` def bodies with the
//! declared param types, minting fresh checker dim vars (`dN`) into the
//! body's `type:` metadata. When `grad(f)` inlines the wrapper,
//! `type_from_meta` applied precision and rank substitutions but not
//! dim substitutions, so backward-pass nodes synthesized from the
//! annotated type (relu's `Const 0.0` mask, gelu's tanh-approx
//! constant) carried `Named("dN")` dims that nothing ever rebound to
//! the concrete call-site extent. The `symbolic_occurrences` guard in
//! `chelis-ir/src/dag.rs` then panicked.
//!
//! The producer was retired by PR #346: every elementwise lowering arm
//! (relu/gelu/sigmoid/tanh/silu/neg/recip/sub/div + transcendentals)
//! now derives its output type from the lowered OPERAND
//! (`elementwise_out_ty`), not the body annotation, so the backward
//! masks inherit the concrete inlined dims. This file pins the full
//! wrapper-declaration matrix from the issue thread so the family
//! cannot regress silently again:
//!
//! - row A — separate-`sig` wrapper (the 0.7.24 regression)
//! - row B — bare inline-annotated wrapper (latent: ICEd on 0.7.23 too)
//! - row C — explicit-quantifier wrapper (latent, same as B)
//! - row D — `[n]`-quantified shim composing a sig-form verb (green on
//!   0.7.23, ICEd on 0.7.24)
//! - row E — Tier-2 `..r` rank-polymorphic wrapper (green on 0.7.24;
//!   must stay green — call-site rank monomorphization is the working
//!   model the dim-var routes were missing)
//! - gelu and a rank-2 wrapper — the non-relu / non-rank-1 members of
//!   the same family from the issue report
//!
//! Per the backend-numerics discipline each positive case runs BOTH
//! lanes: `chelis eval`, and `chelis build --target c` followed by a
//! native compile and execute, asserting the analytic gradient on each
//! lane and cross-lane agreement. The op-internal half of the issue (the guard's
//! `Expand { size: Sym(_) }` blind spot) is owned by the Bucket 4d
//! sweep unit tests in `crates/chelis-ir/src/dag.rs`; the compile step
//! here doubles as the end-to-end "no undeclared identifier reaches
//! cc" oracle.

use assert_cmd::Command;
use serde_json::Value;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::{link_generated, parse_tensor_data, write_file};

/// d/dx sum(relu(x)) at x = [2, -1] is the step function [1, 0].
const RELU_INPUT: &str = "to_tensor([cast(2.0, f32), cast(-1.0, f32)])";
const RELU_GRAD: [f64; 2] = [1.0, 0.0];

/// Row A — the issue's verbatim shape: separate `sig` carrying the
/// symbolic dim, bare `def`, concrete `tensor[2, f32]` call site.
/// Keeps the issue reproducer's `cast(0, i32)` axis form.
fn row_a_sig_form() -> String {
    format!(
        "module Repro.Issue345RowA\n\
         sig relu_fwd[a]: tensor[a, f32] -> tensor[a, f32]\n\
         def relu_fwd(x) = relu(x)\n\
         def f(x: tensor[2, f32]) -> f32 = sum(relu_fwd(x), cast(0, i32)) |> tensor_to_scalar\n\
         out = grad(f)({RELU_INPUT})\n"
    )
}

/// Row B — bare contextual inline annotation. Latent sibling: this
/// form ICEd on 0.7.23 as well (the sig form only survived 0.7.23 via
/// the accidental operand-type fallback).
fn row_b_inline_form() -> String {
    format!(
        "module Repro.Issue345RowB\n\
         def relu_fwd[a](x: tensor[a, f32]) -> tensor[a, f32] = relu(x)\n\
         def f(x: tensor[2, f32]) -> f32 = sum(relu_fwd(x), 0) |> tensor_to_scalar\n\
         out = grad(f)({RELU_INPUT})\n"
    )
}

/// Row C — explicit def-level quantifier `[a]`.
fn row_c_quantifier_form() -> String {
    format!(
        "module Repro.Issue345RowC\n\
         def relu_fwd[a](x: tensor[a, f32]) -> tensor[a, f32] = relu(x)\n\
         def f(x: tensor[2, f32]) -> f32 = sum(relu_fwd(x), 0) |> tensor_to_scalar\n\
         out = grad(f)({RELU_INPUT})\n"
    )
}

/// Row D — `[n]`-quantified shim whose body composes a sig-form verb
/// (the downstream School idiom that masked the latent rows).
fn row_d_shim_form() -> String {
    format!(
        "module Repro.Issue345RowD\n\
         sig relu_sig[a]: tensor[a, f32] -> tensor[a, f32]\n\
         def relu_sig(x) = relu(x)\n\
         def shim[n](x: tensor[n, f32]) -> tensor[n, f32] = relu_sig(x)\n\
         def f(x: tensor[2, f32]) -> f32 = sum(shim(x), 0) |> tensor_to_scalar\n\
         out = grad(f)({RELU_INPUT})\n"
    )
}

/// Row E — Tier-2 `..r` rank-polymorphic wrapper. Green on 0.7.24 via
/// call-site rank monomorphization; pinned must-stay-green because the
/// dim-var rows share the `type_from_meta` extraction plumbing.
fn row_e_rank_poly_form() -> String {
    format!(
        "module Repro.Issue345RowE\n\
         def relu_rp[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n\
         def f(x: tensor[2, f32]) -> f32 = sum(relu_rp(x), 0) |> tensor_to_scalar\n\
         out = grad(f)({RELU_INPUT})\n"
    )
}

/// gelu_tanh sibling — the backward materializes the tanh-approx constant
/// `Const 0.7978845608028654` at the annotated type, the second node
/// kind the issue reported (rank-1, sig form).
fn gelu_sig_form() -> String {
    format!(
        "module Repro.Issue345Gelu\n\
         sig gelu_fwd[a]: tensor[a, f32] -> tensor[a, f32]\n\
         def gelu_fwd(x) = gelu_tanh(x)\n\
         def f(x: tensor[2, f32]) -> f32 = sum(gelu_fwd(x), 0) |> tensor_to_scalar\n\
         out = grad(f)({RELU_INPUT})\n"
    )
}

/// d/dx sum(gelu_tanh(x)) at [2, -1] under the tanh approximation, as
/// printed by the host evaluator (the numeric oracle both lanes must
/// match within f32 noise).
const GELU_GRAD: [f64; 2] = [1.0860992566236183, -0.08296408384578256];

/// Rank-2 sig wrapper — two minted dim vars at once (School's
/// `tests/grad/act.ch` rank-2 failure shape).
fn rank2_sig_form() -> String {
    "module Repro.Issue345Rank2\n\
     sig relu_fwd2[a, b]: tensor[a, b, f32] -> tensor[a, b, f32]\n\
     def relu_fwd2(x) = relu(x)\n\
     def f(x: tensor[2, 3, f32]) -> f32 = sum(sum(relu_fwd2(x), 1), 0) |> tensor_to_scalar\n\
     out = grad(f)(to_tensor([[cast(2.0, f32), cast(-1.0, f32), cast(0.5, f32)], [cast(-3.0, f32), cast(4.0, f32), cast(-0.5, f32)]]))\n"
        .to_string()
}

const RANK2_GRAD: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

/// Parse the evaluator's `name = tensor(shape=[..], data=[..])` print.
/// Issue #912 [05-OBS-6]: single roots are now labelled.
fn parse_eval_tensor_data(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find(|l| l.contains("tensor("))
        .unwrap_or_else(|| panic!("eval output has no tensor(...) line:\n{stdout}"));
    let data_marker = "data=[";
    let start = line
        .find(data_marker)
        .unwrap_or_else(|| panic!("no `data=[` in line: {line}"))
        + data_marker.len();
    let end = line[start..]
        .find(']')
        .unwrap_or_else(|| panic!("no closing `]` after data: {line}"));
    line[start..start + end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric"))
        .collect()
}

fn assert_grad_matches(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: expected length {} got {}; actual={actual:?}",
        expected.len(),
        actual.len(),
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{msg}: element {i} mismatch: actual={a} expected={e} diff={diff} tol={tol}",
        );
    }
}

/// Lane 1: `chelis eval --file` must succeed (no `symbolic dim ...
/// referenced by a non-Load node` ICE) and print the analytic gradient.
fn eval_gradient(source: &str, stem: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{stem}: `chelis eval` must succeed (issue #345 ICEd here); \
         stdout={stdout} stderr={stderr}",
    );
    assert!(
        !stderr.contains("symbolic dim"),
        "{stem}: eval stderr must not carry the #345 guard ICE; stderr={stderr}",
    );
    parse_eval_tensor_data(&stdout)
}

/// Lane 2: `chelis build --target c`, native-compile the emitted
/// kernel, run it, and return the printed gradient. The compile step
/// is itself an oracle: an unsubstituted `dN` (in an output type OR an
/// op-internal field like `Expand::size`) emits an undeclared C
/// identifier and the native compiler rejects it.
fn build_and_run_gradient(source: &str, stem: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source_file = format!("{stem}.c");
    let status = link_generated(&out_dir, &source_file, stem);
    assert!(
        status.success(),
        "{stem}: emitted C must compile cleanly (an unsubstituted `dN` \
         would be an undeclared identifier): {status}",
    );

    let run_output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "{stem}: compiled gradient binary failed: {}\nstderr: {}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr),
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    parse_tensor_data(&stdout, "out")
}

/// Both lanes for one matrix row: analytic value on each lane plus
/// cross-lane agreement.
fn assert_row(source: &str, stem: &str, expected: &[f64]) {
    let eval = eval_gradient(source, stem);
    assert_grad_matches(&eval, expected, 1e-4, &format!("{stem}: eval lane"));
    let compiled = build_and_run_gradient(source, stem);
    assert_grad_matches(&compiled, expected, 1e-4, &format!("{stem}: compiled lane"));
    assert_grad_matches(
        &compiled,
        &eval,
        1e-4,
        &format!("{stem}: eval-vs-compiled agreement"),
    );
}

#[test]
fn issue_345_row_a_sig_form_grad_is_step() {
    assert_row(&row_a_sig_form(), "issue345_row_a", &RELU_GRAD);
}

#[test]
fn issue_345_row_b_inline_form_grad_is_step() {
    assert_row(&row_b_inline_form(), "issue345_row_b", &RELU_GRAD);
}

#[test]
fn issue_345_row_c_quantifier_form_grad_is_step() {
    assert_row(&row_c_quantifier_form(), "issue345_row_c", &RELU_GRAD);
}

#[test]
fn issue_345_row_d_shim_over_sig_verb_grad_is_step() {
    assert_row(&row_d_shim_form(), "issue345_row_d", &RELU_GRAD);
}

#[test]
fn issue_345_row_e_rank_poly_wrapper_stays_green() {
    assert_row(&row_e_rank_poly_form(), "issue345_row_e", &RELU_GRAD);
}

#[test]
fn issue_345_gelu_sig_form_grad_matches_eval_oracle() {
    assert_row(&gelu_sig_form(), "issue345_gelu", &GELU_GRAD);
}

#[test]
fn issue_345_rank2_sig_form_grad_is_step_mask() {
    assert_row(&rank2_sig_form(), "issue345_rank2", &RANK2_GRAD);
}

/// Negative parity: the symbolic-dim sig still ENFORCES dimension
/// consistency at concrete call sites. Routing `tensor[2, f32]`
/// through `tensor[a, f32] -> tensor[a, f32]` cannot produce
/// `tensor[3, f32]`; the checker must reject it as a signature
/// mismatch (a type error, never the #345 ICE).
#[test]
fn issue_345_negative_sig_dim_mismatch_is_type_error_not_ice() {
    let source = "module Repro.Issue345Neg\n\
         sig relu_fwd[a]: tensor[a, f32] -> tensor[a, f32]\n\
         def relu_fwd(x) = relu(x)\n\
         def f(x: tensor[2, f32]) -> tensor[3, f32] = relu_fwd(x)\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("issue345_neg.ch");
    write_file(&path, source);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let json: Value = serde_json::from_slice(&output.stdout).expect("check output should be json");
    let messages: Vec<String> = json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        !messages.is_empty(),
        "mismatched concrete dims through the symbolic sig must be a \
         check error; got none",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("doesn't match declared signature")),
        "expected a signature-mismatch diagnostic, got: {messages:?}",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("symbolic dim") && !stderr.contains("internal compiler error"),
        "negative case must fail as a TYPE error, not the #345 ICE; stderr={stderr}",
    );
}

#[test]
fn issue_345_negative_sig_dim_mismatch_via_eval_lane_is_type_error_not_ice() {
    // Review #363 N4: the check-only negative fires pre-lowering by
    // construction; this variant goes through the SAME eval entry point
    // as the positive rows, so a future regression that lets the
    // mismatch reach grad lowering trips this assert instead of hiding
    // behind the check-stage pin. The root would exercise lowering if
    // the checker ever stopped rejecting.
    let source = "module Repro.Issue345NegEval\n\
         sig relu_fwd[a]: tensor[a, f32] -> tensor[a, f32]\n\
         def relu_fwd(x) = relu(x)\n\
         def f(x: tensor[2, f32]) -> tensor[3, f32] = relu_fwd(x)\n\
         out = f(to_tensor([1.0, 2.0]))\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("issue345_neg_eval.ch");
    write_file(&path, source);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "mismatched concrete dims through the symbolic sig must fail under eval",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("doesn't match declared signature"),
        "expected the signature-mismatch reason on the eval lane, got: {stderr}",
    );
    assert!(
        !stderr.contains("symbolic dim") && !stderr.contains("internal compiler error"),
        "eval-lane negative must fail as a TYPE error, not the #345 ICE; stderr={stderr}",
    );
}
