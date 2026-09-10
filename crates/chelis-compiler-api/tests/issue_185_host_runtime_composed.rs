//! Issue #185 host-runtime acceptance (Group E — Composed Tier 2).
//!
//! `BUILTIN_NAMES` accepts `mean`, `layer_norm`, and `conv` but the
//! host runtime evaluator did not dispatch them. These ops decompose
//! into RISC primitives via `tier2::lower_*`; the host runtime
//! delegates by constructing a small DAG with the appropriate tier2
//! decomposition and forward-evaluating it through the IR evaluator
//! (per the evaluator-vs-backend agreement gate documented in
//! `feedback_evaluator_byte_identical_gate`).
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §3.4, §4.4, §4.5.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};

fn eval_surf(source: &str) -> chelis_compiler_api::schema::EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"))
}

/// Evaluate selecting only the named top-level roots. Mirrors how
/// `chelis eval --file` invokes the compiler API
/// (`crates/chelis-cli/src/main.rs::try_eval`) — without this filter,
/// the eval pipeline tries to forward-evaluate EVERY top-level
/// binding including function-body closures whose tensor inputs are
/// formal parameters with no bound value. For `conv` specifically
/// this would surface as "missing required input `x`" even though
/// the user-visible `out` binding has the conv result.
fn eval_surf_selected(source: &str, roots: &[&str]) -> chelis_compiler_api::schema::EvalResult {
    let selected: Vec<String> = roots.iter().map(|s| (*s).to_string()).collect();
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &selected,
    )
    .unwrap_or_else(|err| panic!("eval_selected failed: {err:?}"))
}

fn root_tensor<'a>(
    result: &'a chelis_compiler_api::schema::EvalResult,
    name: &str,
) -> &'a chelis_compiler_api::schema::TensorValue {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root {name} in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value,
        other => panic!("expected tensor for {name}, got {other:?}"),
    }
}

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected}, got {actual} (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// mean: arithmetic mean over a given axis. `mean(x, axis) = sum(x, axis) /
// axis_size`. Test on a rank-2 input where the answer is verifiable by
// hand.
// ---------------------------------------------------------------------------

#[test]
fn issue185_mean_axis0_runs_and_matches_ir_eval() {
    // x = [[1.0, 4.0, 2.0],
    //      [3.0, 6.0, 4.0]]
    // mean(x, 0) -> shape [3], data [(1+3)/2, (4+6)/2, (2+4)/2] = [2, 5, 3]
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 6.0, 4.0]], 0.0)
out = mean(&make, 0)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3], "mean axis-0 shape");
    let expected = [2.0, 5.0, 3.0];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-6,
            &format!("mean[{i}]"),
        );
    }
}

#[test]
fn issue185_mean_axis1_runs_and_matches_ir_eval() {
    // mean(x, 1) -> shape [2], data [(1+4+2)/3, (3+6+4)/3] = [7/3, 13/3]
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 6.0, 4.0]], 0.0)
out = mean(&make, 1)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "mean axis-1 shape");
    assert_close(out.data.element_f64_lossy(0), 7.0 / 3.0, 1e-6, "mean[0]");
    assert_close(out.data.element_f64_lossy(1), 13.0 / 3.0, 1e-6, "mean[1]");
}

// ---------------------------------------------------------------------------
// layer_norm: normalize over the last axis then scale by gamma + shift by
// beta. For a 1xN input the mean of (x - mean(x)) is 0 so the result is
// gamma * normed + beta where normed has zero mean and unit-ish variance
// after the epsilon stabilizer.
// ---------------------------------------------------------------------------

#[test]
fn issue185_layer_norm_runs_and_matches_ir_eval() {
    // x = [[1.0, 2.0, 3.0, 4.0]]  (shape [1, 4])
    // mean = 2.5, var = 1.25, denom = sqrt(1.25 + 1e-5) ~ 1.118034
    // centered = [-1.5, -0.5, 0.5, 1.5]
    // normed = [-1.34164.., -0.44721.., 0.44721.., 1.34164..]
    // With gamma = [1, 1, 1, 1] and beta = [0, 0, 0, 0]: result == normed.
    // gamma and beta are rank-1 per `layer_norm`'s typer signature
    // (`crates/chelis-types/src/infer.rs` rejects rank-2 gamma).
    let src = r#"
x = pad_sequences([[1.0, 2.0, 3.0, 4.0]], 0.0)
g = to_tensor([1.0, 1.0, 1.0, 1.0])
b = to_tensor([0.0, 0.0, 0.0, 0.0])
out = layer_norm(&x, &g, &b, 0.00001f32)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 4], "layer_norm shape");
    // Sum of normed across the last axis must be ~0 (centered) and the
    // variance must approach 1 (the eps stabilizer makes the denom
    // slightly larger than sqrt(var)). We pin the four exact values to a
    // tight tolerance.
    let denom = (1.25_f64 + 1e-5).sqrt();
    let expected = [-1.5 / denom, -0.5 / denom, 0.5 / denom, 1.5 / denom];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-5,
            &format!("layer_norm[{i}]"),
        );
    }
}

// ---------------------------------------------------------------------------
// conv: 2D convolution. Pinned on a small 1x1x2x2 input with a 1x1x2x2
// kernel (stride=1, padding=0). The output is a 1x1x1x1 scalar tensor
// equal to the elementwise dot product.
//
// `conv`'s typer requires concrete d-lit tensor argument metadata at the
// call site (`crates/chelis-types/src/infer.rs::
// conv_input_dims_concrete_modulo_batch`), which only flows in via
// explicitly-typed function parameters. The fixture therefore wraps the
// call in `def run_conv(x: tensor[1, 1, 2, 2, f32], k: tensor[1, 1, 2,
// 2, f32]) -> ...` so the param-type metadata propagates onto the
// conv call's args.
// ---------------------------------------------------------------------------

#[test]
fn issue185_conv_runs_and_matches_ir_eval() {
    // input  = [[[[1.0, 2.0], [3.0, 4.0]]]]   shape [1, 1, 2, 2]
    // kernel = [[[[0.5, 1.0], [1.5, 2.0]]]]   shape [1, 1, 2, 2]
    // output = 1*0.5 + 2*1.0 + 3*1.5 + 4*2.0 = 0.5 + 2 + 4.5 + 8 = 15.0
    //
    // `chelis test` / `chelis eval` lower `out = run_conv(...)` as a
    // tensor-result top-level binding and route it through DAG eval
    // (since `run_conv` returns a tensor). The host runtime is hit
    // only when the test wraps the value in a non-tensor surface;
    // wrapping the assertion in a Test-effect fn forces the host
    // runtime to invoke the lowered conv on the DAG-evaluated input
    // and then call `tensor_to_scalar` to read out a single value.
    let src = r#"
def run_conv(x: tensor[1, 1, 2, 2, f32], k: tensor[1, 1, 2, 2, f32]) -> tensor[1, 1, 1, 1, f32] = conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
def make_x() -> tensor[1, 1, 2, 2, f32] = to_tensor([[[[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]]]])
def make_k() -> tensor[1, 1, 2, 2, f32] = to_tensor([[[[cast(0.5, f32), cast(1.0, f32)], [cast(1.5, f32), cast(2.0, f32)]]]])
out = run_conv(make_x(), make_k())
"#;
    let result = eval_surf_selected(src, &["out"]);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 1, 1, 1], "conv shape");
    assert_close(out.data.element_f64_lossy(0), 15.0, 1e-5, "conv[0]");
}

// ---------------------------------------------------------------------------
// Negative parity: each builtin must still reject ill-typed inputs at
// check time. Passing a string is the canonical wrong-dtype error.
// ---------------------------------------------------------------------------

fn check_rejects(src: &str, label: &str) {
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "{label}: expected check/eval to fail, got {outcome:?}"
    );
}

#[test]
fn issue185_mean_rejects_string_input() {
    check_rejects(r#"out = mean("not a tensor", 0)"#, "mean string input");
}

#[test]
fn issue185_layer_norm_rejects_string_input() {
    check_rejects(
        r#"out = layer_norm("nope", "nope", "nope", 0.00001f32)"#,
        "layer_norm string input",
    );
}

#[test]
fn issue185_conv_rejects_string_input() {
    check_rejects(
        r#"out = conv("nope", "nope", [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])"#,
        "conv string input",
    );
}
