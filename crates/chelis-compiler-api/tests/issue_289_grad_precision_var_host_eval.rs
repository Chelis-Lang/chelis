//! Issue Chelis-Lang/chelis#289 — host-eval reproducer for `grad` over
//! a precision-polymorphic callee.
//!
//! The issue's "Host eval" failure mode panics with
//! `BUG: monomorphization missed precision var p` when `grad` lowers a
//! function whose callee carries a `tensor[..., p]` precision variable,
//! even though the differentiated entry point is fully concrete `f32`.
//!
//! These tests drive the same `chelis_compiler_api::compiler::eval`
//! host pipeline the issue's host-eval used, so they reproduce the panic
//! directly (no C compilation / binary spawn). They are the fast oracle
//! that runs in the default workspace nextest pass; the CLI numeric
//! suite (`crates/chelis-cli/tests/issue_289_grad_precision_var.rs`)
//! additionally pins the compiled-C gradient values.
//!
//! Numeric contract: `loss(x, w) = sum(x * w)`, so the gradient
//! w.r.t. `x` is `w` and w.r.t. `w` is `x`.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};

fn try_eval(source: &str) -> Result<chelis_compiler_api::schema::EvalResult, String> {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .map_err(|err| format!("{err:?}"))
}

fn eval_surf(source: &str) -> chelis_compiler_api::schema::EvalResult {
    try_eval(source).unwrap_or_else(|err| panic!("eval failed: {err}"))
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

fn reproducer_source(proj: &str) -> String {
    format!(
        "def lin_p[p: Numeric](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] = mul(x, w)\n\
         def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(lin_p(x, w), cast(0, i32)))\n\
         def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] =\n\
           (grad(loss)(x, w)).{proj}\n\
         out = dloss(to_tensor([3.0, 4.0], f32), to_tensor([5.0, 6.0], f32))\n",
    )
}

// =====================================================================
// Positive: host eval differentiates through the precision-var callee.
// =====================================================================

#[test]
fn issue289_host_eval_grad_through_precision_var_callee_dx_equals_w() {
    let result = eval_surf(&reproducer_source("0"));
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "issue #289 d/dx shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![5.0, 6.0],
        "issue #289 d/dx = w"
    );
}

#[test]
fn issue289_host_eval_grad_through_precision_var_callee_dw_equals_x() {
    let result = eval_surf(&reproducer_source("1"));
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "issue #289 d/dw shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![3.0, 4.0],
        "issue #289 d/dw = x"
    );
}

// =====================================================================
// Control note: the inline-f32 control (no precision-var callee) is
// covered end-to-end — compiled C, run, and gradient-checked — by
// `crates/chelis-cli/tests/issue_289_grad_precision_var.rs`'s
// `issue_289_control_inline_f32_callee_dx_equals_w`. It is not duplicated
// here because the fully-concrete `dloss` def evaluated through this
// host-eval entry surfaces a pre-existing, #289-unrelated root-selection
// quirk ("missing required input `x`") when every def is concrete-lowered
// as a standalone root. The positive precision-var cases above exercise
// the host-eval path the issue's reproducer used.
// =====================================================================

// =====================================================================
// Negative parity: grad over a genuinely under-determined precision
// (polymorphic callee with no concrete call site) must NOT panic with
// the internal monomorphization-BUG string. A clean error result is
// acceptable; an internal panic is not.
// =====================================================================

#[test]
fn issue289_host_eval_polymorphic_no_site_does_not_internal_panic() {
    let src = "def lin_p[p: Numeric](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] = mul(x, w)\n\
         def loss[p: Float](x: tensor[2, p], w: tensor[2, p]) -> tensor[p] =\n\
           sum(lin_p(x, w), cast(0, i32))\n\
         def dloss[p: Float](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] =\n\
           (grad(loss)(x, w)).0\n";
    // No `out =` call site pins `p`. The fix concretizes only when a
    // concrete call-site precision exists, so this stays under-determined
    // and must surface a clean error (or simply produce no tensor root),
    // never the internal `BUG: monomorphization missed precision var`
    // panic.
    let outcome = try_eval(src);
    if let Err(message) = &outcome {
        assert!(
            !message.contains("BUG: monomorphization missed precision var"),
            "issue #289 negative parity: under-determined precision must \
             yield a clean diagnostic, not the internal monomorphization-BUG \
             panic; got {message}",
        );
    }
}
