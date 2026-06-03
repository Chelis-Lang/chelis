//! Issue #288: `grad` cannot lower the backward pass through the
//! constant-broadcast idiom `expand(scalar_to_tensor(c), axis, n)`
//! (n > 1).
//!
//! Reproducer (Surf):
//! ```chelis
//! module Repro.GradExpandConst
//! def f(x: tensor[2, f32]) -> f32 = {
//!   k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32))
//!   tensor_to_scalar(sum(mul(x, k), cast(0, int32)))
//! }
//! def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)
//! ```
//!
//! `chelis check` type-checks clean, but `chelis build` fails with
//! `Lowering error: grad(...) lowering rejected: grad: failed to
//! construct backward DAG (unsupported op or verification failure)`.
//!
//! Root cause: in `compute_adjoints`, the `RiscOp::Expand` adjoint
//! always inserts a `Sum` over the expanded axis to reduce the
//! cotangent back to the operand's rank. `scalar_to_tensor(c)` lowers
//! to a rank-0 source (`Tensor(vec![], _)`), so the forward `Expand`
//! is rank-increasing (rank 0 -> rank 1). The cotangent into the
//! Expand adjoint is rank 1; the rank-1 source case (covered by the
//! existing `grad_expand_sum_roundtrip` unit test) reduces cleanly.
//! The rank-0-source case is the regression: a constant
//! `scalar_to_tensor` source has no gradient, so its adjoint
//! contribution should be dropped, and the Expand adjoint must never
//! emit a reduction whose operand would be over-reduced. This file
//! pins the forward DAG shape at the IR level so the fix is exercised
//! by both `grad_dag_checked` (construction) and `eval_tensor`
//! (numeric correctness), with no dependency on the front-end parser
//! or the `chelis` binary.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::{AdError, grad_dag_checked};
use chelis_types::types::Prim;
use std::collections::HashMap;

fn vec_n_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn scalar_f32() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F32,
    }
}

/// Build the issue #288 forward DAG:
///   c   = Cast(Const c_val)                       : f32 (rank 0)  [= scalar_to_tensor(cast(c_val, f32))]
///   k   = Expand{axis:0, size:n}(c)               : tensor[n]     [= expand(c, 0, n)]
///   x   = Load("x")                               : tensor[n]
///   m   = Mul(x, k)                               : tensor[n]     [= mul(x, k)]
///   out = Sum{axis:0}(m)                          : f32 (rank 0)  [= tensor_to_scalar(sum(m, 0))]
///
/// Returns `(dag, x, out)`. `f(x) = sum(x * c_val) = c_val * sum(x)`,
/// so `df/dx = [c_val; n]`.
fn build_expand_scalar_forward(n: usize, c_val: f64) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(n);

    // scalar_to_tensor(cast(c_val, f32)) -> rank-0 f32 constant.
    let raw = dag.add_node(RiscOp::Const { value: c_val }, vec![], scalar_f32(), None);
    let c = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![raw],
        scalar_f32(),
        None,
    );

    // expand(c, axis=0, size=n) -> tensor[n] (rank-INCREASING, rank 0 -> rank 1).
    let k = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(n),
        },
        vec![c],
        vec_ty.clone(),
        None,
    );

    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_ty.clone(), None);
    let m = dag.add_node(RiscOp::Mul, vec![x, k], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default for f32"),
        vec![m],
        scalar_f32(),
        None,
    );
    (dag, x, out)
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch: got {} want {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "{label}: elem {i}: got {g}, want {w}",
        );
    }
}

/// Positive: the forward DAG itself evaluates to the scalar
/// `c_val * sum(x)` — establishes that the *forward* is well-formed and
/// the bug is strictly in the backward construction.
#[test]
fn issue_288_forward_expand_scalar_evaluates() {
    let (dag, _x, out) = build_expand_scalar_forward(2, 2.5);
    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
    let vals = eval_tensor(&dag, &inputs).expect("forward eval must succeed");
    // 2.5 * (3 + 4) = 17.5
    assert_close("forward", &vals[&out].data, &[17.5]);
}

/// Positive (the bug): `grad(f, wrt=x)` must construct cleanly through
/// the `expand(scalar_to_tensor(c), 0, n)` idiom. Before the fix this
/// returns `AdError::NotSupported { op: "<unknown>", reason: Other(...)
/// }` ("failed to construct backward DAG").
#[test]
fn issue_288_grad_through_expand_scalar_constructs() {
    let (dag, x, out) = build_expand_scalar_forward(2, 2.5);
    let result = grad_dag_checked(&dag, out, &[x]);
    match result {
        Ok(_) => {}
        Err(AdError::NotSupported { op, reason }) => panic!(
            "grad through expand(scalar_to_tensor(c), 0, n) must succeed \
             (issue #288); got rejection op={op}, reason={reason:?}",
        ),
    }
}

/// Positive numeric: `df/dx = [c_val; n]`. With `c_val = 2.5`, `n = 2`,
/// `df(x) = [2.5, 2.5]` for any `x`. Asserts the exact gradient the
/// issue's Expected section specifies.
#[test]
fn issue_288_grad_through_expand_scalar_is_correct() {
    let (dag, x, out) = build_expand_scalar_forward(2, 2.5);
    let result = grad_dag_checked(&dag, out, &[x])
        .expect("grad must construct (issue #288)");
    let grad_x = result
        .grad_nodes
        .get(&x)
        .copied()
        .expect("gradient w.r.t. x must be present");
    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("grad DAG eval");
    assert_close("grad_x", &vals[&grad_x].data, &[2.5, 2.5]);
    assert_eq!(vals[&grad_x].shape, vec![2], "grad shape must be tensor[2]");
}

/// Finite-difference cross-check: the analytic gradient must agree with
/// a centered finite difference of the forward, per the backend-numerics
/// discipline. (Evaluator-vs-evaluator agreement here; the C-backend
/// agreement is pinned by the CLI-level build/eval test.)
#[test]
fn issue_288_grad_matches_finite_difference() {
    let (dag, x, out) = build_expand_scalar_forward(2, 2.5);
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];
    let base = TensorValue::from_vec(vec![2], vec![0.7, -1.3]);
    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert("x".into(), base.clone());
    let analytic = eval_tensor(&result.dag, &inputs).expect("analytic eval")[&grad_x]
        .data
        .clone();

    let h = 1e-3;
    let mut numerical = [0.0f64; 2];
    for (j, slot) in numerical.iter_mut().enumerate() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus.data[j] += h;
        minus.data[j] -= h;
        let mut ip = HashMap::new();
        ip.insert("x".into(), plus);
        let mut im = HashMap::new();
        im.insert("x".into(), minus);
        let fp = eval_tensor(&dag, &ip).expect("plus eval")[&out].data[0];
        let fm = eval_tensor(&dag, &im).expect("minus eval")[&out].data[0];
        *slot = (fp - fm) / (2.0 * h);
    }
    for (i, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (a - n).abs() < 1e-3,
            "finite-diff mismatch at {i}: analytic {a}, numerical {n}",
        );
    }
}

// --- Controls: these patterns already differentiate and must keep
// passing. They pin that the fix is scoped to the rank-0-source Expand
// adjoint and does not regress ordinary expand / mul-by-real-tensor /
// sum-of-square gradients. ---

/// Control 1: `sum(mul(x, x), 0)` differentiates. `f(x) = sum(x^2)`,
/// `df/dx = 2x`. No expand-of-scalar on the path.
#[test]
fn issue_288_control_sum_mul_x_x() {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_ty.clone(), None);
    let sq = dag.add_node(RiscOp::Mul, vec![x, x], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![sq],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("control sum(x*x) must differentiate");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, -4.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("control eval");
    assert_close("control_sum_mul_x_x", &vals[&grad_x].data, &[6.0, -8.0]);
}

/// Control 2: `sum(mul(x, t), 0)` against a REAL literal tensor `t`
/// (not expand-of-scalar) differentiates. `f(x) = sum(x .* t)`,
/// `df/dx = t`. `t` is a rank-1 constant whose adjoint Expand source is
/// itself rank 1 — the path the existing roundtrip test exercises.
#[test]
fn issue_288_control_mul_by_real_tensor() {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_ty.clone(), None);
    // A rank-1 constant tensor built as Const+Pad+Add would mirror
    // `to_tensor`; for this control a rank-1 Load standing in for the
    // literal tensor is sufficient to pin the non-scalar-expand path.
    let t = dag.add_node(RiscOp::Load { name: "t".into() }, vec![], vec_ty.clone(), None);
    let m = dag.add_node(RiscOp::Mul, vec![x, t], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![m],
        scalar_f32(),
        None,
    );
    let result =
        grad_dag_checked(&dag, out, &[x]).expect("control mul-by-real-tensor must differentiate");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
    inputs.insert("t".into(), TensorValue::from_vec(vec![2], vec![1.5, -2.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("control eval");
    assert_close("control_mul_by_real_tensor", &vals[&grad_x].data, &[1.5, -2.0]);
}
