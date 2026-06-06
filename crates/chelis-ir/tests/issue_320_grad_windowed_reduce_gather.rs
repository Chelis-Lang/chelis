//! Issue #320: `grad` backward fails for `mean` / `max_reduce` / `gather`
//! when the reduced/gathered operand's shape is the runtime-derived shape
//! produced by a `shrink` / `stride` / `reshape` window (the #291
//! follow-up).
//!
//! #291 (PR #301) made the `shrink` and `stride` *movement* adjoints lower
//! cleanly in isolation: `grad(sum(shrink(x, [0,2]))) = [1,1,0,0]` and
//! `grad(sum(stride(x, 2))) = [1,0,1,0]` construct and evaluate. Those
//! adjoints read the forward operand's rank/extent from its `output_type`,
//! and the runtime evaluator binds any symbolic axes from the input shape
//! (`bind_symbolic_dims`).
//!
//! But verbs that COMPOSE a windowing op and then REDUCE or GATHER over the
//! windowed view still failed because they did not carry the operand's
//! runtime-derived extent through to IR lowering / the adjoint:
//!   * `mean`       -> "mean requires a concrete extent for axis 0 ..."
//!     `lower_mean` divided by a compile-time `Const(axis_size)` and so
//!     `require_axis_size` PANICKED whenever the reduced axis was a
//!     runtime-derived `Named(_, None)` dim instead of a literal.
//!   * `max_reduce` / `gather` carry the operand's rank/extent through the
//!     reverse-mode adjoint; this file pins that they grad+eval over a
//!     runtime-derived (symbolic) operand, the same property #291 gave the
//!     bare windowing ops.
//!
//! Each verb grads fine on its own over a LITERAL-shaped tensor — the
//! negative-parity tests below keep that path green.
//!
//! The symbolic axis here is `n`, carried by the `x` load, so the runtime
//! evaluator binds it from the input tensor's shape — the same
//! `symbolic_bindings` machinery the rest of the IR relies on.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::{AdError, grad_dag_checked};
use chelis_ir::tier2;
use chelis_types::types::Prim;
use std::collections::HashMap;

fn f32_dims(dims: Vec<DimInfo>) -> TensorType {
    TensorType {
        dims,
        precision: Prim::F32,
    }
}

fn lit_vec(n: usize) -> TensorType {
    f32_dims(vec![DimInfo::Lit(n)])
}

fn scalar_f32() -> TensorType {
    f32_dims(vec![])
}

/// A runtime-derived (symbolic, unsized) 1-D f32 vector `tensor[name, f32]`.
fn sym_vec(name: &str) -> TensorType {
    f32_dims(vec![DimInfo::Named(name.into(), None)])
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
        assert!((g - w).abs() < 1e-5, "{label}: elem {i}: got {g}, want {w}");
    }
}

// =====================================================================
// MEAN over a runtime-derived (symbolic-extent) operand.
//
// `tier2::lower_mean` is the exact code that emitted
// "mean requires a concrete extent for axis 0 in IR lowering". These pin
// the fix at the lowering boundary and end to end through grad + eval.
// =====================================================================

/// Lowering (the bug): `lower_mean` over a `tensor[n, f32]` operand whose
/// reduced axis 0 is the runtime-derived `Named("n", None)` MUST lower
/// without panicking. Before the fix it called `require_axis_size`, which
/// panicked on the non-concrete extent. After the fix the divisor is a
/// runtime count (`sum(ones_like(x), axis)`), so lowering succeeds.
#[test]
fn issue_320_lower_mean_over_symbolic_extent_does_not_panic() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        sym_vec("n"),
        None,
    );
    // This call previously PANICKED with "mean requires a concrete extent
    // for axis 0 in IR lowering".
    let mean = tier2::lower_mean(&mut dag, x, 0, &sym_vec("n"), None);
    // The lowered mean must reduce to a scalar (axis 0 of a rank-1 operand).
    assert!(
        dag.get(mean)
            .expect("mean node")
            .output_type
            .dims
            .is_empty(),
        "mean over the only axis must reduce to a scalar",
    );
    // The divisor must be a RUNTIME count: a Sum over a Const-of-ones, NOT a
    // baked-in literal `Const(axis_size)`. Two Sum nodes (value + count)
    // must therefore be present.
    let sum_count = dag
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::Sum { .. }))
        .count();
    assert_eq!(
        sum_count, 2,
        "symbolic-extent mean must build a runtime count via a second Sum \
         (value sum + ones sum), got {sum_count} Sum nodes",
    );
}

/// End to end: `mean(x)` over the runtime-derived axis evaluates to the
/// arithmetic mean, and its grad is `[1/n; n]`. Pins both forward value and
/// the carried-extent backward.
#[test]
fn issue_320_grad_through_symbolic_mean_is_exact() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        sym_vec("n"),
        None,
    );
    let mean = tier2::lower_mean(&mut dag, x, 0, &sym_vec("n"), None);

    // Forward value: mean([10,20,30,40]) = 25.
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    let fwd = eval_tensor(&dag, &inputs).expect("symbolic mean forward eval");
    assert_close("symbolic_mean_value", &fwd[&mean].data, &[25.0]);

    // Backward: d/dx mean(x) = 1/n at every slot.
    let result =
        grad_dag_checked(&dag, mean, &[x]).expect("symbolic mean grad must construct (issue #320)");
    let grad_x = result.grad_nodes[&x];
    let vals = eval_tensor(&result.dag, &inputs).expect("symbolic mean grad eval");
    assert_close(
        "grad_symbolic_mean",
        &vals[&grad_x].data,
        &[0.25, 0.25, 0.25, 0.25],
    );
    assert_eq!(vals[&grad_x].shape, vec![4]);
}

/// Negative parity: `lower_mean` over a LITERAL-extent operand still builds
/// the compile-time-`Const` divisor and grads exactly. `df/dx = [1/4; 4]`.
#[test]
fn issue_320_lower_mean_over_literal_extent_still_exact() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], lit_vec(4), None);
    let mean = tier2::lower_mean(&mut dag, x, 0, &lit_vec(4), None);
    // Literal extent keeps the single-Sum (value) form with a Const divisor.
    let sum_count = dag
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::Sum { .. }))
        .count();
    assert_eq!(
        sum_count, 1,
        "literal-extent mean must keep the compile-time Const divisor \
         (one Sum), got {sum_count} Sum nodes",
    );
    let result = grad_dag_checked(&dag, mean, &[x]).expect("literal mean grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("literal mean grad eval");
    assert_close(
        "grad_literal_mean",
        &vals[&grad_x].data,
        &[0.25, 0.25, 0.25, 0.25],
    );
}

// =====================================================================
// MAX_REDUCE over a runtime-derived (symbolic) operand.
//
// The reduced axis is `Named("n", None)` (runtime-derived). The adjoint
// must carry that extent through `Expand`; the evaluator binds `n` from the
// input shape. Pins the issue's "carry the forward operand's rank/extent
// even when those dims are runtime-derived" requirement for max_reduce.
// =====================================================================

/// Positive (the bug): `grad(max_reduce(window, 0))` over a runtime-derived
/// operand must CONSTRUCT and route the subgradient to the argmax slot.
/// `f(x) = max(x)`; `df/dx` is 1 at the unique argmax, 0 elsewhere.
#[test]
fn issue_320_grad_through_symbolic_max_reduce_is_exact() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        sym_vec("n"),
        None,
    );
    // `stride(x, 1)` is an identity window producing a runtime-derived
    // `tensor[n, f32]` view — the operand whose reduced axis is NOT a
    // literal. This is the #320 windowed-operand condition.
    let window = dag.add_node(
        RiscOp::Stride { strides: vec![1] },
        vec![x],
        sym_vec("n"),
        None,
    );
    let out = dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![window],
        scalar_f32(),
        None,
    );

    match grad_dag_checked(&dag, out, &[x]) {
        Ok(_) => {}
        Err(AdError::NotSupported { op, reason }) => panic!(
            "grad through max_reduce over a windowed operand must succeed (issue #320); \
             got rejection op={op}, reason={reason:?}",
        ),
    }
    let result =
        grad_dag_checked(&dag, out, &[x]).expect("symbolic max_reduce grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 9.0, 3.0, 2.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("symbolic max_reduce grad eval");
    assert_close(
        "grad_symbolic_max_reduce",
        &vals[&grad_x].data,
        &[0.0, 1.0, 0.0, 0.0],
    );
    assert_eq!(vals[&grad_x].shape, vec![4]);
}

/// Negative parity: `max_reduce` over a LITERAL-shaped operand still grads.
/// `f(x) = max(x)`, `df/dx` = 1 at argmax.
#[test]
fn issue_320_grad_max_reduce_over_literal_shape_still_exact() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], lit_vec(4), None);
    let out = dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![x], scalar_f32(), None);
    let result = grad_dag_checked(&dag, out, &[x]).expect("literal max_reduce grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 9.0, 3.0, 2.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("literal max_reduce grad eval");
    assert_close(
        "grad_literal_max_reduce",
        &vals[&grad_x].data,
        &[0.0, 1.0, 0.0, 0.0],
    );
}

// =====================================================================
// GATHER over a runtime-derived (symbolic) operand.
//
// The `values` operand axis is `Named("n", None)`. The adjoint scatter-adds
// the cotangent back into the gathered source slots; the evaluator binds `n`
// from the input shape. Pins the issue's carry-rank/extent requirement for
// gather.
// =====================================================================

/// Positive (the bug): `grad(sum(gather(window, idx, 0)))` over a
/// runtime-derived `values` operand must CONSTRUCT and scatter-add the
/// cotangent. `f(x) = x0 + x2` (gather slots 0 and 2), so
/// `df/dx = [1, 0, 1, 0]`.
#[test]
fn issue_320_grad_through_symbolic_gather_is_exact() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        sym_vec("n"),
        None,
    );
    let window = dag.add_node(
        RiscOp::Stride { strides: vec![1] },
        vec![x],
        sym_vec("n"),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        },
        None,
    );
    let gathered = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![window, indices],
        lit_vec(2),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![gathered],
        scalar_f32(),
        None,
    );

    match grad_dag_checked(&dag, out, &[x]) {
        Ok(_) => {}
        Err(AdError::NotSupported { op, reason }) => panic!(
            "grad through gather over a windowed operand must succeed (issue #320); \
             got rejection op={op}, reason={reason:?}",
        ),
    }
    let result = grad_dag_checked(&dag, out, &[x]).expect("symbolic gather grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    inputs.insert("idx".into(), TensorValue::from_vec(vec![2], vec![0.0, 2.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("symbolic gather grad eval");
    assert_close(
        "grad_symbolic_gather",
        &vals[&grad_x].data,
        &[1.0, 0.0, 1.0, 0.0],
    );
    assert_eq!(vals[&grad_x].shape, vec![4]);
}

/// Negative parity: `gather` over a LITERAL-shaped operand still grads.
/// `f(x) = x0 + x2`, `df/dx = [1, 0, 1, 0]`.
#[test]
fn issue_320_grad_gather_over_literal_shape_still_exact() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], lit_vec(4), None);
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        },
        None,
    );
    let gathered = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![x, indices],
        lit_vec(2),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![gathered],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("literal gather grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    inputs.insert("idx".into(), TensorValue::from_vec(vec![2], vec![0.0, 2.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("literal gather grad eval");
    assert_close(
        "grad_literal_gather",
        &vals[&grad_x].data,
        &[1.0, 0.0, 1.0, 0.0],
    );
}
