//! Issue #320: `grad` backward fails for `mean` / `max_reduce` / `gather`
//! when the reduced/gathered operand is the runtime-derived window produced
//! by a `shrink` / `stride` / `reshape` chain (the #291 follow-up).
//!
//! #291 (PR #301) made the `shrink` and `stride` *movement* adjoints lower
//! cleanly in isolation. The three verbs that COMPOSE a window and then
//! REDUCE / GATHER over it had two distinct failures:
//!
//!   * `mean` -> "mean requires a concrete extent for axis 0 in IR
//!     lowering". `tier2::lower_mean` divided by a compile-time
//!     `Const(axis_size)`, so `require_axis_size` PANICKED whenever the
//!     reduced axis was a runtime-derived `Named(_, None)` extent. THIS
//!     FILE pins the `lower_mean` fix (the divisor becomes a runtime count
//!     `sum(ones_like(x), axis)` for symbolic extents) end to end through
//!     grad + eval.
//!
//!   * `max_reduce` / `gather` -> "axis 0 is out of range for an operand of
//!     rank 0". This is a FRONT-END rank-0 collapse: the windowing/stacking
//!     intermediate lowers to a rank-0 `default_type()` IR node, so
//!     `normalize_axis` sees rank 0. That fix lives in `lower.rs`
//!     (`reduction_operand_rank` / `gather_values_rank` +
//!     `recover_collapsed_operand_type`), and the end-to-end grad+eval
//!     reproducers are the in-module tests
//!     `issue_320_grad_eval_windowed_max_reduce_end_to_end` and
//!     `issue_320_grad_eval_windowed_gather_end_to_end` in `lower.rs` (they
//!     need the front-end lowerer, which is crate-private).
//!
//! This file therefore owns the `mean` end-to-end coverage plus the
//! LITERAL-shape negative-parity regression guards for all three verbs.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::tier2;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        sym_vec("n"),
        None,
    );
    // This call previously PANICKED with "mean requires a concrete extent
    // for axis 0 in IR lowering".
    let mean = tier2::lower_mean(decl.into(), &mut dag, x, 0, &sym_vec("n"), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        sym_vec("n"),
        None,
    );
    let mean = tier2::lower_mean(decl.into(), &mut dag, x, 0, &sym_vec("n"), None);

    // Forward value: mean([10,20,30,40]) = 25.
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    let fwd = eval_tensor(&dag, &inputs).expect("symbolic mean forward eval");
    assert_close(
        "symbolic_mean_value",
        &fwd[&mean].to_f64_lossy_vec(),
        &[25.0],
    );

    // Backward: d/dx mean(x) = 1/n at every slot.
    let result =
        grad_dag_checked(&dag, mean, &[x]).expect("symbolic mean grad must construct (issue #320)");
    let grad_x = result.grad_nodes[&x];
    let vals = eval_tensor(&result.dag, &inputs).expect("symbolic mean grad eval");
    assert_close(
        "grad_symbolic_mean",
        &vals[&grad_x].to_f64_lossy_vec(),
        &[0.25, 0.25, 0.25, 0.25],
    );
    assert_eq!(vals[&grad_x].shape, vec![4]);
}

/// Negative parity: `lower_mean` over a LITERAL-extent operand still builds
/// the compile-time-`Const` divisor and grads exactly. `df/dx = [1/4; 4]`.
#[test]
fn issue_320_lower_mean_over_literal_extent_still_exact() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        lit_vec(4),
        None,
    );
    let mean = tier2::lower_mean(decl.into(), &mut dag, x, 0, &lit_vec(4), None);
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
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("literal mean grad eval");
    assert_close(
        "grad_literal_mean",
        &vals[&grad_x].to_f64_lossy_vec(),
        &[0.25, 0.25, 0.25, 0.25],
    );
}

// =====================================================================
// Literal-shape negative parity for max_reduce / gather.
//
// These pass on `main` regardless of this change (no lower_mean, no
// symbolic dims). They are kept ONLY as regression guards that the reduce/
// gather adjoints still grad+eval over literal shapes; the real #320
// max_reduce/gather coverage is the front-end end-to-end tests in
// `lower.rs` (see this file's module docs).
// =====================================================================

/// Negative parity: `max_reduce` over a LITERAL-shaped operand still grads.
/// `f(x) = max(x)`, `df/dx` = 1 at argmax.
#[test]
fn issue_320_grad_max_reduce_over_literal_shape_still_exact() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        lit_vec(4),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 0 },
        vec![x],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("literal max_reduce grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 9.0, 3.0, 2.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("literal max_reduce grad eval");
    assert_close(
        "grad_literal_max_reduce",
        &vals[&grad_x].to_f64_lossy_vec(),
        &[0.0, 1.0, 0.0, 0.0],
    );
}

/// Negative parity: `gather` over a LITERAL-shaped operand still grads.
/// `f(x) = x0 + x2`, `df/dx = [1, 0, 1, 0]`.
#[test]
fn issue_320_grad_gather_over_literal_shape_still_exact() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        lit_vec(4),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load { name: "idx".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        },
        None,
    );
    let gathered = dag.add_node(
        decl,
        RiscOp::Gather { axis: 0 },
        vec![x, indices],
        lit_vec(2),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![gathered],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("literal gather grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    inputs.insert("idx".into(), TensorValue::from_vec(vec![2], vec![0.0, 2.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("literal gather grad eval");
    assert_close(
        "grad_literal_gather",
        &vals[&grad_x].to_f64_lossy_vec(),
        &[1.0, 0.0, 1.0, 0.0],
    );
}
