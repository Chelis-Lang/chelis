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
//! `chelis check` type-checks clean, but `chelis build --target c`
//! fails with `Lowering error: grad(...) lowering rejected: failed to
//! construct backward DAG (...)`.
//!
//! Root cause (confirmed from the verifier diagnostics on the static
//! build path): the `RiscOp::Expand` adjoint in `compute_adjoints`
//! reduced the cotangent with a single `Sum { axis }` and labeled its
//! output with the SOURCE type for *every* expand shape. A `RiscOp::
//! Expand` has two shapes (`verify.rs` C10):
//!
//!   * RANK-INCREASING (`output_rank == source_rank + 1`): a `Sum` over
//!     the inserted axis correctly recovers the source rank — fine.
//!   * SAME-RANK (`output_rank == source_rank`, size-1 axis broadcast to
//!     n): a `Sum { axis }` *removes* the axis, giving rank
//!     `source_rank - 1`, but the old code mislabeled it as the full
//!     rank-`source_rank` source type. The cotangent then flowed on with
//!     the wrong shape and a downstream elementwise op failed
//!     verification (`binary op ... has mismatched dimension at axis 0:
//!     Lit(2) vs Lit(1)`).
//!
//! The static `chelis build` lowering of the reproducer materializes the
//! constant `scalar_to_tensor(c)` as a rank-1 size-1 `tensor[1]` source,
//! so `expand(c, 0, 2)` is a SAME-RANK broadcast — the failing case. The
//! fix branches the Expand adjoint on the forward shape and reshapes the
//! same-rank reduction back to the source shape.
//!
//! This file pins both source shapes at the IR level so the fix is
//! exercised by `grad_dag_checked` (construction) and `eval_tensor`
//! (numeric correctness), independent of the front-end parser and the
//! `chelis` binary; the CLI sibling file pins the end-to-end build.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::{AdError, grad_dag_checked};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

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

/// Build the issue #288 forward DAG. `source_shape` is the shape of the
/// `scalar_to_tensor(cast(c_val, f32))` source node — the heart of the
/// bug is that the *same* surface idiom lowers the constant source two
/// ways depending on context:
///
///   * `[]` (rank 0): the expand `[] -> [n]` is RANK-INCREASING.
///   * `[1]` (rank 1, size 1): the expand `[1] -> [n]` is SAME-RANK
///     (this is what the static `chelis build` lowering of the issue's
///     reproducer actually produces, and the shape that exposed the bug
///     — the old Expand adjoint mislabeled the same-rank reduction and a
///     downstream `Mul` failed verification with `Lit(2) vs Lit(1)`).
///
/// Forward:
///   c   = Cast(Const c_val)        : f32 source_shape  [= scalar_to_tensor(cast(c_val, f32))]
///   k   = Expand{axis:0, size:n}(c): tensor[n]         [= expand(c, 0, n)]
///   x   = Load("x")                : tensor[n]
///   m   = Mul(x, k)                : tensor[n]         [= mul(x, k)]
///   out = Sum{axis:0}(m)           : f32 (rank 0)      [= tensor_to_scalar(sum(m, 0))]
///
/// Returns `(dag, x, out)`. `f(x) = sum(x * c_val) = c_val * sum(x)`,
/// so `df/dx = [c_val; n]` regardless of how the constant source is
/// shaped.
fn build_expand_scalar_forward(
    n: usize,
    c_val: f64,
    source_shape: &[usize],
) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(n);
    let source_ty = TensorType {
        dims: source_shape.iter().map(|&d| DimInfo::Lit(d)).collect(),
        precision: Prim::F32,
    };

    // scalar_to_tensor(cast(c_val, f32)) -> f32 constant of `source_ty`.
    let raw = dag.add_node(
        RiscOp::synth_const(source_ty.precision, c_val),
        vec![],
        source_ty.clone(),
        None,
    );
    let c = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![raw],
        source_ty,
        None,
    );

    // expand(c, axis=0, size=n) -> tensor[n].
    let k = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(n),
        },
        vec![c],
        vec_ty.clone(),
        None,
    );

    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
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
        assert!((g - w).abs() < 1e-5, "{label}: elem {i}: got {g}, want {w}",);
    }
}

// The two constant-source shapes that the `scalar_to_tensor(c)` idiom
// produces. `[1]` (same-rank expand) is the shape the static `chelis
// build` lowering emits and the one that exposed the bug; `[]`
// (rank-increasing expand) is the rank-0 shape from the issue's prose.
// Every backward-construction test runs against both.
const SOURCE_SHAPES: [&[usize]; 2] = [&[], &[1]];

/// Positive: the forward DAG itself evaluates to the scalar
/// `c_val * sum(x)` — establishes that the *forward* is well-formed and
/// the bug is strictly in the backward construction.
#[test]
fn issue_288_forward_expand_scalar_evaluates() {
    for shape in SOURCE_SHAPES {
        let (dag, _x, out) = build_expand_scalar_forward(2, 2.5, shape);
        let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
        inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
        let vals = eval_tensor(&dag, &inputs).expect("forward eval must succeed");
        // 2.5 * (3 + 4) = 17.5
        assert_close(
            &format!("forward source={shape:?}"),
            &vals[&out].to_f64_lossy_vec(),
            &[17.5],
        );
    }
}

/// Positive (the bug): `grad(f, wrt=x)` must construct cleanly through
/// the `expand(scalar_to_tensor(c), 0, n)` idiom for BOTH source shapes.
/// Before the fix the same-rank (`[1]`) source returns
/// `AdError::NotSupported` — the constructed backward DAG fails
/// verification because the Expand adjoint mislabeled the same-rank
/// reduction, leaving a downstream `Mul` with `Lit(2) vs Lit(1)`.
#[test]
fn issue_288_grad_through_expand_scalar_constructs() {
    for shape in SOURCE_SHAPES {
        let (dag, x, out) = build_expand_scalar_forward(2, 2.5, shape);
        match grad_dag_checked(&dag, out, &[x]) {
            Ok(_) => {}
            Err(AdError::NotSupported { op, reason }) => panic!(
                "grad through expand(scalar_to_tensor(c), 0, n) with source \
                 shape {shape:?} must succeed (issue #288); got rejection \
                 op={op}, reason={reason:?}",
            ),
        }
    }
}

/// Positive numeric: `df/dx = [c_val; n]`. With `c_val = 2.5`, `n = 2`,
/// `df(x) = [2.5, 2.5]` for any `x` and either source shape. Asserts the
/// exact gradient the issue's Expected section specifies.
#[test]
fn issue_288_grad_through_expand_scalar_is_correct() {
    for shape in SOURCE_SHAPES {
        let (dag, x, out) = build_expand_scalar_forward(2, 2.5, shape);
        let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct (issue #288)");
        let grad_x = result
            .grad_nodes
            .get(&x)
            .copied()
            .expect("gradient w.r.t. x must be present");
        let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
        inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
        let vals = eval_tensor(&result.dag, &inputs).expect("grad DAG eval");
        assert_close(
            &format!("grad_x source={shape:?}"),
            &vals[&grad_x].to_f64_lossy_vec(),
            &[2.5, 2.5],
        );
        assert_eq!(
            vals[&grad_x].shape,
            vec![2],
            "grad shape must be tensor[2] for source {shape:?}"
        );
    }
}

/// Finite-difference cross-check for both source shapes: the analytic
/// gradient must agree with a centered finite difference of the forward,
/// per the backend-numerics discipline. (Evaluator-vs-evaluator
/// agreement here; the C-backend agreement is pinned by the CLI-level
/// build/eval test.)
#[test]
fn issue_288_grad_matches_finite_difference() {
    for shape in SOURCE_SHAPES {
        let (dag, x, out) = build_expand_scalar_forward(2, 2.5, shape);
        let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
        let grad_x = result.grad_nodes[&x];
        let base = TensorValue::from_vec(vec![2], vec![0.7, -1.3]);
        let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
        inputs.insert("x".into(), base.clone());
        let analytic = eval_tensor(&result.dag, &inputs).expect("analytic eval")[&grad_x]
            .to_f64_lossy_vec()
            .clone();

        let h = 1e-3;
        let mut numerical = [0.0f64; 2];
        for (j, slot) in numerical.iter_mut().enumerate() {
            let mut plus_data = base.to_f64_lossy_vec();
            let mut minus_data = base.to_f64_lossy_vec();
            plus_data[j] += h;
            minus_data[j] -= h;
            let plus = TensorValue::from_vec(base.shape.clone(), plus_data);
            let minus = TensorValue::from_vec(base.shape.clone(), minus_data);
            let mut ip = UnordMap::new();
            ip.insert("x".into(), plus);
            let mut im = UnordMap::new();
            im.insert("x".into(), minus);
            let fp = eval_tensor(&dag, &ip).expect("plus eval")[&out].to_f64_lossy_vec()[0];
            let fm = eval_tensor(&dag, &im).expect("minus eval")[&out].to_f64_lossy_vec()[0];
            *slot = (fp - fm) / (2.0 * h);
        }
        for (i, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
            assert!(
                (a - n).abs() < 1e-3,
                "finite-diff mismatch at {i} (source {shape:?}): analytic {a}, numerical {n}",
            );
        }
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
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
    let sq = dag.add_node(RiscOp::Mul, vec![x, x], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![sq],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("control sum(x*x) must differentiate");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = UnordMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, -4.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("control eval");
    assert_close(
        "control_sum_mul_x_x",
        &vals[&grad_x].to_f64_lossy_vec(),
        &[6.0, -8.0],
    );
}

/// Control 2: `sum(mul(x, t), 0)` against a REAL literal tensor `t`
/// (not expand-of-scalar) differentiates. `f(x) = sum(x .* t)`,
/// `df/dx = t`. `t` is a rank-1 constant whose adjoint Expand source is
/// itself rank 1 — the path the existing roundtrip test exercises.
#[test]
fn issue_288_control_mul_by_real_tensor() {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(2);
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
    // A rank-1 constant tensor built as Const+Pad+Add would mirror
    // `to_tensor`; for this control a rank-1 Load standing in for the
    // literal tensor is sufficient to pin the non-scalar-expand path.
    let t = dag.add_node(
        RiscOp::Load { name: "t".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
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
    let mut inputs = UnordMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
    inputs.insert("t".into(), TensorValue::from_vec(vec![2], vec![1.5, -2.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("control eval");
    assert_close(
        "control_mul_by_real_tensor",
        &vals[&grad_x].to_f64_lossy_vec(),
        &[1.5, -2.0],
    );
}

/// Probe: differentiate w.r.t. the RANK-0 expand source itself. This
/// forces the `Expand` adjoint (the `Sum` that reduces the cotangent
/// back to the rank-0 source) into the requested-output set so it is
/// NOT pruned and must pass verification. `f(s) = sum(x .* expand(s,
/// 0, 2))` with scalar `s`, so `df/ds = sum(x)`. With x = [3, 4],
/// `df/ds = 7`. If the rank-0-source Expand adjoint is the bug, this is
/// where the over-reduction / verification failure surfaces directly.
#[test]
fn issue_288_grad_wrt_rank0_expand_source() {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(2);
    let s = dag.add_node(
        RiscOp::Load { name: "s".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let k = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![s],
        vec_ty.clone(),
        None,
    );
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
    let m = dag.add_node(RiscOp::Mul, vec![x, k], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![m],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[s])
        .expect("grad w.r.t. a rank-0 expand source must construct (issue #288)");
    let grad_s = result.grad_nodes[&s];
    let mut inputs = UnordMap::new();
    inputs.insert("s".into(), TensorValue::from_vec(vec![], vec![2.5]));
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("probe eval");
    assert_close(
        "grad_wrt_rank0_source",
        &vals[&grad_s].to_f64_lossy_vec(),
        &[7.0],
    );
    assert!(
        vals[&grad_s].shape.is_empty(),
        "gradient of a rank-0 source must be rank-0; got shape {:?}",
        vals[&grad_s].shape,
    );
}

/// Probe (the actual #288 shape): differentiate w.r.t. a rank-1, size-1
/// expand source `s: tensor[1]` broadcast by a SAME-RANK expand `[1] ->
/// [2]`. This forces the same-rank Expand adjoint into the requested
/// outputs so it must pass verification, and pins that the gradient is
/// reshaped back to the source's `[1]` shape (not collapsed to a scalar
/// and not left at rank-1 size-2). `f(s) = sum(x .* expand(s, 0, 2))`
/// with `s = [v]`, so `f = v * sum(x)` and `df/ds = [sum(x)] = [7]` for
/// x = [3, 4]. Before the fix this is exactly the configuration whose
/// backward DAG failed verification with `Lit(2) vs Lit(1)`.
#[test]
fn issue_288_grad_wrt_size1_expand_source() {
    let mut dag = Dag::new();
    let vec_ty = vec_n_f32(2);
    let one_ty = vec_n_f32(1);
    let s = dag.add_node(RiscOp::Load { name: "s".into() }, vec![], one_ty, None);
    let k = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![s],
        vec_ty.clone(),
        None,
    );
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
    let m = dag.add_node(RiscOp::Mul, vec![x, k], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![m],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[s])
        .expect("grad w.r.t. a rank-1 size-1 expand source must construct (issue #288)");
    let grad_s = result.grad_nodes[&s];
    let mut inputs = UnordMap::new();
    inputs.insert("s".into(), TensorValue::from_vec(vec![1], vec![2.5]));
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![3.0, 4.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("probe eval");
    assert_close(
        "grad_wrt_size1_source",
        &vals[&grad_s].to_f64_lossy_vec(),
        &[7.0],
    );
    assert_eq!(
        vals[&grad_s].shape,
        vec![1],
        "gradient of a rank-1 size-1 source must keep shape [1]; got {:?}",
        vals[&grad_s].shape,
    );
}

/// Probe (same-rank expand at a NON-ZERO axis): the existing #288 probes
/// all broadcast a size-1 axis at `axis = 0`. This one differentiates
/// w.r.t. a rank-2, size-1-on-axis-1 source `s: tensor[3, 1]` broadcast
/// by a SAME-RANK expand `[3, 1] -> [3, 2]` along `axis = 1`. It locks
/// that the same-rank Expand adjoint's `Sum { axis }` reduction and its
/// `dims.remove(axis)` / reshape-back path are correct when `axis != 0`
/// (the old code's mislabel and the new code's index arithmetic both
/// hinge on the axis position).
///
/// Forward: `f(s) = Σ_{i,j} x[i,j] * expand(s, 1, 2)[i,j]` and
/// `expand(s, 1, 2)[i,j] = s[i, 0]`, so
/// `f(s) = Σ_i s[i,0] * (x[i,0] + x[i,1])` and
/// `df/ds[i,0] = x[i,0] + x[i,1]` (the per-row sum of `x`), shape
/// `[3, 1]`. With `x = [[10, 20], [30, 40], [50, 60]]` the gradient is
/// `[[30], [70], [110]]`. The loss reduces the `[3, 2]` product to a
/// scalar with two `Sum`s (axis 1 then axis 0); the `s` gradient is
/// independent of that reduction order.
#[test]
fn issue_288_grad_wrt_size1_expand_source_nonzero_axis() {
    let row_vec_ty = |rows: usize, cols: usize| TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let mat_ty = row_vec_ty(3, 2); // [3, 2]
    let col_ty = row_vec_ty(3, 1); // [3, 1] (size-1 on axis 1)
    let row_ty = vec_n_f32(3); // [3] after summing axis 1

    let s = dag.add_node(RiscOp::Load { name: "s".into() }, vec![], col_ty, None);
    // expand(s, axis=1, size=2): [3, 1] -> [3, 2], SAME-RANK broadcast of
    // the size-1 axis 1. This is the non-zero-axis path under test.
    let k = dag.add_node(
        RiscOp::Expand {
            axis: 1,
            size: DimExpr::Concrete(2),
        },
        vec![s],
        mat_ty.clone(),
        None,
    );
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_ty.clone(),
        None,
    );
    let m = dag.add_node(RiscOp::Mul, vec![x, k], mat_ty, None);
    // Reduce [3, 2] -> scalar with two Sums (Sum removes one axis each).
    let row_sums = dag.add_node(
        RiscOp::sum_default(1, Prim::F32).expect("sum_default axis 1"),
        vec![m],
        row_ty,
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default axis 0"),
        vec![row_sums],
        scalar_f32(),
        None,
    );

    let result = grad_dag_checked(&dag, out, &[s]).expect(
        "grad w.r.t. a rank-2 size-1-on-axis-1 expand source must construct (issue #288, axis > 0)",
    );
    let grad_s = result.grad_nodes[&s];
    let mut inputs = UnordMap::new();
    inputs.insert(
        "s".into(),
        TensorValue::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]),
    );
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("probe eval");
    // df/ds[i,0] = x[i,0] + x[i,1]: row sums of x = [30, 70, 110].
    assert_close(
        "grad_wrt_size1_source_axis1",
        &vals[&grad_s].to_f64_lossy_vec(),
        &[30.0, 70.0, 110.0],
    );
    assert_eq!(
        vals[&grad_s].shape,
        vec![3, 1],
        "gradient of a rank-2 size-1-on-axis-1 source must keep shape [3, 1]; got {:?}",
        vals[&grad_s].shape,
    );
}
