//! Issue #291: `grad` cannot lower the backward pass for the
//! slicing/windowing movement ops `shrink` and `stride`.
//!
//! Reproducers (Surf), verbatim from the issue:
//! ```chelis
//! def f(x: tensor[4, f32]) -> f32 =
//!   tensor_to_scalar(sum(shrink(x, [[cast(0, int32), cast(2, int32)]]), cast(0, int32)))
//! def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)
//! ```
//! ```chelis
//! def f(x: tensor[4, f32]) -> f32 =
//!   tensor_to_scalar(sum(stride(x, cast(2, int32)), cast(0, int32)))
//! def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(f)(x)
//! ```
//!
//! Both reported "failed to construct backward DAG". Two distinct root
//! causes were found:
//!
//!   * `shrink`: the Pad-based adjoint in `compute_adjoints` was already
//!     correct, but the FRONT-END lowering of the issue's bound literal
//!     `[[cast(0, int32), cast(2, int32)]]` dropped the `cast`-wrapped
//!     pair elements (the bound walker accepted only `Atom::Int` /
//!     `(lit ...)`), producing an empty `RiscOp::Shrink { bounds: [] }`
//!     that failed backward-DAG verification (`bounds len 0 != input
//!     rank 1`). PR #296 hardened the matching `stride`/`wrt` integer
//!     extraction against the same shape; this closes the gap for
//!     `shrink`/`pad` bounds. That fix lives in `lower.rs`; the
//!     CLI-level sibling file is its end-to-end oracle.
//!
//!   * `stride`: `compute_adjoints` returned `None` for `RiscOp::Stride`
//!     ("Phase 0 has no scatter/upsample primitive"). The exact adjoint
//!     scatters each cotangent element `g[i]` back to source position
//!     `i .* s` and zeros the skipped slots
//!     (spec/05-risc-primitives.md §2.4: stride adjoint = "appropriate
//!     expand/scatter"). It is expressed with EXISTING primitives only,
//!     as a separable `reshape -> pad -> reshape -> shrink` upsample per
//!     axis, so no new IR op was needed.
//!
//! This file pins the IR-level construction (`grad_dag_checked`) and
//! numeric correctness (`eval_tensor`) for `stride`, independent of the
//! front-end parser, plus the IR-level `shrink` adjoint controls. The
//! CLI sibling file pins the end-to-end `build`/`eval` path including
//! the `shrink` lowering fix.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
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

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn scalar_f32() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F32,
    }
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

/// Build `out = sum(stride(x, strides), 0-then-...)` reducing to a
/// scalar, returning `(dag, x, out)`. The forward sums the strided
/// view; `df/dx` is therefore 1 at every strided source slot and 0
/// elsewhere (the pure scatter pattern).
fn build_stride_sum_1d(n: usize, step: usize) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let in_ty = vec_n_f32(n);
    let out_n = n.div_ceil(step);
    let strided_ty = vec_n_f32(out_n);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![step],
        },
        vec![x],
        strided_ty,
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![strided],
        scalar_f32(),
        None,
    );
    (dag, x, out)
}

// --- STRIDE: the bug ---

/// Positive (the bug): `grad(sum(stride(x, 2)))` must CONSTRUCT cleanly.
/// Before the fix `RiscOp::Stride` returned `None` and the backward DAG
/// failed with "no reverse-mode adjoint is defined for `stride`".
#[test]
fn issue_291_grad_through_stride_constructs() {
    let (dag, x, out) = build_stride_sum_1d(4, 2);
    match grad_dag_checked(&dag, out, &[x]) {
        Ok(_) => {}
        Err(AdError::NotSupported { op, reason }) => panic!(
            "grad through stride(x, 2) must succeed (issue #291); got rejection \
             op={op}, reason={reason:?}",
        ),
    }
}

/// Positive numeric (exact scatter): `f(x) = sum(stride(x, 2)) =
/// x[0] + x[2]`, so `df/dx = [1, 0, 1, 0]` — the cotangent scatters into
/// the strided slots and is zero in the skipped slots. Asserts the exact
/// adjoint the issue's Expected section specifies.
#[test]
fn issue_291_grad_through_stride_is_exact_scatter() {
    let (dag, x, out) = build_stride_sum_1d(4, 2);
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct (issue #291)");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad DAG eval");
    assert_close("grad_stride_2", &vals[&grad_x].data, &[1.0, 0.0, 1.0, 0.0]);
    assert_eq!(
        vals[&grad_x].shape,
        vec![4],
        "stride gradient must keep the source shape tensor[4]",
    );
}

/// Numeric with a non-trivial loss so the cotangent is NOT all-ones:
/// `f(x) = sum(stride(x, 2) .* stride(x, 2)) = x[0]^2 + x[2]^2`, hence
/// `df/dx = [2*x0, 0, 2*x2, 0]`. This pins that the scatter routes the
/// REAL upstream cotangent (here `2*x[i*2]`) to slot `i*2`, not merely a
/// constant 1, and zeros the skipped slots.
#[test]
fn issue_291_grad_stride_routes_nonuniform_cotangent() {
    let mut dag = Dag::new();
    let in_ty = vec_n_f32(4);
    let strided_ty = vec_n_f32(2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let s = dag.add_node(
        RiscOp::Stride { strides: vec![2] },
        vec![x],
        strided_ty.clone(),
        None,
    );
    let sq = dag.add_node(RiscOp::Mul, vec![s, s], strided_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![sq],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![3.0, 99.0, 5.0, 99.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    // df/dx = [2*3, 0, 2*5, 0] = [6, 0, 10, 0].
    assert_close(
        "grad_stride_nonuniform",
        &vals[&grad_x].data,
        &[6.0, 0.0, 10.0, 0.0],
    );
}

/// Step that does not evenly divide the axis: `n = 5`, `step = 2`, so the
/// strided size is `ceil(5/2) = 3` and the sampled source slots are
/// `{0, 2, 4}`. `f(x) = sum(stride(x, 2)) = x0 + x2 + x4`, so
/// `df/dx = [1, 0, 1, 0, 1]`. Pins the trailing-overshoot trim (the
/// final group's pad would reach index 5, and the adjoint shrink must
/// cut it back to 5 elements).
#[test]
fn issue_291_grad_stride_non_dividing_step() {
    let (dag, x, out) = build_stride_sum_1d(5, 2);
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    assert_close(
        "grad_stride_5_by_2",
        &vals[&grad_x].data,
        &[1.0, 0.0, 1.0, 0.0, 1.0],
    );
    assert_eq!(vals[&grad_x].shape, vec![5]);
}

/// Step 3 over an axis of size 4: strided size `ceil(4/3) = 2`, sampled
/// slots `{0, 3}`. `f(x) = sum(stride(x, 3)) = x0 + x3`,
/// `df/dx = [1, 0, 0, 1]`. Pins that the inserted-zero count tracks
/// `step - 1` (two zeros between kept slots), not a hard-coded one.
#[test]
fn issue_291_grad_stride_step_three() {
    let (dag, x, out) = build_stride_sum_1d(4, 3);
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    assert_close(
        "grad_stride_4_by_3",
        &vals[&grad_x].data,
        &[1.0, 0.0, 0.0, 1.0],
    );
}

/// Multi-axis stride `[2, 2]` over a `[4, 4]` matrix: the adjoint applies
/// the per-axis upsample twice (one axis at a time) and must reconstruct
/// the `[4, 4]` source shape with the cotangent only at
/// `(2i, 2j)` positions. `f(x) = sum_all(stride(x, [2, 2]))` samples
/// `x[0,0], x[0,2], x[2,0], x[2,2]`, so `df/dx` is 1 at exactly those
/// four positions and 0 elsewhere.
#[test]
fn issue_291_grad_stride_two_axes() {
    let mut dag = Dag::new();
    let in_ty = mat_f32(4, 4);
    let strided_ty = mat_f32(2, 2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![2, 2],
        },
        vec![x],
        strided_ty.clone(),
        None,
    );
    // Sum [2,2] -> [2] (axis 1) -> scalar (axis 0).
    let row_sums = dag.add_node(
        RiscOp::sum_default(1, Prim::F32).expect("sum_default axis 1"),
        vec![strided],
        vec_n_f32(2),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default axis 0"),
        vec![row_sums],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("multi-axis stride grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4, 4], (0..16).map(|v| v as f64).collect()),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    #[rustfmt::skip]
    let want = vec![
        1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_close("grad_stride_2x2", &vals[&grad_x].data, &want);
    assert_eq!(vals[&grad_x].shape, vec![4, 4]);
}

/// Mixed steps `[1, 2]` over `[3, 4]`: axis 0 is identity (step 1, the
/// arm skips it) and axis 1 strides by 2. `f = sum_all(stride(x,[1,2]))`
/// samples every row at columns `{0, 2}`, so `df/dx` is 1 at columns
/// 0 and 2 of every row and 0 at columns 1 and 3. Pins that the step-1
/// axis is correctly left untouched while the step-2 axis scatters.
#[test]
fn issue_291_grad_stride_mixed_identity_axis() {
    let mut dag = Dag::new();
    let in_ty = mat_f32(3, 4);
    let strided_ty = mat_f32(3, 2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![1, 2],
        },
        vec![x],
        strided_ty,
        None,
    );
    let row_sums = dag.add_node(
        RiscOp::sum_default(1, Prim::F32).expect("sum axis 1"),
        vec![strided],
        vec_n_f32(3),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum axis 0"),
        vec![row_sums],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("mixed-step stride grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![3, 4], (0..12).map(|v| v as f64).collect()),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    #[rustfmt::skip]
    let want = vec![
        1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
    ];
    assert_close("grad_stride_1x2", &vals[&grad_x].data, &want);
    assert_eq!(vals[&grad_x].shape, vec![3, 4]);
}

/// Finite-difference cross-check (backend-numerics discipline): the
/// analytic stride gradient must agree with a centered finite difference
/// of the forward, against a non-trivial loss and base point.
/// `f(x) = sum(stride(x, 2)^2) = x0^2 + x2^2`.
#[test]
fn issue_291_grad_stride_matches_finite_difference() {
    let mut dag = Dag::new();
    let in_ty = vec_n_f32(4);
    let strided_ty = vec_n_f32(2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let s = dag.add_node(
        RiscOp::Stride { strides: vec![2] },
        vec![x],
        strided_ty.clone(),
        None,
    );
    let sq = dag.add_node(RiscOp::Mul, vec![s, s], strided_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![sq],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];

    let base = TensorValue::from_vec(vec![4], vec![0.7, -1.3, 2.1, 0.4]);
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), base.clone());
    let analytic = eval_tensor(&result.dag, &inputs).expect("analytic eval")[&grad_x]
        .data
        .clone();

    let h = 1e-3;
    let mut numerical = [0.0f64; 4];
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
            "stride finite-diff mismatch at {i}: analytic {a}, numerical {n}",
        );
    }
}

/// Step strictly larger than the axis: `n = 2`, `step = 5`, so the
/// strided size is `ceil(2/5) = 1` and only source slot `0` is sampled.
/// `f(x) = sum(stride(x, 5)) = x0`, so `df/dx = [1, 0]`. Pins the
/// single-group case where the upsample pads one kept element to `step`
/// (5) and the adjoint shrink trims the four-element overshoot back to
/// the source size 2.
#[test]
fn issue_291_grad_stride_step_exceeds_axis() {
    let (dag, x, out) = build_stride_sum_1d(2, 5);
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![10.0, 20.0]));
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    assert_close("grad_stride_2_by_5", &vals[&grad_x].data, &[1.0, 0.0]);
    assert_eq!(vals[&grad_x].shape, vec![2]);
}

/// Step exactly equal to the axis: `n = 4`, `step = 4`, strided size
/// `ceil(4/4) = 1`, only slot `0` sampled. `f(x) = sum(stride(x, 4)) =
/// x0`, so `df/dx = [1, 0, 0, 0]`. Pins the boundary where `m_a * step`
/// (4) equals the source size exactly (no trailing overshoot to trim).
#[test]
fn issue_291_grad_stride_step_equals_axis() {
    let (dag, x, out) = build_stride_sum_1d(4, 4);
    let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    assert_close(
        "grad_stride_4_by_4",
        &vals[&grad_x].data,
        &[1.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(vals[&grad_x].shape, vec![4]);
}

/// Higher-order AD through the stride adjoint (the PR's comment claims
/// this is supported; this pins it). `f(x) = sum(stride(x, 2)^2) =
/// x0^2 + x2^2`. First grad: `g(x) = df/dx = [2*x0, 0, 2*x2, 0]`. Define
/// `h(x) = sum(g(x)) = 2*x0 + 2*x2`; the SECOND grad is then
/// `dh/dx = [2, 0, 2, 0]`. Because the stride adjoint is built entirely
/// from reshape/pad/shrink (each already higher-order differentiable),
/// differentiating through the backward DAG itself must construct and
/// yield the exact second-order scatter.
#[test]
fn issue_291_grad_stride_supports_higher_order_ad() {
    let mut dag = Dag::new();
    let in_ty = vec_n_f32(4);
    let strided_ty = vec_n_f32(2);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let s = dag.add_node(
        RiscOp::Stride { strides: vec![2] },
        vec![x],
        strided_ty.clone(),
        None,
    );
    let sq = dag.add_node(RiscOp::Mul, vec![s, s], strided_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![sq],
        scalar_f32(),
        None,
    );
    // First-order backward DAG.
    let first = grad_dag_checked(&dag, out, &[x]).expect("first grad must construct");
    let grad_x = first.grad_nodes[&x];
    // Reduce the first gradient to a scalar so the second grad is well
    // defined, then differentiate the backward DAG with respect to `x`.
    let mut g2dag = first.dag.clone();
    let sum_grad = g2dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![grad_x],
        scalar_f32(),
        None,
    );
    let second = grad_dag_checked(&g2dag, sum_grad, &[x])
        .expect("second grad must construct (higher-order AD through stride adjoint)");
    let grad2_x = second.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![3.0, 99.0, 5.0, 99.0]),
    );
    let vals = eval_tensor(&second.dag, &inputs).expect("second-grad eval");
    // d/dx (2*x0 + 2*x2) = [2, 0, 2, 0]; independent of the skipped slots.
    assert_close(
        "grad_grad_stride_2",
        &vals[&grad2_x].data,
        &[2.0, 0.0, 2.0, 0.0],
    );
    assert_eq!(vals[&grad2_x].shape, vec![4]);
}

// --- STRIDE: negative parity ---

/// Negative parity: a stride over a SYMBOLIC (unsized) axis cannot be
/// upsampled because the trim size is unknown. The adjoint relies on a
/// concrete source size (`dim_size`), matching the existing Pad/Shrink
/// adjoints. This pins that the construction does not silently fabricate
/// a wrong shape; the symbolic source dim is unrepresentable here.
#[test]
#[should_panic(expected = "symbolic dimension")]
fn issue_291_grad_stride_symbolic_axis_is_unrepresentable() {
    let mut dag = Dag::new();
    let in_ty = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let strided_ty = TensorType {
        dims: vec![DimInfo::Named("m".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    let s = dag.add_node(
        RiscOp::Stride { strides: vec![2] },
        vec![x],
        strided_ty,
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![s],
        scalar_f32(),
        None,
    );
    // The Stride adjoint must read concrete source sizes; a symbolic
    // unsized source dim panics in `dim_size` (same fail-mode as the
    // existing Pad/Shrink adjoints).
    let _ = grad_dag_checked(&dag, out, &[x]);
}

// --- SHRINK: IR-level controls (the adjoint itself was already
// correct; these pin it stays correct alongside the lowering fix). ---

/// Positive: `grad(sum(shrink(x, [(0, 2)])))` constructs and the adjoint
/// pads the cotangent back into the sliced positions, zeros elsewhere.
/// `f(x) = sum(shrink(x, [0,2))) = x0 + x1`, so `df/dx = [1, 1, 0, 0]`.
/// This is the IR-level twin of the issue's headline shrink reproducer.
#[test]
fn issue_291_grad_through_shrink_is_exact_pad() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_n_f32(4),
        None,
    );
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(0, 2)],
        },
        vec![x],
        vec_n_f32(2),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![shrunk],
        scalar_f32(),
        None,
    );
    let result =
        grad_dag_checked(&dag, out, &[x]).expect("shrink grad must construct (issue #291)");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    assert_close(
        "grad_shrink_0_2",
        &vals[&grad_x].data,
        &[1.0, 1.0, 0.0, 0.0],
    );
    assert_eq!(vals[&grad_x].shape, vec![4]);
}

/// Shrink with a non-trivial loss so the padded cotangent is not all
/// ones inside the window: `f(x) = sum(shrink(x, [1,3))^2) = x1^2 + x2^2`,
/// `df/dx = [0, 2*x1, 2*x2, 0]`. Pins the interior slice + non-uniform
/// cotangent routing.
#[test]
fn issue_291_grad_shrink_interior_nonuniform() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_n_f32(4),
        None,
    );
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(1, 3)],
        },
        vec![x],
        vec_n_f32(2),
        None,
    );
    let sq = dag.add_node(RiscOp::Mul, vec![shrunk, shrunk], vec_n_f32(2), None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![sq],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).expect("shrink grad must construct");
    let grad_x = result.grad_nodes[&x];
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![99.0, 4.0, 6.0, 99.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
    // df/dx = [0, 2*4, 2*6, 0] = [0, 8, 12, 0].
    assert_close(
        "grad_shrink_interior",
        &vals[&grad_x].data,
        &[0.0, 8.0, 12.0, 0.0],
    );
}
