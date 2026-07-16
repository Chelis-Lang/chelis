//! Issue #254 — `reduce_window_*` IR primitive.
//!
//! Pins the IR-level semantics for the four reducers (`Max`, `Min`,
//! `Sum`, `Mean`) under `Valid` padding. The shape contract and the
//! per-window arithmetic are the operational complement of
//! `spec/05-risc-primitives.md` §2.3.1.
//!
//! Positive coverage: each reducer matches the spec formula on a small
//! concrete input. Negative coverage: ill-shaped windows / strides
//! panic from the evaluator's defensive asserts (which the type checker
//! is responsible for rejecting before that point).

use chelis_ir::dag::{Dag, DimInfo, ReduceWindowKind, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::verify;
use chelis_types::types::Prim;
use std::collections::HashMap;

fn tensor_type(dims: &[usize]) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

fn build_reduce_window(
    input_shape: &[usize],
    reducer: ReduceWindowKind,
    window_shape: Vec<usize>,
    strides: Vec<usize>,
    out_shape: &[usize],
) -> (Dag, String) {
    let mut dag = Dag::default();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(input_shape),
        None,
    );
    let _out = dag.add_node(
        RiscOp::ReduceWindow {
            reducer,
            window_shape,
            strides,
        },
        vec![load],
        tensor_type(out_shape),
        None,
    );
    (dag, "x".to_string())
}

fn eval_with_input(dag: &Dag, name: &str, shape: Vec<usize>, data: Vec<f64>) -> TensorValue {
    let mut inputs = HashMap::new();
    inputs.insert(name.to_string(), TensorValue::from_vec(shape, data));
    let values = eval_tensor(dag, &inputs).expect("eval succeeds");
    // The output is the second node (id = 1).
    values
        .get(&chelis_ir::dag::NodeId(1))
        .expect("output node present")
        .clone()
}

// -------- Positive coverage --------

/// Acceptance reproducer from issue #254: [2, 3, 8, 8] input + window
/// [2, 2] + strides [1, 1] + Max → [2, 3, 7, 7]. We use a smaller
/// 1x1x4x4 stand-in so the expected values are tractable; the rank-4
/// shape + windowed-trailing-axes contract is identical.
#[test]
fn reduce_window_max_strided_overlap_smoke() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 4, 4],
        ReduceWindowKind::Max,
        vec![2, 2],
        vec![1, 1],
        &[1, 1, 3, 3],
    );
    // Input:
    // [[ 1  2  3  4]
    //  [ 5  6  7  8]
    //  [ 9 10 11 12]
    //  [13 14 15 16]]
    let data = (1u32..=16).map(f64::from).collect();
    let out = eval_with_input(&dag, &name, vec![1, 1, 4, 4], data);
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    // Each 2x2 window's max:
    // [[ 6  7  8]
    //  [10 11 12]
    //  [14 15 16]]
    assert_eq!(
        out.data,
        vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
    );
}

#[test]
fn reduce_window_max_acceptance_shape_matches_issue() {
    // Issue acceptance: [2, 3, 8, 8] + window [2, 2] + strides [1, 1]
    // + Max → [2, 3, 7, 7]. We only check the shape contract here;
    // the per-element values are exercised on the 4x4 smoke above.
    let (dag, _) = build_reduce_window(
        &[2, 3, 8, 8],
        ReduceWindowKind::Max,
        vec![2, 2],
        vec![1, 1],
        &[2, 3, 7, 7],
    );
    assert!(
        verify::verify(&dag).is_empty(),
        "reduce_window IR for the issue acceptance shape must verify clean"
    );
}

#[test]
fn reduce_window_min_strided_overlap_smoke() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 4, 4],
        ReduceWindowKind::Min,
        vec![2, 2],
        vec![1, 1],
        &[1, 1, 3, 3],
    );
    let data = (1u32..=16).map(f64::from).collect();
    let out = eval_with_input(&dag, &name, vec![1, 1, 4, 4], data);
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    assert_eq!(
        out.data,
        vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn reduce_window_sum_strided_overlap_smoke() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 3, 3],
        ReduceWindowKind::Sum,
        vec![2, 2],
        vec![1, 1],
        &[1, 1, 2, 2],
    );
    // [[1 2 3]
    //  [4 5 6]
    //  [7 8 9]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let out = eval_with_input(&dag, &name, vec![1, 1, 3, 3], data);
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // 2x2 sums:
    // [[1+2+4+5=12, 2+3+5+6=16],
    //  [4+5+7+8=24, 5+6+8+9=28]]
    assert_eq!(out.data, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn reduce_window_mean_strided_overlap_smoke() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 3, 3],
        ReduceWindowKind::Mean,
        vec![2, 2],
        vec![1, 1],
        &[1, 1, 2, 2],
    );
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let out = eval_with_input(&dag, &name, vec![1, 1, 3, 3], data);
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Mean = sum / window_volume (= 4): 12/4=3, 16/4=4, 24/4=6, 28/4=7.
    assert_eq!(out.data, vec![3.0, 4.0, 6.0, 7.0]);
}

/// Stride > 1 makes the output spatially non-overlapping: [1,1,4,4]
/// + window [2,2] + strides [2,2] → [1,1,2,2] (the "stride == window"
///   pool2d corner of the issue text).
#[test]
fn reduce_window_max_non_overlapping_pool2d() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 4, 4],
        ReduceWindowKind::Max,
        vec![2, 2],
        vec![2, 2],
        &[1, 1, 2, 2],
    );
    let data = (1u32..=16).map(f64::from).collect();
    let out = eval_with_input(&dag, &name, vec![1, 1, 4, 4], data);
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Four non-overlapping 2x2 blocks:
    // [[ 1  2 | 3  4]   max → 6  | 8
    //  [ 5  6 | 7  8]
    //  -------+------   max →
    //  [ 9 10 | 11 12]   14 | 16
    //  [13 14 | 15 16]]
    assert_eq!(out.data, vec![6.0, 8.0, 14.0, 16.0]);
}

/// Rank-3 input with rank-2 windowed reduction: the leading axis
/// passes through unchanged, matching the spec §2.3.1 leading-axis
/// rule.
#[test]
fn reduce_window_max_passes_through_leading_axes() {
    let (dag, name) = build_reduce_window(
        &[2, 3, 3],
        ReduceWindowKind::Max,
        vec![2, 2],
        vec![1, 1],
        &[2, 2, 2],
    );
    // First slice [0..]: as in the sum-smoke above.
    // Second slice [1..]: same values + 100 so the max windows differ.
    let mut data: Vec<f64> = (1u32..=9).map(f64::from).collect();
    data.extend((1u32..=9).map(|v| f64::from(v) + 100.0));
    let out = eval_with_input(&dag, &name, vec![2, 3, 3], data);
    assert_eq!(out.shape, vec![2, 2, 2]);
    // Slice 0: max windows of 1..9 with window [2,2] stride [1,1]
    // [[5, 6], [8, 9]]; slice 1: same +100.
    assert_eq!(
        out.data,
        vec![5.0, 6.0, 8.0, 9.0, 105.0, 106.0, 108.0, 109.0]
    );
}

// -------- Negative coverage --------

/// Window larger than the input dim is a structural error; the
/// evaluator's defensive assert fires (the type checker rejects this
/// before lowering in normal pipelines).
#[test]
#[should_panic(expected = "axis")]
fn reduce_window_panics_when_window_exceeds_input_dim() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 2, 2],
        ReduceWindowKind::Max,
        vec![3, 3],
        vec![1, 1],
        &[1, 1, 0, 0],
    );
    let _ = eval_with_input(&dag, &name, vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
}

/// Zero stride is rejected. The IR evaluator panics; the type checker
/// rejects `stride <= 0` before lowering in normal pipelines.
#[test]
#[should_panic(expected = "stride")]
fn reduce_window_panics_on_zero_stride() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 3, 3],
        ReduceWindowKind::Max,
        vec![2, 2],
        vec![1, 0],
        &[1, 1, 2, 2],
    );
    let _ = eval_with_input(&dag, &name, vec![1, 1, 3, 3], vec![1.0; 9]);
}

/// Zero window is rejected. The IR evaluator panics; the type checker
/// rejects `window <= 0` before lowering in normal pipelines.
#[test]
#[should_panic(expected = "window")]
fn reduce_window_panics_on_zero_window() {
    let (dag, name) = build_reduce_window(
        &[1, 1, 3, 3],
        ReduceWindowKind::Max,
        vec![0, 2],
        vec![1, 1],
        &[1, 1, 2, 2],
    );
    let _ = eval_with_input(&dag, &name, vec![1, 1, 3, 3], vec![1.0; 9]);
}

/// Window arity larger than input rank is rejected.
#[test]
#[should_panic(expected = "smaller than window arity")]
fn reduce_window_panics_when_window_arity_exceeds_rank() {
    let (dag, name) = build_reduce_window(
        &[3, 3],
        ReduceWindowKind::Max,
        vec![2, 2, 2],
        vec![1, 1, 1],
        &[1, 1, 1],
    );
    let _ = eval_with_input(&dag, &name, vec![3, 3], vec![1.0; 9]);
}

/// AD is supported per spec §2.3.1: `grad` over `reduce_window_*` lowers
/// to a `ReduceWindowGrad` adjoint rather than being rejected. End-to-end
/// check: the gradient of `sum(reduce_window_sum(x))` w.r.t. `x` is the
/// per-position window-cover count (each input position contributes to
/// every window that covers it).
#[test]
fn reduce_window_grad_lowers_to_adjoint_and_evaluates() {
    use chelis_ir::grad::grad_dag_checked;

    // x[4] -> reduce_window_sum(window=[2], stride=[1]) -> out[3] -> sum -> scalar
    let mut dag = Dag::default();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(&[4]),
        None,
    );
    let rw = dag.add_node(
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Sum,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![load],
        tensor_type(&[3]),
        None,
    );
    let scalar = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).unwrap(),
        vec![rw],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );

    let grad_result = grad_dag_checked(&dag, scalar, &[load])
        .expect("reduce_window_* now has a reverse-mode adjoint (no longer rejected)");

    // The combined DAG must carry the windowed adjoint node.
    assert!(
        grad_result.dag.nodes().iter().any(|n| matches!(
            n.op,
            RiscOp::ReduceWindowGrad {
                reducer: ReduceWindowKind::Sum,
                ..
            }
        )),
        "grad of reduce_window_sum must lower to a ReduceWindowGrad adjoint"
    );

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue {
            data: vec![10.0, 20.0, 30.0, 40.0],
            shape: vec![4],
        },
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&load]];
    assert_eq!(grad.shape, vec![4]);
    // Windows [0,1], [1,2], [2,3]: ends covered once, interior twice.
    assert_eq!(grad.data, vec![1.0, 2.0, 2.0, 1.0]);
}
