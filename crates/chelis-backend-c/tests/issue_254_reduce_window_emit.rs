//! Issue #254 — C-backend emission verification for the four
//! `reduce_window_*` primitives.
//!
//! Pin the structural invariants of the emitted C code so we catch
//! silent regressions where (for example) `Mean` forgets to divide by
//! the window volume or `Min` uses `fmaxf`. Numerical
//! evaluator-vs-C-backend parity is exercised through the host-runtime
//! reduce_window tests in `chelis-compiler-api`, which evaluate the
//! same Surf programs through the IR evaluator path. The IR evaluator
//! is the authoritative oracle per `spec/05-risc-primitives.md` §6.

use chelis_backend_c::codegen;
use chelis_ir::dag::{Dag, DimInfo, ReduceWindowKind, RiscOp, TensorType};
use chelis_types::types::Prim;

fn tensor_4d(shape: [usize; 4]) -> TensorType {
    TensorType {
        dims: shape.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

fn build_dag(reducer: ReduceWindowKind) -> Dag {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_4d([1, 1, 4, 4]),
        None,
    );
    dag.add_node(
        RiscOp::ReduceWindow {
            reducer,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![load],
        tensor_4d([1, 1, 3, 3]),
        None,
    );
    dag
}

#[test]
fn issue254_emit_reduce_window_max_uses_fmaxf_and_neg_infinity() {
    let dag = build_dag(ReduceWindowKind::Max);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("acc = fmaxf("),
        "Max emit must combine with fmaxf, got:\n{src}"
    );
    assert!(
        src.contains("-INFINITY"),
        "Max emit must initialize acc to -INFINITY, got:\n{src}"
    );
    // Two windowed axes → two nested `__w` loops with the literal
    // window size 2.
    assert!(
        src.contains("for (int __w0 = 0; __w0 < 2;"),
        "Max emit must have an inner window loop along axis 0, got:\n{src}"
    );
    assert!(
        src.contains("for (int __w1 = 0; __w1 < 2;"),
        "Max emit must have an inner window loop along axis 1, got:\n{src}"
    );
}

#[test]
fn issue254_emit_reduce_window_min_uses_fminf_and_positive_infinity() {
    let dag = build_dag(ReduceWindowKind::Min);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("acc = fminf("),
        "Min emit must combine with fminf, got:\n{src}"
    );
    // We init to INFINITY (no leading `-`).
    assert!(
        src.contains("float acc = INFINITY;"),
        "Min emit must initialize acc to INFINITY, got:\n{src}"
    );
}

#[test]
fn issue254_emit_reduce_window_sum_uses_plus_equals() {
    let dag = build_dag(ReduceWindowKind::Sum);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("acc += t"),
        "Sum emit must use `acc += ...`, got:\n{src}"
    );
    assert!(
        src.contains("float acc = 0.0f;"),
        "Sum emit must initialize acc to 0.0f, got:\n{src}"
    );
    // Sum must NOT emit a division by the window volume; that's the
    // distinguishing tell vs. Mean.
    assert!(
        !src.contains("acc /= "),
        "Sum emit must NOT divide by window volume, got:\n{src}"
    );
}

#[test]
fn issue254_emit_reduce_window_mean_divides_by_window_volume() {
    let dag = build_dag(ReduceWindowKind::Mean);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("acc += t"),
        "Mean emit must combine with sum, got:\n{src}"
    );
    // Window volume = 2 * 2 = 4.
    assert!(
        src.contains("acc /= 4.0f;"),
        "Mean emit must divide by the window volume (4 for 2x2), got:\n{src}"
    );
}

/// Stride > 1 must show up in the source-index arithmetic (the IR
/// node carries strides verbatim, and the emit multiplies by them).
#[test]
fn issue254_emit_reduce_window_max_uses_stride_in_index_arithmetic() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_4d([1, 1, 4, 4]),
        None,
    );
    dag.add_node(
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Max,
            window_shape: vec![2, 2],
            strides: vec![2, 2],
        },
        vec![load],
        tensor_4d([1, 1, 2, 2]),
        None,
    );
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("* 2 + __w0"),
        "stride>1 emit must multiply the output index by the stride along axis 0, got:\n{src}"
    );
    assert!(
        src.contains("* 2 + __w1"),
        "stride>1 emit must multiply the output index by the stride along axis 1, got:\n{src}"
    );
}
