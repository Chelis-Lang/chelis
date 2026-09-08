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

mod support;
use chelis_ir::dag::{Dag, DimInfo, ReduceWindowKind, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen;

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
        src.contains("acc += ((const float*)t"),
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
        src.contains("acc += ((const float*)t"),
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

#[test]
fn non_f32_windowed_reduction_names_its_real_implementation_owner() {
    let mut dag = Dag::new();
    let input_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F16,
    };
    let load = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
    dag.add_node(
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Sum,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![load],
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F16,
        },
        None,
    );

    let error = match codegen(&dag, "kernel") {
        Ok(_) => panic!("non-f32 windowed reduction must reject"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "unsupported: op `reduce_window_*` on `f16` tensors in the C DAG emitter \
         (node 1) (codegen:c); unimplemented chelis#729: the C windowed-reduction \
         emitter is f32-only today; cast to f32 before the windowed reduction \
         (spec/05-risc-primitives.md section 2.3.1)"
    );
}

#[test]
fn runtime_symbolic_window_extent_names_dynamic_shape_owner() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![
                DimInfo::Lit(1),
                DimInfo::Lit(1),
                DimInfo::Named("runtime_height".into(), None),
                DimInfo::Lit(4),
            ],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_node(
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Max,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![load],
        TensorType {
            dims: vec![
                DimInfo::Lit(1),
                DimInfo::Lit(1),
                DimInfo::Named("runtime_height".into(), None),
                DimInfo::Lit(3),
            ],
            precision: Prim::F32,
        },
        None,
    );

    let error = match codegen(&dag, "kernel") {
        Ok(_) => panic!("runtime-symbolic window extent must reject"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "unsupported: a `reduce_window_*` windowed axis 2 with a runtime-only \
         symbolic extent on the C DAG emitter (node 1) (codegen:c); unimplemented \
         chelis#600: the windowed output extent floor((d - window) / stride) + 1 is \
         not statically representable; bind the axis to a concrete size \
         (spec/05-risc-primitives.md section 2.3.1)"
    );
}
