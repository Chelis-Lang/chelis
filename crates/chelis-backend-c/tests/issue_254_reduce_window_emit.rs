//! Issue #254 — C-backend emission verification for the four
//! `reduce_window_*` primitives.
//!
//! Pin the structural invariants of the emitted C code so we catch
//! silent regressions where (for example) `Mean` forgets to divide by
//! the window volume or extrema stop selecting the first stored value. Numerical
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
    let decl = dag.declare("test");
    let load = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_4d([1, 1, 4, 4]),
        None,
    );
    dag.add_node(
        decl,
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
fn issue254_emit_reduce_window_max_selects_first_nan_without_fmaxf() {
    let dag = build_dag(ReduceWindowKind::Max);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        !src.contains("fmaxf("),
        "Max emit must not use NaN-dropping fmaxf, got:\n{src}"
    );
    assert!(
        src.contains("isnan(candidate) || candidate > best_value"),
        "Max emit must select the first NaN or strict greater value, got:\n{src}"
    );
    assert!(
        src.contains(")[outer] = ((const float*)t0_data)[best_src]"),
        "Max emit must copy the selected stored representation, got:\n{src}"
    );
    assert!(src.contains("chelis_window_count("));
    assert!(src.contains("for (int64_t leaf = 0; leaf < t1_window_count;"));
    assert!(src.contains("chelis_window_index("));
    assert!(!src.contains("full_indices[") && !src.contains("out_indices["));
}

#[test]
fn issue254_emit_reduce_window_min_selects_first_nan_without_fminf() {
    let dag = build_dag(ReduceWindowKind::Min);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        !src.contains("fminf("),
        "Min emit must not use NaN-dropping fminf, got:\n{src}"
    );
    assert!(
        src.contains("isnan(candidate) || candidate < best_value"),
        "Min emit must select the first NaN or strict lesser value, got:\n{src}"
    );
}

#[test]
fn issue254_emit_reduce_window_sum_uses_adjacent_pair_tree() {
    let dag = build_dag(ReduceWindowKind::Sum);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("while (level_n > 1)"),
        "Sum emit must use the canonical balanced tree, got:\n{src}"
    );
    assert!(
        src.contains("level[__pair_left] + level[__pair_right]"),
        "Sum emit must combine adjacent pairs, got:\n{src}"
    );
    // Sum must NOT emit a division by the window volume; that's the
    // distinguishing tell vs. Mean.
    assert!(
        !src.contains("result = result / "),
        "Sum emit must NOT divide by window volume, got:\n{src}"
    );
}

#[test]
fn issue254_emit_reduce_window_mean_divides_by_window_volume() {
    let dag = build_dag(ReduceWindowKind::Mean);
    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(
        src.contains("while (level_n > 1)")
            && src.contains("level[__pair_left] + level[__pair_right]"),
        "Mean emit must first use the canonical balanced sum, got:\n{src}"
    );
    // Window volume = 2 * 2 = 4.
    assert!(
        src.contains("result = result / (float)t1_window_count;"),
        "Mean emit must divide by the window volume (4 for 2x2), got:\n{src}"
    );
}

/// Stride > 1 is transported exactly to the checked projection plan.
#[test]
fn issue254_emit_reduce_window_max_uses_stride_in_index_arithmetic() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let load = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_4d([1, 1, 4, 4]),
        None,
    );
    dag.add_node(
        decl,
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
    let plan = src
        .lines()
        .find(|line| line.contains("= chelis_tensor_window_plan("))
        .unwrap();
    assert!(plan.contains("(chelis_scalar[]){chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C(2)), chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C(2))}"));
    assert!(src.contains("chelis_window_index("));
}

#[test]
fn reduced_float_windowed_reduction_uses_f32_arithmetic_and_f16_storage() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let input_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F16,
    };
    let load = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        input_ty,
        None,
    );
    dag.add_node(
        decl,
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

    let src = codegen(&dag, "kernel").unwrap().c_source;
    assert!(src.contains("chelis_f16_to_f32"));
    assert!(src.contains("chelis_f32_to_f16"));
    assert!(src.contains("CHELIS_DTYPE_F32"));
    assert!(
        src.contains("chelis_alloc(1, (int64_t[]){ 3 }, CHELIS_DTYPE_F16)"),
        "the result must retain f16 storage:\n{src}"
    );
}

#[test]
fn runtime_symbolic_window_extent_names_dynamic_shape_owner() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let load = dag.add_node(
        decl,
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
        decl,
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
