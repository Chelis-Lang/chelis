//! BLAS specialization controls for f32, integer, and non-matmul
//! helper DAGs. The sparse recognizer reports `NotEligible` for these
//! non-sparse shapes. Structured BLAS rejection coverage lives in
//! `host_blas_summary_diagnostics.rs`.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::host::{
    HostTensorInput, SparseSummaryAttempt, try_summarize_sparse_helper_for_test,
};
use chelis_types::types::Prim;

fn mat(prim: Prim, r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: prim,
    }
}

fn t3(prim: Prim, a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: prim,
    }
}

/// Build a canonical Tier-2 matmul subgraph (Expand × Expand → Mul → Sum)
/// returning `(dag, [HostTensorInput], output_ty)` that mirrors what the
/// host-lowering passes hand to the summarizers.
fn build_matmul_helper(prim: Prim) -> (Dag, Vec<HostTensorInput>, TensorType) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(prim, 8, 16),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(prim, 16, 4),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t3(prim, 8, 16, 4),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(8),
        },
        vec![b],
        t3(prim, 8, 16, 4),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![ea, eb], t3(prim, 8, 16, 4), None);
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(prim, 8, 4),
        None,
    );
    dag.add_root(sum);

    let inputs = vec![
        HostTensorInput {
            name: "a".into(),
            ty: mat(prim, 8, 16),
        },
        HostTensorInput {
            name: "b".into(),
            ty: mat(prim, 16, 4),
        },
    ];
    (dag, inputs, mat(prim, 8, 4))
}

/// Regression lock — Int32 matmul stays off the BLAS path.
#[test]
fn int32_matmul_helper_specializer_stays_off_blas_path() {
    let (dag, _inputs, _output) = build_matmul_helper(Prim::Int32);
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        blas_node.is_none(),
        "Int32 matmul must not produce RiscOp::BlasMatmul",
    );
}

/// Regression lock — Int64 matmul stays off the BLAS path.
#[test]
fn int64_matmul_helper_specializer_stays_off_blas_path() {
    let (dag, _inputs, _output) = build_matmul_helper(Prim::Int64);
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        blas_node.is_none(),
        "Int64 matmul must not produce RiscOp::BlasMatmul",
    );
}

/// Positive regression: F32 matmul MUST still produce BlasMatmul.
/// Prevents an overshooting fix that accidentally rejects F32 too.
#[test]
fn f32_matmul_helper_specializer_still_hits_blas_path() {
    let (dag, _inputs, _output) = build_matmul_helper(Prim::F32);
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        blas_node.is_some(),
        "F32 matmul must still hit the BLAS specialization path",
    );
    assert_eq!(blas_node.unwrap().output_type.precision, Prim::F32);
}

/// Test 2: A rank-3 matmul helper body (rank-3 inputs producing a rank-3
/// output via a non-rank-2-leading subgraph) — outside the BLAS
/// recognizer's matmul shape rule. The recognizer returns None silently.
///
/// Hand-build a non-matmul-shaped graph that still produces a tensor
/// output: simple `Mul` of two rank-2 inputs of different dtypes — wait,
/// must keep one precision. Use Add as the root: not a matmul-shaped
/// subgraph at all. The BLAS summarizer rejects because the specialized
/// root is not `BlasMatmul`. Silent.
#[test]
fn non_matmul_shape_helper_silently_misses_blas_summary() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 4, 4),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(Prim::F32, 4, 4),
        None,
    );
    // Root is Add — not matmul-shaped.
    let add = dag.add_node(decl, RiscOp::Add, vec![a, b], mat(Prim::F32, 4, 4), None);
    dag.add_root(add);

    let inputs = vec![
        HostTensorInput {
            name: "a".into(),
            ty: mat(Prim::F32, 4, 4),
        },
        HostTensorInput {
            name: "b".into(),
            ty: mat(Prim::F32, 4, 4),
        },
    ];
    let output = mat(Prim::F32, 4, 4);

    let sparse_result = try_summarize_sparse_helper_for_test(&dag, &inputs, &output);
    assert!(
        matches!(sparse_result, Err(SparseSummaryAttempt::NotEligible)),
        "sparse summarizer says NotEligible for Add-rooted helper; got {sparse_result:?}"
    );

    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let has_blas = specialized
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        !has_blas,
        "Add-rooted helper must NOT be recognized as matmul; got BlasMatmul"
    );
}

/// A constant operand is not a direct helper input, so this DAG does
/// not qualify for the direct-input BLAS summary.
#[test]
fn const_operand_helper_silently_misses_blas_summary() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    // Inline constant as the rhs (16x4).
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(mat(Prim::F32, 16, 4).precision, 2.0),
        vec![],
        mat(Prim::F32, 16, 4),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(8),
        },
        vec![b],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![ea, eb],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(sum);

    // The IR-level BLAS recognizer DOES specialize (the const is a
    // contiguous-leaf operand from the matcher's perspective). The
    // *summary* recognizer is what rejects: it requires both operands
    // to map to helper inputs by Load name. Lock the IR-level
    // ground truth here:
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    assert!(
        specialized
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::BlasMatmul { .. })),
        "IR-level specializer must specialize the const-operand matmul to BlasMatmul; \
         the helper-summary rejection happens at a layer above the recognizer"
    );

    // Only one input is exposed (the loaded `a`); the const is internal.
    let inputs = vec![HostTensorInput {
        name: "a".into(),
        ty: mat(Prim::F32, 8, 16),
    }];
    let output = mat(Prim::F32, 8, 4);
    let sparse_result = try_summarize_sparse_helper_for_test(&dag, &inputs, &output);
    // No sparse op present → NotEligible.
    assert!(
        matches!(sparse_result, Err(SparseSummaryAttempt::NotEligible)),
        "sparse summarizer should report NotEligible for a BLAS-shaped const-operand helper; \
         got {sparse_result:?}"
    );
}
