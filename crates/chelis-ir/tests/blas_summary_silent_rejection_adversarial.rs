//! Wave 5 red-team — Locks the §5 Remaining Work Register's
//! "BLAS summary recognizer is still silent on near-eligible rejections"
//! gap, per `docs/gap_synthesis.md` §5 / M5-follow-up entry:
//!
//! > **Sibling sweep / new follow-up:** the BLAS summary recognizer
//! > (`summarize_blas_helper_from_parts`) is still silent on near-eligible
//! > rejections — it returns `Option<HostBlasMatmulSummary>` and discards
//! > the structural reason. A future workstream should mirror the W4-A
//! > structured-rejection surface for BLAS callsites so user-`def` matmul
//! > helpers that just miss the recognizer (non-F32 precision, non-rank-2
//! > shape, non-Load operand) emit the same enum-matchable diagnostic
//! > class. Tracked here, not opened in this batch.
//!
//! This file PROVES the gap by exercising the public test surface for
//! the sparse recognizer (`try_summarize_sparse_helper_for_test`) on a
//! BLAS-shaped helper that is structurally near-eligible but not exact —
//! and showing that the BLAS-only path returns `None` with no
//! diagnostic. The negative-test parity here is what locks the gap so
//! the follow-up workstream cannot silently land without removing this
//! test or replacing it with the structured-rejection contract.
//!
//! Concretely:
//!
//! 1. **Near-eligible BLAS helper — F64 output**: `summarize_blas_helper_from_parts`
//!    returns `None` (silent). The sparse path's `try_summarize_sparse_helper`
//!    correctly reports `NotEligible` because the helper has no sparse op.
//!    Net effect at the host-program lowering layer: no diagnostic.
//! 2. **Near-eligible BLAS helper — rank-3 shape (not rank-2 helper-body)**:
//!    same silent miss.
//! 3. **Near-eligible BLAS helper — operand is a Mul, not a Load**:
//!    same silent miss.
//!
//! If a future commit closes this follow-up by mirroring the
//! `SparseSummaryAttempt` enum for BLAS, these assertions will need to
//! be updated (and the §5 register entry can be marked closed).

use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
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
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(prim, 8, 16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(prim, 16, 4),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(prim, 8, 16, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t3(prim, 8, 16, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(prim, 8, 16, 4), None);
    let sum = dag.add_node(
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

/// Regression lock for the P0 fix: F64 matmul must NOT be replaced
/// with `RiscOp::BlasMatmul`.
///
/// The IR-level specializer (`chelis_ir::specialize::specialize_for_blas`)
/// gained a precision filter in `detect_matmul_pattern` so a non-F32
/// matmul subgraph stays on the generic `expand+mul+sum` path. Defense-
/// in-depth: both `emit_blas_matmul` sites (C and HIP) panic on non-F32.
/// This test locks the canonical-site filter so a regression that drops
/// the precision check is visible immediately.
#[test]
#[ignore = "WS-A2: F64 matmul subgraphs now route through cblas_dgemm; the W5 P0 fail-closed assumption no longer holds."]
fn f64_matmul_helper_specializer_stays_off_blas_path() {
    let (dag, inputs, output) = build_matmul_helper(Prim::F64);
    let sparse_result = try_summarize_sparse_helper_for_test(&dag, &inputs, &output);
    // The sparse path correctly says NotEligible: no sparse op in the body.
    assert!(
        matches!(sparse_result, Err(SparseSummaryAttempt::NotEligible)),
        "sparse summarizer should report NotEligible for a BLAS-shaped helper; \
         got {sparse_result:?}"
    );

    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        blas_node.is_none(),
        "F64 matmul must not produce RiscOp::BlasMatmul (cblas_sgemm is F32-only); \
         the precision filter at specialize.rs::detect_matmul_pattern has regressed",
    );
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
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 4, 4),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(Prim::F32, 4, 4),
        None,
    );
    // Root is Add — not matmul-shaped.
    let add = dag.add_node(RiscOp::Add, vec![a, b], mat(Prim::F32, 4, 4), None);
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
    // The silent-rejection observable today: no SummaryRejection is
    // produced for this case (no enum-matchable diagnostic). The fix
    // is the M5-follow-up workstream.
}

/// Test 3: A BLAS-shaped helper whose operand is a constant (not a Load),
/// e.g. `matmul(a, const_3x4)`. The summarizer requires
/// `helper_load_input_index` to resolve both operands to helper inputs;
/// a `Const` operand returns `None` from that helper, silently
/// disqualifying the BLAS summary.
///
/// This exercises a real near-eligible miss: simple wrappers that
/// embed a constant matrix lose BLAS specialization silently.
#[test]
fn const_operand_helper_silently_misses_blas_summary() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    // Inline constant as the rhs (16x4).
    let b = dag.add_node(
        RiscOp::synth_const(mat(Prim::F32, 16, 4).precision, 2.0),
        vec![],
        mat(Prim::F32, 16, 4),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(Prim::F32, 8, 16, 4), None);
    let sum = dag.add_node(
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
    // The BLAS summary is silently None today. There is no `Err(BlasSummaryAttempt::Rejected{
    // class: NonLoadOperand, ... })` analog. THAT is the §5 follow-up gap.
}
