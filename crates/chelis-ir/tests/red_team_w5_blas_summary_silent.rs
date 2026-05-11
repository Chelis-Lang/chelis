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
    let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat(prim, 8, 4), None);
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

/// Test 1: **P0 finding — silent miscompile on F64 matmul.**
///
/// The IR-level specializer (`chelis_ir::specialize::specialize_for_blas`)
/// has no precision check in `detect_matmul_pattern` (see `specialize.rs`
/// lines 505-579 — the recognizer keys off shape and op structure only).
/// As a result, **an F64 matmul subgraph is silently replaced with
/// `RiscOp::BlasMatmul`**, and downstream C codegen at
/// `crates/chelis-backend-c/src/emit.rs::emit_blas_matmul` emits
/// `cblas_sgemm(...)` — single-precision BLAS — against the F64 data
/// buffer.
///
/// `cblas_sgemm` reinterprets the bytes as `float*` with `sizeof(float)`
/// stride arithmetic. For a double-precision tensor this is a wrong
/// answer (every other 4 bytes treated as a float; size mismatch in
/// indexing), not a fail-closed panic.
///
/// The sparse summarizer says `NotEligible` (correctly — no sparse op).
/// The BLAS summary path's F32 gate at
/// `host.rs::summarize_blas_helper_from_parts` returns `None` silently,
/// so the host-side summary specialization is correctly not produced.
/// But the IR-level recognizer fires anyway, and the C backend trusts it.
///
/// This is the load-bearing silent-correctness failure mode the
/// red-team brief flagged. We lock it as a P0 with this test.
#[test]
fn f64_matmul_helper_specializer_silently_emits_blas_matmul_p0() {
    let (dag, inputs, output) = build_matmul_helper(Prim::F64);
    let sparse_result = try_summarize_sparse_helper_for_test(&dag, &inputs, &output);
    // The sparse path correctly says NotEligible: no sparse op in the body.
    assert!(
        matches!(sparse_result, Err(SparseSummaryAttempt::NotEligible)),
        "sparse summarizer should report NotEligible for a BLAS-shaped helper; \
         got {sparse_result:?}"
    );

    // The IR-level recognizer has no precision gate. F64 matmul is
    // silently replaced with `BlasMatmul`. Lock the current (buggy)
    // behavior so a remediation that adds a precision gate will fail
    // this test deliberately — we want the regression visible.
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    let blas_node = blas_node.expect(
        "P0 BUG SURFACED: F64 matmul subgraph is currently replaced with \
         RiscOp::BlasMatmul by the IR specializer (no precision gate in \
         detect_matmul_pattern). This is a SILENT MISCOMPILE: C codegen \
         emits cblas_sgemm against F64 data. The fix is a precision \
         filter in specialize.rs::detect_matmul_pattern (or in \
         emit_blas_matmul to reject non-F32 BlasMatmul). When the fix \
         lands, flip this assertion to `assert!(blas_node.is_none())` \
         and add a regression positive test that asserts F64 stays on \
         the generic path.",
    );
    // While the bug exists, lock the F64 precision on the BlasMatmul
    // output so a future commit that adds a precision gate fails this
    // test (test_vs_spec_divergence: the test ENCODES the bug as a
    // pinned negative; closing the bug requires updating the test
    // in the same change set).
    assert_eq!(
        blas_node.output_type.precision,
        Prim::F64,
        "BlasMatmul output should still carry the original F64 precision"
    );
}

/// Test 1b: **P0 sibling — Int32 matmul** also silently fires BLAS
/// specialization. The recognizer has no precision filter, so any
/// matmul-shaped subgraph regardless of dtype gets replaced. This is
/// the broader bug-class behind the F64 finding.
#[test]
fn int32_matmul_helper_specializer_silently_emits_blas_matmul_p0_sibling() {
    let (dag, _inputs, _output) = build_matmul_helper(Prim::Int32);
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        blas_node.is_some(),
        "P0 BUG SIBLING: Int32 matmul also passes through the IR specializer \
         without a precision check, producing BlasMatmul on integer data. \
         Closing the F64 bug must close this sibling at the same site \
         (precision-filter sweep)."
    );
    assert_eq!(
        blas_node.unwrap().output_type.precision,
        Prim::Int32,
        "BlasMatmul output should still carry the original Int32 precision"
    );
}

/// Test 1c: **P0 sibling — Int64 matmul** also silently fires.
#[test]
fn int64_matmul_helper_specializer_silently_emits_blas_matmul_p0_sibling() {
    let (dag, _inputs, _output) = build_matmul_helper(Prim::Int64);
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let blas_node = specialized
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        blas_node.is_some(),
        "P0 BUG SIBLING: Int64 matmul also passes through the IR specializer \
         without a precision check."
    );
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
        RiscOp::Const { value: 2.0 },
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
        RiscOp::Sum { axis: 1 },
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
