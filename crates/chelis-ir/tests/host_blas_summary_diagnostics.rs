//! Wave 6 / W6 Task A — BLAS-rejection structured diagnostic oracle.
//!
//! Sibling of `host_sparse_summary_diagnostics.rs`. Drives the BLAS
//! summary recognizer (`try_summarize_blas_helper_for_test`) on
//! synthetic helper DAGs that exercise the six BLAS-prefixed
//! rejection variants added in W6 Task A. The Surf surface cannot
//! reach every one of these (the type checker blocks
//! precision-mismatched matmul calls before lowering); the synthetic
//! DAG path is the only way to drive each rejection case under a
//! pattern-match contract.
//!
//! ## Categories covered here (six BLAS-prefixed variants)
//!
//!   1. `BlasMultipleRoots` — synthetic two-root matmul-near DAG.
//!   2. `BlasOutputPrecisionMismatch` — F64 matmul helper output.
//!   3. `BlasNotMatmulPattern` — root is Sum (matmul-near) but the
//!      recognizer cannot fold to BlasMatmul, OR root is BlasMatmul
//!      with malformed input count / precision.
//!   4. `BlasNonLoadOperand` — matmul operand is not a direct Load.
//!   5. `BlasInputPrecisionMismatch` — helper input precision != F32.
//!   6. `BlasDimensionBindingFailure` — batch/M/N/K cannot bind to inputs.
//!
//! ## Pattern-match contract
//!
//! Every test pattern-matches on:
//!   * the `BlasSummaryAttempt::Rejected(_)` enum variant
//!   * the `HelperSummaryRejection::rejection_class` enum
//!   * the `HelperSummaryRejection::detail` enum variant + structured fields
//!
//! `contains()` on the rendered `Display` string is explicitly
//! rejected.
//!
//! ## NotEligible parity
//!
//! Two negative tests assert that helpers with no matmul-shape body
//! (e.g. pure-elementwise Add helpers, non-F32 elementwise helpers)
//! return `NotEligible` rather than a structured rejection — the
//! contract is "BLAS recognizer only emits diagnostics on BLAS-near
//! shapes."

use chelis_ir::dag::{Dag, DimExpr, RiscOp};
use chelis_ir::host::{
    BlasDimRole, BlasSummaryAttempt, HelperSummaryRejection, HostTensorInput,
    SummaryRejectionClass, SummaryRejectionDetail, try_summarize_blas_helper_for_test,
};
use chelis_ir::{DimInfo, TensorType};
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

fn input(name: &str, ty: TensorType) -> HostTensorInput {
    HostTensorInput {
        name: name.to_string(),
        ty,
    }
}

/// Unwrap a `BlasSummaryAttempt::Rejected(_)` arm and return its
/// carried `HelperSummaryRejection`. Panics with a clear message if
/// the attempt is `Ok(_)` or `NotEligible` — those represent test
/// fixture bugs.
fn expect_rejected(
    attempt: Result<chelis_ir::host::HostBlasMatmulSummary, BlasSummaryAttempt>,
) -> HelperSummaryRejection {
    match attempt {
        Ok(summary) => panic!("expected BlasSummaryAttempt::Rejected, got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => {
            panic!("expected BlasSummaryAttempt::Rejected, got NotEligible")
        }
        Err(BlasSummaryAttempt::Rejected(r)) => r,
    }
}

/// Build a canonical Tier-2 matmul subgraph (Expand × Expand → Mul → Sum)
/// of the given precision: `[8, 16] @ [16, 4] → [8, 4]`. Mirrors the
/// shape `blas_summary_silent_rejection_adversarial.rs` builds.
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

    let inputs = vec![input("a", mat(prim, 8, 16)), input("b", mat(prim, 16, 4))];
    (dag, inputs, mat(prim, 8, 4))
}

// =========================================================================
// Positive: F32 matmul helper → Ok with HostBlasMatmulSummary
// =========================================================================

#[test]
fn f32_matmul_helper_returns_ok_with_summary() {
    let (dag, inputs, output) = build_matmul_helper(Prim::F32);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let summary = match attempt {
        Ok(summary) => summary,
        Err(BlasSummaryAttempt::NotEligible) => {
            panic!("F32 matmul helper must summarize Ok, got NotEligible")
        }
        Err(BlasSummaryAttempt::Rejected(r)) => panic!(
            "F32 matmul helper must summarize Ok, got Rejected({:?} / {:?})",
            r.rejection_class, r.detail,
        ),
    };
    // Lock the recognized summary shape: inputs map by name to lhs=0, rhs=1.
    assert_eq!(summary.lhs_input, 0);
    assert_eq!(summary.rhs_input, 1);
    assert_eq!(summary.output.precision, Prim::F32);
    assert_eq!(summary.input_tys.len(), 2);
    assert_eq!(summary.input_tys[0].precision, Prim::F32);
    assert_eq!(summary.input_tys[1].precision, Prim::F32);
}

// =========================================================================
// Category 2: BlasOutputPrecisionMismatch
// =========================================================================

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted (cblas_dgemm); BlasOutputPrecisionMismatch no longer fires for F64."]
fn blas_output_precision_mismatch_f64_emits_structured_rejection() {
    // F64 matmul subgraph: the recognizer's matmul-near pre-check
    // fires (Sum(Mul(Expand, Expand)) shape) AND the output is F64,
    // so the recognizer emits BlasOutputPrecisionMismatch.
    let (dag, inputs, output) = build_matmul_helper(Prim::F64);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasOutputPrecisionMismatch,
    );
    let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasOutputPrecisionMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(observed, Prim::F64);
}

#[test]
fn blas_output_precision_mismatch_int32_emits_structured_rejection() {
    // Int32 matmul subgraph — same shape, different precision.
    let (dag, inputs, output) = build_matmul_helper(Prim::Int32);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasOutputPrecisionMismatch,
    );
    let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasOutputPrecisionMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(observed, Prim::Int32);
}

// =========================================================================
// Category 1: BlasMultipleRoots
// =========================================================================

#[test]
fn blas_multiple_roots_synthetic_helper_emits_structured_rejection() {
    // Two roots: one canonical F32 matmul, one sibling matmul. After
    // specialize, the DAG has two BlasMatmul roots (or two
    // matmul-near roots — either way, the recognizer's multi-root
    // gate fires).
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(Prim::F32, 16, 4),
        None,
    );
    // First matmul subgraph.
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
    let mul1 = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(Prim::F32, 8, 16, 4), None);
    let sum1 = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul1],
        mat(Prim::F32, 8, 4),
        None,
    );
    // Second matmul subgraph (sibling, identical shape, sharing the
    // same inputs).
    let ea2 = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let eb2 = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t3(Prim::F32, 8, 16, 4),
        None,
    );
    let mul2 = dag.add_node(RiscOp::Mul, vec![ea2, eb2], t3(Prim::F32, 8, 16, 4), None);
    let sum2 = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul2],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(sum1);
    dag.add_root(sum2);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasMultipleRoots,
    );
    let SummaryRejectionDetail::BlasMultipleRoots { root_count } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasMultipleRoots, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(root_count, 2);
}

// =========================================================================
// Category 3: BlasNotMatmulPattern
//
// Subcase A: root is matmul-near (Sum-of-Mul-of-Expand-of-Expand) but
// `specialize_for_blas` could not fold to BlasMatmul because one
// operand is not a contiguous-leaf matrix (the W5 P0 fix keeps non-F32
// matmuls in the same boat; here we trigger it with an embedded
// constant, which makes the IR-level matcher route differently).
//
// Subcase B: root op is BlasMatmul but its rank/precision is malformed
// (we hand-construct a `RiscOp::BlasMatmul` with one input instead of two).
// =========================================================================

#[test]
fn blas_not_matmul_pattern_subcase_b_malformed_blas_root_emits_structured_rejection() {
    // Hand-built single-input BlasMatmul root — the recognizer's
    // shape-validation gate fires because inputs.len() != 2.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    let bogus = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(8),
            n: DimExpr::Concrete(4),
            k: DimExpr::Concrete(16),
            accumulator: Prim::F32,
        },
        vec![a], // Only one input — malformed.
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(bogus);

    let inputs = vec![input("a", mat(Prim::F32, 8, 16))];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasNotMatmulPattern,
    );
    let SummaryRejectionDetail::BlasNotMatmulPattern { tail_op } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasNotMatmulPattern, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(tail_op, "blas_matmul");
}

// =========================================================================
// Category 4: BlasNonLoadOperand
// =========================================================================

#[test]
fn blas_non_load_operand_const_lhs_emits_structured_rejection() {
    // LHS operand is `Const`, not a `Load` of a helper input. The
    // recognizer rejects with BlasNonLoadOperand at operand_index = 0.
    let mut dag = Dag::new();
    // Inline constant as LHS (8x16).
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
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

    // Only `b` is a helper input; `a` is internal Const.
    let inputs = vec![input("b", mat(Prim::F32, 16, 4))];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasNonLoadOperand,
    );
    let SummaryRejectionDetail::BlasNonLoadOperand { operand_index } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasNonLoadOperand, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(operand_index, 0);
}

#[test]
fn blas_non_load_operand_const_rhs_emits_structured_rejection() {
    // Mirror of the lhs case: RHS is Const, recognizer rejects at
    // operand_index = 1. Sibling-sweep coverage.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
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
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(sum);

    let inputs = vec![input("a", mat(Prim::F32, 8, 16))];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasNonLoadOperand,
    );
    let SummaryRejectionDetail::BlasNonLoadOperand { operand_index } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasNonLoadOperand, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(operand_index, 1);
}

// =========================================================================
// Category 5: BlasInputPrecisionMismatch
//
// Helper has an F32 matmul body but one of the *declared* helper
// inputs has a non-F32 precision. The IR-level recognizer specializes
// the body (because the body's nodes are F32), but the summarizer's
// input-precision check catches the mismatch between body and declared
// signature.
//
// Note: this is a synthetic case the Surf type system blocks; the
// IR-level test surface is the only way to drive it.
// =========================================================================

/// Hand-build a BlasMatmul-rooted DAG with the given Load
/// precisions for input[0] and input[1]. Caller declares matching
/// `inputs` so `helper_load_input_index` succeeds and the recognizer
/// reaches the input-precision gate. Useful for driving
/// `BlasInputPrecisionMismatch` without going through
/// `specialize_for_blas`'s W5 P0 precision filter (which would
/// otherwise leave the body as a Sum/Mul/Expand chain and we'd hit
/// `BlasNotMatmulPattern` first).
fn build_handcrafted_blas_with_load_prims(
    lhs_prim: Prim,
    rhs_prim: Prim,
    out_prim: Prim,
) -> (Dag, Vec<HostTensorInput>, TensorType) {
    let mut dag = Dag::new();
    let lhs_ty = mat(lhs_prim, 8, 16);
    let rhs_ty = mat(rhs_prim, 16, 4);
    let out_ty = mat(out_prim, 8, 4);
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        lhs_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        rhs_ty.clone(),
        None,
    );
    let blas = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(8),
            n: DimExpr::Concrete(4),
            k: DimExpr::Concrete(16),
            accumulator: Prim::F32,
        },
        vec![a, b],
        out_ty.clone(),
        None,
    );
    dag.add_root(blas);

    let inputs = vec![input("a", lhs_ty), input("b", rhs_ty)];
    (dag, inputs, out_ty)
}

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted (cblas_dgemm); F64 inputs no longer trigger BlasInputPrecisionMismatch."]
fn blas_input_precision_mismatch_helper_has_f64_declared_input_emits_structured_rejection() {
    // Hand-built BlasMatmul with F32 output but input[0] Load is F64
    // (declared input matches). The recognizer's BlasMatmul-root
    // gate passes, both operands resolve via helper_load_input_index
    // (matching name + type), and the input-precision gate fires for
    // input_index = 0.
    let (dag, inputs, output) =
        build_handcrafted_blas_with_load_prims(Prim::F64, Prim::F32, Prim::F32);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasInputPrecisionMismatch,
    );
    let SummaryRejectionDetail::BlasInputPrecisionMismatch {
        input_index,
        observed,
    } = rejection.detail
    else {
        panic!(
            "expected SummaryRejectionDetail::BlasInputPrecisionMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(input_index, 0);
    assert_eq!(observed, Prim::F64);
}

#[test]
fn blas_input_precision_mismatch_helper_has_int32_declared_rhs_emits_structured_rejection() {
    // Mirror case: input[1] is Int32, input[0] is F32. The
    // recognizer's first-failing-input scan finds input_index = 1.
    let (dag, inputs, output) =
        build_handcrafted_blas_with_load_prims(Prim::F32, Prim::Int32, Prim::F32);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasInputPrecisionMismatch,
    );
    let SummaryRejectionDetail::BlasInputPrecisionMismatch {
        input_index,
        observed,
    } = rejection.detail
    else {
        panic!(
            "expected SummaryRejectionDetail::BlasInputPrecisionMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(input_index, 1);
    assert_eq!(observed, Prim::Int32);
}

// =========================================================================
// Category 6: BlasDimensionBindingFailure
//
// Helper body has a BlasMatmul whose `m`/`n`/`k` symbols don't bind to
// any helper input's dim list (because the inputs are declared with
// numeric `Lit` dims and the matmul uses a Named dim that's never
// referenced on an input). The recognizer reports the first failing
// role.
// =========================================================================

#[test]
fn blas_dimension_binding_failure_unbound_m_emits_structured_rejection() {
    // Hand-built BlasMatmul whose `m` symbol is a Named dim that
    // doesn't appear on any input. Use concrete dims for n/k/batch so
    // the failing role is unambiguously M.
    let mut dag = Dag::new();
    // Inputs use Lit dims (so `m=Named("unbound_m", None)` is not on either input).
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(8), DimInfo::Lit(16)],
            precision: Prim::F32,
        },
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(16), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let blas = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Sym("unbound_m".into()),
            n: DimExpr::Concrete(4),
            k: DimExpr::Concrete(16),
            accumulator: Prim::F32,
        },
        vec![a, b],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(blas);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasDimensionBindingFailure,
    );
    let SummaryRejectionDetail::BlasDimensionBindingFailure { role } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasDimensionBindingFailure, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(role, BlasDimRole::M);
}

#[test]
fn blas_dimension_binding_failure_unbound_k_emits_structured_rejection() {
    // Mirror case: K is the unbound dim symbol. The recognizer
    // checks batch first, then M, then N, then K, so K-only-failure
    // exercises the trailing role.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(8), DimInfo::Lit(16)],
            precision: Prim::F32,
        },
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(16), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let blas = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(8),
            n: DimExpr::Concrete(4),
            k: DimExpr::Sym("unbound_k".into()),
            accumulator: Prim::F32,
        },
        vec![a, b],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(blas);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasDimensionBindingFailure,
    );
    let SummaryRejectionDetail::BlasDimensionBindingFailure { role } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasDimensionBindingFailure, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(role, BlasDimRole::K);
}

// =========================================================================
// Negative: NotEligible vs Rejected
//
// A helper DAG that contains no matmul-near shape at all (e.g. a
// single elementwise Add, or a Sum-of-elementwise that doesn't have
// the matmul Expand×Expand structure) must NOT produce a rejection —
// it returns NotEligible.
// =========================================================================

#[test]
fn elementwise_add_helper_returns_not_eligible() {
    // Pure elementwise Add — not matmul-near, not BLAS-near.
    let mut dag = Dag::new();
    let in_ty = mat(Prim::F32, 4, 4);
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    let root = dag.add_node(RiscOp::Add, vec![a, b], in_ty.clone(), None);
    dag.add_root(root);

    let inputs = vec![input("a", in_ty.clone()), input("b", in_ty.clone())];
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &in_ty);
    match attempt {
        Err(BlasSummaryAttempt::NotEligible) => {}
        Err(BlasSummaryAttempt::Rejected(r)) => {
            panic!("elementwise Add helper must be NotEligible, not Rejected({r:?})")
        }
        Ok(summary) => panic!("elementwise Add helper must NOT be summarized; got {summary:?}"),
    }
}

#[test]
fn elementwise_f64_helper_returns_not_eligible_not_rejected() {
    // F64 elementwise helper — would falsely fire
    // BlasOutputPrecisionMismatch if the BLAS recognizer didn't
    // gate on a matmul-near body. Locks the "BLAS diagnostics
    // ONLY on BLAS-near shapes" invariant.
    let mut dag = Dag::new();
    let in_ty = mat(Prim::F64, 4, 4);
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    let root = dag.add_node(RiscOp::Mul, vec![a, b], in_ty.clone(), None);
    dag.add_root(root);

    let inputs = vec![input("a", in_ty.clone()), input("b", in_ty.clone())];
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &in_ty);
    match attempt {
        Err(BlasSummaryAttempt::NotEligible) => {}
        Err(BlasSummaryAttempt::Rejected(r)) => panic!(
            "F64 elementwise Mul helper must be NotEligible (no matmul-near body); \
             got false-positive BLAS rejection {r:?}",
        ),
        Ok(summary) => panic!("F64 elementwise Mul must NOT be summarized; got {summary:?}"),
    }
}

#[test]
fn sum_without_matmul_shape_returns_not_eligible() {
    // Sum applied to a single-input chain that's NOT Mul(Expand, Expand).
    // The recognizer must NOT report this as BLAS-near.
    let mut dag = Dag::new();
    let in_ty = mat(Prim::F32, 4, 4);
    let out_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    // Sum directly over a Load, no Mul(Expand, Expand) underneath.
    let root = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![a],
        out_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("a", in_ty)];
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &out_ty);
    match attempt {
        Err(BlasSummaryAttempt::NotEligible) => {}
        Err(BlasSummaryAttempt::Rejected(r)) => panic!(
            "Sum-over-Load helper must be NotEligible (not matmul-near); \
             got false-positive BLAS rejection {r:?}",
        ),
        Ok(summary) => panic!("Sum-over-Load helper must NOT be summarized; got {summary:?}"),
    }
}

// =========================================================================
// HelperSummaryRejection structural sanity — body_span propagation
// =========================================================================

#[test]
fn blas_rejection_helper_body_span_is_threaded_through_when_present() {
    // Construct a BlasMatmul-rooted DAG whose root carries a span_id.
    // Drive a BlasNotMatmulPattern rejection (single-input
    // BlasMatmul) and assert the helper_body_span survives.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    let bogus = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(8),
            n: DimExpr::Concrete(4),
            k: DimExpr::Concrete(16),
            accumulator: Prim::F32,
        },
        vec![a],
        mat(Prim::F32, 8, 4),
        Some("surf:111..222".to_string()),
    );
    dag.add_root(bogus);

    let inputs = vec![input("a", mat(Prim::F32, 8, 16))];
    let output = mat(Prim::F32, 8, 4);
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let rejection = expect_rejected(attempt);
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasNotMatmulPattern,
    );
    assert_eq!(
        rejection.helper_body_span.as_deref(),
        Some("surf:111..222"),
        "helper_body_span must be threaded through when the BLAS helper root carries one",
    );
}

// =========================================================================
// Variant distinctness — the six BLAS variants are all distinct, AND
// distinct from the sparse variants (no accidental aliasing).
// =========================================================================

#[test]
fn six_blas_rejection_classes_are_distinct_from_sparse_variants() {
    let blas_classes = [
        SummaryRejectionClass::BlasMultipleRoots,
        SummaryRejectionClass::BlasOutputPrecisionMismatch,
        SummaryRejectionClass::BlasNotMatmulPattern,
        SummaryRejectionClass::BlasNonLoadOperand,
        SummaryRejectionClass::BlasInputPrecisionMismatch,
        SummaryRejectionClass::BlasDimensionBindingFailure,
    ];
    // All six are pairwise distinct.
    for i in 0..blas_classes.len() {
        for j in (i + 1)..blas_classes.len() {
            assert_ne!(
                blas_classes[i], blas_classes[j],
                "BLAS variants {i} and {j} compared equal -- must be distinct",
            );
        }
    }
    // None of the BLAS variants collapse onto a sparse variant.
    let sparse_classes = [
        SummaryRejectionClass::MultipleRoots,
        SummaryRejectionClass::MultipleReturnPaths,
        SummaryRejectionClass::NonLoadOperand,
        SummaryRejectionClass::PostProcessingAfterSparseOp,
        SummaryRejectionClass::IndicesDTypeMismatch,
        SummaryRejectionClass::PayloadDTypeMismatch,
        SummaryRejectionClass::WildcardDim,
    ];
    for blas in &blas_classes {
        for sparse in &sparse_classes {
            assert_ne!(
                blas, sparse,
                "BLAS variant {blas:?} must not collapse onto sparse variant {sparse:?}",
            );
        }
    }
}

#[test]
fn six_blas_rejection_classes_have_distinct_display() {
    // Display strings are part of the human-readable surface; we
    // lock distinctness so a single rendering doesn't collapse two
    // BLAS variants.
    let pairs = [
        (
            SummaryRejectionClass::BlasMultipleRoots,
            "blas-multiple-roots",
        ),
        (
            SummaryRejectionClass::BlasOutputPrecisionMismatch,
            "blas-output-precision-mismatch",
        ),
        (
            SummaryRejectionClass::BlasNotMatmulPattern,
            "blas-not-matmul-pattern",
        ),
        (
            SummaryRejectionClass::BlasNonLoadOperand,
            "blas-non-load-operand",
        ),
        (
            SummaryRejectionClass::BlasInputPrecisionMismatch,
            "blas-input-precision-mismatch",
        ),
        (
            SummaryRejectionClass::BlasDimensionBindingFailure,
            "blas-dimension-binding-failure",
        ),
    ];
    for (class, expected) in &pairs {
        assert_eq!(class.to_string(), *expected);
    }
}
