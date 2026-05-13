//! Wave 7 fresh-context red-team — BLAS rejection diagnostics, non-precision paths.
//!
//! Per the Wave 7 plan § "Attack surface (other) — BLAS rejection diagnostics
//! (W6 Task A) — non-precision paths":
//!
//!   * pattern-match every variant other than the precision ones
//!     (covered separately in `red_team_w7_blas_cross_product.rs`)
//!   * verify no false positives on accepted F32 matmul callsites
//!   * adversarial near-miss helpers:
//!     - rank-3-output-with-rank-2-operands (handled by W6's
//!       `subcase_b_malformed_blas_root`)
//!     - operand that's a `Cast` wrapping a `Load`
//!     - dimension binding that fails because of a renamed symbol
//!     - `BlasMultipleRoots` via a hand-constructed multi-output DAG
//!       with one matmul-near and one non-matmul root
//!     - `BlasMultipleRoots` via two matmul-near roots sharing inputs
//!       (W6 covers identical shape; we add a heterogeneous-shape variant)
//!
//! Plus boundary tests on the `is_matmul_near` pre-eligibility check:
//!
//!   * almost-matmul: `Sum(Mul(Load, Load))` — no Expand wrapping → NotEligible
//!   * almost-matmul: `Sum(Mul(Expand, Load))` — one Expand → NotEligible
//!   * almost-matmul: `Sum(Add(Expand, Expand))` — wrong inner op → NotEligible
//!   * `is_matmul_near` correctly recognizes both BlasMatmul-rooted and
//!     Sum(Mul(Expand,Expand))-rooted helpers as near.
//!
//! ## Pattern-match contract
//!
//! Every test pattern-matches on
//! `BlasSummaryAttempt::{NotEligible, Rejected(_)}`,
//! `HelperSummaryRejection::rejection_class`, and the structured
//! `SummaryRejectionDetail` variant.

use chelis_ir::dag::{Dag, DimExpr, RiscOp};
use chelis_ir::host::{
    BlasDimRole, BlasSummaryAttempt, HelperSummaryRejection, HostBlasMatmulSummary,
    HostTensorInput, SummaryRejectionClass, SummaryRejectionDetail,
    try_summarize_blas_helper_for_test,
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

fn expect_rejected(
    attempt: Result<HostBlasMatmulSummary, BlasSummaryAttempt>,
) -> HelperSummaryRejection {
    match attempt {
        Ok(summary) => panic!("expected Rejected(_), got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => panic!("expected Rejected(_), got NotEligible"),
        Err(BlasSummaryAttempt::Rejected(r)) => r,
    }
}

fn expect_not_eligible(attempt: Result<HostBlasMatmulSummary, BlasSummaryAttempt>) {
    match attempt {
        Err(BlasSummaryAttempt::NotEligible) => {}
        Ok(summary) => panic!("expected NotEligible, got Ok({summary:?})"),
        Err(BlasSummaryAttempt::Rejected(r)) => panic!(
            "expected NotEligible, got Rejected({:?} / {:?})",
            r.rejection_class, r.detail,
        ),
    }
}

// =========================================================================
// BlasMultipleRoots — heterogeneous root shapes (one matmul-near +
// one elementwise). W6 covers two identical matmul-near roots; this
// test covers the more adversarial case where only ONE root is
// matmul-near and the recognizer must still hit the multi-root gate
// (because `is_matmul_near` returns true for any root).
// =========================================================================

#[test]
fn blas_multiple_roots_one_matmul_near_one_elementwise_emits_structured_rejection() {
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
    // First root: canonical matmul-near (becomes BlasMatmul after specialize)
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
    // Second root: pure elementwise Add (not matmul-near). The
    // recognizer's any_matmul_near() check must still see the FIRST
    // root, fire BlasMultipleRoots.
    let other_a = dag.add_node(
        RiscOp::Load { name: "c".into() },
        vec![],
        mat(Prim::F32, 8, 4),
        None,
    );
    let other_b = dag.add_node(
        RiscOp::Load { name: "d".into() },
        vec![],
        mat(Prim::F32, 8, 4),
        None,
    );
    let add_root = dag.add_node(
        RiscOp::Add,
        vec![other_a, other_b],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(sum);
    dag.add_root(add_root);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
        input("c", mat(Prim::F32, 8, 4)),
        input("d", mat(Prim::F32, 8, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    let rejection = expect_rejected(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasMultipleRoots,
    );
    let SummaryRejectionDetail::BlasMultipleRoots { root_count } = rejection.detail else {
        panic!(
            "expected BlasMultipleRoots detail, got {:?}",
            rejection.detail
        );
    };
    assert_eq!(root_count, 2);
}

// =========================================================================
// BlasMultipleRoots negative parity: TWO non-matmul-near roots
// (both elementwise). The recognizer's any_matmul_near() pre-gate
// must say "not BLAS-near" → NotEligible, NOT BlasMultipleRoots.
//
// This is exactly the contract the plan calls out: "Even a multi-root
// helper qualifies as 'BLAS-near' only if at least one root is
// matmul-shape." If a future regression weakens the gate so any
// multi-root helper produces BlasMultipleRoots, this test fails.
// =========================================================================

#[test]
fn two_non_matmul_roots_return_not_eligible_not_rejected() {
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
    let add1 = dag.add_node(RiscOp::Add, vec![a, b], mat(Prim::F32, 4, 4), None);
    let mul1 = dag.add_node(RiscOp::Mul, vec![a, b], mat(Prim::F32, 4, 4), None);
    dag.add_root(add1);
    dag.add_root(mul1);

    let inputs = vec![
        input("a", mat(Prim::F32, 4, 4)),
        input("b", mat(Prim::F32, 4, 4)),
    ];
    let output = mat(Prim::F32, 4, 4);
    expect_not_eligible(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
}

// =========================================================================
// BlasNotMatmulPattern — root is `Reshape` wrapping the matmul.
//
// Today the IR specializer (`specialize_for_blas`) only replaces
// `Sum(Mul(Expand,Expand))` subgraphs with `RiscOp::BlasMatmul` at the
// root level — it does NOT recurse into intermediate ops. So a Reshape
// over a matmul stays as Reshape(Sum(Mul(...))). The W6 recognizer's
// `is_matmul_near` says "root is Reshape, not Sum or BlasMatmul" →
// NotEligible.
//
// The adversarial intent here: if a future change ever makes Reshape
// _transparent_ to `is_matmul_near`, this test catches it. Today it
// must remain NotEligible.
// =========================================================================

#[test]
fn root_reshape_wrapping_matmul_returns_not_eligible() {
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
    // Wrap the matmul in a Reshape (flatten 8x4 → 32).
    let reshape = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(32)],
        },
        vec![sum],
        TensorType {
            dims: vec![DimInfo::Lit(32)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(reshape);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = TensorType {
        dims: vec![DimInfo::Lit(32)],
        precision: Prim::F32,
    };
    // The root is Reshape, which is NOT matmul-near → NotEligible.
    // (Documentation lock: if this becomes Rejected(...) in the
    // future, the contract has been weakened to "any helper wrapping
    // a matmul should be rejected"; that's a new design decision and
    // the plan would need to surface it.)
    expect_not_eligible(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
}

// =========================================================================
// BlasNonLoadOperand — operand is a `Cast` wrapping a `Load`.
//
// Helper has F32 matmul body BUT the LHS operand is `Cast(Load("a"))`
// (where the Load is, say, F64 → F32 cast to satisfy the matmul's
// F32 precision). The recognizer's `helper_load_input_index` looks
// for a direct `Load` whose precision matches the input declared on
// the helper; a Cast intervenes, so the resolution fails →
// BlasNonLoadOperand.
//
// Hand-crafted at the BlasMatmul level (skipping
// `specialize_for_blas` because adding a Cast in the middle of
// Expand×Expand×Mul×Sum would prevent the specializer from folding).
// =========================================================================

#[test]
fn blas_non_load_operand_cast_wrapping_load_emits_structured_rejection() {
    let mut dag = Dag::new();
    // LHS is an F64 Load that gets Cast'd to F32 before reaching the
    // matmul. RHS is a direct F32 Load.
    let a_f64 = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F64, 8, 16),
        None,
    );
    let a_cast = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![a_f64],
        mat(Prim::F32, 8, 16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(Prim::F32, 16, 4),
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
        vec![a_cast, b],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(blas);

    let inputs = vec![
        input("a", mat(Prim::F64, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    let rejection = expect_rejected(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasNonLoadOperand,
    );
    let SummaryRejectionDetail::BlasNonLoadOperand { operand_index } = rejection.detail else {
        panic!(
            "expected BlasNonLoadOperand detail, got {:?}",
            rejection.detail
        );
    };
    assert_eq!(operand_index, 0);
}

// =========================================================================
// BlasDimensionBindingFailure — Batch role (W6 covers M and K; we
// add the Batch role for sibling-sweep coverage).
//
// 3D batched matmul shape, but the `batch_dim` is a Named symbol that
// doesn't appear on either input. The recognizer's per-role binding
// check fires on the Batch role first (in the canonical Batch → M → N
// → K order).
// =========================================================================

#[test]
fn blas_dimension_binding_failure_unbound_batch_emits_structured_rejection() {
    let mut dag = Dag::new();
    // 3D inputs but the batch dim on the BlasMatmul node uses a Named
    // symbol "unbound_b" that does not appear on either input's
    // declared dim list.
    let a_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(8), DimInfo::Lit(16)],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(16), DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let blas = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Sym("unbound_b".into())],
            m: DimExpr::Concrete(8),
            n: DimExpr::Concrete(4),
            k: DimExpr::Concrete(16),
            accumulator: Prim::F32,
        },
        vec![a, b],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(8), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(blas);

    let inputs = vec![input("a", a_ty), input("b", b_ty)];
    let output = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(8), DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let rejection = expect_rejected(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasDimensionBindingFailure,
    );
    let SummaryRejectionDetail::BlasDimensionBindingFailure { role } = rejection.detail else {
        panic!(
            "expected BlasDimensionBindingFailure detail, got {:?}",
            rejection.detail
        );
    };
    assert_eq!(role, BlasDimRole::Batch);
}

#[test]
fn blas_dimension_binding_failure_unbound_n_emits_structured_rejection() {
    // Mirror sibling-sweep for the N role (W6 covers M and K; W7 adds N
    // and Batch).
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
            n: DimExpr::Sym("unbound_n".into()),
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
    let rejection = expect_rejected(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasDimensionBindingFailure,
    );
    let SummaryRejectionDetail::BlasDimensionBindingFailure { role } = rejection.detail else {
        panic!(
            "expected BlasDimensionBindingFailure detail, got {:?}",
            rejection.detail
        );
    };
    assert_eq!(role, BlasDimRole::N);
}

// =========================================================================
// BlasNotMatmulPattern Subcase A — root op is `Sum` (matmul-near per
// `is_matmul_near` because the body is Sum(Mul(Expand,Expand))) but
// the specializer DECLINED to fold to BlasMatmul.
//
// Why decline: today the specializer's `detect_matmul_pattern` rejects
// when the sum axis isn't lead_len + 1 (i.e. wrong axis). The W6
// recognizer's `is_matmul_near` says "matmul-shape", output gate
// passes (F32), then the recognizer reads root_node.op == Sum, hits
// the `other =>` arm, and emits BlasNotMatmulPattern with
// `tail_op = "sum"`.
//
// This locks the "matmul-shape but specializer declined" path
// distinct from the "root op is unrelated" path.
// =========================================================================

#[test]
fn blas_not_matmul_pattern_wrong_sum_axis_emits_structured_rejection_tail_op_sum() {
    // Build the matmul shape but with `Sum { axis: 0 }` instead of
    // axis 1. The specializer's detect_matmul_pattern rejects on the
    // axis check (`sum_axis != lead_len + 1`), so the root stays as
    // `Sum(Mul(Expand,Expand))` post-specialize.
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
    // Wrong axis: sum over axis 0 instead of 1. Output type matches
    // the wrong axis.
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![mul],
        t3(Prim::F32, 1, 16, 4),
        None,
    );
    dag.add_root(sum);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = t3(Prim::F32, 1, 16, 4);
    let rejection = expect_rejected(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasNotMatmulPattern,
    );
    let SummaryRejectionDetail::BlasNotMatmulPattern { tail_op } = rejection.detail else {
        panic!(
            "expected BlasNotMatmulPattern detail, got {:?}",
            rejection.detail
        );
    };
    // The root op was Sum (not BlasMatmul). The canonical name is "sum".
    assert_eq!(tail_op, "sum");
}

// =========================================================================
// `is_matmul_near` boundary tests — almost-matmul shapes that the
// brief calls out as "the boundary worth probing":
//
//   * expand+mul+sum with the wrong axis (Sum axis = 0 instead of 1)
//   * expand+mul+sum with two operands but no Expand wrapping →
//     `Sum(Mul(Load, Load))` shape
//   * expand+mul+sum where one expand is missing (one-sided expand)
//   * Sum-of-Add (wrong inner op)
//
// Each of these must return NotEligible (not BLAS-near).
// =========================================================================

#[test]
fn near_miss_sum_of_mul_of_two_loads_no_expand_returns_not_eligible() {
    // Sum(Mul(Load, Load)) — same precision, same shape, no Expand →
    // not the matmul pattern. is_matmul_near must say false.
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
    let mul = dag.add_node(RiscOp::Mul, vec![a, b], in_ty.clone(), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(sum);

    let inputs = vec![input("a", in_ty.clone()), input("b", in_ty.clone())];
    let output = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    // No Expand → not matmul-near → NotEligible. This locks the
    // is_matmul_near gate so a future regression that drops the
    // Expand check (and treats any Sum-of-Mul-of-two-things as
    // BLAS-near) would emit a false BlasNonLoadOperand here.
    expect_not_eligible(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
}

#[test]
fn near_miss_sum_of_mul_of_expand_and_load_returns_not_eligible() {
    // One-sided expand: only one of the Mul's operands is wrapped in
    // Expand. is_matmul_near requires BOTH operands to be Expand.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F32, 8, 16),
        None,
    );
    // RHS is just a Load matching the post-expand shape.
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t3(Prim::F32, 8, 16, 4),
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
    let mul = dag.add_node(RiscOp::Mul, vec![ea, b], t3(Prim::F32, 8, 16, 4), None);
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

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", t3(Prim::F32, 8, 16, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    expect_not_eligible(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
}

#[test]
fn near_miss_sum_of_add_of_expands_returns_not_eligible() {
    // Sum(Add(Expand, Expand)) — wrong inner op. is_matmul_near
    // requires the inner op to be Mul.
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
    let add = dag.add_node(RiscOp::Add, vec![ea, eb], t3(Prim::F32, 8, 16, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![add],
        mat(Prim::F32, 8, 4),
        None,
    );
    dag.add_root(sum);

    let inputs = vec![
        input("a", mat(Prim::F32, 8, 16)),
        input("b", mat(Prim::F32, 16, 4)),
    ];
    let output = mat(Prim::F32, 8, 4);
    // Inner op is Add, not Mul → not matmul-near → NotEligible.
    expect_not_eligible(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
}

// =========================================================================
// `is_matmul_near` positive boundary — confirm that BOTH accepted
// shapes (BlasMatmul-rooted AND Sum(Mul(Expand,Expand))-rooted) are
// recognized as matmul-near, even when the recognizer subsequently
// rejects for a different reason.
//
// Tested via observation: a non-F32 matmul subgraph (the W5 P0
// rejection case) stays as Sum(Mul(Expand,Expand)) after specialize
// (because the precision filter at specialize.rs:519 declines
// replacement). The W6 recognizer's is_matmul_near gate must still
// say "BLAS-near" so the rejection arm fires —
// `BlasOutputPrecisionMismatch`, NOT a silent skip.
//
// (Already covered by W7 cross-product, but reproduced here as a
// boundary lock independent of that file.)
// =========================================================================

#[test]
#[ignore = "WS-A2: F64 matmul subgraphs are now successfully specialized to BlasMatmul (cblas_dgemm); the W5 P0 fail-closed rejection no longer fires."]
fn near_match_sum_of_mul_of_expand_of_expand_is_blas_near() {
    // F64 matmul subgraph — specialize_for_blas declines (W5 P0).
    // The post-specialize root is the Sum(Mul(Expand,Expand)) shape
    // and is_matmul_near must say true so the rejection fires.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat(Prim::F64, 8, 16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat(Prim::F64, 16, 4),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F64, 8, 16, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t3(Prim::F64, 8, 16, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(Prim::F64, 8, 16, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F64,
        },
        vec![mul],
        mat(Prim::F64, 8, 4),
        None,
    );
    dag.add_root(sum);

    let inputs = vec![
        input("a", mat(Prim::F64, 8, 16)),
        input("b", mat(Prim::F64, 16, 4)),
    ];
    let output = mat(Prim::F64, 8, 4);
    // is_matmul_near MUST say true → recognizer reaches output-precision
    // gate → BlasOutputPrecisionMismatch. If is_matmul_near were
    // weakened, this would silently skip and the test would FAIL with
    // "expected Rejected, got NotEligible".
    let rejection = expect_rejected(try_summarize_blas_helper_for_test(&dag, &inputs, &output));
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::BlasOutputPrecisionMismatch,
    );
}

// =========================================================================
// No false positive: an F32 BlasMatmul-rooted helper accepts.
// Mirror of W6's `f32_matmul_helper_returns_ok_with_summary` but
// driven through the hand-built BlasMatmul-root path (not through
// `specialize_for_blas`).
// =========================================================================

#[test]
fn hand_built_blas_matmul_root_f32_accepts() {
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
    let blas = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(8),
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
    let summary = match attempt {
        Ok(s) => s,
        Err(BlasSummaryAttempt::NotEligible) => {
            panic!("hand-built BlasMatmul-root F32 helper MUST accept; got NotEligible")
        }
        Err(BlasSummaryAttempt::Rejected(r)) => panic!(
            "hand-built BlasMatmul-root F32 helper MUST accept; got Rejected({:?})",
            r.rejection_class,
        ),
    };
    assert_eq!(summary.lhs_input, 0);
    assert_eq!(summary.rhs_input, 1);
    assert_eq!(summary.output.precision, Prim::F32);
}
