//! Wave 7 fresh-context red-team — W5 P0 silent rejection → W6 diagnosed
//! rejection cross-product.
//!
//! Required W7 evidence per the plan
//! `we-just-made-a-crystalline-quasar.md` (Wave 7 § "Cross-product
//! verification: W5 P0 silent rejection → W6 diagnosed rejection
//! (dedicated section)"):
//!
//!   1. Enumerate every user-`def` matmul helper precision the W5
//!      filter rejects — F64, F16, Bf16, F8e4m3, Int8, Int32, Int64,
//!      Bool. (W5 P0 fix at `crates/chelis-ir/src/specialize.rs`
//!      `detect_matmul_pattern` rejects any output precision != F32.)
//!   2. For each precision, build the helper through direct DAG
//!      construction (the IR-level recognizer test entry point) and
//!      verify:
//!        * The IR specializer still does NOT emit
//!          `RiscOp::BlasMatmul` for the matmul subgraph (W5 P0 lock).
//!        * The W6 BLAS summary recognizer emits a structured
//!          rejection with EXACTLY
//!          `SummaryRejectionClass::BlasOutputPrecisionMismatch` (with
//!          `observed = <precision>` matching the input).
//!        * Pattern-match on the enum variant, not `contains()` on
//!          rendered text.
//!   3. No-silent-rejection invariant: assert that EVERY one of the
//!      enumerated precisions produces a `Rejected(_)` arm, never
//!      `Ok(_)` and never `NotEligible`. The trap test
//!      `nonsilent_rejection_invariant_for_every_w5_rejected_precision`
//!      collects all eight precisions and asserts they all emit a
//!      structured rejection — the contract the W6 cross-product
//!      closes. A regression that re-introduced silent skip would
//!      cause that test to fail with an explicit precision named.
//!
//! ## Surf-reachable vs DAG-only paths
//!
//! Surf accepts `matmul(a, b)` for any precision as long as `lhs_prec
//! == rhs_prec` (see
//! `crates/chelis-types/src/infer.rs::check_matmul_signature`). Some
//! exotic precisions (`f8e4m3`, `bf16`) may not lower cleanly through
//! every IR pass; this file's IR-level tests drive the recognizer
//! directly via the public test entry point
//! `try_summarize_blas_helper_for_test`, bypassing pipeline
//! preconditions that might block exotic precisions.
//!
//! A separate Surf-end-to-end cross-product test lives in
//! `crates/chelis-cli/tests/red_team_w7_blas_precision_cross_product.rs`
//! and exercises F64 through the full pipeline (the W6 file's
//! `surface_f64_matmul_helper_emits_blas_output_precision_mismatch_rejection`
//! covers F64 specifically; this red-team file extends it to a
//! matrix of Surf-reachable precisions).

use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::host::{
    BlasSummaryAttempt, HostTensorInput, SummaryRejectionClass, SummaryRejectionDetail,
    try_summarize_blas_helper_for_test,
};
use chelis_ir::specialize::specialize_for_blas;
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

/// Build a canonical Tier-2 matmul subgraph (Expand × Expand → Mul → Sum)
/// for the given precision: `[8, 16] @ [16, 4] → [8, 4]`. Same shape the
/// W6 test file's `build_matmul_helper` uses; reproduced here so this
/// red-team file is self-contained and doesn't import test fixtures
/// from a sibling integration-test crate.
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
            accumulator: prim,
        },
        vec![mul],
        mat(prim, 8, 4),
        None,
    );
    dag.add_root(sum);

    let inputs = vec![input("a", mat(prim, 8, 16)), input("b", mat(prim, 16, 4))];
    (dag, inputs, mat(prim, 8, 4))
}

/// Helper: assert that the IR specializer keeps a non-F32 matmul
/// subgraph OFF the `RiscOp::BlasMatmul` path (the W5 P0 fix lock).
fn assert_specializer_keeps_off_blas_path(prim: Prim) {
    let (dag, _, _) = build_matmul_helper(prim);
    let specialized = specialize_for_blas(&dag);
    for id in specialized.roots() {
        let node = specialized.get(*id).expect("root must resolve");
        assert!(
            !matches!(&node.op, RiscOp::BlasMatmul { .. }),
            "W5 P0 fix regressed: {prim:?} matmul subgraph was replaced with \
             `RiscOp::BlasMatmul`; specializer must keep non-F32 matmul \
             on the generic Expand×Expand→Mul→Sum path. Got root op: {:?}",
            node.op,
        );
    }
}

// =========================================================================
// W5 P0 cross-product: F64
//
// Already covered by W6's
// `blas_output_precision_mismatch_f64_emits_structured_rejection`. This
// red-team test re-verifies the SAME contract under a fresh-context
// adversarial pass: (1) W5 P0 specializer lock, (2) W6 structured
// rejection on the helper recognizer.
// =========================================================================

#[test]
#[ignore = "WS-A2/A3 lift: F64/Bf16/F16 BLAS matmul are now admitted (cblas_dgemm / hipblasGemmEx); the W5 P0 fail-closed assumption no longer holds for these dtypes."]
fn w7_f64_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::F64);
    let (dag, inputs, output) = build_matmul_helper(Prim::F64);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => {
            panic!("F64 matmul helper MUST be rejected (W6 cross-product); got Ok({summary:?})")
        }
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "F64 matmul helper MUST emit BlasOutputPrecisionMismatch (W6 closes \
             the W5 silent-rejection loop); got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::F64);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: F16
// =========================================================================

#[test]
#[ignore = "WS-A2/A3 lift: F64/Bf16/F16 BLAS matmul are now admitted (cblas_dgemm / hipblasGemmEx); the W5 P0 fail-closed assumption no longer holds for these dtypes."]
fn w7_f16_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::F16);
    let (dag, inputs, output) = build_matmul_helper(Prim::F16);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => {
            panic!("F16 matmul helper MUST be rejected (W6 cross-product); got Ok({summary:?})")
        }
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "F16 matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::F16);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: Bf16
// =========================================================================

#[test]
#[ignore = "WS-A2/A3 lift: F64/Bf16/F16 BLAS matmul are now admitted (cblas_dgemm / hipblasGemmEx); the W5 P0 fail-closed assumption no longer holds for these dtypes."]
fn w7_bf16_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::Bf16);
    let (dag, inputs, output) = build_matmul_helper(Prim::Bf16);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => {
            panic!("Bf16 matmul helper MUST be rejected (W6 cross-product); got Ok({summary:?})")
        }
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "Bf16 matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::Bf16);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: F8e4m3
// =========================================================================

#[test]
fn w7_f8e4m3_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::F8e4m3);
    let (dag, inputs, output) = build_matmul_helper(Prim::F8e4m3);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => panic!("F8e4m3 matmul helper MUST be rejected; got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "F8e4m3 matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::F8e4m3);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: Int8
// =========================================================================

#[test]
fn w7_int8_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::Int8);
    let (dag, inputs, output) = build_matmul_helper(Prim::Int8);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => panic!("Int8 matmul helper MUST be rejected; got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "Int8 matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::Int8);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: Int32 (covered by W6's
// `blas_output_precision_mismatch_int32_emits_structured_rejection`;
// re-verifying here under the W7 adversarial pass)
// =========================================================================

#[test]
fn w7_int32_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::Int32);
    let (dag, inputs, output) = build_matmul_helper(Prim::Int32);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => panic!("Int32 matmul helper MUST be rejected; got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "Int32 matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::Int32);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: Int64
// =========================================================================

#[test]
fn w7_int64_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::Int64);
    let (dag, inputs, output) = build_matmul_helper(Prim::Int64);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => panic!("Int64 matmul helper MUST be rejected; got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "Int64 matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::Int64);
        }
    }
}

// =========================================================================
// W5 P0 cross-product: Bool
//
// Bool matmul is exotic but the W5 P0 filter still rejects it (any
// precision != F32). The W6 recognizer must surface
// BlasOutputPrecisionMismatch for it, not silently skip.
// =========================================================================

#[test]
fn w7_bool_matmul_helper_specializer_locked_and_w6_rejection_fires() {
    assert_specializer_keeps_off_blas_path(Prim::Bool);
    let (dag, inputs, output) = build_matmul_helper(Prim::Bool);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => panic!("Bool matmul helper MUST be rejected; got Ok({summary:?})"),
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "Bool matmul helper MUST emit BlasOutputPrecisionMismatch; \
             got NotEligible -- silent skip regressed"
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "expected BlasOutputPrecisionMismatch detail, got {:?}",
                    r.detail
                );
            };
            assert_eq!(observed, Prim::Bool);
        }
    }
}

// =========================================================================
// NO-SILENT-REJECTION INVARIANT
//
// The single test that closes the W5 P0 → W6 diagnostic loop: every
// precision the W5 filter rejects MUST produce a structured rejection,
// never `Ok(_)` or `NotEligible`. This is the trap test the plan
// requires: built so that, under "silent rejection" (the pre-W6
// behavior), it would have passed with an `Ok(_)` / `NotEligible`
// vote and now correctly fails.
//
// If the W6 BLAS recognizer is regressed to silently skip ANY of the
// eight non-F32 precisions, this test names the precision explicitly
// in its panic message.
// =========================================================================

#[test]
#[ignore = "WS-A2/A3 lift: F64/Bf16/F16 BLAS matmul are now admitted (cblas_dgemm / hipblasGemmEx); the W5 P0 fail-closed assumption no longer holds for these dtypes."]
fn nonsilent_rejection_invariant_for_every_w5_rejected_precision() {
    // Every non-F32 precision in the Prim enum that the W5 P0 filter
    // (`detect_matmul_pattern` precision gate at
    // `crates/chelis-ir/src/specialize.rs:519`) rejects. F32 is
    // explicitly excluded (it goes through the accepted BlasMatmul
    // path, not a rejection arm). String is excluded — tensor
    // precision can't be String at the HostTensorInput level.
    let w5_rejected_precisions = [
        Prim::F64,
        Prim::F16,
        Prim::Bf16,
        Prim::F8e4m3,
        Prim::Int8,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
    ];

    // Bookkeeping: track which precisions produced which arm. A silent
    // skip (NotEligible) or accidental acceptance (Ok) is named
    // explicitly so the failure message identifies the regressed
    // precision instead of a generic "test failed."
    let mut silent_skips: Vec<Prim> = Vec::new();
    let mut accidental_acceptances: Vec<Prim> = Vec::new();
    let mut mismatched_observed: Vec<(Prim, Prim)> = Vec::new();

    for prim in &w5_rejected_precisions {
        let (dag, inputs, output) = build_matmul_helper(*prim);
        match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
            Ok(_) => accidental_acceptances.push(*prim),
            Err(BlasSummaryAttempt::NotEligible) => silent_skips.push(*prim),
            Err(BlasSummaryAttempt::Rejected(r)) => {
                // Class must be exactly BlasOutputPrecisionMismatch
                // (the recognizer reaches the output gate FIRST in
                // the rejection ordering — before the BlasMatmul-shape
                // checks — for matmul-near non-F32 helpers).
                assert_eq!(
                    r.rejection_class,
                    SummaryRejectionClass::BlasOutputPrecisionMismatch,
                    "precision {prim:?} produced unexpected rejection class \
                     {:?}; expected BlasOutputPrecisionMismatch (the W5 P0 \
                     filter rejects on output precision FIRST)",
                    r.rejection_class,
                );
                let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail
                else {
                    panic!(
                        "precision {prim:?} rejection class was correct but \
                         detail was not BlasOutputPrecisionMismatch; got {:?}",
                        r.detail,
                    );
                };
                if observed != *prim {
                    mismatched_observed.push((*prim, observed));
                }
            }
        }
    }

    // Fail explicitly on each violation class so any silent
    // regression names the offending precision in the panic.
    let mut violations: Vec<String> = Vec::new();
    if !silent_skips.is_empty() {
        violations.push(format!(
            "SILENT-REJECTION REGRESSED for precisions {silent_skips:?}: \
             W6 BLAS recognizer returned NotEligible (silent skip) for a \
             non-F32 matmul helper. This is EXACTLY the bug the W5 P0 \
             follow-up was meant to close; ship-blocker."
        ));
    }
    if !accidental_acceptances.is_empty() {
        violations.push(format!(
            "ACCIDENTAL ACCEPTANCE for precisions {accidental_acceptances:?}: \
             W6 BLAS recognizer returned Ok(_) for a non-F32 matmul helper. \
             The W5 P0 fix at specialize.rs:519 prevented `RiscOp::BlasMatmul` \
             replacement, but the helper recognizer accepted anyway -- \
             ship-blocker miscompile risk."
        ));
    }
    if !mismatched_observed.is_empty() {
        violations.push(format!(
            "DETAIL-FIELD MISMATCH for {mismatched_observed:?}: \
             the recognizer rejected but `observed` in the detail did not \
             match the helper's actual output precision. \
             (precision, observed) pairs listed."
        ));
    }

    assert!(
        violations.is_empty(),
        "no-silent-rejection invariant FAILED:\n  - {}",
        violations.join("\n  - "),
    );
}

// =========================================================================
// Trap test: would have passed under "silent rejection" (pre-W6).
//
// The plan requires "at least one adversarial test that would have
// passed under 'silent rejection' (e.g., empty
// HostProgram::summary_rejections for an F64 helper) and verify it
// now fails." Implemented at the recognizer level: under the pre-W6
// silent path, the BLAS recognizer returned `Option<...>::None` (no
// diagnostic at all). The W7 contract is that EVERY non-F32 matmul
// helper now returns `Err(Rejected(_))` — never the "silent" arm.
//
// The trap below is constructed so it would fail under silent
// rejection: it directly asserts the structured Rejection arm fires
// for an F64 helper. The pre-W6 path returned None, which under our
// `BlasSummaryAttempt::NotEligible` mapping would be exactly the
// arm that "didn't fire the diagnostic" — i.e. exactly the silent
// behavior. We assert against that.
// =========================================================================

#[test]
#[ignore = "WS-A2/A3 lift: F64/Bf16/F16 BLAS matmul are now admitted (cblas_dgemm / hipblasGemmEx); the W5 P0 fail-closed assumption no longer holds for these dtypes."]
fn trap_f64_helper_would_have_silently_passed_under_pre_w6() {
    // Pre-W6 behavior (silent): `summarize_blas_helper_from_parts`
    // returned `Option::None` for an F64 matmul helper. There was no
    // `BlasSummaryAttempt` enum; the outer pass treated `None` as
    // "no helper to summarize, no rejection to report."
    //
    // Post-W6: the recognizer returns
    // `Err(BlasSummaryAttempt::Rejected(...))` with
    // `BlasOutputPrecisionMismatch`. The trap asserts the post-W6 arm
    // fires. If the recognizer regresses to silent skip (`NotEligible`
    // for F64), this test fails with an explicit "silent regression"
    // panic — i.e. the post-W6 contract requires NotEligible to be
    // reserved for non-matmul-near helpers only.
    let (dag, inputs, output) = build_matmul_helper(Prim::F64);
    match try_summarize_blas_helper_for_test(&dag, &inputs, &output) {
        Ok(summary) => panic!(
            "F64 matmul helper must NOT summarize successfully (W5 P0 \
             precision filter forbids non-F32 BlasMatmul codegen). \
             Got Ok({summary:?})"
        ),
        Err(BlasSummaryAttempt::NotEligible) => panic!(
            "SILENT REJECTION REGRESSED: F64 matmul helper produced \
             BlasSummaryAttempt::NotEligible, which under the outer \
             summary-derivation pass produces NO diagnostic -- exactly \
             the pre-W6 bug. NotEligible is reserved for non-matmul-near \
             helpers (elementwise Add, pure-sparse, etc.), NOT for non-F32 \
             matmul helpers."
        ),
        Err(BlasSummaryAttempt::Rejected(r)) => {
            // Post-W6 contract: a non-F32 matmul helper produces a
            // structured rejection. Pattern-match the class + detail.
            assert_eq!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch,
                "post-W6 contract requires BlasOutputPrecisionMismatch for \
                 a non-F32 matmul helper; got {:?}",
                r.rejection_class,
            );
            let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = r.detail else {
                panic!(
                    "post-W6 contract requires BlasOutputPrecisionMismatch detail; \
                     got {:?}",
                    r.detail,
                );
            };
            assert_eq!(observed, Prim::F64);
        }
    }
}

// =========================================================================
// F32 control: the W5 P0 filter PASSES F32, and the W6 recognizer
// accepts F32 (no rejection). This pair of behaviors is the OTHER
// side of the cross-product — locks that the W6 recognizer doesn't
// over-report.
// =========================================================================

#[test]
fn w7_f32_control_specializer_accepts_and_w6_recognizer_summarizes() {
    // (1) Specializer DOES replace the F32 matmul with `RiscOp::BlasMatmul`.
    let (dag, inputs, output) = build_matmul_helper(Prim::F32);
    let specialized = specialize_for_blas(&dag);
    let any_blas_matmul = specialized
        .roots()
        .iter()
        .filter_map(|id| specialized.get(*id))
        .any(|n| matches!(&n.op, RiscOp::BlasMatmul { .. }));
    assert!(
        any_blas_matmul,
        "F32 matmul helper MUST be replaced with `RiscOp::BlasMatmul` (the \
         W5 P0 fix only rejects NON-F32 matmul). If specializer left F32 \
         on the generic path, the W5 P0 fix over-rejected."
    );
    // (2) W6 recognizer accepts and produces a summary.
    let attempt = try_summarize_blas_helper_for_test(&dag, &inputs, &output);
    let summary = match attempt {
        Ok(s) => s,
        Err(BlasSummaryAttempt::NotEligible) => {
            panic!("F32 matmul must be summarized, got NotEligible")
        }
        Err(BlasSummaryAttempt::Rejected(r)) => panic!(
            "F32 matmul helper MUST be accepted, got Rejected({:?} / {:?})",
            r.rejection_class, r.detail,
        ),
    };
    assert_eq!(summary.output.precision, Prim::F32);
    assert_eq!(summary.input_tys[0].precision, Prim::F32);
    assert_eq!(summary.input_tys[1].precision, Prim::F32);
}
