//! Wave 7 fresh-context red-team — Surf-driven precision matrix for
//! the W5 P0 → W6 diagnosed-rejection closure.
//!
//! W6's `cross_library_semantic_gap_diagnostics::surface_f64_matmul_helper_emits_blas_output_precision_mismatch_rejection`
//! locks the F64 end-to-end Surf path. This file extends it to F64,
//! Int32, Int64 (precisions Surf accepts for matching-precision matmul
//! per `check_matmul_signature` at
//! `crates/chelis-types/src/infer.rs:7958`).
//!
//! F16, Bf16, F8e4m3, Int8 are NOT exercised through the Surf path
//! because some of these precisions can fail at earlier IR passes
//! (linearity / effect / etc.) for synthetic-shaped tests. The
//! IR-level cross-product in
//! `crates/chelis-ir/tests/blas_rejection_cross_product_adversarial.rs` covers
//! all eight precisions at the recognizer level; this file is the
//! Surf-driven complement for the precisions the front-end reliably
//! accepts.
//!
//! ## Pattern-match contract
//!
//! Pattern-match on `SummaryRejectionClass::BlasOutputPrecisionMismatch`
//! and `SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed }`.
//! No `contains()` on rendered text.

use chelis_ir::host::{
    SummaryRejection, SummaryRejectionClass, SummaryRejectionDetail,
    host_program_summary_rejections, try_lower_compiled_program,
};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::types::Prim;
use chelis_types::{check_ir_program, check_linearity};

/// Drive parse → desugar → typecheck → effect → linearity → host
/// lower on the source and return the resulting summary rejections.
fn rejections_for_source(source: &str) -> Vec<SummaryRejection> {
    let decls = parse_str(source).expect("parse_str");
    let deep = desugar_program(&decls);
    let checked = check_ir_program(&deep).expect("check_ir_program");
    let checked = chelis_effects::check_program(&checked).expect("effect check");
    let checked = check_linearity(&checked).expect("linearity check");
    let compiled = try_lower_compiled_program(&checked).expect("checked BLAS fixture must lower");
    let host = compiled
        .host
        .expect("host lowering must produce a HostProgram");
    host_program_summary_rejections(&host).to_vec()
}

/// Helper that asserts a precision-mismatch rejection exists in the
/// vector, matches both class and detail, and names the observed
/// precision exactly.
fn assert_blas_output_precision_mismatch(rejections: &[SummaryRejection], expected: Prim) {
    let matching: Vec<&SummaryRejection> = rejections
        .iter()
        .filter(|r| {
            matches!(
                r.rejection_class,
                SummaryRejectionClass::BlasOutputPrecisionMismatch
            )
        })
        .collect();
    assert!(
        !matching.is_empty(),
        "no BlasOutputPrecisionMismatch rejection found in {rejections:#?}; \
         expected at least one for the helper compiled at precision {expected:?}",
    );
    // At least one matching rejection must have the expected observed precision.
    let mut found_observed = false;
    for r in &matching {
        let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = &r.detail else {
            panic!(
                "rejection has BlasOutputPrecisionMismatch class but wrong detail: {:?}",
                r.detail
            );
        };
        if *observed == expected {
            found_observed = true;
        }
    }
    assert!(
        found_observed,
        "no rejection had observed precision {expected:?}; matching rejections:\n{matching:#?}",
    );
}

// =========================================================================
// Surf-driven cross-product: F64 (W6 covers; W7 re-verifies)
// =========================================================================

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted (cblas_dgemm); BlasOutputPrecisionMismatch no longer fires for F64 helpers."]
fn w7_surf_f64_matmul_helper_emits_blas_output_precision_mismatch() {
    let source = "def my_mm(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    assert_blas_output_precision_mismatch(&rejections, Prim::F64);
}

// =========================================================================
// Surf-driven cross-product: Int32
// =========================================================================

#[test]
#[ignore = "WS-A0 spec lock: integer matmul is now a type error per spec §5.7.2 (PrecisionMismatch at the type checker), so the helper never reaches the BLAS summary-rejection path this test exercises. The IR-level cross-product in blas_rejection_cross_product_adversarial.rs still covers integer matmul rejection at the recognizer level."]
fn w7_surf_int32_matmul_helper_emits_blas_output_precision_mismatch() {
    let source = "def my_mm(a: tensor[8, 16, i32], b: tensor[16, 4, i32]) \
                  -> tensor[8, 4, i32] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, i32], b: tensor[16, 4, i32]) \
                  -> tensor[8, 4, i32] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    assert_blas_output_precision_mismatch(&rejections, Prim::Int32);
}

// =========================================================================
// Surf-driven cross-product: Int64
// =========================================================================

#[test]
#[ignore = "WS-A0 spec lock: integer matmul is now a type error per spec §5.7.2 (PrecisionMismatch at the type checker), so the helper never reaches the BLAS summary-rejection path this test exercises."]
fn w7_surf_int64_matmul_helper_emits_blas_output_precision_mismatch() {
    let source = "def my_mm(a: tensor[8, 16, i64], b: tensor[16, 4, i64]) \
                  -> tensor[8, 4, i64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, i64], b: tensor[16, 4, i64]) \
                  -> tensor[8, 4, i64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    assert_blas_output_precision_mismatch(&rejections, Prim::Int64);
}

// =========================================================================
// No-silent-rejection invariant — Surf path.
//
// Trap test: under "silent rejection" (pre-W6), the F64 matmul
// helper produced `summary_rejections.is_empty() == true`. Now it
// must NOT be empty.
// =========================================================================

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted; helpers no longer accumulate summary rejections for F64."]
fn nonempty_summary_rejections_for_surf_f64_matmul_helper() {
    let source = "def my_mm(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    assert!(
        !rejections.is_empty(),
        "Surf F64 matmul helper MUST produce at least one summary rejection. \
         Under the pre-W6 silent rejection path, this was empty -- exactly \
         the bug W6 closed. If this test fails, the silent path has been \
         reintroduced."
    );
    // And the rejection must be the structured BLAS variant.
    let has_blas = rejections.iter().any(|r| {
        matches!(
            r.rejection_class,
            SummaryRejectionClass::BlasOutputPrecisionMismatch
        )
    });
    assert!(
        has_blas,
        "Surf F64 matmul helper MUST produce a BlasOutputPrecisionMismatch \
         rejection (not just any rejection). Got: {rejections:#?}"
    );
}

// =========================================================================
// No-silent-rejection invariant — Surf path, Int32 trap.
// =========================================================================

#[test]
#[ignore = "WS-A0 spec lock: integer matmul is a type error per spec §5.7.2; the helper never reaches host lowering, so summary rejections never accrue."]
fn nonempty_summary_rejections_for_surf_int32_matmul_helper() {
    let source = "def my_mm(a: tensor[8, 16, i32], b: tensor[16, 4, i32]) \
                  -> tensor[8, 4, i32] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, i32], b: tensor[16, 4, i32]) \
                  -> tensor[8, 4, i32] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    assert!(
        !rejections.is_empty(),
        "Surf Int32 matmul helper MUST produce at least one summary rejection; \
         pre-W6 returned empty (silent rejection)."
    );
    let has_blas = rejections.iter().any(|r| {
        matches!(
            r.rejection_class,
            SummaryRejectionClass::BlasOutputPrecisionMismatch
        )
    });
    assert!(
        has_blas,
        "Surf Int32 matmul helper MUST produce a BlasOutputPrecisionMismatch \
         rejection. Got: {rejections:#?}"
    );
}

// =========================================================================
// F32 control — Surf path. Must NOT emit any Blas* rejection.
// =========================================================================

#[test]
fn surf_f32_matmul_helper_emits_no_blas_rejection() {
    let source = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    for r in &rejections {
        assert!(
            !matches!(
                r.rejection_class,
                SummaryRejectionClass::BlasMultipleRoots
                    | SummaryRejectionClass::BlasOutputPrecisionMismatch
                    | SummaryRejectionClass::BlasNotMatmulPattern
                    | SummaryRejectionClass::BlasNonLoadOperand
                    | SummaryRejectionClass::BlasInputPrecisionMismatch
                    | SummaryRejectionClass::BlasDimensionBindingFailure
            ),
            "F32 matmul helper must NOT emit any Blas* rejection (it should \
             summarize successfully via BlasMatmul); got {r:?}",
        );
    }
}
