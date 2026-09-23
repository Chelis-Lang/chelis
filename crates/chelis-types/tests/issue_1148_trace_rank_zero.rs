//! Issue #1148: tracing both axes of a matrix returns a rank-zero tensor.
//!
//! `spec/04-type-system.md` section 4.3 requires `trace` to remove both
//! selected axes while preserving the tensor result carrier. A rank-two
//! input therefore produces `tensor[f32]`, not scalar `f32`.
//! This suite locks that f32 inference contract only; it is not evidence for
//! the full per-dtype runtime contract in [05-OP-33].

use chelis_deep::Expr;

use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

fn parse_core(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("valid core fixture")
}

#[test]
fn matrix_trace_typechecks_as_rank_zero_tensor() {
    let deep = parse_core(include_str!(
        "fixtures/core/tests/issue_1148_trace_rank_zero/matrix_trace_typechecks_as_rank_zero_tensor.0.deep"
    ));

    if let Err(report) = check_ir_program(&deep) {
        panic!(
            "matrix trace must type-check as rank-zero tensor[f32], got {:?}",
            report.errors
        );
    }
}

#[test]
fn matrix_trace_rejects_scalar_return_signature() {
    let deep = parse_core(include_str!(
        "fixtures/core/tests/issue_1148_trace_rank_zero/matrix_trace_rejects_scalar_return_signature.0.deep"
    ));

    let report = check_ir_program(&deep)
        .expect_err("matrix trace must not satisfy a scalar f32 return signature");
    let [error] = report.errors.as_slice() else {
        panic!(
            "expected exactly one declared-signature mismatch, got {:?}",
            report.errors
        );
    };
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
}

#[test]
fn wrong_trace_input_type_mismatch_is_not_signature_evidence() {
    let deep = parse_core(include_str!(
        "fixtures/core/tests/issue_1148_trace_rank_zero/wrong_trace_input_type_mismatch_is_not_signature_evidence.0.deep"
    ));

    let report = check_ir_program(&deep).expect_err("trace must reject a scalar input");
    let [error] = report.errors.as_slice() else {
        panic!(
            "expected exactly one wrong-input TypeMismatch, got {:?}",
            report.errors
        );
    };
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
}
