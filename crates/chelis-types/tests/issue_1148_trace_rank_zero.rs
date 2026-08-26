//! Issue #1148: tracing both axes of a matrix returns a rank-zero tensor.
//!
//! `spec/04-type-system.md` section 4.3 requires `trace` to remove both
//! selected axes while preserving the tensor result carrier. A rank-two
//! input therefore produces `tensor[f32]`, not scalar `f32`.
//! This suite locks that f32 inference contract only; it is not evidence for
//! the full per-dtype runtime contract in [05-OP-33].

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

const SCALAR_SIGNATURE_MISMATCH: &str = "def 'f' body doesn't match declared signature: \
body has type `(tensor[4, 4, f32]) -> tensor[, f32]`, declared type is \
`(tensor[4, 4, f32]) -> f32`";

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

#[test]
fn matrix_trace_typechecks_as_rank_zero_tensor() {
    let deep = surf_to_deep("def f(m: tensor[4, 4, f32]) -> tensor[f32] = trace(m, 0, 1)\n");

    if let Err(report) = check_ir_program(&deep) {
        panic!(
            "matrix trace must type-check as rank-zero tensor[f32], got {:?}",
            report.errors
        );
    }
}

#[test]
fn matrix_trace_rejects_scalar_return_signature() {
    let deep = surf_to_deep("def f(m: tensor[4, 4, f32]) -> f32 = trace(m, 0, 1)\n");

    let report = check_ir_program(&deep)
        .expect_err("matrix trace must not satisfy a scalar f32 return signature");
    let [error] = report.errors.as_slice() else {
        panic!(
            "expected exactly one declared-signature mismatch, got {:?}",
            report.errors
        );
    };
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert_eq!(
        error.message, SCALAR_SIGNATURE_MISMATCH,
        "the negative must prove rank-zero tensor[, f32] body versus scalar f32 declaration"
    );
}

#[test]
fn wrong_trace_input_type_mismatch_is_not_signature_evidence() {
    let deep = surf_to_deep("def f(m: tensor[4, 4, f32]) -> f32 = trace(cast(1.0, f32), 0, 1)\n");

    let report = check_ir_program(&deep).expect_err("trace must reject a scalar input");
    let [error] = report.errors.as_slice() else {
        panic!(
            "expected exactly one wrong-input TypeMismatch, got {:?}",
            report.errors
        );
    };
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert_eq!(error.message, "trace expects tensor input, got f32");
    assert_ne!(
        error.message, SCALAR_SIGNATURE_MISMATCH,
        "a generic TypeMismatch must not stand in for the declared-signature contract"
    );
}
