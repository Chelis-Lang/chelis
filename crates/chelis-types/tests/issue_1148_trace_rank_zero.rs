//! Issue #1148: tracing both axes of a matrix returns a rank-zero tensor.
//!
//! `spec/04-type-system.md` section 4.3 requires `trace` to remove both
//! selected axes while preserving the tensor result carrier. A rank-two
//! input therefore produces `tensor[f32]`, not scalar `f32`.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

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
    assert!(
        report
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "expected TypeMismatch for tensor[f32] versus f32, got {:?}",
        report.errors
    );
    assert!(
        report
            .errors
            .iter()
            .all(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "valid trace axes must not introduce unrelated diagnostics, got {:?}",
        report.errors
    );
}
