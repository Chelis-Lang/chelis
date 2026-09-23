//! Let ascriptions constrain generic RHS inference and shared dimensions.

fn fixture(name: &str) -> Vec<chelis_deep::Expr> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/core/tests/issue_143b_let_ascription_no_propagation")
        .join(format!("{name}.deep"));
    let source = std::fs::read_to_string(path).expect("core fixture");
    chelis_deep::parser::parse_str(&source).expect("valid Deep fixture")
}

use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn typecheck_fixture(name: &str) -> Vec<CheckError> {
    let deep = fixture(name);
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(prog_errors) => prog_errors.errors,
    }
}

fn has_dimension_mismatch(errors: &[CheckError]) -> bool {
    errors
        .iter()
        .any(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch))
}

fn errors_summary(errors: &[CheckError]) -> String {
    if errors.is_empty() {
        "(no errors)".to_string()
    } else {
        errors
            .iter()
            .map(|e| format!("[{:?}] {}", e.kind, e.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[test]
fn let_binding_ascription_propagates_to_to_tensor_rhs() {
    // Desired post-fix behavior: the ascription `a: tensor[3, f32]`
    // should unify with the RHS `to_tensor([1.0, 2.0, 3.0])`, binding
    // `a` to `tensor[Lit(3), f32]`. The follow-on `pair_id(&a, &b)`
    // then unifies sig's `n` against `Lit(3)`, then `Lit(3)` vs
    // `Lit(5)` from `b` → DimensionMismatch.
    let errors = typecheck_fixture("case_01");
    assert!(
        has_dimension_mismatch(&errors),
        "expected DimensionMismatch - ascriptions should propagate to to_tensor RHS, \
         allowing sig `n` to bind to 3 from `a` and conflict with 5 from `b`; \
         got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn parameter_ascription_path_works_today() {
    // Counter-probe: the parameter-ascription path is fully functional.
    // This rules out "the type system can't handle ascription at all"
    // and isolates the bug above to the let-binding path specifically.
    let errors = typecheck_fixture("case_02");
    assert!(
        has_dimension_mismatch(&errors),
        "parameter-ascription path should trip DimensionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_binding_ascription_with_matched_shapes_does_not_spuriously_fail() {
    // Sanity: even with the bug present, a let-binding ascription
    // whose RHS dims would match the sig must not produce a spurious
    // DimensionMismatch. Pin this so a future fix doesn't over-correct
    // and introduce false positives for the matched-shape case.
    let errors = typecheck_fixture("case_03");
    assert!(
        !has_dimension_mismatch(&errors),
        "matched-shape let-binding ascriptions must not spuriously trip \
         DimensionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}

// Direct classification checks for incompatible ascriptions.
//
// The three tests above validate the patch indirectly via downstream
// shared-sig-dim unification. The tests below exercise the patch's
// most direct property: a let-binding whose declared type literally
// disagrees with the inferred RHS type produces a clean diagnostic
// with the right `CheckErrorKind`.

fn errors_contain_kind(errors: &[CheckError], kind: &CheckErrorKind) -> bool {
    errors
        .iter()
        .any(|e| std::mem::discriminant(&e.kind) == std::mem::discriminant(kind))
}

#[test]
fn let_ascription_dim_mismatch_reports_dimension_mismatch() {
    // Direct dim mismatch: declared `tensor[3, f32]`, RHS has concrete
    // dim 2 (via a parameter ascription). Must report
    // DimensionMismatch.
    let errors = typecheck_fixture("case_04");
    assert!(
        errors_contain_kind(&errors, &CheckErrorKind::DimensionMismatch,),
        "expected DimensionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_ascription_precision_mismatch_reports_precision_mismatch() {
    // Direct precision mismatch: declared `f64`, RHS is an explicitly
    // f32-cast scalar. Must report PrecisionMismatch.
    let errors = typecheck_fixture("case_05");
    assert!(
        errors_contain_kind(&errors, &CheckErrorKind::PrecisionMismatch,),
        "expected PrecisionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_ascription_type_mismatch_reports_type_mismatch() {
    // Direct top-level type mismatch: declared `tensor[3, f32]`, RHS
    // is a scalar `f32`. Tensor versus primitive is TypeMismatch,
    // not DimensionMismatch or PrecisionMismatch.
    let errors = typecheck_fixture("case_06");
    assert!(
        errors_contain_kind(&errors, &CheckErrorKind::TypeMismatch,),
        "expected TypeMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}
