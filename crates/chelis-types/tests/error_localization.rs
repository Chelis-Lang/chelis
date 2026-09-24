//! Error localization tests: verify that check diagnostics carry
//! `span_offset` so downstream tools can pinpoint WHERE an error occurs.

use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

/// Check a span-preserving core AST fixture, returning the full
/// CheckError structs for inspection.
fn check_errors(source: &str) -> Vec<CheckError> {
    let exprs: Vec<chelis_deep::Expr> = serde_json::from_str(source).expect("core AST fixture");
    match check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked) {
        Ok(_) => Vec::new(),
        Err(rep) => rep.errors,
    }
}

fn check_errors_at_both_ingresses(source: &str) -> Vec<(&'static str, Vec<CheckError>)> {
    let exprs: Vec<chelis_deep::Expr> = serde_json::from_str(source).expect("core AST fixture");
    [
        (
            "normalized IR",
            check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
        ),
        (
            "typed Deep",
            check_typed_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
        ),
    ]
    .into_iter()
    .map(|(ingress, result)| {
        (
            ingress,
            match result {
                Ok(_) => Vec::new(),
                Err(report) => report.errors,
            },
        )
    })
    .collect()
}

fn assert_one_located_ascription_mismatch(source: &str) {
    for (ingress, errors) in check_errors_at_both_ingresses(source) {
        let mismatches = errors
            .iter()
            .filter(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch))
            .collect::<Vec<_>>();
        assert_eq!(
            mismatches.len(),
            1,
            "{ingress} must report one ascription mismatch: {errors:?}"
        );
        let error = mismatches[0];
        assert!(
            error.span_offset.is_some(),
            "{ingress} ascription mismatch must carry span_offset: {error:?}"
        );
        assert!(
            error.span_id.is_some(),
            "{ingress} ascription mismatch must carry span_id: {error:?}"
        );
    }
}

// ── Precision mismatch carries span_offset ──────────────────────────

#[test]
fn precision_mismatch_has_span_offset() {
    // add(f32, i32) is a precision mismatch; the span_offset should be
    // present and non-zero because the call site is not at byte 0.
    let source = include_str!(
        "fixtures/core/tests/error_localization/precision_mismatch_has_span_offset.json"
    );
    let errors = check_errors(source);
    let prec_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e.kind, CheckErrorKind::PrecisionMismatch))
        .collect();
    assert!(
        !prec_errors.is_empty(),
        "expected at least one PrecisionMismatch error; got: {errors:?}"
    );
    let e = &prec_errors[0];
    assert!(
        e.span_offset.is_some(),
        "PrecisionMismatch error should carry span_offset; got None. error: {e:?}"
    );
    assert!(
        e.span_offset.unwrap() > 0,
        "span_offset should be non-zero for a call not at byte 0; got: {}",
        e.span_offset.unwrap()
    );
    // The span_id should survive the core AST ingress.
    assert!(
        e.span_id.is_some(),
        "PrecisionMismatch error should carry span_id; got None. error: {e:?}"
    );
}

// ── Arity mismatch carries span_offset ──────────────────────────────

#[test]
fn arity_mismatch_has_span_offset() {
    // Calling a unary function with two arguments.
    let source =
        include_str!("fixtures/core/tests/error_localization/arity_mismatch_has_span_offset.json");
    let errors = check_errors(source);
    let arity_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e.kind, CheckErrorKind::ArityMismatch))
        .collect();
    assert!(
        !arity_errors.is_empty(),
        "expected at least one ArityMismatch error; got: {errors:?}"
    );
    let e = &arity_errors[0];
    assert!(
        e.span_offset.is_some(),
        "ArityMismatch error should carry span_offset; got None. error: {e:?}"
    );
    assert!(
        e.span_offset.unwrap() > 0,
        "span_offset should be non-zero; got: {}",
        e.span_offset.unwrap()
    );
    assert!(
        e.span_id.is_some(),
        "ArityMismatch error should carry span_id; got None. error: {e:?}"
    );
}

// ── Type mismatch in if-branches carries span_offset ────────────────

#[test]
fn if_branch_mismatch_has_span_offset() {
    let source = include_str!(
        "fixtures/core/tests/error_localization/if_branch_mismatch_has_span_offset.json"
    );
    let errors = check_errors(source);
    let type_errors: Vec<_> = errors
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::PrecisionMismatch
            )
        })
        .collect();
    assert!(
        !type_errors.is_empty(),
        "expected type/precision mismatch in if-branches; got: {errors:?}"
    );
    let e = &type_errors[0];
    assert!(
        e.span_offset.is_some(),
        "if-branch mismatch should carry span_offset; got None. error: {e:?}"
    );
    assert!(
        e.span_offset.unwrap() > 0,
        "span_offset should be non-zero; got: {}",
        e.span_offset.unwrap()
    );
    assert!(
        e.span_id.is_some(),
        "if-branch mismatch should carry span_id; got None. error: {e:?}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// RED TEAM: adversarial error localization tests
// ══════════════════════════════════════════════════════════════════════

// ── Clean program produces no span_offset/span_id (backward compat) ─

#[test]
fn clean_program_has_no_span_fields() {
    let source = include_str!(
        "fixtures/core/tests/error_localization/clean_program_has_no_span_fields.json"
    );
    let errors = check_errors(source);
    assert!(
        errors.is_empty(),
        "clean program should produce no errors; got: {errors:?}"
    );
}

// ── Multi-error programs: all errors get span_offset ────────────────

#[test]
fn multi_error_program_all_get_span_offset() {
    // Two distinct type errors in separate function bodies.
    let source = include_str!(
        "fixtures/core/tests/error_localization/multi_error_program_all_get_span_offset.json"
    );
    let errors = check_errors(source);
    assert!(
        errors.len() >= 2,
        "expected at least 2 errors; got {} errors: {errors:?}",
        errors.len()
    );
    for (i, e) in errors.iter().enumerate() {
        assert!(
            e.span_offset.is_some(),
            "error #{i} ({:?}) should have span_offset; got None. error: {e:?}",
            e.kind
        );
        assert!(
            e.span_offset.unwrap() > 0,
            "error #{i} span_offset should be non-zero; got 0"
        );
    }
}

// ── DimensionMismatch gets a span ───────────────────────────────────

#[test]
fn dimension_mismatch_has_span_offset() {
    // Adding tensors of different shapes is a dimension mismatch.
    let source = include_str!(
        "fixtures/core/tests/error_localization/dimension_mismatch_has_span_offset.json"
    );
    let errors = check_errors(source);
    let dim_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch))
        .collect();
    assert!(
        !dim_errors.is_empty(),
        "expected at least one DimensionMismatch error; got: {errors:?}"
    );
    let e = &dim_errors[0];
    assert!(
        e.span_offset.is_some(),
        "DimensionMismatch error should carry span_offset; got None. error: {e:?}"
    );
    assert!(
        e.span_offset.unwrap() > 0,
        "span_offset should be non-zero; got: {}",
        e.span_offset.unwrap()
    );
}

// ── Deeply nested error reports byte offset in the source ───────────

#[test]
fn deeply_nested_error_has_nonzero_span() {
    // The error is inside a block with multiple bindings.
    let source = include_str!(
        "fixtures/core/tests/error_localization/deeply_nested_error_has_nonzero_span.json"
    );
    let errors = check_errors(source);
    assert!(!errors.is_empty(), "expected at least one error; got none");
    let e = &errors[0];
    assert!(
        e.span_offset.is_some(),
        "nested error should carry span_offset; got None. error: {e:?}"
    );
    // The offset should be > 0 because the error is deep in the source.
    assert!(
        e.span_offset.unwrap() > 0,
        "nested span_offset should be non-zero; got: {}",
        e.span_offset.unwrap()
    );
    // It should be a reasonable byte offset (pointing into the source, not some
    // absurdly large number). The captured source was 76 bytes long.
    assert!(
        e.span_offset.unwrap() < 76,
        "span_offset {} exceeds source length {}; likely not a byte offset",
        e.span_offset.unwrap(),
        76
    );
}

// ── span_offset values are distinct for different error sites ───────

#[test]
fn different_error_sites_have_distinct_offsets() {
    let source = include_str!(
        "fixtures/core/tests/error_localization/different_error_sites_have_distinct_offsets.json"
    );
    let errors = check_errors(source);
    assert!(
        errors.len() >= 2,
        "expected at least 2 errors; got {} errors: {errors:?}",
        errors.len()
    );
    let offsets: Vec<usize> = errors.iter().filter_map(|e| e.span_offset).collect();
    assert!(
        offsets.len() >= 2,
        "expected at least 2 errors with span_offset; got offsets: {offsets:?}"
    );
    // The two errors are at different source positions, so offsets should differ.
    assert_ne!(
        offsets[0], offsets[1],
        "two errors at different source locations should have different span_offsets: {:?}",
        offsets
    );
}

// ── UnboundVariable error carries span_offset ───────────────────────

#[test]
fn unbound_variable_has_span_offset() {
    let source = include_str!(
        "fixtures/core/tests/error_localization/unbound_variable_has_span_offset.json"
    );
    let errors = check_errors(source);
    let unbound_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e.kind, CheckErrorKind::UnboundVariable { .. }))
        .collect();
    assert!(
        !unbound_errors.is_empty(),
        "expected at least one UnboundVariable error; got: {errors:?}"
    );
    let e = &unbound_errors[0];
    assert!(
        e.span_offset.is_some(),
        "UnboundVariable error should carry span_offset; got None. error: {e:?}"
    );
    assert!(
        e.span_offset.unwrap() > 0,
        "span_offset should be non-zero; got: {}",
        e.span_offset.unwrap()
    );
}

#[test]
fn expression_ascription_mismatch_has_source_location_at_both_ingresses() {
    assert_one_located_ascription_mismatch(include_str!(
        "fixtures/core/tests/error_localization/expression_ascription_mismatch_has_source_location_at_both_ingresses.json"
    ));
}

#[test]
fn let_ascription_mismatch_has_source_location_at_both_ingresses() {
    assert_one_located_ascription_mismatch(include_str!(
        "fixtures/core/tests/error_localization/let_ascription_mismatch_has_source_location_at_both_ingresses.json"
    ));
}
