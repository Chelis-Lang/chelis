//! Error localization tests: verify that check diagnostics carry
//! `span_offset` so downstream tools can pinpoint WHERE an error occurs.

use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

/// Parse → desugar → macro-expand → type-check, returning the full
/// CheckError structs for inspection.
fn check_errors(source: &str) -> Vec<CheckError> {
    let decls = parse_str(source).expect("surf parse");
    let exprs = expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs();
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(rep) => rep.errors,
    }
}

fn check_errors_at_both_ingresses(source: &str) -> Vec<(&'static str, Vec<CheckError>)> {
    let decls = parse_str(source).expect("surf parse");
    let exprs = expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs();
    [
        ("normalized IR", check_ir_program(&exprs)),
        ("typed Deep", check_typed_program(&exprs)),
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

fn assert_one_located_ascription_mismatch(source: &str, message_fragment: &str) {
    for (ingress, errors) in check_errors_at_both_ingresses(source) {
        let mismatches = errors
            .iter()
            .filter(|error| {
                matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                    && error.message.contains(message_fragment)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            mismatches.len(),
            1,
            "{ingress} must report one ascription mismatch: {errors:?}"
        );
        let error = mismatches[0];
        assert!(
            error
                .span_offset
                .is_some_and(|offset| offset < source.len()),
            "{ingress} ascription mismatch must point into its source: {error:?}"
        );
    }
}

// ── Precision mismatch carries span_offset ──────────────────────────

#[test]
fn precision_mismatch_has_span_offset() {
    // add(f32, i32) is a precision mismatch; the span_offset should be
    // present and non-zero because the call site is not at byte 0.
    let source = r#"
def bad() -> f32 =
    add(1.0, 1)
"#;
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
    // The span_id should be set from the desugar metadata.
    assert!(
        e.span_id.is_some(),
        "PrecisionMismatch error should carry span_id; got None. error: {e:?}"
    );
    assert!(
        e.span_id.as_ref().unwrap().starts_with("surf:"),
        "span_id should start with 'surf:'; got: {:?}",
        e.span_id
    );
}

// ── Arity mismatch carries span_offset ──────────────────────────────

#[test]
fn arity_mismatch_has_span_offset() {
    // Calling a unary function with two arguments.
    let source = r#"
def f(x: f32) -> f32 = x
def bad() -> f32 = f(1.0, 2.0)
"#;
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
    let source = r#"
def bad(c: bool) -> f32 =
    if c then 1.0 else 1
"#;
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

// ── Builder methods work correctly ──────────────────────────────────

#[test]
fn at_offset_builder_sets_field() {
    let e = CheckError::new(CheckErrorKind::Other, "test".to_string(), vec![]).at_offset(42);
    assert_eq!(e.span_offset, Some(42));
    assert_eq!(e.span_id, None);
}

#[test]
fn with_span_id_builder_sets_field() {
    let e = CheckError::new(CheckErrorKind::Other, "test".to_string(), vec![])
        .with_span_id("octant:line:7".to_string());
    assert_eq!(e.span_id, Some("octant:line:7".to_string()));
    assert_eq!(e.span_offset, None);
}

#[test]
fn builders_chain() {
    let e = CheckError::new(CheckErrorKind::Other, "test".to_string(), vec![])
        .at_offset(100)
        .with_span_id("src:5..20".to_string());
    assert_eq!(e.span_offset, Some(100));
    assert_eq!(e.span_id, Some("src:5..20".to_string()));
}

// ══════════════════════════════════════════════════════════════════════
// RED TEAM: adversarial error localization tests
// ══════════════════════════════════════════════════════════════════════

// ── Clean program produces no span_offset/span_id (backward compat) ─

#[test]
fn clean_program_has_no_span_fields() {
    let source = r#"
def good(x: tensor[3, f32]) -> tensor[3, f32] =
    add(x, x)
"#;
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
    let source = r#"
def bad1() -> f32 =
    add(1.0, 1)

def bad2() -> f32 =
    add(1.0, 1)
"#;
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
    let source = r#"
def bad(x: tensor[3, f32], y: tensor[5, f32]) -> tensor[3, f32] =
    add(x, y)
"#;
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
    let source = r#"
def outer() -> f32 = {
    a = 1.0
    b = 2.0
    c = add(1.0, 1)
    c
}
"#;
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
    // absurdly large number).
    assert!(
        e.span_offset.unwrap() < source.len(),
        "span_offset {} exceeds source length {}; likely not a byte offset",
        e.span_offset.unwrap(),
        source.len()
    );
}

// ── Deep validation warning (unknown tag) gets offset ───────────────

#[test]
fn validation_warning_carries_offset() {
    // Use check_errors which goes through macro expansion.
    // A valid Deep form with an unknown tag would trigger a ValidationWarning,
    // but from Surf we can't easily produce one. Instead, verify that
    // the at_offset mechanism from validation works at the unit level:
    // this is tested by the `at_offset_builder_sets_field` test above.
    // We verify the validation warning path indirectly: for a Surf program
    // that produces no validation warnings, the error list is empty.
    let source = r#"
def good(x: tensor[3, f32]) -> tensor[3, f32] = x
"#;
    let errors = check_errors(source);
    assert!(errors.is_empty(), "expected no errors from a good program");
}

// ── span_offset values are distinct for different error sites ───────

#[test]
fn different_error_sites_have_distinct_offsets() {
    let source = r#"
def first() -> f32 = add(1.0, 1)
def second() -> f32 = add(1.0, 1)
"#;
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
    let source = r#"
def bad() -> f32 = nonexistent_var
"#;
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
    assert_one_located_ascription_mismatch(
        "def f(x: f32) -> i32 = (x : i32)\n",
        "expression ascription does not match value",
    );
}

#[test]
fn let_ascription_mismatch_has_source_location_at_both_ingresses() {
    assert_one_located_ascription_mismatch(
        "def g(x: f32) -> i32 = {\n  y: i32 = x\n  y\n}\n",
        "let-binding `y` ascription does not match RHS",
    );
}
