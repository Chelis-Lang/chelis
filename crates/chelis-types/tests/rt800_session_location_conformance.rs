//! Retained adversarial conformance probes from PR #800's final review.
//! The intentional `infer_var` mutation is represented by `errors::report`'s
//! compile-fail oracle instead of being planted in production source.

mod support;

use chelis_deep::parser::parse_str as parse_deep_lenient;
use chelis_types::check_ir_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn deep_errors(source: &str) -> Vec<CheckError> {
    let exprs = parse_deep_lenient(source).expect("Deep fixture must parse");
    check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("adversarial Deep fixture must be rejected")
        .errors
}

fn fixture_errors(source: &str) -> Vec<CheckError> {
    let exprs: Vec<chelis_deep::Expr> = serde_json::from_str(source).expect("core AST fixture");
    check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("adversarial core fixture must be rejected")
        .errors
}

#[test]
fn einsum_with_only_equation_and_one_operand_reports_one_arity_root() {
    let errors = deep_errors(
        r#"(def {} bad
              (app {}
                (var {} einsum)
                (lit {type: (t-prim {} string)} "i->i")
                (lit {type: (t-prim {} f32)} 1.0)))"#,
    );
    assert_eq!(
        errors.len(),
        1,
        "einsum arity must have one root: {errors:?}"
    );
    assert!(matches!(errors[0].kind, CheckErrorKind::ArityMismatch));
    assert!(
        errors[0].message.contains("expected 3 args, got 2"),
        "{errors:?}"
    );
}

#[test]
fn cast_operand_and_target_failures_remain_two_independent_roots() {
    let exprs = support::parse_unchecked_legacy(
        "(def {} bad (cast {} (var {} missing_value) (t-prim {} f32 extra)))",
    );
    let errors = check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("adversarial Deep fixture must be rejected")
        .errors;
    assert_eq!(
        errors.len(),
        2,
        "the bad operand and malformed target are independent roots: {errors:?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("missing_value")),
        "{errors:?}"
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::MalformedForm) && error.message.contains("t-prim")
        }),
        "{errors:?}"
    );
}

#[test]
fn tuple_projection_of_failed_cast_keeps_one_root_without_a_cascade() {
    let errors = deep_errors(
        r#"(def {} bad
              (tuple-get {}
                (tuple {}
                  (cast {}
                    (lit {type: (t-prim {} f32)} 1.0)
                    (t-prim {} Missing))
                  (lit {type: (t-prim {} i32)} 2))
                0))"#,
    );
    assert_eq!(
        errors.len(),
        1,
        "nested cast -> tuple -> projection must preserve one root: {errors:?}"
    );
    assert!(errors[0].message.contains("Missing"), "{errors:?}");
}

#[test]
fn unknown_alias_target_has_a_complete_source_location() {
    let errors = fixture_errors(include_str!(
        "fixtures/core/tests/rt800_session_location_conformance/case_001.json"
    ));
    assert_eq!(
        errors.len(),
        1,
        "invalid alias header must report once without downstream cascade: {errors:?}"
    );
    let error = &errors[0];
    assert!(error.message.contains("Missing"), "{errors:?}");
    assert!(
        error.span_offset.is_some() && error.span_id.is_some(),
        "the retained core AST preserves both byte offset and synthesized span id: {error:?}"
    );
}
