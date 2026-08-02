//! Retained adversarial conformance probes from PR #800's final review.
//! The intentional `infer_var` mutation is represented by `errors::report`'s
//! compile-fail oracle instead of being planted in production source.

mod support;

use chelis_deep::parser::parse_str as parse_deep_lenient;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;
use chelis_types::errors::{CheckError, CheckErrorKind};
use support::parse_unchecked_legacy;

fn deep_errors(source: &str) -> Vec<CheckError> {
    let exprs = parse_deep_lenient(source).expect("Deep fixture must parse");
    check_ir_program(&exprs)
        .expect_err("adversarial Deep fixture must be rejected")
        .errors
}

fn surf_errors(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let exprs = expand_program(&desugar_program(&decls), &ExpansionOptions::default())
        .expect("Surf fixture must macro-expand")
        .into_exprs();
    check_ir_program(&exprs)
        .expect_err("adversarial Surf fixture must be rejected")
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
    let exprs = parse_unchecked_legacy(
        "(def {} bad (cast {} (var {} missing_value) (t-prim {} f32 extra)))",
    );
    let errors = check_ir_program(&exprs)
        .expect_err("adversarial unchecked Deep fixture must be rejected")
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
                  (lit {type: (t-prim {} int32)} 2))
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
fn surf_unknown_alias_target_has_a_complete_source_location() {
    let errors = surf_errors(
        r#"
type Broken = Missing
def keep(x: Broken) -> Broken = x
"#,
    );
    assert_eq!(
        errors.len(),
        1,
        "invalid alias header must report once without downstream cascade: {errors:?}"
    );
    let error = &errors[0];
    assert!(error.message.contains("Missing"), "{errors:?}");
    assert!(
        error.span_offset.is_some() && error.span_id.is_some(),
        "Surf desugaring preserves both byte offset and synthesized span id: {error:?}"
    );
}
