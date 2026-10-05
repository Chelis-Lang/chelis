//! Retained adversarial conformance probes from PR #800's final review.
//! The intentional `infer_var` mutation is represented by `errors::report`'s
//! compile-fail oracle instead of being planted in production source.

use chelis_deep::parser::parse_str as parse_deep_lenient;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn deep_errors(source: &str) -> Vec<CheckError> {
    let exprs = parse_deep_lenient(source).expect("Deep fixture must parse");
    check_ir_program(&exprs)
        .expect_err("adversarial Deep fixture must be rejected")
        .errors
}

fn surf_errors(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let exprs = expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &ExpansionOptions::default(),
    )
    .expect("Surf fixture must macro-expand")
    .into_exprs();
    check_ir_program(&exprs)
        .expect_err("adversarial Surf fixture must be rejected")
        .errors
}

#[test]
fn einsum_with_only_equation_and_one_operand_reports_one_arity_root() {
    let source = r#"(def {} bad
              (app {}
                (var {} einsum)
                (lit {type: (t-prim {} string)} "i->i")
                (lit {type: (t-prim {} f32)} 1.0)))"#;
    let errors = deep_errors(source);
    assert_eq!(
        errors.len(),
        1,
        "einsum arity must have one root: {errors:?}"
    );
    assert!(matches!(errors[0].kind, CheckErrorKind::ArityMismatch));
    assert_eq!(
        errors[0].expected.as_deref(),
        Some("3 arguments"),
        "{errors:?}"
    );
    assert_eq!(errors[0].got.as_deref(), Some("2 arguments"), "{errors:?}");
    assert_eq!(
        errors[0].span_offset,
        source.find("(app"),
        "native Deep has a measured application coordinate even without an external identity"
    );
    assert_eq!(errors[0].span_id, None);
    assert!(errors[0].message.contains("einsum"), "{errors:?}");
}

#[test]
fn cast_operand_and_target_failures_remain_two_independent_roots() {
    // A wrong-arity target such as `(t-prim {} f32 extra)` has no admitted
    // carrier (the stamped `Node` constructor rejects it at ingress), so the
    // failing target here is an unknown primitive.
    let errors =
        deep_errors("(def {} bad (cast {} (var {} missing_value) (t-prim {} MissingTarget)))");
    assert_eq!(
        errors.len(),
        2,
        "the bad operand and unknown target are independent roots: {errors:?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("missing_value")),
        "{errors:?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("MissingTarget")),
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
fn surf_unknown_alias_target_has_a_complete_source_location() {
    let source = r#"
type Broken = Missing
def keep(x: Broken) -> Broken = x
"#;
    let errors = surf_errors(source);
    assert_eq!(
        errors.len(),
        1,
        "invalid alias header must report once without downstream cascade: {errors:?}"
    );
    let error = &errors[0];
    assert!(error.message.contains("Missing"), "{errors:?}");
    assert_eq!(error.span_offset, source.find("Missing"), "{error:?}");
    assert_eq!(
        error.span_id, None,
        "the alias has an authored coordinate but no external identity: {error:?}"
    );
}
