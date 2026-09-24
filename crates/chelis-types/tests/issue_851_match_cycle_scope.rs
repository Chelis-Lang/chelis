//! Match-arm binders are lexical locals, not eager top-level references.

use chelis_types::check_typed_program;

fn diagnostics(source: &str) -> Vec<String> {
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    match check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| error.message.clone())
            .collect(),
    }
}

#[test]
fn arm_binder_does_not_create_a_top_level_cycle() {
    let errors = diagnostics(include_str!(
        "fixtures/core/tests/issue_851_match_cycle_scope/arm_binder_does_not_create_a_top_level_cycle.0.deep"
    ));
    assert!(errors.is_empty(), "lexical arm binder leaked: {errors:#?}");
}

#[test]
fn nested_pattern_binders_scope_guard_and_body() {
    let errors = diagnostics(include_str!(
        "fixtures/core/tests/issue_851_match_cycle_scope/nested_pattern_binders_scope_guard_and_body.0.deep"
    ));
    assert!(errors.is_empty(), "nested arm binders leaked: {errors:#?}");
}

#[test]
fn genuine_top_level_cycle_still_rejects() {
    let source = include_str!(
        "fixtures/core/tests/issue_851_match_cycle_scope/genuine_top_level_cycle_still_rejects.0.deep"
    );
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    let errors: Vec<_> = chelis_types::infer_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .errors
    .into_iter()
    .map(|error| error.message)
    .collect();
    assert!(
        errors.iter().any(|error| error.contains("binding cycle")),
        "a genuine cycle must still reject: {errors:#?}"
    );
}
