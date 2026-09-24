use chelis_deep::parser::parse_str;
use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context};

fn core_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    parse_str(source).expect("Deep fixture parses")
}

fn assert_checks(source: &str) {
    let result = check_ir_program(&core_fixture(source), chelis_types::ProgramOrigin::Unlinked);
    assert!(result.is_ok(), "known recursion must check: {result:?}");
}

#[test]
fn self_recursive_function_is_prebound_without_diagnostic_erasure() {
    assert_checks(include_str!(
        "fixtures/core/tests/rt800_append_only_cycle_resolution/case_001.deep"
    ));
}

#[test]
fn mutually_recursive_functions_are_prebound_without_diagnostic_erasure() {
    assert_checks(include_str!(
        "fixtures/core/tests/rt800_append_only_cycle_resolution/case_002.deep"
    ));
}

#[test]
fn module_mutual_cycle_uses_the_canonical_dependency_schedule() {
    assert_checks(include_str!(
        "fixtures/core/tests/rt800_append_only_cycle_resolution/case_003.deep"
    ));
}

#[test]
fn context_check_prebinds_new_cycle_and_keeps_library_names_visible() {
    let library = core_fixture(include_str!(
        "fixtures/core/tests/rt800_append_only_cycle_resolution/case_004.deep"
    ));
    let context = build_type_env_from_library(&library, chelis_types::ProgramOrigin::Unlinked)
        .expect("library checks");
    let new_code = core_fixture(include_str!(
        "fixtures/core/tests/rt800_append_only_cycle_resolution/case_005.deep"
    ));
    let result = check_ir_with_context(&context, &new_code, chelis_types::ProgramOrigin::Unlinked);
    assert!(result.is_ok(), "context recursion must check: {result:?}");
}

#[test]
fn genuinely_unknown_callable_still_reports_exactly_once() {
    let result = check_ir_with_context(
        &TypeEnv::empty(),
        &core_fixture(include_str!(
            "fixtures/core/tests/rt800_append_only_cycle_resolution/case_006.deep"
        )),
        chelis_types::ProgramOrigin::Unlinked,
    )
    .expect_err("unknown call must reject");
    let unbound = result
        .errors
        .iter()
        .filter(|error| {
            matches!(
                error.kind,
                chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        unbound.len(),
        1,
        "unknown root owns one diagnostic: {result:?}"
    );
    assert!(unbound[0].message.contains("missing"));
}

#[test]
fn acyclic_generic_helper_remains_generalized_for_downstream_calls() {
    assert_checks(include_str!(
        "fixtures/core/tests/rt800_append_only_cycle_resolution/case_007.deep"
    ));
}

#[test]
fn bare_acyclic_later_helper_remains_textually_unavailable() {
    // Raw Deep carries no synthesized `defsig`; [04-INF-4] does not widen
    // the existing function-inference contract while aligning eager values.
    let deep = chelis_deep::parser::parse_str(
        r#"
(def {} caller
  (fn {} (params {} n)
    (app {} (var {} later) (var {} n))))
(def {} later
  (fn {} (params {} n)
    (app {} (var {} add) (var {} n) (lit {type: (t-prim {} i32)} 1))))
"#,
    )
    .expect("Deep fixture parses");
    let result = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("bare forward helpers retain textual semantics");
    let unbound = result
        .errors
        .iter()
        .filter(|error| {
            matches!(
                error.kind,
                chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        unbound.len(),
        1,
        "forward root owns one diagnostic: {result:?}"
    );
    assert!(unbound[0].message.contains("later"));
}
