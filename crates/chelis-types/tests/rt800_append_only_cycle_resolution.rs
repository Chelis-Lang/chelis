use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context};

fn surf(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_str(source).expect("Surf fixture parses"))
}

fn assert_checks(source: &str) {
    let result = check_ir_program(&surf(source));
    assert!(result.is_ok(), "known recursion must check: {result:?}");
}

#[test]
fn self_recursive_function_is_prebound_without_diagnostic_erasure() {
    assert_checks(
        r#"
def countdown(n: int32) =
  if eq(n, 0) then 0 else countdown(sub(n, 1))
"#,
    );
}

#[test]
fn mutually_recursive_functions_are_prebound_without_diagnostic_erasure() {
    assert_checks(
        r#"
def is_even(n: int32) =
  if eq(n, 0) then true else is_odd(sub(n, 1))
def is_odd(n: int32) =
  if eq(n, 0) then false else is_even(sub(n, 1))
"#,
    );
}

#[test]
fn module_mutual_cycle_uses_the_canonical_dependency_schedule() {
    assert_checks(
        r#"
module Cycle
def is_even(n: int32) =
  if eq(n, 0) then true else is_odd(sub(n, 1))
def is_odd(n: int32) =
  if eq(n, 0) then false else is_even(sub(n, 1))
"#,
    );
}

#[test]
fn context_check_prebinds_new_cycle_and_keeps_library_names_visible() {
    let library = surf("def dec(n: int32) = sub(n, 1)");
    let context = build_type_env_from_library(&library).expect("library checks");
    let new_code = surf(
        r#"
def is_even(n: int32) =
  if eq(n, 0) then true else is_odd(dec(n))
def is_odd(n: int32) =
  if eq(n, 0) then false else is_even(dec(n))
"#,
    );
    let result = check_ir_with_context(&context, &new_code);
    assert!(result.is_ok(), "context recursion must check: {result:?}");
}

#[test]
fn genuinely_unknown_callable_still_reports_exactly_once() {
    let result = check_ir_with_context(
        &TypeEnv::empty(),
        &surf("def caller(n: int32) -> int32 = missing(n)"),
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
    assert_checks(
        r#"
def identity(x) = x
def use_int() = identity(1)
def use_bool() = identity(true)
"#,
    );
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
    (app {} (var {} add) (var {} n) (lit {type: (t-prim {} int32)} 1))))
"#,
    )
    .expect("Deep fixture parses");
    let result =
        check_ir_program(&deep).expect_err("bare forward helpers retain textual semantics");
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
