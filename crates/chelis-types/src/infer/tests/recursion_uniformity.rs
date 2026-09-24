//! Unit tests for the spec/04 §3.1.1 uniform-recursive-instantiation rule
//! ([04-INF-2]/[04-INF-3]): recursive binding groups must type every
//! in-group call at the caller's own instantiation. Positive/negative
//! parity per the `add-bounded-monomorphization` change (chelis#1158).

use super::*;

fn legacy_errors(src: &str) -> Vec<CheckError> {
    let exprs: Vec<deep::Expr> = serde_json::from_str(src).expect("lossless AST fixture");
    infer_program(
        &exprs,
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    )
    .errors
}

fn legacy_ir_errors(src: &str) -> Vec<CheckError> {
    let exprs: Vec<deep::Expr> = serde_json::from_str(src).expect("lossless AST fixture");
    match check_ir_program(
        &exprs,
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn assert_no_polymorphic_recursion_error(errors: &[CheckError]) {
    assert!(
        !errors
            .iter()
            .any(|error| error.message.contains("polymorphic recursion")),
        "uniform recursion must not trip [04-INF-3]: {errors:?}"
    );
}

fn polymorphic_recursion_errors(errors: &[CheckError]) -> Vec<&CheckError> {
    errors
        .iter()
        .filter(|error| error.message.contains("polymorphic recursion"))
        .collect()
}

#[test]
fn direct_uniform_recursion_is_accepted() {
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/direct_uniform_recursion_is_accepted.json"
    ));
    assert!(
        errors.is_empty(),
        "uniform twin must check clean: {errors:?}"
    );
}

#[test]
fn direct_polymorphic_recursion_is_rejected_with_the_atom() {
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/direct_polymorphic_recursion_is_rejected_with_the_atom.json"
    ));
    let poly = polymorphic_recursion_errors(&errors);
    assert_eq!(
        poly.len(),
        1,
        "exactly one polymorphic-recursion error expected: {errors:?}"
    );
    let error = poly[0];
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert!(error.message.contains("`f`"), "must name the function");
    assert_eq!(error.got.as_deref(), Some("[Box[a]]"));
    assert_eq!(error.expected.as_deref(), Some("[a]"));
    assert!(error.span_offset.is_some(), "must carry a source span");
}

#[test]
fn mutual_uniform_recursion_is_accepted() {
    let source = include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/mutual_uniform_recursion_is_accepted.json"
    );
    let errors = legacy_errors(source);
    assert!(
        errors.is_empty(),
        "uniform twin must check clean: {errors:?}"
    );
    let ir_errors = legacy_ir_errors(source);
    assert!(
        ir_errors.is_empty(),
        "IR recursive driver must generalize authored members after SCC exit: {ir_errors:?}"
    );
}

#[test]
fn inferred_recursive_members_generalize_after_both_driver_scc_boundaries() {
    let source = "(def {} left
            (fn {} (params {} x stop)
              (if {} (var {} stop) (var {} x)
                (app {} (var {} right) (var {} x)
                  (lit {type: (t-prim {} bool)} true)))))
         (def {} right
            (fn {} (params {} x stop)
              (if {} (var {} stop) (var {} x)
                (app {} (var {} left) (var {} x)
                  (lit {type: (t-prim {} bool)} true)))))
         (def {} int_use
            (app {} (var {} left) (lit {type: (t-prim {} i32)} 1)
              (lit {type: (t-prim {} bool)} false)))
         (def {} bool_use
            (app {} (var {} left) (lit {type: (t-prim {} bool)} true)
              (lit {type: (t-prim {} bool)} false)))";
    let exprs = chelis_deep::parser::parse_str(source).expect("Deep fixture parses");
    let inferred = infer_program(
        &exprs,
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    );
    assert!(
        inferred.errors.is_empty(),
        "primary driver must generalize inferred SCC members: {:?}",
        inferred.errors
    );
    check_ir_program(
        &exprs,
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    )
    .expect("IR driver must generalize inferred SCC members");
}

#[test]
fn recursive_group_errors_clear_scope_before_the_next_check_in_both_drivers() {
    let broken = include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/recursive_group_errors_clear_scope_before_the_next_check_in_both_drivers.json"
    );
    assert!(!legacy_errors(broken).is_empty());
    assert!(!legacy_ir_errors(broken).is_empty());

    let clean = include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/recursive_group_errors_clear_scope_before_the_next_check_in_both_drivers_2.json"
    );
    assert!(legacy_errors(clean).is_empty());
    assert!(legacy_ir_errors(clean).is_empty());
}

#[test]
fn mutual_polymorphic_recursion_growing_across_the_cycle_is_rejected() {
    // The ping -> pong edge grows the instantiation (`Box[a]`); the
    // pong -> ping edge is uniform on its own. The cycle is unbounded, and
    // the growing edge is the one named.
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/mutual_polymorphic_recursion_growing_across_the_cycle_is_rejected.json"
    ));
    let poly = polymorphic_recursion_errors(&errors);
    assert_eq!(
        poly.len(),
        1,
        "exactly the growing edge must be rejected: {errors:?}"
    );
    let error = poly[0];
    assert!(
        error.message.contains("`ping`") && error.message.contains("`pong`"),
        "must name caller and callee: {}",
        error.message
    );
}

#[test]
fn polymorphic_recursion_behind_a_let_alias_is_rejected() {
    // A let-bound alias of a group member must not re-generalize the
    // member's in-group instantiation; using the alias at a grown
    // instantiation is the same [04-INF-3] violation.
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/polymorphic_recursion_behind_a_let_alias_is_rejected.json"
    ));
    let poly = polymorphic_recursion_errors(&errors);
    assert!(
        !poly.is_empty(),
        "an aliased grown recursive call must reject: {errors:?}"
    );
}

#[test]
fn uniform_recursion_behind_a_let_alias_is_accepted() {
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/uniform_recursion_behind_a_let_alias_is_accepted.json"
    ));
    assert_no_polymorphic_recursion_error(&errors);
}

#[test]
fn monomorphic_caller_in_group_names_the_missing_type_parameters() {
    // Red-team NIT (2026-08-06): a monomorphic group member calling an
    // authored generic at a concrete instantiation is rejected by the
    // strict authored-binder rule, and the diagnostic must not advise
    // reusing type parameters the caller does not have.
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/monomorphic_caller_in_group_names_the_missing_type_parameters.json"
    ));
    let poly = polymorphic_recursion_errors(&errors);
    assert_eq!(poly.len(), 1, "exactly one rejection expected: {errors:?}");
    let error = poly[0];
    assert!(
        error.message.contains("`mono`") && error.message.contains("`gen`"),
        "must name caller and callee: {}",
        error.message
    );
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert_eq!(error.expected.as_deref(), Some("[]"));
}

#[test]
fn out_of_group_calls_at_fresh_instantiations_are_unrestricted() {
    // `main` is not a member of `depth`'s recursive group, so its call may
    // instantiate freely — twice at different types.
    let errors = legacy_errors(include_str!(
        "../../../tests/fixtures/core/src/recursion_uniformity/out_of_group_calls_at_fresh_instantiations_are_unrestricted.json"
    ));
    assert!(
        errors.is_empty(),
        "out-of-group instantiations are unrestricted: {errors:?}"
    );
}
