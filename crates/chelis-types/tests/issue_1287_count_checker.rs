//! First-class `count` checker contract (chelis#1287 / [05-OP-29]).
//!
//! Positive and negative cases are paired deliberately: `count` accepts only
//! bool tensors, one or more compile-time i32 axes, and returns an i64
//! tensor with every selected axis removed.

use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn errors(source: &str) -> Vec<CheckError> {
    let deep = chelis_deep::parse_and_stamp_file(source).expect("Deep fixture");
    match check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked) {
        Ok(_) => Vec::new(),
        Err(report) => report.errors,
    }
}

fn assert_clean(source: &str) {
    let found = errors(source);
    assert!(found.is_empty(), "expected clean typecheck, got {found:#?}");
}

fn assert_rejects(source: &str, needle: &str) {
    let found = errors(source);
    assert!(
        !found.is_empty(),
        "expected type error, but program checked"
    );
    assert!(
        found.iter().any(|error| error.message.contains(needle)),
        "expected diagnostic containing {needle:?}, got {found:#?}"
    );
}

#[test]
fn concrete_positional_axes_are_variadic_and_order_independent() {
    assert_clean(include_str!(
        "fixtures/core/tests/issue_1287_count_checker/concrete_positional_axes_are_variadic_and_order_independent_1.deep"
    ));
}

#[test]
fn bool_tensor_is_required_not_numeric_tensor_or_bool_scalar() {
    assert_clean(include_str!(
        "fixtures/core/tests/issue_1287_count_checker/bool_tensor_is_required_not_numeric_tensor_or_bool_scalar_1.deep"
    ));
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/bool_tensor_is_required_not_numeric_tensor_or_bool_scalar_2.deep"
        ),
        "bool tensor",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/bool_tensor_is_required_not_numeric_tensor_or_bool_scalar_3.deep"
        ),
        "tensor input",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/bool_tensor_is_required_not_numeric_tensor_or_bool_scalar_4.deep"
        ),
        "exactly a bool tensor",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/bool_tensor_is_required_not_numeric_tensor_or_bool_scalar_5.deep"
        ),
        "exactly a bool tensor",
    );
}

#[test]
fn axes_are_required_unique_static_int32_and_in_range() {
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/axes_are_required_unique_static_int32_and_in_range_1.deep"
        ),
        "expected 2 args, got 1",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/axes_are_required_unique_static_int32_and_in_range_2.deep"
        ),
        "duplicate",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/axes_are_required_unique_static_int32_and_in_range_3.deep"
        ),
        "i32 axis",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/axes_are_required_unique_static_int32_and_in_range_4.deep"
        ),
        "compile-time constant",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/axes_are_required_unique_static_int32_and_in_range_5.deep"
        ),
        "out of bounds",
    );
}

#[test]
fn variadic_axes_enforce_int32_in_every_position() {
    assert_clean(include_str!(
        "fixtures/core/tests/issue_1287_count_checker/variadic_axes_enforce_int32_in_every_position_1.deep"
    ));

    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/variadic_axes_enforce_int32_in_every_position_2.deep"
        ),
        "i32 axis",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/variadic_axes_enforce_int32_in_every_position_3.deep"
        ),
        "i32 axis",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/variadic_axes_enforce_int32_in_every_position_4.deep"
        ),
        "i32 axis",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/variadic_axes_enforce_int32_in_every_position_5.deep"
        ),
        "i32 axis",
    );
}

#[test]
fn rank_polymorphic_count_uses_only_named_axes() {
    assert_clean(include_str!(
        "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_1.deep"
    ));
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_2.deep"
        ),
        "rank-spread",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_3.deep"
        ),
        "duplicate",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_4.deep"
        ),
        "no named `missing` axis",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_5.deep"
        ),
        "ambiguous",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_6.deep"
        ),
        "positional and named axes cannot be mixed",
    );
    assert_rejects(
        include_str!(
            "fixtures/core/tests/issue_1287_count_checker/rank_polymorphic_count_uses_only_named_axes_7.deep"
        ),
        "concrete-rank operand requires",
    );
}
