//! chelis#1112: the [05-DIM-1]/[05-DIM-2] extent-domain dtype rule.
//!
//! Extent-domain quantities (dimension extents, slice bounds, pad amounts,
//! stride steps, expand sizes) are `i64`; axis-domain quantities (rank
//! indices, permutation entries) are `i32`. `shape()` returns `i64`
//! and its axis parameter stays `i32`.
//!
//! Spec authority: `spec/05-risc-primitives.md` §2.4 ([05-DIM-1],
//! [05-DIM-2] and the movement signature table) and
//! `spec/04-type-system.md` §4.7 (shape/expand forms), §4.7.3 (the
//! shape-list recognizer, bare and cast element forms), §4.7.5 (the
//! reshape precision rule and its diagnostics).
//!
//! Both polarities throughout: every accepted `i64` form has the
//! rejected `i32` counterpart beside it.

use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn typecheck_core(source: &str) -> Vec<CheckError> {
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    match check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked) {
        Ok(_) => Vec::new(),
        Err(prog_errors) => prog_errors.errors,
    }
}

fn errors_summary(errors: &[CheckError]) -> String {
    if errors.is_empty() {
        "(no errors)".to_string()
    } else {
        errors
            .iter()
            .map(|e| format!("[{:?}] {}", e.kind, e.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn assert_clean(source: &str, what: &str) {
    let errors = typecheck_core(source);
    assert!(
        errors.is_empty(),
        "{what} must type-check clean; got:\n{}",
        errors_summary(&errors)
    );
}

fn assert_rejected_with(source: &str, needle: &str, what: &str) {
    let errors = typecheck_core(source);
    assert!(
        !errors.is_empty(),
        "{what} must be rejected, but checked clean"
    );
    assert!(
        errors.iter().any(|e| e.message.contains(needle)),
        "{what} rejection must mention {needle:?}; got:\n{}",
        errors_summary(&errors)
    );
}

// ---------------------------------------------------------------------------
// [05-DIM-2]: shape() returns i64; its axis parameter stays i32.
// ---------------------------------------------------------------------------

#[test]
fn shape_read_types_int64() {
    assert_clean(
        include_str!("fixtures/core/tests/issue_1112_extent_dtypes/shape_read_types_int64.0.deep"),
        "shape() read declared i64",
    );
}

#[test]
fn shape_read_is_not_int32() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_1112_extent_dtypes/shape_read_is_not_int32.0.deep"
    ));
    assert!(
        !errors.is_empty(),
        "shape() read declared i32 must be rejected under [05-DIM-2]"
    );
}

#[test]
fn shape_axis_stays_int32_and_rejects_int64() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/shape_axis_stays_int32_and_rejects_int64.0.deep"
        ),
        "shape() with cast(0, i32) axis",
    );
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/shape_axis_stays_int32_and_rejects_int64.1.deep"
        ),
        "i32 axis",
        "shape() with an i64 axis",
    );
}

// ---------------------------------------------------------------------------
// shrink: bounds are List<List<i64>>.
// ---------------------------------------------------------------------------

#[test]
fn shrink_int64_bounds_typecheck() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/shrink_int64_bounds_typecheck.0.deep"
        ),
        "shrink with i64 bounds pairs",
    );
}

#[test]
fn shrink_int32_bounds_rejected_naming_int64() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/shrink_int32_bounds_rejected_naming_int64.0.deep"
        ),
        "i64",
        "shrink with bare i32 bounds pairs",
    );
}

// ---------------------------------------------------------------------------
// pad: padding pairs are List<List<i64>>.
// ---------------------------------------------------------------------------

#[test]
fn pad_int64_pairs_typecheck() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/pad_int64_pairs_typecheck.0.deep"
        ),
        "pad with i64 padding pairs",
    );
}

#[test]
fn pad_int32_pairs_rejected_naming_int64() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/pad_int32_pairs_rejected_naming_int64.0.deep"
        ),
        "i64",
        "pad with bare i32 padding pairs",
    );
}

// ---------------------------------------------------------------------------
// stride: steps are i64.
// ---------------------------------------------------------------------------

#[test]
fn stride_int64_steps_typecheck() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/stride_int64_steps_typecheck.0.deep"
        ),
        "stride with i64 steps",
    );
}

#[test]
fn stride_int32_steps_rejected_naming_int64() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/stride_int32_steps_rejected_naming_int64.0.deep"
        ),
        "i64",
        "stride with bare i32 steps",
    );
}

// ---------------------------------------------------------------------------
// expand (positional): axis stays i32, size becomes i64.
// ---------------------------------------------------------------------------

#[test]
fn expand_int64_size_typechecks() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/expand_int64_size_typechecks.0.deep"
        ),
        "expand with an i64 literal size",
    );
}

#[test]
fn expand_int32_size_rejected_naming_int64() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/expand_int32_size_rejected_naming_int64.0.deep"
        ),
        "write Ni64",
        "expand with a bare i32 size",
    );
}

#[test]
fn expand_int64_axis_rejected() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/expand_int64_axis_rejected.0.deep"
        ),
        "i32",
        "expand with an i64 axis",
    );
}

/// The canonical broadcast idiom: a `shape()` read feeding the size slot.
/// Under [05-DIM-2] both sides are i64, so the idiom is well-typed by
/// construction with no cast.
#[test]
fn expand_shape_sourced_size_typechecks() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/expand_shape_sourced_size_typechecks.0.deep"
        ),
        "expand with a shape()-sourced size",
    );
}

// ---------------------------------------------------------------------------
// expand (named-axis, spec/04 §4.5.3): size is an i64 literal.
// ---------------------------------------------------------------------------

#[test]
fn named_axis_expand_int64_size_typechecks() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/named_axis_expand_int64_size_typechecks.0.deep"
        ),
        "named-axis expand with an i64 literal size",
    );
}

#[test]
fn named_axis_expand_int32_size_rejected() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/named_axis_expand_int32_size_rejected.0.deep"
        ),
        "i64",
        "named-axis expand with a bare i32 size",
    );
}

// ---------------------------------------------------------------------------
// reshape (spec/04 §4.7.3/§4.7.5): the bare shape() element is recognized
// and propagates the input's symbolic dim; the cast spelling stays
// accepted; an unsuffixed i32 list is rejected with a fix-naming
// diagnostic.
// ---------------------------------------------------------------------------

/// The recognizer must fire on the bare element: if it does not, the
/// output dim falls back to a fresh wildcard and fails to unify with the
/// declared `batch` dim (the issue-206 oracle shape).
#[test]
fn reshape_bare_shape_element_propagates_symbolic_dim() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reshape_bare_shape_element_propagates_symbolic_dim.0.deep"
        ),
        "reshape with a bare shape() element and i64 literal",
    );
}

/// Continuity: the pre-[05-DIM-2] cast spelling is an identity cast and
/// keeps its propagation.
#[test]
fn reshape_cast_shape_element_still_propagates() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reshape_cast_shape_element_still_propagates.0.deep"
        ),
        "reshape with the cast-wrapped shape() element",
    );
}

#[test]
fn reshape_suffixed_literal_list_typechecks() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reshape_suffixed_literal_list_typechecks.0.deep"
        ),
        "reshape with an i64-suffixed literal list",
    );
}

#[test]
fn reshape_unsuffixed_list_rejected_naming_the_fix() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reshape_unsuffixed_list_rejected_naming_the_fix.0.deep"
        ),
        "i64",
        "reshape with an unsuffixed i32 shape list",
    );
}

// ---------------------------------------------------------------------------
// Axis-domain pins: [05-DIM-1]'s other half. These lock that the i64
// extent flip did NOT move axis-domain surfaces, and that a future change
// cannot flip them silently.
// ---------------------------------------------------------------------------

#[test]
fn permute_axes_stay_int32() {
    assert_clean(
        include_str!("fixtures/core/tests/issue_1112_extent_dtypes/permute_axes_stay_int32.0.deep"),
        "permute with bare i32 axes",
    );
    assert_rejected_with(
        include_str!("fixtures/core/tests/issue_1112_extent_dtypes/permute_axes_stay_int32.1.deep"),
        "i32 axis indices",
        "permute with i64 axes",
    );
}

/// [05-DIM-3]: reduce_window window_shape/strides are extent-domain
/// `List[i64]`; the old `List[i32]` spelling is rejected.
#[test]
fn reduce_window_lists_are_int64() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reduce_window_lists_are_int64.0.deep"
        ),
        "reduce_window_max with i64 window/strides",
    );
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reduce_window_lists_are_int64.1.deep"
        ),
        "List[i64]",
        "reduce_window_max with i32 window",
    );
}

// ---------------------------------------------------------------------------
// spec/04 §4.7.5: a genuinely mixed shape list is an intra-list error no
// defaulting rule can resolve.
// ---------------------------------------------------------------------------

#[test]
fn reshape_mixed_suffix_list_rejected() {
    assert_rejected_with(
        include_str!(
            "fixtures/core/tests/issue_1112_extent_dtypes/reshape_mixed_suffix_list_rejected.0.deep"
        ),
        "i32",
        "reshape with a mixed-suffix shape list",
    );
}
