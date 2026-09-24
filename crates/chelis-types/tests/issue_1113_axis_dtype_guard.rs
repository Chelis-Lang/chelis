//! chelis#1113 fail-closed guard: axis-argument dtype acceptance is
//! consistent with spec/05-risc-primitives.md [05-DIM-3].
//!
//! An axis names a rank position and is i32 in every covered surface.
//! [05-DIM-1] states the movement/shape split and [05-DIM-3] applies it to
//! reductions, concatenation, and the remaining axis-taking builtins. Before
//! this guard,
//! `sum` rejected an i64 axis while `cumsum` and `concat` silently
//! accepted both dtypes. Every builtin now carries a required axis-layout
//! classification, and the shared guard closes that acceptance on the
//! `resolve_builtin_axis` path (cumsum, sort, gather, scatter,
//! scatter_replace), the `resolve_axis_pair_member` path (trace,
//! diagonal), and the inline `concat`/`split` arms now reject a
//! non-i32 axis for the same reason `sum` does.
//!
//! Both polarities per op: the i32 axis stays accepted.

use chelis_deep::parser::parse_str as parse_core;
use chelis_types::errors::CheckError;
use chelis_types::{AxisArgumentLayout, BUILTIN_NAMES, BUILTINS, check_typed_program};

fn typecheck_core(source: &str) -> Vec<CheckError> {
    let deep = if source.trim_start().starts_with('[') {
        serde_json::from_str(source).expect("lossless AST fixture")
    } else {
        parse_core(source).expect("Deep fixture parse should succeed")
    };
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

fn assert_rejects_axis(source: &str, op: &str, axis_dtype: &str) {
    let errors = typecheck_core(source);
    assert!(
        !errors.is_empty(),
        "{op} with a {axis_dtype} axis must be rejected, but checked clean"
    );
    assert!(
        errors.iter().any(|e| e.message.contains("i32 axis")),
        "{op} rejection must name the i32 axis contract; got:\n{}",
        errors_summary(&errors)
    );
}

fn assert_rejects_int64_axis(source: &str, op: &str) {
    assert_rejects_axis(source, op, "i64");
}

#[test]
fn cumsum_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/cumsum_int32_axis_accepted_int64_rejected_73.deep"
        ),
        "cumsum with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/cumsum_int32_axis_accepted_int64_rejected_79.deep"
        ),
        "cumsum",
    );
}

#[test]
fn softmax_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/softmax_int32_axis_accepted_int64_rejected_89.deep"
        ),
        "softmax with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/softmax_int32_axis_accepted_int64_rejected_95.deep"
        ),
        "softmax",
    );
}

#[test]
fn concat_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/concat_int32_axis_accepted_int64_rejected_105.deep"
        ),
        "concat with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/concat_int32_axis_accepted_int64_rejected_111.deep"
        ),
        "concat",
    );
}

#[test]
fn concat_list_overload_does_not_apply_the_axis_contract() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/concat_list_overload_does_not_apply_the_axis_contract_121.deep"
        ),
        "ordinary List concat has no axis slot",
    );
}

#[test]
fn sort_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/sort_int32_axis_accepted_int64_rejected_132.deep"
        ),
        "sort with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/sort_int32_axis_accepted_int64_rejected_138.deep"
        ),
        "sort",
    );
}

/// A cast-wrapped i64 axis is the same rejection: the guard reads the
/// resolved type, not the literal spelling, so `cast(1, i64)` cannot
/// slip past the way it slipped past the value extractor.
#[test]
fn cumsum_cast_int64_axis_rejected() {
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/cumsum_cast_int64_axis_rejected_151.deep"
        ),
        "cumsum",
    );
}

/// Reference behavior: `sum` already rejected an i64 axis before the
/// guard existed. Pinned here so the consistency claim has its anchor.
#[test]
fn sum_int64_axis_still_rejected() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_1113_axis_dtype_guard/sum_int64_axis_still_rejected_163.deep"
    ));
    assert!(
        !errors.is_empty(),
        "sum with an i64 axis must stay rejected"
    );
}

#[test]
fn count_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/count_int32_axis_accepted_int64_rejected_176.deep"
        ),
        "count with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/count_int32_axis_accepted_int64_rejected_182.deep"
        ),
        "count",
    );
}

/// The indexing builtins reach the guard through the same shared
/// `resolve_builtin_axis` helper as `cumsum`/`sort`, so their acceptance
/// is pinned on its own rather than inferred from the helper's other
/// callers.
#[test]
fn gather_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/gather_int32_axis_accepted_int64_rejected_196.deep"
        ),
        "gather with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/gather_int32_axis_accepted_int64_rejected_202.deep"
        ),
        "gather",
    );
}

#[test]
fn scatter_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/scatter_int32_axis_accepted_int64_rejected_212.json"
        ),
        "scatter with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/scatter_int32_axis_accepted_int64_rejected_219.json"
        ),
        "scatter",
    );
}

#[test]
fn scatter_replace_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/scatter_replace_int32_axis_accepted_int64_rejected_230.deep"
        ),
        "scatter_replace with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/scatter_replace_int32_axis_accepted_int64_rejected_237.deep"
        ),
        "scatter_replace",
    );
}

#[test]
fn scatter_elements_int32_axis_accepted_int64_rejected() {
    let setup = include_str!(
        "fixtures/core/tests/issue_1113_axis_dtype_guard/scatter_elements_int32_axis_accepted_int64_rejected_247.deep"
    );
    assert_clean(
        &format!(
            r###"{setup}
(def {{}}
  out
  (app {{span: "surf:6..49"}}
    (var {{span: "surf:6..22"}} scatter_elements)
    (var {{span: "surf:23..27"}} data)
    (var {{span: "surf:29..36"}} indices)
    (var {{span: "surf:38..45"}} updates)
    (lit {{span: "surf:47..48", type: (t-prim {{}} i32)}} 0)))
"###
        ),
        "scatter_elements with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        &format!(
            r###"{setup}
(def {{}}
  out
  (app {{span: "surf:6..52"}}
    (var {{span: "surf:6..22"}} scatter_elements)
    (var {{span: "surf:23..27"}} data)
    (var {{span: "surf:29..36"}} indices)
    (var {{span: "surf:38..45"}} updates)
    (lit {{span: "surf:47..51", type: (t-prim {{}} i64)}} 0)))
"###
        ),
        "scatter_elements",
    );
}

/// `split`'s inline arm carried the same `precision.is_integer()`
/// acceptance `concat` did, so an i64 axis checked clean there too.
#[test]
fn split_int32_axis_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/split_int32_axis_accepted_int64_rejected_267.deep"
        ),
        "split with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/split_int32_axis_accepted_int64_rejected_274.deep"
        ),
        "split",
    );
}

/// `trace` and `diagonal` take their axis pair through
/// `resolve_axis_pair_member`, which screened neither dtype nor kind: an
/// i64, float, or string axis checked clean.
#[test]
fn trace_int32_axis_pair_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/trace_int32_axis_pair_accepted_int64_rejected_288.deep"
        ),
        "trace with a bare i32 axis pair",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/trace_int32_axis_pair_accepted_int64_rejected_295.deep"
        ),
        "trace",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/trace_int32_axis_pair_accepted_int64_rejected_302.deep"
        ),
        "trace second axis",
    );
}

#[test]
fn diagonal_int32_axis_pair_accepted_int64_rejected() {
    assert_clean(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/diagonal_int32_axis_pair_accepted_int64_rejected_313.deep"
        ),
        "diagonal with a bare i32 axis pair",
    );
    assert_rejects_int64_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/diagonal_int32_axis_pair_accepted_int64_rejected_320.deep"
        ),
        "diagonal",
    );
}

/// The pair path's fail-open was worse than a dtype mismatch: a
/// non-integer axis extracted to `None` and silently became the
/// POSITIONAL DEFAULT, so `diagonal(m, 9.0, 0)` reported "axes 0 and 0"
/// -- naming an axis the caller never wrote -- and a string axis checked
/// clean outright. Both must now be rejected on the axis dtype.
#[test]
fn diagonal_non_integer_axis_rejected_on_dtype() {
    assert_rejects_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/diagonal_non_integer_axis_rejected_on_dtype_336.deep"
        ),
        "diagonal",
        "f32",
    );
    assert_rejects_axis(
        include_str!(
            "fixtures/core/tests/issue_1113_axis_dtype_guard/diagonal_non_integer_axis_rejected_on_dtype_344.deep"
        ),
        "diagonal",
        "string",
    );
}

/// Structural class guard: each active axis-taking builtin has exactly one
/// axis layout. The checker consumes this table for every
/// unambiguous generic application; operation-specific paths share the same
/// `reject_non_int32_axis` gate for value/rank handling.
#[test]
fn every_axis_builtin_has_one_axis_layout() {
    let mut actual = BUILTINS
        .iter()
        .filter(|decl| decl.axis_arguments != AxisArgumentLayout::NoAxes)
        .map(|decl| decl.name)
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut deduped = actual.clone();
    deduped.dedup();
    assert_eq!(actual, deduped, "axis registrations must be unique");

    let mut expected = vec![
        "argmax_reduce",
        "argmin_reduce",
        "concat",
        "count",
        "cumsum",
        "diagonal",
        "expand",
        "gather",
        "insert",
        "max_reduce",
        "mean",
        "min_reduce",
        "permute",
        "prod_reduce",
        "scatter",
        "scatter_elements",
        "scatter_replace",
        "shape",
        "softmax",
        "sort",
        "split",
        "sum",
        "trace",
    ];
    expected.sort_unstable();
    assert_eq!(actual, expected, "axis builtin inventory drifted");
    assert!(
        actual.iter().all(|name| BUILTIN_NAMES.contains(name)),
        "axis registrations must name active builtins"
    );
}
