//! chelis#1113 fail-closed guard: axis-argument dtype acceptance is
//! consistent while the classification atom is pending.
//!
//! An axis names a rank position and is int32 in every enforced surface
//! (spec/05-risc-primitives.md [05-DIM-1] states the split for the
//! movement/shape surface; chelis#1113 owns extending the classification
//! to reduction, concatenation, and windowed ops). Before this guard,
//! `sum` rejected an int64 axis while `cumsum` and `concat` silently
//! accepted both dtypes. The guard closes that acceptance without
//! deciding chelis#1113: every axis-taking builtin on the
//! `resolve_builtin_axis` path (cumsum, sort, gather, scatter,
//! scatter_replace) plus concat's inline path now rejects a non-int32
//! axis for the same reason `sum` does.
//!
//! Both polarities per op: the int32 axis stays accepted.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn typecheck_surf(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
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
    let errors = typecheck_surf(source);
    assert!(
        errors.is_empty(),
        "{what} must type-check clean; got:\n{}",
        errors_summary(&errors)
    );
}

fn assert_rejects_int64_axis(source: &str, op: &str) {
    let errors = typecheck_surf(source);
    assert!(
        !errors.is_empty(),
        "{op} with an int64 axis must be rejected, but checked clean"
    );
    assert!(
        errors.iter().any(|e| e.message.contains("int32 axis")),
        "{op} rejection must name the int32 axis contract; got:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn cumsum_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = cumsum(&x, 1)
"#,
        "cumsum with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = cumsum(&x, 1i64)
"#,
        "cumsum",
    );
}

#[test]
fn concat_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32], y: tensor[2, 4, f32]) -> tensor[4, 4, f32] = concat([x, y], 0)
"#,
        "concat with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32], y: tensor[2, 4, f32]) -> tensor[4, 4, f32] = concat([x, y], 0i64)
"#,
        "concat",
    );
}

#[test]
fn sort_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> (tensor[2, 4, f32], tensor[2, 4, int64]) = sort(&x, 1)
"#,
        "sort with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32]) -> (tensor[2, 4, f32], tensor[2, 4, int64]) = sort(&x, 1i64)
"#,
        "sort",
    );
}

/// A cast-wrapped int64 axis is the same rejection: the guard reads the
/// resolved type, not the literal spelling, so `cast(1, int64)` cannot
/// slip past the way it slipped past the value extractor.
#[test]
fn cumsum_cast_int64_axis_rejected() {
    assert_rejects_int64_axis(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = cumsum(&x, cast(1, int64))
"#,
        "cumsum",
    );
}

/// Reference behavior: `sum` already rejected an int64 axis before the
/// guard existed. Pinned here so the consistency claim has its anchor.
#[test]
fn sum_int64_axis_still_rejected() {
    let errors = typecheck_surf(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[4, f32] = sum(&x, 0i64)
"#,
    );
    assert!(
        !errors.is_empty(),
        "sum with an int64 axis must stay rejected"
    );
}
