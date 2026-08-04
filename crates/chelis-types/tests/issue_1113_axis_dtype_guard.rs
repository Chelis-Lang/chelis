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
//! scatter_replace), the `resolve_axis_pair_member` path (trace,
//! diagonal), and the inline `concat`/`split` arms now reject a
//! non-int32 axis for the same reason `sum` does.
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

fn assert_rejects_axis(source: &str, op: &str, axis_dtype: &str) {
    let errors = typecheck_surf(source);
    assert!(
        !errors.is_empty(),
        "{op} with a {axis_dtype} axis must be rejected, but checked clean"
    );
    assert!(
        errors.iter().any(|e| e.message.contains("int32 axis")),
        "{op} rejection must name the int32 axis contract; got:\n{}",
        errors_summary(&errors)
    );
}

fn assert_rejects_int64_axis(source: &str, op: &str) {
    assert_rejects_axis(source, op, "int64");
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

/// The indexing builtins reach the guard through the same shared
/// `resolve_builtin_axis` helper as `cumsum`/`sort`, so their acceptance
/// is pinned on its own rather than inferred from the helper's other
/// callers.
#[test]
fn gather_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[4, 3, f32], i: tensor[2, int32]) -> tensor[2, 3, f32] = gather(&x, &i, 0)
"#,
        "gather with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[4, 3, f32], i: tensor[2, int32]) -> tensor[2, 3, f32] = gather(&x, &i, 0i64)
"#,
        "gather",
    );
}

#[test]
fn scatter_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(b: tensor[4, 3, f32], i: tensor[2, int32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter(b, i, u, 0, "add")
"#,
        "scatter with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(b: tensor[4, 3, f32], i: tensor[2, int32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter(b, i, u, 0i64, "add")
"#,
        "scatter",
    );
}

#[test]
fn scatter_replace_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(b: tensor[4, 3, f32], i: tensor[2, int32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter_replace(b, i, u, 0)
"#,
        "scatter_replace with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(b: tensor[4, 3, f32], i: tensor[2, int32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter_replace(b, i, u, 0i64)
"#,
        "scatter_replace",
    );
}

/// `split`'s inline arm carried the same `precision.is_integer()`
/// acceptance `concat` did, so an int64 axis checked clean there too.
#[test]
fn split_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
pieces = split(m, 1, [cast(1, int64), cast(1, int64)])
"#,
        "split with a bare int32 axis",
    );
    assert_rejects_int64_axis(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
pieces = split(m, 1i64, [cast(1, int64), cast(1, int64)])
"#,
        "split",
    );
}

/// `trace` and `diagonal` take their axis pair through
/// `resolve_axis_pair_member`, which screened neither dtype nor kind: an
/// int64, float, or string axis checked clean.
#[test]
fn trace_int32_axis_pair_accepted_int64_rejected() {
    assert_clean(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
t = trace(m, 0, 1)
"#,
        "trace with a bare int32 axis pair",
    );
    assert_rejects_int64_axis(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
t = trace(m, 0i64, 1)
"#,
        "trace",
    );
    assert_rejects_int64_axis(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
t = trace(m, 0, 1i64)
"#,
        "trace second axis",
    );
}

#[test]
fn diagonal_int32_axis_pair_accepted_int64_rejected() {
    assert_clean(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
d = diagonal(m, 0, 1)
"#,
        "diagonal with a bare int32 axis pair",
    );
    assert_rejects_int64_axis(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
d = diagonal(m, 0i64, 1)
"#,
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
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
d = diagonal(m, 0.0, 1)
"#,
        "diagonal",
        "f32",
    );
    assert_rejects_axis(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
d = diagonal(m, "zero", 1)
"#,
        "diagonal",
        "string",
    );
}
