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

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{AxisArgumentLayout, BUILTIN_NAMES, BUILTINS, check_typed_program};

fn typecheck_surf(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
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
        errors.iter().any(|error| {
            error.kind.diagnostic_name() == "TypeMismatch"
                && error.expected.as_deref() == Some("i32")
                && error.got.as_deref() == Some(axis_dtype)
                && error
                    .message
                    .contains(op.split_whitespace().next().unwrap_or(op))
                && error.message.contains("axis")
        }),
        "{op} must reject a {axis_dtype} axis as i32; got:\n{}",
        errors_summary(&errors)
    );
}

fn assert_rejects_int64_axis(source: &str, op: &str) {
    assert_rejects_axis(source, op, "i64");
}

#[test]
fn cumsum_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = cumsum(&x, 1)
"#,
        "cumsum with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = cumsum(&x, 1i64)
"#,
        "cumsum",
    );
}

#[test]
fn softmax_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = softmax(&x, 1)
"#,
        "softmax with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = softmax(&x, 1i64)
"#,
        "softmax",
    );
}

#[test]
fn concat_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32], y: tensor[2, 4, f32]) -> tensor[4, 4, f32] = concat([x, y], 0)
"#,
        "concat with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32], y: tensor[2, 4, f32]) -> tensor[4, 4, f32] = concat([x, y], 0i64)
"#,
        "concat",
    );
}

#[test]
fn concat_list_overload_does_not_apply_the_axis_contract() {
    assert_clean(
        r#"
xs: List[f32] = [1.0, 2.0]
ys: List[f32] = concat(xs, [3.0, 4.0])
"#,
        "ordinary List concat has no axis slot",
    );
}

#[test]
fn sort_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> (tensor[2, 4, f32], tensor[2, 4, i64]) = sort(&x, 1)
"#,
        "sort with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, f32]) -> (tensor[2, 4, f32], tensor[2, 4, i64]) = sort(&x, 1i64)
"#,
        "sort",
    );
}

/// A cast-wrapped i64 axis is the same rejection: the guard reads the
/// resolved type, not the literal spelling, so `cast(1, i64)` cannot
/// slip past the way it slipped past the value extractor.
#[test]
fn cumsum_cast_int64_axis_rejected() {
    assert_rejects_int64_axis(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = cumsum(&x, cast(1, i64))
"#,
        "cumsum",
    );
}

/// Reference behavior: `sum` already rejected an i64 axis before the
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
        "sum with an i64 axis must stay rejected"
    );
}

#[test]
fn count_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, bool]) -> tensor[2, i64] = count(&x, 1)
"#,
        "count with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[2, 4, bool]) -> tensor[2, i64] = count(&x, 1i64)
"#,
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
        r#"
def f(x: tensor[4, 3, f32], i: tensor[2, i32]) -> tensor[2, 3, f32] = gather(&x, &i, 0)
"#,
        "gather with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(x: tensor[4, 3, f32], i: tensor[2, i32]) -> tensor[2, 3, f32] = gather(&x, &i, 0i64)
"#,
        "gather",
    );
}

#[test]
fn scatter_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(b: tensor[4, 3, f32], i: tensor[2, i32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter(b, i, u, 0, "add")
"#,
        "scatter with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(b: tensor[4, 3, f32], i: tensor[2, i32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter(b, i, u, 0i64, "add")
"#,
        "scatter",
    );
}

#[test]
fn scatter_replace_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
def f(b: tensor[4, 3, f32], i: tensor[2, i32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter_replace(b, i, u, 0)
"#,
        "scatter_replace with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
def g(b: tensor[4, 3, f32], i: tensor[2, i32], u: tensor[2, 3, f32]) -> tensor[4, 3, f32] =
  scatter_replace(b, i, u, 0i64)
"#,
        "scatter_replace",
    );
}

#[test]
fn scatter_elements_int32_axis_accepted_int64_rejected() {
    let setup = r#"
data = to_tensor([[0.0f32, 0.0f32], [0.0f32, 0.0f32]])
indices = to_tensor([[1, 0], [0, 1]], i32)
updates = to_tensor([[5.0f32, 6.0f32], [7.0f32, 8.0f32]])
"#;
    assert_clean(
        &format!("{setup}\nout = scatter_elements(data, indices, updates, 0)"),
        "scatter_elements with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        &format!("{setup}\nout = scatter_elements(data, indices, updates, 0i64)"),
        "scatter_elements",
    );
}

/// `split`'s inline arm carried the same `precision.is_integer()`
/// acceptance `concat` did, so an i64 axis checked clean there too.
#[test]
fn split_int32_axis_accepted_int64_rejected() {
    assert_clean(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
pieces = split(m, 1, [cast(1, i64), cast(1, i64)])
"#,
        "split with a bare i32 axis",
    );
    assert_rejects_int64_axis(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
pieces = split(m, 1i64, [cast(1, i64), cast(1, i64)])
"#,
        "split",
    );
}

/// `trace` and `diagonal` take their axis pair through
/// `resolve_axis_pair_member`, which screened neither dtype nor kind: an
/// i64, float, or string axis checked clean.
#[test]
fn trace_int32_axis_pair_accepted_int64_rejected() {
    assert_clean(
        r#"
m = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
t = trace(m, 0, 1)
"#,
        "trace with a bare i32 axis pair",
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
        "diagonal with a bare i32 axis pair",
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
