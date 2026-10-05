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

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

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

fn assert_rejected_with(source: &str, needle: &str, what: &str) {
    let errors = typecheck_surf(source);
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

fn assert_rejected_axis_dtype(source: &str, callee: &str) {
    let errors = typecheck_surf(source);
    assert!(
        errors.iter().any(|error| {
            error.kind.diagnostic_name() == "TypeMismatch"
                && error.expected.as_deref() == Some("i32")
                && error.got.as_deref() == Some("i64")
                && error.message.contains(callee)
                && error.message.contains("axis")
        }),
        "{callee}: i64 axes must be rejected as i32 axes: {}",
        errors_summary(&errors)
    );
}

// ---------------------------------------------------------------------------
// [05-DIM-2]: shape() returns i64; its axis parameter stays i32.
// ---------------------------------------------------------------------------

#[test]
fn shape_read_types_int64() {
    assert_clean(
        r#"
def f(x: tensor[3, f32]) -> i64 = shape(x, 0)
"#,
        "shape() read declared i64",
    );
}

#[test]
fn shape_read_is_not_int32() {
    let errors = typecheck_surf(
        r#"
def f(x: tensor[3, f32]) -> i32 = shape(x, 0)
"#,
    );
    assert!(
        !errors.is_empty(),
        "shape() read declared i32 must be rejected under [05-DIM-2]"
    );
}

#[test]
fn shape_axis_stays_int32_and_rejects_int64() {
    assert_clean(
        r#"
def f(x: tensor[3, f32]) -> i64 = shape(x, cast(0, i32))
"#,
        "shape() with cast(0, i32) axis",
    );
    assert_rejected_axis_dtype(
        r#"
def g(x: tensor[3, f32]) -> i64 = shape(x, 0i64)
"#,
        "shape",
    );
}

// ---------------------------------------------------------------------------
// shrink: bounds are List<List<i64>>.
// ---------------------------------------------------------------------------

#[test]
fn shrink_int64_bounds_typecheck() {
    assert_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0i64, 1i64], [1i64, 3i64]])
"#,
        "shrink with i64 bounds pairs",
    );
}

#[test]
fn shrink_int32_bounds_rejected_naming_int64() {
    assert_rejected_with(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0, 1], [1, 3]])
"#,
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
        r#"
def f(x: tensor[2, f32]) -> tensor[4, f32] = pad(&x, [[1i64, 1i64]], 0.0)
"#,
        "pad with i64 padding pairs",
    );
}

#[test]
fn pad_int32_pairs_rejected_naming_int64() {
    assert_rejected_with(
        r#"
def f(x: tensor[2, f32]) -> tensor[4, f32] = pad(&x, [[1, 1]], 0.0)
"#,
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
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 1i64, 2i64)
"#,
        "stride with i64 steps",
    );
}

#[test]
fn stride_int32_steps_rejected_naming_int64() {
    assert_rejected_with(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 1, 2)
"#,
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
        r#"
def f(b: tensor[1, 4, f32]) -> tensor[8, 4, f32] = expand(&b, 0, 8i64)
"#,
        "expand with an i64 literal size",
    );
}

#[test]
fn expand_int32_size_rejected_naming_int64() {
    assert_rejected_with(
        r#"
def f(b: tensor[1, 4, f32]) -> tensor[8, 4, f32] = expand(&b, 0, 8)
"#,
        "write Ni64",
        "expand with a bare i32 size",
    );
}

#[test]
fn expand_int64_axis_rejected() {
    assert_rejected_with(
        r#"
def f(b: tensor[1, 4, f32]) -> tensor[8, 4, f32] = expand(&b, 0i64, 8i64)
"#,
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
        r#"
def f[n](x: tensor[n, 4, f32], b: tensor[1, 4, f32]) -> tensor[n, 4, f32] = expand(&b, 0, shape(x, 0))
"#,
        "expand with a shape()-sourced size",
    );
}

// ---------------------------------------------------------------------------
// expand (named-axis, spec/04 §4.5.3): size is an i64 literal.
// ---------------------------------------------------------------------------

#[test]
fn named_axis_expand_int64_size_typechecks() {
    assert_clean(
        r#"
sig add_axis[rest]: &tensor[..rest, f32] -> tensor[..rest, one, f32]
def add_axis(x) = insert(x, one, 1i64)
"#,
        "named-axis expand with an i64 literal size",
    );
}

#[test]
fn named_axis_expand_int32_size_rejected() {
    assert_rejected_with(
        r#"
sig add_axis[rest]: &tensor[..rest, f32] -> tensor[..rest, one, f32]
def add_axis(x) = insert(x, one, 1)
"#,
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
        r#"
sig flat_view: &tensor[batch, 4, f32] -> tensor[batch, 4, f32]
def flat_view(x) = reshape(x, [shape(x, 0), 4i64])
"#,
        "reshape with a bare shape() element and i64 literal",
    );
}

/// Continuity: the pre-[05-DIM-2] cast spelling is an identity cast and
/// keeps its propagation.
#[test]
fn reshape_cast_shape_element_still_propagates() {
    assert_clean(
        r#"
sig flat_view: &tensor[batch, 4, f32] -> tensor[batch, 4, f32]
def flat_view(x) = reshape(x, [cast(shape(x, cast(0, i32)), i64), cast(4, i64)])
"#,
        "reshape with the cast-wrapped shape() element",
    );
}

#[test]
fn reshape_suffixed_literal_list_typechecks() {
    assert_clean(
        r#"
def f(x: tensor[2, 2, f32]) -> tensor[4, f32] = reshape(&x, [4i64])
"#,
        "reshape with an i64-suffixed literal list",
    );
}

#[test]
fn reshape_unsuffixed_list_rejected_naming_the_fix() {
    assert_rejected_with(
        r#"
def f(x: tensor[2, 2, f32]) -> tensor[4, f32] = reshape(&x, [4])
"#,
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
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[4, 2, f32] = permute(&x, 1, 0)
"#,
        "permute with bare i32 axes",
    );
    assert_rejected_axis_dtype(
        r#"
def g(x: tensor[2, 4, f32]) -> tensor[4, 2, f32] = permute(&x, 1i64, 0i64)
"#,
        "permute",
    );
}

/// [05-DIM-3]: reduce_window window_shape/strides are extent-domain
/// `List[i64]`; the old `List[i32]` spelling is rejected.
#[test]
fn reduce_window_lists_are_int64() {
    assert_clean(
        r#"
def f(x: tensor[4, 4, f32]) -> tensor[2, 2, f32] = reduce_window_max(&x, [2i64, 2i64], [2i64, 2i64])
"#,
        "reduce_window_max with i64 window/strides",
    );
    let source = r#"
def g(x: tensor[4, 4, f32]) -> tensor[2, 2, f32] = reduce_window_max(&x, [2, 2], [2, 2])
"#;
    let errors = typecheck_surf(source);
    assert!(
        errors.iter().any(|error| {
            error.kind.diagnostic_name() == "TypeMismatch"
                && error.expected.as_deref() == Some("List i64")
                && error.got.as_deref() == Some("List i32")
                && error.span_offset == source.find("reduce_window_max(")
                && error.message.contains("argument 2")
        }),
        "reduce_window_max must reject the i32 window list in argument 2: {}",
        errors_summary(&errors)
    );
}

// ---------------------------------------------------------------------------
// spec/04 §4.7.5: a genuinely mixed shape list is an intra-list error no
// defaulting rule can resolve.
// ---------------------------------------------------------------------------

#[test]
fn reshape_mixed_suffix_list_rejected() {
    assert_rejected_with(
        r#"
def f(x: tensor[2, 2, f32]) -> tensor[4, f32] = reshape(&x, [2i64, 2i32])
"#,
        "i32",
        "reshape with a mixed-suffix shape list",
    );
}
