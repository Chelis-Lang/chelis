//! `scatter_elements` must be checked through a registered inference rule.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;

fn diagnostics(source: &str) -> Vec<String> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| format!("{:?}: {}", error.kind, error.message))
            .collect(),
    }
}

fn assert_rejected(source: &str, needle: &str) {
    let errors = diagnostics(source);
    assert!(
        errors.iter().any(|error| error.contains(needle)),
        "expected a rejection containing {needle:?}; got {errors:#?}"
    );
}

const PREFIX: &str = r#"
def apply(data: tensor[2, 3, f32], indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] =
"#;

#[test]
fn valid_scatter_elements_is_accepted() {
    let source = format!("{PREFIX}  scatter_elements(data, indices, updates, 1)\n");
    let errors = diagnostics(&source);
    assert!(
        errors.is_empty(),
        "valid scatter_elements rejected: {errors:#?}"
    );
}

#[test]
fn string_axis_is_rejected() {
    let source = format!("{PREFIX}  scatter_elements(data, indices, updates, \"zero\")\n");
    assert_rejected(&source, "i32 axis");
}

#[test]
fn out_of_bounds_axis_is_rejected() {
    let source = format!("{PREFIX}  scatter_elements(data, indices, updates, 99)\n");
    assert_rejected(&source, "out of bounds");
}

#[test]
fn floating_point_indices_are_rejected() {
    let errors = diagnostics(
        r#"
def apply(data: tensor[2, 3, f32], indices: tensor[2, 2, f32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] =
  scatter_elements(data, indices, updates, 1)
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("i32") || error.contains("i64")),
        "floating-point indices checked clean: {errors:#?}"
    );
}

#[test]
fn string_data_is_rejected() {
    assert_rejected(
        r#"
def apply(indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) =
  scatter_elements("notatensor", indices, updates, 0)
"#,
        "tensor data",
    );
}

#[test]
fn updates_must_match_indices_shape_and_data_precision() {
    assert_rejected(
        r#"
def apply(data: tensor[2, 3, f32], indices: tensor[2, 2, i32], updates: tensor[2, 3, f64]) -> tensor[2, 3, f32] =
  scatter_elements(data, indices, updates, 1)
"#,
        "updates",
    );
}

#[test]
fn symbolic_containment_diagnostic_does_not_leak_rust_dimension_internals() {
    let errors = diagnostics(
        r#"
def apply[n, k](data: tensor[n, 3, f32], indices: tensor[k, 3, i32], updates: tensor[k, 3, f32]) -> tensor[n, 3, f32] =
  scatter_elements(data, indices, updates, 1)
"#,
    );
    assert!(
        errors.iter().any(|error| error.contains(
            "scatter_elements cannot prove that the symbolic indices extent fits the symbolic data extent on non-axis dimension 0"
        )),
        "expected a stable symbolic-containment diagnostic; got {errors:#?}"
    );
    assert!(
        errors
            .iter()
            .all(|error| !error.contains("DimVar(") && !error.contains("Var(DimVar")),
        "user-facing diagnostics may not expose Rust dimension internals: {errors:#?}"
    );
}

/// chelis#1147: `check_scatter_elements` must reach the data and updates
/// precisions through `unify_tensor_prec`, not compare their representations.
///
/// The vehicle is a declared precision binder `p` meeting the concrete `f32`
/// of `updates`. Under [04-INF-6] that program is now rejected, because the
/// body narrows an authored binder, and the rejection is what makes this a
/// sharper test rather than a weaker one: the declaration's rigid-binder error
/// is the ONLY diagnostic. A representation-comparing `scatter_elements` would
/// have added its own `updates precision must match data precision` before
/// ever reaching the declaration check, so the absence of that second
/// diagnostic is the evidence that the unifier ran and bound `p := f32`.
///
/// The test previously asserted that this program was accepted, which
/// [04-INF-6] decided otherwise; the property it was written for is unchanged
/// and is asserted on the same source.
#[test]
fn precision_variable_is_unified_by_the_same_rule_as_other_tensor_ops() {
    let errors = diagnostics(
        r#"
def apply[p](data: tensor[2, 3, p], indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) -> tensor[2, 3, p] =
  scatter_elements(data, indices, updates, 1)
"#,
    );
    assert!(
        errors
            .iter()
            .all(|error| !error.contains("updates precision must match data precision")),
        "a precision variable must unify with the updates precision instead of being \
         compared by representation: {errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declared type parameter `p` of `apply` was narrowed")),
        "the body narrows the authored binder `p` to `f32`, which [04-INF-6] rejects at \
         the declaration: {errors:#?}"
    );
}

/// The failure twin: a precision binder that both operands share is never
/// narrowed, so the same builtin accepts it. Without this the assertion above
/// could be satisfied by rejecting every polymorphic precision.
#[test]
fn a_precision_binder_shared_by_data_and_updates_stays_accepted() {
    let errors = diagnostics(
        r#"
def apply[p](data: tensor[2, 3, p], indices: tensor[2, 2, i32], updates: tensor[2, 2, p]) -> tensor[2, 3, p] =
  scatter_elements(data, indices, updates, 1)
"#,
    );
    assert!(
        errors.is_empty(),
        "a shared precision binder must stay polymorphic: {errors:#?}"
    );
}
