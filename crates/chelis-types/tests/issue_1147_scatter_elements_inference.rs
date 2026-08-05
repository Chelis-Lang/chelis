//! `scatter_elements` must be checked through a registered inference rule.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;

fn diagnostics(source: &str) -> Vec<String> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let deep = desugar_program(&decls);
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
def apply(data: tensor[2, 3, f32], indices: tensor[2, 2, int32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] =
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
    assert_rejected(&source, "int32 axis");
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
            .any(|error| error.contains("int32") || error.contains("int64")),
        "floating-point indices checked clean: {errors:#?}"
    );
}

#[test]
fn string_data_is_rejected() {
    assert_rejected(
        r#"
def apply(indices: tensor[2, 2, int32], updates: tensor[2, 2, f32]) =
  scatter_elements("notatensor", indices, updates, 0)
"#,
        "tensor data",
    );
}

#[test]
fn updates_must_match_indices_shape_and_data_precision() {
    assert_rejected(
        r#"
def apply(data: tensor[2, 3, f32], indices: tensor[2, 2, int32], updates: tensor[2, 3, f64]) -> tensor[2, 3, f32] =
  scatter_elements(data, indices, updates, 1)
"#,
        "updates",
    );
}
