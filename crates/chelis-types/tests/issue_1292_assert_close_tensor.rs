//! chelis#1292: `test_assert_close_tensor` is float-only and its tolerance
//! has exactly the tensor element dtype.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

const FLOAT_DTYPES: &[&str] = &["f16", "bf16", "f32", "f64"];
const REJECTED_DTYPES: &[&str] = &["int8", "int16", "int32", "int64", "bool"];

fn diagnostics(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(report) => report.errors,
    }
}

fn rendered(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_active_float_dtype_and_same_dtype_tolerance_are_accepted() {
    for dtype in FLOAT_DTYPES {
        let source = format!(
            r#"
def check(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit =
  test_assert_close_tensor(actual, expected, tol, "own-width")
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.is_empty(),
            "{dtype} must be accepted with its own tolerance dtype; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn shared_precision_generic_wrapper_is_accepted_for_every_active_float_dtype() {
    for dtype in FLOAT_DTYPES {
        let source = format!(
            r#"
def close[p](actual: &tensor[2, p], expected: &tensor[2, p], tol: p) -> unit ! {{ Test }} =
  test_assert_close_tensor(actual, expected, tol, "generic")

def check(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  close(actual, expected, tol)
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.is_empty(),
            "shared-precision generic wrapper must accept {dtype}; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn arbitrary_tensor_ranks_are_accepted_with_one_shared_shape() {
    for tensor_type in ["tensor[f32]", "tensor[2, 3, f32]", "tensor[2, 3, 4, f64]"] {
        let tolerance_type = if tensor_type.ends_with("f64]") {
            "f64"
        } else {
            "f32"
        };
        let source = format!(
            r#"
def check(actual: &{tensor_type}, expected: &{tensor_type}, tol: {tolerance_type}) -> unit ! {{ Test }} =
  test_assert_close_tensor(actual, expected, tol, "rank")
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.is_empty(),
            "{tensor_type} must satisfy the arbitrary-rank signature; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn independently_polymorphic_tolerance_cannot_bypass_same_dtype_constraint() {
    for (tensor_dtype, tolerance_dtype) in [
        ("f16", "bf16"),
        ("bf16", "f32"),
        ("f32", "f64"),
        ("f64", "f32"),
    ] {
        let source = format!(
            r#"
def unsafe_close[p, q](
  actual: &tensor[2, p],
  expected: &tensor[2, p],
  tol: q
) -> unit ! {{ Test }} =
  test_assert_close_tensor(actual, expected, tol, "generic")

def bad(
  actual: &tensor[2, {tensor_dtype}],
  expected: &tensor[2, {tensor_dtype}],
  tol: {tolerance_dtype}
) -> unit ! {{ Test }} =
  unsafe_close(actual, expected, tol)
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            !errors.is_empty(),
            "an independently-polymorphic {tolerance_dtype} tolerance must not bypass the {tensor_dtype} tensor constraint"
        );
    }
}

#[test]
fn shared_precision_generic_wrapper_rejects_non_float_instantiations() {
    for dtype in REJECTED_DTYPES {
        let source = format!(
            r#"
def close[p](actual: &tensor[2, p], expected: &tensor[2, p], tol: p) -> unit ! {{ Test }} =
  test_assert_close_tensor(actual, expected, tol, "generic")

def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  close(actual, expected, tol)
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                    && error.message.contains("active float dtype")
                    && error.message.contains(dtype)
            }),
            "generic {dtype} instantiation must receive the float-domain diagnostic; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn every_non_float_tensor_dtype_is_rejected_at_check_time() {
    for dtype in REJECTED_DTYPES {
        let source = format!(
            r#"
def check(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}]) -> unit =
  test_assert_close_tensor(actual, expected, cast(0.0, f32), "float-only")
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                    && error.message.contains("active float dtype")
                    && error.message.contains(dtype)
            }),
            "{dtype} must receive the float-domain diagnostic; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn tolerance_dtype_must_equal_the_tensor_dtype() {
    let errors = diagnostics(
        r#"
def check(actual: &tensor[2, f64], expected: &tensor[2, f64], tol: f32) -> unit =
  test_assert_close_tensor(actual, expected, tol, "same dtype")
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                && error.message.contains("tolerance dtype `f32`")
                && error.message.contains("tensor dtype `f64`")
        }),
        "a mixed f64/f32 application must name both dtypes; got:\n{}",
        rendered(&errors)
    );
}

#[test]
fn non_float_tolerance_is_rejected_at_check_time() {
    for dtype in ["int32", "bool"] {
        let source = format!(
            r#"
def check(actual: &tensor[2, f32], expected: &tensor[2, f32], tol: {dtype}) -> unit =
  test_assert_close_tensor(actual, expected, tol, "float tolerance")
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                    && error.message.contains("tolerance must have")
                    && error.message.contains(dtype)
            }),
            "{dtype} tolerance must receive the float-domain diagnostic; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn tensor_shapes_and_dtypes_must_match() {
    for source in [
        r#"
def bad(actual: &tensor[2, f32], expected: &tensor[3, f32]) -> unit =
  test_assert_close_tensor(actual, expected, cast(0.0, f32), "shape")
"#,
        r#"
def bad(actual: &tensor[2, f32], expected: &tensor[2, f64]) -> unit =
  test_assert_close_tensor(actual, expected, cast(0.0, f32), "dtype")
"#,
        r#"
def bad(actual: &tensor[2, f32], expected: &tensor[2, 1, f32]) -> unit =
  test_assert_close_tensor(actual, expected, cast(0.0, f32), "rank")
"#,
    ] {
        let errors = diagnostics(source);
        assert!(
            !errors.is_empty(),
            "mismatched tensors must be rejected before evaluation"
        );
    }
}
