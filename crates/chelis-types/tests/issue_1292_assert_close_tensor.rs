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

fn assert_one_active_float_error(errors: &[CheckError], dtype: &str, route: &str) {
    assert_eq!(
        errors.len(),
        1,
        "{route} at {dtype} must emit exactly one diagnostic, never an empty-error fallback or a cascade; got:\n{}",
        rendered(errors)
    );
    let error = &errors[0];
    assert!(
        matches!(error.kind, CheckErrorKind::PrecisionMismatch)
            && error.message.contains("active float dtype")
            && error.message.contains(dtype),
        "{route} at {dtype} must emit the active-float PrecisionMismatch; got:\n{}",
        rendered(errors)
    );
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
def close[p: Float](actual: &tensor[2, p], expected: &tensor[2, p], tol: p) -> unit ! {{ Test }} =
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
fn function_value_aliases_preserve_the_active_float_domain() {
    for dtype in REJECTED_DTYPES {
        let fixtures = [
            (
                "top-level alias",
                format!(
                    r#"
close_alias = test_assert_close_tensor

def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  close_alias(actual, expected, tol, "top")
"#,
                ),
            ),
            (
                "local alias",
                format!(
                    r#"
def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} = {{
  close_alias = test_assert_close_tensor
  close_alias(actual, expected, tol, "local")
}}
"#,
                ),
            ),
            (
                "nested alias",
                format!(
                    r#"
close_alias = test_assert_close_tensor
nested_alias = close_alias

def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  nested_alias(actual, expected, tol, "nested")
"#,
                ),
            ),
            (
                "higher-order alias",
                format!(
                    r#"
close_alias = test_assert_close_tensor

def invoke(f, actual, expected, tol) = f(actual, expected, tol, "higher-order")

def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  invoke(close_alias, actual, expected, tol)
"#,
                ),
            ),
            (
                "returned alias",
                format!(
                    r#"
def return_close() = test_assert_close_tensor
returned_alias = return_close()

def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  returned_alias(actual, expected, tol, "returned")
"#,
                ),
            ),
            (
                "stored alias",
                format!(
                    r#"
close_pair = (test_assert_close_tensor, test_assert_close_tensor)
stored_alias = close_pair.0

def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} =
  stored_alias(actual, expected, tol, "stored")
"#,
                ),
            ),
        ];

        for (route, source) in fixtures {
            assert_one_active_float_error(&diagnostics(&source), dtype, route);
        }
    }
}

#[test]
fn aliased_higher_order_calls_accept_every_float_at_rank_zero_and_multiple_ranks() {
    for dtype in FLOAT_DTYPES {
        for tensor_type in [
            format!("tensor[{dtype}]"),
            format!("tensor[2, {dtype}]"),
            format!("tensor[2, 3, 4, {dtype}]"),
        ] {
            let source = format!(
                r#"
close_alias = test_assert_close_tensor
nested_alias = close_alias
def return_close() = nested_alias
returned_alias = return_close()
close_pair = (returned_alias, nested_alias)
stored_alias = close_pair.0

def invoke(f, actual, expected, tol) = f(actual, expected, tol, "positive")

def check(actual: &{tensor_type}, expected: &{tensor_type}, tol: {dtype}) -> unit ! {{ Test }} =
  invoke(stored_alias, actual, expected, tol)
"#,
            );
            let errors = diagnostics(&source);
            assert!(
                errors.is_empty(),
                "aliased higher-order call must accept {tensor_type}; got:\n{}",
                rendered(&errors)
            );
        }
    }
}

#[test]
fn aliases_still_reject_mixed_precision_and_rank() {
    for (label, source) in [
        (
            "mixed precision",
            r#"
close_alias = test_assert_close_tensor
def bad(actual: &tensor[2, f32], expected: &tensor[2, f64], tol: f32) -> unit ! { Test } =
  close_alias(actual, expected, tol, "mixed precision")
"#,
        ),
        (
            "mixed rank",
            r#"
close_alias = test_assert_close_tensor
def bad(actual: &tensor[2, f32], expected: &tensor[2, 1, f32], tol: f32) -> unit ! { Test } =
  close_alias(actual, expected, tol, "mixed rank")
"#,
        ),
    ] {
        let errors = diagnostics(source);
        assert!(
            errors.len() == 1
                && matches!(
                    errors[0].kind,
                    CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch
                ),
            "{label} through an alias must produce one structural mismatch; got:\n{}",
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
def close[p: Float](actual: &tensor[2, p], expected: &tensor[2, p], tol: p) -> unit ! {{ Test }} =
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
