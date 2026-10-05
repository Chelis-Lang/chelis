//! chelis#3180: [05-OP-67] makes the operand of `drop` owned, so a borrowed
//! operand is a type error rather than an implicit consume of its owner. Before
//! the fix `drop(&x)` and a `drop` of a borrowed parameter type checked.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn type_errors(source: &str) -> Vec<CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

#[track_caller]
fn assert_borrowed_drop_operand(source: &str, what: &str) {
    let errors = type_errors(source);
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::TypeMismatch)
                && error
                    .message
                    .contains("drop argument 1: expected an owned value")
                && error.message.contains("[05-OP-67]")
                && error.expected.as_deref() == Some("an owned value")
                && error.got.as_deref().is_some_and(|got| got.starts_with('&'))
        }),
        "{what}: expected a TypeMismatch for the borrowed drop operand; got {errors:?}"
    );
}

#[track_caller]
fn assert_type_checks(source: &str, what: &str) {
    let errors = type_errors(source);
    assert!(
        errors.is_empty(),
        "{what}: expected no type errors; got {errors:?}"
    );
}

#[test]
fn dropping_an_explicit_borrow_is_a_type_error() {
    assert_borrowed_drop_operand(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(&x)
    x
  }
"#,
        "drop(&x)",
    );
}

#[test]
fn dropping_a_borrowed_parameter_is_a_type_error() {
    assert_borrowed_drop_operand(
        r#"
def f(x: &tensor[4, f32]) -> i32 =
  {
    c = drop(x)
    1
  }
"#,
        "drop of a borrowed parameter",
    );
}

#[test]
fn dropping_a_borrow_at_the_top_level_is_a_type_error() {
    assert_borrowed_drop_operand(
        r#"
x = to_tensor([1.5f32, 2.5f32, 3.5f32])
d = drop(&x)
"#,
        "top-level drop(&x)",
    );
}

// Negative twins.

#[test]
fn dropping_an_owned_local_type_checks() {
    assert_type_checks(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    y = exp(x)
    c = drop(y)
    1
  }
"#,
        "drop of an owned local",
    );
}

#[test]
fn dropping_an_owned_parameter_type_checks() {
    assert_type_checks(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    c = drop(x)
    1
  }
"#,
        "drop of an owned parameter",
    );
}

#[test]
fn dropping_a_copy_of_a_borrowed_parameter_type_checks() {
    // `copy` turns a borrow into an independent owned value, which `drop` takes.
    assert_type_checks(
        r#"
def f(x: &tensor[4, f32]) -> i32 =
  {
    c = drop(copy(x))
    1
  }
"#,
        "drop of a copy of a borrowed parameter",
    );
}
