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

// A `drop` operand that is still an inference variable at the call is
// re-checked against the declaration's final substitution, so a borrow that
// pins that same variable later is refused too. A borrow that reaches `drop`
// by instantiating a generalized binding's type scheme is chelis#3200.

#[test]
fn an_ascribed_lambda_dropping_a_borrowed_parameter_is_a_type_error() {
    assert_borrowed_drop_operand(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    g: (&tensor[4, f32]) -> unit = fn (y) -> drop(y)
    c = g(&x)
    x
  }
"#,
        "ascribed lambda over a borrow",
    );
}

#[test]
fn an_immediately_applied_lambda_dropping_a_borrow_is_a_type_error() {
    assert_borrowed_drop_operand(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = (fn (y) -> drop(y))(&x)
    x
  }
"#,
        "immediately applied lambda over &x",
    );
}

#[test]
fn an_ascribed_lambda_dropping_an_owned_parameter_type_checks() {
    assert_type_checks(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    g: (tensor[4, f32]) -> unit = fn (y) -> drop(y)
    c = g(x)
    1
  }
"#,
        "ascribed lambda over an owned tensor",
    );
}

#[test]
fn an_immediately_applied_lambda_dropping_an_owned_value_type_checks() {
    assert_type_checks(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    c = (fn (y) -> drop(y))(x)
    1
  }
"#,
        "immediately applied lambda over an owned tensor",
    );
}

#[test]
fn a_generalized_dropping_lambda_stays_polymorphic() {
    // The re-check must not hold the lambda's operand variable monomorphic.
    assert_type_checks(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    g = fn (y) -> drop(y)
    a = g(1)
    b = g(true)
    c = g(x)
    1
  }
"#,
        "one dropping lambda used at i32, bool and an owned tensor",
    );
}

#[test]
fn a_generic_dropping_function_at_an_owned_type_type_checks() {
    assert_type_checks(
        r#"
def kill[T](y: T) -> unit = drop(y)
def f(x: tensor[4, f32]) -> i32 =
  {
    a = kill(1)
    c = kill(x)
    1
  }
"#,
        "generic kill at i32 and an owned tensor",
    );
}
