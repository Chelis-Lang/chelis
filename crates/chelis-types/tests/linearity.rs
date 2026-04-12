use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn check_surf(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).map(|_| ())
}

#[test]
fn detects_use_after_consume() {
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = relu(x)
    add(x, y)
  }
"#,
    )
    .expect_err("linearity should reject reusing a consumed tensor");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
            && error.message.contains("relu")
    }));
}

#[test]
fn copy_allows_reuse() {
    check_surf(
        r#"
def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = relu(copy(x))
    add(x, y)
  }
"#,
    )
    .expect("copy should preserve a later consuming use");
}

#[test]
fn borrow_preserves_tensor_for_later_consumption() {
    check_surf(
        r#"
def keep(x: tensor[4, f32]): int32 = 1

def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    n: int32 = keep(&x)
    relu(x)
  }
"#,
    )
    .expect("borrowed call should not consume x");
}

#[test]
fn borrow_cannot_be_stored() {
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]): tensor[4, f32] =
  {
    y = &x
    x
  }
"#,
    )
    .expect_err("borrow binding should be rejected");

    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::InvalidBorrow))
    );
}

#[test]
fn closure_capture_consumes_outer_tensor() {
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]): tensor[4, f32] =
  {
    f = fn () -> x
    relu(x)
  }
"#,
    )
    .expect_err("capturing a tensor should consume it");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("closure capture")
    }));
}

#[test]
fn match_consumes_tuple_scrutinee() {
    let errors = check_surf(
        r#"
def bad(pair: (tensor[4, f32], int32)): int32 =
  {
    n: int32 = match pair with {
      | (x, _) => 1
    }
    again: (tensor[4, f32], int32) = pair
    n
  }
"#,
    )
    .expect_err("reusing a tuple carrying a tensor after match should fail");

    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume))
    );
}

#[test]
fn tensor_shape_queries_do_not_consume_tensor_inputs() {
    check_surf(
        r#"
def ok(x: tensor[2, 3, f32]): int32 =
  {
    r: int32 = rank(x)
    c: int32 = shape(x, 1)
    n: int64 = numel(x)
    c
  }
"#,
    )
    .expect("shape queries should be observational, not consuming");
}
