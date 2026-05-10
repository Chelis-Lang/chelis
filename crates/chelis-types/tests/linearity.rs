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

fn typecheck_surf(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    check_typed_program(&deep)
        .map(|_| ())
        .map_err(|result| result.errors)
}

#[test]
fn detects_use_after_consume() {
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = realize(x)
    add(x, y)
  }
"#,
    )
    .expect_err("linearity should reject reusing a consumed tensor");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
            && error.message.contains("realize")
    }));
}

#[test]
fn copy_allows_reuse() {
    check_surf(
        r#"
def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = relu(copy(x))
    out: tensor[4, f32] = add(x, y)
    _ = drop(x)
    _ = drop(y)
    out
  }
"#,
    )
    .expect("copy should preserve a later consuming use");
}

#[test]
fn auto_borrowed_read_only_primitives_allow_fanout() {
    check_surf(
        r#"
def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = relu(x)
    z: tensor[4, f32] = sigmoid(x)
    out: tensor[4, f32] = add(y, z)
    _ = drop(x)
    _ = drop(y)
    _ = drop(z)
    out
  }
"#,
    )
    .expect("read-only tensor primitives should borrow their tensor inputs");
}

#[test]
fn pipe_auto_borrows_read_only_stage_input() {
    check_surf(
        r#"
def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = x |> relu |> sigmoid
    _ = drop(x)
    y
  }
"#,
    )
    .expect("pipe stages should use the same call-site auto-borrow rule");
}

#[test]
fn explicit_borrow_does_not_satisfy_owned_parameter() {
    let errors = typecheck_surf(
        r#"
def take(x: tensor[4, f32]): tensor[4, f32] = x

def bad(x: tensor[4, f32]): tensor[4, f32] =
  {
    take(&x)
  }
"#,
    )
    .expect_err("borrow-to-owned must require an explicit copy");

    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch))
    );
}

#[test]
fn copy_accepts_explicit_borrow_and_returns_owned_tensor() {
    check_surf(
        r#"
def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = copy(&x)
    _ = drop(x)
    y
  }
"#,
    )
    .expect("copy(&x) should fork ownership from a borrowed tensor");
}

#[test]
fn borrowed_but_never_consumed_is_auto_dropped() {
    check_surf(
        r#"
def ok(): tensor[4, f32] =
  {
    x: tensor[4, f32] = to_tensor([1.0, 2.0, 3.0, 4.0])
    y: tensor[4, f32] = relu(x)
    y
  }
"#,
    )
    .expect("implicit drop handles borrowed local owners at scope end");
}

#[test]
fn borrow_cannot_escape_as_function_result() {
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]): &tensor[4, f32] = &x
"#,
    )
    .expect_err("borrows cannot be returned from functions");

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
    realize(x)
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
    _ = drop(x)
    c
  }
"#,
    )
    .expect("shape queries should be observational, not consuming");
}

#[test]
fn to_list_does_not_consume_tensor_input() {
    check_surf(
        r#"
def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    xs: List[f32] = to_list(x)
    _ = drop(xs)
    y: tensor[4, f32] = relu(x)
    _ = drop(x)
    y
  }
"#,
    )
    .expect("to_list reads the tensor without freeing it, so x must remain live");
}

#[test]
fn to_list_still_flags_use_after_genuine_consume() {
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = realize(x)
    xs: List[f32] = to_list(x)
    y
  }
"#,
    )
    .expect_err("to_list after a consuming use of x should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
    }));
}

#[test]
fn tensor_to_scalar_does_not_consume_tensor_input() {
    check_surf(
        r#"
def ok(x: tensor[f32]): tensor[f32] =
  {
    v: f64 = tensor_to_scalar(x)
    _ = drop(v)
    y: tensor[f32] = relu(x)
    _ = drop(x)
    y
  }
"#,
    )
    .expect("tensor_to_scalar reads the tensor without freeing it, so x must remain live");
}

#[test]
fn tensor_to_scalar_still_flags_use_after_genuine_consume() {
    let errors = check_surf(
        r#"
def bad(x: tensor[f32]): tensor[f32] =
  {
    y: tensor[f32] = realize(x)
    v: f64 = tensor_to_scalar(x)
    y
  }
"#,
    )
    .expect_err("tensor_to_scalar after a consuming use of x should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
    }));
}

#[test]
fn grad_accepts_function_with_borrowed_tensor_parameter() {
    check_surf(
        r#"
def loss(x: &tensor[4, f32]): tensor[f32] = sum(mul(x, x), 0)

def ok(x: tensor[4, f32]): tensor[4, f32] =
  {
    g: tensor[4, f32] = grad(loss)(x)
    _ = drop(x)
    g
  }
"#,
    )
    .expect("grad(f)(x) should auto-borrow when f takes a borrowed tensor");
}

#[test]
fn vmap_lifts_function_with_borrowed_tensor_parameter() {
    check_surf(
        r#"
def activate(x: &tensor[4, f32]): tensor[4, f32] = relu(x)

def ok(xs: tensor[batch, 4, f32]): tensor[batch, 4, f32] =
  {
    ys: tensor[batch, 4, f32] = vmap(activate, axis=0)(xs)
    _ = drop(xs)
    ys
  }
"#,
    )
    .expect("vmap should preserve borrowed tensor parameters while adding the batch axis");
}
