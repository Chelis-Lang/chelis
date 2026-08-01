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
def bad(x: tensor[4, f32]) -> tensor[4, f32] =
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
def ok(x: tensor[4, f32]) -> tensor[4, f32] =
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
def ok(x: tensor[4, f32]) -> tensor[4, f32] =
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
def ok(x: tensor[4, f32]) -> tensor[4, f32] =
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
    // Fixture renamed from `take` for chelis#353: `take` is a builtin
    // name and bare shadowing defs are now rejected at declaration time.
    let errors = typecheck_surf(
        r#"
def grab(x: tensor[4, f32]) -> tensor[4, f32] = x

def bad(x: tensor[4, f32]) -> tensor[4, f32] = grab(&x)
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
def ok(x: tensor[4, f32]) -> tensor[4, f32] =
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
def ok() -> tensor[4, f32] =
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
def bad(x: tensor[4, f32]) -> &tensor[4, f32] = &x
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
def bad(x: tensor[4, f32]) -> tensor[4, f32] =
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
fn closure_capture_consumes_outer_tensor_carrying_adt() {
    // Mirrors `closure_capture_consumes_outer_tensor` but for a
    // record-style ADT whose tensor lives in a variant field, not in
    // a type-argument position. Locks the parameterized
    // `type_expr_contains_tensor` call inside `check_fn`'s capture
    // pass at `linearity.rs:737-740`: if that predicate ever stops
    // consulting `tensor_carrying_adts`, this case would silently
    // regress — the outer `p` would survive the closure capture and
    // the second `use_params(p)` would type-check, hiding a use-after-
    // consume.
    let errors = check_surf(
        r#"
type Params[n] =
  | Params { weight: tensor[n, f32] }

sig use_params: Params[n] -> bool
def use_params(p) = true

def bad[n](p: Params[n]) -> bool =
  {
    f = fn () -> use_params(p)
    use_params(p)
  }
"#,
    )
    .expect_err("capturing a tensor-carrying ADT must consume it");

    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("closure capture")
        }),
        "expected UseAfterConsume with closure-capture site; got {errors:?}"
    );
}

#[test]
fn match_consumes_tuple_scrutinee() {
    let errors = check_surf(
        r#"
def bad(pair: (tensor[4, f32], int32)) -> int32 =
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
def ok(x: tensor[2, 3, f32]) -> int32 =
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
def ok(x: tensor[4, f32]) -> tensor[4, f32] =
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
def bad(x: tensor[4, f32]) -> tensor[4, f32] =
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
def ok(x: tensor[f32]) -> tensor[f32] =
  {
    v: f32 = tensor_to_scalar(x)
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
def bad(x: tensor[f32]) -> tensor[f32] =
  {
    y: tensor[f32] = realize(x)
    v: f32 = tensor_to_scalar(x)
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
fn len_does_not_consume_list_argument() {
    // chelis#527: `len` reads a `List`'s length via `chelis_list_len`
    // (a `const *` that never frees), so the caller still owns the
    // list afterwards.  The parameter is named `params` — a
    // Deep-tag-colliding identifier that desugars through the MetaExpr
    // param form #343 began consume-tracking — so this is a faithful
    // regression for the School `_step_list` read-then-reuse shape.
    check_surf(
        r#"
def ok(params: List[tensor[k, f32]]) -> List[tensor[k, f32]] =
  {
    n: int64 = len(params)
    params
  }
"#,
    )
    .expect("len reads the list without freeing it, so params must remain live (chelis#527)");
}

#[test]
fn index_does_not_consume_list_argument() {
    // chelis#527: `index` retains the element it returns and reads the
    // list via a `const *` (`chelis_list_index`), never freeing it.
    check_surf(
        r#"
def ok(params: List[tensor[k, f32]]) -> List[tensor[k, f32]] =
  {
    first: tensor[k, f32] = index(params, 0)
    _ = drop(first)
    params
  }
"#,
    )
    .expect(
        "index reads an element without freeing the list, so params must remain live (chelis#527)",
    );
}

#[test]
fn list_len_then_index_then_reuse_compiles() {
    // The full School `_step_list` read-then-reuse shape: read the
    // length and an element, then hand the list onward.  Compiled
    // clean at 0.10.0, regressed at 0.10.1 (chelis#527).
    check_surf(
        r#"
def step(params: List[tensor[k, f32]]) -> List[tensor[k, f32]] =
  {
    n: int64 = len(params)
    first: tensor[k, f32] = index(params, 0)
    _ = drop(first)
    params
  }
"#,
    )
    .expect("read length + element then reuse the list must compile (chelis#527)");
}

#[test]
fn len_still_flags_use_after_genuine_consume() {
    // Negative parity: borrowing on `len` must not blind the checker
    // to a real prior consume.  An explicit `drop` frees `params`, so
    // the later `len(params)` borrow-read is a use-after-consume.
    let errors = check_surf(
        r#"
def bad(params: List[tensor[k, f32]]) -> int64 =
  {
    _ = drop(params)
    n: int64 = len(params)
    n
  }
"#,
    )
    .expect_err("len after dropping params should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `params`")
    }));
}

#[test]
fn index_still_flags_use_after_genuine_consume() {
    // Negative parity for `index`, mirroring the `len` case above.
    let errors = check_surf(
        r#"
def bad(params: List[tensor[k, f32]]) -> tensor[k, f32] =
  {
    _ = drop(params)
    first: tensor[k, f32] = index(params, 0)
    first
  }
"#,
    )
    .expect_err("index after dropping params should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `params`")
    }));
}

#[test]
fn len_explicit_container_borrow_is_a_type_error() {
    // chelis#527: auto-borrow is the idiom for the read-only container
    // queries; the explicit `len(&xs)` surface form is intentionally
    // *not* supported and is rejected at check time (spec
    // `05-risc-primitives.md` §1.3.1). An explicit `&List` would
    // type-check past the front end but the host-value lane does not
    // erase the borrow wrapper for container values, so it would fail C
    // codegen — a worse footgun than a clear front-end error. This locks
    // the rejection so a future signature change cannot silently admit
    // `len(&xs)`.
    let errors = typecheck_surf(
        r#"
def bad(params: List[tensor[k, f32]]) -> int64 =
  {
    n: int64 = len(&params)
    n
  }
"#,
    )
    .expect_err("explicit &List is not a supported surface form for len");

    // The diagnostic must point at the real problem — `len` auto-borrows, so
    // the explicit `&` is redundant/unsupported — not the misleading bare
    // "expects List or Dict input, got &List …" (the input *is* a List).
    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error
                .message
                .contains("len auto-borrows its List/Dict argument")
            && error.message.contains("write `len(xs)`, not `len(&xs)`")
    }));
}

#[test]
fn index_explicit_container_borrow_is_a_type_error() {
    // Negative parity for `index`, mirroring the `len` case above:
    // `index(&xs, i)` is rejected at check time (chelis#527).
    let errors = typecheck_surf(
        r#"
def bad(params: List[tensor[k, f32]]) -> tensor[k, f32] =
  {
    first: tensor[k, f32] = index(&params, 0)
    first
  }
"#,
    )
    .expect_err("explicit &List is not a supported surface form for index");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error
                .message
                .contains("index auto-borrows its List argument")
            && error
                .message
                .contains("write `index(xs, i)`, not `index(&xs, i)`")
    }));
}

#[test]
fn len_of_non_container_does_not_mention_auto_borrow() {
    // The auto-borrow guidance must fire only for a genuine `&List`/`&Dict`
    // (a valid container with a redundant `&`). A non-container input is a
    // different mistake, so it must keep the plain "expects List or Dict
    // input" diagnostic and must NOT misleadingly claim `len` auto-borrows
    // an argument that is not even a container.
    let errors = typecheck_surf(
        r#"
def bad(x: f32) -> int64 =
  {
    n: int64 = len(x)
    n
  }
"#,
    )
    .expect_err("len of a scalar is a type error");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error.message.contains("len expects List or Dict input")
            && !error.message.contains("auto-borrows")
    }));
}

#[test]
fn grad_accepts_function_with_borrowed_tensor_parameter() {
    check_surf(
        r#"
def loss(x: &tensor[4, f32]) -> tensor[f32] = sum(mul(x, x), 0)

def ok(x: tensor[4, f32]) -> tensor[4, f32] =
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
def activate(x: &tensor[4, f32]) -> tensor[4, f32] = relu(x)

def ok(xs: tensor[batch, 4, f32]) -> tensor[batch, 4, f32] =
  {
    ys: tensor[batch, 4, f32] = vmap(activate)(xs)
    _ = drop(xs)
    ys
  }
"#,
    )
    .expect("vmap should preserve borrowed tensor parameters while adding the batch axis");
}
