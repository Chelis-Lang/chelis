//! Regression tests for issue #226.
//!
//! Before the fix, `check_pipe` only recognized bare-var pipe stages
//! (`x |> f`); any stage with explicit arguments (`x |> f(y)`) was
//! desugared into a synthesized `(fn (params __chelis_pipe) (app f
//! ... __chelis_pipe ...))` lambda whose top tag is `fn`, not `var`.
//! `check_pipe` then called `arg_is_borrowed(None, 0, ...)` which fell
//! back to inspecting the lambda's type (not the inner callee's), so
//! the piped value was tagged with a structural-pipe consume even for
//! known borrow-arg builtins like `shape`, `add`, `mul`, `matmul`,
//! etc. Subsequent uses of the same variable then tripped
//! `UseAfterConsume` against a "pipe into stage at offset 0 from
//! offset 0" diagnostic (zero spans because the lambda is synthetic).
//!
//! The fix teaches `check_pipe` to peer through the synthesized
//! `__chelis_pipe` lambda, reach the inner callee, and use that for
//! the borrow check. These tests pin both arms of the contract:
//!
//! - positive: a pipe whose stage calls a borrow-arg builtin with
//!   explicit args (e.g. `table |> shape(0)`) must not consume the
//!   piped variable;
//! - negative: a pipe whose stage actually consumes the piped value
//!   (e.g. `x |> realize`) still trips `UseAfterConsume` on a later
//!   read.

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
#[ignore = "issue 226 fixture: enabled by the upcoming check_pipe fix"]
fn pipe_into_shape_with_explicit_axis_does_not_consume() {
    // Minimal repro of issue #226. `table |> shape(cast(0, int32))`
    // calls `shape(table, 0)` which is a borrow-arg builtin at arg 0.
    // The piped value (`table`) lands at the borrow position, so the
    // pipe must not consume `table`, and the later `gather(table, ...)`
    // must succeed.
    check_surf(
        r#"
def f[batch, seq, max_seq, hidden](
    table: tensor[max_seq, hidden, f32],
    ids: &tensor[batch, seq, int64]
) -> tensor[batch, seq, hidden, f32] = {
  max_seq_dim = table |> shape(cast(0, int32))
  gather(table, ids, 0)
}
"#,
    )
    .expect(
        "issue #226: pipe into a borrow-arg builtin must read, not consume; \
         later use of the piped variable must remain legal",
    );
}

#[test]
#[ignore = "issue 226 fixture: enabled by the upcoming check_pipe fix"]
fn pipe_into_add_with_explicit_other_arg_does_not_consume() {
    // `add` borrows both args; `x |> add(y)` must not consume `x`.
    // Covers the multi-arg borrow-arg builtin shape that downstream
    // shells use heavily for tensor-scalar compose patterns.
    check_surf(
        r#"
def f(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = {
  z = x |> add(y)
  add(x, z)
}
"#,
    )
    .expect(
        "issue #226: pipe into a multi-arg borrow-arg builtin must read, not consume",
    );
}

#[test]
#[ignore = "issue 226 fixture: enabled by the upcoming check_pipe fix"]
fn pipe_into_mul_chain_with_explicit_args_does_not_consume() {
    // Composed pipe chain `x |> mul(k) |> mul(k)` — both stages are
    // synthesized lambdas. The piped tensor must survive all stages
    // and remain readable afterward.
    check_surf(
        r#"
def f(x: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> mul(k) |> mul(k)
  add(x, y)
}
"#,
    )
    .expect(
        "issue #226: composed pipe stages into borrow-arg builtins must not \
         consume the piped variable",
    );
}

#[test]
fn pipe_into_realize_still_consumes() {
    // Negative parity: `realize` IS a structural consume (returns an
    // owned tensor that destroys the input view).  After the fix the
    // peering must not over-permit; `x |> realize` followed by a later
    // read of `x` must still trip `UseAfterConsume`.
    let errors = check_surf(
        r#"
def bad(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> realize
  add(x, y)
}
"#,
    )
    .expect_err(
        "issue #226 fix must preserve consume detection: pipe into `realize` \
         is still a structural consume of the piped variable",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("variable `x`")),
        "expected UseAfterConsume on `x`, got: {errors:?}"
    );
}

#[test]
fn pipe_into_user_consuming_function_still_consumes() {
    // A user-defined function whose first parameter is owned-linear
    // (`tensor[4, f32]`, not `&tensor[...]`) consumes its argument by
    // signature. The pipe must propagate that — `x |> take` consumes
    // `x`, and a later `add(x, y)` must trip `UseAfterConsume`.
    let errors = check_surf(
        r#"
def take(t: tensor[4, f32]) -> tensor[4, f32] = t
def bad(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> take
  add(x, y)
}
"#,
    )
    .expect_err(
        "issue #226 fix must preserve consume detection: pipe into a user fn \
         whose first param is owned-linear still consumes the piped variable",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume diagnostic, got: {errors:?}"
    );
}

#[test]
fn pipe_into_user_consuming_function_with_explicit_args_still_consumes() {
    // Same as above but the pipe stage carries explicit args, so the
    // desugarer emits a synthesized lambda. `x |> consume_two(k)`
    // expands to `(fn (params __chelis_pipe) (app consume_two
    // __chelis_pipe k))`. The peering must look up `consume_two`'s
    // signature, NOT the synthesized lambda's, and still mark `x` as
    // consumed.
    let errors = check_surf(
        r#"
def consume_two(t: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = t
def bad(x: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> consume_two(k)
  add(x, y)
}
"#,
    )
    .expect_err(
        "issue #226 fix must preserve consume detection inside synthesized \
         pipe-stage lambdas wrapping a user fn that consumes its first arg",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume diagnostic, got: {errors:?}"
    );
}
