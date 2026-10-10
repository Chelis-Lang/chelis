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
use chelis_types::CopyRepairUseKind;
use chelis_types::{check_linearity, check_typed_program};
use copy_repair::assert_copy_repaired;

#[path = "support/copy_repair.rs"]
mod copy_repair;

fn check_surf(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).map(|_| ())
}

#[test]
fn pipe_into_shape_with_explicit_axis_does_not_consume() {
    // Minimal repro of issue #226. `table |> shape(cast(0, i32))`
    // calls `shape(table, 0)` which is a borrow-arg builtin at arg 0.
    // The piped value (`table`) lands at the borrow position, so the
    // pipe must not consume `table`, and the later `gather(table, ...)`
    // must succeed.
    check_surf(
        r#"
def f[batch, seq, max_seq, hidden](
    table: tensor[max_seq, hidden, f32],
    ids: &tensor[batch, seq, i64]
) -> tensor[batch, seq, hidden, f32] = {
  max_seq_dim = table |> shape(cast(0, i32))
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
    .expect("issue #226: pipe into a multi-arg borrow-arg builtin must read, not consume");
}

#[test]
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
    // `x |> realize` consumes `x`: the later borrow is fan-out repaired at the
    // pipe stage (spec/04 section 8.3), which a borrowing stage would not record.
    assert_copy_repaired(
        r#"
def bad(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> realize
  add(x, y)
}
"#,
        "x",
        "realize",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn pipe_into_user_consuming_function_still_consumes() {
    assert_copy_repaired(
        r#"
def grab(t: tensor[4, f32]) -> tensor[4, f32] = t
def bad(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> grab
  add(x, y)
}
"#,
        "x",
        "grab",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn pipe_into_user_consuming_function_with_explicit_args_still_consumes() {
    // The peering looks up `consume_two`'s signature, not the synthesized
    // lambda's, so the copy sits at the call to `consume_two`.
    assert_copy_repaired(
        r#"
def consume_two(t: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = t
def bad(x: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = {
  y = x |> consume_two(k)
  add(x, y)
}
"#,
        "x",
        "consume_two",
        CopyRepairUseKind::Borrow,
    );
}
