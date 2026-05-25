//! Regression tests for issue #237 (sweep) and #229 (folded in).
//!
//! # Diagnosis
//!
//! Issue #237 reported a "zero-offset spurious-consume" pile in three
//! consumer paths beyond what PR #228's chelis #226 fix patched. After
//! reproducing each shape minimally, the bug class splits cleanly into:
//!
//! 1. **Closure capture (genuine chelis bug)** — `check_fn` in
//!    `crates/chelis-types/src/linearity.rs` unconditionally
//!    structurally consumes every captured tensor-carrying name,
//!    regardless of how the closure body actually uses the capture.
//!    A closure whose body only borrow-reads a capture (e.g.
//!    `fn (i) -> add(c, c)`) should be a *borrow* of the outer `c`,
//!    not a *consume*. The fix mirrors the user-function auto-borrow
//!    inference at `infer.rs:1454`: classify the capture as a read
//!    whenever the closure body has no consuming use of the name.
//!
//! 2. **Pipe-stage auto-borrow inference (#229, sibling of #226)** —
//!    `pipe_consumes_param` in `crates/chelis-types/src/infer.rs:1934`
//!    only recognizes the bare-var pipe-stage shape (`(var f)`); for
//!    explicit-arg pipe stages (`x |> add(k)`) the desugarer emits a
//!    synthesized `(fn (params __chelis_pipe) (app callee ... (var
//!    __chelis_pipe) ...))` lambda. Without peering through that
//!    lambda the inferencer mis-classifies pipe-stage borrow-reads
//!    as consumes, so the wrapping function never gets auto-borrow
//!    inferred. Fix: reuse the `resolve_pipe_stage_callee` helper
//!    from linearity.rs (lifted to a small shared utility).
//!
//! 3. **Direct call / match scrutinee diagnostics in coral** — these
//!    fire on `gf: GroupedFrame[n]` and `df: Frame[n]` parameters that
//!    are *annotated owned-linear* and *actually consumed inside the
//!    callee body* (e.g. `nrows`'s `columns(df)` consumes `df`; later
//!    `get_column(df, …)` is a real use-after-consume per
//!    `spec/04-type-system.md` §8.3 and
//!    `spec/design/implicit_linearity.md` "Signature Inference"
//!    which states "Written `&T` or `T` annotations are
//!    authoritative" and "Inference applies only to unannotated
//!    parameters"). Surfacing those as chelis bugs would require a
//!    spec change to extend auto-borrow inference to annotated
//!    parameters; that is out of scope for this PR and is being
//!    surfaced to the orchestrator as a downstream-source fix on
//!    coral, not a chelis fix.

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

// -------------------------------------------------------------------
// Closure-capture consumer class (#237)
// -------------------------------------------------------------------

#[test]
fn closure_that_only_borrow_reads_capture_does_not_consume_outer_var() {
    // Minimal reproducer for the coral-side closure-capture shape:
    // `inner = fn (i) -> add(c, c)` captures `c` and borrow-reads it
    // (`add` borrows both args). The closure must not structurally
    // consume `c`, so the later `add(c, inner(...))` still sees `c`
    // as live.
    check_surf(
        r#"
type Frame[n] =
  | Frame { col: tensor[n, f32] }
def get_col[n](df: Frame[n]) -> tensor[n, f32] = {
  match df with {
    | Frame { col: c } => c
  }
}
def f[n](df: Frame[n]) -> tensor[n, f32] = {
  c = get_col(df)
  inner = fn (i: int64) -> add(c, c)
  add(c, inner(cast(0, int64)))
}
"#,
    )
    .expect(
        "issue #237: a closure that only borrow-reads its capture must not \
         structurally consume the captured variable",
    );
}

#[test]
fn closure_that_only_reads_capture_via_outer_let_binding_is_a_borrow_capture() {
    // Variant of the above with a flat let binding (no destructure)
    // and a direct tensor parameter. Locks the simpler shape so a
    // regression in the destructure path does not mask a regression
    // in the bare-binding path.
    check_surf(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] = {
  g = fn (i: int64) -> add(x, x)
  add(x, g(cast(0, int64)))
}
"#,
    )
    .expect(
        "issue #237: bare-binding closure capture that only borrow-reads must \
         not structurally consume the outer var",
    );
}

#[test]
fn closure_chain_with_only_borrow_reads_does_not_consume_outer() {
    // Multiple sibling closures each borrow-read the same outer var;
    // none consumes it. The final `add(x, ...)` outside both closures
    // must still see `x` as live.
    check_surf(
        r#"
def f(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = {
  g = fn (v: tensor[4, f32]) -> add(x, v)
  h = fn (v: tensor[4, f32]) -> mul(x, v)
  a = g(y)
  b = h(y)
  add(x, add(a, b))
}
"#,
    )
    .expect(
        "issue #237: multiple closures each borrow-reading the same outer var \
         must not consume that var",
    );
}

#[test]
fn closure_that_actually_consumes_capture_still_trips_use_after_consume() {
    // Negative parity: if the closure body actually consumes `x`
    // (via `realize`, a structural consume), the later use of `x`
    // outside the closure must still trip `UseAfterConsume`. The fix
    // must not over-permit.
    let errors = check_surf(
        r#"
def f(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = {
  g = fn (other: tensor[4, f32]) -> realize(x)
  z = g(y)
  add(x, z)
}
"#,
    )
    .expect_err(
        "issue #237 fix must preserve detection: a closure that structurally \
         consumes its capture must still flag the outer var as consumed",
    );
    assert!(
        errors.iter().any(
            |error| matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("variable `x`")
        ),
        "expected UseAfterConsume on `x`, got: {errors:?}"
    );
}

#[test]
fn closure_that_consumes_via_returning_capture_still_trips_use_after_consume() {
    // The closure's body returns the captured `x` as-is — that's a
    // structural consume (returning an owned tensor moves it out of
    // the closure scope). The outer scope must still see `x` as
    // consumed.
    let errors = check_surf(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] = {
  g = fn (i: int64) -> x
  z = g(cast(0, int64))
  add(x, z)
}
"#,
    )
    .expect_err(
        "issue #237 fix must preserve detection: returning a captured tensor \
         from a closure body is a structural consume of the capture",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume diagnostic, got: {errors:?}"
    );
}

#[test]
fn closure_that_consumes_via_app_arg_still_trips_use_after_consume() {
    // The closure body passes the captured tensor to a user fn whose
    // first param is owned-linear, so the body structurally consumes
    // the capture. Later use outside must still be flagged.
    let errors = check_surf(
        r#"
def take(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(x: tensor[4, f32]) -> tensor[4, f32] = {
  g = fn (i: int64) -> take(x)
  z = g(cast(0, int64))
  add(x, z)
}
"#,
    )
    .expect_err(
        "issue #237 fix must preserve detection: a closure that passes its \
         capture to a consuming user fn must still flag the outer var",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume diagnostic, got: {errors:?}"
    );
}

// -------------------------------------------------------------------
// Auto-borrow inference (issue #229) — pipe_consumes_param
// -------------------------------------------------------------------

#[test]
fn auto_borrow_inference_through_pipe_stage_with_explicit_args() {
    // Smoking-gun shape for #229. `reader` only uses `t` through a
    // pipe stage that wraps a borrow-arg builtin call (`add` borrows
    // both args, so `t |> add(k)` is a borrow-read). `reader.t` is
    // unannotated, so auto-borrow inference applies.
    //
    // Without peering through the synthesized `__chelis_pipe` lambda,
    // `pipe_consumes_param` returns `true` for `t`, so the inferencer
    // marks `reader.t` as owned, the call `reader(x, k)` consumes
    // `x`, and the later `add(x, y)` trips `UseAfterConsume`.
    check_surf(
        r#"
def reader(t, k: tensor[4, f32]) = t |> add(k)
def caller(x: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = {
  y = reader(x, k)
  add(x, y)
}
"#,
    )
    .expect(
        "issue #229: pipe_consumes_param must peer through the synthesized \
         pipe-stage lambda so the auto-borrow inference sees the inner \
         borrow-arg builtin and infers `reader`'s param as Ref",
    );
}

#[test]
fn auto_borrow_inference_through_pipe_stage_chain_with_explicit_args() {
    // Composed pipe chain `t |> add(k) |> add(k)` — each stage is a
    // synthesized lambda. The inferencer must see both stages as
    // borrow-reads and conclude `reader.t` is read-only.
    check_surf(
        r#"
def reader(t, k: tensor[4, f32]) = t |> add(k) |> add(k)
def caller(x: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = {
  y = reader(x, k)
  add(x, y)
}
"#,
    )
    .expect(
        "issue #229: composed pipe-stage chains with explicit args must be \
         classified as borrow-reads by auto-borrow inference",
    );
}

#[test]
fn auto_borrow_pipe_stage_consuming_user_fn_keeps_param_owned() {
    // Negative parity: if the pipe stage wraps a USER fn whose first
    // param is owned-linear (so the pipe DOES consume `t`),
    // pipe_consumes_param must report `t` as consumed. The inferencer
    // then keeps `reader.t` owned, and `reader(x, k)` legitimately
    // consumes `x`. A later `add(x, y)` IS a use-after-consume.
    let errors = check_surf(
        r#"
def consume_two(t: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = t
def reader(t, k: tensor[4, f32]) = t |> consume_two(k)
def caller(x: tensor[4, f32], k: tensor[4, f32]) -> tensor[4, f32] = {
  y = reader(x, k)
  add(x, y)
}
"#,
    )
    .expect_err(
        "issue #229 fix must preserve detection: a pipe stage wrapping a \
         user fn that consumes its first arg still consumes the piped value",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume on `x`, got: {errors:?}"
    );
}
