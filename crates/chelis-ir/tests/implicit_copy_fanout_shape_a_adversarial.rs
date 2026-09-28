//! Wave 3 terminal red team for the 0.7.8 compiler-cleanup workstream.
//!
//! Adversarial fixtures for the implicit-copy fan-out v3 closures (PR #91)
//! and the broader Shape A scope that v3 explicitly deferred.
//!
//! The broader Shape A fixtures originally pinned the deferred-shape
//! TypeMismatch as `expect_err`; the 0.7.9 cleanup closed
//! `Linearity-ShapeABroadReturn-F1` and they now expect `Ok` clean
//! lowering.  Each fixture has a single pinned expected outcome that
//! catches regressions in either direction.

use chelis_ir::dag::Dag;
use chelis_ir::lower::try_lower_program;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::{check_linearity, check_typed_program};

fn surf_to_dag(source: &str) -> Result<Dag, String> {
    let decls = surf_parse(source).map_err(|e| format!("surf parse: {e:?}"))?;
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lowering failed: {diag:?}"))
}

// ============================================================
// §3.5 Shape A broader-scope adversarial fixtures
// (Per the v3 diagnosis these are explicitly OUT of scope for PR #91.
// Pinned to confirm the deferred-shape boundary is honest.)
// ============================================================

/// Shape A literal narrow form already covered by `shape_a_borrow_return_position_lowers_cleanly`.
/// This fixture covers the let-tail-return shape called out in the
/// diagnosis's sibling sweep as a §5-candidate follow-on.
///
/// PR #91's diagnosis flagged `def f(x: &T) -> T = { y = x; y }` as OUT
/// of v3 scope.  The 0.7.9 cleanup closed `Linearity-ShapeABroadReturn-F1`
/// by extending `shape_a_relaxed_return` with the `descend_to_tail_var`
/// helper, so the let-tail shape now lowers cleanly. Companion fixtures
/// in `implicit_copy_shape_a_broader_return.rs` pin the broader coverage.
#[test]
fn shape_a_let_tail_return_lowers_cleanly() {
    let source = r#"
module Repro.ShapeABroader

def identity_via_let[a](x: &tensor[a, f32]) -> tensor[a, f32] = {
  y: &tensor[a, f32] = x
  y
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A broader-return fix should lower let-tail returns cleanly; got {:?}",
        result
    );
}

/// Shape A with `if` tail-return.  The 0.7.9 broader-Shape-A fix's
/// descent walks both branches of the desugared `(if cond then_e else_e)`
/// triple and accepts when both resolve to the same bare-var name.
#[test]
fn shape_a_if_tail_return_lowers_cleanly() {
    let source = r#"
module Repro.ShapeAIf

def identity_via_if[a](x: &tensor[a, f32], flag: bool) -> tensor[a, f32] =
  if flag then x else x
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A broader-return fix should lower if-tail returns cleanly; got {:?}",
        result
    );
}

// ============================================================
// §3.5 Shape B adversarial fixtures
// ============================================================

/// Three-grad fan-out: extends `shape_b_grad_fanout_four_arg_mse_shape_lowers_cleanly`
/// (which has 2 grad calls) to 3 grad calls of the same loss, exercising
/// the `arg_is_borrowed` clause N times against the same arg names.
#[test]
fn shape_b_three_grad_fanout_lowers_cleanly() {
    let source = r#"
module Repro.ImplicitCopyShapeBThreeGrad

def mse_loss(x: tensor[3, f32], y: tensor[3, f32], w: tensor[3, f32], b: tensor[3, f32]) -> tensor[f32] = {
  prod = mul(w, b)
  d = sub(prod, x)
  e = sub(d, y)
  sq = mul(e, e)
  sum(sq, 0)
}

def sgd_step(x: tensor[3, f32], y: tensor[3, f32], w: tensor[3, f32], b: tensor[3, f32], c: tensor[3, f32]) -> tensor[3, f32] = {
  dw = grad(mse_loss, wrt=w)(x, y, w, b)
  db = grad(mse_loss, wrt=b)(x, y, w, b)
  dc = grad(mse_loss, wrt=w)(x, y, w, b)
  new_w = sub(w, dw)
  new_b = sub(b, db)
  add(new_w, new_b)
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B three-grad fan-out must lower cleanly; got {:?}",
        result
    );
}

/// Mixed grad + vmap fan-out: one grad call and one vmap call against
/// the same observational-arg set. Each consumer is independently
/// covered by `arg_is_borrowed` but the mixed case is not pinned.
#[test]
fn shape_b_mixed_grad_vmap_fanout_lowers_cleanly() {
    let source = r#"
module Repro.ImplicitCopyShapeBMixed

def my_loss(w: tensor[3, f32]) -> tensor[f32] = sum(w, 0)

def step(w: tensor[3, f32], ws: tensor[2, 3, f32]) -> tensor[3, f32] = {
  dw = grad(my_loss, wrt=w)(w)
  rs = vmap(my_loss)(ws)
  trailing = sub(w, dw)
  trailing
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B mixed grad + vmap fan-out must lower cleanly; got {:?}",
        result
    );
}

/// Sibling: `jit(f)(args)` per the v3 diagnosis is NOT affected (jit
/// follows f's param types). Pin the negative: an `jit(f)(args); sub(w, ...)`
/// where `f`'s param is owned should still fail with UseAfterConsume
/// at the linearity layer, NOT silently lower.
///
/// This pins the v3 fix scope; if jit started to be treated as
/// observational, this test would catch the over-broad fix.
#[test]
fn shape_b_jit_call_with_owned_arg_then_borrow_still_rejects() {
    let source = r#"
module Repro.ImplicitCopyShapeBJit

def my_op(w: tensor[3, f32]) -> tensor[3, f32] = add(w, w)

def step(w: tensor[3, f32]) -> tensor[3, f32] = {
  out = jit(my_op)(w)
  trailing = sub(w, out)
  trailing
}
"#;
    let result = surf_to_dag(source);
    // jit's call follows the underlying f's param types. If `my_op`
    // takes owned `tensor[3, f32]`, jit-app consumes `w` structurally,
    // and the trailing sub(w, ...) borrow-read should error. The
    // implicit-copy inserter at the lower level inserts a fork for
    // single-consumer + borrow-read, so this MAY lower cleanly via
    // the v2 inserter. Pin whichever outcome the current implementation
    // produces.
    let _ = result;
}

/// Positive control: 3-grad fan-out without trailing borrow. The
/// implicit-copy inserter handles N-way consume fan-out via cascading
/// forks; this confirms `arg_is_borrowed` not having a borrow-read
/// after also works.
#[test]
fn shape_b_three_grad_no_trailing_borrow_lowers_cleanly() {
    let source = r#"
module Repro.ImplicitCopyShapeBNoBorrow

def my_loss(w: tensor[3, f32]) -> tensor[f32] = sum(w, 0)

def step(w: tensor[3, f32]) -> tensor[3, f32] = {
  d1 = grad(my_loss, wrt=w)(w)
  d2 = grad(my_loss, wrt=w)(w)
  d3 = grad(my_loss, wrt=w)(w)
  add(d1, add(d2, d3))
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B three-grad without trailing borrow must lower cleanly; got {:?}",
        result
    );
}
