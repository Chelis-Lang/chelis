//! Implicit-copy fan-out v3: two remaining shapes after PR #29 (v1) and
//! PR #60 (V2-F4).
//!
//! Shape A: borrow-to-owned at return position.
//! `def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x` fails
//! today with a type mismatch ("def 'identity_dim' body doesn't match
//! declared signature"). The body's inferred type is `&tensor[a, f32]`
//! and the declared return is owned `tensor[a, f32]`. The existing
//! auto-borrow path at `chelis_types::infer::auto_borrow_call_arg_types`
//! coerces owned->borrow at argument positions only. The mirror coercion
//! at return position (body borrow -> owned via inserted copy) was
//! missing.
//!
//! Shape B: fan-out across `grad(f, wrt=p)(args)` call sites with a
//! later borrow-read of the same args. The hello-chelis `linreg.ch`
//! shape is the source repro: two grad calls of the same loss followed
//! by a `sub(w, ...)` borrow-read of the parameter. Today fails with
//! `UseAfterConsume` because the grad-app site is treated as a
//! consuming structural use of every arg (matching the inferred fn
//! signature), and the third borrow-read trips `read_or_error`. The
//! grad-app is observational at the linearity level: lowering computes
//! both forward and backward DAGs from synthesized loads, so the call
//! site does not actually destroy its arguments. Treating
//! `grad(f, ...)`-app args as borrows fixes Shape B at the linearity
//! level.
//!
//! Both fixtures gated `#[ignore]` until the fix lands.
//!
//! See `docs/investigations/implicit_copy_fanout_v3_diagnosis.md` for
//! the diagnosis and chosen fix sites.

use chelis_ir::dag::Dag;
use chelis_ir::lower::try_lower_program;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::{check_linearity, check_typed_program};

/// Full Surf-to-DAG pipeline. Mirrors `cross_statement_fanout::surf_to_dag`.
fn surf_to_dag(source: &str) -> Result<Dag, String> {
    let decls = surf_parse(source).map_err(|e| format!("surf parse: {e:?}"))?;
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lowering failed: {diag:?}"))
}

#[test]
#[ignore = "implicit-copy fan-out v3 Shape A, blocked on return-position borrow->owned coercion"]
fn shape_a_borrow_return_position_lowers_cleanly() {
    // Shape A: body is a bare `(var x)` whose type is `&tensor[a, f32]`,
    // and the declared return type is the owned `tensor[a, f32]`. The
    // user expects the compiler to insert an implicit copy at the return
    // position so the function value is owned, matching the declared
    // signature.
    let source = r#"
module Repro.ImplicitCopyShapeA

def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
#[ignore = "implicit-copy fan-out v3 Shape A use-site, blocked on return-position borrow->owned coercion"]
fn shape_a_borrow_return_with_use_site_lowers_cleanly() {
    // Shape A with a downstream caller that consumes the returned owned
    // tensor. Confirms the inserted copy participates as a normal owned
    // value at the call site.
    let source = r#"
module Repro.ImplicitCopyShapeAUse

def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x

def driver(x: tensor[3, f32]) -> tensor[3, f32] = {
  y = identity_dim(&x)
  add(y, y)
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A use-site composition must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
#[ignore = "implicit-copy fan-out v3 Shape B, blocked on grad-call observational arg handling"]
fn shape_b_grad_fanout_with_trailing_borrow_lowers_cleanly() {
    // Shape B (minimal): two grad calls of the same loss, followed by a
    // borrow-read of the same arg. Mirrors the hello-chelis linreg.ch
    // `sgd_step` shape that today requires explicit `copy(w)` /
    // `copy(b)` workaround wrappers at the grad call sites.
    let source = r#"
module Repro.ImplicitCopyShapeB

def my_loss(w: tensor[3, f32], b: tensor[3, f32]) -> tensor[f32] = {
  d = sub(w, b)
  sq = mul(d, d)
  sum(sq, 0)
}

def step(w: tensor[3, f32], b: tensor[3, f32]) -> tensor[3, f32] = {
  dw = grad(my_loss, wrt=w)(w, b)
  db = grad(my_loss, wrt=b)(w, b)
  trailing = sub(w, dw)
  add(trailing, db)
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B grad-fan-out with trailing borrow must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
#[ignore = "implicit-copy fan-out v3 Shape B linreg pattern, blocked on grad-call observational arg handling"]
fn shape_b_grad_fanout_linreg_pattern_lowers_cleanly() {
    // Shape B (linreg.ch shape): the precise hello-chelis pattern that
    // surfaced this bug. Four-arg mse_loss, two grad calls fanning out
    // every arg, followed by a borrow-read of `w` and `b` for the
    // SGD parameter update.
    let source = r#"
module Repro.ImplicitCopyShapeBLinReg

def predict(x: tensor[64, 64, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[64, 1, f32] = {
  m = matmul(x, w)
  e = expand(b, 0, 64)
  add(m, e)
}

def mse_loss(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[f32] = {
  pred = predict(x, w, b)
  err = sub(pred, y)
  err_copy = err
  sum(sum(mul(err, err_copy), 1), 0)
}

def sgd_step(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32], lr: f32) -> (tensor[64, 1, f32], tensor[1, f32]) = {
  dw = grad(mse_loss, wrt=w)(x, y, w, b)
  db = grad(mse_loss, wrt=b)(x, y, w, b)
  lr_t = to_tensor([lr])
  new_w = sub(w, mul(expand(expand(lr_t, 0, 64), 1, 1), dw))
  new_b = sub(b, mul(lr_t, db))
  (new_w, new_b)
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B linreg pattern must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
fn shape_b_single_grad_call_already_lowers_today_control() {
    // Positive control: a single grad-call followed by a borrow-read of
    // the same arg. This works today because there is only one consuming
    // use of `w` at the grad-app site and the borrow-read happens after
    // a single Structural consume which (per spec) the implicit-copy
    // pass would fork.  This fixture pins that the v3 fix does not
    // regress this already-working case.
    let source = r#"
module Repro.ImplicitCopyShapeBControl

def my_loss(w: tensor[3, f32], b: tensor[3, f32]) -> tensor[f32] = {
  d = sub(w, b)
  sq = mul(d, d)
  sum(sq, 0)
}

def step(w: tensor[3, f32], b: tensor[3, f32]) -> tensor[3, f32] = {
  dw = grad(my_loss, wrt=w)(copy(w), copy(b))
  trailing = sub(w, dw)
  add(trailing, b)
}
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B single-grad control should lower cleanly today and after the fix; got {:?}",
        result
    );
}
