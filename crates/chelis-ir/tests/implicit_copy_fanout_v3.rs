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

use chelis_ir::dag::Dag;
use chelis_ir::lower::try_lower_program;
use chelis_types::{check_linearity, check_typed_program};

/// Load the original unexpanded frontend AST, preserving legacy carriers and metadata.
fn fixture_to_dag(source: &str) -> Result<Dag, String> {
    let deep: Vec<chelis_deep::Expr> =
        serde_json::from_str(source).map_err(|e| format!("fixture JSON: {e}"))?;
    let checked = check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked =
        chelis_effects::check_program(&checked, &mut chelis_types::NativeServices::new(None))
            .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked = check_linearity(&checked, &mut chelis_types::NativeServices::new(None))
        .map_err(|errs| format!("linearity failed: {errs:?}"))?;
    chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        try_lower_program(&checked, runtime)
    })
    .map_err(|diag| format!("lowering failed: {diag:?}"))
}

#[test]
fn shape_a_borrow_return_position_lowers_cleanly() {
    // Shape A: body is a bare `(var x)` whose type is `&tensor[a, f32]`,
    // and the declared return type is the owned `tensor[a, f32]`. The
    // user expects the compiler to insert an implicit copy at the return
    // position so the function value is owned, matching the declared
    // signature.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_v3/shape_a_borrow_return_position_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
fn shape_a_borrow_return_with_use_site_lowers_cleanly() {
    // Shape A with a downstream caller that consumes the returned owned
    // tensor. Confirms the inserted copy participates as a normal owned
    // value at the call site.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_v3/shape_a_borrow_return_with_use_site_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A use-site composition must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
fn shape_b_grad_fanout_with_trailing_borrow_lowers_cleanly() {
    // Shape B (minimal): two grad calls of the same loss, followed by a
    // borrow-read of the same arg. Mirrors the hello-chelis linreg.ch
    // `sgd_step` shape that today requires explicit `copy(w)` /
    // `copy(b)` workaround wrappers at the grad call sites.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_v3/shape_b_grad_fanout_with_trailing_borrow_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B grad-fan-out with trailing borrow must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
fn shape_b_grad_fanout_four_arg_mse_shape_lowers_cleanly() {
    // Shape B (4-arg mse-shape): mirrors the hello-chelis linreg.ch
    // sgd_step pattern at vector arity 3.  Four-arg mse_loss, two grad
    // calls fanning out every arg, followed by a borrow-read of `w` and
    // `b` after both grad calls.  The reduction in mse_loss uses a
    // single sum(..., axis) so the test isolates the implicit-copy
    // fan-out behavior without depending on the nested-sum lowering
    // path used by the larger linreg fixture.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_v3/shape_b_grad_fanout_four_arg_mse_shape_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B 4-arg mse-shape must lower cleanly after the v3 fix; got {:?}",
        result
    );
}

#[test]
fn shape_b_vmap_call_with_trailing_borrow_lowers_cleanly() {
    // Sibling: `vmap(f)(args)` shares grad's observational semantics in
    // the linearity checker per the v3 fix.  This fixture pins that the
    // arg-is-borrowed promotion also covers vmap-app, so a vmap-app
    // followed by a borrow-read of the same arg lowers cleanly.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_v3/shape_b_vmap_call_with_trailing_borrow_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B vmap fan-out must lower cleanly after the v3 fix; got {:?}",
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
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_v3/shape_b_single_grad_call_already_lowers_today_control.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B single-grad control should lower cleanly today and after the fix; got {:?}",
        result
    );
}
