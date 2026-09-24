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

use chelis_types::{check_linearity, check_typed_program};

/// Load the original unexpanded frontend AST, preserving legacy carriers and metadata.
fn fixture_to_dag(source: &str) -> Result<Dag, String> {
    let deep: Vec<chelis_deep::Expr> =
        serde_json::from_str(source).map_err(|e| format!("fixture JSON: {e}"))?;
    let checked = check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        try_lower_program(&checked, runtime)
    })
    .map_err(|diag| format!("lowering failed: {diag:?}"))
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
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_shape_a_adversarial/shape_a_let_tail_return_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
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
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_shape_a_adversarial/shape_a_if_tail_return_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
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
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_shape_a_adversarial/shape_b_three_grad_fanout_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
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
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_shape_a_adversarial/shape_b_mixed_grad_vmap_fanout_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B mixed grad + vmap fan-out must lower cleanly; got {:?}",
        result
    );
}

/// Positive control: 3-grad fan-out without trailing borrow. The
/// implicit-copy inserter handles N-way consume fan-out via cascading
/// forks; this confirms `arg_is_borrowed` not having a borrow-read
/// after also works.
#[test]
fn shape_b_three_grad_no_trailing_borrow_lowers_cleanly() {
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_fanout_shape_a_adversarial/shape_b_three_grad_no_trailing_borrow_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape B three-grad without trailing borrow must lower cleanly; got {:?}",
        result
    );
}
