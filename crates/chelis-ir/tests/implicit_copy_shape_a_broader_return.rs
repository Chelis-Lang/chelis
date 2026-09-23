//! Implicit-copy Shape A broader return: tail-position borrow-to-owned
//! coercion in `let`, `if`, and `match` bodies.

use chelis_ir::dag::Dag;
use chelis_ir::lower::try_lower_program;
use chelis_types::{check_linearity, check_typed_program};

/// Load the original unexpanded frontend AST, preserving legacy carriers and metadata.
fn fixture_to_dag(source: &str) -> Result<Dag, String> {
    let deep: Vec<chelis_deep::Expr> =
        serde_json::from_str(source).map_err(|e| format!("fixture JSON: {e}"))?;
    let checked = check_typed_program(&deep)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lowering failed: {diag:?}"))
}

#[test]
fn shape_a_broader_let_tail_return_lowers_cleanly() {
    // Body is a block `{ y = x; y }` whose tail expression resolves to
    // `&tensor[n, f32]` while the declared return is owned. The relaxed
    // retry must descend through the desugared `(let bind body)` chain
    // and recognise the tail var-ref before unifying.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_broader_let_tail_return_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A let-tail return must lower cleanly after the broader-return fix; got {:?}",
        result
    );
}

#[test]
fn shape_a_broader_if_tail_return_lowers_cleanly() {
    // Both `if` branches are `&tensor[n, f32]` borrow-refs. The relaxed
    // retry must descend into both branches and accept the conjunction.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_broader_if_tail_return_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A if-tail return must lower cleanly after the broader-return fix; got {:?}",
        result
    );
}

#[test]
fn shape_a_broader_match_tail_return_lowers_cleanly() {
    // Both match arms are `&tensor[n, f32]` borrow-refs. The relaxed
    // retry must descend into the arm bodies (deep `(arm pattern guard
    // body)` triples) and accept the conjunction.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_broader_match_tail_return_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A match-tail return must lower cleanly after the broader-return fix; got {:?}",
        result
    );
}

#[test]
fn shape_a_broader_nested_let_if_tail_return_lowers_cleanly() {
    // Nested form: a `let` whose body is itself an `if`. Both descent
    // arms must compose for the relaxed retry to fire.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_broader_nested_let_if_tail_return_lowers_cleanly.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A nested let-then-if tail return must lower cleanly after the broader-return fix; got {:?}",
        result
    );
}

#[test]
fn shape_a_bare_var_positive_control_still_lowers() {
    // Positive control mirroring PR #91's `shape_a_borrow_return_position_lowers_cleanly`:
    // the bare-var body shape continues to work after the broader-return
    // extension. The descent helper returns the bare-var case as a leaf,
    // so this exact shape is still covered.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_bare_var_positive_control_still_lowers.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_ok(),
        "Shape A bare-var positive control must continue to lower cleanly; got {:?}",
        result
    );
}

#[test]
fn shape_a_broader_negative_control_real_mismatch_still_errors() {
    // Negative control: body genuinely returns the wrong tensor (a
    // different precision), not a borrow-vs-owned mismatch. The
    // relaxed-retry must NOT mask this; the existing TypeMismatch error
    // must still fire.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_broader_negative_control_real_mismatch_still_errors.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_err(),
        "Shape A negative control must still surface TypeMismatch (precision mismatch \
         is unrelated to borrow-to-owned coercion); got Ok"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("TypeMismatch") || msg.contains("doesn't match declared signature"),
        "expected TypeMismatch diagnostic for genuine precision mismatch; got {msg}"
    );
}

#[test]
fn shape_a_broader_if_one_branch_genuinely_wrong_still_errors() {
    // Mixed negative control: one branch is the expected tail-var, the
    // other is a genuinely wrong-typed expression. The descent must
    // reject when the branches disagree, so the existing TypeMismatch
    // surfaces unchanged.
    let source = include_str!(
        "fixtures/core/tests/implicit_copy_shape_a_broader_return/shape_a_broader_if_one_branch_genuinely_wrong_still_errors.json"
    );
    let result = fixture_to_dag(source);
    assert!(
        result.is_err(),
        "Shape A if-tail with mismatched branches must still surface TypeMismatch; got Ok"
    );
}
