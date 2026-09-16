//! Implicit-copy Shape A broader return: tail-position borrow-to-owned
//! coercion in `let`, `if`, and `match` bodies.
//!
//! PR #91 (W4-A) introduced `shape_a_relaxed_return` in
//! `crates/chelis-types/src/infer.rs` to handle the bare-var body shape:
//!
//! ```ignore
//! def f(x: &T) -> T = x
//! ```
//!
//! Tail-position returns nested inside `let`, `if`, or `match` still fail
//! with `TypeMismatch` because the relaxed-retry gate only fires when the
//! inferred body's inner expression is a bare `(var ...)`. The §5 entry
//! `Linearity-ShapeABroadReturn-F1` tracks the broader coverage. This
//! file pins the gap and the post-fix expectations.
//!
//! See `docs/investigations/implicit_copy_shape_a_broader_return.md` for
//! the diagnosis and chosen fix.

use chelis_ir::dag::Dag;
use chelis_ir::lower::try_lower_program;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::{check_linearity, check_typed_program};

/// Full Surf-to-DAG pipeline. Mirrors `implicit_copy_fanout_v3::surf_to_dag`.
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
fn shape_a_broader_let_tail_return_lowers_cleanly() {
    // Body is a block `{ y = x; y }` whose tail expression resolves to
    // `&tensor[n, f32]` while the declared return is owned. The relaxed
    // retry must descend through the desugared `(let bind body)` chain
    // and recognise the tail var-ref before unifying.
    let source = r#"
module Repro.ShapeABroaderLet

def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = {
  y = x
  y
}
"#;
    let result = surf_to_dag(source);
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
    let source = r#"
module Repro.ShapeABroaderIf

def g[n](c: bool, x: &tensor[n, f32]) -> tensor[n, f32] = if c then x else x
"#;
    let result = surf_to_dag(source);
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
    let source = r#"
module Repro.ShapeABroaderMatch

type Choice =
  | Left
  | Right

def h[n](c: Choice, x: &tensor[n, f32]) -> tensor[n, f32] = match c with {
  | Left => x
  | Right => x
}
"#;
    let result = surf_to_dag(source);
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
    let source = r#"
module Repro.ShapeABroaderNested

def k[n](c: bool, x: &tensor[n, f32]) -> tensor[n, f32] = {
  y = x
  if c then y else y
}
"#;
    let result = surf_to_dag(source);
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
    let source = r#"
module Repro.ShapeABareVarControl

def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x
"#;
    let result = surf_to_dag(source);
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
    let source = r#"
module Repro.ShapeABroaderNegative

def bad[n](x: &tensor[n, f32], y: &tensor[n, i32]) -> tensor[n, f32] = {
  z = y
  z
}
"#;
    let result = surf_to_dag(source);
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
    let source = r#"
module Repro.ShapeABroaderIfMixed

def bad[n](c: bool, x: &tensor[n, f32], y: &tensor[n, i32]) -> tensor[n, f32] = if c then x else y
"#;
    let result = surf_to_dag(source);
    assert!(
        result.is_err(),
        "Shape A if-tail with mismatched branches must still surface TypeMismatch; got Ok"
    );
}
