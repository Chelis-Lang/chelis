//! Negative-axis normalization for axis-taking ops in IR lowering, and
//! the rank-0 standalone-parameter regression that shared the same
//! downstream-blocker symptom.
//!
//! Two bugs are pinned here:
//!
//! Bug B (negative axes): `softmax(t, -1)` and the reduction family
//! used `*n as usize` in `extract_axis`, so a negative axis literal
//! like `-1` became `usize::MAX`. `tier2::lower_softmax` /
//! `lower_*reduce` then indexed past the operand rank and `require_dim`
//! panicked. The fix normalizes `-1` to `rank - 1` (the last axis),
//! uniformly across softmax / mean / sum / max_reduce / min_reduce /
//! prod_reduce / argmax_reduce / argmin_reduce / gather / scatter, and
//! turns a still-out-of-range axis into a clean lowering diagnostic
//! rather than a panic. The type checker normalizes the same way so
//! checker and lowering agree.
//!
//! Bug A (rank-0 standalone params): a top-level def whose parameter
//! types come from a separate `sig` declaration desugars to bare,
//! untyped `fn` params. Standalone lowering then bound them to a
//! rank-0 `default_type()` Load, so any shape-sensitive op on such a
//! param hit the same `require_dim` panic -- even with a non-negative
//! axis. The fix has the type checker stamp the declared signature's
//! parameter type expressions onto the `(params ...)` node.
//!
//! Oracle: both `packages/chelis-std/src/loss/crossentropy.ch` (Bug A,
//! positive axis on a symbolic-dim param) and
//! `packages/chelis-std/src/nn/attention.ch` (Bug B, `softmax(_, -1)`)
//! must lower without panicking.

use chelis_unord::UnordMap;

use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::lower::{lower_program, try_lower_subexpr_program};
use chelis_ir::verify;
use chelis_types::check_ir_program;
use chelis_types::types::Prim;

/// Surf source -> desugar -> macro-expand -> typecheck -> effects ->
/// linearity -> lower. Returns the lowered DAG, or a stage-tagged error
/// string. Mirrors the `lower_surf` helper in `reduce_sum_lowering_adversarial`.
fn lower_surf(src: &str) -> Result<chelis_ir::dag::Dag, String> {
    let decls = chelis_surf::parser::parse_str(src).map_err(|e| format!("parse: {e:?}"))?;
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .map_err(|e| format!("expand: {e:?}"))?
    .into_exprs();
    let checked = check_ir_program(&exprs).map_err(|r| {
        format!(
            "check: {:?}",
            r.errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
        )
    })?;
    let checked = chelis_effects::check_program(&checked).map_err(|e| format!("effects: {e:?}"))?;
    let checked =
        chelis_types::check_linearity(&checked).map_err(|e| format!("linearity: {e:?}"))?;
    Ok(lower_program(&checked))
}

/// Just the typecheck stage, surfacing the error messages. Used for the
/// checker-side negative-axis assertions so checker and lowering can be
/// pinned to agree.
fn check_surf(src: &str) -> Result<(), Vec<String>> {
    let decls = chelis_surf::parser::parse_str(src).expect("parse failed");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("expand failed")
    .into_exprs();
    check_ir_program(&exprs)
        .map(|_| ())
        .map_err(|r| r.errors.iter().map(|e| e.message.clone()).collect())
}

// ---------------------------------------------------------------------
// Bug B: negative axes lower and evaluate -- softmax + every reduction.
// Each op is exercised with `-1` (last axis) on a rank-2 operand; the
// positive-axis control proves the negative form is not just silently
// degrading to axis 0.
// ---------------------------------------------------------------------

/// softmax over the last axis via `-1` lowers cleanly and the lowered
/// IR passes verify (no `require_dim` panic, no malformed node).
#[test]
fn softmax_negative_last_axis_lowers_and_verifies() {
    let src = r#"
sig run: tensor[2, 3, f32] -> tensor[2, 3, f32]
def run(x) = softmax(x, -1)
"#;
    let dag = lower_surf(src).expect("softmax(x, -1) must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered softmax(_, -1) IR must verify clean"
    );
}

/// Positive-axis control: `softmax(x, 1)` on the same rank-2 operand is
/// the explicit form of `-1` and must lower identically clean.
#[test]
fn softmax_positive_axis_control_lowers_and_verifies() {
    let src = r#"
sig run: tensor[2, 3, f32] -> tensor[2, 3, f32]
def run(x) = softmax(x, 1)
"#;
    let dag = lower_surf(src).expect("softmax(x, 1) must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered softmax(_, 1) IR must verify clean"
    );
}

/// Every reduction op accepts `-1` (last axis) and lowers clean. The
/// reductions previously went through `check_reduction_signature`,
/// which rejected negative axes outright with `requires non-negative
/// axis`; that rejection was the checker-side half of the same bug.
///
/// Per issue #230, `argmax_reduce` / `argmin_reduce` produce `i64`
/// indices regardless of input dtype, matching the std-package
/// signature `tensor[a, b, p] -> i32 -> tensor[b, i64]`. The other
/// reductions preserve the input dtype.
#[test]
fn reductions_negative_last_axis_lower_and_verify() {
    for (op, out_dtype) in [
        ("sum", "f32"),
        ("mean", "f32"),
        ("max_reduce", "f32"),
        ("min_reduce", "f32"),
        ("prod_reduce", "f32"),
        ("argmax_reduce", "i64"),
        ("argmin_reduce", "i64"),
    ] {
        let src = format!(
            r#"
sig run: tensor[2, 3, f32] -> tensor[2, {out_dtype}]
def run(x) = {op}(x, -1)
"#
        );
        let dag = lower_surf(&src).unwrap_or_else(|e| panic!("{op}(x, -1) must lower: {e}"));
        assert!(
            verify::verify(&dag).is_empty(),
            "lowered {op}(_, -1) IR must verify clean"
        );
    }
}

/// Positive-axis control for the reductions: `op(x, 1)` is the explicit
/// last axis and must lower identically clean.
#[test]
fn reductions_positive_axis_control_lower_and_verify() {
    for (op, out_dtype) in [
        ("sum", "f32"),
        ("mean", "f32"),
        ("max_reduce", "f32"),
        ("min_reduce", "f32"),
        ("prod_reduce", "f32"),
        ("argmax_reduce", "i64"),
        ("argmin_reduce", "i64"),
    ] {
        let src = format!(
            r#"
sig run: tensor[2, 3, f32] -> tensor[2, {out_dtype}]
def run(x) = {op}(x, 1)
"#
        );
        let dag = lower_surf(&src).unwrap_or_else(|e| panic!("{op}(x, 1) must lower: {e}"));
        assert!(
            verify::verify(&dag).is_empty(),
            "lowered {op}(_, 1) IR must verify clean"
        );
    }
}

/// `gather` with a negative axis lowers clean. gather already
/// normalized negative axes in the type checker via
/// `normalize_static_axis`; this pins that IR lowering now agrees
/// instead of mapping `-1` to `usize::MAX`.
#[test]
fn gather_negative_axis_lowers_and_verifies() {
    let src = r#"
sig run: tensor[2, 3, f32] -> tensor[2, i64] -> tensor[2, 2, f32]
def run(values, idx) = gather(values, idx, -1)
"#;
    let dag = lower_surf(src).expect("gather(values, idx, -1) must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered gather(_, _, -1) IR must verify clean"
    );
}

// ---------------------------------------------------------------------
// Bug B negative parity: an axis still out of range after normalization
// is a clean typed error, NOT a panic. Both the type checker and IR
// lowering must reject it; `require_dim` must stay unreachable.
// ---------------------------------------------------------------------

/// A too-negative axis (`-3` on a rank-2 operand normalizes to `-1`,
/// still out of `0..2`) is rejected by the type checker, not panicked.
#[test]
fn softmax_axis_too_negative_is_a_checker_error() {
    let src = r#"
sig run: tensor[2, 3, f32] -> tensor[2, 3, f32]
def run(x) = softmax(x, -3)
"#;
    let errors = check_surf(src).expect_err("softmax(x, -3) must be a type error");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("softmax") && e.contains("out of bounds")),
        "expected an out-of-bounds softmax axis diagnostic, got {errors:?}"
    );
}

/// A too-large positive axis is likewise a clean checker error.
#[test]
fn reduction_axis_too_large_is_a_checker_error() {
    let src = r#"
sig run: tensor[2, 3, f32] -> tensor[2, f32]
def run(x) = sum(x, 5)
"#;
    let errors = check_surf(src).expect_err("sum(x, 5) must be a type error");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("sum") && e.contains("out of bounds")),
        "expected an out-of-bounds sum axis diagnostic, got {errors:?}"
    );
}

/// If a malformed axis reaches IR lowering directly -- the
/// integration-test corpus synthesizes ad-hoc Deep that bypasses the
/// checker -- lowering must surface a `LowerDiagnostic`, never panic
/// through `require_dim`. This pins the dispatch's hard requirement
/// directly on the lowering boundary, independent of the checker's
/// own (now also present) axis validation.
#[test]
fn lowering_rejects_out_of_range_axis_without_panicking() {
    // `(app softmax (var x) (lit 9))` with `x` scoped as a rank-2
    // tensor: axis 9 is out of `0..2`. `try_lower_subexpr_program`
    // lowers a bare expr with `scoped_tensor_types` and no checker in
    // the loop, so this exercises `normalize_axis` at the lowering
    // boundary directly.
    let deep_src = r#"
(app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
  (var {} softmax)
  (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
  (lit {type: (t-prim {} i32)} 9))
"#;
    let expr = chelis_deep::parser::parse_str(deep_src)
        .expect("parse failed")
        .into_iter()
        .next()
        .expect("one expr");
    let scoped = UnordMap::from([(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        },
    )]);
    // `try_lower_subexpr_program` returns a `LowerDiagnostic` instead
    // of panicking when lowering rejects a form. An out-of-range
    // softmax axis must take that path -- it must not unwind through
    // `require_dim`.
    let result = try_lower_subexpr_program(&expr, scoped, UnordMap::new(), UnordMap::new());
    let diagnostic = result.expect_err("out-of-range softmax axis must be a lowering diagnostic");
    let rendered = diagnostic.to_string();
    assert!(
        rendered.contains("softmax") && rendered.contains("out of range"),
        "expected a clean out-of-range softmax axis diagnostic, got `{rendered}`"
    );
}

// ---------------------------------------------------------------------
// Bug A: a def whose parameter types come from a separate `sig` is
// lowered standalone without the params collapsing to rank-0. The
// shape-sensitive op on the symbolic-dim borrowed param must lower
// clean -- this is the exact shape of `crossentropy.ch`'s `loss`.
// ---------------------------------------------------------------------

/// Regression for Bug A: `softmax(logits, 1)` on a borrowed,
/// symbolic-dim parameter whose type comes from a separate `sig` must
/// lower without the `require_dim` panic. Before the fix the param
/// bound to a rank-0 `default_type()` and `dims.get(1)` was `None`.
#[test]
fn standalone_def_with_separate_sig_lowers_shape_sensitive_param() {
    let src = r#"
sig run: &tensor[a, b, f32] -> tensor[a, b, f32]
def run(logits) = softmax(logits, 1)
"#;
    let dag =
        lower_surf(src).expect("softmax on a separate-sig symbolic-dim param must lower (Bug A)");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered standalone-def IR must verify clean"
    );
}

/// Bug A is independent of axis sign: the same separate-sig borrowed
/// param with a negative axis must also lower clean (Bug A x Bug B).
#[test]
fn standalone_def_with_separate_sig_lowers_negative_axis_param() {
    let src = r#"
sig run: &tensor[a, b, f32] -> tensor[a, b, f32]
def run(logits) = softmax(logits, -1)
"#;
    let dag =
        lower_surf(src).expect("softmax(_, -1) on a separate-sig symbolic-dim param must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered standalone-def IR with negative axis must verify clean"
    );
}

/// The borrowed param must stay borrowed: a separate-sig `&tensor`
/// parameter read twice in the body must still type/linearity-check.
/// This pins that the Bug A fix stamps the declared `&` (t-ref) form
/// onto the params node, not a bare owned tensor type -- otherwise the
/// linearity checker would see the second read as use-after-consume.
#[test]
fn standalone_def_borrowed_param_stays_borrowed_under_fanout() {
    let src = r#"
sig run: &tensor[2, 3, f32] -> tensor[2, 3, f32]
def run(x) = add(softmax(x, -1), softmax(x, 0))
"#;
    let dag = lower_surf(src)
        .expect("two reads of a borrowed separate-sig param must lower (borrow preserved)");
    assert!(
        verify::verify(&dag).is_empty(),
        "borrowed-param fan-out IR must verify clean"
    );
}
