//! Wave 3 terminal red team for the 0.7.8 compiler-cleanup workstream.
//!
//! Adversarial fixtures probing the linearity §5 closures shipped in
//! PRs #83 and #90: typed `ConsumeKind`, multi-level alias chains,
//! and tuple-destructure linearity (single-level + nested).
//!
//! Each fixture has a pinned expected outcome; see the surrounding
//! doc comment for the shape under test.

use chelis_deep::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("Deep fixture parse should succeed");
    let deep = decls;
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

fn assert_linearity_clean(source: &str) {
    let decls = parse_str(source).expect("Deep fixture parse should succeed");
    let deep = decls;
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect("linearity check should not error");
}

// ============================================================
// §3.2 Linearity-AliasedConsume-F1 multi-level alias chains
// ============================================================

/// Three-level alias chain: `let z = y; let y_x = x; let w = z`.
/// Then `realize(w)` consumes the underlying `x` value through three
/// chain hops, so a later `add(x, w)` must error.
///
/// This pushes past the existing two-level fixture in
/// `linearity_aliased_consume.rs::aliased_consume_bypass_errors_after_fix`
/// to confirm `resolve_alias_chain` walks past depth 1.
#[test]
fn alias_chain_three_levels_propagates_consume() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/alias_chain_three_levels_propagates_consume_43.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `x`")
        }),
        "expected UseAfterConsume on `x` after consume traverses three alias links; got {errors:?}"
    );
}

/// Four-level alias chain. Stress the chain walk one step further.
#[test]
fn alias_chain_four_levels_propagates_consume() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/alias_chain_four_levels_propagates_consume_65.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `x`")
        }),
        "expected UseAfterConsume on `x` after consume traverses four alias links; got {errors:?}"
    );
}

/// Aliased read alone is still legal — chain consume on an unused
/// chain tail should not error.
#[test]
fn alias_chain_four_levels_consume_at_tail_only_passes() {
    assert_linearity_clean(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/alias_chain_four_levels_consume_at_tail_only_passes_90.deep"
    ));
}

// ============================================================
// §3.2 Match-scrutinee + alias (Structural kind under match shape)
// ============================================================

/// Pipe-stage alias: `let y = x; y |> realize`. Pipe stage is one of
/// the eight Structural producer sites in the linearity refactor.
/// The alias must forward through the pipe-stage consume so a later
/// `add(x, ...)` errors.
///
/// Per the Phase 0 design doc §"Producer-site assignment table",
/// `pipe_site` is `Structural` and should forward through the alias
/// map. This fixture confirms the contract.
#[test]
fn alias_then_pipe_stage_consume_propagates() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/alias_then_pipe_stage_consume_propagates_118.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `x`")
        }),
        "expected UseAfterConsume on `x` consumed via pipe-stage on alias `y`; got {errors:?}"
    );
}

/// App-arg alias: `let y = x; realize_via_app(y); add(x, ...)`. Same
/// shape as pipe but the alias is consumed as an app-arg.
#[test]
fn alias_then_app_arg_consume_propagates() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/alias_then_app_arg_consume_propagates_140.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `x`")
        }),
        "expected UseAfterConsume on `x` consumed via app-arg on alias `y`; got {errors:?}"
    );
}

// ============================================================
// §3.3 Linearity-F2 nested destructure
// ============================================================

/// Nested tuple destructure: `let (inner, c) = pair; let (a, b) =
/// inner; realize(a); realize(a)` should error on double consume of
/// the inner-tuple component `a`.
///
/// Pin: this is the central Linearity-F2 generalization. PR #83's
/// fixture only covers single-level destructure; the diagnosis cites
/// `tuple_get_element_type` as recursive — verify behavior.
#[test]
fn nested_tuple_destructure_double_realize_errors() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/nested_tuple_destructure_double_realize_errors_171.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "expected UseAfterConsume on `a` (component of nested destructure) after double realize; \
         got {errors:?}"
    );
}

/// Nested destructure, single-consume-each positive control. Should
/// pass: every leaf component is realized exactly once.
#[test]
fn nested_tuple_destructure_single_consume_each_passes() {
    assert_linearity_clean(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/nested_tuple_destructure_single_consume_each_passes_195.deep"
    ));
}

/// Triple-nested destructure (3-level) — error on duplicate consume of
/// innermost component.
#[test]
fn triple_nested_destructure_double_consume_errors() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/triple_nested_destructure_double_consume_errors_213.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "expected UseAfterConsume on innermost `a` of triple-nested destructure; got {errors:?}"
    );
}

// ============================================================
// §3.3 Mixed nested destructure + alias chain
// ============================================================

/// Nested destructure then alias then consume. After destructuring,
/// aliasing the inner component and consuming the alias should
/// propagate to the inner component.
#[test]
fn nested_destructure_alias_consume_errors() {
    let errors = linearity_errors(include_str!(
        "fixtures/core/tests/linearity_alias_destructure_adversarial/nested_destructure_alias_consume_errors_242.deep"
    ));
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "expected UseAfterConsume on destructured `a` via alias `y`; got {errors:?}"
    );
}
