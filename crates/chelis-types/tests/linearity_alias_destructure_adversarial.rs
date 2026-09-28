//! Multi-level alias chains and tuple-destructure components obey
//! the same consume and borrow rules as direct bindings.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

fn assert_linearity_clean(source: &str) {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect("linearity check should not error");
}

// ============================================================
// Multi-level alias chains
// ============================================================

/// Three-level alias chain: `let z = y; let y_x = x; let w = z`.
/// Then `realize(w)` consumes the underlying `x` value through three
/// chain hops, so a later `add(x, w)` must error.
///
/// `resolve_alias_chain` must follow every link to the source.
#[test]
fn alias_chain_three_levels_propagates_consume() {
    let errors = linearity_errors(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = x
    z: tensor[4, f32] = y
    r: tensor[4, f32] = realize(z)
    add(x, r)
  }
"#,
    );
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
    let errors = linearity_errors(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    a: tensor[4, f32] = x
    b: tensor[4, f32] = a
    c: tensor[4, f32] = b
    d: tensor[4, f32] = c
    r: tensor[4, f32] = realize(d)
    add(x, r)
  }
"#,
    );
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
    assert_linearity_clean(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    a: tensor[4, f32] = x
    b: tensor[4, f32] = a
    c: tensor[4, f32] = b
    d: tensor[4, f32] = c
    realize(d)
  }
"#,
    );
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
    let errors = linearity_errors(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = x
    r: tensor[4, f32] = y |> realize
    add(x, r)
  }
"#,
    );
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
    let errors = linearity_errors(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = x
    r: tensor[4, f32] = realize(y)
    add(x, r)
  }
"#,
    );
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
/// This exercises recursive component tracking across two levels.
#[test]
fn nested_tuple_destructure_double_realize_errors() {
    let errors = linearity_errors(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> tensor[4, f32] =
  {
    (inner, c) = p
    (a, b) = inner
    r1: tensor[4, f32] = realize(a)
    realize(a)
  }
"#,
    );
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
    assert_linearity_clean(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> tensor[4, f32] =
  {
    (inner, c) = p
    (a, b) = inner
    r1: tensor[4, f32] = realize(a)
    r2: tensor[4, f32] = realize(b)
    add(r1, r2)
  }
"#,
    );
}

/// Triple-nested destructure (3-level) — error on duplicate consume of
/// innermost component.
#[test]
fn triple_nested_destructure_double_consume_errors() {
    let errors = linearity_errors(
        r#"
def f(p: (((tensor[4, f32], tensor[4, f32]), tensor[4, f32]), tensor[4, f32])) -> tensor[4, f32] =
  {
    (mid, w) = p
    (inner, c) = mid
    (a, b) = inner
    r1: tensor[4, f32] = realize(a)
    realize(a)
  }
"#,
    );
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
    let errors = linearity_errors(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> tensor[4, f32] =
  {
    (inner, c) = p
    (a, b) = inner
    y: tensor[4, f32] = a
    r: tensor[4, f32] = realize(y)
    add(a, r)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "expected UseAfterConsume on destructured `a` via alias `y`; got {errors:?}"
    );
}
