//! Chelis-Lang/chelis#1200 — the Linearity-F2 gate is per-name, not
//! per-block.
//!
//! # Diagnosis
//!
//! `chelis_surf::desugar::desugar_let_bindings` routes every `let` whose
//! pattern is not a bare `Var` — including a bare `_` discard — through
//! `bind_destructure_value`, which stamps `destructure: true` on the
//! synthesized bind. Before this fix `Checker::check_let` turned that
//! marker into a `destructure_scope_depth` bump around the let *body*.
//! In a block, every later statement is nested inside that body, so the
//! depth stayed non-zero for the remainder of the block and the
//! `consume_var_expr` already-consumed arm fired Linearity-F2 for ANY
//! variable — including ordinary bindings and the destructure's own
//! source — instead of falling through to implicit Copy insertion.
//!
//! Downstream cost: coral's suite went 74 passed / 0 failed on 0.18.3 to
//! 17 passed / 7 failed on 0.18.4 purely from this, because coral's test
//! bodies open with a run of `_ = assert_*(...)` discards.
//!
//! The fix marks the names a `destructure: true` bind introduces
//! (`LinearScope::mark_destructured`) and gates the F2 arm on membership
//! of the consumed target. The diagnostic text is unchanged; under the
//! new guard its "(from a destructured binding)" claim is accurate.
//!
//! The three block-poisoning shapes below (B, C, D) all compiled on
//! 0.18.3, failed on 0.18.4, and must compile again. The true positive
//! F2 protects — reuse of a destructured component without `copy()` —
//! must keep failing.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

fn assert_linearity_clean(source: &str) {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    if let Err(errors) = check_linearity(&checked) {
        panic!("linearity check should not error; got {errors:?}");
    }
}

// ============================================================
// Shape B — `_ = eat(v)` then `eat(v)` (the filed shape)
// ============================================================

/// A bare `_` discard binds nothing the user can name, yet its
/// synthesized temp carries `destructure: true`. A later
/// consume-after-consume on the discarded expression's own source must
/// route to implicit Copy insertion, exactly as it does without the
/// discard.
#[test]
fn wildcard_discard_does_not_poison_its_own_source() {
    assert_linearity_clean(
        r#"
def f(v: tensor[4, f32]) -> tensor[4, f32] = {
  _ = realize(v)
  realize(v)
}
"#,
    );
}

/// Control: the same double consume without the discard was always
/// accepted. Pins that shape B's expected verdict is the no-discard
/// verdict, not a newly-invented leniency.
#[test]
fn double_consume_without_discard_is_the_same_verdict() {
    assert_linearity_clean(
        r#"
def f(v: tensor[4, f32]) -> tensor[4, f32] = {
  r: tensor[4, f32] = realize(v)
  realize(v)
}
"#,
    );
}

// ============================================================
// Shape C — a discard on an UNRELATED value poisons the block
// ============================================================

/// `p` and `q` are independent parameters. The discard names `q`; `p` is
/// an ordinary binding with no relationship to any destructure. Under the
/// scope-depth gate the `_ = realize(q)` statement made the second
/// `realize(p)` a hard error.
#[test]
fn discard_does_not_poison_an_unrelated_binding() {
    assert_linearity_clean(
        r#"
def f(p: tensor[4, f32], q: tensor[4, f32]) -> tensor[4, f32] = {
  r1: tensor[4, f32] = realize(p)
  _ = realize(q)
  r2: tensor[4, f32] = realize(p)
  add(r1, r2)
}
"#,
    );
}

/// Coral's actual shape: a run of leading discards followed by ordinary
/// work. Every statement after the first discard sat inside the poisoned
/// scope.
#[test]
fn a_run_of_discards_does_not_poison_the_rest_of_the_block() {
    assert_linearity_clean(
        r#"
def f(p: tensor[4, f32], q: tensor[4, f32], s: tensor[4, f32]) -> tensor[4, f32] = {
  _ = realize(q)
  _ = realize(s)
  r1: tensor[4, f32] = realize(p)
  r2: tensor[4, f32] = realize(p)
  add(r1, r2)
}
"#,
    );
}

// ============================================================
// Shape D — a fully NAMED tuple destructure poisons its own source
// ============================================================

/// No wildcard anywhere: `(a, b) = (realize(v), realize(w))` consumes `v`
/// while binding two named components, and the later `realize(v)` was
/// rejected. This is the proof that the wildcard was a symptom and the
/// block-scoped gate was the cause.
#[test]
fn named_tuple_destructure_does_not_poison_its_source() {
    assert_linearity_clean(
        r#"
def f(v: tensor[4, f32], w: tensor[4, f32]) -> tensor[4, f32] = {
  (a, b) = (realize(v), realize(w))
  r: tensor[4, f32] = realize(v)
  add(add(a, b), r)
}
"#,
    );
}

// ============================================================
// The true positive Linearity-F2 protects — must KEEP failing
// ============================================================

/// Reuse of a destructured component without `copy()` is still a hard
/// error, with the same diagnostic. Implicit Copy insertion does not
/// apply to a component: `tuple-get` yields a fresh owned value, not an
/// aliased borrow.
#[test]
fn component_reuse_without_copy_still_errors() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  r1: tensor[4, f32] = realize(a)
  realize(a)
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message
                    .contains("variable `a` (from a destructured binding)")
        }),
        "expected the Linearity-F2 destructured-component diagnostic on `a`; got {errors:?}"
    );
}

/// The escape hatch the diagnostic names still works.
#[test]
fn component_reuse_with_copy_compiles() {
    assert_linearity_clean(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  r1: tensor[4, f32] = realize(copy(a))
  realize(a)
}
"#,
    );
}

// ============================================================
// Wildcard inside a tuple pattern
// ============================================================

/// `(a, _) = ...` — the wildcard arm of `destructure_pattern` introduces
/// no user-visible name, but the enclosing tuple still marks components.
/// The destructure's SOURCE is not one of them.
#[test]
fn tuple_wildcard_does_not_poison_its_source() {
    assert_linearity_clean(
        r#"
def f(v: tensor[4, f32], w: tensor[4, f32]) -> tensor[4, f32] = {
  (a, _) = (realize(v), realize(w))
  r: tensor[4, f32] = realize(v)
  add(a, r)
}
"#,
    );
}

/// Same pattern, but now the named component `a` is consumed twice: still
/// an error.
#[test]
fn tuple_wildcard_component_reuse_still_errors() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, _) = p
  r1: tensor[4, f32] = realize(a)
  realize(a)
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message
                    .contains("variable `a` (from a destructured binding)")
        }),
        "expected the Linearity-F2 destructured-component diagnostic on `a`; got {errors:?}"
    );
}

// ============================================================
// Shadowing
// ============================================================

/// The component mark rides the scope entry, so an ordinary `let` that
/// re-binds a component name pushes an UNMARKED entry over it. The new
/// binding is an ordinary one and gets ordinary implicit-Copy treatment.
#[test]
fn ordinary_rebinding_of_a_component_name_drops_the_mark() {
    assert_linearity_clean(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32]), w: tensor[4, f32]) -> tensor[4, f32] = {
  (a, b) = p
  a: tensor[4, f32] = realize(w)
  r1: tensor[4, f32] = realize(a)
  r2: tensor[4, f32] = realize(a)
  add(r1, r2)
}
"#,
    );
}

/// The converse direction: a name bound ordinarily first and re-bound as
/// a destructure component afterwards DOES pick the mark up, so the
/// component's double consume errors.
#[test]
fn destructure_rebinding_of_an_ordinary_name_takes_the_mark() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32]), w: tensor[4, f32]) -> tensor[4, f32] = {
  a: tensor[4, f32] = realize(w)
  (a, b) = p
  r1: tensor[4, f32] = realize(a)
  realize(a)
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message
                    .contains("variable `a` (from a destructured binding)")
        }),
        "expected the Linearity-F2 destructured-component diagnostic on the re-bound `a`; \
         got {errors:?}"
    );
}

/// A component consumed through an alias still lands on the component's
/// marked scope entry, because the F2 guard tests the alias chain's
/// terminal target. Pins that the per-name gate did not lose the
/// alias-forwarded true positive.
#[test]
fn component_consumed_through_an_alias_still_errors() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  y: tensor[4, f32] = a
  r1: tensor[4, f32] = realize(y)
  realize(y)
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("(from a destructured binding)")),
        "expected the Linearity-F2 destructured-component diagnostic through alias `y`; \
         got {errors:?}"
    );
}
