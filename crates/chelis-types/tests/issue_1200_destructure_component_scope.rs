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
//! of the consumed target. A genuine component keeps the original
//! wording, whose "(from a destructured binding)" claim is now accurate;
//! an ordinary binding that merely aliases one says so instead.
//!
//! The three block-poisoning shapes below (B, C, D) all compiled on
//! 0.18.3, failed on 0.18.4, and must compile again. The true positive
//! F2 protects — reuse of a destructured component without `copy()` —
//! must keep failing.
//!
//! The Pass A section further down covers the review pass on that fix:
//! the rule is per BINDING, and while the checker still resolved
//! bindings through names, each case there was a place where that
//! resolution reached the wrong binding. The Pass B section at the
//! bottom pins the generation-crossing identity fix (chelis#1209) that
//! removed name resolution from checker state entirely: every binding
//! event has a unique generation id and all alias links, consumption
//! marks, and component marks attach to ids. The normative rule is
//! `spec/04-type-system.md` §8.3 ([04-LIN-1], [04-LIN-2]); the
//! mechanism is `spec/design/implicit_linearity.md` §"Destructured
//! components".

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
///
/// The diagnostic names `y` — the binding the user actually wrote — and
/// says it is an *alias of* a destructured binding. `y` is an ordinary
/// `let`, so calling it a destructured binding would be a false
/// statement about the user's own source, and naming the alias-chain
/// terminal instead would leak `__chelis_tmpN`.
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
                && e.message
                    .contains("variable `y` (an alias of a destructured binding)")),
        "expected the Linearity-F2 destructured-component diagnostic through alias `y`; \
         got {errors:?}"
    );
}

// ============================================================
// Pass A — the review's executed misroutes (chelis#1200 addendum)
//
// An adversarial pass confirmed by execution that resolving marks and
// consumes through bare string names in mutable alias stacks misroutes in
// six ways. Five are closed here; the generation-crossing pair is Pass B,
// delivered by chelis#1209 and pinned at the bottom of this file.
// ============================================================

// ------------------------------------------------------------
// Q1 — a branch body is a new declaration region
//
// Reviewer ruling: "match arms should work like closures. New
// declarations shadow the destructure mark." A closure body accepted a
// component's double consume (because `check_fn` re-`declare`s every
// capture in the closure's scope) while the identical match arm rejected
// it, since arm scopes clone without re-declaring. Branch scopes now drop
// the inherited marks on entry.
// ------------------------------------------------------------

/// A component consumed twice directly inside a match arm: accepted, to
/// agree with the closure-body verdict. 0.18.3 accepted this too.
#[test]
fn component_double_consume_inside_a_match_arm_is_accepted() {
    assert_linearity_clean(
        r#"
type Flag =
  | On
  | Off
def f(c: Flag, p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  match c with {
    | On => add(realize(a), realize(a))
    | Off => b
  }
}
"#,
    );
}

/// Same for an `if` branch. `check_if` clones without re-declaring for
/// exactly the same reason `check_match` does, so leaving `if` gated would
/// have replaced the old match-vs-closure split with an if-vs-match one.
#[test]
fn component_double_consume_inside_an_if_branch_is_accepted() {
    assert_linearity_clean(
        r#"
def f(c: bool, p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  if c then add(realize(a), realize(a)) else b
}
"#,
    );
}

/// NEGATIVE SIDE: only marks INHERITED from outside the branch are
/// dropped. A destructure authored inside the arm marks its own
/// components there, so its double consume is still the F2 error. Without
/// this the Q1 change would have been a blanket amnesty for anything
/// inside a `match`.
#[test]
fn destructure_authored_inside_a_match_arm_still_errors() {
    let errors = linearity_errors(
        r#"
type Flag =
  | On
  | Off
def f(c: Flag, t: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = match c with {
  | On => {
    (a, b) = t
    add(realize(a), realize(a))
  }
  | Off => to_tensor([0.0, 0.0, 0.0, 0.0])
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message
                    .contains("variable `a` (from a destructured binding)")
        }),
        "a destructure authored inside the arm must still be gated there; got {errors:?}"
    );
}

// ------------------------------------------------------------
// Q2 — branch joins propagate Structural consumes
//
// Reviewer ruling: "Accepted as proposed. Just a bug and clearly the only
// way it can be resolved." A component's carrier temp is never `Live` (the
// component's own bind records an `Aliasing` consume on it), so a join
// gated on `Live` dropped the branch's Structural consume outright.
// ------------------------------------------------------------

/// Component consumed in the THEN branch, then again after the join.
/// Silently accepted before this pass — which falsified the very true
/// positive F2 exists to protect.
#[test]
fn component_consumed_in_then_branch_then_after_join_errors() {
    let errors = linearity_errors(
        r#"
def f(c: bool, p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  r: tensor[4, f32] = if c then realize(a) else b
  add(r, realize(a))
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "the then-branch consume must survive the join; got {errors:?}"
    );
}

/// Same through the ELSE branch, so the join is not accidentally
/// order-sensitive.
#[test]
fn component_consumed_in_else_branch_then_after_join_errors() {
    let errors = linearity_errors(
        r#"
def f(c: bool, p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  r: tensor[4, f32] = if c then b else realize(a)
  add(r, realize(a))
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "the else-branch consume must survive the join; got {errors:?}"
    );
}

/// Same through a match arm.
#[test]
fn component_consumed_in_match_arm_then_after_join_errors() {
    let errors = linearity_errors(
        r#"
type Flag =
  | On
  | Off
def f(c: Flag, p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  r: tensor[4, f32] = match c with {
    | On => realize(a)
    | Off => b
  }
  add(r, realize(a))
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "the arm's consume must survive the join; got {errors:?}"
    );
}

/// POSITIVE CONTROL for the join change: consuming a component in a
/// branch and NOT again afterwards stays clean. The join now writes onto
/// `Consumed(Aliasing)` entries, so this pins that it did not start
/// reporting single consumes.
#[test]
fn component_consumed_only_inside_a_branch_stays_clean() {
    assert_linearity_clean(
        r#"
def f(c: bool, p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  if c then realize(a) else b
}
"#,
    );
}

// ------------------------------------------------------------
// Aliasing consumes are not Linearity-F2 events
// ------------------------------------------------------------

/// A second alias bind of the same component is a fan-out of borrows, not
/// a destruction: `lower_let` maps both names onto the component's node.
/// Gating it turned `(a, b) = p; y = a; z = a` into a hard error, which
/// 0.18.3 accepted.
#[test]
fn two_alias_binds_of_one_component_are_accepted() {
    assert_linearity_clean(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  y: tensor[4, f32] = a
  z: tensor[4, f32] = a
  add(y, z)
}
"#,
    );
}

// ------------------------------------------------------------
// Capture consumes forward through the alias chain
// ------------------------------------------------------------

/// Capture-then-reuse of a COMPONENT. A component is an alias of a
/// `__chelis_tmpN` carrier, and that carrier is the component's only
/// identity — so consuming the component's own entry at capture time left
/// the carrier `Live` and the later reuse was silently accepted, while the
/// same reuse without the closure errored. Distinct from Q1, which is
/// about a double consume INSIDE the body.
///
/// The carrier name is *synthesized*, not unforgeable: authored source can
/// spell `__chelis_tmp0`, and `fresh_destructure_temp` only screens the
/// block it is desugaring, so an authored name in an ENCLOSING block can
/// still produce the same spelling. That was chelis#1212; with id-keyed
/// checker state the collision is ordinary shadowing (see the cell at the
/// bottom of this file). Do not read this test as proving whole-context
/// uniqueness of carrier names; it proves carrier *forwarding*.
#[test]
fn component_captured_then_reused_outside_the_closure_errors() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  g = fn () -> realize(a)
  add(g(), realize(a))
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "capture-then-reuse of a component must error; got {errors:?}"
    );
}

/// The control the component case is measured against: capture-then-reuse
/// of an ORDINARY binding has always errored, and still does. Pins that
/// the component fix aligned the two rather than inventing a new rule.
#[test]
fn capture_then_reuse_of_an_ordinary_binding_still_errors() {
    let errors = linearity_errors(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] = {
  g = fn () -> realize(x)
  add(g(), realize(x))
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("closure capture")
        }),
        "capture-then-reuse of an ordinary binding must error; got {errors:?}"
    );
}

/// NEGATIVE PARITY for the capture forwarding: it is scoped to
/// components and must not touch ordinary aliases.
///
/// `y = x` gives two user-visible bindings for one value, and a capture
/// has consumed the name it captured since before this issue — so handing
/// two closures their own name compiles, and has since 0.18.3. Downstream
/// code spells exactly this (nautilus `lu_solve`: `lu_fwd = lu`, then one
/// closure per name), and coral's suite does not compile without it.
/// Forwarding every capture through the alias chain would be an unrelated
/// ecosystem-breaking tightening — the class of change chelis#1200 exists
/// to undo — and the component misroute does not need it.
#[test]
fn two_closures_capturing_one_value_through_an_alias_still_compile() {
    assert_linearity_clean(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] = {
  y: tensor[4, f32] = x
  g = fn () -> realize(y)
  h = fn () -> realize(x)
  add(g(), h())
}
"#,
    );
}

/// chelis#1211 item 2, the later-consume half: an ordinary alias stays
/// consumable after a branch closure consumes its source. The join's
/// Aliasing-to-Structural promotion is carrier-only, so the parent's
/// `Consumed(Aliasing)` record on `x` survives the join and the
/// after-join `realize(y)` forwards onto it as ordinary implicit-Copy
/// fan-out. The later-borrow half (`add(y, y)` after a branch consume)
/// is pinned end-to-end by the chelis-cli lane-parity cell
/// `ordinary_alias_survives_a_branch_consume_of_its_source_in_both_lanes`.
/// The verdict is recorded in `spec/design/implicit_linearity.md` §"New
/// declaration regions" and rides [04-LIN-2]'s identity ruling.
#[test]
fn ordinary_alias_still_consumable_after_a_branch_closure_consumes_its_source() {
    assert_linearity_clean(
        r#"
def f(c: bool, x: tensor[4, f32], t: tensor[4, f32]) -> tensor[4, f32] = {
  y: tensor[4, f32] = x
  r: tensor[4, f32] = if c then {
    g = fn () -> realize(x)
    g()
  } else t
  add(r, realize(y))
}
"#,
    );
}

/// A synthesized temp is never named to the user. `a`'s carrier is
/// `__chelis_tmpN`, and the closure-capture arm used to print the
/// alias-chain terminal — a name that appears nowhere in the source and
/// that the user cannot act on.
#[test]
fn diagnostics_never_name_a_synthesized_temp() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  g = fn () -> realize(a)
  add(g(), realize(a))
}
"#,
    );
    assert!(
        !errors.is_empty(),
        "fixture must produce a diagnostic to inspect"
    );
    for error in &errors {
        assert!(
            !error.message.contains("__chelis_tmp"),
            "diagnostic leaks a desugarer temp: {}",
            error.message
        );
        for suggestion in &error.suggestions {
            assert!(
                !suggestion.contains("__chelis_tmp"),
                "suggestion leaks a desugarer temp: {suggestion}"
            );
        }
    }
}

// ------------------------------------------------------------
// Pass B — generation-crossing identity (chelis#1209, delivered)
//
// An alias used to resolve by NAME, following the name to whatever
// binding was on top of the stack at consume time rather than to the
// generation it was taken against, and both directions misrouted. The
// checker now keys all state on per-binding generation ids
// (spec/04-type-system.md [04-LIN-1]): `BindingOrigin.alias` stores the
// `BindingId` resolved when the alias bind was recorded, so a later
// re-binding of the source's name neither re-points the chain nor lets
// a later destructure of that name capture the alias. These two cells
// are the acceptance oracle for that identity rule, one per misroute
// direction.
// ------------------------------------------------------------

/// Direction 1: alias taken, then its SOURCE shadowed.
///
/// `y` aliases the component `a`; an ordinary `let` then re-binds `a`.
/// The chain must keep pointing at the generation `y` was taken
/// against, so `y`'s double consume lands on the component's carrier
/// (reported against `y`) and the fresh, unrelated `a` stays live for
/// the legal read in the result. Under name-keyed resolution both went
/// wrong at once: the double consume escaped and the fresh `a` was
/// blamed.
#[test]
fn pass_b_alias_survives_shadowing_of_its_source() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32]), w: tensor[4, f32]) -> tensor[4, f32] = {
  (a, b) = p
  y: tensor[4, f32] = a
  a: tensor[4, f32] = w
  r1: tensor[4, f32] = realize(y)
  r2: tensor[4, f32] = realize(y)
  add(add(r1, r2), a)
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `y`")
        }),
        "the double consume of `y` must be reported against `y`, not against the shadowing `a`; \
         got {errors:?}"
    );
}

/// Direction 2: source aliased, then its NAME re-bound as a
/// destructured component.
///
/// `y` aliases an ordinary `x`, so double-consuming `y` is implicit-Copy
/// fan-out and must compile — the control without the re-bind does. The
/// chain terminates at the ordinary generation `y` was taken against;
/// the later destructure's component mark lives on a new generation the
/// chain never reaches. Under name-keyed resolution the walk landed on
/// the marked component and F2 fired: a false positive that existed
/// only because the alias crossed a generation.
#[test]
fn pass_b_alias_is_not_captured_by_a_later_destructure_of_its_source_name() {
    assert_linearity_clean(
        r#"
def f(t: (tensor[4, f32], tensor[4, f32]), s: tensor[4, f32]) -> tensor[4, f32] = {
  x: tensor[4, f32] = s
  y: tensor[4, f32] = x
  (x, c) = t
  r1: tensor[4, f32] = realize(y)
  r2: tensor[4, f32] = realize(y)
  add(add(r1, r2), x)
}
"#,
    );
}

/// The one verdict change generation identity brings beyond the two
/// directions above: a self-rebind `a = a` of a COMPONENT records a real
/// alias to the older generation (the source id is resolved before the
/// new binding is declared), so double-consuming the re-bound name
/// forwards to the component's carrier and is the same F2 error as any
/// other component alias. Under name-keyed resolution the self-link
/// walked back onto its own name, the cycle guard killed it, and the
/// fan-out silently fell through to implicit copy.
#[test]
fn component_self_rebind_alias_double_consume_errors() {
    let errors = linearity_errors(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] = {
  (a, b) = p
  a: tensor[4, f32] = a
  r1: tensor[4, f32] = realize(a)
  r2: tensor[4, f32] = realize(a)
  add(r1, r2)
}
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `a`")
                && e.message.contains("(an alias of a destructured binding)")
        }),
        "a component self-rebind is an alias of the component and its \
         double consume must be the F2 error; got {errors:?}"
    );
}

/// NEGATIVE PARITY for the cell above: the same self-rebind of an
/// ORDINARY binding is plain aliasing fan-out and stays copyable. The
/// alias chain terminates at an unmarked generation, so nothing changed
/// for this shape.
#[test]
fn ordinary_self_rebind_double_consume_stays_clean() {
    assert_linearity_clean(
        r#"
def f(s: tensor[4, f32]) -> tensor[4, f32] = {
  x: tensor[4, f32] = s
  x: tensor[4, f32] = x
  r1: tensor[4, f32] = realize(x)
  r2: tensor[4, f32] = realize(x)
  add(r1, r2)
}
"#,
    );
}

// ------------------------------------------------------------
// chelis#1212 — authored/synthesized carrier-name collision (closed by
// generation identity, chelis#1209)
//
// `fresh_destructure_temp` screens a candidate against the block it is
// desugaring only, so an authored `__chelis_tmpN` in an ENCLOSING block
// is invisible and the mint still produces the same SPELLING. That
// collision used to hide a genuine use-after-consume, because the
// name-keyed checker resolved the alias through whichever binding
// currently owned the name. With id-keyed state the authored binding
// and the synthesized carrier are distinct generations that merely
// share a spelling — ordinary shadowing — so the alias keeps pointing
// at the authored generation and the violation is reported. No desugar
// change was needed; the residual textual collision is cosmetic.
// ------------------------------------------------------------

/// The authored name used to be the ONLY difference from a program that
/// correctly errors: renaming `__chelis_tmp0` to `user_temp` reported
/// `y`'s use-after-consume while the colliding spelling made the error
/// disappear. Both spellings must now report it.
#[test]
fn authored_destructure_temp_name_does_not_hide_an_outer_double_consume() {
    let errors = linearity_errors(
        r#"
def two(t: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = (t, t)
def f(x: tensor[4, f32], w: tensor[4, f32]) -> tensor[4, f32] = {
  __chelis_tmp0 = x
  y = __chelis_tmp0
  r: tensor[4, f32] = {
    (a, b) = two(w)
    add(realize(y), add(a, b))
  }
  add(r, add(y, y))
}
"#,
    );
    assert!(
        errors.iter().any(|error| error.message.contains("`y`")),
        "the authored/synthesized collision must not hide `y`'s \
         use-after-consume; got {errors:?}"
    );
}
