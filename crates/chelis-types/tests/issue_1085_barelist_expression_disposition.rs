//! chelis#1085 -- `Expr::BareList` in expression position is a loud rejection.
//!
//! Spec authority: spec/design/checker_totality.md §C1.1/§C1.2 and
//! spec/04-type-system.md §10 [04-TOT-1].
//!
//! `infer_expr`'s `BareList` arm used to infer each child and return the LAST
//! child's type, which for the empty list is `Type::Unit`, and count the node
//! as typed. That made the two checker ingresses disagree about the same
//! program: `check_ir_program` normalizes `BareList` into a tagless
//! `Expr::List` before inference and rejects it on the unknown-tag arm, while
//! `check_typed_program` walks stamped Deep directly and accepted
//! `(def {} f ())` with zero errors. The permissive side scored a form with no
//! honest type as checked -- the chelis#873-shape fail-open.
//!
//! This file locks both halves:
//!
//! * a headless list reaching expression dispatch is an `UnknownForm`
//!   rejection on BOTH ingresses, which are required to agree;
//! * the structural bare lists that a well-formed program legitimately
//!   contains -- `params` entries, a match arm's absent guard, an empty
//!   `deftype` type-parameter list -- are consumed by their owning form and
//!   must keep checking cleanly. Those are the over-rejection controls.

use chelis_deep::{Atom, DeepTag, Expr, MetaMap, Span, parse_and_stamp};
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn zero() -> Span {
    Span::new(0, 0)
}

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp(source).unwrap_or_else(|e| panic!("fixture must stamp: {source}\n{e}"))
}

/// Errors from the stamped-Deep ingress, which preserves `Expr::BareList`
/// rather than normalizing it to `Expr::List` first.
fn typed_errors(exprs: &[Expr]) -> Vec<CheckError> {
    match check_typed_program(exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn ir_errors(exprs: &[Expr]) -> Vec<CheckError> {
    match check_ir_program(exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn assert_unknown_form_naming_headless_list(errors: &[CheckError], context: &str) {
    let unknown_form = errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::UnknownForm))
        .unwrap_or_else(|| panic!("{context} must be rejected as UnknownForm; got {errors:?}"));
    assert!(
        unknown_form.message.contains("has no checker disposition"),
        "{context} must state the real disposition; got: {}",
        unknown_form.message
    );
    assert!(
        unknown_form.message.contains("headless list"),
        "{context} must name the headless-list shape so the author can see \
         what to write instead; got: {}",
        unknown_form.message
    );
}

// ── Negative polarity: a headless list in expression position rejects ───────

#[test]
fn empty_list_as_def_body_is_rejected_loudly() {
    let errors = typed_errors(&stamped("(def {} f ())"));
    assert!(
        !errors.is_empty(),
        "`()` as a def body has no honest type and must push a diagnostic, \
         not type as unit (chelis#1085)"
    );
    assert_unknown_form_naming_headless_list(&errors, "`()` as a def body");
    let unknown_form = errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::UnknownForm))
        .expect("checked above");
    assert!(
        unknown_form.message.contains("empty list"),
        "the empty case must be named as such; got: {}",
        unknown_form.message
    );
}

#[test]
fn empty_list_as_block_child_is_rejected_loudly() {
    let errors = typed_errors(&stamped("(def {} f (block {} ()))"));
    assert_unknown_form_naming_headless_list(&errors, "`()` as a block child");
}

#[test]
fn empty_list_as_bind_rhs_is_rejected_loudly() {
    let errors = typed_errors(&stamped("(def {} f (let {} (bind {} x ()) (var {} x)))"));
    assert_unknown_form_naming_headless_list(&errors, "`()` as a let-bind right-hand side");
}

#[test]
fn empty_list_as_app_callee_is_rejected_loudly() {
    let errors = typed_errors(&stamped("(def {} f (app {} () (lit {} 1)))"));
    assert_unknown_form_naming_headless_list(&errors, "`()` as an app callee");
}

/// The stamp pass never puts a NON-empty `BareList` at a runtime-expression
/// slot -- a non-empty list there decodes to `Node` or `UnknownForm`. This
/// case is therefore defensive, reachable only by programmatic construction,
/// and it must reject rather than type as its last child.
#[test]
fn programmatic_non_empty_bare_list_in_expression_position_is_rejected_loudly() {
    let body = Expr::BareList(
        vec![
            Expr::node(
                DeepTag::Lit,
                MetaMap::default(),
                vec![Expr::Atom(Atom::Int(1), zero())],
                zero(),
            ),
            Expr::node(
                DeepTag::Lit,
                MetaMap::default(),
                vec![Expr::Atom(Atom::Int(2), zero())],
                zero(),
            ),
        ],
        zero(),
    );
    let def = Expr::node(
        DeepTag::Def,
        MetaMap::default(),
        vec![Expr::Atom(Atom::Name("f".to_string()), zero()), body],
        zero(),
    );

    let errors = typed_errors(&[def]);
    assert_unknown_form_naming_headless_list(&errors, "a non-empty headless list");
    let unknown_form = errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::UnknownForm))
        .expect("checked above");
    assert!(
        unknown_form.message.contains("2 element(s)"),
        "the diagnostic must report the element count so the shape is \
         identifiable; got: {}",
        unknown_form.message
    );
}

/// The defect was the two ingresses disagreeing, so agreement is the
/// invariant, not merely "the stamped path rejects".
#[test]
fn both_checker_ingresses_agree_that_a_headless_list_rejects() {
    for source in [
        "(def {} f ())",
        "(def {} f (block {} ()))",
        "(def {} f (let {} (bind {} x ()) (var {} x)))",
    ] {
        let exprs = stamped(source);
        let typed = typed_errors(&exprs);
        let ir = ir_errors(&exprs);
        assert!(
            !typed.is_empty(),
            "check_typed_program must reject `{source}`"
        );
        assert!(!ir.is_empty(), "check_ir_program must reject `{source}`");
        assert!(
            typed
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::UnknownForm))
                && ir
                    .iter()
                    .any(|e| matches!(e.kind, CheckErrorKind::UnknownForm)),
            "both ingresses must reject `{source}` with the same UnknownForm \
             kind; typed={typed:?} ir={ir:?}"
        );
    }
}

// ── Positive polarity: legitimate structural bare lists still check ─────────

/// A `params` entry `(x {type: ...})` IS an `Expr::BareList` -- it has no
/// vocabulary head. It is consumed structurally by `infer_fn`, never inferred
/// as an expression, so the rejection above must not touch it. This is the
/// primary over-rejection control.
#[test]
fn function_parameter_bare_lists_still_check_cleanly() {
    let exprs = stamped(
        "(defsig {} identity (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n\
         (def {} identity (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
    );
    let errors = typed_errors(&exprs);
    assert!(
        errors.is_empty(),
        "a `params` entry is a structural bare list consumed by its owning \
         `fn`, not an expression; it must not be rejected: {errors:?}"
    );
    assert!(
        ir_errors(&exprs).is_empty(),
        "the same must hold on the normalizing ingress"
    );
}

/// A match arm's absent guard is the one legitimate empty `()` at a
/// runtime-expression slot. `infer_match` screens it before expression
/// dispatch, so it must keep checking cleanly.
#[test]
fn match_arm_absent_guard_still_checks_cleanly() {
    let exprs = stamped("(def {} f (match {} (lit {} 1) (arm {} (pat-wild {}) () (lit {} 2))))");
    let errors = typed_errors(&exprs);
    assert!(
        errors.is_empty(),
        "an arm's absent guard `()` is legitimate and is screened by \
         `infer_match` before expression dispatch: {errors:?}"
    );
    assert!(
        ir_errors(&exprs).is_empty(),
        "the same must hold on the normalizing ingress"
    );
}

/// An empty `deftype` type-parameter list is also a `BareList`, at a syntax
/// slot its owning declaration reads structurally.
#[test]
fn empty_deftype_type_parameter_list_still_checks_cleanly() {
    let exprs = stamped("(deftype {} Foo () (variant {} A))\n(def {} f (lit {} 1))");
    let errors = typed_errors(&exprs);
    assert!(
        errors.is_empty(),
        "an empty `deftype` type-parameter list is structural syntax, not an \
         expression: {errors:?}"
    );
    assert!(
        ir_errors(&exprs).is_empty(),
        "the same must hold on the normalizing ingress"
    );
}

#[test]
fn well_typed_control_program_stays_clean() {
    let exprs = stamped(
        "(def {} f (lit {type: (t-prim {} f32)} 1.0))\n\
         (def {} g (if {} (lit {type: (t-prim {} bool)} true) \
          (var {} f) (lit {type: (t-prim {} f32)} 2.0)))",
    );
    let errors = typed_errors(&exprs);
    assert!(
        errors.is_empty(),
        "the well-typed control must stay clean; got {errors:?}"
    );
}
