//! chelis#1512: the reconcile step, at the level where its two branches are
//! both reachable.
//!
//! A suspended route publishes a type before its rule has run. When the rule
//! finally runs it produces an answer, and those two must agree. The failure
//! mode this guards is not "the wrong diagnostic" but a silent one: a
//! relocated decision quietly overwriting the published type with the
//! replayed one, so a program that disagreed with itself would check clean.

use super::*;

fn tensor(dim: i64) -> Type {
    Type::Tensor(vec![Dim::Lit(dim)], TensorPrec::Concrete(Prim::F32))
}

/// REGRESSION TEST for the disagreement branch: reconciling must push exactly
/// one typed error, and must hand back the PRODUCED type. Returning the
/// published type instead would be the silent override.
#[test]
fn a_replayed_answer_that_disagrees_is_reported_and_never_silently_overridden() {
    let mut subst = Subst::new();
    let (produced, errors) = crate::session::with_test_sink(|sink| {
        reconcile_replayed_result("permute", &tensor(3), tensor(4), &mut subst, sink)
    });

    assert_eq!(
        produced,
        tensor(4),
        "the replayed rule's own answer is what the call produces"
    );
    assert_eq!(
        errors.len(),
        1,
        "a disagreement is one diagnostic, got {errors:?}"
    );
    assert!(matches!(errors[0].kind, CheckErrorKind::TypeMismatch));
    assert!(
        errors[0].message.contains(
            "`permute` result does not match the type this call produces once its operand is known"
        ),
        "the diagnostic must say which call disagreed, got {:?}",
        errors[0].message
    );
}

/// NEGATIVE TWIN. Agreement is silent, and the published variable is bound to
/// the produced type rather than left free -- a suspended call publishes a
/// fresh variable, so leaving it unbound would strand every consumer.
#[test]
fn a_replayed_answer_that_agrees_reports_nothing_and_binds_the_published_variable() {
    let mut subst = Subst::new();
    let mut vg = VarGen::default();
    let published = Type::Var(vg.fresh_tvar());
    let (produced, errors) = crate::session::with_test_sink(|sink| {
        reconcile_replayed_result("permute", &published, tensor(3), &mut subst, sink)
    });

    assert_eq!(produced, tensor(3));
    assert!(
        errors.is_empty(),
        "agreement must be silent, got {errors:?}"
    );
    assert_eq!(
        subst.apply(&published),
        tensor(3),
        "the published variable must be bound to the rule's answer"
    );
}
