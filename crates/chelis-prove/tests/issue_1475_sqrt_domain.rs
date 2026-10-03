//! Regression oracle for chelis#1475: cvc5's partial `sqrt` may only be
//! lowered after its argument's real-domain obligation has been proved from
//! the user's other, total preconditions.

#![cfg(feature = "smt")]

mod support;

use chelis_prove::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use chelis_prove::tier_b::{SmtProperty, TierBResult, lower_to_cvc5, solve_property};
use chelis_prove::{
    CompositeVerdict,
    property_runner::{
        PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus, PropertyTier,
        run_surf_source_properties,
    },
};

fn var(name: &str) -> SmtExpr {
    SmtExpr::Var(name.to_string())
}

fn real(value: f64) -> SmtExpr {
    SmtExpr::RealLit(value)
}

fn arith(op: ArithOp, left: SmtExpr, right: SmtExpr) -> SmtExpr {
    SmtExpr::Arith(op, Box::new(left), Box::new(right))
}

fn cmp(op: CmpOp, left: SmtExpr, right: SmtExpr) -> SmtExpr {
    SmtExpr::Cmp(op, Box::new(left), Box::new(right))
}

fn sqrt(argument: SmtExpr) -> SmtExpr {
    SmtExpr::Apply("sqrt".to_string(), vec![argument])
}

fn real_property(preconditions: Vec<SmtExpr>, postcondition: SmtExpr) -> SmtProperty {
    SmtProperty {
        variables: vec![
            ("x".to_string(), SmtSort::Real),
            ("y".to_string(), SmtSort::Real),
        ],
        preconditions,
        postcondition,
    }
}

fn assert_domain_error(result: TierBResult) {
    let TierBResult::Error(reason) = result else {
        panic!("expected a loud sqrt-domain Error, got {result:?}");
    };
    assert!(reason.contains("sqrt"), "reason names sqrt: {reason}");
    assert!(
        reason.contains("non-negative"),
        "reason names the missing domain proof: {reason}"
    );
    assert!(reason.contains("Tier C"), "reason names fallback: {reason}");
    assert!(reason.contains("#1475"), "reason cites the issue: {reason}");
}

fn run_one(source: &str, tier: &str, samples: usize) -> PropertyOutcome {
    let options = PropertyRunOptions {
        tier: tier.to_string(),
        samples,
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(mut outcomes) =
        run_surf_source_properties(source, &options).expect("source parses and runs");
    assert_eq!(outcomes.len(), 1, "exactly one property: {outcomes:?}");
    outcomes.pop().expect("length checked")
}

#[test]
fn reported_negative_argument_counterexample_is_refused() {
    crate::support::isolate();
    let property = real_property(
        vec![
            cmp(CmpOp::Gt, sqrt(var("x")), real(1.5)),
            cmp(CmpOp::Lt, var("x"), real(100.0)),
        ],
        cmp(CmpOp::Gt, var("x"), real(0.0)),
    );

    assert_domain_error(solve_property(&property, 5_000));
}

#[test]
fn reported_property_proves_when_user_supplies_the_domain() {
    crate::support::isolate();
    let property = real_property(
        vec![
            cmp(CmpOp::Gt, sqrt(var("x")), real(1.5)),
            cmp(CmpOp::Lt, var("x"), real(100.0)),
            cmp(CmpOp::Ge, var("x"), real(0.0)),
        ],
        cmp(CmpOp::Gt, var("x"), real(0.0)),
    );

    assert_eq!(solve_property(&property, 5_000), TierBResult::Proved);
}

#[test]
fn arithmetic_negative_argument_counterexample_is_refused() {
    crate::support::isolate();
    let argument = arith(
        ArithOp::Sub,
        arith(ArithOp::Mul, var("x"), var("x")),
        real(1.0),
    );
    let property = real_property(
        vec![
            cmp(CmpOp::Gt, sqrt(argument), real(1.5)),
            cmp(CmpOp::Lt, var("x"), real(100.0)),
        ],
        cmp(CmpOp::Gt, var("x"), real(0.0)),
    );

    assert_domain_error(solve_property(&property, 5_000));
}

#[test]
fn user_nonnegative_guard_keeps_sqrt_in_tier_b() {
    crate::support::isolate();
    let property = real_property(
        vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
        cmp(CmpOp::Ge, sqrt(var("x")), real(0.0)),
    );

    assert_eq!(solve_property(&property, 5_000), TierBResult::Proved);
}

#[test]
fn tiny_request_timeout_refuses_a_zero_sqrt_sub_budget() {
    crate::support::isolate();
    let property = real_property(
        vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
        cmp(CmpOp::Ge, sqrt(var("x")), real(0.0)),
    );

    for timeout_ms in [0, 1] {
        let result = solve_property(&property, timeout_ms);
        assert_domain_error(result);
    }
}

#[test]
fn algebraically_nonnegative_argument_keeps_sqrt_in_tier_b() {
    crate::support::isolate();
    let square = arith(ArithOp::Mul, var("x"), var("x"));
    let property = real_property(vec![], cmp(CmpOp::Ge, sqrt(square), real(0.0)));

    assert_eq!(solve_property(&property, 5_000), TierBResult::Proved);
}

#[test]
fn genuinely_false_in_domain_property_still_disproves() {
    crate::support::isolate();
    let property = real_property(
        vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
        cmp(CmpOp::Lt, sqrt(var("x")), real(0.0)),
    );

    assert!(
        matches!(solve_property(&property, 5_000), TierBResult::Disproved(_)),
        "the domain gate must not turn an in-domain disproof into a proof"
    );
}

#[test]
fn unproved_sqrt_domain_is_not_silently_assumed() {
    crate::support::isolate();
    // Over concrete reals-as-floats this is false at every negative x because
    // sqrt(x) is NaN and the comparison is false. Adding x >= 0 to the query
    // would hide those witnesses and forge a proof, so Tier B must refuse.
    let property = real_property(vec![], cmp(CmpOp::Ge, sqrt(var("x")), real(0.0)));

    assert_domain_error(solve_property(&property, 5_000));
}

#[test]
fn every_distinct_sqrt_argument_needs_its_own_proof() {
    crate::support::isolate();
    let sum = arith(ArithOp::Add, sqrt(var("x")), sqrt(var("y")));
    let property = real_property(
        vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
        cmp(CmpOp::Ge, sum, real(0.0)),
    );

    assert_domain_error(solve_property(&property, 5_000));
}

#[test]
fn proving_every_distinct_sqrt_argument_preserves_tier_b_reach() {
    crate::support::isolate();
    let sum = arith(ArithOp::Add, sqrt(var("x")), sqrt(var("y")));
    let property = real_property(
        vec![
            cmp(CmpOp::Ge, var("x"), real(0.0)),
            cmp(CmpOp::Ge, var("y"), real(0.0)),
        ],
        cmp(CmpOp::Ge, sum, real(0.0)),
    );

    assert_eq!(solve_property(&property, 5_000), TierBResult::Proved);
}

#[test]
fn top_level_conjunctions_supply_domain_evidence() {
    crate::support::isolate();
    let property = real_property(
        vec![SmtExpr::Bool(
            BoolOp::And,
            vec![
                cmp(CmpOp::Ge, var("x"), real(0.0)),
                cmp(CmpOp::Lt, var("x"), real(100.0)),
            ],
        )],
        cmp(CmpOp::Ge, sqrt(var("x")), real(0.0)),
    );

    assert_eq!(solve_property(&property, 5_000), TierBResult::Proved);
}

#[test]
fn nested_sqrt_and_partial_domain_evidence_fail_closed() {
    crate::support::isolate();
    let nested = real_property(
        vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
        cmp(CmpOp::Ge, sqrt(sqrt(var("x"))), real(0.0)),
    );
    assert_domain_error(solve_property(&nested, 5_000));

    let quotient = arith(ArithOp::Div, var("x"), var("y"));
    let partial = real_property(
        vec![cmp(CmpOp::Ge, quotient.clone(), real(0.0))],
        cmp(CmpOp::Ge, sqrt(quotient), real(0.0)),
    );
    assert_domain_error(solve_property(&partial, 5_000));
}

#[test]
fn quantified_domain_evidence_fails_closed() {
    crate::support::isolate();
    let property = real_property(
        vec![SmtExpr::Forall(
            vec![("x".to_string(), SmtSort::Real)],
            Box::new(cmp(CmpOp::Ge, var("x"), real(0.0))),
        )],
        cmp(CmpOp::Ge, sqrt(var("x")), real(0.0)),
    );

    assert_domain_error(solve_property(&property, 5_000));

    let quantified_sqrt = real_property(
        vec![],
        SmtExpr::Forall(
            vec![("z".to_string(), SmtSort::Real)],
            Box::new(cmp(CmpOp::Ge, sqrt(var("z")), real(0.0))),
        ),
    );
    assert_domain_error(solve_property(&quantified_sqrt, 5_000));
}

#[test]
fn exp_and_abs_controls_keep_their_existing_results() {
    crate::support::isolate();
    let exp_property = real_property(
        vec![],
        cmp(
            CmpOp::Gt,
            SmtExpr::Apply("exp".to_string(), vec![var("x")]),
            real(1.5),
        ),
    );
    assert!(matches!(
        solve_property(&exp_property, 5_000),
        TierBResult::Disproved(_)
    ));

    let abs_property = real_property(
        vec![],
        cmp(
            CmpOp::Gt,
            SmtExpr::Apply("abs".to_string(), vec![var("x")]),
            real(1.5),
        ),
    );
    assert!(matches!(
        solve_property(&abs_property, 5_000),
        TierBResult::Disproved(_)
    ));
}

#[test]
fn unguarded_public_lowering_refuses_sqrt() {
    crate::support::isolate();
    let term_manager = cvc5_rs::TermManager::new();
    let real_sort = term_manager.real_sort();
    let mut variables = std::collections::BTreeMap::new();
    variables.insert("x".to_string(), term_manager.mk_const(real_sort, "x"));
    let mut sorts = chelis_unord::UnordMap::new();
    sorts.insert("x".to_string(), SmtSort::Real);

    let result = lower_to_cvc5(&term_manager, &sqrt(var("x")), &variables, &sorts);
    let reason = result.expect_err("unguarded sqrt lowering must fail closed");
    assert!(reason.contains("sqrt"), "reason names sqrt: {reason}");
    assert!(
        reason.contains("domain"),
        "reason names the authorization boundary: {reason}"
    );
}

#[test]
fn surf_smt_only_reports_the_domain_boundary_as_unsupported() {
    crate::support::isolate();
    let source = r#"module M
@property unsafe_sqrt forall(x: f32) where sqrt(x) > 1.5, x < 100.0:
  (x > 0.0)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(reason.contains("sqrt"), "reason names sqrt: {reason}");
    assert!(
        reason.contains("non-negative"),
        "reason names the missing domain proof: {reason}"
    );
}

#[test]
fn surf_auto_falls_through_without_claiming_an_smt_verdict() {
    crate::support::isolate();
    let source = r#"module M
@property unsafe_sqrt forall(x: f32) where sqrt(x) > 1.5, x < 100.0:
  (x > 0.0)
"#;
    let outcome = run_one(source, "auto", 64);

    assert_eq!(
        outcome.status,
        PropertyStatus::Unsupported,
        "Tier C cannot generate this compound guard, so fallback is loud: {outcome:?}"
    );
    assert_eq!(outcome.proof_tier, PropertyTier::None, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::Unsupported,
        "{outcome:?}"
    );
    assert!(
        outcome
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("unsupported scalar guard"),
        "Tier C reports why it cannot sample the guard: {outcome:?}"
    );
}
