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

/// chelis#1475's guarantee: NO outcome short of a proved domain obligation
/// authorizes lowering `sqrt`, and the refusal says why in the user's terms.
///
/// Two outcomes satisfy it, and the suite distinguishes them deliberately
/// (chelis#3635): an obligation the solver could not decide is a bare
/// [`TierBResult::Error`], while one it decided FALSE additionally carries the
/// model that falsified it. Collapsing the second back onto the first is the
/// pre-chelis#3635 behaviour that discarded the evidence.
fn assert_domain_refusal_reason(reason: &str) {
    assert!(reason.contains("sqrt"), "reason names sqrt: {reason}");
    assert!(
        reason.contains("non-negative"),
        "reason names the missing domain proof: {reason}"
    );
    assert!(reason.contains("Tier C"), "reason names fallback: {reason}");
    assert!(reason.contains("#1475"), "reason cites the issue: {reason}");
}

/// An obligation the solver did not decide: a refusal with NO witness.
fn assert_domain_error(result: TierBResult) {
    let TierBResult::Error(reason) = &result else {
        panic!("expected an undecided sqrt-domain Error, got {result:?}");
    };
    assert_domain_refusal_reason(reason);
}

/// An obligation the solver decided FALSE: the same refusal, plus the witness
/// (chelis#3635). Returns it so a caller can assert what it contains.
fn assert_domain_disproved(result: TierBResult) -> serde_json::Value {
    let TierBResult::DomainDisproved { reason, witness } = &result else {
        panic!("expected a witness-carrying sqrt-domain refusal, got {result:?}");
    };
    assert_domain_refusal_reason(reason);
    assert!(
        reason.contains("the domain obligation is false"),
        "reason names the decided obligation: {reason}"
    );
    witness.clone()
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

    // `x < 100` admits a negative `x`, so the obligation is DECIDED false and
    // its witness travels with the refusal (chelis#3635).
    let witness = assert_domain_disproved(solve_property(&property, 5_000));
    let x = witness["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("the witness carries a numeric `x`: {witness}"));
    assert!(x < 0.0, "the witness is the negative argument: {witness}");
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

    assert_domain_disproved(solve_property(&property, 5_000));
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

    assert_domain_disproved(solve_property(&property, 5_000));
}

#[test]
fn every_distinct_sqrt_argument_needs_its_own_proof() {
    crate::support::isolate();
    let sum = arith(ArithOp::Add, sqrt(var("x")), sqrt(var("y")));
    let property = real_property(
        vec![cmp(CmpOp::Ge, var("x"), real(0.0))],
        cmp(CmpOp::Ge, sum, real(0.0)),
    );

    // `y` has no guard, so the conjoined obligation is decided false on it.
    let witness = assert_domain_disproved(solve_property(&property, 5_000));
    let y = witness["y"]
        .as_f64()
        .unwrap_or_else(|| panic!("the witness carries a numeric `y`: {witness}"));
    assert!(y < 0.0, "the unguarded argument is the witness: {witness}");
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

    // A quantified precondition supplies NO domain evidence, so the obligation
    // runs with an empty evidence set and is decided false.
    assert_domain_disproved(solve_property(&property, 5_000));

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

// ---------------------------------------------------------------------------
// chelis#3635: a FALSE domain obligation is a failure with its witness, not an
// unsupported result.
//
// The witness's only authority is the concrete host-lane evaluation, so every
// test below pairs the confirmed case with the two ways confirmation declines:
// the witness violates a precondition the evidence set dropped, and the
// property body is true at the witness in machine arithmetic. In both the
// verdict stays exactly what it was before this lane existed.
// ---------------------------------------------------------------------------

#[test]
fn issue3635_confirmed_domain_witness_fails_with_its_counterexample() {
    crate::support::isolate();
    let source = r#"module M
def k(x: f64) -> f64 = sqrt(x)

@property k_nonneg forall(x: f64) where x > -1.0f64, x < 2.0f64:
  (k(x) >= 0.0f64)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    // The counterexample is a concrete machine value the host lane evaluated,
    // not an over-the-reals model, so the badge is a DEFINITE failure rather
    // than `disproved_modulo_real_arithmetic`, and it discloses no
    // `real_arithmetic` caveat.
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::Failed,
        "{outcome:?}"
    );
    assert!(
        outcome.disclosed_qualifiers().is_empty(),
        "a machine-confirmed failure discloses no real-arithmetic caveat: {outcome:?}"
    );
    let counterexample = outcome
        .counterexample
        .as_ref()
        .unwrap_or_else(|| panic!("a confirmed witness is reported: {outcome:?}"));
    let x = counterexample["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("the counterexample carries a numeric `x`: {counterexample}"));
    assert!(
        x < 0.0,
        "the counterexample is the negative `sqrt` argument: {counterexample}"
    );
    assert!(
        (-1.0..2.0).contains(&x),
        "the counterexample is admitted by the property's own guards: {counterexample}"
    );
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("#3635"),
        "the failure says where the counterexample came from: {reason}"
    );
}

#[test]
fn issue3635_auto_reports_the_same_smt_failure_without_sampling() {
    crate::support::isolate();
    let source = r#"module M
def k(x: f64) -> f64 = sqrt(x)

@property k_nonneg forall(x: f64) where x > -1.0f64, x < 2.0f64:
  (k(x) >= 0.0f64)
"#;
    let outcome = run_one(source, "auto", 64);

    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert_eq!(
        outcome.proof_tier,
        PropertyTier::Smt,
        "the confirmed witness settles it before Tier C runs: {outcome:?}"
    );
    assert_eq!(outcome.samples, 0, "{outcome:?}");
}

#[test]
fn issue3635_a_witness_violating_a_dropped_guard_stays_unsupported() {
    crate::support::isolate();
    // `x / 1.0` is outside the total algebraic evidence fragment, so the
    // obligation is proved from an EMPTY evidence set and its witness is
    // negative -- but the property's real guard excludes it. The host lane
    // rejects it, and the verdict is the pre-chelis#3635 unsupported.
    let source = r#"module M
def k(x: f64) -> f64 = sqrt(x)

@property k_guarded forall(x: f64) where x / 1.0f64 > 0.0f64, x < 2.0f64:
  (k(x) >= 0.0f64)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert!(outcome.counterexample.is_none(), "{outcome:?}");
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert_domain_refusal_reason(reason);
    assert!(
        reason.contains("does not satisfy the property's own preconditions"),
        "the unsupported result says the witness was rejected: {reason}"
    );
}

#[test]
fn issue3635_a_witness_where_the_body_holds_stays_unsupported() {
    crate::support::isolate();
    // The domain obligation is still false -- `sqrt` does take a negative
    // argument -- but at that argument `k(x)` is NaN, so `k(x) != k(x)` makes
    // the body TRUE. Only the concrete evaluation can see that, which is
    // exactly why it, and not the solver, decides.
    let source = r#"module M
def k(x: f64) -> f64 = sqrt(x)

@property k_nan_or_nonneg forall(x: f64) where x > -1.0f64, x < 2.0f64:
  (k(x) != k(x) || k(x) >= 0.0f64)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert!(outcome.counterexample.is_none(), "{outcome:?}");
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert_domain_refusal_reason(reason);
    assert!(
        reason.contains("the property body holds"),
        "the unsupported result says the body survived the witness: {reason}"
    );
}

#[test]
fn issue3635_an_undecided_domain_obligation_reports_no_counterexample() {
    crate::support::isolate();
    // A nested `sqrt` argument never reaches the solver, so there is no
    // witness to confirm and the verdict must stay unsupported. The negative
    // control for the whole lane: `unsupported` is still reserved for an
    // obligation nothing decided.
    let source = r#"module M
def k(x: f64) -> f64 = sqrt(sqrt(x))

@property k_nonneg forall(x: f64) where x > -1.0f64, x < 2.0f64:
  (k(x) >= 0.0f64)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert!(outcome.counterexample.is_none(), "{outcome:?}");
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("nested `sqrt` argument"),
        "the undecided obligation keeps its own reason: {reason}"
    );
}

#[test]
fn issue3635_an_f32_binder_confirms_at_its_declared_width() {
    crate::support::isolate();
    // The witness is an exact rational; the host lane rounds it ONCE to the
    // binder's declared width before evaluating, so an `f32` property is
    // confirmed at `f32` rather than at the solver's precision.
    let source = r#"module M
def k(x: f32) -> f32 = sqrt(x)

@property k_nonneg forall(x: f32) where x > -1.0, x < 2.0:
  (k(x) >= 0.0)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    let counterexample = outcome
        .counterexample
        .as_ref()
        .unwrap_or_else(|| panic!("a confirmed witness is reported: {outcome:?}"));
    let x = counterexample["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("the counterexample carries a numeric `x`: {counterexample}"));
    assert!(x < 0.0, "{counterexample}");
    assert_eq!(
        x,
        f64::from(x as f32),
        "the reported counterexample is exactly representable at f32: {counterexample}"
    );
}

#[test]
fn issue3635_the_deep_lane_confirms_its_own_witness() {
    crate::support::isolate();
    // The Deep lane carries the same domain phase and the same confirmation.
    // Without this the fix would be Surf-only while `chelis prove` accepts
    // `.dp` input on the same tier.
    let source = r#"(module {} M
      (defsig {} k (t-fn {} (t-prim {} f64) (t-prim {} f64)))
      (def {} k
        (fn {} (params {} (x {type: (t-prim {} f64)}))
          (app {} (var {} sqrt) (var {} x))))
      (def {chelis_role: "property",
            property_preconditions: (tuple {}
              (app {} (var {} gt) (var {} x)
                (app {} (var {} neg) (lit {type: (t-prim {} f64)} 1.0)))
              (app {} (var {} cmplt) (var {} x) (lit {type: (t-prim {} f64)} 2.0))),
            property_quantifiers: (params {} (x {type: (t-prim {} f64)})),
            property_source_kind: "user"} k_nonneg
        (fn {} (params {} (x {type: (t-prim {} f64)}))
          (app {} (var {} gte)
            (app {} (var {} k) (var {} x))
            (lit {type: (t-prim {} f64)} 0.0)))))"#;
    let options = PropertyRunOptions {
        tier: "smt-only".to_string(),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(outcomes) =
        chelis_prove::property_runner::run_deep_source_properties(source, &options)
            .expect("the Deep source parses and runs");
    assert_eq!(outcomes.len(), 1, "exactly one property: {outcomes:?}");
    let outcome = &outcomes[0];

    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::Failed,
        "{outcome:?}"
    );
    let counterexample = outcome
        .counterexample
        .as_ref()
        .unwrap_or_else(|| panic!("a confirmed witness is reported: {outcome:?}"));
    let x = counterexample["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("the counterexample carries a numeric `x`: {counterexample}"));
    assert!(x < 0.0, "{counterexample}");
    assert!((-1.0..2.0).contains(&x), "{counterexample}");
}

#[test]
fn issue3635_an_integer_binder_is_rebuilt_at_its_declared_width() {
    crate::support::isolate();
    // The witness names every binder, not just the one feeding `sqrt`. An
    // integer binder must be rebuilt as an INTEGER at its declared width --
    // `int_sample` clamps out of range, and a clamped value would be a
    // different input from the one evaluated.
    let source = r#"module M
def k(x: f64, n: i32) -> f64 = sqrt(x)

@property k_nonneg forall(x: f64, n: i32) where x > -1.0f64, x < 2.0f64, n > 0, n < 5:
  (k(x, n) >= 0.0f64)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    let counterexample = outcome
        .counterexample
        .as_ref()
        .unwrap_or_else(|| panic!("a confirmed witness is reported: {outcome:?}"));
    let n = counterexample["n"]
        .as_i64()
        .unwrap_or_else(|| panic!("`n` is reported as an integer, not a float: {counterexample}"));
    assert!(
        (1..5).contains(&n),
        "the integer binder is admitted by its own guards: {counterexample}"
    );
    let x = counterexample["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("`x` is numeric: {counterexample}"));
    assert!(x < 0.0, "{counterexample}");
}

#[test]
fn issue3635_a_constant_sqrt_argument_confirms_against_an_unrelated_binder() {
    crate::support::isolate();
    // The negative argument is a CONSTANT, so the obligation is false with no
    // dependence on the binder at all, and the witness's `x` is whatever the
    // guards admit. The confirmation still has to evaluate the body -- which
    // is false because `sqrt` of a negative constant is NaN -- so the failure
    // is earned by the evaluation rather than by the binder's value.
    let source = r#"module M
def k() -> f64 = sqrt(0.0f64 - 1.0f64)

@property k_nonneg forall(x: f64) where x > 0.0f64, x < 1.0f64:
  (k() >= 0.0f64)
"#;
    let outcome = run_one(source, "smt-only", 0);

    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    let counterexample = outcome
        .counterexample
        .as_ref()
        .unwrap_or_else(|| panic!("a confirmed witness is reported: {outcome:?}"));
    let x = counterexample["x"]
        .as_f64()
        .unwrap_or_else(|| panic!("`x` is numeric: {counterexample}"));
    assert!(
        x > 0.0 && x < 1.0,
        "the binder is admitted by its own guards even though it is irrelevant: {counterexample}"
    );
}
