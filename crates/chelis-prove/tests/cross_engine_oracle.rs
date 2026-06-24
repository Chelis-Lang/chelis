//! WI-12 (WS-5) cross-engine oracle: Z3 and cvc5 must AGREE.
//!
//! This battery is the authoritative correctness oracle for the Z3 NRA engine.
//! It runs the SAME [`Goal`]s through BOTH the cvc5 engine and the Z3 engine and
//! asserts they agree:
//!
//! - the same VERDICT CATEGORY (Proved / Disproved / non-verdict), and
//! - byte-identical `(soundness, qualifier_set)` on each discharge (the shared
//!   [`classify_smt_outcome`] guarantees this for any matching outcome).
//!
//! The two engines lower the SAME [`SmtExpr`] to the SAME exact-f64 rationals
//! (`BigRational::from_float`, the #444 cvc5 RealLit fix mirrored in the Z3
//! lowering), so they reason about IDENTICAL numbers. A divergence here is a
//! real bug in one lowering, not a modelling difference.
//!
//! This test compiles ONLY with BOTH `--features smt` (cvc5) AND `--features z3`
//! on: it is the dual-engine acceptance surface. Under a single feature it is
//! cfg'd out (there is no second engine to cross-check against).
#![cfg(all(feature = "smt", feature = "z3"))]

use chelis_prove::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
use chelis_prove::tier_b::TierBResult;
use chelis_prove::{Cvc5Engine, DischargeEngine, DischargeRegistry, Goal, SmtProperty, Z3Engine};

const TIMEOUT_MS: u64 = 10_000;

fn var(name: &str) -> SmtExpr {
    SmtExpr::Var(name.to_string())
}

fn cmp(op: CmpOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
    SmtExpr::Cmp(op, Box::new(l), Box::new(r))
}

fn arith(op: ArithOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
    SmtExpr::Arith(op, Box::new(l), Box::new(r))
}

fn real_prop(vars: &[&str], pre: Vec<SmtExpr>, post: SmtExpr) -> SmtProperty {
    SmtProperty {
        variables: vars
            .iter()
            .map(|v| (v.to_string(), SmtSort::Real))
            .collect(),
        preconditions: pre,
        postcondition: post,
    }
}

/// The verdict CATEGORY, collapsing the model payload (the two engines produce
/// different counterexample renderings, but the same SAT/UNSAT category).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Category {
    Proved,
    Disproved,
    /// Timeout / Unknown / Error -- a non-verdict (not a definite Proved or
    /// Disproved). The oracle does not require the two engines pick the SAME
    /// non-verdict (a timeout vs an unknown is engine-internal), only that
    /// neither claims a DEFINITE verdict the other contradicts.
    NonVerdict,
}

fn category(result: &TierBResult) -> Category {
    match result {
        TierBResult::Proved => Category::Proved,
        TierBResult::Disproved(_) => Category::Disproved,
        TierBResult::Timeout | TierBResult::Unknown | TierBResult::Error(_) => Category::NonVerdict,
    }
}

/// Run one goal through both engines and assert they agree. `expected` pins the
/// category the goal SHOULD reach in the shared decidable (algebraic) fragment
/// where both engines are complete; the two engines must BOTH reach it AND
/// carry byte-identical `(soundness, qualifier_set)`.
fn assert_engines_agree(prop: SmtProperty, expected: Category) {
    let goal = Goal::smt(prop);
    let cvc5 = Cvc5Engine::new().discharge(&goal, TIMEOUT_MS);
    let z3 = Z3Engine::new().discharge(&goal, TIMEOUT_MS);

    let cvc5_cat = category(cvc5.result());
    let z3_cat = category(z3.result());

    assert_eq!(
        cvc5_cat,
        expected,
        "cvc5 must reach {expected:?}, got {:?} ({:?})",
        cvc5_cat,
        cvc5.result()
    );
    assert_eq!(
        z3_cat,
        expected,
        "z3 must reach {expected:?}, got {:?} ({:?})",
        z3_cat,
        z3.result()
    );
    assert_eq!(
        cvc5_cat, z3_cat,
        "cross-engine DISAGREEMENT: cvc5 {cvc5_cat:?} vs z3 {z3_cat:?}"
    );

    // On a matching DEFINITE verdict, the shared classification must give both
    // engines byte-identical soundness + qualifier set: this is what lets the
    // dispatcher fall through cvc5 -> z3 (or vice versa) without one engine
    // laundering the other's guarantee.
    assert_eq!(
        cvc5.soundness(),
        z3.soundness(),
        "engines must carry identical soundness for an agreed verdict"
    );
    assert_eq!(
        cvc5.qualifier_set(),
        z3.qualifier_set(),
        "engines must carry identical qualifier set for an agreed verdict"
    );
}

#[test]
fn oracle_x_squared_non_negative_both_prove() {
    // forall x: x*x >= 0
    assert_engines_agree(
        real_prop(
            &["x"],
            vec![],
            cmp(
                CmpOp::Ge,
                arith(ArithOp::Mul, var("x"), var("x")),
                SmtExpr::RealLit(0.0),
            ),
        ),
        Category::Proved,
    );
}

#[test]
fn oracle_x_always_positive_both_disprove() {
    // forall x: x > 0 -- false (x = 0).
    assert_engines_agree(
        real_prop(
            &["x"],
            vec![],
            cmp(CmpOp::Gt, var("x"), SmtExpr::RealLit(0.0)),
        ),
        Category::Disproved,
    );
}

#[test]
fn oracle_precondition_implies_both_prove() {
    // forall x where x > 0: x >= 0
    assert_engines_agree(
        real_prop(
            &["x"],
            vec![cmp(CmpOp::Gt, var("x"), SmtExpr::RealLit(0.0))],
            cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(0.0)),
        ),
        Category::Proved,
    );
}

#[test]
fn oracle_intrinsic_value_non_negative_both_prove() {
    // if S > K then S - K else 0  >=  0
    let post = cmp(
        CmpOp::Ge,
        SmtExpr::Ite(
            Box::new(cmp(CmpOp::Gt, var("S"), var("K"))),
            Box::new(arith(ArithOp::Sub, var("S"), var("K"))),
            Box::new(SmtExpr::RealLit(0.0)),
        ),
        SmtExpr::RealLit(0.0),
    );
    assert_engines_agree(real_prop(&["S", "K"], vec![], post), Category::Proved);
}

#[test]
fn oracle_max_with_zero_non_negative_both_prove() {
    assert_engines_agree(
        real_prop(
            &["x"],
            vec![],
            cmp(
                CmpOp::Ge,
                SmtExpr::Apply("max".to_string(), vec![var("x"), SmtExpr::RealLit(0.0)]),
                SmtExpr::RealLit(0.0),
            ),
        ),
        Category::Proved,
    );
}

#[test]
fn oracle_abs_non_negative_both_prove() {
    assert_engines_agree(
        real_prop(
            &["x"],
            vec![],
            cmp(
                CmpOp::Ge,
                SmtExpr::Apply("abs".to_string(), vec![var("x")]),
                SmtExpr::RealLit(0.0),
            ),
        ),
        Category::Proved,
    );
}

#[test]
fn oracle_min_le_left_both_prove() {
    // min(x, y) <= x
    assert_engines_agree(
        real_prop(
            &["x", "y"],
            vec![],
            cmp(
                CmpOp::Le,
                SmtExpr::Apply("min".to_string(), vec![var("x"), var("y")]),
                var("x"),
            ),
        ),
        Category::Proved,
    );
}

#[test]
fn oracle_exact_f64_decimal_goal_both_disprove() {
    // THE exact-f64 parity oracle: 0.1 + 0.2 == 0.3 is FALSE at runtime f64.
    // Both engines lower the literals to the SAME exact rationals, so BOTH must
    // DISPROVE -- proving the two lowerings produce identical numbers. A
    // divergence here (one proves, one disproves) would mean the lowerings
    // disagree on the literal value, the exact bug #444 closed for cvc5.
    assert_engines_agree(
        real_prop(
            &[],
            vec![],
            cmp(
                CmpOp::Eq,
                arith(ArithOp::Add, SmtExpr::RealLit(0.1), SmtExpr::RealLit(0.2)),
                SmtExpr::RealLit(0.3),
            ),
        ),
        Category::Disproved,
    );
}

#[test]
fn oracle_exact_f64_dyadic_goal_both_prove() {
    // Negative parity: 0.5 + 0.25 == 0.75 is exact in f64, so BOTH prove.
    assert_engines_agree(
        real_prop(
            &[],
            vec![],
            cmp(
                CmpOp::Eq,
                arith(ArithOp::Add, SmtExpr::RealLit(0.5), SmtExpr::RealLit(0.25)),
                SmtExpr::RealLit(0.75),
            ),
        ),
        Category::Proved,
    );
}

#[test]
fn oracle_nonlinear_false_goal_both_disprove() {
    // forall x: x*x > x -- false (x = 0 and x = 1 are counterexamples).
    assert_engines_agree(
        real_prop(
            &["x"],
            vec![],
            cmp(CmpOp::Gt, arith(ArithOp::Mul, var("x"), var("x")), var("x")),
        ),
        Category::Disproved,
    );
}

#[test]
fn oracle_disproof_carries_real_arith_on_both() {
    // A disproof over the reals carries the SAME RealArith qualifier on both
    // engines (the hedged disproved_modulo_real_arithmetic, symmetric to the
    // proof side). Pin it explicitly: both engines' disproof discharges expose
    // the machine-arithmetic caveat identically.
    use chelis_prove::Qualifier;
    let goal = Goal::smt(real_prop(
        &["x"],
        vec![],
        cmp(CmpOp::Gt, var("x"), SmtExpr::RealLit(0.0)),
    ));
    let cvc5 = Cvc5Engine::new().discharge(&goal, TIMEOUT_MS);
    let z3 = Z3Engine::new().discharge(&goal, TIMEOUT_MS);
    assert!(matches!(cvc5.result(), TierBResult::Disproved(_)));
    assert!(matches!(z3.result(), TierBResult::Disproved(_)));
    assert!(cvc5.qualifier_set().contains(Qualifier::RealArith));
    assert!(z3.qualifier_set().contains(Qualifier::RealArith));
    assert_eq!(cvc5.qualifier_set(), z3.qualifier_set());
}

#[test]
fn oracle_capability_split_transcendental_z3_unsupported_cvc5_attempts() {
    // The capability split the try-until-discharge dispatcher exploits: a
    // transcendental goal Z3 has no kind for is a NON-VERDICT (Unsupported)
    // here, while cvc5 attempts QF_NRAT. exp(x) >= 0 is true, so cvc5 PROVES
    // it (over reals) and Z3 declines -- a DIVERGENCE that is EXPECTED and
    // correct (not all goals are in the shared fragment). This pins that the
    // split is real: Z3 must be a non-verdict, cvc5 must NOT be.
    let goal = Goal::smt(real_prop(
        &["x"],
        vec![],
        cmp(
            CmpOp::Ge,
            SmtExpr::Apply("exp".to_string(), vec![var("x")]),
            SmtExpr::RealLit(0.0),
        ),
    ));
    let cvc5 = Cvc5Engine::new().discharge(&goal, TIMEOUT_MS);
    let z3 = Z3Engine::new().discharge(&goal, TIMEOUT_MS);
    assert_eq!(
        category(z3.result()),
        Category::NonVerdict,
        "Z3 has no transcendental kind: exp must be a non-verdict (Unsupported)"
    );
    assert_eq!(
        category(cvc5.result()),
        Category::Proved,
        "cvc5 QF_NRAT proves exp(x) >= 0 over the reals"
    );
}

#[test]
fn dispatcher_falls_through_z3_unsupported_to_cvc5_proved() {
    // END-TO-END fall-through through the REAL registry with the REAL engines:
    // register Z3 FIRST, cvc5 SECOND. A transcendental goal (exp(x) >= 0) is a
    // NON-VERDICT for Z3 (no transcendental kind -> Unsupported Error), so the
    // dispatcher FALLS THROUGH to cvc5, which PROVES it over the reals. The
    // returned discharge is cvc5's Proved carrying cvc5's own
    // SoundApproximate / RealArith -- Z3's non-verdict neither blocks the
    // discharge nor launders cvc5's badge.
    let mut registry = DischargeRegistry::new();
    registry.register(Box::new(Z3Engine::new())); // first; will decline
    registry.register(Box::new(Cvc5Engine::new())); // second; will prove

    let goal = Goal::smt(real_prop(
        &["x"],
        vec![],
        cmp(
            CmpOp::Ge,
            SmtExpr::Apply("exp".to_string(), vec![var("x")]),
            SmtExpr::RealLit(0.0),
        ),
    ));
    // Z3 is the first fitting engine (the name diagnostic reports it), but the
    // VERDICT comes from cvc5 after fall-through.
    assert_eq!(registry.selected_engine_name(&goal), Some("z3"));
    let discharge = registry.dispatch(&goal, TIMEOUT_MS);
    assert_eq!(
        *discharge.result(),
        TierBResult::Proved,
        "the fall-through must reach cvc5's Proved for the transcendental goal"
    );
    assert_eq!(
        discharge.evidence().get("solver").and_then(|v| v.as_str()),
        Some("cvc5"),
        "the discharging engine is cvc5 (Z3 fell through)"
    );
}

#[test]
fn dispatcher_z3_first_proves_a_polynomial_goal_without_consulting_cvc5() {
    // Negative parity to the fall-through: when the FIRST engine (Z3) DOES
    // discharge a definite verdict (a polynomial goal it decides), the
    // dispatcher returns Z3's verdict and never consults cvc5 -- it does not
    // fall through past a definite verdict.
    let mut registry = DischargeRegistry::new();
    registry.register(Box::new(Z3Engine::new()));
    registry.register(Box::new(Cvc5Engine::new()));

    let goal = Goal::smt(real_prop(
        &["x"],
        vec![],
        cmp(
            CmpOp::Ge,
            arith(ArithOp::Mul, var("x"), var("x")),
            SmtExpr::RealLit(0.0),
        ),
    ));
    let discharge = registry.dispatch(&goal, TIMEOUT_MS);
    assert_eq!(*discharge.result(), TierBResult::Proved);
    assert_eq!(
        discharge.evidence().get("solver").and_then(|v| v.as_str()),
        Some("z3"),
        "Z3 decides the polynomial goal; cvc5 is never consulted"
    );
}
