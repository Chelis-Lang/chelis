//! chelis#496 oracle: every discharge producer attributes itself through ONE
//! canonical evidence key. The raw discharge evidence must carry a single
//! top-level `"engine"` key naming the producer (a `DischargeEngine::name()` or
//! a dispatcher pseudo-source), with solver-/backend-specific detail nested
//! under `"backend"` -- never the old per-engine `"solver"` / `"dispatch"`
//! attribution split.
//!
//! This is the cross-engine schema lock: a new engine that minted its own
//! attribution key (or left detail at the top level) fails here. The engines
//! covered depend on which features are linked; the always-present dispatcher
//! pseudo-discharges (no-fit / exhausted) are checked unconditionally.

use chelis_prove::engine_registry::exhausted_discharge;
use chelis_prove::solver::{CmpOp, SmtExpr, SmtSort};
use chelis_prove::{Discharge, Goal, SmtProperty, no_fit_discharge};

#[allow(unused_imports)]
use chelis_prove::DischargeEngine;

fn trivial_smt_goal() -> Goal {
    // forall x: Real . x == x
    Goal::smt(SmtProperty {
        variables: vec![("x".to_string(), SmtSort::Real)],
        preconditions: vec![],
        postcondition: SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::Var("x".to_string())),
            Box::new(SmtExpr::Var("x".to_string())),
        ),
    })
}

/// The canonical-attribution invariant every producer must satisfy: a single
/// top-level `"engine"` string equal to `expected`, and NONE of the legacy
/// per-engine attribution keys at the top level.
fn assert_canonical_attribution(discharge: &Discharge, expected: &str) {
    let evidence = discharge.evidence();
    assert_eq!(
        evidence.get("engine").and_then(|v| v.as_str()),
        Some(expected),
        "canonical top-level `engine` attribution key (chelis#496): {evidence:?}"
    );
    assert!(
        evidence.get("solver").is_none(),
        "legacy top-level `solver` attribution key must be gone: {evidence:?}"
    );
    assert!(
        evidence.get("dispatch").is_none(),
        "legacy top-level `dispatch` attribution key must be gone: {evidence:?}"
    );
}

#[test]
fn no_fit_discharge_uses_canonical_engine_key() {
    let discharge = no_fit_discharge(&trivial_smt_goal());
    assert_canonical_attribution(&discharge, "no_fit");
}

#[test]
fn exhausted_discharge_uses_canonical_engine_key() {
    let discharge = exhausted_discharge(&trivial_smt_goal(), None);
    assert_canonical_attribution(&discharge, "exhausted_fallthrough");
}

#[cfg(not(feature = "smt"))]
#[test]
fn solve_property_engine_uses_canonical_engine_key() {
    use chelis_prove::SolvePropertyEngine;
    let discharge = SolvePropertyEngine::new().discharge(&trivial_smt_goal(), 1_000);
    assert_canonical_attribution(&discharge, "solve_property");
}

#[cfg(feature = "smt")]
#[test]
fn cvc5_engine_uses_canonical_engine_key() {
    use chelis_prove::Cvc5Engine;
    let discharge = Cvc5Engine::new().discharge(&trivial_smt_goal(), 5_000);
    assert_canonical_attribution(&discharge, "cvc5");
}

#[cfg(feature = "z3")]
#[test]
fn z3_engine_uses_canonical_engine_key() {
    use chelis_prove::Z3Engine;
    let discharge = Z3Engine::new().discharge(&trivial_smt_goal(), 5_000);
    assert_canonical_attribution(&discharge, "z3");
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_sos_engine_uses_canonical_engine_key() {
    use chelis_prove::clarabel_sos::ClarabelSosEngine;
    // The trivial `x == x` goal is not a poly-nonneg-on-interval shape, so the
    // engine returns an honest Unknown -- which still routes its evidence
    // through the canonical attribution helper.
    let discharge = ClarabelSosEngine::unwired().discharge(&trivial_smt_goal(), 1_000);
    assert_canonical_attribution(&discharge, "clarabel_sos");
}
