use super::*;
use crate::obligations::{ObligationProperty, collect_obligations};
use crate::opaque::collect_opaque_invariants;
use chelis_types::types::Type;
use std::collections::BTreeMap;

fn deep_of(surf: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(surf).expect("parse surf");
    chelis_surf::desugar::desugar_program(&decls)
}

fn inferred_sigs(exprs: &[Expr]) -> BTreeMap<String, Type> {
    let checked = chelis_types::check_typed_program(exprs)
        .unwrap_or_else(|e| panic!("check: {:?}", e.errors));
    checked
        .signature_inference()
        .functions
        .iter()
        .map(|(n, s)| (n.clone(), s.checked_signature.clone()))
        .collect()
}

const GUARDED_OPTION: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";

#[test]
fn flagship_guarded_option_lowers_to_smt_property() {
    let exprs = deep_of(GUARDED_OPTION);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob: &ObligationProperty = col
        .obligations
        .iter()
        .find(|o| o.producer == "probability")
        .expect("obligation exists");
    let inv = &invs[0];

    // The producer has one scalar param; name it to match the producer's
    // own param so the body's free vars line up.
    let pparams = vec![(
        producer_first_param_name(&exprs, "probability"),
        ProducerParamType::Scalar("f32".to_string()),
    )];
    let lowered = lower_obligation(&exprs, inv, ob, &pparams, &crate::opaque::ConstEnv::new())
        .expect("guarded-Option obligation must lower to an SmtProperty");

    // The lowered postcondition must be an ite over the guard with the
    // invariant in the then-branch and `true` in the None branch.
    let s = format!("{:?}", lowered.property.postcondition);
    assert!(s.contains("Ite"), "expected case-of-ctor ite, got {s}");
    assert_eq!(lowered.property.variables.len(), 1);
}

#[cfg(feature = "smt")]
#[test]
fn flagship_guarded_option_proves_at_smt_tier() {
    use crate::tier_b::{TierBResult, solve_property};
    let exprs = deep_of(GUARDED_OPTION);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "probability")
        .unwrap();
    let inv = &invs[0];
    let pparams = vec![(
        producer_first_param_name(&exprs, "probability"),
        ProducerParamType::Scalar("f32".to_string()),
    )];
    let lowered =
        lower_obligation(&exprs, inv, ob, &pparams, &crate::opaque::ConstEnv::new()).unwrap();
    // The acceptance bar: SMT proves the guarded-Option obligation.
    assert_eq!(
        solve_property(&lowered.property, 5000),
        TierBResult::Proved,
        "guarded-Option obligation must prove at smt tier (D-TIERB)"
    );
}

#[cfg(feature = "smt")]
#[test]
fn clamping_constructor_proves_at_smt_tier() {
    // A clamping constructor always returns a valid value (nested-if
    // clamp; `min`/`max` are predicate-grammar intrinsics but not Surf
    // value builtins, so the clamp is expressed structurally).
    let surf = "module Stats.Prob
export (clamp_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability =
  Probability { value: if x >= 0.0 then (if x <= 1.0 then x else 1.0) else 0.0 }
";
    use crate::tier_b::{TierBResult, solve_property};
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "clamp_prob")
        .expect("clamp producer obligation");
    let inv = &invs[0];
    let pparams = vec![(
        producer_first_param_name(&exprs, "clamp_prob"),
        ProducerParamType::Scalar("f32".to_string()),
    )];
    let lowered =
        lower_obligation(&exprs, inv, ob, &pparams, &crate::opaque::ConstEnv::new()).unwrap();
    assert_eq!(solve_property(&lowered.property, 5000), TierBResult::Proved);
}

#[cfg(feature = "smt")]
#[test]
fn non_validating_constructor_disproves_at_smt_tier() {
    // This constructor returns Some(Probability{x}) for x > 1.0 too — it
    // does not establish the invariant, so the obligation is disproved.
    let surf = "module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Option[Probability] =
  if x >= 0.0 then Some(Probability { value: x }) else None
";
    use crate::tier_b::{TierBResult, solve_property};
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "bad_prob")
        .unwrap();
    let inv = &invs[0];
    let pparams = vec![(
        producer_first_param_name(&exprs, "bad_prob"),
        ProducerParamType::Scalar("f32".to_string()),
    )];
    let lowered =
        lower_obligation(&exprs, inv, ob, &pparams, &crate::opaque::ConstEnv::new()).unwrap();
    assert!(
        matches!(
            solve_property(&lowered.property, 5000),
            TierBResult::Disproved(_)
        ),
        "non-validating constructor must be disproved"
    );
}

/// Resolve the producer's first param name from the Deep program (the
/// lowering uses the property's quantified var name = the producer param
/// name).
fn producer_first_param_name(exprs: &[Expr], producer: &str) -> String {
    let prod = super::lookup_producer(exprs, producer).expect("producer body");
    prod.params.first().cloned().expect("at least one param")
}
