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

/// Build the producer-param list for an update-shaped producer whose first
/// param is the opaque input `Probability` and second is a scalar.
#[cfg(feature = "smt")]
fn update_pparams(
    exprs: &[Expr],
    inv: &crate::opaque::OpaqueInvariant,
    producer: &str,
) -> Vec<(String, ProducerParamType)> {
    let prod = super::lookup_producer(exprs, producer).expect("producer body");
    vec![
        (
            prod.params[0].clone(),
            ProducerParamType::Opaque(inv.clone()),
        ),
        (
            prod.params[1].clone(),
            ProducerParamType::Scalar("f32".to_string()),
        ),
    ]
}

#[cfg(feature = "smt")]
const PROB_DEFS: &str = "module Stats.Prob
export (scale_down)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def prob_value(p: Probability) -> f32 = p.value
";

#[cfg(feature = "smt")]
#[test]
fn update_shaped_producer_proves_under_input_invariant_at_smt() {
    use crate::tier_b::{TierBResult, solve_property};
    // scale_down preserves [0,1] GIVEN the input is in [0,1]: for k in
    // [0,1], p.value * k stays in [0,1] when p.value is in [0,1]. The
    // input invariant is the injected assumption (D-SOUND inductive step).
    let surf = format!(
        "{PROB_DEFS}def scale_down(p: Probability, k: f32) -> Probability =\n  \
         Probability {{ value: if k >= 0.0 then (if k <= 1.0 then prob_value(p) * k else prob_value(p)) else prob_value(p) }}\n"
    );
    let exprs = deep_of(&surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "scale_down")
        .expect("update-shaped producer is an obligation");
    let inv = &invs[0];
    let pparams = update_pparams(&exprs, inv, "scale_down");
    let lowered = lower_obligation(&exprs, inv, ob, &pparams, &crate::opaque::ConstEnv::new())
        .expect("update-shaped obligation lowers with input-invariant injection");
    // The precondition is the input invariant (the assumption).
    assert_eq!(lowered.property.preconditions.len(), 1);
    assert_eq!(
        solve_property(&lowered.property, 5000),
        TierBResult::Proved,
        "invariant-preserving update proves GIVEN the input assumption"
    );
}

#[cfg(feature = "smt")]
#[test]
fn update_shaped_violating_twin_is_disproved_at_smt() {
    use crate::tier_b::{TierBResult, solve_property};
    // bad_scale adds k UNGUARDED (k can exceed the band), so even with a
    // valid input p.value+k can leave [0,1]: disproved.
    let surf = "module Stats.Prob
export (bad_scale)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def prob_value(p: Probability) -> f32 = p.value
def bad_scale(p: Probability, k: f32) -> Probability = Probability { value: prob_value(p) + k }
"
    .to_string();
    let exprs = deep_of(&surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "bad_scale")
        .unwrap();
    let inv = &invs[0];
    let pparams = update_pparams(&exprs, inv, "bad_scale");
    let lowered =
        lower_obligation(&exprs, inv, ob, &pparams, &crate::opaque::ConstEnv::new()).unwrap();
    assert!(
        matches!(
            solve_property(&lowered.property, 5000),
            TierBResult::Disproved(_)
        ),
        "unguarded update must be disproved even under the input assumption"
    );
}

#[cfg(feature = "smt")]
const GUARDED_OPTION_CONST: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
hi = 1.0
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= hi then Some(Probability { value: x }) else None
";

#[cfg(feature = "smt")]
#[test]
fn cr8_guard_with_module_constant_proves_at_smt_tier() {
    // CR-8: a producer guard comparing against an in-module zero-arg
    // constant (`hi = 1.0`) lowered with the bare `(var hi)` unresolved,
    // so its Tier B obligation was wrong (and the SMT solver paniced on
    // the undeclared variable). With the constant resolved, the
    // guard-then-Option obligation proves at smt.
    use crate::tier_b::{TierBResult, solve_property};
    let exprs = deep_of(GUARDED_OPTION_CONST);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "probability")
        .expect("obligation exists");
    let inv = &invs[0];
    let pparams = vec![(
        producer_first_param_name(&exprs, "probability"),
        ProducerParamType::Scalar("f32".to_string()),
    )];
    let mut consts = crate::opaque::ConstEnv::new();
    consts.insert("hi".to_string(), 1.0);
    let lowered = lower_obligation(&exprs, inv, ob, &pparams, &consts)
        .expect("guard-with-constant obligation must lower");
    assert_eq!(
        solve_property(&lowered.property, 5000),
        TierBResult::Proved,
        "guard against a module constant proves at smt once resolved"
    );
}

#[cfg(feature = "smt")]
#[test]
fn cr2_4_int_constant_guard_proves_at_smt_tier() {
    // CR2-4 end-to-end: an int-field opaque type whose producer guards an
    // int param against an int-typed module constant must lower with the
    // constant as an INTEGER literal (SmtSort::Int), so the obligation
    // proves at the SMT tier. Inlining the constant as f32 would mix
    // int/real sorts and mis-lower the guard.
    use crate::tier_b::{TierBResult, solve_property};
    let surf = "module M
export (mk_counter)
@opaque
@invariant(c) c.n >= 0
type Counter =
  | Counter { n: int32 }
lo = 0
def mk_counter(x: int32) -> Option[Counter] =
  if x >= lo then Some(Counter { n: x }) else None
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "mk_counter")
        .expect("counter producer obligation");
    let inv = &invs[0];
    let pparams = vec![(
        producer_first_param_name(&exprs, "mk_counter"),
        ProducerParamType::Scalar("int32".to_string()),
    )];
    let mut consts = crate::opaque::ConstEnv::new();
    consts.insert("lo".to_string(), 0.0);
    let lowered = lower_obligation(&exprs, inv, ob, &pparams, &consts)
        .expect("int-constant guard obligation must lower");
    assert_eq!(
        solve_property(&lowered.property, 5000),
        TierBResult::Proved,
        "int-guarded counter proves at smt with the constant kept integral"
    );
}

/// Resolve the producer's first param name from the Deep program (the
/// lowering uses the property's quantified var name = the producer param
/// name).
fn producer_first_param_name(exprs: &[Expr], producer: &str) -> String {
    let prod = super::lookup_producer(exprs, producer).expect("producer body");
    prod.params.first().cloned().expect("at least one param")
}

#[test]
fn cr2_4_int_typed_constant_inlines_as_integer_literal_not_f32() {
    // CR2-4: a constant declared with an integer type must inline as an
    // integer literal, not be silently retyped to f32. The ConstEnv only
    // carries an f64, so the type is recovered from the const def's
    // declared literal type in the module.
    let surf = "module M
n = 3
def n_fn() -> int32 = 7
m = 3.0
";
    let exprs = deep_of(surf);

    // Value-binding int constant `n = 3`.
    let node = super::const_lit_node(&exprs, "n", 3.0);
    assert_eq!(super::tag(&node), Some("lit"), "is a lit node");
    let lit_value = super::children(&node).first().cloned().expect("lit value");
    assert!(
        matches!(lit_value, Expr::Atom(Atom::Int(3), _)),
        "int-typed `n = 3` inlines as Atom::Int(3), got {lit_value:?}"
    );
    assert_eq!(
        super::const_declared_int_type(&exprs, "n").as_deref(),
        Some("int32"),
        "n is declared int32"
    );

    // Zero-arg int constant fn `def n_fn() -> int32 = 7`.
    let fn_node = super::const_lit_node(&exprs, "n_fn", 7.0);
    let fn_value = super::children(&fn_node)
        .first()
        .cloned()
        .expect("lit value");
    assert!(
        matches!(fn_value, Expr::Atom(Atom::Int(7), _)),
        "int-typed `def n_fn() -> int32 = 7` inlines as Atom::Int(7), got {fn_value:?}"
    );

    // Float constant `m = 3.0` still inlines as an f32 literal (CR-8).
    let float_node = super::const_lit_node(&exprs, "m", 3.0);
    let float_value = super::children(&float_node)
        .first()
        .cloned()
        .expect("lit value");
    assert!(
        matches!(float_value, Expr::Atom(Atom::Float(_), _)),
        "f32-typed `m = 3.0` still inlines as Atom::Float, got {float_value:?}"
    );
    assert_eq!(
        super::const_declared_int_type(&exprs, "m"),
        None,
        "m is not an integer type"
    );
}
