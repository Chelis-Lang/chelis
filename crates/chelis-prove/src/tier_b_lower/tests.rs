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

/// F3 (review 4): the int-width -> SmtSort::Int decision must be single
/// source across ALL sites (field `scalar_sort`, producer-param sort, the
/// constant recognizer). An int8/int16 opaque field + int8/int16 producer
/// param + an int8/int16 constant used in BOTH the guard AND the invariant
/// must lower CONSISTENTLY (every int operand is `SmtSort::Int` / `IntLit`)
/// and prove at Tier B -- not mix IntLit-const against a Real-sorted field
/// var (which would route to Tier C, the regression the unification missed).
#[cfg(feature = "smt")]
#[test]
fn f3_int_width_field_param_const_lowers_consistently_at_tier_b() {
    use crate::tier_b::{TierBResult, solve_property};
    // Loop over EVERY integer width. The bound `0` is cast to the field
    // width so the module type-checks (integer literals default to int32, so
    // a bare `0` against an int8 field is a precision mismatch).
    for width in ["int8", "int16", "int32", "int64"] {
        let surf = format!(
            "module M
export (mk_counter)
@opaque
@invariant(c) c.n >= ({lo} : {width})
type Counter =
  | Counter {{ n: {width} }}
def mk_counter(x: {width}) -> Option[Counter] =
  if x >= ({lo} : {width}) then Some(Counter {{ n: x }}) else None
",
            lo = 0,
            width = width
        );
        let exprs = deep_of(&surf);
        let invs = collect_opaque_invariants(&exprs);
        assert_eq!(
            invs.len(),
            1,
            "an `{width}` opaque field is in the value class (collected)"
        );
        let sigs = inferred_sigs(&exprs);
        let col = collect_obligations(&exprs, &invs, &sigs);
        let ob = col
            .obligations
            .iter()
            .find(|o| o.producer == "mk_counter")
            .unwrap_or_else(|| panic!("counter producer obligation for {width}"));
        let inv = &invs[0];
        let pparams = vec![(
            producer_first_param_name(&exprs, "mk_counter"),
            ProducerParamType::Scalar(width.to_string()),
        )];
        let consts = crate::opaque::ConstEnv::new();
        let lowered = lower_obligation(&exprs, inv, ob, &pparams, &consts)
            .unwrap_or_else(|| panic!("`{width}` obligation must lower to Tier B"));
        // Every solver variable for this all-int module must be Int-sorted
        // (no int field/param silently lowered to Real).
        for (vname, sort) in &lowered.property.variables {
            assert_eq!(
                *sort,
                crate::solver::SmtSort::Int,
                "var `{vname}` of an all-{width} module must be Int-sorted, got {sort:?}"
            );
        }
        assert_eq!(
            solve_property(&lowered.property, 5000),
            TierBResult::Proved,
            "all-{width} field/param/const proves at Tier B (no sort mismatch)"
        );
    }
}

/// F4 (review 4): the producer-body comparison lowering
/// (`lower_pred_bool`) routes through the SAME operand-sort reconciliation
/// as the flattened-predicate path, so an int binder field compared against
/// an integral constant lowers with consistent sorts (no RealLit against an
/// Int field var). The producer-body path previously had NO reconciliation,
/// so it was the parallel-path divergence the unification missed.
#[cfg(feature = "smt")]
#[test]
fn f4_producer_body_int_field_comparison_lowers_consistently() {
    use crate::tier_b::{TierBResult, solve_property};
    // An int32 field whose invariant compares it against a module constant
    // `lo`. The GUARDED Option producer constructs the value only when
    // `x >= lo`, so the obligation lowers through the case-of-known-ctor
    // reduction -> apply_invariant -> lower_pred_bool, the F4 path, and
    // PROVES. The constant and the field must lower as the SAME sort.
    let surf = "module M
export (mk)
@opaque
@invariant(c) c.n >= lo
type Counter =
  | Counter { n: int32 }
lo = 0
def mk(x: int32) -> Option[Counter] =
  if x >= lo then Some(Counter { n: x }) else None
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "mk")
        .expect("mk producer obligation");
    let inv = &invs[0];
    let pparams = vec![(
        producer_first_param_name(&exprs, "mk"),
        ProducerParamType::Scalar("int32".to_string()),
    )];
    let mut consts = crate::opaque::ConstEnv::new();
    consts.insert("lo".to_string(), 0.0);
    let lowered = lower_obligation(&exprs, inv, ob, &pparams, &consts)
        .expect("int-field producer-body obligation must lower");
    // The postcondition compares the int field (an Int var) against the int
    // constant; the reconciliation keeps both Int (no RealLit anywhere).
    assert!(
        !smt_contains_real_lit(&lowered.property.postcondition),
        "producer-body int comparison carries no RealLit: {:?}",
        lowered.property.postcondition
    );
    // And it proves at Tier B (a sort mismatch would route it to Tier C or
    // abort cvc5 before the U3 pre-check even runs).
    assert_eq!(
        solve_property(&lowered.property, 5000),
        TierBResult::Proved,
        "consistent-sort int field comparison proves at Tier B"
    );
}

/// F4: the shared reconciliation rewrites an integral RealLit on the other
/// side when one operand is integer-sorted, in both operand orders, and is
/// a no-op for a fractional real or two reals.
#[test]
fn f4_shared_reconcile_coerces_integral_real_against_int_only() {
    use crate::solver::SmtExpr;
    // int var on the left, integral real on the right => right becomes Int.
    let mut l = SmtExpr::Var("n".into());
    let mut r = SmtExpr::RealLit(3.0);
    crate::opaque::reconcile_cmp_operands(&mut l, &mut r, true, false);
    assert!(matches!(r, SmtExpr::IntLit(3)), "integral real -> IntLit: {r:?}");
    // Reverse order.
    let mut l = SmtExpr::RealLit(5.0);
    let mut r = SmtExpr::Var("n".into());
    crate::opaque::reconcile_cmp_operands(&mut l, &mut r, false, true);
    assert!(matches!(l, SmtExpr::IntLit(5)), "integral real -> IntLit (rev): {l:?}");
    // A FRACTIONAL real against an int operand is NOT coerced (it would
    // change the value); it stays Real (the U3 pre-check then routes out).
    let mut l = SmtExpr::Var("n".into());
    let mut r = SmtExpr::RealLit(2.5);
    crate::opaque::reconcile_cmp_operands(&mut l, &mut r, true, false);
    assert!(matches!(r, SmtExpr::RealLit(_)), "fractional real stays Real: {r:?}");
    // Two non-int operands: no coercion.
    let mut l = SmtExpr::RealLit(1.0);
    let mut r = SmtExpr::RealLit(2.0);
    crate::opaque::reconcile_cmp_operands(&mut l, &mut r, false, false);
    assert!(matches!((&l, &r), (SmtExpr::RealLit(_), SmtExpr::RealLit(_))));
}

/// True if any leaf of an SmtExpr is a `RealLit`.
#[cfg(feature = "smt")]
fn smt_contains_real_lit(e: &SmtExpr) -> bool {
    use crate::solver::SmtExpr as E;
    match e {
        E::RealLit(_) => true,
        E::Arith(_, l, r) | E::Cmp(_, l, r) => {
            smt_contains_real_lit(l) || smt_contains_real_lit(r)
        }
        E::Bool(_, kids) => kids.iter().any(smt_contains_real_lit),
        E::Not(inner) | E::Forall(_, inner) | E::Exists(_, inner) => smt_contains_real_lit(inner),
        E::Apply(_, args) => args.iter().any(smt_contains_real_lit),
        E::Ite(c, t, el) => {
            smt_contains_real_lit(c) || smt_contains_real_lit(t) || smt_contains_real_lit(el)
        }
        _ => false,
    }
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
        crate::opaque::const_declared_int_type(&exprs, "n").as_deref(),
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
        crate::opaque::const_declared_int_type(&exprs, "m"),
        None,
        "m is not an integer type"
    );
}

// ===========================================================================
// U2 (review 3): ONE type-aware constant lowering, used by BOTH the
// producer-body path AND the invariant-application path.
// ===========================================================================

/// True if any leaf of an SmtExpr is a `RealLit`.
fn contains_real_lit(e: &SmtExpr) -> bool {
    match e {
        SmtExpr::RealLit(_) => true,
        SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
            contains_real_lit(l) || contains_real_lit(r)
        }
        SmtExpr::Bool(_, kids) => kids.iter().any(contains_real_lit),
        SmtExpr::Not(inner) | SmtExpr::Forall(_, inner) | SmtExpr::Exists(_, inner) => {
            contains_real_lit(inner)
        }
        SmtExpr::Apply(_, args) => args.iter().any(contains_real_lit),
        SmtExpr::Ite(c, t, e) => {
            contains_real_lit(c) || contains_real_lit(t) || contains_real_lit(e)
        }
        _ => false,
    }
}

/// True if any leaf of an SmtExpr is an `IntLit`.
fn contains_int_lit(e: &SmtExpr) -> bool {
    match e {
        SmtExpr::IntLit(_) => true,
        SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
            contains_int_lit(l) || contains_int_lit(r)
        }
        SmtExpr::Bool(_, kids) => kids.iter().any(contains_int_lit),
        SmtExpr::Not(inner) | SmtExpr::Forall(_, inner) | SmtExpr::Exists(_, inner) => {
            contains_int_lit(inner)
        }
        SmtExpr::Apply(_, args) => args.iter().any(contains_int_lit),
        SmtExpr::Ite(c, t, e) => contains_int_lit(c) || contains_int_lit(t) || contains_int_lit(e),
        _ => false,
    }
}

/// An int-field opaque type whose INVARIANT compares the int field against
/// an int-typed module constant `lo`, AND whose producer guards the int
/// param against the SAME constant. Under the old code the producer-body
/// path lowered `lo` as IntLit (CR2-4) while the invariant path hardcoded
/// `RealLit`, so the SAME constant lowered with two different sorts within
/// one property -- comparing an Int var against a Real literal -- and cvc5
/// ABORTED the process ("Subexpressions must have the same type: Int/Real").
const INT_CONST_IN_INVARIANT: &str = "module M
export (mk_counter)
@opaque
@invariant(c) c.n >= lo
type Counter =
  | Counter { n: int32 }
lo = 0
def mk_counter(x: int32) -> Option[Counter] =
  if x >= lo then Some(Counter { n: x }) else None
";

fn lower_int_const_obligation(surf: &str) -> LoweredObligation {
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
    lower_obligation(&exprs, inv, ob, &pparams, &consts)
        .expect("int-constant-in-invariant obligation must lower")
}

#[test]
fn u2_int_constant_in_invariant_lowers_as_int_not_real() {
    // The SAME int constant must lower identically (as an integer literal)
    // on BOTH the producer-body guard AND the invariant predicate. The
    // postcondition must carry NO RealLit (which would force a Real sort and
    // a mismatch against the Int-sorted field var).
    let lowered = lower_int_const_obligation(INT_CONST_IN_INVARIANT);
    assert!(
        !contains_real_lit(&lowered.property.postcondition),
        "int constant in the invariant must NOT lower as RealLit: {:?}",
        lowered.property.postcondition
    );
    assert!(
        contains_int_lit(&lowered.property.postcondition),
        "int constant in the invariant lowers as IntLit: {:?}",
        lowered.property.postcondition
    );
}

#[cfg(feature = "smt")]
#[test]
fn u2_int_constant_in_invariant_proves_without_sort_mismatch() {
    // End-to-end: with the constant kept integral on both paths the
    // obligation proves at SMT. Under the old code this ABORTED cvc5.
    use crate::tier_b::{TierBResult, solve_property};
    let lowered = lower_int_const_obligation(INT_CONST_IN_INVARIANT);
    assert_eq!(
        solve_property(&lowered.property, 5000),
        TierBResult::Proved,
        "int constant in both guard and invariant proves, no sort-mismatch abort"
    );
}

#[test]
fn u2_const_declared_int_type_covers_int8_and_int16() {
    // The declared-type reader must recognize EVERY integer width, not just
    // int32/int64. int8/int16 constants used in a guard or invariant must
    // lower to SmtSort::Int, not be silently retyped to Real.
    let surf = "module M
def a() -> int8 = 1
def b() -> int16 = 2
def c() -> int32 = 3
def d() -> int64 = 4
def e() -> f32 = 5.0
";
    let exprs = deep_of(surf);
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "a").as_deref(),
        Some("int8"),
        "int8 is recognized as an integer constant type"
    );
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "b").as_deref(),
        Some("int16"),
        "int16 is recognized as an integer constant type"
    );
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "c").as_deref(),
        Some("int32")
    );
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "d").as_deref(),
        Some("int64")
    );
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "e"),
        None,
        "an f32 constant is not an integer type"
    );

    // int8/int16 inline as integer literals.
    let a_node = super::const_lit_node(&exprs, "a", 1.0);
    let a_val = super::children(&a_node)
        .first()
        .cloned()
        .expect("lit value");
    assert!(
        matches!(a_val, Expr::Atom(Atom::Int(1), _)),
        "int8 `a = 1` inlines as Atom::Int(1), got {a_val:?}"
    );
    let b_node = super::const_lit_node(&exprs, "b", 2.0);
    let b_val = super::children(&b_node)
        .first()
        .cloned()
        .expect("lit value");
    assert!(
        matches!(b_val, Expr::Atom(Atom::Int(2), _)),
        "int16 `b = 2` inlines as Atom::Int(2), got {b_val:?}"
    );
}

#[test]
fn u2_constant_body_referencing_another_constant_keeps_int_type() {
    // A constant whose body references ANOTHER constant must resolve its
    // type transitively. The hard case is an UNTYPED value binding whose
    // body is a bare `(var other)`: there is no defsig on the alias and no
    // literal on its body, so the type must be followed through the chain to
    // `base`'s declared int type. The original reader returned None here and
    // the constant lost its int type.
    let surf = "module M
def base() -> int32 = 7
typed_alias = base
def fn_alias() -> int32 = base
";
    let exprs = deep_of(surf);
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "typed_alias").as_deref(),
        Some("int32"),
        "an untyped value binding `typed_alias = base` follows the chain to int32"
    );
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs, "fn_alias").as_deref(),
        Some("int32"),
        "a typed alias resolves its declared int type"
    );
    // An f32 alias chain must NOT become integer-typed.
    let surf_f = "module M
def fbase() -> f32 = 1.5
falias = fbase
";
    let exprs_f = deep_of(surf_f);
    assert_eq!(
        crate::opaque::const_declared_int_type(&exprs_f, "falias"),
        None,
        "an f32 alias chain stays Real (None)"
    );
}
