use super::*;
use chelis_pred::PredAmenability;

/// Desugar a Surf module string to Deep for collection tests.
fn deep_of(surf: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(surf).expect("parse surf");
    chelis_surf::desugar::desugar_program(&decls)
}

const PROB: &str = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";

#[test]
fn collects_opaque_invariant_with_record_field() {
    let exprs = deep_of(PROB);
    let invs = collect_opaque_invariants(&exprs);
    assert_eq!(invs.len(), 1);
    let inv = &invs[0];
    assert_eq!(inv.type_name, "Probability");
    assert_eq!(inv.ctor_name, "Probability");
    assert_eq!(inv.binder, "p");
    assert_eq!(inv.fields.len(), 1);
    assert_eq!(inv.fields[0].0, "value");
    assert_eq!(inv.fields[0].1, FieldType::Scalar("f32".to_string()));
    assert_eq!(inv.amenability, PredAmenability::Linear);
    assert_eq!(inv.scalar_count(), 1);
}

#[test]
fn skips_opaque_without_invariant() {
    // Plain opacity (no @invariant) is unaffected by injection (D-INJECT).
    let surf = "module M.Plain
@opaque
type Token =
  | Token { id: int32 }
def make(i: int32) -> Token = Token { id: i }
";
    let exprs = deep_of(surf);
    assert!(collect_opaque_invariants(&exprs).is_empty());
}

#[test]
fn skips_non_opaque_type() {
    let surf = "module M.Open
type Pair =
  | Pair { a: f32, b: f32 }
";
    let exprs = deep_of(surf);
    assert!(collect_opaque_invariants(&exprs).is_empty());
}

#[test]
fn recomputes_amenability_not_trusting_recorded_metadata() {
    // Polynomial predicate: product of two binder fields.
    let surf = "module M.Poly
@opaque
@invariant(p) p.a * p.b >= 0.0
type Poly =
  | Poly { a: f32, b: f32 }
def make(a: f32, b: f32) -> Poly = Poly { a: a, b: b }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].amenability, PredAmenability::Polynomial);
}

#[test]
fn lowers_linear_predicate_to_flattened_smt() {
    let exprs = deep_of(PROB);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let smt = lower_predicate_flattened(inv, "p", &ConstEnv::new()).expect("lowerable");
    // The lowered form must reference the flattened var "p.value".
    let s = format!("{smt:?}");
    assert!(s.contains("p.value"), "expected flattened var, got {s}");
}

#[test]
fn flattened_predicate_validates_with_concrete_eval() {
    use crate::concrete_eval::eval_bool;
    use std::collections::HashMap;
    let exprs = deep_of(PROB);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let smt = lower_predicate_flattened(inv, "p", &ConstEnv::new()).expect("lowerable");
    let mut env: HashMap<String, f64> = HashMap::new();
    env.insert("p.value".to_string(), 0.5);
    assert!(eval_bool(&smt, &env), "0.5 satisfies 0<=v<=1");
    env.insert("p.value".to_string(), 1.5);
    assert!(!eval_bool(&smt, &env), "1.5 violates v<=1");
    env.insert("p.value".to_string(), -0.1);
    assert!(!eval_bool(&smt, &env), "-0.1 violates 0<=v");
}

#[test]
fn lowers_sum_with_constant_for_simplex_band() {
    use crate::concrete_eval::eval_bool;
    use std::collections::HashMap;
    // Tolerance-band simplex invariant referencing a module constant `eps`.
    let surf = "module M.Simplex
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def make(w: tensor[3, f32]) -> Simplex = Simplex { weights: w }
";
    let exprs = deep_of(surf);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let mut consts = ConstEnv::new();
    consts.insert("eps".to_string(), 0.001);
    let smt = lower_predicate_flattened(inv, "p", &consts).expect("lowerable with sum+const");
    let s = format!("{smt:?}");
    assert!(s.contains("p.weights.0"), "expected expanded sum: {s}");
    assert!(s.contains("p.weights.2"), "expected all 3 elements: {s}");

    // A weight vector summing to exactly 1.0 satisfies the band.
    let mut env: HashMap<String, f64> = HashMap::new();
    env.insert("p.weights.0".to_string(), 0.2);
    env.insert("p.weights.1".to_string(), 0.3);
    env.insert("p.weights.2".to_string(), 0.5);
    assert!(eval_bool(&smt, &env), "sum 1.0 is within band");
    // A vector summing to 1.5 violates the upper band.
    env.insert("p.weights.2".to_string(), 1.0);
    assert!(!eval_bool(&smt, &env), "sum 1.5 violates band");
}

#[test]
fn u2_int_field_constant_in_precondition_lowers_int_not_real() {
    // U2 (precondition path): an int-field invariant comparing the field
    // against an int-typed module constant must lower the constant as an
    // INTEGER, not a Real, so the flattened precondition does not mix Int and
    // Real sorts against the int-sorted field var (which would abort cvc5 in
    // the update-shaped inductive step where this predicate is a precondition).
    let surf = "module M
@opaque
@invariant(c) c.n >= lo
type Counter =
  | Counter { n: int32 }
def make(x: int32) -> Counter = Counter { n: x }
";
    let exprs = deep_of(surf);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let mut consts = ConstEnv::new();
    consts.insert("lo".to_string(), 0.0);
    // The cvc5 precondition path passes the defining program, so the
    // constant lowers with its declared type via the shared resolver.
    let smt = lower_predicate_flattened_in(inv, "c", &consts, &exprs).expect("lowerable");
    let s = format!("{smt:?}");
    assert!(
        s.contains("IntLit"),
        "int constant against an int field lowers as IntLit: {s}"
    );
    assert!(
        !s.contains("RealLit"),
        "no RealLit against an Int-sorted field var: {s}"
    );
}

#[test]
fn unresolved_constant_makes_predicate_not_lowerable() {
    let surf = "module M.Simplex
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def make(w: tensor[3, f32]) -> Simplex = Simplex { weights: w }
";
    let exprs = deep_of(surf);
    let inv = &collect_opaque_invariants(&exprs)[0];
    // No `eps` in the const env => not lowerable (falls to Tier C).
    assert!(lower_predicate_flattened(inv, "p", &ConstEnv::new()).is_none());
}

#[test]
fn tensor_field_in_value_class() {
    let surf = "module M.Simplex
@opaque
@invariant(p) sum(p.weights) >= 0.0
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def make(w: tensor[3, f32]) -> Simplex = Simplex { weights: w }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    assert_eq!(invs.len(), 1);
    assert_eq!(
        invs[0].fields[0].1,
        FieldType::Tensor {
            dims: vec![3],
            precision: "f32".to_string()
        }
    );
    assert_eq!(invs[0].scalar_count(), 3);
}

#[test]
fn cr2_2_validate_env_rejects_non_finite_under_negation_invariant() {
    // CR2-2 HIGH (injection path): a `!=`-shaped invariant satisfied by
    // `NaN != C == true` would accept a non-finite sample fail-OPEN.
    // `validate_env` must reject any non-finite field unconditionally,
    // independent of the predicate's truth value.
    let surf = "module M.Prob
@opaque
@invariant(p) p.value != 0.5
type Probability =
  | Probability { value: f32 }
def make(x: f32) -> Probability = Probability { value: x }
";
    let exprs = deep_of(surf);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let consts = ConstEnv::new();
    let pred = lower_predicate_flattened(inv, &inv.binder, &consts);

    // A finite value that satisfies `!= 0.5` is accepted.
    let mut ok = BTreeMap::new();
    ok.insert("p.value".to_string(), 0.25);
    assert!(
        validate_env(&ok, inv, &pred, &consts),
        "a finite satisfying value is accepted"
    );

    // NaN: `NaN != 0.5` is true under strict IEEE, but the field is
    // non-finite, so it must be REJECTED (fail-closed), not accepted.
    let mut nan = BTreeMap::new();
    nan.insert("p.value".to_string(), f64::NAN);
    assert!(
        !validate_env(&nan, inv, &pred, &consts),
        "a NaN field is rejected even though `NaN != 0.5` is true"
    );

    // +Inf is likewise non-finite and rejected.
    let mut inf = BTreeMap::new();
    inf.insert("p.value".to_string(), f64::INFINITY);
    assert!(
        !validate_env(&inf, inv, &pred, &consts),
        "an infinite field is rejected"
    );
}
