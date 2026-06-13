use super::*;
use crate::opaque::collect_opaque_invariants;
use chelis_types::types::Type;
use std::collections::BTreeMap;

/// Desugar a Surf module string to Deep.
fn deep_of(surf: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(surf).expect("parse surf");
    chelis_surf::desugar::desugar_program(&decls)
}

/// Run the checker and build the def-name -> inferred-type map the
/// obligation collector consumes (the D-PRODUCER inferred-type path).
fn inferred_sigs(exprs: &[Expr]) -> BTreeMap<String, Type> {
    let checked = chelis_types::check_typed_program(exprs)
        .unwrap_or_else(|e| panic!("check failed: {:?}", e.errors));
    checked
        .signature_inference()
        .functions
        .iter()
        .map(|(name, sig)| (name.clone(), sig.checked_signature.clone()))
        .collect()
}

const PROB: &str = "module Stats.Prob
export (probability, prob_value)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
def prob_value(p: Probability) -> f32 = p.value
";

#[test]
fn direct_producer_yields_obligation() {
    let exprs = deep_of(PROB);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(col.errors.is_empty(), "no errors: {:?}", col.errors);
    // `probability` is a direct producer; `prob_value` returns f32 (not a
    // producer); both are exported.
    let names: Vec<_> = col.obligations.iter().map(|o| o.name.as_str()).collect();
    assert!(names.contains(&"invariant:Probability:probability"));
    assert!(!names.iter().any(|n| n.contains("prob_value")));
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "probability")
        .unwrap();
    assert_eq!(ob.position, ProducedPosition::Direct);
    assert_eq!(ob.meta.obligation_kind, "invariant_producer");
    assert_eq!(ob.meta.source_type, "Probability");
}

#[test]
fn option_producer_decomposes() {
    let surf = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(col.errors.is_empty(), "no errors: {:?}", col.errors);
    let ob = col
        .obligations
        .iter()
        .find(|o| o.producer == "probability")
        .expect("option producer is an obligation");
    assert_eq!(
        ob.position,
        ProducedPosition::InsideOption(Box::new(ProducedPosition::Direct))
    );
}

#[test]
fn unannotated_producer_still_in_set_via_inferred_return() {
    // No `-> Probability` annotation; the inferred return must still place
    // it in the producer set (D-PRODUCER inferred-type path).
    let surf = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) = Probability { value: x }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(col.errors.is_empty(), "no errors: {:?}", col.errors);
    assert!(
        col.obligations.iter().any(|o| o.producer == "probability"),
        "inferred-return producer must be in the set"
    );
}

#[test]
fn no_export_means_zero_obligations() {
    let surf = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(col.obligations.is_empty(), "no export => no producers");
    assert!(col.errors.is_empty());
}

// --- direct decompose unit tests over Type ---

fn adt(name: &str) -> Type {
    Type::Adt(name.to_string(), vec![])
}
fn opt(inner: Type) -> Type {
    Type::Adt("Option".to_string(), vec![inner])
}

#[test]
fn decompose_direct() {
    assert_eq!(
        decompose_return(&adt("Probability"), "Probability").unwrap(),
        Some(ProducedPosition::Direct)
    );
}

#[test]
fn decompose_option_of_tuple() {
    // Option[(Probability, f32)]
    let ty = opt(Type::Tuple(vec![
        adt("Probability"),
        Type::Prim(chelis_types::types::Prim::F32),
    ]));
    let pos = decompose_return(&ty, "Probability").unwrap().unwrap();
    assert_eq!(
        pos,
        ProducedPosition::InsideOption(Box::new(ProducedPosition::TupleComponents(vec![(
            0,
            ProducedPosition::Direct
        )])))
    );
}

#[test]
fn decompose_unsupported_container_rejects() {
    // List[Probability] is an unsupported container.
    let ty = Type::Adt("List".to_string(), vec![adt("Probability")]);
    assert!(decompose_return(&ty, "Probability").is_err());
}

#[test]
fn decompose_type_absent_is_none() {
    assert_eq!(
        decompose_return(&Type::Prim(chelis_types::types::Prim::F32), "Probability").unwrap(),
        None
    );
}

#[test]
fn caller_receives_function_param_rejected() {
    // Exported `with_t(f: Probability -> f32) -> f32`: the domain of `f`
    // is caller-receives.
    let surf = "module Stats.Prob
export (with_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def with_prob(f: (Probability) -> f32) -> f32 = f(default_prob())
def default_prob() -> Probability = Probability { value: 0.0 }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(
        col.errors
            .iter()
            .any(|e| matches!(e, ObligationError::CallerReceives { function, .. } if function == "with_prob")),
        "caller-receives signature must be rejected, got {:?}",
        col.errors
    );
}

#[test]
fn module_receives_record_param_legal() {
    // A record parameter with a T field is module-receives (legal); the
    // function returns f32 (not a producer). No error, no obligation.
    let surf = "module Stats.Prob
export (read_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def read_prob(p: Probability) -> f32 = prob_value(p)
def prob_value(p: Probability) -> f32 = p.value
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(
        col.errors.is_empty(),
        "module-receives is legal: {:?}",
        col.errors
    );
}
