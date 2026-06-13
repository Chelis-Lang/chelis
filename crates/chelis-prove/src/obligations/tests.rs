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
        decompose_return(&adt("Probability"), "Probability", &empty_records()).unwrap(),
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
    let pos = decompose_return(&ty, "Probability", &empty_records())
        .unwrap()
        .unwrap();
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
    assert!(decompose_return(&ty, "Probability", &empty_records()).is_err());
}

#[test]
fn decompose_type_absent_is_none() {
    assert_eq!(
        decompose_return(
            &Type::Prim(chelis_types::types::Prim::F32),
            "Probability",
            &empty_records()
        )
        .unwrap(),
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

fn empty_records() -> BTreeMap<String, Vec<Type>> {
    BTreeMap::new()
}

#[test]
fn record_wrapper_producer_is_covered_or_rejected() {
    // RT-2 CRITICAL: a non-generic record wrapping the opaque type in a
    // produced position must be a covered-or-rejected error, never a
    // silent miss.
    let surf = "module M
export (make_wrapped)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper = | Wrapper { inner: T }
def make_wrapped(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(
        col.errors.iter().any(|e| matches!(
            e,
            ObligationError::UnsupportedContainer { producer, container, .. }
                if producer == "make_wrapped" && container.contains("Wrapper")
        )),
        "record-wrapper producer must be covered-or-rejected, got {:?}",
        col.errors
    );
    assert!(
        col.obligations.is_empty(),
        "no silent obligation for the wrapper"
    );
}

#[test]
fn record_param_with_t_field_stays_legal() {
    // Module-receives: a record PARAMETER carrying T is legal (D-PRODUCER);
    // only produced positions and caller-receives reject.
    let surf = "module M
export (read_wrapped)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper = | Wrapper { inner: T }
def read_wrapped(w: Wrapper) -> f32 = 0.0
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(
        col.errors.is_empty(),
        "record param is module-receives: {:?}",
        col.errors
    );
    assert!(col.obligations.is_empty());
}

/// Assert that a producer reaching the opaque type through a record field
/// spelled with any type-alias chain is covered-or-rejected (RT3-F1), the
/// same as the direct-type case. The record field reads syntactically from
/// Deep, so alias resolution must happen there.
fn assert_record_alias_rejected(surf: &str, producer: &str) {
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(
        col.errors.iter().any(|e| matches!(
            e,
            ObligationError::UnsupportedContainer { producer: p, .. } if p == producer
        )),
        "alias-wrapped record producer `{producer}` must be covered-or-rejected, got {:?}",
        col.errors
    );
    assert!(
        col.obligations.is_empty(),
        "no silent obligation for the alias-wrapped record"
    );
}

#[test]
fn rt3_f1_record_field_typealias_is_covered_or_rejected() {
    // RT3-F1 CRITICAL: a record field spelled with a type alias of the
    // opaque type silently defeated the producer set (exit 0, zero
    // obligations) -- a violating T escaped. It must be covered-or-rejected
    // identically to the direct-type field.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type TA = T
type Wrapper = | Wrapper { inner: TA }
def make_w(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }
",
        "make_w",
    );
}

#[test]
fn rt3_f1_record_field_alias_of_alias_chain_is_rejected() {
    // TB = TA = T reached through a record field.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type TA = T
type TB = TA
type Wrapper = | Wrapper { inner: TB }
def make_w(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }
",
        "make_w",
    );
}

#[test]
fn rt3_f1_record_field_alias_inside_option_is_rejected() {
    // An alias reached through an Option field of a record container.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type TA = T
type Wrapper = | Wrapper { inner: Option[TA] }
def make_w(x: f32) -> Wrapper = Wrapper { inner: Some(T { value: 99.0 }) }
",
        "make_w",
    );
}

#[test]
fn rt3_f1_record_field_alias_inside_tuple_is_rejected() {
    // An alias reached through a tuple field of a record container.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type TA = T
type Wrapper = | Wrapper { inner: (TA, f32) }
def make_w(x: f32) -> Wrapper = Wrapper { inner: (T { value: 99.0 }, x) }
",
        "make_w",
    );
}

#[test]
fn rt3_f1_record_field_alias_nested_two_records_deep_is_rejected() {
    // An alias reached two record containers deep (Outer { mid: Mid { inner: TA } }).
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type TA = T
type Mid = | Mid { inner: TA }
type Outer = | Outer { mid: Mid }
def make_w(x: f32) -> Outer = Outer { mid: Mid { inner: T { value: 99.0 } } }
",
        "make_w",
    );
}

#[test]
fn cr15_multi_variant_non_first_variant_record_field_is_rejected() {
    // CR-15 HIGH: a producer returning a MULTI-VARIANT ADT whose NON-FIRST
    // variant wraps the opaque type in a record field was silently missed
    // because collect_record_fields recorded only the FIRST variant's
    // fields. It must be covered-or-rejected, same as the single-variant
    // wrapper.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper =
  | Empty { }
  | Full { inner: T }
def make_w(x: f32) -> Wrapper = Full { inner: T { value: 99.0 } }
",
        "make_w",
    );
}

#[test]
fn cr15_multi_variant_non_first_positional_payload_is_rejected() {
    // The opaque type as a POSITIONAL payload of a non-first variant.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper =
  | None
  | Some(T)
def make_w(x: f32) -> Wrapper = Some(T { value: 99.0 })
",
        "make_w",
    );
}

#[test]
fn cr15_multi_variant_non_first_variant_tuple_field_is_rejected() {
    // The opaque type inside a TUPLE field of a non-first variant.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper =
  | Empty { }
  | Pair { both: (T, f32) }
def make_w(x: f32) -> Wrapper = Pair { both: (T { value: 99.0 }, x) }
",
        "make_w",
    );
}

#[test]
fn cr15_first_variant_record_field_still_rejected() {
    // Negative-parity control: the opaque type in the FIRST variant must
    // also be rejected (it worked before CR-15 too; this guards against a
    // regression that only handles non-first variants).
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper =
  | Full { inner: T }
  | Empty { }
def make_w(x: f32) -> Wrapper = Full { inner: T { value: 99.0 } }
",
        "make_w",
    );
}
