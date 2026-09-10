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

fn traversal_module(aliases: usize, records: usize, carries_opaque: bool) -> String {
    let mut source = String::from(
        "module M\nexport (make_w)\n@opaque\n@invariant(p) p.value >= 0.0 && p.value <= 1.0\ntype T = | T { value: f32 }\n",
    );
    let mut ty = if carries_opaque { "T" } else { "f32" }.to_string();
    let mut value = if carries_opaque {
        "T { value: 99.0 }"
    } else {
        "99.0"
    }
    .to_string();
    for i in 0..aliases {
        source.push_str(&format!("type A{i} = {ty}\n"));
        ty = format!("A{i}");
    }
    for i in 0..records {
        source.push_str(&format!("type W{i} = | W{i} {{ inner: {ty} }}\n"));
        ty = format!("W{i}");
        value = format!("W{i} {{ inner: {value} }}");
    }
    source.push_str(&format!("def make_w(x: f32) -> {ty} = {value}\n"));
    source
}

#[test]
fn issue_872_alias_threshold_never_drops_a_producer() {
    assert_traversal_thresholds(&[(31, 1), (32, 1), (33, 1), (40, 1)]);
}

#[test]
fn issue_872_record_threshold_never_drops_a_producer() {
    assert_traversal_thresholds(&[(0, 15), (0, 16), (0, 17), (0, 24)]);
}

fn assert_traversal_thresholds(cases: &[(usize, usize)]) {
    for &(aliases, records) in cases {
        for carries_opaque in [true, false] {
            let exprs = deep_of(&traversal_module(aliases, records, carries_opaque));
            let invs = collect_opaque_invariants(&exprs);
            let col = collect_obligations(&exprs, &invs, &inferred_sigs(&exprs));
            assert_eq!(
                col.errors.len(),
                usize::from(carries_opaque),
                "aliases={aliases}, records={records}, opaque={carries_opaque}: {col:?}"
            );
            assert!(col.obligations.is_empty(), "{col:?}");
            if carries_opaque {
                assert!(
                    matches!(&col.errors[0], ObligationError::UnsupportedContainer { producer, .. } if producer == "make_w"),
                    "{col:?}"
                );
            }
        }
    }
}

#[test]
fn issue_872_direct_alias_producer_remains_obligated() {
    for aliases in [31, 32, 33, 40] {
        let exprs = deep_of(&traversal_module(aliases, 0, true));
        let col = collect_obligations(
            &exprs,
            &collect_opaque_invariants(&exprs),
            &inferred_sigs(&exprs),
        );
        assert!(col.errors.is_empty(), "{col:?}");
        assert_eq!(col.obligations.len(), 1, "{col:?}");
        assert_eq!(col.obligations[0].position, ProducedPosition::Direct);
    }
}

fn proof_node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(
        tag,
        Default::default(),
        children,
        chelis_deep::Span::new(0, 0),
    )
}

fn proof_name(name: &str) -> Expr {
    Expr::Atom(Atom::Name(name.to_string()), chelis_deep::Span::new(0, 0))
}

fn proof_adt(name: &str) -> Expr {
    proof_node(DeepTag::TAdt, vec![proof_name(name)])
}

// No parser or checker participates: this is collect_obligations' actual
// public embedding ingress, including cyclic raw alias/record declarations.
fn parser_free_cycle(
    opaque: bool,
    alias: bool,
) -> (Vec<Expr>, Vec<OpaqueInvariant>, BTreeMap<String, Type>) {
    let payload = if opaque {
        proof_adt("T")
    } else {
        proof_node(DeepTag::TUnit, vec![])
    };
    let mut exprs = vec![
        proof_node(DeepTag::Export, vec![proof_name("make")]),
        proof_node(
            DeepTag::Def,
            vec![proof_name("make"), proof_node(DeepTag::Tuple, vec![])],
        ),
    ];
    if alias {
        exprs.push(proof_node(
            DeepTag::Typealias,
            vec![
                proof_name("Cycle"),
                proof_node(DeepTag::Params, vec![]),
                proof_node(DeepTag::TTuple, vec![proof_adt("Cycle"), payload]),
            ],
        ));
    } else {
        exprs.push(proof_node(
            DeepTag::Deftype,
            vec![
                proof_name("Cycle"),
                proof_node(DeepTag::Params, vec![]),
                proof_node(
                    DeepTag::Variant,
                    vec![proof_name("Cycle"), proof_adt("Cycle"), payload],
                ),
            ],
        ));
    }
    let invs = vec![OpaqueInvariant {
        type_name: "T".into(),
        ctor_name: "T".into(),
        fields: vec![],
        predicate: proof_node(DeepTag::Tuple, vec![]),
        binder: "p".into(),
        amenability: chelis_pred::PredAmenability::Opaque,
    }];
    (exprs, invs, BTreeMap::from([("make".into(), adt("Cycle"))]))
}

#[test]
fn issue_872_parser_free_cycles_converge_without_hiding_opaque_fields() {
    for alias in [false, true] {
        for opaque in [false, true] {
            let (exprs, invs, sigs) = parser_free_cycle(opaque, alias);
            let col = collect_obligations(&exprs, &invs, &sigs);
            assert!(col.obligations.is_empty(), "{col:?}");
            assert_eq!(col.errors.len(), usize::from(opaque), "{col:?}");
            if opaque {
                assert!(
                    matches!(col.errors[0], ObligationError::UnsupportedContainer { .. }),
                    "{col:?}"
                );
            }
        }
    }
}

#[test]
fn issue_872_low_budget_is_a_collection_error_not_an_empty_success() {
    for opaque in [false, true] {
        let (exprs, invs, sigs) = parser_free_cycle(opaque, false);
        let col = collect_with_budget(&exprs, &invs, &sigs, 1);
        assert!(
            matches!(col.errors.as_slice(), [ObligationError::TypeTraversal {
            type_name, producer, source: TraversalError::Exhausted { limit: 1 }
        }] if type_name == "T" && producer == "make"),
            "{col:?}"
        );
        assert!(col.obligations.is_empty());
        // A completed no-opaque cycle is an honest empty set, not exhaustion.
        let completed = collect_with_budget(&exprs, &invs, &sigs, 100);
        assert_eq!(completed.errors.len(), usize::from(opaque), "{completed:?}");
    }
}

#[test]
fn issue_872_kinded_type_arguments_are_edges_but_dimensions_are_not() {
    use chelis_types::types::{Dim, NominalArg};
    let records = BTreeMap::new();
    let dims = vec![NominalArg::Dimension(Dim::Lit(3))];
    assert_eq!(
        decompose_return(
            &Type::KindedAdt("Sized".into(), dims.clone()),
            "T",
            &records
        )
        .unwrap(),
        None
    );
    for opaque in [false, true] {
        let mut args = dims.clone();
        args.push(NominalArg::Type(if opaque { adt("T") } else { Type::Unit }));
        let result = decompose_return(&Type::KindedAdt("Sized".into(), args), "T", &records);
        assert_eq!(result.is_err(), opaque, "{result:?}");
        let raw = proof_node(
            DeepTag::TAdt,
            vec![
                proof_name("Sized"),
                proof_node(DeepTag::DName, vec![proof_name("n")]),
                if opaque {
                    proof_adt("T")
                } else {
                    proof_node(DeepTag::TUnit, vec![])
                },
            ],
        );
        let mut graph = ProofTypes::new(BTreeMap::new(), BTreeMap::new(), Budget::default());
        let root = graph
            .project(Source::Deep(&raw))
            .expect("dimension arguments are not malformed types");
        assert_eq!(graph.decompose(root, "T").is_err(), opaque);
    }
}

#[test]
fn issue_872_supported_shared_alias_paths_keep_every_position() {
    let source = "module M\nexport (make)\n@opaque\n@invariant(p) p.value >= 0.0\ntype T = | T { value: f32 }\ntype A = Option[T]\ndef make(x: f32) -> (A, A) = (None, None)\n";
    let exprs = deep_of(source);
    let col = collect_obligations(
        &exprs,
        &collect_opaque_invariants(&exprs),
        &inferred_sigs(&exprs),
    );
    assert!(col.errors.is_empty(), "{col:?}");
    assert_eq!(col.obligations.len(), 1, "{col:?}");
    assert_eq!(
        col.obligations[0].position,
        ProducedPosition::TupleComponents(vec![
            (
                0,
                ProducedPosition::InsideOption(Box::new(ProducedPosition::Direct))
            ),
            (
                1,
                ProducedPosition::InsideOption(Box::new(ProducedPosition::Direct))
            ),
        ])
    );
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
  | Empty
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
  | Empty
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
  | Empty
def make_w(x: f32) -> Wrapper = Full { inner: T { value: 99.0 } }
",
        "make_w",
    );
}

#[test]
fn cr14_record_field_ref_to_opaque_is_covered_or_rejected() {
    // CR-14: a record field of type `&T` (a borrow of the opaque type) was
    // previously projected to an inert leaf, hiding the T inside the borrow
    // and silently missing the producer. The graph follows the `t-ref` edge,
    // so the record-with-ref-field producer is covered-or-rejected.
    assert_record_alias_rejected(
        "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper = | Wrapper { inner: &T }
def make_w(t: &T) -> Wrapper = Wrapper { inner: t }
",
        "make_w",
    );
}

#[test]
fn cr14_tensor_record_field_is_not_a_false_positive() {
    // Negative-parity: a record field of a plain numeric tensor (which
    // cannot contain the opaque type) must NOT be flagged -- the leaf
    // classification for `t-tensor` is safe. The producer returns a record with a
    // tensor field and no opaque type, so no obligation and no error.
    let surf = "module M
export (make_w)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Box = | Box { data: tensor[3, f32] }
def make_w(d: tensor[3, f32]) -> Box = Box { data: d }
";
    let exprs = deep_of(surf);
    let invs = collect_opaque_invariants(&exprs);
    let sigs = inferred_sigs(&exprs);
    let col = collect_obligations(&exprs, &invs, &sigs);
    assert!(
        col.errors.is_empty(),
        "a tensor-field record is not a false positive: {:?}",
        col.errors
    );
    assert!(col.obligations.is_empty());
}

// ── Review follow-up: covered-or-rejected. An invariant-carrying opaque type
//    the prover cannot model must be a REJECTION, never silently dropped.

#[test]
fn unmodelable_invariant_type_is_rejected_not_silently_dropped() {
    // A `string` scalar field is outside the prover's value class. The
    // collector must report it as a rejection, not drop it -- a dropped
    // invariant lets a VIOLATING producer pass with zero obligations.
    let exprs = deep_of(
        "module M\nexport (make)\n@opaque\n@invariant(p) p.value >= 0.0\n\
         type Tagged = | Tagged { tag: string, value: f32 }\n\
         def make() -> Tagged = Tagged { tag: \"bad\", value: 0.0 }",
    );
    let (oks, rejections) = crate::opaque::collect_opaque_invariants_and_rejections(&exprs);
    assert!(
        oks.is_empty(),
        "an unmodelable type must not be collected as an OK invariant: {oks:?}"
    );
    assert_eq!(
        rejections.len(),
        1,
        "an unmodelable invariant-carrying type must be a covered-or-rejected rejection: {rejections:?}"
    );
    assert_eq!(rejections[0].type_name, "Tagged");
}

#[test]
fn nested_record_invariant_type_is_modeled_as_a_record_field() {
    // A nested single-variant record field resolves to `FieldType::Record`, so
    // the invariant over `p.inner.value` is modelable (it was silently dropped
    // before, leaving a violating producer unchecked).
    let exprs = deep_of(
        "module M\nexport (make)\ntype Inner = | Inner { value: f32 }\n\
         @opaque\n@invariant(p) p.inner.value >= 0.0\n\
         type Boxed = | Boxed { inner: Inner }\n\
         def make() -> Boxed = Boxed { inner: Inner { value: 0.0 } }",
    );
    let (oks, rejections) = crate::opaque::collect_opaque_invariants_and_rejections(&exprs);
    assert!(
        rejections.is_empty(),
        "a nested record of value-class fields must be modeled, not rejected: {rejections:?}"
    );
    assert_eq!(oks.len(), 1);
    assert!(
        matches!(
            oks[0].fields.first(),
            Some((name, crate::opaque::FieldType::Record(_))) if name == "inner"
        ),
        "the `inner` field must model as a nested Record: {:?}",
        oks[0].fields
    );
}
