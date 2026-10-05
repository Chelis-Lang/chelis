//! Spec-first corpus for [03-META-1/2], chelis#1478 and chelis#1330.
use chelis_deep::parser::{parse_raw_str, parse_str};
use chelis_deep::{Atom, DeepTag, Expr, Metadata, Span, node::Node};

use chelis_deep::annotations::{
    MetadataValue as M, PipeStageOrigin, PositiveInteger, RuntimeExpression, SpanId, Spanned,
    TypeSyntax,
};

const ZERO: Span = Span { offset: 0, len: 0 };

// Each row independently states a normative key and an admissible owner/value.
const CASES: &[(&str, &str)] = &[
    ("type", "(var {type: (t-prim {} f32)} x)"),
    ("loc", "(var {loc: (loc \"source.dp\" 1 2)} x)"),
    ("eff", "(t-fn {eff: (effects {})} (t-unit {}))"),
    (
        "dtype_bounds",
        "(defsig {dtype_bounds: {p: float}} f (p) (t-var {} p))",
    ),
    (
        "effects",
        "(fn {effects: (effects {})} (params {}) (lit {} 1))",
    ),
    ("source", "(var {source: (macro_name x)} x)"),
    ("wrt", "(grad {wrt: (var {} x)} (var {} f))"),
    ("span", "(var {span: \"\"} x)"),
    (
        "chelis_role",
        "(def {chelis_role: \"custom\"} f (lit {} 1))",
    ),
    (
        "property_source_kind",
        "(def {property_source_kind: \"user\"} f (lit {} 1))",
    ),
    (
        "property_quantifiers",
        "(def {property_quantifiers: (params {} x)} f (lit {} 1))",
    ),
    (
        "property_preconditions",
        "(def {property_preconditions: (tuple {} true)} f (lit {} 1))",
    ),
    (
        "property_source_id",
        "(def {property_source_id: \"ears:1\"} f (lit {} 1))",
    ),
    (
        "property_tolerance",
        "(def {property_tolerance: (var {} tolerance)} f (lit {} 1))",
    ),
    (
        "property_seed",
        "(def {property_seed: (lit {} 1)} f (lit {} 1))",
    ),
    (
        "property_samples",
        "(def {property_samples: (lit {} 2)} f (lit {} 1))",
    ),
    (
        "property_contracts",
        "(def {property_contracts: (tuple {} \"a\" \"b\")} f (lit {} 1))",
    ),
    ("opaque", "(deftype {opaque: true} T () (variant {} T))"),
    (
        "invariant",
        "(deftype {opaque: true, invariant: (fn {} (params {} x) true)} T () (variant {} T))",
    ),
    (
        "invariant_amenability",
        "(deftype {opaque: true, invariant: (fn {} (params {} x) true), invariant_amenability: \"linear\"} T () (variant {} T))",
    ),
    ("surf_path", "(module {surf_path: \"M.Path\"} m.path)"),
    (
        "surf_dim_group_size",
        "(defdim {surf_dim_group_size: 2} n) (defdim {} m)",
    ),
    (
        "surf_pipe_stage",
        "(pipe {} 1 (fn {surf_pipe_stage: \"call-first\"} (params {} x) (var {} x)))",
    ),
    (
        "surf_literal_style",
        "(lit {surf_literal_style: \"explicit\"} 1)",
    ),
    (
        "surf_binding_type",
        "(bind {} x (lit {surf_binding_type: \"inferred\"} 1))",
    ),
    ("lin", "(var {lin: borrow} x)"),
    ("doc", "(var {doc: \"documentation\"} x)"),
    (
        "effect",
        "(handle-effect {effect: resource} (lit {} 1) (lit {} 2))",
    ),
    ("literal_source", "(lit {literal_source: integer} 1)"),
    ("destructure", "(bind {destructure: true} x (lit {} 1))"),
    (
        "accumulator",
        "(app {accumulator: (t-prim {} f64)} (var {} sum) (var {} x) (lit {} 0))",
    ),
];

fn metadata_value_span(expr: &chelis_deep::RawExpr, key: &str) -> Option<Span> {
    use chelis_deep::RawExpr;
    match expr {
        RawExpr::Map(entries, _) | RawExpr::MetaExpr { entries, .. } => {
            entries.iter().find_map(|(k, v)| {
                if k == key {
                    Some(v.span())
                } else {
                    metadata_value_span(v, key)
                }
            })
        }
        RawExpr::List(items, _) => items.iter().find_map(|v| metadata_value_span(v, key)),
        RawExpr::Atom(..) | RawExpr::ExtensionData(_) => None,
    }
}

#[test]
fn every_declared_key_has_positive_and_negative_shape_coverage() {
    for (key, source) in CASES {
        parse_raw_str(source).unwrap_or_else(|e| panic!("raw {key}: {e}"));
        parse_str(source).unwrap_or_else(|e| panic!("stamped {key}: {e}"));
        let raw = parse_raw_str(source).unwrap();
        let value = raw
            .iter()
            .find_map(|expr| metadata_value_span(expr, key))
            .unwrap();
        let bad = format!(
            "{}metadata_bad_value{}",
            &source[..value.offset],
            &source[value.end()..]
        );
        let error = parse_raw_str(&bad).expect_err(key).to_string();
        assert!(error.contains(key), "{key}: {error}");
    }
}

#[test]
fn issue_1478_paths_fail_closed() {
    for value in ["1", "(lit {} 1)", "\"Totally.Different\""] {
        let source = format!("(module {{surf_path: {value}}} m.path)");
        let error = chelis_deep::parse_and_stamp_file(&source)
            .unwrap_err()
            .to_string();
        assert!(error.contains("surf_path"), "{error}");
    }
    chelis_deep::parse_and_stamp_file("(module {surf_path: \"M.Path\"} m.path)").unwrap();
}

#[test]
fn metadata_expression_roles_reject_bare_names_with_remediation() {
    for key in [
        "wrt",
        "property_tolerance",
        "property_seed",
        "property_samples",
    ] {
        let source = if key == "wrt" {
            "(def {} g (grad {wrt: unwrapped} (var {} f)))".to_string()
        } else {
            format!("(def {{{key}: unwrapped}} f (lit {{}} 1))")
        };
        let error = chelis_deep::parse_and_stamp_file(&source)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(key)
                && error.contains("unwrapped")
                && error.contains("(var {} unwrapped)"),
            "{error}"
        );
    }
    for value in ["(var {} x)", "(tuple {} (var {} x) (var {} y))"] {
        parse_str(&format!("(grad {{wrt: {value}}} (var {{}} f))")).unwrap();
    }
    for value in [
        "(tuple {})",
        "(tuple {} (lit {} 1))",
        "(tuple {} x y)",
        "(lit {} 1)",
    ] {
        assert!(parse_str(&format!("(grad {{wrt: {value}}} (var {{}} f))")).is_err());
    }
}

#[test]
fn data_and_provenance_are_not_metadata_or_expression_roles() {
    parse_str(
        "(defsig {dtype_bounds: {type: float, span: int, surf_future: numeric}} f (type span surf_future) (t-var {} type))",
    )
    .unwrap();
    let source = "(var {source: (macro_name {surf_future: 1, span: 2} bare_name), custom: {nested: (lit {span: \"id\"} 1)}, span_future: (a b)} x)";
    parse_str(source).unwrap();
    assert!(parse_str("(var {custom: {nested: (lit {span: 1} 1)}} x)").is_ok());
}

#[test]
fn constructors_and_transactional_replacement_enforce_local_contracts() {
    let name = Expr::Atom(Atom::Name("x".into()), ZERO);
    assert!(SpanId::try_new("invalid\nspan".into(), ZERO).is_err());
    let bad = Metadata::from(M::SurfPath(Spanned::new("Wrong.Owner".into(), ZERO)));
    assert!(Node::try_new(DeepTag::Var, bad.clone(), vec![name.clone()]).is_err());
    let mut node = Node::try_new(DeepTag::Var, Metadata::default(), vec![name]).unwrap();
    let before = node.clone();
    assert!(node.try_replace_meta(bad).is_err());
    assert_eq!(node, before);
    let mut wire = serde_json::to_value(&node).unwrap();
    wire["meta"] = serde_json::json!({"entries": [["span", Expr::Atom(Atom::Int(1), ZERO)]]});
    assert!(serde_json::from_value::<Node>(wire).is_err());
}

#[test]
fn placements_and_singleton_keys_are_validated() {
    for source in [
        "(var {span: \"a\", span: \"b\"} x)",
        "(var {surf_path: \"X\"} x)",
        "{surf_literal_style: \"explicit\"}",
        "(fn {surf_pipe_stage: \"call-first\"} (params {} x) (var {} x))",
        "(lit {surf_binding_type: \"inferred\"} 1)",
        "(defdim {surf_dim_group_size: 2} n)",
        "(defdim {surf_dim_group_size: 2} n) (defdim {surf_dim_group_size: 1} m)",
    ] {
        assert!(parse_str(source).is_err(), "{source}");
    }
}

#[test]
fn registry_and_corpus_cover_exactly_the_normative_inventory() {
    let spec = include_str!("../../../spec/03-deep-syntax.md");
    let section = spec
        .split("**Defined keys:**")
        .nth(1)
        .unwrap()
        .split("> **[03-META-1]**")
        .next()
        .unwrap();
    let mut normative: Vec<_> = section
        .lines()
        .filter_map(|line| line.strip_prefix("| `")?.split('`').next())
        .filter(|key| *key != "span_*")
        .collect();
    normative.sort_unstable();
    let mut registered = chelis_deep::metadata::REGISTERED_METADATA_KEYS.to_vec();
    registered.sort_unstable();
    let mut corpus: Vec<_> = CASES.iter().map(|(key, _)| *key).collect();
    corpus.sort_unstable();
    assert_eq!(registered, normative);
    assert_eq!(corpus, normative);
}

#[test]
fn legacy_programmatic_carriers_cannot_hide_malformed_metadata() {
    for source in [
        "(var {span: 1} x)",
        "(def {property_seed: (lit {span: 1} 2)} x 1)",
        "(future {property_seed: (lit {span: 1} 2)} (var {} x))",
        "(var {span: \"ok\", span: 2} x)",
    ] {
        let error = parse_str(source).unwrap_err().to_string();
        assert!(
            error.contains("span") || error.contains("property_seed"),
            "{source}: {error}"
        );
    }
}

#[test]
fn parent_placement_is_checked_without_rejecting_unattached_fragments() {
    let marked = Node::try_new(
        DeepTag::Fn,
        Metadata::from(M::SurfPipeStage(Spanned::new(
            PipeStageOrigin::CallFirst,
            ZERO,
        ))),
        vec![
            Expr::node(DeepTag::Params, Metadata::default(), vec![], ZERO),
            Expr::Atom(Atom::Int(1), ZERO),
        ],
    )
    .unwrap();
    let marked = Expr::Node(Box::new(marked), ZERO);
    assert!(chelis_deep::metadata::validate_metadata(std::slice::from_ref(&marked)).is_err());
    let mut pipe = Node::try_new(
        DeepTag::Pipe,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(1), ZERO), marked.clone()],
    )
    .unwrap();
    let before = pipe.clone();
    assert!(pipe.try_replace_child(0, marked).is_err());
    assert_eq!(pipe, before);
    chelis_deep::metadata::validate_metadata(&[Expr::Node(Box::new(pipe), ZERO)]).unwrap();
}

#[test]
fn each_registered_key_rejects_duplicates_at_the_second_value() {
    for (key, source) in CASES {
        let raw = parse_raw_str(source).unwrap();
        let span = raw
            .iter()
            .find_map(|v| metadata_value_span(v, key))
            .unwrap();
        let original = &source[span.offset..span.end()];
        let bad = format!(
            "{}, {key}: {original}{}",
            &source[..span.end()],
            &source[span.end()..]
        );
        let error = parse_raw_str(&bad).unwrap_err();
        assert!(error.to_string().contains(key), "{error}");
        let chelis_deep::parser::ParseError::Metadata(error) = error else {
            panic!("{error}");
        };
        assert_eq!(error.span.offset, span.end() + key.len() + 4);
    }
}

#[test]
fn structured_shapes_reject_wrong_members_and_closed_enum_values() {
    for source in [
        "(def {property_source_kind: \"other\"} f (lit {} 1))",
        "(def {property_contracts: (tuple {} \"a\" (lit {} 1))} f (lit {} 1))",
        "(def {property_quantifiers: (params {} 1)} f (lit {} 1))",
        "(def {property_preconditions: (tuple {} unwrapped)} f (lit {} 1))",
        "(deftype {opaque: false} T () (variant {} T))",
        "(deftype {invariant: (fn {} (params {} x) true)} T () (variant {} T))",
        "(deftype {opaque: true, invariant_amenability: \"linear\"} T () (variant {} T))",
        "(var {loc: (loc \"a.dp\" 1.0 2)} x)",
        "(var {type: (t-prim {} (lit {} 1))} x)",
        "(defsig {dtype_bounds: {p: int, p: float}} f (p) (t-var {} p))",
        "(defsig {dtype_bounds: {p: {nested: float}}} f (p) (t-var {} p))",
    ] {
        assert!(parse_str(source).is_err(), "{source}");
    }
}

#[test]
fn programmatic_type_metadata_obeys_recursive_type_roles() {
    let malformed_type = Expr::node(
        DeepTag::TPrim,
        Metadata::default(),
        vec![Expr::node(
            DeepTag::Lit,
            Metadata::default(),
            vec![Expr::Atom(Atom::Int(1), ZERO)],
            ZERO,
        )],
        ZERO,
    );
    assert!(TypeSyntax::try_new(malformed_type).is_err());
}

#[test]
fn property_metadata_has_its_required_fields_and_matching_binders() {
    let valid = "(def {chelis_role: \"property\", property_source_kind: \"user\", property_quantifiers: (params {} (x {type: (t-prim {} f32)})), property_preconditions: (tuple {})} p (fn {} (params {} (x {type: (t-prim {} f32)})) true))";
    parse_str(valid).unwrap();
    for bad in [
        valid.replacen("property_source_kind: \"user\", ", "", 1),
        valid.replacen(
            "property_quantifiers: (params {} (x",
            "property_quantifiers: (params {} (y",
            1,
        ),
        valid.replacen("f32", "i32", 1),
    ] {
        assert!(parse_str(&bad).is_err(), "{bad}");
    }
}

#[test]
fn declaration_roles_and_property_binder_spellings_follow_the_contract() {
    parse_str("(defsig {chelis_role: \"producer_signature\"} f (t-prim {} f32))").unwrap();
    for source in [
        "(var {chelis_role: \"producer_signature\"} f)",
        "(defsig {chelis_role: \"property\"} f (t-prim {} f32))",
    ] {
        assert!(parse_str(source).is_err(), "{source}");
    }
    let source = "(def {chelis_role: \"property\", property_source_kind: \"user\", property_quantifiers: (params {} ^{:type (t-prim {} f32)} x), property_preconditions: (tuple {})} p (fn {} (params {} (x {type: (t-prim {} f32), doc: \"binder\"})) true))";
    parse_raw_str(source).unwrap();
    parse_str(source).unwrap();
    assert!(parse_str(&source.replacen("f32", "i32", 1)).is_err());
}

#[test]
fn each_registered_key_rejects_invalid_serialized_replacement() {
    fn find<'a>(expr: &'a Expr, key: &str) -> Option<&'a Node> {
        let Expr::Node(node, _) = expr else {
            return None;
        };
        if node
            .meta()
            .values()
            .any(|value| value.key().spelling() == key)
        {
            return Some(node);
        }
        node.children_slice().iter().find_map(|v| find(v, key))
    }
    for (key, source) in CASES {
        let parsed = parse_str(source).unwrap();
        let node = parsed.iter().find_map(|v| find(v, key)).expect(key).clone();
        let before = node.clone();
        let mut wire = serde_json::to_value(node.meta()).unwrap();
        let entry = wire["entries"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry[0] == *key)
            .unwrap();
        entry[1] = serde_json::to_value(Expr::Atom(Atom::Name("unwrapped".into()), ZERO)).unwrap();
        let error = serde_json::from_value::<Metadata>(wire).expect_err(key);
        assert!(error.to_string().contains(key), "{error}");
        assert_eq!(
            node, before,
            "{key}: failed replacement must preserve the node"
        );
    }
}

#[test]
fn module_construction_checks_the_siblings_it_already_contains() {
    let mut first = Node::try_new(
        DeepTag::Defdim,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("n".into()), ZERO)],
    )
    .unwrap();
    first
        .try_replace_meta(Metadata::from(M::SurfDimGroupSize(
            PositiveInteger::try_new(Expr::Atom(Atom::Int(2), ZERO)).unwrap(),
        )))
        .unwrap();
    let mut children = vec![
        Expr::Atom(Atom::Name("m".into()), ZERO),
        Expr::Node(Box::new(first), ZERO),
    ];
    assert!(Node::try_new(DeepTag::Module, Metadata::default(), children.clone()).is_err());
    children.push(Expr::node(
        DeepTag::Defdim,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("k".into()), ZERO)],
        ZERO,
    ));
    Node::try_new(DeepTag::Module, Metadata::default(), children).unwrap();
}

#[test]
fn resource_effect_metadata_requires_string_devices_at_every_ingress() {
    for (owner, key, children) in [
        ("t-fn", "eff", "(t-prim {} i32)"),
        ("fn", "effects", "(params {}) (lit {} 1)"),
    ] {
        for device in ["\"gpu:0\"", "42", "unwrapped", "(lit {} 42)"] {
            let source =
                format!("({owner} {{{key}: (effects {{}} (resource {{}} {device}))}} {children})");
            let valid = device == "\"gpu:0\"";
            assert_eq!(parse_raw_str(&source).is_ok(), valid, "raw: {source}");
            assert_eq!(parse_str(&source).is_ok(), valid, "stamped: {source}");
            if valid {
                let parsed = parse_str(&source).unwrap();
                let wire = serde_json::to_value(&parsed).unwrap();
                assert_eq!(serde_json::from_value::<Vec<Expr>>(wire).unwrap(), parsed);
            }
        }
    }
}

/// A nested runtime child cannot hide below a typed ancestor in an
/// unvalidated carrier: the only spelling of a nested `app` is a validated
/// `Node`, which refuses a bare name at a runtime position when it is built,
/// so every payload the constructor sees already carries its roles.
#[test]
fn expression_metadata_nested_runtime_roles_are_enforced_at_construction() {
    for key in [
        "property_seed",
        "property_samples",
        "property_tolerance",
        "property_preconditions",
    ] {
        let child = Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("missing".into()), ZERO)],
            ZERO,
        );
        let app = Expr::node(DeepTag::App, Metadata::default(), vec![child], ZERO);
        let mut payload = Expr::node(DeepTag::Tuple, Metadata::default(), vec![app], ZERO);
        if key == "property_preconditions" {
            payload = Expr::node(DeepTag::Tuple, Metadata::default(), vec![payload], ZERO);
        }
        let admitted = RuntimeExpression::try_new(payload);
        assert!(admitted.is_ok(), "constructor {key}: {admitted:?}");
    }
    assert!(matches!(
        Node::try_new(
            DeepTag::App,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("missing".into()), ZERO)],
        ),
        Err(chelis_deep::node::NodeError::NameAtExprSlot { .. })
    ));
}

#[test]
fn metadata_expressions_follow_runtime_roles_through_helpers() {
    for payload in [
        "(fn {} (params {} x) (app {} (var {} missing)))",
        "(let {} (bind {} x (app {} (var {} missing))) (var {} x))",
        "(match {} 1 (arm {} (pat-wild {}) () (app {} (var {} missing))))",
        "(record {} T (kv {} x (app {} (var {} missing))))",
        "(pipe {} 1 (app {} (var {} missing)))",
        "(app {} (var {} missing))",
    ] {
        let valid = parse_str(payload).unwrap().remove(0);
        assert!(RuntimeExpression::try_new(valid).is_ok());
        let invalid = payload.replace("(var {} missing)", "missing");
        let source = format!("(def {{property_seed: {invalid}}} f 1)");
        assert!(parse_str(&source).is_err(), "{source}");
    }
}
