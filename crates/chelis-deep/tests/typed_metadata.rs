//! [03-META-1/2]: annotation admission, distinct from preserved syntax data.
use chelis_deep::parser::parse_str;

#[test]
fn every_annotation_key_is_unique_including_extensions() {
    for key in [
        "custom",
        "span_file",
        "literal_source",
        "effect",
        "destructure",
    ] {
        let (tag, value, children) = match key {
            "literal_source" => ("lit", "integer", "1"),
            "effect" => ("handle-effect", "resource", "(lit {} 1) (lit {} 2)"),
            "destructure" => ("bind", "true", "x (lit {} 1)"),
            _ => ("var", "\"a\"", "x"),
        };
        let single = format!("({tag} {{{key}: {value}}} {children})");
        parse_str(&single).unwrap_or_else(|e| panic!("{single}: {e}"));
        let duplicate = format!("({tag} {{{key}: {value}, {key}: {value}}} {children})");
        let error = parse_str(&duplicate).expect_err(&duplicate).to_string();
        assert!(error.contains(key), "{error}");
    }
}

#[test]
fn additional_core_payloads_reject_bad_shapes_and_owners() {
    for source in [
        "(lit {literal_source: float} 1)",
        "(var {literal_source: integer} x)",
        "(handle-effect {effect: imaginary} (lit {} 1) (lit {} 2))",
        "(var {effect: resource} x)",
        "(handle-effect {effect: random} (lit {} 1) (lit {} 2))",
        "(bind {destructure: false} x (lit {} 1))",
        "(var {destructure: true} x)",
    ] {
        assert!(parse_str(source).is_err(), "accepted {source}");
    }
}

#[test]
fn source_arguments_and_extension_values_are_data() {
    let historical = "(var {source: (macro_name {custom: 1, custom: 2, surf_future: false})} x)";
    let parsed = parse_str(historical).unwrap();
    let wire = serde_json::to_string(&parsed).unwrap();
    let decoded: Vec<chelis_deep::Expr> = serde_json::from_str(&wire).unwrap();
    assert_eq!(parsed, decoded);
    assert!(parse_str("(var {custom: {custom: 1, custom: 2}} x)").is_ok());
    assert!(parse_str("(var {custom: {surf_future: false}} x)").is_ok());
}

#[test]
fn registered_inventory_contains_thirty_compiler_owned_keys() {
    let keys = chelis_deep::metadata::REGISTERED_METADATA_KEYS;
    assert_eq!(keys.len(), 30);
    assert_eq!(TYPED_CASES.len(), keys.len());
    for key in keys {
        assert!(
            TYPED_CASES.iter().any(|(covered, _)| covered == key),
            "{key}"
        );
    }
    for key in ["effect", "literal_source", "destructure"] {
        assert!(keys.contains(&key), "{key}");
    }
}

#[test]
fn typed_insertion_replacement_and_extension_names_are_separate() {
    use chelis_deep::Span;
    use chelis_deep::annotations::{Metadata, MetadataValue, Spanned};
    let span = Span::new(0, 0);
    let mut meta = Metadata::default();
    meta.insert(MetadataValue::Doc(Spanned::new("first".into(), span)))
        .unwrap();
    assert!(
        meta.insert(MetadataValue::Doc(Spanned::new("second".into(), span)))
            .is_err()
    );
    assert_eq!(meta.doc().unwrap().value(), "first");
    meta.replace(MetadataValue::Doc(Spanned::new("second".into(), span)));
    assert_eq!(meta.doc().unwrap().value(), "second");
    for key in [
        "type",
        "span",
        "literal_source",
        "effect",
        "destructure",
        "surf_future",
    ] {
        assert!(
            meta.extensions_mut()
                .insert(
                    key.into(),
                    chelis_deep::ExtensionData::parse("true").unwrap()
                )
                .is_err(),
            "{key}"
        );
    }
    meta.extensions_mut()
        .insert(
            "custom".into(),
            chelis_deep::ExtensionData::parse("true").unwrap(),
        )
        .unwrap();
    assert!(
        meta.extensions_mut()
            .insert(
                "custom".into(),
                chelis_deep::ExtensionData::parse("false").unwrap()
            )
            .is_err()
    );
}

#[test]
fn structural_containers_preserve_presence_without_an_expression_root() {
    use chelis_deep::Span;
    use chelis_deep::annotations::{Metadata, MetadataValue, PropertyPreconditions};
    let span = Span::new(7, 8);
    let mut metadata = Metadata::default();
    assert!(metadata.property_preconditions().is_none());
    metadata
        .insert(MetadataValue::PropertyPreconditions(
            PropertyPreconditions::new(Metadata::default(), vec![], span),
        ))
        .unwrap();
    let preconditions = metadata.property_preconditions().unwrap();
    assert!(preconditions.values().is_empty());
    assert_eq!(preconditions.span(), span);
    assert!(std::mem::size_of::<Metadata>() <= std::mem::size_of::<usize>());
}

const TYPED_CASES: &[(&str, &str)] = &[
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
];

#[test]
fn typed_payloads_round_trip_all_registered_shapes() {
    fn dedicated(expr: &chelis_deep::Expr, key: &str) -> bool {
        let chelis_deep::Expr::Node(node, _) = expr else {
            return false;
        };
        assert!(
            node.meta().extensions().get(key).is_none(),
            "registered key leaked to extensions"
        );
        node.meta()
            .values()
            .any(|value| value.key().spelling() == key)
            || node
                .children_slice()
                .iter()
                .any(|child| dedicated(child, key))
    }
    for (key, source) in TYPED_CASES {
        let parsed = parse_str(source).unwrap_or_else(|e| panic!("{key}: {e}"));
        assert!(
            parsed.iter().any(|expr| dedicated(expr, key)),
            "{key} was not decoded to a dedicated variant"
        );
        let binary = bincode::serialize(&parsed).unwrap();
        assert_eq!(
            bincode::deserialize::<Vec<chelis_deep::Expr>>(&binary).unwrap(),
            parsed,
            "current binary: {key}"
        );
        let serialized = serde_json::to_string(&parsed).unwrap();
        let decoded: Vec<chelis_deep::Expr> =
            serde_json::from_str(&serialized).unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(parsed, decoded, "{key}");
        let canonical = chelis_deep::printer::print_canonical_flat(&parsed);
        assert_eq!(
            canonical,
            chelis_deep::printer::print_canonical_flat(&parse_str(&canonical).unwrap()),
            "{key}"
        );
    }
}

#[test]
fn binder_annotation_spans_survive_structural_payload_decoding() {
    let source = "(def {property_quantifiers: (params {} (x {type: (t-prim {} f32)}))} p 1)";
    let parsed = parse_str(source).unwrap();
    let wire = serde_json::to_value(&parsed).unwrap();
    let encoded = wire.to_string();
    let map_offset = source.find("{type:").unwrap();
    assert!(
        encoded.contains(&format!("\"offset\":{map_offset}")),
        "binder map span was erased: {encoded}"
    );
}

#[test]
fn empty_storage_and_invalid_extension_names_have_no_hidden_state() {
    use chelis_deep::annotations::Metadata;
    let mut metadata = Metadata::default();
    for key in ["", "bad-key", "1invalid", "not.key"] {
        assert!(
            metadata
                .extensions_mut()
                .insert(
                    key.into(),
                    chelis_deep::ExtensionData::parse("true").unwrap()
                )
                .is_err(),
            "{key}"
        );
    }
    assert_eq!(metadata, Metadata::default());
}

#[test]
fn wire_export_preserves_structures_and_source_data_without_admitting_it() {
    let program =
        parse_str("(t-fn {source: (macro {k: 1, k: 2}), eff: (effects {} io)} (t-unit {}))")
            .unwrap();
    let raw = program[0].to_raw();
    let chelis_deep::raw::RawExpr::List(items, _) = raw else {
        panic!("node exports as raw list")
    };
    let chelis_deep::raw::RawExpr::Map(entries, _) = &items[1] else {
        panic!("annotation slot")
    };
    assert_eq!(
        entries
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["eff", "source"]
    );
    let chelis_deep::raw::RawExpr::List(source, _) = &entries[1].1 else {
        panic!("source invocation")
    };
    let chelis_deep::raw::RawExpr::Map(arguments, _) = &source[1] else {
        panic!("argument data")
    };
    assert_eq!(arguments.len(), 2);
    assert!(parse_str("(t-fn {eff: (effects {}), eff: (effects {})} (t-unit {}))").is_err());
}

#[test]
fn metadata_rewrites_visit_structural_annotations_without_rewriting_roots() {
    use chelis_deep::annotations::MetadataKey;
    use chelis_deep::{DeepTag, Expr};
    let program = parse_str("(t-fn {eff: (effects {span: \"effects\"} (resource {span: \"resource\"} \"cpu\"))} (t-unit {}))").unwrap();
    let Expr::Node(node, _) = &program[0] else {
        panic!("typed node")
    };
    let mut containers = Vec::new();
    let result = node
        .meta()
        .try_map_expressions_with_annotations::<chelis_deep::metadata::MetadataError>(
            &mut |_, _| panic!("effect roots and names are not expression leaves"),
            &mut |mut annotations, owner| {
                containers.push(owner);
                annotations.remove(MetadataKey::Span);
                Ok(annotations)
            },
        )
        .unwrap();
    assert_eq!(
        containers,
        [Some(DeepTag::Effects), Some(DeepTag::Resource)]
    );
    assert!(result.eff().unwrap().metadata().span_id().is_none());
    let failure = node
        .meta()
        .try_map_expressions_with_annotations::<chelis_deep::metadata::MetadataError>(
            &mut |value, _| Ok(value.clone()),
            &mut |mut annotations, _| {
                annotations.replace(chelis_deep::annotations::MetadataValue::Opaque(
                    chelis_deep::annotations::Present::new(chelis_deep::Span::new(0, 0)),
                ));
                Ok(annotations)
            },
        );
    assert!(
        failure.is_err(),
        "structural annotation placement is checked after rebuilding"
    );
    assert!(
        node.meta().eff().unwrap().metadata().span_id().is_some(),
        "failed replacement leaves the original untouched"
    );
}

#[test]
fn expression_payload_admission_rejects_undecoded_vocabulary() {
    use chelis_deep::annotations::{RuntimeExpression, TypeSyntax};
    use chelis_deep::{Atom, Expr, Metadata, Span, UnknownFormData};
    let span = Span::new(0, 0);
    // A vocabulary head that bypassed stamping: only a hand-built
    // `UnknownForm` can carry one.
    let raw = |tag: &str| {
        Expr::UnknownForm(Box::new(UnknownFormData {
            head: tag.into(),
            meta: Metadata::default(),
            children: vec![Expr::Atom(Atom::Name("x".into()), span)],
            span,
        }))
    };
    assert!(TypeSyntax::try_new(raw("t-var")).is_err());
    assert!(RuntimeExpression::try_new(raw("var")).is_err());
    let typed = parse_str("(var {} x)").unwrap().remove(0);
    assert!(RuntimeExpression::try_new(typed).is_ok());
    let mut annotations = Metadata::default();
    annotations
        .extensions_mut()
        .insert(
            "custom".into(),
            chelis_deep::ExtensionData::parse("(var {} x extra)").unwrap(),
        )
        .unwrap();
    let before = annotations.clone();
    assert!(
        annotations
            .extensions_mut()
            .replace(
                "surf_future".into(),
                chelis_deep::ExtensionData::parse("true").unwrap()
            )
            .is_err()
    );
    assert_eq!(annotations, before);
}

#[test]
fn serde_metadata_does_not_restamp_an_invalid_typed_payload() {
    use chelis_deep::{Atom, Expr, Metadata, Span, UnknownFormData};
    let span = Span::new(0, 0);
    let raw = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "var".into(),
        meta: Metadata::default(),
        children: vec![Expr::Atom(Atom::Name("x".into()), span)],
        span,
    }));
    let wire = serde_json::json!({"entries": [["custom", raw]]});
    assert!(serde_json::from_value::<Metadata>(wire).is_err());
    let typed = parse_str("(var {custom: (var {} x), source: (m (var {} historical))} x)").unwrap();
    assert_eq!(
        serde_json::from_value::<Vec<Expr>>(serde_json::to_value(&typed).unwrap()).unwrap(),
        typed
    );
}

#[test]
fn prefix_metadata_supports_binary_checkpoint_round_trips() {
    let program = parse_str("^{:type (t-prim {} f32)} x").unwrap();
    let bytes = bincode::serialize(&program).expect("prefix metadata serializes in binary formats");
    let restored: Vec<chelis_deep::Expr> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(restored, program);
}

#[test]
fn serde_preserves_existing_expression_leaf_carriers() {
    use chelis_deep::annotations::{
        MetadataValue as M, PropertyPreconditions, RuntimeExpression, TypeSyntax,
    };
    use chelis_deep::{Atom, DeepTag, Expr, Metadata, Span};
    let span = Span::new(3, 4);
    let stamped = |tag, child| {
        let node = chelis_deep::node::Node::new(tag, Metadata::default(), vec![child]);
        Expr::Node(Box::new(node), span)
    };
    let variable = RuntimeExpression::try_new(stamped(
        DeepTag::Var,
        Expr::Atom(Atom::Name("x".into()), span),
    ))
    .unwrap();
    let ty = TypeSyntax::try_new(stamped(
        DeepTag::TPrim,
        Expr::Atom(Atom::Name("f32".into()), span),
    ))
    .unwrap();
    let metadata = Metadata::try_from_values([
        M::Type(ty),
        M::PropertySeed(variable.clone()),
        M::PropertyPreconditions(PropertyPreconditions::new(
            Metadata::default(),
            vec![variable],
            span,
        )),
    ])
    .unwrap();
    let json = serde_json::to_value(&metadata).unwrap();
    assert_eq!(serde_json::from_value::<Metadata>(json).unwrap(), metadata);
    let binary = bincode::serialize(&metadata).unwrap();
    assert_eq!(bincode::deserialize::<Metadata>(&binary).unwrap(), metadata);
}

#[test]
fn previous_extension_checkpoints_reject_and_core_json_remains_readable() {
    let sources: Vec<String> =
        serde_json::from_str(include_str!("fixtures/metadata_df5daab/sources.json")).unwrap();
    let old: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("fixtures/metadata_df5daab/ast.json")).unwrap();
    assert_eq!(sources.len(), 32);
    assert_eq!(old.len(), sources.len());
    let mut rejected = 0;
    let mut accepted = 0;
    let mut retired = 0;
    for (source, json) in sources.iter().zip(old) {
        // The `random` handler kind was retired with the counter stream
        // (#2413): its source no longer parses and its old core JSON no
        // longer decodes, each naming the retired kind.
        if source.contains("{effect: random}") {
            assert!(parse_str(source).is_err(), "retired kind parsed: {source}");
            let error = serde_json::from_value::<Vec<chelis_deep::Expr>>(json)
                .unwrap_err()
                .to_string();
            assert!(error.contains("effect kind `random` is retired"), "{error}");
            retired += 1;
            continue;
        }
        let expected = parse_str(source).unwrap();
        let current = serde_json::to_value(&expected).unwrap();
        let decoded = serde_json::from_value::<Vec<chelis_deep::Expr>>(json);
        if current.to_string().contains("ExtensionData") {
            assert!(
                decoded
                    .unwrap_err()
                    .to_string()
                    .contains("opaque extension-data encoding")
            );
            rejected += 1;
        } else {
            assert_eq!(decoded.unwrap(), expected, "old core: {source}");
            accepted += 1;
        }
        let current_binary = bincode::serialize(&expected).unwrap();
        assert_eq!(
            bincode::deserialize::<Vec<chelis_deep::Expr>>(&current_binary).unwrap(),
            expected
        );
    }
    assert!(accepted > 0 && rejected > 0);
    assert_eq!(retired, 1);
    assert!(
        bincode::deserialize::<Vec<Vec<chelis_deep::Expr>>>(include_bytes!(
            "fixtures/metadata_df5daab/ast.bin"
        ))
        .is_err()
    );
}

#[test]
fn serde_rejects_duplicate_core_and_extension_entries_in_json_and_binary() {
    // Independent hostile wire producer. The positive controls lock the enum
    // tags before the duplicate-key controls exercise admission.
    #[derive(Clone, serde::Serialize)]
    #[allow(dead_code)]
    enum Wire {
        Atom(chelis_deep::Atom, chelis_deep::Span),
        Map,
        MetaExpr,
        Node,
        BareList,
        UnknownForm,
        ExtensionData(chelis_deep::ExtensionData),
    }
    #[derive(serde::Serialize)]
    struct Entries {
        entries: Vec<(String, Wire)>,
    }
    for key in ["doc", "custom", "span_file"] {
        let value = if key == "doc" {
            Wire::Atom(
                chelis_deep::Atom::Str("value".into()),
                chelis_deep::Span::new(0, 0),
            )
        } else {
            Wire::ExtensionData(chelis_deep::ExtensionData::parse("\"value\"").unwrap())
        };
        let entry = (key.to_string(), value);
        let single = Entries {
            entries: vec![entry.clone()],
        };
        assert!(
            serde_json::from_slice::<chelis_deep::Metadata>(&serde_json::to_vec(&single).unwrap())
                .is_ok()
        );
        assert!(
            bincode::deserialize::<chelis_deep::Metadata>(&bincode::serialize(&single).unwrap())
                .is_ok()
        );
        let duplicate = Entries {
            entries: vec![entry.clone(), entry],
        };
        for error in [
            serde_json::from_slice::<chelis_deep::Metadata>(
                &serde_json::to_vec(&duplicate).unwrap(),
            )
            .unwrap_err()
            .to_string(),
            bincode::deserialize::<chelis_deep::Metadata>(&bincode::serialize(&duplicate).unwrap())
                .unwrap_err()
                .to_string(),
        ] {
            assert!(
                error.contains(key) && error.contains("exactly one occurrence"),
                "{error}"
            );
        }
    }
}

#[test]
fn unknown_effect_diagnostics_preserve_the_kind_at_text_and_serde_ingress() {
    use chelis_deep::{Atom, Expr, Metadata, Span};
    parse_str("(handle-effect {effect: resource} (lit {} 1) (lit {} 2))").unwrap();
    let value = Expr::Atom(Atom::Name("resource".into()), Span::new(0, 0));
    let wire = serde_json::json!({"entries": [["effect", value]]});
    assert!(serde_json::from_value::<Metadata>(wire).is_ok());
    // `random` was a handler kind until the counter stream was retired
    // (#2413); both ingresses refuse it as a typed retired spelling, and
    // `teleport` as an unknown kind.
    for (kind, expected) in [
        ("teleport", "unknown effect kind `teleport`"),
        ("random", "effect kind `random` is retired"),
    ] {
        let source = format!("(handle-effect {{effect: {kind}}} (lit {{}} 1) (lit {{}} 2))");
        let value = Expr::Atom(Atom::Name(kind.into()), Span::new(0, 0));
        let wire = serde_json::json!({"entries": [["effect", value]]});
        for error in [
            parse_str(&source).unwrap_err().to_string(),
            serde_json::from_value::<Metadata>(wire)
                .unwrap_err()
                .to_string(),
        ] {
            assert!(error.contains("metadata `effect`"), "{error}");
            assert!(error.contains(expected), "{error}");
        }
    }
    let wrong_owner = parse_str("(var {effect: resource} x)")
        .unwrap_err()
        .to_string();
    assert!(wrong_owner.contains("handle-effect"), "{wrong_owner}");
    assert!(
        !wrong_owner.contains("unknown effect kind"),
        "{wrong_owner}"
    );
}
