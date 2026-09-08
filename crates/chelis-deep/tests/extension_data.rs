use chelis_deep::{metadata::validate_metadata, parser::parse_str, printer::print_expr_flat};

#[test]
fn owner_combination_preserves_data_and_rejects_conflicts_transactionally() {
    let owner = parse_str("(var {tool_data: {id: 1}} x)").unwrap().remove(0);
    let replacement = parse_str("(var {other: 2} y)").unwrap().remove(0);
    let merged = replacement.clone().try_inherit_extensions(&owner).unwrap();
    assert!(print_expr_flat(&merged).contains("tool_data: {id: 1}"));
    assert!(print_expr_flat(&merged).contains("other: 2"));
    let identical = parse_str("(var {tool_data: {id: 1}} y)").unwrap().remove(0);
    assert!(identical.try_inherit_extensions(&owner).is_ok());
    let conflicting = parse_str("(var {tool_data: {id: 2}} y)").unwrap().remove(0);
    let before = conflicting.clone();
    assert!(
        conflicting
            .clone()
            .try_inherit_extensions(&owner)
            .unwrap_err()
            .to_string()
            .contains("tool_data")
    );
    assert_eq!(conflicting, before);
    let atom = parse_str("1")
        .unwrap()
        .remove(0)
        .try_inherit_extensions(&owner)
        .unwrap();
    validate_metadata(std::slice::from_ref(&atom)).unwrap();
    assert!(print_expr_flat(&atom).contains("tool_data"));
}

#[test]
fn extension_data_format_and_serialization_are_checked() {
    use chelis_deep::ExtensionData;
    for source in ["", "1 2", "(x", "{x 1}", "{x:}", "^{:x 1}", "{1x: 1}"] {
        assert!(ExtensionData::parse(source).is_err(), "{source}");
        let wire = serde_json::json!([source, {"offset": 4, "len": source.len()}]);
        assert!(
            serde_json::from_value::<ExtensionData>(wire).is_err(),
            "{source}"
        );
    }
    for source in [
        "()",
        "{}",
        "^{:type false :type 2} (var {})",
        "(lit {} 7f32)",
        "{type: false, type: 2}",
        "-0.0",
        "1e-3",
        "\"\\n\"",
    ] {
        let data = ExtensionData::parse(source).unwrap();
        assert_eq!(data.syntax(), source);
        assert_eq!(
            serde_json::from_value::<ExtensionData>(serde_json::to_value(&data).unwrap()).unwrap(),
            data
        );
        assert_eq!(
            bincode::deserialize::<ExtensionData>(&bincode::serialize(&data).unwrap()).unwrap(),
            data
        );
    }
    let nested = format!("{}x{}", "(".repeat(4000), ")".repeat(4000));
    assert_eq!(ExtensionData::parse(&nested).unwrap().syntax(), nested);
}

#[test]
fn producer_data_is_not_compiler_syntax() {
    for payload in [
        "{type: \"ablation\", type: 2, surf_future: false, span: 7}",
        "(var {} x y)",
        "(lit {} (lit {} 7f32))",
        "(unknown_macro {type: false} bare_name)",
    ] {
        let source = format!("(var {{tool_data: {payload}}} x)");
        let program = parse_str(&source).unwrap_or_else(|e| panic!("{source}: {e}"));
        validate_metadata(&program).unwrap();
        assert!(print_expr_flat(&program[0]).contains(payload));
    }
    for source in [
        "(var {type: \"ablation\"} x)",
        "(var {span: 7} x)",
        "(var {surf_future: false} x)",
        "(var {tool_data: 1, tool_data: 2} x)",
    ] {
        assert!(parse_str(source).is_err(), "accepted {source}");
    }
}

#[test]
fn semantic_visitors_skip_extensions_but_visit_registered_expressions() {
    let program =
        parse_str("(def {tool_data: (var {} historical), property_seed: (var {} live)} f 1)")
            .unwrap();
    let chelis_deep::Expr::Node(node, _) = &program[0] else {
        panic!("node")
    };
    let mut visited = Vec::new();
    node.meta()
        .visit_expressions(&mut |value, _| visited.push(print_expr_flat(value)));
    assert_eq!(visited, ["(var {} live)"]);
    let mapped = node
        .meta()
        .map_expressions(&mut |value, _| {
            assert!(!print_expr_flat(value).contains("historical"));
            value.clone()
        })
        .unwrap();
    assert_eq!(&mapped, node.meta());
}

#[test]
fn metadata_replacements_retain_expression_and_binder_owners() {
    let parsed = parse_str("(def {property_seed: (var {tool_data: 1} old), property_quantifiers: (params {container_data: 2} (x {binder_data: 3}))} f 1)").unwrap();
    let chelis_deep::Expr::Node(node, _) = &parsed[0] else {
        panic!("node")
    };
    let mapped = node
        .meta()
        .try_map_expressions_with_annotations::<chelis_deep::metadata::MetadataError>(
            &mut |value, _| {
                Ok(if print_expr_flat(value).starts_with("(var ") {
                    parse_str("(var {} new)").unwrap().remove(0)
                } else if print_expr_flat(value).starts_with("(x ") {
                    parse_str("y").unwrap().remove(0)
                } else {
                    value.clone()
                })
            },
            &mut |_, _| Ok(chelis_deep::Metadata::default()),
        )
        .unwrap();
    let mut rendered = node.clone();
    rendered.try_replace_meta(mapped).unwrap();
    let text = print_expr_flat(&chelis_deep::Expr::Node(rendered, parsed[0].span()));
    for data in ["tool_data: 1", ":binder_data 3", "container_data: 2"] {
        assert!(text.contains(data), "{text}");
    }
    assert!(
        node.meta()
            .try_map_expressions::<chelis_deep::metadata::MetadataError>(&mut |value, _| Ok(
                if print_expr_flat(value).starts_with("(var ") {
                    parse_str("(var {tool_data: 9} new)").unwrap().remove(0)
                } else {
                    value.clone()
                }
            ))
            .is_err()
    );
}

#[test]
fn authoring_preserves_replaced_declaration_owners_and_conflicts_fail() {
    let original = chelis_deep::parse_and_stamp_file("(module {} m (defsig {sig_data: 1} f (t-fn {} (t-prim {} int32))) (def {def_data: 2} f (fn {} (params {params_data: 3}) (lit {} 1))))").unwrap();
    let replacement = chelis_deep::parse_and_stamp_file("(defsig {} f (t-fn {} (t-prim {} int32))) (def {new_data: 4} f (fn {} (params {}) (lit {} 2)))").unwrap();
    let replaced =
        chelis_deep::authoring::replace_function(&original, "m.f", &replacement).unwrap();
    let text = replaced
        .module
        .iter()
        .map(print_expr_flat)
        .collect::<String>();
    for data in ["sig_data: 1", "def_data: 2", "new_data: 4"] {
        assert!(text.contains(data), "{text}");
    }
    assert!(
        !text.contains("params_data"),
        "removed body nodes lose their data: {text}"
    );
    let bad =
        chelis_deep::parse_and_stamp_file("(def {def_data: 9} f (fn {} (params {}) (lit {} 2)))")
            .unwrap();
    assert!(
        chelis_deep::authoring::replace_function(&original, "m.f", &bad)
            .unwrap_err()
            .to_string()
            .contains("def_data")
    );
    let params = parse_str("(params {})").unwrap().remove(0);
    let changed = chelis_deep::authoring::change_signature(
        &original,
        "m.f",
        &replacement[0],
        &params,
        &[],
        &[],
    )
    .unwrap();
    let text = changed
        .module
        .iter()
        .map(print_expr_flat)
        .collect::<String>();
    for data in ["sig_data: 1", "def_data: 2", "params_data: 3"] {
        assert!(text.contains(data), "{text}");
    }
}

#[test]
fn raw_and_token_ingress_cannot_forge_unvalidated_data() {
    use chelis_deep::{
        ExtensionData, RawAtom, RawExpr, Span,
        lexer::{Token, TokenKind},
    };
    let span = Span::new(0, 0);
    for atom in [
        RawAtom::Float(f64::INFINITY),
        RawAtom::Float(f64::NAN),
        RawAtom::Symbol("true".into()),
        RawAtom::Symbol("two words".into()),
    ] {
        assert!(ExtensionData::from_raw(&RawExpr::Atom(atom, span)).is_err());
    }
    assert_eq!(
        ExtensionData::from_raw(&RawExpr::Atom(RawAtom::Int(7), span))
            .unwrap()
            .syntax(),
        "7"
    );
    let mut tokens = chelis_deep::lexer::lex("(var {data: x} y)").unwrap();
    let token = tokens
        .iter_mut()
        .find(|t| t.kind == TokenKind::Symbol("x".into()))
        .unwrap();
    *token = Token {
        kind: TokenKind::Symbol("a )".into()),
        span,
    };
    assert!(chelis_deep::parser::parse(&tokens).is_err());
    let data = RawExpr::ExtensionData(ExtensionData::parse("(var {} x)").unwrap());
    assert!(chelis_deep::stamp_to_typed(vec![data.clone()]).is_err());
    assert!(chelis_deep::stamp_runtime_exprs(vec![data.clone()]).is_err());
    assert!(chelis_deep::stamp_as_tagged(vec![data.clone()], chelis_deep::DeepTag::Var).is_err());
    assert!(chelis_deep::stamp_to_typed::stamp_exprs_lenient(vec![data]).is_err());
}

#[test]
fn raw_data_conversion_preserves_map_key_identity() {
    use chelis_deep::{ExtensionData, RawAtom, RawExpr, Span};
    let span = Span::new(7, 3);
    let scalar = RawExpr::Atom(RawAtom::Int(2), span);
    for key in ["a: 1, b", "a 1 :b", "", "two words"] {
        let entries = vec![(key.into(), scalar.clone())];
        for raw in [
            RawExpr::Map(entries.clone(), span),
            RawExpr::MetaExpr {
                entries,
                expr: Box::new(scalar.clone()),
                span,
            },
        ] {
            assert!(ExtensionData::from_raw(&raw).is_err(), "{raw:?}");
            let nested = RawExpr::List(vec![scalar.clone(), raw], span);
            assert!(ExtensionData::from_raw(&nested).is_err(), "{nested:?}");
        }
    }
    let entries = vec![
        ("type".into(), scalar.clone()),
        ("type".into(), scalar.clone()),
        ("_key2".into(), scalar.clone()),
    ];
    for (raw, expected) in [
        (
            RawExpr::Map(entries.clone(), span),
            "{type: 2, type: 2, _key2: 2}",
        ),
        (
            RawExpr::MetaExpr {
                entries,
                expr: Box::new(scalar),
                span,
            },
            "^{:type 2 :type 2 :_key2 2} 2",
        ),
    ] {
        let data = ExtensionData::from_raw(&raw).unwrap();
        assert_eq!(data.syntax(), expected);
        assert_eq!(data.span(), span);
        let nested = RawExpr::List(
            vec![
                raw,
                RawExpr::ExtensionData(ExtensionData::parse("1e-3f32").unwrap()),
            ],
            span,
        );
        assert_eq!(
            ExtensionData::from_raw(&nested).unwrap().syntax(),
            format!("({expected} 1e-3f32)")
        );
    }
}

#[test]
fn authoring_renames_live_references_and_preserves_recorded_names() {
    let original = chelis_deep::parse_and_stamp_file("(module {} m (def {} f (fn {} (params {}) (lit {} 1))) (def {tool_data: (var {} f)} g (fn {} (params {}) (var {} f))))").unwrap();
    let renamed = chelis_deep::authoring::rename_function(&original, "m.f", "renamed").unwrap();
    let text = renamed
        .module
        .iter()
        .map(print_expr_flat)
        .collect::<String>();
    assert!(text.contains("tool_data: (var {} f)"), "{text}");
    assert!(text.contains("(var {} renamed)"), "{text}");
    assert_eq!(renamed.renamed_references, 1);
    assert!(chelis_deep::authoring::rename_function(&original, "m.absent", "renamed").is_err());
}
