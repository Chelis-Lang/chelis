use chelis_deep::{
    Atom, DeepTag, Expr, List, Metadata, Span, UnknownFormData,
    metadata::same_semantic_type_syntax, parse_and_stamp_type,
};

fn parse_type_at_offset(source: &str, offset: usize) -> Expr {
    parse_and_stamp_type(&format!("{}{source}", " ".repeat(offset)))
        .unwrap_or_else(|error| panic!("valid type syntax {source:?}: {error}"))
}

fn legacy_list_root(expr: &Expr) -> Expr {
    let Expr::Node(node, span) = expr else {
        panic!("fixture must be a stamped node: {expr:?}");
    };
    let mut elements = vec![
        Expr::Atom(Atom::Tag(node.tag()), Span::new(span.offset + 1, 1)),
        Expr::Map(node.meta().clone(), Span::new(span.offset + 2, 2)),
    ];
    elements.extend(node.children_slice().iter().cloned());
    Expr::List(List { elements }, *span)
}

fn bare_list_root(expr: &Expr) -> Expr {
    let Expr::List(list, span) = legacy_list_root(expr) else {
        unreachable!("legacy_list_root returns a list")
    };
    Expr::BareList(list.elements, span)
}

#[test]
fn semantic_type_syntax_erases_source_storage_spans_recursively() {
    for name in ["f8e4m3", "f8e5m2"] {
        for source in [
            format!("(t-prim {{doc: \"same\"}} {name})"),
            format!(
                "(t-tensor {{doc: \"same\"}} \
                   (d-lit {{doc: \"dimension\"}} 3) \
                   (t-prim {{doc: \"precision\"}} {name}))"
            ),
            format!(
                "(t-fn {{eff: (effects {{doc: \"set\"}} random)}} \
                   (t-prim {{doc: \"parameter\"}} {name}) \
                   (t-prim {{doc: \"result\"}} bool))"
            ),
        ] {
            let left = parse_type_at_offset(&source, 0);
            let right = parse_type_at_offset(&source, 43);
            assert!(
                same_semantic_type_syntax(&left, &right),
                "storage spans changed the semantic view: {source}"
            );
        }
    }
}

#[test]
fn semantic_type_syntax_uses_the_metadata_contract_boundary() {
    let left = parse_type_at_offset(
        "(t-prim {doc: \"same\", span: \"left\", \
          loc: (loc \"left.dp\" 1 2), source: (macro left), \
          tool_data: {nested: (payload \"same\")}} f8e4m3)",
        0,
    );
    let right = parse_type_at_offset(
        "(t-prim {doc: \"same\", span: \"right\", \
          loc: (loc \"right.dp\" 9 8), source: (other right), \
          tool_data: {nested: (payload \"same\")}} f8e4m3)",
        37,
    );
    assert!(
        same_semantic_type_syntax(&left, &right),
        "source-only metadata and diagnostic offsets are not type semantics"
    );

    let different_doc = parse_type_at_offset(
        "(t-prim {doc: \"different\", tool_data: {nested: (payload \"same\")}} f8e4m3)",
        11,
    );
    assert!(
        !same_semantic_type_syntax(&left, &different_doc),
        "retained compiler metadata must remain distinguishable"
    );

    let different_extension = parse_type_at_offset(
        "(t-prim {doc: \"same\", tool_data: {nested: (payload \"different\")}} f8e4m3)",
        19,
    );
    assert!(
        !same_semantic_type_syntax(&left, &different_extension),
        "opaque producer payloads must remain distinguishable"
    );

    let payload_key_is_semantic = parse_type_at_offset(
        "(t-prim {doc: \"same\", tool_data: {span: \"payload-different\"}} f8e4m3)",
        29,
    );
    assert!(
        !same_semantic_type_syntax(&left, &payload_key_is_semantic),
        "a source-only metadata key does not erase the same spelling inside producer data"
    );
}

#[test]
fn semantic_type_syntax_erases_the_complete_top_level_span_namespace() {
    let left = parse_type_at_offset(
        "(t-prim {doc: \"same\", span_start: 11, span_end: 17, \
          span_file: \"left.dp\"} f8e4m3)",
        0,
    );
    let right = parse_type_at_offset(
        "(t-prim {doc: \"same\", span_start: 101, span_end: 107, \
          span_file: \"right.dp\"} f8e4m3)",
        41,
    );
    assert!(
        same_semantic_type_syntax(&left, &right),
        "the reserved top-level span_* namespace is source provenance"
    );

    let nested_left = parse_type_at_offset(
        "(t-prim {doc: \"same\", tool_data: {span_start: 11}} f8e4m3)",
        0,
    );
    let nested_right = parse_type_at_offset(
        "(t-prim {doc: \"same\", tool_data: {span_start: 101}} f8e4m3)",
        41,
    );
    assert!(
        !same_semantic_type_syntax(&nested_left, &nested_right),
        "span_* inside extension data remains opaque payload data"
    );
}

#[test]
fn semantic_type_syntax_accepts_only_the_typed_and_transitional_list_carriers() {
    for name in ["f8e4m3", "f8e5m2"] {
        for source in [
            format!("(t-prim {{doc: \"same\"}} {name})"),
            format!(
                "(t-tensor {{doc: \"same\"}} \
                   (d-lit {{}} 3) (t-prim {{}} {name}))"
            ),
        ] {
            let node = parse_type_at_offset(&source, 0);
            let list = legacy_list_root(&node);
            assert!(same_semantic_type_syntax(&node, &list), "{source}");
            assert!(same_semantic_type_syntax(&list, &node), "{source}");

            let bare = bare_list_root(&node);
            assert!(
                !same_semantic_type_syntax(&node, &bare),
                "a non-type BareList carrier must not acquire generated-copy authority: {source}"
            );
        }
    }

    let malformed = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::TPrim), Span::new(0, 1)),
                Expr::Atom(Atom::Name("f8e4m3".into()), Span::new(1, 7)),
            ],
        },
        Span::new(0, 8),
    );
    assert!(
        !same_semantic_type_syntax(&parse_type_at_offset("(t-prim {} f8e4m3)", 0), &malformed),
        "a malformed legacy list must remain independent"
    );

    let unknown = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "t-prim".into(),
        meta: Metadata::default(),
        children: vec![Expr::Atom(Atom::Name("f8e4m3".into()), Span::new(0, 7))],
        span: Span::new(0, 7),
    }));
    assert!(
        !same_semantic_type_syntax(&parse_type_at_offset("(t-prim {} f8e4m3)", 0), &unknown),
        "an unknown-form carrier must remain independent"
    );
}

#[test]
fn semantic_type_syntax_preserves_arity_and_authored_type_differences() {
    let one = parse_type_at_offset("(t-tuple {} (t-prim {} f8e4m3))", 0);
    let two = parse_type_at_offset("(t-tuple {} (t-prim {} f8e4m3) (t-prim {} bool))", 23);
    assert!(!same_semantic_type_syntax(&one, &two));

    let f64 = parse_type_at_offset("(t-prim {doc: \"same\"} f64)", 0);
    let f32 = parse_type_at_offset("(t-prim {doc: \"same\"} f32)", 31);
    assert!(!same_semantic_type_syntax(&f64, &f32));
}
