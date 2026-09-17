use chelis_deep::{
    Atom, DeepTag, Expr, ExprCarrier, List, MetaExpr, Metadata, Span, UnknownFormData,
    authoring::rename_function, parse_and_stamp_file,
};

fn span() -> Span {
    Span::new(0, 0)
}

fn name(value: &str) -> Expr {
    Expr::Atom(Atom::Name(value.to_string()), span())
}

fn legacy_node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::List(
        List {
            elements: std::iter::once(Expr::Atom(Atom::Tag(tag), span()))
                .chain(std::iter::once(Expr::Map(Metadata::default(), span())))
                .chain(children)
                .collect(),
        },
        span(),
    )
}

fn legacy_unknown(head: &str, children: Vec<Expr>) -> Expr {
    Expr::List(
        List {
            elements: std::iter::once(name(head))
                .chain(std::iter::once(Expr::Map(Metadata::default(), span())))
                .chain(children)
                .collect(),
        },
        span(),
    )
}

fn decoded_parts(expr: &Expr) -> (DeepTag, &Metadata, &[Expr]) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => (tag, metadata, children),
        other => panic!("expected decoded node carrier, found {other:?}"),
    }
}

fn undecodable_parts(expr: &Expr) -> (&str, &Metadata, &[Expr]) {
    match expr.carrier() {
        ExprCarrier::UndecodableHead(head, metadata, children) => (head, metadata, children),
        other => panic!("expected undecodable-head carrier, found {other:?}"),
    }
}

fn structural_names_match(elements: &[Expr], expected: &[&str]) -> bool {
    elements.len() == expected.len()
        && elements
            .iter()
            .zip(expected)
            .all(|(expr, expected)| matches!(expr, Expr::Atom(Atom::Name(found), _) if found == expected))
}

fn structural_head_matches(elements: &[Expr], expected: &str) -> bool {
    matches!(
        elements,
        [Expr::Atom(Atom::Name(found), _), Expr::Map(_, _)] if found == expected
    )
}

fn contains_structural_names(expr: &Expr, expected: &[&str]) -> bool {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, children)
        | ExprCarrier::UndecodableHead(_, metadata, children) => {
            let mut found = false;
            metadata.visit_syntax(&mut |_, value| {
                found |= contains_structural_names(value, expected);
            });
            found
                || children
                    .iter()
                    .any(|child| contains_structural_names(child, expected))
        }
        ExprCarrier::StructuralList(elements) => {
            structural_names_match(elements, expected)
                || elements
                    .iter()
                    .any(|child| contains_structural_names(child, expected))
        }
        ExprCarrier::MetadataMap(metadata) => {
            let mut found = false;
            metadata.visit_syntax(&mut |_, value| {
                found |= contains_structural_names(value, expected);
            });
            found
        }
        ExprCarrier::MetadataExpression(metadata_expr) => {
            let mut found = contains_structural_names(&metadata_expr.expr, expected);
            metadata_expr.metadata.visit_syntax(&mut |_, value| {
                found |= contains_structural_names(value, expected);
            });
            found
        }
        ExprCarrier::MalformedLegacyList(list) => list
            .elements
            .iter()
            .any(|child| contains_structural_names(child, expected)),
        ExprCarrier::Atom(_) => false,
    }
}

fn contains_structural_head(expr: &Expr, expected: &str) -> bool {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, children)
        | ExprCarrier::UndecodableHead(_, metadata, children) => {
            let mut found = false;
            metadata.visit_syntax(&mut |_, value| {
                found |= contains_structural_head(value, expected);
            });
            found
                || children
                    .iter()
                    .any(|child| contains_structural_head(child, expected))
        }
        ExprCarrier::StructuralList(elements) => {
            structural_head_matches(elements, expected)
                || elements
                    .iter()
                    .any(|child| contains_structural_head(child, expected))
        }
        ExprCarrier::MetadataMap(metadata) => {
            let mut found = false;
            metadata.visit_syntax(&mut |_, value| {
                found |= contains_structural_head(value, expected);
            });
            found
        }
        ExprCarrier::MetadataExpression(metadata_expr) => {
            let mut found = contains_structural_head(&metadata_expr.expr, expected);
            metadata_expr.metadata.visit_syntax(&mut |_, value| {
                found |= contains_structural_head(value, expected);
            });
            found
        }
        ExprCarrier::MalformedLegacyList(list) => list
            .elements
            .iter()
            .any(|child| contains_structural_head(child, expected)),
        ExprCarrier::Atom(_) => false,
    }
}

#[test]
fn decoded_node_read_is_identical_for_successor_and_legacy_carriers() {
    let successor = Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![name("successor")],
        span(),
    );
    let legacy = legacy_node(DeepTag::Var, vec![name("legacy")]);

    for (expr, expected_name) in [(&successor, "successor"), (&legacy, "legacy")] {
        let (tag, metadata, children) = decoded_parts(expr);
        assert_eq!(tag, DeepTag::Var);
        assert!(metadata.is_empty());
        assert!(matches!(
            children,
            [Expr::Atom(Atom::Name(found), _)] if found == expected_name
        ));
    }
}

#[test]
fn legal_legacy_structural_lists_share_the_structural_disposition() {
    let source = "\
(module {} demo.structural
  (import {} std.linalg (copy fill))
  (def {} target
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (var {} x))))";
    let stamped = parse_and_stamp_file(source).expect("fixture must stamp");
    let renamed =
        rename_function(&stamped, "target", "renamed").expect("authoring normalization succeeds");

    assert!(
        renamed
            .module
            .iter()
            .any(|expr| contains_structural_head(expr, "x")),
        "an annotated binder must remain a structural-list carrier after authoring normalization"
    );
    assert!(
        renamed
            .module
            .iter()
            .any(|expr| contains_structural_names(expr, &["copy", "fill"])),
        "an import name list must remain a structural-list carrier after authoring normalization"
    );

    let undecodable = legacy_unknown("future-form", vec![]);
    assert!(matches!(
        undecodable.carrier(),
        ExprCarrier::UndecodableHead("future-form", _, [])
    ));
    let malformed = Expr::List(
        List {
            elements: vec![Expr::Atom(Atom::Tag(DeepTag::Copy), span()), name("value")],
        },
        span(),
    );
    assert!(matches!(
        malformed.carrier(),
        ExprCarrier::MalformedLegacyList(_)
    ));
}

#[test]
fn undecodable_head_read_is_identical_for_successor_and_legacy_carriers() {
    let successor = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-successor".to_string(),
        meta: Metadata::default(),
        children: vec![name("successor-payload")],
        span: span(),
    }));
    let legacy = legacy_unknown("future-legacy", vec![name("legacy-payload")]);

    for (expr, expected_head, expected_payload) in [
        (&successor, "future-successor", "successor-payload"),
        (&legacy, "future-legacy", "legacy-payload"),
    ] {
        let (head, metadata, children) = undecodable_parts(expr);
        assert_eq!(head, expected_head);
        assert!(metadata.is_empty());
        assert!(matches!(
            children,
            [Expr::Atom(Atom::Name(found), _)] if found == expected_payload
        ));
    }
}

#[test]
fn undecodable_legacy_head_retains_its_external_span_id() {
    let parsed =
        chelis_deep::parser::parse_str(r#"(future-form {span: "legacy-unknown"} payload)"#)
            .expect("legacy metadata fixture parses")
            .remove(0);
    let Expr::BareList(elements, _) = parsed else {
        panic!("unstamped source list must retain its structural role");
    };
    let Expr::Map(metadata, _) = &elements[1] else {
        panic!("fixture must contain metadata in the legacy metadata slot");
    };
    let metadata = metadata.clone();

    let undecodable = Expr::List(
        List {
            elements: vec![
                name("future-form"),
                Expr::Map(metadata.clone(), span()),
                name("payload"),
            ],
        },
        span(),
    );
    assert!(matches!(
        undecodable.carrier(),
        ExprCarrier::UndecodableHead("future-form", _, _)
    ));
    assert_eq!(undecodable.span_id(), Some("legacy-unknown"));

    let known = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), span()),
                Expr::Map(metadata, span()),
                name("value"),
            ],
        },
        span(),
    );
    assert_eq!(known.span_id(), Some("legacy-unknown"));

    let malformed = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), span()),
                name("not-metadata"),
            ],
        },
        span(),
    );
    assert_eq!(malformed.span_id(), None);
}

#[test]
fn accessor_has_an_explicit_disposition_for_every_non_node_carrier() {
    let bare = Expr::BareList(vec![name("item")], span());
    let unknown = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: Metadata::default(),
        children: vec![name("payload")],
        span: span(),
    }));
    let atom = name("leaf");
    let map = Expr::Map(Metadata::default(), span());
    let annotated = Expr::MetaExpr(
        MetaExpr {
            metadata: Metadata::default(),
            expr: Box::new(name("wrapped")),
        },
        span(),
    );
    let malformed = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), span()),
                name("not-metadata"),
                name("payload"),
            ],
        },
        span(),
    );

    assert!(matches!(
        bare.carrier(),
        ExprCarrier::StructuralList([Expr::Atom(Atom::Name(found), _)]) if found == "item"
    ));
    assert!(matches!(
        unknown.carrier(),
        ExprCarrier::UndecodableHead("future-form", _, [Expr::Atom(Atom::Name(found), _)])
            if found == "payload"
    ));
    assert!(matches!(
        atom.carrier(),
        ExprCarrier::Atom(Atom::Name(found)) if found == "leaf"
    ));
    assert!(matches!(
        map.carrier(),
        ExprCarrier::MetadataMap(metadata) if metadata.is_empty()
    ));
    assert!(matches!(
        annotated.carrier(),
        ExprCarrier::MetadataExpression(MetaExpr { expr, .. })
            if matches!(expr.as_ref(), Expr::Atom(Atom::Name(found), _) if found == "wrapped")
    ));
    assert!(matches!(
        malformed.carrier(),
        ExprCarrier::MalformedLegacyList(list) if list.elements.len() == 3
    ));
}

#[test]
fn malformed_legacy_heads_are_not_silently_decoded_or_treated_as_undecodable() {
    let tagged_missing_metadata = Expr::List(
        List {
            elements: vec![Expr::Atom(Atom::Tag(DeepTag::Var), span()), name("x")],
        },
        span(),
    );
    let tagged_wrong_metadata = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), span()),
                name("not-metadata"),
                name("x"),
            ],
        },
        span(),
    );
    let unknown_missing_metadata = Expr::List(
        List {
            elements: vec![name("future-form")],
        },
        span(),
    );
    let unknown_wrong_metadata = Expr::List(
        List {
            elements: vec![name("future-form"), name("not-metadata"), name("payload")],
        },
        span(),
    );

    for malformed in [
        &tagged_missing_metadata,
        &tagged_wrong_metadata,
        &unknown_missing_metadata,
        &unknown_wrong_metadata,
    ] {
        assert!(matches!(
            malformed.carrier(),
            ExprCarrier::MalformedLegacyList(_)
        ));
    }
}
