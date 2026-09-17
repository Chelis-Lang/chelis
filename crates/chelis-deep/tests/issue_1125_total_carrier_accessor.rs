use chelis_deep::{
    Atom, DeepTag, Expr, ExprCarrier, List, MetaExpr, Metadata, Span, UnknownFormData,
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
