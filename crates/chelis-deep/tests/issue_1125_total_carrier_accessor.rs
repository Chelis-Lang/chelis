use chelis_deep::{
    Atom, DeepTag, Expr, ExprCarrier, MetaExpr, Metadata, Span, UnknownFormData,
    authoring::rename_function, parse_and_stamp_file, parse_and_stamp_runtime_exprs,
};

fn span() -> Span {
    Span::new(0, 0)
}

fn name(value: &str) -> Expr {
    Expr::Atom(Atom::Name(value.to_string()), span())
}

fn parse_runtime_expr(source: &str) -> Expr {
    let mut exprs = parse_and_stamp_runtime_exprs(source).expect("fixture must stamp");
    assert_eq!(exprs.len(), 1, "one expression");
    exprs.remove(0)
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
        ExprCarrier::Atom(_) => false,
    }
}

#[test]
fn decoded_node_read_is_identical_for_constructed_and_parsed_nodes() {
    let constructed = Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![name("constructed")],
        span(),
    );
    let parsed = parse_runtime_expr("(var {} parsed)");

    for (expr, expected_name) in [(&constructed, "constructed"), (&parsed, "parsed")] {
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
        rename_function(&stamped, "target", "renamed").expect("authoring rewrite succeeds");

    assert!(
        renamed
            .module
            .iter()
            .any(|expr| contains_structural_head(expr, "x")),
        "an annotated binder must remain a structural-list carrier after an authoring rewrite"
    );
    assert!(
        renamed
            .module
            .iter()
            .any(|expr| contains_structural_names(expr, &["copy", "fill"])),
        "an import name list must remain a structural-list carrier after an authoring rewrite"
    );
}

#[test]
fn undecodable_head_read_is_identical_for_constructed_and_parsed_forms() {
    let constructed = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-constructed".to_string(),
        meta: Metadata::default(),
        children: vec![name("constructed-payload")],
        span: span(),
    }));
    let parsed = parse_runtime_expr("(future-parsed {} parsed-payload)");

    for (expr, expected_head, expected_payload) in [
        (&constructed, "future-constructed", "constructed-payload"),
        (&parsed, "future-parsed", "parsed-payload"),
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
    let Expr::BareList(elements, _) = &parsed else {
        panic!("unstamped source list must retain its structural role");
    };
    let Expr::Map(metadata, _) = &elements[1] else {
        panic!("fixture must contain metadata in the legacy metadata slot");
    };

    let undecodable = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: metadata.clone(),
        children: vec![name("payload")],
        span: span(),
    }));
    assert!(matches!(
        undecodable.carrier(),
        ExprCarrier::UndecodableHead("future-form", _, _)
    ));
    assert_eq!(undecodable.span_id(), Some("legacy-unknown"));

    let known = parse_runtime_expr(r#"(var {span: "legacy-unknown"} value)"#);
    assert_eq!(known.span_id(), Some("legacy-unknown"));

    // A structural list carries no node metadata, even when one of its
    // elements is a map holding a span.
    assert_eq!(parsed.span_id(), None);
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
}

/// A node or unknown form without its metadata slot has no in-memory
/// spelling: the stamper rejects it at ingress, so no reader can decode it
/// silently or mistake it for an undecodable head.
#[test]
fn malformed_legacy_heads_are_not_silently_decoded_or_treated_as_undecodable() {
    for malformed in [
        "(var x)",
        "(var not-metadata x)",
        "(future-form)",
        "(future-form not-metadata payload)",
    ] {
        assert!(
            parse_and_stamp_runtime_exprs(malformed).is_err(),
            "`{malformed}` must be rejected at ingress"
        );
    }
}
