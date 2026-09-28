use chelis_deep::node::Node;
use chelis_deep::parser::{parse_raw_str, parse_str};
use chelis_deep::validate::validate;
use chelis_deep::{Atom, DeepTag, Expr, Metadata, Span};

fn name(value: &str) -> Expr {
    Expr::Atom(Atom::Name(value.to_string()), Span::new(0, 0))
}

fn string(value: &str) -> Expr {
    Expr::Atom(Atom::Str(value.to_string()), Span::new(0, 0))
}

#[test]
fn parsers_reject_unknown_keys_in_the_closed_surf_namespace() {
    let typed = parse_str("(var {surf_future: true} x)")
        .expect_err("unknown surface metadata must fail at the typed parse boundary");
    assert!(typed.to_string().contains("closed Surf metadata namespace"));

    let raw = parse_raw_str("(var {surf_future: true} x)")
        .expect_err("unknown surface metadata must fail at the raw parse boundary");
    assert!(raw.to_string().contains("closed Surf metadata namespace"));
}

#[test]
fn programmatic_validation_accepts_known_surface_metadata_and_rejects_unknown_keys() {
    let known = parse_str("(module {surf_path: \"M.Path\"} m.path)").unwrap();
    assert!(validate(&known).is_empty());
    assert!(
        Node::try_new(
            DeepTag::Var,
            Metadata::from(chelis_deep::annotations::MetadataValue::SurfPath(
                chelis_deep::annotations::Spanned::new("M.Path".into(), Span::new(0, 0))
            )),
            vec![name("x")]
        )
        .is_err()
    );

    let mut metadata = Metadata::default();
    let error = metadata
        .extensions_mut()
        .insert(
            "surf_future".into(),
            chelis_deep::ExtensionData::parse("\"value\"").unwrap(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("surf_future"));
}

#[test]
fn legacy_expression_encoding_cannot_enter_the_extension_data_map() {
    let raw = serde_json::json!({"entries": [["outer", {"Map": [{"entries": [["surf_nested", string("value")]]}, Span::new(0, 0)]}]]});
    let error = serde_json::from_value::<Metadata>(raw).unwrap_err();
    assert!(error.to_string().contains("opaque extension-data encoding"));
}
