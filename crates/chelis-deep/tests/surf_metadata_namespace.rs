use chelis_deep::node::Node;
use chelis_deep::parser::{parse_raw_str, parse_str};
use chelis_deep::validate::validate;
use chelis_deep::{Atom, DeepTag, Expr, MetaMap, Span};

fn name(value: &str) -> Expr {
    Expr::Atom(Atom::Name(value.to_string()), Span::new(0, 0))
}

fn string(value: &str) -> Expr {
    Expr::Atom(Atom::Str(value.to_string()), Span::new(0, 0))
}

fn variable_with_metadata(entries: Vec<(String, Expr)>) -> Expr {
    Expr::List(
        chelis_deep::List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), Span::new(0, 0)),
                Expr::Map(MetaMap { entries }, Span::new(0, 0)),
                name("x"),
            ],
        },
        Span::new(0, 0),
    )
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
            MetaMap {
                entries: vec![("surf_path".into(), string("M.Path"))]
            },
            vec![name("x")]
        )
        .is_err()
    );

    let unknown = variable_with_metadata(vec![("surf_future".to_string(), string("value"))]);
    let warnings = validate(&[unknown]);
    assert_eq!(
        warnings
            .iter()
            .filter(|warning| warning.message.contains("closed Surf metadata namespace"))
            .count(),
        1,
        "programmatic producers must hit the same closed namespace gate: {warnings:?}"
    );
}

#[test]
fn programmatic_validation_recurses_through_structural_containers() {
    let unknown = variable_with_metadata(vec![("surf_nested".to_string(), string("value"))]);
    let program = Expr::BareList(vec![unknown], Span::new(0, 0));

    assert!(
        validate(&[program])
            .iter()
            .any(|warning| warning.message.contains("surf_nested")),
        "a structural container must not hide an unknown surface key"
    );
}
