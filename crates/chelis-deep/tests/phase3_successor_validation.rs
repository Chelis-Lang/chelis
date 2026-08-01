//! Phase 3 successor-carrier regressions for chelis#731.
//!
//! The decode-once oracle and validator must traverse every expression-bearing
//! edge introduced by the gated `Node` representation.  Otherwise a legacy
//! raw vocabulary tag can hide below a newly stamped carrier and restore the
//! silent-dispatch class that Phase 3 closed.

use chelis_deep::node::{Node, NodeError};
use chelis_deep::span::Span;
use chelis_deep::tag::DeepTag;
use chelis_deep::validate::{WarningKind, find_raw_vocabulary_tag, validate};
use chelis_deep::{Atom, Expr, List, MetaMap, UnknownFormData, parse_and_stamp};

fn sp() -> Span {
    Span::new(0, 0)
}

fn raw_vocabulary_form(tag: &str) -> Expr {
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Name(tag.to_string()), sp()),
                Expr::Map(MetaMap::default(), sp()),
                Expr::Atom(Atom::Int(0), sp()),
            ],
        },
        sp(),
    )
}

fn assert_raw_tag_is_found_and_validated(expr: Expr, expected: &str) {
    assert_eq!(
        find_raw_vocabulary_tag(std::slice::from_ref(&expr)).as_deref(),
        Some(expected)
    );
    assert!(
        validate(&[expr]).iter().any(|warning| {
            matches!(warning.kind, WarningKind::UnknownTag) && warning.message.contains(expected)
        }),
        "the recursive validator must report the nested raw vocabulary tag"
    );
}

#[test]
fn node_metadata_and_children_are_recursive_validation_edges() {
    let node_with_raw_metadata = Node::try_new(
        DeepTag::Var,
        MetaMap {
            entries: vec![("probe".to_string(), raw_vocabulary_form("lit"))],
        },
        vec![Expr::Atom(Atom::Name("x".to_string()), sp())],
    );
    assert!(matches!(
        node_with_raw_metadata,
        Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "lit"
    ));

    let node_with_raw_child = Node::try_new(
        DeepTag::Var,
        MetaMap::default(),
        vec![raw_vocabulary_form("app")],
    );
    assert!(matches!(
        node_with_raw_child,
        Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "app"
    ));
}

#[test]
fn bare_list_and_unknown_form_edges_are_recursive_validation_edges() {
    let bare = Expr::BareList(vec![raw_vocabulary_form("if")], sp());
    assert_raw_tag_is_found_and_validated(bare, "if");

    let unknown_with_raw_metadata = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: MetaMap {
            entries: vec![("probe".to_string(), raw_vocabulary_form("let"))],
        },
        children: vec![],
        span: sp(),
    }));
    assert_raw_tag_is_found_and_validated(unknown_with_raw_metadata, "let");

    let unknown_with_raw_child = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: MetaMap::default(),
        children: vec![raw_vocabulary_form("tuple")],
        span: sp(),
    }));
    assert_raw_tag_is_found_and_validated(unknown_with_raw_child, "tuple");
}

#[test]
fn clean_successor_carriers_remain_clean() {
    let stamped = parse_and_stamp("(def {} f (lit {} 1))").expect("valid stamped declaration");
    assert_eq!(find_raw_vocabulary_tag(&stamped), None);
    assert!(validate(&stamped).is_empty());

    let unknown = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: MetaMap::default(),
        children: stamped,
        span: sp(),
    }));
    assert_eq!(
        find_raw_vocabulary_tag(std::slice::from_ref(&unknown)),
        None
    );
    let warnings = validate(&[unknown]);
    assert_eq!(warnings.len(), 1);
    assert!(matches!(warnings[0].kind, WarningKind::UnknownTag));
    assert!(warnings[0].message.contains("future-form"));
}

#[test]
fn node_constructor_rejects_raw_vocabulary_below_its_gate() {
    let result = Node::try_new(
        DeepTag::Def,
        MetaMap::default(),
        vec![
            Expr::Atom(Atom::Name("f".to_string()), sp()),
            raw_vocabulary_form("lit"),
        ],
    );

    assert!(
        matches!(
            result,
            Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "lit"
        ),
        "the gated carrier must reject a raw closed-vocabulary tag: {result:?}"
    );
}

#[test]
fn typed_expr_constructor_produces_a_gated_node() {
    let expr = Expr::node(
        DeepTag::Lit,
        MetaMap::default(),
        vec![Expr::Atom(Atom::Int(1), sp())],
        sp(),
    );
    assert!(matches!(expr, Expr::Node(node, _) if node.tag() == DeepTag::Lit));
}

#[test]
fn typed_expr_constructor_cannot_bypass_runtime_roles() {
    let result = std::panic::catch_unwind(|| {
        Expr::node(
            DeepTag::Def,
            MetaMap::default(),
            vec![
                Expr::Atom(Atom::Name("f".to_string()), sp()),
                Expr::Atom(Atom::Name("unwrapped_name".to_string()), sp()),
            ],
            sp(),
        )
    });
    assert!(
        result.is_err(),
        "the typed producer must enforce Node roles"
    );
}
