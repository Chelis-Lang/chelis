//! Phase 3 successor-carrier regressions for chelis#731.
//!
//! The decode-once oracle and validator must traverse every expression-bearing
//! edge introduced by the gated `Node` representation.  Otherwise a legacy
//! raw vocabulary tag can hide below a newly stamped carrier and restore the
//! silent-dispatch class that Phase 3 closed.

use chelis_deep::node::{Node, NodeError};
use chelis_deep::role::AritySpec;
use chelis_deep::span::Span;
use chelis_deep::tag::DeepTag;
use chelis_deep::validate::{WarningKind, find_raw_vocabulary_tag, validate};
use chelis_deep::{Atom, Expr, Metadata, UnknownFormData, parse_and_stamp};

fn probe(value: Expr) -> Result<Metadata, chelis_deep::metadata::MetadataError> {
    let mut metadata = Metadata::default();
    metadata.insert(chelis_deep::annotations::MetadataValue::PropertySeed(
        chelis_deep::annotations::RuntimeExpression::try_new(value)?,
    ))?;
    Ok(metadata)
}

fn sp() -> Span {
    Span::new(0, 0)
}

/// A vocabulary head that bypassed stamping. The stamper builds a `Node` for
/// every decodable head, so only a hand-built `UnknownForm` can carry one.
fn raw_vocabulary_form(tag: &str) -> Expr {
    Expr::UnknownForm(Box::new(UnknownFormData {
        head: tag.to_string(),
        meta: Metadata::default(),
        children: vec![Expr::Atom(Atom::Int(0), sp())],
        span: sp(),
    }))
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
    assert!(
        probe(raw_vocabulary_form("lit"))
            .unwrap_err()
            .to_string()
            .contains("lit")
    );

    let node_with_raw_child = Node::try_new(
        DeepTag::Var,
        Metadata::default(),
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

    assert!(
        probe(raw_vocabulary_form("let"))
            .unwrap_err()
            .to_string()
            .contains("let")
    );

    let unknown_with_raw_child = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: Metadata::default(),
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
        meta: Metadata::default(),
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
fn canonical_stamped_function_with_typed_params_has_no_structural_warning() {
    let exprs = chelis_deep::parse_and_stamp_file(
        "(def {} negate (fn {} (params {} (x {type: (t-prim {} f32)})) \
         (app {} (var {} neg) (var {} x))))",
    )
    .expect("canonical function must stamp");

    let warnings = validate(&exprs);
    assert!(
        warnings.is_empty(),
        "canonical stamped function must remain structurally clean: {warnings:#?}"
    );
}

#[test]
fn canonical_stamped_effect_and_property_metadata_have_no_structural_warning() {
    let exprs = chelis_deep::parse_and_stamp_file(
        r#"(defsig {}
  always_true
  (t-fn {eff: (effects {} diff (resource {} "gpu:0"))}
    (t-prim {} f32)
    (t-prim {} bool)))
(def {chelis_role: "property",
       property_preconditions: (tuple {}),
       property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
       property_source_kind: "user"}
  always_true
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (var {} x))))"#,
    )
    .expect("canonical property declarations must stamp");

    let warnings = validate(&exprs);
    assert!(
        warnings.is_empty(),
        "canonical stamped metadata nodes must remain structurally clean: {warnings:#?}"
    );
}

#[test]
fn node_constructor_rejects_raw_vocabulary_below_its_gate() {
    let result = Node::try_new(
        DeepTag::Def,
        Metadata::default(),
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
fn construction_gate_descends_through_every_unvalidated_carrier() {
    // chelis#1109 stops the construction scan at a stamped `Node` child,
    // whose subtree cleared the same scan already. Nothing validated the
    // other carriers, so the scan must still reach a raw tag buried at any
    // depth below them, in children and in metadata alike.
    let clean = Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(1), sp())],
        sp(),
    );

    for (tag, payload) in [
        (
            "if",
            Expr::BareList(
                vec![Expr::BareList(vec![raw_vocabulary_form("if")], sp())],
                sp(),
            ),
        ),
        (
            "tuple",
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "future-form".into(),
                meta: Metadata::default(),
                children: vec![Expr::BareList(vec![raw_vocabulary_form("tuple")], sp())],
                span: sp(),
            })),
        ),
    ] {
        assert!(
            probe(payload.clone())
                .unwrap_err()
                .to_string()
                .contains(tag)
        );
        assert!(matches!(
            Node::try_new(
                DeepTag::App,
                Metadata::default(),
                vec![clean.clone(), payload]
            ),
            Err(NodeError::RawVocabularyTag { .. })
        ));
    }

    // Stopping at one stamped child must not stop the scan of its siblings.
    assert!(
        matches!(
            Node::try_new(
                DeepTag::App,
                Metadata::default(),
                vec![clean, raw_vocabulary_form("let")],
            ),
            Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "let"
        ),
        "a stamped sibling must not shadow a raw tag later in the child list"
    );
}

#[test]
fn deeply_nested_stamped_construction_scans_each_level_once() {
    // Bottom-up stamping constructs one gated node per level, and before
    // chelis#1109 each construction re-walked the entire subtree below it,
    // so the total cost was quadratic in nesting depth (measured in
    // release: 3,200 levels took 24.4ms, versus 0.14ms after the fix).
    //
    // There is no parser depth limit to lean on here: `parse_raw_str`
    // parses a 1,000-level chain and simply overflows the stack somewhere
    // before 2,000. The depth below is bounded by the DESCENDING oracle
    // this test asserts, which recurses once per level and overflows a
    // debug test thread between 2,500 and 3,000; construction itself is
    // iterative and stays fine past 8,000, and dropping the tree recurses
    // too (fine at 3,000, overflows by 8,000). 1,000 keeps the margin.
    //
    // This is a shape regression test, not a wall-clock gate -- the
    // behavioral lock on the boundary skip lives in node.rs's
    // `construction_scan_stops_at_a_stamped_node_boundary`.
    let mut expr = Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(1), sp())],
        sp(),
    );
    for _ in 0..1000 {
        expr = Expr::node(DeepTag::App, Metadata::default(), vec![expr], sp());
    }

    assert_eq!(find_raw_vocabulary_tag(std::slice::from_ref(&expr)), None);
}

#[test]
fn node_metadata_replacement_revalidates_before_commit() {
    let mut node = Node::try_new(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("x".to_string()), sp())],
    )
    .expect("clean node must construct");

    let candidate = node.meta().clone();
    let rejected = chelis_deep::annotations::RuntimeExpression::try_new(raw_vocabulary_form("lit"));
    assert!(rejected.unwrap_err().to_string().contains("lit"));
    assert!(candidate.is_empty());
    assert!(node.meta().is_empty());
    node.try_replace_meta(Metadata::from(
        chelis_deep::annotations::MetadataValue::Span(
            chelis_deep::annotations::SpanId::try_new("source:1".into(), sp()).unwrap(),
        ),
    ))
    .unwrap();
    assert_eq!(node.meta().span_id().unwrap().value(), "source:1");
}

#[test]
fn node_child_replacement_revalidates_before_commit() {
    let original = Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(1), sp())],
        sp(),
    );
    let mut node = Node::try_new(
        DeepTag::Def,
        Metadata::default(),
        vec![
            Expr::Atom(Atom::Name("f".to_string()), sp()),
            original.clone(),
        ],
    )
    .expect("clean node must construct");

    let rejected = node.try_replace_child(1, raw_vocabulary_form("lit"));
    assert!(matches!(
        rejected,
        Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "lit"
    ));
    assert_eq!(
        node.expr_child(1),
        &original,
        "a rejected child replacement must leave the original node intact"
    );
}

#[test]
fn node_children_replacement_revalidates_arity_before_commit() {
    let original = Expr::Atom(Atom::Name("x".to_string()), sp());
    let mut node = Node::try_new(DeepTag::Var, Metadata::default(), vec![original.clone()])
        .expect("clean node must construct");

    let rejected = node.try_replace_children(Vec::new());
    assert!(matches!(rejected, Err(NodeError::ArityViolation { .. })));
    assert_eq!(node.child_count(), 1);
    assert!(matches!(
        node.children_iter().next(),
        Some(chelis_deep::node::ChildRef::Syntax(child)) if child == &original
    ));
}

#[test]
fn grad_admission_accepts_one_optional_selector_and_rejects_extra_children() {
    let target = Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(1), sp())],
        sp(),
    );
    let selector = Expr::Atom(Atom::Int(0), sp());

    assert!(Node::try_new(DeepTag::Grad, Metadata::default(), vec![target.clone()]).is_ok());
    assert!(
        Node::try_new(
            DeepTag::Grad,
            Metadata::default(),
            vec![target.clone(), selector.clone()],
        )
        .is_ok()
    );

    let rejected = Node::try_new(
        DeepTag::Grad,
        Metadata::default(),
        vec![target, selector, Expr::Atom(Atom::Int(1), sp())],
    );
    assert!(matches!(
        rejected,
        Err(NodeError::ArityViolation {
            tag: DeepTag::Grad,
            expected: AritySpec::Range(1, 2),
            actual: 3,
        })
    ));
}

/// A vocabulary word has one atom spelling, a name, and a name is never a
/// runtime expression.
#[test]
fn node_constructor_rejects_tag_atom_at_runtime_expr_slot() {
    let result = Node::try_new(
        DeepTag::Def,
        Metadata::default(),
        vec![
            Expr::Atom(Atom::Name("f".to_string()), sp()),
            Expr::Atom(Atom::Name("lit".to_string()), sp()),
        ],
    );

    assert!(matches!(
        result,
        Err(NodeError::NameAtExprSlot { ref name, .. }) if name == "lit"
    ));
}

#[test]
fn node_public_api_exposes_no_raw_mutable_child_borrows() {
    let source = include_str!("../src/node.rs");
    assert!(!source.contains("pub fn children_slice_mut("));
    assert!(!source.contains("pub fn children_vec_mut("));
}

#[test]
fn typed_expr_constructor_produces_a_gated_node() {
    let expr = Expr::node(
        DeepTag::Lit,
        Metadata::default(),
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
            Metadata::default(),
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
