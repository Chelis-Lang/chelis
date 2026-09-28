//! [03-META-1]: typed payload admission and checker placement defenses.
use chelis_deep::annotations::{DtypeBounds, MetadataValue as M, RuntimeExpression};
use chelis_deep::node::Node;
use chelis_deep::{Atom, DeepTag, Expr, Metadata, Span};
const SPAN: Span = Span { offset: 0, len: 0 };
fn verdicts(source: &str) -> Vec<Vec<chelis_types::errors::CheckError>> {
    verdicts_for(&chelis_deep::parser::parse_str(source).unwrap())
}
fn verdicts_for(exprs: &[chelis_deep::Expr]) -> Vec<Vec<chelis_types::errors::CheckError>> {
    use chelis_types::*;
    let library = chelis_deep::parser::parse_str("(def {} library_value (lit {} 1))").unwrap();
    let context = build_type_env_from_library(&library).unwrap();
    vec![
        check_ir_program(exprs)
            .err()
            .map(|e| e.errors)
            .unwrap_or_default(),
        check_typed_program(exprs)
            .err()
            .map(|e| e.errors)
            .unwrap_or_default(),
        build_type_env_from_library(exprs)
            .err()
            .map(|e| e.errors)
            .unwrap_or_default(),
        build_compiled_library_context(exprs)
            .err()
            .map(|e| e.errors)
            .unwrap_or_default(),
        build_compiled_library_context_with_base(&context, exprs)
            .err()
            .map(|e| e.errors)
            .unwrap_or_default(),
        check_ir_with_context(&context, exprs)
            .err()
            .map(|e| e.errors)
            .unwrap_or_default(),
        infer_ir_program(exprs).errors,
        infer_program(exprs).errors,
    ]
}

#[test]
fn checked_effect_replacement_also_checks_metadata_before_publication() {
    let original =
        chelis_deep::parser::parse_str("(def {doc: \"keep\"} f (fn {} (params {}) (lit {} 1)))")
            .unwrap();
    let checked = chelis_types::check_ir_program(&original).unwrap();
    checked
        .try_with_effect_annotations(checked.exprs().to_vec())
        .unwrap();
    let mut changed = checked.exprs().to_vec();
    match &mut changed[0] {
        Expr::Node(node, _) => {
            let mut meta = node.meta().clone();
            meta.remove(chelis_deep::annotations::MetadataKey::Doc);
            node.try_replace_meta(meta).unwrap();
        }
        _ => panic!("declaration"),
    }
    checked
        .try_with_effect_annotations(changed)
        .expect_err("derived-effect rewrite cannot discard source annotations");
    let malformed = serde_json::json!({"entries": [["effects", Expr::Atom(Atom::Int(1), SPAN)]]});
    assert!(serde_json::from_value::<Metadata>(malformed).is_err());
}

#[test]
fn metadata_admission_precedes_inference_on_every_checker_session() {
    for (source, key) in [
        ("(def {span: 1} f (lit {} 1))", "span"),
        ("(def {} f (lit {type: unknown_type_syntax} 1))", "type"),
        ("(def {} f (grad {wrt: x} (var {} missing)))", "wrt"),
    ] {
        let error = chelis_deep::parser::parse_str(source).unwrap_err();
        assert!(error.to_string().contains(key), "{error}");
    }
    // A locally misplaced, well-shaped payload has no admitted carrier: the
    // programmatic `Node` constructor checks placement before any checker
    // session can resolve the unbound body.
    let metadata = Metadata::from(M::DtypeBounds(DtypeBounds::try_new([], SPAN).unwrap()));
    let error = Node::try_new(
        DeepTag::Def,
        metadata,
        vec![
            Expr::Atom(Atom::Name("f".into()), SPAN),
            Expr::node(
                DeepTag::Var,
                Metadata::default(),
                vec![Expr::Atom(Atom::Name("missing".into()), SPAN)],
                SPAN,
            ),
        ],
    )
    .expect_err("a def cannot carry dtype_bounds");
    assert!(error.to_string().contains("dtype_bounds"), "{error}");
}

#[test]
fn valid_metadata_remains_admissible_on_every_checker_session() {
    for errors in verdicts("(def {span: \"producer_id\", doc: \"example\"} f (lit {} 1))") {
        assert!(errors.is_empty(), "{errors:?}");
    }
}

#[test]
fn nested_expression_roles_are_checked_before_publication() {
    let node = |tag, children: Vec<Expr>| Node::try_new(tag, Metadata::default(), children);
    let name = || Expr::Atom(Atom::Name("missing".into()), SPAN);

    // A bare name in the callee slot has no admitted carrier: the Node
    // constructor rejects it before any payload or checker can see it.
    let error =
        node(DeepTag::App, vec![name()]).expect_err("a bare callee name is not an expression");
    assert!(error.to_string().contains("missing"), "{error}");

    let callee = Expr::Node(
        Box::new(node(DeepTag::Var, vec![name()]).expect("var node")),
        SPAN,
    );
    let app = Expr::Node(
        Box::new(node(DeepTag::App, vec![callee]).expect("app node")),
        SPAN,
    );
    let payload = RuntimeExpression::try_new(app).expect("a stamped app is a runtime payload");
    let metadata = Metadata::from(M::PropertySeed(payload));
    let expr = Expr::node(
        DeepTag::Def,
        metadata,
        vec![
            Expr::Atom(Atom::Name("f".into()), SPAN),
            Expr::node(
                DeepTag::Lit,
                Metadata::default(),
                vec![Expr::Atom(Atom::Int(1), SPAN)],
                SPAN,
            ),
        ],
        SPAN,
    );
    for errors in verdicts_for(&[expr]) {
        assert!(errors.is_empty(), "{errors:?}");
    }
}
