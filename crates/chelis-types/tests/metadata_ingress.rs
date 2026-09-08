//! [03-META-1]: typed payload admission and checker placement defenses.
use chelis_deep::annotations::{DtypeBounds, MetadataValue as M, RuntimeExpression};
use chelis_deep::{Atom, DeepTag, Expr, List, Metadata, Span};
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
        Expr::List(list, _) => {
            let Expr::Map(meta, _) = &mut list.elements[1] else {
                panic!("metadata slot")
            };
            meta.remove(chelis_deep::annotations::MetadataKey::Doc);
        }
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
    // Legacy Lists can still assemble a locally misplaced, well-shaped payload.
    // Every checker ingress must reject it before resolving the unbound body.
    let metadata = Metadata::from(M::DtypeBounds(DtypeBounds::try_new([], SPAN).unwrap()));
    let expr = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Def), SPAN),
                Expr::Map(metadata, SPAN),
                Expr::Atom(Atom::Name("f".into()), SPAN),
                Expr::node(
                    DeepTag::Var,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Name("missing".into()), SPAN)],
                    SPAN,
                ),
            ],
        },
        SPAN,
    );
    for errors in verdicts_for(&[expr]) {
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(matches!(
            errors[0].kind,
            chelis_types::errors::CheckErrorKind::MalformedForm
        ));
        assert!(errors[0].message.contains("dtype_bounds"));
        assert!(errors[0].span_offset.is_some());
    }
}

#[test]
fn valid_metadata_remains_admissible_on_every_checker_session() {
    for errors in verdicts("(def {span: \"producer_id\", doc: \"example\"} f (lit {} 1))") {
        assert!(errors.is_empty(), "{errors:?}");
    }
}

#[test]
fn nested_legacy_expression_roles_are_checked_before_publication() {
    let list = |tag, children: Vec<Expr>| {
        let mut elements = vec![
            Expr::Atom(Atom::Tag(tag), SPAN),
            Expr::Map(Metadata::default(), SPAN),
        ];
        elements.extend(children);
        Expr::List(List { elements }, SPAN)
    };
    for valid in [true, false] {
        let name = Expr::Atom(Atom::Name("missing".into()), SPAN);
        let callee = if valid {
            list(DeepTag::Var, vec![name])
        } else {
            name
        };
        let payload = RuntimeExpression::try_new(list(DeepTag::App, vec![callee]));
        assert_eq!(payload.is_ok(), valid);
        if let Ok(payload) = payload {
            let metadata = Metadata::from(M::PropertySeed(payload));
            let expr = Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Tag(DeepTag::Def), SPAN),
                        Expr::Map(metadata, SPAN),
                        Expr::Atom(Atom::Name("f".into()), SPAN),
                        list(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), SPAN)]),
                    ],
                },
                SPAN,
            );
            for errors in verdicts_for(&[expr]) {
                assert!(errors.is_empty(), "{errors:?}");
            }
        }
    }
}
