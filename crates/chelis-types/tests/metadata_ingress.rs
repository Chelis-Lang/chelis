//! [03-META-1]: ungated legacy input cannot bypass checker admission.
#[path = "../../../tests/support/legacy_metadata.rs"]
mod legacy_metadata;

fn verdicts(source: &str) -> Vec<Vec<chelis_types::errors::CheckError>> {
    verdicts_for(&legacy_metadata::legacy_metadata_fixture(source))
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
        chelis_deep::parser::parse_str("(def {} f (fn {} (params {}) (lit {} 1)))").unwrap();
    let checked = chelis_types::check_ir_program(&original).unwrap();
    checked
        .try_with_effect_annotations(checked.exprs().to_vec())
        .unwrap();
    let bad = legacy_metadata::legacy_metadata_fixture(
        "(def {} f (fn {effects: 1} (params {}) (lit {} 1)))",
    );
    let result = checked.try_with_effect_annotations(bad).unwrap_err();
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].message.contains("metadata `effects`"));
}

#[test]
fn metadata_admission_precedes_inference_on_every_checker_session() {
    for (source, key) in [
        ("(def {span: 1} f (lit {} 1))", "span"),
        ("(def {} f (lit {type: unknown_type_syntax} 1))", "type"),
        ("(def {} f (grad {wrt: x} (var {} missing)))", "wrt"),
        (
            "(def {dtype_bounds: {p: float}} f (lit {} 1))",
            "dtype_bounds",
        ),
    ] {
        for errors in verdicts(source) {
            assert_eq!(errors.len(), 1, "{source}: {errors:?}");
            assert!(matches!(
                errors[0].kind,
                chelis_types::errors::CheckErrorKind::MalformedForm
            ));
            assert!(errors[0].message.contains(key), "{:?}", errors[0]);
            assert!(errors[0].span_offset.is_some());
        }
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
    use chelis_deep::{Atom, DeepTag, Expr, List, MetaMap, Span};
    let span = Span { offset: 0, len: 0 };
    let list = |tag, children: Vec<Expr>| {
        let mut elements = vec![
            Expr::Atom(Atom::Tag(tag), span),
            Expr::Map(MetaMap::default(), span),
        ];
        elements.extend(children);
        Expr::List(List { elements }, span)
    };
    for valid in [true, false] {
        let name = Expr::Atom(Atom::Name("missing".into()), span);
        let callee = if valid {
            list(DeepTag::Var, vec![name])
        } else {
            name
        };
        let meta = MetaMap {
            entries: vec![("property_seed".into(), list(DeepTag::App, vec![callee]))],
        };
        let exprs = vec![Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Def), span),
                    Expr::Map(meta, span),
                    Expr::Atom(Atom::Name("f".into()), span),
                    list(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), span)]),
                ],
            },
            span,
        )];
        for errors in verdicts_for(&exprs) {
            if valid {
                assert!(errors.is_empty(), "{errors:?}");
            } else {
                assert_eq!(errors.len(), 1, "{errors:?}");
                assert!(
                    errors[0].message.contains("metadata `property_seed`"),
                    "{errors:?}"
                );
            }
        }
    }
}
