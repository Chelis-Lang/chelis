//! Chelis #1125 PP7 E5e: macro readers consume the total Deep carrier view.

use chelis_deep::ast::{Atom, Expr, ExprCarrier, MetaExpr, Metadata, UnknownFormData};
use chelis_deep::{DeepTag, Span};
use chelis_macros::{ExpansionError, ExpansionOptions, expand_program};

fn sp() -> Span {
    Span::new(7, 11)
}

fn name(value: &str) -> Expr {
    Expr::Atom(Atom::Name(value.to_string()), sp())
}

fn decoded(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(tag, Metadata::default(), children, sp())
}

/// A head that did not decode, with its metadata and children: the one
/// in-memory spelling of a raw-string form.
fn raw_form(head: &str, children: Vec<Expr>) -> Expr {
    Expr::UnknownForm(Box::new(UnknownFormData {
        head: head.to_string(),
        meta: Metadata::default(),
        children,
        span: sp(),
    }))
}

fn options() -> ExpansionOptions {
    ExpansionOptions {
        max_iterations: 100,
        load_std_prelude: false,
    }
}

fn macro_definition(name_value: &str, body: Expr) -> Expr {
    raw_form(
        "defmacro",
        vec![
            name(name_value),
            decoded(DeepTag::Params, vec![name("value")]),
            body,
        ],
    )
}

fn macro_call(name_value: &str, argument: Expr) -> Expr {
    decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name(name_value)]), argument],
    )
}

/// A zero-parameter macro, so its invalid body is admitted under
/// [02-MACRO-4] and the rejection comes from successor reconstruction.
fn direct_invalid_body_result_error(body: Expr) -> ExpansionError {
    let definition = raw_form(
        "defmacro",
        vec![
            name("invalid_body"),
            decoded(DeepTag::Params, Vec::new()),
            body,
        ],
    );
    let nested_call = decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name("invalid_body")])],
    );
    let successor_parent = decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name("sink")]), nested_call],
    );
    let program = vec![definition, successor_parent];

    expand_program(&program, &options())
        .expect_err("an invalid macro body result must reject at successor reconstruction")
}

fn assert_successor_decoded_carriers(expr: &Expr) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, children) => {
            assert!(
                matches!(expr, Expr::Node(_, _)),
                "a decoded node must remain a stamped Node: {expr:?}"
            );
            metadata.visit_syntax(&mut |_, value| assert_successor_decoded_carriers(value));
            for child in children {
                assert_successor_decoded_carriers(child);
            }
        }
        ExprCarrier::StructuralList(elements) => {
            for element in elements {
                assert_successor_decoded_carriers(element);
            }
        }
        ExprCarrier::UndecodableHead(_, metadata, children) => {
            metadata.visit_syntax(&mut |_, value| assert_successor_decoded_carriers(value));
            for child in children {
                assert_successor_decoded_carriers(child);
            }
        }
        ExprCarrier::MetadataMap(metadata) => {
            metadata.visit_syntax(&mut |_, value| assert_successor_decoded_carriers(value));
        }
        ExprCarrier::MetadataExpression(meta) => {
            meta.metadata
                .visit_syntax(&mut |_, value| assert_successor_decoded_carriers(value));
            assert_successor_decoded_carriers(&meta.expr);
        }
        ExprCarrier::Atom(_) => {}
    }
}

#[test]
fn macro_pipeline_preserves_successor_carriers_across_reader_walks() {
    let source = "\
(defmacro {} wrap (params {} value)
  (let {}
    (bind {} local (var {} value))
    (match {} (var {} local)
      (arm {} (pat-var {} item) () (var {} item)))))
(def {} result
  (app {audit: (var {} metadata_name)} (var {} wrap) (lit {} 9)))";
    let parsed = chelis_deep::parser::parse_str(source).expect("fixture stamps");
    assert!(
        parsed.iter().any(|expr| matches!(expr, Expr::Node(_, _))),
        "the fixture must reach the successor Node carrier"
    );

    let expanded = expand_program(&parsed, &options()).expect("macro expansion succeeds");

    for expr in expanded.exprs() {
        assert_successor_decoded_carriers(expr);
    }
}

#[test]
fn macro_walks_preserve_every_explicit_nondecoded_carrier_role() {
    let carriers = vec![
        name("atom"),
        Expr::Map(Metadata::default(), sp()),
        Expr::MetaExpr(
            MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(name("wrapped")),
            },
            sp(),
        ),
        Expr::BareList(vec![name("structural")], sp()),
        Expr::UnknownForm(Box::new(UnknownFormData {
            head: "future-form".to_string(),
            meta: Metadata::default(),
            children: vec![name("payload")],
            span: sp(),
        })),
    ];

    let expanded = expand_program(&carriers, &options()).expect("nondecoded carrier walk succeeds");

    assert_eq!(
        expanded.exprs(),
        carriers.as_slice(),
        "a macro-free walk must preserve every nondecoded carrier, span, and payload exactly"
    );
    assert!(matches!(
        expanded.exprs()[0].carrier(),
        ExprCarrier::Atom(_)
    ));
    assert!(matches!(
        expanded.exprs()[1].carrier(),
        ExprCarrier::MetadataMap(_)
    ));
    assert!(matches!(
        expanded.exprs()[2].carrier(),
        ExprCarrier::MetadataExpression(_)
    ));
    assert!(matches!(
        expanded.exprs()[3].carrier(),
        ExprCarrier::StructuralList(_)
    ));
    assert!(matches!(
        expanded.exprs()[4].carrier(),
        ExprCarrier::UndecodableHead("future-form", _, _)
    ));
}

#[test]
fn nonlegacy_macro_invoke_carriers_are_not_reclassified_as_the_raw_boundary() {
    let mut program =
        chelis_deep::parser::parse_str("(defmacro {} keep (params {} value) (var {} value))")
            .expect("macro definition stamps");
    program.push(Expr::UnknownForm(Box::new(UnknownFormData {
        head: "macro-invoke".to_string(),
        meta: Metadata::default(),
        children: vec![name("keep"), name("argument")],
        span: sp(),
    })));
    program.push(Expr::BareList(
        vec![
            name("macro-invoke"),
            Expr::Map(Metadata::default(), sp()),
            name("keep"),
            name("argument"),
        ],
        sp(),
    ));

    let expanded = expand_program(&program, &options()).expect("unknown form walk succeeds");

    assert!(matches!(
        expanded.exprs(),
        [Expr::UnknownForm(data), Expr::BareList(elements, _)]
            if data.head == "macro-invoke"
                && matches!(
                    data.children.as_slice(),
                    [Expr::Atom(Atom::Name(name), _), ..] if name == "keep"
                )
                && matches!(
                    elements.as_slice(),
                    [Expr::Atom(Atom::Name(head), _), _, Expr::Atom(Atom::Name(name), _), ..]
                        if head == "macro-invoke" && name == "keep"
                )
    ));
}

#[test]
fn macro_free_odd_bind_keeps_its_unpaired_child() {
    let malformed_let = decoded(
        DeepTag::Let,
        vec![
            decoded(
                DeepTag::Bind,
                vec![
                    name("x"),
                    decoded(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), sp())]),
                    name("dangling"),
                ],
            ),
            decoded(DeepTag::Var, vec![name("x")]),
        ],
    );

    let expanded =
        expand_program(std::slice::from_ref(&malformed_let), &options()).expect("walk succeeds");
    assert_eq!(
        expanded.exprs(),
        std::slice::from_ref(&malformed_let),
        "macro expansion must not truncate an odd bind"
    );
}

#[test]
fn macro_body_odd_bind_survives_substitution_and_hygiene() {
    let malformed_let = decoded(
        DeepTag::Let,
        vec![
            decoded(
                DeepTag::Bind,
                vec![
                    name("x"),
                    decoded(DeepTag::Var, vec![name("value")]),
                    name("dangling"),
                ],
            ),
            decoded(DeepTag::Var, vec![name("x")]),
        ],
    );
    // Substitution does not enter the malformed let, so the body references
    // the parameter outside it as well ([02-MACRO-4]).
    let body = decoded(
        DeepTag::App,
        vec![
            decoded(DeepTag::Var, vec![name("pair")]),
            malformed_let,
            decoded(DeepTag::Var, vec![name("value")]),
        ],
    );
    let program = vec![
        macro_definition("odd_bind", body),
        macro_call("odd_bind", decoded(DeepTag::Var, vec![name("argument")])),
    ];

    let expanded = expand_program(&program, &options()).expect("macro expansion succeeds");
    let [expanded_app] = expanded.exprs() else {
        panic!("expected one expanded macro body");
    };
    let ExprCarrier::DecodedNode(DeepTag::App, _, app_children) = expanded_app.carrier() else {
        panic!("expected the body call to remain a call");
    };
    let ExprCarrier::DecodedNode(DeepTag::Let, _, let_children) = app_children[1].carrier() else {
        panic!("expected the malformed let to remain a let");
    };
    let ExprCarrier::DecodedNode(DeepTag::Bind, _, bind_children) = let_children[0].carrier()
    else {
        panic!("expected the malformed bind to remain a bind");
    };
    assert_eq!(
        bind_children.len(),
        3,
        "substitution and hygiene must not discard the unpaired bind child"
    );
    assert_eq!(bind_children[2], name("dangling"));
}

/// A macro whose body is a bare name places that name at its caller's
/// runtime slot; the stamped parent refuses it with a typed error rather
/// than panicking during reconstruction.
#[test]
fn direct_name_result_under_successor_parent_is_typed_rejection() {
    let error = direct_invalid_body_result_error(name("bare"));
    assert!(matches!(error, ExpansionError::InvalidNode(_)));
    assert!(
        error
            .to_string()
            .contains("structural name `bare` at RuntimeExpr child position"),
        "the rejection must retain the deciding successor-node invariant: {error}"
    );
}

#[test]
fn direct_raw_vocabulary_result_under_successor_parent_is_typed_rejection() {
    assert!(matches!(
        direct_invalid_body_result_error(raw_form("lit", vec![Expr::Atom(Atom::Int(1), sp())],)),
        ExpansionError::InvalidNode(_)
    ));
}
