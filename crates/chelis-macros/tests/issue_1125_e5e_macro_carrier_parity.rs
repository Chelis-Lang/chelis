//! Chelis #1125 PP7 E5e: macro readers consume the total Deep carrier view.

use chelis_deep::ast::{Atom, Expr, ExprCarrier, List, MetaExpr, Metadata, UnknownFormData};
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

fn raw_form(head: &str, children: Vec<Expr>) -> Expr {
    let mut elements = vec![name(head), Expr::Map(Metadata::default(), sp())];
    elements.extend(children);
    Expr::List(List { elements }, sp())
}

fn transitional_decoded(tag: DeepTag, children: Vec<Expr>) -> Expr {
    let mut elements = vec![
        Expr::Atom(Atom::Tag(tag), sp()),
        Expr::Map(Metadata::default(), sp()),
    ];
    elements.extend(children);
    Expr::List(List { elements }, sp())
}

fn malformed_form(head: Expr, children: Vec<Expr>) -> Expr {
    let mut elements = vec![head, name("not-metadata")];
    elements.extend(children);
    Expr::List(List { elements }, sp())
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

fn transitional_macro_call(name_value: &str, argument: Expr) -> Expr {
    transitional_decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name(name_value)]), argument],
    )
}

fn direct_invalid_result_error(argument: Expr) -> ExpansionError {
    let nested_call = transitional_macro_call("identity", argument);
    let successor_parent = decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name("sink")]), nested_call],
    );
    let program = vec![
        macro_definition("identity", decoded(DeepTag::Var, vec![name("value")])),
        successor_parent,
    ];

    expand_program(&program, &options())
        .expect_err("an invalid direct macro result must reject at successor reconstruction")
}

fn direct_invalid_body_result_error(body: Expr) -> ExpansionError {
    let nested_call = transitional_macro_call(
        "invalid_body",
        decoded(DeepTag::Lit, vec![Expr::Atom(Atom::Int(0), sp())]),
    );
    let successor_parent = decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name("sink")]), nested_call],
    );
    let program = vec![macro_definition("invalid_body", body), successor_parent];

    expand_program(&program, &options())
        .expect_err("an invalid macro body result must reject at successor reconstruction")
}

fn assert_successor_decoded_carriers(expr: &Expr) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, children) => {
            assert!(
                matches!(expr, Expr::Node(_, _)),
                "a successor decoded node must not be rewritten into a legacy List: {expr:?}"
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
        ExprCarrier::MalformedLegacyList(list) => {
            for element in &list.elements {
                assert_successor_decoded_carriers(element);
            }
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
        Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Var), sp()),
                    name("not-metadata"),
                ],
            },
            sp(),
        ),
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
    assert!(matches!(
        expanded.exprs()[5].carrier(),
        ExprCarrier::MalformedLegacyList(_)
    ));
}

#[test]
fn production_macro_readers_have_no_node_to_list_bridge() {
    let source = include_str!("../src/lib.rs");
    assert!(
        !source.contains(".to_list("),
        "macro reader walks must consume ExprCarrier directly"
    );
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
fn malformed_macro_body_is_opaque_to_substitution_hygiene_and_source_annotation() {
    let malformed_fn = malformed_form(
        Expr::Atom(Atom::Tag(DeepTag::Fn), sp()),
        vec![
            decoded(DeepTag::Params, vec![name("x")]),
            decoded(
                DeepTag::Var,
                vec![decoded(DeepTag::Var, vec![name("value")])],
            ),
        ],
    );
    let program = vec![
        macro_definition("opaque", malformed_fn.clone()),
        macro_call("opaque", decoded(DeepTag::Var, vec![name("argument")])),
    ];

    let expanded = expand_program(&program, &options()).expect("macro expansion succeeds");
    assert_eq!(
        expanded.exprs(),
        std::slice::from_ref(&malformed_fn),
        "a malformed semantic carrier must remain byte-for-byte structurally opaque"
    );
}

#[test]
fn macro_body_odd_bind_survives_substitution_and_hygiene() {
    let body = decoded(
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
    let program = vec![
        macro_definition("odd_bind", body),
        macro_call("odd_bind", decoded(DeepTag::Var, vec![name("argument")])),
    ];

    let expanded = expand_program(&program, &options()).expect("macro expansion succeeds");
    let [expanded_let] = expanded.exprs() else {
        panic!("expected one expanded macro body");
    };
    let ExprCarrier::DecodedNode(DeepTag::Let, _, let_children) = expanded_let.carrier() else {
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

#[test]
fn malformed_raw_macro_call_is_not_admitted_or_rewritten() {
    let malformed_call = malformed_form(name("macro-invoke"), vec![name("keep"), name("argument")]);
    let program = vec![
        macro_definition("keep", decoded(DeepTag::Var, vec![name("value")])),
        malformed_call.clone(),
    ];

    let expanded = expand_program(&program, &options()).expect("malformed raw call stays opaque");
    assert_eq!(
        expanded.exprs(),
        std::slice::from_ref(&malformed_call),
        "a malformed raw call must not be admitted or rewritten"
    );
}

#[test]
fn malformed_transitional_call_rejects_instead_of_panicking_during_reconstruction() {
    let body = decoded(
        DeepTag::App,
        vec![
            decoded(DeepTag::Var, vec![name("callee")]),
            decoded(DeepTag::Var, vec![name("value")]),
        ],
    );
    let malformed_call = transitional_decoded(
        DeepTag::App,
        vec![decoded(DeepTag::Var, vec![name("wrap")]), name("bare")],
    );
    let program = vec![macro_definition("wrap", body), malformed_call];

    let error = expand_program(&program, &options())
        .expect_err("a bare runtime argument must reject without reconstruction panic");
    assert!(
        error
            .to_string()
            .contains("structural name `bare` at RuntimeExpr child position"),
        "the rejection must retain the deciding successor-node invariant: {error}"
    );
}

#[test]
fn direct_name_result_under_successor_parent_is_typed_rejection() {
    assert!(matches!(
        direct_invalid_result_error(name("bare")),
        ExpansionError::InvalidNode(_)
    ));
}

#[test]
fn direct_tag_result_under_successor_parent_is_typed_rejection() {
    assert!(matches!(
        direct_invalid_result_error(Expr::Atom(Atom::Tag(DeepTag::Lit), sp())),
        ExpansionError::InvalidNode(_)
    ));
}

#[test]
fn direct_raw_vocabulary_result_under_successor_parent_is_typed_rejection() {
    assert!(matches!(
        direct_invalid_body_result_error(raw_form("lit", vec![Expr::Atom(Atom::Int(1), sp())],)),
        ExpansionError::InvalidNode(_)
    ));
}
