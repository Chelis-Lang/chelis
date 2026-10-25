use super::*;
use chelis_deep::parser::parse_str;

fn effect_error_messages(errors: Vec<EffectError>) -> Vec<String> {
    errors
        .into_iter()
        .map(|error| format!("{:?}: {}", error.kind, error.message))
        .collect()
}

#[test]
fn effects_reader_classes_read_a_recursive_stamped_program() {
    let successor = chelis_deep::parse_and_stamp_file(
        r#"(module {} Test
         (defsig {} entry
           (t-fn {eff: (effects {})} (t-prim {} unit)))
         (def {} entry
           (fn {}
             (params {})
             (let {}
               (bind {} local
                 (fn {}
                   (params {})
                   (app {} (var {} debug) (lit {} 1))))
               (app {} (var {} local))))))"#,
        &chelis_syntax_host::NativeStack,
    )
    .expect("recursive effects fixture stamps");

    let successor_bodies = top_level_def_bodies(&successor);
    assert_eq!(successor_bodies.keys().collect::<Vec<_>>(), ["entry"]);

    let (successor_effects, successor_callables) =
        Analysis::default().infer_program_effects(&successor);
    assert!(
        successor_effects
            .get("entry")
            .is_some_and(|effects| effects.contains(&Effect::Io))
    );

    let mut successor_errors = Vec::new();
    validate_declared_vs_inferred(&successor, &successor_effects, &mut successor_errors);
    assert!(
        effect_error_messages(successor_errors)
            .iter()
            .any(|message| message.contains("`entry`")),
        "the declared-pure `entry` must be rejected for its local closure's IO"
    );

    let successor_annotated = successor
        .iter()
        .map(|expr| {
            Analysis::default().annotate_effects(
                expr,
                &successor_effects,
                &successor_callables,
                &mut Locals::new(),
            )
        })
        .collect::<Vec<_>>();
    assert!(
        chelis_deep::printer::print_canonical(&successor_annotated).contains("eff"),
        "the annotated closure carries its inferred effect row"
    );
}

#[test]
fn guarded_var_and_literal_readers_read_decoded_nodes() {
    let span = Span::new(0, 0);
    let successor_var = Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("value".into()), span)],
        span,
    );
    let successor_lit = Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Str("gpu:0".into()), span)],
        span,
    );
    assert_eq!(var_name(&successor_var), Some("value"));
    assert_eq!(string_literal(&successor_lit), Some("gpu:0"));

    let wrong_tag = Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Str("gpu:0".into()), span)],
        span,
    );
    assert_eq!(string_literal(&wrong_tag), None);
    assert_eq!(
        var_name(&Expr::BareList(
            vec![Expr::Atom(Atom::Name("value".into()), span)],
            span,
        )),
        None
    );
}

#[test]
fn let_effect_reader_traverses_every_non_bind_slot_carrier() {
    let span = Span::new(0, 0);
    let name = |value: &str| Expr::Atom(Atom::Name(value.into()), span);
    let io_expr = || {
        Expr::node(
            DeepTag::App,
            Metadata::default(),
            vec![
                Expr::node(DeepTag::Var, Metadata::default(), vec![name("debug")], span),
                Expr::node(
                    DeepTag::Lit,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Int(1), span)],
                    span,
                ),
            ],
            span,
        )
    };
    let pure_expr = || {
        Expr::node(
            DeepTag::Lit,
            Metadata::default(),
            vec![Expr::Atom(Atom::Int(0), span)],
            span,
        )
    };

    let carrier_cases = |child: Expr| {
        [
            ("structural-list", Expr::BareList(vec![child.clone()], span)),
            (
                "undecodable-head",
                Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                    head: "future-bind".into(),
                    meta: Metadata::default(),
                    children: vec![child.clone()],
                    span,
                })),
            ),
            (
                "metadata-expression",
                Expr::MetaExpr(
                    chelis_deep::MetaExpr {
                        metadata: Metadata::default(),
                        expr: Box::new(child.clone()),
                    },
                    span,
                ),
            ),
            (
                "decoded-non-bind",
                Expr::node(DeepTag::Tuple, Metadata::default(), vec![child], span),
            ),
        ]
    };
    let effectful_cases = carrier_cases(io_expr());
    assert!(matches!(
        effectful_cases[0].1.carrier(),
        ExprCarrier::StructuralList(_)
    ));
    assert!(matches!(
        effectful_cases[1].1.carrier(),
        ExprCarrier::UndecodableHead(..)
    ));
    assert!(matches!(
        effectful_cases[2].1.carrier(),
        ExprCarrier::MetadataExpression(_)
    ));
    assert!(matches!(
        effectful_cases[3].1.carrier(),
        ExprCarrier::DecodedNode(DeepTag::Tuple, ..)
    ));

    let hidden = effectful_cases
        .iter()
        .filter_map(|(label, bind_slot)| {
            let effects = Analysis::default().infer_let_effects(
                &[bind_slot.clone(), pure_expr()],
                &BTreeMap::new(),
                &BTreeSet::new(),
                &mut Locals::new(),
            );
            (!effects.contains(&Effect::Io)).then_some(*label)
        })
        .collect::<Vec<_>>();
    assert!(
        hidden.is_empty(),
        "bind-slot carriers hide effectful content: {hidden:?}"
    );

    let pure_cases = carrier_cases(pure_expr());
    let invented = pure_cases
        .iter()
        .filter_map(|(label, bind_slot)| {
            let effects = Analysis::default().infer_let_effects(
                &[bind_slot.clone(), pure_expr()],
                &BTreeMap::new(),
                &BTreeSet::new(),
                &mut Locals::new(),
            );
            effects.contains(&Effect::Io).then_some(*label)
        })
        .collect::<Vec<_>>();
    assert!(
        invented.is_empty(),
        "bind-slot carrier traversal invents effects: {invented:?}"
    );
}

#[test]
fn declared_effect_reader_reads_a_decoded_defsig() {
    let mut parsed = chelis_deep::parse_and_stamp(
        "(defsig {} entry \
       (t-fn {eff: (effects {} accum io)} \
         (t-prim {} i32) \
         (t-prim {} i32)))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("canonical defsig fixture stamps");
    let successor = parsed.remove(0);

    let successor_effects =
        declared_effects_from_defsig(&successor).expect("successor declaration");
    assert!(successor_effects.contains(&Effect::Accum));
    assert!(successor_effects.contains(&Effect::Io));
}

#[test]
fn declared_effect_reader_declines_nondecoded_carriers() {
    let span = Span::new(0, 0);
    for expr in [
        Expr::BareList(vec![], span),
        Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "future-defsig".into(),
            meta: Metadata::default(),
            children: vec![],
            span,
        })),
    ] {
        assert_eq!(declared_effects_from_defsig(&expr), None);
    }
}

#[test]
fn keyed_dropout_introduces_no_effect() {
    // [05-RNG-1]: a draw is a pure function of its key, so a keyed
    // `dropout` carries an empty effect row (the `Random` effect was
    // retired with the counter stream, #2413).
    let exprs = parse_str(
        "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
     (def {} y (app {} (var {} dropout)
       (app {} (var {} key_from_seed) (lit {type: (t-prim {} i64)} 7))
       (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let (inferred, _) = Analysis::default().infer_program_effects(&exprs);
    assert!(
        inferred.get("y").is_some_and(|effects| effects.is_empty()),
        "keyed dropout must infer an empty effect row, got {:?}",
        inferred.get("y")
    );
}

#[test]
fn contextual_rows_exclude_library_and_preserve_shadowing() {
    let library = parse_str(
        "(def {} helper (fn {} (params {}) (app {} (var {} debug) (lit {} 1))))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("library parses");
    let original = library.clone();
    let extension = parse_str(
        "(def {} entry (fn {} (params {}) (app {} (var {} helper))))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("extension parses");
    let rows = infer_effect_rows(&extension, Some(&library));
    assert_eq!(rows.len(), 1);
    assert!(rows["entry"].contains(&Effect::Io));
    let shadow = parse_str(
        "(def {} helper (fn {} (params {}) (lit {} 0))) \
         (def {} entry (fn {} (params {}) (app {} (var {} helper))))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("shadow extension parses");
    let shadow_rows = infer_effect_rows(&shadow, Some(&library));
    assert_eq!(shadow_rows.len(), 2);
    assert!(shadow_rows.values().all(EffectSet::is_empty));
    assert_eq!(library, original);
    assert!(infer_effect_rows(&extension, Some(&library))["entry"].contains(&Effect::Io));
}

#[test]
fn contextual_analysis_validates_only_owned_expressions() {
    let library = parse_str(
        "(def {} old (handle-effect {effect: resource} (var {} device) (lit {} 1)))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("library parses");
    let extension = parse_str(
        "(def {} new (fn {} (params {}) (lit {} 0)))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("extension parses");
    assert!(analyze_effects(&library, None).is_err());
    let annotated = analyze_effects(&extension, Some(&library))
        .expect("library validation belongs to its original host check");
    assert_eq!(annotated, extension);
}

#[test]
fn handler_errors_precede_declared_row_errors() {
    let expressions = parse_str(
        "(defsig {} entry (t-fn {eff: (effects {})} (t-prim {} i32))) \
         (def {} entry (fn {} (params {}) (app {} (var {} debug) (lit {} 1)))) \
         (def {} invalid (handle-effect {effect: resource} (var {} device) (lit {} 1)))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("invalid analysis fixture parses");
    let errors = analyze_effects(&expressions, None).expect_err("both validations must reject");
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].kind, EffectErrorKind::InvalidHandler);
    assert_eq!(errors[1].kind, EffectErrorKind::UnhandledEffect);
    assert!(errors[1].message.contains("`entry`"));
    assert!(errors.iter().all(|error| !error.suggestions.is_empty()));
}

#[cfg(feature = "test-instrumentation")]
#[test]
fn work_profiles_are_explicit_and_isolated() {
    let expressions = parse_str(
        "(def {} entry (fn {} (params {}) (app {} (var {} debug) (lit {} 1))))",
        &chelis_syntax_host::NativeStack,
    )
    .expect("profile fixture parses");
    let (first_result, first) = analyze_effects_with_profile(&expressions, None);
    let (second_result, second) = analyze_effects_with_profile(&expressions, None);
    assert_eq!(first_result, second_result);
    assert!(first_result.is_ok());
    assert_eq!(
        first, second,
        "each invocation starts with its own counters"
    );
    assert!(first.infer_expr_visits > 0);
    assert!(first.annotation_expr_visits > 0);
    assert_eq!(first.metadata_rewrites, 1);
    assert_eq!(
        first.node_bridge_clone_nodes
            + first.list_rewrite_clone_nodes
            + first.top_level_body_clone_nodes,
        0
    );
}
