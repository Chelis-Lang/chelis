use chelis_deep::ast::{Atom, Expr};
use chelis_deep::{DeepTag, Span};
use chelis_types::{CheckedProgram, check_ir_program, check_linearity};

fn checked_program() -> CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(
        r#"(def {span: "rt800:root"} apply_relu
              (fn {} (params {} (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
                (app {} (var {} relu) (var {} x))))"#,
    )
    .expect("fixture parses");
    let typed = check_ir_program(&exprs).expect("fixture type-checks");
    check_linearity(&typed).expect("fixture passes linearity")
}

fn mutate_annotation_leaves(
    metadata: &mut chelis_deep::Metadata,
    mut f: impl FnMut(&mut Expr) -> bool,
) -> bool {
    let mut changed = false;
    *metadata = metadata
        .map_expressions(&mut |value, _| {
            let mut value = value.clone();
            if !changed {
                changed = f(&mut value);
            }
            value
        })
        .expect("mutation preserves the payload shape");
    changed
}

fn mutate_first_decoded_node(
    expr: &mut Expr,
    target: DeepTag,
    mutator: &mut impl FnMut(&mut chelis_deep::Metadata, &mut Vec<Expr>) -> bool,
) -> bool {
    match expr {
        Expr::Node(node, span) => {
            let tag = node.tag();
            let mut metadata = node.meta().clone();
            let mut children = node.children_slice().to_vec();
            let span = *span;
            let changed = (tag == target && mutator(&mut metadata, &mut children))
                || mutate_annotation_leaves(&mut metadata, |value| {
                    mutate_first_decoded_node(value, target, mutator)
                })
                || children
                    .iter_mut()
                    .any(|child| mutate_first_decoded_node(child, target, mutator));
            if changed {
                *expr = Expr::node(tag, metadata, children, span);
            }
            changed
        }
        Expr::BareList(elements, _) => elements
            .iter_mut()
            .any(|element| mutate_first_decoded_node(element, target, mutator)),
        Expr::UnknownForm(data) => {
            mutate_annotation_leaves(&mut data.meta, |value| {
                mutate_first_decoded_node(value, target, mutator)
            }) || data
                .children
                .iter_mut()
                .any(|child| mutate_first_decoded_node(child, target, mutator))
        }
        Expr::Map(metadata, _) => mutate_annotation_leaves(metadata, |value| {
            mutate_first_decoded_node(value, target, mutator)
        }),
        Expr::MetaExpr(metadata_expr, _) => {
            mutate_first_decoded_node(&mut metadata_expr.expr, target, mutator)
                || mutate_annotation_leaves(&mut metadata_expr.metadata, |value| {
                    mutate_first_decoded_node(value, target, mutator)
                })
        }
        Expr::Atom(_, _) => false,
    }
}

fn replace_symbol(expr: &mut Expr, from: &str, to: &str) -> bool {
    match expr {
        Expr::Atom(Atom::Name(name), _) if name == from => {
            *name = to.to_string();
            true
        }
        Expr::Map(map, _) => mutate_annotation_leaves(map, |value| replace_symbol(value, from, to)),
        Expr::MetaExpr(meta, _) => {
            replace_symbol(&mut meta.expr, from, to)
                || mutate_annotation_leaves(&mut meta.metadata, |value| {
                    replace_symbol(value, from, to)
                })
        }
        Expr::Node(node, span) => {
            let tag = node.tag();
            let mut metadata = node.meta().clone();
            let mut children = node.children_slice().to_vec();
            let span = *span;
            let changed =
                mutate_annotation_leaves(&mut metadata, |value| replace_symbol(value, from, to))
                    || children
                        .iter_mut()
                        .any(|child| replace_symbol(child, from, to));
            if changed {
                *expr = Expr::node(tag, metadata, children, span);
            }
            changed
        }
        Expr::BareList(elements, _) => elements
            .iter_mut()
            .any(|element| replace_symbol(element, from, to)),
        Expr::UnknownForm(data) => {
            mutate_annotation_leaves(&mut data.meta, |value| replace_symbol(value, from, to))
                || data
                    .children
                    .iter_mut()
                    .any(|child| replace_symbol(child, from, to))
        }
        Expr::Atom(_, _) => false,
    }
}

fn remove_first_metadata_key(expr: &mut Expr, key: chelis_deep::annotations::MetadataKey) -> bool {
    match expr {
        Expr::Map(map, _) => {
            mutate_annotation_leaves(map, |value| remove_first_metadata_key(value, key))
        }
        Expr::MetaExpr(meta, _) => {
            remove_first_metadata_key(&mut meta.expr, key)
                || mutate_annotation_leaves(&mut meta.metadata, |value| {
                    remove_first_metadata_key(value, key)
                })
        }
        Expr::Node(node, span) => {
            let tag = node.tag();
            let mut metadata = node.meta().clone();
            let mut children = node.children_slice().to_vec();
            let span = *span;
            let changed = metadata.remove(key).is_some()
                || mutate_annotation_leaves(&mut metadata, |value| {
                    remove_first_metadata_key(value, key)
                })
                || children
                    .iter_mut()
                    .any(|child| remove_first_metadata_key(child, key));
            if changed {
                *expr = Expr::node(tag, metadata, children, span);
            }
            changed
        }
        Expr::BareList(elements, _) => elements
            .iter_mut()
            .any(|element| remove_first_metadata_key(element, key)),
        Expr::UnknownForm(data) => {
            data.meta.remove(key).is_some()
                || mutate_annotation_leaves(&mut data.meta, |value| {
                    remove_first_metadata_key(value, key)
                })
                || data
                    .children
                    .iter_mut()
                    .any(|child| remove_first_metadata_key(child, key))
        }
        Expr::Atom(_, _) => false,
    }
}

fn add_effects_metadata(exprs: &mut [Expr]) {
    assert!(mutate_first_decoded_node(
        &mut exprs[0],
        DeepTag::Fn,
        &mut |metadata, _| {
            metadata
                .insert(chelis_deep::annotations::MetadataValue::Effects(
                    chelis_deep::annotations::EffectSet::new(
                        chelis_deep::Metadata::default(),
                        vec![chelis_deep::annotations::EffectMember::Name(
                            chelis_deep::annotations::Spanned::new("io".into(), Span::new(0, 0)),
                        )],
                        Span::new(0, 0),
                    ),
                ))
                .unwrap();
            true
        }
    ));
}

fn mutate_body(exprs: &mut [Expr]) {
    assert!(replace_symbol(&mut exprs[0], "relu", "sigmoid"));
}

fn mutate_structure(exprs: &mut [Expr]) {
    assert!(mutate_first_decoded_node(
        &mut exprs[0],
        DeepTag::App,
        &mut |_, children| {
            children.pop();
            true
        }
    ));
}

fn mutate_type_metadata(exprs: &mut [Expr]) {
    assert!(replace_symbol(&mut exprs[0], "f32", "f64"));
}

/// `eff` belongs on a `t-fn` type, so a `fn` node carrying it has no admitted
/// spelling: return the `Node` constructor's rejection of that candidate.
fn eff_metadata_rejection(exprs: &mut [Expr]) -> String {
    let mut rejection = None;
    mutate_first_decoded_node(&mut exprs[0], DeepTag::Fn, &mut |metadata, children| {
        let mut metadata = metadata.clone();
        metadata
            .insert(chelis_deep::annotations::MetadataValue::Eff(
                chelis_deep::annotations::EffectSet::new(
                    chelis_deep::Metadata::default(),
                    vec![],
                    Span::new(0, 0),
                ),
            ))
            .unwrap();
        let error = chelis_deep::node::Node::try_new(DeepTag::Fn, metadata, children.clone())
            .expect_err("a fn node cannot carry eff metadata");
        rejection = Some(error.to_string());
        false
    });
    rejection.expect("fixture contains a fn node")
}

fn mutate_span(exprs: &mut [Expr]) {
    match &mut exprs[0] {
        Expr::Node(_, span) => *span = Span::new(span.offset + 1, span.len),
        other => panic!("expected node root, got {other:?}"),
    }
}

fn remove_type_stamp(exprs: &mut [Expr]) {
    assert!(remove_first_metadata_key(
        &mut exprs[0],
        chelis_deep::annotations::MetadataKey::Type
    ));
}

fn root_metadata(exprs: &[Expr]) -> chelis_deep::Metadata {
    let Expr::Node(root, _) = &exprs[0] else {
        panic!("expected node root");
    };
    root.meta().clone()
}

fn replace_root_metadata(exprs: &mut [Expr], metadata: chelis_deep::Metadata) {
    let Expr::Node(root, _) = &mut exprs[0] else {
        panic!("expected node root");
    };
    root.try_replace_meta(metadata)
        .expect("root metadata replacement preserves the Node invariant");
}

#[test]
fn effect_only_reannotation_preserves_every_checked_context() {
    let checked = checked_program();
    let mut annotated = checked.annotated_exprs().to_vec();
    add_effects_metadata(&mut annotated);
    let expected_annotated = annotated.clone();

    let rewritten = checked
        .try_with_effect_annotations(annotated)
        .unwrap_or_else(|errors| panic!("effects rewrite failed: {errors:?}"));

    assert_eq!(rewritten.annotated_exprs(), expected_annotated);
    assert_eq!(rewritten.type_env(), checked.type_env());
    assert_eq!(
        rewritten.signature_inference(),
        checked.signature_inference()
    );
    assert_eq!(rewritten.linearity(), checked.linearity());
}

#[test]
fn non_effect_reannotation_mutations_are_rejected_exactly_once() {
    type MutationCase = (&'static str, fn(&mut [Expr]), &'static str);

    let checked = checked_program();
    let cases: [MutationCase; 5] = [
        ("body", mutate_body, "effects-only"),
        ("structure", mutate_structure, "effects-only"),
        ("type metadata", mutate_type_metadata, "effects-only"),
        ("span", mutate_span, "effects-only"),
        ("missing type stamp", remove_type_stamp, "effects-only"),
    ];

    let candidate = checked.annotated_exprs().to_vec();
    let eff = eff_metadata_rejection(&mut candidate.clone());
    assert!(
        eff.contains("metadata `eff`"),
        "eff metadata mutation must name its metadata boundary: {eff}"
    );
    for (label, mutate, diagnostic) in cases {
        let mut forged = candidate.clone();
        mutate(&mut forged);
        let result = checked
            .try_with_effect_annotations(forged)
            .expect_err("non-effects mutation must not reconstruct success");
        assert_eq!(
            result.errors.len(),
            1,
            "{label} mutation must report exactly once: {:?}",
            result.errors
        );
        assert!(
            result.errors[0].message.contains(diagnostic),
            "{label} mutation must name its metadata or ownership boundary: {:?}",
            result.errors
        );
    }
}

#[test]
fn typed_metadata_canonicalizes_order_and_rejects_duplicate_effect_rows() {
    let checked = checked_program();
    let mut reordered = checked.annotated_exprs().to_vec();
    let root = root_metadata(&reordered);
    let mut values = root.values().cloned().collect::<Vec<_>>();
    values.reverse();
    let reversed = chelis_deep::Metadata::try_from_values(values).unwrap();
    assert_eq!(root, reversed);
    replace_root_metadata(&mut reordered, reversed);
    checked.try_with_effect_annotations(reordered).unwrap();

    let mut candidate = checked.annotated_exprs().to_vec();
    let mut root = root_metadata(&candidate);
    let before = root.clone();
    let duplicate = root.values().next().unwrap().clone();
    assert!(root.insert(duplicate).is_err());
    assert_eq!(root, before);
    replace_root_metadata(&mut candidate, root);
    add_effects_metadata(&mut candidate);
    assert!(mutate_first_decoded_node(
        &mut candidate[0],
        DeepTag::Fn,
        &mut |metadata, _| {
            let duplicate = metadata.effects().unwrap().clone();
            assert!(
                metadata
                    .insert(chelis_deep::annotations::MetadataValue::Effects(duplicate))
                    .is_err()
            );
            true
        }
    ));
    checked.try_with_effect_annotations(candidate).unwrap();
}

#[test]
fn mutation_helpers_preserve_structural_and_unknown_carriers() {
    let span = Span::new(0, 0);
    let mut structural = Expr::BareList(vec![Expr::Atom(Atom::Name("relu".into()), span)], span);
    assert!(replace_symbol(&mut structural, "relu", "sigmoid"));
    assert!(matches!(
        structural,
        Expr::BareList(ref elements, _)
            if matches!(
                elements.as_slice(),
                [Expr::Atom(Atom::Name(name), _)] if name == "sigmoid"
            )
    ));

    let mut unknown = Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
        head: "future-form".into(),
        meta: chelis_deep::Metadata::default(),
        children: vec![Expr::Atom(Atom::Name("relu".into()), span)],
        span,
    }));
    assert!(replace_symbol(&mut unknown, "relu", "sigmoid"));
    assert!(matches!(
        unknown,
        Expr::UnknownForm(ref data)
            if matches!(
                data.children.as_slice(),
                [Expr::Atom(Atom::Name(name), _)] if name == "sigmoid"
            )
    ));
}
