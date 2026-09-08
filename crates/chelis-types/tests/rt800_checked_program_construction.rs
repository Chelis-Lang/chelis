use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List};
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

fn find_list_mut<'a>(expr: &'a mut Expr, tag: &str) -> Option<&'a mut List> {
    match expr {
        Expr::List(list, _) => {
            let matches = matches!(
                list.elements.first(),
                Some(Expr::Atom(Atom::Tag(found), _)) if found.as_str() == tag
            );
            if matches {
                return Some(list);
            }
            for element in &mut list.elements {
                if let Some(found) = find_list_mut(element, tag) {
                    return Some(found);
                }
            }
            None
        }
        // These mutation targets are ordinary function/body nodes. Typed
        // metadata exposes no mutable Expr reference that could invalidate it.
        Expr::Map(_, _) => None,
        Expr::MetaExpr(meta, _) => find_list_mut(&mut meta.expr, tag),
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(_) => {
            panic!("rt800 list mutation helper requires the legacy checked-AST representation")
        }
        Expr::Atom(_, _) => None,
    }
}

fn replace_symbol(expr: &mut Expr, from: &str, to: &str) -> bool {
    match expr {
        Expr::Atom(Atom::Name(name), _) if name == from => {
            *name = to.to_string();
            true
        }
        Expr::List(list, _) => list
            .elements
            .iter_mut()
            .any(|element| replace_symbol(element, from, to)),
        Expr::Map(map, _) => mutate_annotation_leaves(map, |value| replace_symbol(value, from, to)),
        Expr::MetaExpr(meta, _) => {
            replace_symbol(&mut meta.expr, from, to)
                || mutate_annotation_leaves(&mut meta.metadata, |value| {
                    replace_symbol(value, from, to)
                })
        }
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(_) => {
            panic!("rt800 symbol mutation helper requires the legacy checked-AST representation")
        }
        Expr::Atom(_, _) => false,
    }
}

fn remove_first_metadata_key(expr: &mut Expr, key: chelis_deep::annotations::MetadataKey) -> bool {
    match expr {
        Expr::List(list, _) => {
            if let Some(Expr::Map(meta, _)) = list.elements.get_mut(1)
                && meta.remove(key).is_some()
            {
                return true;
            }
            list.elements
                .iter_mut()
                .any(|element| remove_first_metadata_key(element, key))
        }
        Expr::Map(map, _) => {
            mutate_annotation_leaves(map, |value| remove_first_metadata_key(value, key))
        }
        Expr::MetaExpr(meta, _) => {
            remove_first_metadata_key(&mut meta.expr, key)
                || mutate_annotation_leaves(&mut meta.metadata, |value| {
                    remove_first_metadata_key(value, key)
                })
        }
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(_) => {
            panic!("rt800 metadata mutation helper requires the legacy checked-AST representation")
        }
        Expr::Atom(_, _) => false,
    }
}

fn add_effects_metadata(exprs: &mut [Expr]) {
    let function = exprs
        .iter_mut()
        .find_map(|expr| find_list_mut(expr, "fn"))
        .expect("fixture has a function");
    let Expr::Map(meta, _) = function
        .elements
        .get_mut(1)
        .expect("canonical function metadata")
    else {
        panic!("function metadata must be a map");
    };
    meta.insert(chelis_deep::annotations::MetadataValue::Effects(
        chelis_deep::annotations::EffectSet::new(
            chelis_deep::Metadata::default(),
            vec![chelis_deep::annotations::EffectMember::Name(
                chelis_deep::annotations::Spanned::new("io".into(), Span::new(0, 0)),
            )],
            Span::new(0, 0),
        ),
    ))
    .unwrap();
}

fn mutate_body(exprs: &mut [Expr]) {
    assert!(replace_symbol(&mut exprs[0], "relu", "sigmoid"));
}

fn mutate_structure(exprs: &mut [Expr]) {
    let app = find_list_mut(&mut exprs[0], "app").expect("fixture has app");
    app.elements.pop();
}

fn mutate_type_metadata(exprs: &mut [Expr]) {
    assert!(replace_symbol(&mut exprs[0], "f32", "f64"));
}

fn mutate_eff_metadata(exprs: &mut [Expr]) {
    let function = find_list_mut(&mut exprs[0], "fn").expect("fixture has fn");
    let Expr::Map(meta, _) = &mut function.elements[1] else {
        panic!("function metadata must be a map");
    };
    meta.insert(chelis_deep::annotations::MetadataValue::Eff(
        chelis_deep::annotations::EffectSet::new(
            chelis_deep::Metadata::default(),
            vec![],
            Span::new(0, 0),
        ),
    ))
    .unwrap();
}

fn mutate_span(exprs: &mut [Expr]) {
    match &mut exprs[0] {
        Expr::List(_, span) => *span = Span::new(span.offset + 1, span.len),
        other => panic!("expected list root, got {other:?}"),
    }
}

fn remove_type_stamp(exprs: &mut [Expr]) {
    assert!(remove_first_metadata_key(
        &mut exprs[0],
        chelis_deep::annotations::MetadataKey::Type
    ));
}

fn root_metadata_mut(exprs: &mut [Expr]) -> &mut chelis_deep::Metadata {
    let Expr::List(root, _) = &mut exprs[0] else {
        panic!("expected list root");
    };
    let Expr::Map(meta, _) = &mut root.elements[1] else {
        panic!("expected canonical root metadata");
    };
    meta
}

#[test]
fn effect_only_reannotation_preserves_every_checked_context() {
    let checked = checked_program();
    let mut annotated = checked.annotated_exprs().to_vec();
    add_effects_metadata(&mut annotated);
    let expected_annotated = annotated.clone();

    let rewritten = checked
        .try_with_effect_annotations(annotated)
        .expect("effects-owned metadata is the only legal rewrite");

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
    let cases: [MutationCase; 6] = [
        ("body", mutate_body, "effects-only"),
        ("structure", mutate_structure, "effects-only"),
        ("type metadata", mutate_type_metadata, "effects-only"),
        ("eff metadata", mutate_eff_metadata, "metadata `eff`"),
        ("span", mutate_span, "effects-only"),
        ("missing type stamp", remove_type_stamp, "effects-only"),
    ];

    for (label, mutate, diagnostic) in cases {
        let mut forged = checked.annotated_exprs().to_vec();
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
    let root = root_metadata_mut(&mut reordered);
    let mut values = root.values().cloned().collect::<Vec<_>>();
    values.reverse();
    let reversed = chelis_deep::Metadata::try_from_values(values).unwrap();
    assert_eq!(*root, reversed);
    *root = reversed;
    checked.try_with_effect_annotations(reordered).unwrap();

    let mut candidate = checked.annotated_exprs().to_vec();
    let root = root_metadata_mut(&mut candidate);
    let before = root.clone();
    let duplicate = root.values().next().unwrap().clone();
    assert!(root.insert(duplicate).is_err());
    assert_eq!(*root, before);
    add_effects_metadata(&mut candidate);
    let function = find_list_mut(&mut candidate[0], "fn").unwrap();
    let Expr::Map(metadata, _) = &mut function.elements[1] else {
        panic!("metadata slot")
    };
    let duplicate = metadata.effects().unwrap().clone();
    assert!(
        metadata
            .insert(chelis_deep::annotations::MetadataValue::Effects(duplicate))
            .is_err()
    );
    checked.try_with_effect_annotations(candidate).unwrap();
}
