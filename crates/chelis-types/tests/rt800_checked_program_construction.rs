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
        Expr::Map(map, _) => {
            for (_, value) in &mut map.entries {
                if let Some(found) = find_list_mut(value, tag) {
                    return Some(found);
                }
            }
            None
        }
        Expr::MetaExpr(meta, _) => find_list_mut(&mut meta.expr, tag),
        Expr::Atom(_, _) => None,
    }
}

fn replace_symbol(expr: &mut Expr, from: &str, to: &str) -> bool {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) if name == from => {
            *name = to.to_string();
            true
        }
        Expr::List(list, _) => list
            .elements
            .iter_mut()
            .any(|element| replace_symbol(element, from, to)),
        Expr::Map(map, _) => map
            .entries
            .iter_mut()
            .any(|(_, value)| replace_symbol(value, from, to)),
        Expr::MetaExpr(meta, _) => {
            replace_symbol(&mut meta.expr, from, to)
                || meta
                    .entries
                    .iter_mut()
                    .any(|(_, value)| replace_symbol(value, from, to))
        }
        Expr::Atom(_, _) => false,
    }
}

fn remove_first_metadata_key(expr: &mut Expr, key: &str) -> bool {
    match expr {
        Expr::List(list, _) => {
            if let Some(Expr::Map(meta, _)) = list.elements.get_mut(1)
                && let Some(index) = meta.entries.iter().position(|(name, _)| name == key)
            {
                meta.entries.remove(index);
                return true;
            }
            list.elements
                .iter_mut()
                .any(|element| remove_first_metadata_key(element, key))
        }
        Expr::Map(map, _) => map
            .entries
            .iter_mut()
            .any(|(_, value)| remove_first_metadata_key(value, key)),
        Expr::MetaExpr(meta, _) => {
            remove_first_metadata_key(&mut meta.expr, key)
                || meta
                    .entries
                    .iter_mut()
                    .any(|(_, value)| remove_first_metadata_key(value, key))
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
    let effect_row = chelis_deep::parser::parse_str("(effects {} io)")
        .expect("effect row parses")
        .remove(0);
    meta.entries.push(("effects".to_string(), effect_row));
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
    meta.entries.push((
        "eff".to_string(),
        Expr::Atom(Atom::Symbol("forged".to_string()), Span::new(0, 0)),
    ));
}

fn mutate_span(exprs: &mut [Expr]) {
    match &mut exprs[0] {
        Expr::List(_, span) => *span = Span::new(span.offset + 1, span.len),
        other => panic!("expected list root, got {other:?}"),
    }
}

fn remove_type_stamp(exprs: &mut [Expr]) {
    assert!(remove_first_metadata_key(&mut exprs[0], "type"));
}

fn root_metadata_mut(exprs: &mut [Expr]) -> &mut Vec<(String, Expr)> {
    let Expr::List(root, _) = &mut exprs[0] else {
        panic!("expected list root");
    };
    let Expr::Map(meta, _) = &mut root.elements[1] else {
        panic!("expected canonical root metadata");
    };
    &mut meta.entries
}

fn reorder_non_effect_metadata(exprs: &mut [Expr]) {
    let entries = root_metadata_mut(exprs);
    assert!(
        entries.len() >= 2,
        "fixture must retain span and type metadata"
    );
    entries.swap(0, 1);
}

fn duplicate_non_effect_metadata(exprs: &mut [Expr]) {
    let entries = root_metadata_mut(exprs);
    let duplicate = entries
        .iter()
        .find(|(key, _)| key != "effects")
        .expect("fixture has non-effects metadata")
        .clone();
    entries.push(duplicate);
}

fn duplicate_effects_metadata(exprs: &mut [Expr]) {
    add_effects_metadata(exprs);
    add_effects_metadata(exprs);
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
    type MutationCase = (&'static str, fn(&mut [Expr]));

    let checked = checked_program();
    let cases: [MutationCase; 6] = [
        ("body", mutate_body),
        ("structure", mutate_structure),
        ("type metadata", mutate_type_metadata),
        ("eff metadata", mutate_eff_metadata),
        ("span", mutate_span),
        ("missing type stamp", remove_type_stamp),
    ];

    for (label, mutate) in cases {
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
            result.errors[0].message.contains("effects-only"),
            "{label} mutation must name the narrow ownership boundary: {:?}",
            result.errors
        );
    }
}

#[test]
fn effects_only_boundary_rejects_order_duplicates_and_multiple_effect_rows() {
    type MutationCase = (&'static str, fn(&mut [Expr]));

    let checked = checked_program();
    let cases: [MutationCase; 3] = [
        (
            "reordered non-effects metadata",
            reorder_non_effect_metadata,
        ),
        (
            "duplicated non-effects metadata",
            duplicate_non_effect_metadata,
        ),
        ("duplicate effects rows", duplicate_effects_metadata),
    ];

    for (label, mutate) in cases {
        let mut forged = checked.annotated_exprs().to_vec();
        mutate(&mut forged);
        let result = checked
            .try_with_effect_annotations(forged)
            .expect_err("the narrow effects-only boundary must reject the mutation");
        assert_eq!(
            result.errors.len(),
            1,
            "{label} must report exactly once: {:?}",
            result.errors
        );
        assert!(
            result.errors[0].message.contains("effects-only"),
            "{label} must name the narrow ownership boundary: {:?}",
            result.errors
        );
    }
}
