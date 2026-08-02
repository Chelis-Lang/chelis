//! The public checker boundary must consume and preserve stamped Deep.
//!
//! Authority: `spec/design/checker_totality.md` section C4.2 successor
//! acceptance and `spec/design/unrepresentable_ast_domain.md` Tasks 6-8.

use chelis_deep::{Atom, DeepTag, Expr, MetaMap, Span, parse_and_stamp};
use chelis_types::{check_linearity, check_typed_program, errors::CheckErrorKind};

fn assert_no_legacy_list(expr: &Expr) {
    match expr {
        Expr::List(_, _) => panic!("checker output normalized stamped Deep back to Expr::List"),
        Expr::Node(node, _) => {
            for (_, value) in &node.meta().entries {
                assert_no_legacy_list(value);
            }
            for child in node.children_slice() {
                assert_no_legacy_list(child);
            }
        }
        Expr::BareList(elements, _) => {
            for child in elements {
                assert_no_legacy_list(child);
            }
        }
        Expr::UnknownForm(data) => {
            for (_, value) in &data.meta.entries {
                assert_no_legacy_list(value);
            }
            for child in &data.children {
                assert_no_legacy_list(child);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_no_legacy_list(value);
            }
        }
        Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                assert_no_legacy_list(value);
            }
            assert_no_legacy_list(&meta.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn find_node_mut(expr: &mut Expr, tag: DeepTag) -> Option<&mut chelis_deep::node::Node> {
    match expr {
        Expr::Node(node, _) => {
            if node.tag() == tag {
                return Some(node);
            }
            for child in node.children_slice_mut() {
                if let Some(found) = find_node_mut(child, tag) {
                    return Some(found);
                }
            }
            None
        }
        Expr::BareList(elements, _) => elements
            .iter_mut()
            .find_map(|child| find_node_mut(child, tag)),
        Expr::UnknownForm(data) => data
            .children
            .iter_mut()
            .find_map(|child| find_node_mut(child, tag)),
        Expr::Map(map, _) => map
            .entries
            .iter_mut()
            .find_map(|(_, value)| find_node_mut(value, tag)),
        Expr::MetaExpr(meta, _) => find_node_mut(&mut meta.expr, tag),
        Expr::Atom(_, _) | Expr::List(_, _) => None,
    }
}

#[test]
fn accepted_stamped_program_stays_stamped_through_checker_output() {
    let program =
        parse_and_stamp("(def {} answer (lit {} 1))").expect("canonical Deep program must stamp");

    let checked = check_typed_program(&program).expect("stamped program must type-check");

    assert!(
        matches!(checked.annotated_exprs().first(), Some(Expr::Node(..))),
        "top-level stamped declaration must remain a Node"
    );
    for expr in checked.annotated_exprs() {
        assert_no_legacy_list(expr);
    }
}

#[test]
fn stamped_comparison_keeps_its_boolean_result_contract() {
    let program = parse_and_stamp(
        r#"(defsig {} always_true (t-fn {} (t-prim {} f32) (t-prim {} bool)))
           (def {} always_true
             (fn {} (params {} (x {type: (t-prim {} f32)}))
               (app {} (var {} gte) (var {} x) (var {} x))))"#,
    )
    .expect("canonical comparison program must stamp");

    let checked = check_typed_program(&program)
        .expect("stamped application dispatch must preserve comparison semantics");
    assert!(
        checked
            .signature_inference()
            .functions
            .contains_key("always_true"),
        "signature inference must consume stamped Def/Fn declarations"
    );
}

#[test]
fn stamped_let_bindings_enter_scope_and_unbound_names_still_reject() {
    let accepted = parse_and_stamp("(def {} answer (let {} (bind {} x (lit {} 1)) (var {} x)))")
        .expect("canonical let program must stamp");
    check_typed_program(&accepted).expect("stamped bind child must introduce x into let scope");

    let rejected = parse_and_stamp("(def {} answer (let {} (bind {}) (var {} missing)))")
        .expect("canonical negative let program must stamp");
    let errors = check_typed_program(&rejected).expect_err("unbound let body name must reject");
    assert!(
        errors
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UnboundVariable)),
        "negative stamped let must retain the unbound-variable disposition: {:?}",
        errors.errors
    );
}

#[test]
fn stamped_linearity_preserves_positive_and_negative_tensor_ownership() {
    let accepted = parse_and_stamp(
        r#"(defsig {} identity
             (t-fn {}
               (t-tensor {} (d-lit {} 4) (t-prim {} f32))
               (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
           (def {} identity
             (fn {}
               (params {} (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
               (var {} x)))"#,
    )
    .expect("positive linearity fixture must stamp");
    let accepted = check_typed_program(&accepted).expect("positive fixture type-checks");
    check_linearity(&accepted).expect("single tensor consume must pass linearity");

    let rejected = parse_and_stamp(
        r#"(defsig {} alias_twice
             (t-fn {}
               (t-tensor {} (d-lit {} 4) (t-prim {} f32))
               (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
           (def {} alias_twice
             (fn {}
               (params {} (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
               (let {}
                 (bind {} y (realize {} (var {} x)))
                 (app {} (var {} add) (var {} x) (var {} y)))))"#,
    )
    .expect("negative linearity fixture must stamp");
    let rejected = check_typed_program(&rejected).expect("negative fixture type-checks first");
    check_linearity(&rejected).expect_err("use after tensor consumption must reject");
}

#[test]
fn effects_only_reannotation_accepts_stamped_nodes_and_rejects_other_metadata() {
    let program = parse_and_stamp("(def {} answer (fn {} (params {}) (lit {} 1)))")
        .expect("canonical Deep program must stamp");
    let checked = check_typed_program(&program).expect("stamped program must type-check");

    let mut effects_only = checked.annotated_exprs().to_vec();
    let function = effects_only
        .iter_mut()
        .find_map(|expr| find_node_mut(expr, DeepTag::Fn))
        .expect("fixture contains a stamped function");
    let effect_row = Expr::node(
        DeepTag::Effects,
        MetaMap::default(),
        vec![Expr::Atom(Atom::Name("io".to_string()), Span::new(0, 0))],
        Span::new(0, 0),
    );
    function
        .meta_mut()
        .entries
        .push(("effects".to_string(), effect_row));
    checked
        .try_with_effect_annotations(effects_only)
        .expect("effects-owned metadata is the only legal stamped rewrite");

    let mut forged = checked.annotated_exprs().to_vec();
    let root = forged
        .first_mut()
        .and_then(|expr| find_node_mut(expr, DeepTag::Def))
        .expect("fixture contains a stamped declaration");
    root.meta_mut().entries.push((
        "forged".to_string(),
        Expr::Atom(Atom::Bool(true), Span::new(0, 0)),
    ));
    checked
        .try_with_effect_annotations(forged)
        .expect_err("non-effects stamped metadata must not reconstruct success");
}

#[test]
fn unknown_stamped_form_is_rejected_at_checker_boundary() {
    let program = parse_and_stamp("(def {} answer (apl {} (lit {} 1)))")
        .expect("lenient unknown runtime form must stamp as UnknownForm");

    let errors = check_typed_program(&program).expect_err("unknown form must not type-check");

    assert!(
        errors
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UnknownForm)),
        "checker must preserve the loud UnknownForm disposition: {:?}",
        errors.errors
    );
}
