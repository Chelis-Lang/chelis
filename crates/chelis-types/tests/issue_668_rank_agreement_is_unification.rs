//! Elementwise tensor rank agreement is enforced by unification at both ingresses.
use chelis_deep::{Atom, DeepTag, Expr, ExprCarrier};
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_ir_program, check_typed_program};

fn template(name: &str, args: &[&str]) -> Vec<chelis_deep::Expr> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/core/tests/issue_668_rank_agreement_is_unification")
        .join(format!("{name}.deep"));
    let mut source = std::fs::read_to_string(path).expect("core fixture");
    for (index, value) in args.iter().enumerate() {
        source = source.replace(&format!("__CORE_ARG_{index}__"), value);
    }
    chelis_deep::parser::parse_str(&source).expect("valid parameterized Deep fixture")
}

fn agreed_diagnostics(name: &str, args: &[&str]) -> Vec<String> {
    let program = template(name, args);
    let kinds = |result: Result<_, chelis_types::InferResult>| {
        let mut kinds = match result {
            Ok(_) => Vec::new(),
            Err(report) => report
                .errors
                .into_iter()
                .map(|e| format!("{:?}", e.kind))
                .collect::<Vec<_>>(),
        };
        kinds.sort();
        kinds
    };
    let ir = kinds(check_ir_program(
        &program,
        chelis_types::ProgramOrigin::Unlinked,
    ));
    let typed = kinds(check_typed_program(
        &program,
        chelis_types::ProgramOrigin::Unlinked,
    ));
    assert_eq!(ir, typed, "rank diagnostics must agree: {name}, {args:?}");
    ir
}

fn check(name: &str, args: &[&str], expected: Option<CheckErrorKind>) {
    let errors = agreed_diagnostics(name, args);
    match expected {
        Some(kind) => assert!(
            errors.contains(&format!("{kind:?}")),
            "{name}, {args:?}: expected {kind:?}, got {errors:?}"
        ),
        None => assert!(errors.is_empty(), "{name}, {args:?}: {errors:?}"),
    }
}

#[test]
fn a_rank_divergent_insert_operand_is_refused_in_both_orders() {
    for (args, expected) in [
        (&["s", "e"][..], Some(CheckErrorKind::DimensionMismatch)),
        (&["e", "s"][..], Some(CheckErrorKind::DimensionMismatch)),
    ] {
        check(
            "a_rank_divergent_insert_operand_is_refused_in_both_orders_template_0",
            args,
            expected,
        );
    }
}

#[test]
fn an_inline_insert_operand_is_refused_in_both_orders() {
    check(
        "an_inline_insert_operand_is_refused_in_both_orders_template_0",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    check(
        "an_inline_insert_operand_is_refused_in_both_orders_template_1",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn an_intervening_identity_call_does_not_erase_the_rank_disagreement() {
    check(
        "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_0",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    for (args, expected) in [
        (&["ss", "add"][..], Some(CheckErrorKind::DimensionMismatch)),
        (
            &["fd", "floor_div"][..],
            Some(CheckErrorKind::DimensionMismatch),
        ),
    ] {
        check(
            "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_1",
            args,
            expected,
        );
    }
    check(
        "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_2",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    for (args, expected) in [
        (&["mul"][..], Some(CheckErrorKind::DimensionMismatch)),
        (&["floor_div"][..], Some(CheckErrorKind::DimensionMismatch)),
    ] {
        check(
            "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_3",
            args,
            expected,
        );
    }
    check(
        "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_4",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    check(
        "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_5",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    check(
        "an_intervening_identity_call_does_not_erase_the_rank_disagreement_template_6",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn a_discarded_comparison_still_unifies_its_operand_ranks() {
    check(
        "a_discarded_comparison_still_unifies_its_operand_ranks_template_0",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn a_declared_rank_two_user_def_result_is_refused_in_both_orders() {
    check(
        "a_declared_rank_two_user_def_result_is_refused_in_both_orders_template_0",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    check(
        "a_declared_rank_two_user_def_result_is_refused_in_both_orders_template_1",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn a_rank_zero_reduction_result_is_refused_beside_its_rank_one_input() {
    check(
        "a_rank_zero_reduction_result_is_refused_beside_its_rank_one_input_template_0",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
    check(
        "a_rank_zero_reduction_result_is_refused_beside_its_rank_one_input_template_1",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn a_scalar_beside_a_tensor_is_refused_by_add_and_max_elem() {
    for (args, expected) in [
        (&["add"][..], Some(CheckErrorKind::TypeMismatch)),
        (&["max_elem"][..], Some(CheckErrorKind::TypeMismatch)),
    ] {
        check(
            "a_scalar_beside_a_tensor_is_refused_by_add_and_max_elem_template_0",
            args,
            expected,
        );
    }
    for (args, expected) in [
        (&["add"][..], Some(CheckErrorKind::TypeMismatch)),
        (&["max_elem"][..], Some(CheckErrorKind::TypeMismatch)),
    ] {
        check(
            "a_scalar_beside_a_tensor_is_refused_by_add_and_max_elem_template_1",
            args,
            expected,
        );
    }
}

#[test]
fn a_rank_two_where_branch_is_reported_once_by_wheres_own_rule() {
    assert_eq!(
        agreed_diagnostics(
            "a_rank_two_where_branch_is_reported_once_by_wheres_own_rule_template_0",
            &[]
        ),
        vec!["TypeMismatch".to_string()]
    );
}

#[test]
fn a_rebinding_to_a_divergent_rank_is_still_refused() {
    check(
        "a_rebinding_to_a_divergent_rank_is_still_refused_template_0",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn matching_ranks_are_accepted() {
    check("matching_ranks_are_accepted_template_0", &[], None);
    check("matching_ranks_are_accepted_template_1", &[], None);
    check("matching_ranks_are_accepted_template_2", &[], None);
}

#[test]
fn the_expand_built_reproducer_is_accepted_in_both_orders() {
    for (args, expected) in [(&["s", "e"][..], None), (&["e", "s"][..], None)] {
        check(
            "the_expand_built_reproducer_is_accepted_in_both_orders_template_0",
            args,
            expected,
        );
    }
    check(
        "the_expand_built_reproducer_is_accepted_in_both_orders_template_1",
        &[],
        None,
    );
    check(
        "the_expand_built_reproducer_is_accepted_in_both_orders_template_2",
        &[],
        None,
    );
}

#[test]
fn a_lexically_shadowed_builtin_name_keeps_its_parameter_type() {
    check(
        "a_lexically_shadowed_builtin_name_keeps_its_parameter_type_template_0",
        &[],
        None,
    );
}

#[test]
fn a_rebinding_does_not_inherit_the_previous_bindings_shape() {
    check(
        "a_rebinding_does_not_inherit_the_previous_bindings_shape_template_0",
        &[],
        None,
    );
}

#[test]
fn an_authored_rank_only_prefix_dimension_is_an_ordinary_name() {
    check(
        "an_authored_rank_only_prefix_dimension_is_an_ordinary_name_template_0",
        &[],
        None,
    );
}

fn tensor_stamps(expr: &Expr, out: &mut Vec<(usize, bool)>) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, children) => {
            if let Some(ty) = metadata.ty()
                && let ExprCarrier::DecodedNode(DeepTag::TTensor, _, axes) =
                    ty.expression().carrier()
            {
                let extent_two = axes.first().is_some_and(|axis| {
                    matches!(
                        axis.carrier(),
                        ExprCarrier::DecodedNode(DeepTag::DLit, _, [Expr::Atom(Atom::Int(2), _)])
                    )
                });
                let f32_precision = axes.last().is_some_and(|precision| matches!(precision.carrier(),
                    ExprCarrier::DecodedNode(DeepTag::TPrim, _, [Expr::Atom(Atom::Name(name), _)]) if name == "f32"));
                out.push((axes.len() - 1, extent_two && f32_precision));
            }
            for child in children {
                tensor_stamps(child, out);
            }
        }
        ExprCarrier::StructuralList(children) => {
            for child in children {
                tensor_stamps(child, out);
            }
        }
        ExprCarrier::MalformedLegacyList(list) => {
            for child in &list.elements {
                tensor_stamps(child, out);
            }
        }
        _ => {}
    }
}

#[test]
fn the_expand_reproducer_stamps_rank_one_at_both_ingresses() {
    let exprs = template("expand_rank_stamp", &[]);
    for checked in [
        check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
        check_typed_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
    ] {
        let checked = checked.expect("rank-preserving expand must check");
        let mut stamps = Vec::new();
        for expr in checked.annotated_exprs() {
            tensor_stamps(expr, &mut stamps);
        }
        assert!(
            stamps.contains(&(1, true)),
            "expand must stamp tensor[2, f32]: {stamps:?}"
        );
        assert!(
            stamps.iter().all(|(rank, _)| *rank != 2),
            "expand must never add an axis: {stamps:?}"
        );
    }
}
