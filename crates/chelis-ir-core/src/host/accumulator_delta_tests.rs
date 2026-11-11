#![cfg(test)]

use super::*;

fn find_fn(expr: &Expr) -> Option<&Expr> {
    if let Expr::MetaExpr(meta, _) = expr {
        return find_fn(&meta.expr);
    }
    let node = as_node(expr)?;
    if node.tag() == DeepTag::Fn {
        return Some(expr);
    }
    node.children_slice().iter().find_map(find_fn)
}

fn proof(body: &str) -> Option<AccumulatorDeltaProof> {
    let source = format!(
        "out = fold(fn (acc: List[tensor[3, f32]], i: i64) -> {body}, [to_tensor([1.0f32, 2.0f32, 3.0f32])], range(0i64, 2i64))\n"
    );
    let declarations =
        chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    crate::test_support::with_operation(|operation| {
        accumulator_delta_proof(
            checked.facts().exprs().iter().find_map(find_fn).unwrap(),
            checked.facts().adt_registry(),
            operation,
        )
        .unwrap()
    })
}

#[test]
fn nested_alias_identity_analysis_has_bounded_source_and_lowered_work() {
    for length in [40, 80, 160] {
        let mut bindings = "old0 = acc\n".to_owned();
        for index in 1..length {
            bindings.push_str(&format!("old{index} = old{}\n", index - 1));
        }
        let body = format!(
            "{{ {bindings} append(old{}, to_tensor([1.0f32, 2.0f32, 3.0f32])) }}",
            length - 1
        );
        ACCUMULATOR_IDENTITY_WORK.with(|work| work.set(0));
        let proof = proof(&body).unwrap();
        let source_work = ACCUMULATOR_IDENTITY_WORK.with(core::cell::Cell::get);
        let ty = proof.formal_type.clone();
        let var = |name: String| HostExpr::new(HostExprKind::Var(name, ty.clone()));
        let mut lowered = HostExpr::new(HostExprKind::Builtin {
            name: "append".into(),
            args: vec![
                var(format!("old{}", length - 1)),
                HostExpr::new(HostExprKind::Unit),
            ],
            ty: ty.clone(),
        });
        for index in (0..length).rev() {
            lowered = HostExpr::new(HostExprKind::Let {
                bindings: vec![HostBinding {
                    name: format!("old{index}"),
                    display_name: None,
                    display_roots: Vec::new(),
                    ty: ty.clone(),
                    value: var(if index == 0 {
                        "acc".to_owned()
                    } else {
                        format!("old{}", index - 1)
                    }),
                }],
                body: Box::new(lowered),
                ty: ty.clone(),
            });
        }
        ACCUMULATOR_IDENTITY_WORK.with(|work| work.set(0));
        assert!(proof.validates_body(&lowered));
        let lowered_work = ACCUMULATOR_IDENTITY_WORK.with(core::cell::Cell::get);
        std::eprintln!("lexical identity L={length}: source={source_work} lowered={lowered_work}");
        assert!(source_work <= 16 * length + 100, "source={source_work}");
        assert!(lowered_work <= 16 * length + 100, "lowered={lowered_work}");
        let HostExprKind::Let { bindings, .. } = &mut lowered.kind else {
            unreachable!()
        };
        bindings[0].value = var("equal_shape_unproved".to_owned());
        assert!(!proof.validates_body(&lowered));
    }
}

#[test]
fn lexical_identity_undo_preserves_sibling_constructor_components() {
    let source = "type Node[T] =\n | End { v: T }\n | Link { values: List[T], next: Node[T] }\ndef step(acc: Node[tensor[3, f32]], i: i64) -> Node[tensor[*, f32]] = { old = acc\n Link { values: { old = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n old }, next: old } }\nout = 3i64\n";
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    crate::test_support::with_operation(|operation| {
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, callee) = session.def_named("step").unwrap();
        let proof = accumulator_construction_proof(
            callee,
            checked.facts().adt_registry(),
            session.operation,
        )
        .unwrap()
        .unwrap();
        let host = try_lower_compiled_program(checked.facts(), session.operation)
            .unwrap()
            .host
            .unwrap();
        let body = &host
            .functions
            .iter()
            .find(|function| function.name == "step")
            .unwrap()
            .body;
        assert!(
            proof.validates_body(body),
            "the fresh field's shadow must not escape to its sibling"
        );
        fn change(expr: &mut ConcreteHostExpr) -> bool {
            match &mut expr.kind {
                HostExprKind::Let { body, .. }
                | HostExprKind::ResultClaimScope { body, .. }
                | HostExprKind::ExtentSites { value: body, .. } => change(body),
                HostExprKind::AdtConstruct { ctor, fields, ty } if ctor == "Link" => {
                    fields[1] = ConcreteHostExpr::new(HostExprKind::Var(
                        "equal_shape_unproved".into(),
                        ty.clone(),
                    ));
                    true
                }
                _ => false,
            }
        }
        let mut changed = body.clone();
        assert!(change(&mut changed));
        assert!(!proof.validates_body(&changed));
    });
}

#[test]
fn delta_proof_rejects_a_transplanted_formal_obligation() {
    let proof = proof("append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32]))").unwrap();
    assert!(proof.validates_formal_type(&proof.formal_type));
    let mut changed = proof.formal_type.clone();
    let HostTypeTerm::List(element) = &mut changed else {
        panic!("List proof");
    };
    let HostTypeTerm::Tensor(tensor) = element.as_mut() else {
        panic!("tensor element");
    };
    tensor.dims[0] = DimInfo::Lit(5);
    assert!(!proof.validates_formal_type(&changed));
    assert!(!proof.validates_formal_type(&HostTypeTerm::Unit));
    assert!(proof.validates_pattern(proof.pattern()));
    let changed = chelis_deep::parser::parse_str(
        "(t-adt {} List (t-tensor {} (d-lit {} 5) (t-prim {} f32)))",
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let changed = crate::claim_pattern::ClaimPattern::derive(
        &changed[0],
        &chelis_checker_core::adt::AdtRegistry::default(),
    )
    .unwrap();
    assert!(!proof.validates_pattern(&changed));
}

#[test]
fn delta_proof_transports_only_sequential_immutable_root_aliases() {
    for body in [
        "{ old = acc\n append(old, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
        "{ old = acc\n older = old\n concat(older, [to_tensor([1.0f32, 2.0f32, 3.0f32])]) }",
        "{ old = acc\n older = old\n concat([to_tensor([1.0f32, 2.0f32, 3.0f32])], older) }",
    ] {
        assert!(
            proof(body).is_some(),
            "immutable original-root alias: {body}"
        );
    }
}

#[test]
fn delta_proof_rejects_rebound_pattern_shadowed_and_equal_value_aliases() {
    for body in [
        "{ old = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n append(old, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
        "{ old = acc\n old = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n append(old, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
        "{ old = if i == 0i64 then acc else acc\n append(old, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
        "{ old = acc\n (old, ignored) = ([to_tensor([1.0f32, 2.0f32, 3.0f32])], i)\n append(old, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
        "{ old = acc\n append = fn (xs: List[tensor[3, f32]], x: tensor[3, f32]) -> xs\n append(old, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
    ] {
        assert!(proof(body).is_none(), "no original-root authority: {body}");
    }
}

#[test]
fn delta_proof_checks_lowered_alias_identity_independently() {
    let proof =
        proof("{ old = acc\n older = old\n append(older, to_tensor([1.0f32, 2.0f32, 3.0f32])) }")
            .unwrap();
    let variable =
        |name: &str| HostExpr::new(HostExprKind::Var(name.into(), proof.formal_type.clone()));
    let binding = |name: &str, value| HostBinding {
        name: name.into(),
        display_name: None,
        display_roots: Vec::new(),
        ty: proof.formal_type.clone(),
        value,
    };
    let lowered = |source: &str, operation: &str| {
        HostExpr::new(HostExprKind::Let {
            bindings: vec![
                binding("old", variable(source)),
                binding("older", variable("old")),
            ],
            body: Box::new(HostExpr::new(HostExprKind::Builtin {
                name: operation.into(),
                args: vec![variable("older"), HostExpr::new(HostExprKind::Unit)],
                ty: proof.formal_type.clone(),
            })),
            ty: proof.formal_type.clone(),
        })
    };
    assert!(proof.validates_body(&lowered("acc", "append")));
    assert!(
        !proof.validates_body(&lowered("same_shape", "append")),
        "a lowered source mutation cannot borrow checked alias authority"
    );
    assert!(
        !proof.validates_body(&lowered("acc", "concat")),
        "operation identity remains independent evidence"
    );
    let mut rebound = lowered("acc", "append");
    let HostExprKind::Let { bindings, .. } = &mut rebound.kind else {
        unreachable!()
    };
    bindings.push(binding("older", variable("same_shape")));
    assert!(
        !proof.validates_body(&rebound),
        "a lowered rebind invalidates the alias"
    );
}

#[test]
fn delta_proof_admits_only_structural_original_components() {
    for body in [
        "append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32]))",
        "concat([to_tensor([1.0f32, 2.0f32, 3.0f32])], acc)",
    ] {
        let proof = proof(body).expect("structurally inherited List interval");
        assert_eq!(proof.parameter(), "acc");
        let changed = HostExpr::new(HostExprKind::Var("acc".into(), HostTypeTerm::Unit));
        assert!(
            !proof.validates_body(&changed),
            "changed callback cannot reuse a construction proof"
        );
    }
    for body in [
        "[to_tensor([1.0f32, 2.0f32, 3.0f32])]",
        "{ acc = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
        "if i == 0i64 then acc else append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32]))",
        "{ append = fn (xs: List[tensor[3, f32]], x: tensor[3, f32]) -> xs\n append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32])) }",
    ] {
        assert!(
            proof(body).is_none(),
            "unproved result path must retain full entry walk"
        );
    }
}
