use super::*;
fn find_lambda(expr: &Expr) -> Option<&Expr> {
    if let Expr::MetaExpr(meta, _) = expr {
        return find_lambda(&meta.expr);
    }
    let node = as_node(expr)?;
    if node.tag() == DeepTag::Fn {
        return Some(expr);
    }
    node.children_slice().iter().find_map(find_lambda)
}
fn proof(body: &str) -> Option<AccumulatorCompositionProof> {
    let source = format!(
        "def step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32]))\nout = fold(fn (acc: List[tensor[*, f32]], i: i64) -> {body}, [to_tensor([1.0f32, 2.0f32, 3.0f32])], range(0i64, 2i64))\n"
    );
    let declarations =
        chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    crate::test_support::with_operation(|operation| {
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, callee) = session.def_named("step").unwrap();
        accumulator_composition_proof(
            checked
                .facts()
                .exprs()
                .last()
                .and_then(find_lambda)
                .unwrap(),
            callee,
            checked.facts().adt_registry(),
            session.operation,
        )
        .unwrap()
    })
}
fn returned_plan(body: &str) -> Option<BoundAccumulatorForwardingPlan> {
    let source = format!(
        "def step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32]))\nout = fold(fn (acc: List[tensor[*, f32]], i: i64) -> {body}, [to_tensor([1.0f32, 2.0f32, 3.0f32])], range(0i64, 2i64))\n"
    );
    let declarations =
        chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    crate::test_support::with_operation(|operation| {
        bound_accumulator_forwarding_plan(
            checked
                .facts()
                .exprs()
                .last()
                .and_then(find_lambda)
                .unwrap(),
            operation,
        )
        .unwrap()
    })
}

#[test]
fn returned_invocation_plan_transports_only_whole_immutable_results() {
    for body in [
        "{ old = acc\n item = i\n result = step(old, item)\n result }",
        "{ result = step(acc, i)\n saved = result\n result = acc\n saved }",
        "{ result = step(acc, i)\n alias = result\n alias2 = alias\n alias2 }",
    ] {
        assert!(returned_plan(body).is_some(), "{body}");
    }
    for body in [
        "{ result = step(acc, i)\n result = acc\n result }",
        "{ result = step(acc, i)\n if i == 0i64 then result else result }",
        "{ item = if i == 0i64 then i else i\n result = step(acc, item)\n result }",
        "{ step = fn (xs: List[tensor[*, f32]], n: i64) -> xs\n result = step(acc, i)\n result }",
    ] {
        assert!(returned_plan(body).is_none(), "{body}");
    }
}

#[test]
fn returned_invocation_plan_selects_occurrences_not_expression_equality() {
    let first = returned_plan("{ result = step(acc, i)\n _ = step(acc, i)\n result }").unwrap();
    let second = returned_plan("{ _ = step(acc, i)\n result = step(acc, i)\n result }").unwrap();
    assert_eq!(first.callee_name(), second.callee_name());
    assert_eq!(first.analysis_records().0, 2);
    assert_eq!(second.analysis_records().0, 2);
    assert_ne!(first.occurrence(), second.occurrence());
    assert_ne!(first.route(), second.route());
}

#[test]
fn returned_invocation_plan_counts_alias_and_candidate_records_separately() {
    for length in [40, 80, 160] {
        let mut body = String::from("{ result = step(acc, i)\n");
        for index in 0..length {
            let previous = if index == 0 {
                "result".to_owned()
            } else {
                format!("alias{}", index - 1)
            };
            body.push_str(&format!("alias{index} = {previous}\n"));
            body.push_str("_ = step(acc, i)\n");
        }
        body.push_str(&format!("alias{} }}", length - 1));
        ACCUMULATOR_IDENTITY_WORK.with(|work| work.set(0));
        let plan = returned_plan(&body).unwrap();
        let work = ACCUMULATOR_IDENTITY_WORK.with(core::cell::Cell::get);
        let (candidates, routes) = plan.analysis_records();
        assert_eq!(plan.occurrence(), 0);
        assert_eq!(candidates, length + 1);
        eprintln!(
            "returned source L={length}: visits={work} candidates={candidates} route_records={routes} selected_route={}",
            plan.route().len()
        );
        assert!(routes <= 16 * length + 64, "L={length}: routes={routes}");
        assert!(work <= 128 * length + 2048, "L={length}: work={work}");
        // Whole callback/callee byte comparisons and ordered-map
        // comparisons are separate from these structural visit units.
    }
}

#[test]
fn returned_invocation_plan_keeps_post_call_effects_and_saved_roles_opaque() {
    for body in [
        "{ old = acc\n item = i\n result = step(old, item)\n _ = print(\"after\")\n saved = result\n result = old\n saved }",
        "{ result = step(acc, i)\n _ = step(acc, i)\n result }",
    ] {
        assert!(returned_plan(body).is_some(), "{body}");
    }
    for body in [
        "{ result = step(acc, i)\n alias = result\n alias = acc\n alias }",
        "{ old = acc\n old = []\n result = step(old, i)\n result }",
    ] {
        assert!(returned_plan(body).is_none(), "{body}");
    }
}

#[test]
fn lowered_returned_invocation_requires_exact_marker_and_whole_lineage() {
    crate::test_support::with_operation(|operation| {
        let plan = accumulator_forwarding_plan(
            returned_plan("{ result = step(acc, i)\n result }")
                .unwrap()
                .owner(),
            operation,
        )
        .unwrap()
        .unwrap();
        let other = accumulator_forwarding_plan(
            returned_plan("{ _ = step(acc, i)\n result = step(acc, i)\n result }")
                .unwrap()
                .owner(),
            operation,
        )
        .unwrap()
        .unwrap();
        let ty = HostTypeTerm::Unit;
        let marked = |authority: AccumulatorForwardingPlan| {
            HostExpr::new(HostExprKind::RetainedInvocation {
                bindings: Vec::new(),
                body: Box::new(HostExpr::new(HostExprKind::Unit)),
                ty: ty.clone(),
                forwarding: Some(Arc::new(authority)),
            })
        };
        let binding = |name: &str, value| HostBinding {
            name: name.to_owned(),
            display_name: None,
            display_roots: Vec::new(),
            ty: ty.clone(),
            value,
        };
        let make = |return_name: &str, second: Option<HostExpr>| {
            let mut bindings = vec![
                binding("result", marked(plan.clone())),
                binding(
                    "saved",
                    HostExpr::new(HostExprKind::Var("result".into(), ty.clone())),
                ),
            ];
            if let Some(second) = second {
                bindings.push(binding("later", second));
            }
            HostExpr::new(HostExprKind::Let {
                bindings,
                body: Box::new(HostExpr::new(HostExprKind::Var(
                    return_name.into(),
                    ty.clone(),
                ))),
                ty: ty.clone(),
            })
        };
        assert!(returned_accumulator_invocation(&make("saved", None), &plan).is_some());
        assert!(returned_accumulator_invocation(&make("unrelated", None), &plan).is_none());
        assert!(
            returned_accumulator_invocation(&make("later", Some(marked(other.clone()))), &plan)
                .is_none()
        );
        assert!(
            returned_accumulator_invocation(&make("saved", Some(marked(plan.clone()))), &plan)
                .is_none()
        );
        assert!(returned_accumulator_invocation(&make("saved", None), &other).is_none());
    });
}

#[test]
fn lowered_returned_alias_work_counts_success_and_failed_lineage() {
    crate::test_support::with_operation(|operation| {
        let plan = accumulator_forwarding_plan(
            returned_plan("{ result = step(acc, i)\n result }")
                .unwrap()
                .owner(),
            operation,
        )
        .unwrap()
        .unwrap();
        let ty = HostTypeTerm::Unit;
        for length in [40, 80, 160] {
            let mut bindings = vec![HostBinding {
                name: "result".into(),
                display_name: None,
                display_roots: Vec::new(),
                ty: ty.clone(),
                value: HostExpr::new(HostExprKind::RetainedInvocation {
                    bindings: Vec::new(),
                    body: Box::new(HostExpr::new(HostExprKind::Unit)),
                    ty: ty.clone(),
                    forwarding: Some(Arc::new(plan.clone())),
                }),
            }];
            for index in 0..length {
                bindings.push(HostBinding {
                    name: format!("alias{index}"),
                    display_name: None,
                    display_roots: Vec::new(),
                    ty: ty.clone(),
                    value: HostExpr::new(HostExprKind::Var(
                        if index == 0 {
                            "result".into()
                        } else {
                            format!("alias{}", index - 1)
                        },
                        ty.clone(),
                    )),
                });
            }
            for valid in [true, false] {
                let body = HostExpr::new(HostExprKind::Let {
                    bindings: bindings.clone(),
                    body: Box::new(HostExpr::new(HostExprKind::Var(
                        if valid {
                            format!("alias{}", length - 1)
                        } else {
                            "unproved_equal_value".into()
                        },
                        ty.clone(),
                    ))),
                    ty: ty.clone(),
                });
                ACCUMULATOR_IDENTITY_WORK.with(|work| work.set(0));
                assert_eq!(
                    returned_accumulator_invocation(&body, &plan).is_some(),
                    valid
                );
                let work = ACCUMULATOR_IDENTITY_WORK.with(core::cell::Cell::get);
                eprintln!("returned lowered L={length} valid={valid}: visits={work}");
                assert!(work <= 16 * length + 100, "L={length}: visits={work}");
            }
        }
    });
}

#[test]
fn returned_invocation_is_not_original_accumulator_identity() {
    assert!(proof("{ result = step(acc, i)\n step(result, i) }").is_none());
    assert!(returned_plan("{ result = step(acc, i)\n step(result, i) }").is_none());
}

#[test]
fn checked_composition_preserves_sequential_forwarding_roles() {
    for body in [
        "step(acc, i)",
        "{ old = acc\n item = i\n step(old, item) }",
        "{ old = acc\n old = old\n older = old\n item = i\n item2 = item\n step(older, item2) }",
        "{ old = acc\n item = i\n acc = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n i = 0i64\n step(old, item) }",
    ] {
        assert!(proof(body).is_some(), "{body}");
    }
}
#[test]
fn checked_composition_rejects_changed_roles_and_lexical_callees() {
    for body in [
        "{ old = if i == 0i64 then acc else acc\n step(old, i) }",
        "{ item = if i == 0i64 then i else i\n step(acc, item) }",
        "{ step = fn (xs: List[tensor[*, f32]], n: i64) -> xs\n step(acc, i) }",
        "{ old = acc\n old = []\n step(old, i) }",
        "{ old = acc\n acc = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n step(acc, i) }",
    ] {
        assert!(proof(body).is_none(), "{body}");
    }
}

#[test]
fn forwarding_path_selects_only_the_tail_occurrence() {
    let proof = proof("{ old = acc\n item = i\n _ = step(old, item)\n step(old, item) }").unwrap();
    let plan = proof.forwarding();
    assert!(
        !plan.is_target(0),
        "the initializer is not the selected call"
    );
    assert!(plan.advance(0, DeepTag::App, 1).is_none());
    let mut position = 0;
    while !plan.is_target(position) {
        position = plan.advance(position, DeepTag::Let, 1).unwrap();
    }
    assert!(plan.advance(position, DeepTag::Let, 1).is_none());
}

#[test]
fn forwarding_plan_retains_the_application_beneath_legacy_metadata() {
    let source = "def step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = acc\nout = fold(fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i), [to_tensor([1.0f32, 2.0f32, 3.0f32])], range(0i64, 1i64))\n";
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    crate::test_support::with_operation(|operation| {
        let callback = checked
            .facts()
            .exprs()
            .last()
            .and_then(find_lambda)
            .unwrap();
        let node = as_node(callback).unwrap();
        let mut children = node.children_slice().to_vec();
        children[1] = Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(children[1].clone()),
            },
            children[1].span(),
        );
        let wrapped = Expr::node(DeepTag::Fn, node.meta().clone(), children, callback.span());
        let plan = accumulator_forwarding_plan(&wrapped, operation)
            .unwrap()
            .unwrap();
        assert_eq!(plan.callsite().tag(), Some(DeepTag::App));
        let app = as_node(&node.children_slice()[1]).unwrap();
        let mut actuals = app.children_slice().to_vec();
        actuals[0] = Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(actuals[0].clone()),
            },
            actuals[0].span(),
        );
        let mut children = node.children_slice().to_vec();
        children[1] = Expr::node(
            DeepTag::App,
            app.meta().clone(),
            actuals,
            children[1].span(),
        );
        let callee_wrapper =
            Expr::node(DeepTag::Fn, node.meta().clone(), children, callback.span());
        assert!(
            accumulator_forwarding_plan(&callee_wrapper, operation)
                .unwrap()
                .is_none(),
            "callable wrappers retain their own full admission"
        );
    });
}
