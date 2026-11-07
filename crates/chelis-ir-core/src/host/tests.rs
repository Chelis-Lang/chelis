use super::*;
use chelis_types::CheckedProgram;
struct TestManifest<'a>(&'a chelis_types::manifest::RootManifest);
impl ManifestView for TestManifest<'_> {
    fn len(&self) -> usize {
        self.0.entries.len()
    }
    fn entry(&self, index: usize) -> ManifestEntry<'_> {
        let entry = &self.0.entries[index];
        ManifestEntry {
            def_name: &entry.def_name,
            tensor_lane: entry.lane == chelis_types::types::Lane::Tensor,
        }
    }
}

std::thread_local! {
    static TEST_SUMMARY_PROBE_BUILDS: Cell<u64> = const { Cell::new(0) };
}
struct TestObserver;
impl crate::resources::Observer for TestObserver {
    fn host_summary_probe_build(&self) {
        TEST_SUMMARY_PROBE_BUILDS.with(|count| {
            let previous = count.get();
            count.set(if previous == u64::MAX {
                previous
            } else {
                previous + 1
            });
        });
    }
}
fn test_resources() -> crate::resources::Resources<'static, chelis_syntax_host::NativeStack> {
    let mut resources = crate::resources::Resources::new(&chelis_syntax_host::NativeStack);
    resources.observer = Some(&TestObserver);
    resources
}
fn reset_host_summary_probe_builds() {
    TEST_SUMMARY_PROBE_BUILDS.with(|count| count.set(0));
}
fn host_summary_probe_builds() -> u64 {
    TEST_SUMMARY_PROBE_BUILDS.with(Cell::get)
}

/// [04-INF-6], §4.7: actualize executable metadata without turning the
/// independently authored result claim into a physical argument fact.
#[test]
fn retained_callable_adapter_actualizes_its_whole_signature_per_invocation() {
    test_resources().run(|operation| {
        let authored = deep_expr(
            r#"(t-fn {eff: (effects {} IO), span: "callable-origin"}
            (t-ref {span: "borrow-origin"}
                (t-tensor {} (d-name {} *) (t-var {} p)))
            (t-tensor {span: "result-origin"} (d-lit {} 2) (t-var {} p)))"#,
        );
        let contract = NormalizedCallableContract::from_authored(
            &chelis_checker_core::adt::AdtRegistry::default(),
            &authored,
            operation,
        )
        .unwrap()
        .unwrap();
        let original = contract.signature().clone();
        for precision in [Prim::F32, Prim::F64] {
            let formal = HostTypeTerm::Fn(
                vec![HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Named("*".into(), None)],
                    precision,
                })],
                Box::new(HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision,
                })),
            );
            let adapter = retain_callable_invocation_contract(
                &deep_expr("(var {} abs)"),
                &formal,
                &contract,
                0,
                &mut UnordSet::new(),
                authored.span(),
            )
            .unwrap();
            let signature = checked_function_type_expr(&adapter).unwrap();
            let (actualized_params, _) = parse_fn_type_expr_parts(&signature).unwrap();
            let (authored_params, _) = parse_fn_type_expr_parts(&original).unwrap();
            assert_eq!(actualized_params[0].tag(), Some(DeepTag::TRef));
            assert_eq!(
                as_node(&actualized_params[0]).unwrap().meta(),
                as_node(&authored_params[0]).unwrap().meta(),
                "borrow/source parameter metadata changed"
            );
            assert!(as_node(&original).unwrap().meta().eff().is_some());
            assert_eq!(
                as_node(&signature).unwrap().meta(),
                as_node(&original).unwrap().meta(),
                "authored effects/source metadata changed"
            );
            assert_eq!(decode_host_type(&signature).unwrap(), formal);
            assert_eq!(
                contract.signature(),
                &original,
                "cached authored frame changed"
            );
        }
    })
}

/// [04-INF-6]: specializing a tensor's executable shape retains the
/// provenance of every independently authored child and expanded spread.
#[test]
fn retained_callable_actualization_keeps_tensor_child_provenance() {
    test_resources().run(|operation| {
        let authored = deep_expr(
            r#"(t-fn {eff: (effects {} IO), span: "callable-origin"}
            (t-ref {span: "borrow-origin"}
                (t-tensor {span: "tensor-origin"}
                    (d-lit {span: "literal-origin"} 2)
                    (d-name {span: "anchor-origin"} anchor)
                    (d-rank {span: "spread-origin"} rest)
                    (t-var {span: "precision-origin"} p)))
            (t-tensor {span: "result-origin"} (d-lit {} 2) (t-var {} p)))"#,
        );
        let contract = NormalizedCallableContract::from_authored(
            &chelis_checker_core::adt::AdtRegistry::default(),
            &authored,
            operation,
        )
        .unwrap()
        .unwrap();
        let (parameters, _) = parse_fn_type_expr_parts(contract.signature()).unwrap();
        let (_, _, reference) = stamped_parts(&parameters[0]).unwrap();
        let (_, _, original_children) = stamped_parts(&reference[0]).unwrap();
        for precision in [Prim::F32, Prim::F64] {
            let formal = HostTypeTerm::Fn(
                vec![HostTypeTerm::Tensor(TensorType {
                    dims: vec![
                        DimInfo::Lit(2),
                        DimInfo::Named("anchor".into(), None),
                        DimInfo::Lit(3),
                        DimInfo::Lit(4),
                    ],
                    precision,
                })],
                Box::new(HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision,
                })),
            );
            let actualized = actualize_callable_invocation_contract(&contract, &formal).unwrap();
            let (parameters, _) = parse_fn_type_expr_parts(actualized.signature()).unwrap();
            let (_, _, reference) = stamped_parts(&parameters[0]).unwrap();
            let (_, _, children) = stamped_parts(&reference[0]).unwrap();
            assert_eq!(children.len(), 5);
            for (axis, original) in [0, 1, 2, 2, 3].into_iter().enumerate() {
                assert_eq!(
                    as_node(&children[axis]).unwrap().meta(),
                    as_node(&original_children[original]).unwrap().meta(),
                    "actualized child {axis} lost authored provenance"
                );
            }
            assert_eq!(actualized.authored(), contract.authored());
            let HostTypeTerm::Fn(parameters, result) =
                decode_host_type(actualized.signature()).unwrap()
            else {
                panic!("callable signature lost");
            };
            let HostTypeTerm::Tensor(input) = &parameters[0] else {
                panic!("tensor input lost")
            };
            assert_eq!(input.dims.len(), 4);
            assert_eq!(input.precision, precision);
            assert_eq!(
                *result,
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision
                })
            );
        }
    })
}

#[test]
fn retained_callable_actualization_refuses_missing_rank_correspondence() {
    test_resources()
        .run(|operation| { let authored = deep_expr(
        "(t-fn {} (t-tensor {} (d-lit {} 2) (d-name {} anchor) (d-rank {} rest) (t-var {} p)) (t-tensor {} (d-lit {} 2) (t-var {} p)))",
    );
    let contract = NormalizedCallableContract::from_authored(
        &chelis_checker_core::adt::AdtRegistry::default(),
        &authored,
     operation)
    .unwrap()
    .unwrap();
    let formal = HostTypeTerm::Fn(
        vec![HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        })],
        Box::new(HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        })),
    );
    assert!(
        actualize_callable_invocation_contract(&contract, &formal).is_err(),
        "missing anchor/rank evidence must not synthesize an executable frame"
    ); })
}

/// §4.7 literal refutation is decided by the existing whole-call DAG
/// owner, after caller values are prepared and before body execution.
#[test]
fn whole_named_call_preserves_closed_list_value_shape() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "def g(a: List[f32], b: List[f32]) -> tensor[*, f32] = add(to_tensor(a), to_tensor(b))\n\
         def main() -> tensor[*, f32] = g([1.0f32, 2.0f32], [3.0f32, 4.0f32])\n",
    );
    let session = HostLoweringSession::new(program.facts(), operation);
    let (name, body) = session.def_named("main").unwrap();
    let signature = host_def_signature(name, body, None, &session).expect("valid checked admission facts").unwrap();
    let context = cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
    let dag = crate::lower::try_lower_tensor_helper_program_with_ordered_inputs(
        &signature.body_expr,
        Vec::new(),
        &context,
        None,
        false,
        None,
     operation)
    .expect("a whole checked call must retain its closed List actual values");
    assert!(kernel_dag_loads_builtin(&dag, &HostScope::new()).is_none());
    assert_eq!(
        dag.get(dag.roots()[0]).unwrap().output_type.dims,
        [DimInfo::Lit(2)]
    ); })
}

#[test]
fn whole_named_call_refutes_conflicting_closed_list_values() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "def g(a: List[f32], b: List[f32]) -> tensor[*, f32] = add(to_tensor(a), to_tensor(b))\n\
         def main() -> tensor[*, f32] = g([1.0f32, 2.0f32], [3.0f32, 4.0f32, 5.0f32])\n",
    );
    let session = HostLoweringSession::new(program.facts(), operation);
    let (name, body) = session.def_named("main").unwrap();
    let signature = host_def_signature(name, body, None, &session).expect("valid checked admission facts").unwrap();
    let context = cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
    let error = crate::lower::try_lower_tensor_helper_program_with_ordered_inputs(
        &signature.body_expr,
        Vec::new(),
        &context,
        None,
        false,
        None,
     operation)
    .expect_err("closed unequal List actuals must refute before execution");
    assert!(error.fatal, "{error:?}");
    assert!(error.is_dimension_mismatch(), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("DimensionMismatch: `add` argument 2, axis 0: expected 2, got 3"),
        "{error:?}"
    ); })
}

/// §4.7 applies to a complete call with an inferred result just as it
/// applies to a fully annotated definition. Parameter annotations do not
/// leave the result as an executable inference hole after checking.
#[test]
fn whole_named_call_inferred_result_keeps_literal_refutation() {
    test_resources().run(|operation| {
        for (mismatch, right) in [
            (false, "[3.0f32, 4.0f32]"),
            (true, "[3.0f32, 4.0f32, 5.0f32]"),
        ] {
            let program = surf_check(&format!(
                "def g(a: List[f32], b: List[f32]) = add(to_tensor(a), to_tensor(b))\n\
             before = print(\"before\")\n\
             out = g([1.0f32, 2.0f32], {right})\n"
            ));
            let session = HostLoweringSession::new(program.facts(), operation);
            let lowered = top_level_lowering_map(program.exprs(), program.type_env(), operation)
                .expect("checked fixture classification");
            let result = lower_host_program_with_execution(&session, &lowered, None, true, false);
            if mismatch {
                let error = match result {
                    Err(error) => error,
                    Ok(_) => panic!(
                        "inferred call must refute; declared={:?}, checked={:?}",
                        lookup_declared_fn_type(&session, "g").expect("valid checked metadata"),
                        session.signature_inference().functions.get("g")
                    ),
                };
                assert!(error.fatal && error.is_dimension_mismatch(), "{error:?}");
                assert!(
                    error
                        .to_string()
                        .contains("DimensionMismatch: `add` argument 2, axis 0: expected 2, got 3"),
                    "{error:?}"
                );
            } else {
                result.expect("agreeing inferred-result call remains executable");
            }
        }
    })
}

/// [04-INF-6], §4.7: a pass-through body cannot replace the actual
/// definition's shared-row entry, even behind a wider callable formal.
#[test]
fn retained_definition_entry_precedes_body_only_helper_reduction() {
    test_resources()
        .run(|operation| { use chelis_dag_core::axis_sources::{
        EntryExtentGuard, ExtentRecord, entry_extent_guards, witness_entry_obligations,
    };
    use chelis_dag_core::dag::{NodeId, RtAxis};
    fn shared_rank(guards: &[EntryExtentGuard]) -> bool {
        guards.iter().any(|guard| {
            matches!(guard, EntryExtentGuard::Named {
            claim, canonical: (NodeId(0), 1), observed: (NodeId(1), 1),
        } if claim == "rest[0]")
        })
    }
    fn rank_entries<T>(expr: &HostExpr<T>, output: &mut Vec<Vec<EntryExtentGuard>>) {
        match &expr.kind {
            HostExprKind::SignatureEntry {
                plan,
                args,
                positions,
                ..
            } => {
                if shared_rank(plan.guards()) {
                    assert_eq!(
                        positions,
                        &[0, 1],
                        "both declaration positions are required"
                    );
                    assert_eq!(args.len(), 2, "unused input observation was dropped");
                }
                output.push(plan.guards().to_vec());
            }
            HostExprKind::RetainedInvocation { bindings, body, .. }
            | HostExprKind::Let { bindings, body, .. } => {
                for binding in bindings {
                    rank_entries(&binding.value, output);
                }
                rank_entries(body, output);
            }
            HostExprKind::ResultClaimScope { body, .. }
            | HostExprKind::FormalIngress { value: body, .. }
            | HostExprKind::ExtentSites { value: body, .. } => rank_entries(body, output),
            _ => {}
        }
    }
    for invoke in [false, true] {
        for width in [2, 3] {
            let call = if invoke { "f(value, ignored)" } else { "value" };
            let row = vec!["1.0f64"; width].join(",");
            let source = format!(
                "def first[rest, p: Float](value: tensor[extent, ..rest, p], ignored: tensor[extent, ..rest, p]) -> tensor[extent, ..rest, p] = value\n\
                 def invoke[p: Float](f: tensor[*, *, p] -> tensor[*, *, p] -> tensor[*, *, p], value: tensor[*, *, p], ignored: tensor[*, *, p]) -> tensor[*, *, p] ! {{IO}} = {{\n\
                 _ = print(\"caller-before\")\n\
                 result = {call}\n\
                 _ = print(\"caller-after\")\n\
                 result\n}}\n\
                 out = invoke(first, to_tensor([[1.0f64,2.0f64],[3.0f64,4.0f64],[5.0f64,6.0f64]]), to_tensor([[{row}],[{row}],[{row}]]))\n"
            );
            let program = surf_check(&source);
            let realizability = chelis_effects::realizability::infer_realizability(
                &program,
                &[Prim::F32, Prim::Bool],
            );
            let manifest =
                chelis_effects::realizability::compute_root_manifest(&program, &realizability);
            let (compiled, _) = try_lower_compiled_program_with_options(
                program.facts(),
                Some(&TestManifest(&manifest)),
                true,
                false, operation)
            .unwrap();
            let mut plans = Vec::new();
            let host = compiled.host.as_ref().unwrap();
            for binding in &host.globals {
                rank_entries(&binding.value, &mut plans);
            }
            let mut rank_retained = plans.iter().any(|guards| shared_rank(guards));
            let mut witness_plans = Vec::new();
            for helper in &host.global_tensor_helpers {
                let guards = entry_extent_guards(&helper.dag);
                let mut obligations = Vec::new();
                for node in helper.dag.nodes() {
                    // The ordinary C lane also retains these witnesses as
                    // shape dependencies and emits Drop value edges. The
                    // HIP prologue-only classification excludes value-used
                    // witnesses; the shared input-read projection still
                    // owns their ordered entry claims.
                    if let Some(projected) = witness_entry_obligations(&helper.dag, node.id) {
                        obligations.extend(projected);
                    }
                }
                let rank_witness = obligations.iter().any(|obligation| {
                    obligation.label == "rest[0]"
                        && obligation.operation == "load"
                        && matches!(&obligation.records, (
                            ExtentRecord::Read { load: NodeId(0), axis: 1, parameter: first },
                            ExtentRecord::Read { load: NodeId(1), axis: 1, parameter: second },
                        ) if first == "value" && second == "ignored")
                });
                if shared_rank(&guards) || rank_witness {
                    rank_retained = true;
                    assert_eq!(helper.inputs.len(), 2, "both callable inputs are required");
                    let mut reachable = BTreeSet::new();
                    let mut pending = helper.dag.roots().to_vec();
                    while let Some(id) = pending.pop() {
                        if reachable.insert(id) {
                            let node = helper.dag.get(id).unwrap();
                            pending.extend(node.inputs.iter().copied());
                            pending.extend(node.shape_deps.iter().copied());
                            pending.extend(node.result_claim_deps.iter().copied());
                        }
                    }
                    let ignored = helper.dag.nodes().iter().find(|node| matches!(
                        &node.op, RiscOp::ExtentWitness {
                            parameter, axis: RtAxis::Lit(1), claims, ..
                        } if parameter == "ignored" && claims.iter().any(|claim| claim.claim == "rest[0]")
                    )).expect("unused parameter's rank witness is required");
                    assert!(
                        reachable.contains(&ignored.id),
                        "unused input witness is not retained"
                    );
                    assert!(
                        reachable.contains(&NodeId(1)),
                        "unused input dependency is not retained"
                    );
                }
                plans.push(guards);
                witness_plans.extend(obligations);
            }
            if invoke {
                assert!(
                    rank_retained,
                    "invoked definition lost shared-row entry at width {width}: {plans:?}, {witness_plans:?}"
                );
            } else {
                assert!(
                    plans.iter().all(Vec::is_empty) && witness_plans.is_empty(),
                    "an uninvoked callback manufactured obligations at width {width}: {plans:?}, {witness_plans:?}"
                );
            }
        }
    } })
}

/// spec/06 §3: a concrete supplied target specializes a template even
/// when the template has no standalone transformed callable ABI.
#[test]
fn retained_callable_static_target_classifies_the_concrete_grad_caller() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "def apply(h: tensor[1, f32] -> tensor[1, f32], v: tensor[2, 1, f32]) -> tensor[2, 1, f32] = {\n a = to_tensor([100.0f32])\n vmap(h)(v)\n}\n\
         def main() -> tensor[1, f32] = grad(fn (x: tensor[1, f32]) -> {\n a = mul(x, x)\n sum(sum(apply(fn (v: tensor[1, f32]) -> mul(v, a), insert(x, 0i32, 2i64)), 0i32), 0i32)\n})(to_tensor([5.0f32]))\n",
    );
    let lowered = top_level_lowering_map(program.exprs(), program.type_env(), operation).expect("checked fixture classification");
    assert_eq!(
        lowered.get("apply"),
        Some(&false),
        "unresolved standalone template"
    );
    assert_eq!(
        lowered.get("main"),
        Some(&true),
        "concrete lexical target was lost"
    );
    let compiled = try_lower_compiled_program(program.facts(), operation).unwrap();
    assert!(
        compiled
            .host
            .as_ref()
            .is_none_or(|host| { host_program_unresolved_transform_sites(host).is_empty() }),
        "a specialized DAG entry must not emit an unresolved template wrapper"
    );
    let realizability =
        chelis_effects::realizability::infer_realizability(&program, &[Prim::F32, Prim::Bool]);
    let manifest =
        chelis_effects::realizability::compute_root_manifest(&program, &realizability);
    let compiled = try_lower_compiled_program_with_manifest(program.facts(), &TestManifest(&manifest), operation).unwrap();
    assert!(
        compiled
            .host
            .as_ref()
            .is_none_or(|host| { host_program_unresolved_transform_sites(host).is_empty() }),
        "manifest-selected specialized DAG entry must not emit an unresolved template wrapper"
    ); })
}

#[test]
fn retained_callable_runtime_target_does_not_gain_static_transform_identity() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "def apply(h: tensor[1, f32] -> tensor[1, f32], v: tensor[2, 1, f32]) -> tensor[2, 1, f32] = vmap(h)(v)\n\
         def selected(flag: bool, x: tensor[2, 1, f32]) -> tensor[2, 1, f32] = apply(if flag then (fn (v: tensor[1, f32]) -> neg(v)) else (fn (v: tensor[1, f32]) -> abs(v)), x)\n",
    );
    let lowered = top_level_lowering_map(program.exprs(), program.type_env(), operation).expect("checked fixture classification");
    assert_eq!(lowered.get("selected"), Some(&false)); })
}

/// spec/04 §4.7: an invocation's result is an obligation, not evidence
/// that specializes a recursive callee's own runtime extent witness.
#[test]
fn retained_type_substitution_does_not_bind_activation_dimensions() {
    test_resources().run(|operation| {
        let dimension =
            |dim| HostTypeTerm::Dimension(crate::host_type_state::HostShapeSlot::Dim(dim));
        let formal = HostTypeTerm::Adt(
            "Box".into(),
            vec![dimension(DimInfo::Named("m".into(), None))],
        );
        let actual = HostTypeTerm::Adt("Box".into(), vec![dimension(DimInfo::Lit(3))]);
        let mut types = UnordMap::new();
        solve_host_type_vars(&formal, &actual, &mut types, operation);
        assert!(
            types.is_empty(),
            "nominal dimensions are not type variables"
        );
        solve_host_type_vars(
            &HostTypeTerm::TypeVariable("m".into()),
            &dimension(DimInfo::Lit(3)),
            &mut types,
            operation,
        );
        assert!(types.is_empty(), "type variables cannot capture dimensions");
        assert_eq!(substitute_host_type_term(formal.clone(), &types), formal);

        let mut binders = UnordSet::new();
        binders.insert("m".into());
        let mut dimensions = UnordMap::new();
        solve_checked_dimensions(&formal, &formal, &binders, &mut dimensions);
        assert!(
            dimensions.is_empty(),
            "a binder observing itself is not extent evidence"
        );
        solve_checked_dimensions(&formal, &actual, &binders, &mut dimensions);
        assert_eq!(dimensions.get("m"), Some(&DimInfo::Lit(3)));

        solve_host_type_vars(
            &HostTypeTerm::TypeVariable("t".into()),
            &HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::F32)),
            &mut types,
            operation,
        );
        assert_eq!(
            types.get("t"),
            Some(&HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                Prim::F32
            )))
        );
        assert!(!types.contains_key("m"));
    })
}

#[test]
fn authored_nominal_arguments_keep_the_registry_parameter_kind() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "type Box[n] = | Box { v: tensor[n, f32] }\n\
         type Value[t] = | Value { v: t }\n\
         def boxed[m](u: tensor[m, f32]) -> Box[m] = Box { v: u }\n\
         def value[t](u: t) -> Value[t] = Value { v: u }\n",
        );
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, boxed) = lookup_declared_fn_type(&session, "boxed")
            .expect("valid checked metadata")
            .unwrap();
        assert_eq!(
            boxed,
            HostTypeTerm::Adt(
                "Box".into(),
                vec![HostTypeTerm::Dimension(
                    crate::host_type_state::HostShapeSlot::Dim(DimInfo::Named("m".into(), None),)
                )],
            ),
        );
        let (_, value) = lookup_declared_fn_type(&session, "value")
            .expect("valid checked metadata")
            .unwrap();
        assert_eq!(
            value,
            HostTypeTerm::Adt("Value".into(), vec![HostTypeTerm::TypeVariable("t".into())]),
        );
    })
}

#[test]
fn callable_conformance_preserves_physical_descriptor_beside_logical_contract() {
    test_resources().run(|operation| {
        let builtin_operation = CheckedBuiltinCallable::from_symbol("is_finite").unwrap();
        let physical = HostTypeTerm::CheckedBuiltinCallable(builtin_operation);
        let value = HostExpr::new(HostExprKind::Builtin {
            name: builtin_operation.into(),
            args: Vec::new(),
            ty: physical.clone(),
        });
        let logical = HostTypeTerm::Fn(
            vec![HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F64,
            })],
            Box::new(HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::Bool,
            })),
        );
        assert_eq!(
            host_expr_type(&force_host_expr_type(value, logical, operation)),
            physical
        );
    })
}

#[test]
fn checked_callable_join_preserves_branch_identity_through_conformance() {
    test_resources().run(|operation| {
        let nan = CheckedBuiltinCallable::from_symbol("is_nan").unwrap();
        let finite = CheckedBuiltinCallable::from_symbol("is_finite").unwrap();
        let ty = |op| HostTypeTerm::CheckedBuiltinCallable(op);
        let joined = HostTypeTerm::checked_callable_join([ty(nan), ty(finite)]).unwrap();
        assert_eq!(
            joined,
            HostTypeTerm::checked_callable_join([ty(finite), ty(nan)]).unwrap()
        );
        assert_eq!(
            joined,
            HostTypeTerm::checked_callable_join([joined.clone(), ty(nan)]).unwrap()
        );
        assert_eq!(
            HostTypeTerm::checked_callable_join([ty(nan), ty(nan)]),
            Some(ty(nan))
        );
        assert!(HostTypeTerm::checked_callable_join([ty(nan), HostTypeTerm::Bool]).is_none());
        let branch = |op: CheckedBuiltinCallable| {
            HostExpr::new(HostExprKind::Builtin {
                name: op.into(),
                args: Vec::new(),
                ty: ty(op),
            })
        };
        let expr = force_host_expr_type(
            HostExpr::new(HostExprKind::If {
                cond: Box::new(HostExpr::new(HostExprKind::Bool(false))),
                then_expr: Box::new(branch(nan)),
                else_expr: Box::new(branch(finite)),
                ty: joined.clone(),
            }),
            joined.clone(),
            operation,
        );
        let HostExprKind::If {
            then_expr,
            else_expr,
            ty: actual,
            ..
        } = expr.kind
        else {
            panic!("if")
        };
        assert_eq!(actual, joined);
        assert_eq!(host_expr_type(&then_expr), ty(nan));
        assert_eq!(host_expr_type(&else_expr), ty(finite));
        assert!(matches!(
            joined.into_concrete(),
            Ok(ConcreteHostType::CheckedCallableChoice(_))
        ));
    })
}

#[test]
fn stationary_accumulator_proof_is_checked_and_revalidated_after_host_transforms() {
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
    fn proof(body: &str, item_type: &str, items: &str) -> Option<StationaryAccumulatorProof> {
        let source = format!(
            "out = fold(fn (acc: List[tensor[3, f32]], i: {item_type}) -> {body}, [to_tensor([1.0f32, 2.0f32, 3.0f32])], {items})\n"
        );
        let declarations =
            chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap();
        let deep =
            chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
                .unwrap();
        let checked = chelis_types::check_ir_program(&deep).unwrap();
        let function = checked.exprs().iter().find_map(find_fn).unwrap();
        stationary_accumulator_proof(function, checked.adt_registry())
            .expect("valid checked accumulator metadata")
    }
    for body in [
        "acc",
        "{ _ = print(i)\n acc }",
        "if i == 0i64 then acc else acc",
    ] {
        let proof = proof(body, "i64", "range(0i64, 2i64)")
            .expect("same checked formal on every result path");
        assert_eq!(proof.parameter(), "acc");
        assert!(proof.pattern().nested_root().is_some());
        let same = HostExpr::new(HostExprKind::Var("acc".into(), HostTypeTerm::Unit));
        assert!(proof.validates_body(&same));
        let other = HostExpr::new(HostExprKind::Var("other".into(), HostTypeTerm::Unit));
        assert!(
            !proof.validates_body(&other),
            "a transported proof cannot name another formal"
        );
        let fresh = HostExpr::new(HostExprKind::List(Vec::new(), HostTypeTerm::Unit));
        assert!(
            !proof.validates_body(&fresh),
            "a transported proof cannot approve a constructed result"
        );
    }
    for body in [
        "[to_tensor([1.0f32, 2.0f32, 3.0f32])]",
        "{ acc = [to_tensor([1.0f32, 2.0f32, 3.0f32])]\n acc }",
        "if i == 0i64 then acc else [to_tensor([1.0f32, 2.0f32, 3.0f32])]",
    ] {
        assert!(proof(body, "i64", "range(0i64, 2i64)").is_none());
    }
    assert!(
        proof(
            "acc",
            "tensor[3, f32]",
            "[to_tensor([1.0f32, 2.0f32, 3.0f32])]"
        )
        .is_none(),
        "another tensor formal keeps its independent entry checks"
    );
}
use chelis_dag_core::{DimInfo, RiscOp};
use chelis_numeric_core::prim::Prim;

/// [05-OP-46]: a unary producer retains the actual operand's complete
/// scalar/tensor carrier, including unresolved precision and rank rows.
/// Its classification consumer must not recover that information from
/// the Boolean result annotation.
#[test]
fn type_preserving_builtin_host_inference_retains_actual_carrier() {
    test_resources().run(|_| {
        for name in [
            "neg", "recip", "exp", "log", "sin", "sqrt", "cos", "tan", "atan", "tanh", "erf",
            "erfc", "abs", "floor", "ceil", "round",
        ] {
            let mut inputs = Vec::new();
            for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
                inputs.push(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)));
                for dims in [
                    vec![],
                    vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
                ] {
                    inputs.push(HostTypeTerm::Tensor(TensorType { dims, precision }));
                }
                inputs.push(HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                    shape: HostShapeTerm::Polymorphic(vec![
                        HostShapeSlot::RankVariable("before".into()),
                        HostShapeSlot::Dim(DimInfo::Named("seq".into(), None)),
                        HostShapeSlot::RankVariable("after".into()),
                    ]),
                    precision: HostPrecisionTerm::Concrete(precision),
                }));
            }
            inputs.push(HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                shape: HostShapeTerm::Polymorphic(vec![HostShapeSlot::RankVariable("r".into())]),
                precision: HostPrecisionTerm::Variable("p".into()),
            }));
            for input in inputs {
                assert_eq!(
                    infer_builtin_host_type_from_arg_tys(name, core::slice::from_ref(&input))
                        .expect("valid builtin fixture"),
                    Some(input.clone()),
                    "{name} lost actual operand {input:?}"
                );
            }
        }
    })
}

#[test]
fn type_preserving_builtin_host_inference_requires_scheme_identity_and_arity() {
    test_resources().run(|_| {
        for name in ["abs", "cos", "recip", "floor"] {
            assert_eq!(type_preserving_builtin_argument(name), Some((1, 0)));
            assert!(infer_builtin_host_type_from_arg_tys(name, &[]).is_err());
            assert!(
                infer_builtin_host_type_from_arg_tys(
                    name,
                    &[HostTypeTerm::Float32, HostTypeTerm::Float32]
                )
                .is_err()
            );
        }
        // Shape identity alone cannot turn a precision-changing predicate
        // into its float operand, and a shared scheme variable alone cannot
        // assert that a movement operation preserves the operand's shape.
        for name in [
            "is_nan",
            "is_finite",
            "is_infinite",
            "cmplt",
            "reshape",
            "permute",
            "sum",
            "not_a_builtin",
        ] {
            assert_eq!(type_preserving_builtin_argument(name), None, "{name}");
        }
        let input = HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        });
        assert_eq!(
            infer_builtin_host_type_from_arg_tys("cmplt", &[input.clone(), input])
                .expect("valid builtin fixture"),
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: Prim::Bool
            }))
        );
    })
}

#[test]
fn predicate_host_application_owns_actual_surface_and_axes() {
    test_resources().run(|_| {
        for kind in [
            chelis_numeric_core::dtype_semantics::FloatClassify::Nan,
            chelis_numeric_core::dtype_semantics::FloatClassify::Finite,
            chelis_numeric_core::dtype_semantics::FloatClassify::Infinite,
        ] {
            for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
                let axes = vec![
                    DimInfo::Named("batch".into(), None),
                    DimInfo::Named("_rt_producer_axis".into(), None),
                ];
                let input = HostExpr::new(HostExprKind::Var(
                    "input".into(),
                    HostTypeTerm::Tensor(TensorType {
                        dims: axes.clone(),
                        precision,
                    }),
                ));
                let coarse = HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Named("*".into(), None); 2],
                    precision: Prim::Bool,
                });
                let actual = CheckedApplicationRelation(kind)
                    .instantiate(&[input], &coarse)
                    .unwrap();
                assert_eq!(
                    actual,
                    HostTypeTerm::Tensor(TensorType {
                        dims: axes,
                        precision: Prim::Bool
                    })
                );
                let scalar = HostExpr::new(HostExprKind::Var(
                    "input".into(),
                    HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)),
                ));
                assert_eq!(
                    CheckedApplicationRelation(kind)
                        .instantiate(&[scalar], &HostTypeTerm::Bool)
                        .unwrap(),
                    HostTypeTerm::Bool
                );
            }
        }
    })
}

#[test]
fn predicate_shape_conformance_keeps_known_conflicts_for_rejection() {
    test_resources().run(|operation| {
        for kind in [
            chelis_numeric_core::dtype_semantics::FloatClassify::Nan,
            chelis_numeric_core::dtype_semantics::FloatClassify::Finite,
            chelis_numeric_core::dtype_semantics::FloatClassify::Infinite,
        ] {
            for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
                let input_ty = HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision,
                });
                let wrong = HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::Bool,
                });
                let value = force_host_expr_type(
                    HostExpr::new(HostExprKind::Builtin {
                        name: CheckedBuiltinCallable::FloatClassify(kind).into(),
                        args: vec![HostExpr::new(HostExprKind::Var(
                            "x".into(),
                            input_ty.clone(),
                        ))],
                        ty: wrong.clone(),
                    }),
                    wrong.clone(),
                    operation,
                );
                let HostExprKind::Builtin { args, ty, .. } = value.kind else {
                    panic!("closed operation must retain identity");
                };
                assert_eq!(host_expr_type(&args[0]), input_ty);
                assert_eq!(ty, wrong);
                assert!(
                    CheckedApplicationRelation(kind)
                        .instantiate(&args, &ty)
                        .is_err()
                );
            }
        }
    })
}

#[test]
fn predicate_host_relation_rejects_wrong_family_surface_precision_rank_and_enclosing_extent() {
    test_resources().run(|_| {
        for kind in [
            chelis_numeric_core::dtype_semantics::FloatClassify::Nan,
            chelis_numeric_core::dtype_semantics::FloatClassify::Finite,
            chelis_numeric_core::dtype_semantics::FloatClassify::Infinite,
        ] {
            let relation = CheckedApplicationRelation(kind);
            let input = HostExpr::new(HostExprKind::Var(
                "x".into(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision: Prim::F32,
                }),
            ));
            for wrong in [
                HostTypeTerm::Bool,
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision: Prim::F32,
                }),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![],
                    precision: Prim::Bool,
                }),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::Bool,
                }),
            ] {
                assert!(
                    relation
                        .instantiate(core::slice::from_ref(&input), &wrong)
                        .is_err()
                );
            }
            for precision in [Prim::Int32, Prim::Bool] {
                let input = HostExpr::new(HostExprKind::Var(
                    "x".into(),
                    HostTypeTerm::Tensor(TensorType {
                        dims: vec![DimInfo::Lit(2)],
                        precision,
                    }),
                ));
                assert!(
                    relation
                        .instantiate(&[input], &fresh_host_inference())
                        .is_err()
                );
            }
            assert!(relation.instantiate(&[], &HostTypeTerm::Bool).is_err());
            assert!(
                relation
                    .instantiate(&[input.clone(), input], &HostTypeTerm::Bool)
                    .is_err()
            );
            let actual = HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::Bool,
            });
            let stale = HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: Prim::Bool,
            });
            for (expected, actual) in [
                (
                    HostTypeTerm::List(Box::new(stale.clone())),
                    HostTypeTerm::List(Box::new(actual.clone())),
                ),
                (
                    HostTypeTerm::Option(Box::new(stale.clone())),
                    HostTypeTerm::Option(Box::new(actual.clone())),
                ),
                (
                    HostTypeTerm::Tuple(vec![stale]),
                    HostTypeTerm::Tuple(vec![actual]),
                ),
            ] {
                assert!(
                    validate_checked_result_requirement(&expected, &actual, kind.name()).is_err()
                );
            }
        }
    })
}

#[test]
fn refined_predicate_condition_cannot_change_where_data_precision() {
    test_resources().run(|_| {
        for precision in [
            Prim::F16,
            Prim::Bf16,
            Prim::F32,
            Prim::F64,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ] {
            let dims = vec![DimInfo::Named("_rt_data".into(), None)];
            let data = HostTypeTerm::Tensor(TensorType {
                dims: dims.clone(),
                precision,
            });
            let condition = HostTypeTerm::Tensor(TensorType {
                dims,
                precision: Prim::Bool,
            });
            assert_eq!(
                infer_builtin_host_type_from_arg_tys(
                    "where",
                    &[condition.clone(), data.clone(), data.clone()]
                )
                .unwrap(),
                Some(data.clone())
            );
            for wrong in [
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![],
                    precision,
                }),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision: if precision == Prim::Bool {
                        Prim::F32
                    } else {
                        Prim::Bool
                    },
                }),
            ] {
                assert!(
                    infer_builtin_host_type_from_arg_tys(
                        "where",
                        &[condition.clone(), data.clone(), wrong]
                    )
                    .is_err()
                );
            }
            let data = HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision,
            });
            let wrong_extent = HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision,
            });
            assert!(
                infer_builtin_host_type_from_arg_tys(
                    "where",
                    &[condition.clone(), data, wrong_extent]
                )
                .is_err()
            );
            let bad_condition = HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("_rt_data".into(), None)],
                precision: Prim::Int32,
            });
            let data = HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("_rt_data".into(), None)],
                precision,
            });
            assert!(
                infer_builtin_host_type_from_arg_tys("where", &[bad_condition, data.clone(), data])
                    .is_err()
            );
            assert!(infer_builtin_host_type_from_arg_tys("where", &[condition]).is_err());
        }
    })
}

#[test]
fn inferred_type_holes_are_not_authored_binders() {
    test_resources().run(|_| {
        let parse = |source: &str| {
            chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack)
                .expect("Deep type parses")
                .into_iter()
                .next()
                .expect("one type")
        };
        let authored = parse("(t-fn {} (t-var {} _) (t-var {} p))");
        let checked = parse("(t-fn {} (t-var {} t7) (t-var {} t8))");
        let mut identities = UnordMap::new();
        pair_authored_binder_identities(&authored, &checked, &mut identities, &mut UnordMap::new());
        assert_eq!(identities.get("_"), None);
        assert_eq!(identities.get("p").map(String::as_str), Some("t8"));

        let body = parse("(t-fn {} (t-var {} _) (t-var {} p))");
        let respelled = respell_type_binders(&body, &identities, &UnordMap::new());
        assert_eq!(
            chelis_deep::printer::print_expr_flat(&respelled),
            "(t-fn {} (t-var {} _) (t-var {} t8))"
        );
    })
}

#[test]
fn retained_body_respells_dimension_binders_in_callback_annotations() {
    test_resources().run(|_| {
        let parse = |source: &str| {
            chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack)
                .expect("Deep type parses")
                .into_iter()
                .next()
                .expect("one type")
        };
        let authored =
            parse("(t-fn {} (t-tensor {} (d-var {} c) (d-var {} f)) (t-tensor {} (d-var {} c)))");
        let checked = parse(
            "(t-fn {} (t-tensor {} (d-var {} d1) (d-var {} d2)) (t-tensor {} (d-var {} d1)))",
        );
        let body = parse("(t-tensor {} (d-var {} c) (d-var {} f))");
        let mut identities = UnordMap::new();
        pair_authored_binder_identities(&authored, &checked, &mut UnordMap::new(), &mut identities);
        let respelled = respell_type_binders(&body, &UnordMap::new(), &identities);
        assert_eq!(
            chelis_deep::printer::print_expr_flat(&respelled),
            "(t-tensor {} (d-var {} d1) (d-var {} d2))"
        );
    })
}

#[test]
fn issue_1248_comparison_family_preserves_operand_surface() {
    test_resources().run(|_| {
        let dims = vec![DimInfo::Named("n".into(), None), DimInfo::Lit(3)];
        let shape = HostShapeTerm::Polymorphic(vec![
            crate::host_type_state::HostShapeSlot::RankVariable("r".into()),
            crate::host_type_state::HostShapeSlot::Dim(DimInfo::Named("n".into(), None)),
        ]);
        let fixtures = [
            (HostTypeTerm::Float32, HostTypeTerm::Bool),
            (
                HostTypeTerm::Tensor(TensorType {
                    dims: dims.clone(),
                    precision: Prim::F32,
                }),
                HostTypeTerm::Tensor(TensorType {
                    dims,
                    precision: Prim::Bool,
                }),
            ),
            (
                HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                    precision: HostPrecisionTerm::Variable("p".into()),
                    shape: shape.clone(),
                }),
                HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                    precision: HostPrecisionTerm::Concrete(Prim::Bool),
                    shape,
                }),
            ),
        ];
        for name in ["cmplt", "lt", "gt", "gte", "lte", "eq", "neq"] {
            for (input, expected) in &fixtures {
                assert_eq!(
                    infer_builtin_host_type_from_arg_tys(name, &[input.clone(), input.clone()])
                        .expect("valid builtin fixture"),
                    Some(expected.clone()),
                    "{name}: {input:?}"
                );
            }
        }
    })
}

#[test]
fn helper_operand_views_widen_axes_without_refining_physical_claims() {
    test_resources().run(|_| {
        let physical = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let checked = TensorType {
            dims: vec![
                DimInfo::Named("*".into(), None),
                DimInfo::Named("n".into(), None),
            ],
            precision: Prim::F32,
        };
        assert_eq!(tensor_helper_operand_view(&physical, &checked), checked);
        assert_eq!(physical.dims, vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
        let mut wrong_rank = checked.clone();
        wrong_rank.dims.pop();
        assert_eq!(tensor_helper_operand_view(&physical, &wrong_rank), physical);
        let mut wrong_precision = checked.clone();
        wrong_precision.precision = Prim::F64;
        assert_eq!(
            tensor_helper_operand_view(&physical, &wrong_precision),
            physical
        );
        let unknown = TensorType {
            dims: vec![DimInfo::Named("runtime".into(), None)],
            precision: Prim::F32,
        };
        let claimed_unit = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        assert_eq!(tensor_helper_operand_view(&unknown, &claimed_unit), unknown);
    })
}

#[test]
fn operand_name_reservations_cover_unread_source_names_and_visible_bindings() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "__host_tensor_arg_1_1 = 11i64\ndef unused(__host_tensor_arg_1_0: i64) -> i64 = {\n __host_tensor_arg_1_2 = 12i64\n 0i64\n}\n",
    );
    let session = HostLoweringSession::new(checked.facts(), operation);
    let scope = HostScope::from([("__host_tensor_arg_1_3".into(), HostTypeTerm::Int64)]);
    let mut names = OperandNameSupply::default();
    assert_eq!(names.fresh(1, &session, &scope), "__host_tensor_arg_1_4");
    assert_eq!(names.fresh(1, &session, &scope), "__host_tensor_arg_1_5");
    assert_eq!(names.fresh(2, &session, &scope), "__host_tensor_arg_2_6");
    let other = surf_check("out = 0i64\n");
    let mut independent = OperandNameSupply::default();
    assert_eq!(
        independent.fresh(1, &HostLoweringSession::new(other.facts(), operation), &HostScope::new()),
        "__host_tensor_arg_1_0"
    ); })
}

#[test]
fn operand_name_reservations_include_metadata_carriers() {
    test_resources().run(|_| {
        let name = "__host_tensor_arg_1_0";
        let reference = deep_expr("(var {} __host_tensor_arg_1_0)");
        let metadata = Metadata::from(chelis_deep::annotations::MetadataValue::PropertySeed(
            chelis_deep::annotations::RuntimeExpression::try_new(reference)
                .expect("runtime name metadata"),
        ));
        let span = chelis_deep::Span::new(0, 0);
        for expr in [
            Expr::Map(metadata.clone(), span),
            Expr::MetaExpr(
                chelis_deep::ast::MetaExpr {
                    metadata: metadata.clone(),
                    expr: Box::new(deep_expr("(lit {} 1)")),
                },
                span,
            ),
            Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
                head: "future-form".into(),
                meta: metadata,
                children: Vec::new(),
                span,
            })),
        ] {
            let mut names = UnordSet::new();
            collect_operand_source_names(&expr, &mut names);
            assert!(names.contains(name));
            assert!(!names.contains("__host_tensor_arg_1_1"));
        }
    })
}

#[test]
fn operand_name_reservations_repeat_across_sessions_and_unwind() {
    test_resources().run(|operation| {
        let checked = surf_check("__host_tensor_arg_1_0 = 7i64\nout = 0i64\n");
        let prepared = PreparedHostFacts::new(checked.facts());
        let scope = HostScope::new();
        take_top_level_name_index_builds(operation);
        let allocate = |session: &HostLoweringSession<'_, chelis_syntax_host::NativeStack>| {
            let mut names = OperandNameSupply::default();
            (
                names.fresh(1, session, &scope),
                names.fresh(1, session, &scope),
            )
        };
        let expected = allocate(&prepared.session(operation));
        assert_eq!(
            expected,
            (
                "__host_tensor_arg_1_1".into(),
                "__host_tensor_arg_1_2".into()
            )
        );
        let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let session = prepared.session(operation);
            let mut names = OperandNameSupply::default();
            assert_eq!(names.fresh(1, &session, &scope), expected.0);
            panic!("interrupt an operand-preparation attempt");
        }));
        assert!(interrupted.is_err());
        assert_eq!(allocate(&prepared.session(operation)), expected);
        assert_eq!(allocate(&prepared.session(operation)), expected);
        assert_eq!(take_top_level_name_index_builds(operation), 1);
    })
}

#[test]
fn persistent_host_scope_preserves_order_shadowing_and_dimension_identity() {
    test_resources().run(|_| {
        let tensor = |names: &[&str]| {
            HostTypeTerm::Tensor(TensorType {
                dims: names
                    .iter()
                    .map(|name| DimInfo::Named((*name).into(), None))
                    .collect(),
                precision: Prim::F32,
            })
        };
        let mut scope = HostScope::from([
            ("b".into(), tensor(&["n"])),
            ("a".into(), tensor(&["n", "other"])),
            (
                LoadStoreName::top_level("global").as_str().into(),
                tensor(&["n", "global_only"]),
            ),
        ]);
        let snapshot = scope.clone();
        assert_eq!(scope.dimension_source("n").unwrap().0, "a");
        assert!(
            scope.has_dimension("global_only"),
            "private globals reserve identities"
        );
        assert!(
            scope.dimension_source("global_only").is_none(),
            "private global is not an input"
        );
        scope.insert("a".into(), HostTypeTerm::Int64);
        assert_eq!(scope.dimension_source("n").unwrap().0, "b");
        assert!(!scope.has_dimension("other"));
        assert_eq!(snapshot.dimension_source("n").unwrap().0, "a");
        assert!(snapshot.has_dimension("other"));
        scope.remove("b");
        assert!(scope.dimension_source("n").is_none());
        let normalized = "\0rank-axis:r:0".to_string();
        scope.insert("a".into(), tensor(&[&normalized]));
        scope.insert("z".into(), tensor(&["r[0]"]));
        assert_eq!(
            scope.dimension_source("r[0]").unwrap().0,
            "a",
            "source selection uses the old sorted-name then axis order across aliases"
        );
        scope.insert("wild".into(), tensor(&["*", ""]));
        for wildcard in ["*", ""] {
            assert!(
                scope.has_dimension(wildcard),
                "wildcards remain namespace reservations"
            );
            assert!(
                scope.dimension_source(wildcard).is_none(),
                "wildcards do not declare binders"
            );
        }
        assert!(!scope.has_dimension("absent"));
    })
}

#[test]
fn tensor_helper_scope_keeps_shape_only_binders_and_excludes_global_aliases() {
    test_resources().run(|operation| {
        let checked = surf_check("w = to_tensor([1.0f32])\n");
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, site) = session.def_named("w").expect("checked declaration");
        let scope = HostScope::from([
            (
                "n".to_string(),
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int64)),
            ),
            (
                "x".to_string(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Named("n".to_string(), None)],
                    precision: Prim::F32,
                }),
            ),
            (
                LoadStoreName::top_level("w").as_str().to_string(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(1)],
                    precision: Prim::F32,
                }),
            ),
        ]);
        let inputs = collect_tensor_scope(&session, &scope, Some(site));
        assert!(
            inputs.contains_key("n"),
            "a shape witness need not be a runtime read"
        );
        assert!(
            inputs.contains_key("x"),
            "a claim witness need not be a runtime read"
        );
        assert_eq!(inputs.len(), 2, "resolved global is not a lexical input");
    })
}

#[test]
fn kernel_builtin_load_rejection_requires_a_typed_lexical_input() {
    test_resources().run(|_| {
        for name in ["mean", "fold", "map"] {
            let mut dag = chelis_dag_core::Dag::new();
            let decl = dag.declare("test");
            dag.add_node(
                decl,
                RiscOp::Load { name: name.into() },
                vec![],
                TensorType {
                    dims: vec![],
                    precision: Prim::F32,
                },
                None,
            );
            let mut scope = HostScope::new();
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), Some(name.into()));
            scope.insert(
                "unrelated".into(),
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::F32)),
            );
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), Some(name.into()));
            scope.insert(name.into(), HostTypeTerm::String);
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), Some(name.into()));
            scope.insert(
                name.into(),
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::F32)),
            );
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), None);
        }
    })
}

fn parse_one_expr(source: &str) -> Expr {
    deep_expr(source)
}

#[test]
fn direct_builtin_expr_reader_covers_each_recursive_carrier() {
    test_resources().run(|_| {
        let span = chelis_deep::Span::new(0, 0);
        let call = parse_one_expr("(app {} (var {} abs) (lit {} 1))");
        let metadata = Metadata::from(chelis_deep::annotations::MetadataValue::PropertySeed(
            chelis_deep::annotations::RuntimeExpression::try_new(call.clone())
                .expect("a builtin call is valid runtime metadata"),
        ));
        let carriers = [
            Expr::BareList(vec![call.clone()], span),
            Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
                head: "future-form".to_string(),
                meta: Metadata::default(),
                children: vec![call.clone()],
                span,
            })),
            Expr::Map(metadata, span),
            Expr::MetaExpr(
                chelis_deep::ast::MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(call),
                },
                span,
            ),
        ];

        for carrier in carriers {
            assert_eq!(
                find_direct_builtin_call_in_expr(&carrier, &["abs"]),
                Some("abs".to_string())
            );
            assert_eq!(find_direct_builtin_call_in_expr(&carrier, &["sqrt"]), None);
        }
    })
}

#[test]
fn literal_result_claim_transfer_requires_a_pure_called_helper() {
    test_resources().run(|operation| {
        // A keyed draw is a pure function of its key (spec/05 §2.7), so its
        // helper transfers like any other pure helper; an `IO` helper is the
        // effecting case that keeps the direct lowering contract.
        let checked = surf_check(
            "def pure[n](x: tensor[n, f32]) -> tensor[2, f32] = \
             shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def keyed[n](k: key, x: tensor[n, f32]) -> tensor[2, f32] = \
             dropout(k, shrink(x, [[1i64, shape(x, 0i32)]]), 0.5f32)\n\
         def loud[n](x: tensor[n, f32]) -> tensor[2, f32] ! { IO } = {\n\
           _ = print(\"loud\")\n\
           shrink(x, [[1i64, shape(x, 0i32)]])\n\
         }\n\
         def caller[n](k: key, x: tensor[n, f32]) = \
             (pure(copy(x)), keyed(k, copy(x)), loud(x))\n",
        );
        let session = HostLoweringSession::new(checked.facts(), operation);
        assert!(top_level_fn_transfers_literal_result_claims(
            &session, "pure"
        ));
        assert!(top_level_fn_transfers_literal_result_claims(
            &session, "keyed"
        ));
        assert!(!top_level_fn_transfers_literal_result_claims(
            &session, "loud"
        ));
    })
}

#[test]
fn tuple_staging_preserves_nonbridge_host_payload_routes() {
    test_resources().run(|operation| {
        // A fixed tuple does not confer a scalar/tensor representation on a
        // nominal, container or callable leaf, even when that leaf is unused.
        // Check the real checked-signature staging decision, not a type-name
        // predicate in isolation. Other parameters participate in admission.
        let mut failures = Vec::new();
        for (params, body, expected) in [
            ("p: (tensor[1,f32],i64)", "p.0", true),
            ("p: ((tensor[1,f32],bool),i64)", "(p.0).0", true),
            ("p: (tensor[1,f32],Box)", "p.0", false),
            ("p: (tensor[1,f32],(i64,Box))", "p.0", false),
            ("p: (tensor[1,f32],List[i64])", "p.0", false),
            ("p: (tensor[1,f32],Option[i64])", "p.0", false),
            ("p: (tensor[1,f32],Dict[string,i64])", "p.0", false),
            ("p: (tensor[1,f32],string)", "p.0", false),
            ("p: (tensor[1,f32],(i64 -> i64))", "p.0", false),
            ("p: (tensor[1,f32],i64), other: Box", "p.0", false),
            ("p: (tensor[1,f32],i64), other: (Box,i64)", "p.0", false),
        ] {
            let checked = surf_check(&format!(
                "type Box = | Box {{ value: i64 }}\ndef first({params}) -> tensor[1,f32] = {body}\n"
            ));
            let session = HostLoweringSession::new(checked.facts(), operation);
            let (_, body) = session.def_named("first").unwrap();
            let signature = host_def_signature("first", body, None, &session)
                .expect("valid checked admission facts")
                .unwrap();
            let attempt = staged_def_kernel(&session, &signature).unwrap();
            let admitted = matches!(attempt, staged::StagingAttempt::Ready(_));
            if admitted != expected {
                failures.push(format!("{params}: staged={admitted}, expected={expected}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    })
}

#[test]
fn fixed_aggregate_result_claims_have_exactly_one_owner() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "def staged[n](p: (tensor[n,f32],tensor[*,f32])) -> tensor[n,f32] = p.1\n\
         def selected[n](p: (tensor[n,f32],tensor[*,f32]), choose: bool) -> tensor[n,f32] = if choose then p.1 else p.0\n",
    );
    let session = HostLoweringSession::new(checked.facts(), operation);
    for (name, expected) in [
        ("staged", vec![chelis_dag_core::dag::RtAxis::Lit(0)]),
        ("selected", Vec::new()),
    ] {
        let (_, body) = session.def_named(name).unwrap();
        let lowered = lower_host_function(name, body, None, &session, false)
            .unwrap()
            .unwrap();
        assert_eq!(
            lowered.function.helper_result_claim_axes, expected,
            "{name}: staged claims transfer to the helper; host-control claims remain on the result frame"
        );
    } })
}

#[test]
fn identity_tensor_helper_requires_its_checked_result_type() {
    test_resources().run(|_| {
        let scalar = TensorType::scalar_f32();
        let vector = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let mut dag = chelis_dag_core::Dag::new();
        let decl = dag.declare("test");
        let root = dag.add_node(
            decl,
            RiscOp::Load {
                name: "index".into(),
            },
            vec![],
            scalar.clone(),
            None,
        );
        dag.add_root(root);

        let scalar_scope = HostScope::from([(
            "index".to_string(),
            HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int64)),
        )]);
        assert!(identity_tensor_helper_matches_expected(
            &dag,
            &scalar,
            &scalar_scope
        ));
        assert!(!identity_tensor_helper_matches_expected(
            &dag,
            &vector,
            &scalar_scope
        ));
        assert!(identity_tensor_helper_matches_expected(
            &dag,
            &vector,
            &HostScope::new()
        ));
    })
}

#[test]
fn concat_admission_uses_static_named_extent_metadata() {
    test_resources()
        .run(|operation| { let body = parse_one_expr(
        "(app {} (var {} concat) (app {} (var {} Cons) (var {} x) (app {} (var {} Cons) (var {} x) (var {} Nil))) (lit {} 1))",
    );
    let defs = BTreeMap::new();
    for (width, host) in [
        (DimInfo::Lit(2), false),
        (DimInfo::Named("width".to_string(), Some(2)), false),
        (DimInfo::Named("width".to_string(), None), true),
    ] {
        let tensor = TensorType {
            dims: vec![DimInfo::Lit(2), width],
            precision: Prim::F32,
        };
        let mut walk = UncarriableWalk {
            defs: &defs,
            declaration_type: &no_declaration_type,
            incoming_scope: None,
            operand_capture: None,
            active: UnordSet::new(),
            completed: UnordMap::new(),
            cycle_cutoff: false,
            def_visits: 0,
            admit_dropout: false,
            scopes: vec![UnordMap::from([(
                "x".to_string(),
                AdmissionBinding {
                    tuple: None,
                    callable: CallableFacts::UNKNOWN,
                    constructor: false,
                    concat: ConcatInputFact::Tensor(tensor.clone()),
                },
            )])],
        };
        assert_eq!(walk.expr(&body, operation).expect("valid admission metadata").is_some(), host, "{tensor:?}");
        if !host {
            crate::lower::try_lower_subexpr_program(
                &body,
                UnordMap::from([("x".to_string(), tensor)]),
                UnordMap::new(),
                UnordMap::new(),
             operation)
            .expect("the same known extent is representable by static concat");
        }
    } })
}

#[test]
fn concat_admission_uses_checked_copy_and_add_producers() {
    test_resources().run(|operation| {
        for (width, producer, host) in [
            ("2", "copy(x)", false),
            ("2", "add(x, x)", false),
            ("2", "mul(x, x)", false),
            ("s", "copy(x)", true),
            ("s", "add(x, x)", true),
            ("s", "mul(x, x)", true),
        ] {
            let source = format!(
                "def join[s](x: tensor[s, {width}, f32]) = concat([{producer}, {producer}], 1i32)\n"
            );
            let checked = surf_check(&source);
            let session = HostLoweringSession::new(checked.facts(), operation);
            let defs = cached_program_defs(&session);
            let (_, _, fn_kids) = stamped_parts(&defs["join"]).unwrap();
            let (_, _, binders) = stamped_parts(&fn_kids[0]).unwrap();
            let params = binders
                .iter()
                .map(|binder| HostParam {
                    name: param_name(binder).unwrap(),
                    ty: param_host_type(binder, operation)
                        .expect("valid checked metadata")
                        .unwrap(),
                })
                .collect::<Vec<_>>();
            assert_eq!(
                body_form_the_dag_cannot_carry(&session, &fn_kids[1], &params, false)
                    .expect("valid admission metadata")
                    .is_some(),
                host,
                "{source}"
            );
        }
    })
}

#[test]
fn concat_admission_does_not_treat_shadowed_callee_as_builtin_identity() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "def run[s](x: tensor[s, s, f32]) = {\n z = mul(x, x)\n concat([z, z], 1i32)\n}\n",
        );
        let session = HostLoweringSession::new(checked.facts(), operation);
        let defs = cached_program_defs(&session);
        let (_, _, run) = stamped_parts(&defs["run"]).unwrap();
        let (_, _, body) = stamped_parts(&run[1]).unwrap();
        let (_, _, bindings) = stamped_parts(&body[0]).unwrap();
        let producer = &bindings[1];
        let tensor = TensorType {
            dims: vec![DimInfo::Named("s".to_string(), None); 2],
            precision: Prim::F32,
        };
        let scope = UnordMap::from([(
            "x".to_string(),
            AdmissionBinding {
                tuple: None,
                callable: CallableFacts::UNKNOWN,
                constructor: false,
                concat: ConcatInputFact::Tensor(tensor),
            },
        )]);
        let mut walk = UncarriableWalk {
            defs: &defs,
            declaration_type: &no_declaration_type,
            incoming_scope: None,
            operand_capture: None,
            active: UnordSet::new(),
            completed: UnordMap::new(),
            cycle_cutoff: false,
            def_visits: 0,
            admit_dropout: false,
            scopes: vec![scope],
        };
        assert!(matches!(
            walk.concat_input_fact(producer, operation)
                .expect("valid admission metadata"),
            ConcatInputFact::Tensor(_)
        ));
        walk.scopes
            .last_mut()
            .unwrap()
            .insert("mul".to_string(), AdmissionBinding::default());
        assert!(matches!(
            walk.concat_input_fact(producer, operation)
                .expect("valid admission metadata"),
            ConcatInputFact::Unknown
        ));
    })
}

#[test]
fn concat_admission_does_not_project_destructuring_aliases() {
    test_resources()
        .run(|operation| { // Private structural control: the walk has no tuple/list projection
    // proof. A nested binder must not inherit its RHS's whole List fact.
    let body = parse_one_expr(
        "(let {} (bind {} (tuple {} (var {} a) (var {} b)) (var {} xs)) (app {} (var {} concat) (var {} b) (lit {} 1)))",
    );
    let defs = BTreeMap::new();
    let mut walk = UncarriableWalk {
        defs: &defs,
        declaration_type: &no_declaration_type,
        incoming_scope: None,
        operand_capture: None,
        active: UnordSet::new(),
        completed: UnordMap::new(),
        cycle_cutoff: false,
        def_visits: 0,
        admit_dropout: false,
        scopes: vec![UnordMap::from([(
            "xs".to_string(),
            AdmissionBinding {
                tuple: None,
                callable: CallableFacts::UNKNOWN,
                constructor: false,
                concat: ConcatInputFact::List(vec![ConcatInputFact::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2), DimInfo::Named("width".to_string(), None)],
                    precision: Prim::F32,
                })]),
            },
        )])],
    };
    assert!(walk.expr(&body, operation).expect("valid admission metadata").is_none()); })
}

#[test]
fn concat_admission_captures_actuals_before_formal_shadowing() {
    test_resources()
        .run(|operation| { // Admission-only: the older tensor lowerer has a separate x/y
    // substitution collision for this source. This test does not claim
    // kernel execution; the fresh-formal integration twin covers that.
    for (x_width, y_width, host) in [("*", "2", true), ("2", "*", false)] {
        let source = format!(
            "def second[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([y, y], 1i32)\ndef run[s](x: tensor[s, {x_width}, f32], y: tensor[s, {y_width}, f32]) -> tensor[s, *, f32] = softmax(second(y, x), -1)\n"
        );
        let program = surf_check(&source);
        let defs = cached_program_defs(&HostLoweringSession::new(program.facts(), operation));
        let (_, _, run) = stamped_parts(&defs["run"]).unwrap();
        let (_, _, params) = stamped_parts(&run[0]).unwrap();
        let params: Vec<_> = params
            .iter()
            .map(|param| HostParam {
                name: param_name(param).unwrap(),
                ty: param_host_type(param, operation).expect("valid checked metadata").unwrap(),
            })
            .collect();
        assert_eq!(
            body_form_the_dag_cannot_carry(
                &HostLoweringSession::new(program.facts(), operation),
                &run[1],
                &params,
                false
            ).expect("valid admission metadata")
            .is_some(),
            host,
            "{source}"
        );
    } })
}

fn no_declaration_type(_: &str) -> Result<Option<HostTypeTerm>, crate::lower::LowerDiagnostic> {
    Ok(None)
}

fn admission_test_walk(defs: &BTreeMap<String, Expr>) -> UncarriableWalk<'_> {
    UncarriableWalk {
        defs,
        declaration_type: &no_declaration_type,
        incoming_scope: None,
        operand_capture: None,
        active: UnordSet::new(),
        completed: UnordMap::new(),
        cycle_cutoff: false,
        def_visits: 0,
        admit_dropout: false,
        scopes: Vec::new(),
    }
}

#[test]
fn captured_load_admission_requires_checked_noncallability() {
    test_resources().run(|operation| {
        let body =
            parse_one_expr("(block {} (app {} (var {} print) (lit {} \"initialize\")) (lit {} 1))");
        let defs = BTreeMap::from([("captured".to_owned(), body)]);
        let mut walk = admission_test_walk(&defs);
        // An unresolved reference has no proof that its value is a numeric
        // capture; syntax alone cannot turn it into a Load boundary.
        let reference = parse_one_expr("(var {} captured)");
        assert!(
            walk.expr(&reference, operation)
                .expect("valid admission metadata")
                .is_some()
        );
        let callable = parse_one_expr("(var {type: (t-fn {} (t-prim {} i32))} captured)");
        assert!(
            walk.expr(&callable, operation)
                .expect("valid admission metadata")
                .is_some()
        );
        let noncallable = parse_one_expr("(var {type: (t-prim {} i32)} captured)");
        assert!(
            walk.expr(&noncallable, operation)
                .expect("valid admission metadata")
                .is_some(),
            "a reference stamp cannot supply missing checked declaration authority"
        );
    })
}

#[test]
fn captured_load_admission_uses_checked_declaration_for_unstamped_reference() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "captured: tensor[2, f32] = { _ = print(\"initialize\")\n to_tensor([1.0f32,2.0f32]) }\n",
    );
    let session = HostLoweringSession::new(program.facts(), operation);
    let defs = cached_program_defs(&session);
    assert!(
        expr_type(&defs["captured"], operation).expect("valid checked metadata").is_none(),
        "Let carries no ABI stamp"
    );
    assert!(matches!(
        lookup_declared_host_type(&session, "captured").expect("valid checked metadata"),
        Some(HostTypeTerm::Tensor(_))
    ));
    let reference = parse_one_expr("(var {} captured)");
    assert!(
        expr_type(&reference, operation).expect("valid checked metadata").is_none(),
        "ordinary Var nodes have no ABI stamp"
    );
    assert!(
        body_form_the_dag_cannot_carry(&session, &reference, &[], false).expect("valid admission metadata").is_none(),
        "checked declaration values form a Load boundary"
    ); })
}

#[test]
fn captured_load_admission_rejects_contradictory_declaration_reference_evidence() {
    test_resources()
        .run(|operation| { let program = surf_check("number = to_tensor([1.0f32,2.0f32])\ncaptured = is_nan\n");
    let session = HostLoweringSession::new(program.facts(), operation);
    let declared = lookup_declared_type_expr(&session, "number").unwrap();
    let metadata = Metadata::from(chelis_deep::annotations::MetadataValue::Type(
        chelis_deep::annotations::TypeSyntax::try_new(declared).unwrap(),
    ));
    let reference = Expr::node(
        DeepTag::Var,
        metadata,
        vec![Expr::Atom(
            Atom::Name("captured".into()),
            chelis_deep::Span::new(0, 0),
        )],
        chelis_deep::Span::new(0, 0),
    );
    let declaration = lookup_declared_host_type(&session, "captured").expect("valid checked metadata");
    let reference_type = expr_type(&reference, operation).expect("valid checked metadata");
    let defs = cached_program_defs(&session);
    assert!(
        body_form_the_dag_cannot_carry(&session, &reference, &[], false).expect("valid admission metadata").is_some(),
        "a callable declaration cannot be admitted as a noncallable capture: declaration={declaration:?}, reference={reference_type:?}, body={:?}",
        stamped_parts(&defs["captured"]).map(|(tag, _, _)| tag)
    ); })
}

#[test]
fn captured_load_admission_retains_checked_callable_initializers() {
    test_resources().run(|operation| {
        let program = surf_check("captured = if true then is_nan else is_finite\n");
        let session = HostLoweringSession::new(program.facts(), operation);
        let defs = cached_program_defs(&session);
        assert_eq!(
            CallableFacts::checked_type(
                expr_type(&defs["captured"], operation)
                    .expect("valid checked metadata")
                    .as_ref()
            ),
            CallabilityEvidence::Callable
        );
        let reference = parse_one_expr("(var {} captured)");
        let mut walk = admission_test_walk(&defs);
        assert!(
            walk.expr(&reference, operation)
                .expect("valid admission metadata")
                .is_some(),
            "a checked callable initializer still requires admission"
        );
    })
}

#[test]
fn callable_admission_required_nullary_body_retains_scope_and_cache() {
    test_resources().run(|operation| {
        let defs = BTreeMap::from([
            (
                "predicate".to_string(),
                parse_one_expr("(fn {} (params {}) (var {} is_nan))"),
            ),
            (
                "number".to_string(),
                parse_one_expr("(fn {} (params {}) (lit {} 7))"),
            ),
        ]);
        let mut walk = admission_test_walk(&defs);
        walk.scopes.push(UnordMap::from_iter([(
            "is_nan".to_string(),
            AdmissionBinding {
                tuple: None,
                callable: CallableFacts::NONCALLABLE,
                ..AdmissionBinding::default()
            },
        )]));
        for _ in 0..2 {
            let predicate = walk
                .analyze_def("predicate", &defs["predicate"], None, operation)
                .expect("valid admission metadata");
            assert!(predicate.reason.is_none());
            assert_eq!(predicate.callable, CallableFacts::STATIC);
            let number = walk
                .analyze_def("number", &defs["number"], None, operation)
                .expect("valid admission metadata");
            assert!(number.reason.is_none());
            assert_eq!(number.callable, CallableFacts::NONCALLABLE);
            assert_eq!(
                walk.scopes[0]["is_nan"].callable,
                CallableFacts::NONCALLABLE
            );
            assert!(walk.active.is_empty());
        }
        assert_eq!(
            walk.def_visits, 2,
            "each complete nullary body is analyzed once"
        );
        assert_eq!(walk.completed["predicate"].len(), 1);
        assert_eq!(walk.completed["number"].len(), 1);
        assert!(!walk.cycle_cutoff);
    })
}

#[test]
fn callable_admission_preserves_repeated_closed_relation_branches() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "alias = is_finite\ndef selected(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = {\n f = if flag then alias else is_finite\n f(x)\n}\n",
    );
    let session = HostLoweringSession::new(program.facts(), operation);
    let defs = cached_program_defs(&session);
    let signature = host_def_signature("selected", &defs["selected"], None, &session).expect("valid checked admission facts").unwrap();
    let (_, _, body) = stamped_parts(&signature.body_expr).unwrap();
    let (_, _, bindings) = stamped_parts(&body[0]).unwrap();
    let mut walk = admission_test_walk(&defs);
    let choice = &bindings[1];
    assert_eq!(
        walk.callable_admission(choice, operation).expect("valid admission metadata"),
        CallableFacts::RUNTIME,
        "choice: {choice:#?}\ndefs: {defs:#?}"
    ); })
}

#[test]
fn callable_admission_requires_consistent_checked_control_metadata() {
    test_resources()
        .run(|operation| { let defs = BTreeMap::new();
    let mut walk = admission_test_walk(&defs);
    for (source, expected) in [
        (
            "(if {} (lit {} true) (var {} is_nan) (var {} is_finite))",
            CallableFacts::UNKNOWN,
        ),
        (
            "(if {type: (t-fn {} (t-prim {} f32) (t-prim {} bool))} (lit {type: (t-prim {} bool)} true) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0))",
            CallableFacts::INCONSISTENT,
        ),
    ] {
        let expression = parse_one_expr(source);
        assert_eq!(walk.callable_admission(&expression, operation).expect("valid admission metadata"), expected, "{source}");
        assert_eq!(
            walk.expr(&expression, operation).expect("valid admission metadata").is_some(),
            expected == CallableFacts::INCONSISTENT,
            "inconsistent checked evidence must retain a host refusal: {source}"
        );
    } })
}

#[test]
fn callable_admission_static_tensor_literals_keep_existing_dag_owner() {
    test_resources().run(|operation| {
        for (source, expected) in [
            (
                "def run() -> tensor[2, f64] = to_tensor([-0.0f64, 1.0000000000000002f64])\n",
                true,
            ),
            (
                "def run(x: List[f64]) -> tensor[*, f64] = to_tensor(x)\n",
                false,
            ),
        ] {
            let program = surf_check(source);
            let session = HostLoweringSession::new(program.facts(), operation);
            let defs = cached_program_defs(&session);
            assert_eq!(
                crate::lower::top_level_lowering_map(
                    program.exprs(),
                    program.type_env(),
                    operation
                )
                .expect("checked fixture classification")["run"],
                expected,
                "{source}"
            );
            let mut walk = admission_test_walk(&defs);
            assert_eq!(
                walk.def("run", &defs["run"], None, operation)
                    .expect("valid admission metadata")
                    .is_none(),
                expected,
                "{source}"
            );
        }
    })
}

#[test]
fn static_product_admission_keeps_numeric_projection_and_host_callable_owner() {
    test_resources()
        .run(|operation| { for (source, numeric) in [
        (
            "def run(x: tensor[2,f32]) -> tensor[2,f32] = { pair = (x, neg(x))\n (first, second) = copy(pair)\n first }\n",
            true,
        ),
        (
            "def run(x: tensor[*,f64]) -> tensor[*,bool] = { f: &tensor[2,f64] -> tensor[*,bool] = is_finite\n (g,n) = (copy(f),1i64)\n g(x) }\n",
            false,
        ),
        (
            "def run(x: tensor[3,f64]) -> tensor[3,f64] = { f: &tensor[2,f64] -> tensor[*,bool] = is_finite\n (g,n) = (copy(f),1i64)\n neg(x) }\n",
            true,
        ),
    ] {
        let program = surf_check(source);
        let map = crate::lower::top_level_lowering_map(program.exprs(), program.type_env(), operation).expect("checked fixture classification");
        assert_eq!(map.get("run"), Some(&numeric), "{source}: {map:?}");
        let session = HostLoweringSession::new(program.facts(), operation);
        let kernel =
            host_def_kernel(&session, "run").expect("the selected lane lowers without retry");
        assert_eq!(kernel.is_some(), numeric, "{source}");
    } })
}

#[test]
fn callable_admission_host_realizability_preserves_numeric_ir_owners() {
    test_resources()
        .run(|operation| { for (source, expected) in [
        (
            "def run(x: tensor[rows, f32], y: tensor[cols, f32]) -> tensor[rows, *, f32] = insert(x, 1i32, shape(y, 0i32))\n",
            true,
        ),
        (
            "def run(x: tensor[f32]) -> tensor[f32] = scalar_to_tensor(tensor_to_scalar(x))\n",
            true,
        ),
        (
            "def run(x: tensor[2, f32], y: tensor[3, f32]) -> tensor[5, f32] = concat([x, y], 0i32)\n",
            true,
        ),
        (
            "def run(x: tensor[2, f32]) -> tensor[2, f32] = { y = neg(x)\n _ = drop(x)\n y }\n",
            true,
        ),
        ("def run(x: List[f32]) -> i64 = len(x)\n", false),
    ] {
        let program = surf_check(source);
        let session = HostLoweringSession::new(program.facts(), operation);
        let defs = cached_program_defs(&session);
        let mut walk = admission_test_walk(&defs);
        assert_eq!(
            walk.def("run", &defs["run"], None, operation).expect("valid admission metadata").is_none(),
            expected,
            "{source}"
        );
    } })
}

#[test]
fn callable_admission_host_only_top_level_numeric_projection_refuses_invocation() {
    test_resources()
        .run(|operation| { for source in [
        "def run(x: List[f32]) -> i64 = { f = len\n g = copy(copy(f))\n g(x) }\n",
        "def run(x: List[f32]) -> f32 = { f = index\n g = copy(copy(f))\n g(x, 0i64) }\n",
        "def keep(h: (&List[f32] -> i64), x: List[f32]) -> i64 = len(x)\ndef run(x: List[f32]) -> i64 = keep(len, x)\n",
    ] {
        let program = surf_check(source);
        let session = HostLoweringSession::new(program.facts(), operation);
        let defs = cached_program_defs(&session);
        assert!(
            !crate::lower::expr_is_dag_lowerable(&defs["run"], program.facts(), operation).expect("checked fixture admission"),
            "{source}"
        );
    } })
}

#[test]
fn callable_admission_host_only_top_level_projection_keeps_unused_and_shadowed_values() {
    test_resources()
        .run(|operation| { for body in [
        "{ f = len\n g = copy(f)\n add(x, x) }",
        "{ f = neg\n g = copy(copy(f))\n g(x) }",
        "{ len = neg\n g = copy(len)\n g(x) }",
    ] {
        let program = surf_check(&format!(
            "def run(x: tensor[2, f32]) -> tensor[2, f32] = {body}\n"
        ));
        let session = HostLoweringSession::new(program.facts(), operation);
        let defs = cached_program_defs(&session);
        assert!(
            crate::lower::expr_is_dag_lowerable(&defs["run"], program.facts(), operation).expect("checked fixture admission"),
            "{body}"
        );
    }
    let program = surf_check(
        "def keep(h: (&List[f32] -> i64), x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\ndef run(x: tensor[2, f32]) -> tensor[2, f32] = keep(len, x)\n",
    );
    let session = HostLoweringSession::new(program.facts(), operation);
    let defs = cached_program_defs(&session);
    assert!(crate::lower::expr_is_dag_lowerable(&defs["run"], program.facts(), operation).expect("checked fixture admission")); })
}

#[test]
fn callable_admission_host_only_copy_invocation_retains_execution_lane() {
    test_resources().run(|operation| {
        for builtin_operation in ["len", "index"] {
            let source = if builtin_operation == "len" {
                "def run(x: List[f32]) -> i64 = { f = len\n g = copy(copy(f))\n g(x) }\n"
            } else {
                "def run(x: List[f32]) -> f32 = { f = index\n g = copy(copy(f))\n g(x, 0i64) }\n"
            };
            let program = surf_check(source);
            let session = HostLoweringSession::new(program.facts(), operation);
            let defs = cached_program_defs(&session);
            let mut walk = admission_test_walk(&defs);
            assert!(
                walk.def("run", &defs["run"], None, operation)
                    .expect("valid admission metadata")
                    .is_some(),
                "a copied {builtin_operation} invocation requires its host implementation"
            );
        }
    })
}

#[test]
fn callable_admission_host_only_joins_never_manufacture_numeric_templates() {
    test_resources()
        .run(|operation| { for choice in [
        "if flag then len else len",
        "if flag then len else (fn (v: List[f32]) -> 7i64)",
    ] {
        let program = surf_check(&format!(
            "def run(flag: bool, x: List[f32]) -> i64 = {{ f = {choice}\n g = copy(f)\n g(x) }}\n"
        ));
        let session = HostLoweringSession::new(program.facts(), operation);
        let defs = cached_program_defs(&session);
        let mut walk = admission_test_walk(&defs);
        assert!(walk.def("run", &defs["run"], None, operation).expect("valid admission metadata").is_some(), "{choice}");
    } })
}

#[test]
fn callable_admission_host_only_unused_values_do_not_exclude_numeric_kernel() {
    test_resources().run(|operation| {
        for body in [
            "{ f = len\n g = copy(f)\n add(x, x) }",
            "{ f = neg\n g = copy(copy(f))\n g(x) }",
        ] {
            let program = surf_check(&format!(
                "def run(x: tensor[2, f32]) -> tensor[2, f32] = {body}\n"
            ));
            let session = HostLoweringSession::new(program.facts(), operation);
            let defs = cached_program_defs(&session);
            let mut walk = admission_test_walk(&defs);
            assert!(
                walk.def("run", &defs["run"], None, operation)
                    .expect("valid admission metadata")
                    .is_none(),
                "{body}"
            );
        }
        let defs = BTreeMap::new();
        let mut walk = admission_test_walk(&defs);
        walk.scopes.push(UnordMap::from_iter([(
            "len".into(),
            AdmissionBinding {
                tuple: None,
                callable: CallableFacts::NONCALLABLE,
                ..AdmissionBinding::default()
            },
        )]));
        let shadowed = walk
            .analyze_expr(&parse_one_expr("(var {} len)"), operation)
            .expect("valid admission metadata");
        assert!(shadowed.reason.is_none());
        assert_eq!(shadowed.callable, CallableFacts::NONCALLABLE);
    })
}

#[test]
fn callable_admission_retains_alias_and_copy_dispositions() {
    test_resources().run(|operation| {
        let defs = BTreeMap::from([("global".to_string(), parse_one_expr("(var {} is_nan)"))]);
        let mut walk = admission_test_walk(&defs);
        walk.scopes.push(UnordMap::from([
            (
                "f".to_string(),
                AdmissionBinding {
                    tuple: None,
                    callable: CallableFacts::RUNTIME,
                    ..AdmissionBinding::default()
                },
            ),
            (
                "is_nan".to_string(),
                AdmissionBinding {
                    tuple: None,
                    callable: CallableFacts::NONCALLABLE,
                    ..AdmissionBinding::default()
                },
            ),
        ]));
        for source in ["(var {} f)", "(copy {} (var {} f))"] {
            assert_eq!(
                walk.callable_admission(&parse_one_expr(source), operation)
                    .expect("valid admission metadata"),
                CallableFacts::RUNTIME
            );
        }
        assert_eq!(
            walk.callable_admission(&parse_one_expr("(var {} is_nan)"), operation)
                .expect("valid admission metadata"),
            CallableFacts::NONCALLABLE
        );
        assert_eq!(
            walk.callable_admission(&parse_one_expr("(var {} global)"), operation)
                .expect("valid admission metadata"),
            CallableFacts::STATIC
        );
    })
}

#[test]
fn callable_admission_composes_sparse_lexical_tail_evidence() {
    test_resources()
        .run(|operation| { let defs = BTreeMap::new();
    let mut walk = admission_test_walk(&defs);
    for source in [
        "(let {} (bind {} local (var {} is_nan)) (var {} local))",
        "(block {} (lit {} true) (let {} (bind {} local (var {} is_nan)) (copy {} (copy {} (var {} local)))))",
        "(block {} (grad {} (fn {} (params {} x) (var {} x))))",
        "(block {} (vmap {} (fn {} (params {} x) (var {} x))))",
    ] {
        // Callable evidence composes through the tail; a block result is
        // a selected value the numeric resolver cannot specialize.
        assert_eq!(
            walk.callable_admission(&parse_one_expr(source), operation).expect("valid admission metadata"),
            CallableFacts::HOST_PROJECTION,
            "{source}"
        );
    }
    assert_eq!(
        walk.callable_admission(&parse_one_expr(
            "(copy {} (copy {} (vmap {} (fn {} (params {} x) (var {} x)))))"
        ), operation).expect("valid admission metadata"),
        CallableFacts::STATIC,
        "aliases, copies and transforms keep the static template"
    ); })
}

#[test]
fn callable_admission_declaration_formals_take_each_actuals_facts() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "def use(h: (&tensor[2,f64] -> tensor[2,bool]), x: tensor[2,f64]) -> tensor[2,bool] = h(x)\n",
    );
    let defs = cached_program_defs(&HostLoweringSession::new(program.facts(), operation));
    let mut walk = admission_test_walk(&defs);
    let call = |actual: &str| {
        parse_one_expr(&format!("(app {{}} (var {{}} use) {actual} (var {{}} x))"))
    };
    walk.scopes.push(UnordMap::from([(
        "x".to_string(),
        AdmissionBinding::default(),
    )]));
    let alias = call("(var {} is_finite)");
    let projected = call("(tuple-get {} (tuple {} (var {} is_finite) (lit {} 1)) (lit {} 0))");
    let block = call("(let {} (bind {} h (var {} is_finite)) (var {} h))");
    assert!(
        walk.expr(&alias, operation).expect("valid admission metadata").is_none(),
        "a static template stays numeric"
    );
    assert!(
        walk.expr(&projected, operation).expect("valid admission metadata").is_some(),
        "a projected actual keeps host invocation"
    );
    assert!(
        walk.expr(&block, operation).expect("valid admission metadata").is_some(),
        "a block-result actual keeps host invocation"
    );
    assert!(
        walk.expr(&alias, operation).expect("valid admission metadata").is_none(),
        "distinct actual facts never share a completed declaration context"
    );
    assert_eq!(walk.completed["use"].len(), 2); })
}

#[test]
fn callable_admission_unknown_declarations_and_cycles_prove_nothing() {
    test_resources().run(|operation| {
        let defs = BTreeMap::from([
            ("left".to_string(), parse_one_expr("(var {} right)")),
            ("right".to_string(), parse_one_expr("(var {} left)")),
        ]);
        let mut walk = admission_test_walk(&defs);
        for source in ["(var {} missing)", "(var {} left)"] {
            assert_eq!(
                walk.callable_admission(&parse_one_expr(source), operation)
                    .expect("valid admission metadata"),
                CallableFacts::UNKNOWN,
                "{source}"
            );
        }
    })
}

#[test]
fn concat_admission_shared_context_visits_each_helper_once() {
    test_resources()
        .run(|operation| { // Sharing must not expand into every call path. Count body visits,
    // not wall time; the trailing rank must still choose Host.
    let depth = 12;
    let mut source = "def level_0(x: tensor[2, f32]) -> tensor[2, f32] = x\n".to_string();
    for level in 1..=depth {
        source.push_str(&format!(
            "def level_{level}(x: tensor[2, f32]) -> tensor[2, f32] = add(level_{}(copy(x)), level_{}(copy(x)))\n",
            level - 1, level - 1
        ));
    }
    source.push_str(&format!(
        "def run(x: tensor[2, f32]) -> tensor[2, f32] = {{\n z = level_{depth}(copy(x))\n _ = rank(x)\n z\n}}\n"
    ));
    let program = surf_check(&source);
    let defs = cached_program_defs(&HostLoweringSession::new(program.facts(), operation));
    let mut walk = admission_test_walk(&defs);
    assert!(
        walk.def("run", &defs["run"], None, operation).expect("valid admission metadata")
            .unwrap()
            .contains("rank")
    );
    assert_eq!(walk.def_visits, depth + 2); })
}

#[test]
fn concat_admission_completed_context_retains_order_and_exact_facts() {
    test_resources().run(|operation| {
        let program = surf_check(
            "def second[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([y, y], 1i32)\n",
        );
        let defs = cached_program_defs(&HostLoweringSession::new(program.facts(), operation));
        let fixed = ConcatInputFact::Tensor(TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
            precision: Prim::F32,
        });
        let dynamic = ConcatInputFact::Tensor(TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Named("width".to_string(), None)],
            precision: Prim::F32,
        });
        for order in [[false, true, false, true], [true, false, true, false]] {
            let mut walk = admission_test_walk(&defs);
            for host in order {
                let actuals = if host {
                    vec![fixed.clone(), dynamic.clone()]
                } else {
                    vec![dynamic.clone(), fixed.clone()]
                };
                assert_eq!(
                    walk.def("second", &defs["second"], Some(actuals), operation)
                        .expect("valid admission metadata")
                        .is_some(),
                    host
                );
            }
            assert_eq!(
                walk.def_visits, 2,
                "two distinct ordered contexts, not four calls"
            );
        }
    })
}

#[test]
fn concat_admission_completed_callee_does_not_capture_caller_bindings() {
    test_resources()
        .run(|operation| { let program = surf_check(
        "def global(x: tensor[2, f32]) -> i32 = rank(x)\ndef call(x: tensor[2, f32]) -> i32 = global(x)\n",
    );
    let defs = cached_program_defs(&HostLoweringSession::new(program.facts(), operation));
    let mut walk = admission_test_walk(&defs);
    // A free callee name resolves against program definitions, not a
    // same-spelled caller alias, including its constructor classification.
    for constructor in [false, true] {
        walk.scopes = vec![UnordMap::from([(
            "global".to_string(),
            AdmissionBinding {
                tuple: None,
                callable: CallableFacts::UNKNOWN,
                constructor,
                concat: ConcatInputFact::Unknown,
            },
        )])];
        assert!(
            walk.def("call", &defs["call"], None, operation).expect("valid admission metadata")
                .unwrap()
                .contains("rank")
        );
        assert_eq!(walk.bound("global", operation), Some(constructor));
    }
    assert_eq!(walk.def_visits, 2, "call and global each visited once"); })
}

#[test]
fn concat_admission_cycle_cutoff_is_not_a_completed_negative() {
    test_resources()
        .run(|operation| { // Private walk control: b reached under active a cannot conclude that
    // a lacks Host forms. A later fresh b must still see a's trailing rank.
    let defs = BTreeMap::from([
        (
            "a".to_string(),
            parse_one_expr(
                "(fn {} (params {}) (tuple {} (app {} (var {} b)) (app {} (var {} rank) (lit {} 1))))",
            ),
        ),
        (
            "b".to_string(),
            parse_one_expr("(fn {} (params {}) (app {} (var {} a)))"),
        ),
    ]);
    let mut walk = admission_test_walk(&defs);
    for name in ["a", "b", "a", "b"] {
        assert!(
            walk.def(name, &defs[name], Some(Vec::new()), operation).expect("valid admission metadata")
                .unwrap()
                .contains("rank")
        );
        assert!(walk.active.is_empty(), "active guard must unwind");
    } })
}

#[test]
fn linked_alias_resolution_accepts_one_terminal_match() {
    test_resources().run(|_| {
        let mut registry = AdtRegistry::new();
        registry.register_alias(
            "PairAlias".to_string(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)]),
        );

        let resolved = resolve_host_type_alias(
            &registry,
            "Pkg__issue__1293__recursive__cotangents__Demo__Main__PairAlias",
        )
        .expect("a qualified linked reference must find its unique transparent alias");
        assert!(matches!(resolved.body, Type::Tuple(_)));
    })
}

#[test]
fn linked_alias_resolution_refuses_an_ambiguous_terminal_match() {
    test_resources().run(|_| {
        let mut registry = AdtRegistry::new();
        for name in ["Adep__PairAlias", "Blib__PairAlias"] {
            registry.register_alias(
                name.to_string(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Type::tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)]),
            );
        }

        assert!(
            resolve_host_type_alias(&registry, "Pkg__PairAlias").is_none(),
            "a non-exact terminal spelling must not silently select one colliding alias"
        );
        assert!(
            resolve_host_type_alias(&registry, "Adep__PairAlias").is_some(),
            "an exact linked alias identity remains authoritative"
        );
    })
}

#[test]
fn linked_alias_resolution_preserves_an_exact_adt_identity() {
    test_resources().run(|_| {
        let mut registry = AdtRegistry::new();
        registry.defs.insert(
            "Pkg__collidelib__Demo__Main__Wrapped".to_string(),
            AdtDef {
                name: "Pkg__collidelib__Demo__Main__Wrapped".to_string(),
                type_params: Vec::new(),
                param_kinds: Vec::new(),
                param_vars: Vec::new(),
                param_args: Vec::new(),
                variants: Vec::new(),
                opaque: false,
                defining_module: None,
            },
        );
        registry.register_alias(
            "Pkg__collidedep__Demo__Main__Wrapped".to_string(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)]),
        );

        assert!(
            resolve_host_type_alias(&registry, "Pkg__collidelib__Demo__Main__Wrapped").is_none(),
            "an exact ADT identity must not be reinterpreted as another package's alias"
        );
    })
}

#[test]
fn issue_662_recursive_def_cycle_does_not_hide_a_later_forward_fail() {
    test_resources().run(|_| {
        let defs = BTreeMap::from([
            (
                "a".to_string(),
                parse_one_expr("(tuple {} (var {} b) (var {} fail))"),
            ),
            ("b".to_string(), parse_one_expr("(var {} a)")),
        ]);
        assert!(
            expr_reaches_forward_fail(&parse_one_expr("(var {} a)"), &defs, &mut UnordSet::new()),
            "breaking the a -> b -> a cycle must continue with a's fail sibling"
        );
    })
}

#[test]
fn issue_662_recursive_def_cycle_without_fail_terminates_negative() {
    test_resources().run(|_| {
        let defs = BTreeMap::from([
            ("a".to_string(), parse_one_expr("(var {} b)")),
            ("b".to_string(), parse_one_expr("(var {} a)")),
        ]);
        assert!(
            !expr_reaches_forward_fail(&parse_one_expr("(var {} a)"), &defs, &mut UnordSet::new()),
            "a fail-free recursive cycle must terminate without inventing reachability"
        );
    })
}

// chelis#1087's two `substitute_var` pass-through locks went with the
// function, which chelis#1923 deleted along with this lane's own pipe
// beta-reduction: the checker now folds a pipe into the application
// spec/02-surf-syntax.md §0.1 says it denotes, so no lane substitutes a
// stage parameter any more. The property they pinned, that a
// transitional variant is passed through unchanged rather than rebuilt,
// is a property of the surviving substitution and is pinned there by
// `chelis_deep::pipe::tests::a_transitional_variant_is_passed_through`.

// ── harden-bounded-monomorphization D1/D2/D4 unit locks ──

/// Provenance, not symbol spelling, identifies an internal
/// specialization. A valid authored snake_case name can exactly match the
/// mangling grammar.
#[test]
fn mono_specialization_provenance_is_explicit() {
    test_resources().run(|_| {
        let name = "authored__mono_0123456789abcdef";
        let authored = synthetic_tensor_function(name, HostFunctionOrigin::Authored);
        let generated = synthetic_tensor_function(name, HostFunctionOrigin::Monomorphized);
        assert!(!authored.is_monomorphized_specialization());
        assert!(generated.is_monomorphized_specialization());
        let minted = mono_specialization_symbol(
            "depth",
            &mono_specialization_key("depth", &[HostTypeTerm::Int64], &HostTypeTerm::Int64),
        );
        assert!(minted.starts_with("depth__mono_"));
        assert_eq!(minted.len(), "depth__mono_".len() + 16);
    })
}

/// The interning key is a purpose-built canonical rendering with a
/// golden spelling — never `derive(Debug)` output, whose shape is stable
/// only by accident.
#[test]
fn mono_specialization_key_is_stable() {
    test_resources().run(|_| {
        let box_int32 = HostTypeTerm::Adt(
            "Box".to_string(),
            vec![HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                Prim::Int32,
            ))],
        );
        let i32 = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let key = mono_specialization_key("depth", &[box_int32, i32.clone()], &i32);
        assert_eq!(key, "depth\u{1}adt:Box[s:i32,]\u{1}s:i32\u{1}\u{1}s:i32");
        let tensor = HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Lit(4), DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        });
        let mut rendered = String::new();
        write_canonical_host_type_key(&tensor, &mut rendered);
        assert_eq!(rendered, "tensor[4,n,;f32]");

        let polymorphic = HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
            precision: HostPrecisionTerm::Variable("p".to_string()),
            shape: HostShapeTerm::Polymorphic(vec![
                crate::host_type_state::HostShapeSlot::RankVariable("pre".to_string()),
                crate::host_type_state::HostShapeSlot::Dim(DimInfo::Named("seq".to_string(), None)),
            ]),
        });
        rendered.clear();
        write_canonical_host_type_key(&polymorphic, &mut rendered);
        assert_eq!(
            rendered,
            "ptensor[poly(rank:pre,dim:named:seq,);precision-var:p]"
        );
        rendered.clear();
        write_canonical_host_type_key(
            &HostTypeTerm::InferenceVariable(HostInferenceVar(17)),
            &mut rendered,
        );
        assert_eq!(rendered, "iv:17");
    })
}

/// An FNV-1a collision must fail loudly, never collapse two
/// instantiations into one C definition.
#[test]
fn mono_symbol_collision_is_loud() {
    test_resources().run(|_| {
        let mut state = MonoSpecializationState::default();
        register_mono_symbol_key(&mut state, "depth__mono_0123456789abcdef", "key-a")
            .expect("first registration succeeds");
        register_mono_symbol_key(&mut state, "depth__mono_0123456789abcdef", "key-a")
            .expect("re-registration of the same key is a no-op");
        let collision =
            register_mono_symbol_key(&mut state, "depth__mono_0123456789abcdef", "key-b");
        assert_eq!(collision, Err("key-a".to_string()));
    })
}

/// A speculative lowering error reports no summary rejection and restores
/// every field in the specialization state. The same error must surface
/// when the function lowers outside the probe.
#[test]
fn failed_mono_probe_restores_state_and_defers_the_real_error() {
    test_resources().run(|operation| {
        let checked = surf_check(
            r#"
    type Box[a] =
      | Empty
      | Full { value: a }
    def bad_wrap() -> bool = bad(Empty)
    def bad[b](box: Box[b]) -> bool =
      match box with {
        | Empty => true
        | Full { value: item } => bad(Empty)
      }
    "#,
        );
        let i32 = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let seeded = MonoSpecializationState {
            memo: UnordMap::from([("seed-key".to_string(), "seed-symbol".to_string())]),
            symbol_keys: UnordMap::from([("seed-symbol".to_string(), "seed-key".to_string())]),
            functions: vec![LoweredHostFunction {
                function: HostFunction {
                    helper_result_claim_axes: Vec::new(),
                    result_claim: None,
                    entry_claims: Vec::new(),
                    name: "seeded__mono_0123456789abcdef".to_string(),
                    entry_contract: EntryContract::default(),
                    params: Vec::new(),
                    ret_ty: HostTypeTerm::Unit,
                    body: HostExpr::new(HostExprKind::Unit),
                    tensor_helpers: Vec::new(),
                    origin: HostFunctionOrigin::Monomorphized,
                    specialization: None,
                    summary_rejections: Vec::new(),
                },
                products: Vec::new(),
            }],
            in_progress: vec![InProgressMonoSpecialization {
                def_name: "seeded".to_string(),
                param_tys: vec![i32.clone()],
                ret_ty: i32,
            }],
        };
        operation
            .host
            .MONO_SPECIALIZATIONS
            .with(|state| *state.borrow_mut() = seeded.clone());

        assert_eq!(
            top_level_fn_helper_summary_rejects(
                &HostLoweringSession::new(checked.facts(), operation),
                "bad_wrap"
            ),
            Ok(false),
            "a speculative lowering error is not a summary rejection"
        );
        operation.host.MONO_SPECIALIZATIONS.with(|state| {
            let restored = state.borrow();
            assert_eq!(restored.memo, seeded.memo);
            assert_eq!(restored.symbol_keys, seeded.symbol_keys);
            assert_eq!(
                format!("{:#?}", restored.functions),
                format!("{:#?}", seeded.functions),
                "the complete emitted-function state must be restored"
            );
            assert_eq!(restored.in_progress.len(), seeded.in_progress.len());
            for (restored, expected) in restored.in_progress.iter().zip(&seeded.in_progress) {
                assert_eq!(restored.def_name, expected.def_name);
                assert_eq!(restored.param_tys, expected.param_tys);
                assert_eq!(restored.ret_ty, expected.ret_ty);
            }
        });

        let body = find_top_level_def_expr(checked.exprs(), "bad_wrap", operation)
            .expect("the checked program contains bad_wrap");
        let error = lower_host_function(
            "bad_wrap",
            body,
            None,
            &HostLoweringSession::new(checked.facts(), operation),
            false,
        )
        .expect_err("real lowering must report the genuine bad call");
        assert_eq!(
            error.message,
            "unsupported: Deep expression `app` on host expression lowering: generic host \
         call `bad` has no concrete checked type application to specialize \
         (chelis#1226; [05-UNS-1]) (lowering); deliberate [04-TOT-3]: a malformed or \
         unhandled Deep form cannot lower to a substitute host value"
        );
        operation
            .host
            .MONO_SPECIALIZATIONS
            .with(|state| *state.borrow_mut() = MonoSpecializationState::default());
    })
}

/// A nested probe's already-returned error retains the existing deferral.
/// The public source still rejects in real lowering (API/CLI regression).
#[test]
fn issue_1922_nested_returned_probe_error_retains_deferral_and_cleanup() {
    test_resources().run(|operation| {
        assert_issue_1922_summary_probe("sink_output", false, operation);
    })
}

/// A fatal unwind raised directly in this probe returns the original
/// diagnostic only after cleanup, and never installs a false cache row.
#[test]
fn issue_1922_direct_fatal_summary_probe_restores_state_before_returning_error() {
    test_resources().run(|operation| {
        assert_issue_1922_summary_probe("causal_sdpa_with_sink", true, operation);
    })
}

/// The shared fixture with a direct fatal in `causal_sdpa_with_sink`'s own
/// body: a runtime `reduce_window_sum` window list, a deliberate unimplemented
/// rejection owned by chelis#1058. The fixture's original direct fatal was
/// its `insert` sized by `seq`, which chelis#469 made an ordinary runtime
/// extent; the probe mechanics under test need a fatal raised in the probed
/// body, not that particular one.
const ISSUE_1922_WINDOW: &str =
    "  pooled = reduce_window_sum(&masked, [seq, 1i64], [1i64, 1i64])\n";

fn issue_1922_probe_fixture() -> String {
    let fixture = include_str!("../../../../tests/support/helper_summary_fatal.ch");
    let anchor = "  sink_scalar = scalar_to_tensor(sink)\n";
    assert_eq!(fixture.matches(anchor).count(), 1, "fixture anchor");
    fixture
        .replace(anchor, &format!("{anchor}{ISSUE_1922_WINDOW}"))
        .replace("  _ = masked\n", "  _ = masked\n  _ = pooled\n")
}

fn assert_issue_1922_summary_probe(
    name: &str,
    raises_here: bool,
    operation: &Operation<'_, chelis_syntax_host::NativeStack>,
) {
    let fixture = issue_1922_probe_fixture();
    let checked = surf_check(&fixture);
    let session = HostLoweringSession::new(checked.facts(), operation);
    let _restore = MonoProbeGuard::begin(operation);
    operation.host.MONO_SPECIALIZATIONS.with(|state| {
        state
            .borrow_mut()
            .memo
            .insert("seed-key".into(), "seed-symbol".into());
    });
    let seeded = operation
        .host
        .MONO_SPECIALIZATIONS
        .with(|state| state.borrow().clone());
    for _ in 0..2 {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            top_level_fn_helper_summary_rejects(&session, name)
        }));
        if let Err(payload) = &caught {
            eprintln!(
                "escaped diagnostic: {:?}",
                payload.downcast_ref::<crate::lower::LowerDiagnostic>()
            );
        }
        let result = caught.expect("summary probe must not unwind");
        if raises_here {
            let error = result.expect_err("direct fatal diagnostic must not be deferred");
            assert!(error.fatal, "{error:?}");
            assert!(
                error
                    .message
                    .starts_with("unsupported: a non-literal window list for `reduce_window_sum`")
                    && error.message.contains("unimplemented chelis#1058"),
                "{error:?}"
            );
            let list = "[seq, 1i64]";
            let start = fixture.find(list).expect("window list");
            assert_eq!(
                error.span_id.as_deref(),
                Some(format!("surf:{start}..{}", start + list.len()).as_str())
            );
        } else {
            assert_eq!(
                result,
                Ok(false),
                "preserve already-returned error deferral"
            );
        }
        assert!(!is_inlining("sink_output", operation));
        assert!(!is_inlining("causal_sdpa_with_sink", operation));
        assert_eq!(
            session
                .facts
                .helper_summary_rejects
                .borrow()
                .get(name)
                .copied(),
            if raises_here { None } else { Some(false) }
        );
        operation.host.MONO_SPECIALIZATIONS.with(|state| {
            let restored = state.borrow();
            assert_eq!(restored.memo, seeded.memo);
            assert_eq!(restored.symbol_keys, seeded.symbol_keys);
            assert_eq!(
                format!("{:#?}", restored.functions),
                format!("{:#?}", seeded.functions)
            );
            assert_eq!(restored.in_progress.len(), seeded.in_progress.len());
        });
    }
}

#[test]
fn named_entry_lowering_keeps_authored_unused_interface_obligations() {
    test_resources().run(|operation| {
        let checked = surf_check(
            r#"
    def main[n](x: tensor[n, f32], unused: tensor[n, f32]) -> tensor[n, f32] =
      dropout(key_from_seed(0i64), x, 0.5f32)
    "#,
        );
        let dag = lower_named_tensor_entry_dag(checked.facts(), "main", operation)
            .expect("valid checked metadata")
            .expect("scoped entry lowers");
        let loads = dag
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                chelis_dag_core::dag::RiscOp::Load { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            loads,
            vec!["x", "unused"],
            "an authored repeated-dimension obligation keeps an unread parameter"
        );
        assert!(dag.nodes().iter().any(|node| matches!(
            &node.op,
            chelis_dag_core::dag::RiscOp::ExtentWitness { parameter, claims, .. }
                if parameter == "unused" && !claims.is_empty()
        )));
    })
}

#[test]
fn named_tensor_entry_lowers_a_runtime_dropout_rate_as_an_operand() {
    test_resources().run(|operation| {
        let checked = surf_check(
            r#"
    def main(x: tensor[4, f32], rate: f32) -> tensor[4, f32] =
      dropout(key_from_seed(0i64), x, rate)
    "#,
        );
        let dag = lower_named_tensor_entry_dag(checked.facts(), "main", operation)
            .expect("valid checked metadata")
            .expect("a runtime rate is an operand");
        let dropout = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, chelis_dag_core::dag::RiscOp::Dropout))
            .expect("the entry draws");
        assert!(matches!(
            &dag.get(dropout.inputs[1]).expect("rate operand").op,
            chelis_dag_core::dag::RiscOp::Load { name } if name.as_str() == "rate"
        ));
    })
}

fn synthetic_tensor_function(name: &str, origin: HostFunctionOrigin) -> ConcreteHostFunction {
    let tensor = || {
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        })
    };
    HostFunction {
        helper_result_claim_axes: Vec::new(),
        result_claim: None,
        entry_claims: Vec::new(),
        name: name.to_string(),
        entry_contract: EntryContract::default(),
        params: vec![HostParam {
            name: "x".to_string(),
            ty: tensor(),
        }],
        ret_ty: tensor(),
        body: HostExpr::new(HostExprKind::Unit),
        tensor_helpers: Vec::new(),
        origin,
        specialization: None,
        summary_rejections: Vec::new(),
    }
}

/// A tensor-shaped specialization must not displace the authored
/// preferred tensor entry (the PR #1202-measured ABI flip).
#[test]
fn preferred_tensor_entry_ignores_specializations() {
    test_resources().run(|_| {
        let mut program = ConcreteHostProgram::default();
        program.functions.push(synthetic_tensor_function(
            "entry",
            HostFunctionOrigin::Authored,
        ));
        program.functions.push(synthetic_tensor_function(
            "tloop__mono_0123456789abcdef",
            HostFunctionOrigin::Monomorphized,
        ));
        assert_eq!(preferred_tensor_entry_name(&program), Some("entry"));
    })
}

#[test]
fn tensor_signature_classification_ignores_specializations() {
    test_resources().run(|_| {
        let mut program = ConcreteHostProgram::default();
        program.functions.push(synthetic_tensor_function(
            "entry",
            HostFunctionOrigin::Authored,
        ));
        program.functions.push(synthetic_tensor_function(
            "tloop__mono_0123456789abcdef",
            HostFunctionOrigin::Monomorphized,
        ));
        assert!(function_has_tensor_signature(&program, "entry"));
        assert!(!function_has_tensor_signature(
            &program,
            "tloop__mono_0123456789abcdef"
        ));
    })
}

fn parse_and_check(src: &str) -> CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(src, &chelis_syntax_host::NativeStack)
        .expect("parse failed");
    let checked = chelis_types::check_ir_program(&exprs)
        .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
    let checked = chelis_effects::check_program(&checked)
        .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
    chelis_types::check_linearity(&checked)
        .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
}

fn surf_check(src: &str) -> CheckedProgram {
    let decls = chelis_surf::parser::parse_str(src, &chelis_syntax_host::NativeStack)
        .expect("surf parse failed");
    let deep = chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
        .expect("Surf fixture must desugar");
    chelis_types::check_ir_program(&deep)
        .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors))
}

#[test]
fn tensor_helpers_preserve_lexical_host_value_boundaries() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "type Model = | Model { value: tensor[2, f32] }\n\
         m = Model { value: to_tensor([1.0f32, 2.0f32]) }\n\
         def inspect(m: &Model) -> f32 = tensor_to_scalar(sum(m.value, 0i32))\n\
         def forwarded(m: &Model) -> f32 = inspect(m)\n\
         def scaled(m: &Model, scale: f32) -> f32 = mul(inspect(m), scale)\n\
         def differentiated(m: &Model, scale: f32) -> f32 = grad(scaled, wrt=scale)(m, scale)\n\
         def matched(m: &Model, scale: f32) -> f32 = match m with {\n\
           | Model { value: v } => mul(tensor_to_scalar(sum(v, 0i32)), scale)\n\
         }\n\
         def differentiated_match(m: &Model, scale: f32) -> f32 = grad(matched, wrt=scale)(m, scale)\n\
         def nested_read(p: (string, Model), x: f32) -> f32 = mul(inspect(p.1), x)\n\
         def nested_read_bridge(s: string, m: Model, x: f32) -> f32 = grad(nested_read, wrt=x)((s, m), x)\n\
         def selected(p: (Option[f32], f32), x: f32) -> f32 = mul(p.1, x)\n\
         def selected_bridge(o: Option[f32], c: f32, x: f32) -> f32 = (grad(selected, wrt=p)((o, c), x)).1\n",
    );
    let session = HostLoweringSession::new(checked.facts(), operation);
    let context = cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
    // A lexical host parameter cannot become the same-spelled global
    // just because the helper's numeric input list omits its ADT type.
    // Forwarding and exclusion from `wrt` do not make a payload read safe.
    for name in [
        "inspect",
        "forwarded",
        "differentiated",
        "differentiated_match",
        "nested_read_bridge",
        "selected_bridge",
    ] {
        let (_, body) = session.def_named(name).unwrap();
        let signature = host_def_signature(name, body, None, &session).expect("valid checked admission facts").unwrap();
        assert!(
            lower_tensor_helper_dag(
                &signature.body_expr,
                &session,
                &signature.scope,
                &TensorType::scalar_f32(),
                &context,
                None,
            ).expect("valid tensor helper lowering")
            .is_none(),
            "{name} must preserve the payload boundary"
        );
        assert!(
            lower_tensor_helper_dag_with_controls(
                &signature.body_expr,
                &session,
                &signature.scope,
                &TensorType::scalar_f32(),
            ).expect("valid tensor helper lowering")
            .is_none(),
            "{name} with controls"
        );
        #[cfg(feature = "lowering-trace")]
        assert!(
            lower_tensor_helper_product(
                &signature.body_expr,
                &session,
                &signature.scope,
                &TensorType::scalar_f32(),
                true,
                Some(&context),
                None,
            ).expect("valid tensor helper lowering")
            .is_none(),
            "{name} traced"
        );
    } })
}

#[test]
fn tensor_helpers_keep_numeric_inner_binders_and_callable_operands() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "def square(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0i32))\n\
         def inner(x: tensor[2, f32]) -> tensor[2, f32] = {\n  m = x\n  mul(m, m)\n}\n\
         def differentiated(x: tensor[2, f32]) -> tensor[2, f32] = grad(square)(x)\n",
        );
        let session = HostLoweringSession::new(checked.facts(), operation);
        let context =
            cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
        for name in ["inner", "differentiated"] {
            let (_, body) = session.def_named(name).unwrap();
            let mut signature = host_def_signature(name, body, None, &session)
                .expect("valid checked admission facts")
                .unwrap();
            // The inner tensor binder shadows this unrelated outer host
            // value; its opaque binding must yield to the numeric binder.
            signature
                .scope
                .insert("m".into(), HostTypeTerm::Adt("Model".into(), Vec::new()));
            signature.scope.insert(
                "square".into(),
                HostTypeTerm::Fn(
                    vec![signature.params[0].ty.clone()],
                    Box::new(HostTypeTerm::Float32),
                ),
            );
            let expected = tensor_type_from_host_input(&signature.ret_ty).unwrap();
            assert!(
                lower_tensor_helper_dag(
                    &signature.body_expr,
                    &session,
                    &signature.scope,
                    &expected,
                    &context,
                    None,
                )
                .expect("valid tensor helper lowering")
                .is_some(),
                "{name} must retain its numeric helper"
            );
            #[cfg(feature = "lowering-trace")]
            assert!(
                lower_tensor_helper_product(
                    &signature.body_expr,
                    &session,
                    &signature.scope,
                    &expected,
                    true,
                    Some(&context),
                    None,
                )
                .expect("valid tensor helper lowering")
                .is_some(),
                "{name} must retain its traced numeric helper"
            );
        }
    })
}

#[test]
fn tensor_helpers_keep_checked_operation_specialization() {
    test_resources().run(|operation| {
        let checked =
            surf_check("def finite(x: tensor[2, f32]) -> tensor[2, bool] = is_finite(x)\n");
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, body) = session.def_named("finite").unwrap();
        let mut signature = host_def_signature("finite", body, None, &session)
            .expect("valid checked admission facts")
            .unwrap();
        let value = StaticBuiltinValue::resolve_bare("is_finite").unwrap();
        let builtin_operation = value.checked_identity();
        signature.scope.insert(
            "is_finite".into(),
            HostTypeTerm::CheckedBuiltinCallable(builtin_operation),
        );
        let context =
            cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
        let expected = tensor_type_from_host_input(&signature.ret_ty).unwrap();
        // A checked callable is selected by the operation authority, rather
        // than captured as either a numeric input or an aggregate payload.
        assert!(
            lower_tensor_helper_dag(
                &signature.body_expr,
                &session,
                &signature.scope,
                &expected,
                &context,
                Some(value.clone()),
            )
            .expect("valid tensor helper lowering")
            .is_some()
        );
        #[cfg(feature = "lowering-trace")]
        assert!(
            lower_tensor_helper_product(
                &signature.body_expr,
                &session,
                &signature.scope,
                &expected,
                true,
                Some(&context),
                Some(value.clone()),
            )
            .expect("valid tensor helper lowering")
            .is_some()
        );
    })
}

#[test]
fn tensor_helpers_specialize_ignored_host_payloads_without_reading_them() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "type Model = | Model { value: tensor[2, f32] }\n\
         m = Model { value: to_tensor([1.0f32, 2.0f32]) }\n\
         def ignored(m: &Model, x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)\n\
         def forwarded(m: &Model, x: tensor[2, f32]) -> tensor[2, f32] = ignored(m, x)\n\
         def square(xs: List[f32], scale: f32) -> f32 = mul(scale, scale)\n\
         def differentiated(xs: List[f32], scale: f32) -> f32 = grad(square, wrt=scale)(xs, scale)\n\
         def mixed(p: (string, (Option[f32], f32)), x: f32) -> f32 = mul(mul(x, x), (p.1).1)\n\
         def mixed_bridge(s: string, o: Option[f32], c: f32, x: f32) -> f32 = grad(mixed, wrt=x)((s, (o, c)), x)\n\
         def selected_discrete(p: (string, f32), x: f32) -> f32 = mul(p.1, x)\n\
         def selected_discrete_bridge(s: string, c: f32, x: f32) -> f32 = (grad(selected_discrete, wrt=p)((s, c), x)).1\n",
    );
    let session = HostLoweringSession::new(checked.facts(), operation);
    let context = cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
    for name in [
        "forwarded",
        "differentiated",
        "mixed_bridge",
        "selected_discrete_bridge",
    ] {
        let (_, body) = session.def_named(name).unwrap();
        let signature = host_def_signature(name, body, None, &session).expect("valid checked admission facts").unwrap();
        let expected = tensor_type_from_host_input(&signature.ret_ty).unwrap();
        assert!(
            lower_tensor_helper_dag(
                &signature.body_expr,
                &session,
                &signature.scope,
                &expected,
                &context,
                None
            ).expect("valid tensor helper lowering")
            .is_some(),
            "{name}"
        );
        #[cfg(feature = "lowering-trace")]
        assert!(
            lower_tensor_helper_product(
                &signature.body_expr,
                &session,
                &signature.scope,
                &expected,
                true,
                Some(&context),
                None
            ).expect("valid tensor helper lowering")
            .is_some(),
            "{name} traced"
        );
    } })
}

/// A typed local callable carries `surf_binding_type` on its bind value.
/// Inlining moves that value into callee position, where the node gate
/// refuses the key, so the binding origin must stay with the binding.
#[test]
fn typed_local_callable_inlines_without_its_binding_origin() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "def f(x: f32) -> f32 = {\n  \
           g: (f32) -> f32 = fn (y: f32) -> add(y, y)\n  \
           g(x)\n\
         }\n\
         r = f(cast(2.0, f32))\n",
        );
        let binding_origin = chelis_deep::printer::print_canonical(checked.exprs());
        assert!(
            binding_origin.contains("surf_binding_type"),
            "the fixture must carry a binding origin on its callable: {binding_origin}"
        );
        if let Err(diagnostic) = try_lower_compiled_program(checked.facts(), operation) {
            panic!("a typed local callable must inline: {}", diagnostic.message);
        }
    })
}

/// chelis#2181: a callee that takes a function-typed parameter, whose
/// body is a chain of `binders` `let` bindings.
///
/// The C lane has no function-pointer representation, so this callee is
/// inlined at its call site rather than emitted as a C function. That
/// makes the inliner's cost, not the emitter's, the quantity under test.
fn issue_2181_binder_chain_source(binders: usize) -> String {
    let mut lines = vec![
        "module Demo.BinderChain".to_string(),
        "def sq(x: f32) -> f32 = mul(x, x)".to_string(),
        "def body(f: f32 -> f32, x: f32) -> f32 = {".to_string(),
        "  v1 = f(x)".to_string(),
    ];
    for index in 2..=binders {
        lines.push(format!("  v{index} = add(v{prev}, x)", prev = index - 1));
    }
    lines.push(format!("  v{binders}"));
    lines.push("}".to_string());
    lines.push("def main() -> f32 = body(sq, cast(2.0, f32))".to_string());
    lines.join("\n") + "\n"
}

/// A chain of mutually recursive defs, each calling the next from two
/// argument positions. The kernel-decision probe expands that call graph
/// as a tree, so an unmemoized run costs 2^depth summary builds while a
/// memoized one costs one per definition.
fn issue_1829_fanout_source(depth: usize) -> String {
    let mut lines = vec!["module Demo.Fanout".to_string(), String::new()];
    for level in 0..depth {
        lines.push(format!(
            "def f{level}(x: i64) -> i64 = add(f{next}(x), f{next}(x))",
            next = level + 1
        ));
    }
    lines.push(format!("def f{depth}(x: i64) -> i64 = x"));
    lines.push(String::new());
    lines.join("\n")
}

/// chelis#1829 class receipt, reshaped for the session. The interpreter
/// reaches `host_def_kernel` once per applied definition, and the summary
/// probes behind those calls must be bounded by the definition count
/// rather than expanding the call graph as a tree.
///
/// What this pins that `issue_1835_kernel_decision_work_is_linear` does
/// not: the always-compiled probe counter and the `cfg(test)` profile
/// count the SAME misses. The counted receipt in `chelis-compiler-api`
/// reads only the former, so a divergence there would let that receipt
/// pass on the wrong number.
///
/// Evidentiary status: REGRESSION TEST against the base sha, where
/// `host_def_kernel` took a bare program, nothing armed the memo, and
/// this fixture cost 398,574 builds for thirteen definitions. It was a
/// disposition lock on chelis#1843's armed scope; the session makes the
/// same assertion hold with no scope to arm.
#[test]
fn issue_1829_interpreter_session_bounds_kernel_decision_probes() {
    test_resources().run(|_| {
        let depth = 12;
        std::thread::Builder::new()
            .name("issue-1829-fanout".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                test_resources()
        .run(|operation|{let checked = surf_check(&issue_1829_fanout_source(depth));
    let names = (0..=depth)
        .map(|level| format!("f{level}"))
        .collect::<Vec<_>>();
    reset_host_work_profile(operation);
    reset_host_summary_probe_builds();
    {
        let session = HostLoweringSession::new(checked.facts(), operation);
        for name in &names {
            host_def_kernel(&session, name)
                .expect("#1829 fixture reaches a kernel decision");
        }
    }
    let profile = take_host_work_profile(operation);
    let probes = host_summary_probe_builds();
    eprintln!(
        "#1829 depth={depth} defs={} helper_summary_builds={} probe_builds={probes}",
        names.len(),
        profile.helper_summary_builds
    );
    assert!(
        profile.helper_summary_builds <= names.len(),
        "#1829: a session must build at most one summary per definition; \
         {} definitions produced {} builds",
        names.len(),
        profile.helper_summary_builds
    );
    assert_eq!(
        probes,
        u64::try_from(profile.helper_summary_builds).expect("probe count fits"),
        "#1829: the always-compiled probe counter and the test-only profile \
         must count the same misses, or a downstream receipt reads the wrong number"
    );})
            })
            .expect("#1829 probe thread starts")
            .join()
            .expect("#1829 probe thread completes");
    })
}

/// chelis#2181: inlining a callable-parameter callee is not exponential in
/// its binder count.
///
/// The claim is deliberately "not exponential" rather than "linear":
/// substitution work is superlinear for other shapes (a lambda every
/// fourth binder measures about 7.4n^2 on this tree), which this test
/// does not and should not assert against.
///
/// `substitute_expr`'s `let` arm used to substitute every child into a
/// `elements` vector, then discard it and redo the work as `rebuilt` for
/// the ordinary two-child `let`. Every nested binder therefore substituted
/// its own subtree twice, so one inline cost 2^(binders) and a 40-binder
/// callee such as nautilus `Roots.brent_rec` never finished.
///
/// The assertion is a ratio rather than an absolute count, so it states
/// the asymptotic promise directly and needs no machine-specific budget:
/// doubling the binder count may at most triple the substitution work.
///
/// Evidentiary status: REGRESSION TEST, established by restoring the
/// discarded pass on this tree and keeping this counter. Unfixed, 8
/// binders cost 6,119 substituted nodes and 16 cost 1,572,839, a ratio of
/// 257.0x = 2^8, and this assertion fails. Fixed, the same fixture costs
/// a ratio near 2.
#[test]
fn issue_2181_callable_parameter_inlining_is_not_exponential_in_binder_count() {
    test_resources().run(|operation| {
        fn substitution_nodes(
            binders: usize,
            operation: &Operation<'_, chelis_syntax_host::NativeStack>,
        ) -> usize {
            let checked = surf_check(&issue_2181_binder_chain_source(binders));
            let lowered = top_level_lowering_map(checked.exprs(), checked.type_env(), operation)
                .expect("checked fixture classification");
            reset_host_work_profile(operation);
            lower_host_program(
                &HostLoweringSession::new(checked.facts(), operation),
                &lowered,
            )
            .expect("#2181 fixture must lower");
            take_host_work_profile(operation).substitution_nodes
        }
        let small = substitution_nodes(8, operation);
        let doubled = substitution_nodes(16, operation);
        assert!(
            doubled <= small.saturating_mul(3),
            "#2181: doubling the binder count must at most triple substitution \
         work; 8 binders cost {small} nodes and 16 cost {doubled}, a \
         factor of {factor}. An exponential inliner shows about 2^8 here.",
            factor = doubled / small.max(1),
        );
    })
}

/// chelis#1829 hazard (1), restated for the session. The interpreter holds
/// one session across a whole evaluation, and a whole-program lowering
/// reached from inside it builds a second. Under the thread-local memo
/// that inner entry could disarm the flag and clear every cache, so the
/// hazard needed a refcount to describe. Two sessions share nothing, so
/// the property to hold is now the simpler one: an inner session neither
/// clears the enclosing session's facts nor reads them.
///
/// Evidentiary status: REGRESSION TEST against the base sha, where both
/// programs' facts lived in one thread-local map and the outer program's
/// rows were observable from the inner entry (and, before chelis#1843's
/// refcount, cleared by its drop). It replaces
/// `issue_1829_nested_scope_keeps_the_outer_memo_armed_and_populated`,
/// whose flag and refcount no longer exist.
#[test]
fn issue_1829_a_nested_session_neither_clears_nor_reads_the_enclosing_one() {
    test_resources().run(|operation| {
        let outer = surf_check("module Demo.Outer\n\ndef outer_fn(x: i64) -> i64 = x\n");
        let inner = surf_check("module Demo.Inner\n\ndef inner_fn(x: i64) -> i64 = x\n");
        let outer_session = HostLoweringSession::new(outer.facts(), operation);

        assert!(
            cached_program_defs(&outer_session).contains_key("outer_fn"),
            "#1829: the enclosing session must actually derive facts, or this \
         receipt would pass without measuring anything"
        );

        {
            let inner_session = HostLoweringSession::new(inner.facts(), operation);
            let inner_defs = cached_program_defs(&inner_session);
            assert!(
                inner_defs.contains_key("inner_fn") && !inner_defs.contains_key("outer_fn"),
                "#1829: an inner session's facts describe its own program only"
            );
            assert!(
                !cached_program_defs(&outer_session).contains_key("inner_fn"),
                "#1829: an inner session must not write into the enclosing one"
            );
        }

        // The inner session is gone. The enclosing one is untouched, and the
        // borrow checker, not a refcount, is what made that true.
        assert!(
            cached_program_defs(&outer_session).contains_key("outer_fn"),
            "#1829: an inner session's drop must not clear the enclosing session"
        );
    })
}

/// chelis#1835 fixture: a fan-out chain of defs, each calling the next
/// from two argument positions.
///
/// Two shapes, because the kernel decision costs differently on each. A
/// TENSOR-returning chain reaches the helper-summary probe and stays on
/// the kernel path. A SCALAR-returning chain reaches the same probe
/// first, because `def_body_decision_impl` asks it before the declared
/// result type can rule the def out, and the probe's own lowering then
/// takes the host lane, whose inliner duplicates the callee body per call
/// site. The second shape is therefore the expensive one, and both are
/// the same defect: a per-definition fact recomputed per ask.
fn issue_1835_fanout_source(depth: usize, tensor_result: bool) -> String {
    let (module, ty, leaf) = if tensor_result {
        ("Demo.KernelFanout", "tensor[4, f32]", "mul(x, x)")
    } else {
        ("Demo.ScalarFanout", "i64", "mul(x, x)")
    };
    let mut lines = vec![format!("module {module}"), String::new()];
    for level in 0..depth {
        lines.push(format!(
            "def f{level}(x: {ty}) -> {ty} = add(f{next}(x), f{next}(x))",
            next = level + 1
        ));
    }
    lines.push(format!("def f{depth}(x: {ty}) -> {ty} = {leaf}"));
    lines.push(String::new());
    lines.join("\n")
}

/// Which defs the C lane lowered through the tensor DAG. A kernel body
/// becomes a call to an extracted tensor helper attached to the emitted
/// host function; a host-lane body has none.
fn issue_1835_c_lane_kernel_defs(
    checked: &CheckedProgram,
    names: &[String],
    operation: &Operation<'_, chelis_syntax_host::NativeStack>,
) -> Vec<bool> {
    let compiled =
        try_lower_compiled_program(checked.facts(), operation).expect("#1835 fixture lowers on C");
    let host = compiled.host.expect("#1835 fixture emits a host program");
    names
        .iter()
        .map(|name| {
            host.functions
                .iter()
                .find(|function| &function.name == name)
                .is_some_and(|function| !function.tensor_helpers.is_empty())
        })
        .collect()
}

/// chelis#1835 class receipt. `host_def_kernel` is the entry every lane
/// asks for one def's kernel decision, and its cost must be linear in the
/// definition count with NO opt-in: no flag to arm, no scope to hold, no
/// argument beyond the session every caller must now build. A second full
/// pass must add no work at all, and the answers must agree with the lane
/// that compiles the whole program (chelis#1277 B2h).
///
/// Evidentiary status: REGRESSION TEST for every assertion, watched red on
/// the base sha `6abca2406`, where the memo is gated on a thread-local
/// flag that no `host_def_kernel` caller arms. Thirteen definitions cost
/// 78 summary builds and 728 definition collections on the tensor chain,
/// and 398,574 builds and 797,174 collections on the scalar chain; the
/// second pass repeated both figures exactly, which is the memo never
/// being read. The pull request records the runs.
#[test]
fn issue_1835_kernel_decision_work_is_linear() {
    test_resources()
        .run(|operation| { const DEPTH: usize = 12;
    for tensor_result in [true, false] {
        std::thread::Builder::new()
            .name(format!("issue-1835-fanout-{tensor_result}"))
            .stack_size(512 * 1024 * 1024).spawn(move ||{
        test_resources()
        .run(|operation|{let typed = surf_check(&issue_1835_fanout_source(DEPTH, tensor_result));
        let effected = chelis_effects::check_program(&typed).expect("effect check");
        let checked =
            chelis_types::check_linearity(&effected).expect("linearity check");
        let names = (0..=DEPTH)
            .map(|level| format!("f{level}"))
            .collect::<Vec<_>>();
        // One session, established by the caller because the type
        // leaves it no choice, and NO other opt-in.
        let session = HostLoweringSession::new(checked.facts(), operation);
        reset_host_work_profile(operation);
        let first_answers = names
            .iter()
            .map(|name| {
                host_def_kernel(&session, name)
                    .expect("#1835 fixture reaches a kernel decision")
                    .is_some()
            })
            .collect::<Vec<_>>();
        let first = take_host_work_profile(operation);
        let second_answers = names
            .iter()
            .map(|name| {
                host_def_kernel(&session, name)
                    .expect("#1835 fixture reaches a kernel decision")
                    .is_some()
            })
            .collect::<Vec<_>>();
        let second = take_host_work_profile(operation);
        eprintln!(
            "#1835 tensor_result={tensor_result} depth={DEPTH} defs={} \
             helper_summary_builds={}+{} program_def_collections={}+{}",
            names.len(),
            first.helper_summary_builds,
            second.helper_summary_builds,
            first.program_def_collections,
            second.program_def_collections
        );
        assert!(
            first.helper_summary_builds <= names.len(),
            "#1835: an unopted `host_def_kernel` caller must build at most one \
             summary per definition; {} definitions produced {} builds \
             (tensor_result={tensor_result})",
            names.len(),
            first.helper_summary_builds
        );
        assert_eq!(
            second.helper_summary_builds, 0,
            "#1835: a second full pass over the same program must build no \
             further summaries (tensor_result={tensor_result})"
        );
        assert_eq!(
            first.program_def_collections + second.program_def_collections,
            1,
            "#1835: program definitions must be collected once for the session, \
             not once per decision (tensor_result={tensor_result})"
        );
        assert_eq!(
            first_answers, second_answers,
            "#1835: a memoized decision must not differ from the one that built \
             it (tensor_result={tensor_result})"
        );
        assert_eq!(
            first_answers,
            issue_1835_c_lane_kernel_defs(&checked, &names, operation),
            "#1835: the shared decision must answer what the C lane compiles \
             (chelis#1277 B2h; tensor_result={tensor_result})"
        );})})
            .expect("#1835 fanout thread starts")
            .join()
            .expect("#1835 fanout thread completes");
    }
    // Both chains above answer uniformly, all-kernel or all-host, so their
    // cross-lane comparison cannot fail for a PER-DEFINITION disagreement:
    // two constant vectors of the same constant are equal whatever the
    // decision did. This program's answers are non-uniform by
    // construction, one definition per documented class, so the
    // comparison has something to disagree about.
    let mixed = surf_check(
        "module Demo.MixedLanes\n\n             def k(x: tensor[4, f32]) -> tensor[4, f32] = mul(x, x)\n             def loud(x: tensor[4, f32]) -> tensor[4, f32] ! {IO} = {\n               _ = print(x)\n  x\n}\n             def s(x: i64) -> i64 = add(x, x)\n             def k2(x: tensor[4, f32]) -> tensor[4, f32] = add(k(x), k(x))\n",
    );
    let mixed = chelis_types::check_linearity(
        &chelis_effects::check_program(&mixed).expect("mixed effect check"),
    )
    .expect("mixed linearity check");
    let mixed_names = ["k", "loud", "s", "k2"].map(String::from).to_vec();
    let mixed_session = HostLoweringSession::new(mixed.facts(), operation);
    let mixed_answers = mixed_names
        .iter()
        .map(|name| {
            host_def_kernel(&mixed_session, name)
                .expect("#1835 mixed fixture reaches a kernel decision")
                .is_some()
        })
        .collect::<Vec<_>>();
    eprintln!("#1835 mixed answers={mixed_answers:?}");
    assert!(
        mixed_answers.iter().any(|kernel| *kernel)
            && mixed_answers.iter().any(|kernel| !*kernel),
        "#1835: the mixed fixture must answer non-uniformly, or the cross-lane \
         comparison below is two constant vectors again; got {mixed_answers:?}"
    );
    assert_eq!(
        mixed_answers,
        vec![true, false, false, true],
        "#1835: a pure tensor body is a kernel, an IO effect row and a non-tensor \
         result are host code (chelis#1277 B2h's documented classes)"
    );
    assert_eq!(
        mixed_answers,
        issue_1835_c_lane_kernel_defs(&mixed, &mixed_names, operation),
        "#1835: the shared decision must answer what the C lane compiles, per \
         definition and not merely in aggregate (chelis#1277 B2h)"
    ); })
}

#[test]
fn record_operand_facts_match_the_scoped_reader_and_stop_at_callbacks() {
    test_resources()
        .run(|operation| { for width in ["width(x)", "(fn (z: i64) -> add(z, width(x)))(0i64)"] {
        let checked = surf_check(&format!(
            "type Box = | Box {{ value: i64 }}\ndef width(b: Box) -> i64 = b.value\ndef run(x: Box, y: tensor[1, f32]) -> tensor[*, f32] = reshape(y, [{width}])\n"
        ));
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, body) = session.def_named("run").unwrap();
        let signature = host_def_signature("run", body, None, &session).expect("valid checked admission facts").unwrap();
        let root = issue_1205_find_named_app(&signature.body_expr, "reshape").unwrap();
        let facts = scoped_record_operand_facts(root, &session, &signature.scope).expect("valid checked admission facts");
        let mut pending = vec![root];
        while let Some(expr) = pending.pop() {
            let key = expr as *const Expr as usize;
            assert_eq!(
                facts[&key],
                operand_reads_host_record(expr, &session, &signature.scope).expect("valid checked record metadata")
            );
            match expr {
                Expr::MetaExpr(meta, _) => pending.push(&meta.expr),
                _ => {
                    if let Some((tag, _, children)) = stamped_parts(expr)
                        && !matches!(tag, DeepTag::Var | DeepTag::Fn)
                    {
                        pending.extend(children);
                    }
                }
            }
        }
        assert!(facts.len() <= deep_expr_nodes(root));
    } })
}

#[test]
fn operand_admission_facts_are_bounded_to_a_lowering_scope() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "def width(v: tensor[*, f32]) -> i64 = numel(v)\ndef run[n](x: tensor[n, f32]) -> tensor[*, f32] = reshape(x, [width(x)])\n",
    );
    let session = HostLoweringSession::new(checked.facts(), operation);
    let (_, body) = session.def_named("run").unwrap();
    let signature = host_def_signature("run", body, None, &session).expect("valid checked admission facts").unwrap();
    let parent = issue_1205_find_named_app(&signature.body_expr, "reshape").unwrap();
    let operand = &stamped_parts(parent).unwrap().2[2];
    let scope = signature
        .params
        .iter()
        .map(|param| (param.name.clone(), param.ty.clone()))
        .collect::<HostScope>();
    let before = operation.host.OPERAND_ADMISSION_STACK.with(|stack| stack.borrow().len());
    assert_eq!(cached_operand_admission(parent, operand, &scope, operation), None);
    {
        let _outer = OperandAdmissionGuard::begin(parent, &session, &scope).expect("valid checked admission facts");
        assert_eq!(
            cached_operand_admission(parent, operand, &scope, operation),
            Some(true)
        );
        operation.host.OPERAND_ADMISSION_STACK.with(|stack| {
            assert!(stack.borrow().last().unwrap().operands.len() <= deep_expr_nodes(parent));
        });
        let independent = scope.clone();
        assert_eq!(
            cached_operand_admission(parent, operand, &independent, operation),
            None
        );
        {
            let _inner = OperandAdmissionGuard::begin(parent, &session, &independent).expect("valid checked admission facts");
            assert_eq!(
                cached_operand_admission(parent, operand, &independent, operation),
                Some(true)
            );
        }
        assert_eq!(
            cached_operand_admission(parent, operand, &scope, operation),
            Some(true)
        );
        assert_eq!(
            cached_operand_admission(parent, operand, &independent, operation),
            None
        );
    }
    assert_eq!(cached_operand_admission(parent, operand, &scope, operation), None);
    assert_eq!(
        operation.host.OPERAND_ADMISSION_STACK.with(|stack| stack.borrow().len()),
        before
    ); })
}

#[test]
fn operand_admission_does_not_cache_callee_or_callback_nodes() {
    test_resources()
        .run(|operation| { for anonymous in [false, true] {
        let target = if anonymous {
            "(fn (y: tensor[n, f32]) -> numel(y))(x)"
        } else {
            "width(x)"
        };
        let checked = surf_check(&format!(
            "def width(v: tensor[*, f32]) -> i64 = numel(v)\ndef run[n](x: tensor[n, f32]) -> tensor[*, f32] = reshape(x, [{target}])\n"
        ));
        let session = HostLoweringSession::new(checked.facts(), operation);
        let (_, body) = session.def_named("run").unwrap();
        let signature = host_def_signature("run", body, None, &session).expect("valid checked admission facts").unwrap();
        let parent = issue_1205_find_named_app(&signature.body_expr, "reshape").unwrap();
        let scope = signature
            .params
            .iter()
            .map(|param| (param.name.clone(), param.ty.clone()))
            .collect::<HostScope>();
        let defs = cached_program_defs(&session);
        let callee_body = if anonymous {
            &signature.body_expr
        } else {
            &defs["width"]
        };
        let numel = issue_1205_find_named_app(callee_body, "numel").unwrap();
        let _guard = OperandAdmissionGuard::begin(parent, &session, &scope).expect("valid checked admission facts");
        operation.host.OPERAND_ADMISSION_STACK.with(|stack| {
            assert!(
                !stack
                    .borrow()
                    .last()
                    .unwrap()
                    .operands
                    .contains_key(&(numel as *const Expr as usize))
            );
        });
    } })
}

// [05-OP-54] observes the exact size of an eagerly constructed List.
// A scalar consumer has no tensor helper to consume prepared operands;
// preparing them would lower and discard their work before the host call
// lowers the original actuals again (spec/03 §4.4).
#[test]
fn scalar_list_calls_do_not_prepare_unused_tensor_operands() {
    test_resources().run(|_| {
        std::thread::Builder::new()
            .name("scalar-list-host-work".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                test_resources().run(|operation| {
                    let mut previous = None;
                    for depth in [2, 4, 8, 16, 32, 40] {
                        let expression = (0..depth).fold("1i64".to_string(), |inner, _| {
                            format!("f(len([{inner}, 2i64]))")
                        });
                        let typed =
                            surf_check(&format!("def f(n: i64) -> i64 = n\na = {expression}\n"));
                        let effected = chelis_effects::check_program(&typed).expect("effect check");
                        let checked =
                            chelis_types::check_linearity(&effected).expect("linearity check");
                        reset_host_work_profile(operation);
                        try_lower_compiled_program(checked.facts(), operation)
                            .expect("nested scalar List calls lower");
                        let profile = take_host_work_profile(operation);
                        eprintln!("scalar List depth={depth} profile={profile:?}");
                        assert_eq!(
                            profile.operand_view_axes, 0,
                            "scalar host calls do not prepare tensor axis views"
                        );
                        assert_eq!(
                            profile.app_clone_nodes, 0,
                            "no tensor application is prepared"
                        );
                        assert_eq!(
                            profile.operand_fact_nodes, 0,
                            "no tensor operand admission is needed"
                        );
                        if let Some(visits) = previous {
                            assert!(
                                profile.host_expr_visits * 10 <= visits * 22,
                                "ordinary host work grows with the nested source"
                            );
                        }
                        previous = Some(profile.host_expr_visits);
                    }
                })
            })
            .expect("scalar List profile thread starts")
            .join()
            .expect("scalar List profile thread completes");
    })
}

fn issue_1205_source(operations: usize, flat: bool) -> String {
    let module = if flat { "Flat" } else { "Nested" };
    let mut lines = vec![
        format!("module FrontEndPerformance.{module}N{operations}"),
        "def bc(c: f32) -> tensor[8, f32] = \
             reshape(insert(to_tensor([c]), 0, 8i64), [8i64])"
            .to_string(),
    ];
    if flat {
        lines.push(
            "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else {"
                .to_string(),
        );
        let mut previous = "s".to_string();
        for index in 0..operations {
            lines.push(format!(
                "  t{index} = mul(add({previous}, bc(cast(1.0, f32))), \
                     bc(cast(0.5, f32)))"
            ));
            previous = format!("t{index}");
        }
        lines.extend([format!("  st({previous}, add(i, 1i64))"), "}".to_string()]);
    } else {
        let mut body = "s".to_string();
        for _ in 0..operations {
            body = format!("mul(add({body}, bc(cast(1.0, f32))), bc(cast(0.5, f32)))");
        }
        lines.push(format!(
            "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else st({body}, add(i, 1i64))"
        ));
    }
    lines.push("r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)".to_string());
    lines.join("\n") + "\n"
}

fn issue_1205_host_profile(operations: usize, flat: bool) -> HostWorkProfile {
    std::thread::Builder::new()
        .name(format!("issue-1205-host-{operations}"))
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            test_resources().run(|operation| {
                let typed = surf_check(&issue_1205_source(operations, flat));
                let effected = chelis_effects::check_program(&typed).expect("effect check");
                let checked = chelis_types::check_linearity(&effected).expect("linearity check");
                reset_host_work_profile(operation);
                try_lower_compiled_program(checked.facts(), operation)
                    .expect("#1205 fixture lowers");
                take_host_work_profile(operation)
            })
        })
        .expect("#1205 host profile thread starts")
        .join()
        .expect("#1205 host profile thread completes")
}

fn issue_1205_find_named_app<'expr>(expr: &'expr Expr, expected: &str) -> Option<&'expr Expr> {
    if let Some((DeepTag::App, _, kids)) = stamped_parts(expr)
        && let Some((DeepTag::Var, _, callee_kids)) = kids.first().and_then(stamped_parts)
        && callee_kids.first().and_then(symbol_name) == Some(expected)
    {
        return Some(expr);
    }
    match expr {
        Expr::Node(node, _) => node
            .children_slice()
            .iter()
            .find_map(|child| issue_1205_find_named_app(child, expected)),
        Expr::MetaExpr(meta, _) => issue_1205_find_named_app(&meta.expr, expected),
        Expr::BareList(elements, _) => elements
            .iter()
            .find_map(|child| issue_1205_find_named_app(child, expected)),
        Expr::UnknownForm(data) => data
            .children
            .iter()
            .find_map(|child| issue_1205_find_named_app(child, expected)),
        Expr::Map(map, _) => {
            map.find_expression(|value| issue_1205_find_named_app(value, expected))
        }
        Expr::Atom(_, _) => None,
    }
}

#[test]
fn issue_1205_host_lowering_work_is_linear() {
    test_resources()
        .run(|_| { for flat in [false, true] {
        let mut rows = Vec::new();
        for operations in [20, 40, 80, 160] {
            let profile = issue_1205_host_profile(operations, flat);
            let total_work = profile.host_expr_visits
                + profile.admission_expr_visits
                + profile.reservation_expr_visits
                + profile.reservation_lookups
                + profile.reservation_name_bytes
                + profile.operand_param_entries
                + profile.operand_view_axes
                + profile.operand_scope_clone_entries
                + profile.host_scope_lookups
                + profile.host_scope_transport_entries
                + profile.host_scope_bindings
                + profile.host_scope_clones
                + profile.helper_input_loads
                + profile.helper_global_lookup_steps
                + profile.helper_scope_expr_nodes
                + profile.operand_fact_nodes
                + profile.record_operand_visits
                + profile.app_clone_nodes
                + profile.tensor_helper_input_nodes
                + profile.tensor_helper_preflight_nodes
                + profile.callable_scope_work
                + profile.grad_scan_nodes
                + profile.fail_scan_nodes
                + profile.program_def_clone_nodes
                + profile.type_env_clone_nodes;
            eprintln!(
                "#1205 host shape={} n={operations} profile={profile:?} total_work={total_work}",
                if flat { "flat" } else { "nested" }
            );
            assert_eq!(
                profile.program_def_collections, 1,
                "#1205 program definitions must be collected once per compiled program"
            );
            assert_eq!(
                profile.helper_summary_builds, 2,
                "#1205 helper summaries must be cached per definition"
            );
            assert_eq!(
                profile.tensor_helper_builtin_load_rejections, 0,
                "#1205 dynamic to_tensor candidates must be rejected before DAG lowering"
            );
            assert_eq!(
                profile.tensor_helper_preflight_lookup_misses, 0,
                "#1205 every tensor-helper candidate must have precomputed facts"
            );
            assert!(
                profile.tensor_helper_preflight_rejections > 0,
                "#1205 fixture must exercise the dynamic to_tensor preflight"
            );
            assert!(
                profile.reservation_expr_visits > 0
                    && profile.reservation_lookups > 0
                    && profile.reservation_name_bytes > 0
                    && profile.helper_global_lookup_steps > 0
                    && profile.operand_view_axes > 0,
                "#1205 fixture exercises source reservation, lookup, axis views and name ownership"
            );
            assert!(
                profile.app_clone_nodes > 0 && profile.max_app_clone_nodes > 0,
                "#1205 fixture must exercise prepared application cloning"
            );
            assert!(
                profile.host_scope_clones > 0
                    && profile.host_scope_lookups > 0
                    && profile.helper_input_loads > 0
                    && profile.helper_scope_expr_nodes > 0
                    && profile.record_operand_visits > 0,
                "#1205 fixture exercises the input/scope work being bounded"
            );
            rows.push((operations, total_work, profile));
        }
        for pair in rows.windows(2) {
            let (previous_n, previous, previous_profile) = pair[0];
            let (current_n, current, profile) = pair[1];
            assert!(
                profile.admission_expr_visits * 10
                    <= previous_profile.admission_expr_visits * 22,
                "#1205 scoped admission visits must grow linearly"
            );
            assert!(
                profile.operand_param_entries * 10
                    <= previous_profile.operand_param_entries * 22,
                "#1205 operand scope materialization must grow linearly"
            );
            assert_eq!(
                profile.operand_scope_clone_entries, 0,
                "#1205 persistent host scopes do not copy prefixes"
            );
            for (category, current, previous) in [
                (
                    "name reservation visits",
                    profile.reservation_expr_visits,
                    previous_profile.reservation_expr_visits,
                ),
                (
                    "name reservation lookups",
                    profile.reservation_lookups,
                    previous_profile.reservation_lookups,
                ),
                (
                    "name reservation bytes",
                    profile.reservation_name_bytes,
                    previous_profile.reservation_name_bytes,
                ),
                (
                    "operand view axes",
                    profile.operand_view_axes,
                    previous_profile.operand_view_axes,
                ),
                (
                    "record operand visits",
                    profile.record_operand_visits,
                    previous_profile.record_operand_visits,
                ),
                (
                    "global type lookup steps",
                    profile.helper_global_lookup_steps,
                    previous_profile.helper_global_lookup_steps,
                ),
                (
                    "input scope scan",
                    profile.helper_scope_expr_nodes,
                    previous_profile.helper_scope_expr_nodes,
                ),
                (
                    "scope lookups",
                    profile.host_scope_lookups,
                    previous_profile.host_scope_lookups,
                ),
                (
                    "scope transport",
                    profile.host_scope_transport_entries,
                    previous_profile.host_scope_transport_entries,
                ),
                (
                    "scope bindings",
                    profile.host_scope_bindings,
                    previous_profile.host_scope_bindings,
                ),
                (
                    "scope clones",
                    profile.host_scope_clones,
                    previous_profile.host_scope_clones,
                ),
                (
                    "input loads",
                    profile.helper_input_loads,
                    previous_profile.helper_input_loads,
                ),
            ] {
                assert!(
                    current * 10 <= previous * 22,
                    "#1205 {category} must grow linearly: {previous}->{current}"
                );
            }
            // Operand preparation rebuilds one small application at each
            // host boundary. Its count may grow with source operations;
            // the copied subtree must not grow with expression nesting.
            assert_eq!(
                profile.max_app_clone_nodes, previous_profile.max_app_clone_nodes,
                "#1205 each prepared application must have bounded descendant cloning"
            );
            assert!(
                profile.app_clone_nodes * 10 <= previous_profile.app_clone_nodes * 22,
                "#1205 total prepared application cloning must grow linearly"
            );
            assert!(
                current * 10 <= previous * 22,
                "#1205 host work must grow linearly: shape={} {previous_n}->{current_n} \
                 previous={previous} current={current} profile={profile:?}",
                if flat { "flat" } else { "nested" }
            );
        }
    } })
}

#[test]
fn issue_1205_callable_scope_work_is_linear() {
    test_resources().run(|_| {
        std::thread::Builder::new()
            .name("issue-1205-nested-callable-scopes".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                test_resources().run(|operation| {
                    for nested_fns in [true, false] {
                        let mut rows = Vec::new();
                        for depth in [20, 40, 80, 160] {
                            let body = if nested_fns {
                                let mut body = "bc(c)".to_string();
                                for index in (0..depth).rev() {
                                    body = format!("(fn (p{index}: f32) -> {body})(c)");
                                }
                                body
                            } else {
                                let mut body = format!("f{}(c)", depth - 1);
                                for index in (0..depth).rev() {
                                    let target = if index == 0 {
                                        "bc".to_string()
                                    } else {
                                        format!("f{}", index - 1)
                                    };
                                    body = format!("{{ f{index} = {target}\n{body}\n}}");
                                }
                                body
                            };
                            let shape = if nested_fns { "Fn" } else { "Let" };
                            let source = format!(
                                "module FrontEndPerformance.Nested{shape}{depth}\n\
                         def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
                         def nested(c: f32) -> tensor[1, f32] = {body}\n\
                         out = index(to_list(nested(cast(1.0, f32))), 0i64)\n"
                            );
                            let typed = surf_check(&source);
                            let defs = cached_program_defs(&HostLoweringSession::new(
                                typed.facts(),
                                operation,
                            ));
                            let nested_body = lookup_program_def(&defs, "nested", operation)
                                .expect("nested definition is present");

                            reset_host_work_profile(operation);
                            let summaries = cached_tensor_helper_def_summaries(
                                &HostLoweringSession::new(typed.facts(), operation),
                            )
                            .expect("valid checked admission facts");
                            assert!(summaries["nested"].reaches_dynamic_to_tensor);
                            let summary_profile = take_host_work_profile(operation);
                            let summary_work = summary_profile.tensor_helper_preflight_nodes
                                + summary_profile.callable_scope_work;

                            reset_host_work_profile(operation);
                            let mut facts = UnordMap::new();
                            let nested_facts = analyze_tensor_helper_preflight(
                                nested_body,
                                &summaries,
                                &mut facts,
                            );
                            assert!(nested_facts.reaches_dynamic_to_tensor);
                            let expression_profile = take_host_work_profile(operation);
                            let expression_work = expression_profile.tensor_helper_preflight_nodes
                                + expression_profile.callable_scope_work;

                            eprintln!(
                                "#1205 callable scopes shape={shape} depth={depth} \
                         summary={summary_profile:?} summary_work={summary_work} \
                         expression={expression_profile:?} expression_work={expression_work}"
                            );
                            rows.push((depth, summary_work, expression_work));
                        }
                        for pair in rows.windows(2) {
                            let (previous_depth, previous_summary, previous_expression) = pair[0];
                            let (current_depth, current_summary, current_expression) = pair[1];
                            for (path, previous_work, current_work) in [
                                ("definition summary", previous_summary, current_summary),
                                (
                                    "per-expression facts",
                                    previous_expression,
                                    current_expression,
                                ),
                            ] {
                                assert!(
                                    current_work * 10 <= previous_work * 23,
                                    "#1205 callable-scope work must grow linearly: \
                             shape={} path={path} depth {previous_depth}->{current_depth} \
                             previous={previous_work} current={current_work}",
                                    if nested_fns { "Fn" } else { "Let" }
                                );
                            }
                        }
                    }
                })
            })
            .expect("#1205 nested callable scope profile thread starts")
            .join()
            .expect("#1205 nested callable scope profile thread completes");
    })
}

#[test]
fn issue_1205_callable_scope_restore_semantics() {
    test_resources()
        .run(|operation| { let mut direct = CallableScope::default();
    let outer = direct.bind("f".to_string(), Some("bc".to_string()));
    let duplicate = direct.bind("f".to_string(), Some("other".to_string()));
    let inner = direct.bind("f".to_string(), None);
    direct.restore([inner, duplicate, outer]);
    assert!(
        direct.get("f").is_none(),
        "restoring repeated shadow bindings must recover the absent outer scope"
    );
    let typed = surf_check(
        "module FrontEndPerformance.ScopeRestore\n\
         def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
         def sibling(c: f32) -> tensor[1, f32] = {\n\
           probe = (fn (bc: f32) -> bc)(c)\n\
           bc(c)\n\
         }\n\
         def nested(c: f32) -> tensor[1, f32] = {\n\
           f = bc\n\
           probe = (fn (f: f32) -> (fn (f: f32) -> f)(f))(c)\n\
           f(c)\n\
         }\n\
         def duplicate(c: f32) -> tensor[1, f32] = {\n\
           f = bc\n\
           f = f\n\
           f(c)\n\
         }\n\
         def sequential(c: f32) -> tensor[1, f32] = {\n\
           f = bc\n\
           g = f\n\
           h = g\n\
           h(c)\n\
         }\n\
         out = index(to_list(sibling(cast(1.0, f32))), 0i64)\n",
    );
    let summaries = cached_tensor_helper_def_summaries(&HostLoweringSession::new(typed.facts(), operation)).expect("valid checked admission facts");
    let defs = cached_program_defs(&HostLoweringSession::new(typed.facts(), operation));
    for (definition, final_callee) in [
        ("sibling", "bc"),
        ("nested", "f"),
        ("duplicate", "f"),
        ("sequential", "h"),
    ] {
        assert_eq!(
            summaries
                .get(definition)
                .map(|summary| summary.reaches_dynamic_to_tensor),
            Some(true),
            "lexical shadow restoration must preserve the final dynamic call in {definition}"
        );
        let body = lookup_program_def(&defs, definition, operation).expect("fixture definition exists");
        let final_app = issue_1205_find_named_app(body, final_callee)
            .expect("fixture contains its final named application");
        let mut facts = UnordMap::new();
        let body_facts = analyze_tensor_helper_preflight(body, &summaries, &mut facts);
        assert!(
            body_facts.reaches_dynamic_to_tensor,
            "per-expression analysis must preserve the final call in {definition}"
        );
        assert!(
            facts[&(final_app as *const Expr as usize)].reaches_dynamic_to_tensor,
            "the final {final_callee} application in {definition} must inherit its dynamic summary"
        );
    } })
}

#[test]
fn issue_1205_preflight_preserves_static_to_tensor_literals() {
    test_resources().run(|operation| {
        let typed = surf_check(
            "module FrontEndPerformance.StaticLiteral\n\
         def lit() -> tensor[2, f32] = \
           to_tensor([cast(1.0, f32), cast(2.0, f32)])\n\
         r = index(to_list(lit()), 0i64)\n",
        );
        fn find_to_tensor(expr: &Expr) -> Option<&Expr> {
            if let Some((DeepTag::App, _, kids)) = stamped_parts(expr)
                && let Some((DeepTag::Var, _, callee_kids)) = kids.first().and_then(stamped_parts)
                && callee_kids.first().and_then(symbol_name) == Some("to_tensor")
            {
                return Some(expr);
            }
            match expr {
                Expr::Node(node, _) => node.children_slice().iter().find_map(find_to_tensor),
                Expr::MetaExpr(meta, _) => find_to_tensor(&meta.expr),
                Expr::BareList(elements, _) => elements.iter().find_map(find_to_tensor),
                Expr::UnknownForm(data) => data.children.iter().find_map(find_to_tensor),
                Expr::Map(map, _) => map.find_expression(|value| find_to_tensor(value)),
                Expr::Atom(_, _) => None,
            }
        }
        let literal = typed
            .exprs()
            .iter()
            .find_map(find_to_tensor)
            .expect("fixture contains a to_tensor application");
        assert!(
            crate::lower::is_static_to_tensor_literal(literal),
            "fixture must exercise the lowerer's literal recognizer"
        );
        assert!(
            !expr_is_runtime_shaped_to_tensor(literal),
            "the preflight must share the lowerer's static-literal exemption"
        );
        let effected = chelis_effects::check_program(&typed).expect("effect check");
        let checked = chelis_types::check_linearity(&effected).expect("linearity check");
        reset_host_work_profile(operation);
        try_lower_compiled_program(checked.facts(), operation)
            .expect("static to_tensor fixture lowers");
        let profile = take_host_work_profile(operation);
        assert_eq!(
            profile.tensor_helper_preflight_rejections, 0,
            "the runtime-shaped preflight must not reject a static literal: {profile:?}"
        );
        assert_eq!(profile.tensor_helper_builtin_load_rejections, 0);
    })
}

#[test]
fn issue_1205_preflight_does_not_confuse_shadowed_operands_with_calls() {
    test_resources().run(|operation| {
        let typed = surf_check(
            "module FrontEndPerformance.ShadowedOperand\n\
         def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
         def twice(bc: tensor[1, f32]) -> tensor[1, f32] = add(bc, bc)\n\
         r = index(to_list(twice(to_tensor([cast(1.0, f32)]))), 0i64)\n",
        );
        let summaries =
            cached_tensor_helper_def_summaries(&HostLoweringSession::new(typed.facts(), operation))
                .expect("valid checked admission facts");
        assert_eq!(
            summaries
                .get("bc")
                .map(|summary| summary.reaches_dynamic_to_tensor),
            Some(true),
            "the top-level bc definition must exercise the transitive summary"
        );

        let add = typed
            .exprs()
            .iter()
            .find_map(|expr| issue_1205_find_named_app(expr, "add"))
            .expect("fixture contains the twice body");
        let mut facts = UnordMap::new();
        let add_facts = analyze_tensor_helper_preflight(add, &summaries, &mut facts);
        assert!(
            !add_facts.reaches_dynamic_to_tensor,
            "a tensor operand named bc is not a call to the top-level bc definition"
        );
    })
}

#[test]
fn issue_1205_preflight_tracks_callable_shadowing_and_aliases() {
    test_resources()
        .run(|operation| { let shadowed = surf_check(
        "module FrontEndPerformance.ShadowedCallable\n\
         def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
         def local(x: tensor[1, f32]) -> tensor[1, f32] = {\n\
           bc = fn (y: tensor[1, f32]) -> add(y, y)\n\
           bc(x)\n\
         }\n\
         def wrapper(x: tensor[1, f32]) -> tensor[1, f32] = local(x)\n\
         out = index(to_list(wrapper(to_tensor([cast(1.0, f32)]))), 0i64)\n",
    );
    let shadowed_summaries =
        cached_tensor_helper_def_summaries(&HostLoweringSession::new(shadowed.facts(), operation)).expect("valid checked admission facts");
    assert!(
        shadowed_summaries["bc"].reaches_dynamic_to_tensor,
        "the top-level bc definition must reach dynamic to_tensor"
    );
    for name in ["local", "wrapper", "out"] {
        assert_eq!(
            shadowed_summaries
                .get(name)
                .map(|summary| summary.reaches_dynamic_to_tensor),
            Some(false),
            "a local callable named bc must shadow the top-level helper in {name}"
        );
    }
    let shadowed_defs = cached_program_defs(&HostLoweringSession::new(shadowed.facts(), operation));
    let local_body = lookup_program_def(&shadowed_defs, "local", operation).expect("local definition");
    let local_call = issue_1205_find_named_app(local_body, "bc").expect("local bc call");
    let mut shadowed_facts = UnordMap::new();
    analyze_tensor_helper_preflight(local_body, &shadowed_summaries, &mut shadowed_facts);
    assert!(
        !shadowed_facts[&(local_call as *const Expr as usize)].reaches_dynamic_to_tensor,
        "the local bc call must not inherit the top-level bc summary"
    );
    let aliased = surf_check(
        "module FrontEndPerformance.AliasedCallable\n\
         def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
         def aliased(c: f32) -> tensor[1, f32] = {\n\
           f = bc\n\
           f(c)\n\
         }\n\
         def wrapper(c: f32) -> tensor[1, f32] = aliased(c)\n\
         out = index(to_list(wrapper(cast(1.0, f32))), 0i64)\n",
    );
    let aliased_summaries =
        cached_tensor_helper_def_summaries(&HostLoweringSession::new(aliased.facts(), operation)).expect("valid checked admission facts");
    for name in ["bc", "aliased", "wrapper", "out"] {
        assert_eq!(
            aliased_summaries
                .get(name)
                .map(|summary| summary.reaches_dynamic_to_tensor),
            Some(true),
            "the alias f = bc must propagate the dynamic helper summary through {name}"
        );
    }
    let aliased_defs = cached_program_defs(&HostLoweringSession::new(aliased.facts(), operation));
    let aliased_body =
        lookup_program_def(&aliased_defs, "aliased", operation).expect("aliased definition");
    let alias_call = issue_1205_find_named_app(aliased_body, "f").expect("local f call");
    let mut aliased_facts = UnordMap::new();
    analyze_tensor_helper_preflight(aliased_body, &aliased_summaries, &mut aliased_facts);
    assert!(
        aliased_facts[&(alias_call as *const Expr as usize)].reaches_dynamic_to_tensor,
        "the local f call must inherit the aliased top-level bc summary"
    );
    let effected = chelis_effects::check_program(&aliased).expect("effect check");
    let checked = chelis_types::check_linearity(&effected).expect("linearity check");
    reset_host_work_profile(operation);
    try_lower_compiled_program(checked.facts(), operation).expect("aliased fixture lowers");
    let profile = take_host_work_profile(operation);
    assert_eq!(
        profile.tensor_helper_builtin_load_rejections, 0,
        "aliased dynamic helpers must be rejected by preflight, not after DAG lowering: {profile:?}"
    ); })
}

#[test]
fn list_entry_preflight_propagates_through_tensor_callers() {
    test_resources().run(|operation| {
        let typed = surf_check(
            "def choose[n](xs: List[tensor[n, f32]], y: tensor[*, f32]) -> tensor[n, f32] = y\n\
         def run(xs: List[tensor[*, f32]], y: tensor[*, f32]) -> tensor[*, f32] = choose(xs, y)\n\
         def outer(xs: List[tensor[*, f32]], y: tensor[*, f32]) -> tensor[*, f32] = run(xs, y)\n\
         def plain(y: tensor[*, f32]) -> tensor[*, f32] = y\n\
         out = outer([], plain(to_tensor([1.0f32, 2.0f32])))\n",
        );
        let session = HostLoweringSession::new(typed.facts(), operation);
        let summaries =
            cached_tensor_helper_def_summaries(&session).expect("valid checked admission facts");
        for name in ["choose", "run", "outer", "out"] {
            assert!(
                summaries[name].reaches_list_entry,
                "{name} reaches the named List entry obligation"
            );
        }
        assert!(!summaries["plain"].reaches_list_entry);
        let (_, body) = session.def_named("outer").expect("checked declaration");
        let mut facts = UnordMap::new();
        assert!(analyze_tensor_helper_preflight(body, &summaries, &mut facts).reaches_list_entry);
    })
}

/// harden-bounded-monomorphization D4: the interning identity is the
/// definition's own name, so a short and a qualified spelling of one def
/// produce one canonical key and one minted symbol per instantiation.
/// chelis#2393: the session's name index must resolve every spelling
/// exactly as the linear scans it replaced, over handcrafted collisions
/// and every program in the repository corpus that checks, and must be
/// built once per session.
///
/// Evidentiary status: DISPOSITION LOCK for resolution (the rule is
/// unchanged) and REGRESSION TEST for the build count.
#[test]
fn issue_2393_name_index_resolves_like_the_scans() {
    test_resources()
        .run(|operation| { let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut sources = vec![
        "module Demo.Collide\n\
         def x(a: f32) -> f32 = a\n\
         def y(a: f32) -> f32 = x(a)\n"
            .to_owned(),
    ];
    let mut stack = ["examples", "packages/chelis-std/src", "tests/corpus"]
        .map(|dir| root.join(dir))
        .to_vec();
    while let Some(dir) = stack.pop() {
        let mut entries = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "ch") {
                sources.push(std::fs::read_to_string(&path).unwrap());
            }
        }
    }
    let mut programs = sources
        .iter()
        .filter_map(|source| {
            let decls =
                chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
                    .ok()?;
            let deep =
                chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
                    .ok()?;
            chelis_types::check_ir_program(&deep).ok()
        })
        .collect::<Vec<_>>();
    // Collisions the corpus lacks: an ambiguous terminal across two
    // spellings, and a defsig beside its def. A duplicate definition is
    // not covered: the checker rejects it, so no session can hold one.
    programs.push(parse_and_check(
        "(defsig {} Demo.depth (a) (t-fn {} (t-var {} a) (t-prim {} i32)))\n\
         (def {} Demo.depth (fn {} (params {} (x {type: (t-var {} a)})) (lit {type: (t-prim {} i32)} 1)))\n\
         (def {} A__twin (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))\n\
         (def {} B.twin (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))\n\
         (def {} C__solo (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))\n",
    ));
    // chelis#3854: an authored signature whose terminal segment is a
    // builtin's name, and two authored signatures sharing a terminal.
    programs.push(parse_and_check(
        "(defsig {} M__add (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n\
         (def {} M__add (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))\n\
         (defsig {} A__pair (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n\
         (def {} A__pair (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))\n\
         (defsig {} B.pair (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n\
         (def {} B.pair (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))\n",
    ));
    let mut compared = 0usize;
    for checked in &programs {
        let session = HostLoweringSession::new(checked.facts(), operation);
        take_top_level_name_index_builds(operation);
        let mut names = top_level_items(checked.exprs())
            .into_iter()
            .filter_map(|item| {
                named_item(item, DeepTag::Def).or_else(|| named_item(item, DeepTag::Defsig))
            })
            .map(|(name, _)| name.to_owned())
            .collect::<Vec<_>>();
        for name in names.clone() {
            let terminal = terminal_name(&name, operation).to_owned();
            names.push(format!("Q__{terminal}"));
            names.push(format!("Q.{terminal}"));
            names.push(terminal);
        }
        // The signature and type-environment keys too, builtins included.
        let mut keyed = checked
            .signature_inference()
            .functions
            .keys()
            .chain(checked.type_env().keys())
            .cloned()
            .collect::<Vec<_>>();
        for name in keyed.clone() {
            let terminal = terminal_name(&name, operation).to_owned();
            keyed.push(format!("Q__{terminal}"));
            keyed.push(terminal);
        }
        names.extend(keyed);
        names.push("definitely_missing".into());
        for name in &names {
            assert_eq!(
                checked_authored_function_signature(&session, name).map(std::ptr::from_ref),
                scan_checked_authored_function_signature(&session, name)
                    .map(std::ptr::from_ref),
                "authored signature resolution of `{name}`"
            );
            if !session.declares_exactly(name) {
                assert_eq!(
                    lookup_session_type_expr(&session, name).map(std::ptr::from_ref),
                    lookup_type_expr(checked.type_env(), name, operation).map(std::ptr::from_ref),
                    "type-environment resolution of `{name}`"
                );
            }
            let indexed = session.def_named(name);
            let scanned = find_top_level_def_named(checked.exprs(), name, operation);
            assert_eq!(
                indexed.map(|(key, body)| (key, body as *const Expr)),
                scanned.map(|(key, body)| (key, body as *const Expr)),
                "def resolution of `{name}`"
            );
            assert_eq!(
                lookup_authored_defsig_type_expr(&session, name),
                scan_authored_defsig_type_expr(checked.exprs(), name, operation),
                "defsig resolution of `{name}`"
            );
            compared += 1;
        }
        assert!(take_top_level_name_index_builds(operation) <= 1);
    }
    assert!(
        programs.len() > 20,
        "only {} programs checked",
        programs.len()
    );
    assert!(compared > 1_000, "compared only {compared} spellings");
    let shadows = programs.last().expect("handcrafted terminal collisions");
    let session = HostLoweringSession::new(shadows.facts(), operation);
    assert!(
        checked_authored_function_signature(&session, "Z__add").is_some(),
        "`M__add` is the only authored `add`"
    );
    assert!(
        checked_authored_function_signature(&session, "pair").is_none(),
        "`pair` is ambiguous between `A__pair` and `B.pair`"
    );
    take_top_level_name_index_builds(operation);
    let collisions = &programs[programs.len() - 2];
    let session = HostLoweringSession::new(collisions.facts(), operation);
    assert_eq!(session.def_named("twin"), None, "`twin` is ambiguous");
    assert_eq!(
        session.def_named("Z.solo").map(|(key, _)| key),
        Some("C__solo")
    );
    assert_eq!(
        session.def_named("depth").map(|(key, _)| key),
        Some("Demo.depth")
    );
    assert!(session.defsig_named("depth").is_some());
    assert_eq!(take_top_level_name_index_builds(operation), 1); })
}

#[test]
fn mono_interning_identity_is_the_definitions_own_name() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            "(defsig {} Demo.depth (a) (t-fn {} (t-var {} a) (t-prim {} i32)))\n\
         (def {} Demo.depth\n\
           (fn {} (params {} (x {type: (t-var {} a)}))\n\
             (lit {type: (t-prim {} i32)} 1)))\n",
        );
        let (canonical_short, _) = find_top_level_def_named(checked.exprs(), "depth", operation)
            .expect("short spelling resolves to the def");
        let (canonical_qualified, _) =
            find_top_level_def_named(checked.exprs(), "Demo.depth", operation)
                .expect("qualified spelling resolves to the def");
        assert_eq!(canonical_short, "Demo.depth");
        assert_eq!(canonical_qualified, "Demo.depth");
        let i32 = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let key = mono_specialization_key(canonical_short, core::slice::from_ref(&i32), &i32);
        assert_eq!(
            key,
            mono_specialization_key(canonical_qualified, core::slice::from_ref(&i32), &i32)
        );
        let symbol = mono_specialization_symbol(canonical_short, &key);
        assert!(symbol.starts_with("Demo.depth__mono_"));
        assert_eq!(symbol.len(), "Demo.depth__mono_".len() + 16);
    })
}

#[test]
fn adt_constructor_layouts_come_from_checked_registry() {
    test_resources().run(|operation| {
        let checked = surf_check(
            r#"
    type Column[n, a] = | Column(tensor[n, a])
    type Hamt[a] = | Leaf { value: a }
    type Frame[n, a] = | Frame { cols: Hamt[Column[n, a]] }
    def singleton[a](value: a) -> Hamt[a] = Leaf { value }
    def from_column[n, a](column: Column[n, a]) -> Frame[n, a] =
      Frame { cols: singleton(column) }
    "#,
        );
        let definitions =
            adt_constructor_definitions(&HostLoweringSession::new(checked.facts(), operation))
                .expect("valid checked metadata");
        assert!(
            definitions
                .iter()
                .all(|definition| definition.adt_name != "Option"),
            "prelude ADTs with dedicated host layouts must not enter the generic constructor table"
        );
        let column = definitions
            .iter()
            .find(|definition| definition.adt_name == "Column")
            .expect("Column constructor metadata");
        assert_eq!(column.parameters.len(), 2);
        assert_eq!(
            column.stored_parameters,
            [false, true],
            "the dimension parameter is erased while the dtype is stored"
        );
        assert!(matches!(
            column.fields[0].term,
            HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                precision: HostPrecisionTerm::Variable(_),
                shape: HostShapeTerm::Concrete(_),
            })
        ));
        let frame = definitions
            .iter()
            .find(|definition| definition.adt_name == "Frame")
            .expect("Frame constructor metadata");
        assert_eq!(
            frame.stored_parameters,
            [false, true],
            "nested Frame -> Hamt -> Column storage classification"
        );
        assert!(
            checked
                .signature_inference()
                .functions
                .get("from_column")
                .is_some_and(|signature| {
                    signature.authored_signature && signature.authored_signature_type.is_some()
                }),
            "checker metadata must record the authored generic signature"
        );
        assert!(
            top_level_fn_is_type_polymorphic(
                &HostLoweringSession::new(checked.facts(), operation),
                "from_column"
            ),
            "the stored dtype behind Frame -> Hamt -> Column must classify from checker metadata"
        );
    })
}

#[test]
fn generalized_unannotated_callback_is_not_classified_as_authored_polymorphism() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "def apply(callback, value) = callback(value)\n\
         def increment(value: i32) -> i32 = add(value, 1)\n\
         def use_callback() -> i32 = apply(increment, 1)\n",
        );
        let apply = checked
            .signature_inference()
            .functions
            .get("apply")
            .expect("generalized callback metadata");
        assert!(!apply.authored_signature);
        assert!(apply.authored_signature_type.is_none());
        assert!(
            !top_level_fn_is_type_polymorphic(
                &HostLoweringSession::new(checked.facts(), operation),
                "apply"
            ),
            "an inferred/generalized callback is not an authored generic ABI"
        );
    })
}

#[test]
fn authored_signature_classifier_requires_exact_or_unambiguous_name() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            r#"
            (defsig {} Left.identity (a)
              (t-fn {} (t-var {} a) (t-var {} a)))
            (def {} Left.identity
              (fn {} (params {} (x {type: (t-var {} a)})) (var {} x)))
            (defsig {} Right.identity (b)
              (t-fn {} (t-var {} b) (t-var {} b)))
            (def {} Right.identity
              (fn {} (params {} (x {type: (t-var {} b)})) (var {} x)))
        "#,
        );
        assert!(
            top_level_fn_is_type_polymorphic(
                &HostLoweringSession::new(checked.facts(), operation),
                "Left.identity"
            ),
            "an exact qualified checker record must classify"
        );
        assert!(
            top_level_fn_is_type_polymorphic(
                &HostLoweringSession::new(checked.facts(), operation),
                "Right.identity"
            ),
            "the other exact qualified checker record must classify"
        );
        assert!(
            !top_level_fn_is_type_polymorphic(
                &HostLoweringSession::new(checked.facts(), operation),
                "identity"
            ),
            "an ambiguous terminal name must not select either qualified checker record"
        );
    })
}

/// chelis#1754 (#2390), moved to check by chelis#2518: a `to_tensor([])`
/// whose element type nothing determines is rejected by the checker, so
/// no checked program exists for lowering or code generation to receive,
/// and no dtype can be chosen for it downstream.
#[test]
fn unresolved_empty_to_tensor_is_rejected_at_check_before_lowering() {
    test_resources().run(|_| {
        let decls = chelis_surf::parser::parse_str(
            "result = numel(to_tensor([]))\n",
            &chelis_syntax_host::NativeStack,
        )
        .expect("surf parse failed");
        let deep = chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let rejected = chelis_types::check_ir_program(&deep)
            .expect_err("an empty tensor without a checked dtype must be rejected before lowering");
        assert!(
            rejected.errors.iter().any(|error| {
                error
                    .message
                    .contains("`to_tensor` admits only some operand types")
                    && error.message.contains("never determined within `result`")
            }),
            "{:?}",
            rejected.errors
        );
    })
}

/// The code-generation boundary still rejects a checked program whose host
/// type never resolved: an empty list nothing gives an element type
/// passes check and must not reach codegen with an invented type.
#[test]
fn unresolved_empty_list_rejects_at_the_code_generation_boundary() {
    test_resources().run(|operation| {
        let checked = surf_check("values = []\n");
        let error = try_lower_compiled_program(checked.facts(), operation)
            .expect_err("a host type that never resolved must not reach codegen");
        assert!(
            error
                .message
                .contains("host type did not resolve before the code-generation boundary"),
            "{error:?}"
        );
        assert!(
            error.message.contains("unresolved host inference variable"),
            "{error:?}"
        );
    })
}

/// chelis#336: restore the guard that a function-typed parameter survives
/// as `HostTypeTerm::Fn` through `try_lower_compiled_program`. #331 deleted the
/// only test asserting this (it relied on a reef-package HOF). A
/// *same-module* HOF like `def apply(f, x) = f(x)` is inlined away, so a
/// surviving fn-param needs a non-inlined HOF: here `apply_each` is kept
/// as a real host function because it is called with a fn argument and
/// uses the `map` combinator. The lowered `apply_each` must keep its
/// `f: f32 -> f32` parameter typed `HostTypeTerm::Fn`, not collapsed to a
/// value type.
#[test]
fn host_program_preserves_fn_typed_param_through_lowering() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "def apply_each(f: f32 -> f32, xs: List[f32]) -> List[f32] = map(fn (v: f32) -> f(v), xs)\n\
         def double(x: f32) -> f32 = mul(x, 2.0)\n\
         out = apply_each(double, [1.0, 2.0, 3.0])\n",
    );
    let compiled = try_lower_compiled_program(checked.facts(), operation).expect("fixture must lower");
    let host = compiled.host.expect("host program present");
    let apply_each = host
        .functions
        .iter()
        .find(|f| f.name == "apply_each")
        .expect("apply_each must survive as a host function (not inlined)");
    let f_param = apply_each
        .params
        .iter()
        .find(|p| p.name == "f")
        .expect("apply_each must keep its `f` parameter");
    match &f_param.ty {
        ConcreteHostType::Function(params, ret) => {
            assert_eq!(params.len(), 1, "f takes one scalar arg, got {params:?}");
            assert!(
                matches!(params[0], ConcreteHostType::Scalar(Prim::F32 | Prim::F64)),
                "f's arg must be a scalar, got {:?}",
                params[0]
            );
            assert!(
                matches!(**ret, ConcreteHostType::Scalar(Prim::F32 | Prim::F64)),
                "f's return must be a scalar, got {ret:?}"
            );
        }
        other => {
            panic!("fn-typed param `f` must lower to ConcreteHostType::Function, got {other:?}")
        }
    } })
}

/// chelis#336 negative parity: a program with no higher-order parameter
/// must produce no `HostTypeTerm::Fn` parameter, so the positive guard above
/// is not vacuously satisfied by some unrelated fn-typed param.
#[test]
fn host_program_has_no_fn_typed_param_without_hof() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "def double(x: f32) -> f32 = mul(x, 2.0)\n\
         out = double(2.0)\n",
        );
        let compiled =
            try_lower_compiled_program(checked.facts(), operation).expect("fixture must lower");
        let host = compiled.host.expect("host program present");
        let has_fn_param = host
            .functions
            .iter()
            .flat_map(|f| f.params.iter())
            .any(|p| matches!(p.ty, ConcreteHostType::Function(_, _)));
        assert!(
            !has_fn_param,
            "a non-HOF program must not lower any HostTypeTerm::Fn parameter"
        );
    })
}

// ── Issue #308: scalar_to_tensor operand-precision plumbing ──
//
// `HostTypeTerm::Float64` collapses f32 and f64, so the Deep-level
// `infer_app_expr_host_type` must recover the operand precision
// before the coarse arg-ty fallback erases it to f32.

fn parse_deep_app(src: &str) -> Expr {
    chelis_deep::parser::parse_str(src, &chelis_syntax_host::NativeStack)
        .expect("parse failed")
        .into_iter()
        .next()
        .expect("one expr")
}

fn infer_scalar_to_tensor_host_type(
    arg_src: &str,
    operation: &Operation<'_, chelis_syntax_host::NativeStack>,
) -> Option<HostTypeTerm> {
    let app = parse_deep_app(&format!("(app {{}} (var {{}} scalar_to_tensor) {arg_src})"));
    // Empty checked program: the arm under test must not depend on
    // program-level lookups for the precision recovery.
    let program = surf_check("unrelated = 1\n");
    infer_app_expr_host_type(
        &app,
        &HostLoweringSession::new(program.facts(), operation),
        &HostScope::new(),
    )
    .expect("valid checked metadata")
}

#[test]
fn scalar_to_tensor_infers_f64_from_cast_target() {
    test_resources().run(|operation| {
        // The #308 reproducer shape: `scalar_to_tensor(cast(1.1, f64))`
        // with NO checker annotation on the app or the cast — the cast
        // target alone must drive the result precision.
        let inferred = infer_scalar_to_tensor_host_type(
            "(cast {} (lit {type: (t-prim {} f32)} 1.1) (t-prim {} f64))",
            operation,
        );
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F64,
            })),
            "scalar_to_tensor(cast(_, f64)) must infer a rank-0 f64 tensor",
        );
    })
}

#[test]
fn scalar_to_tensor_infers_f64_from_checker_annotation() {
    test_resources().run(|operation| {
        let inferred =
            infer_scalar_to_tensor_host_type("(var {type: (t-prim {} f64)} c)", operation);
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F64,
            })),
            "scalar_to_tensor of an f64-annotated operand must infer a rank-0 f64 tensor",
        );
    })
}

#[test]
fn scalar_to_tensor_keeps_f32_default_for_f32_operand() {
    test_resources().run(|operation| {
        // Negative parity (pins the #300/#306 f32 path): an f32 operand
        // — whether via cast target or bare default literal — must keep
        // the rank-0 f32 result so `chelis_scalar_tensor_from_f32`
        // storage and the consuming helper's f32 read stay paired.
        for arg_src in [
            "(cast {} (lit {type: (t-prim {} f32)} 2.5) (t-prim {} f32))",
            "(lit {type: (t-prim {} f32)} 2.5)",
        ] {
            let inferred = infer_scalar_to_tensor_host_type(arg_src, operation);
            assert_eq!(
                inferred,
                Some(HostTypeTerm::Tensor(TensorType {
                    dims: vec![],
                    precision: Prim::F32,
                })),
                "scalar_to_tensor({arg_src}) must keep the f32 default",
            );
        }
    })
}

#[test]
fn scalar_to_tensor_float_recovery_does_not_hijack_integer_operands() {
    test_resources().run(|operation| {
        // Negative parity: the #308 float-precision recovery must not
        // claim integer operands. At this Deep-expr layer a
        // cast-wrapped int infers `None` (pre-#308 behavior:
        // `expr_host_type` does not see through `cast`, so the coarse
        // fallback gets `Unknown` and abstains); the Int64 tensor
        // typing happens post-lowering via `host_expr_type` on the
        // lowered operand and the `infer_builtin_host_type_from_arg_tys`
        // Int64 arm.
        let inferred = infer_scalar_to_tensor_host_type(
            "(cast {} (lit {type: (t-prim {} i32)} 3) (t-prim {} i32))",
            operation,
        );
        assert_eq!(
            inferred, None,
            "integer operands must fall through unchanged (no float hijack)",
        );
        // The coarse arm still owns the lowered-lane integer answer.
        assert_eq!(
            infer_builtin_host_type_from_arg_tys("scalar_to_tensor", &[HostTypeTerm::Int64]),
            Ok(Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::Int64,
            }))),
        );
    })
}

fn rank_zero_key() -> HostTypeTerm {
    HostTypeTerm::Tensor(TensorType {
        dims: vec![],
        precision: Prim::Key,
    })
}

fn named_f32_vector(name: &str) -> HostTypeTerm {
    HostTypeTerm::Tensor(TensorType {
        dims: vec![DimInfo::Named(name.to_owned(), None)],
        precision: Prim::F32,
    })
}

/// [05-OP-8] and [05-OP-37]: a draw's result is its data operand's type.
/// A key projected from `split_key` is a rank-zero key tensor, which
/// precedes the data operand, so it must never be taken as the result.
#[test]
fn draw_result_types_follow_the_data_operand_not_the_key() {
    test_resources().run(|_| {
        let scalar_key = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Key));
        for key in [rank_zero_key(), scalar_key] {
            assert_eq!(
                infer_builtin_host_type_from_arg_tys(
                    "uniform_like",
                    &[
                        key.clone(),
                        named_f32_vector("n"),
                        HostTypeTerm::Float32,
                        HostTypeTerm::Float32,
                    ]
                ),
                Ok(Some(named_f32_vector("n"))),
                "uniform_like keyed by {key:?}",
            );
            assert_eq!(
                infer_builtin_host_type_from_arg_tys(
                    "dropout",
                    &[key.clone(), named_f32_vector("n"), HostTypeTerm::Float32]
                ),
                Ok(Some(named_f32_vector("n"))),
                "dropout keyed by {key:?}",
            );
            // A scalar template draws a scalar of the template's type.
            assert_eq!(
                infer_builtin_host_type_from_arg_tys(
                    "uniform_like",
                    &[
                        key.clone(),
                        HostTypeTerm::Float32,
                        HostTypeTerm::Float32,
                        HostTypeTerm::Float32,
                    ]
                ),
                Ok(Some(HostTypeTerm::Float32)),
                "scalar uniform_like keyed by {key:?}",
            );
        }
    })
}

/// The operand-derived application type may name dimensions the checker
/// left synthetic, never change the checker's element dtype.
#[test]
fn operand_inference_names_synthetic_dims_without_changing_the_dtype() {
    test_resources().run(|_| {
        let checked = named_f32_vector("d47");
        assert!(
            should_prefer_inferred_app_type(&checked, &named_f32_vector("n")),
            "a same-dtype operand type names the synthetic dimension",
        );
        assert!(
            !should_prefer_inferred_app_type(&checked, &rank_zero_key()),
            "a key operand's type must not displace a checked f32 draw",
        );
        assert!(
            !should_prefer_inferred_app_type(
                &checked,
                &HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Named("n".to_owned(), None)],
                    precision: Prim::F64,
                }),
            ),
            "an operand of another float dtype must not displace the checked dtype",
        );
    })
}

// ── chelis#631: host-lane concat result typing ──
//
// The host lane must never carry the ELEMENT's extent on the concat
// axis: `chelis_tensor_concat` sizes that axis at run time, and a
// baked element extent aborts the binary at the runtime-dim guard.

fn rank2_element() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(1), DimInfo::Lit(2)],
        precision: Prim::F32,
    }
}

fn anon_dim() -> DimInfo {
    DimInfo::Named("*".to_string(), None)
}

fn infer_concat_host_type(
    axis_src: &str,
    operation: &Operation<'_, chelis_syntax_host::NativeStack>,
) -> Option<HostTypeTerm> {
    let app = parse_deep_app(&format!(
        "(app {{}} (var {{}} concat) (var {{}} rows) {axis_src})"
    ));
    let program = surf_check("unrelated = 1\n");
    let mut scope = HostScope::new();
    scope.insert(
        "rows".to_string(),
        HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
    );
    infer_app_expr_host_type(
        &app,
        &HostLoweringSession::new(program.facts(), operation),
        &scope,
    )
    .expect("valid checked metadata")
}

#[test]
fn concat_app_expr_host_type_wildcards_concat_axis_only() {
    test_resources().run(|operation| {
        // Literal axis 0 (bare and cast-wrapped, the canonical spelling):
        // the concat axis is anon, the trailing element extent survives.
        for axis_src in ["(lit {} 0)", "(cast {} (lit {} 0) (t-prim {} i32))"] {
            let inferred = infer_concat_host_type(axis_src, operation);
            assert_eq!(
                inferred,
                Some(HostTypeTerm::Tensor(TensorType {
                    dims: vec![anon_dim(), DimInfo::Lit(2)],
                    precision: Prim::F32,
                })),
                "concat(rows, {axis_src}) must wildcard only axis 0",
            );
        }
    })
}

#[test]
fn concat_app_expr_host_type_negative_axis_indexes_from_end() {
    test_resources().run(|operation| {
        let inferred = infer_concat_host_type("(lit {} -1)", operation);
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(1), anon_dim()],
                precision: Prim::F32,
            })),
            "concat(rows, -1) must wildcard the LAST axis and keep axis 0",
        );
    })
}

#[test]
fn concat_app_expr_host_type_unknown_axis_wildcards_all_axes() {
    test_resources().run(|operation| {
        // Negative parity: a non-literal axis leaves no per-axis claim.
        let inferred = infer_concat_host_type("(var {} ax)", operation);
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), anon_dim()],
                precision: Prim::F32,
            })),
            "a runtime concat axis must wildcard every axis",
        );
    })
}

#[test]
fn concat_arg_tys_fallback_wildcards_all_axes() {
    test_resources().run(|_| {
        // The arg-tys-only lane cannot see the axis value: pre-chelis#631
        // it returned the element type VERBATIM (dims [1, 2]) — the baked
        // extent that aborted guarded forward binaries.
        let inferred = infer_builtin_host_type_from_arg_tys(
            "concat",
            &[
                HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
                HostTypeTerm::Int64,
            ],
        );
        assert_eq!(
            inferred,
            Ok(Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), anon_dim()],
                precision: Prim::F32,
            }))),
        );
    })
}

#[test]
fn tuple_projection_retains_a_typed_literal_index() {
    test_resources().run(|_| {
        let pair = HostExpr::new(HostExprKind::Var(
            "pair".to_string(),
            HostTypeTerm::Tuple(vec![
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32)),
                HostTypeTerm::Bool,
            ]),
        ));
        for index in [1, -1, 2] {
            let index_expr = HostExpr::new(HostExprKind::Builtin {
                name: "cast".into(),
                args: vec![HostExpr::new(HostExprKind::Int(index))],
                ty: HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32)),
            });
            let result = infer_builtin_host_type("tuple-get", &[pair.clone(), index_expr])
                .expect("valid checked metadata");
            if index == 1 {
                assert_eq!(result, Some(HostTypeTerm::Bool));
            } else {
                assert!(result.is_some_and(|ty| ty.is_unresolved()));
            }
        }
    })
}

#[test]
fn concat_host_expr_lane_peels_cast_wrapped_axis() {
    test_resources().run(|_| {
        // infer_builtin_host_type sees HostExpr args: a cast-wrapped int
        // axis still selects the single concat axis.
        let list_arg = HostExpr {
            kind: HostExprKind::Var(
                "rows".to_string(),
                HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
            ),
            span_id: None,
            merged_spans: Vec::new(),
        };
        let axis_arg = HostExpr {
            kind: HostExprKind::Builtin {
                name: "cast".into(),
                args: vec![HostExpr {
                    kind: HostExprKind::Int(0),
                    span_id: None,
                    merged_spans: Vec::new(),
                }],
                ty: HostTypeTerm::Int64,
            },
            span_id: None,
            merged_spans: Vec::new(),
        };
        let inferred = infer_builtin_host_type("concat", &[list_arg, axis_arg])
            .expect("valid checked metadata");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), DimInfo::Lit(2)],
                precision: Prim::F32,
            })),
        );
    })
}

/// E2 (WS-A0 RT-1 fixup): the `to_list` host classification arm
/// returns a typed error for f8e4m3 rather than panicking or silently
/// classifying the deferred dtype as Float64.
#[test]
fn to_list_classifier_rejects_f8e4m3_tensor_per_spec_1_1_1() {
    test_resources().run(|_| {
        let f8_tensor = HostTypeTerm::Tensor(chelis_dag_core::dag::TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F8e4m3,
        });
        let error = infer_builtin_host_type_from_arg_tys("to_list", &[f8_tensor])
            .expect_err("the deferred dtype must be a typed inference error");
        assert!(error.to_string().contains("f8e4m3 is deferred"));
    })
}

#[test]
fn invalid_builtin_shape_is_not_an_inference_variable() {
    test_resources().run(|_| {
        let error = infer_builtin_host_type_from_arg_tys(
            "sum",
            &[HostTypeTerm::List(Box::new(HostTypeTerm::Int64))],
        )
        .expect_err("sum over a list is invalid, not underconstrained");
        assert_eq!(error.builtin, "sum");
        assert!(error.requirement.contains("must be a tensor"));
    })
}

#[test]
fn genuinely_underconstrained_tuple_get_keeps_a_fresh_identity() {
    test_resources().run(|_| {
        let first = infer_builtin_host_type_from_arg_tys(
            "tuple-get",
            &[HostTypeTerm::Tuple(vec![HostTypeTerm::Int64])],
        )
        .expect("the arg-types-only lane lacks the index, but is not invalid")
        .expect("tuple-get is a known builtin");
        let second = infer_builtin_host_type_from_arg_tys(
            "tuple-get",
            &[HostTypeTerm::Tuple(vec![HostTypeTerm::Int64])],
        )
        .expect("the arg-types-only lane lacks the index, but is not invalid")
        .expect("tuple-get is a known builtin");
        assert!(matches!(first, HostTypeTerm::InferenceVariable(_)));
        assert!(matches!(second, HostTypeTerm::InferenceVariable(_)));
        assert_ne!(first, second, "fresh holes must retain distinct identity");
    })
}

#[test]
fn named_tensor_entry_inserts_copy_for_consuming_fanout() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            r#"
            (def {} consume
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                (realize {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                  (var {} x))))
            (def {} double_it
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                  (var {} add)
                  (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                    (var {} consume)
                    (var {} x))
                  (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                    (var {} consume)
                    (var {} x)))))
        "#,
        );
        let dag = lower_named_tensor_entry_dag(checked.facts(), "double_it", operation)
            .expect("valid checked metadata")
            .expect("lower entry");
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1, "{:?}", dag.nodes());
    })
}

#[test]
fn named_entry_dag_accepts_scalar_numeric_params() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            r#"
            (def {} add_scalar
              (fn {type: (t-fn {}
                            (t-prim {} f32)
                            (t-prim {} f32)
                            (t-prim {} f32))}
                (params {}
                  (x {type: (t-prim {} f32)})
                  (y {type: (t-prim {} f32)}))
                (app {type: (t-prim {} f32)}
                  (var {} add)
                  (var {} x)
                  (var {} y))))
        "#,
        );

        let dag = lower_named_tensor_entry_dag(checked.facts(), "add_scalar", operation)
            .expect("valid checked metadata")
            .expect("lower scalar entry");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();

        assert_eq!(root.op, RiscOp::Add, "{:?}", dag.nodes());
        let scalar_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F32
            })
            .count();
        assert_eq!(scalar_loads, 2, "{:?}", dag.nodes());
    })
}

// ── WS-4: declared float width survives entry-param lowering ──
//
// `tensor_type_from_host_input` previously mapped every
// `HostTypeTerm::Float64` (which collapsed both f32 and f64) to
// `Prim::F32`, silently truncating an f64 scalar entry parameter to
// 4-byte storage. The declared `f32`/`f64` width now threads through
// `parse_host_type_with_subst` so each lowers to a `Load` of its own
// precision.

#[test]
fn named_entry_dag_lowers_f64_scalar_params_to_f64_loads() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            r#"
            (def {} add_scalar
              (fn {type: (t-fn {}
                            (t-prim {} f64)
                            (t-prim {} f64)
                            (t-prim {} f64))}
                (params {}
                  (x {type: (t-prim {} f64)})
                  (y {type: (t-prim {} f64)}))
                (app {type: (t-prim {} f64)}
                  (var {} add)
                  (var {} x)
                  (var {} y))))
        "#,
        );

        let dag = lower_named_tensor_entry_dag(checked.facts(), "add_scalar", operation)
            .expect("valid checked metadata")
            .expect("lower f64 scalar entry");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();
        assert_eq!(root.op, RiscOp::Add, "{:?}", dag.nodes());

        let f64_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F64
            })
            .count();
        assert_eq!(
            f64_loads,
            2,
            "an f64 scalar entry param must lower to an f64 Load, not f32: {:?}",
            dag.nodes()
        );
    })
}

#[test]
fn named_entry_dag_does_not_downgrade_f64_scalar_params_to_f32() {
    test_resources().run(|operation| {
        // Negative twin: the historical bug. NO scalar Load produced for
        // a declared-f64 entry param may carry `Prim::F32`.
        let checked = parse_and_check(
            r#"
            (def {} add_scalar
              (fn {type: (t-fn {}
                            (t-prim {} f64)
                            (t-prim {} f64)
                            (t-prim {} f64))}
                (params {}
                  (x {type: (t-prim {} f64)})
                  (y {type: (t-prim {} f64)}))
                (app {type: (t-prim {} f64)}
                  (var {} add)
                  (var {} x)
                  (var {} y))))
        "#,
        );

        let dag = lower_named_tensor_entry_dag(checked.facts(), "add_scalar", operation)
            .expect("valid checked metadata")
            .expect("lower f64 scalar entry");
        let f32_scalar_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F32
            })
            .count();
        assert_eq!(
            f32_scalar_loads,
            0,
            "an f64 scalar entry param must NOT be silently downgraded to an f32 Load: {:?}",
            dag.nodes()
        );
    })
}

#[test]
fn parse_host_type_preserves_declared_float_width() {
    test_resources().run(|_| {
        // Unit-level pin on the precision-threading site: the `t-prim`
        // syntax for f32 and f64 must map to distinct host scalar
        // variants so the entry-param boundary can recover the width.
        let f32_ty = parse_deep_app("(t-prim {} f32)");
        assert_eq!(
            decode_host_type_with_subst(&f32_ty, &UnordMap::new()),
            Ok(HostTypeTerm::Float32),
        );

        let f64_ty = parse_deep_app("(t-prim {} f64)");
        assert_eq!(
            decode_host_type_with_subst(&f64_ty, &UnordMap::new()),
            Ok(HostTypeTerm::Float64),
        );
    })
}

#[test]
fn host_input_tensor_round_trip_preserves_every_active_scalar_dtype() {
    test_resources().run(|_| {
        for prim in [
            Prim::F16,
            Prim::Bf16,
            Prim::F32,
            Prim::F64,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ] {
            let host = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(prim));
            let tensor = TensorType {
                dims: vec![],
                precision: prim,
            };
            assert_eq!(
                tensor_type_from_host_input(&host),
                Some(tensor.clone()),
                "{prim:?}"
            );
            assert_eq!(host_type_from_tensor_input(&tensor), host, "{prim:?}");
        }
    })
}

#[test]
fn host_input_tensor_boundary_rejects_non_tensor_scalar_domains() {
    test_resources().run(|_| {
        assert_eq!(tensor_type_from_host_input(&HostTypeTerm::String), None);
        assert_eq!(
            tensor_type_from_host_input(&HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                Prim::F8e4m3
            ))),
            None
        );
    })
}

#[test]
fn call_graph_recursion_detection_does_not_mark_every_function_recursive() {
    test_resources().run(|_| {
        let graph = BTreeMap::from([
            ("plain".to_string(), BTreeSet::from(["leaf".to_string()])),
            ("leaf".to_string(), BTreeSet::new()),
            (
                "self_rec".to_string(),
                BTreeSet::from(["self_rec".to_string()]),
            ),
            ("mut_a".to_string(), BTreeSet::from(["mut_b".to_string()])),
            ("mut_b".to_string(), BTreeSet::from(["mut_a".to_string()])),
        ]);

        let recursive = recursive_top_level_fn_names_from_graph(&graph);

        assert!(
            !recursive.contains("plain"),
            "ordinary top-level callers must not be classified as recursive"
        );
        assert!(
            !recursive.contains("leaf"),
            "leaf functions must not be classified as recursive"
        );
        assert!(
            recursive.contains("self_rec"),
            "direct self-recursion must still be detected"
        );
        assert!(
            recursive.contains("mut_a") && recursive.contains("mut_b"),
            "mutual recursion must still be detected"
        );
    })
}

/// The per-question walk the derived call-graph facts replace: whether one
/// or more calls from `start` reach a name in `targets`.
fn reference_reaches_any(graph: &CallGraph, start: &str, targets: &BTreeSet<String>) -> bool {
    let mut visited = BTreeSet::new();
    let mut stack = graph
        .get(start)
        .into_iter()
        .flat_map(|callees| callees.iter().cloned())
        .collect::<Vec<_>>();
    while let Some(name) = stack.pop() {
        if targets.contains(&name) {
            return true;
        }
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(next) = graph.get(&name) {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

#[test]
fn derived_call_graph_facts_answer_every_four_function_graph_as_the_walks_did() {
    test_resources().run(|_| {
        // Every graph on four functions, self-calls included: 2^16 graphs, so
        // the derived facts are checked on every shape of cycle, chain and
        // fan the per-question walks could see at this size.
        let names = ["a", "b", "c", "d"].map(str::to_string);
        for edges in 0u32..(1 << 16) {
            let graph: CallGraph = names
                .iter()
                .enumerate()
                .map(|(from, name)| {
                    let callees = names
                        .iter()
                        .enumerate()
                        .filter(|(to, _)| edges & (1 << (from * 4 + to)) != 0)
                        .map(|(_, callee)| callee.clone())
                        .collect();
                    (name.clone(), callees)
                })
                .collect();
            let recursive = names
                .iter()
                .filter(|name| {
                    reference_reaches_any(&graph, name, &BTreeSet::from([(*name).clone()]))
                })
                .cloned()
                .collect::<BTreeSet<_>>();
            assert_eq!(
                recursive_top_level_fn_names_from_graph(&graph),
                recursive,
                "recursive functions of graph {edges:#06x}"
            );
            let facts = derive_call_graph_facts(&graph);
            let reaches = names
                .iter()
                .filter(|name| reference_reaches_any(&graph, name, &recursive))
                .cloned()
                .collect::<BTreeSet<_>>();
            assert_eq!(
                facts.reaches_recursive, reaches,
                "functions reaching recursion in graph {edges:#06x}"
            );
            let called_by_another = names
                .iter()
                .filter(|name| {
                    graph
                        .iter()
                        .any(|(caller, callees)| caller != *name && callees.contains(*name))
                })
                .cloned()
                .collect::<BTreeSet<_>>();
            assert_eq!(
                facts.called_by_another, called_by_another,
                "functions called by another in graph {edges:#06x}"
            );
        }
    })
}

#[test]
fn recursion_detection_handles_a_hundred_thousand_function_chain() {
    test_resources().run(|_| {
        // The iterative walk holds a call chain far deeper than a thread's
        // stack, and finds the one cycle at its end.
        let length = 100_000;
        let mut graph = (0..length)
            .map(|index| {
                (
                    format!("f{index}"),
                    BTreeSet::from([format!("f{}", index + 1)]),
                )
            })
            .collect::<CallGraph>();
        graph.insert(format!("f{length}"), BTreeSet::from(["f0".to_string()]));
        assert_eq!(
            recursive_top_level_fn_names_from_graph(&graph).len(),
            length + 1
        );
        graph.insert(format!("f{length}"), BTreeSet::new());
        assert!(recursive_top_level_fn_names_from_graph(&graph).is_empty());
    })
}

#[test]
fn tensor_helper_exp_body_keeps_exp_root() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            r#"
            (def {} softplus
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 4) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 4) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
                (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
                  (var {} exp)
                  (var {} x))))
        "#,
        );
        let defs = collect_program_defs(checked.exprs());
        let body = lookup_program_def(&defs, "softplus", operation).unwrap();
        let fn_node = as_node(body).unwrap();
        let body = fn_node.children_slice().get(1).unwrap();
        let mut scope = HostScope::new();
        scope.insert(
            "x".to_string(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            }),
        );
        let expected = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let session = HostLoweringSession::new(checked.facts(), operation);
        let context =
            cached_subexpr_lowering_context(&session).expect("valid checked lowering context");
        let dag = lower_tensor_helper_dag(body, &session, &scope, &expected, &context, None)
            .expect("valid tensor helper lowering")
            .expect("helper dag");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();
        assert_eq!(root.op, RiscOp::Exp, "{:?}", dag.nodes());
    })
}

#[test]
fn tensor_helper_hoists_host_lane_tensor_args_with_f32_type() {
    test_resources().run(|operation| {
        let checked = parse_and_check(
            r#"
            (defsig {}
              jac_row
              (n)
              (t-fn {}
                (t-fn {}
                  (t-tensor {} (d-var {} n) (t-prim {} f32))
                  (t-prim {} f32)
                  (t-prim {} f32)
                  (t-prim {} f32))
                (t-tensor {} (d-var {} n) (t-prim {} f32))
                (t-prim {} f32)
                (t-prim {} f32)
                (t-tensor {} (d-var {} n) (t-prim {} f32))))
            (def {}
              jac_row
              (fn {}
                (params {}
                  (model {type: (t-fn {}
                                   (t-tensor {} (d-var {} n) (t-prim {} f32))
                                   (t-prim {} f32)
                                   (t-prim {} f32)
                                   (t-prim {} f32))})
                  (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                  (x {type: (t-prim {} f32)})
                  (y {type: (t-prim {} f32)}))
                (let {}
                  (bind {}
                    target
                    (fn {}
                      (params {}
                        (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                      (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                  (app {}
                    (grad {wrt: (var {} theta_local)}
                      (var {} target)
                      (lit {type: (t-prim {} i32)} 0))
                    (var {} theta)))))
            (defsig {}
              lm_model
              (t-fn {}
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                (t-prim {} f32)
                (t-prim {} f32)
                (t-prim {} f32)))
            (def {}
              lm_model
              (fn {}
                (params {}
                  (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                  (x {type: (t-prim {} f32)})
                  (y {type: (t-prim {} f32)}))
                (let {}
                  (bind {}
                    y_hat
                    (if {}
                      (app {}
                        (var {} lt)
                        (var {} x)
                        (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                      (app {}
                        (var {} tensor_to_scalar)
                        (app {}
                          (var {} sum)
                          (copy {} (var {} theta))
                          (lit {type: (t-prim {} i32)} 0)))
                      (app {}
                        (var {} add)
                        (app {}
                          (var {} tensor_to_scalar)
                          (app {}
                            (var {} sum)
                            (copy {} (var {} theta))
                            (lit {type: (t-prim {} i32)} 0)))
                        (var {} x))))
                  (app {} (var {} sub) (var {} y) (var {} y_hat)))))
            (def {}
              out
              (app {}
                (var {} jac_row)
                (var {} lm_model)
                (app {}
                  (var {} to_tensor)
                  (app {}
                    (var {} Cons)
                    (lit {type: (t-prim {} f32)} 1.0)
                    (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
        "#,
        );
        let lowered = top_level_lowering_map(checked.exprs(), checked.type_env(), operation)
            .expect("checked fixture classification");
        let host = lower_host_program(
            &HostLoweringSession::new(checked.facts(), operation),
            &lowered,
        )
        .expect("host program must lower");
        let out_binding = host
            .globals
            .iter()
            .find(|binding| binding.name == "out")
            .expect("out host binding");
        let (bindings, body) = match &out_binding.value.kind {
            HostExprKind::RetainedInvocation { bindings, body, .. } => (bindings, body.as_ref()),
            other => {
                panic!("expected out to retain its prepared invocation boundary, got {other:?}")
            }
        };
        let actual_binding = bindings
            .iter()
            .find(|binding| {
                binding.name.starts_with("__chelis_entry_actual_")
                    && matches!(
                        &binding.ty,
                        HostTypeTerm::Tensor(tensor) if tensor.precision == Prim::F32
                    )
            })
            .unwrap_or_else(|| {
                panic!(
                    "prepared f32 tensor actual missing; retained bindings were {:?}",
                    bindings
                        .iter()
                        .map(|binding| (&binding.name, &binding.ty))
                        .collect::<Vec<_>>()
                )
            });
        match &actual_binding.ty {
            HostTypeTerm::Tensor(tensor) => assert_eq!(tensor.precision, Prim::F32),
            other => panic!("expected prepared theta actual to be tensor-typed, got {other:?}"),
        }
        let formal_binding = bindings
            .iter()
            .find(|binding| match &binding.value.kind {
                HostExprKind::FormalIngress { value, ty } => {
                    matches!(
                        ty,
                        HostTypeTerm::Tensor(tensor) if tensor.precision == Prim::F32
                    ) && matches!(
                        &value.kind,
                        HostExprKind::Var(name, HostTypeTerm::Tensor(tensor))
                            if name == &actual_binding.name && tensor.precision == Prim::F32
                    )
                }
                _ => false,
            })
            .unwrap_or_else(|| {
                panic!(
                    "f32 formal ingress for prepared actual `{}` missing: {:?}",
                    actual_binding.name,
                    bindings
                        .iter()
                        .map(|binding| (&binding.name, &binding.value.kind))
                        .collect::<Vec<_>>()
                )
            });
        let scoped_body = match &body.kind {
            HostExprKind::ResultClaimScope { plan, body, .. } => {
                assert_eq!(plan.axes().len(), 1);
                match &plan.axes()[0] {
                    (
                        0,
                        HostResultRequirementPlan::NamedDirect {
                            claim,
                            source,
                            prepared,
                            axis,
                        },
                    ) => {
                        assert_eq!(claim, "n");
                        assert_eq!(source, "theta");
                        assert_eq!(prepared, &formal_binding.name);
                        assert_eq!(*axis, 0);
                    }
                    other => panic!("expected named result obligation on theta, got {other:?}"),
                }
                body.as_ref()
            }
            other => panic!("expected retained result-claim scope, got {other:?}"),
        };
        let helper_index = match &scoped_body.kind {
            HostExprKind::TensorCall { helper, .. } => *helper,
            other => panic!("expected retained body to call tensor helper, got {other:?}"),
        };
        let helper = host
            .global_tensor_helpers
            .get(helper_index)
            .expect("helper index in range");
        assert!(
            helper
                .inputs
                .iter()
                .any(|input| input.name == formal_binding.name && input.ty.precision == Prim::F32),
            "helper inputs should reference the f32 formal-ingress binding: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
        assert!(
            helper.inputs.iter().all(|input| input.name != "to_tensor"),
            "helper inputs must not contain raw `to_tensor` load: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
    })
}

#[test]
fn host_type_parser_unwraps_borrowed_tensor_refs() {
    test_resources().run(|_| {
        let expr = chelis_deep::parser::parse_str(
            "(t-ref {} (t-tensor {} (d-name {} n) (t-prim {} f32)))",
            &chelis_syntax_host::NativeStack,
        )
        .expect("parse ref type")
        .into_iter()
        .next()
        .expect("one type expr");

        match decode_host_type(&expr).expect("borrowed tensor type decodes") {
            HostTypeTerm::Tensor(tensor) => {
                assert_eq!(
                    tensor.dims,
                    vec![chelis_dag_core::dag::DimInfo::Named("n".into(), None)]
                );
                assert_eq!(tensor.precision, Prim::F32);
            }
            other => panic!("expected borrowed tensor ref to parse as tensor, got {other:?}"),
        }
    })
}

#[test]
fn helper_dimension_rebinding_boundary_has_a_nonidentity_discriminator() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, TensorType};

        let symbolic = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let concrete = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let mut before = Dag::new();
        let before_decl = before.declare("test");
        let root = before.add_node(
            before_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            symbolic,
            None,
        );
        before.add_root(root);
        let mut scope = HostScope::new();
        scope.insert("x".into(), HostTypeTerm::Tensor(concrete.clone()));

        let after = remap_tensor_helper_dim_symbols(&before, &scope, &concrete)
            .expect("remap actualizes the helper");
        assert_ne!(
            bincode::serialize(&before).unwrap(),
            bincode::serialize(&after).unwrap(),
            "the actual host boundary must rewrite the symbolic helper"
        );
        assert_eq!(after.get(after.roots()[0]).unwrap().output_type, concrete);
        assert_eq!(
            before.get(before.roots()[0]).unwrap().output_type.dims,
            vec![DimInfo::Named("n".into(), None)],
            "the pass input remains an immutable historical snapshot"
        );
        #[cfg(feature = "lowering-trace")]
        {
            let collector = crate::lowering_trace::Collector::new_helper();
            collector.normalization(
                &before,
                &before,
                before.clone(),
                &before,
                &UnordMap::new(),
                &UnordMap::new(),
            );
            collector.helper_result(before.roots(), crate::lowering_trace::Value::Node(root));
            let mut trace = collector.finish_helper();
            trace.record_dimension_rebinding(&after);
            let normalization = &trace.lowering.normalization;
            for historical in [
                &normalization.before_dce,
                &normalization.after_dce,
                &normalization.after_copies,
                &normalization.after_drops,
            ] {
                assert_eq!(
                    bincode::serialize(historical).unwrap(),
                    bincode::serialize(&before).unwrap()
                );
            }
            assert_eq!(
                bincode::serialize(trace.after_dimension_rebinding.as_ref().unwrap()).unwrap(),
                bincode::serialize(&after).unwrap(),
            );
        }
    })
}

#[test]
fn host_permutation_retains_ordered_physical_axes() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{DimInfo, TensorType};
        for name in ["n", "d77"] {
            for dimension in [
                DimInfo::Named(name.into(), None),
                DimInfo::Lit(2),
                DimInfo::Lit(3),
            ] {
                let input = TensorType {
                    dims: vec![DimInfo::Lit(4), dimension.clone(), DimInfo::Lit(7)],
                    precision: Prim::F64,
                };
                let permuted = permute_host_tensor_type(&input, &[2, 0, 1]).unwrap();
                assert_eq!(
                    permuted.dims,
                    vec![DimInfo::Lit(7), DimInfo::Lit(4), dimension]
                );
                assert_eq!(permuted.precision, Prim::F64);
                assert_eq!(
                    permute_host_tensor_type(&permuted, &[1, 2, 0]).unwrap(),
                    input
                );
            }
        }
    })
}

#[test]
fn host_permutation_rejects_incomplete_or_nonbijective_axes() {
    test_resources().run(|_| {
        let input = TensorType {
            dims: vec![DimInfo::Lit(4), DimInfo::Lit(2)],
            precision: Prim::F64,
        };
        for axes in [&[0][..], &[0, 0], &[0, 2], &[-1, 0], &[0, 1, 2]] {
            assert!(permute_host_tensor_type(&input, axes).is_none(), "{axes:?}");
        }
    })
}

#[test]
fn tensor_helper_actualization_merges_matmul_synthetic_expand_dims() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let in_dim = DimInfo::Named("in_dim".into(), None);
        let out_dim = DimInfo::Named("out_dim".into(), None);
        let d417 = DimInfo::Named("d417".into(), None);
        let d420 = DimInfo::Named("d420".into(), None);
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let w = dag.add_node(
            decl,
            RiscOp::Load { name: "w".into() },
            vec![],
            TensorType {
                dims: vec![in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded_x = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(1),
                },
            },
            vec![x, w],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone(), d420.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded_w = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![w, x],
            TensorType {
                dims: vec![d417, in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let product = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![expanded_x, expanded_w],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone(), d420],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_numeric_core::prim::Prim::F32,
            },
            vec![product],
            TensorType {
                dims: vec![batch.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = HostScope::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![batch.clone(), in_dim.clone()],
                precision: Prim::F32,
            }),
        );
        scope.insert(
            "w".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        for node in actualized.nodes() {
            assert!(
                !node.output_type.dims.iter().any(|dim| {
                    matches!(dim, DimInfo::Named(name, None) if name == "d417" || name == "d420")
                }),
                "node retained synthetic dims: {node:?}"
            );
        }
    })
}

/// Actualization may rewrite a checker-minted output dimension, but the
/// structural InputAxis carrier continues to reference the same tensor
/// input slot and axis.
#[test]
fn tensor_helper_actualization_preserves_structural_expand_size() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let n = DimInfo::Named("n".into(), None);
        let d47 = DimInfo::Named("d47".into(), None);
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        // Load typed with the minted alias; the scope knows the
        // user-facing symbol.
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![d47.clone()],
                precision: Prim::F32,
            },
            None,
        );
        // Scalar upstream gradient, as the Sum adjoint produces.
        let g = dag.add_node(
            decl,
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        // The Sum adjoint's expand-back reads the original tensor shape.
        let expanded_g = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![g, x],
            TensorType {
                dims: vec![d47.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![expanded_g, x],
            TensorType {
                dims: vec![d47],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = HostScope::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![n.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        let expand = actualized.get(expanded_g).expect("expand node");
        assert_eq!(
            expand.output_type.dims,
            vec![n],
            "expand output dim must actualize to the user-facing symbol"
        );
        assert_eq!(
            expand.op,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            "actualization must preserve the structural witness slot"
        );
        let errors = chelis_dag_core::verify::verify(&actualized);
        assert!(
            errors.is_empty(),
            "actualized helper must verify: {errors:?}"
        );
    })
}

/// Negative parity: an already user-facing witness remains unchanged.
#[test]
fn tensor_helper_actualization_leaves_user_facing_input_axis_alone() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let g = dag.add_node(
            decl,
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![g, x],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![expanded, x],
            TensorType {
                dims: vec![batch],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = HostScope::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("batch".into(), None)],
                precision: Prim::F32,
            }),
        );

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        let expand = actualized.get(expanded).expect("expand node");
        assert_eq!(
            expand.op,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            "user-facing structural sizes must not be rewritten"
        );
    })
}

#[test]
fn tensor_helper_actualization_repairs_shrink_rank_from_bounds() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load {
                name: "expanded".into(),
            },
            vec![],
            TensorType {
                // The checked helper arrived with the stale pre-expand rank.
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let shrink = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![
                    (RtDim::Lit(0), RtDim::Lit(1)),
                    (RtDim::Lit(0), RtDim::Lit(2)),
                ],
            },
            vec![input],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrink);

        let mut scope = HostScope::new();
        scope.insert(
            "expanded".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
                precision: Prim::F32,
            }),
        );

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        assert_eq!(
            actualized.get(shrink).expect("shrink node").output_type,
            TensorType {
                dims: vec![DimInfo::Lit(1), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            "the shrink result rank and literal extents come from its actual input and bounds"
        );
    })
}

/// chelis#2906: a checker binder two activations bring into one helper
/// graph with different extents is a typed refusal in every build,
/// never a debug assertion that a release build skips by keeping the
/// first extent.
#[test]
fn conflicting_checker_dimension_is_a_typed_refusal() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, TensorType};

        let shared = DimInfo::Named("d7".into(), None);
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        for name in ["x", "y"] {
            let load = dag.add_node(
                decl,
                RiscOp::Load { name: name.into() },
                vec![],
                TensorType {
                    dims: vec![shared.clone()],
                    precision: Prim::Int64,
                },
                None,
            );
            dag.add_root(load);
        }
        let scope = HostScope::from([
            (
                "x".to_string(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::Int64,
                }),
            ),
            (
                "y".to_string(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(4)],
                    precision: Prim::Int64,
                }),
            ),
        ]);
        let refused = actualize_tensor_helper_types(&dag, &scope)
            .expect_err("two extents for one checker dimension");
        assert!(refused.fatal, "{refused:?}");
        assert!(
            refused
                .message
                .contains("gives checker dimension `d7` the extents 3 and 4")
                && refused.message.contains("unimplemented chelis#1277"),
            "{}",
            refused.message
        );

        // chelis#3340: the remap that every helper route calls after its
        // lowering entry has returned hands the refusal back as a value. No
        // enclosing catch is needed for it to reach the caller.
        let returned = remap_tensor_helper_dim_symbols(
            &dag,
            &scope,
            &TensorType {
                dims: vec![shared.clone()],
                precision: Prim::Int64,
            },
        )
        .expect_err("the refusal is returned, not unwound");
        assert!(returned.fatal, "{returned:?}");
        assert!(
            returned.message.contains("checker dimension `d7`"),
            "{}",
            returned.message
        );

        // The control: one extent for the binder is a rename, not a refusal.
        let agreeing = HostScope::from([
            ("x".to_string(), scope["x"].clone()),
            ("y".to_string(), scope["x"].clone()),
        ]);
        let actualized =
            actualize_tensor_helper_types(&dag, &agreeing).expect("consistent checker dimensions");
        for root in actualized.roots() {
            assert_eq!(
                actualized.get(*root).expect("root").output_type.dims,
                vec![DimInfo::Lit(3)]
            );
        }
        let remapped = remap_tensor_helper_dim_symbols(
            &dag,
            &agreeing,
            &TensorType {
                dims: vec![shared],
                precision: Prim::Int64,
            },
        )
        .expect("matching extents remain valid through the helper route");
        for root in remapped.roots() {
            assert_eq!(
                remapped.get(*root).expect("root").output_type.dims,
                vec![DimInfo::Lit(3)]
            );
        }
    })
}

#[test]
fn tensor_helper_actualization_preserves_full_axis_shrink_extent() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let shrink = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::ToEnd)],
            },
            vec![input],
            TensorType {
                dims: vec![DimInfo::Named("*".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrink);

        let mut scope = HostScope::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        assert_eq!(
            actualized
                .get(shrink)
                .expect("full-axis shrink")
                .output_type
                .dims,
            vec![batch],
            "the full-axis sentinel preserves the input axis identity"
        );
    })
}

/// NEGATIVE PARITY: a `ToEnd` sentinel under a nonzero start is malformed,
/// not a shape actualization may interpret.
///
/// `grad`'s Shrink adjoint panics on this exact shape and
/// `shrink_adjoint_malformed_sentinel_fails_loud` is its `#[should_panic]`
/// control; `bind_symbolic_dims` rejects the non-literal-start form.
/// This is the third component's half of that rule. No producer builds
/// one - `lower::lower_one_bound` gives the front end only `Lit`/`Node`,
/// and every `grad` adjoint emitting the sentinel pairs it with `Lit(0)` -
/// so the DAG is hand-built, and the assertion is about disposition: the
/// exhaustive `Shrink` arm declines and leaves the node's declared type
/// alone. It must not forward the input axis identity the way the
/// well-formed `(0, ToEnd)` sentinel does, which would silently claim
/// extent 4 for a slice that starts at 1.
#[test]
fn tensor_helper_actualization_declines_a_nonzero_start_shrink_sentinel() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let declared = TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::F32,
        };
        let shrink = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::ToEnd)],
            },
            vec![input],
            declared.clone(),
            None,
        );
        dag.add_root(shrink);

        let scope = HostScope::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            }),
        )]);

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        assert_eq!(
            actualized
                .get(shrink)
                .expect("malformed sentinel shrink")
                .output_type,
            declared,
            "a malformed sentinel must not be given an extent by actualization"
        );
    })
}

/// NEGATIVE PARITY: [05-MOV-1]'s owner matrix reserves `InputAxis` for
/// `expand` and `reshape`. Host helper actualization must therefore
/// decline it in either `shrink` bound and in a `stride`, preserving the
/// declared type until verification reports the malformed carrier.
#[test]
fn tensor_helper_actualization_declines_input_axis_for_shrink_and_stride() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("actual_input".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let declared = TensorType {
            dims: vec![DimInfo::Named("d47".into(), None)],
            precision: Prim::F32,
        };
        let input_axis = || RtDim::InputAxis {
            tensor: 1,
            axis: RtAxis::Lit(0),
        };
        let malformed = [
            dag.add_node(
                decl,
                RiscOp::Shrink {
                    bounds: vec![(input_axis(), RtDim::Lit(2))],
                },
                vec![input, input],
                declared.clone(),
                None,
            ),
            dag.add_node(
                decl,
                RiscOp::Shrink {
                    bounds: vec![(RtDim::Lit(0), input_axis())],
                },
                vec![input, input],
                declared.clone(),
                None,
            ),
            dag.add_node(
                decl,
                RiscOp::Stride {
                    strides: vec![input_axis()],
                },
                vec![input, input],
                declared.clone(),
                None,
            ),
        ];
        for node in malformed {
            // `reusable_input` is a storage-reuse hint, not shape authority.
            // Exercise the generic fallback explicitly: a declined malformed
            // movement carrier must not inherit its reusable buffer's shape.
            dag.set_reusable_input(node, input);
            dag.add_root(node);
        }

        let scope = HostScope::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("actual_input".into(), None)],
                precision: Prim::F32,
            }),
        )]);

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        for node in malformed {
            assert_eq!(
                actualized
                    .get(node)
                    .expect("malformed movement node")
                    .output_type,
                declared,
                "InputAxis outside expand/reshape must not be actualized"
            );
        }
    })
}

fn runtime_shrink_bound_nodes(
    dag: &mut chelis_dag_core::dag::Dag,
    decl: chelis_dag_core::dag::DeclId,
) -> (chelis_dag_core::dag::NodeId, chelis_dag_core::dag::NodeId) {
    use chelis_dag_core::dag::{RiscOp, TensorType};

    let scalar_i64 = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let start = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Int64, 1.0),
        vec![],
        scalar_i64.clone(),
        None,
    );
    let end = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Int64, 3.0),
        vec![],
        scalar_i64,
        None,
    );
    (start, end)
}

#[test]
fn runtime_shrink_name_reserves_scope_actualized_base_and_suffixes() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            },
            None,
        );
        let (start, end) = runtime_shrink_bound_nodes(&mut dag, decl);
        let shrink = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
            },
            vec![input, start, end],
            TensorType {
                dims: vec![DimInfo::Named("*".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrink);

        let base = format!("_rt_shrink_dim_{}_0", shrink.0);
        let suffixed = format!("{base}_1");
        let mut scope = HostScope::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            }),
        )]);
        scope.insert(
            "scope_only_reservations".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![
                    DimInfo::Named(base.clone(), None),
                    DimInfo::Named(suffixed, None),
                ],
                precision: Prim::F32,
            }),
        );

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        assert_eq!(
            actualized
                .get(shrink)
                .expect("runtime shrink")
                .output_type
                .dims,
            vec![DimInfo::Named(format!("{base}_2"), None)],
            "names introduced by host-scope actualization share the same identity namespace"
        );
    })
}

#[test]
fn runtime_shrink_name_reserves_op_internal_symbols_in_both_node_orders() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        for internal_before_shrink in [false, true] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                TensorType {
                    dims: vec![DimInfo::Lit(8)],
                    precision: Prim::F32,
                },
                None,
            );
            let (start, end) = runtime_shrink_bound_nodes(&mut dag, decl);
            let expected_shrink_id = if internal_before_shrink { 4 } else { 3 };
            let base = format!("_rt_shrink_dim_{expected_shrink_id}_0");
            let internal_shape = vec![RtDim::Sym(base.clone()), RtDim::Sym(format!("{base}_1"))];
            let add_internal_carrier = |dag: &mut Dag| {
                dag.add_node(
                    decl,
                    RiscOp::Reshape {
                        new_shape: internal_shape.clone(),
                    },
                    vec![input],
                    TensorType {
                        dims: vec![DimInfo::Lit(1), DimInfo::Lit(8)],
                        precision: Prim::F32,
                    },
                    None,
                )
            };
            if internal_before_shrink {
                let _ = add_internal_carrier(&mut dag);
            }
            let shrink = dag.add_node(
                decl,
                RiscOp::Shrink {
                    bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
                },
                vec![input, start, end],
                TensorType {
                    dims: vec![DimInfo::Named("*".into(), None)],
                    precision: Prim::F32,
                },
                None,
            );
            if !internal_before_shrink {
                let _ = add_internal_carrier(&mut dag);
            }
            dag.add_root(shrink);
            assert_eq!(shrink.0, expected_shrink_id);

            let scope = HostScope::from([(
                "x".into(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(8)],
                    precision: Prim::F32,
                }),
            )]);
            let actualized =
                actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
            assert_eq!(
                actualized
                    .get(shrink)
                    .expect("runtime shrink")
                    .output_type
                    .dims,
                vec![DimInfo::Named(format!("{base}_2"), None)],
                "op-internal symbols must be reserved before allocation regardless of node order"
            );
        }
    })
}

#[test]
fn multiple_runtime_shrinks_keep_distinct_names_through_fanout_consumers() {
    test_resources().run(|_| {
        use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            },
            None,
        );
        let (start, end) = runtime_shrink_bound_nodes(&mut dag, decl);
        let wildcard = TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::F32,
        };
        let shrink_a = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
            },
            vec![input, start, end],
            wildcard.clone(),
            None,
        );
        let shrink_b = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
            },
            vec![input, start, end],
            wildcard.clone(),
            None,
        );
        let synthetic = |name: &str| TensorType {
            dims: vec![DimInfo::Named(name.into(), None)],
            precision: Prim::F32,
        };
        let consumers = [
            dag.add_node(
                decl,
                RiscOp::Add,
                vec![shrink_a, shrink_a],
                synthetic("d701"),
                None,
            ),
            dag.add_node(
                decl,
                RiscOp::Mul,
                vec![shrink_a, shrink_a],
                synthetic("d702"),
                None,
            ),
            dag.add_node(
                decl,
                RiscOp::Add,
                vec![shrink_b, shrink_b],
                synthetic("d703"),
                None,
            ),
            dag.add_node(
                decl,
                RiscOp::Mul,
                vec![shrink_b, shrink_b],
                synthetic("d704"),
                None,
            ),
        ];
        for consumer in consumers {
            dag.add_root(consumer);
        }
        let scope = HostScope::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            }),
        )]);

        let actualized =
            actualize_tensor_helper_types(&dag, &scope).expect("consistent checker dimensions");
        let dims_a = actualized
            .get(shrink_a)
            .expect("first shrink")
            .output_type
            .dims
            .clone();
        let dims_b = actualized
            .get(shrink_b)
            .expect("second shrink")
            .output_type
            .dims
            .clone();
        assert_ne!(
            dims_a, dims_b,
            "each runtime Shrink owns a distinct extent identity"
        );
        assert_eq!(
            actualized
                .get(consumers[0])
                .expect("a/add")
                .output_type
                .dims,
            dims_a
        );
        assert_eq!(
            actualized
                .get(consumers[1])
                .expect("a/mul")
                .output_type
                .dims,
            dims_a
        );
        assert_eq!(
            actualized
                .get(consumers[2])
                .expect("b/add")
                .output_type
                .dims,
            dims_b
        );
        assert_eq!(
            actualized
                .get(consumers[3])
                .expect("b/mul")
                .output_type
                .dims,
            dims_b
        );
    })
}

fn program_with_global_builtin(builtin: &str) -> HostProgram {
    let mut program = HostProgram::default();
    program.globals.push(HostBinding {
        name: "probe".into(),
        display_name: None,
        display_roots: Vec::new(),
        ty: HostTypeTerm::Unit,
        value: HostExpr::new(HostExprKind::Builtin {
            name: builtin.into(),
            args: Vec::new(),
            ty: HostTypeTerm::Unit,
        }),
    });
    program
}

/// Only the unspellable markers are unresolved call sites; names a
/// user could spell (`call`, `__unresolved_grad`) are ordinary
/// identifiers (chelis#841). The transform marker routes to the
/// transform scan; the callable marker does not.
#[test]
fn unresolved_call_sites_match_only_the_unspellable_markers() {
    test_resources().run(|_| {
        let callable = program_with_global_builtin(HOST_UNRESOLVED_CALLABLE_MARKER);
        assert_eq!(
            host_program_unresolved_call_sites(&callable),
            vec!["probe".to_string()]
        );
        assert!(host_program_unresolved_transform_sites(&callable).is_empty());

        let transform = program_with_global_builtin(HOST_UNRESOLVED_TRANSFORM_MARKER);
        assert_eq!(
            host_program_unresolved_call_sites(&transform),
            vec!["probe".to_string()]
        );
        assert_eq!(
            host_program_unresolved_transform_sites(&transform),
            vec!["probe".to_string()]
        );

        for spellable in ["call", "__unresolved_grad"] {
            let program = program_with_global_builtin(spellable);
            assert!(
                host_program_unresolved_call_sites(&program).is_empty(),
                "user-spellable name {spellable:?} must not read as a marker"
            );
        }

        let mut marker_call = HostProgram::default();
        marker_call.globals.push(HostBinding {
            name: "probe".into(),
            display_name: None,
            display_roots: Vec::new(),
            ty: HostTypeTerm::Unit,
            value: HostExpr::new(HostExprKind::Call {
                callee: HostCallee::unresolved_callable(),
                args: Vec::new(),
                arg_tys: Vec::new(),
                ty: HostTypeTerm::Unit,
            }),
        });
        assert_eq!(
            host_program_unresolved_call_sites(&marker_call),
            vec!["probe".to_string()],
            "a typed-but-unresolved callee carries a marker in `Call` position"
        );
    })
}

#[test]
fn nullary_generic_constructor_retains_named_term_until_checked_expectation() {
    test_resources().run(|operation| {
        let definition = GenericAdtConstructor {
            adt_name: "Box".into(),
            ctor_name: "Empty".into(),
            parameters: vec!["a".into()],
            stored_parameters: vec![false],
            fields: Vec::new(),
        };
        let carried = definition
            .instantiate_nullary_term(&fresh_host_inference(), operation)
            .expect("nullary generic term may await its checked context");
        assert_eq!(
            carried.ty,
            HostTypeTerm::Adt("Box".into(), vec![HostTypeTerm::TypeVariable("a".into())])
        );
        let concrete = force_host_expr_type(
            HostExpr::new(HostExprKind::AdtConstruct {
                ctor: "Empty".into(),
                fields: Vec::new(),
                ty: carried.ty,
            }),
            HostTypeTerm::Adt("Box".into(), vec![HostTypeTerm::Float32]),
            operation,
        );
        assert_eq!(
            host_expr_type(&concrete),
            HostTypeTerm::Adt("Box".into(), vec![HostTypeTerm::Float32])
        );
    })
}

#[test]
fn nullary_generic_constructor_still_rejects_wrong_name_and_arity() {
    test_resources().run(|operation| {
        let definition = GenericAdtConstructor {
            adt_name: "Box".into(),
            ctor_name: "Empty".into(),
            parameters: vec!["a".into()],
            stored_parameters: vec![false],
            fields: Vec::new(),
        };
        assert!(matches!(
            definition.instantiate_nullary_term(
                &HostTypeTerm::Adt("Other".into(), vec![HostTypeTerm::Float32]),
                operation
            ),
            Err(AdtInstantiationError::WrongAppliedType { .. })
        ));
        assert!(matches!(
            definition
                .instantiate_nullary_term(&HostTypeTerm::Adt("Box".into(), Vec::new()), operation),
            Err(AdtInstantiationError::Arity {
                expected: 1,
                got: 0,
                ..
            })
        ));
    })
}

/// chelis#1201: a non-recursive generic is no longer inlined — it is
/// specialized through the same bounded monomorphization the recursive
/// path uses. The lowering assertion is the contract; the predicate that
/// used to gate inlining is gone with the inlining.
#[test]
fn nonrecursive_generic_match_is_a_bounded_callsite_specialization() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "type Box[a] =\n\
           | Empty\n\
           | Full { value: a }\n\
         def empty[a]() -> Box[a] = Empty\n\
         def is_empty[a](value: Box[a]) -> bool = match value with {\n\
           | Empty => true\n\
           | Full { value: item } => false\n\
         }\n\
         def concrete() -> Box[f32] = empty()\n\
         def main() -> bool = is_empty(concrete())\n",
        );
        let lowered = try_lower_compiled_program(checked.facts(), operation)
            .expect("the bounded generic match must lower through its concrete caller");
        let host = lowered
            .host
            .as_ref()
            .expect("the program lowers a host lane");
        assert!(
            host.functions
                .iter()
                .any(|function| function.origin == HostFunctionOrigin::Monomorphized),
            "the non-recursive generic must reach lowering as a specialization, not an inline paste"
        );
    })
}

#[test]
fn manifested_host_lane_overrides_legacy_tensor_root_classification() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "x = insert(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)\n\
         y = mul(x, x)\n",
        );
        let realizability = chelis_effects::realizability::infer_realizability(
            &checked,
            &[
                Prim::F32,
                Prim::Bool,
                Prim::Bf16,
                Prim::F16,
                Prim::Int32,
                Prim::Int64,
            ],
        );
        let manifest =
            chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
        assert!(
            manifest
                .entries
                .iter()
                .any(|entry| entry.name == "y" && entry.lane == chelis_types::types::Lane::Host),
            "the C-like capability set must route the f64 root to Host"
        );

        let legacy =
            try_lower_compiled_program(checked.facts(), operation).expect("legacy lowering");
        assert!(
            legacy.host.is_none(),
            "the fixture must exercise the legacy classifier's Tensor decision"
        );
        let manifested = try_lower_compiled_program_with_manifest(
            checked.facts(),
            &TestManifest(&manifest),
            operation,
        )
        .expect("manifested lowering");
        assert!(
            manifested.host.is_some(),
            "manifested lowering must retain the C-target Host realization"
        );
    })
}

/// chelis#1201 / coral#26: a generic ADT constructed in a position the
/// caller's expected type cannot reach.
///
/// The arm-body and `if`-branch propagation resolves a constructor that
/// sits in a RESULT position. A block-local binding is not one, so this
/// resolves only through the specialization's own type substitution —
/// and that substitution only matches if it is keyed on the same
/// variables the checker stamped into the body. This is the exact shape
/// coral's `Hamt` node constructors take.
#[test]
fn generic_adt_built_outside_a_result_position_specializes() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "type Store[a] =\n\
           | Vacant\n\
           | Held { key: string, value: a }\n\
         def unwrap[a](s: Store[a], fallback: a) -> a = match s with {\n\
           | Vacant => fallback\n\
           | Held { key: k, value: v } => v\n\
         }\n\
         def put[a](s: Store[a], key: string, value: a) -> a = {\n\
           fresh = Held { key, value }\n\
           unwrap(fresh, value)\n\
         }\n\
         def read() -> i64 = put(Vacant, \"a\", cast(1, i64))\n",
        );
        try_lower_compiled_program(checked.facts(), operation).expect(
            "a generic ADT built in a block-local binding must resolve from the specialization",
        );
    })
}

/// chelis#1201: a bare nullary generic constructor used as a call
/// ARGUMENT. Nothing at its own site pins the parameter; it resolves
/// only from the callee's declared parameter type.
///
/// This already passes without the substitution fix above — it is
/// coverage for the shape #1201's body describes, not a guard on that
/// fix. The guard is
/// `generic_adt_built_outside_a_result_position_specializes`.
#[test]
fn nullary_generic_constructor_in_argument_position_specializes() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "type Store[a] =\n\
           | Vacant\n\
           | Held { key: string, value: a }\n\
         def fresh_store[a]() -> Store[a] = Vacant\n\
         def unwrap[a](s: Store[a], fallback: a) -> a = match s with {\n\
           | Vacant => fallback\n\
           | Held { key: k, value: v } => v\n\
         }\n\
         def seed[a](value: a) -> a = unwrap(fresh_store(), value)\n\
         def read() -> i64 = seed(cast(7, i64))\n",
        );
        try_lower_compiled_program(checked.facts(), operation)
            .expect("a nullary generic constructor in argument position must lower");
    })
}

/// chelis#1201: a generic container instantiated at a DIMENSION-generic
/// element type.
///
/// #1201's body asserts that dim-generic tensor code and generic
/// container ADTs are disjoint populations. coral's `Frame` is both —
/// `Hamt[Column[n]]` — so this combination is pinned here directly.
///
/// It already passes without the substitution fix above; it exists
/// because the claim it refutes was load-bearing in #1201's triage, not
/// as a guard on that fix.
#[test]
fn generic_container_over_a_dimension_generic_adt_specializes() {
    test_resources().run(|operation| {
        let checked = surf_check(
            "type Col[n] =\n\
           | FloatCol(tensor[n, f32])\n\
         type Box[a] =\n\
           | Nothing\n\
           | Just { item: a }\n\
         def box_it[a](item: a) -> Box[a] = Just { item }\n\
         def unbox[a](b: Box[a], fallback: a) -> a = match b with {\n\
           | Nothing => fallback\n\
           | Just { item: i } => i\n\
         }\n\
         def roundtrip[n](c: Col[n]) -> Col[n] = unbox(box_it(c), c)\n\
         def total[n](c: Col[n]) -> tensor[n, f32] = match roundtrip(c) with {\n\
           | FloatCol(t) => t\n\
         }\n\
         def main() -> tensor[2, f32] =\n\
           total(FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))\n",
        );
        try_lower_compiled_program(checked.facts(), operation)
            .expect("a generic container over a dim-generic element must lower");
    })
}

/// chelis#1216: a RECURSIVE function generic over an erased ADT
/// dimension.
///
/// The inline path specializes the outer call but refuses the recursive
/// edge, and that edge then fell through to a plain call to a symbol the
/// emitter elided, carrying the ADT's own parameter variable to the
/// code-generation boundary. Such a callee is now monomorphized instead,
/// so the recursive edge has an in-progress specialization to complete
/// its slots from. coral's `column_lengths_match` -> `all_eq_len` is
/// this shape.
#[test]
fn recursive_dimension_generic_call_is_monomorphized() {
    test_resources()
        .run(|operation| { let checked = surf_check(
        "type Col[n] =\n\
           | FloatCol(tensor[n, f32])\n\
         def zero_i64() -> i64 = cast(0, i64)\n\
         def one_i64() -> i64 = cast(1, i64)\n\
         def col_len[n](col: Col[n]) -> i64 = match col with {\n\
           | FloatCol(xs) => numel(xs)\n\
         }\n\
         def all_eq_len[n](pairs: List[(string, Col[n])], expected: i64) -> bool =\n\
           if eq(len(pairs), zero_i64()) then true else {\n\
             entry = index(pairs, zero_i64())\n\
             if neq(col_len(entry.1), expected) then false else all_eq_len(skip(pairs, one_i64()), expected)\n\
           }\n\
         def main() -> bool = all_eq_len([(\"a\", FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))], cast(2, i64))\n",
    );
    let lowered = try_lower_compiled_program(checked.facts(), operation)
        .expect("a recursive dimension-generic call must lower through monomorphization");
    let host = lowered
        .host
        .as_ref()
        .expect("the program lowers a host lane");
    // PR #1218 review: any-Monomorphized was too weak — pin the
    // specialization to `all_eq_len` itself...
    let specialization = host
        .functions
        .iter()
        .find(|function| {
            function.origin == HostFunctionOrigin::Monomorphized
                && function.name.starts_with("all_eq_len__mono_")
        })
        .expect("the specialization must belong to `all_eq_len`, not merely exist");
    // ...and pin its recursive edge to the OWNING specialized symbol:
    // the reported defect was precisely that edge falling through to a
    // plain call to the elided generic symbol. Direct recursion at one
    // instantiation memoizes to one symbol, so the edge must name the
    // specialization it lives in.
    let body = format!("{:?}", specialization.body);
    assert!(
        body.contains(&format!("callee: Function(\"{}\")", specialization.name)),
        "the recursive edge must call the owning specialized symbol \
         `{}`, got body:\n{body}",
        specialization.name
    );
    assert!(
        !body.contains("callee: Function(\"all_eq_len\")"),
        "no edge may still call the elided generic symbol `all_eq_len`, \
         got body:\n{body}"
    ); })
}

// ── chelis#1271: cross-package constructor-name collision ────────
//
// Two packages in one dependency graph may declare types whose
// constructors share an unqualified name. The reef linker gives each
// a distinct `Pkg__<pkg>__<Module>__<Name>` identity, so lowering
// must resolve on the full spelling and must not silently pick one
// when only a short spelling is available. The end-to-end faces live
// in `crates/chelis-cli/tests/issue_1271_cross_package_ctor_collision.rs`;
// these cover the resolution rule itself and the ambiguity arm.
//
// No whole program reaches that arm, and the reason is structural
// rather than a property of today's checker: it needs mangled
// DECLARATIONS together with a SHORT reference, mangled declarations
// exist only in reef linker output, and there every reference is
// mangled too. A hand-authored mangled declaration is a
// `ReservedLinkerName` declaration error (spec/04-type-system.md,
// "Reserved linker name format"). The arm is therefore a fail-closed
// guard, and unit tests are the only place it can be exercised.

/// Two declarations whose mangled names differ only in their package
/// segment, with `Adep__` sorting ahead of `Blib__` so a first-match
/// rule always answers with the wrong one.
const COLLIDING_DECLARATIONS: &str = concat!(
    "(deftype {} Adep__Wrapped () (variant {} Adep__Wrapped ",
    "(field {} amount (t-prim {} f32)) (field {} extra (t-prim {} f32))))\n",
    "(deftype {} Blib__Wrapped () (variant {} Blib__Wrapped ",
    "(field {} extra (t-prim {} f32)) (field {} amount (t-prim {} f32))))\n"
);

/// Parse one Deep expression into the stamped carrier that
/// `lower_host_expr_kind` dispatches on.
fn deep_expr(source: &str) -> Expr {
    chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("parse failed")
        .into_iter()
        .next()
        .expect("one expression")
}

fn lower_against(
    program: &CheckedProgram,
    source: &str,
    operation: &Operation<'_, chelis_syntax_host::NativeStack>,
) -> Result<HostExpr, String> {
    let program = &HostLoweringSession::new(program.facts(), operation);
    let mut helpers = TensorHelperSink::new(false);
    lower_host_expr(&deep_expr(source), program, &HostScope::new(), &mut helpers)
        .map_err(|diagnostic| diagnostic.message)
}

#[test]
fn constructor_resolution_prefers_the_exact_spelling() {
    test_resources().run(|operation| {
        // The repair: the mangled name IS the identity
        // (spec/04-type-system.md, "Module identity"), so the
        // later-sorting package still resolves to its own declaration.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let AdtConstructorResolution::Unique(definition) = resolve_adt_constructor_definition(
            &HostLoweringSession::new(checked.facts(), operation),
            "Blib__Wrapped",
        )
        .expect("valid checked metadata") else {
            panic!("an exactly-spelled constructor must resolve uniquely");
        };
        assert_eq!(definition.adt_name, "Blib__Wrapped");
        assert_eq!(
            definition
                .fields
                .iter()
                .map(|field| field.name.as_deref())
                .collect::<Vec<_>>(),
            [Some("extra"), Some("amount")],
            "the exact spelling must carry its own field list and order"
        );
    })
}

#[test]
fn constructor_resolution_still_answers_a_unique_terminal_spelling() {
    test_resources().run(|operation| {
        // The terminal fallback is what lets a short, unqualified
        // spelling reach its declaration; the repair narrows it, it does
        // not remove it.
        let checked = parse_and_check(concat!(
            "(deftype {} Blib__Wrapped () (variant {} Blib__Wrapped ",
            "(field {} amount (t-prim {} f32))))\n"
        ));
        let AdtConstructorResolution::Unique(definition) = resolve_adt_constructor_definition(
            &HostLoweringSession::new(checked.facts(), operation),
            "Wrapped",
        )
        .expect("valid checked metadata") else {
            panic!("an uncontested terminal spelling must still resolve");
        };
        assert_eq!(definition.adt_name, "Blib__Wrapped");
    })
}

#[test]
fn constructor_resolution_refuses_a_contested_terminal_spelling() {
    test_resources().run(|operation| {
        // Negative parity: when the short spelling really is ambiguous,
        // lowering must say so and name both sides rather than restore
        // the defect under a different first-match rule.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let AdtConstructorResolution::Ambiguous(candidates) = resolve_adt_constructor_definition(
            &HostLoweringSession::new(checked.facts(), operation),
            "Wrapped",
        )
        .expect("valid checked metadata") else {
            panic!("a contested terminal spelling must not resolve");
        };
        assert_eq!(candidates, ["Adep__Wrapped", "Blib__Wrapped"]);
        let message =
            ambiguous_constructor_error(&deep_expr("(var {} Wrapped)"), "Wrapped", &candidates)
                .message;
        assert!(
            message.contains("`Adep__Wrapped`") && message.contains("`Blib__Wrapped`"),
            "the diagnostic must name both candidates; got: {message}"
        );
        assert!(
            message.contains("unsupported:") && message.contains("[04-TOT-3]"),
            "the diagnostic must keep the branded lowering shape; got: {message}"
        );
    })
}

#[test]
fn constructor_resolution_keeps_same_name_declarations_deterministic() {
    test_resources().run(|operation| {
        // Settled by decision, with no open owner: two declarations
        // carrying the SAME name are not a qualified-versus-unqualified
        // ambiguity, so no name rule can separate them. The gate is the
        // checker, which resolves a same-name constructor by call shape
        // or rejects it, and this repair leaves the case exactly as it
        // was. Locked so a later change to that decision is deliberate.
        let checked = parse_and_check(concat!(
            "(deftype {} AaaBox () (variant {} Boxed (field {} amount (t-prim {} f32))))\n",
            "(deftype {} BbbBox () (variant {} Boxed (field {} amount (t-prim {} f32))))\n"
        ));
        let AdtConstructorResolution::Unique(definition) = resolve_adt_constructor_definition(
            &HostLoweringSession::new(checked.facts(), operation),
            "Boxed",
        )
        .expect("valid checked metadata") else {
            panic!("same-name declarations keep their existing deterministic choice");
        };
        assert_eq!(definition.adt_name, "AaaBox");
    })
}

#[test]
fn adt_field_lookup_prefers_the_exact_type_name() {
    test_resources().run(|operation| {
        // Field access resolves through the ADT name and had the same
        // collision: admitting both declarations made them disagree on
        // the index of `amount` and rejected a valid access.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let found = lookup_adt_field_on_type(
            &HostLoweringSession::new(checked.facts(), operation),
            "Blib__Wrapped",
            &[],
            "amount",
        )
        .expect("valid checked metadata")
        .expect("no instantiation error")
        .expect("`amount` is declared on Blib__Wrapped");
        assert_eq!(found.0, 1, "`amount` is field 1 in the exact declaration");
    })
}

#[test]
fn adt_parameter_storage_prefers_the_exact_type_name() {
    test_resources().run(|operation| {
        // Same narrowing on the storage classification: the erased
        // parameter of one declaration must not be reported as stored
        // because a terminal-colliding declaration stores its own. The
        // declarations are built directly so the collision is the only
        // variable.
        let definitions = vec![
            GenericAdtConstructor {
                adt_name: "Adep__Column".to_string(),
                ctor_name: "Adep__Col".to_string(),
                parameters: vec!["a".to_string()],
                stored_parameters: vec![true],
                fields: Vec::new(),
            },
            GenericAdtConstructor {
                adt_name: "Blib__Column".to_string(),
                ctor_name: "Blib__Col".to_string(),
                parameters: vec!["n".to_string()],
                stored_parameters: vec![false],
                fields: Vec::new(),
            },
        ];
        assert!(
            adt_parameter_is_stored("Adep__Column", 0, &definitions, operation),
            "the stored parameter of `Adep__Column` is stored"
        );
        assert!(
            !adt_parameter_is_stored("Blib__Column", 0, &definitions, operation),
            "the erased parameter of `Blib__Column` must not inherit the \
         terminal-colliding declaration's storage classification"
        );
    })
}

#[test]
fn bare_constructor_reference_refuses_a_contested_terminal_spelling() {
    test_resources().run(|operation| {
        // Caller disposition: whether a bare reference is a construction
        // at all depends on which declaration answers it, so ambiguity
        // must reject rather than fall through to the variable path and
        // emit a bare C identifier nothing declares.
        let checked = parse_and_check(concat!(
            "(deftype {} Adep__Flag () (variant {} Adep__On (field {} weight (t-prim {} f32))))\n",
            "(deftype {} Blib__Flag () (variant {} Blib__On))\n"
        ));
        let message = lower_against(&checked, "(var {} On)", operation)
            .expect_err("a contested bare constructor reference must reject");
        assert!(
            message.contains("`Adep__On`") && message.contains("`Blib__On`"),
            "the rejection must name both candidates; got: {message}"
        );
    })
}

#[test]
fn record_construction_refuses_a_contested_terminal_spelling() {
    test_resources().run(|operation| {
        // Caller disposition: the field-name check reports the AUTHORED
        // constructor, so validating against another declaration is what
        // produced the self-contradicting "has no field" rejection.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let message = lower_against(
            &checked,
            "(record {} Wrapped (kv {} extra (lit {} 1.0)) (kv {} amount (lit {} 2.0)))",
            operation,
        )
        .expect_err("a contested record constructor must reject");
        assert!(
            message.contains("`Adep__Wrapped`") && message.contains("`Blib__Wrapped`"),
            "the rejection must name both candidates; got: {message}"
        );
        assert!(
            !message.contains("has no field"),
            "the contested case must not borrow the missing-field wording; got: {message}"
        );
    })
}

#[test]
fn positional_construction_refuses_a_contested_terminal_spelling() {
    test_resources().run(|operation| {
        // Caller disposition: the resolved declaration supplies each
        // argument's expected field type, which lowering then forces onto
        // the argument, so a wrong answer rewrites the payload's dtype.
        let checked = parse_and_check(concat!(
            "(deftype {} Adep__Box () (variant {} Adep__Boxed (t-prim {} f64)))\n",
            "(deftype {} Blib__Box () (variant {} Blib__Boxed (t-prim {} f32)))\n"
        ));
        let message = lower_against(&checked, "(app {} (var {} Boxed) (lit {} 1.0))", operation)
            .expect_err("a contested positional constructor must reject");
        assert!(
            message.contains("`Adep__Boxed`") && message.contains("`Blib__Boxed`"),
            "the rejection must name both candidates; got: {message}"
        );
    })
}

#[test]
fn match_pattern_refuses_a_contested_terminal_spelling() {
    test_resources().run(|operation| {
        // Caller disposition: the resolved field list decides which index
        // each binder reads, so a wrong answer silently binds the other
        // package's field. The `type_env` fallback for names that are not
        // constructors must not absorb this.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let message = lower_against(
            &checked,
            "(match {} (var {} w) (arm {} (pat-record {} Wrapped \
         (kv {} amount (pat-var {} a))) () (var {} a)))",
            operation,
        )
        .expect_err("a contested match constructor must reject");
        assert!(
            message.contains("`Adep__Wrapped`") && message.contains("`Blib__Wrapped`"),
            "the rejection must name both candidates; got: {message}"
        );
    })
}

fn option_i64() -> HostTypeTerm {
    HostTypeTerm::Option(Box::new(HostTypeTerm::Int64))
}

/// `[Some(1), None]` as lowering leaves it: the list and `Some` are
/// typed, the bare `None` still holds an inference hole.
fn sibling_typed_option_list() -> HostExpr {
    HostExpr::new(HostExprKind::List(
        vec![
            HostExpr::new(HostExprKind::Builtin {
                name: "Some".into(),
                args: vec![HostExpr::new(HostExprKind::Int(1))],
                ty: option_i64(),
            }),
            HostExpr::new(HostExprKind::Var("None".into(), fresh_host_inference())),
        ],
        HostTypeTerm::List(Box::new(option_i64())),
    ))
}

fn list_items(expr: &HostExpr) -> &[HostExpr] {
    match &expr.kind {
        HostExprKind::List(items, _) => items,
        other => panic!("expected a list, got {other:?}"),
    }
}

// chelis#2599: a builtin with no expected-type rule fixes no type for its
// argument. The argument keeps its own resolved type, shaped unlike the
// parent's `i64`, and that type reaches the `None` inside it.
#[test]
fn a_resolved_child_keeps_its_own_type_under_a_differently_typed_parent() {
    test_resources().run(|operation| {
        let forced = force_host_expr_type(
            HostExpr::new(HostExprKind::Builtin {
                name: "len".into(),
                args: vec![sibling_typed_option_list()],
                ty: fresh_host_inference(),
            }),
            HostTypeTerm::Int64,
            operation,
        );
        assert_eq!(host_expr_type(&forced), HostTypeTerm::Int64);
        let HostExprKind::Builtin { args, .. } = &forced.kind else {
            panic!("expected the builtin, got {:?}", forced.kind);
        };
        assert_eq!(
            host_expr_type(&args[0]),
            HostTypeTerm::List(Box::new(option_i64()))
        );
        assert_eq!(host_expr_type(&list_items(&args[0])[1]), option_i64());
    })
}

// A resolved own type wins even over a resolved expectation of another
// shape; the expectation fills only a hole.
#[test]
fn a_resolved_own_type_wins_over_a_resolved_expectation() {
    test_resources().run(|operation| {
        let list_ty = HostTypeTerm::List(Box::new(option_i64()));
        let kept = conform_host_subexpr(
            sibling_typed_option_list(),
            Some(&HostTypeTerm::Int64),
            operation,
        );
        assert_eq!(host_expr_type(&kept), list_ty);
        assert_eq!(host_expr_type(&list_items(&kept)[1]), option_i64());

        let filled = conform_host_subexpr(
            HostExpr::new(HostExprKind::Var("None".into(), fresh_host_inference())),
            Some(&option_i64()),
            operation,
        );
        assert_eq!(host_expr_type(&filled), option_i64());
    })
}

// Nothing resolves the child, so it is returned exactly as it was: no
// unresolved type is written and the hole survives to the rejection.
#[test]
fn a_child_nothing_resolves_is_left_unchanged() {
    test_resources().run(|operation| {
        let hole = fresh_host_inference();
        let other_hole = fresh_host_inference();
        let child = HostExpr::new(HostExprKind::Var("None".into(), hole.clone()));
        assert_eq!(
            host_expr_type(&conform_host_subexpr(child.clone(), None, operation)),
            hole
        );
        assert_eq!(
            host_expr_type(&conform_host_subexpr(child, Some(&other_hole), operation)),
            hole
        );
    })
}

// A fold's unresolved seed and callback result type take the fold's
// type, and the callback body is conformed, which reaches the `None` in
// its `if`. The list keeps its own type.
#[test]
fn a_fold_conforms_its_seed_and_callback_body() {
    test_resources().run(|operation| {
        let callback = HostCallback {
            stationary_accumulator: None,
            accumulator_delta: None,
            accumulator_composition: None,
            kind: HostCallbackKind::Inline {
                params: vec![
                    HostParam {
                        name: "acc".into(),
                        ty: option_i64(),
                    },
                    HostParam {
                        name: "x".into(),
                        ty: HostTypeTerm::Int64,
                    },
                ],
                body: Box::new(HostExpr::new(HostExprKind::If {
                    cond: Box::new(HostExpr::new(HostExprKind::Bool(true))),
                    then_expr: Box::new(HostExpr::new(HostExprKind::Var(
                        "None".into(),
                        fresh_host_inference(),
                    ))),
                    else_expr: Box::new(HostExpr::new(HostExprKind::Var(
                        "acc".into(),
                        option_i64(),
                    ))),
                    ty: option_i64(),
                })),
            },
            ret_ty: fresh_host_inference(),
        };
        let forced = force_host_expr_type(
            HostExpr::new(HostExprKind::Fold {
                callback,
                init: Box::new(HostExpr::new(HostExprKind::Var(
                    "None".into(),
                    fresh_host_inference(),
                ))),
                list: Box::new(HostExpr::new(HostExprKind::Var(
                    "xs".into(),
                    HostTypeTerm::List(Box::new(HostTypeTerm::Int64)),
                ))),
                ty: fresh_host_inference(),
            }),
            option_i64(),
            operation,
        );
        let HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        } = &forced.kind
        else {
            panic!("expected the fold, got {:?}", forced.kind);
        };
        assert_eq!(ty, &option_i64());
        assert_eq!(host_expr_type(init), option_i64());
        assert_eq!(
            host_expr_type(list),
            HostTypeTerm::List(Box::new(HostTypeTerm::Int64))
        );
        assert_eq!(callback.ret_ty, option_i64());
        let HostCallbackKind::Inline { body, .. } = &callback.kind else {
            panic!("expected an inline callback");
        };
        let HostExprKind::If { then_expr, .. } = &body.kind else {
            panic!("expected the callback's `if`, got {:?}", body.kind);
        };
        assert_eq!(host_expr_type(then_expr), option_i64());
    })
}

// Every node kind hands each child to the conform pass, including the
// children whose type the node does not fix. Each case holds a `None`
// hole below such a child; conforming must fill it so the node crosses
// the code-generation boundary.
#[test]
fn every_node_kind_conforms_its_children() {
    test_resources().run(|operation| {
        let list_ty = HostTypeTerm::List(Box::new(option_i64()));
        let holder = HostTypeTerm::Adt("Holder".into(), Vec::new());
        let constructed = || {
            HostExpr::new(HostExprKind::AdtConstruct {
                ctor: "Holder".into(),
                fields: vec![sibling_typed_option_list()],
                ty: holder.clone(),
            })
        };
        let signature_entry = |args: Vec<HostExpr>, lists: Vec<HostListEntry>| {
            HostExpr::new(HostExprKind::SignatureEntry {
                contract: EntryContract::default(),
                plan: SignatureEntryPlan::new(Vec::<HostTensorInput>::new()),
                args,
                positions: Vec::new(),
                lists,
            })
        };
        let cases = vec![
            (
                "builtin argument without an expected-type rule",
                HostExpr::new(HostExprKind::Builtin {
                    name: "len".into(),
                    args: vec![sibling_typed_option_list()],
                    ty: HostTypeTerm::Int64,
                }),
                HostTypeTerm::Int64,
            ),
            (
                "list item under a type of another shape",
                HostExpr::new(HostExprKind::List(
                    vec![sibling_typed_option_list()],
                    fresh_host_inference(),
                )),
                HostTypeTerm::Int64,
            ),
            (
                "tuple item under a tuple of another arity",
                HostExpr::new(HostExprKind::Tuple(
                    vec![sibling_typed_option_list()],
                    fresh_host_inference(),
                )),
                HostTypeTerm::Tuple(Vec::new()),
            ),
            (
                "call argument without a parameter type",
                HostExpr::new(HostExprKind::Call {
                    callee: HostCallee::test_function("f"),
                    args: vec![sibling_typed_option_list()],
                    arg_tys: Vec::new(),
                    ty: HostTypeTerm::Int64,
                }),
                HostTypeTerm::Int64,
            ),
            ("constructor field", constructed(), holder.clone()),
            (
                "field access base",
                HostExpr::new(HostExprKind::AdtFieldAccess {
                    base: Box::new(constructed()),
                    field_index: 0,
                    ty: list_ty.clone(),
                }),
                list_ty.clone(),
            ),
            (
                "option match scrutinee",
                HostExpr::new(HostExprKind::MatchOption {
                    scrutinee: Box::new(HostExpr::new(HostExprKind::Builtin {
                        name: "Some".into(),
                        args: vec![sibling_typed_option_list()],
                        ty: HostTypeTerm::Option(Box::new(list_ty.clone())),
                    })),
                    bind_name: "xs".into(),
                    some_expr: Box::new(HostExpr::new(HostExprKind::Int(1))),
                    none_expr: Box::new(HostExpr::new(HostExprKind::Int(0))),
                    ty: HostTypeTerm::Int64,
                }),
                HostTypeTerm::Int64,
            ),
            (
                "data-type match scrutinee",
                HostExpr::new(HostExprKind::MatchAdt {
                    scrutinee: Box::new(sibling_typed_option_list()),
                    arms: Vec::new(),
                    default_expr: Some(Box::new(HostExpr::new(HostExprKind::Int(0)))),
                    ty: HostTypeTerm::Int64,
                }),
                HostTypeTerm::Int64,
            ),
            (
                "tensor-call argument",
                HostExpr::new(HostExprKind::TensorCall {
                    helper: 0,
                    args: vec![sibling_typed_option_list()],
                    ty: HostTypeTerm::Int64,
                }),
                HostTypeTerm::Int64,
            ),
            (
                "signature-entry argument",
                signature_entry(vec![sibling_typed_option_list()], Vec::new()),
                HostTypeTerm::Unit,
            ),
            (
                "signature-entry list value",
                signature_entry(
                    Vec::new(),
                    vec![HostListEntry {
                        position: 0,
                        name: "xs".into(),
                        claim: None,
                        ty: list_ty.clone(),
                        value: HostExpr::new(HostExprKind::List(
                            vec![HostExpr::new(HostExprKind::Var(
                                "None".into(),
                                fresh_host_inference(),
                            ))],
                            fresh_host_inference(),
                        )),
                    }],
                ),
                HostTypeTerm::Unit,
            ),
        ];
        for (name, expr, ty) in cases {
            assert!(
                resolve_host_expr(expr.clone()).is_err(),
                "{name}: the case must hold an unresolved hole before conforming"
            );
            if let Err(error) = resolve_host_expr(force_host_expr_type(expr, ty, operation)) {
                panic!("{name}: conforming left the hole unresolved: {error}");
            }
        }
    })
}

/// The record construction a function body reduces to: the expression
/// whose kind is the `S` construct, or the `let` that sequences its
/// field values ahead of it.
fn record_construction(expr: &HostExpr) -> &HostExpr {
    match &expr.kind {
        HostExprKind::AdtConstruct { .. } => expr,
        HostExprKind::Let { body, .. }
            if matches!(body.kind, HostExprKind::AdtConstruct { .. }) =>
        {
            expr
        }
        HostExprKind::ResultClaimScope { body, .. }
        | HostExprKind::Let { body, .. }
        | HostExprKind::RetainedInvocation { body, .. } => record_construction(body),
        HostExprKind::FormalIngress { value, .. } => record_construction(value),
        other => panic!("no record construction under {other:?}"),
    }
}

fn operation_name(expr: &HostExpr) -> &str {
    match &expr.kind {
        HostExprKind::Builtin { name, .. } => name,
        HostExprKind::Call { callee, .. } => callee.name(),
        other => panic!("expected an operation, got {other:?}"),
    }
}

fn lowered_record_body(
    def: &str,
    operation: &Operation<'_, chelis_syntax_host::NativeStack>,
) -> HostExpr {
    let checked = surf_check(&format!(
        "type S =\n  | S {{ lo: f32, hi: f32 }}\n{def}\nout = mk(3.0f32)\n"
    ));
    let lowered = top_level_lowering_map(checked.exprs(), checked.type_env(), operation)
        .expect("checked fixture classification");
    let host = lower_host_program(
        &HostLoweringSession::new(checked.facts(), operation),
        &lowered,
    )
    .expect("record fixture must lower");
    host.functions
        .into_iter()
        .find(|function| function.name == "mk")
        .expect("mk host function")
        .body
}

#[test]
fn a_permuted_record_literal_evaluates_its_fields_in_written_order() {
    test_resources().run(|operation| {
        // chelis#3265: `hi` is written first and declared second. Its value
        // must be bound first, and the constructor must still receive the
        // declared slots `lo, hi`.
        let body = lowered_record_body(
            "def mk(x: f32) -> S = S { hi: mul(x, 2.0f32), lo: add(x, 1.0f32) }",
            operation,
        );
        let HostExprKind::Let {
            bindings,
            body: construct,
            ..
        } = &record_construction(&body).kind
        else {
            panic!("a permuted record literal must sequence its fields: {body:?}");
        };
        assert_eq!(
            bindings
                .iter()
                .map(|binding| operation_name(&binding.value))
                .collect::<Vec<_>>(),
            ["mul", "add"],
            "field values bind in written order"
        );
        let HostExprKind::AdtConstruct { fields, .. } = &construct.kind else {
            unreachable!("record_construction returns a let over the construct");
        };
        let slots = fields
            .iter()
            .map(|field| match &field.kind {
                HostExprKind::Var(name, _) => name.as_str(),
                other => panic!("a sequenced slot must read its local, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            slots,
            [bindings[1].name.as_str(), bindings[0].name.as_str()],
            "slots stay in declared order"
        );
    })
}

/// chelis#3268: the pattern-binder callee fence is lexical. A call
/// through the binder takes the unresolved-callable marker, an inner
/// binding that reuses the binder's name stops the fence for its scope
/// only, and a named callback over the binder becomes the inline callback
/// that applies the marker to its own parameters.
#[test]
fn pattern_bound_callee_fence_is_lexical_and_rewrites_named_callbacks() {
    test_resources().run(|_| {
        let int = || HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let call_to = |callee: HostCallee| {
            HostExpr::new(HostExprKind::Call {
                callee,
                args: vec![HostExpr::new(HostExprKind::Int(1))],
                arg_tys: vec![int()],
                ty: int(),
            })
        };
        let call = |name: &str| call_to(HostCallee::test_local(name));
        let callee = |expr: &HostExpr| match &expr.kind {
            HostExprKind::Call { callee, .. } => callee.clone(),
            other => panic!("expected a call, got {other:?}"),
        };
        let marker = HostCallee::unresolved_callable;
        let binders = UnordSet::from(["h".to_string()]);

        // `let g = h(1) in let h = .. in h(1)` beside a call to an unrelated
        // callback `f`.
        let mut shadowed = HostExpr::new(HostExprKind::Let {
            bindings: vec![
                HostBinding {
                    name: "g".into(),
                    display_name: None,
                    display_roots: Vec::new(),
                    ty: int(),
                    value: call("h"),
                },
                HostBinding {
                    name: "h".into(),
                    display_name: None,
                    display_roots: Vec::new(),
                    ty: int(),
                    value: call("f"),
                },
            ],
            body: Box::new(call("h")),
            ty: int(),
        });
        fence_pattern_bound_callees(&mut shadowed, &binders);
        let HostExprKind::Let { bindings, body, .. } = &shadowed.kind else {
            unreachable!("the fence keeps the node kind");
        };
        assert_eq!(callee(&bindings[0].value), marker());
        assert_eq!(callee(&bindings[1].value), HostCallee::test_local("f"));
        assert_eq!(
            callee(body),
            HostCallee::test_local("h"),
            "the inner `h` is not the pattern binder"
        );

        // A call of the program function `h` is not a call through the
        // pattern binder that shares its spelling.
        let mut function_call = call_to(HostCallee::test_function("h"));
        fence_pattern_bound_callees(&mut function_call, &binders);
        assert_eq!(callee(&function_call), HostCallee::test_function("h"));

        let named = |function: &str| {
            HostExpr::new(HostExprKind::Map {
                callback: HostCallback {
                    stationary_accumulator: None,
                    accumulator_delta: None,
                    accumulator_composition: None,
                    kind: HostCallbackKind::Named {
                        callee: HostCallee::test_local(function),
                        params: vec![HostParam {
                            name: "x".into(),
                            ty: int(),
                        }],
                    },
                    ret_ty: int(),
                },
                list: Box::new(HostExpr::new(HostExprKind::List(Vec::new(), int()))),
                ty: int(),
            })
        };
        let mut through_binder = named("h");
        fence_pattern_bound_callees(&mut through_binder, &binders);
        let HostExprKind::Map { callback, .. } = &through_binder.kind else {
            unreachable!("the fence keeps the node kind");
        };
        let HostCallbackKind::Inline { params, body } = &callback.kind else {
            panic!("a named callback over a binder must become inline");
        };
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "x");
        let HostExprKind::Call { callee, args, .. } = &body.kind else {
            panic!("the inline body must apply the marker");
        };
        assert_eq!(*callee, marker());
        assert!(
            matches!(&args[..], [HostExpr { kind: HostExprKind::Var(name, _), .. }] if name == "x")
        );

        let mut through_formal = named("f");
        fence_pattern_bound_callees(&mut through_formal, &binders);
        let HostExprKind::Map { callback, .. } = &through_formal.kind else {
            unreachable!("the fence keeps the node kind");
        };
        assert!(matches!(
            &callback.kind,
            HostCallbackKind::Named { callee, .. } if callee.view() == HostCalleeView::Local("f")
        ));
    })
}
