//! IR plan probes: [05-OP-37], [05-RNG-1], and spec/10 §3.2.
//! Baseline defects were first reproduced through the legacy Dag evaluator.
//! These conformance probes now exercise the additive source-plan boundary;
//! legacy APIs are intentionally not relabeled as repaired.
use chelis_ir::dag::{DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_plan_with_strict};
use chelis_ir::evaluation::{EvaluationPlan, RandomExecutionContext};
use chelis_ir::host::RandomLoweringState;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

// #1764: admission and lowering must specialize the same typed rate, before AD.
fn typed_rate_plan(source: &str) -> Result<EvaluationPlan, String> {
    let expressions = chelis_deep::parser::parse_and_stamp(source).unwrap();
    typed_rate_plan_from_ir(&expressions)
}

fn typed_rate_plan_from_ir(expressions: &[chelis_deep::Expr]) -> Result<EvaluationPlan, String> {
    let checked =
        chelis_types::check_ir_program(expressions, chelis_types::ProgramOrigin::Unlinked).unwrap();
    let context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 0,
    });
    let product = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        chelis_ir::host::host_def_evaluation_plan(
            &chelis_ir::host::HostLoweringSession::new(&checked, runtime),
            "sample",
            &context,
        )
    })
    .map_err(|error| error.to_string())?
    .ok_or("no tensor kernel")?;
    if product.profile() != chelis_ir::evaluation::EvaluationProfile::FixedControl {
        return Err(format!("unexpected profile: {:?}", product.profile()));
    }
    product
        .plan()
        .cloned()
        .ok_or("no fixed-control plan".into())
}

#[test]
fn typed_static_rate_checks_only_actual_transitive_captures() {
    // Preserve the original static-capture carriers and metadata without
    // normalizing the legacy AST through the canonical Deep printer.
    let fixture_plan = |source: &str| {
        let expressions: Vec<chelis_deep::Expr> =
            serde_json::from_str(source).expect("original static-capture AST");
        typed_rate_plan_from_ir(&expressions)
    };
    let reference =
        fixture_plan(include_str!("fixtures/core/tests/dropout_fixed_stream_ir/typed_static_rate_checks_only_actual_transitive_captures.json"))
            .unwrap();
    let execute = |plan: &EvaluationPlan| {
        let mut context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 0,
        });
        let values = eval_tensor_plan_with_strict(plan, &mut context, |_| {
            Some(TensorValue::from_vec(vec![8], vec![1.0; 8]))
        })
        .unwrap();
        (
            values[&plan.dag_for_inspection().roots()[0]].to_f64_lossy_vec(),
            context.state().counter,
        )
    };
    for source in [
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_0_case_0.json"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_0_case_1.json"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_0_case_2.json"),
    ] {
        let plan = fixture_plan(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
        assert_eq!(execute(&plan), execute(&reference), "{source}");
    }
    for source in [
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_1_case_0.json"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_1_case_1.json"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_1_case_2.json"),
    ] {
        assert!(fixture_plan(source).is_err(), "{source}");
    }
}

#[test]
fn typed_static_rate_specializes_helpers_and_gradient_captures() {
    for (dtype, prim) in [
        ("f16", Prim::F16),
        ("bf16", Prim::Bf16),
        ("f32", Prim::F32),
        ("f64", Prim::F64),
    ] {
        for template in [
            include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_2_case_0.deep"),
            include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_2_case_1.deep"),
            include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_2_case_2.deep"),
        ] {
            let source = template.replace("DTYPE", dtype);
            let plan = typed_rate_plan(&source).unwrap_or_else(|error| panic!("{source}\n{error}"));
            let input = chelis_types::finalize_tensor(
                "test",
                prim,
                chelis_types::RawTensor::Float(vec![1.0; 8]),
            )
            .unwrap();
            let mut context = RandomExecutionContext::new(RandomLoweringState {
                seed: Some(42),
                counter: 0,
            });
            for ordinal in 0..2 {
                let values = eval_tensor_plan_with_strict(&plan, &mut context, |_| {
                    Some(TensorValue::from_storage(vec![8], input.clone()))
                })
                .unwrap();
                let expected = (0..8)
                    .map(|index| {
                        let word = splitmix64(
                            42 ^ splitmix64(ordinal).rotate_left(17)
                                ^ splitmix64(index).rotate_left(41),
                        );
                        if word >> 63 == 0 { 0.0 } else { 2.0 }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    values[&plan.dag_for_inspection().roots()[0]].to_f64_lossy_vec(),
                    expected
                );
                assert_eq!(context.state().counter, ordinal + 1);
            }
        }
    }
}

#[test]
fn typed_static_rate_preserves_source_width_and_distinct_call_bindings() {
    let source = include_str!(
        "fixtures/core/tests/dropout_fixed_stream_ir/typed_static_rate_preserves_source_width_and_distinct_call_bindings.deep"
    );
    let plan = typed_rate_plan(source).unwrap();
    let rates = plan
        .dag_for_inspection()
        .nodes()
        .iter()
        .filter_map(|node| {
            if let RiscOp::Dropout { rate, .. } = node.op {
                Some(rate)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(rates, vec![f64::from(0.1f32), 0.1f64]);
}

#[test]
fn typed_static_rate_does_not_duplicate_arguments() {
    let expected = vec![0.5, 0.25];
    let source =
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/distinct_argument_calls.deep");
    let plan = typed_rate_plan(source).unwrap();
    let rates = plan
        .dag_for_inspection()
        .nodes()
        .iter()
        .filter_map(|node| {
            if let RiscOp::Dropout { rate, .. } = node.op {
                Some(rate)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(rates, expected, "{source}");
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 0,
    });
    eval_tensor_plan_with_strict(&plan, &mut context, |_| {
        Some(TensorValue::from_vec(vec![8], vec![1.0; 8]))
    })
    .unwrap();
    assert_eq!(context.state().counter, expected.len() as u64);
}

#[test]
fn typed_static_rate_rejects_runtime_formals_and_shadowed_aliases() {
    for source in [
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_3_case_0.deep"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_3_case_1.deep"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_3_case_2.deep"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_3_case_3.deep"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_3_case_4.deep"),
        include_str!("fixtures/core/tests/dropout_fixed_stream_ir/rate_group_3_case_5.deep"),
    ] {
        let error = typed_rate_plan(source).unwrap_err();
        assert!(
            error.contains("RuntimeRate")
                || error.contains("statically-resolvable")
                || error == "no tensor kernel",
            "{source}\n{error}"
        );
    }
}

fn evaluate(rate: f64, seed: u64, count: usize) -> Result<Vec<f64>, String> {
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(seed),
        counter: 0,
    });
    let plan = lower(
        &format!(
            "(app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} f32)}} {rate:?}))"
        ),
        count,
        UnordMap::new(),
        &context,
    );
    run(&plan, count, &mut context)
}

fn lower(
    source: &str,
    count: usize,
    defs: UnordMap<String, chelis_deep::Expr>,
    context: &RandomExecutionContext,
) -> EvaluationPlan {
    let expression = chelis_deep::parser::parse_str(source)
        .unwrap()
        .pop()
        .unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorType {
            dims: vec![DimInfo::Lit(count)],
            precision: Prim::F32,
        },
    );
    chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        chelis_ir::lower::try_lower_subexpr_evaluation_plan(
            &expression,
            inputs,
            UnordMap::new(),
            defs,
            context,
            runtime,
        )
    })
    .unwrap()
}

fn run(
    plan: &EvaluationPlan,
    count: usize,
    context: &mut RandomExecutionContext,
) -> Result<Vec<f64>, String> {
    let values = eval_tensor_plan_with_strict(plan, context, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![count], vec![1.0; count]))
    })?;
    Ok(values[&plan.dag_for_inspection().roots()[0]].to_f64_lossy_vec())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[test]
fn staged_claim_failure_commits_the_preceding_draw_and_reuse_keeps_live_ordinals() {
    use chelis_ir::host::staged::HostStage;
    let source = include_str!(
        "fixtures/core/tests/dropout_fixed_stream_ir/staged_claim_failure_commits_the_preceding_draw_and_reuse_keeps_live_ordinals.deep"
    );
    let expressions = chelis_deep::parser::parse_and_stamp(source).unwrap();
    let checked =
        chelis_types::check_ir_program(&expressions, chelis_types::ProgramOrigin::Unlinked)
            .unwrap();
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 5,
    });
    let product = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        chelis_ir::host::host_def_evaluation_plan(
            &chelis_ir::host::HostLoweringSession::new(&checked, runtime),
            "sample",
            &context,
        )
    })
    .unwrap()
    .unwrap();
    let staged = product.kernel_for_inspection().staged.as_ref().unwrap();
    let execution = product
        .staged_plan()
        .expect("checked stages retain Random provenance");
    assert!(product.plan().is_none());
    for count in [2, 3, 2] {
        let before = context.state().counter;
        let mut values = UnordMap::new();
        values.insert(
            "source".into(),
            TensorValue::from_vec(vec![count], vec![1.0; count]),
        );
        values.insert(
            "x".into(),
            TensorValue::from_vec(vec![count * 2], vec![1.0; count * 2]),
        );
        let result = (|| {
            let mut frame = execution.frame(&mut context)?;
            for stage in staged.stages() {
                match stage {
                    HostStage::Source {
                        captures, output, ..
                    } => {
                        // This source is numel(dead): read the actually executed
                        // capture, and prove its draw preceded this host cut.
                        frame.with_context(|context| {
                            assert_eq!(context.state().counter, before + 1)
                        });
                        let captured = &values[&captures[0].value];
                        assert_eq!(captured.to_f64_lossy_vec(), vec![1.0; count]);
                        let extent = chelis_types::scalar_from_i64(
                            "numel",
                            Prim::Int64,
                            captured.storage().len() as i64,
                        )
                        .unwrap();
                        values.insert(
                            output.clone(),
                            TensorValue::from_storage(
                                vec![],
                                chelis_types::tensor_from_scalars(Prim::Int64, &[extent]),
                            ),
                        );
                    }
                    HostStage::Kernel { dag, outputs } => {
                        let computed = frame.eval_next_kernel(|name| values.get(name).cloned())?;
                        for (output, root) in outputs.iter().zip(dag.roots()) {
                            values.insert(output.clone(), computed[root].clone());
                        }
                    }
                }
            }
            Ok::<_, String>(values[staged.output()].clone())
        })();
        if count == 3 {
            let error = result.unwrap_err();
            assert!(
                error.contains("claimed = 2") && error.contains("reshape axis 0 = 3"),
                "{error}"
            );
            assert_eq!(context.state().counter, before + 1);
        } else {
            let actual = result.unwrap();
            assert_eq!(actual.shape, vec![2, 2]);
            let expected = (0..4)
                .map(|index| {
                    let word = splitmix64(
                        42 ^ splitmix64(before + 1).rotate_left(17)
                            ^ splitmix64(index).rotate_left(41),
                    );
                    if (((word >> 11) as f64 / 9007199254740992.0) as f32) < 0.5 {
                        0.0
                    } else {
                        2.0
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(actual.to_f64_lossy_vec(), expected);
            assert_eq!(context.state().counter, before + 2);
        }
    }
    let mut wrong = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(7),
        counter: 19,
    });
    assert!(
        execution
            .frame(&mut wrong)
            .err()
            .unwrap()
            .contains("inherited seed")
    );
    assert_eq!(wrong.state().counter, 19);
}

#[test]
fn source_profile_names_exclusions_before_plan_construction() {
    use chelis_ir::evaluation::{EvaluationProfile, LegacyEvaluationReason as Reason};
    let draw = "(app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))";
    for (source, reason) in [
        (format!("(vmap {{}} {draw})"), Reason::RandomVmap),
        (
            format!("(grad {{}} (grad {{}} {draw}))"),
            Reason::HigherOrderAd,
        ),
        (
            format!("(handle-effect {{effect: random}} (var {{}} seed) {draw})"),
            Reason::RuntimeSeed,
        ),
        (
            format!("(handle-effect {{effect: resource}} (var {{}} token) {draw})"),
            Reason::ResourceScope,
        ),
        (
            format!("(if {{}} (var {{}} condition) {draw} (var {{}} x))"),
            Reason::DynamicControl,
        ),
        (
            "(app {} (var {} dropout) (var {} x) (var {} rate))".into(),
            Reason::RuntimeRate,
        ),
        // An excluded pure declaration must not become NoDropout and then
        // silently join a selected fixed-control declaration's plan.
        (
            "(if {} (var {} condition) (var {} x) (var {} y))".into(),
            Reason::DynamicControl,
        ),
    ] {
        let expression = chelis_deep::parser::parse_str(&source)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                chelis_ir::lower::evaluation_profile(&expression, &UnordMap::new(), runtime)
            }),
            EvaluationProfile::Legacy(reason),
            "{source}"
        );
        let context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 9,
        });
        let error = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            chelis_ir::lower::try_lower_subexpr_evaluation_plan(
                &expression,
                UnordMap::new(),
                UnordMap::new(),
                UnordMap::new(),
                &context,
                runtime,
            )
        })
        .unwrap_err();
        assert!(error.message.contains(&format!("{reason:?}")), "{error:?}");
        assert_eq!(context.state().counter, 9);
    }
}

#[test]
fn source_profile_respects_lexical_and_top_level_dropout_shadowing() {
    use chelis_ir::evaluation::{EvaluationProfile, LegacyEvaluationReason as Reason};
    for source in [
        "(fn {} (params {} dropout x) (app {} (var {} dropout) (var {} x) (lit {} 0.5)))",
        "(let {} (bind {} dropout (fn {} (params {} x rate) (var {} x))) (app {} (var {} dropout) (var {} x) (lit {} 0.5)))",
    ] {
        let expression = chelis_deep::parser::parse_str(source)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                chelis_ir::lower::evaluation_profile(&expression, &UnordMap::new(), runtime)
            }),
            EvaluationProfile::Legacy(Reason::NoDropout)
        );
    }
    let expression =
        chelis_deep::parser::parse_str("(app {} (var {} dropout) (var {} x) (lit {} 0.5))")
            .unwrap()
            .pop()
            .unwrap();
    let body = chelis_deep::parser::parse_str("(fn {} (params {} x rate) (var {} x))")
        .unwrap()
        .pop()
        .unwrap();
    let mut defs = UnordMap::new();
    defs.insert("dropout".into(), body);
    assert_eq!(
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            chelis_ir::lower::evaluation_profile(&expression, &defs, runtime)
        }),
        EvaluationProfile::Legacy(Reason::NoDropout)
    );
}

#[test]
fn zero_rate_is_an_identity_at_empty_and_nonempty_shapes() {
    for count in [0, 1, 32] {
        assert_eq!(evaluate(0.0, 42, count).unwrap(), vec![1.0; count]);
    }
}

#[test]
fn invalid_rates_trap_even_when_the_tensor_is_empty() {
    // Nonfinite stored rates are covered directly at the private plan boundary;
    // these are finite source literals, not invented NaN/Infinity syntax.
    for rate in [-0.5, 1.0, 2.0] {
        for count in [0, 1, 32] {
            let result = evaluate(rate, 42, count);
            assert!(result.is_err(), "rate={rate:?}, count={count}: {result:?}");
            let error = result.unwrap_err();
            assert_eq!(error, "numeric trap: domain in dropout at f32");
        }
    }
}

#[test]
fn ordinal_zero_uses_the_canonical_source_word_and_f32_mask() {
    let expected = (0..32u64)
        .map(|index| {
            let word =
                splitmix64(42 ^ splitmix64(0).rotate_left(17) ^ splitmix64(index).rotate_left(41));
            let unit = ((word >> 11) as f64 / (1u64 << 53) as f64) as f32;
            if unit < 0.5 { 0.0 } else { 2.0 }
        })
        .collect::<Vec<_>>();
    assert_eq!(evaluate(0.5, 42, 32).unwrap(), expected);
}

#[test]
fn lowering_does_not_relabel_a_call_key_as_the_raw_seed() {
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 5,
    });
    let plan = lower(
        "(app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))",
        2,
        UnordMap::new(),
        &context,
    );
    assert_eq!(context.state().counter, 5, "lowering does not enter Random");
    let seeds = plan
        .dag_for_inspection()
        .nodes()
        .iter()
        .filter_map(|node| match node.op {
            RiscOp::Dropout { seed, .. } => Some(seed),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(seeds, [42], "spec/10 seed is the raw signed-seed bit image");
    run(&plan, 2, &mut context).unwrap();
    assert_eq!(context.state().counter, 6);
}

#[test]
fn later_invalid_rate_does_not_erase_an_earlier_dead_draw() {
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 5,
    });
    let plan = lower(
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 1.0)))",
        0,
        UnordMap::new(),
        &context,
    );
    assert_eq!(
        run(&plan, 0, &mut context).unwrap_err(),
        "numeric trap: domain in dropout at f32"
    );
    assert_eq!(context.state().counter, 6);
}

#[test]
fn nested_equal_seed_scope_does_not_advance_its_parent() {
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 5,
    });
    let plan = lower(
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)) nested (handle-effect {effect: random} (lit {type: (t-prim {} i64)} 42) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))",
        0,
        UnordMap::new(),
        &context,
    );
    assert!(run(&plan, 0, &mut context).unwrap().is_empty());
    assert_eq!(context.state().counter, 7);
}

#[test]
fn valid_uniform_like_occupies_one_shared_ordinal_between_dropout_calls() {
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 0,
    });
    let plan = lower(
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)) uniform (app {} (var {} uniform_like) (var {} x) (lit {type: (t-prim {} f32)} 0.0) (lit {type: (t-prim {} f32)} 1.0))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        32,
        UnordMap::new(),
        &context,
    );
    let expected = (0..32)
        .map(|index| {
            let word =
                splitmix64(42 ^ splitmix64(2).rotate_left(17) ^ splitmix64(index).rotate_left(41));
            let unit = ((word >> 11) as f64 / 9007199254740992.0) as f32;
            if unit < 0.5 { 0.0 } else { 2.0 }
        })
        .collect::<Vec<_>>();
    assert_eq!(run(&plan, 32, &mut context).unwrap(), expected);
    assert_eq!(context.state().counter, 3);
}

#[test]
fn local_movement_failure_occurs_after_the_earlier_entered_draw() {
    let expressions = chelis_deep::parser::parse_and_stamp(include_str!("fixtures/core/tests/dropout_fixed_stream_ir/local_movement_failure_occurs_after_the_earlier_entered_draw.deep")).unwrap();
    let checked =
        chelis_types::check_ir_program(&expressions, chelis_types::ProgramOrigin::Unlinked)
            .unwrap();
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 5,
    });
    let kernel = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        chelis_ir::host::host_def_evaluation_plan(
            &chelis_ir::host::HostLoweringSession::new(&checked, runtime),
            "sample",
            &context,
        )
    })
    .unwrap()
    .unwrap();
    let plan = kernel
        .plan()
        .expect("fixed-control helper has execution transport");
    let error = eval_tensor_plan_with_strict(plan, &mut context, |_| {
        Some(TensorValue::from_vec(vec![2], vec![1.0; 2]))
    })
    .unwrap_err();
    assert!(error.contains("shrink"), "{error}");
    assert_eq!(
        context.state().counter,
        6,
        "a later local bounds failure cannot be hoisted before the dead draw"
    );
}

#[test]
fn explicit_drop_in_gradient_retains_draws_and_verified_terminal_ownership() {
    let source = include_str!(
        "fixtures/core/tests/dropout_fixed_stream_ir/explicit_drop_in_gradient_retains_draws_and_verified_terminal_ownership.deep"
    );
    let expressions = chelis_deep::parser::parse_and_stamp(source).unwrap();
    let checked =
        chelis_types::check_ir_program(&expressions, chelis_types::ProgramOrigin::Unlinked)
            .unwrap();
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 0,
    });
    let kernel = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        chelis_ir::host::host_def_evaluation_plan(
            &chelis_ir::host::HostLoweringSession::new(&checked, runtime),
            "sample",
            &context,
        )
    })
    .unwrap()
    .unwrap();
    let plan = kernel.plan().unwrap();
    let dag = plan.dag_for_inspection();
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Dropout { .. }))
            .count(),
        3,
        "two actual forwards and one live backward replay"
    );
    assert!(
        dag.nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Drop))
    );
    assert!(
        dag.roots()
            .iter()
            .all(|id| !matches!(dag.get(*id).unwrap().op, RiscOp::Drop))
    );
    let ownership = chelis_ir::ownership::lower_dag_ownership(dag.clone()).unwrap();
    let verified = chelis_ir::ownership::verify_ownership(ownership).unwrap();
    assert!(verified.emission().actions().any(|action| matches!(
        action,
        chelis_ir::ownership::VerifiedDagAction::OwnedDrop { .. }
    )));
    let expected = (0..32)
        .map(|index| {
            let word =
                splitmix64(42 ^ splitmix64(1).rotate_left(17) ^ splitmix64(index).rotate_left(41));
            if (((word >> 11) as f64 / 9007199254740992.0) as f32) < 0.5 {
                0.0
            } else {
                2.0
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(run(plan, 32, &mut context).unwrap(), expected);
    assert_eq!(
        context.state().counter,
        2,
        "Drop and replay consume no ordinal"
    );
}

#[test]
fn source_ad_replays_a_mask_with_no_extra_draw_and_preserves_dead_forward_calls() {
    for dead in [false, true] {
        let body = if dead {
            "(let {} (bind {} dead (app {} (var {} dropout) (var {} t) (lit {type: (t-prim {} f32)} 0.0))) (var {} t))"
        } else {
            "(app {} (var {} dropout) (var {} t) (lit {type: (t-prim {} f32)} 0.5))"
        };
        let function = format!(
            "(fn {{}} (params {{}} (t {{type: (t-tensor {{}} (d-lit {{}} 32) (t-prim {{}} f32))}})) (app {{type: (t-tensor {{}} (t-prim {{}} f32))}} (var {{}} sum) {body} (lit {{type: (t-prim {{}} i32)}} 0)))"
        );
        let mut defs = UnordMap::new();
        defs.insert(
            "loss".into(),
            chelis_deep::parser::parse_str(&function)
                .unwrap()
                .pop()
                .unwrap(),
        );
        let mut context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 0,
        });
        let plan = lower(
            "(app {} (grad {} (var {} loss)) (var {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} x))",
            32,
            defs,
            &context,
        );
        let values = run(&plan, 32, &mut context).unwrap();
        let expected = if dead {
            vec![1.0; 32]
        } else {
            evaluate(0.5, 42, 32).unwrap()
        };
        assert_eq!(values, expected);
        assert_eq!(context.state().counter, 1);
    }
}
