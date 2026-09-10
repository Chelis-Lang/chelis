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
    chelis_ir::lower::try_lower_subexpr_evaluation_plan(
        &expression,
        inputs,
        UnordMap::new(),
        defs,
        context,
    )
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
            chelis_ir::lower::evaluation_profile(&expression, &UnordMap::new()),
            EvaluationProfile::Legacy(reason),
            "{source}"
        );
        let context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 9,
        });
        let error = chelis_ir::lower::try_lower_subexpr_evaluation_plan(
            &expression,
            UnordMap::new(),
            UnordMap::new(),
            UnordMap::new(),
            &context,
        )
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
            chelis_ir::lower::evaluation_profile(&expression, &UnordMap::new()),
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
        chelis_ir::lower::evaluation_profile(&expression, &defs),
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
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)) nested (handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))",
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
    let declarations = chelis_surf::parser::parse_str("def sample(x: tensor[n, f32]) -> tensor[3, f32] = {\n dead = dropout(x, 0.0f32)\n shrink(x, [[0i64, 3i64]])\n}\n").unwrap();
    let expressions = chelis_surf::desugar::desugar_program(&declarations);
    let checked = chelis_types::check_ir_program(&expressions).unwrap();
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 5,
    });
    let kernel = chelis_ir::host::host_def_evaluation_plan(&checked, "sample", &context)
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
    let source = "def loss(x: tensor[32, f32]) -> tensor[f32] = {\n dead = dropout(x, 0.0f32)\n _ = drop(dead)\n sum(dropout(x, 0.5f32), 0)\n}\ndef sample(x: tensor[32, f32]) -> tensor[32, f32] = grad(loss)(x)\n";
    let declarations = chelis_surf::parser::parse_str(source).unwrap();
    let expressions = chelis_surf::desugar::desugar_program(&declarations);
    let checked = chelis_types::check_ir_program(&expressions).unwrap();
    let mut context = RandomExecutionContext::new(RandomLoweringState {
        seed: Some(42),
        counter: 0,
    });
    let kernel = chelis_ir::host::host_def_evaluation_plan(&checked, "sample", &context)
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
            "(fn {{}} (params {{}} (t {{type: (t-tensor {{}} (d-lit {{}} 32) (t-prim {{}} f32))}})) (app {{type: (t-tensor {{}} (t-prim {{}} f32))}} (var {{}} sum) {body} (lit {{type: (t-prim {{}} int32)}} 0)))"
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
