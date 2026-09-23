//! IR draw-key probes: [05-OP-37], [05-RNG-1], and spec/10 §3.2.
//! Baseline defects were first reproduced through the legacy Dag evaluator.
//! These conformance probes evaluate the lowered graph with a `RandomFrame`,
//! whose draw keys take the ordinals an independent [05-RNG-1] transcription
//! predicts.
use chelis_ir::Dag;
use chelis_ir::dag::{DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{RandomFrame, TensorValue, eval_tensor_roots_with_frame};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn evaluate(rate: f64, seed: u64, count: usize) -> Result<Vec<f64>, String> {
    let mut frame = RandomFrame::inherited(seed, 0);
    let dag = lower(
        &format!(
            "(app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} f32)}} {rate:?}))"
        ),
        count,
        UnordMap::new(),
    );
    run(&dag, count, &mut frame)
}

fn lower(source: &str, count: usize, defs: UnordMap<String, chelis_deep::Expr>) -> Dag {
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
    chelis_ir::lower::try_lower_subexpr_program(&expression, inputs, UnordMap::new(), defs).unwrap()
}

fn run(dag: &Dag, count: usize, frame: &mut RandomFrame) -> Result<Vec<f64>, String> {
    let values = eval_tensor_roots_with_frame(dag, dag.roots(), frame, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![count], vec![1.0; count]))
    })?;
    Ok(values[&dag.roots()[0]].to_f64_lossy_vec())
}

fn counter(frame: &RandomFrame) -> u64 {
    frame.inherited_counter().expect("an inherited frame")
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn checked_surf(source: &str) -> chelis_types::CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).unwrap();
    let expressions =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    chelis_types::check_ir_program(&expressions).unwrap()
}

#[test]
fn staged_claim_failure_commits_the_preceding_draw_and_reuse_keeps_live_ordinals() {
    use chelis_ir::host::staged::HostStage;
    let checked = checked_surf(
        "def sample[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {\n dead = dropout(source, 0.0f32)\n dropout(reshape(x, [numel(dead), 2i64]), 0.5f32)\n}",
    );
    let kernel = chelis_ir::host::host_def_kernel(
        &chelis_ir::host::HostLoweringSession::new(&checked),
        "sample",
    )
    .unwrap()
    .unwrap();
    let staged = kernel.staged.as_ref().expect("numel-fed reshape stages");
    let mut next = 5;
    for count in [2, 3, 2] {
        let before = next;
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
            for stage in staged.stages() {
                match stage {
                    HostStage::Source {
                        captures, output, ..
                    } => {
                        // This source is numel(dead): read the actually executed
                        // capture, and prove its draw preceded this host cut.
                        assert_eq!(next, before + 1);
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
                        let mut frame = RandomFrame::inherited(42, next);
                        let computed =
                            eval_tensor_roots_with_frame(dag, dag.roots(), &mut frame, |name| {
                                values.get(name).cloned()
                            });
                        next = counter(&frame);
                        let computed = computed?;
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
            assert_eq!(next, before + 1);
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
            assert_eq!(next, before + 2);
        }
    }
}

#[test]
fn zero_rate_is_an_identity_at_empty_and_nonempty_shapes() {
    for count in [0, 1, 32] {
        assert_eq!(evaluate(0.0, 42, count).unwrap(), vec![1.0; count]);
    }
}

/// [05-OP-37]: an accepted call enters once and takes one ordinal whatever
/// its shape, the empty tensor and the zero rate included. Ported from the
/// fixed-control plan test `accepted_empty_and_zero_rate_calls_each_enter_once`,
/// which the key-operand switch deleted with the plans.
///
/// Evidentiary status: COVERAGE LOCK; it passes at 3b5f029d8. A zero-rate
/// call that skipped its draw key fails the counter assertion.
#[test]
fn accepted_empty_and_zero_rate_calls_each_take_one_ordinal() {
    let zero = "(lit {type: (t-prim {} f32)} 0.0)";
    let source = format!(
        "(app {{}} (var {{}} dropout) (app {{}} (var {{}} dropout) (var {{}} x) {zero}) {zero})"
    );
    for count in [0, 1, 32] {
        let dag = lower(&source, count, UnordMap::new());
        let mut frame = RandomFrame::inherited(42, 0);
        assert_eq!(run(&dag, count, &mut frame).unwrap(), vec![1.0; count]);
        assert_eq!(counter(&frame), 2, "count={count}");
    }
}

#[test]
fn invalid_rates_trap_even_when_the_tensor_is_empty() {
    // Nonfinite rates are covered at the kernel boundary; these are finite
    // source literals, not invented NaN/Infinity syntax.
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
fn lowering_bakes_no_seed_and_the_frame_supplies_the_key() {
    let dag = lower(
        "(app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))",
        2,
        UnordMap::new(),
    );
    let handlers = dag
        .nodes()
        .iter()
        .filter_map(|node| match node.op {
            RiscOp::DrawKey { handler, .. } => Some(handler),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        handlers,
        [chelis_ir::dag::RandomHandler::Inherited],
        "an unhandled draw reads its caller's stream, never a lowering-time seed"
    );
    let mut frame = RandomFrame::inherited(42, 5);
    run(&dag, 2, &mut frame).unwrap();
    assert_eq!(counter(&frame), 6);
}

#[test]
fn later_invalid_rate_does_not_erase_an_earlier_dead_draw() {
    let mut frame = RandomFrame::inherited(42, 5);
    let dag = lower(
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 1.0)))",
        0,
        UnordMap::new(),
    );
    assert_eq!(
        run(&dag, 0, &mut frame).unwrap_err(),
        "numeric trap: domain in dropout at f32"
    );
    assert_eq!(counter(&frame), 6);
}

#[test]
fn nested_equal_seed_scope_does_not_advance_its_parent() {
    let mut frame = RandomFrame::inherited(42, 5);
    let dag = lower(
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)) nested (handle-effect {effect: random} (lit {type: (t-prim {} i64)} 42) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))",
        0,
        UnordMap::new(),
    );
    assert!(run(&dag, 0, &mut frame).unwrap().is_empty());
    assert_eq!(counter(&frame), 7);
}

#[test]
fn valid_uniform_like_occupies_one_shared_ordinal_between_dropout_calls() {
    let mut frame = RandomFrame::inherited(42, 0);
    let dag = lower(
        "(let {} (bind {} dead (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.0)) uniform (app {} (var {} uniform_like) (var {} x) (lit {type: (t-prim {} f32)} 0.0) (lit {type: (t-prim {} f32)} 1.0))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        32,
        UnordMap::new(),
    );
    let expected = (0..32)
        .map(|index| {
            let word =
                splitmix64(42 ^ splitmix64(2).rotate_left(17) ^ splitmix64(index).rotate_left(41));
            let unit = ((word >> 11) as f64 / 9007199254740992.0) as f32;
            if unit < 0.5 { 0.0 } else { 2.0 }
        })
        .collect::<Vec<_>>();
    assert_eq!(run(&dag, 32, &mut frame).unwrap(), expected);
    assert_eq!(counter(&frame), 3);
}

#[test]
fn local_movement_failure_occurs_after_the_earlier_entered_draw() {
    let checked = checked_surf(
        "def sample[n](x: tensor[n, f32]) -> tensor[3, f32] = {\n dead = dropout(x, 0.0f32)\n shrink(x, [[0i64, 3i64]])\n}\n",
    );
    let kernel = chelis_ir::host::host_def_kernel(
        &chelis_ir::host::HostLoweringSession::new(&checked),
        "sample",
    )
    .unwrap()
    .expect("the helper is a kernel");
    let mut frame = RandomFrame::inherited(42, 5);
    let error = eval_tensor_roots_with_frame(&kernel.dag, kernel.dag.roots(), &mut frame, |_| {
        Some(TensorValue::from_vec(vec![2], vec![1.0; 2]))
    })
    .unwrap_err();
    assert!(error.contains("shrink"), "{error}");
    assert_eq!(
        counter(&frame),
        6,
        "a later local bounds failure cannot be hoisted before the dead draw"
    );

    // The earlier draw's own trap is the one reported: an invalid rate is
    // validated at its draw, before the later bounds failure, and consumes no
    // ordinal.
    let checked = checked_surf(
        "def sample[n](x: tensor[n, f32]) -> tensor[3, f32] = {\n dead = dropout(x, 2.0f32)\n shrink(x, [[0i64, 3i64]])\n}\n",
    );
    let kernel = chelis_ir::host::host_def_kernel(
        &chelis_ir::host::HostLoweringSession::new(&checked),
        "sample",
    )
    .unwrap()
    .expect("the helper is a kernel");
    let mut frame = RandomFrame::inherited(42, 5);
    let error = eval_tensor_roots_with_frame(&kernel.dag, kernel.dag.roots(), &mut frame, |_| {
        Some(TensorValue::from_vec(vec![2], vec![1.0; 2]))
    })
    .unwrap_err();
    assert_eq!(error, "numeric trap: domain in dropout at f32");
    assert_eq!(counter(&frame), 5);
}

#[test]
fn explicit_drop_in_gradient_retains_draws_and_verified_terminal_ownership() {
    let checked = checked_surf(
        "def loss(x: tensor[32, f32]) -> tensor[f32] = {\n dead = dropout(x, 0.0f32)\n _ = drop(dead)\n sum(dropout(x, 0.5f32), 0)\n}\ndef sample(x: tensor[32, f32]) -> tensor[32, f32] = grad(loss)(x)\n",
    );
    let mut frame = RandomFrame::inherited(42, 0);
    let kernel = chelis_ir::host::host_def_kernel(
        &chelis_ir::host::HostLoweringSession::new(&checked),
        "sample",
    )
    .unwrap()
    .unwrap();
    let dag = &kernel.dag;
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::DrawKey { .. }))
            .count(),
        2,
        "two actual forward draws"
    );
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::DropoutReplay))
            .count(),
        1,
        "one live backward replay"
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
    assert_eq!(run(dag, 32, &mut frame).unwrap(), expected);
    assert_eq!(counter(&frame), 2, "Drop and replay consume no ordinal");
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
        let mut frame = RandomFrame::inherited(42, 0);
        let dag = lower(
            "(app {} (grad {} (var {} loss)) (var {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} x))",
            32,
            defs,
        );
        let values = run(&dag, 32, &mut frame).unwrap();
        let expected = if dead {
            vec![1.0; 32]
        } else {
            evaluate(0.5, 42, 32).unwrap()
        };
        assert_eq!(values, expected);
        assert_eq!(counter(&frame), 1);
    }
}

/// [05-RNG-1] enters only the selected arm of a runtime `if`. A kernel
/// `where` computes both arms, so every draw lowered in an arm carries the
/// arm's path condition as its activation, on its draw key and on its
/// primitive: an unselected draw neither validates its controls nor takes an
/// ordinal (chelis#2410). Rows cover both primitives, both arms, a nested
/// arm, and a runtime bound that traps when it is validated. A literal
/// condition lowers only its taken arm, with no activation.
///
/// Evidentiary status: REGRESSION TEST. At dcc9256c4 lowering refused every
/// `dropout` row with a #2410 rejection, and the `uniform_like` rows drew
/// with no activation: the unselected arm took an ordinal and trapped on
/// its invalid bound.
#[test]
fn a_draw_in_a_runtime_arm_is_activated_by_its_arm_path() {
    let scalar = |precision| TensorType {
        dims: Vec::new(),
        precision,
    };
    let lower_arm = |source: &str| {
        let expression = chelis_deep::parser::parse_str(source)
            .unwrap()
            .pop()
            .unwrap();
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        for name in ["flag", "inner"] {
            inputs.insert(name.into(), scalar(Prim::Bool));
        }
        inputs.insert("low".into(), scalar(Prim::F32));
        chelis_ir::lower::try_lower_subexpr_program(
            &expression,
            inputs,
            UnordMap::new(),
            UnordMap::new(),
        )
        .unwrap()
    };
    let run_arm = |dag: &Dag, flag: bool, inner: bool, low: f64, frame: &mut RandomFrame| {
        let values = eval_tensor_roots_with_frame(dag, dag.roots(), frame, |name| match name {
            "x" => Some(TensorValue::from_vec(vec![4], vec![1.0; 4])),
            "flag" | "inner" => Some(
                TensorValue::finalize_from_wide_int(
                    "arm flag",
                    Prim::Bool,
                    Vec::new(),
                    vec![i64::from(if name == "flag" { flag } else { inner })],
                )
                .unwrap(),
            ),
            "low" => Some(TensorValue::from_storage(
                Vec::new(),
                finalize_tensor("arm bound", Prim::F32, RawTensor::Float(vec![low])).unwrap(),
            )),
            _ => None,
        })?;
        Ok::<_, String>(values[&dag.roots()[0]].to_f64_lossy_vec())
    };
    let dropout = "(app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))";
    let uniform =
        "(app {} (var {} uniform_like) (var {} x) (var {} low) (lit {type: (t-prim {} f32)} 1.0))";
    let x = "(var {} x)";
    for draw in [dropout, uniform] {
        let rows = [
            (
                format!("(if {{}} (var {{}} flag) {draw} {x})"),
                [true, true],
            ),
            (
                format!("(if {{}} (var {{}} flag) {x} {draw})"),
                [false, true],
            ),
            (
                format!("(if {{}} (var {{}} flag) (if {{}} (var {{}} inner) {draw} {x}) {x})"),
                [true, true],
            ),
            (
                format!("(if {{}} (var {{}} flag) {x} (if {{}} (var {{}} inner) {x} {draw}))"),
                [false, false],
            ),
        ];
        for (source, [selected_flag, selected_inner]) in rows {
            let dag = lower_arm(&source);
            let mut activated = 0;
            for node in dag.nodes() {
                let fixed = match node.op {
                    RiscOp::DrawKey { draw, .. } => draw.control_count(),
                    RiscOp::Dropout => 3,
                    RiscOp::UniformLike => 4,
                    _ => continue,
                };
                let activation = node.inputs.get(fixed).copied();
                assert!(
                    activation.is_some(),
                    "{source}: {:?} is unactivated",
                    node.op
                );
                assert_eq!(
                    dag.get(activation.unwrap()).unwrap().output_type,
                    scalar(Prim::Bool)
                );
                activated += 1;
            }
            assert_eq!(activated, 2, "{source}");
            for flag in [false, true] {
                for inner in [false, true] {
                    let selected = flag == selected_flag
                        && (!source.contains("inner") || inner == selected_inner);
                    // A valid bound: the selected arm takes one ordinal.
                    let mut frame = RandomFrame::inherited(42, 0);
                    let values = run_arm(&dag, flag, inner, 0.0, &mut frame).unwrap();
                    assert_eq!(
                        counter(&frame),
                        u64::from(selected),
                        "{source} {flag} {inner}"
                    );
                    if !selected {
                        assert_eq!(values, vec![1.0; 4], "{source} {flag} {inner}");
                    }
                    // A bound above `high`: validated, and so trapping, only
                    // in the selected arm. `dropout`'s literal rate is valid.
                    let mut frame = RandomFrame::inherited(42, 0);
                    let invalid = run_arm(&dag, flag, inner, 2.0, &mut frame);
                    if selected && draw == uniform {
                        let error = invalid.unwrap_err();
                        assert!(error.contains("domain in uniform_like"), "{error}");
                        assert_eq!(counter(&frame), 0, "validation consumes no ordinal");
                    } else {
                        invalid.unwrap_or_else(|error| panic!("{source} {flag} {inner}: {error}"));
                    }
                }
            }
        }
    }
    let literal = lower_arm(&format!(
        "(if {{}} (lit {{type: (t-prim {{}} bool)}} true) {dropout} {x})"
    ));
    assert!(
        !literal
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Where))
    );
    let key = literal
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::DrawKey { .. }))
        .expect("the taken arm draws");
    assert_eq!(
        key.inputs.len(),
        1,
        "a literal condition adds no activation"
    );
}
