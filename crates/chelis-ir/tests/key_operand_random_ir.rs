//! Phase 3 of chelis#2413: the key-operand random IR on hand-built graphs.
//!
//! Every expected value here is recomputed from the spec text of [05-RNG-1],
//! [05-OP-37] and [05-OP-8] (the `spec_*` helpers below), never from an
//! evaluator or kernel helper. The graphs exercise the bridge `DrawKey`
//! (inherited and scoped handlers, activation, validation before consumption,
//! unused results), runtime controls, the verifier's key rules, `grad`
//! through the key edge, and the `RandomSelectionParameter` rejection of
//! chelis#2421.

use chelis_ir::dag::{
    Dag, DimInfo, NodeId, RandomDraw, RandomHandler, RiscOp, TensorType, UniformBound,
};
use chelis_ir::eval::{RandomFrame, TensorValue, eval_tensor_roots_with_frame};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_ir::verify::verify;
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::scalar_from_f64;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

// ---- [05-RNG-1], [05-OP-37] and [05-OP-8], transcribed from the spec ----

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn spec_unit(seed: u64, ordinal: u64, index: u64) -> f64 {
    let word =
        splitmix64(seed ^ splitmix64(ordinal).rotate_left(17) ^ splitmix64(index).rotate_left(41));
    (word >> 11) as f64 / (1_u64 << 53) as f64
}

fn stored(prim: Prim, value: f64) -> f64 {
    scalar_from_f64("test", prim, value).unwrap().as_f64_lossy()
}

/// [05-OP-37] at `p`: drop when the arithmetic-width unit is below the rate;
/// otherwise `div(x, sub(1p, rate))`, each primitive finalized at `p` and
/// f16/bf16 computed in f32.
fn spec_dropout(prim: Prim, input: &[f64], rate: f64, seed: u64, ordinal: u64) -> Vec<f64> {
    let rate = stored(prim, rate);
    input
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let unit = spec_unit(seed, ordinal, i as u64);
            let unit = if prim == Prim::F64 {
                unit
            } else {
                f64::from(unit as f32)
            };
            if unit < rate {
                0.0
            } else if prim == Prim::F64 {
                x / (1.0 - rate)
            } else {
                let denominator = stored(prim, f64::from(1.0f32 - rate as f32));
                stored(prim, f64::from(*x as f32 / denominator as f32))
            }
        })
        .collect()
}

/// [05-OP-8] at `p` over f32 bounds.
fn spec_uniform(prim: Prim, len: usize, low: f32, high: f32, seed: u64, ordinal: u64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let unit = spec_unit(seed, ordinal, i as u64);
            if prim == Prim::F64 {
                (f64::from(high) - f64::from(low)).mul_add(unit, f64::from(low))
            } else {
                stored(prim, f64::from((high - low).mul_add(unit as f32, low)))
            }
        })
        .collect()
}

// ---- graph construction ----

fn tensor(prim: Prim, len: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(len)],
        precision: prim,
    }
}

fn scalar(prim: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: prim,
    }
}

fn load(dag: &mut Dag, name: &str, ty: TensorType) -> NodeId {
    dag.add_node(RiscOp::Load { name: name.into() }, vec![], ty, None)
}

fn constant(dag: &mut Dag, prim: Prim, value: f64) -> NodeId {
    dag.add_node(RiscOp::synth_const(prim, value), vec![], scalar(prim), None)
}

fn draw_key(
    dag: &mut Dag,
    handler: RandomHandler,
    seed: Option<NodeId>,
    draw: RandomDraw,
    dtype: Prim,
    controls: &[NodeId],
    active: Option<NodeId>,
) -> NodeId {
    let inputs = seed
        .into_iter()
        .chain(controls.iter().copied())
        .chain(active)
        .collect();
    dag.add_node(
        RiscOp::DrawKey {
            handler,
            draw,
            dtype,
        },
        inputs,
        scalar(Prim::Key),
        None,
    )
}

fn dropout(
    dag: &mut Dag,
    x: NodeId,
    rate: NodeId,
    handler: RandomHandler,
    seed: Option<NodeId>,
    active: Option<NodeId>,
) -> NodeId {
    let ty = dag.get(x).unwrap().output_type.clone();
    let key = draw_key(
        dag,
        handler,
        seed,
        RandomDraw::Dropout,
        ty.precision,
        &[rate],
        active,
    );
    let inputs = [x, rate, key].into_iter().chain(active).collect();
    dag.add_node(RiscOp::Dropout, inputs, ty, None)
}

fn uniform(
    dag: &mut Dag,
    template: NodeId,
    low: NodeId,
    high: NodeId,
    handler: RandomHandler,
    seed: Option<NodeId>,
) -> NodeId {
    let ty = dag.get(template).unwrap().output_type.clone();
    let key = draw_key(
        dag,
        handler,
        seed,
        RandomDraw::UniformLike,
        ty.precision,
        &[low, high],
        None,
    );
    dag.add_node(
        RiscOp::UniformLike,
        vec![template, low, high, key],
        ty,
        None,
    )
}

fn value(prim: Prim, shape: Vec<usize>, data: Vec<f64>) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor("test", prim, RawTensor::Float(data)).unwrap(),
    )
}

fn run(
    dag: &Dag,
    frame: &mut RandomFrame,
    inputs: &UnordMap<&str, TensorValue>,
) -> Result<UnordMap<NodeId, TensorValue>, String> {
    assert!(verify(dag).is_empty(), "{:?}", verify(dag));
    eval_tensor_roots_with_frame(dag, dag.roots(), frame, |name| inputs.get(name).cloned())
}

fn bits(value: &TensorValue) -> Vec<u64> {
    value
        .to_f64_lossy_vec()
        .into_iter()
        .map(f64::to_bits)
        .collect()
}

fn f64_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

const ACTIVE_FLOATS: [Prim; 4] = [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64];

// ---- evaluation ----

#[test]
fn inherited_dropout_takes_a_runtime_rate_and_the_handlers_next_ordinal() {
    for prim in ACTIVE_FLOATS {
        let data = (0..12)
            .map(|i| 0.5 + f64::from(i) / 4.0)
            .collect::<Vec<_>>();
        let mut dag = Dag::new();
        let x = load(&mut dag, "x", tensor(prim, 12));
        let rate = load(&mut dag, "rate", scalar(prim));
        let out = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
        dag.add_root(out);
        let inputs = UnordMap::from([
            ("x", value(prim, vec![12], data.clone())),
            ("rate", value(prim, vec![], vec![0.375])),
        ]);
        let mut frame = RandomFrame::inherited(42, 5);
        let values = run(&dag, &mut frame, &inputs).unwrap();
        let stored_input = value(prim, vec![12], data).to_f64_lossy_vec();
        assert_eq!(
            bits(&values[&out]),
            f64_bits(&spec_dropout(prim, &stored_input, 0.375, 42, 5)),
            "{prim:?}"
        );
        assert_eq!(frame.inherited_counter(), Some(6));
    }
}

#[test]
fn scoped_draws_count_from_zero_and_leave_the_inherited_stream_alone() {
    for prim in ACTIVE_FLOATS {
        let mut dag = Dag::new();
        let template = load(&mut dag, "t", tensor(prim, 9));
        let low = load(&mut dag, "low", scalar(Prim::F32));
        let high = load(&mut dag, "high", scalar(Prim::F32));
        let seed = constant(&mut dag, Prim::Int64, 7.0);
        let scoped = RandomHandler::Scoped { instance: 3 };
        let first = uniform(&mut dag, template, low, high, scoped, Some(seed));
        let inherited = uniform(
            &mut dag,
            template,
            low,
            high,
            RandomHandler::Inherited,
            None,
        );
        let second = uniform(&mut dag, template, low, high, scoped, Some(seed));
        for root in [first, inherited, second] {
            dag.add_root(root);
        }
        let inputs = UnordMap::from([
            ("t", value(prim, vec![9], vec![0.0; 9])),
            ("low", value(Prim::F32, vec![], vec![-1.5])),
            ("high", value(Prim::F32, vec![], vec![2.25])),
        ]);
        let mut frame = RandomFrame::inherited(42, 11);
        let values = run(&dag, &mut frame, &inputs).unwrap();
        for (node, seed, ordinal) in [(first, 7, 0), (inherited, 42, 11), (second, 7, 1)] {
            assert_eq!(
                bits(&values[&node]),
                f64_bits(&spec_uniform(prim, 9, -1.5, 2.25, seed, ordinal)),
                "{prim:?} seed {seed} ordinal {ordinal}"
            );
        }
        assert_eq!(frame.inherited_counter(), Some(12));
    }
}

#[test]
fn an_unused_draw_still_consumes_its_ordinal() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let _unused = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
    let used = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
    dag.add_root(used);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let inputs = UnordMap::from([("x", value(Prim::F32, vec![4], data.clone()))]);
    let pruned = chelis_ir::optimize::dead_code_eliminate(&dag);
    let mut frame = RandomFrame::inherited(-1_i64 as u64, 0);
    let values = run(&pruned, &mut frame, &inputs).unwrap();
    assert_eq!(
        bits(&values[&pruned.roots()[0]]),
        f64_bits(&spec_dropout(Prim::F32, &data, 0.5, -1_i64 as u64, 1)),
        "the used draw takes ordinal 1 behind the dead-result draw"
    );
    assert_eq!(frame.inherited_counter(), Some(2));
}

#[test]
fn validation_precedes_consumption_and_an_inactive_draw_does_neither() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 3));
    let rate = load(&mut dag, "rate", scalar(Prim::F32));
    let active = load(&mut dag, "active", scalar(Prim::Bool));
    let out = dropout(
        &mut dag,
        x,
        rate,
        RandomHandler::Inherited,
        None,
        Some(active),
    );
    dag.add_root(out);
    let inputs = |rate: f64, active: bool| {
        UnordMap::from([
            ("x", value(Prim::F32, vec![3], vec![1.0, 2.0, 3.0])),
            ("rate", value(Prim::F32, vec![], vec![rate])),
            (
                "active",
                TensorValue::from_storage(
                    vec![],
                    finalize_tensor("test", Prim::Bool, RawTensor::Int(vec![i64::from(active)]))
                        .unwrap(),
                ),
            ),
        ])
    };
    let mut frame = RandomFrame::inherited(42, 9);
    let error = run(&dag, &mut frame, &inputs(1.5, true)).unwrap_err();
    assert!(error.contains("domain in dropout"), "{error}");
    assert_eq!(
        frame.inherited_counter(),
        Some(9),
        "a failed rate consumes nothing"
    );

    let values = run(&dag, &mut frame, &inputs(1.5, false)).unwrap();
    assert_eq!(values[&out].to_f64_lossy_vec(), vec![0.0; 3]);
    assert_eq!(
        frame.inherited_counter(),
        Some(9),
        "an inactive draw consumes nothing"
    );

    let values = run(&dag, &mut frame, &inputs(0.25, true)).unwrap();
    assert_eq!(
        bits(&values[&out]),
        f64_bits(&spec_dropout(Prim::F32, &[1.0, 2.0, 3.0], 0.25, 42, 9))
    );
    assert_eq!(frame.inherited_counter(), Some(10));
}

#[test]
fn uniform_bounds_validate_before_consumption() {
    let mut dag = Dag::new();
    let template = load(&mut dag, "t", tensor(Prim::F64, 2));
    let low = load(&mut dag, "low", scalar(Prim::F32));
    let high = load(&mut dag, "high", scalar(Prim::F32));
    let out = uniform(
        &mut dag,
        template,
        low,
        high,
        RandomHandler::Inherited,
        None,
    );
    dag.add_root(out);
    for (lo, hi) in [(1.0, 0.0), (f64::NAN, 1.0), (0.0, f64::INFINITY)] {
        let inputs = UnordMap::from([
            ("t", value(Prim::F64, vec![2], vec![0.0; 2])),
            ("low", value(Prim::F32, vec![], vec![lo])),
            ("high", value(Prim::F32, vec![], vec![hi])),
        ]);
        let mut frame = RandomFrame::inherited(1, 4);
        let error = run(&dag, &mut frame, &inputs).unwrap_err();
        assert!(error.contains("domain in uniform_like"), "{error}");
        assert_eq!(frame.inherited_counter(), Some(4));
    }
}

#[test]
fn an_inherited_draw_without_a_handler_is_refused() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 2));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let out = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
    dag.add_root(out);
    let inputs = UnordMap::from([("x", value(Prim::F32, vec![2], vec![1.0, 2.0]))]);
    let error = run(&dag, &mut RandomFrame::unhandled(), &inputs).unwrap_err();
    assert!(error.contains("no active handler"), "{error}");
}

// ---- grad ----

fn loss_of(dag: &mut Dag, value: NodeId, prim: Prim) -> NodeId {
    let accumulator = if prim == Prim::F64 {
        Prim::F64
    } else {
        Prim::F32
    };
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator,
        },
        vec![value],
        scalar(accumulator),
        None,
    )
}

#[test]
fn dropout_input_adjoint_replays_the_forward_mask_through_the_key() {
    for prim in [Prim::F32, Prim::F64] {
        let mut dag = Dag::new();
        let x = load(&mut dag, "x", tensor(prim, 16));
        let rate = load(&mut dag, "rate", scalar(prim));
        let out = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
        let loss = loss_of(&mut dag, out, prim);
        dag.add_root(loss);
        let grad = grad_dag_checked(&dag, loss, &[x]).expect("the rate is not a parameter");
        let dx = grad.grad_nodes[&x];
        let data = (0..16).map(|i| 1.0 + f64::from(i)).collect::<Vec<_>>();
        let inputs = UnordMap::from([
            ("x", value(prim, vec![16], data)),
            ("rate", value(prim, vec![], vec![0.25])),
        ]);
        let mut frame = RandomFrame::inherited(42, 3);
        let values = eval_tensor_roots_with_frame(&grad.dag, &[dx], &mut frame, |name| {
            inputs.get(name).cloned()
        })
        .unwrap();
        let expected = spec_dropout(prim, &[1.0; 16], 0.25, 42, 3);
        assert_eq!(bits(&values[&dx]), f64_bits(&expected), "{prim:?}");
        assert_eq!(
            frame.inherited_counter(),
            Some(4),
            "replay reads the key without a second draw"
        );
    }
}

#[test]
fn uniform_bound_adjoints_match_the_05_op_8_transcription() {
    for prim in [Prim::F32, Prim::F64] {
        let len = 11;
        let mut dag = Dag::new();
        let template = load(&mut dag, "t", tensor(prim, len));
        let weights = load(&mut dag, "w", tensor(prim, len));
        let low = load(&mut dag, "low", scalar(Prim::F32));
        let high = load(&mut dag, "high", scalar(Prim::F32));
        let sample = uniform(
            &mut dag,
            template,
            low,
            high,
            RandomHandler::Inherited,
            None,
        );
        let weighted = dag.add_node(RiscOp::Mul, vec![sample, weights], tensor(prim, len), None);
        let loss = loss_of(&mut dag, weighted, prim);
        dag.add_root(loss);
        let grad = grad_dag_checked(&dag, loss, &[low, high]).expect("bounds differentiate");
        let weights_data = (0..len)
            .map(|i| (i as f64 - 4.0) * 0.375)
            .collect::<Vec<_>>();
        let inputs = UnordMap::from([
            ("t", value(prim, vec![len], vec![0.0; len])),
            ("w", value(prim, vec![len], weights_data.clone())),
            ("low", value(Prim::F32, vec![], vec![-0.5])),
            ("high", value(Prim::F32, vec![], vec![1.75])),
        ]);
        let roots = [grad.grad_nodes[&low], grad.grad_nodes[&high]];
        let mut frame = RandomFrame::inherited(9, 2);
        let values = eval_tensor_roots_with_frame(&grad.dag, &roots, &mut frame, |name| {
            inputs.get(name).cloned()
        })
        .unwrap();
        // [05-OP-8]: g_i * (1-u_i) to low and g_i * u_i to high, at the
        // arithmetic width, combined by the adjacent-pair tree; the f32
        // bound then takes the checked cast of the template-dtype sum.
        let tree = |mut level: Vec<f64>| {
            while level.len() > 1 {
                level = level
                    .chunks(2)
                    .map(|pair| {
                        if pair.len() == 2 {
                            if prim == Prim::F64 {
                                pair[0] + pair[1]
                            } else {
                                f64::from(pair[0] as f32 + pair[1] as f32)
                            }
                        } else {
                            pair[0]
                        }
                    })
                    .collect();
            }
            level[0]
        };
        for (root, high_bound) in [(roots[0], false), (roots[1], true)] {
            let leaves = (0..len)
                .map(|i| {
                    let unit = spec_unit(9, 2, i as u64);
                    let g = weights_data[i];
                    if prim == Prim::F64 {
                        g * if high_bound { unit } else { 1.0 - unit }
                    } else {
                        let unit = unit as f32;
                        f64::from(g as f32 * if high_bound { unit } else { 1.0 - unit })
                    }
                })
                .collect::<Vec<_>>();
            let expected = stored(Prim::F32, tree(leaves));
            assert_eq!(
                values[&root].to_f64_lossy_vec()[0].to_bits(),
                expected.to_bits(),
                "{prim:?} high={high_bound}"
            );
        }
        assert_eq!(frame.inherited_counter(), Some(3));
    }
}

fn rejection(result: Result<chelis_ir::grad::GradResult, AdError>) -> Option<AdRejectionReason> {
    match result {
        Ok(_) => None,
        Err(AdError::NotSupported { reason, .. }) => Some(reason),
    }
}

#[test]
fn a_parameter_reaching_the_rate_through_adjoint_slots_is_rejected() {
    // rate = p * c, and p is differentiated.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let p = load(&mut dag, "p", scalar(Prim::F32));
    let c = constant(&mut dag, Prim::F32, 0.5);
    let rate = dag.add_node(RiscOp::Mul, vec![p, c], scalar(Prim::F32), None);
    let out = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
    let loss = loss_of(&mut dag, out, Prim::F32);
    dag.add_root(loss);
    assert_eq!(
        rejection(grad_dag_checked(&dag, loss, &[p])),
        Some(AdRejectionReason::RandomSelectionParameter)
    );
    assert_eq!(
        rejection(grad_dag_checked(&dag, loss, &[x, p])),
        Some(AdRejectionReason::RandomSelectionParameter)
    );
    // The same rate is a constant when only x is differentiated.
    assert_eq!(rejection(grad_dag_checked(&dag, loss, &[x])), None);
}

#[test]
fn a_rate_reached_only_through_a_zero_cotangent_slot_differentiates() {
    // rate = sum(uniform_like(p, 0.1, 0.2)) / 4: p reaches the rate only
    // through the uniform template, whose cotangent is zero.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let p = load(&mut dag, "p", tensor(Prim::F32, 4));
    let low = constant(&mut dag, Prim::F32, 0.0);
    let high = constant(&mut dag, Prim::F32, 0.25);
    let noise = uniform(&mut dag, p, low, high, RandomHandler::Inherited, None);
    let total = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![noise],
        scalar(Prim::F32),
        None,
    );
    let four = constant(&mut dag, Prim::F32, 4.0);
    let rate = dag.add_node(RiscOp::Div, vec![total, four], scalar(Prim::F32), None);
    let out = dropout(&mut dag, x, rate, RandomHandler::Inherited, None, None);
    let scaled = dag.add_node(RiscOp::Mul, vec![out, p], tensor(Prim::F32, 4), None);
    let loss = loss_of(&mut dag, scaled, Prim::F32);
    dag.add_root(loss);
    assert_eq!(rejection(grad_dag_checked(&dag, loss, &[p])), None);
    // Through a bound, which carries an adjoint, the same shape rejects.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let t = load(&mut dag, "t", tensor(Prim::F32, 4));
    let bound = load(&mut dag, "b", scalar(Prim::F32));
    let low = constant(&mut dag, Prim::F32, 0.0);
    let noise = uniform(&mut dag, t, low, bound, RandomHandler::Inherited, None);
    let total = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![noise],
        scalar(Prim::F32),
        None,
    );
    let out = dropout(&mut dag, x, total, RandomHandler::Inherited, None, None);
    let loss = loss_of(&mut dag, out, Prim::F32);
    dag.add_root(loss);
    assert_eq!(
        rejection(grad_dag_checked(&dag, loss, &[bound])),
        Some(AdRejectionReason::RandomSelectionParameter)
    );
}

// ---- verifier key rules ----

fn dropout_graph() -> (Dag, NodeId, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let key = draw_key(
        &mut dag,
        RandomHandler::Inherited,
        None,
        RandomDraw::Dropout,
        Prim::F32,
        &[rate],
        None,
    );
    let out = dag.add_node(
        RiscOp::Dropout,
        vec![x, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    (dag, x, rate, key, out)
}

fn assert_rejected(dag: &Dag, needle: &str) {
    let errors = verify(dag);
    assert!(
        errors.iter().any(|error| error.contains(needle)),
        "expected `{needle}` in {errors:?}"
    );
}

#[test]
fn the_verifier_accepts_a_consumed_key_and_its_replays() {
    let (mut dag, _, rate, key, out) = dropout_graph();
    let g = load(&mut dag, "g", tensor(Prim::F32, 4));
    let replay = dag.add_node(
        RiscOp::DropoutReplay,
        vec![g, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    let again = dag.add_node(
        RiscOp::DropoutReplay,
        vec![g, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    for root in [out, replay, again] {
        dag.add_root(root);
    }
    assert_eq!(verify(&dag), Vec::<String>::new());
}

#[test]
fn the_verifier_rejects_a_double_consume() {
    let (mut dag, x, rate, key, out) = dropout_graph();
    let twice = dag.add_node(
        RiscOp::Dropout,
        vec![x, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(out);
    dag.add_root(twice);
    assert_rejected(&dag, "is consumed twice");
}

#[test]
fn the_verifier_rejects_a_key_fed_to_another_operation() {
    let (mut dag, _, _, key, out) = dropout_graph();
    let added = dag.add_node(RiscOp::Add, vec![key, key], scalar(Prim::Key), None);
    dag.add_root(out);
    dag.add_root(added);
    assert_rejected(
        &dag,
        "only a key operation or a random primitive consumes a key",
    );
    // chelis#2413 step 1, rule V2: a root is a use of its key, and a draw
    // key's key is its draw's alone, so a consumed draw key is no root.
    let (mut dag, _, _, key, out) = dropout_graph();
    dag.add_root(out);
    dag.add_root(key);
    assert_rejected(&dag, "is a graph root and is also consumed");
    assert_rejected(&dag, "a draw key's key feeds only its draw");
}

#[test]
fn the_verifier_rejects_a_replay_that_changes_the_mask_contract() {
    let (mut dag, _, _, key, out) = dropout_graph();
    let g = load(&mut dag, "g", tensor(Prim::F32, 4));
    let other_rate = constant(&mut dag, Prim::F32, 0.25);
    let replay = dag.add_node(
        RiscOp::DropoutReplay,
        vec![g, other_rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(out);
    dag.add_root(replay);
    assert_rejected(&dag, "changes its forward node");
}

#[test]
fn the_verifier_rejects_a_replay_of_an_unconsumed_key() {
    let mut dag = Dag::new();
    let g = load(&mut dag, "g", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let key = draw_key(
        &mut dag,
        RandomHandler::Inherited,
        None,
        RandomDraw::Dropout,
        Prim::F32,
        &[rate],
        None,
    );
    let replay = dag.add_node(
        RiscOp::DropoutReplay,
        vec![g, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(replay);
    assert_rejected(&dag, "that no forward random primitive consumes");
}

#[test]
fn the_verifier_rejects_a_constant_key_and_mismatched_controls() {
    // Rule V1: a key comes from a key operation, a draw key, or a Load; a
    // key constant would be literal bits, which no carrier admits.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let forged = dag.add_node(
        RiscOp::Const {
            value: chelis_types::ScalarValue::from_key(chelis_types::RandomKey::from_counter(7, 0)),
        },
        vec![],
        scalar(Prim::Key),
        None,
    );
    let out = dag.add_node(
        RiscOp::Dropout,
        vec![x, rate, forged],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(out);
    assert_rejected(
        &dag,
        "only a key operation, a draw key or a Load produces one",
    );

    // The key validates a different rate than its consumer uses.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let other = constant(&mut dag, Prim::F32, 0.75);
    let key = draw_key(
        &mut dag,
        RandomHandler::Inherited,
        None,
        RandomDraw::Dropout,
        Prim::F32,
        &[other],
        None,
    );
    let out = dag.add_node(
        RiscOp::Dropout,
        vec![x, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(out);
    assert_rejected(&dag, "does not validate the controls");

    // A scoped key's seed must be a literal i64.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, Prim::F32, 0.5);
    let runtime_seed = load(&mut dag, "seed", scalar(Prim::Int64));
    let out = dropout(
        &mut dag,
        x,
        rate,
        RandomHandler::Scoped { instance: 0 },
        Some(runtime_seed),
        None,
    );
    dag.add_root(out);
    assert_rejected(&dag, "literal seed");
}

#[test]
fn bound_adjoint_nodes_verify_against_their_forward_template() {
    let mut dag = Dag::new();
    let template = load(&mut dag, "t", tensor(Prim::F32, 4));
    let low = constant(&mut dag, Prim::F32, 0.0);
    let high = constant(&mut dag, Prim::F32, 1.0);
    let out = uniform(
        &mut dag,
        template,
        low,
        high,
        RandomHandler::Inherited,
        None,
    );
    let key = dag.get(out).unwrap().inputs[3];
    let g = load(&mut dag, "g", tensor(Prim::F32, 4));
    let adjoint = dag.add_node(
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::High,
        },
        vec![template, g, key],
        scalar(Prim::F32),
        None,
    );
    dag.add_root(out);
    dag.add_root(adjoint);
    assert_eq!(verify(&dag), Vec::<String>::new());
    let wide = load(&mut dag, "wide", tensor(Prim::F32, 5));
    let g5 = load(&mut dag, "g5", tensor(Prim::F32, 5));
    let mismatched = dag.add_node(
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::Low,
        },
        vec![wide, g5, key],
        scalar(Prim::F32),
        None,
    );
    dag.add_root(mismatched);
    assert_rejected(&dag, "changes its forward node");
}
