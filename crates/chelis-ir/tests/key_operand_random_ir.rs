//! Phase 3 of chelis#2413: the key-operand random IR on hand-built graphs.
//!
//! Every expected value here is recomputed from the spec text of [05-RNG-1],
//! [05-OP-37] and [05-OP-8] (the `spec_*` helpers below), never from an
//! evaluator or kernel helper. Each draw's key is `key_from_seed` of a literal
//! seed. The graphs exercise runtime controls, the verifier's key rules,
//! `grad` through the key edge, and the `RandomSelectionParameter` rejection
//! of chelis#2421.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType, UniformBound};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_exact};
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

/// [05-RNG-1]'s unit value of `word(key, index)` for the key whose bits are
/// `key` (`key_from_seed` keeps a seed's two's-complement bits).
fn spec_unit(key: u64, index: u64) -> f64 {
    let word = splitmix64(key ^ splitmix64(index).rotate_left(41));
    (word >> 11) as f64 / (1_u64 << 53) as f64
}

fn stored(prim: Prim, value: f64) -> f64 {
    scalar_from_f64("test", prim, value).unwrap().as_f64_lossy()
}

/// [05-OP-37] at `p`: drop when the arithmetic-width unit is below the rate;
/// otherwise `div(x, sub(1p, rate))`, each primitive finalized at `p` and
/// f16/bf16 computed in f32.
fn spec_dropout(prim: Prim, input: &[f64], rate: f64, key: u64) -> Vec<f64> {
    let rate = stored(prim, rate);
    input
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let unit = spec_unit(key, i as u64);
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

fn load(dag: &mut Dag, decl: chelis_ir::dag::DeclId, name: &str, ty: TensorType) -> NodeId {
    dag.add_node(decl, RiscOp::Load { name: name.into() }, vec![], ty, None)
}

fn constant(dag: &mut Dag, decl: chelis_ir::dag::DeclId, prim: Prim, value: f64) -> NodeId {
    dag.add_node(
        decl,
        RiscOp::synth_const(prim, value),
        vec![],
        scalar(prim),
        None,
    )
}

/// `key_from_seed(seed)`: a rank-0 key whose bits are the seed's.
fn seeded_key(dag: &mut Dag, decl: chelis_ir::dag::DeclId, seed: i64) -> NodeId {
    let seed = dag.add_node(
        decl,
        RiscOp::Const {
            value: chelis_types::scalar_from_i64("test", Prim::Int64, seed).unwrap(),
        },
        vec![],
        scalar(Prim::Int64),
        None,
    );
    dag.add_node(
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        scalar(Prim::Key),
        None,
    )
}

fn dropout(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    x: NodeId,
    rate: NodeId,
    seed: i64,
    active: Option<NodeId>,
) -> NodeId {
    let ty = dag.get(x).unwrap().output_type.clone();
    let key = seeded_key(dag, decl, seed);
    let inputs = [x, rate, key].into_iter().chain(active).collect();
    dag.add_node(decl, RiscOp::Dropout, inputs, ty, None)
}

fn uniform(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    template: NodeId,
    low: NodeId,
    high: NodeId,
    seed: i64,
) -> NodeId {
    let ty = dag.get(template).unwrap().output_type.clone();
    let key = seeded_key(dag, decl, seed);
    dag.add_node(
        decl,
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

// ---- evaluation ----

// ---- grad ----

fn loss_of(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: NodeId, prim: Prim) -> NodeId {
    let accumulator = if prim == Prim::F64 {
        Prim::F64
    } else {
        Prim::F32
    };
    dag.add_node(
        decl,
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
        let decl = dag.declare("test");
        let x = load(&mut dag, decl, "x", tensor(prim, 16));
        let rate = load(&mut dag, decl, "rate", scalar(prim));
        let out = dropout(&mut dag, decl, x, rate, 42, None);
        let loss = loss_of(&mut dag, decl, out, prim);
        dag.add_root(loss);
        let grad = grad_dag_checked(&dag, loss, &[x]).expect("the rate is not a parameter");
        let dx = grad.grad_nodes[&x];
        let data = (0..16).map(|i| 1.0 + f64::from(i)).collect::<Vec<_>>();
        let inputs = UnordMap::from([
            ("x", value(prim, vec![16], data)),
            ("rate", value(prim, vec![], vec![0.25])),
        ]);
        let values =
            eval_tensor_roots_exact(&grad.dag, &[dx], |name| inputs.get(name).cloned()).unwrap();
        let expected = spec_dropout(prim, &[1.0; 16], 0.25, 42);
        assert_eq!(bits(&values[&dx]), f64_bits(&expected), "{prim:?}");
    }
}

#[test]
fn uniform_bound_adjoints_match_the_05_op_8_transcription() {
    for prim in [Prim::F32, Prim::F64] {
        let len = 11;
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let template = load(&mut dag, decl, "t", tensor(prim, len));
        let weights = load(&mut dag, decl, "w", tensor(prim, len));
        let low = load(&mut dag, decl, "low", scalar(prim));
        let high = load(&mut dag, decl, "high", scalar(prim));
        let sample = uniform(&mut dag, decl, template, low, high, 9);
        let weighted = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![sample, weights],
            tensor(prim, len),
            None,
        );
        let loss = loss_of(&mut dag, decl, weighted, prim);
        dag.add_root(loss);
        let grad = grad_dag_checked(&dag, loss, &[low, high]).expect("bounds differentiate");
        let weights_data = (0..len)
            .map(|i| (i as f64 - 4.0) * 0.375)
            .collect::<Vec<_>>();
        let inputs = UnordMap::from([
            ("t", value(prim, vec![len], vec![0.0; len])),
            ("w", value(prim, vec![len], weights_data.clone())),
            ("low", value(prim, vec![], vec![-0.5])),
            ("high", value(prim, vec![], vec![1.75])),
        ]);
        let roots = [grad.grad_nodes[&low], grad.grad_nodes[&high]];
        let values =
            eval_tensor_roots_exact(&grad.dag, &roots, |name| inputs.get(name).cloned()).unwrap();
        // [05-OP-8]: g_i * (1-u_i) to low and g_i * u_i to high, at the
        // arithmetic width, combined by the adjacent-pair tree and stored at
        // the template's dtype, which is the bounds' dtype.
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
                    let unit = spec_unit(9, i as u64);
                    let g = weights_data[i];
                    if prim == Prim::F64 {
                        g * if high_bound { unit } else { 1.0 - unit }
                    } else {
                        let unit = unit as f32;
                        f64::from(g as f32 * if high_bound { unit } else { 1.0 - unit })
                    }
                })
                .collect::<Vec<_>>();
            let expected = stored(prim, tree(leaves));
            assert_eq!(
                values[&root].to_f64_lossy_vec()[0].to_bits(),
                expected.to_bits(),
                "{prim:?} high={high_bound}"
            );
        }
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
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", tensor(Prim::F32, 4));
    let p = load(&mut dag, decl, "p", scalar(Prim::F32));
    let c = constant(&mut dag, decl, Prim::F32, 0.5);
    let rate = dag.add_node(decl, RiscOp::Mul, vec![p, c], scalar(Prim::F32), None);
    let out = dropout(&mut dag, decl, x, rate, 7, None);
    let loss = loss_of(&mut dag, decl, out, Prim::F32);
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
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", tensor(Prim::F32, 4));
    let p = load(&mut dag, decl, "p", tensor(Prim::F32, 4));
    let low = constant(&mut dag, decl, Prim::F32, 0.0);
    let high = constant(&mut dag, decl, Prim::F32, 0.25);
    let noise = uniform(&mut dag, decl, p, low, high, 7);
    let total = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![noise],
        scalar(Prim::F32),
        None,
    );
    let four = constant(&mut dag, decl, Prim::F32, 4.0);
    let rate = dag.add_node(
        decl,
        RiscOp::Div,
        vec![total, four],
        scalar(Prim::F32),
        None,
    );
    let out = dropout(&mut dag, decl, x, rate, 7, None);
    let scaled = dag.add_node(decl, RiscOp::Mul, vec![out, p], tensor(Prim::F32, 4), None);
    let loss = loss_of(&mut dag, decl, scaled, Prim::F32);
    dag.add_root(loss);
    assert_eq!(rejection(grad_dag_checked(&dag, loss, &[p])), None);
    // Through a bound, which carries an adjoint, the same shape rejects.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", tensor(Prim::F32, 4));
    let t = load(&mut dag, decl, "t", tensor(Prim::F32, 4));
    let bound = load(&mut dag, decl, "b", scalar(Prim::F32));
    let low = constant(&mut dag, decl, Prim::F32, 0.0);
    let noise = uniform(&mut dag, decl, t, low, bound, 7);
    let total = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![noise],
        scalar(Prim::F32),
        None,
    );
    let out = dropout(&mut dag, decl, x, total, 7, None);
    let loss = loss_of(&mut dag, decl, out, Prim::F32);
    dag.add_root(loss);
    assert_eq!(
        rejection(grad_dag_checked(&dag, loss, &[bound])),
        Some(AdRejectionReason::RandomSelectionParameter)
    );
}

// ---- verifier key rules ----

fn dropout_graph() -> (Dag, NodeId, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, decl, Prim::F32, 0.5);
    let key = seeded_key(&mut dag, decl, 7);
    let out = dag.add_node(
        decl,
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
    let decl = dag.nodes()[0].owner.decl;
    let g = load(&mut dag, decl, "g", tensor(Prim::F32, 4));
    let replay = dag.add_node(
        decl,
        RiscOp::DropoutReplay,
        vec![g, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    let again = dag.add_node(
        decl,
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
    let decl = dag.nodes()[0].owner.decl;
    let twice = dag.add_node(
        decl,
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
    let decl = dag.nodes()[0].owner.decl;
    let added = dag.add_node(decl, RiscOp::Add, vec![key, key], scalar(Prim::Key), None);
    dag.add_root(out);
    dag.add_root(added);
    assert_rejected(
        &dag,
        "only a key operation, a join or a random primitive consumes a key",
    );
    // chelis#2413 step 1, rule V2: a root is a use of its key, so a
    // consumed key is no root.
    let (mut dag, _, _, key, out) = dropout_graph();
    dag.add_root(out);
    dag.add_root(key);
    assert_rejected(&dag, "is a graph root and is also consumed");
}

#[test]
fn the_verifier_rejects_a_replay_that_changes_the_mask_contract() {
    let (mut dag, _, _, key, out) = dropout_graph();
    let decl = dag.nodes()[0].owner.decl;
    let g = load(&mut dag, decl, "g", tensor(Prim::F32, 4));
    let other_rate = constant(&mut dag, decl, Prim::F32, 0.25);
    let replay = dag.add_node(
        decl,
        RiscOp::DropoutReplay,
        vec![g, other_rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(out);
    dag.add_root(replay);
    assert_rejected(&dag, "changes the mask contract of");
}

#[test]
fn the_verifier_rejects_a_replay_of_an_unconsumed_key() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let g = load(&mut dag, decl, "g", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, decl, Prim::F32, 0.5);
    let key = seeded_key(&mut dag, decl, 7);
    let replay = dag.add_node(
        decl,
        RiscOp::DropoutReplay,
        vec![g, rate, key],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(replay);
    assert_rejected(&dag, "which no forward random primitive consumes");
}

#[test]
fn the_verifier_rejects_a_constant_key() {
    // Rule V1: a key comes from a key operation or a Load; a key constant
    // would be literal bits, which no carrier admits.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", tensor(Prim::F32, 4));
    let rate = constant(&mut dag, decl, Prim::F32, 0.5);
    let forged = dag.add_node(
        decl,
        RiscOp::Const {
            value: chelis_types::ScalarValue::from_key(
                chelis_types::RandomKey::from_seed(
                    chelis_types::scalar_from_i64("test", Prim::Int64, 7).unwrap(),
                )
                .unwrap(),
            ),
        },
        vec![],
        scalar(Prim::Key),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Dropout,
        vec![x, rate, forged],
        tensor(Prim::F32, 4),
        None,
    );
    dag.add_root(out);
    assert_rejected(&dag, "only a key operation, a join or a Load produces one");
}

#[test]
fn bound_adjoint_nodes_verify_against_their_forward_template() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let template = load(&mut dag, decl, "t", tensor(Prim::F32, 4));
    let low = constant(&mut dag, decl, Prim::F32, 0.0);
    let high = constant(&mut dag, decl, Prim::F32, 1.0);
    let out = uniform(&mut dag, decl, template, low, high, 7);
    let key = dag.get(out).unwrap().inputs[3];
    let g = load(&mut dag, decl, "g", tensor(Prim::F32, 4));
    let adjoint = dag.add_node(
        decl,
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
    let wide = load(&mut dag, decl, "wide", tensor(Prim::F32, 5));
    let g5 = load(&mut dag, decl, "g5", tensor(Prim::F32, 5));
    let mismatched = dag.add_node(
        decl,
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::Low,
        },
        vec![wide, g5, key],
        scalar(Prim::F32),
        None,
    );
    dag.add_root(mismatched);
    assert_rejected(&dag, "changes the mask contract of");
}
