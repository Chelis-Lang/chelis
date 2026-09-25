//! Phase 3 of chelis#2413: native C execution of the key-operand random IR
//! on hand-built graphs, against a transcription of [05-RNG-1], [05-OP-37]
//! and [05-OP-8] from the spec text, and against the DAG evaluator.
mod ownership_support;

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType, UniformBound};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_exact};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::scalar_from_f64;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// [05-RNG-1]'s unit value of `word(key, index)`.
fn spec_unit(key: u64, index: u64) -> f64 {
    let word = splitmix64(key ^ splitmix64(index).rotate_left(41));
    (word >> 11) as f64 / (1_u64 << 53) as f64
}

fn stored(prim: Prim, value: f64) -> f64 {
    scalar_from_f64("test", prim, value).unwrap().as_f64_lossy()
}

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

fn spec_uniform(prim: Prim, len: usize, low: f32, high: f32, key: u64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let unit = spec_unit(key, i as u64);
            if prim == Prim::F64 {
                (f64::from(high) - f64::from(low)).mul_add(unit, f64::from(low))
            } else {
                stored(prim, f64::from((high - low).mul_add(unit as f32, low)))
            }
        })
        .collect()
}

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

/// `key_from_seed` of the rank-0 i64 `seed` node.
fn key(dag: &mut Dag, seed: NodeId) -> NodeId {
    dag.add_node(RiscOp::KeyFromSeed, vec![seed], scalar(Prim::Key), None)
}

fn dtype_tag(prim: Prim) -> (&'static str, usize) {
    match prim {
        Prim::F64 => ("CHELIS_DTYPE_F64", 8),
        Prim::F32 => ("CHELIS_DTYPE_F32", 4),
        Prim::F16 => ("CHELIS_DTYPE_F16", 2),
        Prim::Bf16 => ("CHELIS_DTYPE_BF16", 2),
        other => panic!("{other:?}"),
    }
}

fn storage_bits(prim: Prim, value: f64) -> u64 {
    match prim {
        Prim::F64 => value.to_bits(),
        Prim::F32 => u64::from((value as f32).to_bits()),
        Prim::F16 => u64::from(half::f16::from_f64(value).to_bits()),
        Prim::Bf16 => u64::from(half::bf16::from_f64(value).to_bits()),
        other => panic!("{other:?}"),
    }
}

/// Compile `dag` as a public C entry, feed `inputs` (name, dtype, shape,
/// values) in the artifact's label order, and return each root's stored
/// element bits.
fn run_c(dag: Dag, inputs: &[(&str, Prim, Vec<usize>, Vec<f64>)]) -> Vec<Vec<u64>> {
    let roots = dag
        .roots()
        .iter()
        .map(|root| dag.get(*root).unwrap().output_type.clone())
        .collect::<Vec<_>>();
    let verified = chelis_ir::ownership::lower_dag_ownership(dag)
        .and_then(chelis_ir::ownership::verify_ownership)
        .expect("hand-built random graph passes ownership");
    let artifact = chelis_backend_c::codegen(verified, "sample").expect("C emission");
    let generated = ownership_support::GeneratedProgram::from_codegen(&artifact);
    let mut setup = String::new();
    for (slot, label) in artifact.input_labels.iter().enumerate() {
        let (_, prim, shape, values) = inputs
            .iter()
            .find(|(name, ..)| name == label)
            .unwrap_or_else(|| panic!("no input `{label}`"));
        let (tag, width) = dtype_tag(*prim);
        let dims = if shape.is_empty() {
            "NULL".to_string()
        } else {
            format!(
                "(int64_t[]){{{}}}",
                shape
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let bits = values
            .iter()
            .map(|value| format!("UINT64_C(0x{:x})", storage_bits(*prim, *value)))
            .collect::<Vec<_>>()
            .join(",");
        setup.push_str(&format!(
            "    inputs[{slot}] = chelis_alloc({rank}, {dims}, {tag});\n    {{\n        const uint64_t bits[] = {{{bits}0}};\n        chelis_tensor_write *write = chelis_tensor_begin_write(inputs[{slot}]);\n        unsigned char *data = (unsigned char*)chelis_tensor_write_view(write).data;\n        for (int i = 0; i < {count}; ++i) memcpy(data + i * {width}, &bits[i], {width});\n        chelis_tensor_end_write(write);\n    }}\n",
            rank = shape.len(),
            count = values.len(),
            bits = if bits.is_empty() { String::new() } else { format!("{bits},") },
        ));
    }
    let mut readout = String::new();
    for (slot, ty) in roots.iter().enumerate() {
        let (_, width) = dtype_tag(ty.precision);
        readout.push_str(&format!(
            "    {{\n        chelis_read_view view = chelis_tensor_read_view(outputs[{slot}]);\n        for (int64_t i = 0; i < view.count; ++i) {{ uint64_t word = 0; memcpy(&word, (const unsigned char*)view.data + i * {width}, {width}); printf(\"%llx \", (unsigned long long)word); }}\n        printf(\"\\n\");\n        chelis_tensor_release(outputs[{slot}]);\n    }}\n"
        ));
    }
    let n_in = artifact.input_labels.len();
    let n_out = roots.len();
    let driver = format!(
        "int main(void) {{\n    chelis_tensor *inputs[{alloc_in}];\n{setup}    chelis_tensor *outputs[{alloc_out}];\n    sample(inputs, {n_in}, outputs, {n_out});\n{readout}    for (int i = 0; i < {n_in}; ++i) chelis_tensor_release(inputs[i]);\n    return 0;\n}}\n",
        alloc_in = n_in.max(1),
        alloc_out = n_out.max(1),
    );
    let (summary, stdout) = ownership_support::run_with_stdout(&generated, &driver);
    ownership_support::balanced(&summary);
    stdout
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|word| u64::from_str_radix(word, 16).unwrap())
                .collect()
        })
        .collect()
}

fn run_eval(dag: &Dag, inputs: &[(&str, Prim, Vec<usize>, Vec<f64>)]) -> Vec<Vec<u64>> {
    let values = inputs
        .iter()
        .map(|(name, prim, shape, data)| {
            (
                *name,
                TensorValue::from_storage(
                    shape.clone(),
                    finalize_tensor("test", *prim, RawTensor::Float(data.clone())).unwrap(),
                ),
            )
        })
        .collect::<UnordMap<_, _>>();
    let out = eval_tensor_roots_exact(dag, dag.roots(), |name| values.get(name).cloned()).unwrap();
    dag.roots()
        .iter()
        .map(|root| {
            let value = &out[root];
            value
                .to_f64_lossy_vec()
                .into_iter()
                .map(|element| storage_bits(value.prim(), element))
                .collect()
        })
        .collect()
}

#[test]
fn seeded_runtime_controls_match_the_spec_in_c_and_eval() {
    for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for (seed, len) in [(7_i64, 13_usize), (-1, 1), (42, 0)] {
            let mut dag = Dag::new();
            let x = load(&mut dag, "x", tensor(prim, len));
            let rate = load(&mut dag, "rate", scalar(prim));
            let low = load(&mut dag, "low", scalar(Prim::F32));
            let high = load(&mut dag, "high", scalar(Prim::F32));
            let seed_node = dag.add_node(
                RiscOp::synth_const(Prim::Int64, seed as f64),
                vec![],
                scalar(Prim::Int64),
                None,
            );
            let dropout_key = key(&mut dag, seed_node);
            let dropped = dag.add_node(
                RiscOp::Dropout,
                vec![x, rate, dropout_key],
                tensor(prim, len),
                None,
            );
            let uniform_key = key(&mut dag, seed_node);
            let sampled = dag.add_node(
                RiscOp::UniformLike,
                vec![x, low, high, uniform_key],
                tensor(prim, len),
                None,
            );
            dag.add_root(dropped);
            dag.add_root(sampled);
            let data = (0..len).map(|i| 0.75 + i as f64 / 8.0).collect::<Vec<_>>();
            let stored_data = data
                .iter()
                .map(|value| stored(prim, *value))
                .collect::<Vec<_>>();
            let inputs = [
                ("x", prim, vec![len], data),
                ("rate", prim, vec![], vec![0.3125]),
                ("low", Prim::F32, vec![], vec![-2.5]),
                ("high", Prim::F32, vec![], vec![0.75]),
            ];
            let seed_bits = seed as u64;
            let expected = vec![
                spec_dropout(prim, &stored_data, 0.3125, seed_bits)
                    .into_iter()
                    .map(|value| storage_bits(prim, value))
                    .collect::<Vec<_>>(),
                spec_uniform(prim, len, -2.5, 0.75, seed_bits)
                    .into_iter()
                    .map(|value| storage_bits(prim, value))
                    .collect::<Vec<_>>(),
            ];
            assert_eq!(
                run_eval(&dag, &inputs),
                expected,
                "eval {prim:?} seed {seed} len {len}"
            );
            assert_eq!(
                run_c(dag, &inputs),
                expected,
                "C {prim:?} seed {seed} len {len}"
            );
        }
    }
}

#[test]
fn native_replay_and_bound_adjoints_match_eval() {
    for prim in [Prim::F32, Prim::F64] {
        let len = 9;
        let mut dag = Dag::new();
        let x = load(&mut dag, "x", tensor(prim, len));
        let w = load(&mut dag, "w", tensor(prim, len));
        let rate = load(&mut dag, "rate", scalar(prim));
        let low = load(&mut dag, "low", scalar(Prim::F32));
        let high = load(&mut dag, "high", scalar(Prim::F32));
        let seed = dag.add_node(
            RiscOp::synth_const(Prim::Int64, 5.0),
            vec![],
            scalar(Prim::Int64),
            None,
        );
        let dropout_key = key(&mut dag, seed);
        let dropped = dag.add_node(
            RiscOp::Dropout,
            vec![x, rate, dropout_key],
            tensor(prim, len),
            None,
        );
        let uniform_key = key(&mut dag, seed);
        let sampled = dag.add_node(
            RiscOp::UniformLike,
            vec![x, low, high, uniform_key],
            tensor(prim, len),
            None,
        );
        let mixed = dag.add_node(RiscOp::Mul, vec![dropped, sampled], tensor(prim, len), None);
        let weighted = dag.add_node(RiscOp::Mul, vec![mixed, w], tensor(prim, len), None);
        let loss = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: prim,
            },
            vec![weighted],
            scalar(prim),
            None,
        );
        dag.add_root(loss);
        let grad = chelis_ir::grad::grad_dag_checked(&dag, loss, &[x, low, high]).unwrap();
        let mut backward = grad.dag.clone();
        backward.set_roots(vec![
            grad.grad_nodes[&x],
            grad.grad_nodes[&low],
            grad.grad_nodes[&high],
        ]);
        let backward = chelis_ir::optimize::dead_code_eliminate(&backward);
        assert!(
            backward
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::DropoutReplay))
                && backward.nodes().iter().any(|node| matches!(
                    node.op,
                    RiscOp::UniformBoundAdjoint {
                        bound: UniformBound::Low
                    }
                )),
            "the gradient reads the forward keys through the replay nodes"
        );
        let inputs = [
            (
                "x",
                prim,
                vec![len],
                (0..len).map(|i| 1.0 + i as f64 / 4.0).collect(),
            ),
            (
                "w",
                prim,
                vec![len],
                (0..len).map(|i| (i as f64 - 3.0) / 2.0).collect(),
            ),
            ("rate", prim, vec![], vec![0.25]),
            ("low", Prim::F32, vec![], vec![-1.0]),
            ("high", Prim::F32, vec![], vec![3.5]),
        ];
        let eval = run_eval(&backward, &inputs);
        assert_eq!(run_c(backward, &inputs), eval, "{prim:?}");
    }
}
