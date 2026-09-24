//! Step 1 of chelis#2413: native C execution of the explicit key operations
//! on hand-built graphs, against `key_ref_ext.py`'s independent
//! transcription of [05-RNG-2], [05-OP-8] and [05-OP-37], against the
//! counter-stream C draws, and against the DAG evaluator.
mod ownership_support;

use chelis_ir::dag::{
    Dag, DimInfo, KeyBranch, NodeId, RandomDraw, RandomHandler, RiscOp, RtDim, TensorType,
    UniformBound,
};
use chelis_ir::eval::{RandomFrame, TensorValue, eval_tensor_roots_with_frame};
use chelis_types::dtype_semantics::{RawTensor, StorageView, TensorStorage, finalize_tensor};
use chelis_types::types::Prim;
use chelis_types::{RandomKey, scalar_from_i64};
use chelis_unord::UnordMap;

// ---- key_ref_ext.py: k = key(-3); (L, R) = split(k); F = fold_in(L, -5);
// S = split_n(F, 3); G = fold_in(R, 9) ----

const S_KEYS: [u64; 3] = [
    0xfa91_e16d_2c37_662f,
    0xa561_587d_4583_1567,
    0xa67b_6073_3765_a1bc,
];
const G_KEY: u64 = 0x2334_cf03_8b09_85b4;

/// key_ref_ext.py's `uniform01(key, 4, p)` stored bits for S0, S1, S2 and G.
fn uniform01_bits(prim: Prim) -> [[u64; 4]; 4] {
    match prim {
        Prim::F16 => [
            [0x2bb9, 0x36aa, 0x2c51, 0x379b],
            [0x31ac, 0x386c, 0x38a0, 0x2002],
            [0x3a75, 0x39df, 0x399b, 0x394b],
            [0x3930, 0x3024, 0x39ea, 0x39fe],
        ],
        Prim::Bf16 => [
            [0x3d77, 0x3ed5, 0x3d8a, 0x3ef3],
            [0x3e36, 0x3f0d, 0x3f14, 0x3c00],
            [0x3f4f, 0x3f3c, 0x3f33, 0x3f29],
            [0x3f26, 0x3e04, 0x3f3d, 0x3f40],
        ],
        Prim::F32 => [
            [0x3d77_27f8, 0x3ed5_3b14, 0x3d8a_16fd, 0x3ef3_5934],
            [0x3e35_8a2e, 0x3f0d_78fc, 0x3f13_f5cd, 0x3c00_3538],
            [0x3f4e_a96b, 0x3f3b_d3e4, 0x3f33_600c, 0x3f29_6b8d],
            [0x3f26_0626, 0x3e04_77f5, 0x3f3d_4491, 0x3f3f_c69e],
        ],
        Prim::F64 => [
            [
                0x3fae_e4ff_0d11_4150,
                0x3fda_a762_7ea5_1cac,
                0x3fb1_42df_ada4_6e50,
                0x3fde_6b26_7b7c_d914,
            ],
            [
                0x3fc6_b145_ce97_2d4c,
                0x3fe1_af1f_8d43_0c4a,
                0x3fe2_7eb9_9d12_90b9,
                0x3f80_06a7_08d2_dd00,
            ],
            [
                0x3fe9_d52d_5bec_29b5,
                0x3fe7_7a7c_8cd7_7700,
                0x3fe6_6c01_8e61_d20e,
                0x3fe5_2d71_a31b_67be,
            ],
            [
                0x3fe4_c0c4_b4c0_c228,
                0x3fc0_8efe_9dc0_18c0,
                0x3fe7_a892_2127_d3e4,
                0x3fe7_f8d3_bc3d_7919,
            ],
        ],
        other => panic!("no reference for {other:?}"),
    }
}

/// key_ref_ext.py's `dropout_half` keep pattern for S0, S1, S2 and G.
const DROPOUT_KEPT: [[bool; 4]; 4] = [
    [false, false, false, false],
    [false, true, true, false],
    [true, true, true, true],
    [true, false, true, true],
];

const FLOATS: [Prim; 4] = [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64];

// ---- graph construction ----

fn ty(dims: &[usize], prim: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().map(|extent| DimInfo::Lit(*extent)).collect(),
        precision: prim,
    }
}

fn node(dag: &mut Dag, op: RiscOp, inputs: Vec<NodeId>, dims: &[usize], prim: Prim) -> NodeId {
    dag.add_node(op, inputs, ty(dims, prim), None)
}

fn i64_const(dag: &mut Dag, value: i64) -> NodeId {
    node(
        dag,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Int64, value).unwrap(),
        },
        vec![],
        &[],
        Prim::Int64,
    )
}

fn float_const(dag: &mut Dag, prim: Prim, value: f64) -> NodeId {
    node(dag, RiscOp::synth_const(prim, value), vec![], &[], prim)
}

fn load(dag: &mut Dag, name: &str, dims: &[usize], prim: Prim) -> NodeId {
    node(dag, RiscOp::Load { name: name.into() }, vec![], dims, prim)
}

struct Chain {
    rows: NodeId,
    g: NodeId,
}

fn build_chain(dag: &mut Dag) -> Chain {
    let seed = i64_const(dag, -3);
    let root = node(dag, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
    let left = node(
        dag,
        RiscOp::Split {
            branch: KeyBranch::Left,
        },
        vec![root],
        &[],
        Prim::Key,
    );
    let right = node(
        dag,
        RiscOp::Split {
            branch: KeyBranch::Right,
        },
        vec![root],
        &[],
        Prim::Key,
    );
    let minus_five = i64_const(dag, -5);
    let folded = node(dag, RiscOp::FoldIn, vec![left, minus_five], &[], Prim::Key);
    let rows = node(
        dag,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![folded],
        &[3],
        Prim::Key,
    );
    let nine = i64_const(dag, 9);
    let g = node(dag, RiscOp::FoldIn, vec![right, nine], &[], Prim::Key);
    Chain { rows, g }
}

// ---- inputs and the two lanes ----

#[derive(Clone)]
enum Input {
    Floats(Prim, Vec<usize>, Vec<f64>),
    Bools(Vec<usize>, Vec<i64>),
    Ints(Vec<usize>, Vec<i64>),
    Keys(Vec<usize>, Vec<u64>),
}

impl Input {
    fn prim(&self) -> Prim {
        match self {
            Self::Floats(prim, ..) => *prim,
            Self::Bools(..) => Prim::Bool,
            Self::Ints(..) => Prim::Int64,
            Self::Keys(..) => Prim::Key,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Floats(_, shape, _)
            | Self::Bools(shape, _)
            | Self::Ints(shape, _)
            | Self::Keys(shape, _) => shape,
        }
    }

    fn storage(&self) -> TensorStorage {
        match self {
            Self::Floats(prim, _, data) => {
                finalize_tensor("test", *prim, RawTensor::Float(data.clone())).unwrap()
            }
            Self::Bools(_, data) => {
                finalize_tensor("test", Prim::Bool, RawTensor::Int(data.clone())).unwrap()
            }
            Self::Ints(_, data) => {
                finalize_tensor("test", Prim::Int64, RawTensor::Int(data.clone())).unwrap()
            }
            Self::Keys(_, bits) => TensorStorage::from_keys(
                bits.iter()
                    .map(|bits| {
                        // A key has no literal: derive each from its bits'
                        // seed, which key_from_seed returns unmixed.
                        RandomKey::from_seed(
                            scalar_from_i64("test", Prim::Int64, *bits as i64).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            ),
        }
    }
}

fn dtype_tag(prim: Prim) -> (&'static str, usize) {
    match prim {
        Prim::F64 => ("CHELIS_DTYPE_F64", 8),
        Prim::F32 => ("CHELIS_DTYPE_F32", 4),
        Prim::F16 => ("CHELIS_DTYPE_F16", 2),
        Prim::Bf16 => ("CHELIS_DTYPE_BF16", 2),
        Prim::Bool => ("CHELIS_DTYPE_BOOL", 1),
        Prim::Int64 => ("CHELIS_DTYPE_I64", 8),
        Prim::Key => ("CHELIS_DTYPE_KEY", 8),
        other => panic!("{other:?}"),
    }
}

/// Every element's stored bits, from the value's own storage.
fn value_bits(value: &TensorValue) -> Vec<u64> {
    match value.storage().view() {
        StorageView::F64(v) => v.iter().map(|x| x.to_bits()).collect(),
        StorageView::F32(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
        StorageView::F16(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
        StorageView::Bf16(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
        StorageView::I64(v) => v.iter().map(|x| *x as u64).collect(),
        StorageView::Bool(v) => v.iter().map(|x| u64::from(*x)).collect(),
        StorageView::Key(v) => v.iter().map(|key| key.bits()).collect(),
        other => panic!("{other:?}"),
    }
}

fn input_bits(input: &Input) -> Vec<u64> {
    value_bits(&TensorValue::from_storage(
        input.shape().to_vec(),
        input.storage(),
    ))
}

fn generated(dag: Dag) -> (ownership_support::GeneratedProgram, Vec<String>, Vec<Prim>) {
    let roots = dag
        .roots()
        .iter()
        .map(|root| dag.get(*root).unwrap().output_type.precision)
        .collect::<Vec<_>>();
    let verified = chelis_ir::ownership::lower_dag_ownership(dag)
        .and_then(chelis_ir::ownership::verify_ownership)
        .expect("hand-built key graph passes ownership");
    let artifact = chelis_backend_c::codegen(verified, "sample").expect("C emission");
    (
        ownership_support::GeneratedProgram::from_codegen(&artifact),
        artifact.input_labels.clone(),
        roots,
    )
}

fn driver(labels: &[String], roots: &[Prim], inputs: &[(&str, Input)]) -> String {
    let mut setup = String::new();
    for (slot, label) in labels.iter().enumerate() {
        let (_, input) = inputs
            .iter()
            .find(|(name, _)| name == label)
            .unwrap_or_else(|| panic!("no input `{label}`"));
        let (tag, width) = dtype_tag(input.prim());
        let shape = input.shape();
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
        let bits = input_bits(input);
        let listed = bits
            .iter()
            .map(|bits| format!("UINT64_C(0x{bits:x}),"))
            .collect::<String>();
        setup.push_str(&format!(
            "    inputs[{slot}] = chelis_alloc({rank}, {dims}, {tag});\n    {{\n        const uint64_t bits[] = {{{listed}0}};\n        chelis_tensor_write *write = chelis_tensor_begin_write(inputs[{slot}]);\n        unsigned char *data = (unsigned char*)chelis_tensor_write_view(write).data;\n        for (int i = 0; i < {count}; ++i) memcpy(data + i * {width}, &bits[i], {width});\n        chelis_tensor_end_write(write);\n    }}\n",
            rank = shape.len(),
            count = bits.len(),
        ));
    }
    let mut readout = String::new();
    for (slot, prim) in roots.iter().enumerate() {
        let (_, width) = dtype_tag(*prim);
        readout.push_str(&format!(
            "    {{\n        chelis_read_view view = chelis_tensor_read_view(outputs[{slot}]);\n        printf(\"=\");\n        for (int64_t i = 0; i < view.count; ++i) {{ uint64_t word = 0; memcpy(&word, (const unsigned char*)view.data + i * {width}, {width}); printf(\" %llx\", (unsigned long long)word); }}\n        printf(\"\\n\");\n        chelis_tensor_release(outputs[{slot}]);\n    }}\n"
        ));
    }
    let n_in = labels.len();
    let n_out = roots.len();
    format!(
        "int main(void) {{\n    chelis_tensor *inputs[{alloc_in}];\n{setup}    chelis_tensor *outputs[{alloc_out}];\n    sample(inputs, {n_in}, outputs, {n_out});\n{readout}    for (int i = 0; i < {n_in}; ++i) chelis_tensor_release(inputs[i]);\n    return 0;\n}}\n",
        alloc_in = n_in.max(1),
        alloc_out = n_out.max(1),
    )
}

/// Compile `dag` as a public C entry and return each root's stored bits.
fn run_c(dag: Dag, inputs: &[(&str, Input)]) -> Vec<Vec<u64>> {
    let (program, labels, roots) = generated(dag);
    let (summary, stdout) =
        ownership_support::run_with_stdout(&program, &driver(&labels, &roots, inputs));
    ownership_support::balanced(&summary);
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix('='))
        .map(|line| {
            line.split_whitespace()
                .map(|word| u64::from_str_radix(word, 16).unwrap())
                .collect()
        })
        .collect()
}

fn run_c_failure(dag: Dag, inputs: &[(&str, Input)]) -> String {
    let (program, labels, roots) = generated(dag);
    ownership_support::run_failure_stderr(&program, &driver(&labels, &roots, inputs))
}

fn run_eval(dag: &Dag, inputs: &[(&str, Input)]) -> Result<Vec<Vec<u64>>, String> {
    assert_eq!(chelis_ir::verify::verify(dag), Vec::<String>::new());
    let values = inputs
        .iter()
        .map(|(name, input)| {
            (
                *name,
                TensorValue::from_storage(input.shape().to_vec(), input.storage()),
            )
        })
        .collect::<UnordMap<_, _>>();
    let out =
        eval_tensor_roots_with_frame(dag, dag.roots(), &mut RandomFrame::unhandled(), |name| {
            values.get(name).cloned()
        })?;
    Ok(dag
        .roots()
        .iter()
        .map(|root| value_bits(&out[root]))
        .collect())
}

fn float_bits(prim: Prim, values: &[f64]) -> Vec<u64> {
    input_bits(&Input::Floats(prim, vec![values.len()], values.to_vec()))
}

// ---- oracle (b), C lane ----

#[test]
fn the_key_chain_runs_to_the_reference_keys_in_c() {
    let mut dag = Dag::new();
    let chain = build_chain(&mut dag);
    dag.add_root(chain.rows);
    dag.add_root(chain.g);
    let expected = vec![S_KEYS.to_vec(), vec![G_KEY]];
    assert_eq!(run_eval(&dag, &[]).unwrap(), expected);
    assert_eq!(run_c(dag, &[]), expected);
}

#[test]
fn chained_draws_match_the_reference_in_c_at_every_float_dtype() {
    for prim in FLOATS {
        let mut dag = Dag::new();
        let chain = build_chain(&mut dag);
        let template = load(&mut dag, "t", &[3, 4], prim);
        let low = float_const(&mut dag, prim, 0.0);
        let high = float_const(&mut dag, prim, 1.0);
        let batched = node(
            &mut dag,
            RiscOp::UniformLike,
            vec![template, low, high, chain.rows],
            &[3, 4],
            prim,
        );
        let scalar_template = load(&mut dag, "s", &[4], prim);
        let scalar = node(
            &mut dag,
            RiscOp::UniformLike,
            vec![scalar_template, low, high, chain.g],
            &[4],
            prim,
        );
        dag.add_root(batched);
        dag.add_root(scalar);
        let inputs = [
            ("t", Input::Floats(prim, vec![3, 4], vec![0.0; 12])),
            ("s", Input::Floats(prim, vec![4], vec![0.0; 4])),
        ];
        let reference = uniform01_bits(prim);
        let expected = vec![reference[..3].concat().to_vec(), reference[3].to_vec()];
        assert_eq!(run_eval(&dag, &inputs).unwrap(), expected, "{prim:?} eval");
        assert_eq!(run_c(dag, &inputs), expected, "{prim:?} C uniform");

        let mut dag = Dag::new();
        let chain = build_chain(&mut dag);
        let x = load(&mut dag, "x", &[3, 4], prim);
        let rate = float_const(&mut dag, prim, 0.5);
        let batched = node(
            &mut dag,
            RiscOp::Dropout,
            vec![x, rate, chain.rows],
            &[3, 4],
            prim,
        );
        let xs = load(&mut dag, "xs", &[4], prim);
        let scalar = node(
            &mut dag,
            RiscOp::Dropout,
            vec![xs, rate, chain.g],
            &[4],
            prim,
        );
        dag.add_root(batched);
        dag.add_root(scalar);
        let row = [1.0, 2.0, 3.0, 4.0];
        let inputs = [
            ("x", Input::Floats(prim, vec![3, 4], row.repeat(3))),
            ("xs", Input::Floats(prim, vec![4], row.to_vec())),
        ];
        let kept = |pattern: &[bool; 4]| {
            row.iter()
                .zip(pattern)
                .map(|(x, keep)| if *keep { 2.0 * x } else { 0.0 })
                .collect::<Vec<f64>>()
        };
        let batched_values: Vec<f64> = DROPOUT_KEPT[..3].iter().flat_map(kept).collect();
        let expected = vec![
            float_bits(prim, &batched_values),
            float_bits(prim, &kept(&DROPOUT_KEPT[3])),
        ];
        assert_eq!(run_eval(&dag, &inputs).unwrap(), expected, "{prim:?} eval");
        assert_eq!(run_c(dag, &inputs), expected, "{prim:?} C dropout");
    }
}

#[test]
fn per_row_controls_activations_and_bound_adjoints_agree_in_c_and_eval() {
    for prim in FLOATS {
        let mut dag = Dag::new();
        let chain = build_chain(&mut dag);
        let x = load(&mut dag, "x", &[3, 5], prim);
        let rates = load(&mut dag, "rates", &[3], prim);
        let active = load(&mut dag, "active", &[3], Prim::Bool);
        let dropped = node(
            &mut dag,
            RiscOp::Dropout,
            vec![x, rates, chain.rows, active],
            &[3, 5],
            prim,
        );
        let g = load(&mut dag, "g", &[3, 5], prim);
        let replay = node(
            &mut dag,
            RiscOp::DropoutReplay,
            vec![g, rates, chain.rows, active],
            &[3, 5],
            prim,
        );
        dag.add_root(dropped);
        dag.add_root(replay);
        dag.add_root(chain.g);
        let data: Vec<f64> = (1..=15).map(f64::from).collect();
        let inputs = [
            ("x", Input::Floats(prim, vec![3, 5], data.clone())),
            (
                "g",
                Input::Floats(prim, vec![3, 5], data.iter().map(|v| -v).collect()),
            ),
            ("rates", Input::Floats(prim, vec![3], vec![0.25, 0.0, 0.75])),
            ("active", Input::Bools(vec![3], vec![1, 0, 1])),
        ];
        let eval = run_eval(&dag, &inputs).unwrap();
        // Row 1 is inactive: positive zeros in both nodes.
        assert!(eval[0][5..10].iter().all(|bits| *bits == 0), "{prim:?}");
        assert_eq!(run_c(dag, &inputs), eval, "{prim:?} dropout and replay");

        for per_row in [false, true] {
            let mut dag = Dag::new();
            let chain = build_chain(&mut dag);
            let template = load(&mut dag, "t", &[3, 5], prim);
            let g = load(&mut dag, "g", &[3, 5], prim);
            let active = load(&mut dag, "active", &[3], Prim::Bool);
            let (low, high) = if per_row {
                (
                    load(&mut dag, "lo", &[3], prim),
                    load(&mut dag, "hi", &[3], prim),
                )
            } else {
                (
                    float_const(&mut dag, prim, -1.0),
                    float_const(&mut dag, prim, 3.0),
                )
            };
            let forward = node(
                &mut dag,
                RiscOp::UniformLike,
                vec![template, low, high, chain.rows, active],
                &[3, 5],
                prim,
            );
            let out_dims: &[usize] = if per_row { &[3] } else { &[] };
            let low_adjoint = node(
                &mut dag,
                RiscOp::UniformBoundAdjoint {
                    bound: UniformBound::Low,
                },
                vec![template, g, chain.rows, active],
                out_dims,
                prim,
            );
            let high_adjoint = node(
                &mut dag,
                RiscOp::UniformBoundAdjoint {
                    bound: UniformBound::High,
                },
                vec![template, g, chain.rows, active],
                out_dims,
                prim,
            );
            for root in [forward, low_adjoint, high_adjoint, chain.g] {
                dag.add_root(root);
            }
            let inputs = [
                ("t", Input::Floats(prim, vec![3, 5], vec![0.0; 15])),
                ("g", Input::Floats(prim, vec![3, 5], data.clone())),
                ("active", Input::Bools(vec![3], vec![1, 1, 0])),
                ("lo", Input::Floats(prim, vec![3], vec![-1.0, 0.0, 2.0])),
                ("hi", Input::Floats(prim, vec![3], vec![3.0, 0.5, 4.0])),
            ];
            let eval = run_eval(&dag, &inputs).unwrap();
            assert_eq!(
                run_c(dag, &inputs),
                eval,
                "{prim:?} uniform per_row={per_row}"
            );
        }
    }
}

/// `split_keys(split_keys(key(seed), 2)[i], 3)`: the `tensor[2, 3, key]`
/// whose element `(i, j)` is key_ref.py's
/// `split_n(split_n(key(seed), 2)[i], 3)[j]`.
fn rank_two_keys(dag: &mut Dag, seed: i64) -> NodeId {
    let seed = i64_const(dag, seed);
    let root = node(dag, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
    let rows = node(
        dag,
        RiscOp::SplitN {
            count: RtDim::Lit(2),
        },
        vec![root],
        &[2],
        Prim::Key,
    );
    node(
        dag,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![rows],
        &[2, 3],
        Prim::Key,
    )
}

/// key_ref_ext.py's `uniform01(K7[i][j], 4, "f32")`, row-major over `(i, j)`.
const RANK_TWO_UNIFORM_F32: [u64; 24] = [
    0x3f64_799a,
    0x3ed8_0a73,
    0x3dd4_668c,
    0x3f7f_4030,
    0x3f0f_6914,
    0x3bfe_1697,
    0x3f40_d9de,
    0x3f5b_136f,
    0x3f75_04eb,
    0x3f4e_767a,
    0x3dc5_4695,
    0x3ea5_d879,
    0x3f56_4547,
    0x3f5a_3cbc,
    0x3ebc_4ac1,
    0x3e25_d5fd,
    0x3f57_ebf1,
    0x3e5a_cd04,
    0x3e27_5c0a,
    0x3ddd_e418,
    0x3ef0_8b68,
    0x3f5d_2d1f,
    0x3f67_7f50,
    0x3ee4_1732,
];

/// V5 at rank 2 in C: row `(i, j)` draws with key `(i, j)`; a rate and an
/// activation shaped like the key's leading axis serve rows `(i, *)`; and a
/// bound adjoint of that shape folds the rows that share each bound.
#[test]
fn a_rank_two_key_batch_agrees_in_c_and_eval() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let uniform_keys = rank_two_keys(&mut dag, 7);
    let template = load(&mut dag, "t", &[2, 3, 4], prim);
    let low = float_const(&mut dag, prim, 0.0);
    let high = float_const(&mut dag, prim, 1.0);
    let sampled = node(
        &mut dag,
        RiscOp::UniformLike,
        vec![template, low, high, uniform_keys],
        &[2, 3, 4],
        prim,
    );
    let dropout_keys = rank_two_keys(&mut dag, 8);
    let x = load(&mut dag, "x", &[2, 3, 4], prim);
    let rates = load(&mut dag, "rates", &[2], prim);
    let active = load(&mut dag, "active", &[2], Prim::Bool);
    let dropped = node(
        &mut dag,
        RiscOp::Dropout,
        vec![x, rates, dropout_keys, active],
        &[2, 3, 4],
        prim,
    );
    dag.add_root(sampled);
    dag.add_root(dropped);
    let data: Vec<f64> = (1..=24).map(f64::from).collect();
    let inputs = [
        ("t", Input::Floats(prim, vec![2, 3, 4], vec![0.0; 24])),
        ("x", Input::Floats(prim, vec![2, 3, 4], data.clone())),
        ("rates", Input::Floats(prim, vec![2], vec![0.5, 0.0])),
        ("active", Input::Bools(vec![2], vec![0, 1])),
    ];
    let eval = run_eval(&dag, &inputs).unwrap();
    assert_eq!(eval[0], RANK_TWO_UNIFORM_F32);
    // Rows (0, *) are inactive; rows (1, *) draw at rate 0 and keep x.
    let mut kept = vec![0.0; 12];
    kept.extend(&data[12..]);
    assert_eq!(eval[1], float_bits(prim, &kept));
    assert_eq!(run_c(dag, &inputs), eval);

    for out_dims in [&[][..], &[2][..], &[2, 3][..]] {
        let mut dag = Dag::new();
        let keys = rank_two_keys(&mut dag, 7);
        let template = load(&mut dag, "t", &[2, 3, 4], prim);
        let g = load(&mut dag, "g", &[2, 3, 4], prim);
        let (low, high) = if out_dims.is_empty() {
            (
                float_const(&mut dag, prim, -1.0),
                float_const(&mut dag, prim, 3.0),
            )
        } else {
            (
                load(&mut dag, "lo", out_dims, prim),
                load(&mut dag, "hi", out_dims, prim),
            )
        };
        let forward = node(
            &mut dag,
            RiscOp::UniformLike,
            vec![template, low, high, keys],
            &[2, 3, 4],
            prim,
        );
        let adjoints = [UniformBound::Low, UniformBound::High].map(|bound| {
            node(
                &mut dag,
                RiscOp::UniformBoundAdjoint { bound },
                vec![template, g, keys],
                out_dims,
                prim,
            )
        });
        dag.add_root(forward);
        for adjoint in adjoints {
            dag.add_root(adjoint);
        }
        let groups: usize = out_dims.iter().product();
        let inputs = [
            ("t", Input::Floats(prim, vec![2, 3, 4], vec![0.0; 24])),
            ("g", Input::Floats(prim, vec![2, 3, 4], data.clone())),
            (
                "lo",
                Input::Floats(prim, out_dims.to_vec(), vec![-1.0; groups]),
            ),
            (
                "hi",
                Input::Floats(prim, out_dims.to_vec(), vec![3.0; groups]),
            ),
        ];
        let eval = run_eval(&dag, &inputs).unwrap();
        assert_eq!(eval[1].len(), groups.max(1), "{out_dims:?}");
        assert_eq!(run_c(dag, &inputs), eval, "{out_dims:?}");
    }
}

/// `vmap` twice over a draw keyed by a scalar key compiles to a rank-2 key
/// batch whose rows draw key_ref_ext.py's words in C as in eval.
#[test]
fn vmap_of_vmap_of_a_draw_agrees_in_c_and_eval() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let key = load(&mut dag, "k", &[], Prim::Key);
    let template = load(&mut dag, "t", &[4], prim);
    let low = float_const(&mut dag, prim, 0.0);
    let high = float_const(&mut dag, prim, 1.0);
    let sampled = node(
        &mut dag,
        RiscOp::UniformLike,
        vec![template, low, high, key],
        &[4],
        prim,
    );
    dag.add_root(sampled);
    let once = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(3)).unwrap();
    let twice = chelis_ir::vmap::vectorize_axis0(&once, DimInfo::Lit(2)).unwrap();
    // key_ref.py's split_n(split_n(key(7), 2)[i], 3)[j].
    let keys = [
        0x3460_149a_1d8e_2fa2,
        0x832c_74f4_3998_a706,
        0x1f97_59f0_1867_fc24,
        0xa587_cbf4_1f03_a05f,
        0x4cab_8638_d3ee_2fba,
        0xb3cc_d76d_1f33_866a,
    ];
    let inputs = [
        ("k", Input::Keys(vec![2, 3], keys.to_vec())),
        ("t", Input::Floats(prim, vec![2, 3, 4], vec![0.0; 24])),
    ];
    let eval = run_eval(&twice, &inputs).unwrap();
    assert_eq!(eval[0], RANK_TWO_UNIFORM_F32);
    assert_eq!(run_c(twice, &inputs), eval);
}

#[test]
fn a_negative_runtime_split_count_traps_in_c_as_in_eval() {
    let build = || {
        let mut dag = Dag::new();
        let seed = i64_const(&mut dag, 7);
        let root = node(&mut dag, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
        let n = load(&mut dag, "n", &[], Prim::Int64);
        let rows = dag.add_node(
            RiscOp::SplitN {
                count: RtDim::Node(1),
            },
            vec![root, n],
            TensorType {
                dims: vec![DimInfo::Named("keys".into(), None)],
                precision: Prim::Key,
            },
            None,
        );
        dag.add_root(rows);
        dag
    };
    for count in [0i64, 2] {
        let inputs = [("n", Input::Ints(vec![], vec![count]))];
        let expected = [0x25ea_33e6_1c10_576f, 0x7071_24fb_ecd5_f054];
        let want = vec![expected[..count as usize].to_vec()];
        assert_eq!(run_eval(&build(), &inputs).unwrap(), want);
        assert_eq!(run_c(build(), &inputs), want);
    }
    let inputs = [("n", Input::Ints(vec![], vec![-1]))];
    let trap = "numeric trap: domain in split_keys at i64";
    assert_eq!(run_eval(&build(), &inputs).unwrap_err(), trap);
    assert!(run_c_failure(build(), &inputs).contains(trap));
}

/// `split_keys(key(7), count)` whose count axis is declared `n`, which the
/// input `d` also binds, and, when `draw`, a dropout of `d` batched by those
/// keys.
fn split_declaring_n(count: RtDim, draw: bool) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let seed = i64_const(&mut dag, 7);
    let root = node(&mut dag, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
    let mut inputs = vec![root];
    if matches!(count, RtDim::Node(_)) {
        inputs.push(load(&mut dag, "cnt", &[], Prim::Int64));
    }
    let n = || DimInfo::Named("n".into(), None);
    let rows = dag.add_node(
        RiscOp::SplitN { count },
        inputs,
        TensorType {
            dims: vec![n()],
            precision: Prim::Key,
        },
        None,
    );
    let d = dag.add_node(
        RiscOp::Load { name: "d".into() },
        vec![],
        TensorType {
            dims: vec![n(), DimInfo::Lit(2)],
            precision: Prim::F32,
        },
        None,
    );
    if draw {
        let rate = float_const(&mut dag, Prim::F32, 0.5);
        let drawn = dag.add_node(
            RiscOp::Dropout,
            vec![d, rate, rows],
            TensorType {
                dims: vec![n(), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(drawn);
    } else {
        dag.add_root(rows);
        dag.add_root(d);
    }
    (dag, rows)
}

/// [05-OP-71]'s count is the result's extent. A count that disagrees with
/// its declared, elsewhere-bound axis traps before the result is allocated,
/// so nothing is written past it, and a batched draw keyed by those keys
/// never runs; the agreeing count gives key_ref.py's `split_n(key(7), 4)`.
#[test]
fn a_split_count_that_disagrees_with_its_declared_extent_traps_in_c() {
    let trap = "numeric trap: domain in split_keys at i64";
    let inputs = |count: i64| {
        [
            ("cnt", Input::Ints(vec![], vec![count])),
            ("d", Input::Floats(Prim::F32, vec![4, 2], vec![1.0; 8])),
        ]
    };
    for (count, draw) in [(3, false), (6, false), (40, false), (3, true)] {
        let (dag, _) = split_declaring_n(RtDim::Node(1), draw);
        let stderr = run_c_failure(dag, &inputs(count));
        assert!(stderr.contains(trap), "count {count}: {stderr}");
        assert!(
            stderr.contains(&format!(
                "extent `n`: claimed = 4, split_keys axis 0 = {count}"
            )),
            "count {count}: {stderr}"
        );
    }
    let (dag, rows) = split_declaring_n(RtDim::Node(1), false);
    let (program, ..) = generated(dag.clone());
    let guard = program
        .find("extent `n`: claimed")
        .expect("the split guards its declared extent");
    let allocation = program
        .find(&format!("chelis_tensor *t{} = chelis_alloc(", rows.0))
        .expect("the split allocates its result");
    assert!(guard < allocation, "the guard must precede the allocation");
    let keys = [
        0x25ea_33e6_1c10_576f,
        0x7071_24fb_ecd5_f054,
        0x8239_3615_3a56_5205,
        0x53c6_f7e8_3810_b049,
    ];
    assert_eq!(run_c(dag, &inputs(4))[0], keys);
    // A literal count is checked the same way against a runtime-bound axis.
    let (dag, _) = split_declaring_n(RtDim::Lit(6), false);
    let stderr = run_c_failure(dag, &inputs(0));
    assert!(
        stderr.contains("extent `n`: claimed = 4, split_keys axis 0 = 6") && stderr.contains(trap),
        "{stderr}"
    );
    let (dag, _) = split_declaring_n(RtDim::Lit(4), false);
    assert_eq!(run_c(dag, &inputs(0))[0], keys);
}

#[test]
fn an_explicitly_keyed_draw_validates_its_own_controls_in_c() {
    for (rate, traps) in [(0.5, false), (1.0, true), (-0.25, true)] {
        let build = || {
            let mut dag = Dag::new();
            let key = load(&mut dag, "k", &[], Prim::Key);
            let x = load(&mut dag, "x", &[3], Prim::F32);
            let rate = load(&mut dag, "rate", &[], Prim::F32);
            let dropped = node(
                &mut dag,
                RiscOp::Dropout,
                vec![x, rate, key],
                &[3],
                Prim::F32,
            );
            dag.add_root(dropped);
            dag
        };
        let inputs = [
            ("k", Input::Keys(vec![], vec![G_KEY])),
            ("x", Input::Floats(Prim::F32, vec![3], vec![1.0, 2.0, 3.0])),
            ("rate", Input::Floats(Prim::F32, vec![], vec![rate])),
        ];
        let trap = "numeric trap: domain in dropout at f32";
        if traps {
            assert_eq!(run_eval(&build(), &inputs).unwrap_err(), trap);
            assert!(run_c_failure(build(), &inputs).contains(trap));
        } else {
            assert_eq!(
                run_c(build(), &inputs),
                run_eval(&build(), &inputs).unwrap()
            );
        }
    }
}

/// A control or activation of a batched draw: rank 0, or one element per
/// key row.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Operand {
    Rank0,
    PerRow,
}

/// A load whose leading axis is the key batch's `n`, followed by `trailing`.
fn load_n(dag: &mut Dag, name: &str, trailing: &[usize], prim: Prim) -> NodeId {
    let mut dims = vec![DimInfo::Named("n".into(), None)];
    dims.extend(trailing.iter().map(|extent| DimInfo::Lit(*extent)));
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims,
            precision: prim,
        },
        None,
    )
}

fn batched_operand(dag: &mut Dag, name: &str, prim: Prim, shape: Operand) -> NodeId {
    match shape {
        Operand::Rank0 => load(dag, name, &[], prim),
        Operand::PerRow => load_n(dag, name, &[], prim),
    }
}

/// spec/10 §3.2: each row of a batched draw is one draw and validates the
/// controls it reads only when it is active, so a batch with no rows
/// validates nothing. The rank-0 control is invalid; the per-row control is
/// invalid in row 0 and valid in row 1, and the per-row activation turns
/// row 0 off. Row 1's draw is then key_ref_ext.py's S1 value.
#[test]
fn a_batched_draw_validates_each_active_rows_controls_in_c_and_eval() {
    let prim = Prim::F32;
    for uniform in [false, true] {
        for rows in [0usize, 2] {
            for control in [Operand::Rank0, Operand::PerRow] {
                for activation in [None, Some(Operand::Rank0), Some(Operand::PerRow)] {
                    let build = || {
                        let mut dag = Dag::new();
                        let keys = load_n(&mut dag, "k", &[], Prim::Key);
                        let x = load_n(&mut dag, "x", &[4], prim);
                        let mut inputs = vec![x];
                        if uniform {
                            inputs.push(batched_operand(&mut dag, "lo", prim, control));
                            inputs.push(batched_operand(&mut dag, "hi", prim, control));
                        } else {
                            inputs.push(batched_operand(&mut dag, "rate", prim, control));
                        }
                        inputs.push(keys);
                        if let Some(shape) = activation {
                            inputs.push(batched_operand(&mut dag, "on", Prim::Bool, shape));
                        }
                        let op = if uniform {
                            RiscOp::UniformLike
                        } else {
                            RiscOp::Dropout
                        };
                        let drawn = dag.add_node(
                            op,
                            inputs,
                            TensorType {
                                dims: vec![DimInfo::Named("n".into(), None), DimInfo::Lit(4)],
                                precision: prim,
                            },
                            None,
                        );
                        dag.add_root(drawn);
                        dag
                    };
                    let control_input = |invalid: f64, valid: f64| match control {
                        Operand::Rank0 => Input::Floats(prim, vec![], vec![invalid]),
                        Operand::PerRow => {
                            Input::Floats(prim, vec![rows], [invalid, valid][..rows].to_vec())
                        }
                    };
                    let on = match activation {
                        Some(Operand::PerRow) => Input::Bools(vec![rows], [0, 1][..rows].to_vec()),
                        _ => Input::Bools(vec![], vec![1]),
                    };
                    let mut inputs = vec![
                        ("k", Input::Keys(vec![rows], S_KEYS[..rows].to_vec())),
                        (
                            "x",
                            Input::Floats(prim, vec![rows, 4], [1.0, 2.0, 3.0, 4.0].repeat(rows)),
                        ),
                        ("on", on),
                    ];
                    if uniform {
                        inputs.push(("lo", control_input(2.0, 0.0)));
                        inputs.push(("hi", control_input(1.0, 1.0)));
                    } else {
                        inputs.push(("rate", control_input(1.5, 0.5)));
                    }
                    let row_one = if uniform {
                        uniform01_bits(prim)[1].to_vec()
                    } else {
                        let kept = [1.0, 2.0, 3.0, 4.0]
                            .into_iter()
                            .zip(DROPOUT_KEPT[1])
                            .map(|(x, keep)| if keep { 2.0 * x } else { 0.0 })
                            .collect::<Vec<_>>();
                        float_bits(prim, &kept)
                    };
                    let case = format!(
                        "uniform={uniform} rows={rows} control={control:?} activation={activation:?}"
                    );
                    if rows == 0 {
                        assert_eq!(run_eval(&build(), &inputs), Ok(vec![Vec::new()]), "{case}");
                        assert_eq!(run_c(build(), &inputs), vec![Vec::<u64>::new()], "{case}");
                    } else if control == Operand::PerRow && activation == Some(Operand::PerRow) {
                        let expected = vec![[vec![0; 4], row_one].concat()];
                        assert_eq!(run_eval(&build(), &inputs), Ok(expected.clone()), "{case}");
                        assert_eq!(run_c(build(), &inputs), expected, "{case}");
                    } else {
                        let trap = format!(
                            "numeric trap: domain in {} at f32",
                            if uniform { "uniform_like" } else { "dropout" }
                        );
                        assert_eq!(run_eval(&build(), &inputs), Err(trap.clone()), "{case}");
                        let stderr = run_c_failure(build(), &inputs);
                        assert!(stderr.contains(&trap), "{case}: {stderr}");
                    }
                }
            }
        }
    }
}

/// Round 2's probe: keys split from `key(7)` by a runtime count, and an
/// invalid rank-0 control. No rows validate nothing; two rows trap.
#[test]
fn an_empty_split_batch_with_an_invalid_control_is_empty_in_c_and_eval() {
    for uniform in [false, true] {
        for count in [0i64, 2] {
            let build = || {
                let mut dag = Dag::new();
                let seed = i64_const(&mut dag, 7);
                let root = node(&mut dag, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
                let cnt = load(&mut dag, "cnt", &[], Prim::Int64);
                let rows = dag.add_node(
                    RiscOp::SplitN {
                        count: RtDim::Node(1),
                    },
                    vec![root, cnt],
                    TensorType {
                        dims: vec![DimInfo::Named("n".into(), None)],
                        precision: Prim::Key,
                    },
                    None,
                );
                let x = load_n(&mut dag, "x", &[4], Prim::F32);
                let (op, mut inputs) = if uniform {
                    let low = float_const(&mut dag, Prim::F32, 2.0);
                    let high = float_const(&mut dag, Prim::F32, 1.0);
                    (RiscOp::UniformLike, vec![x, low, high])
                } else {
                    let rate = float_const(&mut dag, Prim::F32, 1.5);
                    (RiscOp::Dropout, vec![x, rate])
                };
                inputs.push(rows);
                let drawn = dag.add_node(
                    op,
                    inputs,
                    TensorType {
                        dims: vec![DimInfo::Named("n".into(), None), DimInfo::Lit(4)],
                        precision: Prim::F32,
                    },
                    None,
                );
                dag.add_root(drawn);
                dag
            };
            let rows = count as usize;
            let inputs = [
                ("cnt", Input::Ints(vec![], vec![count])),
                (
                    "x",
                    Input::Floats(Prim::F32, vec![rows, 4], vec![1.0; rows * 4]),
                ),
            ];
            let case = format!("uniform={uniform} count={count}");
            if count == 0 {
                assert_eq!(run_eval(&build(), &inputs), Ok(vec![Vec::new()]), "{case}");
                assert_eq!(run_c(build(), &inputs), vec![Vec::<u64>::new()], "{case}");
            } else {
                let trap = format!(
                    "numeric trap: domain in {} at f32",
                    if uniform { "uniform_like" } else { "dropout" }
                );
                assert_eq!(run_eval(&build(), &inputs), Err(trap.clone()), "{case}");
                assert!(run_c_failure(build(), &inputs).contains(&trap), "{case}");
            }
        }
    }
}

// ---- oracle (c): the counter bridge, C lane ----

#[test]
fn a_loaded_bridge_key_draws_the_counter_streams_bits_in_c() {
    let cases: Vec<(i64, u64, f64, usize)> = [(42i64, 0u64), (7, 1), (-1, 0), (0, 3)]
        .into_iter()
        .flat_map(|(seed, ordinal)| {
            [(0.5, 4usize), (0.0, 3), (0.25, 32), (0.9, 1), (0.5, 0)]
                .into_iter()
                .map(move |(rate, len)| (seed, ordinal, rate, len))
        })
        .collect();
    for prim in FLOATS {
        // Every case in one graph per lane: case `c` owns scoped handler `c`,
        // which takes `ordinal` earlier draws before the compared one.
        let mut counter = Dag::new();
        let mut keyed = Dag::new();
        let mut inputs = Vec::new();
        let names: Vec<(String, String)> = (0..cases.len())
            .map(|case| (format!("x{case}"), format!("k{case}")))
            .collect();
        for (case, (seed, ordinal, rate, len)) in cases.iter().copied().enumerate() {
            let data: Vec<f64> = (0..len).map(|i| 1.0 + i as f64).collect();
            let x = load(&mut counter, &names[case].0, &[len], prim);
            let rate_node = float_const(&mut counter, prim, rate);
            let seed_node = i64_const(&mut counter, seed);
            let mut compared = x;
            for index in 0..=ordinal {
                let key = node(
                    &mut counter,
                    RiscOp::DrawKey {
                        handler: RandomHandler::Scoped {
                            instance: case as u32,
                        },
                        draw: RandomDraw::Dropout,
                        dtype: prim,
                    },
                    vec![seed_node, rate_node],
                    &[],
                    Prim::Key,
                );
                let drawn = node(
                    &mut counter,
                    RiscOp::Dropout,
                    vec![x, rate_node, key],
                    &[len],
                    prim,
                );
                if index < ordinal {
                    counter.add_root(drawn);
                } else {
                    compared = drawn;
                }
            }
            counter.add_root(compared);

            let x = load(&mut keyed, &names[case].0, &[len], prim);
            let rate_node = float_const(&mut keyed, prim, rate);
            let key = load(&mut keyed, &names[case].1, &[], Prim::Key);
            let drawn = node(
                &mut keyed,
                RiscOp::Dropout,
                vec![x, rate_node, key],
                &[len],
                prim,
            );
            keyed.add_root(drawn);
            let bridge = RandomKey::from_counter(seed as u64, ordinal).bits();
            inputs.push((case, Input::Floats(prim, vec![len], data), bridge));
        }
        let mut named: Vec<(&str, Input)> = Vec::new();
        for (case, data, bridge) in &inputs {
            named.push((names[*case].0.as_str(), data.clone()));
            named.push((names[*case].1.as_str(), Input::Keys(vec![], vec![*bridge])));
        }
        // The compared draw of each case is the last root it added.
        let mut compared_roots = Vec::new();
        let mut root_index = 0;
        for (_, ordinal, _, _) in &cases {
            root_index += *ordinal as usize;
            compared_roots.push(root_index);
            root_index += 1;
        }
        let counter_bits = run_c(counter, &named);
        let keyed_eval = run_eval(&keyed, &named).unwrap();
        let keyed_c = run_c(keyed, &named);
        for (case, root) in compared_roots.into_iter().enumerate() {
            assert_eq!(
                keyed_c[case], counter_bits[root],
                "{prim:?} case {:?}",
                cases[case]
            );
            assert_eq!(
                keyed_eval[case], counter_bits[root],
                "{prim:?} case {:?}",
                cases[case]
            );
        }
    }
}

// ---- rule V5 at run time: extents a draw checks before it reads ----

fn named_dims(name: &str, trailing: &[usize]) -> Vec<DimInfo> {
    std::iter::once(DimInfo::Named(name.into(), None))
        .chain(trailing.iter().map(|extent| DimInfo::Lit(*extent)))
        .collect()
}

/// A value declared `[m, trailing..]` whose rows are those of the load
/// `name`, declared `[p, trailing..]`: `op` (`neg`, or `not` for Bool) of
/// that load. The verifier admits the renamed axis, and both lanes size the
/// result from the load (chelis#1277's class), so at run time its rows can
/// disagree with an `m` bound elsewhere. It is how a verified graph reaches
/// a draw with operands whose extents disagree.
fn rows_of_p(dag: &mut Dag, name: &str, trailing: &[usize], prim: Prim) -> NodeId {
    let source = dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims: named_dims("p", trailing),
            precision: prim,
        },
        None,
    );
    let op = if prim == Prim::Bool {
        RiscOp::Logical(chelis_ir::dag::LogicalKind::Not)
    } else {
        RiscOp::Neg
    };
    dag.add_node(
        op,
        vec![source],
        TensorType {
            dims: named_dims("m", trailing),
            precision: prim,
        },
        None,
    )
}

fn load_m(dag: &mut Dag, name: &str, trailing: &[usize], prim: Prim) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims: named_dims("m", trailing),
            precision: prim,
        },
        None,
    )
}

/// Which operand of a draw keyed by `k: tensor[m, key]` has `p` rows.
#[derive(Clone, Copy, Debug)]
enum Short {
    DropoutData,
    DropoutRate,
    DropoutActivation,
    UniformTemplate,
    UniformHigh,
    ReplayCotangent,
    AdjointCotangent,
}

/// A draw whose `Short` operand has `p` rows while its key batch, and every
/// other operand, has `m`. The short operand comes first, so neither lane
/// has bound `m` when it sizes that operand from its load.
fn short_operand_draw(short: Short) -> Dag {
    let mut dag = Dag::new();
    let f32_const = |dag: &mut Dag, value: f64| float_const(dag, Prim::F32, value);
    let short_data = matches!(short, Short::DropoutData | Short::UniformTemplate);
    let short_control = match short {
        Short::DropoutRate => Some(rows_of_p(&mut dag, "r", &[], Prim::F32)),
        Short::DropoutActivation => Some(rows_of_p(&mut dag, "on", &[], Prim::Bool)),
        Short::UniformHigh => Some(rows_of_p(&mut dag, "h", &[], Prim::F32)),
        _ => None,
    };
    let g = matches!(short, Short::ReplayCotangent | Short::AdjointCotangent)
        .then(|| rows_of_p(&mut dag, "g", &[4], Prim::F32));
    let data = if short_data {
        rows_of_p(&mut dag, "x", &[4], Prim::F32)
    } else {
        load_m(&mut dag, "x", &[4], Prim::F32)
    };
    let (first, second) = match short {
        Short::DropoutRate => (short_control.unwrap(), None),
        Short::DropoutActivation => (f32_const(&mut dag, 0.5), short_control),
        Short::UniformTemplate | Short::AdjointCotangent => {
            (f32_const(&mut dag, 0.0), Some(f32_const(&mut dag, 1.0)))
        }
        Short::UniformHigh => (f32_const(&mut dag, 0.0), short_control),
        _ => (f32_const(&mut dag, 0.5), None),
    };
    let keys = load_m(&mut dag, "k", &[], Prim::Key);
    let drawn_ty = TensorType {
        dims: named_dims("m", &[4]),
        precision: Prim::F32,
    };
    let add = |dag: &mut Dag, op: RiscOp, inputs: Vec<NodeId>, ty: TensorType| {
        let id = dag.add_node(op, inputs, ty, None);
        dag.add_root(id);
    };
    match short {
        Short::DropoutData | Short::DropoutRate => {
            add(&mut dag, RiscOp::Dropout, vec![data, first, keys], drawn_ty);
        }
        Short::DropoutActivation => {
            let on = second.unwrap();
            add(
                &mut dag,
                RiscOp::Dropout,
                vec![data, first, keys, on],
                drawn_ty,
            );
        }
        Short::UniformTemplate | Short::UniformHigh => {
            let inputs = vec![data, first, second.unwrap(), keys];
            add(&mut dag, RiscOp::UniformLike, inputs, drawn_ty);
        }
        // A replay runs before its forward draw here, so the forward's own
        // result claim, which C checks against the `m` its entry bound from
        // the short cotangent, is not what traps first.
        Short::ReplayCotangent => {
            let replay = vec![g.unwrap(), first, keys];
            add(&mut dag, RiscOp::DropoutReplay, replay, drawn_ty.clone());
            add(&mut dag, RiscOp::Dropout, vec![data, first, keys], drawn_ty);
        }
        Short::AdjointCotangent => {
            add(
                &mut dag,
                RiscOp::UniformBoundAdjoint {
                    bound: UniformBound::High,
                },
                vec![data, g.unwrap(), keys],
                TensorType {
                    dims: vec![],
                    precision: Prim::F32,
                },
            );
            let forward = vec![data, first, second.unwrap(), keys];
            add(&mut dag, RiscOp::UniformLike, forward, drawn_ty);
        }
    }
    dag
}

/// Inputs for [`short_operand_draw`]: three keys, so `m` is 3, and `p`
/// rows in each load declared `[p, ..]`.
fn short_operand_inputs(short: Short, p: usize) -> Vec<(&'static str, Input)> {
    let x = if matches!(short, Short::DropoutData | Short::UniformTemplate) {
        p
    } else {
        3
    };
    vec![
        ("k", Input::Keys(vec![3], S_KEYS.to_vec())),
        ("x", Input::Floats(Prim::F32, vec![x, 4], vec![1.0; x * 4])),
        ("r", Input::Floats(Prim::F32, vec![p], vec![-0.5; p])),
        ("on", Input::Bools(vec![p], vec![0; p])),
        ("h", Input::Floats(Prim::F32, vec![p], vec![-1.0; p])),
        ("g", Input::Floats(Prim::F32, vec![p, 4], vec![1.0; p * 4])),
    ]
}

/// Only the inputs `dag` loads.
fn inputs_for(dag: &Dag, inputs: &[(&'static str, Input)]) -> Vec<(&'static str, Input)> {
    inputs
        .iter()
        .filter(|(name, _)| {
            dag.nodes().iter().any(
                |node| matches!(&node.op, RiscOp::Load { name: loaded } if loaded.as_str() == *name),
            )
        })
        .cloned()
        .collect()
}

/// Spec/10 §3.2's rule V5 relates a draw's declared dims, which the verifier
/// checks; this is the run-time half. A draw whose operand's rows disagree
/// with its key batch traps before it reads anything, with the same extent
/// report and `Domain` trap in the DAG evaluator and in C; with agreeing
/// rows both lanes draw the same bits. Each draw's row length is its data's
/// element count over its key's, never its declared result's.
///
/// Evidentiary status: REGRESSION TEST. At c23ec448a eval reported untyped
/// errors, and C read past the short operand or the key batch.
#[test]
fn a_draw_whose_operand_rows_disagree_with_its_keys_traps_in_c_as_in_eval() {
    let cases = [
        (
            Short::DropoutData,
            "extent `m`: claimed = 3, dropout input 0 axis 0 = 2",
            "dropout",
        ),
        (
            Short::DropoutRate,
            "extent `m`: claimed = 3, dropout input 1 axis 0 = 2",
            "dropout",
        ),
        (
            Short::DropoutActivation,
            "extent `m`: claimed = 3, dropout input 3 axis 0 = 2",
            "dropout",
        ),
        (
            Short::UniformTemplate,
            "extent `m`: claimed = 3, uniform_like input 0 axis 0 = 2",
            "uniform_like",
        ),
        (
            Short::UniformHigh,
            "extent `m`: claimed = 3, uniform_like input 2 axis 0 = 2",
            "uniform_like",
        ),
        (
            Short::ReplayCotangent,
            "extent `m`: claimed = 3, dropout input 0 axis 0 = 2",
            "dropout",
        ),
        (
            Short::AdjointCotangent,
            "extent `m`: claimed = 3, uniform_like input 1 axis 0 = 2",
            "uniform_like",
        ),
    ];
    for (short, line, op) in cases {
        let report = format!("{line}\nnumeric trap: domain in {op} at i64");
        let dag = short_operand_draw(short);
        let inputs = inputs_for(&dag, &short_operand_inputs(short, 2));
        assert_eq!(run_eval(&dag, &inputs), Err(report.clone()), "{short:?}");
        let stderr = run_c_failure(dag, &inputs);
        assert!(stderr.contains(&report), "{short:?}: {stderr}");
    }
    // With three rows everywhere the same graphs draw, in both lanes alike.
    for short in [
        Short::DropoutData,
        Short::DropoutRate,
        Short::UniformTemplate,
        Short::ReplayCotangent,
        Short::AdjointCotangent,
    ] {
        let dag = short_operand_draw(short);
        let inputs = inputs_for(&dag, &short_operand_inputs(short, 3));
        let eval = run_eval(&dag, &inputs).unwrap();
        assert!(eval.iter().any(|root| !root.is_empty()), "{short:?}");
        assert_eq!(run_c(dag, &inputs), eval, "{short:?}");
    }
    // The guard precedes the draw's allocation, and its row length is the
    // data's element count over the key's.
    let dag = short_operand_draw(Short::DropoutData);
    let data = dag.get(dag.roots()[0]).unwrap().inputs[0].0;
    let drawn = dag.roots()[0].0;
    let (program, ..) = generated(dag);
    let guard = program
        .find("dropout input 0 axis 0 = %lld")
        .expect("the draw guards its data's leading axes");
    let allocation = program
        .find(&format!("chelis_tensor *t{drawn} = chelis_alloc("))
        .expect("the draw allocates its result");
    assert!(guard < allocation, "the guard must precede the allocation");
    assert!(
        program.contains(&format!(
            "t{drawn}_row_len = t{drawn}_rows > 0 ? t{data}_size / t{drawn}_rows : 0;"
        )),
        "the row length is the data's"
    );
}

/// C allocates a draw's result from its declared dims, so it checks each
/// declared extent against the data's before the allocation, with the local
/// extent guard's report. Here `m` is bound from `neg(a)`'s axis, three, and
/// the later load `x: tensor[m, 4, f32]` holds two rows, which C's entry
/// does not check against `m` (chelis#1277's class). The rank-0-keyed
/// dropout of `x` then traps instead of reading twelve elements of eight.
/// The DAG evaluator builds the result from `x` and reads no declared
/// extent, so it is not compared here.
///
/// Evidentiary status: REGRESSION TEST. At c23ec448a C read past `x`.
#[test]
fn a_draw_whose_declared_result_disagrees_with_its_data_traps_in_c() {
    let build = || {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            TensorType {
                dims: named_dims("p", &[]),
                precision: Prim::F32,
            },
            None,
        );
        let negated = dag.add_node(
            RiscOp::Neg,
            vec![a],
            TensorType {
                dims: named_dims("m", &[]),
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(negated);
        let x = load_m(&mut dag, "x", &[4], Prim::F32);
        let rate = float_const(&mut dag, Prim::F32, 0.5);
        let key = load(&mut dag, "k", &[], Prim::Key);
        let drawn = dag.add_node(
            RiscOp::Dropout,
            vec![x, rate, key],
            TensorType {
                dims: named_dims("m", &[4]),
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(drawn);
        (dag, drawn)
    };
    let inputs = |rows: usize| {
        [
            ("a", Input::Floats(Prim::F32, vec![3], vec![1.0; 3])),
            (
                "x",
                Input::Floats(Prim::F32, vec![rows, 4], vec![1.0; rows * 4]),
            ),
            ("k", Input::Keys(vec![], vec![G_KEY])),
        ]
    };
    let (dag, drawn) = build();
    let stderr = run_c_failure(dag.clone(), &inputs(2));
    assert!(
        stderr.contains(
            "extent `m`: claimed = 3, dropout axis 0 = 2\nnumeric trap: domain in dropout at i64"
        ),
        "{stderr}"
    );
    let (program, ..) = generated(dag.clone());
    let guard = program
        .find("dropout axis 0 = %lld")
        .expect("the draw guards its declared extents");
    let allocation = program
        .find(&format!("chelis_tensor *t{} = chelis_alloc(", drawn.0))
        .expect("the draw allocates its result");
    assert!(guard < allocation, "the guard must precede the allocation");
    // Three rows agree with `m`, and the lanes draw alike.
    assert_eq!(
        run_c(dag.clone(), &inputs(3)),
        run_eval(&dag, &inputs(3)).unwrap()
    );
}

// ---- extents a key operation checks before it reads ----

/// An elementwise key operation: [05-OP-69], one half of [05-OP-70], or
/// [05-OP-72].
#[derive(Clone, Copy, Debug)]
enum KeyOperation {
    FromSeed,
    Split,
    FoldIn,
}

impl KeyOperation {
    fn name(self) -> &'static str {
        match self {
            Self::FromSeed => "key_from_seed",
            Self::Split => "split_key",
            Self::FoldIn => "fold_in",
        }
    }
}

/// A value declared `[m]` whose extent is that of the load `name`, declared
/// `[p]`: `neg` of that load, as in [`rows_of_p`].
fn neg_of_p(dag: &mut Dag, name: &str, prim: Prim) -> NodeId {
    rows_of_p(dag, name, &[], prim)
}

/// `op` over loads declared `[m]`, its result declared `[m]`, after the root
/// `neg(a)` declared `[m]` over `a: tensor[p, f32]`. C's entry binds `m`
/// from that `neg`'s axis, which is `a`'s, and does not check the later
/// loads against it (chelis#1277's class), so at run time the key
/// operation's declared extent can differ from its operands'.
fn key_operation_after_a(op: KeyOperation) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let negated = neg_of_p(&mut dag, "a", Prim::F32);
    dag.add_root(negated);
    let declared = || TensorType {
        dims: named_dims("m", &[]),
        precision: Prim::Key,
    };
    let result = match op {
        KeyOperation::FromSeed => {
            let seeds = load_m(&mut dag, "s", &[], Prim::Int64);
            dag.add_node(RiscOp::KeyFromSeed, vec![seeds], declared(), None)
        }
        KeyOperation::Split => {
            let keys = load_m(&mut dag, "k", &[], Prim::Key);
            let branch = KeyBranch::Left;
            dag.add_node(RiscOp::Split { branch }, vec![keys], declared(), None)
        }
        KeyOperation::FoldIn => {
            let keys = load_m(&mut dag, "k", &[], Prim::Key);
            let indices = load_m(&mut dag, "i", &[], Prim::Int64);
            dag.add_node(RiscOp::FoldIn, vec![keys, indices], declared(), None)
        }
    };
    dag.add_root(result);
    (dag, result)
}

/// Inputs for [`key_operation_after_a`]: `a` holds `m` elements, and every
/// operand two.
fn key_operation_inputs(m: usize) -> Vec<(&'static str, Input)> {
    vec![
        ("a", Input::Floats(Prim::F32, vec![m], vec![1.0; m])),
        ("s", Input::Ints(vec![2], vec![-3, 7])),
        ("k", Input::Keys(vec![2], vec![1, 2])),
        ("i", Input::Ints(vec![2], vec![5, 6])),
    ]
}

/// key_ref.py's `fold_in(key(1), 5)` and `fold_in(key(2), 6)`.
const FOLDED: [u64; 2] = [0x3fcf_c3fe_583c_3f8c, 0x6595_59e7_5fe8_debb];

/// C allocates an elementwise key operation's result from its declared dims
/// and reads every operand at each index of it, so before the allocation it
/// checks each declared extent against its operands', with the local extent
/// guard's report. Here `m` is three and each operand holds two elements.
/// The DAG evaluator builds the result from its operands and reads no
/// declared extent, so on this graph it returns two keys where C traps, as
/// it does for a generic operation; it is not compared here. With `m` two
/// the lanes agree on key_ref.py's keys.
///
/// Evidentiary status: REGRESSION TEST. At 8a79e7c95 C exited normally
/// after reading a third element past each two-element operand.
#[test]
fn a_key_operation_whose_declared_result_exceeds_its_operands_traps_in_c() {
    for (op, reference) in [
        // key_ref.py's key(-3) and key(7).
        (KeyOperation::FromSeed, [0xffff_ffff_ffff_fffd, 7]),
        // key_ref.py's split(key(1)).0 and split(key(2)).0.
        (
            KeyOperation::Split,
            [0x187d_9c3f_8064_0697, 0x000a_32cd_565f_bcd0],
        ),
        (KeyOperation::FoldIn, FOLDED),
    ] {
        let name = op.name();
        let (dag, result) = key_operation_after_a(op);
        let inputs = inputs_for(&dag, &key_operation_inputs(3));
        let stderr = run_c_failure(dag.clone(), &inputs);
        let report = format!(
            "extent `m`: claimed = 3, {name} axis 0 = 2\nnumeric trap: domain in {name} at i64"
        );
        assert!(stderr.contains(&report), "{op:?}: {stderr}");
        let (program, ..) = generated(dag.clone());
        let guard = program
            .find(&format!("{name} axis 0 = %lld"))
            .expect("the key operation guards its declared extents");
        let allocation = program
            .find(&format!("chelis_tensor *t{} = chelis_alloc(", result.0))
            .expect("the key operation allocates its result");
        assert!(
            guard < allocation,
            "{op:?}: the guard must precede the allocation"
        );
        let inputs = inputs_for(&dag, &key_operation_inputs(2));
        let eval = run_eval(&dag, &inputs).unwrap();
        assert_eq!(eval[1], reference.to_vec(), "{op:?}");
        assert_eq!(run_c(dag, &inputs), eval, "{op:?}");
    }
}

/// `fold_in(k, i)` whose key and index disagree at run time. The operand
/// that is `neg` of a load declared `[p]` comes first, so neither lane has
/// bound `m` when it sizes it, and C's entry binds `m` from it. Both lanes
/// check the index's extents against the key's before reading either, with
/// one report; with agreeing extents both return key_ref.py's keys.
///
/// Evidentiary status: REGRESSION TEST. At 8a79e7c95 the DAG evaluator
/// reported an untyped shape error, and C read past the shorter operand.
#[test]
fn a_fold_in_whose_key_and_index_disagree_traps_in_c_as_in_eval() {
    // `index_first`: `i = neg(j)` with `j: tensor[p, i64]` and the load
    // `k: tensor[m, key]`; otherwise `k = key_from_seed(neg(s))` with
    // `s: tensor[p, i64]` and the load `i: tensor[m, i64]`.
    let build = |index_first: bool| {
        let mut dag = Dag::new();
        let declared = || TensorType {
            dims: named_dims("m", &[]),
            precision: Prim::Key,
        };
        let (keys, indices) = if index_first {
            let indices = neg_of_p(&mut dag, "j", Prim::Int64);
            (load_m(&mut dag, "k", &[], Prim::Key), indices)
        } else {
            let seeds = neg_of_p(&mut dag, "s", Prim::Int64);
            let keys = dag.add_node(RiscOp::KeyFromSeed, vec![seeds], declared(), None);
            (keys, load_m(&mut dag, "i", &[], Prim::Int64))
        };
        let folded = dag.add_node(RiscOp::FoldIn, vec![keys, indices], declared(), None);
        dag.add_root(folded);
        dag
    };
    let inputs = |index_first: bool, p: usize| {
        if index_first {
            vec![
                ("j", Input::Ints(vec![p], vec![-5, -6, -7][..p].to_vec())),
                ("k", Input::Keys(vec![2], vec![1, 2])),
            ]
        } else {
            vec![
                ("s", Input::Ints(vec![p], vec![-1, -2, -3][..p].to_vec())),
                ("i", Input::Ints(vec![2], vec![5, 6])),
            ]
        }
    };
    for (index_first, line) in [
        (true, "extent `m`: claimed = 2, fold_in input 1 axis 0 = 3"),
        (false, "extent `m`: claimed = 3, fold_in input 1 axis 0 = 2"),
    ] {
        let report = format!("{line}\nnumeric trap: domain in fold_in at i64");
        let dag = build(index_first);
        let inputs3 = inputs(index_first, 3);
        assert_eq!(
            run_eval(&dag, &inputs3),
            Err(report.clone()),
            "{index_first}"
        );
        let stderr = run_c_failure(dag.clone(), &inputs3);
        assert!(stderr.contains(&report), "{index_first}: {stderr}");
        let inputs2 = inputs(index_first, 2);
        let eval = run_eval(&dag, &inputs2).unwrap();
        assert_eq!(eval, vec![FOLDED.to_vec()], "{index_first}");
        assert_eq!(run_c(dag, &inputs2), eval, "{index_first}");
    }
}
