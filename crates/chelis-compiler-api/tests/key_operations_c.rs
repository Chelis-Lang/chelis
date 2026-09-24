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
