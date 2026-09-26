//! Step 1 of chelis#2413: the explicit key operations in the IR, on
//! hand-built graphs.
//!
//! Every expected key and draw bit pattern comes from `key_ref_ext.py`, an
//! independent Python transcription of [05-RNG-2], [05-OP-8] and [05-OP-37]
//! (the design pass's `key_ref.py` plus the draw rules), or from the `spec_*`
//! transcriptions below, never from an evaluator or kernel. The graphs cover
//! evaluation of `KeyFromSeed -> Split -> FoldIn -> SplitN -> draw` at every
//! float dtype, row-batched draws, the counter-stream bridge, the verifier's
//! key rules V1 to V5, `grad`, the optimizer, and `vmap`.

use chelis_ir::dag::{
    Dag, DimInfo, KeyBranch, LogicalKind, NodeId, Owner, RiscOp, RtDim, TensorType, UniformBound,
    record_runtime_dim_shape_deps,
};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_exact};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::optimize::{common_subexpr_eliminate, constant_fold, dead_code_eliminate};
use chelis_ir::verify::verify;
use chelis_ir::vmap::vectorize_axis0;
use chelis_types::dtype_semantics::{RawTensor, StorageView, TensorStorage, finalize_tensor};
use chelis_types::types::Prim;
use chelis_types::{RandomKey, ScalarValue, scalar_from_i64};
use chelis_unord::UnordMap;

// ---- key_ref_ext.py: k = key(-3); (L, R) = split(k, k_decl); F = fold_in(L, -5);
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

/// key_ref_ext.py's `dropout_half(key, [1,2,3,4])` keep pattern for S0, S1,
/// S2 and G: rate 0.5 keeps element i exactly when its unit is >= 0.5, and a
/// kept element stores 2x. The pattern is the same at every float dtype.
const DROPOUT_KEPT: [[bool; 4]; 4] = [
    [false, false, false, false],
    [false, true, true, false],
    [true, true, true, true],
    [true, false, true, true],
];

// ---- graph construction ----

fn ty(dims: &[usize], prim: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().map(|extent| DimInfo::Lit(*extent)).collect(),
        precision: prim,
    }
}

fn node(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    op: RiscOp,
    inputs: Vec<NodeId>,
    dims: &[usize],
    prim: Prim,
) -> NodeId {
    dag.add_node(decl, op, inputs, ty(dims, prim), None)
}

fn i64_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: i64) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Int64, value).unwrap(),
        },
        vec![],
        &[],
        Prim::Int64,
    )
}

fn float_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, prim: Prim, value: f64) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::synth_const(prim, value),
        vec![],
        &[],
        prim,
    )
}

fn load(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    name: &str,
    dims: &[usize],
    prim: Prim,
) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Load { name: name.into() },
        vec![],
        dims,
        prim,
    )
}

struct Chain {
    /// `split_n(fold_in(split(key(-3)).0, -5), 3)`: rank 1, three keys.
    rows: NodeId,
    /// `fold_in(split(key(-3)).1, 9)`: rank 0.
    g: NodeId,
}

fn build_chain(dag: &mut Dag, decl: chelis_ir::dag::DeclId) -> Chain {
    let seed = i64_const(dag, decl, -3);
    let root = node(dag, decl, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
    let left = node(
        dag,
        decl,
        RiscOp::Split {
            branch: KeyBranch::Left,
        },
        vec![root],
        &[],
        Prim::Key,
    );
    let right = node(
        dag,
        decl,
        RiscOp::Split {
            branch: KeyBranch::Right,
        },
        vec![root],
        &[],
        Prim::Key,
    );
    let minus_five = i64_const(dag, decl, -5);
    let folded = node(
        dag,
        decl,
        RiscOp::FoldIn,
        vec![left, minus_five],
        &[],
        Prim::Key,
    );
    let rows = node(
        dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![folded],
        &[3],
        Prim::Key,
    );
    let nine = i64_const(dag, decl, 9);
    let g = node(dag, decl, RiscOp::FoldIn, vec![right, nine], &[], Prim::Key);
    Chain { rows, g }
}

fn run(dag: &Dag, inputs: &UnordMap<&str, TensorValue>) -> UnordMap<NodeId, TensorValue> {
    assert_eq!(verify(dag), Vec::<String>::new());
    eval_tensor_roots_exact(dag, dag.roots(), |name| inputs.get(name).cloned()).unwrap()
}

fn stored_bits(value: &TensorValue) -> Vec<u64> {
    match value.storage().view() {
        StorageView::F16(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
        StorageView::Bf16(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
        StorageView::F32(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
        StorageView::F64(values) => values.iter().map(|v| v.to_bits()).collect(),
        other => panic!("not a float draw: {other:?}"),
    }
}

fn key_bits(value: &TensorValue) -> Vec<u64> {
    value
        .storage()
        .keys()
        .expect("a key value")
        .iter()
        .map(|key| key.bits())
        .collect()
}

fn floats(prim: Prim, shape: Vec<usize>, data: Vec<f64>) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor("test", prim, RawTensor::Float(data)).unwrap(),
    )
}

fn keys_value(shape: Vec<usize>, keys: Vec<RandomKey>) -> TensorValue {
    TensorValue::from_storage(shape, TensorStorage::from_keys(keys))
}

const FLOATS: [Prim; 4] = [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64];

// ---- oracle (b), eval lane ----

#[test]
fn the_key_chain_evaluates_to_the_reference_keys() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let chain = build_chain(&mut dag, decl);
    dag.add_root(chain.rows);
    dag.add_root(chain.g);
    let out = run(&dag, &UnordMap::new());
    assert_eq!(key_bits(&out[&chain.rows]), S_KEYS);
    assert_eq!(out[&chain.rows].shape, vec![3]);
    assert_eq!(key_bits(&out[&chain.g]), [G_KEY]);
    assert!(out[&chain.g].shape.is_empty());
}

#[test]
fn chained_draws_match_the_reference_at_every_float_dtype() {
    for prim in FLOATS {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let chain = build_chain(&mut dag, decl);
        let template = load(&mut dag, decl, "t", &[3, 4], prim);
        let low = float_const(&mut dag, decl, prim, 0.0);
        let high = float_const(&mut dag, decl, prim, 1.0);
        let batched = node(
            &mut dag,
            decl,
            RiscOp::UniformLike,
            vec![template, low, high, chain.rows],
            &[3, 4],
            prim,
        );
        let scalar_template = load(&mut dag, decl, "s", &[4], prim);
        let scalar = node(
            &mut dag,
            decl,
            RiscOp::UniformLike,
            vec![scalar_template, low, high, chain.g],
            &[4],
            prim,
        );
        dag.add_root(batched);
        dag.add_root(scalar);
        let inputs = UnordMap::from_iter([
            ("t", floats(prim, vec![3, 4], vec![0.0; 12])),
            ("s", floats(prim, vec![4], vec![0.0; 4])),
        ]);
        let out = run(&dag, &inputs);
        let expected = uniform01_bits(prim);
        let batched_bits = stored_bits(&out[&batched]);
        for (row, expect) in expected[..3].iter().enumerate() {
            assert_eq!(
                &batched_bits[row * 4..row * 4 + 4],
                expect,
                "{prim:?} uniform row {row}"
            );
        }
        assert_eq!(
            stored_bits(&out[&scalar]),
            expected[3],
            "{prim:?} uniform G"
        );

        // Dropout at rate 0.5 over rows [1, 2, 3, 4].
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let chain = build_chain(&mut dag, decl);
        let x = load(&mut dag, decl, "x", &[3, 4], prim);
        let rate = float_const(&mut dag, decl, prim, 0.5);
        let batched = node(
            &mut dag,
            decl,
            RiscOp::Dropout,
            vec![x, rate, chain.rows],
            &[3, 4],
            prim,
        );
        let xs = load(&mut dag, decl, "xs", &[4], prim);
        let scalar = node(
            &mut dag,
            decl,
            RiscOp::Dropout,
            vec![xs, rate, chain.g],
            &[4],
            prim,
        );
        dag.add_root(batched);
        dag.add_root(scalar);
        let row = [1.0, 2.0, 3.0, 4.0];
        let inputs = UnordMap::from_iter([
            ("x", floats(prim, vec![3, 4], row.repeat(3))),
            ("xs", floats(prim, vec![4], row.to_vec())),
        ]);
        let out = run(&dag, &inputs);
        let expect = |kept: &[bool; 4]| {
            row.iter()
                .zip(kept)
                .map(|(x, keep)| if *keep { 2.0 * x } else { 0.0 })
                .collect::<Vec<f64>>()
        };
        let mut batched_expected = Vec::new();
        for kept in &DROPOUT_KEPT[..3] {
            batched_expected.extend(expect(kept));
        }
        assert_eq!(
            out[&batched].to_f64_lossy_vec(),
            batched_expected,
            "{prim:?}"
        );
        assert_eq!(
            out[&scalar].to_f64_lossy_vec(),
            expect(&DROPOUT_KEPT[3]),
            "{prim:?}"
        );
    }
}

#[test]
fn a_batched_draw_equals_its_rows_drawn_with_scalar_keys() {
    // [05-OP-71]: row j of split_keys(F, 3) is fold_in(F, j), so three
    // graphs that each fold one index into a fresh F give the rows.
    for prim in FLOATS {
        let mut batched_dag = Dag::new();
        let batched_dag_decl = batched_dag.declare("test");
        let chain = build_chain(&mut batched_dag, batched_dag_decl);
        let x = load(&mut batched_dag, batched_dag_decl, "x", &[3, 5], prim);
        let rate = float_const(&mut batched_dag, batched_dag_decl, prim, 0.25);
        let batched = node(
            &mut batched_dag,
            batched_dag_decl,
            RiscOp::Dropout,
            vec![x, rate, chain.rows],
            &[3, 5],
            prim,
        );
        batched_dag.add_root(batched);
        batched_dag.add_root(chain.g);
        let data: Vec<f64> = (1..=15).map(f64::from).collect();
        let inputs = UnordMap::from_iter([("x", floats(prim, vec![3, 5], data.clone()))]);
        let batched_out = run(&batched_dag, &inputs)[&batched].to_f64_lossy_vec();

        let mut stacked = Vec::new();
        for j in 0..3 {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let seed = i64_const(&mut dag, decl, -3);
            let root = node(
                &mut dag,
                decl,
                RiscOp::KeyFromSeed,
                vec![seed],
                &[],
                Prim::Key,
            );
            let left = node(
                &mut dag,
                decl,
                RiscOp::Split {
                    branch: KeyBranch::Left,
                },
                vec![root],
                &[],
                Prim::Key,
            );
            let minus_five = i64_const(&mut dag, decl, -5);
            let folded = node(
                &mut dag,
                decl,
                RiscOp::FoldIn,
                vec![left, minus_five],
                &[],
                Prim::Key,
            );
            let index = i64_const(&mut dag, decl, j);
            let row_key = node(
                &mut dag,
                decl,
                RiscOp::FoldIn,
                vec![folded, index],
                &[],
                Prim::Key,
            );
            let x = load(&mut dag, decl, "x", &[5], prim);
            let rate = float_const(&mut dag, decl, prim, 0.25);
            let drawn = node(
                &mut dag,
                decl,
                RiscOp::Dropout,
                vec![x, rate, row_key],
                &[5],
                prim,
            );
            dag.add_root(drawn);
            let row = data[j as usize * 5..j as usize * 5 + 5].to_vec();
            let inputs = UnordMap::from_iter([("x", floats(prim, vec![5], row))]);
            stacked.extend(run(&dag, &inputs)[&drawn].to_f64_lossy_vec());
        }
        assert_eq!(batched_out, stacked, "{prim:?}");
    }
}

#[test]
fn per_row_controls_and_activations_select_each_row_independently() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let chain = build_chain(&mut dag, decl);
    let x = load(&mut dag, decl, "x", &[3, 4], prim);
    let rates = load(&mut dag, decl, "rates", &[3], prim);
    let active = load(&mut dag, decl, "active", &[3], Prim::Bool);
    let drawn = dag.add_node(
        Owner::new(decl, Some(active)),
        RiscOp::Dropout,
        vec![x, rates, chain.rows],
        ty(&[3, 4], prim),
        None,
    );
    dag.add_root(drawn);
    dag.add_root(chain.g);
    let row = [1.0, 2.0, 3.0, 4.0];
    let inputs = UnordMap::from_iter([
        ("x", floats(prim, vec![3, 4], row.repeat(3))),
        ("rates", floats(prim, vec![3], vec![0.5, 0.0, 0.5])),
        (
            "active",
            TensorValue::from_storage(
                vec![3],
                finalize_tensor("test", Prim::Bool, RawTensor::Int(vec![1, 1, 0])).unwrap(),
            ),
        ),
    ]);
    let out = run(&dag, &inputs)[&drawn].to_f64_lossy_vec();
    // Row 0 keeps S0's pattern at 0.5 (nothing); row 1 at rate 0 keeps
    // everything, scaled by 1/(1-0) = 1; row 2 is inactive: positive zeros.
    assert_eq!(
        out,
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    );
    // A trap in an active row's rate is reported; an inactive row's bad rate
    // is never validated.
    let bad = UnordMap::from_iter([
        ("x", floats(prim, vec![3, 4], row.repeat(3))),
        ("rates", floats(prim, vec![3], vec![0.5, 0.5, 2.0])),
        (
            "active",
            TensorValue::from_storage(
                vec![3],
                finalize_tensor("test", Prim::Bool, RawTensor::Int(vec![1, 1, 0])).unwrap(),
            ),
        ),
    ]);
    assert!(eval_tensor_roots_exact(&dag, dag.roots(), |n| bad.get(n).cloned()).is_ok());
    let trapping = UnordMap::from_iter([
        ("x", floats(prim, vec![3, 4], row.repeat(3))),
        ("rates", floats(prim, vec![3], vec![0.5, 2.0, 0.5])),
        (
            "active",
            TensorValue::from_storage(
                vec![3],
                finalize_tensor("test", Prim::Bool, RawTensor::Int(vec![1, 1, 0])).unwrap(),
            ),
        ),
    ]);
    let error =
        eval_tensor_roots_exact(&dag, dag.roots(), |n| trapping.get(n).cloned()).unwrap_err();
    assert!(error.contains("dropout"), "{error}");
}

#[test]
fn a_key_load_draws_with_its_bits_and_refuses_raw_ingress() {
    // Oracle (c), eval lane: a draw keyed by a loaded key draws with the
    // loaded bits, the same draw as `key_from_seed` of those bits.
    for seed in [7i64, -1, 0, i64::MAX] {
        for prim in FLOATS {
            let row: Vec<f64> = (1..=6).map(f64::from).collect();
            let mut seeded = Dag::new();
            let seeded_decl = seeded.declare("test");
            let x = load(&mut seeded, seeded_decl, "x", &[6], prim);
            let rate = float_const(&mut seeded, seeded_decl, prim, 0.3);
            let seed_node = i64_const(&mut seeded, seeded_decl, seed);
            let key = node(
                &mut seeded,
                seeded_decl,
                RiscOp::KeyFromSeed,
                vec![seed_node],
                &[],
                Prim::Key,
            );
            let drawn = node(
                &mut seeded,
                seeded_decl,
                RiscOp::Dropout,
                vec![x, rate, key],
                &[6],
                prim,
            );
            seeded.add_root(drawn);
            let inputs = UnordMap::from_iter([("x", floats(prim, vec![6], row.clone()))]);
            let seeded_bits = stored_bits(&run(&seeded, &inputs)[&drawn]);

            let mut keyed = Dag::new();
            let keyed_decl = keyed.declare("test");
            let x = load(&mut keyed, keyed_decl, "x", &[6], prim);
            let rate = float_const(&mut keyed, keyed_decl, prim, 0.3);
            let key = load(&mut keyed, keyed_decl, "k", &[], Prim::Key);
            let drawn = node(
                &mut keyed,
                keyed_decl,
                RiscOp::Dropout,
                vec![x, rate, key],
                &[6],
                prim,
            );
            keyed.add_root(drawn);
            let loaded =
                RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, seed).unwrap()).unwrap();
            let inputs = UnordMap::from_iter([
                ("x", floats(prim, vec![6], row.clone())),
                ("k", keys_value(vec![], vec![loaded])),
            ]);
            assert_eq!(
                stored_bits(&run(&keyed, &inputs)[&drawn]),
                seeded_bits,
                "seed {seed} {prim:?}"
            );
        }
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", &[2], Prim::F32);
    let rate = float_const(&mut dag, decl, Prim::F32, 0.3);
    let key = load(&mut dag, decl, "k", &[], Prim::Key);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, key],
        &[2],
        Prim::F32,
    );
    dag.add_root(drawn);
    let error = eval_tensor_roots_exact(&dag, dag.roots(), |name| match name {
        "x" => Some(floats(Prim::F32, vec![2], vec![1.0, 2.0])),
        _ => Some(TensorValue::scalar(7.0)),
    })
    .unwrap_err();
    assert!(error.contains("no raw numeric ingress"), "{error}");
}

#[test]
fn a_negative_runtime_split_count_traps_and_zero_is_empty() {
    for (count, ok) in [(-1i64, false), (0, true), (2, true)] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let seed = i64_const(&mut dag, decl, 7);
        let root = node(
            &mut dag,
            decl,
            RiscOp::KeyFromSeed,
            vec![seed],
            &[],
            Prim::Key,
        );
        let n = load(&mut dag, decl, "n", &[], Prim::Int64);
        let rows = dag.add_node(
            decl,
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
        assert_eq!(verify(&dag), Vec::<String>::new());
        let result = eval_tensor_roots_exact(&dag, dag.roots(), |_| {
            Some(TensorValue::from_storage(
                vec![],
                finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![count])).unwrap(),
            ))
        });
        match ok {
            false => assert_eq!(
                result.unwrap_err(),
                "numeric trap: domain in split_keys at i64",
                "count {count}"
            ),
            true => {
                let out = result.unwrap();
                assert_eq!(out[&rows].shape, vec![count as usize]);
                // key_ref.py: split_n(key(7), 3) begins 25ea33e6..., 707124fb...
                let expected = [0x25ea_33e6_1c10_576f, 0x7071_24fb_ecd5_f054];
                assert_eq!(key_bits(&out[&rows]), expected[..count as usize]);
            }
        }
    }
}

/// `split_keys(key(7), count)` whose count axis is declared `n`, the axis
/// the input `d` also binds, and, when `draw`, a dropout of `d` batched by
/// those keys. `[05-OP-71]`'s count is the extent; `n` is a claim about it.
fn split_declaring_n(count: RtDim, draw: bool) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seed = i64_const(&mut dag, decl, 7);
    let root = node(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        &[],
        Prim::Key,
    );
    let mut inputs = vec![root];
    if matches!(count, RtDim::Node(_)) {
        inputs.push(load(&mut dag, decl, "cnt", &[], Prim::Int64));
    }
    let n = || DimInfo::Named("n".into(), None);
    let rows = dag.add_node(
        decl,
        RiscOp::SplitN { count },
        inputs,
        TensorType {
            dims: vec![n()],
            precision: Prim::Key,
        },
        None,
    );
    let d = dag.add_node(
        decl,
        RiscOp::Load { name: "d".into() },
        vec![],
        TensorType {
            dims: vec![n(), DimInfo::Lit(2)],
            precision: Prim::F32,
        },
        None,
    );
    let out = if draw {
        let rate = float_const(&mut dag, decl, Prim::F32, 0.5);
        let drawn = dag.add_node(
            decl,
            RiscOp::Dropout,
            vec![d, rate, rows],
            TensorType {
                dims: vec![n(), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(drawn);
        drawn
    } else {
        dag.add_root(rows);
        dag.add_root(d);
        rows
    };
    (dag, out)
}

fn eval_split(dag: &Dag, count: i64) -> Result<UnordMap<NodeId, TensorValue>, String> {
    assert_eq!(verify(dag), Vec::<String>::new());
    let inputs = UnordMap::from_iter([
        (
            "cnt",
            TensorValue::from_storage(
                vec![],
                finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![count])).unwrap(),
            ),
        ),
        ("d", floats(Prim::F32, vec![4, 2], vec![1.0; 8])),
    ]);
    eval_tensor_roots_exact(dag, dag.roots(), |name| inputs.get(name).cloned())
}

/// A count that disagrees with its declared, elsewhere-bound axis traps
/// before any key exists, with the `Domain` trap a negative count raises,
/// and a batched draw keyed by those keys never runs. The count that agrees
/// gives key_ref.py's `split_n(key(7), 4)`.
#[test]
fn a_split_count_that_disagrees_with_its_declared_extent_traps() {
    let trap = "numeric trap: domain in split_keys at i64";
    for (count, draw) in [(3, false), (6, false), (40, false), (3, true)] {
        let (dag, _) = split_declaring_n(RtDim::Node(1), draw);
        let error = eval_split(&dag, count).unwrap_err();
        assert!(error.ends_with(trap), "count {count}: {error}");
        assert!(
            error.contains(&format!(
                "extent `n`: claimed = 4, split_keys axis 0 = {count}"
            )),
            "count {count}: {error}"
        );
    }
    let (dag, rows) = split_declaring_n(RtDim::Node(1), false);
    assert_eq!(
        key_bits(&eval_split(&dag, 4).unwrap()[&rows]),
        [
            0x25ea_33e6_1c10_576f,
            0x7071_24fb_ecd5_f054,
            0x8239_3615_3a56_5205,
            0x53c6_f7e8_3810_b049
        ]
    );
    // A literal count is checked the same way against a runtime-bound axis.
    let (dag, _) = split_declaring_n(RtDim::Lit(6), false);
    let error = eval_split(&dag, 0).unwrap_err();
    assert!(
        error.contains("extent `n`: claimed = 4, split_keys axis 0 = 6") && error.ends_with(trap),
        "{error}"
    );
    let (dag, rows) = split_declaring_n(RtDim::Lit(4), false);
    assert_eq!(eval_split(&dag, 0).unwrap()[&rows].shape, vec![4]);
}

/// The key's axes are the leading extents of a split's result. The verifier
/// requires the result to declare the key's own dims, so a leading axis
/// named for another extent is rejected; the evaluator, handed that graph
/// unverified, still checks the declared axis against the key before any key
/// exists.
#[test]
fn a_split_whose_declared_leading_axis_disagrees_with_its_key_traps() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = dag.add_node(
        decl,
        RiscOp::Load { name: "k".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("a".into(), None)],
            precision: Prim::Key,
        },
        None,
    );
    let rows = dag.add_node(
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(2),
        },
        vec![key],
        TensorType {
            dims: vec![DimInfo::Named("b".into(), None), DimInfo::Lit(2)],
            precision: Prim::Key,
        },
        None,
    );
    let e = dag.add_node(
        decl,
        RiscOp::Load { name: "e".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("b".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(rows);
    dag.add_root(e);
    assert_rejected(&dag, "with a split's count axis appended last");
    let one = RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, 1).unwrap()).unwrap();
    let run = |b: usize| {
        let inputs = UnordMap::from_iter([
            ("k", keys_value(vec![3], vec![one; 3])),
            ("e", floats(Prim::F32, vec![b], vec![0.0; b])),
        ]);
        eval_tensor_roots_exact(&dag, dag.roots(), |name| inputs.get(name).cloned())
    };
    let error = run(2).unwrap_err();
    assert!(
        error.contains("extent `b`: claimed = 2, split_keys axis 0 = 3")
            && error.ends_with("numeric trap: domain in split_keys at i64"),
        "{error}"
    );
    assert_eq!(run(3).unwrap()[&rows].shape, vec![3, 2]);
}

// ---- trap liveness: spec/06 §5.2, "purity alone does not make a possible
// trap dead" ----

/// A key-sourced draw validates its own rate, so a discarded one with an
/// invalid rate still traps. With a valid literal rate it cannot trap
/// (decisions section 6 item 5), so it is dead code the verifier reports as
/// dangling, and the same graph succeeds.
#[test]
fn a_discarded_key_sourced_draw_with_an_invalid_rate_still_traps() {
    for (rate, traps) in [(1.5, true), (0.5, false)] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let key = root_key(&mut dag, decl);
        let x = load(&mut dag, decl, "x", &[4], Prim::F32);
        let bad = float_const(&mut dag, decl, Prim::F32, rate);
        let draw = node(
            &mut dag,
            decl,
            RiscOp::Dropout,
            vec![x, bad, key],
            &[4],
            Prim::F32,
        );
        let out = node(&mut dag, decl, RiscOp::Neg, vec![x], &[4], Prim::F32);
        dag.add_root(out);
        let dangling = (!traps).then(|| {
            format!(
                "node {} is dangling: it has no consumers and is not a DAG root",
                draw.0
            )
        });
        assert_eq!(verify(&dag), Vec::from_iter(dangling));
        let result = eval_tensor_roots_exact(&dag, &[out], |_| {
            Some(floats(Prim::F32, vec![4], vec![1.0, 2.0, 3.0, 4.0]))
        });
        assert_eq!(result.is_err(), traps, "rate {rate}: {result:?}");
        if traps {
            assert!(
                result.unwrap_err().contains("numeric trap: domain"),
                "rate {rate}"
            );
        }
    }
}

/// Dead-code elimination keeps exactly the random nodes that can trap by
/// themselves: a draw with a runtime rate and a runtime-count `SplitN`. A
/// literal-count `SplitN` and a `FoldIn` cannot trap, so both are removed
/// when dead. The verifier draws the same line: only those two are dangling.
#[test]
fn dead_code_elimination_keeps_only_the_random_nodes_that_can_trap() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", &[4], Prim::F32);
    let rate = load(&mut dag, decl, "rate", &[], Prim::F32);
    let keyed = root_key(&mut dag, decl);
    node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, keyed],
        &[4],
        Prim::F32,
    );
    let n = load(&mut dag, decl, "n", &[], Prim::Int64);
    let counted = root_key(&mut dag, decl);
    dag.add_node(
        decl,
        RiscOp::SplitN {
            count: RtDim::Node(1),
        },
        vec![counted, n],
        TensorType {
            dims: vec![DimInfo::Named("keys".into(), None)],
            precision: Prim::Key,
        },
        None,
    );
    let literal = root_key(&mut dag, decl);
    let literal_split = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(2),
        },
        vec![literal],
        &[2],
        Prim::Key,
    );
    let folded = root_key(&mut dag, decl);
    let three = i64_const(&mut dag, decl, 3);
    let fold = node(
        &mut dag,
        decl,
        RiscOp::FoldIn,
        vec![folded, three],
        &[],
        Prim::Key,
    );
    let out = node(&mut dag, decl, RiscOp::Neg, vec![x], &[4], Prim::F32);
    dag.add_root(out);
    let dangling = |id: NodeId| {
        format!(
            "node {} is dangling: it has no consumers and is not a DAG root",
            id.0
        )
    };
    assert_eq!(verify(&dag), vec![dangling(literal_split), dangling(fold)]);

    let kept = dead_code_eliminate(&dag);
    assert_eq!(verify(&kept), Vec::<String>::new());
    let count =
        |matches: fn(&RiscOp) -> bool| kept.nodes().iter().filter(|n| matches(&n.op)).count();
    assert_eq!(count(|op| matches!(op, RiscOp::Dropout)), 1);
    assert_eq!(
        count(|op| matches!(
            op,
            RiscOp::SplitN {
                count: RtDim::Node(_)
            }
        )),
        1
    );
    assert_eq!(
        count(|op| matches!(
            op,
            RiscOp::SplitN {
                count: RtDim::Lit(_)
            }
        )),
        0
    );
    assert_eq!(count(|op| matches!(op, RiscOp::FoldIn)), 0);
    let draw = kept
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Dropout))
        .unwrap();
    assert!(matches!(
        kept.nodes()[draw.inputs[2].0].op,
        RiscOp::KeyFromSeed
    ));
}

/// The gradient pruner seeds the same random nodes: a discarded key-sourced
/// draw with an invalid rate survives `grad` and still traps there.
#[test]
fn grad_keeps_a_discarded_key_sourced_draw_that_can_trap() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let x = load(&mut dag, decl, "x", &[4], prim);
    let bad = float_const(&mut dag, decl, prim, 1.5);
    node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, bad, key],
        &[4],
        prim,
    );
    let total = node(
        &mut dag,
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: prim,
        },
        vec![x],
        &[],
        prim,
    );
    dag.add_root(total);
    let grad = grad_dag_checked(&dag, total, &[x]).unwrap();
    assert!(
        grad.dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Dropout)),
        "the discarded draw was pruned"
    );
    let gradient = grad.grad_nodes[&x];
    let result = eval_tensor_roots_exact(&grad.dag, &[gradient], |_| {
        Some(floats(prim, vec![4], vec![1.0, 2.0, 3.0, 4.0]))
    });
    assert!(
        result.unwrap_err().contains("numeric trap: domain"),
        "the retained draw must trap"
    );
}

// ---- oracle (d): the verifier's key rules ----

fn assert_rejected(dag: &Dag, needle: &str) {
    let errors = verify(dag);
    assert!(
        errors.iter().any(|error| error.contains(needle)),
        "expected `{needle}` in {errors:?}"
    );
}

fn root_key(dag: &mut Dag, decl: chelis_ir::dag::DeclId) -> NodeId {
    let seed = i64_const(dag, decl, 7);
    node(dag, decl, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key)
}

fn draw(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    key: NodeId,
    active: Option<NodeId>,
) -> NodeId {
    let x = load(dag, decl, "x", &[4], Prim::F32);
    let rate = float_const(dag, decl, Prim::F32, 0.5);
    dag.add_node(
        Owner::new(decl, active),
        RiscOp::Dropout,
        vec![x, rate, key],
        ty(&[4], Prim::F32),
        None,
    )
}

fn split(dag: &mut Dag, decl: chelis_ir::dag::DeclId, key: NodeId, branch: KeyBranch) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Split { branch },
        vec![key],
        &[],
        Prim::Key,
    )
}

#[test]
fn a_key_consumed_twice_is_rejected_for_every_consumer_kind() {
    // Two draws.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    for _ in 0..2 {
        let drawn = draw(&mut dag, decl, key, None);
        dag.add_root(drawn);
    }
    assert_rejected(&dag, "is consumed twice");
    // A fold and a split-n.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let three = i64_const(&mut dag, decl, 3);
    let folded = node(
        &mut dag,
        decl,
        RiscOp::FoldIn,
        vec![key, three],
        &[],
        Prim::Key,
    );
    let rows = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(2),
        },
        vec![key],
        &[2],
        Prim::Key,
    );
    dag.add_root(folded);
    dag.add_root(rows);
    assert_rejected(&dag, "is consumed twice");
    // A split and a fold on one parent.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let left = split(&mut dag, decl, key, KeyBranch::Left);
    let three = i64_const(&mut dag, decl, 3);
    let folded = node(
        &mut dag,
        decl,
        RiscOp::FoldIn,
        vec![key, three],
        &[],
        Prim::Key,
    );
    dag.add_root(left);
    dag.add_root(folded);
    assert_rejected(&dag, "is consumed twice");
    // Two lefts.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let first = split(&mut dag, decl, key, KeyBranch::Left);
    let second = split(&mut dag, decl, key, KeyBranch::Left);
    dag.add_root(first);
    dag.add_root(second);
    assert_rejected(&dag, "split twice for one branch");
    // A split-n key reused by a draw after its derivation (consumed twice).
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let rows = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(4),
        },
        vec![key],
        &[4],
        Prim::Key,
    );
    let drawn = draw(&mut dag, decl, key, None);
    dag.add_root(rows);
    dag.add_root(drawn);
    assert_rejected(&dag, "is consumed twice");
}

#[test]
fn one_split_per_branch_a_key_root_and_a_key_load_are_accepted() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let left = split(&mut dag, decl, key, KeyBranch::Left);
    let right = split(&mut dag, decl, key, KeyBranch::Right);
    let drawn = draw(&mut dag, decl, left, None);
    dag.add_root(drawn);
    dag.add_root(right);
    assert_eq!(verify(&dag), Vec::<String>::new());
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let loaded = load(&mut dag, decl, "k", &[], Prim::Key);
    let drawn = draw(&mut dag, decl, loaded, None);
    dag.add_root(drawn);
    assert_eq!(verify(&dag), Vec::<String>::new());
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let loaded = load(&mut dag, decl, "k", &[], Prim::Key);
    dag.add_root(loaded);
    assert_eq!(verify(&dag), Vec::<String>::new());
}

/// V2 counts uses of a key, not consumers of a node: returning a key is a
/// use, and two `Load`s of one parameter are one key.
#[test]
fn a_root_and_every_load_of_one_parameter_count_as_uses_of_one_key() {
    // A loaded key that is returned and drawn with.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let loaded = load(&mut dag, decl, "k", &[], Prim::Key);
    let drawn = draw(&mut dag, decl, loaded, None);
    dag.add_root(drawn);
    dag.add_root(loaded);
    assert_rejected(&dag, "is a graph root and is also consumed");
    // A derived key that is returned and split.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let left = split(&mut dag, decl, key, KeyBranch::Left);
    dag.add_root(key);
    dag.add_root(left);
    assert_rejected(&dag, "is a graph root and is also consumed");
    // A key returned twice.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    dag.set_roots(vec![key, key]);
    assert_rejected(&dag, "is a graph root twice");
    // Two loads of one parameter, each drawn with.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    for _ in 0..2 {
        let loaded = load(&mut dag, decl, "k", &[], Prim::Key);
        let drawn = draw(&mut dag, decl, loaded, None);
        dag.add_root(drawn);
    }
    assert_rejected(&dag, "is consumed twice");
    // Two loads of one parameter, one drawn with and one returned.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let first = load(&mut dag, decl, "k", &[], Prim::Key);
    let second = load(&mut dag, decl, "k", &[], Prim::Key);
    let drawn = draw(&mut dag, decl, first, None);
    dag.add_root(drawn);
    dag.add_root(second);
    assert_rejected(&dag, "is a graph root and is also consumed");
    // Loads of two parameters are two keys.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    for name in ["k", "j"] {
        let loaded = load(&mut dag, decl, name, &[], Prim::Key);
        let drawn = draw(&mut dag, decl, loaded, None);
        dag.add_root(drawn);
    }
    assert_eq!(verify(&dag), Vec::<String>::new());
}

#[test]
fn a_key_reaching_arithmetic_selection_or_a_shape_dependency_is_rejected() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let other = root_key(&mut dag, decl);
    let added = node(
        &mut dag,
        decl,
        RiscOp::Add,
        vec![key, other],
        &[],
        Prim::Key,
    );
    dag.add_root(added);
    assert_rejected(
        &dag,
        "only a key operation, a join or a random primitive consumes a key",
    );

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let other = root_key(&mut dag, decl);
    let condition = load(&mut dag, decl, "c", &[], Prim::Bool);
    let selected = node(
        &mut dag,
        decl,
        RiscOp::Where,
        vec![condition, key, other],
        &[],
        Prim::Key,
    );
    dag.add_root(selected);
    assert_rejected(
        &dag,
        "only a key operation, a join or a random primitive consumes a key",
    );
    // `key` is an active tensor element dtype, but `where` names no `key`,
    // so its own scheme rejects key branches too, independently of V4.
    assert_rejected(&dag, "branches must use an active data element dtype");

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let drawn = draw(&mut dag, decl, key, None);
    let x = load(&mut dag, decl, "y", &[4], Prim::F32);
    let negated = node(&mut dag, decl, RiscOp::Neg, vec![x], &[4], Prim::F32);
    dag.node_mut(negated).unwrap().shape_deps.push(key);
    dag.add_root(drawn);
    dag.add_root(negated);
    assert_rejected(&dag, "as a dependency");
}

#[test]
fn a_replay_must_read_its_own_forward_draws_key() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let other = root_key(&mut dag, decl);
    let forward = draw(&mut dag, decl, key, None);
    let other_forward = draw(&mut dag, decl, other, None);
    let g = load(&mut dag, decl, "g", &[4], Prim::F32);
    let rate = dag.get(forward).unwrap().inputs[1];
    // The replay reads `other`'s key while claiming `forward`'s rate; the
    // two rates are distinct constants, so the mask contract changes.
    let replay = node(
        &mut dag,
        decl,
        RiscOp::DropoutReplay,
        vec![g, rate, other],
        &[4],
        Prim::F32,
    );
    for root in [forward, other_forward, replay] {
        dag.add_root(root);
    }
    assert_rejected(&dag, "changes the mask contract of");
    // A replay of a key only a derivation consumed has no forward draw.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let left = split(&mut dag, decl, key, KeyBranch::Left);
    let g = load(&mut dag, decl, "g", &[4], Prim::F32);
    let rate = float_const(&mut dag, decl, Prim::F32, 0.5);
    let replay = node(
        &mut dag,
        decl,
        RiscOp::DropoutReplay,
        vec![g, rate, key],
        &[4],
        Prim::F32,
    );
    dag.add_root(left);
    dag.add_root(replay);
    assert_rejected(&dag, "which no forward random primitive consumes");
}

/// `lower_if`'s arm activations over the enclosing path `parent`.
fn arms(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    parent: Option<NodeId>,
) -> (NodeId, NodeId, NodeId) {
    let condition = load(dag, decl, "c", &[], Prim::Bool);
    let not = node(
        dag,
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![condition],
        &[],
        Prim::Bool,
    );
    match parent {
        None => (condition, condition, not),
        Some(parent) => {
            let then_arm = node(
                dag,
                decl,
                RiscOp::Logical(LogicalKind::And),
                vec![parent, condition],
                &[],
                Prim::Bool,
            );
            let else_arm = node(
                dag,
                decl,
                RiscOp::Logical(LogicalKind::And),
                vec![parent, not],
                &[],
                Prim::Bool,
            );
            (condition, then_arm, else_arm)
        }
    }
}

#[test]
fn rule_v3_admits_exclusive_arms_and_rejects_overlapping_ones() {
    // Top-level arms: X and Not(X).
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let (_, then_arm, else_arm) = arms(&mut dag, decl, None);
    let a = draw(&mut dag, decl, key, Some(then_arm));
    let b = draw(&mut dag, decl, key, Some(else_arm));
    dag.add_root(a);
    dag.add_root(b);
    assert_eq!(verify(&dag), Vec::<String>::new());

    // Nested: And(P, X) against And(P, Not X), and a third draw nested in
    // the then arm, And(And(P, X), Y), against the else arm.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let parent = load(&mut dag, decl, "p", &[], Prim::Bool);
    let (_, then_arm, else_arm) = arms(&mut dag, decl, Some(parent));
    let inner = load(&mut dag, decl, "y", &[], Prim::Bool);
    let nested = node(
        &mut dag,
        decl,
        RiscOp::Logical(LogicalKind::And),
        vec![then_arm, inner],
        &[],
        Prim::Bool,
    );
    let a = draw(&mut dag, decl, key, Some(nested));
    let b = draw(&mut dag, decl, key, Some(else_arm));
    dag.add_root(a);
    dag.add_root(b);
    assert_eq!(verify(&dag), Vec::<String>::new());

    // Two draws under the same arm, and under unrelated conditions, overlap.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let (_, then_arm, _) = arms(&mut dag, decl, None);
    let a = draw(&mut dag, decl, key, Some(then_arm));
    let b = draw(&mut dag, decl, key, Some(then_arm));
    dag.add_root(a);
    dag.add_root(b);
    assert_rejected(&dag, "whose activations are not exclusive");
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let first = load(&mut dag, decl, "c1", &[], Prim::Bool);
    let second = load(&mut dag, decl, "c2", &[], Prim::Bool);
    let not_second = node(
        &mut dag,
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![second],
        &[],
        Prim::Bool,
    );
    let a = draw(&mut dag, decl, key, Some(first));
    let b = draw(&mut dag, decl, key, Some(not_second));
    dag.add_root(a);
    dag.add_root(b);
    assert_rejected(&dag, "whose activations are not exclusive");
    // An activated draw beside an unactivated one is a double consume.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let (_, then_arm, _) = arms(&mut dag, decl, None);
    let a = draw(&mut dag, decl, key, Some(then_arm));
    let b = draw(&mut dag, decl, key, None);
    dag.add_root(a);
    dag.add_root(b);
    assert_rejected(&dag, "is consumed twice");
}

fn bool_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: bool) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Bool, i64::from(value)).unwrap(),
        },
        vec![],
        &[],
        Prim::Bool,
    )
}

/// Every root's stored bits, with `x` loaded as `[1, 2, 3, 4]` and the
/// parent condition `p` as true.
fn eval_roots(dag: &Dag) -> Vec<Vec<u64>> {
    let inputs = UnordMap::from_iter([
        ("x", floats(Prim::F32, vec![4], vec![1.0, 2.0, 3.0, 4.0])),
        (
            "p",
            TensorValue::from_storage(
                vec![],
                finalize_tensor("test", Prim::Bool, RawTensor::Int(vec![1])).unwrap(),
            ),
        ),
    ]);
    let out = eval_tensor_roots_exact(dag, dag.roots(), |n| inputs.get(n).cloned()).unwrap();
    dag.roots()
        .iter()
        .map(|root| stored_bits(&out[root]))
        .collect()
}

/// Rule V3 holds after constant folding. When the arms' condition `X` is a
/// constant, one of `X` and `Not(X)` folds to `false`, whose draw never
/// runs, so the folded pair stays exclusive and draws what it drew before.
#[test]
fn rule_v3_holds_after_folding_a_constant_condition() {
    for condition in [true, false] {
        for nested in [false, true] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let key = root_key(&mut dag, decl);
            let x = bool_const(&mut dag, decl, condition);
            let not = node(
                &mut dag,
                decl,
                RiscOp::Logical(LogicalKind::Not),
                vec![x],
                &[],
                Prim::Bool,
            );
            let (then_arm, else_arm) = if nested {
                let parent = load(&mut dag, decl, "p", &[], Prim::Bool);
                let mut and = |arm| {
                    node(
                        &mut dag,
                        decl,
                        RiscOp::Logical(LogicalKind::And),
                        vec![parent, arm],
                        &[],
                        Prim::Bool,
                    )
                };
                (and(x), and(not))
            } else {
                (x, not)
            };
            let a = draw(&mut dag, decl, key, Some(then_arm));
            let b = draw(&mut dag, decl, key, Some(else_arm));
            dag.add_root(a);
            dag.add_root(b);
            let case = format!("condition={condition} nested={nested}");
            assert_eq!(verify(&dag), Vec::<String>::new(), "{case}");
            let before = eval_roots(&dag);
            constant_fold(&mut dag);
            assert!(
                matches!(dag.get(not).unwrap().op, RiscOp::Const { .. }),
                "{case}: Not(X) folds"
            );
            assert_eq!(verify(&dag), Vec::<String>::new(), "{case} after folding");
            assert_eq!(eval_roots(&dag), before, "{case}");
            // The arm whose condition is false never draws.
            let zeros = vec![0u64; 4];
            assert_eq!(before[usize::from(condition)], zeros, "{case}");
            assert_ne!(before[usize::from(!condition)], zeros, "{case}");
        }
    }

    // Two true constants are not exclusive, and a false one does not excuse
    // a draw that carries no activation.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let first = bool_const(&mut dag, decl, true);
    let second = bool_const(&mut dag, decl, true);
    let a = draw(&mut dag, decl, key, Some(first));
    let b = draw(&mut dag, decl, key, Some(second));
    dag.add_root(a);
    dag.add_root(b);
    assert_rejected(&dag, "whose activations are not exclusive");
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let off = bool_const(&mut dag, decl, false);
    let a = draw(&mut dag, decl, key, Some(off));
    let b = draw(&mut dag, decl, key, None);
    dag.add_root(a);
    dag.add_root(b);
    assert_rejected(&dag, "is consumed twice");
}

#[test]
fn key_operation_operands_and_batched_shapes_are_checked() {
    // A key op fed an integer where a key belongs, and a seed of the wrong
    // integer width.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seed = node(
        &mut dag,
        decl,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Int32, 7).unwrap(),
        },
        vec![],
        &[],
        Prim::Int32,
    );
    let key = node(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        &[],
        Prim::Key,
    );
    dag.add_root(key);
    assert_rejected(&dag, "reads the wrong operand dtype");
    // fold_in with unequal shapes: no broadcasting.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3], Prim::Key);
    let n = i64_const(&mut dag, decl, 1);
    let folded = node(
        &mut dag,
        decl,
        RiscOp::FoldIn,
        vec![keys, n],
        &[3],
        Prim::Key,
    );
    dag.add_root(folded);
    assert_rejected(&dag, "element-wise over one exact shape");
    // split_keys whose declared count disagrees with its literal.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let rows = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![key],
        &[4],
        Prim::Key,
    );
    dag.add_root(rows);
    assert_rejected(&dag, "with a split's count axis appended last");
    // V5: a key batch whose shape is not the data's leading shape, and a
    // control whose shape is no leading part of the key's.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3], Prim::Key);
    let x = load(&mut dag, decl, "x", &[2, 4], Prim::F32);
    let rate = float_const(&mut dag, decl, Prim::F32, 0.5);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, keys],
        &[2, 4],
        Prim::F32,
    );
    dag.add_root(drawn);
    assert_rejected(
        &dag,
        "requires a key batch matching its data's leading axes",
    );
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3], Prim::Key);
    let x = load(&mut dag, decl, "x", &[3, 4], Prim::F32);
    let rates = load(&mut dag, decl, "r", &[4], Prim::F32);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rates, keys],
        &[3, 4],
        Prim::F32,
    );
    dag.add_root(drawn);
    assert_rejected(&dag, "random control must be a value of the draw's dtype");
    // A rank-2 key batch is its data's leading two axes, in order.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3, 2], Prim::Key);
    let x = load(&mut dag, decl, "x", &[2, 3], Prim::F32);
    let rate = float_const(&mut dag, decl, Prim::F32, 0.5);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, keys],
        &[2, 3],
        Prim::F32,
    );
    dag.add_root(drawn);
    assert_rejected(
        &dag,
        "requires a key batch matching its data's leading axes",
    );
    // A control shaped like the key's trailing axis, not a leading part.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3, 2], Prim::Key);
    let x = load(&mut dag, decl, "x", &[3, 2, 4], Prim::F32);
    let rates = load(&mut dag, decl, "r", &[2], Prim::F32);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rates, keys],
        &[3, 2, 4],
        Prim::F32,
    );
    dag.add_root(drawn);
    assert_rejected(&dag, "random control must be a value of the draw's dtype");
    // A bound adjoint's result shaped like no leading part of its key.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3, 2], Prim::Key);
    let t = load(&mut dag, decl, "t", &[3, 2, 4], Prim::F32);
    let g = load(&mut dag, decl, "g", &[3, 2, 4], Prim::F32);
    let low = float_const(&mut dag, decl, Prim::F32, 0.0);
    let high = float_const(&mut dag, decl, Prim::F32, 1.0);
    let forward = node(
        &mut dag,
        decl,
        RiscOp::UniformLike,
        vec![t, low, high, keys],
        &[3, 2, 4],
        Prim::F32,
    );
    let adjoint = node(
        &mut dag,
        decl,
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::High,
        },
        vec![t, g, keys],
        &[2],
        Prim::F32,
    );
    dag.add_root(forward);
    dag.add_root(adjoint);
    assert_rejected(&dag, "a leading part of its key's shape");
}

/// Dims of named axes followed by literal ones.
fn named_ty(names: &[&str], trailing: &[usize], prim: Prim) -> TensorType {
    TensorType {
        dims: names
            .iter()
            .map(|name| DimInfo::Named((*name).into(), None))
            .chain(trailing.iter().map(|extent| DimInfo::Lit(*extent)))
            .collect(),
        precision: prim,
    }
}

/// Round 3's witness: keys `split_n(key(7), 3)`, a template `t: [3, 4]`, and
/// a `UniformLike` of them declared `[rows, 4]`.
fn uniform_declaring_rows(rows: usize) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let keys = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![key],
        &[3],
        Prim::Key,
    );
    let t = load(&mut dag, decl, "t", &[3, 4], Prim::F32);
    let low = float_const(&mut dag, decl, Prim::F32, 0.0);
    let high = float_const(&mut dag, decl, Prim::F32, 1.0);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::UniformLike,
        vec![t, low, high, keys],
        &[rows, 4],
        Prim::F32,
    );
    dag.add_root(drawn);
    dag
}

/// A forward `UniformLike` over `t: [3, 5]` keyed by `split_n(key(7), 3)`,
/// and its high-bound adjoint over the cotangent `g` of type `cotangent`.
fn bound_adjoint_over(cotangent: TensorType) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = root_key(&mut dag, decl);
    let keys = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![key],
        &[3],
        Prim::Key,
    );
    let t = load(&mut dag, decl, "t", &[3, 5], Prim::F32);
    let low = float_const(&mut dag, decl, Prim::F32, 0.0);
    let high = float_const(&mut dag, decl, Prim::F32, 1.0);
    let forward = node(
        &mut dag,
        decl,
        RiscOp::UniformLike,
        vec![t, low, high, keys],
        &[3, 5],
        Prim::F32,
    );
    let g = dag.add_node(
        decl,
        RiscOp::Load { name: "g".into() },
        vec![],
        cotangent,
        None,
    );
    let adjoint = node(
        &mut dag,
        decl,
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::High,
        },
        vec![t, g, keys],
        &[],
        Prim::F32,
    );
    dag.add_root(forward);
    dag.add_root(adjoint);
    dag
}

/// spec/10 §3.2: a draw preserves its first input's exact shape, a bound
/// adjoint's cotangent has its template's exact type, and each key operation
/// preserves its operands' exact dims (a split appending its count). The
/// verifier checks them through `verify_random_operands`, which the wire
/// decoder shares, so a declared result or cotangent whose rank agrees but
/// whose dims differ is rejected, and the agreeing graphs verify and run.
///
/// Evidentiary status: REGRESSION TEST. At c23ec448a `verify` accepted every
/// rejected graph here: it compared ranks for the draw and the cotangent,
/// and, for the key operations, extents that might agree.
#[test]
fn a_declared_result_or_cotangent_unlike_its_operand_is_rejected() {
    let uniform = "uniform_like must preserve its float template's exact shape and dtype";
    for rows in [2, 5] {
        assert_rejected(&uniform_declaring_rows(rows), uniform);
    }
    let dag = uniform_declaring_rows(3);
    let t = floats(Prim::F32, vec![3, 4], vec![0.0; 12]);
    let out = run(&dag, &UnordMap::from_iter([("t", t)]));
    assert_eq!(out[&dag.roots()[0]].shape, vec![3, 4]);

    // The same disagreement between named axes: keys and template `[n]`,
    // result `[m]`.
    let named = |out: &str| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let keys = dag.add_node(
            decl,
            RiscOp::Load { name: "k".into() },
            vec![],
            named_ty(&["n"], &[], Prim::Key),
            None,
        );
        let t = dag.add_node(
            decl,
            RiscOp::Load { name: "t".into() },
            vec![],
            named_ty(&["n"], &[4], Prim::F32),
            None,
        );
        let low = float_const(&mut dag, decl, Prim::F32, 0.0);
        let high = float_const(&mut dag, decl, Prim::F32, 1.0);
        let drawn = dag.add_node(
            decl,
            RiscOp::UniformLike,
            vec![t, low, high, keys],
            named_ty(&[out], &[4], Prim::F32),
            None,
        );
        dag.add_root(drawn);
        dag
    };
    assert_rejected(&named("m"), uniform);
    assert_eq!(verify(&named("n")), Vec::<String>::new());

    // A key operation's declared result against its operands: `Split` and
    // `KeyFromSeed` from `[n]` to `[m]`, and `FoldIn` of `[n]` keys by `[n]`
    // indices to `[m]` or by `[m]` indices to `[n]`.
    let key_op = "key operation must be element-wise over one exact shape";
    let unary = |op: RiscOp, operand: Prim, out: &str| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "k".into() },
            vec![],
            named_ty(&["n"], &[], operand),
            None,
        );
        let result = dag.add_node(
            decl,
            op,
            vec![input],
            named_ty(&[out], &[], Prim::Key),
            None,
        );
        dag.add_root(result);
        dag
    };
    let split = || RiscOp::Split {
        branch: KeyBranch::Left,
    };
    assert_rejected(&unary(split(), Prim::Key, "m"), key_op);
    assert_eq!(
        verify(&unary(split(), Prim::Key, "n")),
        Vec::<String>::new()
    );
    assert_rejected(&unary(RiscOp::KeyFromSeed, Prim::Int64, "m"), key_op);
    assert_eq!(
        verify(&unary(RiscOp::KeyFromSeed, Prim::Int64, "n")),
        Vec::<String>::new()
    );
    let fold = |indices: &str, out: &str| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let keys = dag.add_node(
            decl,
            RiscOp::Load { name: "k".into() },
            vec![],
            named_ty(&["n"], &[], Prim::Key),
            None,
        );
        let ns = dag.add_node(
            decl,
            RiscOp::Load { name: "i".into() },
            vec![],
            named_ty(&[indices], &[], Prim::Int64),
            None,
        );
        let result = dag.add_node(
            decl,
            RiscOp::FoldIn,
            vec![keys, ns],
            named_ty(&[out], &[], Prim::Key),
            None,
        );
        dag.add_root(result);
        dag
    };
    assert_rejected(&fold("n", "m"), key_op);
    assert_rejected(&fold("m", "n"), key_op);
    assert_eq!(verify(&fold("n", "n")), Vec::<String>::new());

    // A bound adjoint's cotangent against its template `[3, 5]`: another
    // extent, or another dtype.
    let adjoint = "over a cotangent of its template's exact type";
    assert_rejected(&bound_adjoint_over(ty(&[3, 7], Prim::F32)), adjoint);
    assert_rejected(&bound_adjoint_over(ty(&[3, 5], Prim::F64)), adjoint);
    let dag = bound_adjoint_over(ty(&[3, 5], Prim::F32));
    let inputs = UnordMap::from_iter([
        ("t", floats(Prim::F32, vec![3, 5], vec![0.0; 15])),
        ("g", floats(Prim::F32, vec![3, 5], vec![1.0; 15])),
    ]);
    let out = run(&dag, &inputs);
    assert_eq!(out[&dag.roots()[1]].shape, Vec::<usize>::new());
}

// ---- V5 at every key rank: `vmap` composes over draws ----

/// `[05-OP-71]` twice: `split_keys(split_keys(key(seed), 2)[i], 3)`, the
/// `tensor[2, 3, key]` whose element `(i, j)` is key_ref.py's
/// `split_n(split_n(key(seed), 2)[i], 3)[j]`.
fn rank_two_keys(dag: &mut Dag, decl: chelis_ir::dag::DeclId, seed: i64) -> NodeId {
    let seed = i64_const(dag, decl, seed);
    let root = node(dag, decl, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key);
    let rows = node(
        dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(2),
        },
        vec![root],
        &[2],
        Prim::Key,
    );
    node(
        dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![rows],
        &[2, 3],
        Prim::Key,
    )
}

/// key_ref.py's `split_n(split_n(key(seed), 2)[i], 3)[j]` as key values.
fn rank_two_key_values(seed: i64) -> Vec<RandomKey> {
    RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, seed).unwrap())
        .unwrap()
        .split_n(2)
        .iter()
        .flat_map(|row| row.split_n(3))
        .collect()
}

/// key_ref_ext.py's `uniform01(K7[i][j], 4, "f32")`, row-major over `(i, j)`.
const RANK_TWO_UNIFORM_F32: [[u32; 4]; 6] = [
    [0x3f64_799a, 0x3ed8_0a73, 0x3dd4_668c, 0x3f7f_4030],
    [0x3f0f_6914, 0x3bfe_1697, 0x3f40_d9de, 0x3f5b_136f],
    [0x3f75_04eb, 0x3f4e_767a, 0x3dc5_4695, 0x3ea5_d879],
    [0x3f56_4547, 0x3f5a_3cbc, 0x3ebc_4ac1, 0x3e25_d5fd],
    [0x3f57_ebf1, 0x3e5a_cd04, 0x3e27_5c0a, 0x3ddd_e418],
    [0x3ef0_8b68, 0x3f5d_2d1f, 0x3f67_7f50, 0x3ee4_1732],
];

/// key_ref_ext.py's `dropout_half(K8[i][j], x)` over `x = 1..=24` row-major.
const RANK_TWO_DROPOUT_HALF: [[f64; 4]; 6] = [
    [2.0, 4.0, 6.0, 8.0],
    [0.0, 12.0, 0.0, 16.0],
    [0.0, 0.0, 0.0, 24.0],
    [0.0, 0.0, 30.0, 32.0],
    [0.0, 0.0, 38.0, 40.0],
    [0.0, 0.0, 46.0, 48.0],
];

fn rank_two_uniform_bits() -> Vec<u64> {
    RANK_TWO_UNIFORM_F32
        .iter()
        .flatten()
        .map(|bits| u64::from(*bits))
        .collect()
}

/// V5 at rank 2: row `(i, j)` of the data draws with key `(i, j)`, and a
/// control of the key's leading shape `[2]` gives every row `(i, *)` its
/// element `i`.
#[test]
fn a_rank_two_key_batch_draws_each_row_with_its_own_key() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let uniform_keys = rank_two_keys(&mut dag, decl, 7);
    let template = load(&mut dag, decl, "t", &[2, 3, 4], prim);
    let low = float_const(&mut dag, decl, prim, 0.0);
    let high = float_const(&mut dag, decl, prim, 1.0);
    let sampled = node(
        &mut dag,
        decl,
        RiscOp::UniformLike,
        vec![template, low, high, uniform_keys],
        &[2, 3, 4],
        prim,
    );
    let dropout_keys = rank_two_keys(&mut dag, decl, 8);
    let x = load(&mut dag, decl, "x", &[2, 3, 4], prim);
    let rates = load(&mut dag, decl, "rates", &[2], prim);
    let dropped = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rates, dropout_keys],
        &[2, 3, 4],
        prim,
    );
    dag.add_root(sampled);
    dag.add_root(dropped);
    let data: Vec<f64> = (1..=24).map(f64::from).collect();
    let inputs = UnordMap::from_iter([
        ("t", floats(prim, vec![2, 3, 4], vec![0.0; 24])),
        ("x", floats(prim, vec![2, 3, 4], data.clone())),
        ("rates", floats(prim, vec![2], vec![0.0, 0.5])),
    ]);
    let out = run(&dag, &inputs);
    assert_eq!(stored_bits(&out[&sampled]), rank_two_uniform_bits());
    // Rows (0, *) draw at rate 0 and keep x; rows (1, *) at 0.5 keep
    // key_ref_ext.py's pattern.
    let mut expected = data[..12].to_vec();
    expected.extend(RANK_TWO_DROPOUT_HALF[3..].iter().flatten());
    assert_eq!(out[&dropped].to_f64_lossy_vec(), expected);
}

/// `vmap` over a draw keyed by a scalar key, twice, gives a rank-2 key
/// batch that verifies and draws each row with its own key.
#[test]
fn vmap_of_vmap_of_a_draw_verifies_and_draws_each_row_with_its_key() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = load(&mut dag, decl, "k", &[], Prim::Key);
    let template = load(&mut dag, decl, "t", &[4], prim);
    let low = float_const(&mut dag, decl, prim, 0.0);
    let high = float_const(&mut dag, decl, prim, 1.0);
    let sampled = node(
        &mut dag,
        decl,
        RiscOp::UniformLike,
        vec![template, low, high, key],
        &[4],
        prim,
    );
    dag.add_root(sampled);
    let once = vectorize_axis0(&dag, DimInfo::Lit(3)).unwrap();
    let twice = vectorize_axis0(&once, DimInfo::Lit(2)).unwrap();
    assert_eq!(verify(&twice), Vec::<String>::new());
    let inputs = UnordMap::from_iter([
        ("k", keys_value(vec![2, 3], rank_two_key_values(7))),
        ("t", floats(prim, vec![2, 3, 4], vec![0.0; 24])),
    ]);
    let out = run(&twice, &inputs);
    let root = twice.roots()[0];
    assert_eq!(out[&root].shape, vec![2, 3, 4]);
    assert_eq!(stored_bits(&out[&root]), rank_two_uniform_bits());
}

/// `vmap` over a draw that is already batched keeps its rank-0 rate as one
/// value per mapped row: the key's leading shape.
#[test]
fn vmap_of_a_batched_draw_verifies_and_keeps_each_rows_rate() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let keys = load(&mut dag, decl, "k", &[3], Prim::Key);
    let x = load(&mut dag, decl, "x", &[3, 4], prim);
    let rate = float_const(&mut dag, decl, prim, 0.5);
    let dropped = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, keys],
        &[3, 4],
        prim,
    );
    dag.add_root(dropped);
    let batched = vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
    assert_eq!(verify(&batched), Vec::<String>::new());
    let inputs = UnordMap::from_iter([
        ("k", keys_value(vec![2, 3], rank_two_key_values(8))),
        (
            "x",
            floats(prim, vec![2, 3, 4], (1..=24).map(f64::from).collect()),
        ),
    ]);
    let out = run(&batched, &inputs);
    let expected: Vec<f64> = RANK_TWO_DROPOUT_HALF.iter().flatten().copied().collect();
    assert_eq!(out[&batched.roots()[0]].to_f64_lossy_vec(), expected);
}

/// A bound adjoint over a rank-2 key batch folds each group of rows that
/// shares one bound element: all rows for a rank-0 result, rows `(i, *)`
/// for a `[2]` result, and each row alone for a `[2, 3]` result, every group
/// in one canonical tree over its contributions in row-major order.
#[test]
fn a_rank_two_bound_adjoint_folds_each_group_of_rows_sharing_a_bound() {
    let prim = Prim::F32;
    let units: Vec<f32> = RANK_TWO_UNIFORM_F32
        .iter()
        .flatten()
        .map(|bits| f32::from_bits(*bits))
        .collect();
    let pair_sum = |values: &[f32]| -> f32 {
        let mut level = values.to_vec();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        pair[0] + pair[1]
                    } else {
                        pair[0]
                    }
                })
                .collect();
        }
        level.first().copied().unwrap_or(0.0)
    };
    for out_dims in [&[][..], &[2][..], &[2, 3][..]] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let keys = rank_two_keys(&mut dag, decl, 7);
        let template = load(&mut dag, decl, "t", &[2, 3, 4], prim);
        let g = load(&mut dag, decl, "g", &[2, 3, 4], prim);
        let (low, high) = if out_dims.is_empty() {
            (
                float_const(&mut dag, decl, prim, 0.0),
                float_const(&mut dag, decl, prim, 1.0),
            )
        } else {
            (
                load(&mut dag, decl, "lo", out_dims, prim),
                load(&mut dag, decl, "hi", out_dims, prim),
            )
        };
        let forward = node(
            &mut dag,
            decl,
            RiscOp::UniformLike,
            vec![template, low, high, keys],
            &[2, 3, 4],
            prim,
        );
        let adjoint = node(
            &mut dag,
            decl,
            RiscOp::UniformBoundAdjoint {
                bound: UniformBound::High,
            },
            vec![template, g, keys],
            out_dims,
            prim,
        );
        dag.add_root(forward);
        dag.add_root(adjoint);
        let groups: usize = out_dims.iter().product();
        let inputs = UnordMap::from_iter([
            ("t", floats(prim, vec![2, 3, 4], vec![0.0; 24])),
            ("g", floats(prim, vec![2, 3, 4], vec![1.0; 24])),
            ("lo", floats(prim, out_dims.to_vec(), vec![0.0; groups])),
            ("hi", floats(prim, out_dims.to_vec(), vec![1.0; groups])),
        ]);
        let out = run(&dag, &inputs);
        let expected: Vec<f64> = units
            .chunks(24 / groups)
            .map(|group| f64::from(pair_sum(group)))
            .collect();
        assert_eq!(out[&adjoint].shape, out_dims.to_vec());
        assert_eq!(out[&adjoint].to_f64_lossy_vec(), expected, "{out_dims:?}");
    }
}

#[test]
fn a_key_constant_is_rejected_and_folding_keeps_derivations_symbolic() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let constant = node(
        &mut dag,
        decl,
        RiscOp::Const {
            value: ScalarValue::from_key(
                RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, 7).unwrap()).unwrap(),
            ),
        },
        vec![],
        &[],
        Prim::Key,
    );
    let drawn = draw(&mut dag, decl, constant, None);
    dag.add_root(drawn);
    assert_rejected(&dag, "only a key operation, a join or a Load produces one");

    // Constant folding over literal seeds and indices leaves every key
    // operation in place, and CSE merges no two of them.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = root_key(&mut dag, decl);
    let b = root_key(&mut dag, decl);
    let three = i64_const(&mut dag, decl, 3);
    let folded = node(
        &mut dag,
        decl,
        RiscOp::FoldIn,
        vec![a, three],
        &[],
        Prim::Key,
    );
    let x = draw(&mut dag, decl, folded, None);
    let y = draw(&mut dag, decl, b, None);
    dag.add_root(x);
    dag.add_root(y);
    constant_fold(&mut dag);
    let dag = common_subexpr_eliminate(&dag);
    assert_eq!(verify(&dag), Vec::<String>::new());
    let count = |op: fn(&RiscOp) -> bool| dag.nodes().iter().filter(|node| op(&node.op)).count();
    assert_eq!(count(|op| matches!(op, RiscOp::KeyFromSeed)), 2);
    assert_eq!(count(|op| matches!(op, RiscOp::FoldIn)), 1);
    assert!(
        dag.nodes()
            .iter()
            .all(|node| node.output_type.precision != Prim::Key
                || !matches!(node.op, RiscOp::Const { .. })),
        "no key constant after folding"
    );
}

// ---- grad and vmap ----

#[test]
fn a_key_takes_no_cotangent_and_grad_replays_the_forward_key() {
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let chain = build_chain(&mut dag, decl);
    let x = load(&mut dag, decl, "x", &[4], prim);
    let rate = float_const(&mut dag, decl, prim, 0.5);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, chain.g],
        &[4],
        prim,
    );
    let axis_sum = node(
        &mut dag,
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: prim,
        },
        vec![drawn],
        &[],
        prim,
    );
    dag.add_root(axis_sum);
    dag.add_root(chain.rows);
    let grad = grad_dag_checked(&dag, axis_sum, &[x]).unwrap();
    let grads = grad.dag;
    assert_eq!(verify(&grads), Vec::<String>::new());
    let gradient = grad.grad_nodes[&x];
    let row = vec![1.0, 2.0, 3.0, 4.0];
    let out = eval_tensor_roots_exact(&grads, &[gradient], |_| {
        Some(floats(prim, vec![4], row.clone()))
    })
    .unwrap();
    // d/dx sum(dropout(x)) keeps G's pattern with scale 1/(1-0.5) = 2.
    let expected: Vec<f64> = DROPOUT_KEPT[3]
        .iter()
        .map(|keep| if *keep { 2.0 } else { 0.0 })
        .collect();
    assert_eq!(out[&gradient].to_f64_lossy_vec(), expected);
}

/// A runtime-count split declares its count axis `n`, which the data it
/// batches also binds, but its value is a key, and no node takes a key as a
/// dependency (V4). `grad`'s runtime-extent pass records none on it, whether
/// the count is a parameter or read from the data with `shape(x, 0)`, and the
/// backward graph verifies with the split still in it.
#[test]
fn grad_records_no_dependency_on_a_runtime_count_split() {
    for count_from_data in [false, true] {
        let prim = Prim::F32;
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            named_ty(&["n"], &[4], prim),
            None,
        );
        let count = if count_from_data {
            node(
                &mut dag,
                decl,
                RiscOp::Shape { axis: 0 },
                vec![x],
                &[],
                Prim::Int64,
            )
        } else {
            load(&mut dag, decl, "c", &[], Prim::Int64)
        };
        let seed = i64_const(&mut dag, decl, 11);
        let root = node(
            &mut dag,
            decl,
            RiscOp::KeyFromSeed,
            vec![seed],
            &[],
            Prim::Key,
        );
        let keys = dag.add_node(
            decl,
            RiscOp::SplitN {
                count: RtDim::Node(1),
            },
            vec![root, count],
            named_ty(&["n"], &[], Prim::Key),
            None,
        );
        let rate = float_const(&mut dag, decl, prim, 0.5);
        let drawn = dag.add_node(
            decl,
            RiscOp::Dropout,
            vec![x, rate, keys],
            named_ty(&["n"], &[4], prim),
            None,
        );
        let rows = dag.add_node(
            decl,
            RiscOp::sum_default(1, prim).unwrap(),
            vec![drawn],
            named_ty(&["n"], &[], prim),
            None,
        );
        let total = node(
            &mut dag,
            decl,
            RiscOp::sum_default(0, prim).unwrap(),
            vec![rows],
            &[],
            prim,
        );
        dag.add_root(total);
        assert_eq!(verify(&dag), Vec::<String>::new());

        let mut recorded = dag.clone();
        record_runtime_dim_shape_deps(&mut recorded);
        assert_eq!(
            verify(&recorded),
            Vec::<String>::new(),
            "count from data: {count_from_data}"
        );
        assert!(
            recorded
                .nodes()
                .iter()
                .all(|node| !node.shape_deps.contains(&keys)),
            "count from data: {count_from_data}"
        );

        let grad = grad_dag_checked(&dag, total, &[x])
            .unwrap_or_else(|error| panic!("count from data {count_from_data}: {error:?}"));
        assert_eq!(verify(&grad.dag), Vec::<String>::new());
        assert!(
            grad.dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::SplitN { .. }))
        );
    }
}

#[test]
fn vmap_maps_key_rows_and_splits_every_row() {
    // f(k, x) = dropout(x, 0.5, fold_in(k, 2)) and g(k) = split_keys(k, 2),
    // mapped over a tensor[3, key] of S's rows.
    let prim = Prim::F32;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = load(&mut dag, decl, "k", &[], Prim::Key);
    let two = i64_const(&mut dag, decl, 2);
    let folded = node(
        &mut dag,
        decl,
        RiscOp::FoldIn,
        vec![key, two],
        &[],
        Prim::Key,
    );
    let x = load(&mut dag, decl, "x", &[4], prim);
    let rate = float_const(&mut dag, decl, prim, 0.5);
    let drawn = node(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![x, rate, folded],
        &[4],
        prim,
    );
    let other = load(&mut dag, decl, "k2", &[], Prim::Key);
    let rows = node(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(2),
        },
        vec![other],
        &[2],
        Prim::Key,
    );
    dag.add_root(drawn);
    dag.add_root(rows);
    let batched = vectorize_axis0(&dag, DimInfo::Lit(3)).unwrap();
    assert_eq!(verify(&batched), Vec::<String>::new());
    // S's rows as real key values: a key has no literal, so derive them.
    let s = RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, -3).unwrap())
        .unwrap()
        .split()
        .0
        .fold_in(scalar_from_i64("test", Prim::Int64, -5).unwrap())
        .unwrap()
        .split_n(3);
    assert_eq!(s.iter().map(|key| key.bits()).collect::<Vec<_>>(), S_KEYS);
    let row = [1.0, 2.0, 3.0, 4.0];
    let inputs = UnordMap::from_iter([
        ("k", keys_value(vec![3], s.clone())),
        ("k2", keys_value(vec![3], s.clone())),
        ("x", floats(prim, vec![3, 4], row.repeat(3))),
    ]);
    let out = eval_tensor_roots_exact(&batched, batched.roots(), |name| inputs.get(name).cloned())
        .unwrap();
    let drawn_rows = out[&batched.roots()[0]].to_f64_lossy_vec();
    let split_rows = &out[&batched.roots()[1]];
    assert_eq!(split_rows.shape, vec![3, 2]);
    for (b, key) in s.iter().enumerate() {
        // Row b of the batched draw is the scalar draw keyed by fold_in(S[b], 2).
        let mut scalar = Dag::new();
        let scalar_decl = scalar.declare("test");
        let k = load(&mut scalar, scalar_decl, "k", &[], Prim::Key);
        let two = i64_const(&mut scalar, scalar_decl, 2);
        let folded = node(
            &mut scalar,
            scalar_decl,
            RiscOp::FoldIn,
            vec![k, two],
            &[],
            Prim::Key,
        );
        let x = load(&mut scalar, scalar_decl, "x", &[4], prim);
        let rate = float_const(&mut scalar, scalar_decl, prim, 0.5);
        let drawn = node(
            &mut scalar,
            scalar_decl,
            RiscOp::Dropout,
            vec![x, rate, folded],
            &[4],
            prim,
        );
        scalar.add_root(drawn);
        let inputs = UnordMap::from_iter([
            ("k", keys_value(vec![], vec![*key])),
            ("x", floats(prim, vec![4], row.to_vec())),
        ]);
        let expected = run(&scalar, &inputs)[&drawn].to_f64_lossy_vec();
        assert_eq!(
            &drawn_rows[b * 4..b * 4 + 4],
            expected.as_slice(),
            "row {b}"
        );
        let rows_b: Vec<u64> = key.split_n(2).iter().map(|key| key.bits()).collect();
        assert_eq!(&key_bits(split_rows)[b * 2..b * 2 + 2], rows_b.as_slice());
    }
}

#[test]
fn vmap_refuses_to_broadcast_a_captured_key() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = load(&mut dag, decl, "k", &[], Prim::Key);
    let drawn = draw(&mut dag, decl, key, None);
    dag.add_root(drawn);
    let captured = chelis_unord::UnordSet::from_iter(["k".to_string()]);
    let error = chelis_ir::vmap::vectorize_axis0_with_captures(&dag, DimInfo::Lit(3), &captured)
        .unwrap_err();
    assert!(error.contains("captured key"), "{error}");
}

#[test]
fn a_batched_uniform_bound_adjoint_sums_shared_bounds_and_splits_per_row_bounds() {
    let prim = Prim::F32;
    for per_row in [false, true] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let chain = build_chain(&mut dag, decl);
        let template = load(&mut dag, decl, "t", &[3, 4], prim);
        let g = load(&mut dag, decl, "g", &[3, 4], prim);
        let (low, high) = if per_row {
            (
                load(&mut dag, decl, "lo", &[3], prim),
                load(&mut dag, decl, "hi", &[3], prim),
            )
        } else {
            (
                float_const(&mut dag, decl, prim, 0.0),
                float_const(&mut dag, decl, prim, 1.0),
            )
        };
        let forward = node(
            &mut dag,
            decl,
            RiscOp::UniformLike,
            vec![template, low, high, chain.rows],
            &[3, 4],
            prim,
        );
        let out_dims: &[usize] = if per_row { &[3] } else { &[] };
        let adjoint = node(
            &mut dag,
            decl,
            RiscOp::UniformBoundAdjoint {
                bound: UniformBound::High,
            },
            vec![template, g, chain.rows],
            out_dims,
            prim,
        );
        dag.add_root(forward);
        dag.add_root(adjoint);
        dag.add_root(chain.g);
        let inputs = UnordMap::from_iter([
            ("t", floats(prim, vec![3, 4], vec![0.0; 12])),
            ("g", floats(prim, vec![3, 4], vec![1.0; 12])),
            ("lo", floats(prim, vec![3], vec![0.0; 3])),
            ("hi", floats(prim, vec![3], vec![1.0; 3])),
        ]);
        let out = run(&dag, &inputs);
        // With g = 1 the high adjoint of each row is the f32 balanced sum of
        // its units, which uniform(0, 1) stored exactly (f32 bits above).
        let units = uniform01_bits(prim);
        let row_units = |row: usize| -> Vec<f32> {
            units[row]
                .iter()
                .map(|bits| f32::from_bits(*bits as u32))
                .collect()
        };
        let pair_sum = |values: &[f32]| -> f32 {
            let mut level = values.to_vec();
            while level.len() > 1 {
                level = level
                    .chunks(2)
                    .map(|pair| {
                        if pair.len() == 2 {
                            pair[0] + pair[1]
                        } else {
                            pair[0]
                        }
                    })
                    .collect();
            }
            level.first().copied().unwrap_or(0.0)
        };
        let actual = out[&adjoint].to_f64_lossy_vec();
        if per_row {
            let expected: Vec<f64> = (0..3)
                .map(|row| f64::from(pair_sum(&row_units(row))))
                .collect();
            assert_eq!(actual, expected);
        } else {
            let all: Vec<f32> = (0..3).flat_map(row_units).collect();
            assert_eq!(actual, vec![f64::from(pair_sum(&all))]);
        }
    }
}

/// `dropout(key_from_seed(42i64), x, rate)` over `x = [1; count]` through
/// lowering and the DAG evaluator.
fn lowered_dropout(rate: f64, count: usize) -> Result<Vec<f64>, String> {
    let source = format!(
        "(app {{}} (var {{}} dropout) \
         (app {{}} (var {{}} key_from_seed) (lit {{type: (t-prim {{}} i64)}} 42)) \
         (var {{}} x) (lit {{type: (t-prim {{}} f32)}} {rate:?}))"
    );
    let expression = chelis_deep::parser::parse_str(&source)
        .unwrap()
        .pop()
        .unwrap();
    let inputs = UnordMap::from_iter([(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Lit(count)],
            precision: Prim::F32,
        },
    )]);
    let dag = chelis_ir::lower::try_lower_subexpr_program(
        &expression,
        inputs,
        UnordMap::new(),
        UnordMap::new(),
    )
    .unwrap();
    let values = eval_tensor_roots_exact(&dag, dag.roots(), |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![count], vec![1.0; count]))
    })?;
    Ok(values[&dag.roots()[0]].to_f64_lossy_vec())
}

/// [05-OP-37]: a zero rate keeps every element, at empty and nonempty
/// shapes.
#[test]
fn zero_rate_dropout_is_an_identity_at_empty_and_nonempty_shapes() {
    for count in [0, 1, 32] {
        assert_eq!(lowered_dropout(0.0, count).unwrap(), vec![1.0; count]);
    }
}

/// [05-OP-37]: a rate outside `[0, 1)` traps even when the tensor is empty.
#[test]
fn invalid_dropout_rates_trap_even_when_the_tensor_is_empty() {
    for rate in [-0.5, 1.0, 2.0] {
        for count in [0, 1, 32] {
            let error = lowered_dropout(rate, count).unwrap_err();
            assert_eq!(
                error, "numeric trap: domain in dropout at f32",
                "rate={rate:?}, count={count}"
            );
        }
    }
}
