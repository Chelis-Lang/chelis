//! Incremental `eval-performance` oracle: sparse runtime indices [05-SPARSE-1].
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Allocator;
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, size) };
        if !result.is_null() {
            BYTES.fetch_add(size, Ordering::Relaxed);
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn ty(n: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision,
    }
}

fn shaped_ty(shape: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

#[test]
fn eval_performance_streamed_contraction_bits_and_scratch() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::tier2::lower_matmul_with_accumulator;
    for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for accumulator in [Prim::F32, Prim::F64] {
            if prim == Prim::F64 && accumulator == Prim::F32 {
                continue;
            }
            let (m, k, n) = (33, 17, 31);
            let mut dag = Dag::new();
            let owner = Owner::from(dag.declare("contraction oracle"));
            let a_ty = shaped_ty(&[m, k], prim);
            let b_ty = shaped_ty(&[k, n], prim);
            let a = dag.add_node(
                owner,
                RiscOp::Load { name: "a".into() },
                vec![],
                a_ty.clone(),
                None,
            );
            let b = dag.add_node(
                owner,
                RiscOp::Load { name: "b".into() },
                vec![],
                b_ty.clone(),
                None,
            );
            let root = lower_matmul_with_accumulator(
                owner,
                &mut dag,
                a,
                b,
                &a_ty,
                &b_ty,
                accumulator,
                None,
            );
            dag.add_root(root);
            let inputs = UnordMap::from([
                (
                    "a".into(),
                    TensorValue::from_storage(
                        vec![m, k],
                        finalize_tensor(
                            "oracle",
                            prim,
                            RawTensor::Float(
                                (0..m * k)
                                    .map(|i| {
                                        [2048.0, 0.00006103515625, -2048.0, 1.0, -0.0, 0.0, 2.0]
                                            [i % 7]
                                    })
                                    .collect(),
                            ),
                        )
                        .unwrap(),
                    ),
                ),
                (
                    "b".into(),
                    TensorValue::from_storage(
                        vec![k, n],
                        finalize_tensor(
                            "oracle",
                            prim,
                            RawTensor::Float(
                                (0..k * n)
                                    .map(|i| [-1.0, 0.5, 1.0, -0.0, 2.0][i % 5])
                                    .collect(),
                            ),
                        )
                        .unwrap(),
                    ),
                ),
            ]);
            let before = LIVE.load(Ordering::Relaxed);
            PEAK.store(before, Ordering::Relaxed);
            let start = Instant::now();
            let rooted =
                eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned()).unwrap();
            let elapsed = start.elapsed();
            let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
            assert!(
                peak < 96 * 1024,
                "materialized contraction {prim:?}/{accumulator:?} peak={peak}"
            );
            let dense = eval_tensor(&dag, &inputs).unwrap();
            assert_eq!(
                bincode::serialize(rooted[&root].storage()).unwrap(),
                bincode::serialize(dense[&root].storage()).unwrap()
            );
            let product = dag
                .nodes()
                .iter()
                .find(|node| matches!(node.op, RiscOp::Mul))
                .unwrap()
                .id;
            let selected =
                eval_tensor_roots_with(&dag, &[root, product], |name| inputs.get(name).cloned())
                    .unwrap();
            assert_eq!(selected.len(), 2);
            assert_eq!(
                bincode::serialize(selected[&product].storage()).unwrap(),
                bincode::serialize(dense[&product].storage()).unwrap()
            );
            eprintln!(
                "eval-performance contraction dtype={prim:?} accumulator={accumulator:?} wall={elapsed:?} peak_heap_delta={peak} PASS"
            );
        }
    }
}

#[test]
fn eval_performance_streamed_convolution_bits_and_scratch() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::tier2::lower_conv;
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("streamed convolution"));
    let input_ty = shaped_ty(&[1, 3, 16, 16], Prim::F32);
    let kernel_ty = shaped_ty(&[2, 3, 3, 3], Prim::F32);
    let output_ty = shaped_ty(&[1, 2, 16, 16], Prim::F32);
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        input_ty.clone(),
        None,
    );
    let k = dag.add_node(
        owner,
        RiscOp::Load { name: "k".into() },
        vec![],
        kernel_ty.clone(),
        None,
    );
    let root = lower_conv(
        owner,
        &mut dag,
        x,
        k,
        &input_ty,
        &kernel_ty,
        &output_ty,
        &[1, 1],
        &[(1, 1), (1, 1)],
        None,
    );
    dag.add_root(root);
    let inputs: UnordMap<String, TensorValue> = UnordMap::from([
        (
            "x".into(),
            TensorValue::from_storage(
                vec![1, 3, 16, 16],
                finalize_tensor(
                    "oracle",
                    Prim::F32,
                    RawTensor::Float(
                        (0..768)
                            .map(|i| [2048.0, -2048.0, 0.5, -0.0, 0.00006103515625][i % 5])
                            .collect(),
                    ),
                )
                .unwrap(),
            ),
        ),
        (
            "k".into(),
            TensorValue::from_storage(
                vec![2, 3, 3, 3],
                finalize_tensor(
                    "oracle",
                    Prim::F32,
                    RawTensor::Float((0..54).map(|i| [1.0, -1.0, 0.5][i % 3]).collect()),
                )
                .unwrap(),
            ),
        ),
    ]);
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let rooted = eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned()).unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    assert!(peak < 128 * 1024, "materialized convolution peak={peak}");
    let dense = eval_tensor(&dag, &inputs).unwrap();
    assert_eq!(
        bincode::serialize(rooted[&root].storage()).unwrap(),
        bincode::serialize(dense[&root].storage()).unwrap()
    );
    let indices = dag
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Gather { .. }))
        .unwrap()
        .inputs[1];
    let selected =
        eval_tensor_roots_with(&dag, &[root, indices], |name| inputs.get(name).cloned()).unwrap();
    assert_eq!(
        bincode::serialize(selected[&indices].storage()).unwrap(),
        bincode::serialize(dense[&indices].storage()).unwrap()
    );
    eprintln!("eval-performance convolution peak_heap_delta={peak} PASS");
}

#[test]
#[ignore = "model-sized contraction witness; docs/manual_gates.md eval-performance"]
fn eval_performance_pointwise_pair_witness() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::tier2::{lower_gelu, lower_matmul};
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("pointwise pair"));
    let a_ty = shaped_ty(&[3136, 128], Prim::F32);
    let b_ty = shaped_ty(&[128, 512], Prim::F32);
    let c_ty = shaped_ty(&[512, 128], Prim::F32);
    let a = dag.add_node(
        owner,
        RiscOp::Load { name: "a".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let b = dag.add_node(
        owner,
        RiscOp::Load { name: "b".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let c = dag.add_node(
        owner,
        RiscOp::Load { name: "c".into() },
        vec![],
        c_ty.clone(),
        None,
    );
    let first = lower_matmul(owner, &mut dag, a, b, &a_ty, &b_ty, None);
    let mid_ty = shaped_ty(&[3136, 512], Prim::F32);
    let activation = lower_gelu(owner, &mut dag, first, &mid_ty, None);
    let root = lower_matmul(owner, &mut dag, activation, c, &mid_ty, &c_ty, None);
    dag.add_root(root);
    let value = |shape: Vec<usize>, v: f64| {
        let len = shape.iter().product();
        TensorValue::from_storage(
            shape,
            finalize_tensor("oracle", Prim::F32, RawTensor::Float(vec![v; len])).unwrap(),
        )
    };
    let inputs: UnordMap<String, TensorValue> = UnordMap::from([
        ("a".into(), value(vec![3136, 128], 0.1)),
        ("b".into(), value(vec![128, 512], 0.01)),
        ("c".into(), value(vec![512, 128], 0.01)),
    ]);
    // An independent dense scalar GELU graph supplies only the activation;
    // both contractions use explicit adjacent-pair native-f32 references.
    let first_value = (0..7).fold(0.1_f32 * 0.01_f32, |x, _| x + x);
    let mut scalar = Dag::new();
    let decl = scalar.declare("scalar activation");
    let x = scalar.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, f64::from(first_value)),
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let y = lower_gelu(decl.into(), &mut scalar, x, &TensorType::scalar_f32(), None);
    scalar.add_root(y);
    let activation = eval_tensor(&scalar, &UnordMap::default()).unwrap()[&y]
        .storage()
        .scalar_at(0)
        .as_f64_lossy() as f32;
    let expected = (0..9).fold(activation * 0.01_f32, |x, _| x + x);
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let start = Instant::now();
    let actual = eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned()).unwrap();
    let elapsed = start.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    assert!(peak < 96 * 1024 * 1024, "pointwise pair peak={peak}");
    assert!(
        elapsed.as_secs_f64() < 90.0,
        "pointwise pair wall={elapsed:?}"
    );
    for i in 0..3136 * 128 {
        assert_eq!(
            (actual[&root].storage().scalar_at(i).as_f64_lossy() as f32).to_bits(),
            expected.to_bits()
        );
    }
    eprintln!(
        "eval-performance pointwise [3136,128]x[128,512] gelu x[512,128] wall={elapsed:?} peak_heap_delta={peak} PASS"
    );
}

#[test]
fn eval_performance_cartesian_indices_errors_and_empty_domain() {
    use chelis_ir::dag::RtDim;
    use chelis_ir::eval::eval_tensor_roots_with;
    for (rows, cols, extent) in [
        (vec![0, 3], vec![0, 1], 5),
        (vec![-1, 0], vec![0, 1], 5),
        (vec![3, 0], vec![2, -1], 5),
        (vec![i64::MAX], vec![1], 5),
        (vec![i64::MIN], vec![-1], 5),
        (vec![i64::MAX], vec![], 0),
    ] {
        let mut dag = Dag::new();
        let owner = dag.declare("cartesian trap");
        let base = dag.add_node(
            owner,
            RiscOp::Load {
                name: "base".into(),
            },
            vec![],
            ty(extent, Prim::F32),
            None,
        );
        let r = dag.add_node(
            owner,
            RiscOp::Load { name: "r".into() },
            vec![],
            ty(rows.len(), Prim::Int64),
            None,
        );
        let c = dag.add_node(
            owner,
            RiscOp::Load { name: "c".into() },
            vec![],
            ty(cols.len(), Prim::Int64),
            None,
        );
        let matrix = shaped_ty(&[rows.len(), cols.len()], Prim::Int64);
        let a = dag.add_node(
            owner,
            RiscOp::Expand {
                axis: 1,
                size: RtDim::Lit(cols.len()),
            },
            vec![r],
            matrix.clone(),
            None,
        );
        let b = dag.add_node(
            owner,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(rows.len()),
            },
            vec![c],
            matrix.clone(),
            None,
        );
        let indices = dag.add_node(owner, RiscOp::Add, vec![a, b], matrix, None);
        let root = dag.add_node(
            owner,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
            vec![base, indices],
            shaped_ty(&[rows.len(), cols.len()], Prim::F32),
            None,
        );
        // A second view makes Gather private, while the selected Copy owns
        // the output. Errors must still belong to the original Add/Gather.
        let copy = dag.add_node(
            owner,
            RiscOp::Copy,
            vec![root],
            shaped_ty(&[rows.len(), cols.len()], Prim::F32),
            None,
        );
        dag.add_root(copy);
        let inputs: UnordMap<String, TensorValue> = UnordMap::from([
            (
                "base".into(),
                TensorValue::from_vec(vec![extent], vec![1.0; extent]),
            ),
            (
                "r".into(),
                TensorValue::finalize_from_wide_int(
                    "oracle",
                    Prim::Int64,
                    vec![rows.len()],
                    rows.clone(),
                )
                .unwrap(),
            ),
            (
                "c".into(),
                TensorValue::finalize_from_wide_int(
                    "oracle",
                    Prim::Int64,
                    vec![cols.len()],
                    cols.clone(),
                )
                .unwrap(),
            ),
        ]);
        let dense = eval_tensor(&dag, &inputs);
        let streamed = eval_tensor_roots_with(&dag, &[copy], |name| inputs.get(name).cloned());
        match (dense, streamed) {
            (Ok(dense), Ok(streamed)) => assert_eq!(
                bincode::serialize(dense[&copy].storage()).unwrap(),
                bincode::serialize(streamed[&copy].storage()).unwrap()
            ),
            (Err(a), Err(b)) => assert_eq!(a, b),
            other => panic!("different trap outcomes: {other:?}"),
        }
    }
}

#[test]
fn eval_performance_streamed_convolution_gradients() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::grad::grad_dag_checked;
    use chelis_ir::tier2::lower_conv;
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("convolution gradient"));
    let xt = shaped_ty(&[1, 1, 4, 4], Prim::F32);
    let kt = shaped_ty(&[1, 1, 3, 3], Prim::F32);
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        xt.clone(),
        None,
    );
    let k = dag.add_node(
        owner,
        RiscOp::Load { name: "k".into() },
        vec![],
        kt.clone(),
        None,
    );
    let mut loss = lower_conv(
        owner,
        &mut dag,
        x,
        k,
        &xt,
        &kt,
        &xt,
        &[1, 1],
        &[(1, 1), (1, 1)],
        None,
    );
    for rank in (0..4).rev() {
        loss = dag.add_node(
            owner,
            RiscOp::sum_with_accumulator(rank, Prim::F32, Prim::F32).unwrap(),
            vec![loss],
            shaped_ty(&[1, 1, 4, 4][..rank], Prim::F32),
            None,
        );
    }
    dag.add_root(loss);
    let value = |shape: Vec<usize>, data: Vec<f64>| {
        TensorValue::from_storage(
            shape,
            finalize_tensor("oracle", Prim::F32, RawTensor::Float(data)).unwrap(),
        )
    };
    let inputs: UnordMap<String, TensorValue> = UnordMap::from([
        ("x".into(), value(vec![1, 1, 4, 4], vec![0.25; 16])),
        ("k".into(), value(vec![1, 1, 3, 3], vec![0.5; 9])),
    ]);
    let differentiated = grad_dag_checked(&dag, loss, &[x, k]).unwrap();
    let roots = [differentiated.grad_nodes[&x], differentiated.grad_nodes[&k]];
    let dense = eval_tensor(&differentiated.dag, &inputs).unwrap();
    let streamed = eval_tensor_roots_with(&differentiated.dag, &roots, |name| {
        inputs.get(name).cloned()
    })
    .unwrap();
    for root in roots {
        assert_eq!(
            bincode::serialize(dense[&root].storage()).unwrap(),
            bincode::serialize(streamed[&root].storage()).unwrap()
        );
    }
    let delta = 0.03125;
    let scalar_loss = |offset: f64| {
        let mut perturbed = inputs.clone();
        let mut data = vec![0.25; 16];
        data[5] += offset;
        perturbed.insert("x".into(), value(vec![1, 1, 4, 4], data));
        eval_tensor_roots_with(&dag, &[loss], |name| perturbed.get(name).cloned()).unwrap()[&loss]
            .storage()
            .scalar_at(0)
            .as_f64_lossy()
    };
    let finite_difference = (scalar_loss(delta) - scalar_loss(-delta)) / (2.0 * delta);
    assert!(
        (streamed[&roots[0]].storage().scalar_at(5).as_f64_lossy() - finite_difference).abs()
            < 1e-5
    );
}

#[test]
fn eval_performance_streamed_contraction_nonfinite_bits() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::tier2::lower_matmul;
    for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for values in [
            vec![-0.0, -0.0, -0.0],
            vec![f64::NAN, 1.0, 2.0],
            vec![f64::INFINITY, -f64::INFINITY, 1.0],
            vec![0.0, -0.0, 0.0],
        ] {
            let mut dag = Dag::new();
            let owner = Owner::from(dag.declare("nonfinite contraction"));
            let at = shaped_ty(&[1, 3], prim);
            let bt = shaped_ty(&[3, 1], prim);
            let a = dag.add_node(
                owner,
                RiscOp::Load { name: "a".into() },
                vec![],
                at.clone(),
                None,
            );
            let b = dag.add_node(
                owner,
                RiscOp::Load { name: "b".into() },
                vec![],
                bt.clone(),
                None,
            );
            let root = lower_matmul(owner, &mut dag, a, b, &at, &bt, None);
            dag.add_root(root);
            let inputs: UnordMap<String, TensorValue> = UnordMap::from([
                (
                    "a".into(),
                    TensorValue::from_storage(
                        vec![1, 3],
                        finalize_tensor("oracle", prim, RawTensor::Float(values)).unwrap(),
                    ),
                ),
                (
                    "b".into(),
                    TensorValue::from_storage(
                        vec![3, 1],
                        finalize_tensor("oracle", prim, RawTensor::Float(vec![1.0; 3])).unwrap(),
                    ),
                ),
            ]);
            let dense = eval_tensor(&dag, &inputs).unwrap();
            let streamed =
                eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned()).unwrap();
            assert_eq!(
                bincode::serialize(dense[&root].storage()).unwrap(),
                bincode::serialize(streamed[&root].storage()).unwrap()
            );
        }
    }
}

#[test]
fn eval_performance_many_contraction_plans() {
    use chelis_ir::dag::Owner;
    use chelis_ir::tier2::lower_matmul;
    for count in [16, 256, 4096] {
        let mut dag = Dag::new();
        let owner = Owner::from(dag.declare("many contractions"));
        let ty = shaped_ty(&[1, 1], Prim::F32);
        let a = dag.add_node(
            owner,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(
            owner,
            RiscOp::Load { name: "b".into() },
            vec![],
            ty.clone(),
            None,
        );
        for _ in 0..count {
            let root = lower_matmul(owner, &mut dag, a, b, &ty, &ty, None);
            dag.add_root(root);
        }
        let start = Instant::now();
        let plans = chelis_ir::fuse::reduction_inlined_products(&dag);
        assert_eq!(plans.len(), count);
        eprintln!(
            "eval-performance planner contractions={count} nodes={} wall={:?} PASS",
            dag.len(),
            start.elapsed()
        );
    }
}

#[test]
fn eval_performance_contraction_dense_control() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::tier2::lower_matmul;
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("dense timing control"));
    let at = shaped_ty(&[196, 128], Prim::F32);
    let bt = shaped_ty(&[128, 128], Prim::F32);
    let a = dag.add_node(
        owner,
        RiscOp::Load { name: "a".into() },
        vec![],
        at.clone(),
        None,
    );
    let b = dag.add_node(
        owner,
        RiscOp::Load { name: "b".into() },
        vec![],
        bt.clone(),
        None,
    );
    let root = lower_matmul(owner, &mut dag, a, b, &at, &bt, None);
    dag.add_root(root);
    let inputs: UnordMap<String, TensorValue> = UnordMap::from([
        (
            "a".into(),
            TensorValue::from_vec(vec![196, 128], vec![0.1; 196 * 128]),
        ),
        (
            "b".into(),
            TensorValue::from_vec(vec![128, 128], vec![0.01; 128 * 128]),
        ),
    ]);
    let start = Instant::now();
    let streamed = eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned()).unwrap();
    let stream_time = start.elapsed();
    let start = Instant::now();
    let dense = eval_tensor(&dag, &inputs).unwrap();
    let dense_time = start.elapsed();
    assert_eq!(
        bincode::serialize(streamed[&root].storage()).unwrap(),
        bincode::serialize(dense[&root].storage()).unwrap()
    );
    eprintln!(
        "eval-performance same-head contraction control streamed={stream_time:?} dense={dense_time:?} PASS"
    );
}

fn coordinate_case(
    shape: &[usize],
    out_shape: &[usize],
    op: RiscOp,
    source_index: impl Fn(usize) -> usize,
) {
    let mut dag = Dag::new();
    let decl = dag.declare("coordinate oracle");
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        shaped_ty(shape, Prim::Int64),
        None,
    );
    let root = dag.add_node(
        decl,
        op,
        vec![input],
        shaped_ty(out_shape, Prim::Int64),
        None,
    );
    dag.add_root(root);
    let len: usize = shape.iter().product();
    let out_len: usize = out_shape.iter().product();
    let inputs = UnordMap::from([(
        "x".into(),
        TensorValue::finalize_from_wide_int(
            "oracle",
            Prim::Int64,
            shape.to_vec(),
            (0..len).map(|i| (1_i64 << 53) + i as i64).collect(),
        )
        .unwrap(),
    )]);
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    let before = BYTES.load(Ordering::Relaxed);
    let start = Instant::now();
    let values = eval_tensor(&dag, &inputs).unwrap();
    let elapsed = start.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start_live);
    let allocated = BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(values.len(), 2, "all-values API retains both nodes");
    let rooted =
        chelis_ir::eval::eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned())
            .unwrap();
    assert_eq!(rooted.len(), 1);
    assert_eq!(values[&root].storage(), rooted[&root].storage());
    for i in 0..out_len {
        assert_eq!(
            values[&root].storage().scalar_at(i).as_i64_exact(),
            Some((1_i64 << 53) + source_index(i) as i64)
        );
    }
    // Resolved input, loaded value and copied output are permitted. A usize
    // per output is not.
    assert!(
        peak < (2 * len + out_len) * 8 + 1024 * 1024,
        "coordinate scratch: shape={shape:?}, output={out_shape:?}, peak={peak}"
    );
    eprintln!(
        "eval-performance movement shape={shape:?} output={out_shape:?} wall={elapsed:?} allocated={allocated} peak_heap_delta={peak} PASS"
    );
}

#[test]
fn eval_performance_regular_movement_coordinates() {
    use chelis_ir::dag::RtDim;
    coordinate_case(
        &[32, 64, 512],
        &[512, 32, 64],
        RiscOp::Permute {
            axes: vec![2, 0, 1],
        },
        |i| ((i / 64) % 32 * 64 + i % 64) * 512 + i / (32 * 64),
    );
    coordinate_case(
        &[256, 1, 256],
        &[256, 16, 256],
        RiscOp::Expand {
            axis: 1,
            size: RtDim::Lit(16),
        },
        |i| i / (16 * 256) * 256 + i % 256,
    );
    coordinate_case(
        &[256, 256],
        &[256, 16, 256],
        RiscOp::Expand {
            axis: 1,
            size: RtDim::Lit(16),
        },
        |i| i / (16 * 256) * 256 + i % 256,
    );
    coordinate_case(
        &[1024, 1024],
        &[768, 896],
        RiscOp::Shrink {
            bounds: vec![
                (RtDim::Lit(128), RtDim::Lit(896)),
                (RtDim::Lit(64), RtDim::Lit(960)),
            ],
        },
        |i| (i / 896 + 128) * 1024 + i % 896 + 64,
    );
    coordinate_case(
        &[1024, 1024],
        &[512, 342],
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2), RtDim::Lit(3)],
        },
        |i| i / 342 * 2 * 1024 + i % 342 * 3,
    );
}

#[test]
fn eval_performance_depthwise_window_gather_composition() {
    let mut dag = Dag::new();
    let decl = dag.declare("depthwise window oracle");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        shaped_ty(&[1, 2, 8, 8], Prim::Int64),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        shaped_ty(&[8, 3], Prim::Int64),
        None,
    );
    let rows = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 2,
            batch_rank: 0,
        },
        vec![x, indices],
        shaped_ty(&[1, 2, 8, 3, 8], Prim::Int64),
        None,
    );
    let cols = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 4,
            batch_rank: 0,
        },
        vec![rows, indices],
        shaped_ty(&[1, 2, 8, 3, 8, 3], Prim::Int64),
        None,
    );
    let root = dag.add_node(
        decl,
        RiscOp::Permute {
            axes: vec![0, 1, 2, 4, 3, 5],
        },
        vec![cols],
        shaped_ty(&[1, 2, 8, 8, 3, 3], Prim::Int64),
        None,
    );
    dag.add_root(root);
    let inputs = UnordMap::from([
        (
            "x".into(),
            TensorValue::finalize_from_wide_int(
                "oracle",
                Prim::Int64,
                vec![1, 2, 8, 8],
                (0..128).map(|i| (1_i64 << 53) + i).collect(),
            )
            .unwrap(),
        ),
        (
            "indices".into(),
            TensorValue::finalize_from_wide_int(
                "oracle",
                Prim::Int64,
                vec![8, 3],
                (0..24).map(|i| (i / 3 + i % 3) % 8).collect(),
            )
            .unwrap(),
        ),
    ]);
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    let start = Instant::now();
    let result = eval_tensor(&dag, &inputs).unwrap();
    let elapsed = start.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start_live);
    for i in 0..1152 {
        let kx = i % 3;
        let ky = i / 3 % 3;
        let x = i / 9 % 8;
        let y = i / 72 % 8;
        let c = i / 576;
        let source = (c * 8 + (y + ky) % 8) * 8 + (x + kx) % 8;
        assert_eq!(
            result[&root].storage().scalar_at(i).as_i64_exact(),
            Some((1_i64 << 53) + source as i64)
        );
    }
    assert!(peak < 128 * 1024, "window scratch {peak}");
    eprintln!(
        "eval-performance depthwise gather composition wall={elapsed:?} peak_heap_delta={peak} PASS"
    );
}

#[test]
fn eval_performance_large_gather_coordinates() {
    let mut dag = Dag::new();
    let decl = dag.declare("large gather oracle");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        shaped_ty(&[32, 1024], Prim::Int64),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        ty(32768, Prim::Int64),
        None,
    );
    let root = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 1,
            batch_rank: 0,
        },
        vec![x, indices],
        shaped_ty(&[32, 32768], Prim::Int64),
        None,
    );
    dag.add_root(root);
    let inputs = UnordMap::from([
        (
            "x".into(),
            TensorValue::finalize_from_wide_int(
                "oracle",
                Prim::Int64,
                vec![32, 1024],
                (0..32768).map(|i| (1_i64 << 53) + i).collect(),
            )
            .unwrap(),
        ),
        (
            "indices".into(),
            TensorValue::finalize_from_wide_int(
                "oracle",
                Prim::Int64,
                vec![32768],
                (0..32768).map(|i| (i * 7 + 3) % 1024).collect(),
            )
            .unwrap(),
        ),
    ]);
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    let start = Instant::now();
    let result = eval_tensor(&dag, &inputs).unwrap();
    let elapsed = start.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start_live);
    for flat in 0..1048576 {
        let source = flat / 32768 * 1024 + (flat % 32768 * 7 + 3) % 1024;
        assert_eq!(
            result[&root].storage().scalar_at(flat).as_i64_exact(),
            Some((1_i64 << 53) + source as i64)
        );
    }
    assert!(
        peak < (2 * 65536 + 1048576) * 8 + 1024 * 1024,
        "gather coordinate scratch {peak}"
    );
    eprintln!("eval-performance large gather wall={elapsed:?} peak_heap_delta={peak} PASS");
}

#[test]
#[ignore = "large movement witness; docs/manual_gates.md eval-performance"]
fn eval_performance_large_window_movement_coordinates() {
    use chelis_ir::dag::RtDim;
    coordinate_case(
        &[1, 128, 56, 56, 7, 7],
        &[1, 128, 56, 7, 56, 7],
        RiscOp::Permute {
            axes: vec![0, 1, 2, 4, 3, 5],
        },
        |i| {
            let kx = i % 7;
            let x = i / 7 % 56;
            let ky = i / (7 * 56) % 7;
            let cy = i / (7 * 56 * 7);
            ((cy * 56 + x) * 7 + ky) * 7 + kx
        },
    );
    coordinate_case(
        &[1, 128, 56, 56, 1, 7],
        &[1, 128, 56, 56, 7, 7],
        RiscOp::Expand {
            axis: 4,
            size: RtDim::Lit(7),
        },
        |i| i / 49 * 7 + i % 7,
    );
}

fn axis_sum_case(shape: &[usize], axis: usize) {
    axis_reduction_case(
        shape,
        axis,
        RiscOp::Sum {
            axis,
            accumulator: Prim::Int64,
        },
    );
}

fn axis_reduction_case(shape: &[usize], axis: usize, op: RiscOp) {
    let mut dag = Dag::new();
    let decl = dag.declare("axis oracle");
    let mut out_shape = shape.to_vec();
    let axis_len = out_shape.remove(axis);
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        shaped_ty(shape, Prim::Int64),
        None,
    );
    let root = dag.add_node(
        decl,
        op.clone(),
        vec![input],
        shaped_ty(&out_shape, Prim::Int64),
        None,
    );
    dag.add_root(root);
    let len: usize = shape.iter().product();
    let out_len: usize = out_shape.iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let inputs = UnordMap::from([(
        "x".into(),
        TensorValue::finalize_from_wide_int(
            "oracle",
            Prim::Int64,
            shape.to_vec(),
            (0..len).map(|i| (i % 13) as i64 - 6).collect(),
        )
        .unwrap(),
    )]);
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    let start = Instant::now();
    let values = eval_tensor(&dag, &inputs).unwrap();
    let elapsed = start.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start_live);
    assert_eq!(values.len(), 2, "all-values API retains both nodes");
    let rooted =
        chelis_ir::eval::eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned())
            .unwrap();
    assert_eq!(rooted.len(), 1);
    assert_eq!(values[&root].storage(), rooted[&root].storage());
    for i in 0..out_len {
        let base = i / inner * axis_len * inner + i % inner;
        let elements: Vec<_> = (0..axis_len)
            .map(|j| ((base + j * inner) % 13) as i64 - 6)
            .collect();
        let expected = match op {
            RiscOp::Sum { .. } => elements.iter().sum::<i64>(),
            RiscOp::ProdReduce { .. } => elements.iter().product::<i64>(),
            RiscOp::MaxReduce { .. } => *elements.iter().max().unwrap(),
            RiscOp::MinReduce { .. } => *elements.iter().min().unwrap(),
            RiscOp::Argmax { .. } => elements
                .iter()
                .position(|x| x == elements.iter().max().unwrap())
                .unwrap() as i64,
            RiscOp::Argmin { .. } => elements
                .iter()
                .position(|x| x == elements.iter().min().unwrap())
                .unwrap() as i64,
            _ => unreachable!(),
        };
        assert_eq!(
            values[&root].storage().scalar_at(i).as_i64_exact(),
            Some(expected)
        );
    }
    // Input load, typed result construction and output are permitted, with
    // a fixed margin for graph scheduling; index/leaf arrays are not.
    assert!(
        peak < len * 16 + out_len * 48 + 1024 * 1024,
        "axis scratch shape={shape:?} axis={axis} peak={peak}"
    );
    eprintln!(
        "eval-performance reduction op={op:?} shape={shape:?} axis={axis} wall={elapsed:?} peak_heap_delta={peak} PASS"
    );
}

#[test]
fn eval_performance_axis_reduction_coordinates() {
    axis_sum_case(&[1048576], 0);
    axis_sum_case(&[65536, 16], 1);
    axis_sum_case(&[128, 64, 128], 0);
    axis_sum_case(&[128, 64, 128], 1);
    axis_sum_case(&[128, 64, 128], 2);
    for op in [
        RiscOp::ProdReduce { axis: 1 },
        RiscOp::MaxReduce { axis: 1 },
        RiscOp::MinReduce { axis: 1 },
        RiscOp::Argmax { axis: 1 },
        RiscOp::Argmin { axis: 1 },
    ] {
        axis_reduction_case(&[65536, 16], 1, op);
    }
}

#[test]
#[ignore = "large reduction witness; docs/manual_gates.md eval-performance"]
fn eval_performance_large_axis_reduction_coordinates() {
    axis_sum_case(&[19668992], 0);
    axis_sum_case(&[3136, 6272], 1);
}

fn sparse_case(n: usize, prim: Prim, invalid: Option<i64>) {
    let mut dag = Dag::new();
    let decl = dag.declare("sparse oracle");
    let data = dag.add_node(
        decl,
        RiscOp::Load {
            name: "data".into(),
        },
        vec![],
        ty(16, Prim::Int64),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        ty(n, prim),
        None,
    );
    let root = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![data, indices],
        ty(n, Prim::Int64),
        None,
    );
    dag.add_root(root);
    let expected: Vec<i64> = (0..n).map(|i| ((i * 7 + 3) % 16) as i64).collect();
    let mut source = expected.clone();
    if let Some(index) = invalid {
        source[n / 2] = index;
    }
    let inputs = UnordMap::from([
        (
            "data".into(),
            TensorValue::finalize_from_wide_int(
                "oracle",
                Prim::Int64,
                vec![16],
                (0..16).map(|i| (1_i64 << 53) + i).collect(),
            )
            .unwrap(),
        ),
        (
            "indices".into(),
            TensorValue::from_storage(
                vec![n],
                finalize_tensor("oracle", prim, RawTensor::Int(source)).unwrap(),
            ),
        ),
    ]);
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    let start_bytes = BYTES.load(Ordering::Relaxed);
    let start = Instant::now();
    let result = eval_tensor(&dag, &inputs);
    let elapsed = start.elapsed();
    let bytes = BYTES.load(Ordering::Relaxed) - start_bytes;
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start_live);
    if let Some(index) = invalid {
        let error = result.unwrap_err();
        assert_eq!(
            error,
            chelis_abi::failure::sparse_index_out_of_bounds("gather", index, 0, 16)
        );
    } else {
        assert_eq!(
            result.unwrap()[&root].storage().to_i64_exact_vec().unwrap(),
            expected
                .iter()
                .map(|i| (1_i64 << 53) + i)
                .collect::<Vec<_>>()
        );
        assert!(
            bytes < n * 256 + 131072,
            "whole-buffer copies: n={n}, allocated={bytes}"
        );
        assert!(
            peak < n * 64 + 131072,
            "unbounded sparse scratch: n={n}, peak={peak}"
        );
    }
    eprintln!(
        "eval-performance sparse n={n} dtype={prim:?} wall={elapsed:?} allocated={bytes} peak_heap_delta={peak} PASS"
    );
}

#[test]
fn eval_performance_sparse_runtime_indices() {
    // Nonconstant runtime loads, two lengths, all signed index widths, and
    // negative/upper-bound/large-exact-integer rejection controls.
    for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        for n in [128, 4096] {
            sparse_case(n, prim, None);
        }
        for index in [-1, 16] {
            sparse_case(128, prim, Some(index));
        }
    }
    sparse_case(128, Prim::Int64, Some((1_i64 << 53) + 1));
    for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        sparse_consumers(prim, false);
        sparse_consumers(prim, true);
    }
}

fn sparse_consumers(prim: Prim, invalid: bool) {
    for op in [
        RiscOp::ScatterAdd {
            axis: 0,
            batch_rank: 0,
        },
        RiscOp::Scatter {
            axis: 0,
            batch_rank: 0,
        },
        RiscOp::ScatterElements { axis: 0 },
        RiscOp::OneHot { vocab: 16 },
    ] {
        let one_hot = matches!(op, RiscOp::OneHot { .. });
        let adding = matches!(op, RiscOp::ScatterAdd { .. });
        let mut dag = Dag::new();
        let decl = dag.declare("sparse consumer oracle");
        let indices = dag.add_node(
            decl,
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            ty(5, prim),
            None,
        );
        let data = dag.add_node(
            decl,
            RiscOp::Load {
                name: "data".into(),
            },
            vec![],
            ty(16, Prim::Int64),
            None,
        );
        let updates = dag.add_node(
            decl,
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            ty(5, Prim::Int64),
            None,
        );
        let output = if one_hot {
            TensorType {
                dims: vec![DimInfo::Lit(5), DimInfo::Lit(16)],
                precision: Prim::Int64,
            }
        } else {
            ty(16, Prim::Int64)
        };
        let root = dag.add_node(
            decl,
            op,
            if one_hot {
                vec![indices]
            } else {
                vec![data, indices, updates]
            },
            output,
            None,
        );
        dag.add_root(root);
        let source = if invalid {
            vec![3, 0, -1, 15, 2]
        } else {
            vec![3, 0, 3, 15, 2]
        };
        let inputs = UnordMap::from([
            (
                "indices".into(),
                TensorValue::from_storage(
                    vec![5],
                    finalize_tensor("oracle", prim, RawTensor::Int(source.clone())).unwrap(),
                ),
            ),
            (
                "data".into(),
                TensorValue::finalize_from_wide_int("oracle", Prim::Int64, vec![16], vec![0; 16])
                    .unwrap(),
            ),
            (
                "updates".into(),
                TensorValue::finalize_from_wide_int(
                    "oracle",
                    Prim::Int64,
                    vec![5],
                    vec![11, 12, 13, 14, 15],
                )
                .unwrap(),
            ),
        ]);
        // OneHot's existing evaluator boundary asserts on an invalid class;
        // preserve that behavior here rather than changing its error contract.
        if one_hot && invalid {
            assert!(std::panic::catch_unwind(|| eval_tensor(&dag, &inputs)).is_err());
            continue;
        }
        let result = eval_tensor(&dag, &inputs);
        if invalid {
            let error = result.unwrap_err();
            assert!(error.contains("-1"), "{error}");
        } else {
            let mut expected = vec![0; if one_hot { 80 } else { 16 }];
            for (i, index) in source.into_iter().enumerate() {
                if one_hot {
                    expected[i * 16 + index as usize] = 1;
                } else if adding {
                    expected[index as usize] += i as i64 + 11;
                } else {
                    expected[index as usize] = i as i64 + 11;
                }
            }
            assert_eq!(
                result.unwrap()[&root].storage().to_i64_exact_vec().unwrap(),
                expected
            );
        }
    }
}

#[test]
#[ignore = "full convolution witness; docs/manual_gates.md eval-performance"]
fn eval_performance_convolution_index_witness() {
    use chelis_ir::dag::Owner;
    use chelis_ir::eval::eval_tensor_roots_with;
    use chelis_ir::tier2::lower_conv;
    let tensor = |shape: &[usize]| TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("conv oracle"));
    let input_ty = tensor(&[1, 128, 56, 56]);
    let kernel_ty = tensor(&[1, 128, 7, 7]);
    let output_ty = tensor(&[1, 1, 56, 56]);
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        input_ty.clone(),
        None,
    );
    let k = dag.add_node(
        owner,
        RiscOp::Load { name: "k".into() },
        vec![],
        kernel_ty.clone(),
        None,
    );
    let root = lower_conv(
        owner,
        &mut dag,
        x,
        k,
        &input_ty,
        &kernel_ty,
        &output_ty,
        &[1, 1],
        &[(3, 3), (3, 3)],
        None,
    );
    dag.add_root(root);
    let inputs = UnordMap::from([
        (
            "x".to_string(),
            TensorValue::from_storage(
                vec![1, 128, 56, 56],
                finalize_tensor("oracle", Prim::F32, RawTensor::Float(vec![0.1; 401408])).unwrap(),
            ),
        ),
        (
            "k".to_string(),
            TensorValue::from_storage(
                vec![1, 128, 7, 7],
                finalize_tensor("oracle", Prim::F32, RawTensor::Float(vec![0.01; 6272])).unwrap(),
            ),
        ),
    ]);
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    let start = Instant::now();
    let result = eval_tensor_roots_with(&dag, &[root], |name| inputs.get(name).cloned()).unwrap();
    let elapsed = start.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start_live);
    assert!(peak < 32 * 1024 * 1024, "convolution peak={peak}");
    // Direct [05-OP-51] cross-correlation, with [05-OP-30]'s adjacent-pair
    // tree over channel/kernel-height/kernel-width, including padded zeros.
    for y in 0..56 {
        for x in 0..56 {
            let mut terms = Vec::with_capacity(6272);
            for _ in 0..128 {
                for ky in 0..7 {
                    for kx in 0..7 {
                        let in_bounds = (3..59).contains(&(y + ky)) && (3..59).contains(&(x + kx));
                        terms.push(if in_bounds {
                            0.1_f32 * 0.01_f32
                        } else {
                            0.0_f32
                        });
                    }
                }
            }
            while terms.len() > 1 {
                terms = terms
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
            assert_eq!(
                (result[&root].storage().scalar_at(y * 56 + x).as_f64_lossy() as f32).to_bits(),
                terms[0].to_bits()
            );
        }
    }
    assert!(
        elapsed.as_secs_f64() < 90.0,
        "convolution exceeded 90 s: {elapsed:?}"
    );
    eprintln!(
        "eval-performance conv [1,128,56,56] x [1,128,7,7] wall={elapsed:?} peak_heap_delta={peak} PASS"
    );
}
