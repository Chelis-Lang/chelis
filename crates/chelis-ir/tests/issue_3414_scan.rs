//! [05-OP-33] ordered scans retain first-leaf and accumulator semantics.
use chelis_ir::dag::{Dag, DimInfo, RiscOp, ScanDirection, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_types::dtype_semantics::cumsum_tensor_lanes;
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
fn graph(
    shape: &[usize],
    p: Prim,
    axis: usize,
    direction: ScanDirection,
    runtime: bool,
) -> (Dag, chelis_ir::dag::NodeId) {
    let mut dag = Dag::new();
    let owner = dag.declare("scan");
    let dims = shape
        .iter()
        .map(|n| {
            if runtime {
                DimInfo::Named("*".into(), None)
            } else {
                DimInfo::Lit(*n)
            }
        })
        .collect::<Vec<_>>();
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: dims.clone(),
            precision: p,
        },
        None,
    );
    let root = dag.add_node(
        owner,
        RiscOp::Cumsum { axis, direction },
        vec![x],
        TensorType {
            dims,
            precision: p.default_reduce_sum_result_precision().unwrap_or(p),
        },
        None,
    );
    dag.add_root(root);
    (dag, root)
}
fn bits(v: &chelis_types::TensorStorage) -> Vec<u64> {
    (0..v.len())
        .map(|i| v.scalar_at(i).as_f64_lossy().to_bits())
        .collect()
}
#[test]
fn scan_prefix_suffix_exact_at_all_float_widths_axes_lengths_and_runtime_extents() {
    for p in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for n in [0, 1, 3, 5, 17] {
            for axis in 0..3 {
                for direction in [ScanDirection::Increasing, ScanDirection::Decreasing] {
                    for runtime in [false, true] {
                        let mut shape = vec![2, 2, 3];
                        shape[axis] = n;
                        let len = shape.iter().product();
                        let storage = finalize_tensor(
                            "oracle",
                            p,
                            RawTensor::Float(
                                (0..len)
                                    .map(|i| [-0.0, 0.1, 2048.0, -2048.0, 0.3, f64::NAN][i % 6])
                                    .collect(),
                            ),
                        )
                        .unwrap();
                        let outer = shape[..axis].iter().product::<usize>();
                        let inner = shape[axis + 1..].iter().product::<usize>();
                        let lanes = (0..outer * inner)
                            .map(|lane| {
                                (0..n)
                                    .map(|k| {
                                        let k = if direction == ScanDirection::Decreasing {
                                            n - 1 - k
                                        } else {
                                            k
                                        };
                                        (lane / inner * n + k) * inner + lane % inner
                                    })
                                    .collect()
                            })
                            .collect::<Vec<Vec<usize>>>();
                        let expected = cumsum_tensor_lanes(&storage, &lanes).unwrap();
                        let (dag, root) = graph(&shape, p, axis, direction, runtime);
                        let actual = eval_tensor(
                            &dag,
                            &UnordMap::from([(
                                "x".into(),
                                TensorValue::from_storage(shape.clone(), storage),
                            )]),
                        )
                        .unwrap();
                        assert_eq!(actual[&root].shape, shape);
                        assert_eq!(actual[&root].prim(), p);
                        assert_eq!(
                            bits(actual[&root].storage()),
                            bits(&expected),
                            "{p:?} {n} {axis} {direction:?} runtime={runtime}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn scan_integer_forward_overflow_and_bool_rejection() {
    for p in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let (dag, root) = graph(&[3], p, 0, ScanDirection::Increasing, false);
        let value =
            TensorValue::finalize_from_wide_int("oracle", p, vec![3], vec![1, 2, -1]).unwrap();
        let actual = eval_tensor(&dag, &UnordMap::from([("x".into(), value)])).unwrap();
        assert_eq!(
            (0..3)
                .map(|i| actual[&root].storage().scalar_at(i).as_i64_exact().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
    }
    for (p, max) in [(Prim::Int32, i64::from(i32::MAX)), (Prim::Int64, i64::MAX)] {
        let (dag, _) = graph(&[3], p, 0, ScanDirection::Increasing, false);
        let value =
            TensorValue::finalize_from_wide_int("oracle", p, vec![3], vec![max, 1, 0]).unwrap();
        assert!(
            eval_tensor(&dag, &UnordMap::from([("x".into(), value)]))
                .unwrap_err()
                .contains("overflow")
        );
    }
    let (reverse_integer, _) = graph(&[3], Prim::Int32, 0, ScanDirection::Decreasing, false);
    let values =
        TensorValue::finalize_from_wide_int("oracle", Prim::Int32, vec![3], vec![1, 2, -1])
            .unwrap();
    assert!(eval_tensor(&reverse_integer, &UnordMap::from([("x".into(), values)])).is_err());
    let (dag, _) = graph(&[1], Prim::Bool, 0, ScanDirection::Increasing, false);
    let value = TensorValue::from_storage(
        vec![1],
        finalize_tensor("oracle", Prim::Bool, RawTensor::Int(vec![1])).unwrap(),
    );
    assert!(eval_tensor(&dag, &UnordMap::from([("x".into(), value)])).is_err());
}
#[test]
fn vmap_scan_shifts_axis_and_retains_direction() {
    let (dag, _) = graph(&[5], Prim::F32, 0, ScanDirection::Decreasing, false);
    let (batched, _) =
        chelis_ir::vmap::vectorize_axis0_with_node_map(&dag, DimInfo::Lit(3)).unwrap();
    assert!(batched.nodes().iter().any(|node| matches!(
        node.op,
        RiscOp::Cumsum {
            axis: 1,
            direction: ScanDirection::Decreasing
        }
    )));
}

#[test]
fn reduced_float_scan_keeps_wide_running_state() {
    for (p, tiny, last) in [
        (Prim::F16, 2_f64.powi(-12), 1.0009765625),
        (Prim::Bf16, 2_f64.powi(-9), 1.0078125),
    ] {
        let storage =
            finalize_tensor("oracle", p, RawTensor::Float(vec![1.0, tiny, tiny, tiny])).unwrap();
        let (dag, root) = graph(&[4], p, 0, ScanDirection::Increasing, false);
        let values = eval_tensor(
            &dag,
            &UnordMap::from([("x".into(), TensorValue::from_storage(vec![4], storage))]),
        )
        .unwrap();
        assert_eq!(values[&root].storage().scalar_at(3).as_f64_lossy(), last);
    }
}

#[test]
fn scan_adjoint_reverses_direction_without_zero_seed_at_all_float_widths() {
    use chelis_ir::grad::grad_dag_checked;
    for p in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for n in [0, 1, 5] {
            for direction in [ScanDirection::Increasing, ScanDirection::Decreasing] {
                let (mut dag, scan) = graph(&[n], p, 0, direction, true);
                let x = dag.nodes()[0].id;
                let owner = dag.nodes()[0].owner;
                let ty = dag.get(scan).unwrap().output_type.clone();
                let weights = dag.add_node(
                    owner,
                    RiscOp::Load { name: "w".into() },
                    vec![],
                    ty.clone(),
                    None,
                );
                let weighted = dag.add_node(owner, RiscOp::Mul, vec![scan, weights], ty, None);
                let acc = RiscOp::default_reduce_sum_accumulator(p).unwrap();
                let loss = dag.add_node(
                    owner,
                    RiscOp::Sum {
                        axis: 0,
                        accumulator: acc,
                    },
                    vec![weighted],
                    TensorType {
                        dims: vec![],
                        precision: acc,
                    },
                    None,
                );
                let gradient = grad_dag_checked(&dag, loss, &[x]).unwrap();
                let root = gradient.grad_nodes[&x];
                let inputs = UnordMap::from([
                    (
                        "x".into(),
                        TensorValue::finalize_from_wide("oracle", p, vec![n], vec![0.0; n])
                            .unwrap(),
                    ),
                    (
                        "w".into(),
                        TensorValue::finalize_from_wide("oracle", p, vec![n], vec![-0.0; n])
                            .unwrap(),
                    ),
                ]);
                let actual = eval_tensor(&gradient.dag, &inputs).unwrap();
                assert_eq!(
                    bits(actual[&root].storage()),
                    vec![(-0.0_f64).to_bits(); n],
                    "{p:?} {n} {direction:?}"
                );
                if n == 5 {
                    let (big, tiny) = match p {
                        Prim::F16 => (2048.0, 0.0001220703125),
                        Prim::Bf16 => (256.0, 0.0000152587890625),
                        Prim::F32 => (16777216.0, 1.0),
                        Prim::F64 => (9007199254740992.0, 1.0),
                        _ => unreachable!(),
                    };
                    let weights = TensorValue::finalize_from_wide(
                        "oracle",
                        p,
                        vec![n],
                        vec![big, tiny, -big, 0.0, 0.0],
                    )
                    .unwrap();
                    let lane = if direction == ScanDirection::Increasing {
                        (0..n).rev().collect::<Vec<_>>()
                    } else {
                        (0..n).collect()
                    };
                    let expected = cumsum_tensor_lanes(weights.storage(), &[lane]).unwrap();
                    let mut cancellation = inputs.clone();
                    cancellation.insert("w".into(), weights);
                    let actual = eval_tensor(&gradient.dag, &cancellation).unwrap();
                    assert_eq!(
                        bits(actual[&root].storage()),
                        bits(&expected),
                        "cancellation {p:?} {direction:?}"
                    );
                }
                // A positive-zero boundary cotangent changes the result.
                if n > 0 {
                    let mut positive = inputs.clone();
                    positive.insert(
                        "w".into(),
                        TensorValue::finalize_from_wide("oracle", p, vec![n], vec![0.0; n])
                            .unwrap(),
                    );
                    let actual = eval_tensor(&gradient.dag, &positive).unwrap();
                    assert_eq!(bits(actual[&root].storage()), vec![0.0_f64.to_bits(); n]);
                }
            }
        }
    }
    let (mut dag, scan) = graph(&[3], Prim::Int32, 0, ScanDirection::Increasing, false);
    dag.add_root(scan);
    assert!(grad_dag_checked(&dag, scan, &[dag.nodes()[0].id]).is_err());
}
