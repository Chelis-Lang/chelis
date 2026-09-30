use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::vmap::{
    vectorize_axis0, vectorize_axis0_with_node_map, vectorize_axis0_with_node_map_and_captures,
};
use chelis_types::check_ir_program;
use chelis_types::types::Prim;
use chelis_unord::{UnordMap, UnordSet};

/// #2003: a metadata read is scalar even without a movement-bound consumer.
#[test]
fn vmap_ordinary_shape_roots_and_consumers_keep_the_scalar_operation() {
    for batch in [
        DimInfo::Lit(2),
        DimInfo::Lit(0),
        DimInfo::Named("batch".into(), None),
    ] {
        let count = if batch == DimInfo::Lit(0) { 0 } else { 2 };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(3),
            None,
        );
        let shape = dag.add_node(
            decl,
            RiscOp::Shape { axis: 0 },
            vec![x],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        let cast = dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![shape],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        let sum = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![x],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        let mixed = dag.add_node(
            decl,
            RiscOp::Add,
            vec![sum, cast],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        for root in [shape, cast, mixed] {
            dag.add_root(root);
        }
        let mapped = vectorize_axis0(&dag, batch).unwrap();
        let reads: Vec<_> = mapped
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, RiscOp::Shape { .. }))
            .collect();
        assert_eq!(reads.len(), 1);
        assert!(reads[0].output_type.dims.is_empty());
        assert!(matches!(reads[0].op, RiscOp::Shape { axis: 1 }));
        assert!(chelis_ir::verify::verify(&mapped).is_empty());
        chelis_ir::axis_sources::check_axis_sources(
            &mapped,
            chelis_types::unsupported::Stage::Runtime,
        )
        .unwrap();
        let values = eval_tensor_roots_with_strict(&mapped, mapped.roots(), |name| {
            (name == "x").then(|| TensorValue::from_vec(vec![count, 3], vec![2.0; count * 3]))
        })
        .unwrap();
        for (root, precision, expected) in [
            (mapped.roots()[0], Prim::Int64, 3.0),
            (mapped.roots()[1], Prim::F32, 3.0),
            (mapped.roots()[2], Prim::F32, 9.0),
        ] {
            assert_eq!(values[&root].shape, vec![count]);
            assert_eq!(values[&root].prim(), precision);
            assert_eq!(values[&root].to_f64_lossy_vec(), vec![expected; count]);
        }
    }
}

#[test]
fn vmap_ordinary_shape_rejects_a_malformed_ranked_read() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let bad = dag.add_node(
        decl,
        RiscOp::Shape { axis: 0 },
        vec![x],
        TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_root(bad);
    let error = vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap_err();
    assert!(
        error.contains("produces rank 1 rather than one shared scalar"),
        "{error}"
    );
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn mat_f32(m: usize, n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn eval_root(dag: &Dag, inputs: &UnordMap<String, TensorValue>) -> TensorValue {
    let root = dag.roots()[0];
    let values = eval_tensor_roots_with_strict(dag, &[root], |name| inputs.get(name).cloned())
        .expect("evaluation should succeed");
    values[&root].clone()
}

fn sparse_type(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

#[test]
fn vmap_gather_reduces_each_row_and_preserves_result_rank() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let values = dag.add_node(
        decl,
        RiscOp::Load {
            name: "values".into(),
        },
        vec![],
        sparse_type(&[4], Prim::F32),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        sparse_type(&[2], Prim::Int64),
        None,
    );
    let gather = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![values, indices],
        sparse_type(&[2], Prim::F32),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![gather],
        sparse_type(&[], Prim::F32),
        None,
    );
    dag.add_root(sum);
    let mapped = vectorize_axis0(&dag, DimInfo::Lit(3)).unwrap();
    assert!(matches!(
        mapped.get(gather).unwrap().op,
        RiscOp::Gather {
            axis: 1,
            batch_rank: 1
        }
    ));
    let errors = chelis_ir::verify::verify(&mapped);
    assert!(errors.is_empty(), "{errors:?}");
    let value = eval_root(
        &mapped,
        &UnordMap::from([
            (
                "values".into(),
                TensorValue::from_vec(
                    vec![3, 4],
                    vec![
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ],
                ),
            ),
            (
                "indices".into(),
                TensorValue::from_vec(vec![3, 2], vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0]),
            ),
        ]),
    );
    assert_eq!(value.shape, vec![3]);
    assert_eq!(value.to_f64_lossy_vec(), vec![4.0, 12.0, 20.0]);
}

#[test]
fn vmap_sparse_indices_check_the_row_extent() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let values = dag.add_node(
        decl,
        RiscOp::Load {
            name: "values".into(),
        },
        vec![],
        sparse_type(&[4], Prim::F32),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        sparse_type(&[1], Prim::Int64),
        None,
    );
    let gather = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![values, indices],
        sparse_type(&[1], Prim::F32),
        None,
    );
    dag.add_root(gather);
    let mapped = vectorize_axis0(&dag, DimInfo::Lit(5)).unwrap();
    let inputs = UnordMap::from([
        (
            "values".into(),
            TensorValue::from_vec(vec![5, 4], vec![1.0; 20]),
        ),
        (
            "indices".into(),
            TensorValue::from_vec(vec![5, 1], vec![4.0; 5]),
        ),
    ]);
    // chelis#1636 still renders this as a panic. The check must nevertheless
    // reject index 4 against each size-4 row, not accept it against batch 5.
    assert!(std::panic::catch_unwind(|| eval_root(&mapped, &inputs)).is_err());
}

#[test]
fn vmap_scatter_family_shifts_the_target_axis() {
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
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let base = dag.add_node(
            decl,
            RiscOp::Load {
                name: "base".into(),
            },
            vec![],
            sparse_type(&[4], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            sparse_type(&[2], Prim::Int64),
            None,
        );
        let updates = dag.add_node(
            decl,
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            sparse_type(&[2], Prim::F32),
            None,
        );
        let scatter = dag.add_node(
            decl,
            op.clone(),
            vec![base, indices, updates],
            sparse_type(&[4], Prim::F32),
            None,
        );
        dag.add_root(scatter);
        let mapped = vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
        assert!(
            matches!(
                mapped.get(scatter).unwrap().op,
                RiscOp::ScatterAdd {
                    axis: 1,
                    batch_rank: 1
                } | RiscOp::Scatter {
                    axis: 1,
                    batch_rank: 1
                } | RiscOp::ScatterElements { axis: 1 }
            ),
            "{op:?}"
        );
        let errors = chelis_ir::verify::verify(&mapped);
        assert!(errors.is_empty(), "{op:?}: {errors:?}");
        let value = eval_root(
            &mapped,
            &UnordMap::from([
                (
                    "base".into(),
                    TensorValue::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                ),
                (
                    "indices".into(),
                    TensorValue::from_vec(vec![2, 2], vec![0.0, 2.0, 0.0, 2.0]),
                ),
                (
                    "updates".into(),
                    TensorValue::from_vec(vec![2, 2], vec![9.0, 7.0, 11.0, 13.0]),
                ),
            ]),
        );
        let expected = if matches!(op, RiscOp::ScatterAdd { .. }) {
            vec![10.0, 2.0, 10.0, 4.0, 16.0, 6.0, 20.0, 8.0]
        } else {
            vec![9.0, 2.0, 7.0, 4.0, 11.0, 6.0, 13.0, 8.0]
        };
        assert_eq!(value.shape, vec![2, 4], "{op:?}");
        assert_eq!(value.to_f64_lossy_vec(), expected, "{op:?}");
    }
}

#[test]
fn vmap_one_hot_keeps_the_vocabulary_axis_last() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        sparse_type(&[2], Prim::Int64),
        None,
    );
    let one_hot = dag.add_node(
        decl,
        RiscOp::OneHot { vocab: 4 },
        vec![indices],
        sparse_type(&[2, 4], Prim::F32),
        None,
    );
    dag.add_root(one_hot);
    let mapped = vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
    assert!(chelis_ir::verify::verify(&mapped).is_empty());
    let value = eval_root(
        &mapped,
        &UnordMap::from([(
            "indices".into(),
            TensorValue::from_vec(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]),
        )]),
    );
    assert_eq!(value.shape, vec![2, 2, 4]);
    assert_eq!(
        value.to_f64_lossy_vec(),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]
    );
}

#[test]
fn nested_vmap_pairs_each_sparse_batch_axis_once() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let values = dag.add_node(
        decl,
        RiscOp::Load {
            name: "values".into(),
        },
        vec![],
        sparse_type(&[4], Prim::F32),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        sparse_type(&[2], Prim::Int64),
        None,
    );
    let gather = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![values, indices],
        sparse_type(&[2], Prim::F32),
        None,
    );
    dag.add_root(gather);
    let inner = vectorize_axis0(&dag, DimInfo::Lit(3)).unwrap();
    let outer = vectorize_axis0(&inner, DimInfo::Lit(2)).unwrap();
    let sparse = outer
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Gather { .. }))
        .unwrap();
    assert!(matches!(
        sparse.op,
        RiscOp::Gather {
            axis: 2,
            batch_rank: 2
        }
    ));
    let errors = chelis_ir::verify::verify(&outer);
    assert!(errors.is_empty(), "{errors:?}");
    let value = eval_root(
        &outer,
        &UnordMap::from([
            (
                "values".into(),
                TensorValue::from_vec(vec![2, 3, 4], (1..=24).map(f64::from).collect()),
            ),
            (
                "indices".into(),
                TensorValue::from_vec(vec![2, 3, 2], [0.0, 2.0].repeat(6)),
            ),
        ]),
    );
    assert_eq!(value.shape, vec![2, 3, 2]);
    assert_eq!(
        value.to_f64_lossy_vec(),
        vec![
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0
        ]
    );
}

#[test]
fn sparse_paired_prefix_must_match_in_verifier() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let values = dag.add_node(
        decl,
        RiscOp::Load {
            name: "values".into(),
        },
        vec![],
        sparse_type(&[3, 4], Prim::F32),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        sparse_type(&[2, 2], Prim::Int64),
        None,
    );
    let gather = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 1,
            batch_rank: 1,
        },
        vec![values, indices],
        sparse_type(&[3, 2], Prim::F32),
        None,
    );
    dag.add_root(gather);
    let errors = chelis_ir::verify::verify(&dag);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("invalid paired batch prefix")),
        "{errors:?}"
    );
}

#[test]
fn vmap_elementwise_vectorizes_axis_zero() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let y = dag.add_node(decl, RiscOp::Neg, vec![x], vec_f32(3), None);
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let value = eval_root(
        &vmapped,
        &UnordMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0]),
        )]),
    );
    assert_eq!(value.shape, vec![2, 3]);
    assert_eq!(
        value.to_f64_lossy_vec(),
        vec![-1.0, -2.0, -3.0, -4.0, 5.0, -6.0]
    );
}

#[test]
fn vmap_capture_keeps_authored_load_rank_and_maps_to_explicit_insert() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(2),
        None,
    );
    let w = dag.add_node(
        decl,
        RiscOp::Load { name: "w".into() },
        vec![],
        vec_f32(2),
        None,
    );
    let product = dag.add_node(decl, RiscOp::Mul, vec![x, w], vec_f32(2), None);
    dag.add_root(product);

    let captures = UnordSet::from(["w".to_string()]);
    let (mapped, node_map) =
        vectorize_axis0_with_node_map_and_captures(&dag, DimInfo::Lit(3), &captures)
            .expect("capture-aware vmap");
    let lifted = mapped.get(node_map[w.0]).expect("mapped capture identity");
    assert!(
        matches!(
            lifted.op,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(3)
            }
        ),
        "capture identity must be an explicit rank-inserting movement: {lifted:?}"
    );
    let raw = mapped.get(lifted.inputs[0]).expect("raw capture load");
    assert_eq!(raw.output_type, vec_f32(2));
    assert!(matches!(&raw.op, RiscOp::Load { name } if name == "w"));
    assert_eq!(
        mapped
            .get(node_map[x.0])
            .expect("mapped formal")
            .output_type,
        mat_f32(3, 2),
        "an actual mapped formal still receives the batch axis"
    );
    assert!(chelis_ir::verify::verify(&mapped).is_empty());

    let value = eval_root(
        &mapped,
        &UnordMap::from([
            (
                "x".to_string(),
                TensorValue::from_vec(vec![3, 2], vec![1.0, 1.0, 2.0, 2.0, 0.0, 1.0]),
            ),
            (
                "w".to_string(),
                TensorValue::from_vec(vec![2], vec![10.0, 20.0]),
            ),
        ]),
    );
    assert_eq!(value.shape, vec![3, 2]);
    assert_eq!(
        value.to_f64_lossy_vec(),
        vec![10.0, 20.0, 20.0, 40.0, 0.0, 20.0]
    );
}

#[test]
fn vmap_broadcasts_nonshared_constant_tensors_with_a_real_batched_identity() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let constant = dag.add_node(
        decl,
        RiscOp::synth_const_tensor(Prim::F32, vec![10.0, 20.0, 30.0]),
        vec![],
        vec_f32(3),
        None,
    );
    let sum = dag.add_node(decl, RiscOp::Add, vec![x, constant], vec_f32(3), None);
    dag.add_root(sum);

    let (mapped, node_map) =
        vectorize_axis0_with_node_map(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let mapped_constant = node_map[constant.0];
    let mapped_node = mapped
        .get(mapped_constant)
        .expect("mapped constant identity");
    assert!(
        matches!(
            mapped_node.op,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2)
            }
        ),
        "the original constant identity must map to its batch broadcast, got {mapped_node:?}"
    );
    let raw_constant = mapped_node.inputs[0];
    let raw_node = mapped.get(raw_constant).expect("raw constant payload");
    assert_eq!(raw_node.output_type, vec_f32(3));
    assert!(
        matches!(&raw_node.op, RiscOp::ConstTensor { data } if data.len() == 3),
        "the raw payload must retain its original unbatched type and cardinality"
    );
    assert!(
        chelis_ir::verify::verify(&mapped).is_empty(),
        "mapped graph must satisfy structural verification: {:?}",
        chelis_ir::verify::verify(&mapped)
    );

    let value = eval_root(
        &mapped,
        &UnordMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )]),
    );
    assert_eq!(value.shape, vec![2, 3]);
    assert_eq!(
        value.to_f64_lossy_vec(),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
}

#[test]
fn vmap_rejects_a_shape_dependency_without_a_mapped_identity() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let value = dag.add_node(
        decl,
        RiscOp::synth_const_tensor(Prim::F32, vec![1.0, 2.0]),
        vec![],
        vec_f32(2),
        None,
    );
    dag.node_mut(value)
        .expect("constant")
        .shape_deps
        .push(chelis_ir::dag::NodeId(99));
    dag.add_root(value);

    let error = vectorize_axis0_with_node_map(&dag, DimInfo::Lit(2)).unwrap_err();
    assert!(
        error.contains("shape dependency") && error.contains("no mapped identity"),
        "{error}"
    );
}

#[test]
fn vmap_reduction_shifts_the_reduced_axis() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(2),
        None,
    );
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let value = eval_root(
        &vmapped,
        &UnordMap::from([(
            "x".to_string(),
            TensorValue::from_vec(
                vec![2, 2, 3],
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 8.0, 9.0,
                ],
            ),
        )]),
    );
    assert_eq!(value.shape, vec![2, 2]);
    assert_eq!(value.to_f64_lossy_vec(), vec![6.0, 15.0, 60.0, 24.0]);
}

#[test]
fn vmap_nested_adds_multiple_batch_axes() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    dag.add_root(x);

    let inner = vectorize_axis0(&dag, DimInfo::Lit(3)).expect("inner vmap should succeed");
    let outer = vectorize_axis0(&inner, DimInfo::Lit(2)).expect("outer vmap should succeed");
    let root = outer.roots()[0];
    let node = outer.get(root).expect("root");
    assert_eq!(
        node.output_type,
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        }
    );
}

#[test]
fn vmap_batched_matmul_stays_in_expand_mul_sum_form() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let a_exp = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let b_exp = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let prod = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![a_exp, b_exp],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![prod],
        mat_f32(2, 4),
        None,
    );
    dag.add_root(out);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(5)).expect("vmap should succeed");
    assert!(
        vmapped
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Expand { axis: 3, .. })),
        "batched matmul should stay in the generic expand->mul->sum decomposition"
    );
}

// =====================================================================
// chelis#524: a non-constant `vmap` mapped (batch) axis must FAIL CLOSED
// at lowering — a loud, LOCATED, FATAL diagnostic — never silently
// default to axis 0.
//
// `resolve_callable_expr_inner`'s `Some("vmap")` arm read the axis with
// `kids.get(1).and_then(extract_usize_value).unwrap_or(0)`, which
// collapsed BOTH "axis arg absent" (the documented vmap default) AND
// "axis arg present but non-constant" to axis 0. A runtime mapped axis
// then silently vectorized the WRONG axis — the same default-to-0
// anti-pattern #364 eliminated for reductions / softmax / gather/scatter.
//
// The surf parser only admits a literal `axis=N`, so a runtime axis is
// reachable through the Deep (`.dp`) input path (or any internal IR
// transform): pre-fix `chelis build` succeeded, silently lowering the `vmap`
// over axis 0. Negative parity with #364's `extract_axis_raw`.
//
// chelis#874 (PP8) MOVED THIS REJECTION TO CHECK TIME, and the rest of this
// block is the record of that. #524's program no longer checks clean: the
// `vmap` axis is a `Selector` slot, and reading a bound `i32` parameter
// there is a [04-TOT-3]/[04-TOT-4] malformed form. Three sentences decide it,
// and none of them is either test:
//
//   * spec/06-transformations.md section 3.1 and section 3.4 type `vmap(f, axis=n)` as
//     `tensor[D with batch inserted at n, P] -> tensor[D' with batch inserted
//     at n, P]`. The result type's dimension ORDER is a function of `n`, so an
//     unknown `n` leaves not merely the extents but the positional sequence
//     undetermined. `concat` is the documented contrast: spec/04-type-system.md
//     section 4.5.4 rule 5 admits a dynamic axis precisely because its result can
//     type every axis `*` at a known rank. `vmap` is given no such rule.
//   * spec/06-transformations.md section 3.6 and section 8.4 REQUIRE the
//     `axis_out_of_bounds` diagnostic, "vmap axis 2 is out of bounds for rank 1
//     tensor", keyed on the axis's value against a rank. A check-time
//     diagnostic that names the value presupposes the value at check time.
//   * spec/04-type-system.md section 4.5.3 names the general rule: "a bound
//     `i32` variable is a runtime value and keeps the static-axis rule".
//
// chelis#259 had already reached the same conclusion for the reduce/expand
// family on the same ground -- "the dimension at position `axis` is removed"
// is undeterminable without a concrete axis -- and rejected a non-literal axis
// at check time. `vmap` was simply never given the rejection its siblings got.
// `lower.rs`'s own guard says so: "a non-constant axis stays a check-time
// rejection (#259 family), so reaching this site with an unresolvable axis is
// an internal contract violation".
//
// So the test below inverts: same program, same defect, decided one stage
// earlier. The three constant-axis controls that follow are untouched and
// still reach the lowering path this block was written to protect.
// =====================================================================

/// A `vmap` over `tensor[batch, features]` whose mapped axis is the bound
/// runtime parameter `ax: i32` (a non-constant).
const VMAP_RUNTIME_AXIS_DEEP: &str = r#"
(defsig {} process
  (t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32))
           (t-tensor {} (d-name {} features) (t-prim {} f32))))
(def {} process
  (fn {} (params {} (x {type: (t-tensor {} (d-name {} features) (t-prim {} f32))}))
    (app {} (var {} relu) (var {} x))))
(defsig {} batch_process
  (t-fn {} (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
           (t-prim {} i32)
           (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
(def {} batch_process
  (fn {} (params {}
           (xs {type: (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))})
           (ax {type: (t-prim {} i32)}))
    (pipe {} (var {} xs)
      (vmap {} (var {} process) (var {span: "dp:vmap-runtime-axis"} ax)))))
"#;

/// The same program with a CONSTANT (`lit`) mapped axis — the negative
/// parity control. Identical except the axis node is a literal, so it must
/// lower cleanly.
const VMAP_CONST_AXIS_DEEP: &str = r#"
(defsig {} process
  (t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32))
           (t-tensor {} (d-name {} features) (t-prim {} f32))))
(def {} process
  (fn {} (params {} (x {type: (t-tensor {} (d-name {} features) (t-prim {} f32))}))
    (app {} (var {} relu) (var {} x))))
(defsig {} batch_process
  (t-fn {} (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
           (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
(def {} batch_process
  (fn {} (params {}
           (xs {type: (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))}))
    (pipe {} (var {} xs)
      (vmap {} (var {} process) (lit {type: (t-prim {} i32)} 0)))))
"#;

fn check_effects_linearity_deep(deep_src: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(deep_src).expect("deep parse");
    let checked = check_ir_program(&exprs)
        .unwrap_or_else(|r| panic!("deep program must check clean: {:?}", r.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

fn lower_surf_program(src: &str) -> Result<Dag, chelis_ir::lower::LowerDiagnostic> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("expand")
    .into_exprs();
    let checked = check_ir_program(&exprs)
        .unwrap_or_else(|r| panic!("surf program must check clean: {:?}", r.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    chelis_ir::lower::try_lower_program(&checked)
}

/// REJECT: a present-but-non-constant `vmap` mapped axis is rejected by the
/// CHECKER, naming `vmap` and the shape the slot expects — not silently
/// defaulted to axis 0 (which pre-fix let `chelis build` succeed), and no
/// longer deferred to lowering.
///
/// This is chelis#524's program and chelis#524's defect. What moved is the
/// stage: chelis#874 (PP8) made the `vmap` axis a typed `Selector` read, and
/// the block comment above records the three spec sentences that decide the
/// axis is static. #524's own class -- no silent default to axis 0 -- is
/// preserved and strengthened, since the program is now rejected before
/// effects, linearity, and lowering ever run.
///
/// Renamed from `issue524_runtime_vmap_axis_is_fatal_located_lowering_error`.
/// Its two lost assertions are recorded honestly rather than reconstructed:
/// the checker's rejection is not `fatal`-flagged, because that flag belongs
/// to the lowering diagnostic type and has no analogue here, and it carries no
/// `span_id`, because `MalformedForm` diagnostics from the slot seam are not
/// span-located. Neither is a PP8 claim.
///
/// WHICH GUARD LOST ITS WITNESS, precisely, because a maintainer reading this
/// must not delete the wrong thing. It is the `DeepTag::Vmap` arm of
/// `resolve_callable_expr_inner` (`chelis-ir/src/lower.rs:6716-6742`), which
/// reads the axis with `extract_usize_value` and raises a fatal, located
/// error whose message opens "`vmap` mapped axis is not a compile-time integer
/// constant" and whose `span_id` is the axis node's. That message and that
/// span are exactly what the deleted assertions required.
///
/// It is NOT `extract_axis_raw`, which this file's earlier note misnamed.
/// `extract_axis_raw` is never called for `vmap`: its call sites pass
/// "gather", "scatter_replace", "scatter_elements", "softmax", and the
/// reduction path's forwarded `op`, so its message template cannot contain
/// "vmap" and could never have satisfied the deleted assertion. That
/// function's own lack of a message witness is pre-existing and is not
/// something this change created.
///
/// AND THE GUARD IS NOT DEAD CODE. The chelis#524 block comment above names
/// two routes to it: the `.dp` input path, and "any internal IR transform".
/// Check time closes only the first. "Unreachable from any checked program"
/// is therefore true but strictly narrower than "unreachable": an internal IR
/// transform that synthesizes a `vmap` node with a non-constant axis still
/// reaches this arm after the checker has passed, and that second route is
/// the one now unwitnessed. No fixture is owed for it -- witnessing a
/// fail-loud internal-contract guard would mean synthesizing IR no checked
/// program can produce -- and PR #1602 records it as residual scope.
#[test]
fn issue524_runtime_vmap_axis_is_rejected_at_check_time() {
    let exprs = chelis_deep::parser::parse_str(VMAP_RUNTIME_AXIS_DEEP).expect("deep parse");
    let Err(result) = check_ir_program(&exprs) else {
        panic!("a runtime vmap axis must REJECT at check, not check clean and default to 0");
    };
    let malformed: Vec<&chelis_types::errors::CheckError> = result
        .errors
        .iter()
        .filter(|e| matches!(e.kind, chelis_types::errors::CheckErrorKind::MalformedForm))
        .collect();
    assert!(
        !malformed.is_empty(),
        "the runtime axis must be a MalformedForm ([04-TOT-4]); got {:?}",
        result.errors
    );
    assert!(
        malformed
            .iter()
            .any(|e| e.message.contains("vmap") && e.message.contains("integer axis")),
        "the diagnostic must name `vmap` and the expected shape so the author \
         can see what to write; got {:?}",
        result.errors
    );
}

/// CONTROL (constant axis): the same program with a `lit` axis lowers
/// cleanly — FAIL-CLOSED must not become reject-everything.
#[test]
fn issue524_constant_vmap_axis_still_lowers() {
    let checked = check_effects_linearity_deep(VMAP_CONST_AXIS_DEEP);
    let dag = chelis_ir::lower::try_lower_program(&checked)
        .expect("a constant vmap axis must still lower cleanly");
    assert!(!dag.nodes().is_empty(), "lowered DAG must be non-empty");
}

/// CONTROL (absent axis): the documented `vmap` contract — an OMITTED axis
/// argument defaults to axis 0 — must still hold. `vmap(process)` lowers
/// cleanly (the `None` branch, not the rejected non-constant branch).
#[test]
fn issue524_absent_vmap_axis_defaults_to_zero_and_lowers() {
    let src = "def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)\n\
         def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = xs |> vmap(process)\n";
    let dag = lower_surf_program(src).expect("an absent vmap axis must default to 0 and lower");
    assert!(!dag.nodes().is_empty(), "lowered DAG must be non-empty");
}

/// CONTROL (explicit nonzero constant axis via Surf): the canonical named
/// `axis=1` form lowers cleanly.
#[test]
fn issue524_explicit_nonzero_constant_vmap_axis_via_surf_lowers() {
    let src = "def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)\n\
         def batch_process(xs: tensor[features, batch, f32]) -> tensor[features, batch, f32] = xs |> vmap(process, axis=1)\n";
    let dag = lower_surf_program(src).expect("an explicit constant vmap axis must lower");
    assert!(!dag.nodes().is_empty(), "lowered DAG must be non-empty");
}

/// chelis#1821: `vectorize_axis0_with_node_map` names the batched id of every
/// input node.
///
/// The lowering of `vmap(grad(...))` has to record the batched forward
/// activation as a shape dependency of each batched cotangent, and the only id
/// it holds is the one the unbatched gradient DAG used. The rebuild is not
/// id-preserving: a shared extent scalar consumed as an ordinary value gains a
/// batch-expansion node, which shifts every id after it. The fixture below is
/// exactly that shape, and its `sqrt` node's batched id is one higher than its
/// own, so a caller reusing its input id would name the expansion instead.
///
/// The contract is one entry per input node, in input-node order, each naming a
/// node that exists in the batched DAG and carries the same operator family.
///
/// EVIDENTIARY STATUS: disposition lock on a new API. There is no prior
/// behaviour to regress; the row exists so a future rebuild that inserts nodes
/// cannot silently return a map that has drifted from its own DAG.
#[test]
fn the_vmap_node_map_names_every_input_nodes_batched_id() {
    use chelis_ir::dag::{NodeId, RtDim};
    use chelis_ir::vmap::vectorize_axis0_with_node_map;

    let int64_scalar = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let f32_scalar = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    // One shape read serving a movement bound, which is what marks it shared.
    let extent = dag.add_node(decl, RiscOp::Shape { axis: 0 }, vec![x], int64_scalar, None);
    let shrunk = dag.add_node(
        decl,
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Node(1))],
        },
        vec![x, extent],
        vec_f32(3),
        None,
    );
    // `cast` is a shared scalar operation, so the shared chain reaches here...
    let as_f32 = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![extent],
        f32_scalar.clone(),
        None,
    );
    // ... and stops: `sqrt` is not one, so it is batched and its shared operand
    // is expanded to the batch, inserting a node ahead of it.
    let root = dag.add_node(decl, RiscOp::Sqrt, vec![as_f32], f32_scalar, None);
    dag.add_root(shrunk);
    dag.add_root(root);

    let (batched, map) = vectorize_axis0_with_node_map(&dag, DimInfo::Lit(2))
        .expect("a shared extent scalar with an ordinary consumer vectorizes");

    assert_eq!(
        map.len(),
        dag.len(),
        "one entry per input node, in input-node order"
    );
    for node in dag.nodes() {
        let mapped = map[node.id.0];
        let batched_node = batched.get(mapped).unwrap_or_else(|| {
            panic!(
                "{:?} maps to {mapped:?}, which the batched DAG lacks",
                node.id
            )
        });
        assert_eq!(
            std::mem::discriminant(&batched_node.op),
            std::mem::discriminant(&node.op),
            "{:?} must map to a node of its own operator family, not to whichever \
             node happens to sit at that index after an insertion",
            node.id
        );
    }

    // The insertion is what makes the map necessary rather than decorative.
    assert_eq!(
        batched.len(),
        dag.len() + 1,
        "the ordinary use of the shared extent inserts one batch expansion"
    );
    assert_eq!(
        map[root.0],
        NodeId(root.0 + 1),
        "so the node after the insertion does not keep its own id"
    );
}

/// chelis#1821 on the mapped path: the batched cotangent carries a shape
/// dependency on the batched forward activation.
///
/// The lowering records that edge after `vmap(grad(...))`'s splice, for the
/// same reason the unmapped path does: only the cotangents are rooted in the
/// parent DAG, so the entry-point dead-code elimination would otherwise remove
/// the forward activation and the carrier holding its extent obligations.
///
/// This structural row asserts the edge rather than a program's output.
/// #2112's entry-witness matrix supplies the separate end-to-end Eval/C
/// receipt, including a named witness through `vmap(grad(...))`.
///
/// The program has two roots: the exported `h` kernel's own unbatched
/// reduction, and the mapped cotangent. Only the second carries a shape
/// dependency, and it names the BATCHED reduction, the one whose reduced axis
/// vmap shifted past the batch axis. Asserting that signature rather than a
/// root index is what makes the row independent of root order.
///
/// EVIDENTIARY STATUS: regression test for the dependency's presence. The
/// nonidentity mapped-axis receipt added by #2144 and #2112's end-to-end
/// matrix establish properties this structural row alone cannot prove.
#[test]
fn vmap_grad_records_the_batched_forward_activation_as_a_shape_dep() {
    let source = "def h(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
                  def main() = vmap(grad(h))(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_ir_program(&exprs).expect("check");
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    let dag = chelis_ir::lower::try_lower_program(&checked).expect("lower vmap(grad(h))");

    let carrying: Vec<_> = dag
        .roots()
        .iter()
        .copied()
        .filter(|root| !dag.get(*root).expect("root node").shape_deps.is_empty())
        .collect();
    assert_eq!(
        carrying.len(),
        1,
        "exactly one root, the mapped cotangent, depends on the batched forward \
         activation; found {carrying:?} among roots {:?}",
        dag.roots()
    );
    let cotangent = dag.get(carrying[0]).expect("cotangent node");
    for dep in &cotangent.shape_deps {
        assert!(
            !cotangent.inputs.contains(dep),
            "a dependency the cotangent already reads as a value input would prove \
             nothing about retention; {dep:?} is such an input"
        );
        let node = dag.get(*dep).expect("shape dependency survives lowering");
        assert!(
            matches!(node.op, RiscOp::Sum { axis: 1, .. }),
            "the dependency names the batched forward reduction, got {:?}",
            node.op
        );
    }
}
