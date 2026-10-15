//! [05-OP-33]: storage changes preserve the primitive AD graph and its trees.
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor, eval_tensor_roots_with};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::{tier2, tier2_ordered};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn ty(shape: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}
fn graph(shape: [usize; 3], einsum: bool, precision: Prim) -> (Dag, Vec<NodeId>) {
    let [m, k, n] = shape;
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let at = ty(&[m, k], precision);
    let bt = ty(&[k, n], precision);
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
    let acc = precision.default_reduce_sum_accumulator().unwrap();
    let y = if einsum {
        tier2_ordered::lower_einsum(
            owner.into(),
            &mut dag,
            "ij,jk->ik",
            [(a, &at), (b, &bt)],
            Some(acc),
            None,
        )
        .ok()
        .expect("legal einsum")
    } else {
        tier2::lower_matmul_with_accumulator(owner.into(), &mut dag, a, b, &at, &bt, acc, None)
    };
    let s = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 1,
            accumulator: acc,
        },
        vec![y],
        ty(&[m], acc),
        None,
    );
    let loss = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: acc,
        },
        vec![s],
        ty(&[], acc),
        None,
    );
    let result = grad_dag_checked(&dag, loss, &[a, b]).unwrap();
    let roots = vec![
        result.output_node,
        result.grad_nodes[&a],
        result.grad_nodes[&b],
    ];
    let mut dag = result.dag;
    dag.set_roots(roots.clone());
    (dag, roots)
}
fn inputs([m, k, n]: [usize; 3]) -> UnordMap<String, TensorValue> {
    [
        (
            "a".into(),
            TensorValue::from_vec(
                vec![m, k],
                (0..m * k)
                    .map(|i| ((i * 17 % 29) as f64 - 14.0) / 16.0)
                    .collect(),
            ),
        ),
        (
            "b".into(),
            TensorValue::from_vec(
                vec![k, n],
                (0..k * n)
                    .map(|i| ((i * 13 % 31) as f64 - 15.0) / 16.0)
                    .collect(),
            ),
        ),
    ]
    .into_iter()
    .collect()
}
fn same_bits(a: &TensorValue, b: &TensorValue) {
    use chelis_types::dtype_semantics::StorageView;
    fn bits(value: &TensorValue) -> Vec<u64> {
        match value.storage().view() {
            StorageView::F64(v) => v.iter().map(|x| x.to_bits()).collect(),
            StorageView::F32(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
            StorageView::F16(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
            StorageView::Bf16(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
            _ => panic!("float contraction result"),
        }
    }
    assert_eq!(a.shape, b.shape);
    assert_eq!(a.prim(), b.prim());
    assert_eq!(bits(a), bits(b));
}
#[test]
fn rooted_ad_contractions_preserve_dense_primitive_bits() {
    for precision in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16] {
        for einsum in [false, true] {
            for shape in [[3, 5, 7], [2, 0, 3], [0, 3, 2], [2, 3, 0]] {
                let (dag, roots) = graph(shape, einsum, precision);
                let values = inputs(shape);
                let dense = eval_tensor(&dag, &values).unwrap();
                let rooted =
                    eval_tensor_roots_with(&dag, &roots, |name| values.get(name).cloned()).unwrap();
                for root in roots {
                    same_bits(&dense[&root], &rooted[&root]);
                }
            }
        }
    }
}
#[test]
fn an_exposed_label_product_keeps_the_all_values_contract() {
    let (mut dag, mut roots) = graph([3, 5, 7], true, Prim::F32);
    let product = dag
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Mul))
        .unwrap()
        .id;
    roots.push(product);
    dag.set_roots(roots.clone());
    let values = inputs([3, 5, 7]);
    let dense = eval_tensor(&dag, &values).unwrap();
    let rooted = eval_tensor_roots_with(&dag, &roots, |name| values.get(name).cloned()).unwrap();
    assert_eq!(rooted[&product].len(), 105);
    same_bits(&dense[&product], &rooted[&product]);
}
#[test]
fn a_mismatching_load_still_fails_before_ad_views() {
    let (dag, roots) = graph([3, 5, 7], false, Prim::F32);
    let mut values = inputs([3, 5, 7]);
    values.insert("b".into(), TensorValue::from_vec(vec![4, 7], vec![1.0; 28]));
    let dense = eval_tensor(&dag, &values).unwrap_err();
    let rooted =
        eval_tensor_roots_with(&dag, &roots, |name| values.get(name).cloned()).unwrap_err();
    assert_eq!(dense, rooted);
}

#[test]
fn ad_views_admit_metadata_readers_but_keep_exposed_products_dense() {
    for einsum in [false, true] {
        let (dag, roots) = graph([3, 5, 7], einsum, Prim::F32);
        let plan = chelis_ir::contraction_views::contraction_views(&dag, &roots);
        for node in dag.nodes() {
            if matches!(node.op, RiscOp::Mul) {
                assert!(
                    plan.is_deferred(node.id),
                    "product {} should not store its label domain",
                    node.id.0
                );
            }
        }
        let product = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Mul))
            .unwrap()
            .id;
        let readers: Vec<_> = dag
            .nodes()
            .iter()
            .filter(|node| node.inputs.contains(&product))
            .collect();
        assert_eq!(
            readers.len(),
            2,
            "the original forward product has two readers"
        );
        assert!(
            readers
                .iter()
                .any(|node| matches!(node.op, RiscOp::Sum { .. }))
        );
        assert!(readers.iter().any(|node| matches!(
            &node.op,
            RiscOp::Expand {
                size: chelis_ir::dag::RtDim::InputAxis { tensor: 1, .. },
                ..
            }
        ) && node.inputs[1] == product));
        let exposed = chelis_ir::contraction_views::contraction_views(&dag, &[product]);
        assert!(!exposed.is_deferred(product));
    }
}

#[test]
fn ad_view_nonfinite_and_signed_zero_bits_match_stored_operations() {
    for precision in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16] {
        for einsum in [false, true] {
            let (dag, roots) = graph([3, 5, 7], einsum, precision);
            let mut values = inputs([3, 5, 7]);
            values.insert(
                "a".into(),
                TensorValue::from_vec(
                    vec![3, 5],
                    (0..15)
                        .map(|i| match i % 5 {
                            0 => -0.0,
                            1 => 0.0,
                            2 => f64::INFINITY,
                            3 => f64::NAN,
                            _ => f64::from_bits(1),
                        })
                        .collect(),
                ),
            );
            let dense = eval_tensor(&dag, &values).unwrap();
            let rooted =
                eval_tensor_roots_with(&dag, &roots, |name| values.get(name).cloned()).unwrap();
            for root in roots {
                same_bits(&dense[&root], &rooted[&root]);
            }
        }
    }
}

#[test]
fn shared_numeric_diamonds_have_a_materialization_boundary() {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let tensor = ty(&[17], Prim::F32);
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor.clone(),
        None,
    );
    let mut current = x;
    let mut diamonds = Vec::new();
    for _ in 0..12 {
        let product = dag.add_node(
            owner,
            RiscOp::Mul,
            vec![current, current],
            tensor.clone(),
            None,
        );
        let moved = dag.add_node(
            owner,
            RiscOp::Permute { axes: vec![0] },
            vec![product],
            tensor.clone(),
            None,
        );
        let reshaped = dag.add_node(
            owner,
            RiscOp::Reshape {
                new_shape: vec![chelis_ir::dag::RtDim::Lit(17)],
            },
            vec![moved],
            tensor.clone(),
            None,
        );
        diamonds.push(product);
        current = reshaped;
    }
    let sum = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![current],
        ty(&[], Prim::F32),
        None,
    );
    dag.add_root(sum);
    let plan = chelis_ir::contraction_views::contraction_views(&dag, &[sum]);
    for boundary in diamonds.iter().take(11) {
        assert!(
            !plan.is_deferred(*boundary),
            "shared numeric producer needs cached payload"
        );
    }
    // Count the visits a deferred recipe performs for one requested scalar.
    // A dense boundary costs one read: its arithmetic already ran once at
    // its original DAG node. Deferring all nodes is the counterfactual.
    fn visits(
        dag: &Dag,
        node: NodeId,
        plan: Option<&chelis_ir::contraction_views::ContractionViews>,
    ) -> usize {
        if matches!(dag.get(node).unwrap().op, RiscOp::Load { .. })
            || plan.is_some_and(|p| !p.is_deferred(node))
        {
            return 1;
        }
        1 + dag
            .get(node)
            .unwrap()
            .inputs
            .iter()
            .map(|id| visits(dag, *id, plan))
            .sum::<usize>()
    }
    let bounded = visits(&dag, current, Some(&plan));
    let expanded = visits(&dag, current, None);
    assert!(bounded <= 12, "bounded recipe visits={bounded}");
    assert!(
        expanded > 4_000,
        "counterfactual must expose expression-tree duplication"
    );
}

#[test]
fn scalar_broadcast_multiplication_keeps_its_existing_materialization() {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(&[3], Prim::F32),
        None,
    );
    let scalar = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        ty(&[], Prim::F32),
        None,
    );
    let product = dag.add_node(
        owner,
        RiscOp::Mul,
        vec![x, scalar],
        ty(&[3], Prim::F32),
        None,
    );
    let sum = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![product],
        ty(&[], Prim::F32),
        None,
    );
    dag.add_root(sum);
    let plan = chelis_ir::contraction_views::contraction_views(&dag, &[sum]);
    assert!(!plan.is_deferred(product));
    let values = [(
        "x".into(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    )]
    .into_iter()
    .collect();
    let dense = eval_tensor(&dag, &values).unwrap();
    let rooted = eval_tensor_roots_with(&dag, &[sum], |name| values.get(name).cloned()).unwrap();
    same_bits(&dense[&sum], &rooted[&sum]);
}

#[test]
fn nonlinear_cotangents_cache_arithmetic_before_broadcasting() {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let operand = ty(&[3, 3], Prim::F32);
    let a = dag.add_node(
        owner,
        RiscOp::Load { name: "a".into() },
        vec![],
        operand.clone(),
        None,
    );
    let b = dag.add_node(
        owner,
        RiscOp::Load { name: "b".into() },
        vec![],
        operand.clone(),
        None,
    );
    let y = tier2::lower_matmul_with_accumulator(
        owner.into(),
        &mut dag,
        a,
        b,
        &operand,
        &operand,
        Prim::F32,
        None,
    );
    let sine = dag.add_node(owner, RiscOp::Sin, vec![y], operand.clone(), None);
    let row = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![sine],
        ty(&[3], Prim::F32),
        None,
    );
    let loss = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![row],
        ty(&[], Prim::F32),
        None,
    );
    let result = grad_dag_checked(&dag, loss, &[a, b]).unwrap();
    let dag = result.dag;
    let plan = chelis_ir::contraction_views::contraction_views(&dag, dag.roots());
    for node in dag
        .nodes()
        .iter()
        .filter(|node| node.output_type.dims.len() == 3)
    {
        assert!(
            plan.is_deferred(node.id),
            "nonlinear cotangent must not cache a broadcast domain: {} {:?}",
            node.id.0,
            node.op
        );
    }
    let values = inputs([3, 3, 3]);
    let dense = eval_tensor(&dag, &values).unwrap();
    let rooted =
        eval_tensor_roots_with(&dag, dag.roots(), |name| values.get(name).cloned()).unwrap();
    for root in dag.roots() {
        same_bits(&dense[root], &rooted[root]);
    }
}

#[test]
fn mapped_ad_contractions_preserve_each_batch_tree_and_empty_batches() {
    for einsum in [false, true] {
        for batch in [0usize, 1, 3] {
            for grad_after_map in [false, true] {
                let (dag, _) = graph([3, 5, 7], einsum, Prim::F32);
                let mut mapped =
                    chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(batch)).unwrap();
                if grad_after_map {
                    let owner = mapped.declare("batch_loss");
                    let loss = mapped.add_node(
                        owner,
                        RiscOp::Sum {
                            axis: 0,
                            accumulator: Prim::F32,
                        },
                        vec![mapped.roots()[0]],
                        ty(&[], Prim::F32),
                        None,
                    );
                    let loads = ["a", "b"].map(|label| {
                        mapped
                            .nodes()
                            .iter()
                            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == label))
                            .unwrap()
                            .id
                    });
                    let gradient = grad_dag_checked(&mapped, loss, &loads).unwrap();
                    let roots = vec![
                        gradient.output_node,
                        gradient.grad_nodes[&loads[0]],
                        gradient.grad_nodes[&loads[1]],
                    ];
                    mapped = gradient.dag;
                    mapped.set_roots(roots);
                }
                let source_values = inputs([3, 5, 7]);
                let values: UnordMap<String, TensorValue> = ["a", "b"]
                    .into_iter()
                    .map(|name| {
                        let value = &source_values[name];
                        let mut shape = vec![batch];
                        shape.extend(&value.shape);
                        let data = value.to_f64_lossy_vec();
                        (
                            name.into(),
                            TensorValue::from_vec(
                                shape,
                                (0..batch)
                                    .flat_map(|i| data.iter().map(move |x| x + i as f64 / 16.0))
                                    .collect(),
                            ),
                        )
                    })
                    .collect();
                let dense = eval_tensor(&mapped, &values).unwrap();
                let rooted = eval_tensor_roots_with(&mapped, mapped.roots(), |name| {
                    values.get(name).cloned()
                })
                .unwrap();
                for root in mapped.roots() {
                    same_bits(&rooted[root], &dense[root]);
                }
                let plan = chelis_ir::contraction_views::contraction_views(&mapped, mapped.roots());
                let deferred = mapped
                    .nodes()
                    .iter()
                    .filter(|node| plan.is_deferred(node.id))
                    .count();
                assert!(
                    deferred > 0,
                    "mapped contractions must use the storage plan"
                );
                let mut malformed = values.clone();
                malformed.insert(
                    "a".into(),
                    TensorValue::from_vec(vec![batch, 3, 4], vec![0.0; batch * 12]),
                );
                assert_eq!(
                    eval_tensor(&mapped, &malformed).unwrap_err(),
                    eval_tensor_roots_with(&mapped, mapped.roots(), |name| malformed
                        .get(name)
                        .cloned())
                    .unwrap_err()
                );
            }
        }
    }
}
