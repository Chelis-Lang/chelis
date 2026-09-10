//! [04] section 4.7: independently observed extents precede shape refinement.
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::optimize::{common_subexpr_eliminate, constant_fold, dead_code_eliminate};
use chelis_types::{scalar_from_i64, types::Prim};

fn scalar_type() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::Int64,
    }
}
fn scalar(dag: &mut Dag, value: i64) -> NodeId {
    dag.add_node(
        RiscOp::Const {
            value: scalar_from_i64("reshape", Prim::Int64, value).unwrap(),
        },
        vec![],
        scalar_type(),
        None,
    )
}
fn checked_extent(dag: &mut Dag, actual: i64, required: i64) -> NodeId {
    let actual = scalar(dag, actual);
    let required = scalar(dag, required);
    dag.add_node(
        RiscOp::CheckedReshapeExtent {
            claims: vec!["rows".into()],
            axis: RtAxis::Lit(0),
        },
        vec![actual, required],
        scalar_type(),
        Some("reshape-call".into()),
    )
}
fn unit_axis(dag: &mut Dag) -> (NodeId, NodeId, NodeId) {
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("unit".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let witness = dag.add_node(
        RiscOp::ExtentWitness {
            site: chelis_ir::dag::ExtentWitnessSite::Caller,
            parameter: "b".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![scalar_from_i64("load", Prim::Int64, 1).unwrap()],
        },
        vec![x],
        scalar_type(),
        Some("expand-call".into()),
    );
    let checked = dag.add_node(
        RiscOp::CheckedUnitAxis {
            axis: RtAxis::Lit(0),
        },
        vec![x, witness],
        TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        },
        Some("expand-call".into()),
    );
    (x, witness, checked)
}
#[test]
fn computed_claim_checks_independent_values_before_returning_a_scalar() {
    for actual in [2, 3] {
        let mut dag = Dag::new();
        let checked = checked_extent(&mut dag, actual, 2);
        dag.add_root(checked);
        assert!(chelis_ir::verify::verify(&dag).is_empty());
        let result = eval_tensor_with(&dag, |_| None);
        if actual == 2 {
            let values = result.unwrap();
            assert!(values[&checked].shape.is_empty());
            assert_eq!(
                values[&checked].storage().scalar_at(0).as_i64_exact(),
                Some(2)
            );
        } else {
            let error = result.unwrap_err();
            assert!(error.contains("claimed = 2"), "{error}");
            assert!(error.contains("axis 0 = 3"), "{error}");
            assert!(
                error
                    .lines()
                    .any(|line| line == "numeric trap: domain in reshape at int64"),
                "{error}"
            );
        }
    }
}
#[test]
fn multiple_claims_keep_each_requirement_and_reject_missing_edges() {
    for second in [2, 3] {
        let mut dag = Dag::new();
        let actual = scalar(&mut dag, 2);
        let inner = scalar(&mut dag, 2);
        let outer = scalar(&mut dag, second);
        let checked = dag.add_node(
            RiscOp::CheckedReshapeExtent {
                claims: vec!["inner".into(), "outer".into()],
                axis: RtAxis::Lit(0),
            },
            vec![actual, inner, outer],
            scalar_type(),
            None,
        );
        dag.add_root(checked);
        assert!(chelis_ir::verify::verify(&dag).is_empty());
        let mut rewritten = common_subexpr_eliminate(&dag);
        constant_fold(&mut rewritten);
        let rewritten = dead_code_eliminate(&rewritten);
        for graph in [dag.clone(), rewritten] {
            let result = eval_tensor_with(&graph, |_| None);
            if second == 2 {
                let root = graph.roots()[0];
                assert_eq!(
                    result.unwrap()[&root].storage().scalar_at(0).as_i64_exact(),
                    Some(2)
                );
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("extent `outer`: claimed = 3, reshape axis 0 = 2"),
                    "{error}"
                );
                assert!(
                    error.contains("numeric trap: domain in reshape at int64"),
                    "{error}"
                );
            }
        }
        let mut missing = dag.clone();
        missing.node_mut(checked).unwrap().inputs.pop();
        assert!(!chelis_ir::verify::verify(&missing).is_empty());
        let mut empty = dag;
        empty.node_mut(checked).unwrap().op = RiscOp::CheckedReshapeExtent {
            claims: Vec::new(),
            axis: RtAxis::Lit(0),
        };
        empty.node_mut(checked).unwrap().inputs.truncate(1);
        assert!(!chelis_ir::verify::verify(&empty).is_empty());
    }
}

#[test]
fn checked_unit_refinement_requires_the_same_observed_tensor_axis() {
    for actual in [1, 2] {
        let mut dag = Dag::new();
        let (_, _, checked) = unit_axis(&mut dag);
        dag.add_root(checked);
        assert!(chelis_ir::verify::verify(&dag).is_empty());
        let result = eval_tensor_with(&dag, |_| {
            Some(TensorValue::from_vec(vec![actual], vec![5.0; actual]))
        });
        if actual == 1 {
            let values = result.unwrap();
            assert_eq!(values[&checked].shape, [1]);
            assert_eq!(values[&checked].to_f64_lossy_vec(), [5.0]);
        } else {
            let error = result.unwrap_err();
            assert!(error.contains("claimed = 1, b axis 0 = 2"), "{error}");
            assert!(
                error
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at int64"),
                "{error}"
            );
        }
    }
}
#[test]
fn malformed_checked_carriers_cannot_assert_metadata_as_proof() {
    for mutation in 0..5 {
        let mut dag = Dag::new();
        let (_, witness, checked) = unit_axis(&mut dag);
        dag.add_root(checked);
        match mutation {
            0 => dag
                .node_mut(checked)
                .unwrap()
                .inputs
                .pop()
                .map(|_| ())
                .unwrap(),
            1 => dag.node_mut(checked).unwrap().output_type.dims[0] = DimInfo::Lit(2),
            2 => {
                if let RiscOp::ExtentWitness { requirements, .. } =
                    &mut dag.node_mut(witness).unwrap().op
                {
                    requirements.clear();
                }
            }
            3 => dag.node_mut(witness).unwrap().op = RiscOp::Shape { axis: 0 },
            4 => {
                dag.node_mut(checked).unwrap().op = RiscOp::CheckedUnitAxis {
                    axis: RtAxis::Lit(1),
                }
            }
            _ => unreachable!(),
        }
        assert!(
            !chelis_ir::verify::verify(&dag).is_empty(),
            "unit mutation {mutation}"
        );
    }
    for mutation in 0..3 {
        let mut dag = Dag::new();
        let checked = checked_extent(&mut dag, 2, 2);
        dag.add_root(checked);
        match mutation {
            0 => {
                dag.node_mut(checked).unwrap().inputs.pop();
            }
            1 => dag.node_mut(checked).unwrap().output_type.precision = Prim::F32,
            2 => dag
                .node_mut(checked)
                .unwrap()
                .output_type
                .dims
                .push(DimInfo::Lit(1)),
            _ => unreachable!(),
        }
        assert!(
            !chelis_ir::verify::verify(&dag).is_empty(),
            "scalar mutation {mutation}"
        );
    }
}
#[test]
fn rewriting_preserves_discarded_checks_and_separate_call_provenance() {
    for actual in [2, 3] {
        let mut dag = Dag::new();
        let first = checked_extent(&mut dag, actual, 2);
        let second = checked_extent(&mut dag, actual, 2);
        dag.node_mut(second).unwrap().span_id = Some("second-call".into());
        let result = scalar(&mut dag, 9);
        dag.node_mut(result).unwrap().shape_deps = vec![first, second];
        dag.add_root(result);
        constant_fold(&mut dag);
        let dag = dead_code_eliminate(&common_subexpr_eliminate(&dag));
        let checks: Vec<_> = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::CheckedReshapeExtent { .. }))
            .collect();
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].span_id.as_deref(), Some("reshape-call"));
        assert_eq!(checks[1].span_id.as_deref(), Some("second-call"));
        assert_eq!(eval_tensor_with(&dag, |_| None).is_ok(), actual == 2);
    }
}

fn computed_reshape() -> (Dag, NodeId, NodeId) {
    use chelis_ir::dag::RtDim;
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("size".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let shape = dag.add_node(RiscOp::Shape { axis: 0 }, vec![x], scalar_type(), None);
    let two = scalar(&mut dag, 2);
    let actual = dag.add_node(RiscOp::FloorDiv, vec![shape, two], scalar_type(), None);
    let checked = dag.add_node(
        RiscOp::CheckedReshapeExtent {
            claims: vec!["2".into()],
            axis: RtAxis::Lit(0),
        },
        vec![actual, two],
        scalar_type(),
        Some("reshape-call".into()),
    );
    let result = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1), RtDim::Lit(2)],
        },
        vec![x, checked],
        TensorType {
            dims: vec![DimInfo::Named("checked".into(), None), DimInfo::Lit(2)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(result);
    (dag, x, result)
}

#[test]
fn vectorization_transports_checked_shapes_and_shifts_observed_axes() {
    for actual in [1, 2] {
        let mut dag = Dag::new();
        let (_, _, checked) = unit_axis(&mut dag);
        dag.add_root(checked);
        let mapped = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
        assert!(chelis_ir::verify::verify(&mapped).is_empty());
        let result = eval_tensor_with(&mapped, |_| {
            Some(TensorValue::from_vec(
                vec![2, actual],
                vec![5.0; 2 * actual],
            ))
        });
        if actual == 1 {
            let values = result.unwrap();
            let result = &values[&mapped.roots()[0]];
            assert_eq!(result.shape, [2, 1]);
            assert_eq!(result.to_f64_lossy_vec(), [5.0, 5.0]);
        } else {
            assert!(result.unwrap_err().contains("claimed = 1, b axis 1 = 2"));
        }
    }
    for actual in [4, 6] {
        let (dag, _, _) = computed_reshape();
        let mapped = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
        assert!(chelis_ir::verify::verify(&mapped).is_empty());
        let result = eval_tensor_with(&mapped, |_| {
            Some(TensorValue::from_vec(
                vec![2, actual],
                vec![7.0; 2 * actual],
            ))
        });
        if actual == 4 {
            let values = result.unwrap();
            let result = &values[&mapped.roots()[0]];
            assert_eq!(result.shape, [2, 2, 2]);
            assert_eq!(result.to_f64_lossy_vec(), [7.0; 8]);
        } else {
            let error = result.unwrap_err();
            assert!(error.contains("claimed = 2, reshape axis 1 = 3"), "{error}");
            assert!(
                error.contains("numeric trap: domain in reshape at int64"),
                "{error}"
            );
        }
    }
}

#[test]
fn vectorization_keeps_local_unit_failure_at_the_observed_node() {
    for actual in [1, 2] {
        let mut dag = Dag::new();
        let (_, witness, checked) = unit_axis(&mut dag);
        if let RiscOp::ExtentWitness { site, .. } = &mut dag.node_mut(witness).unwrap().op {
            *site = chelis_ir::dag::ExtentWitnessSite::LocalExpand;
        }
        dag.add_root(checked);
        let mapped = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
        assert!(chelis_ir::verify::verify(&mapped).is_empty());
        let result = eval_tensor_with(&mapped, |_| {
            Some(TensorValue::from_vec(
                vec![2, actual],
                vec![5.0; 2 * actual],
            ))
        });
        if actual == 1 {
            let values = result.unwrap();
            assert_eq!(values[&mapped.roots()[0]].shape, [2, 1]);
            assert_eq!(values[&mapped.roots()[0]].to_f64_lossy_vec(), [5.0, 5.0]);
        } else {
            let error = result.unwrap_err();
            let witness = mapped
                .nodes()
                .iter()
                .find(|node| {
                    matches!(
                        node.op,
                        RiscOp::ExtentWitness {
                            site: chelis_ir::dag::ExtentWitnessSite::LocalExpand,
                            ..
                        }
                    )
                })
                .unwrap();
            assert!(
                error.contains(&format!("node {} axis 1 = 2", witness.inputs[0].0)),
                "{error}"
            );
            assert!(
                error.contains("numeric trap: domain in expand at int64"),
                "{error}"
            );
        }
    }
}

#[test]
fn masked_gradients_keep_unit_and_computed_primal_checks() {
    use chelis_ir::eval::eval_tensor_roots_with_strict;
    for unit in [true, false] {
        for good in [true, false] {
            let (mut dag, input, mut output) = if unit {
                let mut dag = Dag::new();
                let (input, _, output) = unit_axis(&mut dag);
                (dag, input, output)
            } else {
                computed_reshape()
            };
            while !dag.get(output).unwrap().output_type.dims.is_empty() {
                let mut ty = dag.get(output).unwrap().output_type.clone();
                ty.dims.remove(0);
                output = dag.add_node(
                    RiscOp::Sum {
                        axis: 0,
                        accumulator: Prim::F32,
                    },
                    vec![output],
                    ty,
                    None,
                );
            }
            let gradient = chelis_ir::grad::grad_dag(&dag, output, &[input]).unwrap();
            let root = gradient.grad_nodes[&input];
            let actual = match (unit, good) {
                (true, true) => 1,
                (true, false) => 2,
                (false, true) => 4,
                (false, false) => 6,
            };
            let result = eval_tensor_roots_with_strict(&gradient.dag, &[root], |_| {
                Some(TensorValue::from_vec(vec![actual], vec![7.0; actual]))
            });
            if good {
                let values = result.unwrap();
                assert_eq!(values[&root].shape, [actual]);
                assert_eq!(values[&root].to_f64_lossy_vec(), vec![1.0; actual]);
            } else {
                let error = result.unwrap_err();
                let expected = if unit {
                    "claimed = 1, b axis 0 = 2"
                } else {
                    "claimed = 2, reshape axis 0 = 3"
                };
                assert!(error.contains(expected), "{error}");
                assert!(error.contains("numeric trap: domain in"), "{error}");
            }
        }
    }
}

#[test]
fn remainder_rebuild_retains_exact_values_and_rejects_invalid_types() {
    let mut dag = Dag::new();
    let a = scalar(&mut dag, -7);
    let b = scalar(&mut dag, 2);
    let root = dag.add_node(RiscOp::Mod, vec![a, b], scalar_type(), None);
    dag.add_root(root);
    assert!(chelis_ir::verify::verify(&dag).is_empty());
    assert!(chelis_ir::grad::grad_dag_checked(&dag, root, &[a]).is_err());
    for mut rebuilt in [
        dag.clone(),
        common_subexpr_eliminate(&dag),
        dead_code_eliminate(&dag),
        chelis_ir::fuse::fuse(&dag),
    ] {
        constant_fold(&mut rebuilt);
        assert!(chelis_ir::verify::verify(&rebuilt).is_empty());
        let values = eval_tensor_with(&rebuilt, |_| None).unwrap();
        assert_eq!(
            values[&rebuilt.roots()[0]]
                .storage()
                .scalar_at(0)
                .as_i64_exact(),
            Some(-1)
        );
    }
    let mapped = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
    assert!(chelis_ir::verify::verify(&mapped).is_empty());
    let values = eval_tensor_with(&mapped, |_| None).unwrap();
    assert_eq!(values[&mapped.roots()[0]].shape, vec![2]);
    assert_eq!(
        values[&mapped.roots()[0]].to_f64_lossy_vec(),
        vec![-1.0, -1.0]
    );
    for mutation in 0..3 {
        let mut bad = dag.clone();
        let node = bad.node_mut(root).unwrap();
        match mutation {
            0 => node.output_type.precision = Prim::F32,
            1 => node.inputs.pop().map(|_| ()).unwrap(),
            2 => node.output_type.dims.push(DimInfo::Lit(2)),
            _ => unreachable!(),
        }
        assert!(!chelis_ir::verify::verify(&bad).is_empty());
    }
}
