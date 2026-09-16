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
            claims: Vec::new(),
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

fn declared_producer(dag: &mut Dag, parameter: &str, actual: usize) -> NodeId {
    use chelis_ir::dag::ExtentWitnessSite;
    let input = dag.add_node(
        RiscOp::Load {
            name: parameter.into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named(format!("{parameter}_length"), None)],
            precision: Prim::F32,
        },
        None,
    );
    let declared = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: parameter.into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        },
        vec![input],
        scalar_type(),
        None,
    );
    let required = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::ResultClaim {
                claim: "n".into(),
                axis: RtAxis::Lit(0),
            },
            parameter: parameter.into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        },
        vec![input],
        scalar_type(),
        None,
    );
    dag.add_shape_dep(required, declared);
    let value = scalar(dag, 7);
    let actual = scalar(dag, i64::try_from(actual).unwrap());
    let produced = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Node(1),
        },
        vec![value, actual],
        TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_shape_dep(produced, required);
    dag.add_root(produced);
    produced
}

#[test]
fn result_claim_witnesses_survive_rebuilds_without_joining_same_labels() {
    for actual in [3, 4] {
        let mut dag = Dag::new();
        declared_producer(&mut dag, "left", 2);
        declared_producer(&mut dag, "right", actual);
        let mut folded = dag.clone();
        constant_fold(&mut folded);
        let mapped = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
        for (rebuilt, batched) in [
            (dag.clone(), false),
            (common_subexpr_eliminate(&dag), false),
            (dead_code_eliminate(&dag), false),
            (folded, false),
            (
                chelis_ir::specialize::specialize_for_exact_arithmetic(&dag),
                false,
            ),
            (mapped, true),
        ] {
            let errors = chelis_ir::verify::verify(&rebuilt);
            assert!(errors.is_empty(), "{errors:?}");
            let owned = chelis_ir::ownership::verify_ownership(
                chelis_ir::ownership::lower_dag_ownership(rebuilt.clone()).unwrap(),
            )
            .unwrap();
            let plan = chelis_ir::ownership::plan_c_storage(owned).unwrap();
            for node in rebuilt.nodes() {
                for dependency in &node.shape_deps {
                    if matches!(
                        rebuilt.get(*dependency).unwrap().op,
                        RiscOp::ExtentWitness {
                            site: chelis_ir::dag::ExtentWitnessSite::ResultClaim { .. },
                            ..
                        }
                    ) {
                        let slot = plan
                            .slot_for_node(*dependency)
                            .expect("claim scalar owns storage");
                        assert!(
                            plan.slot(slot).unwrap().last_use_index() >= node.id.0,
                            "claim bytes remain live until their producer reads them"
                        );
                        for intermediate in (dependency.0 + 1)..node.id.0 {
                            assert_ne!(
                                plan.slot_for_node(NodeId(intermediate)),
                                Some(slot),
                                "claim storage cannot be overwritten before its producer"
                            );
                        }
                    }
                }
            }
            let result = eval_tensor_with(&rebuilt, |name| {
                let width = if name == "left" { 2 } else { 3 };
                let shape = if batched { vec![2, width] } else { vec![width] };
                let count = if batched { 2 * width } else { width };
                Some(TensorValue::from_vec(shape, vec![0.0; count]))
            });
            if actual == 3 {
                let values = result.unwrap();
                assert_eq!(
                    values[&rebuilt.roots()[0]].shape,
                    if batched { vec![2, 2] } else { vec![2] }
                );
                assert_eq!(
                    values[&rebuilt.roots()[1]].shape,
                    if batched { vec![2, 3] } else { vec![3] }
                );
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains(&format!(
                        "extent `n`: claimed = 3, insert axis {} = 4",
                        usize::from(batched)
                    )),
                    "{error}"
                );
                assert!(
                    error.ends_with("numeric trap: domain in insert at i64"),
                    "{error}"
                );
            }
        }
    }
    // Two declarations can constrain the same produced axis. The dependency
    // list retains both values and the declaration order, even with one label.
    for (first, second, actual, failure) in
        [(3, 3, 3, None), (2, 3, 2, Some(3)), (2, 3, 4, Some(2))]
    {
        let mut dag = Dag::new();
        let left = declared_producer(&mut dag, "left", first);
        let right = declared_producer(&mut dag, "right", actual);
        let first_token = dag.get(left).unwrap().shape_deps[0];
        dag.node_mut(right)
            .unwrap()
            .shape_deps
            .insert(0, first_token);
        assert!(chelis_ir::verify::verify(&dag).is_empty());
        let result = eval_tensor_with(&dag, |name| {
            let width = if name == "left" { first } else { second };
            Some(TensorValue::from_vec(vec![width], vec![0.0; width]))
        });
        if let Some(required) = failure {
            let error = result.unwrap_err();
            assert!(
                error.contains(&format!(
                    "extent `n`: claimed = {required}, insert axis 0 = {actual}"
                )),
                "{error}"
            );
        } else {
            assert_eq!(result.unwrap()[&right].shape, [3]);
        }
    }
    // A captured named obligation cannot discharge an independent literal
    // claim on that same axis. The named claim agrees in both executions.
    for literal in [2, 3] {
        let mut dag = Dag::new();
        let producer = declared_producer(&mut dag, "x", 2);
        dag.node_mut(producer).unwrap().output_type.dims[0] = DimInfo::Lit(literal);
        let errors = chelis_ir::verify::verify(&dag);
        assert!(errors.is_empty(), "{errors:?}");
        let result = eval_tensor_with(&dag, |_| Some(TensorValue::from_vec(vec![2], vec![0.0; 2])));
        if literal == 2 {
            assert_eq!(result.unwrap()[&producer].shape, [2]);
        } else {
            let error =
                result.expect_err("an agreeing named token cannot hide the disagreeing literal");
            assert!(
                error.contains("extent `3`: claimed = 3, insert axis 0 = 2"),
                "{error}"
            );
            assert!(
                error.ends_with("numeric trap: domain in insert at i64"),
                "{error}"
            );
        }
    }
}

#[test]
fn result_claim_admission_requires_its_declaring_observation_and_producing_axis() {
    let mut dag = Dag::new();
    let producer = declared_producer(&mut dag, "x", 2);
    let token = dag.get(producer).unwrap().shape_deps[0];
    assert!(
        !chelis_ir::axis_sources::witness_is_entry_obligation(&dag, token),
        "the producer reads this witness's scalar through its explicit dependency"
    );
    assert!(
        chelis_ir::axis_sources::witness_entry_obligations(&dag, token).is_none(),
        "a local result requirement cannot be waived as an empty entry check"
    );
    let errors = chelis_ir::verify::verify(&dag);
    assert!(errors.is_empty(), "{errors:?}");
    for mutation in ["declaration", "axis", "producer"] {
        let mut bad = dag.clone();
        match mutation {
            "declaration" => bad.node_mut(token).unwrap().shape_deps.clear(),
            "axis" => {
                let RiscOp::ExtentWitness {
                    site: chelis_ir::dag::ExtentWitnessSite::ResultClaim { axis, .. },
                    ..
                } = &mut bad.node_mut(token).unwrap().op
                else {
                    unreachable!()
                };
                *axis = RtAxis::Lit(1);
            }
            "producer" => bad.node_mut(producer).unwrap().op = RiscOp::Copy,
            _ => unreachable!(),
        }
        assert!(!chelis_ir::verify::verify(&bad).is_empty(), "{mutation}");
    }
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
                    .any(|line| line == "numeric trap: domain in reshape at i64"),
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
                    error.contains("numeric trap: domain in reshape at i64"),
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
                    .any(|line| line == "numeric trap: domain in load at i64"),
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
                error.contains("numeric trap: domain in reshape at i64"),
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
                error.contains("numeric trap: domain in expand at i64"),
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

/// [04-NUM-9]: a named result claim reports its declaring observation first,
/// even when that parameter occurs after the value whose extent is claimed.
#[test]
fn an_entry_result_claim_keeps_its_later_declaring_witness() {
    use chelis_ir::axis_sources::{EntryExtentGuard, entry_extent_guards};
    use chelis_ir::dag::{ExtentClaim, ExtentWitnessSite};
    let mut dag = Dag::new();
    let mut loads = Vec::new();
    let mut witnesses = Vec::new();
    for parameter in ["x", "y"] {
        let load = dag.add_node(
            RiscOp::Load {
                name: parameter.into(),
            },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("cols".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        loads.push(load);
        witnesses.push(dag.add_node(
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::Caller,
                parameter: parameter.into(),
                axis: RtAxis::Lit(0),
                requirements: vec![],
                claims: vec![],
            },
            vec![load],
            scalar_type(),
            None,
        ));
    }
    let later = dag.node_mut(witnesses[1]).unwrap();
    later.inputs.push(witnesses[0]);
    let RiscOp::ExtentWitness { claims, .. } = &mut later.op else {
        unreachable!()
    };
    claims.push(ExtentClaim {
        claim: "cols".into(),
        requirement_declares: false,
    });
    let result = dag.add_node(
        RiscOp::Copy,
        vec![loads[0]],
        dag.get(loads[0]).unwrap().output_type.clone(),
        None,
    );
    dag.add_shape_dep(result, witnesses[1]);
    dag.add_root(result);
    assert_eq!(
        entry_extent_guards(&dag),
        vec![EntryExtentGuard::Named {
            claim: "cols".into(),
            canonical: (loads[1], 0),
            observed: (loads[0], 0)
        }]
    );
    let inputs = |width| {
        std::collections::BTreeMap::from([
            (
                "x".to_string(),
                TensorValue::from_vec(vec![2], vec![1.0, 2.0]),
            ),
            (
                "y".to_string(),
                TensorValue::from_vec(vec![width], vec![1.0; width]),
            ),
        ])
    };
    let error = eval_tensor_with(&dag, |name| inputs(3).get(name).cloned())
        .expect_err("later declaration must reject an earlier foreign extent");
    assert_eq!(
        error,
        "extent `cols`: y axis 0 = 3, x axis 0 = 2\nnumeric trap: domain in load at i64"
    );
    let output = eval_tensor_with(&dag, |name| inputs(2).get(name).cloned())
        .expect("agreeing claim executes");
    assert_eq!(output[&result].shape, vec![2]);
    assert_eq!(output[&result].to_f64_lossy_vec(), vec![1.0, 2.0]);
}
