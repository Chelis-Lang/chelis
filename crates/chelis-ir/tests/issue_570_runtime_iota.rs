//! [05-OP-54]: the integer sequence source is exact, runtime-sized and has
//! no cotangent; malformed arity/dtype/rank cannot reach execution.
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};

fn scalar(value: i64) -> TensorValue {
    TensorValue::from_storage(
        vec![],
        finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![value])).unwrap(),
    )
}

fn program() -> Dag {
    program_with_activation(None)
}

fn program_with_activation(active: Option<bool>) -> Dag {
    let mut dag = Dag::new();
    let owner = dag.declare("runtime_iota");
    let scalar = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let start = dag.add_node(
        owner,
        RiscOp::Load {
            name: "start".into(),
        },
        vec![],
        scalar.clone(),
        None,
    );
    let end = dag.add_node(
        owner,
        RiscOp::Load { name: "end".into() },
        vec![],
        scalar,
        None,
    );
    let activation = active.map(|value| {
        dag.add_node(
            owner,
            RiscOp::synth_const(Prim::Bool, if value { 1.0 } else { 0.0 }),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        )
    });
    let output = dag.add_node(
        chelis_ir::dag::Owner::new(owner, activation),
        RiscOp::Iota,
        vec![start, end],
        TensorType {
            dims: vec![DimInfo::Named("count".into(), None)],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_root(output);
    dag
}

#[test]
fn runtime_iota_keeps_exact_i64_values_and_independent_lengths() {
    let dag = program();
    assert!(chelis_ir::verify::verify(&dag).is_empty());
    let root = dag.roots()[0];
    for (start, end, expected) in [
        (0, 3, vec![0, 1, 2]),
        (-2, 2, vec![-2, -1, 0, 1]),
        (7, 7, vec![]),
        (9, 3, vec![]),
        (
            9_007_199_254_740_993,
            9_007_199_254_740_995,
            vec![9_007_199_254_740_993, 9_007_199_254_740_994],
        ),
    ] {
        let result = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
            "start" => Some(scalar(start)),
            "end" => Some(scalar(end)),
            _ => None,
        })
        .unwrap();
        assert_eq!(result[&root].shape, vec![expected.len()]);
        assert_eq!(result[&root].prim(), Prim::Int64);
        assert_eq!(result[&root].storage(), &scalar_storage(expected));
    }
}

fn scalar_storage(values: Vec<i64>) -> chelis_types::TensorStorage {
    finalize_tensor("test", Prim::Int64, RawTensor::Int(values)).unwrap()
}

#[test]
fn runtime_iota_rejects_unrepresentable_length_before_allocation() {
    let dag = program();
    let error = eval_tensor_roots_with_strict(&dag, dag.roots(), |name| match name {
        "start" => Some(scalar(i64::MIN)),
        "end" => Some(scalar(i64::MAX)),
        _ => None,
    })
    .unwrap_err();
    assert!(error.contains("overflow"), "{error}");
}

#[test]
fn runtime_iota_rejects_malformed_integer_sources() {
    for malformed in [
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Int64,
        },
    ] {
        let mut dag = program();
        let input = dag.nodes()[0].clone();
        dag.replace_node(input.id, input.op, input.inputs, malformed);
        assert!(!chelis_ir::verify::verify(&dag).is_empty());
    }
}

#[test]
fn runtime_iota_rejects_a_false_result_extent_claim() {
    let mut dag = program();
    let root = dag.roots()[0];
    let output = dag.get(root).unwrap().clone();
    dag.replace_node(
        root,
        output.op,
        output.inputs,
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        },
    );
    let error = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "start" => Some(scalar(0)),
        "end" => Some(scalar(3)),
        _ => None,
    })
    .unwrap_err();
    assert!(error.contains("domain in range at i64"), "{error}");
}

#[test]
fn runtime_iota_rejects_missing_endpoints_and_wrong_result_dtype() {
    for inputs in [vec![], vec![chelis_ir::dag::NodeId(0)]] {
        let mut dag = program();
        let root = dag.roots()[0];
        let output = dag.get(root).unwrap().clone();
        dag.replace_node(root, output.op, inputs, output.output_type);
        assert!(!chelis_ir::verify::verify(&dag).is_empty());
    }
    let mut dag = program();
    let root = dag.roots()[0];
    let output = dag.get(root).unwrap().clone();
    dag.replace_node(
        root,
        output.op,
        output.inputs,
        TensorType {
            dims: vec![DimInfo::Named("count".into(), None)],
            precision: Prim::F32,
        },
    );
    assert!(!chelis_ir::verify::verify(&dag).is_empty());
}

#[test]
fn anonymous_batch_comparisons_keep_physical_input_authority() {
    let mut scalar = Dag::new();
    let owner = scalar.declare("callback");
    let ty = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let value = scalar.add_node(
        owner,
        RiscOp::Load {
            name: "index".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let selected = scalar.add_node(
        owner,
        RiscOp::Load {
            name: "selected".into(),
        },
        vec![],
        ty,
        None,
    );
    let comparison = scalar.add_node(
        owner,
        RiscOp::Compare(chelis_ir::dag::ComparisonKind::Eq),
        vec![value, selected],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    scalar.add_root(comparison);
    let mut batched = chelis_ir::vmap::vectorize_axis0_with_captures(
        &scalar,
        DimInfo::Named("*".into(), None),
        &["selected".to_string()].into_iter().collect(),
    )
    .unwrap();
    assert!(
        chelis_ir::verify::verify(&batched).is_empty(),
        "{:?}",
        chelis_ir::verify::verify(&batched)
    );
    let root = batched.roots()[0];
    batched.node_mut(root).unwrap().shape_deps.clear();
    assert!(!chelis_ir::verify::verify(&batched).is_empty());
}

#[test]
fn runtime_iota_rejects_a_per_row_activation() {
    let mut dag = program_with_activation(Some(true));
    let root = dag.roots()[0];
    let active = dag.get(root).unwrap().owner.activation.unwrap();
    dag.node_mut(active).unwrap().output_type.dims = vec![DimInfo::Lit(2)];
    assert!(
        chelis_ir::verify::verify(&dag)
            .iter()
            .any(|error| error.contains("requires scalar activation"))
    );
}

#[test]
fn runtime_iota_inactive_scalar_arm_does_not_read_overflowing_endpoints() {
    let dag = program_with_activation(Some(false));
    let root = dag.roots()[0];
    let result = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "start" => Some(scalar(i64::MIN)),
        "end" => Some(scalar(i64::MAX)),
        _ => None,
    })
    .unwrap();
    assert_eq!(result[&root].shape, vec![0]);
}

fn ordered_sum_program(precision: Prim, groups: Vec<usize>, ranks: &[usize]) -> Dag {
    let mut dag = Dag::new();
    let owner = dag.declare("ordered_cotangent");
    let inputs = ranks
        .iter()
        .enumerate()
        .map(|(index, rank)| {
            dag.add_node(
                owner,
                RiscOp::Load {
                    name: format!("x{index}").into(),
                },
                vec![],
                TensorType {
                    precision,
                    dims: (0..*rank)
                        .map(|_| DimInfo::Named("*".into(), None))
                        .collect(),
                },
                None,
            )
        })
        .collect();
    let root = dag.add_node(
        owner,
        RiscOp::OrderedAdjointSum { groups },
        inputs,
        TensorType {
            precision,
            dims: vec![],
        },
        None,
    );
    dag.add_root(root);
    dag
}

#[test]
fn ordered_cotangent_tree_rounds_at_each_active_float_width() {
    for (precision, large, small) in [
        (Prim::F16, 2048.0, 0.5),
        (Prim::Bf16, 256.0, 0.25),
        (Prim::F32, 1.0e20, 3.0),
        (Prim::F64, 1.0e20, 3.0),
    ] {
        let dag = ordered_sum_program(precision, vec![1], &[1]);
        assert!(chelis_ir::verify::verify(&dag).is_empty());
        for data in [vec![large, -large, small], vec![]] {
            let values = eval_tensor_roots_with_strict(&dag, dag.roots(), |_| {
                Some(TensorValue::from_vec(vec![data.len()], data.clone()))
            })
            .unwrap();
            assert_eq!(
                values[&dag.roots()[0]].first_f64_lossy_or_zero().to_bits(),
                0.0f64.to_bits(),
                "{precision:?}"
            );
        }
    }
}

#[test]
fn ordered_cotangent_groups_preserve_rows_and_separate_invocations() {
    for (groups, expected) in [(vec![2], 0.0), (vec![1, 1], 3.0)] {
        let dag = ordered_sum_program(Prim::F32, groups, &[1, 1]);
        let values = eval_tensor_roots_with_strict(&dag, dag.roots(), |_| {
            Some(TensorValue::from_vec(vec![3], vec![1.0e20, -1.0e20, 3.0]))
        })
        .unwrap();
        assert_eq!(values[&dag.roots()[0]].first_f64_lossy_or_zero(), expected);
    }
}

#[test]
fn ordered_cotangent_rejects_malformed_groups_and_dtype_rank() {
    for (precision, groups, ranks) in [
        (Prim::F32, vec![0, 2], vec![1, 1]),
        (Prim::F32, vec![usize::MAX, 2], vec![1, 1]),
        (Prim::F32, vec![1], vec![1, 1]),
        (Prim::F32, vec![2], vec![0, 1]),
        (Prim::F32, vec![1], vec![2]),
        (Prim::Int64, vec![1], vec![1]),
    ] {
        let dag = ordered_sum_program(precision, groups, &ranks);
        assert!(!chelis_ir::verify::verify(&dag).is_empty());
    }
    let dag = ordered_sum_program(Prim::F32, vec![2], &[1, 1]);
    let error = eval_tensor_roots_with_strict(&dag, dag.roots(), |name| {
        let count = if name == "x0" { 2 } else { 3 };
        Some(TensorValue::from_vec(vec![count], vec![1.0; count]))
    })
    .unwrap_err();
    assert!(error.contains("different lengths"), "{error}");
    assert!(
        chelis_ir::grad::grad_dag(&dag, dag.roots()[0], &[chelis_ir::dag::NodeId(0)]).is_none(),
        "higher-order accumulation must not fall back to ordinary Sum"
    );
    let error = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap_err();
    assert!(error.contains("nested invocation provenance"), "{error}");
}

#[test]
fn list_capture_has_an_actual_axis_and_rejects_a_false_claim() {
    let mut dag = Dag::new();
    let owner = dag.declare("list_capture");
    let source = dag.add_node(
        owner,
        RiscOp::Load { name: "s".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let carrier = dag.add_node(
        owner,
        RiscOp::Load { name: "xs".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::Int64,
        },
        None,
    );
    let root = dag.add_node(
        owner,
        RiscOp::ListMapCapture { first: true },
        vec![source, carrier],
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(root);
    assert!(chelis_ir::verify::verify(&dag).is_empty());
    for count in [0, 2, 3, 4] {
        let result = eval_tensor_roots_with_strict(&dag, dag.roots(), |name| match name {
            "s" => Some(TensorValue::scalar(2.0)),
            "xs" => Some(TensorValue::from_storage(
                vec![count],
                scalar_storage(vec![0; count]),
            )),
            _ => None,
        });
        if count == 3 {
            assert_eq!(result.unwrap()[&root].to_f64_lossy_vec(), vec![2.0; 3]);
        } else {
            assert!(
                result.is_err(),
                "accepted carrier length {count} against a literal claim of 3"
            );
        }
    }
    for inputs in [vec![source], vec![carrier, source]] {
        let mut malformed = dag.clone();
        malformed.node_mut(root).unwrap().inputs = inputs;
        assert!(!chelis_ir::verify::verify(&malformed).is_empty());
    }
    let mut malformed = dag.clone();
    malformed.node_mut(root).unwrap().op = RiscOp::ListMapCapture { first: false };
    assert!(!chelis_ir::verify::verify(&malformed).is_empty());
    let second = dag.add_node(
        owner,
        RiscOp::ListMapCapture { first: true },
        vec![source, carrier],
        dag.get(root).unwrap().output_type.clone(),
        None,
    );
    dag.add_root(second);
    let optimized = chelis_ir::optimize::common_subexpr_eliminate(&dag);
    assert_eq!(
        optimized
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::ListMapCapture { .. }))
            .count(),
        2,
        "independent List invocations cannot merge"
    );
    let loss = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![root],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let gradient = chelis_ir::grad::grad_dag(&dag, loss, &[root, source]).unwrap();
    let values =
        eval_tensor_roots_with_strict(&gradient.dag, gradient.dag.roots(), |name| match name {
            "s" => Some(TensorValue::scalar(2.0)),
            "xs" => Some(TensorValue::from_storage(
                vec![3],
                scalar_storage(vec![0; 3]),
            )),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        values[&gradient.grad_nodes[&root]].to_f64_lossy_vec(),
        vec![1.0; 3]
    );
    assert_eq!(
        values[&gradient.grad_nodes[&source]].to_f64_lossy_vec(),
        vec![3.0]
    );
}
