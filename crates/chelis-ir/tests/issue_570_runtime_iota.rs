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
