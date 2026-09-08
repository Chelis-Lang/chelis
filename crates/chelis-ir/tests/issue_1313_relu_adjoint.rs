use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::fuse::fuse;
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::{tier2, verify};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn tensor(precision: Prim, len: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(len)],
        precision,
    }
}

fn value(precision: Prim, values: Vec<f64>) -> TensorValue {
    value_with_shape(precision, vec![values.len()], values)
}

fn value_with_shape(precision: Prim, shape: Vec<usize>, values: Vec<f64>) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor("test", precision, RawTensor::Float(values)).unwrap(),
    )
}

#[test]
fn relu_lowering_remains_one_identity_until_ad() {
    let mut dag = Dag::new();
    let ty = tensor(Prim::F32, 4);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let relu = tier2::lower_relu(&mut dag, x, &ty, Some("relu.expr"));

    assert_eq!(dag.len(), 2);
    assert_eq!(dag.get(relu).unwrap().op, RiscOp::Relu);
    assert_eq!(dag.get(relu).unwrap().inputs, vec![x]);
    assert_eq!(dag.get(relu).unwrap().span_id.as_deref(), Some("relu.expr"));
    assert!(verify::verify(&dag).is_empty());

    dag.add_root(relu);
    let fused = fuse(&dag);
    assert!(
        fused.nodes().iter().any(|node| node.op == RiscOp::Relu),
        "fusion must not erase the semantic identity before AD"
    );
}

#[test]
fn relu_adjoint_uses_dedicated_identity_and_direct_max_keeps_its_tie_rule() {
    let ty = tensor(Prim::F64, 5);
    let mut relu_dag = Dag::new();
    let x = relu_dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let relu = relu_dag.add_node(RiscOp::Relu, vec![x], ty.clone(), None);
    let sum = relu_dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![relu],
        TensorType {
            dims: vec![],
            precision: Prim::F64,
        },
        None,
    );
    let differentiated = grad_dag_checked(&relu_dag, sum, &[x]).unwrap();
    let dx = differentiated.grad_nodes[&x];
    assert!(
        differentiated
            .dag
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::ReluAdjoint && node.inputs.first() == Some(&x)),
        "the accumulated gradient must contain the dedicated ReLU contribution"
    );
    assert!(
        differentiated
            .dag
            .nodes()
            .iter()
            .all(|node| node.op != RiscOp::MaxElem),
        "ReLU AD must not reconstruct the direct-maximum surrogate"
    );

    let nan = f64::from_bits(0xfff8_1234_5678_9abc);
    let inputs = UnordMap::from([(
        "x".to_string(),
        value(Prim::F64, vec![nan, -0.0, 0.0, -1.0, f64::from_bits(1)]),
    )]);
    let values = eval_tensor(&differentiated.dag, &inputs).unwrap();
    let actual = values[&dx].to_f64_lossy_vec();
    assert_eq!(actual, vec![0.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(actual[..4].iter().all(|v| v.to_bits() == 0));

    let mut max_dag = Dag::new();
    let left = max_dag.add_node(
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let right = max_dag.add_node(
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let max = max_dag.add_node(RiscOp::MaxElem, vec![left, right], ty.clone(), None);
    let sum = max_dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![max],
        TensorType {
            dims: vec![],
            precision: Prim::F64,
        },
        None,
    );
    let direct = grad_dag_checked(&max_dag, sum, &[left]).unwrap();
    let inputs = UnordMap::from([
        ("left".to_string(), value(Prim::F64, vec![0.0; 5])),
        ("right".to_string(), value(Prim::F64, vec![0.0; 5])),
    ]);
    let values = eval_tensor(&direct.dag, &inputs).unwrap();
    assert_eq!(
        values[&direct.grad_nodes[&left]].to_f64_lossy_vec(),
        vec![1.0; 5]
    );
}

#[test]
fn relu_adjoint_preserves_non_unit_cotangent_bits_and_is_second_order_in_g_only() {
    let ty = tensor(Prim::F64, 4);
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], ty.clone(), None);
    let adjoint = dag.add_node(RiscOp::ReluAdjoint, vec![x, g], ty.clone(), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![adjoint],
        TensorType {
            dims: vec![],
            precision: Prim::F64,
        },
        None,
    );
    let differentiated = grad_dag_checked(&dag, sum, &[x, g]).unwrap();

    let payload = f64::from_bits(0x7ff8_abcd_1234_5678);
    let inputs = UnordMap::from([
        (
            "x".to_string(),
            value(Prim::F64, vec![0.0, f64::NAN, 1.0, -1.0]),
        ),
        (
            "g".to_string(),
            value(Prim::F64, vec![payload, f64::INFINITY, -0.0, payload]),
        ),
    ]);
    let values = eval_tensor(&differentiated.dag, &inputs).unwrap();
    assert!(
        values[&differentiated.grad_nodes[&x]]
            .to_f64_lossy_vec()
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    assert_eq!(
        values[&differentiated.grad_nodes[&g]].to_f64_lossy_vec(),
        vec![0.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn scalar_relu_matrix_covers_every_float_width_and_boundary_class() {
    for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let positive_subnormal = match precision {
            Prim::F16 => 5.960_464_477_539_063e-8,
            Prim::Bf16 => 9.183_549_615_799_121e-41,
            Prim::F32 => 1.401_298_464_324_817e-45,
            Prim::F64 => f64::from_bits(1),
            _ => unreachable!(),
        };
        for (x_value, g_value, forward_class, adjoint_class) in [
            (f64::NAN, f64::NAN, "nan", "positive_zero"),
            (-0.0, f64::INFINITY, "negative_zero", "positive_zero"),
            (0.0, -0.0, "positive_zero", "positive_zero"),
            (-1.0, f64::NAN, "positive_zero", "positive_zero"),
            (1.0, -0.0, "positive", "negative_zero"),
            (
                positive_subnormal,
                f64::INFINITY,
                "positive_subnormal",
                "positive_infinity",
            ),
        ] {
            let scalar_ty = TensorType {
                dims: vec![],
                precision,
            };
            let mut dag = Dag::new();
            let x = dag.add_node(
                RiscOp::Load { name: "x".into() },
                vec![],
                scalar_ty.clone(),
                None,
            );
            let g = dag.add_node(
                RiscOp::Load { name: "g".into() },
                vec![],
                scalar_ty.clone(),
                None,
            );
            let relu = dag.add_node(RiscOp::Relu, vec![x], scalar_ty.clone(), None);
            let adjoint = dag.add_node(RiscOp::ReluAdjoint, vec![x, g], scalar_ty, None);
            dag.add_root(relu);
            dag.add_root(adjoint);
            assert!(verify::verify(&dag).is_empty());

            let inputs = UnordMap::from([
                (
                    "x".to_string(),
                    value_with_shape(precision, vec![], vec![x_value]),
                ),
                (
                    "g".to_string(),
                    value_with_shape(precision, vec![], vec![g_value]),
                ),
            ]);
            let values = eval_tensor(&dag, &inputs).unwrap();
            assert!(values[&relu].shape.is_empty(), "{precision:?}");
            assert!(values[&adjoint].shape.is_empty(), "{precision:?}");
            let forward = values[&relu].to_f64_lossy_vec()[0];
            let adjoint_value = values[&adjoint].to_f64_lossy_vec()[0];
            match forward_class {
                "nan" => assert!(forward.is_nan(), "{precision:?}"),
                "negative_zero" => assert_eq!(forward.to_bits(), (-0.0_f64).to_bits()),
                "positive_zero" => assert_eq!(forward.to_bits(), 0.0_f64.to_bits()),
                "positive" => assert_eq!(forward, 1.0),
                "positive_subnormal" => assert_eq!(forward, positive_subnormal),
                _ => unreachable!(),
            }
            match adjoint_class {
                "positive_zero" => assert_eq!(adjoint_value.to_bits(), 0.0_f64.to_bits()),
                "negative_zero" => {
                    assert_eq!(adjoint_value.to_bits(), (-0.0_f64).to_bits())
                }
                "positive_infinity" => assert_eq!(adjoint_value, f64::INFINITY),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn verifier_rejects_relu_wrong_arity_and_non_float_domain() {
    let mut wrong_arity = Dag::new();
    let ty = tensor(Prim::F32, 1);
    let x = wrong_arity.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    wrong_arity.add_node(RiscOp::Relu, vec![x, x], ty.clone(), None);
    assert!(
        verify::verify(&wrong_arity)
            .iter()
            .any(|error| error.contains("expected 1"))
    );

    let mut wrong_domain = Dag::new();
    let int_ty = tensor(Prim::Int32, 1);
    let x = wrong_domain.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        int_ty.clone(),
        None,
    );
    wrong_domain.add_node(RiscOp::Relu, vec![x], int_ty, None);
    assert!(
        verify::verify(&wrong_domain)
            .iter()
            .any(|error| error.contains("float"))
    );

    let mut wrong_adjoint = Dag::new();
    let x = wrong_adjoint.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    wrong_adjoint.add_node(RiscOp::ReluAdjoint, vec![x], ty, None);
    assert!(
        verify::verify(&wrong_adjoint)
            .iter()
            .any(|error| error.contains("expected 2"))
    );

    let mut mismatched = Dag::new();
    let x = mismatched.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor(Prim::F32, 1),
        None,
    );
    let g = mismatched.add_node(
        RiscOp::Load { name: "g".into() },
        vec![],
        tensor(Prim::F64, 2),
        None,
    );
    mismatched.add_node(RiscOp::ReluAdjoint, vec![x, g], tensor(Prim::F32, 1), None);
    assert!(
        verify::verify(&mismatched)
            .iter()
            .any(|error| error.contains("same-shape, same-dtype"))
    );
}
