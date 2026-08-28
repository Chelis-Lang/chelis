use std::collections::HashMap;

use chelis_ir::dag::{Dag, ExtremaKind, ExtremaOperand, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_ir::{tier2, verify};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

#[test]
fn sub_lowers_to_one_direct_identity_without_synthetic_negation() {
    let mut dag = Dag::new();
    let ty = scalar_f32();
    let left = dag.add_node(
        RiscOp::synth_const(ty.precision, -1.0),
        vec![],
        ty.clone(),
        None,
    );
    let right = dag.add_node(
        RiscOp::synth_const(ty.precision, 2.0),
        vec![],
        ty.clone(),
        None,
    );

    let result = tier2::lower_sub(&mut dag, left, right, &ty, Some("sub.expr"));

    assert!(verify::verify(&dag).is_empty());
    assert_eq!(dag.len(), 3, "direct sub adds exactly one node");
    assert_eq!(dag.get(result).unwrap().op, RiscOp::Sub);
    assert_eq!(dag.get(result).unwrap().inputs, vec![left, right]);
    assert_eq!(
        dag.get(result).unwrap().span_id.as_deref(),
        Some("sub.expr")
    );
    assert!(
        dag.nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::Neg | RiscOp::Add)),
        "sub must not introduce add or neg"
    );
}

#[test]
fn min_elem_lowers_to_one_direct_selection_without_arithmetic_surrogate() {
    let mut dag = Dag::new();
    let ty = scalar_f32();
    let left = dag.add_node(
        RiscOp::synth_const(ty.precision, -1.0),
        vec![],
        ty.clone(),
        None,
    );
    let right = dag.add_node(
        RiscOp::synth_const(ty.precision, 2.0),
        vec![],
        ty.clone(),
        None,
    );

    let result = tier2::lower_min_elem(&mut dag, left, right, &ty, Some("min.expr"));

    assert!(verify::verify(&dag).is_empty());
    assert_eq!(dag.len(), 3, "direct min_elem adds exactly one node");
    assert_eq!(dag.get(result).unwrap().op, RiscOp::MinElem);
    assert_eq!(dag.get(result).unwrap().inputs, vec![left, right]);
    assert_eq!(
        dag.get(result).unwrap().span_id.as_deref(),
        Some("min.expr")
    );
    assert!(
        dag.nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::Neg | RiscOp::MaxElem)),
        "min_elem must not introduce neg or max_elem"
    );
}

fn scalar_at(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

fn exact_f64(value: f64) -> TensorValue {
    TensorValue::from_storage(
        vec![],
        finalize_tensor("test", Prim::F64, RawTensor::Float(vec![value])).unwrap(),
    )
}

#[test]
fn extrema_adjoint_routes_ties_and_nan_cotangents_to_the_forward_selected_operand() {
    for op in [RiscOp::MaxElem, RiscOp::MinElem] {
        let mut dag = Dag::new();
        let ty = scalar_at(Prim::F64);
        let left = dag.add_node(
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let right = dag.add_node(
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let output = dag.add_node(op.clone(), vec![left, right], ty, None);
        let differentiated = grad_dag_checked(&dag, output, &[left, right]).unwrap();

        let kind = if matches!(op, RiscOp::MaxElem) {
            ExtremaKind::Max
        } else {
            ExtremaKind::Min
        };
        assert!(differentiated.dag.nodes().iter().any(|node| {
            matches!(
                node.op,
                RiscOp::ExtremaAdjoint {
                    kind: observed,
                    operand: ExtremaOperand::Left,
                } if observed == kind
            )
        }));
        assert!(differentiated.dag.nodes().iter().any(|node| {
            matches!(
                node.op,
                RiscOp::ExtremaAdjoint {
                    kind: observed,
                    operand: ExtremaOperand::Right,
                } if observed == kind
            )
        }));
        assert!(
            differentiated
                .dag
                .nodes()
                .iter()
                .all(|node| !matches!(node.op, RiscOp::CmpLt)),
            "NaN selection cannot be reconstructed from ordered comparison"
        );

        let cases = [
            (0.0, -0.0, 1.0, 0.0),
            (f64::from_bits(0xfff8_1234_5678_9abc), 1.0, 1.0, 0.0),
            (1.0, f64::from_bits(0x7ff8_abcd_1234_5678), 0.0, 1.0),
        ];
        for (left_value, right_value, expected_left, expected_right) in cases {
            let inputs = HashMap::from([
                ("left".to_string(), exact_f64(left_value)),
                ("right".to_string(), exact_f64(right_value)),
            ]);
            let values = eval_tensor(&differentiated.dag, &inputs).unwrap();
            assert_eq!(
                values[&differentiated.grad_nodes[&left]].to_f64_lossy_vec(),
                vec![expected_left]
            );
            assert_eq!(
                values[&differentiated.grad_nodes[&right]].to_f64_lossy_vec(),
                vec![expected_right]
            );
        }
    }
}

#[test]
fn integer_direct_sub_and_extrema_are_forward_only() {
    for op in [RiscOp::Sub, RiscOp::MaxElem, RiscOp::MinElem] {
        let mut dag = Dag::new();
        let int_ty = scalar_at(Prim::Int64);
        let left = dag.add_node(
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            int_ty.clone(),
            None,
        );
        let right = dag.add_node(
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            int_ty.clone(),
            None,
        );
        let integer = dag.add_node(op.clone(), vec![left, right], int_ty, None);
        let output = dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F64,
            },
            vec![integer],
            scalar_at(Prim::F64),
            None,
        );

        let error = match grad_dag_checked(&dag, output, &[left]) {
            Ok(_) => panic!("signed-integer arithmetic must reject reverse-mode AD"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            AdError::NotSupported {
                op: match op {
                    RiscOp::Sub => "sub",
                    RiscOp::MaxElem => "max_elem",
                    RiscOp::MinElem => "min_elem",
                    _ => unreachable!(),
                },
                reason: AdRejectionReason::IntegerArithmeticOutput,
            }
        );
    }
}

#[test]
fn signed_integer_extrema_chains_stay_materialized_for_typed_backends() {
    for precision in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let mut dag = Dag::new();
        let ty = scalar_at(precision);
        let left = dag.add_node(
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let right = dag.add_node(
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let cap = dag.add_node(
            RiscOp::Load { name: "cap".into() },
            vec![],
            ty.clone(),
            None,
        );
        let maximum = dag.add_node(RiscOp::MaxElem, vec![left, right], ty.clone(), None);
        let minimum = dag.add_node(RiscOp::MinElem, vec![maximum, cap], ty, None);
        dag.add_root(minimum);

        let fused = chelis_ir::fuse::fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .all(|node| !matches!(node.op, RiscOp::FusedElem { .. })),
            "{precision:?} extrema must stay on the exact typed direct kernels"
        );
        assert!(
            fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::MaxElem)),
            "{precision:?} max_elem disappeared during fusion"
        );
        assert!(
            fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::MinElem)),
            "{precision:?} min_elem disappeared during fusion"
        );
    }
}
