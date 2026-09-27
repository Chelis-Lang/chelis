//! [04] §4.7 / [06] §5.2: literal claims retain independent call witnesses.
use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::optimize::dead_code_eliminate;
use chelis_types::{scalar_from_i64, types::Prim};

fn claimed_witness(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    required: &[i64],
) -> chelis_ir::dag::NodeId {
    let x = dag.add_node(
        decl,
        RiscOp::Load {
            name: "actual".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("rows".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_node(
        decl,
        RiscOp::ExtentWitness {
            site: chelis_ir::dag::ExtentWitnessSite::Caller,
            parameter: "x".into(),
            axis: RtAxis::Lit(0),
            requirements: required
                .iter()
                .map(|n| scalar_from_i64("load", Prim::Int64, *n).unwrap())
                .collect(),
            claims: Vec::new(),
        },
        vec![x],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        Some("call-f".into()),
    )
}

#[test]
fn call_witness_checks_each_literal_and_returns_the_observed_extent() {
    for (required, actual, fails) in [
        (vec![4], 4, None),
        (vec![4], 5, Some(4)),
        (vec![4, 5], 4, Some(5)),
        (vec![4, 5], 5, Some(4)),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let witness = claimed_witness(&mut dag, decl, &required);
        dag.add_root(witness);
        let outcome = eval_tensor_with(&dag, |_| {
            Some(TensorValue::from_vec(vec![actual], vec![7.0; actual]))
        });
        if let Some(required) = fails {
            let error = outcome.unwrap_err();
            assert!(
                error
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at i64"),
                "{error}"
            );
            assert!(
                error.contains(&format!("claimed = {required}, x axis 0 = {actual}")),
                "{error}"
            );
        } else {
            let values = outcome.unwrap();
            assert_eq!(values[&witness].prim(), Prim::Int64);
            assert!(values[&witness].shape.is_empty());
            assert_eq!(values[&witness].to_f64_lossy_vec(), [4.0]);
        }
    }
}

#[test]
fn discarded_result_keeps_a_potentially_failing_witness() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let witness = claimed_witness(&mut dag, decl, &[4]);
    let result = dag.add_node(
        decl,
        RiscOp::Const {
            value: scalar_from_i64("load", Prim::Int64, 9).unwrap(),
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    dag.node_mut(result).unwrap().shape_deps.push(witness);
    dag.add_root(result);
    let dag = dead_code_eliminate(&dag);
    let error =
        eval_tensor_with(&dag, |_| Some(TensorValue::from_vec(vec![5], vec![7.0; 5]))).unwrap_err();
    assert!(error.contains("claimed = 4, x axis 0 = 5"), "{error}");
}

/// A call is stamped with its caller's declaration (spec/10 section 3.2), so
/// another invocation's witness is one a declaration the root does not enter
/// owns: [06] section 5.2 scopes potentially trapping liveness to the program
/// the evaluation runs, and the witness is dropped. A witness the root's own
/// declaration owns is that declaration's discarded call, whose initializer
/// runs whether or not its value is read (spec/03 section 4.4), so its
/// literal claim stays a seed even though no value reaches it.
#[test]
fn unrelated_root_does_not_activate_another_invocations_witness() {
    let build = |witness_is_the_roots: bool| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let other = dag.declare("other");
        claimed_witness(
            &mut dag,
            if witness_is_the_roots { decl } else { other },
            &[4],
        );
        let result = dag.add_node(
            decl,
            RiscOp::Const {
                value: scalar_from_i64("load", Prim::Int64, 9).unwrap(),
            },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        dag.add_root(result);
        dead_code_eliminate(&dag)
    };
    let unrelated = build(false);
    assert!(
        unrelated
            .nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::ExtentWitness { .. }))
    );
    assert!(eval_tensor_with(&unrelated, |_| None).is_ok());
    let discarded = build(true);
    let error = eval_tensor_with(&discarded, |_| {
        Some(TensorValue::from_vec(vec![5], vec![7.0; 5]))
    })
    .unwrap_err();
    assert!(error.contains("claimed = 4, x axis 0 = 5"), "{error}");
}

#[test]
fn malformed_witnesses_are_rejected() {
    for malformed in 0..4 {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let witness = claimed_witness(&mut dag, decl, &[4]);
        dag.add_root(witness);
        assert!(chelis_ir::verify::verify(&dag).is_empty());
        let node = dag.node_mut(witness).unwrap();
        match malformed {
            0 => node.output_type.precision = Prim::F32,
            1 => node.output_type.dims.push(DimInfo::Lit(1)),
            2 => {
                if let RiscOp::ExtentWitness { axis, .. } = &mut node.op {
                    *axis = RtAxis::Lit(1);
                }
            }
            3 => {
                if let RiscOp::ExtentWitness { requirements, .. } = &mut node.op {
                    requirements[0] = scalar_from_i64("load", Prim::Int64, -1).unwrap();
                }
            }
            _ => unreachable!(),
        }
        assert!(
            !chelis_ir::verify::verify(&dag).is_empty(),
            "malformed case {malformed}"
        );
    }
}

#[test]
fn graph_rebuilds_preserve_witness_claims_and_call_provenance() {
    use chelis_ir::optimize::{common_subexpr_eliminate, constant_fold};
    for actual in [4, 5] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let first = claimed_witness(&mut dag, decl, &[4]);
        let original = dag.get(first).unwrap().clone();
        let second = dag.add_node(
            decl,
            original.op.clone(),
            original.inputs.clone(),
            original.output_type.clone(),
            Some("call-g".into()),
        );
        dag.node_mut(second).unwrap().shape_deps.push(first);
        dag.add_root(second);
        constant_fold(&mut dag);
        let dag = dead_code_eliminate(&common_subexpr_eliminate(&dag));
        let witnesses = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::ExtentWitness { .. }))
            .collect::<Vec<_>>();
        assert_eq!(witnesses.len(), 2, "distinct call checks must not CSE");
        assert_eq!(witnesses[0].span_id.as_deref(), Some("call-f"));
        assert_eq!(witnesses[1].span_id.as_deref(), Some("call-g"));
        let outcome = eval_tensor_with(&dag, |_| {
            Some(TensorValue::from_vec(vec![actual], vec![7.0; actual]))
        });
        assert_eq!(outcome.is_ok(), actual == 4);
        if actual == 5 {
            assert!(outcome.unwrap_err().contains("claimed = 4, x axis 0 = 5"));
        }
    }
}

#[test]
fn vectorization_shares_witness_and_shifts_its_source_axis() {
    for actual in [4, 5] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let witness = claimed_witness(&mut dag, decl, &[4]);
        dag.add_root(witness);
        let dag = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
        let witness = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::ExtentWitness { .. }))
            .unwrap();
        assert!(witness.output_type.dims.is_empty());
        assert!(matches!(
            witness.op,
            RiscOp::ExtentWitness {
                axis: RtAxis::Lit(1),
                ..
            }
        ));
        let outcome = eval_tensor_with(&dag, |_| {
            Some(TensorValue::from_vec(
                vec![2, actual],
                vec![7.0; 2 * actual],
            ))
        });
        if actual == 4 {
            let values = outcome.unwrap();
            let value = &values[&dag.roots()[0]];
            assert_eq!(value.shape, [2]);
            assert_eq!(value.to_f64_lossy_vec(), [4.0, 4.0]);
        } else {
            assert!(outcome.unwrap_err().contains("claimed = 4, x axis 1 = 5"));
        }
    }
}

#[test]
fn gradient_retains_primal_shape_checks() {
    for actual in [4, 5] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let witness = claimed_witness(&mut dag, decl, &[4]);
        let input = dag.get(witness).unwrap().inputs[0];
        let output = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![input],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        dag.node_mut(output).unwrap().shape_deps.push(witness);
        dag.add_root(output);
        let gradient =
            chelis_ir::grad::grad_dag(&dag, output, &[input]).expect("differentiable sum");
        let outcome = eval_tensor_with(&gradient.dag, |_| {
            Some(TensorValue::from_vec(vec![actual], vec![7.0; actual]))
        });
        assert_eq!(outcome.is_ok(), actual == 4);
        if actual == 5 {
            assert!(outcome.unwrap_err().contains("claimed = 4, x axis 0 = 5"));
        }
    }
}
