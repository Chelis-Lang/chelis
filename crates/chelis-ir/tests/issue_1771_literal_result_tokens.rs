//! Spec/04 §4.7: declarations own distinct ordered obligations at their producer.
use chelis_ir::dag::{Dag, DimInfo, ExtentWitnessSite, NodeId, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::optimize::{common_subexpr_eliminate, dead_code_eliminate};
use chelis_types::{scalar_from_i64, types::Prim};

fn token(dag: &mut Dag, decl: chelis_ir::dag::DeclId, required: i64) -> NodeId {
    dag.add_node(
        decl,
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: RtAxis::Lit(0),
            requirements: vec![scalar_from_i64("load", Prim::Int64, required).unwrap()],
            claims: vec![],
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    )
}

fn fixture(outer: i64, inner: i64, cast: bool) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    // An enclosing declaration captures first, but executes after the inner
    // declaration's obligation at their common producing expression.
    let outer = token(&mut dag, decl, outer);
    let inner = token(&mut dag, decl, inner);
    let ty = TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    };
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![input, input], ty.clone(), None);
    let producer = if cast {
        dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![add],
            ty,
            None,
        )
    } else {
        add
    };
    dag.add_shape_dep(producer, inner);
    dag.add_shape_dep(producer, outer);
    dag.add_root(producer);
    dag
}

#[test]
fn distinct_literal_tokens_keep_declaration_order_and_primitive_through_rebuilds() {
    for cast in [false, true] {
        for (outer, inner, required) in [(2, 2, None), (3, 2, Some(3)), (3, 4, Some(4))] {
            let dag = fixture(outer, inner, cast);
            for rebuilt in [
                dag.clone(),
                common_subexpr_eliminate(&dag),
                dead_code_eliminate(&dag),
                chelis_ir::specialize::specialize_for_exact_arithmetic(&dag),
            ] {
                assert!(chelis_ir::verify::verify(&rebuilt).is_empty());
                assert_eq!(
                    rebuilt
                        .nodes()
                        .iter()
                        .filter(|node| matches!(
                            node.op,
                            RiscOp::ExtentWitness {
                                site: ExtentWitnessSite::LiteralResultClaim,
                                ..
                            }
                        ))
                        .count(),
                    2
                );
                let result = eval_tensor_with(&rebuilt, |_| {
                    Some(TensorValue::from_vec(vec![2], vec![1.0, 2.0]))
                });
                if let Some(required) = required {
                    let operation = if cast { "cast" } else { "add" };
                    assert_eq!(
                        result.unwrap_err(),
                        format!(
                            "extent `{required}`: claimed = {required}, {operation} axis 0 = 2\nnumeric trap: domain in {operation} at i64"
                        )
                    );
                } else {
                    let values = result.unwrap();
                    let value = &values[&rebuilt.roots()[0]];
                    assert_eq!(value.shape, vec![2]);
                    assert_eq!(value.to_f64_lossy_vec(), vec![2.0, 4.0]);
                }
            }
        }
    }
}

#[test]
fn literal_role_rejects_observations_entry_claims_and_malformed_requirements() {
    for mutation in [
        "input",
        "missing",
        "extra",
        "negative",
        "precision",
        "axis",
        "parameter",
        "dependency",
        "entry-claim",
        "missing-owner",
        "multiple-owners",
    ] {
        let mut dag = fixture(2, 2, false);
        let decl = dag.nodes()[0].decl;
        match mutation {
            "input" => dag.node_mut(NodeId(1)).unwrap().inputs.push(NodeId(0)),
            "dependency" => dag.add_shape_dep(NodeId(1), NodeId(0)),
            "precision" => dag.node_mut(NodeId(1)).unwrap().output_type.precision = Prim::F32,
            "missing-owner" => dag
                .node_mut(NodeId(3))
                .unwrap()
                .shape_deps
                .retain(|id| *id != NodeId(1)),
            "multiple-owners" => {
                let ty = dag.get(NodeId(3)).unwrap().output_type.clone();
                let copy = dag.add_node(decl, RiscOp::Copy, vec![NodeId(3)], ty, None);
                dag.add_shape_dep(copy, NodeId(1));
            }
            _ => {
                let RiscOp::ExtentWitness {
                    requirements,
                    axis,
                    parameter,
                    claims,
                    ..
                } = &mut dag.node_mut(NodeId(1)).unwrap().op
                else {
                    unreachable!()
                };
                match mutation {
                    "missing" => requirements.clear(),
                    "extra" => requirements.push(requirements[0]),
                    "negative" => {
                        requirements[0] = scalar_from_i64("load", Prim::Int64, -1).unwrap()
                    }
                    "axis" => *axis = RtAxis::Lit(1),
                    "parameter" => *parameter = "x".into(),
                    "entry-claim" => claims.push(chelis_ir::dag::ExtentClaim {
                        claim: "n".into(),
                        requirement_declares: true,
                    }),
                    _ => unreachable!(),
                }
            }
        }
        assert!(!chelis_ir::verify::verify(&dag).is_empty(), "{mutation}");
    }
}

/// Administrative return copies retain the numeric operation's attribution;
/// a fusion may not replace it with the fusion implementation's identity.
#[test]
fn administrative_copies_preserve_literal_producer_through_fusion() {
    for copied in [false, true] {
        for required in [2, 3] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let claim = token(&mut dag, decl, required);
            let ty = TensorType {
                dims: vec![DimInfo::Named("*".into(), None)],
                precision: Prim::F32,
            };
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let add = dag.add_node(decl, RiscOp::Add, vec![input, input], ty.clone(), None);
            let mul = dag.add_node(decl, RiscOp::Mul, vec![add, input], ty.clone(), None);
            let owner = if copied {
                dag.add_node(decl, RiscOp::Copy, vec![mul], ty, None)
            } else {
                mul
            };
            dag.add_shape_dep(owner, claim);
            dag.add_root(owner);
            let mut unclaimed = dag.clone();
            unclaimed.node_mut(owner).unwrap().shape_deps.clear();
            let unclaimed = dead_code_eliminate(&unclaimed);
            assert!(
                chelis_ir::fuse::fuse(&unclaimed)
                    .nodes()
                    .iter()
                    .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
                "the control must exercise fusion"
            );
            let mut folded = dag.clone();
            chelis_ir::optimize::constant_fold(&mut folded);
            for rebuilt in [
                dag.clone(),
                chelis_ir::fuse::fuse(&dag),
                common_subexpr_eliminate(&dag),
                dead_code_eliminate(&dag),
                chelis_ir::specialize::specialize_for_exact_arithmetic(&dag),
                folded,
            ] {
                assert!(chelis_ir::verify::verify(&rebuilt).is_empty());
                let result = eval_tensor_with(&rebuilt, |_| {
                    Some(TensorValue::from_vec(vec![2], vec![1.0, 2.0]))
                });
                if required == 2 {
                    let values = result.unwrap();
                    assert_eq!(
                        values[&rebuilt.roots()[0]].to_f64_lossy_vec(),
                        vec![2.0, 8.0]
                    );
                } else {
                    assert_eq!(
                        result.unwrap_err(),
                        "extent `3`: claimed = 3, mul axis 0 = 2\nnumeric trap: domain in mul at i64",
                        "copied={copied}, {rebuilt:?}"
                    );
                }
            }
        }
    }
}
