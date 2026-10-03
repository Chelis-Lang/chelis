use chelis_ir::{Dag, DimInfo, RiscOp, TensorType, compositions, tier2, verify, vmap};
use chelis_types::types::Prim;

fn graph(precision: Prim, axis: usize) -> Dag {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
        precision,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let y = tier2::lower_softmax(owner.into(), &mut dag, x, axis, &ty, Some("softmax.source"));
    dag.add_root(y);
    dag
}
#[test]
fn semantic_identity_survives_batching_and_only_post_ad_decomposes() {
    let dag = graph(Prim::F16, 1);
    assert_eq!(dag.len(), 2);
    assert_eq!(dag.nodes()[1].op, RiscOp::Softmax { axis: 1 });
    assert!(verify::verify(&dag).is_empty());
    let batched = vmap::vectorize_axis0(&dag, DimInfo::Lit(3)).unwrap();
    assert!(
        batched
            .nodes()
            .iter()
            .any(|n| n.op == RiscOp::Softmax { axis: 2 })
    );
    assert!(verify::verify(&batched).is_empty());
    let lowered = compositions::decompose(&batched);
    assert!(
        !lowered
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Softmax { .. }))
    );
    assert!(
        lowered
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::MaxReduce { axis: 2 }))
    );
    assert!(verify::verify(&lowered).is_empty());
    let twice = compositions::decompose(&lowered);
    assert_eq!(
        serde_json::to_value(&lowered).unwrap(),
        serde_json::to_value(&twice).unwrap()
    );
    let root = lowered.get(lowered.roots()[0]).unwrap();
    assert_eq!(root.span_id.as_deref(), Some("softmax.source"));
    assert_eq!(root.owner, batched.get(batched.roots()[0]).unwrap().owner);
}
#[test]
fn malformed_softmax_identities_reject_before_execution() {
    assert!(!verify::verify(&graph(Prim::Int32, 1)).is_empty());
    assert!(!verify::verify(&graph(Prim::F32, 2)).is_empty());
    let mut dag = graph(Prim::F32, 1);
    let root = dag.roots()[0];
    dag.node_mut(root).unwrap().inputs.clear();
    assert!(!verify::verify(&dag).is_empty());
}

#[test]
fn decomposition_preserves_activation_provenance_and_nonvalue_edges() {
    use chelis_ir::dag::Owner;
    let dag = graph(Prim::F32, 1);
    // Rebuild with metadata dependencies preceding the softmax.
    let mut source = Dag::new();
    let owner = source.declare("metadata");
    let predicate = source.add_node(
        owner,
        RiscOp::Load {
            name: "active".into(),
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    let ty = dag.nodes()[0].output_type.clone();
    let x = source.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let shape = source.add_node(
        owner,
        RiscOp::Shape { axis: 0 },
        vec![x],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let claim = source.add_node(
        owner,
        RiscOp::ExtentWitness {
            site: chelis_ir::dag::ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: chelis_ir::dag::RtAxis::Lit(0),
            requirements: vec![chelis_types::scalar_from_i64("test", Prim::Int64, 2).unwrap()],
            claims: vec![],
        },
        vec![],
        TensorType { dims: vec![], precision: Prim::Int64 },
        None,
    );
    let y = tier2::lower_softmax(
        Owner::new(owner, Some(predicate)),
        &mut source,
        x,
        1,
        &ty,
        Some("source"),
    );
    source.add_shape_dep(y, shape);
    source.add_result_claim_dep(y, claim);
    source
        .node_mut(y)
        .unwrap()
        .merged_spans
        .push("merged".into());
    source.add_root(y);
    let lowered = compositions::decompose(&source);
    let result = lowered.get(lowered.roots()[0]).unwrap();
    assert_eq!(result.shape_deps, vec![shape]);
    assert_eq!(result.result_claim_deps, vec![claim]);
    assert_eq!(result.merged_spans, vec!["merged"]);
    assert_eq!(result.owner.activation, Some(predicate));
    assert_eq!(lowered.declarations(), source.declarations());
    assert!(
        lowered
            .nodes()
            .iter()
            .skip(4)
            .all(|n| n.owner.activation == Some(predicate))
    );
    assert!(verify::verify(&lowered).is_empty());
}

#[test]
fn decomposition_refuses_an_unmapped_metadata_dependency() {
    let mut dag = graph(Prim::F32, 1);
    let root = dag.roots()[0];
    dag.add_shape_dep(root, chelis_ir::NodeId(999));
    assert!(std::panic::catch_unwind(|| compositions::decompose(&dag)).is_err());
}

#[test]
fn empty_axis_traps_only_when_the_retained_softmax_is_active() {
    use chelis_ir::{
        dag::Owner,
        eval::{self, TensorValue},
    };
    use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
    let mut dag = Dag::new();
    let owner = dag.declare("empty");
    let active = dag.add_node(
        owner,
        RiscOp::Load {
            name: "active".into(),
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    let ty = TensorType {
        dims: vec![DimInfo::Lit(0), DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let y = tier2::lower_softmax(Owner::new(owner, Some(active)), &mut dag, x, 0, &ty, None);
    dag.add_root(y);
    for flag in [false, true] {
        let result = eval::eval_tensor_roots_with_strict(&dag, &[y], |name| match name {
            "active" => Some(
                TensorValue::finalize_from_wide_int(
                    "test",
                    Prim::Bool,
                    vec![],
                    vec![i64::from(flag)],
                )
                .unwrap(),
            ),
            "x" => Some(TensorValue::from_storage(
                vec![0, 4],
                finalize_tensor("test", Prim::F32, RawTensor::Float(vec![])).unwrap(),
            )),
            _ => None,
        });
        if flag {
            assert!(result.is_err(), "an active empty-axis softmax must trap");
        } else {
            assert_eq!(result.unwrap()[&y].shape, vec![0, 4]);
        }
    }
}

#[test]
fn decomposition_refuses_an_unmapped_result_claim_dependency() {
    let mut dag = graph(Prim::F32, 1);
    let root = dag.roots()[0];
    dag.add_result_claim_dep(root, chelis_ir::NodeId(999));
    assert!(std::panic::catch_unwind(|| compositions::decompose(&dag)).is_err());
}
