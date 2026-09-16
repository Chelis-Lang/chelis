//! chelis#1948 structural control: rank-0 operands are not members of a
//! positive-rank same-shape agreement relation.
//!
//! Surf deliberately rejects scalar-plus-tensor `add`, but the lowered tensor
//! DAG uses rank-0 operands as its internal scalar-broadcast idiom. This
//! hand-built graph therefore owns the control that cannot be expressed as a
//! `.ch` fixture. A literal result token is attached directly to `add`; Eval
//! must ignore the scalar's empty shape, use the positive-rank result extent,
//! and report `add` as the primitive that produced the returned value.

use chelis_ir::axis_sources::same_shape_result_agreement;
use chelis_ir::dag::{Dag, DimInfo, ExtentWitnessSite, NodeId, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::optimize::dead_code_eliminate;
use chelis_ir::verify::verify;
use chelis_types::{scalar_from_f64, scalar_from_i64, types::Prim};

fn wildcard_f32() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    }
}

fn named_claim_fusion_dag(claimed: bool, producer_is_interior: bool) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let declaration_input = dag.add_node(
        RiscOp::Load {
            name: "declared".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let declaration = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: "declared".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        },
        vec![declaration_input],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let claim = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::ResultClaim {
                claim: "n".into(),
                axis: RtAxis::Lit(0),
            },
            parameter: "declared".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        },
        vec![declaration_input],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_shape_dep(claim, declaration);
    let left = dag.add_node(
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        wildcard_f32(),
        None,
    );
    let right = dag.add_node(
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        wildcard_f32(),
        None,
    );
    let left = if producer_is_interior {
        dag.add_node(RiscOp::Exp, vec![left], wildcard_f32(), None)
    } else {
        left
    };
    let add = dag.add_node(RiscOp::Add, vec![left, right], wildcard_f32(), None);
    if claimed {
        dag.add_result_claim_dep(add, claim);
    }
    let neg = dag.add_node(RiscOp::Neg, vec![add], wildcard_f32(), None);
    dag.add_root(neg);
    (dag, add)
}

fn eval_named_claim(dag: &Dag, declared_extent: usize) -> Result<Vec<f64>, String> {
    let values = eval_tensor_with(dag, |name| match name {
        "declared" => Some(TensorValue::from_vec(
            vec![declared_extent],
            vec![0.0; declared_extent],
        )),
        "left" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        "right" => Some(TensorValue::from_vec(vec![3], vec![4.0, 5.0, 6.0])),
        _ => None,
    })?;
    Ok(values[&dag.roots()[0]].to_f64_lossy_vec())
}

#[test]
fn named_result_claim_traps_after_fusion_and_dce() {
    let (unclaimed, _) = named_claim_fusion_dag(false, false);
    let unclaimed = chelis_ir::fuse::fuse(&unclaimed);
    assert!(
        unclaimed
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        "the control must prove that the add-to-neg chain is otherwise fusible"
    );

    let (dag, add) = named_claim_fusion_dag(true, false);
    assert!(verify(&dag).is_empty());
    assert_eq!(
        eval_named_claim(&dag, 2).unwrap_err(),
        "extent `n`: claimed = 2, add axis 0 = 3\n\
         numeric trap: domain in add at int64"
    );

    let fused = dead_code_eliminate(&chelis_ir::fuse::fuse(&dag));
    assert!(verify(&fused).is_empty());
    let retained_add = fused
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Add))
        .expect("the claimed producer must retain its primitive identity");
    assert_eq!(
        retained_add.result_claim_deps.len(),
        1,
        "fusion must retain the named producer claim"
    );
    assert_eq!(
        eval_named_claim(&fused, 2).unwrap_err(),
        "extent `n`: claimed = 2, add axis 0 = 3\n\
         numeric trap: domain in add at int64",
        "fuse plus DCE must preserve both the trap and add attribution"
    );
    assert_eq!(
        retained_add.id, add,
        "the fully live control keeps its topological producer identity"
    );
}

#[test]
fn agreeing_named_result_claim_survives_fusion_and_dce() {
    let (dag, _) = named_claim_fusion_dag(true, false);
    let fused = dead_code_eliminate(&chelis_ir::fuse::fuse(&dag));
    assert!(verify(&fused).is_empty());
    let retained_add = fused
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Add))
        .expect("an agreeing claim must retain the same producer barrier");
    assert_eq!(retained_add.result_claim_deps.len(), 1);
    assert_eq!(eval_named_claim(&fused, 3).unwrap(), vec![-5.0, -7.0, -9.0]);
}

#[test]
fn an_interior_named_result_claim_splits_the_fusion_chain() {
    let (unclaimed, _) = named_claim_fusion_dag(false, true);
    assert!(
        chelis_ir::fuse::fuse(&unclaimed)
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        "the unclaimed exp-to-add-to-neg control must fuse"
    );

    let (claimed, _) = named_claim_fusion_dag(true, true);
    let fused = dead_code_eliminate(&chelis_ir::fuse::fuse(&claimed));
    let retained_add = fused
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Add))
        .expect("a claimed interior producer must split the fusion chain");
    assert_eq!(retained_add.result_claim_deps.len(), 1);
    assert_eq!(
        eval_named_claim(&fused, 2).unwrap_err(),
        "extent `n`: claimed = 2, add axis 0 = 3\n\
         numeric trap: domain in add at int64"
    );
}

#[test]
fn rank_zero_inputs_are_excluded_from_same_shape_result_claim_observation() {
    let mut dag = Dag::new();
    let claim = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: RtAxis::Lit(0),
            requirements: vec![scalar_from_i64("load", Prim::Int64, 2).unwrap()],
            claims: vec![],
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let scalar = dag.add_node(
        RiscOp::Const {
            value: scalar_from_f64("const", Prim::F32, 1.0).unwrap(),
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let tensor_ty = wildcard_f32();
    let tensor = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let add = dag.add_node(RiscOp::Add, vec![scalar, tensor], tensor_ty, None);
    dag.add_result_claim_dep(add, claim);
    dag.add_root(add);

    assert_eq!(
        same_shape_result_agreement(&dag, add)
            .unwrap()
            .unwrap()
            .members(),
        &[tensor],
        "rank-0 inputs are absent from the complete positive-rank relation"
    );

    let error = eval_tensor_with(&dag, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]))
    })
    .expect_err("the declared result extent 2 disagrees with add's extent 3");
    assert_eq!(
        error,
        "extent `2`: claimed = 2, add axis 0 = 3\n\
        numeric trap: domain in add at int64"
    );
}

#[test]
fn identical_members_deduplicate_but_distinct_paths_remain() {
    let mut dag = Dag::new();
    let tensor_ty = TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let y = dag.add_node(
        RiscOp::Load { name: "y".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let repeated = dag.add_node(RiscOp::Add, vec![x, x], tensor_ty.clone(), None);
    let distinct = dag.add_node(RiscOp::Add, vec![x, y], tensor_ty, None);

    assert_eq!(
        same_shape_result_agreement(&dag, repeated)
            .unwrap()
            .unwrap()
            .members(),
        &[x]
    );
    assert_eq!(
        same_shape_result_agreement(&dag, distinct)
            .unwrap()
            .unwrap()
            .members(),
        &[x, y]
    );
}

#[test]
fn malformed_same_shape_relations_are_verifier_errors() {
    let mut dag = Dag::new();
    let scalar_ty = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let result_ty = TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    };
    let left = dag.add_node(
        RiscOp::Const {
            value: scalar_from_f64("const", Prim::F32, 1.0).unwrap(),
        },
        vec![],
        scalar_ty.clone(),
        None,
    );
    let right = dag.add_node(
        RiscOp::Const {
            value: scalar_from_f64("const", Prim::F32, 2.0).unwrap(),
        },
        vec![],
        scalar_ty,
        None,
    );
    let claim = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: RtAxis::Lit(0),
            requirements: vec![scalar_from_i64("load", Prim::Int64, 2).unwrap()],
            claims: vec![],
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let add = dag.add_node(RiscOp::Add, vec![left, right], result_ty, None);
    dag.add_result_claim_dep(add, claim);
    dag.add_root(add);

    let relation = same_shape_result_agreement(&dag, add).unwrap_err();
    assert!(relation.contains("no positive-rank agreement member"));
    let errors = verify(&dag);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("no positive-rank agreement member")),
        "malformed agreement must fail before execution: {errors:?}"
    );
}
