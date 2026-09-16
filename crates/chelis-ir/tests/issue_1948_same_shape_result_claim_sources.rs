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
use chelis_ir::dag::{Dag, DimInfo, ExtentWitnessSite, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::verify::verify;
use chelis_types::{scalar_from_f64, scalar_from_i64, types::Prim};

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
    let tensor_ty = TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    };
    let tensor = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let add = dag.add_node(RiscOp::Add, vec![scalar, tensor], tensor_ty, None);
    dag.add_shape_dep(add, claim);
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
    let add = dag.add_node(RiscOp::Add, vec![left, right], result_ty, None);
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
