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

#[derive(Clone, Copy)]
enum InteriorClaim {
    None,
    Named,
    Literal,
}

fn wildcard_f32() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    }
}

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn vec_i32(len: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(len)],
        precision: Prim::Int32,
    }
}

fn interior_claim_token(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    claim: InteriorClaim,
    required: i64,
) -> Option<NodeId> {
    match claim {
        InteriorClaim::None => None,
        InteriorClaim::Literal => Some(dag.add_node(
            decl,
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::LiteralResultClaim,
                parameter: String::new(),
                axis: RtAxis::Lit(0),
                requirements: vec![scalar_from_i64("test", Prim::Int64, required).unwrap()],
                claims: Vec::new(),
            },
            Vec::new(),
            TensorType {
                dims: Vec::new(),
                precision: Prim::Int64,
            },
            None,
        )),
        InteriorClaim::Named => {
            let declared = dag.add_node(
                decl,
                RiscOp::Load {
                    name: "declared".into(),
                },
                Vec::new(),
                TensorType {
                    dims: vec![DimInfo::Named("n".into(), None)],
                    precision: Prim::F32,
                },
                None,
            );
            let caller = dag.add_node(
                decl,
                RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::Caller,
                    parameter: "declared".into(),
                    axis: RtAxis::Lit(0),
                    requirements: Vec::new(),
                    claims: Vec::new(),
                },
                vec![declared],
                TensorType {
                    dims: Vec::new(),
                    precision: Prim::Int64,
                },
                None,
            );
            let claim = dag.add_node(
                decl,
                RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::ResultClaim {
                        claim: "n".into(),
                        axis: RtAxis::Lit(0),
                    },
                    parameter: "declared".into(),
                    axis: RtAxis::Lit(0),
                    requirements: Vec::new(),
                    claims: Vec::new(),
                },
                vec![declared],
                TensorType {
                    dims: Vec::new(),
                    precision: Prim::Int64,
                },
                None,
            );
            dag.add_shape_dep(claim, caller);
            Some(claim)
        }
    }
}

fn blas_pattern(claim: InteriorClaim) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let claim = interior_claim_token(&mut dag, decl, claim, 5);
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        Vec::new(),
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        Vec::new(),
        mat_f32(3, 4),
        None,
    );
    let expand_a = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
        None,
    );
    let expand_b = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![expand_a, expand_b],
        tensor3_f32(2, 3, 4),
        None,
    );
    if let Some(claim) = claim {
        dag.add_result_claim_dep(mul, claim);
    }
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
    dag.add_root(sum);
    dag
}

fn eval_blas_pattern(dag: &Dag, declared: Option<usize>) -> Result<Vec<f64>, String> {
    let values = eval_tensor_with(dag, |name| match name {
        "declared" => declared.map(|extent| TensorValue::from_vec(vec![extent], vec![0.0; extent])),
        "a" => Some(TensorValue::from_vec(vec![2, 3], vec![1.0; 6])),
        "b" => Some(TensorValue::from_vec(vec![3, 4], vec![1.0; 12])),
        _ => None,
    })?;
    Ok(values[&dag.roots()[0]].to_f64_lossy_vec())
}

fn dense_gather_pattern(claim: InteriorClaim) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let claim = interior_claim_token(&mut dag, decl, claim, 5);
    let values = dag.add_node(
        decl,
        RiscOp::Load {
            name: "values".into(),
        },
        Vec::new(),
        mat_f32(2, 3),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        Vec::new(),
        vec_i32(4),
        None,
    );
    let one_hot = dag.add_node(
        decl,
        RiscOp::OneHot { vocab: 2 },
        vec![indices],
        mat_f32(4, 2),
        None,
    );
    let expand_one_hot = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(3),
        },
        vec![one_hot],
        tensor3_f32(4, 2, 3),
        None,
    );
    let expand_values = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![values],
        tensor3_f32(4, 2, 3),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![expand_one_hot, expand_values],
        tensor3_f32(4, 2, 3),
        None,
    );
    if let Some(claim) = claim {
        dag.add_result_claim_dep(mul, claim);
    }
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat_f32(4, 3),
        None,
    );
    dag.add_root(sum);
    dag
}

fn eval_dense_gather_pattern(dag: &Dag, declared: Option<usize>) -> Result<Vec<f64>, String> {
    let values = eval_tensor_with(dag, |name| match name {
        "declared" => declared.map(|extent| TensorValue::from_vec(vec![extent], vec![0.0; extent])),
        "values" => Some(TensorValue::from_vec(
            vec![2, 3],
            vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        )),
        "indices" => Some(TensorValue::from_vec(vec![4], vec![0.0, 1.0, 0.0, 1.0])),
        _ => None,
    })?;
    Ok(values[&dag.roots()[0]].to_f64_lossy_vec())
}

fn assert_specialized_claim_trap(
    dag: &Dag,
    specialize: impl Fn(&Dag) -> Dag,
    eval: impl Fn(&Dag) -> Result<Vec<f64>, String>,
    expected: &str,
) -> Dag {
    assert!(verify(dag).is_empty());
    assert_eq!(eval(dag).unwrap_err(), expected);
    let specialized = specialize(dag);
    assert!(
        verify(&specialized).is_empty(),
        "a declined replacement must leave a valid executable graph: {:?}",
        verify(&specialized)
    );
    assert_eq!(
        eval(&specialized).unwrap_err(),
        expected,
        "specialization plus DCE must preserve the interior producer trap and attribution"
    );
    specialized
}

#[test]
fn blas_interior_named_claim_blocks_replacement_and_traps_after_dce() {
    let dag = blas_pattern(InteriorClaim::Named);
    let specialized = assert_specialized_claim_trap(
        &dag,
        chelis_ir::specialize::specialize_for_blas,
        |dag| eval_blas_pattern(dag, Some(5)),
        "extent `n`: claimed = 5, mul axis 0 = 2\n\
         numeric trap: domain in mul at i64",
    );
    assert!(
        !specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
        "the complete BLAS match must be declined when an interior producer is claimed"
    );
}

#[test]
fn blas_interior_literal_claim_blocks_replacement_and_traps_after_dce() {
    let dag = blas_pattern(InteriorClaim::Literal);
    let specialized = assert_specialized_claim_trap(
        &dag,
        chelis_ir::specialize::specialize_for_blas,
        |dag| eval_blas_pattern(dag, None),
        "extent `5`: claimed = 5, mul axis 0 = 2\n\
         numeric trap: domain in mul at i64",
    );
    assert!(
        !specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
        "literal and named claims use the same complete BLAS-region barrier"
    );
}

#[test]
fn unclaimed_blas_region_still_specializes_and_executes() {
    let dag = blas_pattern(InteriorClaim::None);
    let expected = eval_blas_pattern(&dag, None).unwrap();
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    assert!(verify(&specialized).is_empty());
    assert!(
        specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. }))
    );
    assert_eq!(eval_blas_pattern(&specialized, None).unwrap(), expected);
}

#[test]
fn dense_gather_interior_named_claim_blocks_replacement_and_traps_after_dce() {
    let dag = dense_gather_pattern(InteriorClaim::Named);
    let specialized = assert_specialized_claim_trap(
        &dag,
        chelis_ir::specialize::specialize_for_exact_arithmetic,
        |dag| eval_dense_gather_pattern(dag, Some(5)),
        "extent `n`: claimed = 5, mul axis 0 = 4\n\
         numeric trap: domain in mul at i64",
    );
    assert!(
        !specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Gather { .. })),
        "the complete dense-gather match must be declined when an interior producer is claimed"
    );
}

#[test]
fn dense_gather_interior_literal_claim_blocks_replacement_and_traps_after_dce() {
    let dag = dense_gather_pattern(InteriorClaim::Literal);
    let specialized = assert_specialized_claim_trap(
        &dag,
        chelis_ir::specialize::specialize_for_exact_arithmetic,
        |dag| eval_dense_gather_pattern(dag, None),
        "extent `5`: claimed = 5, mul axis 0 = 4\n\
         numeric trap: domain in mul at i64",
    );
    assert!(
        !specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Gather { .. })),
        "literal and named claims use the same complete dense-gather-region barrier"
    );
}

#[test]
fn unclaimed_dense_gather_region_still_specializes_and_executes() {
    let dag = dense_gather_pattern(InteriorClaim::None);
    let expected = eval_dense_gather_pattern(&dag, None).unwrap();
    let specialized = chelis_ir::specialize::specialize_for_exact_arithmetic(&dag);
    assert!(verify(&specialized).is_empty());
    assert!(
        specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Gather { axis: 0 }))
    );
    assert_eq!(
        eval_dense_gather_pattern(&specialized, None).unwrap(),
        expected
    );
}

fn named_claim_fusion_dag(claimed: bool, producer_is_interior: bool) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let declaration_input = dag.add_node(
        decl,
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
        decl,
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
        decl,
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
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        wildcard_f32(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        wildcard_f32(),
        None,
    );
    let left = if producer_is_interior {
        dag.add_node(decl, RiscOp::Exp, vec![left], wildcard_f32(), None)
    } else {
        left
    };
    let add = dag.add_node(decl, RiscOp::Add, vec![left, right], wildcard_f32(), None);
    if claimed {
        dag.add_result_claim_dep(add, claim);
    }
    let neg = dag.add_node(decl, RiscOp::Neg, vec![add], wildcard_f32(), None);
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

fn named_claim_identity_cast_dag() -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let declaration_input = dag.add_node(
        decl,
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
        decl,
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
        decl,
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
    let value = dag.add_node(
        decl,
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        wildcard_f32(),
        None,
    );
    let cast = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![value],
        wildcard_f32(),
        None,
    );
    dag.add_result_claim_dep(cast, claim);
    dag.add_root(cast);
    dag
}

fn eval_identity_claim(
    dag: &Dag,
    declared_extent: usize,
    runtime_extent: usize,
) -> Result<Vec<f64>, String> {
    let values = eval_tensor_with(dag, |name| match name {
        "declared" => Some(TensorValue::from_vec(
            vec![declared_extent],
            vec![0.0; declared_extent],
        )),
        "value" => Some(TensorValue::from_vec(
            vec![runtime_extent],
            (0..runtime_extent).map(|index| index as f64).collect(),
        )),
        _ => None,
    })?;
    Ok(values[&dag.roots()[0]].to_f64_lossy_vec())
}

#[test]
fn named_result_claim_traps_after_specialization_and_dce() {
    let dag = named_claim_identity_cast_dag();
    assert!(verify(&dag).is_empty());
    assert_eq!(
        eval_identity_claim(&dag, 2, 3).unwrap_err(),
        "extent `n`: claimed = 2, cast axis 0 = 3\n\
         numeric trap: domain in cast at i64"
    );

    let specialized = chelis_ir::specialize::specialize_for_exact_arithmetic(&dag);
    assert!(verify(&specialized).is_empty());
    let retained_cast = specialized
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Cast { .. }))
        .expect("specialization must not eliminate a named claimed producer");
    assert_eq!(retained_cast.result_claim_deps.len(), 1);
    assert_eq!(
        eval_identity_claim(&specialized, 2, 3).unwrap_err(),
        "extent `n`: claimed = 2, cast axis 0 = 3\n\
         numeric trap: domain in cast at i64",
        "specialization plus DCE must preserve the trap and cast attribution"
    );
}

#[test]
fn agreeing_named_result_claim_survives_specialization_and_dce() {
    let dag = named_claim_identity_cast_dag();
    let specialized = chelis_ir::specialize::specialize_for_exact_arithmetic(&dag);
    assert!(verify(&specialized).is_empty());
    let retained_cast = specialized
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Cast { .. }))
        .expect("an agreeing named claim uses the same claimed-producer barrier");
    assert_eq!(retained_cast.result_claim_deps.len(), 1);
    assert_eq!(
        eval_identity_claim(&specialized, 3, 3).unwrap(),
        vec![0.0, 1.0, 2.0]
    );
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
         numeric trap: domain in add at i64"
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
         numeric trap: domain in add at i64",
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
         numeric trap: domain in add at i64"
    );
}

#[test]
fn rank_zero_inputs_are_excluded_from_same_shape_result_claim_observation() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let claim = dag.add_node(
        decl,
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
        decl,
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
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![scalar, tensor], tensor_ty, None);
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
        numeric trap: domain in add at i64"
    );
}

#[test]
fn identical_members_deduplicate_but_distinct_paths_remain() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let tensor_ty = TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        tensor_ty.clone(),
        None,
    );
    let repeated = dag.add_node(decl, RiscOp::Add, vec![x, x], tensor_ty.clone(), None);
    let distinct = dag.add_node(decl, RiscOp::Add, vec![x, y], tensor_ty, None);

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
    let decl = dag.declare("test");
    let scalar_ty = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let result_ty = TensorType {
        dims: vec![DimInfo::Named("*".into(), None)],
        precision: Prim::F32,
    };
    let left = dag.add_node(
        decl,
        RiscOp::Const {
            value: scalar_from_f64("const", Prim::F32, 1.0).unwrap(),
        },
        vec![],
        scalar_ty.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Const {
            value: scalar_from_f64("const", Prim::F32, 2.0).unwrap(),
        },
        vec![],
        scalar_ty,
        None,
    );
    let claim = dag.add_node(
        decl,
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
    let add = dag.add_node(decl, RiscOp::Add, vec![left, right], result_ty, None);
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
