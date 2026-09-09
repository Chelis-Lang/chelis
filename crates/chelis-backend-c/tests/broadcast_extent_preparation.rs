//! [05-MOV-1], #1619: preparing anonymous broadcast axes must preserve the
//! size source, the operand's unit precondition, and numeric result claims.

use chelis_backend_c::{CodegenOptions, prepare_dag_for_codegen};
use chelis_ir::axis_sources::{
    DimClaim, derive_runtime_dim_classes, derive_unit_extent_claims, member_load_axis,
};
use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};
use chelis_types::types::Prim;

fn named(name: &str) -> DimInfo {
    DimInfo::Named(name.into(), None)
}

fn ty(dims: Vec<DimInfo>) -> TensorType {
    TensorType {
        dims,
        precision: Prim::F32,
    }
}

fn fixture(
    operand: Vec<DimInfo>,
    result: Vec<DimInfo>,
    size: RtDim,
) -> (Dag, chelis_ir::dag::NodeId) {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty(operand), None);
    let sizes = dag.add_node(
        RiscOp::Load {
            name: "sizes".into(),
        },
        vec![],
        ty(vec![DimInfo::Lit(5), named("width")]),
        None,
    );
    let inputs = if matches!(size, RtDim::Lit(_)) {
        vec![x]
    } else {
        vec![x, sizes]
    };
    let result = dag.add_node(RiscOp::Expand { axis: 1, size }, inputs, ty(result), None);
    dag.add_root(result);
    (dag, result)
}

fn source() -> RtDim {
    RtDim::InputAxis {
        tensor: 1,
        axis: RtAxis::Lit(1),
    }
}

#[test]
fn anonymous_broadcast_axes_follow_their_own_sources() {
    for (size, extent) in [
        (source(), named("width")),
        (RtDim::Lit(7), DimInfo::Lit(7)),
        (RtDim::Lit(0), DimInfo::Lit(0)),
    ] {
        let (dag, result) = fixture(
            vec![DimInfo::Lit(2), DimInfo::Lit(1), DimInfo::Lit(4)],
            vec![named("*"), named("*"), named("*")],
            size.clone(),
        );
        let prepared = prepare_dag_for_codegen(dag, CodegenOptions::default());
        let result = prepared.get(result).unwrap();
        assert_eq!(
            result.output_type.dims,
            vec![DimInfo::Lit(2), extent, DimInfo::Lit(4)]
        );
        assert_eq!(result.op, RiscOp::Expand { axis: 1, size });
        assert!(
            derive_unit_extent_claims(&prepared).is_empty(),
            "the operand's literal unit extent proves only its precondition"
        );
    }
}

#[test]
fn a_runtime_unit_claim_still_reads_the_operand_axis() {
    let (dag, result) = fixture(
        vec![DimInfo::Lit(2), named("unit"), DimInfo::Lit(4)],
        vec![named("*"), named("*"), named("*")],
        source(),
    );
    let prepared = prepare_dag_for_codegen(dag, CodegenOptions::default());
    assert_eq!(
        prepared.get(result).unwrap().output_type.dims[1],
        named("width")
    );
    let claims = derive_unit_extent_claims(&prepared);
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].operand, prepared.get(result).unwrap().inputs[0]);
    assert_eq!(claims[0].axis, 1);
}

#[test]
fn an_anonymous_bystander_does_not_erase_the_literal_result_extent() {
    let (dag, result) = fixture(
        vec![DimInfo::Lit(2), DimInfo::Lit(1), DimInfo::Lit(4)],
        vec![named("*"), DimInfo::Lit(4), DimInfo::Lit(4)],
        source(),
    );
    let prepared = prepare_dag_for_codegen(dag, CodegenOptions::default());
    assert_eq!(
        prepared.get(result).unwrap().output_type.dims,
        vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(4)]
    );
    assert!(
        derive_runtime_dim_classes(&prepared)
            .iter()
            .any(|class| class.claim == DimClaim::Literal(4)),
        "the declared 4 is still an obligation against the independent size source"
    );
}

#[test]
fn inserted_axes_shift_bystander_sources() {
    let (dag, result) = fixture(
        vec![DimInfo::Lit(2), DimInfo::Lit(4)],
        vec![named("*"), named("*"), named("*")],
        source(),
    );
    let prepared = prepare_dag_for_codegen(dag, CodegenOptions::default());
    assert_eq!(
        prepared.get(result).unwrap().output_type.dims,
        vec![DimInfo::Lit(2), named("width"), DimInfo::Lit(4)]
    );
    assert!(
        derive_unit_extent_claims(&prepared).is_empty(),
        "insert has no unit precondition"
    );
}

#[test]
fn an_anonymous_resolved_claim_keeps_its_required_number() {
    let (dag, result) = fixture(
        vec![DimInfo::Lit(2), DimInfo::Lit(1), DimInfo::Lit(4)],
        vec![
            named("*"),
            DimInfo::Named("*".into(), Some(4)),
            DimInfo::Lit(4),
        ],
        source(),
    );
    let prepared = prepare_dag_for_codegen(dag, CodegenOptions::default());
    assert_eq!(
        prepared.get(result).unwrap().output_type.dims[1],
        DimInfo::Lit(4)
    );
    assert!(
        derive_runtime_dim_classes(&prepared)
            .iter()
            .any(|class| class.claim == DimClaim::Literal(4))
    );
}

#[test]
fn an_unread_named_claim_keeps_its_existing_rejection_path() {
    let (mut dag, result) = fixture(
        vec![DimInfo::Lit(2), named("unit"), DimInfo::Lit(4)],
        vec![named("*"), named("claimed"), named("*")],
        source(),
    );
    dag.add_node(
        RiscOp::Load {
            name: "expected".into(),
        },
        vec![],
        ty(vec![named("claimed")]),
        None,
    );
    let prepared = prepare_dag_for_codegen(dag, CodegenOptions::default());
    let inputs = &prepared.get(result).unwrap().inputs;
    // Current-behavior boundary, not named-claim acceptance: with x axis 1=1
    // and sizes axis 1=3, the existing preparation rejects via these reads.
    // Dropping that comparison newly executes a wrong [2, claimed, 4].
    // B2b-1 still owes the correctly scoped expected-vs-sizes comparison.
    assert!(
        derive_runtime_dim_classes(&prepared).iter().any(|class| {
            let reads: Vec<_> = class
                .members
                .iter()
                .filter_map(|member| member_load_axis(&prepared, member))
                .collect();
            reads.contains(&(inputs[0], 1)) && reads.contains(&(inputs[1], 1))
        }),
        "the unread named witness must not turn a rejected mismatch into unchecked execution"
    );
}
