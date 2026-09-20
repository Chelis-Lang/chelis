//! Defense-in-depth rejection at the Metal emitter boundary (#1284).

mod support;

use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, LogicalKind, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::{RejectionAuthorityKind, Stage};
use support::try_codegen_metal;

fn vector(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision,
    }
}

fn direct_dag(op: RiscOp, input_prims: &[Prim], output: Prim) -> Dag {
    let mut dag = Dag::new();
    let inputs = input_prims
        .iter()
        .enumerate()
        .map(|(index, precision)| {
            dag.add_node(
                RiscOp::Load {
                    name: format!("input_{index}").into(),
                },
                vec![],
                vector(*precision),
                None,
            )
        })
        .collect();
    let out = dag.add_node(op, inputs, vector(output), None);
    dag.add_root(out);
    dag
}

#[test]
fn metal_emitter_rejects_direct_nonnumeric_nodes_with_issue_2266_authority() {
    let cases = [
        direct_dag(
            RiscOp::Compare(ComparisonKind::Eq),
            &[Prim::F32, Prim::F32],
            Prim::Bool,
        ),
        direct_dag(
            RiscOp::Logical(LogicalKind::And),
            &[Prim::Bool, Prim::Bool],
            Prim::Bool,
        ),
        direct_dag(RiscOp::Logical(LogicalKind::Not), &[Prim::Bool], Prim::Bool),
        direct_dag(
            RiscOp::Where,
            &[Prim::Bool, Prim::F32, Prim::F32],
            Prim::F32,
        ),
    ];

    for (index, dag) in cases.iter().enumerate() {
        let error = try_codegen_metal(dag, &format!("direct_nonnumeric_{index}"))
            .expect_err("Metal emitter must defend the #1284 capability boundary");
        assert_eq!(error.stage, Stage::Codegen("metal"));
        assert_eq!(
            error.authority.kind(),
            RejectionAuthorityKind::Unimplemented
        );
        let message = error.to_string();
        assert!(message.contains("chelis#2266"), "{message}");
        assert!(message.contains("direct nonnumeric"), "{message}");
    }
}
