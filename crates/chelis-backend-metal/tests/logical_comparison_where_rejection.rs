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
    let decl = dag.declare("test");
    let inputs = input_prims
        .iter()
        .enumerate()
        .map(|(index, precision)| {
            dag.add_node(
                decl,
                RiscOp::Load {
                    name: format!("input_{index}").into(),
                },
                vec![],
                vector(*precision),
                None,
            )
        })
        .collect();
    let out = dag.add_node(decl, op, inputs, vector(output), None);
    dag.add_root(out);
    dag
}

#[test]
fn metal_bitwise_kernels_preserve_exact_integer_width_and_shift_traps() {
    for kind in [
        chelis_types::BitwiseKind::And,
        chelis_types::BitwiseKind::Or,
        chelis_types::BitwiseKind::Xor,
        chelis_types::BitwiseKind::ShiftLeft,
        chelis_types::BitwiseKind::ShiftRight,
    ] {
        for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
            let dag = direct_dag(RiscOp::Bitwise(kind), &[prim, prim], prim);
            let source = try_codegen_metal(&dag, "bitwise")
                .expect("[05-OP-47] typed Metal kernel")
                .mm_source;
            assert!(source.contains(&format!("k_bitwise_{}_{}", kind.name(), prim.name())));
            assert!(
                source.contains(match prim {
                    Prim::Int8 => "uchar bits",
                    Prim::Int16 => "ushort bits",
                    Prim::Int32 => "uint bits",
                    Prim::Int64 => "ulong bits",
                    _ => unreachable!(),
                }),
                "bitwise kernels must use width-matched unsigned bits"
            );
            if kind.is_shift() {
                assert!(source.contains("shift amount must be non-negative, got"));
                assert!(source.contains("chelis_metal_alloc_status_word"));
            }
        }
    }
}

#[test]
fn metal_rank_one_kernel_rejects_unrepresentable_device_indices() {
    let extent = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    let mut dag = Dag::new();
    let decl = dag.declare("large_bitwise");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(extent)],
        precision: Prim::Int8,
    };
    let lhs = dag.add_node(
        decl,
        RiscOp::Load { name: "lhs".into() },
        vec![],
        ty.clone(),
        None,
    );
    let rhs = dag.add_node(
        decl,
        RiscOp::Load { name: "rhs".into() },
        vec![],
        ty.clone(),
        None,
    );
    let result = dag.add_node(
        decl,
        RiscOp::Bitwise(chelis_types::BitwiseKind::ShiftLeft),
        vec![lhs, rhs],
        ty,
        None,
    );
    dag.add_root(result);
    let error = try_codegen_metal(&dag, "large_bitwise").unwrap_err();
    assert!(error.to_string().contains("uint32 device indexing limit"));
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
