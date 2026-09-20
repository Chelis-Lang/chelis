//! Public capability admission/rejection coverage for #1284 target cells.

use chelis_compiler_api::compiler::{reject_unsupported_hip_ops, reject_unsupported_metal_ops};
use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, LogicalKind, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_vocab::DiagnosticKind;

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

fn asserted_metal_2266_rejection(dag: &Dag) {
    let error = reject_unsupported_metal_ops(dag)
        .expect_err("Metal must reject direct nonnumeric operations at the typed gate");
    assert_eq!(error.stage, "compile");
    assert_eq!(error.errors[0].kind(), DiagnosticKind::UnsupportedFeature);
    let message = &error.errors[0].message;
    assert!(message.contains("unimplemented chelis#2266"), "{message}");
    assert!(message.contains("--target c"), "{message}");
    assert!(message.contains("--target hip"), "{message}");
}

#[test]
fn hip_admits_the_complete_direct_nonnumeric_dtype_matrix() {
    for kind in [
        ComparisonKind::CmpLt,
        ComparisonKind::Lt,
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gt,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ] {
        for precision in [
            Prim::F16,
            Prim::Bf16,
            Prim::F32,
            Prim::F64,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
        ] {
            reject_unsupported_hip_ops(&direct_dag(
                RiscOp::Compare(kind),
                &[precision, precision],
                Prim::Bool,
            ))
            .unwrap_or_else(|error| {
                panic!(
                    "HIP rejected {} {}: {error:?}",
                    kind.surf_name(),
                    precision.name()
                )
            });
        }
    }

    for kind in [ComparisonKind::Eq, ComparisonKind::Neq] {
        reject_unsupported_hip_ops(&direct_dag(
            RiscOp::Compare(kind),
            &[Prim::Bool, Prim::Bool],
            Prim::Bool,
        ))
        .expect("HIP must admit bool equality");
    }
    for kind in [LogicalKind::And, LogicalKind::Or] {
        reject_unsupported_hip_ops(&direct_dag(
            RiscOp::Logical(kind),
            &[Prim::Bool, Prim::Bool],
            Prim::Bool,
        ))
        .expect("HIP must admit bool binary logic");
    }
    reject_unsupported_hip_ops(&direct_dag(
        RiscOp::Logical(LogicalKind::Not),
        &[Prim::Bool],
        Prim::Bool,
    ))
    .expect("HIP must admit bool not");

    for precision in [
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
    ] {
        reject_unsupported_hip_ops(&direct_dag(
            RiscOp::Where,
            &[Prim::Bool, precision, precision],
            precision,
        ))
        .unwrap_or_else(|error| panic!("HIP rejected where {}: {error:?}", precision.name()));
    }
}

#[test]
fn metal_rejects_every_direct_nonnumeric_identity_with_stable_typed_2266_authority() {
    for kind in [
        ComparisonKind::CmpLt,
        ComparisonKind::Lt,
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gt,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ] {
        asserted_metal_2266_rejection(&direct_dag(
            RiscOp::Compare(kind),
            &[Prim::F32, Prim::F32],
            Prim::Bool,
        ));
    }
    for kind in [LogicalKind::And, LogicalKind::Or] {
        asserted_metal_2266_rejection(&direct_dag(
            RiscOp::Logical(kind),
            &[Prim::Bool, Prim::Bool],
            Prim::Bool,
        ));
    }
    asserted_metal_2266_rejection(&direct_dag(
        RiscOp::Logical(LogicalKind::Not),
        &[Prim::Bool],
        Prim::Bool,
    ));
    asserted_metal_2266_rejection(&direct_dag(
        RiscOp::Where,
        &[Prim::Bool, Prim::F32, Prim::F32],
        Prim::F32,
    ));
}
