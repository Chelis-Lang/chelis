//! Default-CI structural coverage for the direct nonnumeric HIP kernels (#1284).

mod support;

use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, LogicalKind, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen_hip;

fn vector(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(8)],
        precision,
    }
}

fn binary_source(op: RiscOp, input: Prim, output: Prim, name: &str) -> String {
    let mut dag = Dag::new();
    let input_ty = vector(input);
    let lhs = dag.add_node(
        RiscOp::Load { name: "lhs".into() },
        vec![],
        input_ty.clone(),
        None,
    );
    let rhs = dag.add_node(RiscOp::Load { name: "rhs".into() }, vec![], input_ty, None);
    let out = dag.add_node(op, vec![lhs, rhs], vector(output), None);
    dag.add_root(out);
    codegen_hip(&dag, name)
        .expect("direct nonnumeric HIP codegen")
        .c_source
}

#[test]
fn every_comparison_emits_a_bool8_kernel_for_every_admitted_operand_dtype() {
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
            let source = binary_source(
                RiscOp::Compare(kind),
                precision,
                Prim::Bool,
                &format!("{}_{}", kind.surf_name(), precision.name()),
            );
            assert!(
                source.contains("unsigned char *out"),
                "{} {} comparison did not emit Bool8 output:\n{source}",
                kind.surf_name(),
                precision.name()
            );
            assert!(
                source.contains("out[i] = (unsigned char)("),
                "{} {} comparison did not canonicalize Bool8:\n{source}",
                kind.surf_name(),
                precision.name()
            );
        }
    }

    for kind in [ComparisonKind::Eq, ComparisonKind::Neq] {
        let source = binary_source(
            RiscOp::Compare(kind),
            Prim::Bool,
            Prim::Bool,
            &format!("{}_bool", kind.surf_name()),
        );
        assert!(source.contains("const unsigned char *a"), "{source}");
        assert!(source.contains("unsigned char *out"), "{source}");
    }
}

#[test]
fn f16_and_bf16_comparison_decode_values_instead_of_ordering_storage_bits() {
    let f16 = binary_source(
        RiscOp::Compare(ComparisonKind::Gte),
        Prim::F16,
        Prim::Bool,
        "gte_f16",
    );
    assert!(f16.contains("chelis_f16_to_f32(a[idx_a])"), "{f16}");
    assert!(f16.contains("chelis_f16_to_f32(b[idx_b])"), "{f16}");

    let bf16 = binary_source(
        RiscOp::Compare(ComparisonKind::Lte),
        Prim::Bf16,
        Prim::Bool,
        "lte_bf16",
    );
    assert!(bf16.contains("chelis_bf16_to_f32(a[idx_a])"), "{bf16}");
    assert!(bf16.contains("chelis_bf16_to_f32(b[idx_b])"), "{bf16}");
}

#[test]
fn logical_kernels_are_eager_bool8_truth_operations() {
    for (kind, spelling) in [
        (LogicalKind::And, "(a[idx_a] != 0) && (b[idx_b] != 0)"),
        (LogicalKind::Or, "(a[idx_a] != 0) || (b[idx_b] != 0)"),
    ] {
        let source = binary_source(
            RiscOp::Logical(kind),
            Prim::Bool,
            Prim::Bool,
            kind.surf_name(),
        );
        assert!(source.contains(spelling), "{source}");
        assert!(source.contains("unsigned char *out"), "{source}");
    }

    let mut dag = Dag::new();
    let ty = vector(Prim::Bool);
    let input = dag.add_node(
        RiscOp::Load {
            name: "input".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(RiscOp::Logical(LogicalKind::Not), vec![input], ty, None);
    dag.add_root(out);
    let source = codegen_hip(&dag, "logical_not")
        .expect("logical not HIP codegen")
        .c_source;
    assert!(source.contains("a[idx_a] == 0"), "{source}");
    assert!(source.contains("unsigned char *out"), "{source}");
}

#[test]
fn where_selects_raw_stored_bits_for_every_admitted_branch_dtype() {
    for (precision, carrier) in [
        (Prim::F16, "chelis_u16"),
        (Prim::Bf16, "chelis_u16"),
        (Prim::F32, "chelis_u32"),
        (Prim::F64, "chelis_u64"),
        (Prim::Int8, "chelis_i8"),
        (Prim::Int16, "chelis_i16"),
        (Prim::Int32, "chelis_i32"),
        (Prim::Int64, "chelis_i64"),
        (Prim::Bool, "unsigned char"),
    ] {
        let mut dag = Dag::new();
        let cond = dag.add_node(
            RiscOp::Load {
                name: "cond".into(),
            },
            vec![],
            vector(Prim::Bool),
            None,
        );
        let branch_ty = vector(precision);
        let lhs = dag.add_node(
            RiscOp::Load { name: "lhs".into() },
            vec![],
            branch_ty.clone(),
            None,
        );
        let rhs = dag.add_node(
            RiscOp::Load { name: "rhs".into() },
            vec![],
            branch_ty.clone(),
            None,
        );
        let out = dag.add_node(RiscOp::Where, vec![cond, lhs, rhs], branch_ty, None);
        dag.add_root(out);
        let source = codegen_hip(&dag, &format!("where_{}", precision.name()))
            .expect("where HIP codegen")
            .c_source;
        assert!(
            source.contains(&format!("const {carrier} *a"))
                && source.contains(&format!("const {carrier} *b"))
                && source.contains(&format!("{carrier} *out")),
            "{} where did not use its raw storage carrier:\n{source}",
            precision.name()
        );
        assert!(
            source.contains("out[i] = cond[idx_cond] != 0 ? a[idx_a] : b[idx_b];"),
            "{source}"
        );
    }
}
