//! Default-CI structural coverage for the direct nonnumeric HIP kernels (#1284).

mod support;

use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, LogicalKind, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
use support::codegen_hip;

#[test]
fn generated_hip_bitwise_math_matches_exact_width_reference_on_cpu() {
    use chelis_types::{BitwiseKind, bitwise_scalar, scalar_from_i64};
    use std::process::Command;

    for (prim, signed, width, minimum, min_literal) in [
        (Prim::Int8, "int8_t", 8, i8::MIN as i64, "-128LL"),
        (Prim::Int16, "int16_t", 16, i16::MIN as i64, "-32768LL"),
        (Prim::Int32, "int32_t", 32, i32::MIN as i64, "-2147483648LL"),
        (Prim::Int64, "int64_t", 64, i64::MIN, "INT64_MIN"),
    ] {
        let lhs = [-1, 1, minimum, 8];
        let rhs = [1, width - 1, width, width + 1];
        for kind in [
            BitwiseKind::And,
            BitwiseKind::Or,
            BitwiseKind::Xor,
            BitwiseKind::ShiftLeft,
            BitwiseKind::ShiftRight,
        ] {
            let kernel = chelis_backend_hip::kernels::binary_bitwise_typed(
                1,
                "bitwise_cpu",
                kind,
                prim,
                None,
            );
            let expected = lhs
                .into_iter()
                .zip(rhs)
                .map(|(a, b)| {
                    bitwise_scalar(
                        kind,
                        scalar_from_i64("bitwise", prim, a).unwrap(),
                        scalar_from_i64("bitwise", prim, b).unwrap(),
                    )
                    .unwrap()
                    .as_i64_exact()
                    .unwrap()
                    .to_string()
                })
                .collect::<Vec<_>>()
                .join(" ");
            let rhs_init = rhs
                .iter()
                .map(|value| format!("{value}LL"))
                .collect::<Vec<_>>()
                .join(", ");
            let source = format!(
                r#"#include <cstdint>
#include <cstdio>
#define __device__
#define __global__
#define CHELIS_DEBUG_BOUNDS 0
struct Dim {{ long x; }};
Dim blockIdx{{0}}, blockDim{{4}}, threadIdx{{0}};
unsigned long long atomicCAS(unsigned long long* p, unsigned long long old, unsigned long long value) {{ auto prior=*p; if (prior==old) *p=value; return prior; }}
unsigned int atomicExch(unsigned int* p, unsigned int value) {{ auto prior=*p; *p=value; return prior; }}
{kernel}
int main() {{
    {signed} lhs[4] = {{-1, 1, ({signed}){min_literal}, 8}};
    {signed} rhs[4] = {{{rhs_init}}};
    {signed} out[4] = {{0}};
    for (int i=0; i<4; ++i) {{ threadIdx.x=i; bitwise_cpu(lhs, 1, 1, 4, rhs, 1, 1, 4, out, 4, 1, 4); }}
    for (int i=0; i<4; ++i) printf("%s%lld", i ? " " : "", (long long)out[i]);
    printf("\n");
    if ({shift}) {{
        chelis_numeric_failure_flag=0;
        chelis_numeric_failure_index=~0ULL;
        rhs[0]=-3; rhs[1]=-7;
        for (int i=0; i<4; ++i) {{ threadIdx.x=i; bitwise_cpu(lhs, 1, 1, 4, rhs, 1, 1, 4, out, 4, 1, 4); }}
        printf("TRAP %llu %lld\n", chelis_numeric_failure_index, (long long)out[0]);
    }}
    return 0;
}}
"#,
                shift = i32::from(kind.is_shift()),
            );
            let temporary = tempfile::tempdir().expect("tempdir");
            let source_file = temporary.path().join("bitwise.cpp");
            let binary = temporary.path().join("bitwise");
            std::fs::write(&source_file, &source).expect("write HIP CPU projection");
            let compile = Command::new("c++")
                .args(["-std=c++17", "-O2"])
                .arg(&source_file)
                .arg("-o")
                .arg(&binary)
                .output()
                .expect("C++ compiler");
            assert!(
                compile.status.success(),
                "{kind:?} {prim:?}: {}\n{source}",
                String::from_utf8_lossy(&compile.stderr)
            );
            let run = Command::new(&binary)
                .output()
                .expect("run HIP CPU projection");
            assert!(
                run.status.success(),
                "{kind:?} {prim:?}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let stdout = String::from_utf8(run.stdout).expect("utf8");
            assert_eq!(
                stdout.lines().next(),
                Some(expected.as_str()),
                "{kind:?} {prim:?}"
            );
            if kind.is_shift() {
                assert_eq!(
                    stdout.lines().nth(1),
                    Some("TRAP 0 -3"),
                    "{kind:?} {prim:?}"
                );
            }
        }
    }
}

fn vector(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(8)],
        precision,
    }
}

#[test]
fn hip_bitwise_kernels_preserve_exact_integer_width_and_shift_traps() {
    for kind in [
        chelis_types::BitwiseKind::And,
        chelis_types::BitwiseKind::Or,
        chelis_types::BitwiseKind::Xor,
        chelis_types::BitwiseKind::ShiftLeft,
        chelis_types::BitwiseKind::ShiftRight,
    ] {
        for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let lhs = dag.add_node(
                decl,
                RiscOp::Load { name: "lhs".into() },
                vec![],
                vector(prim),
                None,
            );
            let rhs = dag.add_node(
                decl,
                RiscOp::Load { name: "rhs".into() },
                vec![],
                vector(prim),
                None,
            );
            let out = dag.add_node(
                decl,
                RiscOp::Bitwise(kind),
                vec![lhs, rhs],
                vector(prim),
                None,
            );
            dag.add_root(out);
            let source = codegen_hip(&dag, "bitwise")
                .expect("[05-OP-47] typed HIP kernel")
                .c_source;
            assert!(source.contains(&format!("kernel_{}_{}", kind.name(), prim.name())));
            assert!(
                source.contains("unsigned"),
                "bitwise kernels must use unsigned bits"
            );
            if kind.is_shift() {
                assert!(source.contains("chelis_record_numeric_failure"));
                assert!(source.contains("shift amount must be non-negative, got"));
                assert!(source.contains("chelis_numeric_failure_index"));
            }
        }
    }
}

fn binary_source(op: RiscOp, input: Prim, output: Prim, name: &str) -> String {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let input_ty = vector(input);
    let lhs = dag.add_node(
        decl,
        RiscOp::Load { name: "lhs".into() },
        vec![],
        input_ty.clone(),
        None,
    );
    let rhs = dag.add_node(
        decl,
        RiscOp::Load { name: "rhs".into() },
        vec![],
        input_ty,
        None,
    );
    let out = dag.add_node(decl, op, vec![lhs, rhs], vector(output), None);
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
    let decl = dag.declare("test");
    let ty = vector(Prim::Bool);
    let input = dag.add_node(
        decl,
        RiscOp::Load {
            name: "input".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![input],
        ty,
        None,
    );
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
        let decl = dag.declare("test");
        let cond = dag.add_node(
            decl,
            RiscOp::Load {
                name: "cond".into(),
            },
            vec![],
            vector(Prim::Bool),
            None,
        );
        let branch_ty = vector(precision);
        let lhs = dag.add_node(
            decl,
            RiscOp::Load { name: "lhs".into() },
            vec![],
            branch_ty.clone(),
            None,
        );
        let rhs = dag.add_node(
            decl,
            RiscOp::Load { name: "rhs".into() },
            vec![],
            branch_ty.clone(),
            None,
        );
        let out = dag.add_node(decl, RiscOp::Where, vec![cond, lhs, rhs], branch_ty, None);
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

#[test]
fn permuted_stepped_views_feed_comparison_logical_and_where_stride_metadata() {
    let matrix = |rows, cols, precision| TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision,
    };
    let input_f32 = matrix(2, 4, Prim::F32);
    let input_bool = matrix(2, 4, Prim::Bool);
    let permuted_f32 = matrix(4, 2, Prim::F32);
    let permuted_bool = matrix(4, 2, Prim::Bool);
    let stepped_f32 = matrix(2, 2, Prim::F32);
    let stepped_bool = matrix(2, 2, Prim::Bool);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let view = |dag: &mut Dag,
                name: &str,
                input: &TensorType,
                permuted: &TensorType,
                stepped: &TensorType| {
        let load = dag.add_node(
            decl,
            RiscOp::Load { name: name.into() },
            vec![],
            input.clone(),
            None,
        );
        let permute = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![load],
            permuted.clone(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2), RtDim::Lit(1)],
            },
            vec![permute],
            stepped.clone(),
            None,
        )
    };
    let lhs = view(&mut dag, "lhs", &input_f32, &permuted_f32, &stepped_f32);
    let rhs = view(&mut dag, "rhs", &input_f32, &permuted_f32, &stepped_f32);
    let logical_lhs = view(
        &mut dag,
        "logical_lhs",
        &input_bool,
        &permuted_bool,
        &stepped_bool,
    );
    let logical_rhs = view(
        &mut dag,
        "logical_rhs",
        &input_bool,
        &permuted_bool,
        &stepped_bool,
    );
    let condition = view(
        &mut dag,
        "condition",
        &input_bool,
        &permuted_bool,
        &stepped_bool,
    );
    let comparison = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::Gte),
        vec![lhs, rhs],
        stepped_bool.clone(),
        None,
    );
    let logical = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::And),
        vec![logical_lhs, logical_rhs],
        stepped_bool,
        None,
    );
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![condition, lhs, rhs],
        stepped_f32,
        None,
    );
    dag.add_root(comparison);
    dag.add_root(logical);
    dag.add_root(selected);

    let source = codegen_hip(&dag, "permuted_stepped_nonnumeric")
        .expect("permuted/stepped direct nonnumeric HIP codegen")
        .c_source;
    assert!(
        source.contains("chelis_int_checked_mul(d_t")
            && source.contains("INT64_C(2)")
            && source.contains("->strides[1]"),
        "the transpose and stride must remain shared-view metadata:\n{source}"
    );
    assert!(source.contains("kernel_compare_gte"), "{source}");
    assert!(source.contains("kernel_logical_and"), "{source}");
    assert!(source.contains("kernel_where"), "{source}");
    assert!(
        source.matches("chelis_indices_to_flat(indices").count() >= 6,
        "each direct operation must index its non-contiguous operands through supplied strides:\n{source}"
    );
}
