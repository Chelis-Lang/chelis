//! Primitive matmul preserves its dtype and canonical reduction tree in C.
//! Enabling explicit BLAS-node emission does not authorize replacing a
//! primitive contraction with a library-dependent arithmetic graph.

use chelis_backend_c::CodegenOptions;
mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen_with_options;

fn t(prim: Prim, dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: prim,
    }
}

/// Regression lock: end-to-end F64 matmul must NOT emit cblas_sgemm.
#[test]
fn f64_matmul_subgraph_stays_off_blas_path_in_c_backend() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F64, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F64, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(8),
        },
        vec![b],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F64,
        },
        vec![mul],
        t(Prim::F64, vec![8, 4]),
        None,
    );
    dag.add_root(sum);

    // Exercise the option used by C entry points, including its negative
    // guarantee: primitive arithmetic does not acquire a vendor summary.
    let result = codegen_with_options(
        &dag,
        "p0_f64_matmul_miscompile",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    )
    .unwrap();

    assert!(!result.c_source.contains("cblas_sgemm("));
    assert!(!result.c_source.contains("cblas_dgemm("));
    assert!(result.c_source.contains("__sum_level_"));

    // F64 slot allocation should still be present (the data is F64;
    // the generic expand+mul+sum path computes against it).
    assert!(
        result.c_source.contains("CHELIS_DTYPE_F64"),
        "F64 slot allocation must be present on the generic path; got source:\n{}",
        result.c_source
    );

    // With BLAS specialization skipped, the BLAS link requirement must
    // NOT be set — the generic path doesn't need -lopenblas.
    assert!(
        !result.requirements.needs_blas,
        "F64 matmul on the generic path must not surface needs_blas; \
         got requirements = {:?}",
        result.requirements
    );
}

/// F32 has the same exact-arithmetic obligation as F64.
#[test]
fn f32_matmul_subgraph_preserves_canonical_arithmetic_in_c_backend() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F32, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(8),
        },
        vec![b],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        t(Prim::F32, vec![8, 4]),
        None,
    );
    dag.add_root(sum);

    let result = codegen_with_options(
        &dag,
        "f32_matmul_canonical",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    )
    .unwrap();

    assert!(!result.c_source.contains("cblas_sgemm("));
    assert!(!result.requirements.needs_blas);
    assert!(result.c_source.contains("CHELIS_DTYPE_F32"));
    assert!(result.c_source.contains("__sum_level_"));
}
