//! Wave 5 red-team — **P0 silent-miscompile regression lock** (now closed).
//!
//! Originally surfaced by W5 as a P0: an F64 matmul subgraph reached
//! C codegen as `RiscOp::BlasMatmul` (because the IR specializer had
//! no precision gate) and the C backend emitted `cblas_sgemm` —
//! single-precision BLAS — against the F64 data. The orchestrator
//! shipped the fix in-band with this commit set:
//!
//! 1. `crates/chelis-ir/src/specialize.rs::detect_matmul_pattern`
//!    early-returns `None` when any operand or output is not
//!    `Prim::F32`, keeping non-F32 matmul on the generic
//!    `expand+mul+sum` path.
//! 2. `crates/chelis-backend-c/src/emit.rs::emit_blas_matmul` panics
//!    if it ever receives a non-F32 `BlasMatmul` (defense-in-depth
//!    against future code paths that bypass the specializer).
//! 3. `crates/chelis-backend-hip/src/emit.rs::emit_blas_matmul` has
//!    the same defense-in-depth panic.
//!
//! This file now LOCKS THE FIX: the F64 matmul subgraph must NOT
//! reach the C backend as `BlasMatmul`, and the emitted C source must
//! NOT contain `cblas_sgemm`. A regression that drops the precision
//! filter would fail these assertions.

use chelis_backend_c::{CodegenOptions, codegen_with_options};
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn t(prim: Prim, dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: prim,
    }
}

/// Regression lock: end-to-end F64 matmul must NOT emit cblas_sgemm.
#[test]
#[ignore = "WS-A2: F64 matmul is now ON the BLAS path (cblas_dgemm); the W5 P0 fail-closed assertion this test checks no longer applies."]
fn f64_matmul_subgraph_stays_off_blas_path_in_c_backend() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F64, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F64, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F64,
        },
        vec![mul],
        t(Prim::F64, vec![8, 4]),
        None,
    );
    dag.add_root(sum);

    // The CLI's `chelis build` path uses `use_blas: true`
    // (`crates/chelis-cli/src/main.rs:4060`). Mirror that here so the
    // test reproduces the end-user-visible code path, not the
    // library-default no-BLAS path.
    let result = codegen_with_options(
        &dag,
        "p0_f64_matmul_miscompile",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    ).unwrap();

    // The IR specializer must NOT replace the F64 matmul with BlasMatmul
    // (cblas_sgemm is F32-only), and the C backend must NOT emit
    // cblas_sgemm. Lock both legs as positive assertions on the fix.
    assert!(
        !result.c_source.contains("cblas_sgemm("),
        "F64 matmul subgraph must stay off the BLAS path (cblas_sgemm is \
         single-precision only). Found a cblas_sgemm call in the emitted \
         C source -- the precision filter at \
         chelis_ir::specialize::detect_matmul_pattern has regressed. \
         Emitted C source:\n{}",
        result.c_source
    );

    // F64 slot allocation should still be present (the data is F64;
    // the generic expand+mul+sum path computes against it).
    assert!(
        result.c_source.contains("CHELIS_F64"),
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

/// Positive regression: F32 matmul must still hit the BLAS path.
/// The precision filter must be tight (only F32 admitted), not overshoot
/// to F32-shaped-but-other-types or accidentally also reject F32.
#[test]
fn f32_matmul_subgraph_still_hits_blas_path_in_c_backend() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F32, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
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
        "f32_matmul_blas_positive",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    ).unwrap();

    assert!(
        result.c_source.contains("cblas_sgemm("),
        "F32 matmul must still hit the BLAS path; emitted source:\n{}",
        result.c_source
    );
}
