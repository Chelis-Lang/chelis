//! WS-A0 RT-1 fixup acceptance tests.
//!
//! Pins the per-fix done-condition for the F1 `BlasMatmul` tactical
//! precision guard. The other fixups (A1/C1/D1/E1/E2) have their
//! acceptance tests living in the crates that own the fixed surface.
//!
//! These tests document each finding's rejection contract so a future
//! lift (notably WS-A1 lifting F1 once `cblas_dgemm` / `hipblasDgemm`
//! dispatch lands) breaks the test loudly rather than silently
//! reverting the guard.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::verify;
use chelis_types::types::Prim;

fn matrix(rows: usize, cols: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: prec,
    }
}

// ----------------------------------------------------------------
// F1: BlasMatmul supports f32 / f64 / bf16 / f16 (post WS-A1+A2+A3)
// ----------------------------------------------------------------
//
// WS-A1 lifted the f64 arm of the F1 tactical guard once the C
// backend wired `cblas_dgemm` dispatch through
// `MatmulEmitSpec::accumulator`. WS-A2 extended the HIP backend to
// dispatch `hipblasSgemm`/`hipblasDgemm` for f32/f64. WS-A3 wired
// bf16/f16 through `hipblasGemmEx` (HIPBLAS_COMPUTE_32F per
// spec/04-type-system.md §5.7.1). The F1 guard now only blocks the
// integer matmul arm (lifted in WS-A4 per spec §5.7.2).

/// Negative-parity twin of the original f64-rejection test. After
/// WS-A1 (C backend) and WS-A2 (HIP backend), `RiscOp::BlasMatmul`
/// on f64 operands MUST validate cleanly because the C backend
/// dispatches `cblas_dgemm` and the HIP backend dispatches
/// `hipblasDgemm` (both reading/writing `double*` storage).
/// Replaces the prior `blas_matmul_f64_rejected_with_f1_diagnostic`
/// assertion per the WS-A1 brief contract that the test be
/// REPLACED, not silently deleted.
#[test]
fn blas_matmul_f64_validates_cleanly_after_ws_a1_lift() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(2, 3, Prim::F64),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(3, 4, Prim::F64),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F64,
    )
    .expect("f64 matmul constructs (per spec §5.7.1 default = f64)");
    let _ = dag.add_node(matmul_op, vec![a, b], matrix(2, 4, Prim::F64), None);

    let errors = verify::verify(&dag);
    assert!(
        !errors
            .iter()
            .any(|m| m.contains("F1: BlasMatmul on operand precision `f64`")),
        "f64 BlasMatmul must not trip the (now-lifted) F1 guard; got: {errors:?}"
    );
    assert!(
        !errors.iter().any(|m| m.contains("F1:")),
        "f64 BlasMatmul must not trip any remaining F1 guard arm post WS-A1; got: {errors:?}"
    );
    assert!(
        errors.is_empty(),
        "f64 BlasMatmul with default (f64) accumulator must validate cleanly; got: {errors:?}"
    );
}

/// WS-A3 admits `bf16` `BlasMatmul`: the HIP backend now binds the
/// `accumulator` field explicitly (no more destructure-`..` footgun)
/// and routes bf16 + f32-default-accumulator through `hipblasGemmEx`
/// per spec/04-type-system.md §5.7.1. The C backend still rejects
/// bf16 at its own F1 guard; this IR-level test pins that the
/// validation guard no longer catches the bf16 case.
#[test]
fn blas_matmul_bf16_admitted_after_ws_a3_lift() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(3, 4, Prim::Bf16),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Bf16,
    )
    .expect("bf16 matmul constructs (per spec §5.7.1 default accumulator = f32)");
    let _ = dag.add_node(matmul_op, vec![a, b], matrix(2, 4, Prim::Bf16), None);

    let errors = verify::verify(&dag);
    assert!(
        !errors.iter().any(|m| m.contains("F1: BlasMatmul")),
        "WS-A3 lifted bf16 from the F1 guard; bf16 BlasMatmul must validate cleanly. \
         Got: {errors:?}"
    );
}

/// WS-A3 admits `f16` `BlasMatmul` on the same lift as bf16 (also
/// dispatches through `hipblasGemmEx` with `HIPBLAS_COMPUTE_32F` per
/// spec §5.7.1).
#[test]
fn blas_matmul_f16_admitted_after_ws_a3_lift() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(2, 3, Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(3, 4, Prim::F16),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F16,
    )
    .expect("f16 matmul constructs (per spec §5.7.1 default accumulator = f32)");
    let _ = dag.add_node(matmul_op, vec![a, b], matrix(2, 4, Prim::F16), None);

    let errors = verify::verify(&dag);
    assert!(
        !errors.iter().any(|m| m.contains("F1: BlasMatmul")),
        "WS-A3 lifted f16 from the F1 guard; f16 BlasMatmul must validate cleanly. \
         Got: {errors:?}"
    );
}

/// Negative-parity twin: f32 `BlasMatmul` still validates cleanly. The
/// F1 guard is operand-precision specific and must not regress f32.
#[test]
fn blas_matmul_f32_validates_cleanly_under_f1_guard() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(2, 3, Prim::F32),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(3, 4, Prim::F32),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F32,
    )
    .expect("f32 matmul constructs");
    let _ = dag.add_node(matmul_op, vec![a, b], matrix(2, 4, Prim::F32), None);

    let errors = verify::verify(&dag);
    assert!(
        !errors.iter().any(|m| m.contains("F1: BlasMatmul")),
        "f32 BlasMatmul must not trip the F1 guard; got: {errors:?}"
    );
}
