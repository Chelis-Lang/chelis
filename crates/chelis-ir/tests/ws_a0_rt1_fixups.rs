//! WS-A0 RT-1 fixup acceptance tests.
//!
//! Pins the per-fix done-condition for the F1 BlasMatmul tactical
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
// F1: BlasMatmul currently supports only f32 and f64 (post WS-A1)
// ----------------------------------------------------------------
//
// WS-A1 (commit `feat(backend-c,ir): WS-A1 lift F1 guard for f64 matmul`)
// lifted the f64 arm of the F1 tactical guard once the C backend wired
// `cblas_dgemm` dispatch through `MatmulEmitSpec::accumulator`. The
// HIP/Metal backends still destructure `BlasMatmul` with `..` and call
// single-precision GEMM, and bf16/f16 has no native dispatch on any
// backend yet, so the F1 guard remains in place for those precisions.
// WS-A2 lifts HIP f64; WS-A3 lifts bf16/f16.

/// Negative-parity twin of the original f64-rejection test. After
/// WS-A1, `RiscOp::BlasMatmul` on f64 operands MUST validate cleanly
/// because the C backend dispatches `cblas_dgemm` (and reads/writes
/// `double*` storage). Replaces the prior
/// `blas_matmul_f64_rejected_with_f1_diagnostic` assertion per the
/// WS-A1 brief contract that the test be REPLACED, not silently
/// deleted.
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
            .any(|m| m.contains("F1: BlasMatmul currently supports only f32")),
        "f64 BlasMatmul must not trip the (now-lifted) F1 f32-only guard; got: {errors:?}"
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

/// Same shape for `bf16`. The spec §5.7.1 row for bf16 has
/// accumulator=f32, but the BlasMatmul backend dispatch still requires
/// f32 operand storage; bf16 source data would be silently misread by
/// the f32-only GEMM path. Reject at validation.
#[test]
fn blas_matmul_bf16_rejected_with_f1_diagnostic() {
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
        errors
            .iter()
            .any(|m| m.contains("F1: BlasMatmul currently supports only f32")),
        "expected F1 diagnostic for bf16 operand; got: {errors:?}"
    );
}

/// Negative-parity twin: f32 BlasMatmul still validates cleanly. The
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
        !errors
            .iter()
            .any(|m| m.contains("F1: BlasMatmul currently supports only f32")),
        "f32 BlasMatmul must not trip the F1 guard; got: {errors:?}"
    );
}
