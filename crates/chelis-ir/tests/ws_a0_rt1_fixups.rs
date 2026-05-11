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
// F1: BlasMatmul currently supports only f32
// ----------------------------------------------------------------

/// `RiscOp::BlasMatmul` on `tensor[m, k, f64] * tensor[k, n, f64]`
/// must error at IR validation with the F1 diagnostic. The C/HIP
/// backends destructure `BlasMatmul` with `..` and dispatch
/// single-precision GEMM regardless of operand precision; allowing an
/// f64 matmul into the IR would silently lower to `cblas_sgemm` reading
/// `double*` storage. Reject at validation until WS-A1.
#[test]
fn blas_matmul_f64_rejected_with_f1_diagnostic() {
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
        errors
            .iter()
            .any(|m| m.contains("F1: BlasMatmul currently supports only f32")),
        "expected F1 diagnostic; got: {errors:?}"
    );
    assert!(
        errors
            .iter()
            .any(|m| m.contains("spec/04-type-system.md §5.7.1")),
        "F1 diagnostic must cite spec/04-type-system.md §5.7.1; got: {errors:?}"
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
