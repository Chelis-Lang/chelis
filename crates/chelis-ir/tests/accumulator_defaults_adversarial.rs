//! RT-1 adversarial coverage for the WS-A0 IR accumulator surface.
//!
//! Attacks:
//!  - B. Accumulator-default resolution table (§5.7.1) — every row.
//!  - B. Negative twins for accumulator narrowness rule (§5.7.1) for
//!    EVERY operand precision the spec lists (the existing acceptance
//!    file only covered i8, i16, bf16).
//!  - B. matmul rejection for bf16 with operand-narrower-than-default
//!    accumulator (`accumulator=bf16` when default is f32).
//!  - F. The accumulator field destructure-with-`..` pattern in the
//!    backend BlasMatmul detectors.
//!  - F8e4m3 must be rejected by the IR-level default helpers too.

use chelis_ir::dag::{DimExpr, RiscOp};
use chelis_types::types::Prim;

// ---------------------------------------------------------------
// B. Per-row reduce_sum default coverage
// ---------------------------------------------------------------

/// §5.7.1 result-precision rule: result precision = accumulator precision
/// for reduce_sum (across all rows).
#[test]
fn reduce_sum_int64_default_is_int64_self_matching() {
    let acc = RiscOp::default_reduce_sum_accumulator(Prim::Int64)
        .expect("i64 reduce_sum default must succeed");
    assert_eq!(
        acc,
        Prim::Int64,
        "spec §5.7.1: i64 reduce_sum default must be i64, not i32"
    );
}

/// §5.7.1: bool is rejected for reduce_sum entirely.
#[test]
fn reduce_sum_default_bool_rejected_with_workaround_hint() {
    let err = RiscOp::default_reduce_sum_accumulator(Prim::Bool)
        .expect_err("bool reduce_sum has no defined default");
    assert!(
        err.contains("cast to i32"),
        "diagnostic must suggest the i32 cast workaround; got: {err}"
    );
}

/// §1.1.1: deferred f8e4m3 must be rejected by the IR-level default
/// helpers (not just the type checker).
#[test]
fn reduce_sum_default_f8e4m3_rejected_with_spec_diagnostic() {
    let err = RiscOp::default_reduce_sum_accumulator(Prim::F8e4m3)
        .expect_err("f8e4m3 is deferred per §1.1.1");
    assert!(
        err.contains("§1.1.1"),
        "diagnostic must cite §1.1.1; got: {err}"
    );
}

#[test]
fn matmul_default_f8e4m3_rejected_with_spec_diagnostic() {
    let err = RiscOp::default_matmul_accumulator(Prim::F8e4m3)
        .expect_err("f8e4m3 matmul is deferred per §1.1.1");
    assert!(
        err.contains("§1.1.1"),
        "diagnostic must cite §1.1.1; got: {err}"
    );
}

// ---------------------------------------------------------------
// B. Per-row narrowness rule (negative-parity gap for f16, i32, i64)
// ---------------------------------------------------------------

/// §5.7.1: f16 with explicit f16 accumulator (default is f32) must error.
/// NEGATIVE PARITY: existing accumulator_defaults.rs only covers bf16
/// and i8/i16. f16 is a separate row in the §5.7.1 table.
#[test]
fn reduce_sum_f16_with_explicit_f16_accumulator_is_rejected() {
    let err = RiscOp::sum_with_accumulator(0, Prim::F16, Prim::F16)
        .expect_err("f16 operand + f16 accumulator must be rejected per §5.7.1");
    assert!(err.contains("§5.7.1"), "must cite §5.7.1; got: {err}");
    assert!(err.contains("f32"), "must mention default f32; got: {err}");
}

/// §5.7.1: matmul on bf16 with explicit accumulator=bf16 must be rejected
/// (default is f32; bf16 is narrower-than-default).
/// NEGATIVE PARITY: existing accumulator_defaults.rs has no matmul-narrow
/// test at all.
#[test]
fn matmul_bf16_with_explicit_bf16_accumulator_is_rejected() {
    let err = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(3),
        DimExpr::Concrete(4),
        Prim::Bf16,
        Prim::Bf16,
    )
    .expect_err("bf16 matmul + bf16 accumulator must be rejected per §5.7.1");
    assert!(err.contains("§5.7.1"));
    assert!(
        err.contains("f32"),
        "must mention spec default f32 for bf16 matmul; got: {err}"
    );
}

/// §5.7.1: matmul on f16 with explicit accumulator=f16 must be rejected.
#[test]
fn matmul_f16_with_explicit_f16_accumulator_is_rejected() {
    let err = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(3),
        DimExpr::Concrete(4),
        Prim::F16,
        Prim::F16,
    )
    .expect_err("f16 matmul + f16 accumulator must be rejected per §5.7.1");
    assert!(err.contains("§5.7.1"));
}

/// §5.7.1: cross-lane accumulator (f32 operand + i64 accumulator) must
/// be rejected per `accumulator_at_least_as_wide`'s lane check.
/// Existing tests don't pin lane crossing.
#[test]
fn reduce_sum_f32_operand_with_int64_accumulator_rejected_cross_lane() {
    let err = RiscOp::sum_with_accumulator(0, Prim::F32, Prim::Int64)
        .expect_err("f32 operand + i64 accumulator must be rejected (cross-lane)");
    assert!(err.contains("§5.7.1"));
}

/// §5.7.1: cross-lane the other way (i32 operand + f64 accumulator)
/// must also be rejected.
#[test]
fn reduce_sum_int32_operand_with_f64_accumulator_rejected_cross_lane() {
    let err = RiscOp::sum_with_accumulator(0, Prim::Int32, Prim::F64)
        .expect_err("i32 operand + f64 accumulator must be rejected (cross-lane)");
    assert!(err.contains("§5.7.1"));
}

/// §5.7.1 wider-than-default acceptance: reduce_sum on f32 with f64
/// accumulator must be permitted (f64 is wider than the default f32).
/// NEGATIVE PARITY: existing test only covers bf16+f64; pin the f32 row.
#[test]
fn reduce_sum_f32_operand_with_f64_accumulator_accepted() {
    let op = RiscOp::sum_with_accumulator(0, Prim::F32, Prim::F64)
        .expect("f32 operand + f64 accumulator must be accepted (wider than default)");
    match op {
        RiscOp::Sum { accumulator, .. } => assert_eq!(accumulator, Prim::F64),
        other => panic!("expected RiscOp::Sum, got {other:?}"),
    }
}

/// §5.7.1: reduce_sum on i8 with i64 accumulator (wider than i32
/// default) must be accepted.
#[test]
fn reduce_sum_int8_operand_with_int64_accumulator_accepted() {
    let op = RiscOp::sum_with_accumulator(0, Prim::Int8, Prim::Int64)
        .expect("i8 + i64 accumulator must be accepted (wider than i32 default)");
    match op {
        RiscOp::Sum { accumulator, .. } => assert_eq!(accumulator, Prim::Int64),
        other => panic!("expected RiscOp::Sum, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// F. Backend MatmulInfo accumulator-loss
// ---------------------------------------------------------------

// (Backend-side MatmulInfo accumulator-loss is exercised in
// crates/chelis-backend-c/tests/rt1_adversarial.rs to avoid pulling
// chelis-backend-c into chelis-ir's dev-dep graph.)
