//! RT-2 adversarial coverage for Wave 2 IR layer.
//!
//! Attacks:
//!   - End-to-end accumulator plumbing: a hand-built IR with a
//!     non-default accumulator (e.g., i8 reduce_sum + i64
//!     accumulator) must verify and the verify must respect the
//!     non-default field, not silently use the operand precision.
//!   - Result-precision invariant per spec §5.7.1: for reduce_sum,
//!     output_type.precision MUST equal the accumulator field. The
//!     verifier enforces this; pin a hand-built IR that violates it.
//!   - Unusual-but-spec-admitted accumulator (f32 operand + f64
//!     accumulator on reduce_sum) must verify and the output precision
//!     must follow the accumulator.
//!   - F1 BlasMatmul guard residual state: ONLY integer matmul
//!     (§5.7.2) and f8e4m3 matmul (§1.1.1) should remain rejected;
//!     all four float dtypes must be admitted at the verify layer.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::verify;
use chelis_types::types::Prim;

fn tensor(prec: Prim, dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: prec,
    }
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.7.1 result-precision = accumulator invariant.
// The IR verifier enforces this; pin the contract with a hand-built
// IR that violates it.
// ----------------------------------------------------------------

#[test]
fn ir_verify_rejects_sum_with_output_precision_not_matching_accumulator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    // Build a Load of i8 tensor.
    let inp = dag.add_node(
        decl,
        RiscOp::Load {
            name: "xs".to_string().into(),
        },
        vec![],
        tensor(Prim::Int8, vec![3]),
        None,
    );
    // Construct a Sum that VIOLATES the §5.7.1 invariant: output
    // precision is i8 but accumulator field is i32. The verifier
    // must catch this.
    let bad_sum_id = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![inp],
        tensor(Prim::Int8, vec![]),
        None,
    );
    dag.add_root(bad_sum_id);
    let errs = verify::verify(&dag);
    assert!(
        !errs.is_empty(),
        "spec §5.7.1: Sum.output_type.precision must equal accumulator; \
         violator must be rejected by verify"
    );
    assert!(
        errs.iter().any(|e| e.contains("§5.7.1")),
        "verify error must cite §5.7.1; got: {errs:?}"
    );
}

#[test]
fn ir_verify_accepts_sum_int8_with_int32_accumulator_int32_output() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let inp = dag.add_node(
        decl,
        RiscOp::Load {
            name: "xs".to_string().into(),
        },
        vec![],
        tensor(Prim::Int8, vec![3]),
        None,
    );
    let sum_id = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![inp],
        tensor(Prim::Int32, vec![]),
        None,
    );
    dag.add_root(sum_id);
    let errs = verify::verify(&dag);
    assert!(
        errs.is_empty(),
        "i8 sum with i32 accumulator + i32 output must verify; \
         got: {errs:?}"
    );
}

#[test]
fn ir_verify_accepts_sum_f32_with_explicit_f64_wider_accumulator() {
    // Spec §5.7.1: explicit accumulator wider than the default is admitted.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let inp = dag.add_node(
        decl,
        RiscOp::Load {
            name: "xs".to_string().into(),
        },
        vec![],
        tensor(Prim::F32, vec![3]),
        None,
    );
    let sum_id = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![inp],
        tensor(Prim::F64, vec![]),
        None,
    );
    dag.add_root(sum_id);
    let errs = verify::verify(&dag);
    assert!(
        errs.is_empty(),
        "f32 sum with explicit f64 accumulator (wider than default f32) \
         must verify per spec §5.7.1; got: {errs:?}"
    );
}

// ----------------------------------------------------------------
// CRITICAL: F1 BlasMatmul guard — verify residual rejection set.
// ----------------------------------------------------------------

#[test]
fn ir_verify_rejects_blas_matmul_int8_per_spec_5_7_2() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "a".to_string().into(),
        },
        vec![],
        tensor(Prim::Int8, vec![2, 3]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load {
            name: "b".to_string().into(),
        },
        vec![],
        tensor(Prim::Int8, vec![3, 2]),
        None,
    );
    // Hand-build an i8 BlasMatmul. Per §5.7.2 this should be rejected.
    let mm = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::Int32,
        },
        vec![a, b],
        tensor(Prim::Int8, vec![2, 2]),
        None,
    );
    dag.add_root(mm);
    let errs = verify::verify(&dag);
    assert!(
        !errs.is_empty(),
        "spec §5.7.2: integer matmul must be rejected at IR verify"
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("F1") || e.contains("integer")),
        "IR verify rejection of i8 matmul must mention F1 or integer; got: {errs:?}"
    );
}

#[test]
fn ir_verify_admits_blas_matmul_bf16() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "a".to_string().into(),
        },
        vec![],
        tensor(Prim::Bf16, vec![2, 3]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load {
            name: "b".to_string().into(),
        },
        vec![],
        tensor(Prim::Bf16, vec![3, 2]),
        None,
    );
    let mm = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F32,
        },
        vec![a, b],
        tensor(Prim::Bf16, vec![2, 2]),
        None,
    );
    dag.add_root(mm);
    let errs = verify::verify(&dag);
    assert!(
        errs.is_empty(),
        "spec §5.7.1 + WS-A3: bf16 matmul with f32 accumulator must verify; \
         got: {errs:?}"
    );
}

#[test]
fn ir_verify_admits_blas_matmul_f16() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "a".to_string().into(),
        },
        vec![],
        tensor(Prim::F16, vec![2, 3]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load {
            name: "b".to_string().into(),
        },
        vec![],
        tensor(Prim::F16, vec![3, 2]),
        None,
    );
    let mm = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F32,
        },
        vec![a, b],
        tensor(Prim::F16, vec![2, 2]),
        None,
    );
    dag.add_root(mm);
    let errs = verify::verify(&dag);
    assert!(
        errs.is_empty(),
        "spec §5.7.1 + WS-A3: f16 matmul with f32 accumulator must verify; \
         got: {errs:?}"
    );
}

#[test]
fn ir_verify_admits_blas_matmul_f64_with_dgemm_accumulator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "a".to_string().into(),
        },
        vec![],
        tensor(Prim::F64, vec![2, 3]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load {
            name: "b".to_string().into(),
        },
        vec![],
        tensor(Prim::F64, vec![3, 2]),
        None,
    );
    let mm = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F64,
        },
        vec![a, b],
        tensor(Prim::F64, vec![2, 2]),
        None,
    );
    dag.add_root(mm);
    let errs = verify::verify(&dag);
    assert!(
        errs.is_empty(),
        "spec §5.7.1 + WS-A1/A2: f64 matmul with f64 accumulator must verify; \
         got: {errs:?}"
    );
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.7.1 narrowness rule via the constructor.
// `matmul_with_accumulator` should reject narrower-than-default; pin
// each lane.
// ----------------------------------------------------------------

#[test]
fn matmul_with_accumulator_rejects_f32_when_default_is_f32() {
    // f64 operand, f32 accumulator: f32 is narrower than f64 default.
    let err = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(3),
        DimExpr::Concrete(4),
        Prim::F64,
        Prim::F32,
    )
    .expect_err("f64 matmul with f32 accumulator must be rejected (narrower)");
    assert!(err.contains("§5.7.1"), "must cite spec §5.7.1; got: {err}");
}

// ----------------------------------------------------------------
// SPEC-DIVERGENCE: spec §5.7.2 says integer matmul is rejected; the
// `matmul_with_accumulator` constructor enforces this. But the Sum
// constructor allows any wider integer. Verify both per-row.
// ----------------------------------------------------------------

#[test]
fn sum_with_accumulator_int64_operand_with_int32_acc_rejected() {
    // i64 operand, i32 accumulator: narrower than the i64 default.
    let err = RiscOp::sum_with_accumulator(0, Prim::Int64, Prim::Int32)
        .expect_err("i64 sum with i32 accumulator must be rejected (narrower)");
    assert!(err.contains("§5.7.1"));
}

#[test]
fn sum_with_accumulator_f64_operand_with_f32_acc_rejected() {
    // f64 operand, f32 accumulator: narrower than f64 default.
    let err = RiscOp::sum_with_accumulator(0, Prim::F64, Prim::F32)
        .expect_err("f64 sum with f32 accumulator must be rejected (narrower)");
    assert!(err.contains("§5.7.1"));
}
