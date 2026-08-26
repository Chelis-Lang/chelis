//! RT-2 adversarial coverage for the C backend's WS-A1..A4 dtype matrix.
//!
//! Attacks:
//!   - Spec-vs-code matrix: bf16/f16 must error at C codegen with a
//!     clear diagnostic citing bf16/f16 specifically (not a generic
//!     "unsupported precision" or a deep panic).
//!   - i32/i64 reduce_sum (i32 acc / i64 acc) must lower without a
//!     silent f32 downgrade.
//!   - C-backend BlasMatmul accumulator destructure: hand-build an IR
//!     where the accumulator field disagrees with the operand precision
//!     (i.e., f64 matmul — operand f64, accumulator f64 per spec
//!     default) and verify the dispatch picks `cblas_dgemm`, not
//!     `cblas_sgemm`. The pre-WS-A1 footgun was silently calling
//!     `cblas_sgemm` for non-f32.
//!   - i8 reduce_sum end-to-end: source is i8, accumulator is i32,
//!     output dtype is i32; the emitted C must use `int32_t` for the
//!     accumulator.

use chelis_backend_c::emit::CEmitter;
use chelis_backend_c::{CodegenOptions, codegen_with_options};
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::load_store_name::LoadStoreName;
use chelis_types::types::Prim;

fn vec_t(n: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: p,
    }
}

fn mat_t(r: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: p,
    }
}

fn scalar_t(p: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: p,
    }
}

fn ln(s: &str) -> LoadStoreName {
    s.to_string().into()
}

// ----------------------------------------------------------------
// C-backend admission of bf16 / f16 matmul post-WS-1 (cycle: dtype +
// Metal cleanup). The pre-WS-1 behavior was a panic with a citation
// of the F1 guard; post-WS-1 the matmul wrapper routes through
// convert-then-`cblas_sgemm` with f32 scratch buffers per
// spec/04-type-system.md §5.7.1. These tests pin the new routing so
// a future regression that emits `cblas_sgemm` directly on the
// `uint16_t` operand bytes trips here.
// ----------------------------------------------------------------

#[test]
fn c_backend_blas_matmul_bf16_routes_through_convert_then_sgemm_post_ws_1() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: ln("a") },
        vec![],
        mat_t(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: ln("b") },
        vec![],
        mat_t(3, 2, Prim::Bf16),
        None,
    );
    let mm = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F32,
        },
        vec![a, b],
        mat_t(2, 2, Prim::Bf16),
        None,
    );
    dag.add_root(mm);
    let src = CEmitter::emit_dag(&dag, "test_fn").unwrap();
    assert!(
        src.contains("chelis_bf16_to_f32"),
        "WS-1: bf16 matmul must convert operands to f32 before BLAS dispatch; got:\n{src}"
    );
    assert!(
        src.contains("cblas_sgemm"),
        "WS-1: bf16 matmul must dispatch cblas_sgemm against the f32 scratch buffers; got:\n{src}"
    );
    assert!(
        src.contains("chelis_f32_to_bf16"),
        "WS-1: bf16 matmul must downcast the f32 accumulator buffer back to bf16 storage; got:\n{src}"
    );
}

#[test]
fn c_backend_blas_matmul_f16_routes_through_convert_then_sgemm_post_ws_1() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: ln("a") },
        vec![],
        mat_t(2, 3, Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: ln("b") },
        vec![],
        mat_t(3, 2, Prim::F16),
        None,
    );
    let mm = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F32,
        },
        vec![a, b],
        mat_t(2, 2, Prim::F16),
        None,
    );
    dag.add_root(mm);
    let src = CEmitter::emit_dag(&dag, "test_fn").unwrap();
    assert!(
        src.contains("chelis_f16_to_f32"),
        "WS-1: f16 matmul must convert operands to f32 before BLAS dispatch; got:\n{src}"
    );
    assert!(
        src.contains("cblas_sgemm"),
        "WS-1: f16 matmul must dispatch cblas_sgemm against the f32 scratch buffers; got:\n{src}"
    );
    assert!(
        src.contains("chelis_f32_to_f16"),
        "WS-1: f16 matmul must downcast the f32 accumulator buffer back to f16 storage; got:\n{src}"
    );
}

// ----------------------------------------------------------------
// C-backend f64 BlasMatmul dispatch must use cblas_dgemm, not sgemm.
// (Critical: the pre-WS-A1 footgun was the destructure-`..` silent-
// downgrade to sgemm. Verify the actual emitted C contains dgemm.)
// ----------------------------------------------------------------

#[test]
fn c_backend_f64_matmul_emits_cblas_dgemm_not_sgemm() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: ln("a") },
        vec![],
        mat_t(2, 3, Prim::F64),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: ln("b") },
        vec![],
        mat_t(3, 2, Prim::F64),
        None,
    );
    let mm = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F64,
        },
        vec![a, b],
        mat_t(2, 2, Prim::F64),
        None,
    );
    dag.add_root(mm);
    let opts = CodegenOptions {
        use_blas: true,
        ..CodegenOptions::default()
    };
    let result = codegen_with_options(&dag, "f64_mm", opts).unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("cblas_dgemm"),
        "f64 matmul must emit cblas_dgemm; got source:\n{src}"
    );
    assert!(
        !src.contains("cblas_sgemm"),
        "f64 matmul must NOT emit cblas_sgemm (the F1 footgun); got source:\n{src}"
    );
    // Result tensor data must be reinterpreted as `double*`, not `float*`,
    // for the f64 path.
    assert!(
        src.contains("(double*)"),
        "f64 matmul must cast tensor data to double*; got source:\n{src}"
    );
}

// ----------------------------------------------------------------
// C-backend i8 reduce_sum end-to-end: source i8, accumulator i32,
// output i32. The emitted C MUST use int32_t for accumulator.
// ----------------------------------------------------------------

#[test]
fn c_backend_int8_reduce_sum_uses_int32_accumulator_no_silent_overflow() {
    let mut dag = Dag::new();
    let xs = dag.add_node(
        RiscOp::Load { name: ln("xs") },
        vec![],
        vec_t(3, Prim::Int8),
        None,
    );
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![xs],
        scalar_t(Prim::Int32),
        None,
    );
    dag.add_root(s);
    let result = codegen_with_options(&dag, "int8_sum", CodegenOptions::default()).unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("int32_t"),
        "i8 reduce_sum must use int32_t accumulator per spec §5.7.1; got:\n{src}"
    );
    assert!(
        src.contains("int8_t"),
        "i8 reduce_sum must reference int8_t for the source data; got:\n{src}"
    );
}

#[test]
fn c_backend_int16_reduce_sum_uses_int32_accumulator() {
    let mut dag = Dag::new();
    let xs = dag.add_node(
        RiscOp::Load { name: ln("xs") },
        vec![],
        vec_t(3, Prim::Int16),
        None,
    );
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![xs],
        scalar_t(Prim::Int32),
        None,
    );
    dag.add_root(s);
    let result = codegen_with_options(&dag, "int16_sum", CodegenOptions::default()).unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("int32_t"),
        "i16 reduce_sum must use int32_t accumulator per spec §5.7.1; got:\n{src}"
    );
    assert!(
        src.contains("int16_t"),
        "i16 reduce_sum must reference int16_t for source; got:\n{src}"
    );
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.7.1 unusual-but-admitted accumulator combo —
// f32 operand + f64 accumulator. Must produce a sum that uses double
// for the accumulator, not float.
// ----------------------------------------------------------------

#[test]
fn c_backend_f32_operand_f64_accumulator_reduce_sum_uses_double_acc() {
    let mut dag = Dag::new();
    let xs = dag.add_node(
        RiscOp::Load { name: ln("xs") },
        vec![],
        vec_t(3, Prim::F32),
        None,
    );
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![xs],
        scalar_t(Prim::F64),
        None,
    );
    dag.add_root(s);
    let result = codegen_with_options(&dag, "f32_sum_f64_acc", CodegenOptions::default()).unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("double"),
        "f32 sum with explicit f64 accumulator must use double for the \
         running accumulator per spec §5.7.1; got:\n{src}"
    );
}

// ----------------------------------------------------------------
// SANITY: i32 reduce_sum end-to-end (i32 default accumulator).
// ----------------------------------------------------------------

#[test]
fn c_backend_int32_reduce_sum_emits_int32_no_silent_float_downgrade() {
    let mut dag = Dag::new();
    let xs = dag.add_node(
        RiscOp::Load { name: ln("xs") },
        vec![],
        vec_t(3, Prim::Int32),
        None,
    );
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![xs],
        scalar_t(Prim::Int32),
        None,
    );
    dag.add_root(s);
    let result = codegen_with_options(&dag, "int32_sum", CodegenOptions::default()).unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("int32_t"),
        "i32 reduce_sum must use int32_t; got:\n{src}"
    );
    // The actual int32_sum function body must not have a float cast in
    // the accumulator path; the runtime preamble may contain unrelated
    // float helpers, so scope the check to the function's own body.
    let body_start = src
        .find("void int32_sum(")
        .expect("emitted source must contain int32_sum function");
    let body = &src[body_start..];
    assert!(
        !body.contains("(float)"),
        "i32 reduce_sum body must NOT contain a float cast (silent downgrade); \
         body:\n{body}"
    );
}

#[test]
fn c_backend_int64_reduce_sum_emits_int64() {
    let mut dag = Dag::new();
    let xs = dag.add_node(
        RiscOp::Load { name: ln("xs") },
        vec![],
        vec_t(3, Prim::Int64),
        None,
    );
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int64,
        },
        vec![xs],
        scalar_t(Prim::Int64),
        None,
    );
    dag.add_root(s);
    let result = codegen_with_options(&dag, "int64_sum", CodegenOptions::default()).unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("int64_t"),
        "i64 reduce_sum must use int64_t; got:\n{src}"
    );
}
