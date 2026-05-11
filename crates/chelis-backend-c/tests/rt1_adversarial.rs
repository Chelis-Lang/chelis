//! RT-1 adversarial coverage for the C backend's WS-A0 dtype boundary.
//!
//! Attacks:
//!  - F. The IR `RiscOp::BlasMatmul { accumulator, .. }` destructure in
//!    `emit.rs` uses `..` and the C `MatmulInfo` struct has no
//!    `accumulator` field. So matmul over f64 operands (with the
//!    spec-default f64 accumulator per §5.7.1) silently lowers to
//!    `cblas_sgemm` (single-precision GEMM). For f64 operands held in
//!    `double*` arrays, `cblas_sgemm` is a wrong-type call; the
//!    generated C compiles only because the runtime lays out tensors as
//!    `void*`/`float*` and the BLAS prototype takes `float*`. The
//!    semantics would be silently wrong even if it linked.
//!  - E. The default backend arm `_ => "float"` was removed; verify
//!    the new behavior (panic with a useful message) is in fact a
//!    panic, not a downgrade.

use chelis_backend_c::emit::CEmitter;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn mat(r: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: p,
    }
}

// ---------------------------------------------------------------
// E. Removed silent-default-arm verification
// ---------------------------------------------------------------

/// E. Element type for a tensor with f16 precision must NOT be silently
/// emitted as `float`. The pre-WS-A0 fallback `_ => "float"` was removed;
/// the new behavior is a panic. Verify by trying to emit a DAG containing
/// an f16 tensor and asserting the panic.
#[test]
#[should_panic(expected = "f16")]
fn c_backend_panics_on_f16_tensor_no_silent_float_downgrade() {
    let mut dag = Dag::new();
    let _ = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F16,
        },
        None,
    );
    let _ = CEmitter::emit_dag(&dag, "test_fn");
}

#[test]
#[should_panic(expected = "bf16")]
fn c_backend_panics_on_bf16_tensor_no_silent_float_downgrade() {
    let mut dag = Dag::new();
    let _ = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Bf16,
        },
        None,
    );
    let _ = CEmitter::emit_dag(&dag, "test_fn");
}

#[test]
#[should_panic(expected = "int8")]
fn c_backend_panics_on_int8_tensor_no_silent_int32_downgrade() {
    let mut dag = Dag::new();
    let _ = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    let _ = CEmitter::emit_dag(&dag, "test_fn");
}

// ---------------------------------------------------------------
// F. BlasMatmul accumulator silent loss
// ---------------------------------------------------------------

/// F. The structural finding: `MatmulInfo` has no accumulator field,
/// and the destructure in `emit_dag` discards the accumulator from
/// `RiscOp::BlasMatmul { .. }`.
///
/// This test pins the type-level fact: try to construct MatmulInfo with
/// an accumulator field — it will not compile. If a future fix adds the
/// field, this test will fail to compile and must be updated.
#[test]
fn matmul_info_struct_has_no_accumulator_field() {
    use chelis_backend_c::blas::MatmulInfo;
    // Public struct literal — if `accumulator` is later added as a
    // required field, this constructor errors at compile time and
    // surfaces the structural change directly.
    let info = MatmulInfo {
        a: chelis_ir::dag::NodeId(0),
        b: chelis_ir::dag::NodeId(1),
        m: 2,
        n: 4,
        k: 3,
    };
    let _ = info;
}

/// F. The runtime/codegen finding: build a DAG containing
/// `RiscOp::BlasMatmul` over f64 operands with the spec-default f64
/// accumulator per §5.7.1, ask the C backend to emit it, and check the
/// generated source for the BLAS call. The C backend emits
/// `cblas_sgemm` (f32-only GEMM) regardless of operand precision; for
/// f64 operands this is a silent downgrade.
///
/// EXPECTED: either (a) the C backend panics ("does not yet support
/// f64 BLAS matmul") or (b) it emits `cblas_dgemm`. Anything else is
/// silent data loss.
/// ACTUAL: see test output.
#[test]
fn c_backend_blas_matmul_f64_does_not_silently_lower_to_sgemm() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        mat(2, 3, Prim::F64),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        mat(3, 4, Prim::F64),
        None,
    );
    // Construct directly with the spec-default accumulator.
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F64,
    )
    .expect("f64 matmul default constructs");
    // Verify the IR carries an f64 accumulator (per §5.7.1).
    if let RiscOp::BlasMatmul { accumulator, .. } = &matmul_op {
        assert_eq!(
            *accumulator,
            Prim::F64,
            "spec §5.7.1: f64 → f64 accumulator"
        );
    } else {
        panic!("expected BlasMatmul");
    }
    let _matmul = dag.add_node(matmul_op, vec![a, b], mat(2, 4, Prim::F64), None);

    // Catch the panic if the backend rejects; otherwise inspect the source.
    let emit_result = std::panic::catch_unwind(|| CEmitter::emit_dag(&dag, "test_fn"));
    match emit_result {
        Err(_panic) => {
            // Backend rejected — acceptable behavior. Pass.
        }
        Ok(src) => {
            // Backend emitted something. Check whether it's silent
            // sgemm-on-f64-data. The substring `cblas_sgemm` indicates
            // single-precision GEMM. Source data is `double*`; calling
            // sgemm on double data is silent data corruption.
            let src: String = src;
            let mentions_sgemm = src.contains("cblas_sgemm");
            let mentions_dgemm = src.contains("cblas_dgemm");
            assert!(
                !mentions_sgemm || mentions_dgemm,
                "C backend silently emitted cblas_sgemm (single precision) for an \
                 f64 matmul DAG with the spec-default f64 accumulator per §5.7.1. \
                 Generated source:\n{src}"
            );
        }
    }
}

/// F. Same shape as the f64 case, but for bf16/f32 — the bf16 row of
/// §5.7.1 has accumulator=f32, and matmul result precision = operand
/// precision = bf16. The C backend `validate_supported_precisions`
/// will panic on bf16 OUTPUT, which is the right outcome. Pin it so a
/// future regression that allows bf16 output but still emits sgemm
/// without a bf16-aware path is caught.
#[test]
#[should_panic(expected = "bf16")]
fn c_backend_blas_matmul_bf16_panics_until_bf16_output_supported() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        mat(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        mat(3, 4, Prim::Bf16),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Bf16,
    )
    .expect("bf16 matmul default constructs (accumulator=f32 per §5.7.1)");
    let _matmul = dag.add_node(matmul_op, vec![a, b], mat(2, 4, Prim::Bf16), None);
    let _ = CEmitter::emit_dag(&dag, "test_fn");
}
