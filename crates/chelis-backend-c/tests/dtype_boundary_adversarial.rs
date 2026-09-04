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

mod support;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::emit_dag;

fn mat(r: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: p,
    }
}

// ---------------------------------------------------------------
// E. Removed silent-default-arm verification
// ---------------------------------------------------------------

/// E (post-WS-1). f16 tensors are now admitted by the C backend per
/// `spec/04-type-system.md` §1.1.3: storage is `uint16_t`, arithmetic
/// converts to f32 via runtime helpers, matmul routes through
/// convert-then-`cblas_sgemm`. The pre-WS-1 behavior was a panic; the
/// post-WS-1 behavior is admission. This test pins the new behavior
/// by asserting that the emitted C contains the `uint16_t` storage
/// type plus the `CHELIS_DTYPE_F16` runtime dtype tag, so a future
/// regression that re-routes f16 storage through `float*` (the silent
/// 4-byte-per-element downgrade the WS-A0 footgun targeted) trips
/// here immediately.
#[test]
fn c_backend_admits_f16_tensor_with_uint16_storage_post_ws_1() {
    let mut dag = Dag::new();
    let _ = dag.add_node(
        RiscOp::synth_const(
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F16,
            }
            .precision,
            1.0,
        ),
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F16,
        },
        None,
    );
    let src = emit_dag(&dag, "test_fn").unwrap();
    assert!(
        src.contains("CHELIS_DTYPE_F16"),
        "WS-1: C backend must allocate f16 tensors via `CHELIS_DTYPE_F16`; got source:\n{src}"
    );
    assert!(
        src.contains(
            "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F16, UINT16_C(0x3C00)))"
        ),
        "C backend must preserve the exact f16 tag and bits through the single public fill API; got source:\n{src}"
    );
}

/// E sibling: same as above for bf16.
#[test]
fn c_backend_admits_bf16_tensor_with_uint16_storage_post_ws_1() {
    let mut dag = Dag::new();
    let _ = dag.add_node(
        RiscOp::synth_const(
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::Bf16,
            }
            .precision,
            1.0,
        ),
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Bf16,
        },
        None,
    );
    let src = emit_dag(&dag, "test_fn").unwrap();
    assert!(
        src.contains("CHELIS_DTYPE_BF16"),
        "WS-1: C backend must allocate bf16 tensors via `CHELIS_DTYPE_BF16`; got source:\n{src}"
    );
    assert!(
        src.contains(
            "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_BF16, UINT16_C(0x3F80)))"
        ),
        "C backend must preserve the exact bf16 tag and bits through the single public fill API; got source:\n{src}"
    );
}

/// WS-A4 lifts the WS-A0 panic-until-wired guard for i8 tensors:
/// `dtype_macro` and `elem_type` now map `Prim::Int8 → CHELIS_DTYPE_I8 /
/// int8_t`, the runtime allocator sizes the buffer at 1 byte per
/// element, and `chelis_contiguous` mirrors the same per-dtype size.
/// The C source generated for an i8 tensor must NO LONGER panic, and
/// must mention `int8_t` so a future refactor that re-introduces the
/// silent f32 downgrade is caught immediately.
#[test]
fn ws_a4_c_backend_emits_int8_tensor_via_int8_t_no_silent_downgrade() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    // A bare Load+root is not enough — `verify` rejects dangling
    // loads — so wrap it in a Copy that consumes `a`.
    dag.add_node(
        RiscOp::Copy,
        vec![a],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    let src = emit_dag(&dag, "test_fn").unwrap();
    assert!(
        src.contains("int8_t"),
        "WS-A4: C backend must emit i8 tensors via `int8_t` (no silent \
         float downgrade); got source:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I8"),
        "WS-A4: C backend must allocate i8 tensors via `CHELIS_DTYPE_I8`; got source:\n{src}"
    );
}

// ---------------------------------------------------------------
// F. BlasMatmul accumulator silent loss
// ---------------------------------------------------------------

/// F (post-WS-A1). The structural fix: `MatmulInfo` now carries an
/// `accumulator: Prim` field per `spec/04-type-system.md` §5.7.1, and
/// `emit.rs` no longer destructures `RiscOp::BlasMatmul { .. }` without
/// binding `accumulator`. The pre-WS-A1 version of this test pinned
/// the absence of the field as a structural-bug tripwire; WS-A1 added
/// the field and wired it through to BLAS dispatch (sgemm vs dgemm).
///
/// The test now pins the positive shape: the accumulator field IS
/// present, must be set on construction, and is sourced from the
/// originating Sum's `accumulator` per `detect_matmul_pattern`.
#[test]
fn matmul_info_struct_carries_accumulator_field_post_ws_a1() {
    use chelis_backend_c::blas::MatmulInfo;
    use chelis_types::types::Prim;
    // Public struct literal — `accumulator` is now a required field.
    // If a future change drops it, the constructor errors at compile
    // time and surfaces the structural regression directly (the F1
    // footgun would re-emerge if any backend path could read the spec
    // without the accumulator).
    let info = MatmulInfo {
        a: chelis_ir::dag::NodeId(0),
        b: chelis_ir::dag::NodeId(1),
        m: 2,
        n: 4,
        k: 3,
        accumulator: Prim::F64,
    };
    assert_eq!(info.accumulator, Prim::F64);
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
        RiscOp::synth_const(mat(2, 3, Prim::F64).precision, 1.0),
        vec![],
        mat(2, 3, Prim::F64),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(mat(3, 4, Prim::F64).precision, 1.0),
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
    let emit_result = std::panic::catch_unwind(|| emit_dag(&dag, "test_fn"));
    match emit_result {
        Err(_panic) => {
            // Backend rejected via panic — acceptable behavior. Pass.
        }
        // chelis#730 Phase 1: the emitter can now also reject through the
        // Result channel — equally acceptable.
        Ok(Err(_unsupported)) => {}
        Ok(Ok(src)) => {
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

/// F (post-WS-1). bf16 matmul is now admitted by the C backend per
/// `spec/04-type-system.md` §1.1.3 + §5.7.1: operands are bf16,
/// accumulator is f32, output is bf16. The wrapper allocates f32
/// scratch buffers, calls `chelis_bf16_to_f32` element-wise to convert,
/// dispatches `cblas_sgemm` against the f32 buffers, and converts
/// the result back to bf16 via `chelis_f32_to_bf16`. This
/// test pins the new routing so a future regression that emits
/// `cblas_sgemm` directly on the `uint16_t` operand bytes (the
/// silent-data-corruption pattern this whole boundary file targets)
/// trips here immediately.
#[test]
fn c_backend_blas_matmul_bf16_routes_through_convert_then_sgemm_post_ws_1() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(mat(2, 3, Prim::Bf16).precision, 1.0),
        vec![],
        mat(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(mat(3, 4, Prim::Bf16).precision, 1.0),
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
    let src = emit_dag(&dag, "test_fn").unwrap();
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
