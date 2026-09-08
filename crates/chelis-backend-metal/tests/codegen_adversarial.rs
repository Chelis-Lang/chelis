//! Adversarial / red-team test surface for the Metal backend (M7).
//!
//! These test the M0-M6 surface as it actually shipped: rank-1 contiguous
//! elementwise (M2), full-axis rank-1 reductions (M4), tiled matmul (M5).
//! Cases the M-phase explicitly defers (bool-through-where, cast,
//! symbolic dims, broadcasts/strides, partial-axis reductions, oversized
//! reductions) must return a typed unsupported error before artifact
//! emission, not silently miscompile.
//!
//! Mirrors `chelis-backend-hip/tests/codegen_adversarial.rs` in spirit;
//! coverage breadth grows as later phases add real support for each case.
//! When a case here flips from "returns unsupported" to "emits real
//! kernels", that's a signal the corresponding phase has shipped — flip
//! the assertion.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::{codegen_metal, try_codegen_metal};

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn vec_bool(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Bool,
    }
}

fn mat_f32(r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn assert_typed_unsupported(error: &chelis_types::unsupported::Unsupported, label: &str) {
    let rendered = error.to_string();
    assert!(
        rendered.starts_with("unsupported:")
            && error.stage == chelis_types::unsupported::Stage::Codegen("metal"),
        "{label}: expected typed Metal codegen rejection, got:\n{rendered}"
    );
}

fn assert_emits_real_kernel(src: &str, label: &str) {
    assert!(
        src.contains("kernel void") || src.contains("chelis_metal_host_to_device"),
        "{label}: expected a real kernel or concrete transfer emission:\n{src}"
    );
}

// ===========================================================================
// Cases the M-phase handles
// ===========================================================================

#[test]
fn m7_zero_element_tensor_does_not_panic() {
    // n=0 is a corner case for the host-side memcpy and the device buffer.
    // The runtime header bumps zero-length allocations to 1 byte, so the
    // pipeline shouldn't crash; the kernel's `if (tid >= n) return;` guard
    // means no thread does work.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(0), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(0), None);
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(0), None);
    dag.add_root(s);

    let result = codegen_metal(&dag, "zero");
    let src = &result.mm_source;
    assert_emits_real_kernel(src, "zero-element add");
    // Allocation bytes should be `0u * sizeof(float)` — runtime tolerates 0.
    assert!(
        src.contains("0u * sizeof(float)"),
        "expected zero-length alloc reflected in source: {src}"
    );
}

#[test]
fn m7_single_element_reduction_emits_real_kernel() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(1), None);
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(s);

    let result = codegen_metal(&dag, "tiny");
    let src = &result.mm_source;
    assert_emits_real_kernel(src, "single-element sum");
    assert!(
        src.contains("k_reduce_sum_"),
        "single-element sum should still route to the reduction kernel: {src}"
    );
}

#[test]
fn m7_chain_of_elementwise_does_not_collapse_to_one_kernel() {
    // Until M-phase fusion lands, a chain of elementwise ops should emit
    // one MSL kernel per node — not silently merge them. This catches a
    // class of regressions where a future fusion pass over-merges.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(8), None);
    let c = dag.add_node(RiscOp::Load { name: "c".into() }, vec![], vec_f32(8), None);
    let m = dag.add_node(RiscOp::Mul, vec![a, b], vec_f32(8), None);
    let s = dag.add_node(RiscOp::Add, vec![m, c], vec_f32(8), None);
    let e = dag.add_node(RiscOp::Exp, vec![s], vec_f32(8), None);
    dag.add_root(e);

    let result = codegen_metal(&dag, "chain");
    let src = &result.mm_source;
    let kernels = src.matches("kernel void").count();
    assert_eq!(
        kernels, 3,
        "expected 3 separate kernels for unfused chain, got {kernels}: {src}"
    );
}

#[test]
fn wsm1_matmul_at_tile_boundary_routes_to_mps() {
    // M=K=N exactly equals TILE — tests boundary handling of the tiled kernel.

    fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(16, 16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(16, 16),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(16),
        },
        vec![a],
        tensor3_f32(16, 16, 16),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(16),
        },
        vec![b],
        tensor3_f32(16, 16, 16),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(16, 16, 16), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(16, 16),
        None,
    );
    dag.add_root(sum);

    let result = codegen_metal(&dag, "mm_tile");
    let src = &result.mm_source;
    assert_emits_real_kernel(src, "tile-boundary matmul");
    // WS-M1: f32 matmul routes to MPSMatrixMultiplication; the tiled MSL
    // kernel is no longer emitted for f32 dispatch. The boundary case
    // (M=N=K exactly equals TILE) is still meaningful: the MPS path
    // exercises the same shapes, and the emitter must not regress to
    // the stub.
    assert!(
        src.contains("chelis_metal_mps_gemm_f32(buf_0, buf_1, buf_5, 16u, 16u, 16u)"),
        "tile-boundary f32 matmul must dispatch through MPS: {src}"
    );
}

// ===========================================================================
// Cases the M-phase explicitly defers — must return typed unsupported.
// ===========================================================================

#[test]
fn m7_partial_axis_reduction_is_rejected_before_codegen() {
    // Sum{axis:1} with no matmul subgraph behind it — partial-axis
    // reduction is M4.next territory. Emitter must not silently emit a
    // full-axis kernel and lie about the result.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let r = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(r);

    let error = chelis_ir::ownership::lower_dag_ownership(dag)
        .expect_err("out-of-range reduction axis must not cross the verified boundary");
    assert!(
        error
            .to_string()
            .contains("axis 1 but input has 1 dimensions")
    );
}

#[test]
fn m7_oversized_reduction_returns_typed_unsupported() {
    // n > 4096 exceeds the single-threadgroup wrap-loop ceiling. Two-pass
    // reduction is M4.next; until then, this MUST fall through.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(8192),
        None,
    );
    let r = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(r);

    let error = try_codegen_metal(&dag, "big").unwrap_err();
    assert_typed_unsupported(&error, "oversized reduction");
}

#[test]
fn m7_non_power_of_two_reduction_emits_real_kernel() {
    // Regression for the red-team C1 finding: tree reduction silently
    // miscomputed for n not a power of 2 because tg_size was clamped to
    // n.min(256) and the halving loop dropped upper-half entries when
    // tg_size was odd. Fix decouples tg from n: kernel always uses
    // TG_SIZE=256, wrap loop strides over n. This test asserts the
    // emitter still produces real code for the awkward sizes.
    for &n in &[33usize, 50, 100, 200, 333, 1000, 4095] {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(n), None);
        let r = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_root(r);

        let result = codegen_metal(&dag, &format!("sum_{n}"));
        let src = &result.mm_source;
        assert!(src.contains("kernel void"), "n={n}: missing kernel:\n{src}");
        // Dispatch must always use TG_SIZE=256 regardless of n, so the
        // tree-reduce loop terminates exactly at every halving step.
        assert!(
            src.contains("chelis_metal_launch(") && src.contains("256u, 256u"),
            "n={n}: must dispatch fixed TG_SIZE=256: {src}"
        );
    }
}

#[test]
fn m7_rank2_elementwise_without_matmul_returns_typed_unsupported() {
    // M-phase emits rank-2 only in the matmul subgraph specialization.
    // A bare rank-2 add has no broadcast/stride machinery yet.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(4, 4),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(4, 4),
        None,
    );
    let s = dag.add_node(RiscOp::Add, vec![a, b], mat_f32(4, 4), None);
    dag.add_root(s);

    let error = try_codegen_metal(&dag, "rank2add").unwrap_err();
    assert_typed_unsupported(&error, "rank-2 bare elementwise");
}

#[test]
fn m7_bool_load_without_where_emits_real_output() {
    // Bool tensors are emittable as `device const bool*` in MSL kernels
    // (verified in M2 prelude on Apple Silicon family 7+), but the
    // M-phase emitter doesn't yet wire bool inputs through `where` or
    // any other op.
    let mut dag = Dag::new();
    let _cond = dag.add_node(RiscOp::Load { name: "c".into() }, vec![], vec_bool(8), None);
    // `where` doesn't exist as a RiscOp variant in this IR; the closest
    // unsupported case is loading a bool tensor and trying to add it to
    // a float, which the type checker would reject upstream — but a
    // bool Load alone with no op should still emit a real Load and
    // root-writeback (the input lands in outputs[0] unchanged).
    dag.add_root(_cond);

    let result = codegen_metal(&dag, "boolload");
    let src = &result.mm_source;
    assert_emits_real_kernel(src, "bool Load alone");
    assert!(
        src.contains("CHELIS_DTYPE_BOOL"),
        "bool root output must declare CHELIS_DTYPE_BOOL dtype: {src}"
    );
    assert!(
        src.contains("device const bool* a") || src.contains("sizeof(bool)"),
        "bool buffer should reach the .mm via direct emission: {src}"
    );
}

#[test]
fn wsm1_int32_unary_neg_emits_typed_kernel() {
    // WS-M1: int32 is now in the active Metal dtype set
    // (spec/04-type-system.md §1.1.3). Unary Neg on int32 should
    // emit a properly typed kernel (`device const int*`), not the
    // stub and not the f32 kernel that would silently misinterpret
    // the buffer's bytes.
    let int_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::Int32,
    };
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        int_ty.clone(),
        None,
    );
    let n = dag.add_node(RiscOp::Neg, vec![a], int_ty, None);
    dag.add_root(n);

    let result = codegen_metal(&dag, "int_neg");
    let src = &result.mm_source;
    assert!(
        src.contains("device const int* a"),
        "int32 unary neg must emit an int-typed input parameter: {src}"
    );
    assert!(
        src.contains("device int* out"),
        "int32 unary neg must emit an int-typed output parameter: {src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "int32 root output should declare CHELIS_DTYPE_I32 dtype: {src}"
    );
}

#[test]
fn wsm1_unary_transcendental_on_int_rejected_before_codegen() {
    // Defense in depth: the type checker rejects transcendentals on
    // integer precisions per spec §5.4. If a regression admitted such
    // a DAG, the Metal emitter must bail with a structured error, not
    // emit a kernel that calls `exp(int)` and silently miscompiles.
    let int_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::Int32,
    };
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        int_ty.clone(),
        None,
    );
    let n = dag.add_node(RiscOp::Exp, vec![a], int_ty, None);
    dag.add_root(n);

    let error = chelis_ir::ownership::lower_dag_ownership(dag)
        .expect_err("integer transcendental must not cross the verified boundary");
    assert!(error.to_string().contains("requires float input"));
}

#[test]
fn wsm1_f64_matmul_returns_typed_unsupported() {
    // f64 on Metal is hard-rejected per spec §1.1.3. A DAG that
    // reaches the codegen entry must bail; emit_dag's
    // `require_metal_admissible` returns the FP64-ALU diagnostic without
    // creating an artifact.
    let mat_f64 = |r: usize, c: usize| TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F64,
    };
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f64(2, 3),
        None,
    );
    dag.add_root(a);
    let error = try_codegen_metal(&dag, "f64_load").unwrap_err();
    assert_typed_unsupported(&error, "f64 input");
    assert!(error.to_string().contains("[04-TGT-1]"), "{error}");
}
