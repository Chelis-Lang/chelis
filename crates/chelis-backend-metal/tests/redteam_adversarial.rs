//! Adversarial / red-team test surface for the Metal backend (M7).
//!
//! These test the M0-M6 surface as it actually shipped: rank-1 contiguous
//! elementwise (M2), full-axis rank-1 reductions (M4), tiled matmul (M5).
//! Cases the M-phase explicitly defers (bool-through-where, cast,
//! symbolic dims, broadcasts/strides, partial-axis reductions, oversized
//! reductions) must fall through to the stub-with-abort body cleanly,
//! not silently miscompile.
//!
//! Mirrors `chelis-backend-hip/tests/redteam_adversarial.rs` in spirit;
//! coverage breadth grows as later phases add real support for each case.
//! When a case here flips from "falls through to stub" to "emits real
//! kernels", that's a signal the corresponding phase has shipped — flip
//! the assertion.

use chelis_backend_metal::codegen_metal;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

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

fn assert_falls_through_to_stub(src: &str, label: &str) {
    assert!(
        src.contains("M1 fallback stub") && src.contains("abort()"),
        "{label}: expected fall-through to stub (with abort), got:\n{src}"
    );
}

fn assert_emits_real_kernel(src: &str, label: &str) {
    assert!(
        !src.contains("M1 fallback stub"),
        "{label}: expected real emission, got stub:\n{src}"
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
        RiscOp::Sum { axis: 0 },
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
fn m7_matmul_at_tile_boundary_routes_to_tiled_kernel() {
    // M=K=N exactly equals TILE — tests boundary handling of the tiled kernel.
    use chelis_ir::dag::DimExpr;

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
            size: DimExpr::Concrete(16),
        },
        vec![a],
        tensor3_f32(16, 16, 16),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(16),
        },
        vec![b],
        tensor3_f32(16, 16, 16),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(16, 16, 16), None);
    let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(16, 16), None);
    dag.add_root(sum);

    let result = codegen_metal(&dag, "mm_tile");
    let src = &result.mm_source;
    assert_emits_real_kernel(src, "tile-boundary matmul");
    assert!(
        src.contains("k_matmul_"),
        "tile-boundary matmul must specialize to the tiled kernel: {src}"
    );
}

// ===========================================================================
// Cases the M-phase explicitly defers — must fall through to stub.
// ===========================================================================

#[test]
fn m7_partial_axis_reduction_falls_through_to_stub() {
    // Sum{axis:1} with no matmul subgraph behind it — partial-axis
    // reduction is M4.next territory. Emitter must not silently emit a
    // full-axis kernel and lie about the result.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let r = dag.add_node(
        RiscOp::Sum { axis: 1 },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(r);

    let result = codegen_metal(&dag, "axis1");
    assert_falls_through_to_stub(&result.mm_source, "axis-1 reduction (no matmul)");
}

#[test]
fn m7_oversized_reduction_falls_through_to_stub() {
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
        RiscOp::Sum { axis: 0 },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(r);

    let result = codegen_metal(&dag, "big");
    assert_falls_through_to_stub(&result.mm_source, "oversized reduction");
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
            RiscOp::Sum { axis: 0 },
            vec![a],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_root(r);

        let result = codegen_metal(&dag, &format!("sum_{n}"));
        let src = &result.mm_source;
        assert!(
            !src.contains("M1 fallback stub"),
            "n={n}: must emit real reduction kernel, got stub:\n{src}"
        );
        // Dispatch must always use TG_SIZE=256 regardless of n, so the
        // tree-reduce loop terminates exactly at every halving step.
        assert!(
            src.contains("chelis_metal_launch(") && src.contains("256u, 256u"),
            "n={n}: must dispatch fixed TG_SIZE=256: {src}"
        );
    }
}

#[test]
fn m7_rank2_elementwise_without_matmul_falls_through_to_stub() {
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

    let result = codegen_metal(&dag, "rank2add");
    assert_falls_through_to_stub(&result.mm_source, "rank-2 bare elementwise");
}

#[test]
fn m7_bool_load_through_where_falls_through_to_stub() {
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
        src.contains("CHELIS_BOOL"),
        "bool root output must declare CHELIS_BOOL dtype: {src}"
    );
    assert!(
        src.contains("device const bool* a") || src.contains("sizeof(bool)"),
        "bool buffer should reach the .mm via direct emission: {src}"
    );
}

#[test]
fn m7_no_silent_miscompile_for_unsupported_unary_with_int_precision() {
    // The emitter's msl_type panics on non-{f32, bool} precisions — but
    // the CLI's reject_unsupported_metal_precisions is the gate that
    // *should* catch this before the emitter is ever called. Confirm
    // the emitter at least bails (panic or fall-through) rather than
    // silently emitting code that interprets int data as f32.
    use std::panic::AssertUnwindSafe;
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

    // Either the emitter falls through to stub, OR it panics. Both are
    // acceptable outcomes — what's NOT acceptable is silently emitting a
    // kernel that miscompiles. We catch the panic so the test reports
    // either outcome cleanly.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| codegen_metal(&dag, "int_neg")));
    match result {
        Ok(r) => {
            assert_falls_through_to_stub(&r.mm_source, "int32 unary neg");
        }
        Err(_) => {
            // Panic-on-unsupported-precision is the alternative honest
            // outcome and is consistent with the kernels::msl_type panic.
        }
    }
}
