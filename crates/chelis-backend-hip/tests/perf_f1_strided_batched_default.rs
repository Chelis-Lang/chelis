//! Perf-F1 — HIP strided-batched matmul is the default on uniform layouts.
//!
//! This is the orchestrator-visible structural acceptance signal for the
//! Perf-F1 work item from `docs/gap_synthesis.md` §5. It locks two
//! invariants of the HIP batched-matmul dispatch:
//!
//! 1. On a uniform-stride batched layout (rank ≥ 3 with statically
//!    contiguous matrix slices, concrete matrix dimensions, and simple
//!    leading batch axes), the emitted host source dispatches to
//!    `hipblasSgemmStridedBatched` via the
//!    `chelis_hipblas_sgemm_strided_batched_row_major` runtime helper.
//!    The per-batch helper loop (`chelis_hipblas_sgemm_batched_row_major`)
//!    must not appear.
//!
//! 2. On a broadcasted leading-axis layout (the leading batch dim is
//!    produced by a `RiscOp::Expand`, so per-batch slice strides are not
//!    statically uniform), the emitted host source falls back to the
//!    per-batch helper loop. The strided-batched call must not appear.
//!
//! Assertions match exact lines after trim — not loose substrings — per
//! `feedback_redteam_thoroughness.md` ("tests must check exact output").
//! The default workspace test pass is the orchestrator-visible signal;
//! the GPU manual gate
//! (`cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored
//! --test-threads=1` with `HSA_OVERRIDE_GFX_VERSION=11.5.1` per
//! `docs/local_hip_environment.md`) is the numerical oracle.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn trimmed_lines(source: &str) -> Vec<&str> {
    source.lines().map(str::trim).collect()
}

fn assert_line_present(source: &str, expected: &str) {
    let lines = trimmed_lines(source);
    assert!(
        lines.contains(&expected),
        "expected exact line not found:\n  expected: {expected}\n  source:\n{source}"
    );
}

fn assert_no_substring(source: &str, forbidden: &str) {
    assert!(
        !source.contains(forbidden),
        "forbidden substring `{forbidden}` appeared in emitted source:\n{source}"
    );
}

/// W1-A acceptance: uniform-stride batched matmul takes the strided-batched
/// path with the precise stride/batch-count literals.
///
/// Shape: `(3, 4, 5) @ (3, 5, 6) -> (3, 4, 6)` so
/// `m=4, n=6, k=5, batch=3, a_stride=m*k=20, b_stride=k*n=30,
/// out_stride=m*n=24`.
#[test]
fn perf_f1_uniform_rank3_batched_matmul_dispatches_strided_batched() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        tensor3_f32(3, 4, 5),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        tensor3_f32(3, 5, 6),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
        },
        vec![a, b],
        tensor3_f32(3, 4, 6),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "perf_f1_uniform_strided_batched");

    // The exact emitted line is deterministic from `emit_blas_matmul`:
    //   chelis_hipblas_sgemm_strided_batched_row_major(d_t<a>, d_t<b>,
    //       d_t<out>, <m>, <n>, <k>, <batch_count>, <a_stride>LL,
    //       <b_stride>LL, <out_stride>LL);
    let expected_line = format!(
        "chelis_hipblas_sgemm_strided_batched_row_major(d_t{a}, d_t{b}, d_t{out}, 4, 6, 5, 3, 20LL, 30LL, 24LL);",
        a = a.0,
        b = b.0,
        out = out.0,
    );
    assert_line_present(&result.c_source, &expected_line);

    // The helper-loop fallback must not be reached on a uniform layout.
    assert_no_substring(&result.c_source, "chelis_hipblas_sgemm_batched_row_major(");

    // The strided-batched helper still requires the hipBLAS link flag.
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "strided-batched dispatch must surface `-lhipblas`; got {:?}",
        result.link_flags
    );
}

/// W1-A acceptance: a broadcasted leading-axis layout (the lhs leading
/// batch dim is produced by `RiscOp::Expand`, so per-batch matrix slices
/// share a stride-0 column on that axis) falls back to the per-batch
/// helper loop. Strided-batched would be unsound here because it
/// assumes uniform non-zero leading strides.
///
/// Shape:
///   base_a : (4, 5)   (rank-2)
///   a      : expand(base_a, axis=0, size=3) -> (3, 4, 5)
///            (statically non-contiguous; stride[0] == 0 broadcast)
///   b      : (3, 5, 6)
///   out    : (3, 4, 6)
#[test]
fn perf_f1_broadcasted_leading_axis_uses_helper_loop_fallback() {
    let mut dag = Dag::new();
    let base_a = dag.add_node(
        RiscOp::Load {
            name: "base_a".into(),
        },
        vec![],
        mat_f32(4, 5),
        None,
    );
    let a = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(3),
        },
        vec![base_a],
        tensor3_f32(3, 4, 5),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        tensor3_f32(3, 5, 6),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
        },
        vec![a, b],
        tensor3_f32(3, 4, 6),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "perf_f1_broadcasted_helper_loop");

    // The exact emitted fallback line is deterministic from
    // `emit_blas_matmul`:
    //   chelis_hipblas_sgemm_batched_row_major(d_t<a>, d_t<b>, d_t<out>,
    //       <m>, <n>, <k>);
    let expected_line = format!(
        "chelis_hipblas_sgemm_batched_row_major(d_t{a}, d_t{b}, d_t{out}, 4, 6, 5);",
        a = a.0,
        b = b.0,
        out = out.0,
    );
    assert_line_present(&result.c_source, &expected_line);

    // Strided-batched must not appear when broadcasted leading strides
    // would make per-batch offsets non-uniform.
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );

    // Helper-loop fallback still needs the hipBLAS link flag.
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "helper-loop fallback must surface `-lhipblas`; got {:?}",
        result.link_flags
    );

    // Static reference: `Expand` on the leading axis is exactly the
    // broadcasted-leading-stride shape that disqualifies a uniform
    // strided-batched plan. If this assertion is touched, the
    // closed-list contiguity vocabulary in
    // `crates/chelis-backend-hip/src/emit.rs::node_is_statically_contiguous`
    // is what changed; surface the change to the spec.
}
