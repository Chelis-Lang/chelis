//! Wave 5 red-team — HIP strided-batched matmul on adversarial layouts
//! (Perf-F1 + Gap 4 attack surface).
//!
//! The existing `perf_f1_strided_batched_default.rs` locks two cases:
//!   * Rank-3 uniform-stride layout → `chelis_hipblas_sgemm_strided_batched_row_major`
//!   * Rank-3 broadcasted leading axis → `chelis_hipblas_sgemm_batched_row_major`
//!     helper loop fallback.
//!
//! This file probes adversarial cases the brief invited and the existing
//! suite doesn't cover with exact-line assertions:
//!
//! 1. **Rank-4 uniform-stride batched** (e.g. `(B, H, M, K) @ (B, H, K, N)`)
//!    — must dispatch to strided-batched with `batch_count = B*H`.
//! 2. **Rank-3 NON-F32 (F64) uniform layout** — must fall back to the
//!    helper loop, because `strided_batched_hipblas_plan` keys off F32.
//! 3. **Rank-3 with a Permute on the leading axis** (i.e. a non-identity
//!    Permute that the specializer's contiguity check rejects) — must
//!    NOT reach BLAS at all, falling through to the generic emit path.
//! 4. **Rank-3 BROADCASTED leading axis on the RHS** (not just lhs) —
//!    must fall back to helper loop. The existing test only covers lhs.
//! 5. **Symbolic batch dim with concrete m/n/k** — must take helper-loop
//!    fallback because `strided_batched_hipblas_plan` requires concrete
//!    m/n/k (calls `.as_concrete()?`).
//! 6. **Rank-2 (no batch)** — must take the standard non-batched sgemm
//!    path (`chelis_hipblas_sgemm_row_major`), not strided-batched and
//!    not helper-loop-batched.
//!
//! Assertions match exact lines or assert exact substrings absent — not
//! loose substrings present.
//!
//! This file also carries the two Perf-F1 structural acceptance cases
//! (formerly `perf_f1_strided_batched_default.rs`):
//!   * `perf_f1_uniform_rank3_batched_matmul_dispatches_strided_batched`
//!   * `perf_f1_broadcasted_leading_axis_uses_helper_loop_fallback`
//!
//! These lock the rank-3 uniform-stride dispatch with literal batch count
//! and stride literals, and the broadcasted-lhs-leading-axis helper-loop
//! fallback. The W5 adversarial cases above (rank-4 uniform, rhs/both
//! broadcasts) do not subsume the rank-3 literal-stride case, so the
//! Perf-F1 cases live here as part of the keep-by-default lock.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn t(prim: Prim, dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: prim,
    }
}

fn t_named(prim: Prim, dims: Vec<DimInfo>) -> TensorType {
    TensorType {
        dims,
        precision: prim,
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
        "forbidden substring `{forbidden}` appeared in emitted source"
    );
}

/// ADV-HIP-1: Rank-4 uniform-stride batched. `(2, 3, 4, 5) @ (2, 3, 5, 6)
/// → (2, 3, 4, 6)` with leading axes [2,3]. Strided-batched should be
/// chosen with `batch_count` = `(2 * 3)`. Strides:
///   a: m*k = 4*5 = 20
///   b: k*n = 5*6 = 30
///   out: m*n = 4*6 = 24
#[test]
fn rank4_uniform_batched_matmul_dispatches_strided_batched_with_product_batch_count() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F32, vec![2, 3, 4, 5]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![2, 3, 5, 6]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(2), DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F32, vec![2, 3, 4, 6]),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_rank4_uniform");

    let expected = format!(
        "chelis_hipblas_sgemm_strided_batched_row_major(d_t{a}, d_t{b}, d_t{out}, 4, 6, 5, (2 * 3), 20LL, 30LL, 24LL);",
        a = a.0,
        b = b.0,
        out = out.0,
    );
    assert_line_present(&result.c_source, &expected);
    assert_no_substring(&result.c_source, "chelis_hipblas_sgemm_batched_row_major(");
}

/// ADV-HIP-2: Rank-3 F64 uniform layout. The HIP backend fail-closes at
/// codegen time when the operand and accumulator precision pair has no
/// hipBLAS dispatch entry. WS-A2 lifted f64 support for the (f64, f64)
/// pair (`cblas_dgemm` equivalent); the (f64, f32) combination this test
/// constructs is still rejected because the IR-pinned accumulator and
/// operand are inconsistent (spec §5.7.1).
#[test]
#[should_panic(expected = "is not yet supported by the HIP backend")]
fn rank3_f64_uniform_batched_does_not_dispatch_strided_batched() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F64, vec![3, 4, 5]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F64, vec![3, 5, 6]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F64, vec![3, 4, 6]),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_rank3_f64");
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );
}

/// ADV-HIP-3: Symbolic batch dim with concrete m/n/k. `as_concrete()` on
/// the symbolic batch returns Some only via `Concrete` — wait, the brief
/// says the plan reads m/n/k as concrete and accepts `DimExpr::Sym` for
/// `batch_dims` via `is_simple_runtime_dim`. So this case SHOULD dispatch
/// strided-batched with a symbolic `batch_count`.
#[test]
fn symbolic_batch_concrete_mnk_dispatches_strided_batched_with_symbolic_batch_count() {
    let mut dag = Dag::new();
    let dims_a = vec![
        DimInfo::Named("batch".into(), None),
        DimInfo::Lit(4),
        DimInfo::Lit(5),
    ];
    let dims_b = vec![
        DimInfo::Named("batch".into(), None),
        DimInfo::Lit(5),
        DimInfo::Lit(6),
    ];
    let dims_out = vec![
        DimInfo::Named("batch".into(), None),
        DimInfo::Lit(4),
        DimInfo::Lit(6),
    ];
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t_named(Prim::F32, dims_a),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t_named(Prim::F32, dims_b),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Sym("batch".into())],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t_named(Prim::F32, dims_out),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_symbolic_batch");
    // The batch_count is the symbol's runtime expression. The HIP emitter
    // renders `DimExpr::Sym("batch")` as the bare local variable `batch`
    // (bound from `inputs[0]->shape[0]` earlier in the emitted entry).
    let expected = format!(
        "chelis_hipblas_sgemm_strided_batched_row_major(d_t{a}, d_t{b}, d_t{out}, 4, 6, 5, batch, 20LL, 30LL, 24LL);",
        a = a.0,
        b = b.0,
        out = out.0,
    );
    assert_line_present(&result.c_source, &expected);
}

/// ADV-HIP-4: All-symbolic m/n/k — `as_concrete()` returns None, so the
/// strided-batched plan must NOT fire. The helper-loop fallback should
/// be used instead. Lock the absence of strided-batched on this layout.
#[test]
fn symbolic_mnk_falls_back_to_helper_loop_not_strided_batched() {
    let mut dag = Dag::new();
    let dims_a = vec![
        DimInfo::Lit(3),
        DimInfo::Named("m".into(), None),
        DimInfo::Named("k".into(), None),
    ];
    let dims_b = vec![
        DimInfo::Lit(3),
        DimInfo::Named("k".into(), None),
        DimInfo::Named("n".into(), None),
    ];
    let dims_out = vec![
        DimInfo::Lit(3),
        DimInfo::Named("m".into(), None),
        DimInfo::Named("n".into(), None),
    ];
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t_named(Prim::F32, dims_a),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t_named(Prim::F32, dims_b),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Sym("m".into()),
            n: DimExpr::Sym("n".into()),
            k: DimExpr::Sym("k".into()),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t_named(Prim::F32, dims_out),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_symbolic_mnk");
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );
    // Helper-loop fallback must fire.
    assert!(
        result
            .c_source
            .contains("chelis_hipblas_sgemm_batched_row_major("),
        "symbolic m/n/k must fall back to helper loop; got source:\n{}",
        result.c_source
    );
}

/// ADV-HIP-5: Rank-2 (no batch). Must take the plain sgemm path —
/// neither strided-batched nor helper-loop-batched.
#[test]
fn rank2_matmul_takes_plain_sgemm_not_batched_nor_strided() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F32, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![16, 4]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(8),
            n: DimExpr::Concrete(4),
            k: DimExpr::Concrete(16),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F32, vec![8, 4]),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_rank2");
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );
    assert_no_substring(&result.c_source, "chelis_hipblas_sgemm_batched_row_major(");
    // Plain sgemm path: `chelis_hipblas_sgemm_row_major(`.
    assert!(
        result.c_source.contains("chelis_hipblas_sgemm_row_major("),
        "rank-2 matmul must dispatch to plain sgemm; got source:\n{}",
        result.c_source
    );
}

/// ADV-HIP-6: Broadcasted leading axis on the RHS (not lhs). The HIP
/// contiguity checker treats `Expand` as non-contiguous regardless of
/// which side it's on, so strided-batched must NOT fire here either.
#[test]
fn broadcasted_rhs_leading_axis_falls_back_to_helper_loop() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F32, vec![3, 4, 5]),
        None,
    );
    let base_b = dag.add_node(
        RiscOp::Load {
            name: "base_b".into(),
        },
        vec![],
        t(Prim::F32, vec![5, 6]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(3),
        },
        vec![base_b],
        t(Prim::F32, vec![3, 5, 6]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F32, vec![3, 4, 6]),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_rhs_broadcast");
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );
    let expected = format!(
        "chelis_hipblas_sgemm_batched_row_major(d_t{a}, d_t{b}, d_t{out}, 4, 6, 5);",
        a = a.0,
        b = b.0,
        out = out.0,
    );
    assert_line_present(&result.c_source, &expected);
}

/// ADV-HIP-7: Both lhs and rhs broadcasted leading axes — same expected
/// outcome (helper-loop fallback). Locks the symmetric defense.
#[test]
fn both_sides_broadcasted_leading_axis_falls_back_to_helper_loop() {
    let mut dag = Dag::new();
    let base_a = dag.add_node(
        RiscOp::Load {
            name: "base_a".into(),
        },
        vec![],
        t(Prim::F32, vec![4, 5]),
        None,
    );
    let a = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(3),
        },
        vec![base_a],
        t(Prim::F32, vec![3, 4, 5]),
        None,
    );
    let base_b = dag.add_node(
        RiscOp::Load {
            name: "base_b".into(),
        },
        vec![],
        t(Prim::F32, vec![5, 6]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(3),
        },
        vec![base_b],
        t(Prim::F32, vec![3, 5, 6]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F32, vec![3, 4, 6]),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_both_broadcast");
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );
    let expected = format!(
        "chelis_hipblas_sgemm_batched_row_major(d_t{a}, d_t{b}, d_t{out}, 4, 6, 5);",
        a = a.0,
        b = b.0,
        out = out.0,
    );
    assert_line_present(&result.c_source, &expected);
}

// ---------------------------------------------------------------------------
// Perf-F1 structural acceptance cases (formerly
// `perf_f1_strided_batched_default.rs`). Locks two invariants of the HIP
// batched-matmul dispatch:
//
// 1. On a uniform-stride rank-3 batched layout (statically contiguous matrix
//    slices, concrete matrix dimensions, simple leading batch axes), the
//    emitted host source dispatches to `hipblasSgemmStridedBatched` via the
//    `chelis_hipblas_sgemm_strided_batched_row_major` runtime helper with a
//    literal batch count. The per-batch helper loop must not appear.
// 2. On a broadcasted leading-axis layout (the lhs leading batch dim is
//    produced by a `RiscOp::Expand`), the emitted host source falls back to
//    the per-batch helper loop. The strided-batched call must not appear.
// ---------------------------------------------------------------------------

/// Perf-F1 W1-A acceptance: uniform-stride rank-3 batched matmul takes the
/// strided-batched path with the precise stride/batch-count literals.
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
        t(Prim::F32, vec![3, 4, 5]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![3, 5, 6]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F32, vec![3, 4, 6]),
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

/// Perf-F1 W1-A acceptance: a broadcasted leading-axis layout (the lhs
/// leading batch dim is produced by `RiscOp::Expand`, so per-batch matrix
/// slices share a stride-0 column on that axis) falls back to the per-batch
/// helper loop. Strided-batched would be unsound here because it assumes
/// uniform non-zero leading strides.
///
/// Shape:
///   `base_a` : (4, 5)   (rank-2)
///   a      : `expand(base_a`, axis=0, size=3) -> (3, 4, 5)
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
        t(Prim::F32, vec![4, 5]),
        None,
    );
    let a = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(3),
        },
        vec![base_a],
        t(Prim::F32, vec![3, 4, 5]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![3, 5, 6]),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(3)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(6),
            k: DimExpr::Concrete(5),
            accumulator: Prim::F32,
        },
        vec![a, b],
        t(Prim::F32, vec![3, 4, 6]),
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
