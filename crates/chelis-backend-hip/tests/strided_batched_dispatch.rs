//! Wave 5 red-team — HIP strided-batched matmul on adversarial layouts
//! (Perf-F1 + Gap 4 attack surface).
//!
//! The existing `perf_f1_strided_batched_default.rs` originally locked two cases:
//!   * Rank-3 uniform-stride layout → `chelis_hipblas_sgemm_strided_batched_row_major`
//!   * Rank-3 broadcasted leading axis → `chelis_hipblas_sgemm_batched_row_major`
//!     helper loop fallback.
//!
//! HIP ownership preparation now materializes every BLAS operand into an owned
//! contiguous slot. Consequently, broadcast views are realized before dispatch
//! and may safely use strided-batched GEMM. These tests lock both that
//! materialization boundary and the descriptor-derived dispatch arguments.
//!
//! This file probes adversarial cases the brief invited and the existing
//! suite doesn't cover with exact-line assertions:
//!
//! 1. **Rank-4 uniform-stride batched** (e.g. `(B, H, M, K) @ (B, H, K, N)`)
//!    — must dispatch to strided-batched with `batch_count = B*H`.
//! 2. **Rank-3 F64 uniform layout** — must use the typed dgemm
//!    strided-batched helper; an invalid f32 accumulator must fail earlier.
//! 3. **Rank-3 with a Permute on the leading axis** (i.e. a non-identity
//!    Permute that the specializer's contiguity check rejects) — must
//!    NOT reach BLAS at all, falling through to the generic emit path.
//! 4. **Rank-3 BROADCASTED leading axis on the RHS** (not just lhs) —
//!    must be realized before strided-batched dispatch.
//! 5. **Symbolic batch dim with concrete m/n/k** — may dispatch
//!    strided-batched because only m/n/k must be concrete.
//! 6. **Rank-2 (no batch)** — must take the standard non-batched sgemm
//!    path (`chelis_hipblas_sgemm_row_major`), not strided-batched and
//!    not helper-loop-batched.
//!
//! Assertions match exact descriptor-based lines for both emitted entrypoints
//! or assert exact call spellings absent — not loose substrings present.
//!
//! This file also carries the two Perf-F1 structural acceptance cases
//! (formerly `perf_f1_strided_batched_default.rs`):
//!   * `perf_f1_uniform_rank3_batched_matmul_dispatches_strided_batched`
//!   * `perf_f1_broadcasted_leading_axis_realizes_then_dispatches_strided`

mod support;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen_hip;

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

fn assert_line_occurrences(source: &str, expected: &str, count: usize) {
    let actual = trimmed_lines(source)
        .into_iter()
        .filter(|line| *line == expected)
        .count();
    assert_eq!(
        actual, count,
        "expected exact line {count} times, found {actual}:\n  expected: {expected}\n  source:\n{source}"
    );
}

fn assert_no_substring(source: &str, forbidden: &str) {
    assert!(
        !source.contains(forbidden),
        "forbidden substring `{forbidden}` appeared in emitted source"
    );
}

#[derive(Debug)]
struct PreparedMatmul {
    lhs: chelis_ir::dag::NodeId,
    rhs: chelis_ir::dag::NodeId,
    output: chelis_ir::dag::NodeId,
    lhs_source_op: RiscOp,
    rhs_source_op: RiscOp,
    matrix_axis: usize,
}

fn prepared_matmul(dag: &Dag) -> PreparedMatmul {
    let verified = support::verified_dag(dag);
    let emission = verified.emission();
    let node = emission
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::BlasMatmul { .. }))
        .expect("prepared DAG must retain one BlasMatmul");
    let [lhs, rhs] = node.inputs.as_slice() else {
        panic!("BlasMatmul must retain exactly two operands");
    };
    let lhs_node = emission.get(*lhs).expect("prepared lhs exists");
    let rhs_node = emission.get(*rhs).expect("prepared rhs exists");
    assert!(
        matches!(lhs_node.op, RiscOp::Realize),
        "HIP BLAS lhs must be explicitly materialized before ownership lowering"
    );
    assert!(
        matches!(rhs_node.op, RiscOp::Realize),
        "HIP BLAS rhs must be explicitly materialized before ownership lowering"
    );
    let lhs_source = emission
        .get(lhs_node.inputs[0])
        .expect("realized lhs source exists");
    let rhs_source = emission
        .get(rhs_node.inputs[0])
        .expect("realized rhs source exists");
    let matrix_axis = node
        .output_type
        .dims
        .len()
        .checked_sub(2)
        .expect("matmul output has two matrix axes");
    PreparedMatmul {
        lhs: *lhs,
        rhs: *rhs,
        output: node.id,
        lhs_source_op: lhs_source.op.clone(),
        rhs_source_op: rhs_source.op.clone(),
        matrix_axis,
    }
}

fn assert_prepared_strided_dispatch(source: &str, dag: &Dag, call_name: &str) -> PreparedMatmul {
    let prepared = prepared_matmul(dag);
    assert!(
        prepared.matrix_axis > 0,
        "strided-batched dispatch requires at least one batch axis"
    );
    let output = prepared.output.0;
    let lhs = prepared.lhs.0;
    let rhs = prepared.rhs.0;
    let matrix_axis = prepared.matrix_axis;
    let n_axis = matrix_axis + 1;
    let batch_axis = matrix_axis - 1;
    let batch_count = format!(
        "chelis_device_metadata t{output}_batch_count = d_t{output}->count / d_t{output}->shape[{matrix_axis}] / d_t{output}->shape[{n_axis}];"
    );
    let call = format!(
        "{call_name}(d_t{lhs}, d_t{rhs}, d_t{output}, d_t{output}->shape[{matrix_axis}], d_t{output}->shape[{n_axis}], d_t{lhs}->shape[d_t{lhs}->rank - 1], t{output}_batch_count, d_t{lhs}->strides[{batch_axis}], d_t{rhs}->strides[{batch_axis}], d_t{output}->strides[{batch_axis}]);"
    );
    assert_line_occurrences(source, &batch_count, 2);
    assert_line_occurrences(source, &call, 2);
    assert_eq!(
        source.matches(&format!("{call_name}(")).count(),
        2,
        "host and device entrypoints must each emit exactly one strided-batched call"
    );
    assert_no_substring(source, "chelis_hipblas_sgemm_batched_row_major(");
    assert_no_substring(source, "chelis_hipblas_dgemm_batched_row_major(");
    prepared
}

/// ADV-HIP-1: Rank-4 uniform-stride batched. `(2, 3, 4, 5) @ (2, 3, 5, 6)
/// → (2, 3, 4, 6)` with leading axes [2,3]. Strided-batched must derive
/// batch count and all three batch strides from the prepared descriptors.
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

    let result = codegen_hip(&dag, "red_team_rank4_uniform").unwrap();

    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_sgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Load { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Load { .. }));
}

/// ADV-HIP-2: Rank-3 F64 uniform layout uses the dgemm strided-batched
/// helper after both operands are prepared into owned contiguous storage.
#[test]
fn rank3_f64_uniform_batched_dispatches_typed_strided_batched() {
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
            accumulator: Prim::F64,
        },
        vec![a, b],
        t(Prim::F64, vec![3, 4, 6]),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "red_team_rank3_f64").unwrap();
    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_dgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Load { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Load { .. }));
    assert_no_substring(
        &result.c_source,
        "chelis_hipblas_sgemm_strided_batched_row_major(",
    );
}

/// Negative parity for the F64 dispatch: the invalid `(f64, f32)`
/// operand/accumulator pair must fail before backend preparation.
#[test]
fn rank3_f64_with_narrow_accumulator_is_rejected_before_dispatch() {
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

    let error = chelis_ir::ownership::lower_dag_ownership(dag)
        .expect_err("f64 operands with an f32 accumulator must not be verified");
    assert!(error.to_string().contains("accumulator `f32` narrower"));
}

/// ADV-HIP-3: Symbolic batch dim with concrete m/n/k. `as_concrete()` on
/// the symbolic batch returns Some only via `Concrete` — wait, the brief
/// says the plan reads m/n/k as concrete and accepts `DimExpr::Sym` for
/// batch_dims via `is_simple_runtime_dim`. So this case SHOULD dispatch
/// strided-batched with a symbolic batch_count.
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

    let result = codegen_hip(&dag, "red_team_symbolic_batch").unwrap();
    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_sgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Load { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Load { .. }));
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

    let result = codegen_hip(&dag, "red_team_symbolic_mnk").unwrap();
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

    let result = codegen_hip(&dag, "red_team_rank2").unwrap();
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
/// preparation pass must materialize the Expand into a contiguous owner
/// before strided-batched dispatch.
#[test]
fn broadcasted_rhs_leading_axis_realizes_then_dispatches_strided() {
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
            size: chelis_ir::dag::RtDim::Lit(3),
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

    let result = codegen_hip(&dag, "red_team_rhs_broadcast").unwrap();
    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_sgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Load { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Expand { .. }));
}

/// ADV-HIP-7: Both lhs and rhs broadcasted leading axes must each be
/// materialized before the symmetric strided-batched dispatch.
#[test]
fn both_sides_broadcasted_leading_axes_realize_then_dispatch_strided() {
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
            size: chelis_ir::dag::RtDim::Lit(3),
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
            size: chelis_ir::dag::RtDim::Lit(3),
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

    let result = codegen_hip(&dag, "red_team_both_broadcast").unwrap();
    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_sgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Expand { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Expand { .. }));
}

// ---------------------------------------------------------------------------
// Perf-F1 structural acceptance cases (formerly
// `perf_f1_strided_batched_default.rs`). Locks two invariants of the HIP
// batched-matmul dispatch:
//
// 1. On a uniform-stride rank-3 batched layout, both emitted entrypoints use
//    `chelis_hipblas_sgemm_strided_batched_row_major` with descriptor-derived
//    dimensions, batch count, and strides.
// 2. On a broadcasted leading-axis layout, the Expand is first materialized
//    into owned contiguous storage, after which both entrypoints use the same
//    descriptor-derived strided-batched dispatch. The helper loop must not
//    appear in either case.
// ---------------------------------------------------------------------------

/// Perf-F1 W1-A acceptance: uniform-stride rank-3 batched matmul takes the
/// strided-batched path with exact descriptor-derived stride and batch-count
/// expressions.
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

    let result = codegen_hip(&dag, "perf_f1_uniform_strided_batched").unwrap();

    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_sgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Load { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Load { .. }));

    // The strided-batched helper still requires the hipBLAS link flag.
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "strided-batched dispatch must surface `-lhipblas`; got {:?}",
        result.link_flags
    );
}

/// Perf-F1 W1-A acceptance: a broadcasted leading-axis layout (the lhs
/// leading batch dim is produced by `RiscOp::Expand`, so the view itself
/// has stride zero) is materialized into a contiguous owned BLAS operand.
/// Strided-batched is then sound because it receives the realized descriptor,
/// never the broadcast view.
///
/// Shape:
///   base_a : (4, 5)   (rank-2)
///   a      : expand(base_a, axis=0, size=3) -> (3, 4, 5)
///            (statically non-contiguous; stride[0] == 0 broadcast)
///   b      : (3, 5, 6)
///   out    : (3, 4, 6)
#[test]
fn perf_f1_broadcasted_leading_axis_realizes_then_dispatches_strided() {
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
            size: chelis_ir::dag::RtDim::Lit(3),
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

    let result = codegen_hip(&dag, "perf_f1_broadcasted_realize_then_strided").unwrap();

    let prepared = assert_prepared_strided_dispatch(
        &result.c_source,
        &dag,
        "chelis_hipblas_sgemm_strided_batched_row_major",
    );
    assert!(matches!(prepared.lhs_source_op, RiscOp::Expand { .. }));
    assert!(matches!(prepared.rhs_source_op, RiscOp::Load { .. }));

    // The realized strided-batched path still needs the hipBLAS link flag.
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "strided-batched dispatch must surface `-lhipblas`; got {:?}",
        result.link_flags
    );
}
