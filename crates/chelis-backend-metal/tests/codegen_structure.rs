//! Structural tests for Metal code generation.
//!
//! These verify the generated `.mm` source is well-formed without requiring
//! a Metal device or `clang++` to be present. They run in default CI on
//! every platform (Linux/macOS/Windows).
//!
//! M1 ships the stub-output assertions. M2 fills in elementwise structural
//! coverage; M4 reductions; M5 matmul.

use chelis_backend_metal::codegen_metal;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn build_simple_add_dag() -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(8), None);
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![s],
        vec_f32(8),
        None,
    );
    dag.add_root(stored);
    dag
}

#[test]
fn m1_emits_extern_c_signature_and_runtime_import() {
    let dag = build_simple_add_dag();
    let result = codegen_metal(&dag, "f");

    // Header carries the canonical extern "C" ABI string regardless of
    // whether the body is the M1 stub or M2 real emission.
    assert!(
        result
            .h_header
            .contains("extern \"C\" void f(chelis_tensor **inputs"),
        "h_header missing extern \"C\" signature: {}",
        result.h_header
    );

    // Both M1 stub and M2 real emission #import the Metal runtime header
    // so the link recipe stays consistent.
    assert!(
        result
            .mm_source
            .contains("#import \"chelis_metal_runtime.h\""),
        "mm_source missing #import: {}",
        result.mm_source
    );
    // Body emits the function signature with the C ABI.
    assert!(
        result
            .mm_source
            .contains("extern \"C\" void f(chelis_tensor **inputs"),
        "mm_source missing function signature: {}",
        result.mm_source
    );
}

#[test]
fn m1_stub_carries_link_flags_for_metal_and_foundation() {
    let dag = build_simple_add_dag();
    let result = codegen_metal(&dag, "f");

    // Pairs are emitted as ("-framework", "Metal") so a downstream consumer
    // can pass them to clang++ verbatim.
    let flat: Vec<&str> = result.link_flags.iter().map(|s| s.as_str()).collect();
    assert!(
        flat.windows(2).any(|w| w == ["-framework", "Metal"]),
        "link_flags missing -framework Metal: {:?}",
        result.link_flags
    );
    assert!(
        flat.windows(2).any(|w| w == ["-framework", "Foundation"]),
        "link_flags missing -framework Foundation: {:?}",
        result.link_flags
    );
}

#[test]
fn m1_stub_extracts_input_and_output_labels_from_loads_and_stores() {
    let dag = build_simple_add_dag();
    let result = codegen_metal(&dag, "f");
    assert_eq!(result.input_labels, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(result.output_labels, vec!["out".to_string()]);
}

#[test]
fn m1_compile_flags_request_objc_arc_and_cxx17() {
    let dag = build_simple_add_dag();
    let result = codegen_metal(&dag, "f");
    assert!(
        result.compile_flags.iter().any(|f| f == "-fobjc-arc"),
        "compile_flags missing -fobjc-arc: {:?}",
        result.compile_flags
    );
    assert!(
        result.compile_flags.iter().any(|f| f == "-std=c++17"),
        "compile_flags missing -std=c++17: {:?}",
        result.compile_flags
    );
}

// ===========================================================================
// M2: real elementwise emission. Same-shape contiguous rank-1 first cut.
// Structural assertions only; numerical correctness lives in M6.
// ===========================================================================

#[test]
fn m2_simple_add_emits_msl_kernel_void_and_thread_position() {
    let dag = build_simple_add_dag();
    let result = codegen_metal(&dag, "f");
    let src = &result.mm_source;

    // M2 emission MUST replace the M1 stub. abort() in the stub body is the
    // signal the emitter fell through to the fallback; for a supported
    // shape we should emit real kernels and dispatch sites.
    assert!(
        !src.contains("M1 fallback stub"),
        "M2 should emit real code for a simple rank-1 add, but fell back to the stub:\n{src}"
    );

    // MSL kernel signature shape.
    assert!(
        src.contains("kernel void"),
        "expected MSL `kernel void` declaration: {src}"
    );
    assert!(
        src.contains("[[thread_position_in_grid]]"),
        "expected `[[thread_position_in_grid]]` thread index: {src}"
    );
    // Buffer-qualified inputs/outputs at expected indices.
    assert!(
        src.contains("device const float* a [[buffer(0)]]"),
        "expected `device const float* a [[buffer(0)]]`: {src}"
    );
    assert!(
        src.contains("device const float* b [[buffer(1)]]"),
        "expected `device const float* b [[buffer(1)]]`: {src}"
    );
    assert!(
        src.contains("device float* out [[buffer(2)]]"),
        "expected `device float* out [[buffer(2)]]`: {src}"
    );
    assert!(
        src.contains("constant uint& n [[buffer(3)]]"),
        "expected scalar uniform `constant uint& n [[buffer(3)]]`: {src}"
    );
    // Body uses MSL operator `+` directly.
    assert!(
        src.contains("out[tid] = a[tid] + b[tid];"),
        "expected MSL add body: {src}"
    );
}

#[test]
fn m2_simple_add_emits_raw_string_literal_and_dispatch_site() {
    let dag = build_simple_add_dag();
    let result = codegen_metal(&dag, "f");
    let src = &result.mm_source;

    // Kernel sources are embedded as C++11 raw string literals so escapes
    // stay readable. The MSL marker pair `R"MSL(` / `)MSL"` is the cache key
    // for chelis_metal_get_pipeline.
    assert!(
        src.contains("@R\"MSL("),
        "expected raw NSString literal opener `@R\"MSL(`: {src}"
    );
    assert!(
        src.contains(")MSL\""),
        "expected raw NSString literal closer `)MSL\"`: {src}"
    );
    // Dispatch site uses chelis_metal_launch and chelis_metal_get_pipeline.
    assert!(
        src.contains("chelis_metal_get_pipeline("),
        "expected pipeline lookup: {src}"
    );
    assert!(
        src.contains("chelis_metal_launch("),
        "expected dispatch site: {src}"
    );
    // Output is materialized via the host-side chelis_alloc + device->host copy.
    assert!(
        src.contains("chelis_metal_device_to_host(outputs[0]->data,"),
        "expected device->host copy into outputs[0]: {src}"
    );
}

#[test]
fn m2_chained_elementwise_emits_one_kernel_per_compute_node() {
    // exp(add(a, b)) — three compute nodes (add + exp), so two kernel
    // declarations and two dispatch sites until M4 fusion lands.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(8), None);
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8), None);
    let e = dag.add_node(RiscOp::Exp, vec![s], vec_f32(8), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![e],
        vec_f32(8),
        None,
    );
    dag.add_root(stored);

    let result = codegen_metal(&dag, "chain");
    let src = &result.mm_source;
    assert!(
        !src.contains("M1 fallback stub"),
        "M2 should emit real code: {src}"
    );

    let kernel_void_count = src.matches("kernel void").count();
    assert_eq!(
        kernel_void_count, 2,
        "expected 2 kernels for unfused (add, exp); got {kernel_void_count}: {src}"
    );

    // Both kernels exist as raw-string literals (one per node).
    let pso_lookup_count = src.matches("chelis_metal_get_pipeline(").count();
    assert_eq!(
        pso_lookup_count, 2,
        "expected 2 pipeline lookups (one per kernel); got {pso_lookup_count}: {src}"
    );

    // exp body uses MSL's overloaded `exp` (not `expf`).
    assert!(
        src.contains("out[tid] = exp(a[tid]);"),
        "expected MSL `exp(a[tid])` body: {src}"
    );
}

#[test]
fn m2_unary_neg_uses_prefix_op_not_function() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![n],
        vec_f32(4),
        None,
    );
    dag.add_root(stored);

    let result = codegen_metal(&dag, "negf");
    assert!(
        result.mm_source.contains("out[tid] = -a[tid];"),
        "neg should be prefix `-`, not `neg(...)`: {}",
        result.mm_source
    );
}

// ===========================================================================
// M4: full-axis rank-1 reductions to scalar (sum, max, min). Threadgroup
// memory + tree reduction; single-threadgroup dispatch. Two-pass for
// n>4096 lands in M4.next via two-pass reduction.
// ===========================================================================

fn build_reduce_dag(reduce_op: RiscOp, n: usize) -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(n), None);
    let r = dag.add_node(reduce_op, vec![a], TensorType::scalar_f32(), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![r],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(stored);
    dag
}

#[test]
fn m4_sum_reduction_emits_threadgroup_memory_and_barriers() {
    let dag = build_reduce_dag(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        256,
    );
    let result = codegen_metal(&dag, "sumv");
    let src = &result.mm_source;
    assert!(
        !src.contains("M1 fallback stub"),
        "M4 should emit a real reduction kernel for Sum: {src}"
    );

    // Reduction kernel hallmarks.
    assert!(
        src.contains("threadgroup float shared[256]"),
        "expected threadgroup-memory scratch buffer: {src}"
    );
    assert!(
        src.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "expected at least one threadgroup_barrier: {src}"
    );
    // Tree reduction loop shape — uses the fixed power-of-2 TG_SIZE
    // constant (was tg_size from threads_per_threadgroup; the kernel was
    // changed to a compile-time TG_SIZE so the halving loop terminates
    // exactly even when n is not a power of two).
    assert!(
        src.contains("for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1)"),
        "expected tree-reduction loop with compile-time TG_SIZE: {src}"
    );
    assert!(
        src.contains("const uint TG_SIZE = 256;"),
        "expected fixed power-of-2 TG_SIZE=256: {src}"
    );
    // Sum identity is 0.
    assert!(
        src.contains("float acc = 0.0f;"),
        "expected sum identity 0.0f: {src}"
    );
    // Combine step uses + (not max/min).
    assert!(
        src.contains("acc = (acc) + (input[lid + i]);"),
        "expected sum combine `acc + input[...]`: {src}"
    );
}

#[test]
fn m4_max_reduction_uses_neg_infinity_identity() {
    let dag = build_reduce_dag(RiscOp::MaxReduce { axis: 0 }, 256);
    let result = codegen_metal(&dag, "maxv");
    let src = &result.mm_source;
    assert!(
        !src.contains("M1 fallback stub"),
        "M4 should emit a real reduction kernel for Max: {src}"
    );
    assert!(
        src.contains("float acc = -INFINITY;"),
        "expected max identity -INFINITY: {src}"
    );
    // Combine via MSL `max(...)`.
    assert!(
        src.contains("acc = max(acc, input[lid + i]);"),
        "expected `max(acc, input[...])` combine: {src}"
    );
}

#[test]
fn m4_min_reduction_uses_positive_infinity_identity() {
    let dag = build_reduce_dag(RiscOp::MinReduce { axis: 0 }, 256);
    let result = codegen_metal(&dag, "minv");
    let src = &result.mm_source;
    assert!(
        src.contains("float acc = INFINITY;"),
        "expected min identity INFINITY: {src}"
    );
    assert!(
        src.contains("acc = min(acc, input[lid + i]);"),
        "expected `min(acc, input[...])` combine: {src}"
    );
}

#[test]
fn m4_oversize_reduction_falls_through_to_stub_until_two_pass_lands() {
    // n>4096 exceeds the single-threadgroup wrap-loop limit. Two-pass
    // reduction lands in a follow-up phase; until then, fall through to
    // the stub so we never silently emit a wrong reduction.
    let dag = build_reduce_dag(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        4097,
    );
    let result = codegen_metal(&dag, "big");
    let src = &result.mm_source;
    assert!(
        src.contains("M1 fallback stub"),
        "M4 should fall through to stub for n>4096 until two-pass lands. Source:\n{src}"
    );
}

#[test]
fn m2_root_without_store_writes_output_back() {
    // Regression test for the red-team M0-M4 defect: the CLI's regular
    // lowering path emits a DAG with roots but NO `Store` nodes. The
    // emitter must synthesize a write-back to outputs[idx] in that case,
    // not silently skip the materialization.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(8), None);
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8), None);
    dag.add_root(s); // root, but no Store

    let result = codegen_metal(&dag, "f");
    let src = &result.mm_source;

    // Output labels register at least one synthetic root output.
    assert!(
        result.output_labels.iter().any(|l| l.starts_with("root")),
        "expected at least one synthetic `rootN` output label for a no-Store root: {:?}",
        result.output_labels
    );
    // The function MUST write outputs[0]. Without this, the C ABI is
    // silently violated and downstream callers see uninitialized output.
    assert!(
        src.contains("outputs[0] = chelis_alloc(") && src.contains("outputs[0]->data,"),
        "no-Store root must write outputs[0] via chelis_alloc + device->host. Source:\n{src}"
    );
    // And the chelis_metal_device_to_host call must reference outputs[0].
    assert!(
        src.contains("chelis_metal_device_to_host(outputs[0]->data,"),
        "no-Store root must call chelis_metal_device_to_host into outputs[0]: {src}"
    );
}

#[test]
fn m4_reduce_root_emits_rank_zero_scalar_alloc() {
    // Reduction returns a scalar. The output_specs+root_writeback path
    // must emit `chelis_alloc(0, NULL, CHELIS_F32)` for a true rank-0
    // scalar — not `chelis_alloc(1, {1}, CHELIS_F32)` which would lie
    // about ndim to the host.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(256),
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

    let result = codegen_metal(&dag, "sumv");
    let src = &result.mm_source;
    assert!(
        src.contains("outputs[0] = chelis_alloc(0, NULL, CHELIS_F32);"),
        "scalar reduce output must allocate as rank-0 (ndim=0, NULL shape): {src}"
    );
}

// ===========================================================================
// M5/WS-M1: matmul dispatch. The expand+mul+sum(axis=1) subgraph is
// detected and folded; per spec/04-type-system.md §1.1.3 the dispatch
// path now depends on operand precision:
//
//   * f32, f16    -> MPSMatrixMultiplication (`chelis_metal_mps_gemm_*`)
//   * bf16        -> the 16x16 tiled MSL kernel (parameterized over
//                    bfloat under `#if __METAL_VERSION__ >= 320`)
//   * integer     -> rejected at the F1 codegen guard, mirroring the
//                    type-checker's §5.7.2 rejection
//
// HIP's hipBLAS detector is the structural peer; the Metal dispatch
// table differs only in routing f32/f16 through MPS instead of
// hipBLAS.
// ===========================================================================

fn mat_f32(r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn build_matmul_dag(m: usize, k: usize, n: usize) -> Dag {
    use chelis_ir::dag::DimExpr;
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(m, k),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(k, n),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(n),
        },
        vec![a],
        tensor3_f32(m, k, n),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(m),
        },
        vec![b],
        tensor3_f32(m, k, n),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(m, k, n), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(m, n),
        None,
    );
    dag.add_root(sum);
    dag
}

// Per-dtype matmul routing (f32/f16 -> MPS, bf16 -> tiled MSL) is the
// canonical responsibility of `dtype_matrix.rs::matmul_{f32,f16,bf16}_*`.
// The former `wsm1_{f32,f16,bf16}_matmul_routes_to_*` tests here asserted
// the same routing targets and were removed as duplicates. The two tests
// below stay because they cover non-routing facets (subgraph folding and
// the M/N/K uniform packing) that dtype_matrix does not.

#[test]
fn wsm1_matmul_skips_expand_and_mul_intermediates() {
    let dag = build_matmul_dag(32, 16, 16);
    let result = codegen_metal(&dag, "mm");
    let src = &result.mm_source;
    // Expand and Mul nodes are folded into the matmul; no standalone
    // elementwise kernel should be emitted for them.
    assert!(
        !src.contains("k_binary_") || !src.contains("k_unary_"),
        "matmul subgraph should not emit standalone unary/binary kernels: {src}"
    );
    // Exactly one MPS dispatch (no tiled kernel for f32 anymore).
    let mps_count = src.matches("chelis_metal_mps_gemm_f32(").count();
    assert_eq!(
        mps_count, 1,
        "expected exactly 1 MPS f32 dispatch, got {mps_count}: {src}"
    );
}

#[test]
fn wsm1_bf16_matmul_passes_m_n_k_uniforms() {
    // The bf16 path keeps the packed M/N/K uniforms because it emits
    // the parameterized tiled kernel. The f32/f16 paths bypass uniforms
    // entirely (MPS takes M/N/K as helper args).
    let dag = build_matmul_dag_prec(8, 8, 8, Prim::Bf16);
    let result = codegen_metal(&dag, "mm_bf16");
    let src = &result.mm_source;
    assert!(
        src.contains("struct { uint32_t M, N, K; }"),
        "bf16 matmul should carry packed M/N/K uniforms: {src}"
    );
    assert!(
        src.contains("8u, 8u, 8u"),
        "bf16 matmul should pass M=8, N=8, K=8: {src}"
    );
}

fn mat_prec(r: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: p,
    }
}

fn tensor3_prec(a: usize, b: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: p,
    }
}

fn build_matmul_dag_prec(m: usize, k: usize, n: usize, prec: Prim) -> Dag {
    use chelis_ir::dag::DimExpr;
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_prec(m, k, prec),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_prec(k, n, prec),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(n),
        },
        vec![a],
        tensor3_prec(m, k, n, prec),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(m),
        },
        vec![b],
        tensor3_prec(m, k, n, prec),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_prec(m, k, n, prec), None);
    // Per spec §5.7.1 the bf16/f16 sum accumulator is f32.
    let acc = match prec {
        Prim::F16 | Prim::Bf16 => Prim::F32,
        other => other,
    };
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: acc,
        },
        vec![mul],
        mat_prec(m, n, prec),
        None,
    );
    dag.add_root(sum);
    dag
}

#[test]
fn m4_axis_nonzero_reduction_falls_through_to_stub() {
    // The M4 first cut handles full-axis reduce only (axis=0 on rank-1).
    // axis-nonzero falls through to the stub.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(64), None);
    // Construct an axis=1 sum even though our input is rank-1; this
    // simulates a partial-axis reduction the M4 first cut should reject.
    let r = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![r],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(stored);

    let result = codegen_metal(&dag, "axisone");
    assert!(
        result.mm_source.contains("M1 fallback stub"),
        "axis-nonzero reduce should fall through to stub: {}",
        result.mm_source
    );
}
