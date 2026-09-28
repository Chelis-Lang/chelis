## Phase 1d: Optimized Reductions and hipBLAS

**Goal:** Improve GPU utilization for reductions and matrix-shaped contraction patterns
without changing Chelis's regular tensor execution model.

### Shipped Scope

Phase 1d ships inside the HIP backend code generator and runtime header:

- segmented reductions use a single runtime-sized axis-specific kernel on the generic path
- fused elementwise→reduction kernels reuse that same runtime-sized reduction kernel
- scalar contiguous reductions use a staged scratch-chain reduction
- staged partial buffers are allocated/freed inline in generated host code and are **not**
  routed through the Phase 1c slot planner
- the HIP peak-memory reporting includes the worst single staged scratch chain
- supported matmul subgraphs with rank ≥ 2 specialize to hipBLAS-backed helpers:
  rank-2 uses the typed row-major GEMM helper for `f32`, `f64`, `bf16`, and `f16`;
  HIP preparation materializes each BLAS operand into owned contiguous storage;
  eligible `f32`/`f64` batched matmul then uses the typed strided-batched helper;
  remaining batched/symbolic forms fall back to the typed per-batch helper loop
- non-contiguous matmul-shaped DAGs fall back to the generic reduction path

Phase 1d still does **not** implement flattening for irregular nested parallelism, autotuned
kernel threshold selection, or internal hipBLAS workspace estimation.

### Design Notes

**No flattening for Phase 1.**
Chelis tensors are regular. The shipped Phase 1d path keeps the existing tensor-stride model
and focuses on better reduction kernels rather than flattening transformations.

**Segmented reductions are the primary optimization target.**
Softmax-style workloads reduce rows or row-like segments repeatedly, so the backend keeps a
dedicated runtime-sized segmented reduction kernel instead of using the Phase 1a naive loop for
all cases.

**Scalar staged reductions are intentionally narrow.**
The staged scratch-chain path is used only for safe scalar contiguous reductions. Row-wise and
other multi-output reductions stay on the segmented path.

**hipBLAS specialization is deliberately constrained.**
Recognized BLAS matmul operands are explicitly materialized before ownership lowering.
The shared storage planner therefore accounts for their temporary bytes and gives the
emitter dense owned descriptors even when the source was a broadcast or another
non-contiguous view. Rank ≥ 3 `f32`/`f64` batched matmul uses the typed
`hipblasSgemmStridedBatched` helper when matrix dimensions are concrete, batch dimensions
are simple runtime dimensions, and the prepared operand/result shapes match the matmul
contract. The per-batch helper loop remains the fallback for forms outside that plan,
including symbolic matrix dimensions. A non-contiguous decomposed matmul shape that is
not recognized as `BlasMatmul` remains correct via the generic reduction path. The
realize-then-strided invariant is locked by
`crates/chelis-backend-hip/tests/strided_batched_dispatch.rs` and
`crates/chelis-backend-hip/tests/codegen_structure.rs`, with numerical BLAS coverage in
the HIP `gpu_correctness` manual gate.

**Runtime-sized BLAS dimensions.**
Symbolic dimensions are not required to be compile-time constants for BLAS. Generated
host code uses the existing symbolic preamble bindings from input tensor metadata and
passes those runtime integers as `m`, `n`, `k`, and batch-loop bounds. V1 handles symbols
bound by load tensor shapes and shape-preserving transformations that preserve those
bindings; it does not introduce a new symbolic solver.

**Contiguity guard semantics.**
The runtime guard is a stride comparison on the trailing matrix slice:
`stride[-1] == 1` and `stride[-2] == trailing_column_count`. The check is emitted once
per specialized matmul callsite, before any C batch loop. On the C backend a failed
operand check materializes a contiguous copy with `chelis_contiguous(...)` and then
continues through BLAS. On the HIP backend `prepare_dag_for_codegen` inserts explicit
`Realize` operations before ownership lowering; generated materialization kernels copy
logical element order into planned dense device slots, and BLAS receives those prepared
descriptors rather than the original view descriptors.

### Acceptance Oracle

Phase 1d is complete when this manual GPU oracle passes on a HIP-capable machine:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Supporting evidence:

- `cargo test -p chelis-backend-hip --test strided_batched_dispatch`
- `cargo test -p chelis-backend-hip --test codegen_structure`
- `cargo test -p chelis-backend-hip --test codegen_adversarial`
- `cargo test -p chelis-cli --test cli`

### Test Strategy

- [x] tiny/small/large segmented kernels are selected for the expected axis-size ranges
- [x] staged scalar reductions emit inline scratch buffers and extend the peak-memory estimate
- [x] recognized rank-2 and batched/symbolic matmul patterns materialize HIP BLAS
  operands, emit the eligible typed helper calls, and surface `-lhipblas`
- [x] non-contiguous matmul-shaped DAGs stay on the generic reduction path
- [x] manual GPU correctness covers segmented reductions, staged scalar reduction, hipBLAS matmul, and the non-contiguous fallback

### Deferred Follow-Ups

- add monotonic-threshold autotuning once the kernel selection surface is stable
- consider LMAD-style memory-layout reasoning only if profiling shows coalescing/layout is the next bottleneck
