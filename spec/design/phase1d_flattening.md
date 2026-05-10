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
- contiguous `f32` matmul subgraphs with rank ≥ 2 specialize to hipBLAS-backed helpers:
  rank-2 uses `chelis_hipblas_sgemm_row_major(...)`, and batched/symbolic matmul uses
  `chelis_hipblas_sgemm_batched_row_major(...)`
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
Only operands with contiguous trailing matrix slices take the hipBLAS path. Rank ≥ 3
batched matmul is supported by looping over batch slices in the runtime helper; using
`hipblasSgemmStridedBatched` directly for uniformly strided batches remains a performance
follow-up. Non-contiguous matmul-shaped DAGs remain correct via the generic reduction
fallback.

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
continues through BLAS. On the HIP backend the current helper expects specialization to
have proved contiguous matrix slices; a failed runtime check aborts rather than silently
launching the generic lowering. This keeps the HIP ABI narrow until a GPU make-contiguous
fallback is designed.

### Acceptance Oracle

Phase 1d is complete when this manual GPU oracle passes on a HIP-capable machine:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Supporting evidence:

- `cargo test -p chelis-backend-hip --test codegen_structure`
- `cargo test -p chelis-backend-hip --test redteam_adversarial`
- `cargo test -p chelis-cli --test cli`

### Test Strategy

- [x] tiny/small/large segmented kernels are selected for the expected axis-size ranges
- [x] staged scalar reductions emit inline scratch buffers and extend the peak-memory estimate
- [x] contiguous rank-2 and batched/symbolic matmul patterns emit hipBLAS helper calls
  and surface `-lhipblas`
- [x] non-contiguous matmul-shaped DAGs stay on the generic reduction path
- [x] manual GPU correctness covers segmented reductions, staged scalar reduction, hipBLAS matmul, and the non-contiguous fallback

### Deferred Follow-Ups

- benchmark the optimized reduction and hipBLAS paths against the C backend and PyTorch (Phase 1e)
- add monotonic-threshold autotuning once the kernel selection surface is stable
- use `hipblasSgemmStridedBatched` for uniformly strided batched matmul layouts
- consider LMAD-style memory-layout reasoning only if profiling shows coalescing/layout is the next bottleneck
