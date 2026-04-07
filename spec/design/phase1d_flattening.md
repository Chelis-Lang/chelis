## Phase 1d: Optimized Reductions and hipBLAS

**Goal:** Improve GPU utilization for reductions and matrix-shaped contraction patterns
without changing Chelis's regular tensor execution model.

### Shipped Scope

Phase 1d ships inside the HIP backend code generator and runtime header:

- segmented reductions use three strategies keyed by static segment size
  - tiny: `axis_size <= 8`
  - small: `9..=64`
  - large: `>= 65`
- small and large strategies use shared-memory block cooperation
- fused elementwise→reduction kernels reuse the same tiny/small/large split
- scalar contiguous reductions use a staged scratch-chain reduction
- staged partial buffers are allocated/freed inline in generated host code and are **not**
  routed through the Phase 1c slot planner
- `peak_device_bytes_estimate` includes the worst single staged scratch chain
- contiguous rank-2 `f32` matmul subgraphs (`expand + mul + sum(axis=1)`) specialize to
  hipBLAS via `chelis_hipblas_sgemm_row_major(...)`
- non-contiguous matmul-shaped DAGs fall back to the generic reduction path

Phase 1d still does **not** implement flattening for irregular nested parallelism, autotuned
kernel threshold selection, or internal hipBLAS workspace estimation.

### Design Notes

**No flattening for Phase 1.**
Chelis tensors are regular. The shipped Phase 1d path keeps the existing tensor-stride model
and focuses on better reduction kernels rather than flattening transformations.

**Segmented reductions are the primary optimization target.**
Softmax-style workloads reduce rows or row-like segments repeatedly, so the backend now picks
between tiny/small/large segmented kernels instead of using the Phase 1a naive loop for all
cases.

**Scalar staged reductions are intentionally narrow.**
The staged scratch-chain path is used only for safe scalar contiguous reductions. Row-wise and
other multi-output reductions stay on the segmented path.

**hipBLAS specialization is deliberately constrained.**
Only statically contiguous rank-2 `f32` operands take the hipBLAS path. This avoids inventing a
new GPU "make contiguous" runtime surface in Phase 1d. Non-contiguous matmul-shaped DAGs remain
correct via the generic reduction fallback.

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
- [x] contiguous matmul patterns emit the hipBLAS helper call and surface `-lhipblas`
- [x] non-contiguous matmul-shaped DAGs stay on the generic reduction path
- [x] manual GPU correctness covers segmented reductions, staged scalar reduction, hipBLAS matmul, and the non-contiguous fallback

### Deferred Follow-Ups

- benchmark the optimized reduction and hipBLAS paths against the C backend and PyTorch (Phase 1e)
- add monotonic-threshold autotuning once the kernel selection surface is stable
- consider LMAD-style memory-layout reasoning only if profiling shows coalescing/layout is the next bottleneck
