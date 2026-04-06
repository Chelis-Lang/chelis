## Phase 1d: Optimized Reductions and hipBLAS

**Goal:** Optimize GPU utilization for reductions and matrix operations.

### What the Agent Builds

**No flattening needed for Phase 1.** Futhark's incremental flattening addresses irregular nested parallelism (arrays of arrays with varying inner sizes). Chelis's tensor model has regular, predictable parallelism: batch dimension → grid blocks, inner dimensions → threads. Full flattening can actually reduce performance by destroying data locality. Revisit if irregular parallelism is added later.

**Optimized reductions:**

The Phase 1a naive reduction (one thread per output element, inner loop over reduction axis) is correct but slow. Replace with the standard two-phase GPU reduction:

Phase 1: Each thread block cooperatively reduces a chunk of the reduction axis using shared memory. Produces one partial result per block.

Phase 2: A second kernel reduces the partial results.

This is a standard GPU programming pattern, well-documented in the NVIDIA/AMD reduction tutorials. The agent can follow the pattern directly.

**Thread block sizing:**

Phase 1a uses `blockDim = 256` for everything. This is reasonable for elementwise ops but suboptimal for reductions and operations with specific memory access patterns.

- Elementwise: 256 threads/block, coalesced memory access (contiguous threads access contiguous memory)
- Reduction: block size = min(axis_size, 256), rounded down to power of 2 for efficient butterfly reduction
- Matmul (BLAS): don't emit a kernel — use `hipblas` (the HIP equivalent of cuBLAS). Pattern-match the expand+mul+sum DAG subgraph and emit a `hipblasSgemm` call instead. This mirrors the CPU backend's BLAS integration.

**hipBLAS for matmul:**

Same pattern-matching as the CPU backend's BLAS integration, but targeting `hipblasSgemm` instead of `cblas_sgemm`:

```c
hipblasHandle_t handle;
hipblasCreate(&handle);
hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
             n, m, k, &alpha, B_d, n, A_d, k, &beta, C_d, n);
hipblasDestroy(handle);
```

Note: hipBLAS uses column-major by default. The argument order differs from cblas_sgemm. The agent must handle the row-major → column-major translation carefully (swap A and B, swap m and n).

### Test Strategy (~8 tests)

- [ ] Optimized reduction matches naive reduction output
- [ ] Optimized reduction is measurably faster than naive on large tensors (>100K elements)
- [ ] hipBLAS matmul matches CPU BLAS matmul (within tolerance)
- [ ] hipBLAS path is taken for matmul (check generated C for `hipblasSgemm`)
- [ ] Block size selection: elementwise uses 256, reduction uses power-of-2 ≤ axis_size
- [ ] Memory access is coalesced for elementwise ops (can verify via profiler or structurally in generated code)

### Execution Strategy

```
Commit 1: Optimized two-phase reduction kernels
Commit 2: hipBLAS integration for matmul
Commit 3: Thread block sizing heuristics
Commit 4: Performance profiling + correctness tests
```
