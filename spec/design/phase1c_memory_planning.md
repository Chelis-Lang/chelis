## Phase 1c: Memory Planning for GPU

**Goal:** Minimize GPU memory usage and host↔device transfers.

### What the Agent Builds

**Buffer lifetime analysis (`chelis-backend-hip/src/memory.rs`):**

Walk the DAG and compute, for each tensor:
- Birth: the node that first produces this tensor
- Death: the last node that consumes this tensor
- Size: product of shape × sizeof(dtype)

**Buffer reuse:**

When a buffer's lifetime ends, its GPU memory can be reused for a later tensor of the same size (or smaller, with internal fragmentation). This is a graph coloring problem on the interference graph of buffer lifetimes.

For Phase 1c, use a simple greedy algorithm: process buffers in birth order, reuse the first available dead buffer of sufficient size. This is not optimal but is correct and simple. Optimal allocation (minimum total memory) is NP-hard in general but tractable for the DAG sizes Chelis produces.

**Host↔device transfer minimization:**

- **Input tensors:** Transfer to device once at the start. Don't re-transfer between kernel launches.
- **Intermediate tensors:** Live entirely on device. Never transfer to host unless explicitly `realize()`d.
- **Output tensors:** Transfer to host once at the end.
- **Parameter tensors (for training):** Live on device for the entire training loop. Only transfer to host for checkpointing.

The transfer plan is a list of `(tensor, direction, timing)` triples emitted alongside the kernel launch plan.

**Memory coalescing and layout transformations:**

`permute` and `reshape` are currently metadata-only operations (stride reordering, shape change). On GPU, this means data stays in its original memory layout and kernels read via strided indexing. That's correct but can be slow — non-coalesced memory access on GPU is a significant performance penalty. The pragmatic Phase 1 approach: always use strided indexing (correct, simple), profile, and add physical transposition only where non-coalesced access is measured as a bottleneck. Don't try to build a full memory descriptor system for Phase 1. Track coalescing as a profiling target in Phase 1e.

**Memory budget mode (stretch goal):**

Given a VRAM budget, estimate peak memory usage from the buffer plan. If it exceeds the budget, insert recomputation points — trade compute for memory by recomputing an intermediate result instead of keeping it in VRAM. This is activation checkpointing. For Phase 1c, implement the estimation and warning ("this model needs N MB VRAM, you have M MB") but not automatic checkpointing insertion.

### Test Strategy (~8 tests)

- [ ] Buffer reuse: a DAG with non-overlapping lifetimes reuses memory (inspect allocation count)
- [ ] Buffer reuse: a DAG with overlapping lifetimes does NOT reuse (verify no aliasing)
- [ ] No host↔device transfers between consecutive kernels (intermediates stay on device)
- [ ] Input transferred to device exactly once
- [ ] Output transferred to host exactly once
- [ ] Peak memory estimation is within 10% of actual (measure with `hipMemGetInfo`)
- [ ] No memory leaks: alloc count == free count in generated code
- [ ] MNIST model: peak VRAM usage is reasonable (not allocating per-node without reuse)

### Execution Strategy

```
Commit 1: Buffer lifetime analysis
Commit 2: Greedy buffer reuse
Commit 3: Transfer planning (minimize host↔device copies)
Commit 4: Peak memory estimation + warning
Commit 5: Tests
```
