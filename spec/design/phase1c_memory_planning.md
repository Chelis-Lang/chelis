## Phase 1c: Memory Planning for GPU

**Goal:** Minimize GPU memory usage and host↔device transfers.

### Shipped Scope

Phase 1c is a codegen/runtime-planning change inside the HIP backend. The shipped surface is:

- a greedy device-slot planner in `crates/chelis-backend-hip/src/memory.rs`
- planner-driven HIP emission that reuses backing allocations across non-overlapping lifetimes
- unique input copies transferred once, with repeated `Load(name)` nodes aliasing that copy
- metadata-only wrappers for movement ops and `store`
- a structured peak-memory formula on `HipCodegenResult`, plus an optional concrete estimate when all slot sizes are static
- a human-readable formula line in `chelis build --target hip`

Phase 1c does **not** add a persistent GPU execution API, automatic checkpoint insertion, or
runtime budget comparison flags. The estimate is surfaced to the user; policy decisions based on
that estimate remain later work.

### What the Agent Builds

**Buffer lifetime analysis (`chelis-backend-hip/src/memory.rs`):**

Walk the DAG and compute, for each tensor:
- Birth: the node that first produces this tensor
- Death: the last node that consumes this tensor
- Size: product of shape × sizeof(dtype)

**Buffer reuse:**

When a buffer's lifetime ends, its GPU memory can be reused for a later tensor of the same size (or smaller, with internal fragmentation). This is a graph coloring problem on the interference graph of buffer lifetimes.

For Phase 1c, use a simple greedy algorithm: process buffers in birth order, reuse the first available dead buffer of sufficient size. For symbolic sizes, reuse conservatively only when the size expressions are identical. This is not optimal but is correct and simple. Optimal allocation (minimum total memory) is NP-hard in general but tractable for the DAG sizes Chelis produces.

**Host↔device transfer minimization:**

- **Input tensors:** Transfer to device once at the start. Don't re-transfer between kernel launches.
- **Intermediate tensors:** Live entirely on device. Never transfer to host unless explicitly `realize()`d.
- **Output tensors:** Transfer to host once at the end.
- **Parameter tensors (for training):** Live on device for the entire training loop. Only transfer to host for checkpointing.

The transfer plan is a list of `(tensor, direction, timing)` triples emitted alongside the kernel launch plan.

**Memory coalescing and layout transformations:**

`permute` and `reshape` are currently metadata-only operations (stride reordering, shape change). On GPU, this means data stays in its original memory layout and kernels read via strided indexing. That's correct but can be slow — non-coalesced memory access on GPU is a significant performance penalty. The pragmatic Phase 1 approach: always use strided indexing (correct, simple), profile, and add physical transposition only where non-coalesced access is measured as a bottleneck. Don't try to build a full memory descriptor system for Phase 1.

**Memory budget mode (stretch goal):**

Given a VRAM budget, estimate peak memory usage from the buffer plan. If it exceeds the budget, insert recomputation points — trade compute for memory by recomputing an intermediate result instead of keeping it in VRAM. This is activation checkpointing. For Phase 1c, implement the estimation and warning ("this model needs N MB VRAM, you have M MB") but not automatic checkpointing insertion.

The shipped Phase 1c surface stops at the reporting itself. It reports peak-memory formulas
through `HipCodegenResult` and `chelis build --target hip`, plus concrete byte estimates when
all slot sizes are static, but does not compare against live free memory. Phase 1d extends that
reporting to include inline staged-reduction scratch chains; it still does not compare the result
against live device memory.

**Deferred optimization note:** LMAD-style algebraic memory-layout analysis may later help
reason about coalescing, transposes, and layout-sensitive kernel selection. That is not a
Phase 1c correctness requirement. Implement buffer lifetime analysis and reuse first; only
escalate to LMAD-style reasoning if profiling shows memory layout is the bottleneck.

### Acceptance Oracle

Phase 1c is complete when this manual GPU oracle passes on a HIP-capable machine:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Supporting evidence:

- `cargo test -p chelis-backend-hip --test codegen_structure`
- `cargo test -p chelis-backend-hip --test codegen_adversarial`
- `cargo test -p chelis-cli --test cli`

### Test Strategy

- [x] Buffer reuse: non-overlapping lifetimes reuse slots (`memory.rs` unit tests)
- [x] Buffer reuse: overlapping lifetimes stay separate (`memory.rs` unit tests)
- [x] Repeated `Load(name)` shares one device transfer (`codegen_structure.rs`)
- [x] Outputs transfer host↔device only at the function boundary (`codegen_structure.rs`)
- [x] Cleanup frees every wrapper and every backing slot exactly once (`memory.rs`, `codegen_structure.rs`, `codegen_adversarial.rs`)
- [x] Reused slots still iterate over logical tensor size, not slot capacity (`codegen_structure.rs`, `gpu_correctness.rs`)
- [x] Manual GPU correctness covers repeated-load aliasing and reused-slot execution (`gpu_correctness.rs`)
- [ ] Direct estimate-vs-`hipMemGetInfo` comparison remains future validation work if profiling shows the estimate needs tighter calibration

### Execution Strategy

```
Commit 1: Buffer lifetime analysis
Commit 2: Greedy buffer reuse
Commit 3: Transfer planning (minimize host↔device copies)
Commit 4: Peak memory estimation + warning
Commit 5: Tests
```
