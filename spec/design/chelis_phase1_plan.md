# Phase 1: Futhark-Style GPU Backend — Expanded Plan

## Context

Phase 1 extends the Chelis compiler to target GPUs via HIP. The architecture follows Futhark: the compiler emits C host code with embedded HIP kernel strings. At runtime, `hiprtc` JIT-compiles the kernel strings and dispatches them to whatever GPU is present. HIP runs natively on AMD GPUs and targets NVIDIA GPUs via HIP's CUDA translation layer — one kernel emitter, both vendors.

**Prerequisite:** Phase 0 complete. MNIST trains on CPU. Full pipeline proven (Surf → Deep → type check → lower → DAG → grad → eval/codegen → train). All tests pass. CLI works.

**Deliverable:** The same MNIST program (and harder models — transformer block, CNN) compiles to GPU via `chelis build --target hip` and produces correct results. Performance within 2-5x of PyTorch on equivalent models (proving the architecture works, not winning benchmarks).

**What does NOT change:** The RISC DAG, the type checker, the Surf/Deep parsers, the desugarer, the AD engine. Phase 1 adds a new backend alongside the existing C backend. Both coexist. The C backend remains the test oracle — every GPU result must match it numerically.

**Implemented boundary note:** The currently shipped surface reaches Phase 1a plus the
Phase 1b AD-ordering helper. `grad` from Phase 0g still runs on the unfused DAG, and
`chelis-ir::grad_then_fuse` now re-fuses the resulting forward+backward graph. Full
fusion/codegen integration beyond that helper, optimized reductions (1d), and later
memory/layout refinements are still planned work.

---

## Crate Structure

Phase 1 adds one new crate and modifies two existing ones:

```
crates/
  chelis-backend-hip/          ← NEW: HIP GPU backend
    Cargo.toml
    src/
      lib.rs                   — pub fn codegen_hip(dag, name, opts) → HipCodegenResult
      emit.rs                  — RISC DAG → C host code + HIP kernel strings
      kernels.rs               — Kernel string templates for each op pattern
      fuse.rs                  — Fusion pass: merge adjacent ops into single kernels
      memory.rs                — GPU buffer lifetime analysis, host↔device transfer planning
      launch.rs                — Kernel launch configuration (grid size, block size)
    runtime/
      chelis_hip_runtime.h     — GPU tensor struct, hipMalloc/hipFree, host↔device transfer
      chelis_hip_runtime.c     — Runtime implementation
    tests/
      kernel_correctness.rs    — Each op: GPU output matches CPU backend (within tolerance)
      fusion_correctness.rs    — Fused vs unfused: identical results
      memory_tests.rs          — No leaks, no double-free, transfer minimization
      benchmark.rs             — Wall-clock comparisons against CPU backend

  chelis-ir/                   ← MODIFIED: fusion pass, symbolic dimensions
    src/
      fuse.rs                  — NEW: DAG-to-DAG fusion rewrite (shared by HIP backend)
      optimize.rs              — existing DCE/CSE/constant folding (unchanged)

  chelis-cli/                  ← MODIFIED: --target hip flag
    src/
      main.rs                  — add --target flag to build command
```

The fusion pass lives in `chelis-ir` (not the HIP backend) because fusion is a DAG optimization that future backends (StableHLO, FX) would also use. The HIP backend consumes the fused DAG.

## Accepted Update Tracking

| Proposal | Disposition | Owning area | Touches implemented surface? | Current boundary impacted |
|---|---|---|---|---|
| GPU failure variable for bounds/debug checking | Adopt now | Phase 1a runtime | Yes | 1a |
| AD before fusion, then re-fuse | Adopt now | Transform ordering + Phase 1b | Yes | 0g semantics now, 1b pipeline later |
| Segmented reduction strategies | Adopt now | Phase 1d | No | future 1d |
| Lightweight uniqueness over full linear types | Defer note | Phase 2b | No | future 2b |
| Recomputation-based AD for GPU execution | Defer note | Phase 2 AD refinement | No | future |
| Monotonicity-based autotuning | Defer note | Phase 1d+ kernel selection | No | future |
| LMAD-based memory analysis | Defer note | Phase 1c+ memory/layout optimization | No | future |
| Rank polymorphism via ILP (AUTOMAP) | Research note | Phase 3 | No | future 3 |

---

## Sub-Phase Plans

| Sub-phase | Doc | Summary |
|---|---|---|
| Symbolic Dimensions | [phase1_symbolic_dims.md](phase1_symbolic_dims.md) | IR supports `Concrete \| Symbolic` dims; do first to unblock GPU backend |
| 1a: Kernel Codegen | [phase1a_kernel_codegen.md](phase1a_kernel_codegen.md) | Single RISC op → HIP kernel, runs on GPU, correct output |
| 1b: Fusion | [phase1b_fusion.md](phase1b_fusion.md) | Adjacent DAG nodes → single kernel launches |
| 1c: Memory Planning | [phase1c_memory_planning.md](phase1c_memory_planning.md) | Buffer reuse, minimize host↔device transfers |
| 1d: Optimized Reductions + hipBLAS | [phase1d_flattening.md](phase1d_flattening.md) | Optimized reductions, hipBLAS, thread block sizing (no flattening needed) |
| 1e: Benchmarks | [phase1e_benchmarks.md](phase1e_benchmarks.md) | MNIST + transformer + CNN on GPU, perf characterization |
| 1f: Executable Grammar | [phase1f_executable_grammar.md](phase1f_executable_grammar.md) | PEG-based `chelis validate` conformance tool |

---

## Red Team Checkpoint: Phase 1

After all sub-phases, before declaring Phase 1 complete:

- [ ] Every RISC op: GPU output matches CPU output within 1e-5
- [ ] Fusion preserves correctness in all tested cases. Adversarial fusion tests: multi-consumer nodes, reduction boundaries, realize() barriers.
- [ ] Memory planning: peak VRAM for MNIST is reasonable (not 10x the tensor sizes). No leaks.
- [ ] MNIST trains on GPU to >90% accuracy
- [ ] At least one model beyond MNIST (LeNet or transformer block) runs correctly on GPU
- [ ] GPU is faster than CPU for batch ≥ 32 (if not, profile and explain why)
- [ ] Wall-clock within 2-5x of PyTorch (if not, profile and document bottlenecks)
- [ ] `chelis validate` agrees with the compiler on 100% of the spec test suite
- [ ] No regression in CPU backend (all Phase 0 tests still pass)
- [ ] Symbolic dimensions work in both backends

---

## Overall Execution Order

### Phase 1 GPU Pipeline Ordering

For the GPU path, the intended ordering is:

```text
lower -> optimize -> grad -> optimize again -> fuse -> codegen
```

This makes `grad` operate on the pre-fusion RISC DAG, then lets the same fusion pass run
over both the forward and backward graphs. Do not fuse before `grad`; otherwise the
adjoint rules would need to understand fused kernel nodes directly.

The sub-phases have dependencies:

```
1a (kernel codegen) ──→ 1b (fusion) ──→ 1e (benchmarks)
        │                                     ↑
        └──→ 1c (memory planning) ────────────┘
                                              ↑
1d (optimized reductions, hipBLAS) ───────────┘

1f (executable grammar) — independent, can be done anytime

Symbolic dimensions — do first or early in 1a
```

**Recommended order:**
1. Symbolic dimensions (unblocks everything)
2. 1a: kernel codegen (the foundation)
3. 1c: memory planning (needed before real models)
4. 1b: fusion (the big optimization)
5. 1d: optimized reductions + hipBLAS (performance)
6. 1e: benchmarks (prove it works)
7. 1f: executable grammar (independent, can parallel with anything)

**Total estimated scope:** 1a-1e are the GPU backend proper. Each sub-phase is a significant implementation effort. 1f is independent tooling work. Budget Phase 1 as the largest single phase in the project.
