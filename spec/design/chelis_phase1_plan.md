# Phase 1: Futhark-Style GPU Backend — Expanded Plan

## Context

Phase 1 extends the Chelis compiler to target GPUs via HIP. The architecture follows Futhark: the compiler emits C host code with embedded HIP kernel strings. At runtime, `hiprtc` JIT-compiles the kernel strings and dispatches them to whatever GPU is present. HIP runs natively on AMD GPUs and targets NVIDIA GPUs via HIP's CUDA translation layer — one kernel emitter, both vendors.

**Prerequisite:** Phase 0 complete. MNIST trains on CPU. Full pipeline proven (Surf → Deep → type check → lower → DAG → grad → eval/codegen → train). All tests pass. CLI works.

**Deliverable:** fixed benchmark workloads (`mnist`, `linreg`, `transformer_block`) compile to GPU via `chelis build --target hip`, run through the Phase 1e benchmark oracle, and produce recorded correctness/performance results. PyTorch remains a local/manual comparison backend through the repo `py/` environment; correctness against Chelis CPU remains primary.

**What does NOT change:** The RISC DAG, the type checker, the Surf/Deep parsers, the desugarer, the AD engine. Phase 1 adds a new backend alongside the existing C backend. Both coexist. The C backend remains the test oracle — every GPU result must match it numerically.

**Implemented boundary note:** The currently shipped surface reaches Phase 1e. `grad`
from Phase 0g still runs on the unfused DAG, and
`chelis-ir::grad_then_fuse` re-fuses the resulting forward+backward graph. The HIP
backend now reuses backing slots, deduplicates repeated input transfers, reports
estimated peak device bytes through codegen/CLI output, emits segmented reduction
strategies, uses staged scratch buffers for safe scalar reductions, and specializes
contiguous rank ≥ 2 `f32` matmul patterns to hipBLAS-backed helpers. Phase 1e now ships a fixed
benchmark oracle plus checked-in local PyTorch comparison artifacts. Phase 1f now ships
the standalone `chelis-validate` crate plus `chelis validate --surf/--deep/--desugar`
CLI modes and a dedicated conformance oracle at
`cargo test -p chelis-e2e --test phase1f_validate`.

**Status after red-team review:** the shipped fixed-workload Phase 1 deliverable is met.
The Phase 1e benchmark models (`mnist`, `linreg`, `transformer_block`) compile and run
correctly on both backends, and the validator work from 1f is shipped. The remaining
backend gaps are explicit carried-forward limitations rather than hidden blockers:

- HIP does not yet implement `pad` / `shrink`; none of the current Phase 1 benchmark models uses those ops
- symbolic dimensions are implemented on the stable tensor ABI in both backends; the
  supported Phase 1 surface binds symbolic names from input metadata at runtime
- `layer_norm` still requires a concrete normalized-axis extent; symbolic leading dims
  are supported, but a symbolic hidden size remains follow-up debt
- the dotted Deep module/import round-trip gap remains a documented Phase 2 parser/decompiler follow-up

These limitations should stay visible, but they do not block Phase 2 language work.

---

## Crate Structure

Phase 1 adds one new crate and modifies two existing ones:

```
crates/
  chelis-backend-hip/          ← NEW: HIP GPU backend
    Cargo.toml
    src/
      lib.rs                   — pub fn codegen_hip(dag, name) → HipCodegenResult
      emit.rs                  — RISC DAG → C host code + HIP kernel strings
      kernels.rs               — Kernel string templates for each op pattern
      memory.rs                — device slot planning, transfer dedup, cleanup emission
      launch.rs                — Kernel launch configuration (grid size, block size)
    runtime/
      chelis_hip_runtime.h     — GPU tensor/runtime helpers used by generated host code
    tests/
      codegen_structure.rs     — structural/source-emission coverage for HIP codegen
      gpu_correctness.rs       — manual HIP oracle on real GPU hardware
      codegen_adversarial.rs   — adversarial ownership, cleanup, and surface checks

  chelis-ir/                   ← MODIFIED: fusion pass, symbolic dimensions
    src/
      fuse.rs                  — DAG-to-DAG fusion rewrite shared by backends
      optimize.rs              — existing DCE/CSE/constant folding (unchanged)

  chelis-cli/                  ← MODIFIED: --target hip flag
    src/
      main.rs                  — add --target flag to build command and print HIP estimate
```

The fusion pass lives in `chelis-ir` (not the HIP backend) because fusion is a DAG optimization that future backends (StableHLO, FX) would also use. The HIP backend consumes the fused DAG. The current runtime support is header-only (`chelis_hip_runtime.h`), not a separate `.c` implementation file.

## Accepted Update Tracking

| Proposal | Disposition | Owning area | Touches implemented surface? | Current boundary impacted |
|---|---|---|---|---|
| GPU failure variable for bounds/debug checking | Adopt now | Phase 1a runtime | Yes | 1a |
| AD before fusion, then re-fuse | Adopt now | Transform ordering + Phase 1b | Yes | 0g semantics now, 1b pipeline later |
| Segmented reduction strategies | Adopt now | Phase 1d | Yes | 1d |
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
| 1e: Benchmarks | [phase1e_benchmarks.md](phase1e_benchmarks.md) | Fixed MNIST + linreg + transformer workloads, recorded perf/correctness |
| 1f: Executable Grammar | [phase1f_executable_grammar.md](phase1f_executable_grammar.md) | PEG-based `chelis validate` conformance tool |

---

## Red Team Checkpoint: Phase 1

After all sub-phases, before declaring Phase 1 complete:

This is the ideal full-backend checkpoint, not a claim that every item below is already
true on every possible future model. For the shipped fixed-workload deliverable, the
benchmark-model path is proven; the specific carried-forward limitations above remain
documented debt.

- [ ] Every RISC op: GPU output matches CPU output within the documented manual HIP oracle
- [ ] Fusion preserves correctness in all tested cases. Adversarial fusion tests: multi-consumer nodes, reduction boundaries, realize() barriers.
- [ ] Memory planning: peak VRAM estimates are surfaced and cleanup remains leak-free
- [ ] Benchmark workloads run correctly on GPU and are recorded in the Phase 1e artifact set
- [ ] Performance bottlenecks versus CPU/PyTorch are documented honestly in the benchmark results
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
