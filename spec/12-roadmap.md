# Roadmap

**Status:** Living document.
This file summarizes the current phase boundaries and project status.
For detailed execution planning, see `spec/design/chelis_project_plan.md`.

## Phases

| Phase | Deliverable | Status |
|---|---|---|
| **0a** | Project scaffold, spec docs, CI, test infra | ✅ Complete |
| **0b** | Deep parser (s-expressions) | ✅ Complete |
| **0c** | Surf parser + Surf→Deep desugaring | ✅ Complete |
| **0d** | Type checker (ADTs, HM inference, precision, named dims) | ✅ Complete |
| **0e** | RISC DAG construction from typed AST | ✅ Complete |
| **0f** | C backend codegen (host + BLAS + OpenMP) | ✅ Complete |
| **0g** | `grad` transformation (reverse-mode AD on DAG) | ✅ Complete |
| **0h** | End-to-end: MNIST on CPU + spec test suite | ✅ Complete |
| **0i** | Tide v0.1 (REPL, `chelis deep`, `chelis surf`, `chelis fmt`, `chelis eval`) | ✅ Complete |
| **1a** | HIP kernel code generation (`chelis build --target hip`) | Implemented; carried-forward backend limits documented |
| **1b** | Kernel fusion (elem→elem, elem→reduce, multi-consumer split) | Implemented; carried-forward backend limits documented |
| **1c** | GPU memory planning (buffer reuse, transfer minimization, peak estimate) | Implemented; carried-forward backend limits documented |
| **1d** | Optimized reductions, staged scalar scratch, hipBLAS matmul specialization | Implemented; carried-forward backend limits documented |
| **1e** | Benchmarks, reference comparison, checked-in results | Shipped for fixed benchmark models |
| **1f** | Executable grammar | Implemented |
| **2a** | Algebraic effects | Initial subset shipped: effect syntax, annotated checked Deep, `Random` via `dropout` + `with seed(...)`, and `Resource(Device)` build-boundary validation |
| **2** | Remaining Phase 2 work: broader effects, linear types, macros, `vmap`, Tide Agent API + MCP, LSP, TUI (`chelis cove`), seed corpus | In progress; 2e Tide Agent API + MCP, 2f LSP, 2g Cove, and the 2gb Deep pretty-formatting side quest are shipped alongside the initial 2a subset; detailed design in `spec/design/chelis_phase2_plan.md` |
| **3** | Ecosystem foundations: package system (Reef), Python FFI, research type extensions, Lean formalization, pipe-first style pass |  |
| **4** | ML & AI coding: seed corpus, ICL measurement, trajectory collection, local model training, SKILL.md v2, model integration |  |
| **5** | Advanced backends: StableHLO, FX Graph, Triton, multi-GPU |  |

## Red Team Checkpoints

- after 0a
- after 0d
- after 0h
- after each major phase

Each checkpoint reviews spec compliance, test coverage, architectural debt, and
remaining design ambiguity before the project moves forward.

For Phase 0h specifically, the authoritative milestone validation is the release-mode
MNIST runner in `crates/chelis-e2e`, not `cargo test --workspace` alone.

Phase 2 begins with the documented carry-forward fixes from the shipped Phase 1
boundary: the remaining HIP `pad`/`shrink` work, symbolic normalized-axis support for
`layer_norm`/`mean` when needed, and the Deep dotted-path round-trip gap.
