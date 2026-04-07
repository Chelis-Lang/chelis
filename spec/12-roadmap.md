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
| **1a** | HIP kernel code generation (`chelis build --target hip`) | ✅ Complete |
| **1b** | Kernel fusion (elem→elem, elem→reduce, multi-consumer split) | ✅ Complete |
| **1c** | GPU memory planning (buffer reuse, transfer minimization, peak estimate) | ✅ Complete |
| **1d** | Optimized reductions, staged scalar scratch, hipBLAS matmul specialization | ✅ Complete |
| **1e** | Benchmarks, reference comparison, checked-in results | ✅ Complete |
| **1f** | Executable grammar |  |
| **2** | Effects, linear types, macros, Tide Agent API + MCP, LSP, TUI (`chelis cove`) |  |
| **3** | Package ecosystem (Reef), StableHLO/FX backends, Python FFI, research type features, mechanized type system (Lean 4), local coding model (ships with toolchain) |  |

## Red Team Checkpoints

- after 0a
- after 0d
- after 0h
- after each major phase

Each checkpoint reviews spec compliance, test coverage, architectural debt, and
remaining design ambiguity before the project moves forward.

For Phase 0h specifically, the authoritative milestone validation is the release-mode
MNIST runner in `crates/chelis-e2e`, not `cargo test --workspace` alone.
