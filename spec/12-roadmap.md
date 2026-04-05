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
| **0f** | C backend codegen (host + BLAS + OpenMP) | 🔨 In progress |
| **0g** | `grad` transformation (reverse-mode AD on DAG) |  |
| **0h** | End-to-end: MNIST on CPU + spec test suite |  |
| **0i** | Tide v0.1 (REPL, `chelis deep`, `chelis surf`, `chelis fmt`, `chelis eval`) |  |
| **1** | Futhark-style GPU backend (HIP) + executable grammar (`chelis validate`) |  |
| **2** | Effects, linear types, macros, Tide Agent API + MCP, LSP, TUI (`chelis cove`) |  |
| **3** | Package ecosystem (Reef), StableHLO/FX backends, Python FFI, research type features, mechanized type system (Lean 4), first-party coding model |  |

## Red Team Checkpoints

- after 0a
- after 0d
- after 0h
- after each major phase

Each checkpoint reviews spec compliance, test coverage, architectural debt, and
remaining design ambiguity before the project moves forward.
