# Roadmap

> **Status:** Living document. Updated as phases complete.

## Phase 0: Foundation (Current)
- 0a: Project scaffold, spec docs, CI ✓
- 0b: Deep parser (s-expressions)
- 0c: Surf parser + desugaring
- 0d: Type checker (ADTs, HM inference, precision, named dims)
- 0e: RISC DAG construction from typed AST
- 0f: C backend codegen (host + BLAS)
- 0g: `grad` transformation (reverse-mode AD on DAG)
- 0h: End-to-end: MNIST on CPU
- 0i: Tide v0.1 (REPL, CLI subcommands)

## Phase 1: GPU
- Futhark-style GPU backend (CUDA/OpenCL)
- Kernel fusion
- Memory optimization
- Benchmarks against PyTorch/JAX

## Phase 2: Ecosystem
- Effect system
- Linear types
- Macro system
- Tide Agent API + MCP
- vmap implementation
- jit implementation

## Phase 3: Research
- StableHLO/FX backends
- Package ecosystem (Reef)
- Advanced type features
- Self-hosting exploration

## Red Team Checkpoints
- After 0a: Spec review
- After 0d: Type system soundness
- After 0h: End-to-end correctness
- After each major phase: Architecture review
