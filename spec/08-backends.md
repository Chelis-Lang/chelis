# Backends

> **Status:** Stub. C backend specified in Phase 0f. GPU backend in Phase 1.

How RISC DAGs are compiled to target code.

## Sections (planned)
- Backend Interface (trait that all backends implement)
- C Backend (Phase 0): code generation, BLAS integration, memory plan, runtime
- GPU Backend (Phase 1): Futhark-style compilation to CUDA/OpenCL
- StableHLO Backend (Phase 2+): emit StableHLO for XLA ecosystem interop
- FX Graph Backend (Phase 2+): emit PyTorch FX graphs for Python ecosystem interop
- Backend Selection (CLI flags, auto-detection)
