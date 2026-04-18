# Chelis Python Bindings

Phase `3b` / `3b-ii` Python bindings for the shared Chelis compiler API, DLPack tensor
interop, and direct compiled execution via `chelis.compile_and_load(...)` /
`chelis.load(...)`.

Current guarantees:

- `chelis.check(...)`, `compile(...)`, `desugar(...)`, `decompile(...)`, `validate(...)`
  share the same compiler implementation as Tide through `chelis-compiler-api`
- `chelis.from_dlpack(...)` is CPU-only; unsupported GPU tensors fail as `ValueError`
- `chelis.eval(...)` may copy inputs into the evaluator's internal `Vec<f64>`
  representation in `3b`
- `chelis.compile_and_load(...)` is the product path for compiled execution;
  `chelis.load(...)` is the advanced path for existing artifacts
- compiled execution currently supports fully concrete `float32` tensors
- `ChelisError` reports compiler/build/runtime failures; `ValueError` reports bad Python
  arguments such as wrong dtype, wrong shape, or unsupported device placement
- native compile/build and compiled host/device execution release the Python GIL

Compiler selection:

- native C compilation resolves through Chelis's shared platform toolchain:
  `clang` + Accelerate on macOS, `gcc` + OpenBLAS on Linux; override with `CHELIS_CC`
- native HIP compilation prefers `/usr/bin/hipcc` when present; override with
  `CHELIS_HIPCC`
