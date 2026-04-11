# Foreign Function Interface

**Status:** Outline for later phases.
Chelis does not depend on FFI work for Phase 0 completion, but the expected direction is
already clear enough to record.

## 1. Python Interop

The Phase 3 Python path is split into two cuts:

### Phase 3b: Interop Core

- PyO3 bindings for compiler-facing entry points through a shared
  `chelis-compiler-api` crate
- install surface: `uv pip install ./bindings/python`
- CPU-only DLPack interop with PyTorch as the tested guarantee
- safetensors read/write helpers that round-trip with PyTorch
- GIL release during compiler/evaluator work
- `ChelisError` is reserved for compiler/build/runtime failures
- Python-side data mismatches such as unsupported GPU tensors in this cut are surfaced as
  `ValueError`
- `chelis.eval(...)` may copy Python tensor inputs into the evaluator's internal
  `Vec<f64>` representation in this cut
- GPU tensors are rejected as Python-side `ValueError`s and deferred to `3b-ii`

### Phase 3b-ii: Direct Execution + NumPy Guarantee

- `chelis.compile_and_load("model.ch")` as the product path for compiled execution
- `chelis.load("model.so")` as the advanced loader for an existing artifact
- sidecar manifest (`model.json`) recording source path + content hash, with a warning on
  `load()` if the source has changed since compilation
- direct loading/calling of compiled Chelis artifacts from Python
- CPU direct execution verified for PyTorch CPU tensors and NumPy arrays
- HIP device ABI emitted for direct GPU execution, with the Python GPU bridge layered on
  that ABI
- GIL release during native compile/build and compiled host/device execution
- NumPy DLPack guarantee as a documented/tested promise
- compiled execution currently limited to fully concrete `f32` tensors

### Phase 5a

- JAX DLPack guarantee alongside the StableHLO backend

This is a Phase 3 interoperability feature, not a Phase 0 requirement.

## 2. C Interop

Generated C headers and runtime support should make it possible to call compiled Chelis
artifacts from C or C++.
That interoperability follows naturally from the reference backend and does not require a
separate host-language embedding model first.
After Phase `3m`, that C-facing surface is expected to come from `chelis_runtime.h`
plus the shipped Rust static runtime library rather than a generated `chelis_runtime.c`
implementation file.

## 3. Embedding the Compiler

Longer term, the Rust crates should remain usable as libraries so Tide, editor tooling,
and external integrations can embed compiler functionality directly rather than shelling
out to the CLI.
