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
- compiled execution currently limited to fully concrete `f32` / `f64` tensors on the C
  target, and to fully concrete `f32` tensors on the HIP target (chelis#919, chelis#920).
  The per-target admit-list is `supported_execution_dtypes` in `chelis-python`; the
  marshalling layer derives the NumPy dtype and the DLPack element width from it rather
  than assuming f32, and rejects any dtype it cannot describe

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
When compiled-execution emission (the `compile_and_load` / execution-artifact lane)
scopes an artifact to a selected `def`, the emitted C symbol is decoupled from the
def name: the artifact always emits the fixed symbol `chelis_main`. Because each
artifact is scoped to exactly one entry def there is exactly one emitted entry per
translation unit, so a single fixed symbol suffices and is collision-free by
construction — a def literally named `main` no longer redefines the reserved process
entry `int main(int, char**, char**)`, a def named after a libc symbol (`free`,
`malloc`) no longer collides at link time, and neither does a def named after a
runtime symbol in the `chelis_*` namespace (`chelis_runtime.h` declares
`chelis_free`, `chelis_tuple_get`, …). The artifact manifest's `host_entry_name`
carries `chelis_main` so the loader (`dlsym`) and generated header stay consistent.
(The legacy free-form host/single-def path, where `entry_name` passes through as a
raw output symbol, is unchanged and retains the historical behavior — including its
pre-existing lack of collision protection.)

## 3. Embedding the Compiler

Longer term, the Rust crates should remain usable as libraries so Tide, editor tooling,
and external integrations can embed compiler functionality directly rather than shelling
out to the CLI.
