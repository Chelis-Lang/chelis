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
- the JSON entry points (`check`, `desugar`, `decompile`, `compile`, `eval`,
  `validate`) run on a worker thread and stay responsive to Ctrl-C: SIGINT
  raises `KeyboardInterrupt` promptly instead of waiting for the call to
  finish (chelis#914). The worker is always joined, never detached, so a
  cancelled call leaves no evaluation running behind it.

  **This covers the evaluation phase only.** Cancellation is observed at
  node visits, so a signal arriving while the *front end* is still working
  (parse, desugar, type-check, lower) is not noticed until evaluation
  begins. Interrupt latency during that window equals the **remaining
  compile time**, not milliseconds — measured, a 70 KB source with a ~19 s
  front end returns 17.3 s after a SIGINT sent 2 s in, and finishes
  compiling first either way. Sources with a large library context are
  where this bites; front-end cancellation is tracked separately.

  Within the evaluation phase, latency is bounded by the longest single
  uninterruptible step, not by how much work remains — in practice tens of
  milliseconds. The exception is a program holding one very large
  intermediate value, where allocating or freeing it is itself one such
  step: interrupting a fold over a 40M-element list takes a few seconds,
  because that is how long the list takes to tear down.

  Acceptance probe: `bindings/python/tests/manual_eval_interrupt.py`.

Compiler selection:

- native C compilation resolves through Chelis's shared platform toolchain:
  `clang` + Accelerate on macOS, `gcc` + OpenBLAS on Linux; override with `CHELIS_CC`
- native HIP compilation prefers `/usr/bin/hipcc` when present; override with
  `CHELIS_HIPCC`
