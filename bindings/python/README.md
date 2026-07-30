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
  finish (chelis#914, chelis#930). The worker is always joined, never
  detached, so a cancelled call leaves no work running behind it.

  The **front end** is cancellable too (chelis#930): parse, desugar,
  type-check and lowering poll the same token at every phase boundary, and
  at every top-level declaration inside the passes whose cost scales with
  declaration count. A SIGINT arriving during compilation therefore
  abandons the compile instead of waiting it out. Measured on a 70 KB
  source with a ~19 s front end: interrupting 2 s in used to return after
  17.3 s (the *remaining compile time*) and now returns in tens of
  milliseconds.

  In both phases, latency is bounded by the longest single uninterruptible
  step, not by how much work remains. Two shapes make that step large:

  - during evaluation, a program holding one very large intermediate
    value, where allocating or freeing it is itself one step —
    interrupting a fold over a 40M-element list takes a few seconds,
    because that is how long the list takes to tear down;
  - during compilation, one very large top-level declaration, since a
    declaration is the grain at which the front end polls. Many
    declarations are cheap regardless of how many there are; one
    pathological declaration is not.

  Not everything on the path polls. The Reef graph resolution that precedes
  a package compile, and lowering's whole-program walk, run to completion
  once entered.

  Acceptance probe: `bindings/python/tests/manual_eval_interrupt.py`.

Compiler selection:

- native C compilation resolves through Chelis's shared platform toolchain:
  `clang` + Accelerate on macOS, `gcc` + OpenBLAS on Linux; override with `CHELIS_CC`
- native HIP compilation prefers `/usr/bin/hipcc` when present; override with
  `CHELIS_HIPCC`
