# Backends

**Status:** Active outline.
Phase 0 defines the C backend.
Phase 1 adds the GPU backend.
Later backends are integration layers.

## 1. Backend Strategy

Chelis lowers typed programs to a RISC DAG and treats backend emission as a separate
concern from parsing, type checking, and lowering.
The backend strategy is intentionally sequential:

1. make the C backend correct and complete
2. use it as the numerical oracle for later backends
3. add a single GPU code generation path based on HIP
4. add ecosystem integration backends later

The project does **not** plan multiple competing native code generators in Phase 0.

## 2. Phase 0: C Backend

The C backend is the reference implementation.
Its job is to turn the DAG into portable host code that can be compiled with the system
toolchain.

Current design points:

- emit loops for elementwise, reduction, and movement operations
- use OpenMP for elementwise and reduction parallelism
- pattern-match BLAS-friendly subgraphs such as matrix multiplication
- manage temporary buffers with explicit lifetime-aware memory planning
- ship a small C runtime alongside generated code

This backend is the correctness oracle for future GPU and interoperability backends.

## 3. Phase 1: HIP Backend

The GPU plan is Futhark-style source-to-source compilation:

- host-side control remains in generated C
- GPU kernels are emitted as HIP source strings
- `hiprtc` performs runtime compilation of those kernels

This is the only planned native GPU path.
Chelis does **not** plan separate CUDA and OpenCL backend implementations.
HIP is the vendor-facing abstraction layer.

Key work in Phase 1:

- kernel emission
- kernel fusion
- GPU memory planning
- flattening and parallelization strategy
- benchmark and correctness validation against the C backend

## 4. Later Integration Backends

Later backends are additive:

### StableHLO

For TPU/XLA ecosystem access and ML compiler interop.

### FX

For PyTorch ecosystem interop, export, and execution through the FX / TorchInductor
toolchain.

These do not replace the C/HIP path.

## 5. Interactive Execution

Interactive execution is not a separate backend.
Tide and `chelis eval` use the IR evaluator first.
If latency later becomes a problem, the escalation order is:

1. cached C artifacts
2. persistent compiler helper
3. JIT only if measured workloads justify it

No Cranelift-based backend is currently planned.

## 6. Backend Selection

Planned command surface:

- `chelis build app.ch`
- `chelis build app.ch --target hip`

Additional targets may be added later as StableHLO and FX land.

## 7. Invariants

All backends must preserve:

- numerical correctness within documented tolerances
- named-dimension and precision semantics established before lowering
- agreement with the reference C backend on the shared test suite
