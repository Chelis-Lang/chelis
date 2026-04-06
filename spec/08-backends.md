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

### Phase 1a: Kernel Code Generation (complete)

The `chelis-backend-hip` crate generates HIP host source with embedded HIP kernel strings.
Same ABI as the C backend (`chelis_tensor **inputs/outputs`).

Authoritative Phase 1a oracle:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Current implementation:

- kernel source strings for the Phase 1a execution surface:
  elementwise ops, reductions, fill, and cast
- `chelis-ir::grad_then_fuse` preserves the required Phase 1b ordering:
  differentiate the ordinary DAG first, then fuse the resulting gradient DAG
- shapes/strides passed as individual int kernel parameters (not device pointers)
- debug builds reset/check a per-module `chelis_gpu_failure` flag after every kernel launch
- views preserve backing allocation size so debug index guards validate against real storage
- movement ops (reshape, permute, expand, stride) are host-side metadata operations
- naive reductions (one thread per output element, inner loop over axis)
- `chelis_gpu_free` for allocations, `chelis_gpu_free_view` for views
- `chelis build app.ch --target hip` emits compilable `*_hip.cpp` host output
- `pad` and `shrink` remain deferred to a later Phase 1 iteration

### Phase 1b: Kernel Fusion

Greedy elementwise fusion: adjacent single-consumer elementwise ops are merged into
`FusedElem` nodes that emit as single GPU kernels. MNIST drops from 27 to 19 kernel
launches.

- fusion pass in `chelis-ir/src/fuse.rs` (DAG-to-DAG rewrite, shared by all backends)
- `FusedElem` variant in `RiscOp` with `FusedStep`/`FusedStepOp`/`FusedInput` types
- elementwise→reduction fusion: when a FusedElem's sole consumer is a reduction,
  the elementwise chain is inlined into the reduction kernel's inner loop
- HIP emitter generates fused kernel source strings (register-chained computation)
- C backend emits fused `#pragma omp parallel for` loops (wired into CLI for both targets)
- evaluator decomposes `FusedElem` back to individual ops for testing
- multi-consumer split fusion: per spec, multi-consumer nodes are materialized and serve
  as external inputs to downstream chains
- `realize()` lowers to a real DAG materialization barrier and blocks fusion across it
- `egg` evaluation skipped; greedy heuristic sufficient for Phase 1b scope

Key work remaining in Phase 1:

- GPU memory planning with buffer reuse (Phase 1c)
- optimized reductions + hipBLAS (Phase 1d)
- benchmark and correctness validation against the C backend (Phase 1e)
- executable grammar / `chelis validate` (Phase 1f)

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
