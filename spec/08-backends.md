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

### Phase 1c: Memory Planning (complete)

Authoritative Phase 1c oracle:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Current implementation:

- greedy slot reuse for non-overlapping storage lifetimes in `chelis-backend-hip/src/memory.rs`
- unique input copies transferred once, with repeated `Load(name)` nodes aliasing the first copy
- planner-driven cleanup: every metadata wrapper freed once, every backing slot freed once
- movement ops and `store` remain metadata aliases over the chosen backing slot
- kernel outputs iterate over logical element count (`d_t->size`), while input guard checks still use backing `storage_size`
- HIP codegen reports a peak-memory formula plus an optional concrete estimate when every slot size is statically known
- `chelis build --target hip` prints the formula unconditionally and the concrete estimate when available
- the reporting surface includes inline staged-reduction scratch chains used by Phase 1d scalar reductions
- no runtime memory-budget comparison or checkpoint insertion yet; Phase 1c ships reporting-only visibility

### Phase 1d: Optimized Reductions + hipBLAS (complete)

Authoritative Phase 1d oracle:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Current implementation:

- segmented reductions use a single runtime-sized axis-specific kernel for the generic path
- fused elementwise→reduction kernels reuse that same runtime-sized segmented reduction path
- scalar contiguous reductions use a staged scratch-chain reduction with inline `hipMalloc`/`hipFree`, outside the Phase 1c slot planner
- the peak-memory formula includes the worst single staged scratch chain alongside the slot-plan terms
- rank-2 contiguous `f32` matmul subgraphs (`expand + mul + sum(axis=1)`) specialize to `chelis_hipblas_sgemm_row_major(...)`
- non-contiguous matmul-shaped DAGs fall back to the generic reduction path
- `chelis build --target hip` surfaces the required `-lhipblas` link flag when hipBLAS specialization is emitted

### Phase 1e: Benchmarks and Reference Comparison (complete)

Authoritative Phase 1e oracle:

```sh
cargo run --release -p chelis-e2e --bin bench_phase1e -- --model all --emit-json benchmarks/results/latest.json
```

Current implementation:

- fixed-workload benchmark runner in `chelis-e2e`, not a general-purpose harness
- real compiled-backend execution for Chelis CPU and Chelis HIP benchmark lanes
- executable benchmark examples for linear regression and a transformer-block-style forward path
- checked-in PyTorch reference scripts under `benchmarks/pytorch/`
- benchmark PyTorch lane resolves through `CHELIS_BENCH_PYTHON` or the repo-local `py/.venv`
  prepared with the gfx1151 ROCm nightly install command documented in `benchmarks/RESULTS.md`
- PyTorch benchmark invocations strip stale `HSA_OVERRIDE_GFX_VERSION` shell overrides and
  export the ROCm SDK library path needed by the nightly wheel set
- checked-in benchmark artifacts:
  - `benchmarks/results/latest.json`
  - `benchmarks/RESULTS.md`
- benchmark scope constrained to the shipped op surface:
  - `linreg` training
  - `mnist` training + inference on a fixed subset
  - `transformer` forward pass
- missing HIP, PyTorch, or MNIST dataset prerequisites are surfaced as explicit skips in the emitted JSON rather than aborting the oracle
- CI keeps PyTorch out of the default gate; the local checked-in artifact is the PyTorch comparison proof
- the full `bench_phase1e --model all` integration test is `#[ignore]` and run manually via
  `cargo test -p chelis-e2e --test bench_phase1e -- --ignored`; the default workspace gate
  keeps only the fast structural smoke coverage

### Phase 1f: Executable Grammar (complete)

Authoritative Phase 1f oracle:

```sh
cargo test -p chelis-e2e --test phase1f_validate
```

Current implementation:

- `chelis validate --surf file.ch` validates Surf syntax against the PEG conformance grammar
- `chelis validate --deep file.dp` validates Deep syntax plus the closed tag/metadata/arity rules
- `chelis validate --desugar file.ch` validates compiler-desugared canonical Deep output
- the validator is implemented in the standalone `chelis-validate` crate and wired through the CLI
- the oracle suite checks agreement across executable examples, illustrative syntax examples,
  `SKILL.md`, curated positive spec fixtures, and curated negative fixtures

Phase 1 implementation work is now present through 1f.
The shipped fixed-workload Phase 1 deliverable is met: the benchmark models used by
Phase 1e compile and run on both backends, and the executable-grammar surface from 1f
is shipped. Known carried-forward limitations remain explicit:

- HIP does not yet implement `pad` / `shrink`; no current Phase 1 benchmark model uses them
- symbolic dimensions are implemented on the stable tensor ABI for both backends:
  generated functions bind symbolic names from input tensor metadata at runtime and
  validate repeated occurrences across all participating inputs
- `layer_norm` still requires a concrete normalized-axis extent; symbolic leading dims
  are supported, but a symbolic hidden size remains a follow-up
- dotted Deep module/import round-trip remains a separate Phase 2 parser/decompiler follow-up

These are real backend limitations, not hidden caveats, but they do not block Phase 2
language work.

### Phase 2a backend-boundary checks

The first shipped effect surface interacts with backend selection in two explicit ways:

- `chelis build --target c` rejects resource regions such as
  `with device("gpu:0") { ... }`
- `chelis build --target hip` rejects incompatible non-GPU resource regions
- `chelis build` for either target currently rejects lowered `dropout`; seeded dropout
  is implemented on the evaluator path, not yet on emitted C/HIP code

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

Additional targets may be added later as StableHLO, FX, and Triton land.

## 7. Invariants

All backends must preserve:

- numerical correctness within documented tolerances
- named-dimension and precision semantics established before lowering
- agreement with the reference C backend on the shared test suite
