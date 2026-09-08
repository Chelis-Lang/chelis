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

"Reference implementation" here means the first and most complete backend, and the
practical numeric oracle the later backends are checked against. It does not mean the C
backend DEFINES the values: every lane, the evaluator included, owes its results to
`spec/04-type-system.md` §9 - [04-NUM-8]'s arithmetic width, [04-NUM-1..7]'s finalize and
trap rules, and [04-NUM-11]'s exactness guarantee - and where a lane and those atoms
disagree, the lane has the bug. The C backend is the oracle because it is the most
complete conforming lane, not because conformance is defined as agreeing with it.

Current design points:

- emit loops for elementwise, reduction, and movement operations
- use OpenMP for elementwise and reduction parallelism
- pattern-match BLAS-friendly subgraphs such as matrix multiplication
- manage temporary buffers with explicit lifetime-aware memory planning
- ship `chelis_runtime.h` plus a Rust static runtime library alongside generated code
- preserve `with seed(...)` for generated host code with a handler-scoped RNG state;
  direct DAG random operations use their baked seed, while host fallback functions can
  draw from the active handler when stdlib/user calls contain nested `uniform_like`
- tuple-returning host exports use the stable runtime tuple ABI:
  generated headers surface `chelis_tuple*`, drivers construct tuples with
  `chelis_tuple_from_values(...)`, and typed extraction goes through the
  `chelis_tuple_get_*` helpers documented in `chelis_runtime.h`

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
- movement ops (reshape, permute, expand, insert, stride) are host-side metadata operations
- naive reductions (one thread per output element, inner loop over axis)
- `chelis_gpu_free` for allocations, `chelis_gpu_free_view` for views
- `chelis build app.ch --target hip` emits compilable `*_hip.cpp` host output
- `pad` and `shrink` are implemented as typed per-output-element kernels
  (`kernel_pad{_dtype}` / `kernel_shrink{_dtype}`); GPU output is verified
  equal to the `chelis-ir` evaluator by the `g16_pad_*` / `g16_shrink_*`
  cases in the `gpu_correctness` manual oracle

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
- contiguous `f32` matmul subgraphs with rank ≥ 2 specialize to hipBLAS-backed helpers:
  rank-2 emits `chelis_hipblas_sgemm_row_major(...)`, while batched/symbolic matmul emits
  runtime-sized calls through `chelis_hipblas_sgemm_batched_row_major(...)`
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
cargo test -p chelis-e2e --test example_corpus_validate
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

- symbolic dimensions are implemented on the stable tensor ABI for both backends:
  generated functions bind symbolic names from input tensor metadata at runtime and
  validate repeated occurrences across all participating inputs
- symbolic shape normalization v1 is name-stable and conservative: memory planners
  canonicalize product ordering/associativity and arithmetic identities such as
  `n * 1`, but they do not alpha-rename unrelated symbols. Alpha-renaming is only
  valid for future paths that carry explicit same-property `forall` or
  binder-equivalent dimension identity.
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

## 4. Phase M: Metal Backend (macOS GPU peer)

The Metal backend is the macOS-native GPU peer of the HIP backend. It is not a
continuation of HIP work — it is an independent backend track that mirrors HIP's
architecture exactly:

- host-side control remains in generated source, here Objective-C++ in `.mm` files
- GPU kernels are emitted as Metal Shading Language (MSL) source strings
- `[MTLDevice newLibraryWithSource:options:error:]` performs runtime compilation —
  the direct analog of `hiprtc` for the HIP path

Same ABI as the C and HIP backends:
`extern "C" void func_name(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)`.

The Metal backend is **pure string emission** in Rust. The crate has zero macOS-only
Rust dependencies (no `metal-rs`, no `objc`); it builds, tests, and lints clean on
Linux, macOS, and Windows. Apple-SDK integration happens later when the user runs
`clang++ -fobjc-arc -framework Metal -framework Foundation` against the emitted
`.mm`. This keeps `--target metal` available as a cross-compilation target and
mirrors how HIP works (no `hip-rs` dep on the Rust side; the user runs `hipcc`).

Full design in `spec/design/chelis_metal_backend_plan.md`.

### Phase M0: Spec sync (complete)

Authoritative oracle:

```sh
grep -F "[MTLDevice newLibraryWithSource:]" \
  spec/design/chelis_metal_backend_plan.md
```

The grep proves the Metal runtime model has been switched away from the original
`metal-rs` Rust runtime to the string-emission-plus-runtime-source-compilation
model that mirrors HIP.

### Phase M1: Scaffolding + CLI dispatch

`crates/chelis-backend-metal/` is structurally a peer of `crates/chelis-backend-hip/`:
`src/{lib,emit,kernels,launch,memory,blas}.rs`, `runtime/chelis_metal_runtime.h`,
`tests/{codegen_structure,codegen_adversarial,gpu_correctness}.rs`.

`chelis build --target metal` is wired in `crates/chelis-cli/src/main.rs` alongside
`--target c` and `--target hip`. `pad`/`shrink` are implemented as typed MSL
movement kernels (WS-8A) and pass through to codegen; the reject pass no longer
denies them. Per-dtype admit/reject decisions for
the Metal backend are pinned in `spec/04-type-system.md` §1.1.3 (the
per-backend dtype matrix); the CLI gate, the IR validation pass, and the
codegen entry point each consult that matrix. f64 is **hard-rejected** on
Metal with the FP64-ALU hardware diagnostic per §1.1.3; bf16 admits at
codegen but pipeline creation surfaces the Apple7+ requirement at runtime
on pre-Apple7 devices, also per §1.1.3. Sort/argsort/cumsum/cumprod don't
exist as IR variants today — when they're added, both backends' reject
passes will need the corresponding arms; documented as a follow-up rather
than a current guarantee.

Authoritative oracle:

```sh
cargo build --workspace && \
cargo test -p chelis-cli --test cli -- target_metal && \
cargo tree -p chelis-cli --prefix none --no-dedupe \
  | python3 -c 'import sys, re
forbidden = ("metal", "objc", "objc-foundation", "objc_id", "objc_exception",
             "cocoa", "core-graphics", "core-foundation", "block")
pattern = re.compile(r"^(" + "|".join(re.escape(n) for n in forbidden) + r") v")
bad = sorted({l.strip() for l in sys.stdin if pattern.match(l)})
sys.exit(1 if bad else 0)'
```

The `cargo tree` step enforces the no-Apple-SDK-Rust-deps invariant.

### Phase M2: Elementwise emission

`MetalEmitter` mirrors `HipEmitter::emit_dag` (same passes: `collect_kernels`,
`emit_kernel_string_decl`, signature emission, `emit_input_shape_preamble`,
plan-driven slot allocation). MSL replaces HIP C/C++ syntax for kernel
declarations, thread indexing, buffer qualifiers, and math built-ins.

Coverage: add/sub/mul/div/neg/exp/log/sqrt/sin/cast/clamp/where, fill,
fused elementwise chains, reshape (metadata-only), permute, expand, insert.

Authoritative oracle:

```sh
cargo test -p chelis-backend-metal --test codegen_structure
```

### Phase M3: macOS CI compile-and-link smoke

A Python smoke harness (`.github/scripts/smoke_macos_metal.py`) drives
`chelis build --target metal` on a fixed-shape elementwise program and then runs
`clang++ -std=c++17 -fobjc-arc -O2 ... -framework Metal -framework Foundation`
on the emitted `.mm`. Smoke is intentionally compile-and-link only;
`MTLCreateSystemDefaultDevice` may return null on macos-latest VMs, so kernel
dispatch is gated to the workstation (Phase M6), not CI.

Authoritative oracle: after a manual workflow dispatch, the `macos-smoke` GitHub Actions job exits 0.

### Phase M4: Reductions + fused-elementwise-into-reduction

MSL reduction templates (sum/max/min) using threadgroup memory and tree
reduction. Two-pass for arrays larger than one threadgroup. Fused
elementwise-into-reduction reuses the existing `reduction_inlined_fused_elems`
path in `chelis-ir`.

Authoritative oracle:

```sh
cargo test -p chelis-backend-metal --test codegen_structure -- reduction
```

### Phase M5: Tiled matmul

Custom 16×16 tiled MSL matmul kernel (no MPS dep). The specialization rule
mirrors the original HIP hipBLAS detection: rank-2 contiguous f32 matmul
subgraphs (`insert + mul + sum(axis=1)`) route to
`chelis_metal_matmul_tiled`. Rank ≥ 3 and symbolic batched matmul remain a
Metal follow-up.

Authoritative oracle:

```sh
cargo test -p chelis-backend-metal --test codegen_structure -- matmul_tiled
```

### Phase M6: GPU correctness oracle (manual)

`tests/gpu_correctness.rs` with `#[ignore]` on every test. Each test invokes
`codegen_metal`, writes the emitted `.mm` to a tempdir, drives `clang++` against
`-framework Metal -framework Foundation`, runs the resulting binary, and asserts
agreement with the `chelis-ir` evaluator within Metal-specific f32 tolerances.

Authoritative oracle (manual, requires Apple Silicon Mac with a usable Metal
device):

```sh
cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1
```

This is the single oracle for M2–M6 GPU correctness. MSL fast-math semantics may
require widened tolerance versus HIP for `exp`/`log`/`sqrt`-heavy kernels;
specific kernels needing higher precision use `precise::*` qualifiers per-call.

#### Metal numeric-agreement gate (cross-backend oracle)

The C backend is the numeric oracle for every GPU backend (§2). The Metal
numeric-agreement contract is: for each kernel under test, the Metal GPU
result must equal the `chelis-ir` evaluator (which the C backend is verified
against) within the Metal f32 tolerance (`ABS_TOL`/`REL_TOL` in
`tests/gpu_correctness.rs`, `1e-4` each, widened per fast-math note above).
`assert_close` in that file is the agreement check.

This gate is **manual and workstation-only** — it is NOT part of default CI,
because `MTLCreateSystemDefaultDevice` returns null on the macos-latest CI VMs.
The manual `macos-smoke` job only compiles and links. This gate mirrors the manual HIP `gpu_correctness` oracle.

- Owning phase: Phase M6.
- Command (Apple Silicon Mac with a usable Metal device):
  `cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1`
- Success condition: every `#[ignore]` test passes (exit 0); each asserts
  Metal GPU == evaluator within tolerance.

WS-8A added `m6_pad_*` / `m6_shrink_*` cases (1-D and 2-D pad/shrink, non-zero
fill, and a pad→shrink roundtrip) to this gate, mirroring the HIP `g16_*`
cases one-for-one.

### Phase M7: Adversarial test surface

`tests/codegen_adversarial.rs` mirrors `crates/chelis-backend-hip/tests/codegen_adversarial.rs`.
Cases: zero-element tensors, rank-0 scalars, symbolic dims of 0/1, bool through
`where`, cast f32→bool→f32 round-trip, very large grids (>2^16 threadgroups),
single-element reductions, matmul with degenerate dimensions.

Authoritative oracle:

```sh
cargo test -p chelis-backend-metal --test codegen_adversarial
```

### Carried-forward limitations

- `sort`, `argsort`, `cumsum`, `cumprod`, `diagonal`, `trace` not on the GPU path
- f64 is **hard-rejected** on Metal because Apple Silicon GPUs have no FP64 ALUs;
  software emulation is out of scope. The diagnostic and rationale are pinned in
  `spec/04-type-system.md` §1.1.3. f64 workloads must use `--target c` or
  `--target hip`.
- bf16 on Metal requires Apple7+ GPU family (M3 or later); the kernel template
  guards `bfloat` on `__METAL_VERSION__ >= 320`, and runtime pipeline creation
  surfaces a clean diagnostic on M1/M2 devices. See `spec/04-type-system.md`
  §1.1.3 for both surfaces.
- MPS integration for f32 and f16 matmul is the wrapper-helper plan from
  WS-M1; `chelis_metal_runtime.h` exposes the helpers under the ARC
  ownership model pinned in `spec/04-type-system.md` §1.1.3 ("Metal runtime
  header: ARC vs MRC and MPS wrapper ownership model"). bf16, int8, int16,
  int32, and int64 matmul (where admitted by §5.7.2) routes through the
  parameterized 16x16 tiled MSL kernel rather than MPS.
- Async dispatch deferred; M-phase uses `waitUntilCompleted` for synchronous launches
- `peak_device_bytes_formula` semantically reports peak system RAM for tensor
  storage on Apple Silicon (no separate VRAM); the CLI prefixes the formula with
  a one-line note so users do not double-count

These are real backend limitations, not hidden caveats.

## 5. Later Integration Backends

Later backends are additive:

### StableHLO

For TPU/XLA ecosystem access and ML compiler interop.

### FX

For PyTorch ecosystem interop, export, and execution through the FX / TorchInductor
toolchain.

These do not replace the C/HIP/Metal path.

## 6. Interactive Execution

Interactive execution is not a separate backend.
Tide and `chelis eval` use the IR evaluator first.
If latency later becomes a problem, the escalation order is:

1. cached C artifacts
2. persistent compiler helper
3. JIT only if measured workloads justify it

No Cranelift-based backend is currently planned.

## 7. Backend Selection

Planned command surface:

- `chelis build app.ch`
- `chelis build app.ch --target hip`
- `chelis build app.ch --target metal`

Additional targets may be added later as StableHLO, FX, and Triton land.

## 8. Invariants

All backends must preserve:

- the numeric semantics of `spec/04-type-system.md` §9: finalize at the declared dtype,
  compute at the arithmetic width [04-NUM-8] declares, trap per [04-NUM-9]/[04-NUM-10],
  and carry values without collapse per [04-NUM-11]
- numerical correctness within documented tolerances, which under [04-NUM-8] cover
  implementation variance at a single width (libm against SLEEF against vForce) and never
  a structural width mismatch between lanes
- named-dimension and precision semantics established before lowering
- agreement with the reference C backend on the shared test suite - a practical oracle
  for the invariants above, not a substitute for them
