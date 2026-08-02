# Backends

## 1. Backend Boundary

Chelis lowers a checked program to a RISC DAG before backend selection. Backend
emission consumes that DAG; it does not re-parse source or redefine typing, effects,
numeric values, observation, or unsupported-case semantics.

The numbered language specifications define program meaning. A backend is conforming
when it implements those semantics. The C backend is the reference lane used for
cross-backend validation, but disagreement with a normative rule is a defect in the C
backend rather than a change to the language.

All native backends are source-to-source compilers. `chelis build` emits source,
headers, runtime artifacts, and the flags needed to compile them. It does not invoke the
platform C, HIP, or Objective-C++ compiler.

## 2. Shared Native ABI

The C, HIP, and Metal backends share the exported tensor ABI:

```c
extern "C" void func(
    chelis_tensor **inputs,
    int n_in,
    chelis_tensor **outputs,
    int n_out
);
```

The ABI is configuration-invariant. Generated functions bind symbolic dimensions from
input tensor metadata and validate repeated occurrences. Runtime and header artifacts
use the tagged dtype and representation vocabulary defined by
`spec/04-type-system.md`; no backend may substitute a bare floating carrier for a
typed numeric value.

## 3. C Reference Backend

The C backend emits portable host code. It:

- emits loops for elementwise, reduction, and movement operations;
- uses OpenMP for independent loop iterations;
- recognizes BLAS-compatible matrix operations;
- plans temporary-buffer lifetimes explicitly;
- links generated code against `chelis_runtime.h` and the Chelis static runtime; and
- preserves handler-scoped random state in generated host code.

The C lane and the IR evaluator form the reference agreement pair for native-backend
validation. Agreement is evidence, not semantic authority: both lanes owe their
behavior to the same language rules.

## 4. HIP Backend

The HIP backend emits C++ host code with embedded HIP kernel source. `hiprtc` compiles
the kernels for the selected device. The Rust backend crate performs pure string
emission and has no dependency on a vendor SDK.

The HIP backend:

- emits dtype-correct kernels for elementwise, reduction, movement, fill, and cast
  operations;
- differentiates an ordinary DAG before applying fusion to the resulting forward and
  gradient DAGs;
- greedily fuses eligible single-consumer elementwise chains, including
  elementwise-to-reduction regions;
- treats `realize()` as a materialization barrier;
- uses lifetime analysis to reuse non-overlapping storage slots;
- accounts for transfer buffers and reduction scratch in its peak-memory formula;
- specializes compatible matrix multiplication through hipBLAS; and
- exposes every required native link flag in build output.

Shapes and strides pass to kernels as scalar parameters. Views free only their metadata;
owning allocations free device storage.

## 5. Metal Backend

The Metal backend is the Apple GPU peer of the HIP backend. It emits Objective-C++ host
code with embedded Metal Shading Language source and compiles kernels through
`[MTLDevice newLibraryWithSource:options:error:]`. The Rust backend crate performs
pure string emission and depends on no Apple SDK crate, Objective-C bridge, or
`metal-rs`.

The Metal backend:

- shares the native tensor ABI and lowering structure with the C and HIP backends;
- emits elementwise, reduction, movement, cast, and matrix-multiplication kernels;
- fuses eligible elementwise and elementwise-to-reduction regions;
- uses the runtime-header ownership rules in `spec/04-type-system.md`;
- uses unified-memory accounting without presenting system RAM as separate device
  memory; and
- validates generated Objective-C++ independently of GPU availability.

Metal dtype admission is capability-based. `f64` requires native FP64 arithmetic and
must never be replaced by software-emulated or narrowed arithmetic. `bf16` requires an
MSL language version exposing `bfloat` and a runtime device with native bfloat
operations. A target missing either capability reports the missing capability rather
than emitting or loading a substitute kernel.

## 6. Backend Capability And Rejection

A backend must implement every operation and dtype it admits with the exact semantics
defined by the owning spec. Unsupported target capability is never permission to emit a
placeholder, substitute a different dtype, return a default value, or fall through to a
wrong kernel.

Every rejection must satisfy `spec/05-risc-primitives.md` [05-UNS-1..6]:

- it occurs before an unusable artifact is reported as successful;
- it names the operation, target, and authoritative reason;
- a hardware prohibition is distinguishable from an implementation gap; and
- all public build and embedding entry points produce the same rejection.

Backend gates are defense in depth. The codegen boundary remains total and rejects any
unsupported node even if an earlier gate is absent or permissive.

## 7. Effects And Resources

Resource handlers constrain backend selection:

- a CPU target rejects a region that requires a GPU resource;
- a GPU target rejects a region that requires an incompatible CPU-only resource; and
- a compatible resource region preserves the body's value and effects.

Random operations consume the explicit handler-scoped seed and obey
`spec/05-risc-primitives.md` [05-RNG-1]. A backend that cannot implement an admitted
effect must reject it under the unsupported-case contract; it may not silently erase
the effect.

## 8. Backend Selection

The build surface is:

- `chelis build app.ch` for the default C target;
- `chelis build app.ch --target hip` for HIP; and
- `chelis build app.ch --target metal` for Metal.

StableHLO, FX, and Triton are integration targets. They do not redefine Chelis
semantics and do not replace the native reference lane.

Interactive execution is not a separate backend. Tide and `chelis eval` use the IR
evaluator. Latency optimization follows this order: evaluator, cached C artifact,
persistent compiler helper, then JIT when measured workloads justify it.

## 9. Conformance Invariants

Every backend must preserve:

- the type, named-dimension, effect, and precision judgments established before
  lowering;
- the per-dtype arithmetic width, finalization, trap, and transport rules of
  `spec/04-type-system.md`;
- the operation semantics, accumulator rules, and unsupported-case contract of
  `spec/05-risc-primitives.md`;
- deterministic observation and root rendering under [05-OBS-1..6];
- configuration-invariant public ABI declarations; and
- numerical agreement within the per-operation tolerances defined by the language
  spec.

Tolerance covers implementation variance at one declared arithmetic width. It never
licenses a backend to compute at a different width, collapse a tagged value, suppress a
trap, or change the selected operation.
