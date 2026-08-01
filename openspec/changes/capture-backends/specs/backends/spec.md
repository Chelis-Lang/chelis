## ADDED Requirements

### Requirement: Backend strategy and separation

Chelis SHALL lower typed programs to a RISC DAG and treat backend emission as a separate concern
from parsing, type checking, and lowering. The C, HIP, and Metal backends SHALL share that
lowering and the C backend SHALL remain the reference implementation.

#### Scenario: Backend emission is decoupled from the front end

- **WHEN** a program is compiled
- **THEN** backend emission consumes the RISC DAG rather than re-parsing or re-checking

#### Scenario: Backends share one lowering contract

- **WHEN** any native backend is selected
- **THEN** it consumes the same typed RISC DAG and does not redefine language semantics

### Requirement: C reference backend

The C backend SHALL be the reference implementation, turning the DAG into portable host code:
emitting loops for elementwise/reduction/movement ops, using OpenMP for parallelism,
pattern-matching BLAS-friendly subgraphs, managing temporaries with lifetime-aware planning, and
providing `chelis_runtime.h` plus a Rust static runtime library. Normative language semantics,
not C behavior, SHALL decide correctness for every backend.

#### Scenario: C backend emits portable host code

- **WHEN** `chelis build app.ch` compiles a program
- **THEN** it emits portable C plus `chelis_runtime.h` and the runtime library

#### Scenario: C backend is a conformance reference

- **WHEN** a GPU backend result is validated
- **THEN** the GPU, C, and evaluator results are each checked against the normative semantics

### Requirement: Shared backend ABI and source-to-source GPU model

The C, HIP, and Metal backends SHALL share the ABI
`extern "C" void func(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)`.
The HIP and Metal backends SHALL follow a source-to-source model: host control in generated
source, GPU kernels emitted as HIP/MSL source strings, and runtime compilation (`hiprtc` /
`newLibraryWithSource`). The backend crates SHALL be pure string emission with no vendor-SDK
Rust dependencies.

#### Scenario: Backends share the tensor ABI

- **WHEN** a function is emitted for any of the C, HIP, or Metal targets
- **THEN** it uses the `chelis_tensor **inputs/outputs` ABI

#### Scenario: GPU backend has no vendor-SDK Rust dependency

- **WHEN** the Metal or HIP backend crate is built
- **THEN** it builds/tests/lints on Linux, macOS, and Windows with no `metal-rs`/`objc`/`hip-rs` Rust dependency, deferring SDK integration to the user's `hipcc`/`clang++` invocation

### Requirement: Per-backend dtype gating

Each backend SHALL gate dtypes per the §1.1.3 matrix at the CLI gate, the IR validation pass, and
the codegen entry point. f64 on Metal SHALL be hard-rejected with the FP64-ALU hardware
diagnostic; bf16 on Metal SHALL require the Apple7+ GPU family. A backend capability gap
SHALL be rejected before codegen under [05-UNS-1..6].

#### Scenario: f64 on Metal is hard-rejected across surfaces

- **WHEN** an f64 program is built with `--target metal`
- **THEN** the CLI gate, IR validation, and codegen entry each reject it with the FP64-ALU diagnostic and no kernel is emitted

#### Scenario: Unsupported backend capability rejected before codegen

- **WHEN** a selected backend cannot represent a program capability
- **THEN** it rejects before codegen with an authoritative `unsupported_feature` diagnostic rather than emitting a substitute

### Requirement: Backend-boundary effect checks

`chelis build --target c` SHALL reject GPU resource regions (`with device("gpu:0")`) and
GPU targets SHALL reject incompatible non-GPU resource regions. Seeded `dropout` SHALL preserve
[05-RNG-1] and reuse the forward mask in its adjoint on every backend.

#### Scenario: C target rejects a GPU resource region

- **WHEN** a program with `with device("gpu:0")` is built with `--target c`
- **THEN** it is rejected at the backend boundary

#### Scenario: Backend preserves seeded dropout

- **WHEN** a program using `dropout` is built for any target
- **THEN** the emitted program obeys the active seed handler and the backward pass reuses the forward mask

### Requirement: Backend selection and interactive escalation

Backend selection SHALL be via `chelis build app.ch` (default C), `--target hip`, and
`--target metal`. Interactive execution SHALL NOT be a separate backend: Tide and `chelis eval`
SHALL use the IR evaluator first, escalating in the fixed order cached C artifacts → persistent
compiler helper → JIT only if measured workloads justify it.

#### Scenario: Target selection chooses the backend

- **WHEN** `chelis build app.ch --target metal` is run
- **THEN** the Metal backend is selected; the bare `chelis build app.ch` selects the default C backend

#### Scenario: Interactive execution starts with the evaluator

- **WHEN** an interactive workload runs
- **THEN** it uses the IR evaluator first, escalating only per the fixed order if latency demands it

### Requirement: All-backend invariants

All backends SHALL preserve the normative numerical tolerances and the named-dimension and
precision semantics established before lowering. Reference-backend agreement is supporting
evidence, never authority for a behavior that conflicts with the spec.

#### Scenario: GPU agrees with the reference within tolerance

- **WHEN** a GPU backend evaluates the shared test suite
- **THEN** it and the reference C backend each agree with the normative result within the documented tolerances

#### Scenario: A backend that diverges from the reference is a defect

- **WHEN** a backend disagrees with the reference C backend on a shared test
- **THEN** at least one backend is defective; the normative semantics decide which
