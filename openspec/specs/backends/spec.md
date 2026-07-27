# backends

## Purpose

Define the Chelis backend contract: the C-first sequential strategy and the separation of
backend emission from the front end, the C reference backend and its numerical-oracle role, the
HIP and Metal GPU backends with their shared `chelis_tensor **` ABI and source-to-source model,
per-backend dtype gating, backend-boundary effect checks, backend selection and interactive
escalation, and the all-backend numerical-correctness and reference-agreement invariants. This
is the current truth of how Chelis emits target code.

## Requirements

### Requirement: Backend strategy and separation

Chelis SHALL lower typed programs to a RISC DAG and treat backend emission as a separate concern
from parsing, type checking, and lowering. The strategy SHALL be sequential — make the C backend
correct and complete, use it as the numerical oracle, then add a single HIP GPU path — and SHALL
NOT introduce multiple competing native code generators in Phase 0.

#### Scenario: Backend emission is decoupled from the front end

- **WHEN** a program is compiled
- **THEN** backend emission consumes the RISC DAG rather than re-parsing or re-checking

#### Scenario: No competing Phase-0 native generators

- **WHEN** the Phase 0 backend surface is evaluated
- **THEN** there is a single native code generator (C), not multiple competing ones

### Requirement: C reference backend

The C backend SHALL be the reference implementation, turning the DAG into portable host code:
emitting loops for elementwise/reduction/movement ops, using OpenMP for parallelism,
pattern-matching BLAS-friendly subgraphs, managing temporaries with lifetime-aware planning, and
shipping `chelis_runtime.h` plus a Rust static runtime library. It SHALL be the correctness
oracle for the GPU backends.

#### Scenario: C backend emits portable host code

- **WHEN** `chelis build app.ch` compiles a program
- **THEN** it emits portable C plus `chelis_runtime.h` and the runtime library

#### Scenario: C backend is the numerical oracle

- **WHEN** a GPU backend result is validated
- **THEN** it is checked against the C backend / IR evaluator on the shared test suite

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
diagnostic; bf16 on Metal SHALL require the Apple7+ GPU family; unsupported ops (e.g.
`reduce_window_*` on HIP) SHALL be rejected before codegen with a clean diagnostic.

#### Scenario: f64 on Metal is hard-rejected across surfaces

- **WHEN** an f64 program is built with `--target metal`
- **THEN** the CLI gate, IR validation, and codegen entry each reject it with the FP64-ALU diagnostic and no kernel is emitted

#### Scenario: Unsupported HIP op rejected before codegen

- **WHEN** a `reduce_window_*` program is built with `--target hip`
- **THEN** it is rejected before codegen with a clean `unsupported_feature` diagnostic rather than emitting a kernel

### Requirement: Backend-boundary effect checks

`chelis build --target c` SHALL reject GPU resource regions (`with device("gpu:0")`) and
`chelis build --target hip` SHALL reject incompatible non-GPU resource regions. `chelis build`
for either target SHALL currently reject lowered `dropout`, whose seeded implementation is on the
evaluator path, not emitted C/HIP code.

#### Scenario: C target rejects a GPU resource region

- **WHEN** a program with `with device("gpu:0")` is built with `--target c`
- **THEN** it is rejected at the backend boundary

#### Scenario: Build rejects lowered dropout

- **WHEN** a program using `dropout` is built for the C or HIP target
- **THEN** it is rejected because seeded dropout is implemented on the evaluator path, not emitted backend code

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

All backends SHALL preserve numerical correctness within documented tolerances, the
named-dimension and precision semantics established before lowering, and agreement with the
reference C backend on the shared test suite.

#### Scenario: GPU agrees with the reference within tolerance

- **WHEN** a GPU backend evaluates the shared test suite
- **THEN** it agrees with the reference C backend within the documented tolerances

#### Scenario: A backend that diverges from the reference is a defect

- **WHEN** a backend disagrees with the reference C backend on a shared test
- **THEN** it is a backend defect, because reference agreement is an invariant
