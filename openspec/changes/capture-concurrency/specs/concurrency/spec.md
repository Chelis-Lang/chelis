## ADDED Requirements

### Requirement: Implicit DAG parallelism

Chelis SHALL be pure by default and lower to a DAG whose independent branches are eligible for
parallel execution without explicit programmer annotation. This SHALL be the default
concurrency model.

#### Scenario: Independent branches are parallel-eligible

- **WHEN** a program lowers to a DAG with two independent branches
- **THEN** those branches are eligible for parallel execution without any annotation

#### Scenario: Annotation is not required for the default story

- **WHEN** a pure program is compiled
- **THEN** it obtains DAG parallelism with no `par` or other concurrency annotation

### Requirement: Explicit parallelism construct

Chelis SHALL use `par` for explicit fork/join structure the compiler cannot recover
automatically. Sequential and parallel schedules SHALL be observationally identical.

#### Scenario: par expresses explicit fork/join

- **WHEN** a programmer needs fork/join structure the compiler cannot infer
- **THEN** `par` is the reserved construct for expressing it

#### Scenario: par is scheduler-independent

- **WHEN** a backend schedules a `par` region sequentially or in parallel
- **THEN** the observable result is identical

### Requirement: Backend concurrency mapping

Backend concurrency SHALL map without changing language-level semantics: the C backend uses
OpenMP-parallel loops where appropriate, BLAS-backed dense linear algebra uses library-managed
parallelism, and GPU backends map parallel work onto kernels.

#### Scenario: C backend uses OpenMP

- **WHEN** the C backend emits an elementwise or reduction loop
- **THEN** it uses OpenMP parallelism where appropriate without changing language-level semantics

#### Scenario: Backend mapping does not alter semantics

- **WHEN** the same program runs on the C and GPU backends
- **THEN** the language-level semantics are identical regardless of the backend's concurrency mapping

### Requirement: Concurrency non-goals

The concurrency model SHALL NOT include shared mutable-state concurrency, actor
systems, stream-processing semantics, or scatter/gather-focused parallel runtime features.

#### Scenario: Data parallelism via vmap is in scope

- **WHEN** structured data parallelism is needed
- **THEN** `vmap` expresses it as part of the concurrency-adjacent surface

#### Scenario: Actor and shared-mutable-state models are out of scope

- **WHEN** a program requires shared mutable-state concurrency or an actor system
- **THEN** it is outside the Chelis concurrency model
