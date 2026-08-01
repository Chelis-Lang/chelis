# type-inference-architecture

## Purpose

Define the module boundaries, source-size limit, public API, and semantic conformance for the Chelis type-inference implementation.

## Requirements

### Requirement: Role-based inference modules

The type-inference implementation SHALL use a module tree rooted at `crates/chelis-types/src/infer/mod.rs`. Separate modules SHALL own program orchestration, checked-program construction, validation, expression forms, and application inference.

The application dispatcher SHALL separate generic call inference from numeric, tensor, shape, and collection rules. `crates/chelis-types/src/infer.rs` SHALL NOT exist.

#### Scenario: Role modules are present

- **WHEN** the source architecture guard examines the type-inference implementation
- **THEN** it finds the required role modules under `crates/chelis-types/src/infer/`

#### Scenario: Monolithic inference path is forbidden

- **WHEN** `crates/chelis-types/src/infer.rs` exists
- **THEN** the source architecture guard fails and identifies the forbidden path

### Requirement: Inference source-size limit

Each production Rust file under `crates/chelis-types/src/infer/` SHALL contain no more than 3,000 physical lines. A crate-local source architecture guard SHALL enforce this limit without an external parser dependency.

#### Scenario: Modular source tree passes

- **WHEN** each inference source file contains 3,000 physical lines or fewer
- **THEN** the source architecture guard accepts the file sizes

#### Scenario: Oversized source file fails

- **WHEN** an inference source file contains more than 3,000 physical lines
- **THEN** the source architecture guard fails and identifies the file and its line count

### Requirement: Stable public inference API

Every public item re-exported from `chelis_types::infer` SHALL retain its specified name, signature, and module path unless a normative API change explicitly amends this capability.

#### Scenario: Consumer compiles

- **WHEN** a consumer imports the public inference items
- **THEN** the consumer compiles without an import or signature change

#### Scenario: Public path changes

- **WHEN** an implementation removes or renames a specified public inference path
- **THEN** the public API parity test fails

### Requirement: Type-inference behavior conforms to the language spec

Type inference SHALL implement the behavior that `spec/04-type-system.md` and the `type-system` capability define. Repeated checks of the same accepted program SHALL produce equal checked trees, metadata, and inference statistics.

Repeated checks of the same rejected program SHALL produce equal ordered diagnostic kinds, messages, spans, and hints.

#### Scenario: Accepted fixture has deterministic output

- **WHEN** the checker processes an accepted conformance fixture
- **THEN** its checked tree, metadata, and inference statistics equal that fixture's normative expected result

#### Scenario: Rejected fixture has deterministic diagnostics

- **WHEN** the checker processes a rejected conformance fixture
- **THEN** its ordered diagnostics equal that fixture's normative expected kind, message, span, and hints
