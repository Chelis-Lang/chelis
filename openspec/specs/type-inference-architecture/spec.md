# type-inference-architecture

## Purpose

Define the module boundaries, source-size limit, public API parity, and behavior parity for the Chelis type-inference implementation.

## Requirements

### Requirement: Role-based inference modules

The type-inference implementation SHALL use a module tree rooted at `crates/chelis-types/src/infer/mod.rs`. Separate modules SHALL own program orchestration, checked-program construction, validation, expression forms, and application inference.

The application dispatcher SHALL separate generic call inference from numeric, tensor, shape, and collection rules. `crates/chelis-types/src/infer.rs` SHALL NOT exist.

#### Scenario: Role modules are present

- **WHEN** the source architecture guard examines the type-inference implementation
- **THEN** it finds the required role modules under `crates/chelis-types/src/infer/`

#### Scenario: Legacy monolith returns

- **WHEN** `crates/chelis-types/src/infer.rs` exists
- **THEN** the source architecture guard fails and identifies the legacy path

### Requirement: Inference source-size limit

Each production Rust file under `crates/chelis-types/src/infer/` SHALL contain no more than 3,000 physical lines. A crate-local source architecture guard SHALL enforce this limit without an external parser dependency.

#### Scenario: Modular source tree passes

- **WHEN** each inference source file contains 3,000 physical lines or fewer
- **THEN** the source architecture guard accepts the file sizes

#### Scenario: Oversized source file fails

- **WHEN** an inference source file contains more than 3,000 physical lines
- **THEN** the source architecture guard fails and identifies the file and its line count

### Requirement: Public inference API parity

The refactor SHALL preserve every public item re-exported from `chelis_types::infer`. Each item SHALL retain its name, signature, and module path.

#### Scenario: Existing consumer compiles

- **WHEN** an existing consumer imports the public inference items
- **THEN** the consumer compiles without an import or signature change

#### Scenario: Public path changes

- **WHEN** an extraction removes or renames an existing public inference path
- **THEN** the public API parity test fails

### Requirement: Type-inference behavior parity

The refactor SHALL preserve the type-system behavior that `spec/04-type-system.md` and the `type-system` capability define. Accepted programs SHALL produce equal checked trees, metadata, and inference statistics.

Rejected programs SHALL produce equal ordered diagnostic kinds, messages, spans, and hints. The refactor SHALL NOT add, remove, or reorder a diagnostic.

#### Scenario: Accepted fixture retains its output

- **WHEN** the refactored checker processes an accepted parity fixture
- **THEN** its checked tree, metadata, and inference statistics equal the recorded baseline

#### Scenario: Rejected fixture retains its diagnostics

- **WHEN** the refactored checker processes a rejected parity fixture
- **THEN** its ordered diagnostics equal the recorded baseline in kind, message, span, and hints
