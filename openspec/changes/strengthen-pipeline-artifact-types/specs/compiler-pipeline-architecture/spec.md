## ADDED Requirements

### Requirement: Proof-bearing edit success
The edit API SHALL return an opaque `ValidatedModule` only after type, effect, and linearity checks accept the complete edited module. Only the compiler-owned whole-module check function SHALL construct this type.

`ReplacementReport` SHALL contain the proof type and SHALL NOT contain `checks_clean` or a public raw module field. The proof type SHALL provide read-only and consuming access to its Deep expressions.

#### Scenario: A valid edit returns proof
- **WHEN** the whole-module check accepts an edited module
- **THEN** the result contains a `ValidatedModule` whose expressions equal the checked module

#### Scenario: A rejected edit returns no proof
- **WHEN** type, effect, or linearity checks reject an edited module
- **THEN** the edit API returns the existing tagged error and no `ValidatedModule`

#### Scenario: A caller tries to forge success
- **WHEN** a compile-fail fixture constructs `ValidatedModule` or a false successful `ReplacementReport`
- **THEN** Rust rejects the fixture because the proof constructor is private

### Requirement: Typed and aligned root artifacts
The pipeline SHALL represent all root names, tensor root names, declared tensor outputs, and forward node aliases with separate opaque types. These types SHALL use `IrName` instead of untyped `String` keys at the pipeline boundary.

Normal DAG construction SHALL create `NamedRoots` through one smart constructor. This constructor MUST require exact positional alignment between `TensorRootNames` and `Dag::roots()`.

The existing empty host fallback SHALL use an explicit empty constructor. No constructor SHALL truncate a mismatched pair with `zip`.

`LoweredCompilation::into_parts` SHALL return a named `LoweredParts` product. It SHALL NOT return adjacent root maps in a tuple.

#### Scenario: Tensor names align with DAG roots
- **WHEN** a tuple-valued tensor output lowers to two DAG roots
- **THEN** `NamedRoots` binds each `IrName` to the root at the same position

#### Scenario: Root counts differ
- **WHEN** normal DAG construction receives different tensor-name and DAG-root counts
- **THEN** construction returns the existing root-count rejection before it creates `LoweredCompilation`

#### Scenario: A consumer swaps root map roles
- **WHEN** a compile-fail fixture passes `ForwardNodeIndex` where `NamedRoots` is required
- **THEN** Rust rejects the fixture because the map types differ

#### Scenario: A consumer decomposes lowered output
- **WHEN** a consumer takes ownership of `LoweredCompilation`
- **THEN** it receives named `LoweredParts` fields for the checked state, DAG, declared roots, and forward index

#### Scenario: An explicit host fallback has no DAG roots
- **WHEN** the existing host fallback accepts a nonfatal lower diagnostic and creates an empty DAG
- **THEN** `NamedRoots` is empty and public output remains equal to the current output

### Requirement: Exclusive semantic outcomes
`complete_checks` SHALL return a narrow `SemanticRejection` that contains only effect or linearity errors. The full pipeline boundary SHALL convert this rejection into `PipelineRejection`.

`LayeredCheck` SHALL represent clean, effect-rejected, and linearity-rejected states as exclusive enum variants. No variant SHALL contain both effect and linearity error collections.

#### Scenario: Complete semantic checks accept
- **WHEN** effect and linearity checks accept a typed program
- **THEN** `complete_checks` returns `CheckedCompilation` without an error variant

#### Scenario: Effect checks reject
- **WHEN** effect checks reject a typed program
- **THEN** `SemanticRejection` and `LayeredCheck` expose only effect errors

#### Scenario: Linearity checks reject
- **WHEN** effects accept and linearity checks reject a typed program
- **THEN** `SemanticRejection` and `LayeredCheck` expose only linearity errors

#### Scenario: A caller creates two layered error classes
- **WHEN** a compile-fail fixture tries to create one layered result with both error classes
- **THEN** Rust rejects the fixture because no enum variant has that shape

### Requirement: Authoritative artifact-type oracle
The existing `.venv/bin/python scripts/compiler_pipeline_oracle.py` command SHALL remain the authoritative completion oracle. It SHALL run the proof, root, semantic-outcome, checkpoint, consumer-parity, and compile-fail tests.

#### Scenario: All artifact boundaries hold
- **WHEN** the authoritative oracle runs after all consumers migrate
- **THEN** all positive tests, negative tests, compile-fail tests, and parity tests pass

#### Scenario: A weak artifact shape returns
- **WHEN** a negative fixture restores a Boolean success marker, raw root tuple, or parallel layered error vectors
- **THEN** the authoritative oracle fails and identifies the broken boundary
