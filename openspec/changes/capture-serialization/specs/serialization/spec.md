## ADDED Requirements

### Requirement: Text serialization forms

`.ch` SHALL be Surf source text in UTF-8 and `.dp` SHALL be Deep source text in UTF-8, with Deep
canonical printing defined by the deep-syntax capability.

#### Scenario: Deep text is UTF-8 canonical

- **WHEN** a `.dp` file is written by `chelis fmt`
- **THEN** it is UTF-8 Deep text in the canonical form defined by the deep-syntax spec

#### Scenario: Surf text is UTF-8

- **WHEN** a `.ch` file is read
- **THEN** it is interpreted as UTF-8 Surf source text

### Requirement: Binary Shell metadata artifact

`.chb` SHALL be the binary Shell metadata artifact used by the Reef package system, carrying
public package metadata, exported symbol metadata, compiler compatibility metadata, and room for
future cached products. Its exact wire layout SHALL remain owned by the implementation and SHALL
NOT be published as a frozen low-level guarantee while the implementation evolves.

#### Scenario: chb carries package and symbol metadata

- **WHEN** a Reef package is built
- **THEN** its `.chb` records public package metadata, exported symbol metadata, and compiler compatibility metadata

#### Scenario: No premature wire-layout guarantee

- **WHEN** documenting `.chb`
- **THEN** the project does not publish a frozen low-level layout guarantee while the format is still expected to evolve

### Requirement: Additive wire-schema compatibility

The compiler-API JSON wire models SHALL be the machine-facing surface, with new tagged variants
(e.g. `WireRiscOp::Gather`) being additive changes producers may emit after the owning behavior
lands. Consumers SHOULD tolerate unknown additive variants and report a clear unsupported-variant
diagnostic; the transient `OneHot` marker SHALL NOT reach backends after specialization.

#### Scenario: Consumer tolerates an unknown additive variant

- **WHEN** a consumer receives a wire variant newer than its build
- **THEN** it reports a clear unsupported-variant diagnostic rather than failing only because the enum grew

#### Scenario: OneHot marker does not reach backends

- **WHEN** specialization completes
- **THEN** the transient `WireRiscOp::OneHot` marker is not delivered to any backend

### Requirement: Decode-boundary invariant revalidation

Any codec that materializes a value of an invariant-carrying opaque type from an external payload
SHALL revalidate the declared invariant at materialization time via the `decode_adt_value`
chokepoint: structural validation (declared constructor, matching representation) then invariant
revalidation, with a mandatory NaN/non-finite pre-check before predicate evaluation. Decode of a
violating payload SHALL be a failure, never a repair.

#### Scenario: Valid payload passes both passes

- **WHEN** a payload matches the declared representation and satisfies the invariant
- **THEN** `decode_adt_value` materializes the value after structural validation and invariant revalidation

#### Scenario: Violating payload is rejected, not repaired

- **WHEN** a payload's fields structurally match but violate the invariant (or contain a NaN in a numeric field)
- **THEN** decode fails and yields no value, never clamping/normalizing the payload into the admissible set

#### Scenario: Structural mismatch is distinct from invariant violation

- **WHEN** a payload's constructor or field arity/type does not match the declared representation
- **THEN** it is a structural decode failure, reported distinctly from an invariant violation
