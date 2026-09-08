# serialization

## Purpose

Define Chelis serialization: the `.ch`/`.dp` UTF-8 text forms, the `.chb` binary Shell metadata
artifact with its implementation-owned wire layout, the exact-version compiler-API JSON
wire policy, and the normative decode-boundary invariant-revalidation contract for
invariant-carrying opaque types.

**Source:** captured from [`spec/10-serialization.md`](../../../spec/10-serialization.md).

## Requirements

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

### Requirement: Exact WireDag schema version 7

The compiler-API JSON WireDag surface SHALL carry explicit schema version 7,
and version 7 SHALL be the only accepted version. Missing, versionless,
versions 1 through 6, future, unknown-variant, and best-effort payloads SHALL
fail before IR construction. `Count.axes` SHALL already be the complete
non-empty unique normalized original-axis vector in strictly descending order;
encoder and decoder both reject a noncanonical vector. `Pad.fill` SHALL be a
tagged `ScalarValue` whose dtype and bits exactly match the padded tensor.
Runtime movement and reshape metadata SHALL use the structural `WireRtDim`
carrier. [05-OP-43]'s `Relu` and `ReluAdjoint` SHALL cross the wire as distinct
identities with one and two inputs respectively; every input SHALL have the
output's exact float dtype and dimensions.

#### Scenario: Unknown or older schema fails before IR construction

- **WHEN** a consumer receives a versionless, v1-v6, future-version, or unknown-variant WireDag payload
- **THEN** decoding fails before any IR node is materialized

#### Scenario: Noncanonical Count axes are not rewritten

- **WHEN** an encoder or decoder receives empty, duplicate, increasing, source-order, or out-of-range `Count.axes`
- **THEN** it rejects the payload rather than canonicalizing or guessing

#### Scenario: Pad fill is exact and tagged

- **WHEN** `WireRiscOp::Pad` crosses the wire
- **THEN** its fill is a dtype-tagged `ScalarValue` matching the tensor, never a raw number or inferred dtype

#### Scenario: ReLU identities survive the wire

- **WHEN** a `Relu` or `ReluAdjoint` node crosses the compiler-API boundary
- **THEN** its dedicated identity, exact arity, float dtype, and dimensions are validated before IR construction rather than reconstructed as an extrema operation

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
