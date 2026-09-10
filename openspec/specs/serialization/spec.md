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

### Requirement: Exact WireDag schema version 10

The compiler-API JSON WireDag surface SHALL carry explicit schema version 10,
and version 10 SHALL be the only accepted version. Missing, versionless,
versions 1 through 9, future, unknown-variant, and best-effort payloads SHALL
fail before IR construction. `Count.axes` SHALL already be the complete
non-empty unique normalized original-axis vector in strictly descending order;
encoder and decoder both reject a noncanonical vector. `Pad.fill` SHALL be a
tagged `ScalarValue` whose dtype and bits exactly match the padded tensor.
Runtime movement and reshape metadata SHALL use the structural `WireRtDim`
carrier. [05-OP-43]'s `Relu` and `ReluAdjoint` SHALL cross the wire as distinct
identities with one and two inputs respectively; every input SHALL have the
output's exact float dtype and dimensions. `ExtentWitness` SHALL retain its
ordered fixed-int64 requirements and provenance. Its mandatory `site` SHALL
be exactly `caller` or `local_expand`, preserving call-entry `load` diagnostics
with parameter context or local `expand` diagnostics with observed-input-node
context respectively. Missing or unknown sites SHALL fail decoding.
`shape_deps` SHALL be u64 references to earlier nodes. All three provenance/dependency node fields
SHALL be explicit, including empty lists and a null span identity.

`CheckedReshapeExtent` SHALL preserve separate scalar-int64 input edges for
the actual computed extent and the required extent, plus diagnostic claim text
and a normalized result-axis position. `CheckedUnitAxis` SHALL reference the
original tensor and an `ExtentWitness` for that exact tensor axis with an
explicit requirement of one; only that axis may refine to one. Missing fields,
wrong input/output types, a different witness axis or tensor, and unrelated
shape refinements SHALL fail encoding and decoding.

#### Scenario: Unknown or older schema fails before IR construction

- **WHEN** a consumer receives a versionless, v1-v9, future-version, or unknown-variant WireDag payload
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

### Requirement: Exact numeric wire and role admission

Execution payloads SHALL carry exact schema version 3. The value codec and
source, reference, extent and report domains SHALL follow the controlling
[serialization chapter §§3.2–3.5](../../../spec/10-serialization.md#32-exact-numeric-value-codec).
General float payloads SHALL carry dtype-tagged fixed-width lowercase IEEE hex
bits, preserving signed zero and every NaN payload; report JSON numbers SHALL
use the chapter's exact fixed-dtype adapters over sealed numeric carriers.

#### Scenario: Every float bit pattern is transportable

- **WHEN** a numeric value crosses a general execution or DAG value boundary
- **THEN** the codec preserves the exact dtype and stored bits, including infinity and signed zero

#### Scenario: Wrong encoding is rejected

- **WHEN** a general float payload uses a JSON number, null, wrong-width bits, or a mismatched dtype
- **THEN** decoding rejects it without guessing, rounding, or supplying a default

#### Scenario: Structural roles do not spread to numeric siblings

- **WHEN** a report value or source numeric literal shares a container with a source coordinate
- **THEN** each leaf retains its own numeric or source-syntax contract and normal admission

#### Scenario: Inferred parameter references retain their scope

- **WHEN** a check report carries an inferred function signature's ordered parameters
- **THEN** every parameter index is a u64 equal to its zero-based position in that list, independent of its name

#### Scenario: Observed extents retain their exact numeric domain

- **WHEN** a compiled-artifact dimension or inferred literal dimension carries an observed extent
- **THEN** it carries a nonnegative exact int64; negative values, alternate dtypes, and int64 overflow are rejected, while an absent named extent remains unresolved

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
