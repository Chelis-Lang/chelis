# Serialization

## 1. Text Forms

- `.ch` is UTF-8 Surf source text.
- `.dp` is UTF-8 Deep source text.

Deep canonical printing is defined by `spec/03-deep-syntax.md`. Surf formatting and
desugaring are defined by `spec/02-surf-syntax.md`.

## 2. Binary Shell Metadata

`.chb` is Reef's compiler-private binary metadata artifact. It carries:

- public package metadata;
- exported symbol metadata;
- compiler compatibility metadata; and
- optional cached compiler products.

`.chb` is not a language-level interchange format. A producer and consumer must honor
the compatibility metadata, and no reader may infer cross-version layout compatibility
from the filename alone. Any portable or independently implemented `.chb` contract
requires an explicit versioned schema in this specification.

## 3. Compiler API Wire Compatibility

The explicit wire models in `chelis-compiler-api` define the machine-facing JSON
surface. Compiler-internal Rust types are not wire contracts.

Adding a tagged variant is an additive schema change. A producer may emit a variant
only when the owning semantic behavior is defined. A consumer that does not recognize
an additive variant reports a structured unsupported-variant diagnostic rather than
silently substituting a default.

Transient specialization markers, including `WireRiscOp::OneHot`, must be eliminated
before a backend receives the wire DAG.

## 4. Opaque Invariants At Decode Boundaries

An opaque type invariant is a predicate over the representation fields of the declared
ADT. It is checked at smart-constructor boundaries and at every external decode
boundary.

Any codec that materializes an invariant-carrying opaque value from bytes, JSON, or
another wire payload must call the shared `decode_adt_value` boundary. Decode is
two-pass:

1. **Structural validation:** the constructor is declared and the field arity and types
   match the representation.
2. **Invariant validation:** predicate evaluation completes and returns boolean `true`.

Structural mismatch and invariant violation are distinct failures. Neither yields a
value.

### 4.1 Numeric pre-check

Before predicate evaluation, every numeric representation field is sanity-checked;
every floating field must be finite. NaN and infinity fail decode even when the
predicate does not read that field or host-language comparison behavior would make the
predicate appear true.

A predicate that returns `false`, fails to evaluate (including division by zero or a
domain error), contains an unresolved reference, or returns a non-boolean value is an
invariant failure. Every such failure yields no decoded value.

### 4.2 No repair

Decode never clamps, normalizes, substitutes, or otherwise repairs a violating payload.
The producer owns proof of admissibility; the consumer independently revalidates it.

Nested invariant-carrying values are validated recursively before the enclosing value
is returned.

## 5. External Format Boundary

This chapter does not define third-party formats such as ONNX, safetensors, Arrow, or
DLPack. Their adapters must preserve Chelis's type, dtype, exactness, and invariant
contracts at ingress and egress and must not silently narrow numeric data.
