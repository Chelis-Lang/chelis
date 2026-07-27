## Why

`spec/10-serialization.md` records the serialization surface: the UTF-8 text forms (`.ch`,
`.dp`), the `.chb` binary Shell metadata artifact, the compiler-API JSON wire compatibility
policy, and the normative invariant-revalidation rule at decode boundaries. No OpenSpec
capability records these as testable requirements with negative parity.

## What Changes

- Introduce a `serialization` capability recording the text forms, the `.chb` role (with its
  deliberately under-specified wire layout), the additive wire-schema compatibility policy, and
  the decode-boundary invariant-revalidation rule.
- Capture the decode failure cases (structural mismatch, invariant violation, NaN pre-check,
  no-repair rule) as negative-parity scenarios.

## Capabilities

### New Capabilities
- `serialization`: the `.ch`/`.dp` text forms, the `.chb` Shell metadata artifact, the
  compiler-API wire-compatibility policy, and the decode-boundary invariant-revalidation
  contract.

### Modified Capabilities

## Impact

- Source: `spec/10-serialization.md` (read-only) is the authority for this capability.
- Surface: `chelis-compiler-api` (schema, `decode_adt_value`), the Reef `.chb` metadata, and the
  Deep canonical printer.
- No code changes; this change records current behavior as an OpenSpec capability spec.
