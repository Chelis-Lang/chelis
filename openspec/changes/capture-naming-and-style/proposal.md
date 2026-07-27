## Why

The Chelis ecosystem's naming and style conventions are defined in `spec/01-nomenclature.md`
but have no OpenSpec capability spec that records them as testable requirements. Capturing
them as the current truth lets the `chelis lint` gate, formatter, and backends be validated
against explicit normative statements with positive and negative parity.

## What Changes

- Introduce a `naming-and-style` capability that records the hard language constraints on
  identifiers, the settled filesystem/identifier/module/function/documentation conventions,
  and the enforcement contract of the `chelis lint` tool and built-in style gate.
- Capture the Surf lexer case-split and its single-letter value override, the closed Deep
  tag vocabulary, module-path lowering, and backend/formatter identifier fidelity.
- Capture the lint severity model (blocking vs advisory), the `--allow-style-violations`
  and `CHELIS_STYLE_GATE_DISABLE` escape hatches, and the opaque-domain-construction rule.

## Capabilities

### New Capabilities
- `naming-and-style`: identifier grammar, filesystem/manifest naming, Surf/Rust/Python
  identifier conventions, module ladders, function-naming patterns, documentation naming,
  test naming, and the `chelis lint` enforcement contract.

### Modified Capabilities

## Impact

- Source: `spec/01-nomenclature.md` (read-only) is the authority for this capability.
- Enforcement surface: `crates/chelis-lint`, `crates/chelis-surf` (lexer/formatter),
  `crates/chelis-deep` (validate/printer), C/HIP backend symbol emission.
- No code changes; this change only records current behavior as an OpenSpec capability spec.
