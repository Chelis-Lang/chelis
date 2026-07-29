## Why

`spec/04-type-system.md` is the authoritative record of the Chelis type system: primitive
precision types, ADTs, Hindley-Milner inference, named tensor dimensions with no broadcasting,
dimension and rank polymorphism, opaque-type module identity, fitness scoring, the Phase-2a
effect subset, and linearity. No OpenSpec capability records these as testable requirements
with negative parity.

## What Changes

- Introduce a `type-system` capability recording the checked-Deep contract, the active dtype
  set and per-backend support matrix, tensor type algebra, dimension/rank polymorphism rules,
  opaque-type enforcement, precision rules, fitness scoring, effects, and linearity.
- Capture the decided-but-not-yet-implemented numeric value semantics (§9) and checker
  totality (§10) atoms as normative requirements, marking their implementation status
  faithfully.
- Capture the type-error cases (dimension mismatch, precision mismatch, broadcasting,
  opaque violations, rigid-dim collapse, non-exhaustive match, builtin shadowing).

## Capabilities

### New Capabilities
- `type-system`: checked Deep, primitive/tensor/ADT types, HM inference, tensor type algebra
  and named dimensions, dimension/rank polymorphism, opaque types and invariants, precision
  rules and accumulators, runtime shape semantics, fitness scoring, the Phase-2a effect
  subset, linearity, numeric value semantics, and checker totality.

### Modified Capabilities

## Impact

- Source: `spec/04-type-system.md` (read-only) is the authority for this capability.
- Surface: `crates/chelis-types` (infer/unify/builtins), `chelis-ir` lowering, the backend
  dtype matrix, `chelis check`/`build`/`eval` gating.
- No code changes; this change records current and decided behavior as an OpenSpec spec.
