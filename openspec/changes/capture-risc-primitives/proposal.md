## Why

`spec/05-risc-primitives.md` is the authoritative record of the irreducible tensor operation
set: the RISC primitives, their types and AD adjoints, the derived built-ins that lower to
them, and the standard ML-op lowerings. No OpenSpec capability records these as testable
requirements with negative parity.

## What Changes

- Introduce a `risc-primitives` capability recording the RISC philosophy (small primitive set,
  no broadcasting, primitives-are-functions, borrow-typed inputs), the Tier-1 primitives and
  their AD adjoints, the Tier-2 derived built-ins, and the standard lowerings.
- Capture division semantics (float-only `div`, `floor_div`/`trunc_div`, zero-divisor trap),
  reduction axis/accumulator rules, windowed reductions, movement ops and runtime bounds, the
  effectful primitives with seed determinism, scatter determinism/AD policy, and host-only
  builtins.
- Capture the decided unsupported-case-response and observation/formatting contracts.

## Capabilities

### New Capabilities
- `risc-primitives`: the Tier-1 RISC primitives and AD adjoints, Tier-2 derived built-ins,
  division and reduction semantics, windowed reductions, movement and memory ops, effectful
  primitives, scatter determinism, host-only builtins, standard lowerings, AD completeness,
  reference-implementation oracle, and the unsupported-case and observation contracts.

### Modified Capabilities

## Impact

- Source: `spec/05-risc-primitives.md` (read-only) is the authority for this capability.
- Surface: `chelis-ir` (RiscOp, eval, grad), the C/HIP/Metal backends, and the derived-builtin
  lowering.
- No code changes; this change records current and decided behavior as an OpenSpec spec.
