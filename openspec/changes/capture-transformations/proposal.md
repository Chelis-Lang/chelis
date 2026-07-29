## Why

`spec/06-transformations.md` is the authoritative record of the DAG-to-DAG transformations
(`grad`, `vmap`, `jit`), the optimization passes over RISC DAGs, the rules governing how
transformations compose, and the formal semantics of reverse-mode AD. No OpenSpec capability
records these as testable requirements with negative parity.

## What Changes

- Introduce a `transformations` capability recording `grad` (signature, `wrt`, reverse-mode
  algorithm, gradient accumulation, non-differentiable handling, symbolic-dim adjoints, `match`
  and ADT-argument gradients), `vmap`, and `jit`.
- Capture the optimization passes (constant folding, DCE, CSE, algebraic simplification,
  fusion, memory planning), the composition/commutativity rules, and the pass ordering.
- Capture the transform error conditions and the semantics-preservation requirement.

## Capabilities

### New Capabilities
- `transformations`: `grad` reverse-mode AD and its algorithm, `wrt` selection, gradient types,
  non-differentiable handling, symbolic-dim adjoint construction, static `match`/`if` and
  field-wise ADT gradients, `vmap`, `jit`, the DAG optimization passes, transformation
  composition and ordering, and the transform error contract.

### Modified Capabilities

## Impact

- Source: `spec/06-transformations.md` (read-only) is the authority for this capability.
- Surface: `chelis-ir` (grad/vmap/optimize/specialize), the C/HIP backends, `chelis eval`/Tide
  host transform lowering.
- No code changes; this change records current behavior as an OpenSpec capability spec.
