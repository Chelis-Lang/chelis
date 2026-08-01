## Why

The new compiler pipeline uses types for its major phases, but several public artifacts still permit invalid states. Stronger types can remove false success reports, positional swaps, root-name drift, invalid checkpoints, and impossible rejection branches.

## What Changes

- **BREAKING**: Replace `checks_clean: bool` and raw edited modules with an opaque `ValidatedModule` proof type.
- **BREAKING**: Replace raw root-name vectors and root maps with separate opaque domain types.
- **BREAKING**: Replace `LoweredCompilation::into_parts` tuple output with a named `LoweredParts` product.
- Add one smart constructor that binds tensor root names to `Dag::roots()` and rejects count mismatches.
- Add a crate-private `DiagnosticCheckpoint` that replaces raw offsets for diagnostic iteration.
- Replace the wide rejection type of `complete_checks` with an effect-or-linearity rejection type.
- Replace the parallel error vectors in `LayeredCheck` with exclusive typed outcomes.
- Add positive tests and compile-fail tests for each construction boundary.
- Keep all wire conversions at existing schema boundaries.

### Non-Goals

- Do not change Surf, Deep, type, effect, linearity, lowering, runtime, or backend behavior.
- Do not change CLI output, wire schema, generated code, package behavior, or diagnostic text.
- Do not add root-count wrapper types that cannot prove root alignment.
- Do not add an `EntryName` type because unknown entries intentionally preserve the program.
- Do not add `UnitInterval`, private `ContextHash` storage, or `StdLibCacheKey` in this change.
- Record a separate API review before a future `UnitInterval` change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `compiler-pipeline-architecture`: Require proof-bearing edit success, typed root artifacts, named decomposition, and exclusive semantic outcomes.
- `type-inference-architecture`: Require diagnostic iteration to start from a sink-issued checkpoint.

## Impact

The main changes affect `chelis-compiler-api`, `chelis-types`, `chelis-cli`, and `chelis-e2e`. Existing Rust callers of the changed public compiler API must migrate.

No dependency changes are necessary. Existing machine-facing schema types remain unchanged.
