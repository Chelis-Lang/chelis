## Why

The new compiler pipeline uses types for its major phases, but several public artifacts still permit invalid states. Stronger types can remove false success reports, positional swaps, root-name drift, invalid checkpoints, and impossible rejection branches.

## What Changes

- **BREAKING**: Replace `checks_clean: bool` and raw edited modules with an opaque `ValidatedModule` proof type.
- **BREAKING**: Replace raw root-name vectors and root maps with separate opaque domain types.
- **BREAKING**: Replace `LoweredCompilation::into_parts` tuple output with a named `LoweredParts` product.
- Add one smart constructor that binds tensor root names to `Dag::roots()` and rejects count mismatches.
- Require exact root alignment for every successful DAG-backed lower result.
- Represent a selected successful host-backend result with an explicit root-binding mode.
- Reserve the empty-root constructor for that host mode or a selected nonfatal lowering rejection.
- Add a crate-private `DiagnosticCheckpoint` that replaces raw offsets for diagnostic iteration.
- Replace the wide rejection type of `complete_checks` with an effect-or-linearity rejection type.
- Replace the parallel error vectors in `LayeredCheck` with exclusive typed outcomes.
- Extend the source guard through workspace-local helpers, path-specific aliases, higher-order calls, receiver methods, traits, and stage macros.
- Resolve canonical imports inside invoked local macros.
- Classify canonical stages by module and function identity.
- Track repeated loops, labeled control flow, pattern scopes, and uninvoked callable bodies as separate execution paths.
- Exclude unrelated receivers, imports, and external qualified calls that only reuse a canonical stage name.
- Rebase onto the current typed Deep AST and preserve typed ingestion, wire output, root collection, and the realizability manifest observation.
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

- `compiler-pipeline-architecture`: Require proof-bearing edits, exact root alignment, helper-aware ownership checks, target AST integration, named decomposition, and exclusive outcomes.
- `type-inference-architecture`: Require diagnostic iteration to start from a sink-issued checkpoint.

## Impact

The main changes affect `chelis-compiler-api`, `chelis-types`, `chelis-cli`, and `chelis-e2e`. Existing Rust callers of the changed public compiler API must migrate.

The implementation must compile against the current target branch. The migration must retain the existing #912 realizability manifest observation.

The source guard adds `syn` as a direct development dependency. Existing machine-facing schema types remain unchanged.
