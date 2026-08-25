## Why

`chelis build` removes unreachable eval-only-tainted definitions before its authoritative semantic fallback. It can miss an error that `chelis check` reports in the same selected program.

This divergence breaks the front-end gate contract in `spec/04-type-system.md` and leaves chelis#1184 unresolved.

## What Changes

- **BREAKING**: `chelis build` rejects type, effect, and linearity errors in every definition from the selected source or linked package target.
- Complete semantic checks run before eval-only removal and general reachability pruning.
- Build still removes well-typed unreachable eval-only definitions before backend checks and code emission.
- A reachable eval-only builtin still causes the existing `Unsupported` diagnostic.
- `[05-HOST-2]` changes from whole-program backend rejection to rejection in the retained compile target.
- Cold, warm, and cache-disabled builds produce the same acceptance result and diagnostic.
- If structured repairs exist, each build mode preserves the same repair data for the same diagnostic.
- Valid builds retain the same generated code and cache format.
- Positive and negative CLI tests cover direct and transitive eval-only references.

## Non-Goals

- This change does not add runtime termination analysis.
- This change does not change the eval-only builtin roster or evaluator behavior.
- This change does not change Cargo-style target selection for Reef packages.
- This change does not change the separate whole-program `tensor_scan` contract in `spec/05-risc-primitives.md` §3.6.
- This change does not add backend support for an eval-only builtin.
- This change does not define structured repair producers or the SNAFU error architecture.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `type-system`: Require build to complete semantic checks for all definitions in the selected program before reachability pruning.
- `risc-primitives`: Scope eval-only builtin backend rejection to the retained compile target after semantic checks.

## Impact

- Normative specifications: `spec/04-type-system.md` §2.5 and `spec/05-risc-primitives.md` `[05-HOST-2]`.
- Compiler and CLI: `crates/chelis-cli/src/main.rs` and the layered build-check adapter in `chelis-compiler-api`.
- Tests: `crates/chelis-cli/tests/library_cache_oracle.rs` and focused CLI fixtures.
- Documentation: the release notes and the chelis#1184 references.
- Structured suggestions remain downstream of this semantic gate and have no prerequisite edge for core repair types.
- SNAFU can type an operational `Err` result but does not change the `Ok(None)` fallback or pipeline order.
- Runtime behavior does not change.
- Generated code for a valid program does not change.
- The on-disk cache schema and package graph do not change.
