## Why

The canonical semantic pipeline still lives in `chelis-compiler-api`, which also depends on Reef, backends, caches, schemas, and wire libraries. Cargo cannot enforce the dependency boundary, and `chelis-reef` retains one direct semantic sequence as a documented exception.

## What Changes

- Add an unpublished `chelis-pipeline-core` workspace crate below `chelis-compiler-api` and `chelis-reef`.
- Start the core boundary at an owned, expanded Deep program.
- Move typed phase artifacts and semantic phase functions into the core crate.
- Move canonical root derivation and exact root alignment into the core crate.
- Keep source preparation, dynamic goal dispatch, cancellation presentation, host policy, and backend policy in `chelis-compiler-api`.
- Re-export the current public API through `chelis_compiler_api::pipeline`.
- Move the Reef package artifact path to the core transitions and remove its direct semantic sequence.
- Extend dependency guards and source guards to enforce the new boundary.
- Add core compile-fail tests to the compiler pipeline oracle.
- Record the lower-crate dependencies that block future `#![no_std]` support.

### Non-Goals

- Do not add `#![no_std]` support.
- Do not change language semantics or compiler pass order.
- Do not change diagnostics, CLI output, wire bytes, packages, backends, runtime behavior, or generated code.
- Do not move parsing, Surf desugaring, macro expansion, entry pruning, caches, or Reef resolution into the core crate.
- Do not add generic stage traits or a backend abstraction.
- Do not change current compiler API import paths.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `compiler-pipeline-architecture`: Replace the upper-crate owner with a dependency-bottom core, remove the Reef exception, and enforce the new crate boundary.

## Impact

The main code changes affect `chelis-compiler-api`, `chelis-reef`, and the workspace dependency graph. Smaller changes affect the compiler pipeline oracle, architecture guards, documentation, and compile-fail tests.

The new crate depends only on lower compiler crates. Existing consumers continue to use `chelis_compiler_api::pipeline` without a direct core dependency.

This change modifies implementation architecture only. The numbered specifications retain authority for language, compiler, serialization, backend, runtime, and package behavior.
