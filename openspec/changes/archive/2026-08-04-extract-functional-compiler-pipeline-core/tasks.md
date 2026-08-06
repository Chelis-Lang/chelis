## 1. Baseline and Negative Evidence

- [x] 1.1 Add a compiler-only fixture that imports every current pipeline artifact and function through `chelis_compiler_api::pipeline`.
- [x] 1.2 Add accepted Reef fixtures that freeze checked package, archive, schema, and hash output before the migration.
- [x] 1.3 Add rejected Reef fixtures that freeze exact type, effect, and linearity error text and order.
- [x] 1.4 Add dependency-guard test stubs for the approved graph, a forbidden direct dependency, a forbidden transitive path, and an unknown dependency.
- [x] 1.5 Add source-guard test stubs for the five roots and a planted direct Reef semantic sequence.
- [x] 1.6 Add core compile-fail doctest stubs for private success construction, rejection access, and swapped root-map roles.
- [x] 1.7 Add cancellation parity fixtures for direct lower functions and dynamic pipeline goals.
- [x] 1.8 Add an in-memory documentation-guard fixture that rejects a false current `#![no_std]` claim.

## 2. Core Crate and Dependency Boundary

- [x] 2.1 Add unpublished `chelis-pipeline-core` workspace and dependency entries.
- [x] 2.2 Give the core exactly the approved direct production dependencies.
- [x] 2.3 Implement the manifest and resolved-graph dependency guard.
- [x] 2.4 Make all positive and negative dependency-guard fixtures pass.
- [x] 2.5 Add the owned expanded-Deep carrier without parser, macro, schema, Reef, backend, or cache dependencies.
- [x] 2.6 Add core unit tests for carrier ownership and empty expanded Deep.

## 3. Artifact and Root Migration

- [x] 3.1 Move prepared analysis, checked compilation, and lowered compilation artifacts into the core.
- [x] 3.2 Move `IrName`, root-name collections, root metadata, declared roots, and the forward-node index into the core.
- [x] 3.3 Move exact root alignment and explicit empty-root construction into the core.
- [x] 3.4 Move `SemanticRejection` and add the narrow core lower error.
- [x] 3.5 Re-export moved public artifacts through `chelis_compiler_api::pipeline`.
- [x] 3.6 Keep `compose_checked` available only to required cross-crate adapters.
- [x] 3.7 Make core and facade compile-fail doctests pass.
- [x] 3.8 Run root-name, tuple-root, root-count, and map-role parity tests.

## 4. Semantic Transition Migration

- [x] 4.1 Move isolated and contextual type-analysis transitions into the core.
- [x] 4.2 Move isolated and contextual effect and linearity transitions into the core.
- [x] 4.3 Move isolated, contextual, and library lower transitions into the core.
- [x] 4.4 Keep `PipelineRequest`, `PipelineGoal`, `PipelineOutcome`, `PreparationError`, and `PipelineRejection` in the facade.
- [x] 4.5 Map core semantic and lower errors to the current facade rejection variants.
- [x] 4.6 Preserve before-and-after cancellation checks in every current facade entry point.
- [x] 4.7 Preserve facade selection of `LoweringMode` and keep backend selection outside the core.
- [x] 4.8 Migrate compiler API, edit, cache, context, CLI, and E2E adapters without public output changes.
- [x] 4.9 Run all compiler API, CLI, edit, cache, root, and E2E parity fixtures.

## 5. Reef Migration

- [x] 5.1 Add the core dependency to `chelis-reef` without a reverse dependency.
- [x] 5.2 Migrate `checked_program_with_effects` to the owned core carrier and semantic transitions.
- [x] 5.3 Preserve the linked-program guard across the complete core semantic check.
- [x] 5.4 Preserve current Reef adapters for exact type, effect, and linearity error text.
- [x] 5.5 Remove the direct Reef semantic sequence.
- [x] 5.6 Run accepted and rejected Reef package, archive, schema, and constructor-resolution fixtures.

## 6. Guard Activation and Documentation

- [x] 6.1 Expand the source guard to core, compiler API, Reef, CLI, and E2E production roots.
- [x] 6.2 Restrict multi-stage orchestration to the core owner and make the planted Reef duplicate fail.
- [x] 6.3 Update `ARCHITECTURE.md` and the compiler pipeline inventory to remove the Reef exception.
- [x] 6.4 Add the focused `#![no_std]` blocker inventory with direct and transitive blocker classes.
- [x] 6.5 Implement the documentation guard and make its positive and negative fixtures pass.
- [x] 6.6 Add core tests, both doctest suites, dependency guards, source guards, and Reef parity to the compiler pipeline oracle.
- [x] 6.7 Add the core doctest command to the canonical per-PR gate and its command-list tests.

## 7. Acceptance and Review

- [x] 7.1 Format all changed Rust, Python, TOML, and Markdown files with repository tools.
- [x] 7.2 Run `openspec validate extract-functional-compiler-pipeline-core --strict --no-interactive` and correct each structural error.
- [x] 7.3 Run `.venv/bin/python scripts/compiler_pipeline_oracle.py` as the authoritative completion oracle.
- [x] 7.4 Run `.venv/bin/python scripts/gate.py --local` as additional local evidence.
- [x] 7.5 Run a fresh local red-team agent against the specification, code, tests, facade, Reef behavior, and dependency graph.
- [x] 7.6 Correct each confirmed major finding and rerun the authoritative oracle.
- [x] 7.7 Record hosted macOS Smoke, Docs, and changed-crate results for the exact final commit.
  - Hosted run `30914264791` passed for implementation head `1cf91cf8190b4eb9fb4a4f3bf7684aa8a3300b70`.
  - `macOS Smoke`, `Docs`, `Lint and Unit Tests (Linux)`, and `Integration Tests (Linux)` passed.
  - `Workspace Tests (Linux)` covered all four changed crates and passed.
