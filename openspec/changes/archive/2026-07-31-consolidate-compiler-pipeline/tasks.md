## 1. Baseline Evidence

- [x] 1.1 Record every semantic sequence in the upper-consumer scope and classify its current phase boundary.
- [x] 1.2 Add accepted compiler-API parity fixtures for type analysis, full checks, and lowering.
- [x] 1.3 Add compiler-API rejection fixtures for parse, type, effect, linearity, and lowering stages.
- [x] 1.4 Add CLI fixtures that freeze JSON bytes, inferred signatures, diagnostics, and exit codes.
- [x] 1.5 Add E2E fixtures that freeze expanded Deep, ordered roots, tuple roots, and DAG mappings.
- [x] 1.6 Add cold, warm, cache-disabled, and invalid Reef parity fixtures.
- [x] 1.7 Add bounded rejection fixtures for recursion groups, binding cycles, and deep finite input.

## 2. Combined Type Analysis

- [x] 2.1 Add positive and negative tests for the closed `TypeAnalysisOutcome` states.
- [x] 2.2 Add a test-only inference-session counter with accepted and rejected fixtures.
- [x] 2.3 Implement one `chelis-types` analysis entry for structural fitness and IR inference.
- [x] 2.4 Derive accepted fitness from `CheckedProgram::infer_stats()` without a second inference session.
- [x] 2.5 Derive rejected fitness from the failed inference product without a second inference session.
- [x] 2.6 Preserve recursion, cycle, stack, diagnostic-order, and fitness-honesty behavior.
- [x] 2.7 Centralize clean fitness construction and remove copied weights from `layered.rs`.

## 3. Compiler Pipeline Core

- [x] 3.1 Add compile-fail tests for illegal access to success products from rejected states.
- [x] 3.2 Add compile-fail tests that keep internal pipeline types outside Serde wire models.
- [x] 3.3 Add `chelis_compiler_api::pipeline` with closed goals and typed rejection stages.
- [x] 3.4 Implement Surf, Deep, expanded-Deep, and contextual preparation adapters.
- [x] 3.5 Implement `TypeAnalysis` and `FullCheck` transitions in the required pass order.
- [x] 3.6 Implement `Lower` with checked-program ownership and typed lowering failures.
- [x] 3.7 Derive inferred signatures and canonical root metadata from checked Deep.
- [x] 3.8 Add native-error adapters for compiler API, CLI, edit, and E2E result types.

## 4. Compiler API Consumer Migration

- [x] 4.1 Migrate `compiler::check` to the `TypeAnalysis` goal and preserve its result value.
- [x] 4.2 Migrate `compile_source_scoped`, lower, compile, and evaluation helpers to pipeline goals.
- [x] 4.3 Preserve entry pruning and the host-only nonfatal lowering rule.
- [x] 4.4 Migrate whole-module edit validation and preserve its tagged error variants.
- [x] 4.5 Migrate stdlib context construction and contextual checks to shared transitions.
- [x] 4.6 Migrate layered check and build paths while preserving the monolithic error fallback.
- [x] 4.7 Run compiler-API and cache baseline fixtures after each consumer migration.

## 5. CLI Consumer Migration

- [x] 5.1 Migrate Surf and Deep check paths to the `FullCheck` goal.
- [x] 5.2 Preserve `assemble_check_json`, JSON bytes, inferred signatures, and check exit codes.
- [x] 5.3 Migrate build semantic checks to shared checked and lowered products.
- [x] 5.4 Migrate remaining production helpers that orchestrate two or more semantic stages.
- [x] 5.5 Keep style policy, Reef preparation, target selection, and backend emission in the CLI.
- [x] 5.6 Run the CLI baseline fixtures after each path migration.

## 6. E2E Consumer Migration

- [x] 6.1 Add `chelis-compiler-api` as a normal `chelis-e2e` dependency.
- [x] 6.2 Migrate `compile_surf` to the `Lower` goal.
- [x] 6.3 Replace Surf-tree root reconstruction with canonical pipeline root metadata.
- [x] 6.4 Migrate `check_snippet` to the applicable named pipeline goal.
- [x] 6.5 Remove direct semantic dependencies that no remaining E2E code needs.
- [x] 6.6 Run all E2E baseline fixtures after the migration.

## 7. Recurrence Guard and Oracle

- [x] 7.1 Implement the in-memory source-guard core with positive and negative inventories.
- [x] 7.2 Add the guarded-workspace guard with narrow owner and test exclusions.
- [x] 7.3 Activate the guard after all listed upper consumers delegate.
- [x] 7.4 Add `scripts/compiler_pipeline_oracle.py` as the authoritative completion oracle.
- [x] 7.5 Add Python tests for command selection, failure propagation, and success reporting.
- [x] 7.6 Update `ARCHITECTURE.md` with pipeline ownership, goals, state types, and boundaries.
- [x] 7.7 Lock the exact guard roots and document the Reef dependency exception under issue #1012.

## 8. Acceptance and Review

- [x] 8.1 Format all changed Rust, Python, TOML, and Markdown files with repository tools.
- [x] 8.2 Run `openspec validate --strict consolidate-compiler-pipeline` and correct each error.
- [x] 8.3 Run `.venv/bin/python scripts/compiler_pipeline_oracle.py` as the authoritative oracle.
- [x] 8.4 Run `.venv/bin/python scripts/gate.py --local` as additional local evidence.
- [x] 8.5 Run a fresh-context adversarial review against the specification and executable behavior.
- [x] 8.6 Correct each confirmed finding and rerun the authoritative oracle.
- [x] 8.7 Record hosted CI status.
  - All 13 hosted checks passed for exact head `3c133f2f98dfa5420f845637e47570dcedac7c6e`.
  - Run `30768292087` passed macOS Smoke, Docs, Lint and Unit Tests, and Integration Tests.
- [x] 8.8 Correct the ownership scope after PR review found the under-specified Reef exception.
  - The active specification names the three guarded roots and the issue #1012 exception.
  - The exact-root source-guard test passed.
