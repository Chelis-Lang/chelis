## 1. Tests and Negative Fixtures

- [x] 1.1 Add a compile-fail doctest that rejects direct `ValidatedModule` construction.
- [x] 1.2 Add an accepted edit test that compares `ValidatedModule::as_exprs()` with the checked module.
- [x] 1.3 Add type, effect, and linearity tests that return no validation proof after rejection.
- [x] 1.4 Add root tests for ordered tuple names, declared roots, and forward load aliases.
- [x] 1.5 Add a root mismatch test that expects `PipelineRejection::RootCount` without a partial map.
- [x] 1.6 Add a compile-fail doctest that rejects a `ForwardNodeIndex` and `NamedRoots` swap.
- [x] 1.7 Add clean, effect, and linearity tests for the narrow semantic result.
- [x] 1.8 Add a compile-fail doctest that rejects one layered result with both error classes.
- [x] 1.9 Add checkpoint tests that include new diagnostics and exclude earlier diagnostics.
- [x] 1.10 Freeze CLI and E2E parity fixtures for clean, effect, linearity, root, and host-fallback outputs.
- [x] 1.11 Add a negative test where strict successful lowering returns an empty DAG for nonempty `TensorRootNames`.
- [x] 1.12 Add a positive test where successful lowering aligns an empty DAG with empty `TensorRootNames`.
- [x] 1.13 Add source-guard negative tests for helper composition and multi-level helper chains.
- [x] 1.14 Add a source-guard positive test for a focused helper that reaches one semantic stage.
- [x] 1.15 Add target-branch integration evidence for the typed Deep name variant and the #912 realizability manifest observation.
- [x] 1.16 Add positive tests for selected successful host-backend output and an accepted nonfatal rejection.
- [x] 1.17 Add repeated-helper, qualified-helper, and unrelated-receiver source-guard tests.
- [x] 1.18 Add typed `Node` root parity and complete wire-shape tests.
- [x] 1.19 Add a CLI test that rejects a bare runtime name during typed `.dp` formatting.
- [x] 1.20 Add cross-file, import, alias, callable, trait, macro, and return-path source-guard tests.
- [x] 1.21 Add typed, parenthesized, branch-assigned, higher-order, and lexical-shadow source-guard tests.
- [x] 1.22 Add executable compile-fail evidence that rejects a raw diagnostic offset.
- [x] 1.23 Add module-identity, typed-receiver method, qualified-method, and zero-iteration loop source-guard tests.
- [x] 1.24 Add repeated-loop, labeled-control, pattern-shadow, and imported-macro-alias source-guard tests.
- [x] 1.25 Add branch-result, array-iterable, import-alias, type-alias, nested-macro, and stable-loop source-guard tests.
- [x] 1.26 Lock typed Deep properties, producer verification, and identity validation in the authoritative oracle.
- [x] 1.27 Lock typed Deep lint, declaration parsing, and trace traversal in the authoritative oracle.
- [x] 1.28 Lock typed Deep authoring and opaque-value decode behavior in the authoritative oracle.
- [x] 1.29 Lock strict invalid-tag rejection and nested structural-list acceptance in the authoritative oracle.
- [x] 1.30 Lock lenient diagnostics, generic ADT parsing, and typed macro expansion in the authoritative oracle.
- [x] 1.31 Lock the exact upper-consumer guard roots and the issue #1012 Reef exception.

## 2. Edit Proof and Semantic Rejection

- [x] 2.1 Add opaque `ValidatedModule` accessors in `crates/chelis-compiler-api/src/fragment.rs`.
- [x] 2.2 Make `check_whole_module_edit` return `ValidatedModule` only after all semantic checks accept.
- [x] 2.3 Remove `checks_clean` and raw edited-module fields from the edit success reports.
- [x] 2.4 Migrate compiler API edit consumers to `as_exprs()` or `into_exprs()`.
- [x] 2.5 Add `SemanticRejection` with effect and linearity variants.
- [x] 2.6 Change `complete_checks` to return `SemanticRejection`.
- [x] 2.7 Convert `SemanticRejection` to `PipelineRejection` only at full pipeline boundaries.
- [x] 2.8 Remove impossible semantic `unreachable!` arms from fragment, context, cache, CLI, and layered callers.

## 3. Typed Root Artifacts

- [x] 3.1 Add opaque `IrName`, `AllRootNames`, and `TensorRootNames` types with narrow accessors.
- [x] 3.2 Add opaque `NamedRoots` and `ForwardNodeIndex` types with distinct lookup and conversion methods.
- [x] 3.3 Change `RootMetadata` fields and accessors to use the root collection types.
- [x] 3.4 Add the exact `NamedRoots` alignment constructor and the explicit empty host-fallback constructor.
- [x] 3.5 Route root-count failures through the alignment constructor before `LoweredCompilation` construction.
- [x] 3.6 Build `ForwardNodeIndex` from `NamedRoots` plus internal load aliases.
- [x] 3.7 Add non-exhaustive `LoweredParts` and return it from `LoweredCompilation::into_parts`.
- [x] 3.8 Migrate compiler API tuple consumers to named `LoweredParts` fields.
- [x] 3.9 Migrate `chelis-e2e` to named lowered fields and explicit `IrName` conversion.
- [x] 3.10 Rename CLI tensor-root locals that currently use `all_root_names`.
- [x] 3.11 Keep raw string and integer conversion inside existing compiler API schema adapters.
- [x] 3.12 Route every DAG-backed lower result and empty DAG through `NamedRoots::aligned`.
- [x] 3.13 Use `NamedRoots::empty` only for a selected host-backend result or an accepted nonfatal rejection.

## 4. Diagnostic, Layered, Guard, and Target States

- [x] 4.1 Add crate-private `DiagnosticCheckpoint` with a private offset.
- [x] 4.2 Add `DiagnosticSink::checkpoint` and `DiagnosticSink::iter_since`.
- [x] 4.3 Replace the body-inference raw offset with a checkpoint.
- [x] 4.4 Remove `DiagnosticSink::iter_from` and remove `len` if no caller remains.
- [x] 4.5 Replace `LayeredCheck` fields with clean, effect-rejected, and linearity-rejected variants.
- [x] 4.6 Migrate layered check and build paths to the exclusive variants.
- [x] 4.7 Migrate CLI report assembly without a JSON or score change.
- [x] 4.8 Extend the source guard with a local call graph and fixed-point stage propagation.
- [x] 4.9 Rebase onto the current target branch and replace removed `Atom::Symbol` use with the current typed AST API.
- [x] 4.10 Resolve the compiler conflict without deleting the #912 realizability manifest observation.
- [x] 4.11 Preserve source-guard call multiplicity and resolve qualified local helper paths.
- [x] 4.12 Restrict direct stages to known paths and imported aliases.
- [x] 4.13 Preserve typed `.dp` ingestion and add typed `Node` root collection.
- [x] 4.14 Preserve the complete target wire bridge for typed `Node` values.
- [x] 4.15 Build one workspace source inventory and resolve full crate-local callable identities.
- [x] 4.16 Track terminated paths and exclude uninvoked callable bodies from enclosing functions.
- [x] 4.17 Resolve block imports, glob imports, nested functions, closures, and indirect callable expressions.
- [x] 4.18 Keep alias bindings on execution paths and substitute higher-order callable arguments.
- [x] 4.19 Classify full stage identities, resolve typed receiver methods, and preserve zero-iteration loop paths.
- [x] 4.20 Calculate loop fixed points, track labeled control, bind branch patterns, and resolve macro imports.
- [x] 4.21 Preserve abstract values, known array iterations, full aliases, and nested macro scopes.
- [x] 4.22 Preserve typed Deep verification consumers through one complete transitional list bridge.
- [x] 4.23 Preserve structural identity validation through a complete typed `Node` view.
- [x] 4.24 Preserve Deep lint and trace traversal across complete typed `Node` values.
- [x] 4.25 Preserve Deep authoring and opaque-value decode behavior across typed `Node` values.
- [x] 4.26 Preserve strict invalid-tag rejection after typed fallback construction.
- [x] 4.27 Preserve malformed checker diagnostics at the lenient fragment boundary.
- [x] 4.28 Preserve generic ADT parsing and internal macro expansion across typed values.

## 5. Documentation and Deferred Review

- [x] 5.1 Update pipeline and fragment Rust documentation with the new proof and root boundaries.
- [x] 5.2 Update `docs/investigations/compiler_pipeline_inventory.md` with the final consumer map.
- [x] 5.3 Write `docs/investigations/fitness_unit_interval_api_review.md` with public construction, mutation, and wire findings.
- [x] 5.4 Record a separate recommendation for `UnitInterval` without a fitness code change.
- [x] 5.5 Document the Rust API migration from Boolean reports and tuple decomposition.
- [x] 5.6 Update the inventory with these review findings and remove the obsolete no-actionable-finding claim.

## 6. Strict and Authoritative Validation

- [x] 6.1 Add the new tests and compiler-API doctests to `scripts/compiler_pipeline_oracle.py`.
- [x] 6.2 Update `scripts/test_compiler_pipeline_oracle.py` for the new command list and failure cases.
- [x] 6.3 Run `cargo fmt --all -- --check` and correct all format errors.
- [x] 6.4 Run `openspec validate strengthen-pipeline-artifact-types --strict --no-interactive`.
- [x] 6.5 Run `.venv/bin/python scripts/compiler_pipeline_oracle.py` as the authoritative completion oracle.
- [x] 6.6 Keep completion claims inactive until the authoritative oracle passes.
- [x] 6.7 Run `cargo fmt --all -- --check` and strict OpenSpec validation after review remediation.
- [x] 6.8 Run the target-branch build and `.venv/bin/python scripts/compiler_pipeline_oracle.py` after the rebase.
- [x] 6.9 Run `.venv/bin/python scripts/gate.py --local` as additional target-branch evidence.
  - The gate stopped on three `infer_recursion_depth_guard` tests with signal 10.
  - Current `origin/main` reproduced all three failures in a separate worktree.
  - The other 1,010 `chelis-types` tests passed.
- [x] 6.10 Add the real CLI host-backend regression test to the authoritative oracle.
- [x] 6.11 Add the raw diagnostic-offset compile-fail fixture to the authoritative oracle.
- [x] 6.12 Run the current-target local gate and authoritative oracle after all typed Deep target corrections.
  - The authoritative oracle passed with the expanded typed Deep and macro coverage.
  - After `main` moved again, the oracle passed against `fe9e5a6ad35ab0c95720cbb99c184aebba55f71c`.
  - The current-target local gate passed for all six changed crates.
- [x] 6.13 Add compiler-API doctests and the raw-checkpoint fixture to the canonical local and hosted gate.
  - All 29 gate unit tests passed.
  - Five compiler-API compile-fail doctests and one regular doctest passed.
  - The raw-checkpoint fixture passed its exact diagnostic checks.
  - The complete local gate passed after the correction.

## 7. Adversarial Validation

- [x] 7.1 Run a fresh local red-team agent against the specifications, code, tests, and public API.
- [x] 7.2 Add and run each missing positive or negative fixture that the red team identifies.
- [x] 7.3 Correct each confirmed finding and record the disposition of rejected high-severity findings.
- [x] 7.4 Run the authoritative compiler pipeline oracle again after all corrections.
- [x] 7.5 Run a fresh local red-team agent after the rebase and review remediation.
- [x] 7.6 Correct each confirmed finding and rerun the target-branch oracle.
- [x] 7.7 Run a new fresh local red-team agent after the second remediation.
- [x] 7.8 Correct the confirmed source-guard, oracle, and dependency-documentation findings.
- [x] 7.9 Run the target-branch oracle after the third remediation.
- [x] 7.10 Run a final fresh local red-team agent against the corrected tree.
- [x] 7.11 Correct the path-alias, higher-order, checkpoint-test, and documentation findings.
- [x] 7.12 Run the target-branch oracle after the fourth remediation.
- [x] 7.13 Run a final fresh local red-team agent after the fourth remediation.
- [x] 7.14 Correct the stage-identity, receiver-method, loop-path, and documentation findings.
- [x] 7.15 Run the target-branch oracle after the fifth remediation.
- [x] 7.16 Run a final fresh local red-team agent after the fifth remediation.
- [x] 7.17 Correct the loop, control-flow, pattern-scope, macro-import, and documentation findings.
- [x] 7.18 Rebase onto the current remote `main` after the target moved.
  - The target moved again after the first hosted push. The branch now uses `fe9e5a6ad35ab0c95720cbb99c184aebba55f71c`.
- [x] 7.19 Run the authoritative oracle after the sixth remediation and target rebase.
- [x] 7.20 Run a final fresh local red-team agent against the exact rebased tree and require a PASS verdict.
  - A fresh local Pi subagent reviewed exact commit `2ed580dabfe3f23525f013bada28b67e04f408c6`.
  - The review returned PASS with no implementation or specification findings.
- [x] 7.21 Correct the guard-scope and continuous compile-fail defects from the PR review.
  - All 75 source-guard tests passed.
  - The authoritative compiler pipeline oracle passed after the correction.

## 8. Hosted Acceptance

- [x] 8.1 Record hosted CI results separately from local oracle results.
  - Hosted runs `30762284990`, `30762284996`, and `30762284986` passed for exact head `57dfd607a25f8aeb6a2b7a874735845d0165e056`.
  - The hosted results remain separate from the local oracle and gate logs.
- [x] 8.2 Require green macOS Smoke, Docs, and changed-crate jobs before final acceptance.
  - `macOS Smoke`, `Docs`, `Lint and Unit Tests (Linux)`, and `Integration Tests (Linux)` completed successfully.
  - The Linux integration gate covered the complete workspace, including all six locally detected changed crates.
- [x] 8.3 Confirm that hosted output shows no wire, CLI, backend, runtime, package, or generated-code drift.
  - The integration gate, backend sanitizers, generated C and Metal smoke tests, package checks, and conformance gate passed.
  - The hosted output contained no drift finding for the wire, CLI, backend, runtime, package, or generated-code surfaces.
- [x] 8.4 Confirm that hosted CI executes both compile-fail gate commands on the remediated head.
  - All 13 hosted checks passed for exact head `3c133f2f98dfa5420f845637e47570dcedac7c6e`.
  - Run `30768292087` passed one regular and five compile-fail compiler-API doctests.
  - The same run reported `diagnostic checkpoint compile-fail: PASS`.
