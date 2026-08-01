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

## 4. Diagnostic and Layered States

- [x] 4.1 Add crate-private `DiagnosticCheckpoint` with a private offset.
- [x] 4.2 Add `DiagnosticSink::checkpoint` and `DiagnosticSink::iter_since`.
- [x] 4.3 Replace the body-inference raw offset with a checkpoint.
- [x] 4.4 Remove `DiagnosticSink::iter_from` and remove `len` if no caller remains.
- [x] 4.5 Replace `LayeredCheck` fields with clean, effect-rejected, and linearity-rejected variants.
- [x] 4.6 Migrate layered check and build paths to the exclusive variants.
- [x] 4.7 Migrate CLI report assembly without a JSON or score change.

## 5. Documentation and Deferred Review

- [x] 5.1 Update pipeline and fragment Rust documentation with the new proof and root boundaries.
- [x] 5.2 Update `docs/investigations/compiler_pipeline_inventory.md` with the final consumer map.
- [x] 5.3 Write `docs/investigations/fitness_unit_interval_api_review.md` with public construction, mutation, and wire findings.
- [x] 5.4 Record a separate recommendation for `UnitInterval` without a fitness code change.
- [x] 5.5 Document the Rust API migration from Boolean reports and tuple decomposition.

## 6. Strict and Authoritative Validation

- [x] 6.1 Add the new tests and compiler-API doctests to `scripts/compiler_pipeline_oracle.py`.
- [x] 6.2 Update `scripts/test_compiler_pipeline_oracle.py` for the new command list and failure cases.
- [x] 6.3 Run `cargo fmt --all -- --check` and correct all format errors.
- [x] 6.4 Run `openspec validate strengthen-pipeline-artifact-types --strict --no-interactive`.
- [x] 6.5 Run `.venv/bin/python scripts/compiler_pipeline_oracle.py` as the authoritative completion oracle.
- [x] 6.6 Keep completion claims inactive until the authoritative oracle passes.

## 7. Adversarial Validation

- [x] 7.1 Run a fresh local red-team agent against the specifications, code, tests, and public API.
- [x] 7.2 Add and run each missing positive or negative fixture that the red team identifies.
- [x] 7.3 Correct each confirmed finding and record the disposition of rejected high-severity findings.
- [x] 7.4 Run the authoritative compiler pipeline oracle again after all corrections.

## 8. Hosted Acceptance

- [ ] 8.1 Record hosted CI results separately from local oracle results.
- [ ] 8.2 Require green macOS Smoke, Docs, and changed-crate jobs before final acceptance.
- [ ] 8.3 Confirm that hosted output shows no wire, CLI, backend, runtime, package, or generated-code drift.
