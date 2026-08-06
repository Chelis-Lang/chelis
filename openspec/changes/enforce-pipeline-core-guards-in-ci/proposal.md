## Why

Review of the pipeline-core extraction found two enforcement gaps.

1. `false_no_std_claim` in `scripts/pipeline_core_documentation_guard.py` is wrong in
   both directions. It **false-positives** on a true statement such as "supports std
   only, not no_std" (the guard would reject an honest inventory), and it
   **false-negatives** on genuine false claims such as "portable to no_std targets" and
   "works in a no_std environment" (the guard lets them through). The guard's one job is
   catching a false current no_std claim, and it misses common phrasings. Its own test
   suite only exercises phrasings that happen to work, so the gap is untested.

2. The three pipeline-core boundary guards — the dependency guard
   (`scripts/pipeline_core_dependency_guard.py`), the documentation guard
   (`scripts/pipeline_core_documentation_guard.py`), and the cross-crate
   pipeline-artifact compile-fail fixture (`scripts/check_pipeline_core_compile_fail.py`
   over `crates/chelis-compiler-api/tests/compile_fail/pipeline_artifacts/`) — run only
   in the manual `scripts/compiler_pipeline_oracle.py`. None appears in
   `scripts/gate.py` or under `.github/`. The pipeline-core doctests and unit tests are
   in the gate, but a forbidden dependency added to `chelis-pipeline-core`, a broken
   facade compile-fail boundary, or a false portability claim passes hosted CI green and
   is caught only if a human runs the oracle.

## What Changes

- Rewrite the documentation guard's false-claim detection so it reliably rejects a
  current no_std support claim regardless of adjective, verb, or environment phrasing,
  and does not reject a true "requires std" statement or a future-target non-goal.
- Add explicit positive and negative tests for the phrasings that currently fail.
- Add the dependency guard, the documentation guard, and the pipeline-artifact
  compile-fail fixture to the per-PR gate `lint-and-unit` stage in `scripts/gate.py`, so
  hosted CI runs all three. Keep the manual oracle as additional evidence.
- Extend `scripts/test_gate.py` so CI cannot drop a gate command the script produces.

### Non-Goals

- Do not change language, compiler, runtime, CLI, backend, package, or generated-code
  behavior.
- Do not add `#![no_std]` support.
- Do not change the guards' subject matter — the approved dependency set, the required
  inventory sections, or the required compile-fail diagnostics — only the documentation
  heuristic's reliability and the enforcement point.
- Do not remove the manual oracle or weaken any existing gate command.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `compiler-pipeline-architecture`:
  - "Standard-library blocker inventory" — the documentation guard reliably detects a
    false current no_std claim across common phrasings and does not reject a true
    requires-std statement.
  - Add "Continuous pipeline-core boundary guard enforcement" — the per-PR gate runs the
    dependency guard, the documentation guard, and the pipeline-artifact compile-fail
    fixture, so hosted CI enforces the boundary the manual oracle already checks.

## Impact

- `scripts/pipeline_core_documentation_guard.py`,
  `scripts/test_pipeline_core_documentation_guard.py` — reliable false-claim detection
  and its positive and negative tests.
- `scripts/gate.py` — add the three guards to the `lint-and-unit` stage; CI calls
  `python3 scripts/gate.py lint-and-unit`, so the additions run in hosted CI without a
  workflow edit.
- `scripts/test_gate.py` — assert the workflow runs the stage and produces the new
  commands.
- `openspec/specs/compiler-pipeline-architecture/spec.md` (delta), current-state docs.

This change modifies contributor-process enforcement and one guard heuristic only. The
numbered specifications retain authority for language, compiler, serialization, backend,
runtime, and package behavior.
