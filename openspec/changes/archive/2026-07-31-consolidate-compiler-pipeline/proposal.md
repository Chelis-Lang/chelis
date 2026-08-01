## Why

Chelis repeats front-end orchestration across the CLI, compiler API, edit validation, and E2E code. These paths duplicate inference and can disagree on pass order, diagnostics, and acceptance.

## What Changes

- Add one compiler-API pipeline that owns parse, expansion, type checks, effect checks, linearity checks, and optional DAG lowering.
- Add typed outcomes that separate a rejected analysis, a checked program, and a lowered program.
- Derive fitness data and the checked program from one type-inference product on each selected semantic path.
- Migrate compiler-API operations, CLI check and build paths, whole-module edit validation, and E2E compilation to the shared pipeline.
- Keep Reef preparation, style policy, report presentation, exit codes, and target selection outside the semantic pipeline.
- Keep backend emitters as final target-specific correctness boundaries.
- Add parity evidence for accepted output, rejected diagnostics, CLI JSON, exit codes, inferred signatures, lowered DAGs, and root metadata.
- Add a source guard that rejects new production copies of the complete semantic pass sequence.

### Non-Goals

- Do not change Surf, Deep, type, effect, linearity, lowering, or backend semantics.
- Do not change CLI JSON fields, JSON layout, diagnostic text, diagnostic order, severity values, or exit codes.
- Do not change Reef resolution, cache behavior, package linking, or linked-name policy.
- Do not add a common backend emitter trait.
- Do not change generated C, HIP, or Metal source.
- Do not prevent focused unit tests from calling individual compiler passes.

## Capabilities

### New Capabilities

- `compiler-pipeline-architecture`: Define semantic-pipeline ownership, typed phase outcomes, single-product inference, consumer delegation, and parity evidence.

### Modified Capabilities

None. Existing language, backend, Tide, and serialization requirements remain unchanged.

## Impact

The main changes affect `chelis-compiler-api`, `chelis-cli`, and `chelis-e2e`. Smaller changes affect `chelis-types`, `chelis-effects`, and compiler-API edit validation.

The change adds an internal typed compiler API and removes duplicate production orchestration. Existing machine-facing APIs remain compatible.

The authoritative acceptance oracle will compare the shared path with frozen baseline fixtures across compiler-API, CLI, edit, and E2E consumers. The design will name the exact command.
