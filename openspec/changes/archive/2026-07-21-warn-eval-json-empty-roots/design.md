## Context

`chelis eval --json` has two successful-result emission sites. `run_eval_json_emit` serves direct files, Deep files, fallback evaluation, and inline expressions; `run_eval_in_context` serves files routed through the reef-context fast path. Both serialize `EvalResult` directly and print it to stdout without applying the empty-result warning used by their text-mode counterparts.

An evaluation has no observable output only when both `EvalResult.roots` and `EvalResult.transcript` are empty. Empty transcripts are omitted during serialization, so the wire value is `{"roots":[]}`. Stdout is a machine-facing single-document channel and must remain byte-stable; stderr is the existing warning channel. Nullary function declarations remaining non-evaluable is intentional and outside this change.

## Goals / Non-Goals

**Goals:**

- Apply one empty-result diagnostic rule to both JSON dispatch paths.
- Preserve byte-exact JSON stdout and successful exit status.
- Make the condition explicit as empty roots and empty transcript.
- Lock direct-file and reef-context parity with positive and negative tests.
- Synchronize comments and user-facing documentation with shipped behavior.

**Non-Goals:**

- Changing `EvalResult` or its JSON schema.
- Returning a nonzero status for structurally valid input with no result.
- Adding a machine-readable warning field to stdout.
- Auto-invoking nullary functions or changing which declarations are evaluable roots.
- Reopening the package-root diagnosis from chelis#636.

## Decisions

### Determine emptiness from `EvalResult`

The JSON paths will test `result.roots.is_empty() && result.transcript.is_empty()` through shared logic before successful emission. This expresses the real contract directly and avoids inferring semantics from serialized bytes. Reusing `format_eval_result` was rejected because JSON emission must not depend on the human renderer and because the structured fields already provide the complete predicate.

### Reuse the existing stderr warning and exit policy

An empty successful JSON result will emit the existing newline-terminated warning exactly once on stderr and still return success. Adding a JSON field was rejected because it changes the machine wire schema; changing the exit code was rejected because the input is valid and existing consumers rely on exit `0`.

### Preserve serialization and stdout independently

The result will continue through the same `serde_json::to_string` and `println!` path. For an empty result, stdout therefore remains exactly `{"roots":[]}\n`. The warning is emitted only after evaluation succeeds; serialization and evaluation errors continue through their existing error paths without an empty-result warning.

### Test both dispatch paths and neighboring controls

The existing direct-file empty JSON test will be strengthened rather than duplicated. A new reef-context empty JSON test will exercise `run_eval_in_context`. Existing non-empty direct-file and reef-context tests will assert that the no-roots warning is absent. Error coverage will continue to require nonzero status, empty stdout, and the real diagnostic rather than the no-roots warning. These controls prevent a broad `roots.is_empty()` or unconditional-warning implementation from passing.

### Keep documentation claims channel-specific

Source comments that currently describe deliberate JSON suppression will be updated. Public documentation will state that empty JSON remains on stdout while the human/operator breadcrumb is on stderr; it will not claim that stdout-only consumers can distinguish the condition.

## Risks / Trade-offs

- **Some callers treat any stderr output as failure despite exit `0`.** → Preserve stdout and status exactly, document the additive warning, and keep the change limited to the already-diagnosed empty-result case.
- **Only checking roots would warn on a transcript-only result.** → Centralize and test the conjunction of empty roots and empty transcript.
- **Only changing one emitter would make behavior package-layout-dependent.** → Require direct-file and reef-context integration coverage.
- **Exact stderr assertions can conflict with unrelated cache diagnostics.** → Assert the warning occurs exactly once while allowing separately specified diagnostics where the reef cache legitimately emits them.

## Migration Plan

No data or schema migration is required. Ship the additive stderr behavior with the CLI tests and documentation in one change. Rollback consists of removing the two JSON warning calls; stdout and evaluator behavior are otherwise untouched.

## Open Questions

None.
