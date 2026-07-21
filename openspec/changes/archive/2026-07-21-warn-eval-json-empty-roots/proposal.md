## Why

`chelis eval --json --file` returns successful `{"roots":[]}` output without any diagnostic when an input has neither evaluable roots nor transcript output. The text path already emits a warning, so the JSON asymmetry hides a common authoring mistake and contributed to the package-evaluation misdiagnosis in chelis#636; chelis#639 tracks closing that UX gap without breaking the JSON wire contract.

## What Changes

- Emit the existing no-evaluable-roots warning on stderr in JSON mode when both `EvalResult.roots` and `EvalResult.transcript` are empty.
- Apply the behavior to both the legacy/direct-file JSON path and the reef-context JSON fast path.
- Preserve successful exit status `0` and byte-exact stdout `{"roots":[]}\n` for the empty result.
- Preserve existing behavior for non-empty results and evaluation errors, including keeping stdout a single parseable JSON document.
- Add positive and negative CLI coverage for both dispatch paths and synchronize affected CLI documentation.

## Capabilities

### New Capabilities

- `eval-json-diagnostics`: Defines the stderr, stdout, and exit-status contract for empty and non-empty `chelis eval --json` results across direct-file and reef-context dispatch.

### Modified Capabilities

None.

## Impact

- CLI emission logic in `crates/chelis-cli/src/main.rs`.
- Integration coverage in `crates/chelis-cli/tests/cli.rs` and `crates/chelis-cli/tests/eval_in_reef_context.rs`.
- User-facing and investigation documentation describing empty-root evaluation behavior.
- No JSON schema, evaluator semantics, dependency, or exit-code changes; compatibility impact is limited to one new stderr warning for successful empty JSON results.
