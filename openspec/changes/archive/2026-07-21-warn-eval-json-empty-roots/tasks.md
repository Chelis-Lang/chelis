## 1. Write the Contract Tests First

- [x] 1.1 Strengthen `eval_json_def_only_emits_empty_roots_json` to fail on the current implementation by asserting exit `0`, byte-exact `{"roots":[]}\n` stdout, and exactly one warning line on stderr.
- [x] 1.2 Add a reef-context JSON integration test that fails on the current implementation and asserts the same empty-result stdout, stderr, and exit contract through `run_eval_in_context`.
- [x] 1.3 Add negative-parity coverage proving non-empty direct-file and reef-context results do not emit the warning, transcript-only structured results are not classified as empty, and evaluation failures retain their existing error channel.

## 2. Implement Shared Empty-Result Diagnostics

- [x] 2.1 Add shared structured-result logic that classifies an `EvalResult` as empty only when both `roots` and `transcript` are empty and reuses the existing warning text.
- [x] 2.2 Apply the shared warning logic after successful serialization in both `run_eval_json_emit` and the JSON arm of `run_eval_in_context`, preserving the existing stdout serialization and return values.
- [x] 2.3 Run the new tests red-to-green and verify the neighboring non-empty and error controls remain green.

## 3. Synchronize Public Documentation

- [x] 3.1 Update JSON-emitter comments and CLI-facing documentation to describe `{"roots":[]}` on stdout plus the additive stderr breadcrumb, removing claims that JSON deliberately suppresses it.
- [x] 3.2 Reconcile `CHANGELOG.md`, `docs/investigations/cli_eval_empty_roots_diagnosis.md`, and any book text that describes empty-root evaluation without changing nullary-function or package-evaluation semantics.

## 4. Validate the Change

- [x] 4.1 Run `cargo fmt --all -- --check` and the focused `chelis-cli` direct-file and reef-context integration tests in an isolated `CARGO_TARGET_DIR`.
- [x] 4.2 Run the repository acceptance oracle `python3 scripts/gate.py --local` with an isolated `CARGO_TARGET_DIR`; require every selected stage to pass.
- [x] 4.3 Run `openspec validate warn-eval-json-empty-roots --strict --no-interactive`, then perform the required independent adversarial review and resolve every verified finding.
