# PR #2770 — fresh red-team round 2 of 3

**Verdict: satisfied at `6f4c0aa8090bdc5f875da1f20d6bc79a5ad4caf0`.** No in-scope P0, P1, P2, or P3 finding. I reviewed the supplied clean worktree myself. The merged change from round 1 is the native worker receipt repair from PR #2771; I did not review the unrelated macOS-nightly edits.

## Findings

None. No out-of-scope defect was identified in this pass.

## Claim coverage and evidence

1. **Concurrent, source-bound join.** `scripts/capacity_census_typed.py` dispatches compiler-JSON collection followed by native verification on the binding worker, while the calling thread verifies wire. Both targets are resolved inside this worktree's `target/` and rejected if equal, nested, or foreign. The join waits for both results, checks their exact witness types and common source identity against the initial and current source, validates wire and native, then finalizes and validates compiler-JSON authority. The focused controls exercised actual overlap, delayed completion, single and dual failures, changed source, saved wire receipt, and target overlap. My temporary adversarial probe also exercised a successful separate-target join, native worker failure, native source mismatch, and wire failure; none of the failures reached finalization.

2. **Native worker packet after the merge.** `scripts/capacity_census_native_execution.py` projects `TestExecution` to exactly `command`, `selected`, `executed`, and `output_sha256`, rejecting either wire probe field. The parent independently recomputes those four values from the logged Cargo-selected binary and libtest lifecycle and requires exact packet equality before accepting a group. The positive worker control passed. My probe rejected target-only, digest-only, and paired wire probe contamination; the four-field positive packet passed. This verifies the new packet interaction, while actual completion of all six native groups remains subject to the separate full acceptance run.

3. **Selected, mutation, and control verdicts.** The PR changes orchestration and the packet projection; the binding worker still calls the existing compiler-JSON Python and Rust selections, construction and MIR controls, and native authority collector. The wire thread still calls the existing schema, codec, cache, invocation, and acceptance verifier. The native parent still requires 37 exact passed cases and 50 captures. `spec/design/dtype_semantics.md` §C6 requires live wire and binding authority with no metadata shortcut; the Rust binding bridge still invokes `bindings-discovery` and checks the resulting report shape. Six supervised wire-selection tests passed. The CLI rejected supplied wire rustdoc JSON and binding discovery without registration provenance before any build.

## Commands run

From `/Users/robertronan/chelis-worktrees/pr2770-merge-interaction-20260929`:

- `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_typed test_capacity_census_compiler_json test_capacity_census_native_execution.MatrixContractTests test_capacity_census_native_execution.SourceIdentityTests test_capacity_census_native_execution.RetainedCaptureTests` — 51 passed.
- `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_wire_runner.SelectedWireProbe test_capacity_census_wire_runner.SupervisedUnittest` — 6 passed.
- `PYTHONPATH=scripts .venv/bin/python target/round2_join_probe.py` — all five printed adversarial checks passed; the temporary probe was removed afterward. Its first run had a probe-harness type error; I corrected the temporary probe and reran it successfully.
- `.venv/bin/python scripts/capacity_census_typed.py bindings-discovery` — exit 1, expected missing-provenance rejection.
- `.venv/bin/python scripts/capacity_census_typed.py wire --rustdoc-json /tmp/fake-rustdoc.json` — exit 1, expected supplied-artifact rejection.

An initial whole-module unittest invocation was not a valid bounded control: `NativeExecutionIntegration.setUpClass` requires `CHELIS_NATIVE_EXECUTION_TARGET` and would start the cold native build. I used the listed focused classes instead.

## Unvalidated and final state

I did not run the cold full binding/native Cargo acceptance, actual Rust mutation matrix, required CI, or hosted heavy-e2e; the brief says full acceptance is running separately. No claim about those terminal results is made here.

Final supplied worktree: **VERDICT: FREE**, head `6f4c0aa8090bdc5f875da1f20d6bc79a5ad4caf0`, clean with zero modified, staged, untracked, or unmerged paths; lease free and no scoped processes. Checked with `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/pr2770-merge-interaction-20260929/target` at `2026-09-29T18:03:39Z`. I remain available for verification if a repair is requested.
