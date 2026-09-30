# #2356 scoped Linux Extended Validation: local implementation report

Date: 2026-09-29

## State

- Worktree: `/Users/robertronan/chelis-worktrees/issue-2356-scoped-dispatch`
- Branch: `agent/2356-scoped-dispatch`
- Base fetched before worktree creation: `e16b2f795045e26a366cfbba570b73a0c6278ea3`
- Local commit: `070ffb7c1332cf4ae5f80c4b6e466b3681c59efd` (`fix(ci): scope extended validation dispatch and baselines`)
- Worktree status: clean (`git status --porcelain=v1` produced no output). Nothing was pushed; no PR was opened; #2356 remains open and assigned to `rlronan`.
- `origin/main` advanced during work to `58f7a0e8e192516934e376edab1c152741e225a2`. Its intervening change touches only `crates/chelis-compiler-api/src/source_arch.rs`, outside this commit's paths. The task branch was not rebased.

## Changes

- `.github/workflows/heavy-e2e.yml`: added a `validation_scope` choice (`all` by default, or `runtime-representation`). Schedule and `all` select every original execution job. Scoped dispatch selects the runtime-representation oracle and a small, hosted scope-receipt job. The generalization aggregate, telemetry, and main-only failure-tracker report retain their full-run conditions. A full dispatch on `main` requires the receipt job to succeed for the report to consider it complete; a schedule tolerates that job's expected skip.
- `scripts/ci_failure_baseline.py`: selects only schedule runs or dispatches with all four unique, unexpired full-workspace JUnits and a unique, well-formed `all` receipt bound to the API run ID and head SHA. It skips unknown events and incomplete or scoped candidates; old scheduled runs need no receipt. It rejects attempts to narrow the default workflow's required four artifacts. The receipt download follows `gh run download`'s single-artifact extraction layout.
- `scripts/test_ci_failure_baseline.py` and `scripts/test_ci_cadence.py`: positive and negative controls for schedule/all/scoped selection, main-only reporting, receipt generation, scoped runs retaining four JUnits, manual full acceptance, legacy schedule acceptance, wrong run/head, missing/ambiguous/malformed receipt, unknown event, and missing JUnit coverage.
- `docs/ci_validation.md`: documented scope commands, the meaning of scoped verdicts, and baseline eligibility.
- `changelog.d/2356-scoped-extended-validation.changed.md`: release note fragment.

## Evidence

- Test stubs were written before production edits. Four selected tests failed on the original workflow/selector as expected.
- `.venv/bin/python -m unittest scripts.test_ci_failure_baseline scripts.test_ci_cadence scripts.test_gate.DocsOnlySkipTests.test_generalize_sweep_aggregator_is_fail_closed_nightly scripts.test_gate.DocsOnlySkipTests.test_telemetry_skips_unless_every_junit_producer_succeeded scripts.test_gate.DocsOnlySkipTests.test_telemetry_producer_list_matches_the_report_aggregate scripts.test_gate.CiParityTests.test_runtime_representation_phase2_oracle_is_a_dedicated_gate_job` — **46 tests passed**. The cadence suite parses the workflow YAML, checks job conditions and dependencies, and executes the inline receipt writer with valid and invalid scope inputs.
- `.venv/bin/python scripts/changelog.py check` — **PASS**.
- `git diff HEAD^ HEAD --check` — **PASS**.
- `gh run download --help` confirmed that a single named artifact extracts directly into the destination; the parser and test fake use that layout.

Job legs selected by the workflow contract, before conditional runtime failures: scheduled `main` retains the 20 original legs (telemetry runs only after its producers succeed); a full branch dispatch has the 19 original branch legs plus one receipt job (20 total); a full `main` dispatch has the 20 original legs plus one receipt job (21 total); runtime-representation scope has **2** legs on either ref. The main-only report is excluded from scoped runs.

## Remaining acceptance work

The requested no-push constraint prevents a hosted dispatch of this new workflow. After publication is authorized, dispatch the committed candidate with `gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=runtime-representation`, inspect the actual two jobs and receipt artifact, and confirm the oracle verdict. A real scoped run on `main` cannot be tested until the workflow is available there; the local baseline-selection tests establish rejection even when four JUnits are present. Hosted CI and the full package-expansion consumer remain unverified. No Cargo build, full gate, issue closure, push, or PR was performed, as directed.
