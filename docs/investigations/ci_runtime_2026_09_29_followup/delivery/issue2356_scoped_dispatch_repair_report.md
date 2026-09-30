# #2356 local repair report

Date: 2026-09-29

Worktree: `/Users/robertronan/chelis-worktrees/issue-2356-scoped-dispatch`
Branch: `agent/2356-scoped-dispatch`
Prior head: `070ffb7c1332cf4ae5f80c4b6e466b3681c59efd`
Repair head: `eb2e05e149809e87f259c1bfb420472a28c9f97c`
Status: clean (`git status --porcelain=v1` printed nothing). The repair is one local commit; nothing was pushed and no PR or agent was created.

## Repairs

1. `.github/workflows/heavy-e2e.yml` now installs `astral-sh/setup-uv@v8.1.0`, creates a venv with `uv venv --python 3.11`, and writes the dispatch receipt using `.venv/bin/python`. `scripts/test_ci_cadence.py` locks the setup order and exact interpreter command, then executes the inline producer with the test process's interpreter in a temporary directory. It does not require the hosted job's venv.
2. `scripts/ci_failure_baseline.py` treats a failed `gh run download` for a scope receipt as an ineligible candidate and continues to an older baseline. `scripts/test_ci_failure_baseline.py` tests a failed newer dispatch download followed by a complete scheduled run. Scope and run/head validation still fail closed.
3. The event filter, four-JUnit lock, and scope receipt apply only to `heavy-e2e.yml`. Other `--workflow` selections retain their prior event and supplied-artifact behavior. A positive test selects custom `push` and `workflow_dispatch` runs using two supplied JUnit artifacts without a scope receipt.
4. If the recent mixed run window has no usable `heavy-e2e.yml` baseline, the selector queries a second window filtered to schedules, with the same `--search-runs` bound (10 by default). Run IDs are deduplicated across windows. A test puts ten complete-artifact scoped dispatches ahead of an older complete schedule, proves the scoped runs are skipped, and selects the schedule. `docs/ci_validation.md` records the two-window selection rule.
5. The workflow exposes exactly four choices: `all` (default), `runtime-representation`, `dtype-phase3`, and `runtime-extent`. The two added scopes each select their named independent oracle plus the receipt job. Workflow contract tests reject a missing owner or an extra full-workspace job for either scope. The receipt producer accepts both new scope names; the baseline selector still accepts only `all`. The pending changelog fragment and CI documentation describe the added scopes.

The second window is deliberately bounded. If its ten scheduled runs are incomplete, expired, or not contained in the candidate base, baseline selection still fails closed. It does not search unbounded history.

Candidate dispatch commands:

```text
gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=runtime-representation
gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=dtype-phase3
gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=runtime-extent
gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=all
```

## Verification

Tests were added before the production repairs. The selected new tests failed against the prior head for the Python setup, failed receipt download, custom-workflow selection, ten-dispatch crowd-out, and new scope-selection cases.

```text
.venv/bin/python -m unittest scripts.test_ci_failure_baseline scripts.test_ci_cadence scripts.test_gate.DocsOnlySkipTests.test_generalize_sweep_aggregator_is_fail_closed_nightly scripts.test_gate.DocsOnlySkipTests.test_telemetry_skips_unless_every_junit_producer_succeeded scripts.test_gate.DocsOnlySkipTests.test_telemetry_producer_list_matches_the_report_aggregate scripts.test_gate.CiParityTests.test_runtime_representation_phase2_oracle_is_a_dedicated_gate_job
```

Result: **50 tests passed**. The cadence suite parses the workflow YAML.

```text
.venv/bin/python scripts/changelog.py check
```

Result: **PASS**.

```text
git diff --check
git diff --cached --check
git diff HEAD^ HEAD --check
```

Result: **PASS** for each. No Cargo, full gate, hosted dispatch, push, or PR was run. Hosted dispatch and actual artifact selection remain acceptance work after publication is authorized.
