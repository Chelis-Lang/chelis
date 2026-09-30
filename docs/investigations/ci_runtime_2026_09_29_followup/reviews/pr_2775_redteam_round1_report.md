# PR #2775 red-team round 1 — initial report

Reviewed pushed head `aef5b63f83a4dc8084bf732399feda05062f560e` on `agent/2356-scoped-dispatch` in `/Users/robertronan/chelis-worktrees/issue-2356-scoped-dispatch`. Issue #2356 is open; its acceptance asks for a runtime-representation-only dispatch with materially fewer than nineteen jobs and rejection of scoped runs as package-expansion baselines. Reviewed the six named changed files and the owning CI contract in `scripts/test_gate.py`. No tracked source was changed.

## Finding

**P2, in scope — CI-contract oracle migration.** The new `if:` on `heavy-e2e.yml`'s `integration-support` job is required to keep support slices out of scoped dispatches. The older `_assert_support_slice_contract` in `scripts/test_gate.py:1864` still asserts that this job has *no* job-level `if:`. This makes the required changed-CI-contract preflight red, which stops ordinary candidate CI from exercising this PR. The assertion should accept the exact full-run condition while continuing to reject false or weakened conditions and step-level skips. This is a mechanical test-contract migration, not evidence that support slices execute in the wrong scope.

Reproduction at the reviewed head:

```sh
.venv/bin/python -m unittest scripts.test_gate.CiParityTests.test_fast_worker_and_nightly_support_have_separate_owners -v
```

Result: one test, one failure at `test_gate.py:2319` / `_assert_support_slice_contract:1864`, `AssertionError` on `re.search(r"^    if\s*:", block, re.MULTILINE) is None`. Running `.venv/bin/python -m unittest scripts.test_gate.CiParityTests -q` yields 50 tests, one failure: the same case. Hosted CI run `36617706358`, detector job `109574877309`, ran 420 CI-contract tests with this sole failure. The detector's preflight outcome is `failure` (`the changed CI contracts did not validate`); Docs, Lint and Unit Tests, Change-Owned Integration Report, and Integration Tests then failed or skipped as consequences. They are not separate reproduced findings. Required CI remains red.

No P0/P1 or other P2 finding was confirmed at this head.

## Coverage against the PR claims

- **Job routing and receipt:** Executed the actual workflow's `if` expressions across `schedule`, dispatch `all`, each of the three scoped values, and `push`. The full events select every original execution job; each scoped value selects only `dispatch-scope` and its named oracle; `push` selects none. These are expression evaluations, not hosted executions. Live dispatch `36617713749` on this exact SHA started only `Record dispatch scope` and `Runtime Representation Phase 2 Oracle`; other execution jobs were skipped. Downloaded its published `scope.json` with `gh run download 36617713749 --repo Chelis-Lang/chelis --dir /tmp/chelis-2775-review.r8hu8D --name linux-extended-dispatch-scope`. It says `workflow_dispatch`, `runtime-representation`, run `36617713749`, and the exact head SHA. The receipt job succeeded; the oracle was later cancelled, so this is selection evidence only.
- **Baseline provenance:** Executed `ci_failure_baseline.prepare` with the real scoped receipt and fake *all four* JUnit artifacts: rejected. Replaced the fake receipt with exact `all`/run/head: accepted. Replaced its head with a wrong SHA: rejected. The `FakeGh` harness and all output lived in a temporary directory. Also ran five named focused tests below; all passed. These cover legacy schedules without receipts, duplicate/missing receipts, custom workflow events/artifacts, and fallback to a bounded schedule list after ten scoped dispatches.
- **Report semantics:** Executed the actual `report` job's github-script body under Node with synthetic `needs` and GitHub API stubs. A successful schedule with `dispatch-scope=skipped` and a successful full dispatch with `dispatch-scope=success` closed the existing nightly tracker; an execution failure, unexpected successful receipt on schedule, or failed receipt on full dispatch did not. The scoped branch excludes the report by its job condition. The live scoped dispatch did not close a nightly tracker.
- **Docs and changelog:** `docs/ci_validation.md` and the fragment describe the four manual choices, the separate receipt, legacy schedule eligibility, baseline fallback, and scoped result limits consistently with the exercised behavior.

Focused command:

```sh
.venv/bin/python -m unittest -v scripts.test_ci_failure_baseline.BaselineSelectionTests.test_ten_scoped_dispatches_do_not_crowd_out_last_complete_schedule scripts.test_ci_failure_baseline.BaselineSelectionTests.test_custom_workflow_preserves_events_and_supplied_artifacts scripts.test_ci_failure_baseline.BaselineSelectionTests.test_a_legacy_schedule_needs_four_junits_but_no_receipt scripts.test_ci_failure_baseline.BaselineSelectionTests.test_dispatch_missing_or_ambiguous_receipt_is_rejected scripts.test_ci_cadence.ExtendedCadenceTests.test_dispatch_receipt_step_records_the_exact_scope_run_and_head
```

Result: 5/5 pass. Live GitHub API schedule-filter query returned ten scheduled runs; the latest completed schedule has four nonexpired `junit-linux-full-*` artifacts. I did not download its potentially large JUnits.

## Limits and handoff

The runtime-representation oracle's hosted verdict was not obtained because run `36617713749` was cancelled. No hosted `all`, `dtype-phase3`, or `runtime-extent` dispatch and no real package-expansion report on this head were checked. The new schedule report path cannot execute on the PR branch; its JavaScript was exercised with synthetic `needs`. Full required CI was cut off by the P2 assertion above. No heavyweight build was started.

Final `git status --porcelain=v1` was empty; HEAD remained `aef5b63f83a4dc8084bf732399feda05062f560e`. The shared worktree is available for a serial author repair and subsequent exact-head verification by this reviewer.

## Standing-reviewer verification — local repair

**CLOSED** for the round-1 P2 class **CI-contract oracle migration** on committed, unpushed head `0ab65e24cc79af98f7edf5e6bb83231f60604ced`. The only change since the reviewed pushed head is `scripts/test_gate.py`. `_assert_support_slice_contract` now requires the exact schedule-or-dispatch-`all` job condition and still forbids skipped support-slice steps.

Commands run in the supplied worktree:

```sh
.venv/bin/python -m unittest scripts.test_gate.CiParityTests.test_fast_worker_and_nightly_support_have_separate_owners -v
.venv/bin/python -m unittest -v scripts.test_gate.CiParityTests.test_fast_worker_and_nightly_support_have_separate_owners scripts.test_gate.CiParityTests.test_support_slice_contract_rejects_missing_or_weakened_job_scope scripts.test_gate.CiParityTests.test_support_slice_contract_rejects_missing_repeated_or_skipped_work scripts.test_ci_cadence.ExtendedCadenceTests.test_schedule_all_and_scoped_dispatch_job_selection
.venv/bin/python -m unittest scripts.test_gate.CiParityTests -q
```

Results: 1/1, 4/4, and 51/51 pass. The new mutation control rejects a missing job condition, `if: true`, schedule-only selection, and duplicate job `if:`. The older control still rejects missing, repeated, and step-skipped support work. I inspected the helper's other call site and ran the full CI-parity class for a similar mismatch; none appeared. Final HEAD is the exact local SHA above; `git status --porcelain=v1` is empty and `worktree_status.py` reports FREE. No tracked source or target build was modified by this verification. Hosted CI on this repair remains unvalidated until the author pushes it.
