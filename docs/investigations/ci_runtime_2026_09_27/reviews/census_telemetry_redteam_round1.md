# PR #2682 — red-team round 1

Reviewed head: `d30770571bc4237703626cf9224369de1efb0bd3`.
Result: **Satisfied; round closed. No in-scope P0, P1, P2, or P3 findings.** No out-of-scope defect found.

Reviewed `.github/workflows/ci.yml`, `.github/workflows/pr-package-expansion.yml`, `docs/ci_validation.md`, and `scripts/test_change_owned_workflow.py`, plus the existing executor, timing hook, receipt validator, and census entry points relevant to the change.

## Execution and coverage

- `.venv/bin/python -m unittest scripts.test_change_owned_workflow scripts.test_ci_script_tests.TimingTests` — 15 tests passed. This includes upload-path agreement, `always()` upload behavior, subprocess success/failure/exception preservation, and timing output.
- Independent inline Python probe parsed both workflow files and expanded every shard index (0–3). For each of the eight paths, it ran a successful child and a checked exit-23 child under `ci_timing.subprocesses()`. The successful output and failure exception survived; JSONL contained the expected start/finish records and failure return code.
- The same probe launched an instrumented Python subprocess whose child sent SIGKILL to that Python parent. The killed process returned -9 and its start record remained readable without a finish record, inside the exact uploaded shard directory. This proves partial records survive process interruption locally.
- The probe wrote valid receipt sidecars in that directory and loaded them through `ci_change_owned.load_receipts`. Adding fabricated success timing JSONL left the loaded receipt identical. Corrupting `commands.json` still raised `ValueError`; timing records cannot substitute for the authoritative sidecar digest.
- A separate inline Python probe wrapped the existing executor test's injected runner to require `CHELIS_CI_TIMING_DIR` in the effective build/list/run environment. `ShardingAndExecutionTests.test_nonempty_shard_builds_products_once_before_target_commands` passed with all assertions. Three additional `ReportTests` passed: `test_required_report_rejects_missing_digest_duplicate_uncovered_excluded_and_failure`, `test_receipt_digest_mismatch_is_rejected`, and `test_load_receipts_rejects_missing_or_modified_sidecars`.
- Source inspection confirms the relevant census Python entry points opt into the existing hook; the executor copies its environment for nextest. The two workflow additions affect only environment values. Test selection, receipt schema, digest authority, and verdict code are unchanged. The documentation's bounded diagnostic claim matches this evidence.

## Limits

Hosted artifact transport and cancellation behavior were not executed locally. The workflow uploads remain `always()` and target the tested containing directories; a runner lost before artifact upload cannot be rescued by this change. No full census, heavy build, full fast gate, hosted CI status, or runtime reduction was revalidated or claimed. Root's exact pushed-head and initial clean-worktree evidence was accepted as instructed.

## Restoration

All probe data used `TemporaryDirectory`; no tracked source was changed, and monkeypatches were scoped/restored. Final command:

`.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/census-perf-20260927`

At 2026-09-27T18:28:53Z: **FREE**, exact reviewed head above, clean (0 modified/staged/untracked/unmerged), no scoped processes, free gate lease, no git operation. Standing reviewer remains available for repair verification.
