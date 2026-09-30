# PR CI runtime follow-up, September 2026

This archive was assembled September 30 from the September 29 investigation. Read the earlier [September 23–27 investigation](../ci_runtime_2026_09_27/README.md) first for the original twenty-slowest-test audit and PRs #2675, #2678, #2681–#2685, and #2688. The documents here are dated observations and review records, not a current CI contract. [`docs/ci_validation.md`](../../ci_validation.md) owns current selection and execution policy.

## What landed and what the receipts establish

| Area | Merged change | Observation and limit |
| --- | --- | --- |
| Standing Fast | #2753 kept a six-cell claimed-join canary in Fast and moved the unchanged 100-cell matrix to affected-package expansion and nightly; #2684 and #2754 reduced source-scanner copying. | Local testcase improvements do not establish hosted PR finish-time savings. The final #2775 Fast receipt still took 25m39s, including 310.4s in the source guard. |
| Runtime extents | #2683 reused identical compiled C fixtures; #2762 also reused exact-source/caller check and build preparation. | Distinct runtime inputs still execute separately. A sequential local #2762 pair improved 65.493s to 63.358s; expansion wall gain is unmeasured. |
| Capacity | #2685 shared dependency builds inside binding scopes; #2764 reused the wire schema's current-source probe in supervised controls; #2770 overlapped live wire proof with binding-only work inside the binding case. | Wire and binding verdicts remain separate. Exact-head hosted wire and binding cases passed in 786.031s and 1,384.811s; no matched before/after hosted trial establishes a causal gain. #2771 repaired the native-worker packet schema introduced by #2764. |
| Manual validation | #2775 added `all` and three named scopes to manual Linux Extended Validation. | A scoped dispatch started only its scope-receipt and selected oracle jobs. The oracle was then canceled; that run proves selection, not oracle success or time saved. Scheduled full runs and full-baseline eligibility remain separate. |
| Correctness | #2774 made the source guard fail on unreadable guarded directories or entries. | This prevents a false clean guard result; it is not a speed change. |

The [three-week run analysis](reports/pr_ci_3week_evidence.md) counted 531 created PRs and 1,857 completed PR CI runs (September 8–29; September 29 partial). Fast was the last independent worker in 153/162 sampled successful hosted PR runs. In 114 successful recent expansion dispatches, runtime extent preparation, wire census and binding census were selected 66, 62 and 23 times; their workers finished last in 29/66, 44/62 and 23/23. A last-worker observation does not attribute all of that worker's time to one test. The [final-hosted receipt](reports/pr_2775_final_head_slow_receipts.md) records Fast at 25m39s and Script/Python at 21m03s, including a 253.6s native-registration setup and 9m35s distribution smoke step.

## What was tested but did not land

- #2676's Fast/expansion deduplication closed after equal case names failed to establish equivalent Cargo feature closures; the earlier archive has the full comparison.
- #2763's Nextest priority trial kept all 5,376 Fast cases but did not improve hosted Fast latency against two same-base comparisons. The older [Fast assessment](reports/fast_agent_report.md) recommends this trial; its recommendation is superseded by the hosted result.
- A shared Phase 4B fixture prototype was not pushed. Its apparent whole-module improvement was mostly outside fixture preparation; the [subphase profile](experiments/dtype_phase4b_fixture_profile_2026_09_29.md) found about 1.12s of direct setup/teardown saving across 228 cases, plus isolation risks.
- A combined C-initializer fixture batch passed the focused behavioral checks but made a quiet warm pair about 17s slower; see the [rejected experiment](experiments/initializer_batch_rejected.md).
- Exact test-root routing, an exact Fast-unit sidecar route, cross-worker census sharing, and nightly runtime-extent/wire receipt reuse remained assessments. None is an accepted omission rule. The existing broad policy question is #1824; shard balance is already tracked by #2355 and #2253, and lockfile graph routing by #2363.

## New focused discussions

- #2792: share exact-candidate wire work across isolated expansion workers, while retaining separate authority.
- #2793: determine whether a *full scheduled* runtime-extent Phase A can consume the dtype job's same-run wire receipt.
- #2794: decide what a PR canary/full-oracle split would actually guarantee and how an unrun full proof is reported.
- #2795: prove reader closure before routing a modified `chelis-cli` integration root without CLI-wide expansion. The path-only opportunity is at most 24 of 531 historical PRs, not a measured saving.
- #2796: audit `tests/support/c_preprocessor.rs` for an exact owner before narrowing its broad CLI expansion.
- #2797: profile and shorten the remaining Fast source guard without losing its five-root and fail-closed checks.
- #2799: measure source-owned routing for the live native-registration build in Script/Python.
- #2798: profile wheel-smoke phases before deciding whether to split the PR canary from crossed-bundle validation.
- #2800: profile Phase 4B's repeated validation work; retain isolated negative-test fixtures.

These issues record options and rejection conditions. No test is approved for nightly-only selection by this archive.

## Archived documents

### Timing, selection and oracle assessments

- [Three-week PR CI and expansion analysis](reports/pr_ci_3week_evidence.md), [Fast >10s audit](reports/fast_gt10_audit_report.md), [Script/Python >10s audit](reports/script_gt10_audit_report.md), and [final-head slow-test receipt](reports/pr_2775_final_head_slow_receipts.md).
- [Selection plan](reports/selection_assessment.md), [reader-closure audit](reports/selection_agent_report.md), [test-only sidecar feasibility](reports/ci_selection_sidecar_feasibility_20260929.md), [native-registration routing addendum](reports/script_native_registration_routing_addendum.md), and [lockfile dependency audit](reports/lockfile_agent_report.md).
- [Census unification](reports/census_unification_report.md), [nightly wire/extent overlap](reports/nightly_overlap_evidence.md), [wire assessment](reports/wire_agent_report.md), [binding overlap feasibility](reports/binding_parallel_feasibility_report.md), and [extent restructuring](reports/extent_agent_report.md).
- [Fast critical-path assessment](reports/fast_agent_report.md) and [source-guard correctness addendum](reports/fast_source_guard_correctness_addendum.md).

### Experiments, implementation and review

- [Phase 4B shared-fixture experiment](experiments/dtype_phase4b_fixture_reuse_2026_09_29.md), [isolation audit](experiments/dtype_phase4b_fixture_audit_2026_09_29.md), [subphase profile](experiments/dtype_phase4b_fixture_profile_2026_09_29.md), and [rejected initializer batching](experiments/initializer_batch_rejected.md).
- [Fast](delivery/fast_implementation_report.md), [wire](delivery/wire_implementation_report.md), [binding](delivery/binding_parallel_implementation_report.md), [scoped dispatch](delivery/issue2356_scoped_dispatch_implementation_report.md), [dispatch repair](delivery/issue2356_scoped_dispatch_repair_report.md), and [source-guard repair](delivery/source_guard_failclosed_implementation_report.md) implementation records.
- [#2762](reviews/pr2762_redteam_round1.md), [#2763](reviews/pr2763_redteam_round1.md), [#2764](reviews/pr2764_redteam_round1.md), [#2770 initial](reviews/binding_2770_review_report.md), [#2770 base update](reviews/binding_2770_merged_review_report.md), [#2771](reviews/native_hotfix_2771_review_report.md), [#2774](reviews/pr_2774_redteam_round1_report.md), and [#2775](reviews/pr_2775_redteam_round1_report.md) review records.

Some source reports name ignored `target/` files, local absolute worktree paths, or historical heads. Those are provenance for the original measurement; the raw artifacts and temporary scripts are not bundled here. Per-test durations can overlap, and sums of JUnit times are not job wall time. A recommendation in a dated source report may have been implemented, withdrawn or superseded afterward; the status above records the September 30 disposition.
