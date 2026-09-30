# PR CI and expansion evidence, September 8–29, 2026

Collected September 29 from Chelis-Lang/chelis GitHub Actions and PR file APIs. Source checkout for planner interpretation: `465a3623be19bd68ada6c39aa7552b63e0ccf8c3`. September 29 is incomplete. Raw inventories and samples are in this directory (`ci_pr_runs_3weeks.json`, `prs_3weeks.json`, `pr_files_3weeks.json`, `ci_jobs_sample.json`, `expansion_all_plans.json`, `recent_expansion_jobs.json`, `recent_expansion_criticality.json`). The GitHub run listing stopped after 1,000 results for a broad query, so PR CI was partitioned by date and deduplicated by run ID. Job timing, not a long-lived run `updated_at`, is used for the sampled CI stage comparisons. Artifacts are retained 14 days; 30 of 338 older/failed expansion dispatches had no downloadable plan; **all 114 September 22–29 successful dispatches did**.

## Volume and changed paths

| Period UTC | Created PRs | Merged from cohort | Completed PR CI | Success / failure / cancelled | Expansion dispatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sep 8–14 | 265 | 254 | 896 | 480 / 149 / 267 | 0 returned for this workflow |
| Sep 15–21 | 131 | 128 | 538 | 228 / 130 / 180 | 141 |
| Sep 22–29 | 135 | 127 | 423 | 259 / 95 / 69 | 197 |
| Total | 531 | 509 | 1,857 | 967 / 374 / 516 | 338 |

530 of the 531 PR branches appeared in CI; their median was 2.5 completed CI runs (p90 7). These are final PR file lists, not exact files at every intermediate CI head: 263/531 touched `crates/chelis-cli/tests/`, including 256 with a `.rs` test root; 46 touched `crates/chelis-cli/src/`. Only 37 of the test-root PRs touched no `crates/*/src/` path, and 34 also touched no Cargo manifest or root lock. Root `Cargo.lock` appeared in 35 PRs; 16 of those touched at most three crate directories. These counts bound possible new selection policies but do not prove any changed test root is isolated.

## Current routing code

`scripts/ci_change_owned.py:1391-1730` makes an added/directly modified integration target required and also adds its package to `selected_packages`. The package's other eligible targets go to `package_expansion`; `required_package_rule` promotes all targets in its named packages to required. Non-target crate paths add their package for expansion. Targeted rebases add reverse workspace dependencies. `execution_shards()` at line 2008 takes the duration-LPT plan in `target_dispositions`, not the legacy hash compatibility field in `plan.shards.package_expansion`. `.config/ci-test-targets.toml:3088-3102` expands the root lockfile to every workspace package; its separate required rule at line 179 keeps the runtime-bundle evidence.

## Job timings and actual expansion selection

Sixty deterministic random successful CI runs, 20 distinct PR branches per period, were sampled for job/step timing. The workflow changed around September 11 from full workspace jobs to Fast/change-owned, so the early week is not a comparable Fast baseline. In September 15–21, sampled Fast jobs were median 17.6 minutes (20/20). In September 22–29, they were median 18.9 minutes (20/20); the 15 GitHub-hosted jobs were median 21.0 minutes with median 1.3 minutes from run creation to job start. The gate step accounted for median 15.6 minutes of the full 20-run set. This is a small stage sample, not a paired runner experiment.

For a fuller recent critical-path check, job metadata was collected for **all 259 successful PR CI runs** from September 22–29. Excluding six rerun attempts leaves 253 first-attempt successes. Fast ran in 245 of those; eight took a docs or reuse route. Of the 245, 162 used GitHub-hosted runners (median Fast 22.0 minutes) and 83 used warm self-hosted runners (median Fast 9.3 minutes). The difference mixes heads and routing histories and is not a matched runner trial. Fast was the last independent worker in 153/162 hosted runs and 9/83 warm runs. `Integration Tests (Linux)`, which depends on Fast, was the final workflow job in 175/253 first-attempt successes; `Lint and Unit Tests (Linux)` was final in 65, telemetry in seven, and Docs in six. This supports measuring a Fast scheduling change against hosted PR finish time, while showing that a Fast-only improvement cannot shorten every run.

In the same 253 runs, a simple greater-than-one-minute job-duration proxy classifies 180 as having at least one active change-owned shard; the median longest such shard was 6.4 minutes, p90 15.3. Ninety-four had all four shards over one minute. This threshold is a setup-plus-test proxy, not an exact selected-target count. A change-owned shard was the last independent worker in 19 runs; Fast was last in 162. So reducing selected change-owned work often saves paid execution without reducing the PR's final required-check time, while the long extent/census tails still merit specific attention.

Among 114 **successful** September 22–29 expansion dispatches, 96 selected nonempty coverage (median workflow wall 38.4 minutes, p90 68.0), 74 selected `chelis-cli`, 66 selected `chelis-cli::runtime_extent_claim_preparation`, 62 selected `chelis-compiler-api::capacity_census_wire`, and 23 selected `chelis-python::capacity_census_bindings`. Nineteen changed root `Cargo.lock` and selected at least 30 workspace packages. These are counts of dispatches, not distinct PRs; a PR may dispatch more than once. From actual execution assignments and finish times for all 96 nonempty runs: the extent target was on the last finishing worker in 29/66 selections, wire in 44/62, and bindings in 23/23; 83/96 last workers held at least one of those three. Last-worker membership alone does not prove the target was the cause, and cross-worker queue/start differences also matter.

Target-level receipts confirm the principal cases in representative expansions:

| Run ID | Selection | Target run seconds | Last-worker observation |
| ---: | --- | ---: | --- |
| 36533889067 (#2754) | wire | 1,131 | wire worker last, 11-minute margin to next worker |
| 36510529160 (broad lockfile) | bindings | 2,536 | binding worker last, 10-minute margin |
| 36510529160 (same run) | wire | 1,141 | wire on another worker |
| 36531610022 (#2753) | extent preparation | 678 | another CLI worker finished 9.6 minutes later |
| 36382051985 (CLI expansion) | extent preparation | 970 | extent worker last, 3.9-minute margin |

The wire primary JUnit case accounts for virtually all 1,131 seconds; its two acceptance-control cases are under a second each. The bindings primary case likewise accounts for the 2,536 seconds and internally runs a fresh wire verifier. Capacity code's nested JSONL shows repeated Cargo builds, including a 169-second duplicate publication-probe build; nested process intervals overlap, so their durations cannot be summed. Avoid quoting any target-only saving as equal to workflow latency saved: another worker may become the critical path.

## Interpretation for PR priorities

The PR-facing priority is well supported by observed recurrence: the three heavy targets appear in 66/62/23 of 114 successful dispatch plans in the last eight days despite few direct edits to their own roots. Improving the wire proof can also help bindings, because the binding verifier calls the wire verifier. Combining the two tests into one serial target would reduce duplicate runner work only if an in-memory proof can be shared; it could worsen PR finish time now that workers run in parallel. A conservative test-root selection policy or dependency-aware lockfile expansion could save substantially more selected targets, but needs a fail-closed input/dependency proof. Nightly serial-oracle scheduling is a separate lower-frequency opportunity.

## Follow-up design decisions

The right unit for shared work is an exact source-bound **observation**, not a
testcase name. A preparation may be shared when the Surf source, generated
caller, target, toolchain and staged runtime agree; each good or bad runtime
input still needs its own process and assertion. Extent claim observations can
therefore share compilation within one matrix without collapsing the distinct
export, binding and root routes. The wire census can similarly reuse an
already built current-source probe inside its supervised controls, while its
cold codec, source mutation, selected/executed and authority checks remain.

A larger unified proof runner is possible in principle: schedule source-bound
probe nodes, then issue separate extent and capacity verdicts with separate
failure identities and negative controls. Today their input universes differ.
The extent oracle's registered runtime programs and guard ordering do not
enumerate published C headers, Rustdoc wire fields or numeric registration.
Running both as one serial Nextest case therefore does not itself remove
compilation or shorten an expansion's last worker. A cross-worker reuse scheme
would need authenticated same-candidate receipts and an accepted replacement
for the verifier's in-memory witness. A within-one-test overlap of the nested
wire and binding-specific work is being checked separately.

Moving the full oracles to nightly with a PR smoke test is a valid change in
*claim*, not an equivalent optimization. The PR would need to report exact
selected identities and explicitly state that the full oracle has not run on
that candidate. Because 150 PRs per week can merge before the daily run, a
regression outside the smoke coverage could first appear after merge. Existing
timed stages (`target/ci-fast/timing.json` and the census JSONL) already expose
where the current serial time goes; benchmark changes against complete
required-check or expansion finish time, not the sum of testcase durations.

## September 29 delivery checks

- [PR #2753](https://github.com/Chelis-Lang/chelis/pull/2753) moved the unchanged 100-cell claimed-join matrix out of standing Fast selection, retaining three standing regression tests including a six-cell Eval/DAG/C canary. [PR #2754](https://github.com/Chelis-Lang/chelis/pull/2754) reduced repeated source-guard scanner work. Both were merged earlier on September 29.
- [PR #2762](https://github.com/Chelis-Lang/chelis/pull/2762) merged as `11c0049633223ad1b4ff8b88af3a64d312b6afee`. It reuses check/build preparation only for exact matching exported source and caller while retaining separate fresh executions and assertions. Its required runtime-extent target passed and final expansion covered 377 additional targets with zero introduced, inherited, or unrun. A sequential local pair of the 120-case omitted-extent test took 65.493s before and 63.358s after; the latter reused 19 checked builds. This 2.135s local delta is small and is not a hosted workflow estimate. The specific receipt is `target/extent-paired-benchmark.txt` here.
- [PR #2764](https://github.com/Chelis-Lang/chelis/pull/2764) merged as `6d3f8c4b66af2d195bd661454c964932b2dc5a95`. Its final-head full wire case passed locally with a matching source digest and 135/135 supervised mutation controls selected and executed. Its final expansion selected zero targets and reported zero introduced, inherited, or unrun; required hosted CI passed. The 204s warm final-head case and 1,058s earlier cold case do not isolate this patch's wall-time effect.
- [PR #2763](https://github.com/Chelis-Lang/chelis/pull/2763), a Fast source-guard priority trial, was closed unmerged. Its exact selected cases remained 5,376, and local controls confirmed the scheduling change. Its hosted Fast job took 26m32s and its main Nextest group took 1,006s, versus two same-base non-priority Fast runs at 25m36s/941s and 20m35s/699s. The 986-target final expansion was cancelled when the hosted comparisons showed no observed latency benefit.
