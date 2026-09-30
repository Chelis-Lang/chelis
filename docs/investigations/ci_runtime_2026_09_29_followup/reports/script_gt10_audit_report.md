# Script and Python Unit Tests (Linux): >10-second audit

**Scope.** Read-only audit on 2026-09-29. Code citations refer to `origin/main` at `5e2e8630db704a0e9ac64d384ce39d16f763ad2e`; checkout HEAD is older (`465a3623be19bd68ada6c39aa7552b63e0ccf8c3`). From the supplied run inventory, seed `290926` selected 27 distinct-branch successful PR CI runs by date. Job steps were available for 27, `script-tests-pr` for 19, and wheel receipts for four. Eight older selected runs had no script artifact available. Workflow heads differ, so this is not a paired experiment.

**Receipts.** `target/script-gt10-audit/sample.json` has full heads, jobs and artifacts; downloads are under `artifacts/<run-id>/`; calculations are in `derived.json` and `all_jobs.json`. Extraction:

```sh
gh api '/repos/Chelis-Lang/chelis/actions/runs/36531598261/jobs?per_page=100'
gh api '/repos/Chelis-Lang/chelis/actions/runs/36531598261/artifacts?per_page=100'
gh run download 36531598261 -n script-tests-pr -D target/script-gt10-audit/artifacts/36531598261/script-tests-pr -R Chelis-Lang/chelis
gh run download 36531598261 -n python-wheel-smoke -D target/script-gt10-audit/artifacts/36531598261/python-wheel-smoke -R Chelis-Lang/chelis
```

Sample IDs and head prefixes (full heads in `sample.json`); `*` means script timing artifact available:

| Dates | Runs `id:head` |
|---|---|
| Sep 10 | 34460209461:0be95a6949d2, 34538967850:31a0b1e42714, 34438704955:1399ab07396b |
| Sep 11–12 | 34708257937:41b88b2a04af, 34684799146:a9f435f114c6, 34632889630:0d8ec40a3211 |
| Sep 15–17 | 35149343348:57ba7cee7c70*, 35215374105:ed02288b09e8*, 34939445989:851774fa1a0d, 34938053523:0e33df54fcd1 |
| Sep 20–21 | 35631312451:d49b0a1565ba*, 35565992702:f9d8edc57a79*, 35510005359:57ae8224d225*, 35477685380:091bbe1b8ecc* |
| Sep 23–25 | 36030159868:9abcdfc73896*, 36089295688:e58122844ae3*, 36049178763:5050a71607c1*, 36027975197:44793759528f*, 35831882426:fe3279abc1b8* |
| Sep 26–28 | 36232648668:f9d106e58702*, 36227789823:8ecf25b1b5f4*, 36461453296:342fe8c70c08*, 36228209667:7f554ef60522*, 36230532409:1eb1ddd4be65* |
| Sep 29 | 36531598261:768a27b28e35*, 36581803867:4dab9a2afef1*, 36509012547:b67246adfce6* |

## Granularity and observed times

Actions timestamps time jobs and steps, not cases inside `pytest`, binding tests, rejection validation or wheel smoke. Only `ci_script_tests.py pr` writes `selection.json` and JSONL test/setup/subprocess spans (`scripts/ci_script_tests.py:85–144`, `scripts/ci_timing.py:10–65`). Subprocess spans nest in tests. Module totals sum serial tests and class setups, excluding discovery. `artifacts/36531598261/script-tests-pr/selection.json` lists **3,062 PR, 62 nightly, 38 census, 25 profile** identities; sampled PR selections range 2,916–3,200.

All **named Actions steps** with at least one observation strictly greater than 10 seconds in the 27 successful sampled jobs:

| Step | Observations >10 / measured | Median, maximum seconds | Maximum run |
|---|---:|---:|---:|
| Script unit tests | 27/27 | 470, 1,394 | 34460209461 |
| Rejection authority construction boundary | 20/20 | 21, 24 | 36461453296 |
| Free disk space | 13/13 | 92, 365 | 36230532409 |
| Set up project CI | 13/13 | 26, 60 | 36030159868 |
| Build and exercise Python distributions | 4/4 | 480.5, 562 | 36461453296 |

On September 29, script steps took 441–492 seconds, wheel 459–499, and jobs 1,082–1,369. Script plus wheel occupied median 86.4% of the four wheel-bearing jobs. September 10 median job time was 1,388 seconds versus 397 on September 11–12, across different workflow heads. The serial job (`ci.yml:768–838`) feeds required Lint and Unit (`:844–876`). In four wheel runs it ended 241–488 seconds after Rust Policy; the aggregate started 2–4 seconds later. The last job was Integration in three runs, telemetry in one (`all_jobs.json`). Other branches can still limit end-to-end savings.

All **module identities** whose derived test-plus-class-setup total exceeded 10 seconds in at least one of the 19 timed successful runs:

| Module (drop `test_` prefix only for display) | >10 runs | Median, max seconds |
|---|---:|---:|
| `test_capacity_census_native_registration` | 19/19 | 221.9, 262.2 |
| `test_dtype_phase4b_oracle` | 19/19 | 78.6, 92.9 |
| `test_runtime_representation_oracle` | 19/19 | 48.0, 51.4 |
| `test_phase4b_change_report` | 19/19 | 32.3, 62.4 |
| `test_changelog` | 9/19 | 9.9, 13.4 |
| `test_check_configuration_closure` | 6/19 | 9.5, 10.6 |
| `test_ci_rebase_reuse` | 2/19 | 5.2, 11.8 |

All **individual Python test/setup identities** exceeding 10 seconds in those receipts (seconds are median over *all* 19 runs, then maximum):

| Kind and full identity | >10 runs | Median, max |
|---|---:|---:|
| setup `test_capacity_census_native_registration.NativeRegistrationExecution.setUpClass` | 19/19 | 217.3, 256.2 |
| test `test_runtime_representation_oracle.BaselineTests.test_a_retired_foundation_identity_cannot_be_restored` | 19/19 | 31.3, 33.6 |
| test `test_phase4b_change_report.RetainedMutationTests.test_every_required_atom_and_region_edit_keeps_enforcing_review` | 19/19 | 28.4, 38.1 |
| test `test_runtime_representation_oracle.RedTeamRegressionTests.test_a_stale_active_debt_sample_fails` | 17/19 | 14.6, 16.1 |
| test `test_phase4b_change_report.RetainedMutationTests.test_missing_or_ambiguous_required_definitions_and_boundaries_still_fail` | 2/19 | 0.8, 20.7 |

All **exact subprocess-command identities** exceeding 10 seconds (same 19 runs; nested times):

| Command | >10 invocations | Median, max seconds |
|---|---:|---:|
| `cargo build --locked -p chelis-python --example native_binding_probe --message-format=json` | 19/19 | 214.7, 254.0 |
| `/home/runner/work/chelis/chelis/target/debug/chelis-repr-inventory --repo /home/runner/work/chelis/chelis` | 36/38 | 15.3, 20.3 |
| `cargo build --quiet -p chelis-repr-inventory` | 19/38 | 5.2, 15.0 |

Max native timings: `artifacts/36461453296/script-tests-pr/timings/2828.jsonl`. On run 36531598261, 286 `test_dtype_phase4b_oracle` cases sum to 64.2 seconds although none individually exceed 10.

## Boundary judgment and ranked work

**1. Native registration build: largest steady cost.** `NativeRegistrationExecution.setUpClass` always builds and executes a current-source `chelis-python` probe; its two tests check four actual exposures and tampering controls (`scripts/test_capacity_census_native_registration.py:93–125`, `scripts/capacity_census_native_registration.py:19,75–145`). The synthetic packet controls (`test_capacity_census_native_registration.py:32–90`) do not replace that proof. The compiled class is absent from `NIGHTLY_CLASSES`, so default PR selection runs it even on docs-only heads (`ci_script_tests.py:16–39,70–80`; `ci.yml:760–764`). **Next PR:** add fail-closed change ownership for the compiled proof, triggered by probe/test/collector changes, Python native sources, dependency closure, manifests, lock/toolchain and unknown paths; retain synthetic controls every PR and compiled proof at final candidate/nightly when unselected. Test every trigger, unknown-path fallback, docs-only exclusion, and stale-source/changed-binary/missing-process failures. Unconditional nightly moves a possible Python registration break past PR merge. A cached binary or packet alone supplies no proof: Cargo must still build and execute the exact candidate. The wheel uses a separate release build and cannot automatically share this debug example.

**2. Wheel smoke: another serial eight-minute step.** Every PR, including docs-only, runs `--crossed-bundle` (`ci.yml:760–761,803–825`). It checks editable freshness, sealed-wheel compile/reload, developer-target absence, and runtime-distinct cross-wheel rejection (`bindings/python/tests/python_wheel_smoke.py:769–825,830–975,990–1120`). Four wheel receipts under `artifacts/{36461453296,36531598261,36581803867,36509012547}/python-wheel-smoke/` passed with cross-wheel rejection. All report `source_free_build_proven=false`: Linux does not enforce original-checkout denial, which only the Darwin sandbox requests (`python_wheel_smoke.py:478–486,1115–1120`). No phase timers separate the crossed-bundle cost. **Next PR:** add phase durations and a partial-failure receipt, then compare a split with installed sealed-wheel canary on every PR and crossed-bundle on Python/runtime/package changes plus every final exact candidate. Unselected intermediate heads lose immediate cross-wheel mismatch detection. `--wheel` skips a current-source build (`:1140–1153`); reuse requires a candidate-bound producer receipt and mismatched-wheel negative control. A parallel job adds setup/free-disk cost; test same-head hosted finish time and runner-minutes.

**3. Retain live representation/contract guards; reuse preparation carefully.** Two current-source scanner invocations take 10–20 seconds each (`runtime_representation_oracle.py:703–775`). Failed run **36419684817** (head `278f80c1fc…`) rejected a new `load-store-template` in `host_emit.rs` (`artifacts/36419684817/script-tests-pr/timings/2916.jsonl`; `failed-36419684817.log:1573–1609`), showing why the live scan belongs on PRs. `test_dtype_phase4b_oracle.py:30–50` copies contract files per test; its module median is 78.6 seconds. `test_phase4b_change_report.py:469–545` already batches validators while checking each atom/region. **Next PR:** profile copies/snapshots/scans and reuse only immutable same-source observations, preserving invalidation (`runtime_representation_oracle.py:679–687`) and failure identities. Negative controls: mutate a scanned source between reads, remove/duplicate an atom or region marker, and omit one acknowledgement. Compare full script steps and identical selections.

**4. Secondary steps.** Rejection authority takes median 21 seconds, enforcing source privacy/use without mutable issue liveness (`scripts/check_rejection_authority_boundary.py:86–189`; `ci.yml:796–804`). LOC and binding ingress had no >10-second step. Free-disk takes median 92, max 365 seconds, but guards past ENOSPC (`scripts/ci_free_disk.py:1–31,79–114`). Measure this job's disk peak before an adequate-space skip; low/unknown-space controls must still clean. `docs/ci_validation.md:447–453` says 35-minute timeout versus 90 in `ci.yml:762`.

**Limits.** Among 22 sampled failed workflows (`failure_sample.json`), five failed script-unit. Run **36385703372** (head `1260568bc4…`) failed contract-fixture tests after a required WireDag schema contract disappeared (`failed-36385703372.log:1573–1679`); candidate validity was not adjudicated. Run **36419684817** is the live scan above. No failed wheel step was analyzed. Older artifacts and wheel subphase times are unavailable. Measure proposed savings with paired exact heads, selection readback, negative controls, hosted finish time/runner-minutes and final expansion; this sample proves no causal speedup.

**Worktree:** `git status --short` was empty before the audit; only ignored files under `target/script-gt10-audit/` and this ignored report were created. No tracked source was edited and no build or PR was run.
