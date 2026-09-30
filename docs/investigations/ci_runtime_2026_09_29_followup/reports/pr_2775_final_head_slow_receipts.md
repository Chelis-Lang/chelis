# PR #2775 final-head slow-test and delivery receipt (September 29, 2026)

Head `0ab65e24cc79af98f7edf5e6bb83231f60604ced`; required PR CI run `36618923324` completed successfully. PR #2775 squash-merged as `82693f8b97228db1eeed05d0e9c099ebbaca531d`; merged tree `13ceebda5afc67740d1210a2359702686466b950` equals the reviewed synthetic merge tree. Issue #2356 was closed manually afterward.

## Fast Tests (Linux)

Hosted Fast job `109579518880` passed in 25m39s. Its downloaded `junit-linux-fast` artifact contains 5,510 cases, 26 strictly over 10 seconds. Longest cases:

| Test | JUnit seconds |
|---|---:|
| `chelis-compiler-api::source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline` | 310.4 |
| `chelis-cli::runtime_extent_slice_b::independent_grad_entry_failures_are_ordered_on_eval_and_c` | 52.5 |
| `chelis-cli::issue_1739_diagonal_runtime_bound::no_shipped_example_gains_a_host_lane_guard` | 43.4 |
| `chelis-cli::runtime_extent_slice_b::independent_grad_computed_claims_keep_producer_on_eval_and_c` | 40.4 |
| `chelis-cli::dropout_fixed_stream_cli::a_keyed_dropout_under_a_runtime_arm_reads_only_its_own_key_in_eval_and_c` | 40.3 |

Artifact: `target/2775-final-fast-junit/junit.xml` in this research worktree. Case durations overlap in time and cannot be summed to obtain job duration.

## Script and Python Unit Tests (Linux)

Job `109579519063` passed in 21m03s. Its Script unit step ran 19:27:04–19:35:55 UTC (8m51s), then Python distribution smoke ran 19:36:23–19:45:58 (9m35s). The downloaded `script-tests-pr` timing artifact has four individual setup/test spans over 10 seconds:

| Span | seconds |
|---|---:|
| `test_capacity_census_native_registration.NativeRegistrationExecution.setUpClass` | 253.6 |
| `test_runtime_representation_oracle.BaselineTests.test_a_retired_foundation_identity_cannot_be_restored` | 33.8 |
| `test_phase4b_change_report.RetainedMutationTests.test_every_required_atom_and_region_edit_keeps_enforcing_review` | 31.3 |
| `test_runtime_representation_oracle.RedTeamRegressionTests.test_a_stale_active_debt_sample_fails` | 16.6 |

Artifacts: `target/2775-script-timings/timings/2847.jsonl` and `target/2775-wheel-smoke/wheel-smoke.json` in this research worktree. Wheel smoke passed crossed-bundle rejection and reported `source_free_build_proven=false` on Linux, which does not enforce checkout denial.

## Scoped dispatch and baseline

Hosted manual `heavy-e2e.yml` run `36618930443` on the exact head started only `Record dispatch scope` and `Runtime Representation Phase 2 Oracle`; every other execution job was skipped. The first job succeeded and its `scope.json` bound `runtime-representation`, the exact run ID, and the exact head. The oracle was intentionally canceled after verifying selection; this run is job-selection evidence only. Artifact: `target/2775-final-hosted-scope/scope.json`.

Final package expansion run `36618930296` passed, selected zero additional integration targets, and reported zero introduced, inherited, and unrun. It found complete scheduled baseline run `36517834756`, 15 commits behind the candidate base. Artifacts: `target/2775-expansion-plan/plan.json` and `target/2775-expansion-summary/{summary.md,report.json}`. Zero selected does not establish any oracle verdict.

Red-team round 1 found one stale CI-parity assertion, fixed on the final head and verified by the same reviewer. The initial CI-contract preflight failed 1 of 420 tests; the final-head preflight passed 421 of 421. Its red first run is not final-head validation. Review report: `target/2775-redteam-round1-report.md`.
