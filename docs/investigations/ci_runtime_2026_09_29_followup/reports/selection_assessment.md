# CI selection assessment (2026-09-29)

Head `465a3623be19bd68ada6c39aa7552b63e0ccf8c3`. Offline, no-build Cargo metadata and hypothetical one-path `make_plan` calls give exact counts. Seconds are LPT weights, not wall times. Cases are raw `#[test]` names, ignored included. Sources: `scripts/ci_change_owned.py:1391-1730`, `.config/ci-test-targets.toml`, `.config/ci-change-owned-durations.json`, `docs/investigations/ci_runtime_2026_09_27/`.

## Exact one-path selections

| Modified path | Required | Manual expansion | Baseline / case proxy |
| --- | ---: | ---: | --- |
| `crates/chelis-cli/src/main.rs` | 0 | 378 CLI targets | 5,697s total, ~1,424s/shard; 4,148 raw names |
| `crates/chelis-cli/tests/cli.rs` | `chelis-cli::cli` (1) | 377 CLI targets | required: 274 names, 87.6s; expansion: 3,874 names, 5,610s |
| `crates/chelis-cli/tests/issue_1582_vmap_std_scalar.rs` | 1 standing target, reused from Fast | 377 CLI targets | standing: 2 names, 12.7s; expansion: 5,685s |
| `tests/support/c_preprocessor.rs` or `examples/grad_extent_claim.ch` | 0 | 378 CLI targets | same as `main.rs` |
| `tests/support/helper_summary_fatal.ch` | 0 | 592 targets in CLI, compiler API, IR | 10,640s total, ~2,660s/shard |
| `Cargo.lock` | 3 runtime-bundle targets | 986 workspace targets | ~4,809s/shard, around the 80-minute execution deadline before build |

Cargo exposes 381 CLI targets; three have alternative owners, leaving 378 eligible. Forty-five standing targets can recur in expansion. Of 728 baseline target weights, 45 eligible CLI targets use the 30s fallback. #2586 executed 4,854 **actual** cases in 599 targets (`reports/ci_audit_report_overlap.md:16-26`). Ordinary PRs seed their owning or path-rule packages, require added/modified integration roots, and expand to the remaining targets. Targeted rebases additionally take reverse dependents (`scripts/ci_change_owned.py:669-708,1660-1690`). Fast covers default-feature lib/bin units and standing targets.

## Narrowing versus balancing

Existing integration binaries already give exact selection for edits to their roots. Splitting `tests/cli.rs` could shrink later **required** test edits; expansion would stay broad. Case IDs, past failures, durations and semantic labels cannot prove effect: helpers, macros, paired controls and fixture reads cross boundaries. `required-features` seals selected compilation, not omission. The withdrawn Fast/expansion reuse experiment found 466 same-named cases with differing feature closures; just 22 matched package configuration, worth 0.73s of test bodies (`reports/ci_expansion_dedup_pr_report.md`).

`chelis-cli` has one binary and no library target (`crates/chelis-cli/Cargo.toml`). Subcrates isolate their own source; CLI tests still consume the binary. A split needs independent tests. `tests/support/c_preprocessor.rs` has one visible `#[path]` consumer, `chelis-cli::capacity_census_tripwire` (86 raw names, 20s weight). `tests/support/helper_summary_fatal.ch` has two CLI/API integration consumers and one IR lib-unit consumer via `include_str!`; Fast covers the lib unit. Both merit exact owners after a runtime-reader audit. The manifest lacks exact-target owners.

Expansion batches up to 16 same-package/feature targets per command, capped at 300s of group weight; required shards use singletons. Every nonempty shard builds workspace lib/bin products. #2586 made nine isolated builds (Fast + four required + four expansion), plus Rust policy. A 360-target CLI expansion spent 119-195s per shard on builds (~693 runner-seconds total); execution varied 616-1,481s despite ~1,088s assigned each (`reports/ci_shard_analysis_20260927.md`). Fewer jobs save paid builds but may lengthen elapsed time. Measure both.

Case sharding might help `runtime_extent_claim_preparation`: 29 raw names, ~914s target execution in #2586, longest observed case ~283s. It duplicates compilation; Nextest already overlaps cases. It cannot divide one-case wire/binding censuses (~18-19 and ~28-44 minutes); improve their inner work. Fast may remain the PR critical path (`reports/ci_audit_report_timing.md`, `reports/ci_slow_tests_report.md`). Case sharding changes balance, not selection.

## Safeguards and ranked PR sequence

Omission needs checked input owners and fallback for unknown readers. Retain exact-head digests, feature sets, listing-to-JUnit equality, manual gates, and introduced/inherited/**unrun** counts. Classify both sides of renames; new case IDs must appear in the candidate listing. Negative tests should add an importer/glob reader, alter a helper/feature, rename target/case, and remove a receipt. Refresh weights from authenticated complete hosted receipts; disclose age and 30s fallbacks. Stale weights may alter balance, never selection. Green with unrun targets is incomplete coverage.

1. **Exact shared input:** In `.config/ci-test-targets.toml` and `scripts/ci_change_owned.py`, add an exact-target rule for `tests/support/c_preprocessor.rs`; update `scripts/test_ci_change_owned.py`, workflow-routing tests and `docs/ci_validation.md`. Enforce sole ownership or fall back. Goal: 378 expansion -> 0, one required target, complete exact-head receipt and no unexplained introduced failure versus paired full expansion.
2. **Second shared input:** Audit `tests/support/helper_summary_fatal.ch`, then require its two integration targets plus Fast IR lib unit. Update the same manifest, planner tests and docs. Goal: 592 expansion -> 0, two required targets, fewer hosted runner-minutes in paired exact-head runs, no omitted failing owner.
3. **Test-root shadow:** Have `scripts/ci_change_owned.py`, its tests and `docs/ci_validation.md` record hypothetical isolation while retaining full runs. Inventory `include!`, `#[path]`, runtime readers, scanners and scripts. `crates/chelis-cli/tests/cli.rs` could become 1 required / 0 expansion only with an executable closure guard. Require zero unexplained readers or changed-case omissions across 20 exact-candidate shadow plans and second-reader mutation controls; samples do not prove coverage.
4. **Case pilot:** In `scripts/ci_change_owned.py`, its tests, CI shard workflow and `docs/ci_validation.md`, partition `runtime_extent_claim_preparation` after candidate listing; seal case union and features. Across three paired hosted runs require >=20% lower median slowest-shard time, no worse PR completion, <=25% more runner-minutes, and zero missing/duplicate cases. Stop if Fast or compilation erases the gain.