# PR CI runtime investigation, September 2026

This directory preserves the reports produced during the September 23–27 PR CI
runtime investigation. The reports under `reports/` and `reviews/` are dated
snapshots: their measurements, candidate SHAs, and observations remain as
recorded, including statements that a PR was pending when the report was
written. This index records the disposition verified on September 27 after
those reports were written. It does not change the [CI execution contract](../../ci_validation.md).

## Findings and disposition

- A sampled #2586 PR CI run took 28.2 minutes. Its longest change-owned shard
  spent 22.3 minutes executing tests, including 914 seconds on
  `chelis-cli::runtime_extent_claim_preparation`; Fast Tests ran concurrently.
  A matching manual package expansion took 49.8 minutes, dominated by the
  binding and wire censuses at 28.3 and 18.4 minutes. Setup and queue time
  did not explain those critical paths. See the [timing audit](reports/ci_audit_report_timing.md).
- Main's standing integration inventory moved from 96 to 97 targets during the
  sampled period, while five runtime package rules were added. There was no
  wholesale transfer of nightly tests into required PR CI. Fast Tests ran
  slower after moving from a warm runner to `ubuntu-latest`, while the severe
  warm-runner queue waits disappeared. Different candidates and runner states
  limit a causal comparison. See the [selection audit](reports/ci_audit_report_overlap.md).
- Change-owned and manual expansion select disjoint target sets. In the exact
  #2586 candidate, Fast Tests and manual expansion nevertheless executed 466
  same-named cases across 57 standing targets. Those names did not prove
  equivalent Cargo feature closures. [PR #2676](https://github.com/Chelis-Lang/chelis/pull/2676)
  was closed without merge after that safety issue and a small safe-case
  benefit were measured. See the [deduplication report](reports/ci_expansion_dedup_pr_report.md).
- Rebase evidence reuse is designed for both an eligible rebase and a merge of
  `main`. The false parent-count failure came from reading a synthetic commit
  through a shallow graft. [PR #2675](https://github.com/Chelis-Lang/chelis/pull/2675)
  merged the raw-parent repair and local tests for both update forms. The
  quoted #2586 update also lacked a prior eligible receipt, so the fix alone
  would not establish that run could take a cheaper lane. A qualifying hosted
  base update after the repair remains the direct end-to-end observation.
  See the [rebase audit](reports/ci_audit_report_rebase.md).
- [PR #2678](https://github.com/Chelis-Lang/chelis/pull/2678) merged a broader,
  authenticated duration baseline for deterministic target sharding. A later
  360-target expansion still had 1,481-second and 616-second shards despite
  nearly equal assigned weights; the excess was primarily test execution.
  Grouping, old weights, and runner variance remain possible contributors.
  See the [sharding delivery](reports/ci_sharding_pr_report.md) and
  [subsequent receipt audit](reports/ci_shard_analysis_20260927.md).

Five further PRs merged after targeted profiling. These are local measurements,
not claims that an entire hosted workflow fell by the same amount:

| PR | Change | Measured local result |
| --- | --- | --- |
| [#2681](https://github.com/Chelis-Lang/chelis/pull/2681) | Batch sanitizer C fixture links; retain each separate assertion and sanitizer execution | Representative pair 34.49 to 28.14 seconds |
| [#2682](https://github.com/Chelis-Lang/chelis/pull/2682) | Preserve per-command and testcase timing artifacts | Telemetry only |
| [#2683](https://github.com/Chelis-Lang/chelis/pull/2683) | Reuse identical compiled extent fixtures; retain each CLI/C route and trap/value assertion | 120-case test 78.86 to 67.56–70.43 seconds |
| [#2684](https://github.com/Chelis-Lang/chelis/pull/2684) | Avoid copying a discarded scanner path frontier | Production scanner 77.74–79.27 to 72.81 seconds; hosted effect unresolved |
| [#2685](https://github.com/Chelis-Lang/chelis/pull/2685) | Share dependency builds across sequential binding-census scopes; retain fresh selected-crate compiles and verdict checks | Two collectors 311.36 to 160.28 seconds combined; full binding owner test passed in 1,289.58 seconds |

The [top-20 table](reports/current_top20.md) identifies each slow testcase,
its introducing PR, intended assertion, sampled historical failures, and
possible restructuring. The [source audit](reports/ci_slow_tests_report.md)
and [failure-history audit](reports/ci_slow_test_history_report.md) explain the
sample and its limits. No whole-group move to nightly was supported by that
evidence. The deep-chain cost is tracked by
[#2592](https://github.com/Chelis-Lang/chelis/issues/2592); its separate
[PR #2679](https://github.com/Chelis-Lang/chelis/pull/2679) was still open on
September 27 after an introduced census failure in package expansion.

## Report archive

These files preserve the original analysis and measurements. Some retain
absolute paths to the audit machine's task-owned worktrees or artifacts; those
paths are provenance, not portable repository dependencies. Links between
reports in this archive have been made relative.

### Initial audit and slow tests

- [Timing and runner audit](reports/ci_audit_report_timing.md)
- [Selection and overlap audit](reports/ci_audit_report_overlap.md)
- [Rebase reuse and census diagnostic audit](reports/ci_audit_report_rebase.md)
- [Twenty slowest tests](reports/current_top20.md)
- [Slow-test source and purpose audit](reports/ci_slow_tests_report.md)
- [Slow-test failure-history audit](reports/ci_slow_test_history_report.md)

### Repair, scheduling, and profiling

- [Rebase and census diagnostic repair report](reports/ci_repair_pr_report.md)
- [Withdrawn Fast/expansion deduplication report](reports/ci_expansion_dedup_pr_report.md)
- [Sharding delivery report](reports/ci_sharding_pr_report.md)
- [Sharding CI wait snapshot](reports/ci_sharding_ci_wait_report.md)
- [Later four-shard receipt analysis](reports/ci_shard_analysis_20260927.md)
- [Sanitizer fixture profile](reports/c_sanitizer_perf_report.md)
- [Census nested-build profile](reports/census_perf_report.md)
- [CLI extent fixture profile](reports/cli_fixture_perf_report.md)
- [Source-scanner algorithm audit](reports/source_scanner_algorithm_report.md)
- [Binding-census cache experiment](reports/census_cache_experiment_report.md)

### Independent review records

- [#2675 initial round](reviews/redteam_pr2675_round1_874c83d.md) and
  [repair verification](reviews/redteam_pr2675_round1_verify_b9807e8.md)
- [#2676 deduplication round](reviews/ci_expansion_redteam_round1.md)
- [#2678 sharding round](reviews/ci_sharding_redteam_round1.md)
- [#2681 sanitizer round](reviews/c_sanitizer_redteam_round1.md)
- [#2682 telemetry round](reviews/census_telemetry_redteam_round1.md)
- [#2683 CLI fixture round](reviews/cli_fixture_redteam_round1.md)
- [#2684 scanner round](reviews/source_scanner_redteam_round1.md)
- [#2685 census cache round](reviews/census_cache_redteam_round1.md)
