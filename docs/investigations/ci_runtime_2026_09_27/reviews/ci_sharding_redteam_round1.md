# PR #2678 red-team round 1

Reviewed head: b1485224bc5482a59498a3740e0adad3ed6a6478 (base e65735e8c).

Findings: none in scope at P0/P1/P2/P3. I am satisfied; round 1 is closed. I remain available for verification if the author changes the candidate.

Executed validation:
- `.venv/bin/python -m unittest scripts.test_ci_change_owned.DurationBaselineTests`: 7 passed.
- An inline unittest loader selected ShardingAndExecutionTests and ReportTests methods containing tamper, missing, duplicate, uncovered, execution, duration, balanced, or coverage: 11 passed.
- Inline Python adversarial probes against `_expansion_duration_observations`: negative and NaN run times, missing target timing, foreign command-group membership, NaN JUnit time, skipped testcase, and missing JUnit cases all rejected. A modified JUnit sidecar without a matching receipt digest was rejected by `build_duration_baseline`.
- Independently rebuilt the baseline using `git show e65735e8c:.config/ci-change-owned-durations.json` as a temporary seed and all three supplied plan/receipt directories. The resulting complete JSON equals the committed baseline exactly, including five sources, values, and sample counts. Receipt and sidecar authentication ran through the production importer.
- Independently replayed the recorded measurements: each source has 599 targets; refreshed assignment counts are 1/189/203/206, with `chelis-python::capacity_census_bindings` alone. Recorded longest elapsed times reproduce 46.3/48.1/74.7 minutes. Proportional command-work replay reproduces the documented 33.4/32.9-minute first two estimates. The third replay is 47.8 minutes. Using the builder's per-target longest-test floor in the replay instead gives 34.0/33.5/47.8; this is consistent with the documented model being proportional command work rather than exact target runtimes.

Claim coverage: checked grouping without repeated command wall time, consistent group rows and partition validation, JUnit/receipt exact identity agreement, finite nonnegative timing enforcement, sidecar integrity, seed/source retention, baseline reproducibility, deterministic assignment and report coverage negatives. No execution/report implementation changed. Docs correctly distinguish estimates from actual new dispatch results and explain unknown-target fallback limitations.

Unvalidated: actual hosted runtime improvement on this candidate; cross-run training estimate 37.9–39.6 minutes; current hosted CI status. No Rust build or broad CI-contract rerun was needed; the author supplied those prior results. Source artifacts were locally authenticated, not independently redownloaded from GitHub.

Final status: supplied worktree remains clean at the reviewed SHA (`git status --short` empty). Probes used temporary directories and changed no tracked source. No build processes were started.
