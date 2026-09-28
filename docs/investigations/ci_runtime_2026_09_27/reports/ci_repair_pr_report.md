# Core CI repair PR report — 2026-09-27 16:08 UTC

PR: https://github.com/Chelis-Lang/chelis/pull/2675

Final pushed head: `b9807e86450627460c0e6a68c87911418a005a5e` on `agent/ci-shallow-reuse-2673`, based on `origin/main` `71815c2802b96767577fdc0144aab5de68535f77`. Worktree: `/Users/robertronan/chelis-worktrees/ci-reuse-expansion-2673`, clean after push. This PR remains open and unmerged. Issue #2673 is assigned to `rlronan` and remains open.

## Changes

- `scripts/ci_rebase_reuse.py`: read synthetic candidate parents from the raw commit object via the existing candidate-identity helper. A shallow graft no longer hides the two recorded parents; head-parent, receipt, forward-base, candidate-identity and frontier checks remain in place.
- `scripts/test_ci_rebase_reuse.py`: mark a synthetic merge shallow in a temporary Git repository and prove both `base-rebase` and `base-merge` classifications reach the trusted docs-only reuse decision with an eligible prior receipt.
- `scripts/capacity_census_cache_publication.py`: on an unexpected standalone compile outcome, include up to 3,500 characters beginning at rustc's earliest plain or coded error line.
- `scripts/test_capacity_census_cache_publication.py`: verify the bounded diagnostic and the first-error ordering.
- `crates/chelis-compiler-api/src/cache_envelope.rs`: document the standalone census and its fixed external crate set.
- `changelog.d/ci_rebase_cache_diagnostics.fixed.md`: release fragment for the two repairs.

This bounded #2673 repair leaves optional extraction of `cache_envelope` into a dedicated crate for later structural work; the PR body says so. The manual Fast Tests/package-expansion deduplication is a separate PR owned by another agent and is not claimed here.

## Local evidence and review

- At initial head `874c83d8b3f4920a650a0b57e9d6248dbadad76c`: `PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_ci_rebase_reuse scripts.test_capacity_census_cache_publication` passed 30 tests; `python3 scripts/gate.py --fast` passed 6 stages and 83 selected tests, changing no files.
- Red-team round 1 on that pushed head found one in-scope P2, class `wrong primary-diagnostic selection`: a coded rustc error could be shown ahead of an earlier plain error. The reviewer executed real-rustc and shallow-base-update probes. Initial report: `/Users/robertronan/chelis-worktrees/redteam_pr2675_round1_874c83d.md`.
- Local repair commit `b9807e86450627460c0e6a68c87911418a005a5e` added first-error selection and a regression. The same standing reviewer reran its real-rustc reproduction, exercised adjacent stderr cases, passed all 31 focused tests, and stated the P2 closed and round satisfied. Verification report: `/Users/robertronan/chelis-worktrees/redteam_pr2675_round1_verify_b9807e8.md`.
- Before pushing the repair, `python3 scripts/gate.py --fast` passed 6 stages and 83 selected tests in 66.7 seconds, changing no files. PR body records the round and exact heads. Phase 3 and Phase 4B change reports required no acknowledgement lines.

## Hosted evidence and limits

- Final-head CI run [36332003880](https://github.com/Chelis-Lang/chelis/actions/runs/36332003880) was **in progress** at 2026-09-27 16:08 UTC; it has no terminal verdict yet. The initial-head CI run [36331387868](https://github.com/Chelis-Lang/chelis/actions/runs/36331387868) was also in progress, with four change-owned buckets, planner, backend sanitizer, and early policy checks successful. Final-head checks must be read when terminal; initial-head successes are not final-head acceptance.
- Manual package expansion was not dispatched for this open, unmerged PR. Dispatch on the exact final candidate only when content/review is settled and the eventual merge decision is near.
- Because this PR changes a CI-policy script, its own candidate receipt is reuse-ineligible. The shallow-reuse mechanism has executed local Git fixtures for both update forms, but a hosted trusted-base reuse exercise must occur after the verifier becomes main-owned.
