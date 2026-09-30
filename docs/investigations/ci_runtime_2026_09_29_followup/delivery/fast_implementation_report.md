# Fast Nextest priority trial implementation

**Verdict:** The scheduling override is locally committed and its exact test filter is verified. Hosted timing remains for the orchestrator's PR.

- Worktree: `/Users/robertronan/chelis-worktrees/ci-fast-source-priority-20260929`
- Branch: `perf/ci-fast-source-priority-20260929`
- Commit: `e6a2143e35a37c78d1e2634fede819a7cb1bcf92` (`perf(ci): start source guard early in fast tests`)
- Base: freshly fetched `origin/main` at `465a3623be19bd68ada6c39aa7552b63e0ccf8c3`

The sole tracked change adds one `[[profile.ci-fast.overrides]]` rule in `.config/nextest.toml`, assigning priority 100 to the exact `chelis-compiler-api` library binary and `source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline` unit case. No selection, ignored state, timeout, or other profile changed. The #2754 hosted `ci-fast/test-list-0.json` records `binary-id: chelis-compiler-api` and `kind: lib`; the local Nextest listing confirmed the same identity.

## Local verification

- Read `docs/guard_changes_for_pr_authors.md`; created the dedicated worktree from fetched `origin/main` and its own Python 3.11 venv with `uv venv --python 3.11`.
- `python3 scripts/reap_orphans.py` reported no repo build/test processes before the Cargo listing and after it.
- `cargo nextest show-config --profile ci-fast version` accepted the modified configuration; local cargo-nextest is 0.9.136.
- `.venv/bin/python -m unittest scripts.test_ci_test_targets scripts.test_nextest_profile_partition.FilterTextTests scripts.test_ci_cadence`: **45 tests passed**.
- `CARGO_BUILD_JOBS=2 cargo nextest list -p chelis-compiler-api --lib --locked --profile ci-fast --ignore-default-filter -E 'binary_id(/^chelis-compiler-api$/) & test(/^source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline$/)'`: **passed**, listing exactly `chelis-compiler-api source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline`.
- `git diff --check` passed before commit. After commit, `git status --short --branch` reports a clean branch one commit ahead of `origin/main`.

An initial unscoped invocation of the three test modules encountered missing PyYAML in the fresh venv and entered the profile-partition module's broad workspace compile. I stopped that invocation, installed PyYAML into this worktree's venv, and ran the focused checks above. The interrupted invocation is not counted as a pass. I did not run a local performance benchmark while another worktree was building, and did not push, open a PR, dispatch CI, or spawn an agent.
