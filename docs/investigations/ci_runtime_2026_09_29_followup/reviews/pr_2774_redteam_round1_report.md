# PR #2774 red-team round 1

**Verdict:** Satisfied. No in-scope P0, P1, P2, or P3 findings. This round is closed on pushed head `d699109991d5c9f32d6df60d0c0cf2fdacff09da`. I remain available for same-reviewer verification if a later local repair needs it.

## Findings

None. No out-of-scope defect was discovered.

## Execution and coverage

- Exact head: `git rev-parse HEAD` and `git rev-parse origin/fix/source-guard-directory-failclosed-20260929` both returned `d699109991d5c9f32d6df60d0c0cf2fdacff09da`. `git diff origin/main...HEAD -- crates/chelis-compiler-api/src/source_arch.rs` showed the single reviewed file. `python3 scripts/reap_orphans.py` found no repo build/test processes before testing.
- Focused positive and negative suite: `cargo nextest run -p chelis-compiler-api --lib -E 'test(source_arch::)'` passed **83/83** at the reviewed head. This includes the real readable-workspace guard (passed in 102.787 seconds), all five guarded-root inclusion, missing Reef root, nested directory and injected iterator-entry failures, dangling-symlink metadata failure, and planted semantic-pipeline duplicates.
- Adversarial omitted-source probe: `.venv/bin/python /Users/robertronan/chelis-worktrees/ci-selection-assessment/target/2774-redteam-permission-probe.py` created an untracked `omitted.rs` inside a mode-000 directory under `crates/chelis-pipeline-core/src`, then ran `cargo nextest run -p chelis-compiler-api --lib -E 'test(source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline)'`. The test failed as expected (nextest exit 100), reporting `failed to enumerate guarded source directory` with the exact unreadable directory path and `Permission denied (os error 13)`. The probe restored permissions and removed its directory and file in `finally`.
- Contract comparison: `docs/investigations/compiler_pipeline_inventory.md` names the same five roots as `GUARDED_SOURCE_ROOTS` and expressly limits the inventory to those trees. The implementation propagates enumeration, iterator-entry, and metadata errors with a directory or entry path; `actual_workspace_findings` turns them into a failed guard. Its source parsing and duplicate-detection path remains covered by the existing positive and planted negative tests.

## Unvalidated and final state

I did not run `scripts/compiler_pipeline_oracle.py`, the full local gate, or hosted CI in this round. The supplied busy signal records an earlier `gate fast` pass; that is history, not my test result. Iterator-entry errors were exercised by the injected fixture, not a live filesystem race. Each root's error handling uses the same collector; this round did not separately fault-inject every root.

Final `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/source-guard-directory-failclosed-20260929/target` at `2026-09-29T18:22:43Z`: `VERDICT: FREE`, exact head above, `clean (0 modified, 0 staged, 0 untracked, 0 unmerged)`, no scoped processes. `git diff --check` passed; the probe path no longer exists.
