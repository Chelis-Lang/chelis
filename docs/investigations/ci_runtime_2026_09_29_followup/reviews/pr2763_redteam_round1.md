# PR #2763 — red-team round 1/3

**Verdict: satisfied; no in-scope P0–P3 findings.** Reviewed pushed `e6a2143e35a37c78d1e2634fede819a7cb1bcf92` (`perf/ci-fast-source-priority-20260929`). Defect classes: none. No out-of-scope defects observed.

## Execution and coverage

- `cargo nextest --version`: 0.9.136. Nextest parsed the changed profile. The exact expression `binary_id(/^chelis-compiler-api$/) & test(/^source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline$/)` selected **one active, non-ignored** test in the compiler API library; the other 560 cases in that binary were expression mismatches.
- `cargo nextest run -p chelis-compiler-api --lib --locked --profile ci-fast --ignore-default-filter --no-fail-fast --test-threads 1 -E 'binary_id(/^chelis-compiler-api$/) & (test(/^source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline$/) + test(/^source_arch::a_break_path_continues_after_the_loop$/) + test(/^source_arch::a_labeled_break_terminates_the_outer_loop_body$/))'`: 3/3 PASS. The guard started first and passed in 115.833s; both ordinary tests followed.
- The identical three-test command with `--profile ci-full`: 3/3 PASS. Both ordinary tests ran before the guard, which passed in 141.052s. This control supports that priority is effective in `ci-fast` and is not inherited by `ci-full`. These serial probes establish order with one worker; they do not measure hosted wall-clock speedup.
- Ran `cargo nextest list -p chelis-compiler-api --lib --locked --profile ci-fast --ignore-default-filter --message-format json` against both `origin/main`'s config via `--config-file` and the reviewed config. Resolved library testcase maps were identical: 561 matches, zero ignored. The temporary baseline config was removed.
- Parsed both TOML configs with Python `tomllib`: the sole effective delta is this one `ci-fast` override, with priority 100. All other profile, filter, timeout, and JUnit values are identical. `git diff --check origin/main...HEAD` passed.
- Checked `.config/nextest.toml` profile documentation and Nextest override documentation, `docs/ci_validation.md`, `.github/workflows/ci.yml`, and `scripts/ci_test_targets.py`. Hosted Fast runs `chelis-gate ci-fast`; that driver invokes Nextest with `--profile ci-fast --ignore-default-filter`, checks its exact target/test listing, and retains the same JUnit and coverage receipt path. The diff changes no Cargo target selectors, required features, ignored status, or receipt code.

**Unvalidated:** hosted speedup and completion of the running hosted CI checks. The local serial runs establish scheduling behavior only.

**Restoration and final state:** Removed the two focused-probe JUnit files; no tracked files or temporary config remain. `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/ci-fast-source-priority-20260929/target` at 2026-09-29T16:09:45Z: `VERDICT: FREE`, exact reviewed head, clean tree (0 modified/staged/untracked/unmerged), no scoped processes. Available for repair verification.
