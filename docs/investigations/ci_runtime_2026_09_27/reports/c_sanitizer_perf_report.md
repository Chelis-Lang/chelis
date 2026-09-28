# C sanitizer fixture compilation, 2026-09-27

Opened [PR #2681](https://github.com/Chelis-Lang/chelis/pull/2681) at head `4bd1f2d3ba2559f28e17702974bfdcc6c55aa00d`, based on `af5dd14c7ac6187a63e435b73551bbe3376865ce`. Branch/worktree: `agent/c-sanitizer-batch-20260927`, `/Users/robertronan/chelis-worktrees/c-sanitizer-batch-20260927`. `no-changelog` label applied. I did **not** merge.

The three slow `chelis-backend-c::exec_compile` positive matrices launched a complete native compile/link for each cell. The PR groups 198 positive cells into 26 per-dtype compile/link commands: 72 affine, 54 sparse, and 72 non-Int64 permute/expand. Every generated kernel remains its own C translation unit, so its private helpers cannot collide. A group shares one harness translation unit and linked binary, but **each cell executes in a separate ASan/UBSan process**. Harness functions have unique `case_N` entries; sparse helper functions have unique `bits_N`/`fill_N` names. Each successful C harness emits its case index, and Rust checks that index, preventing a misrouted dispatcher from silently repeating one passing cell. Exact bit, dtype, rank, shape, and coordinate-map assertions remain.

The nine Int64 permutation positive cells and all malformed-metadata, inverse-permutation, and coordinate-bypass mutation controls retain the original individual compile/run path and order. Separate affine-bounds and sparse-invalid-domain tests are unchanged.

## Measurements and validation

- Uncontended same-machine affine pair using the saved original test binary: **34.49 s before**, **28.14 s after**, both passed: 6.35 s / 18.4% local reduction. The after measurement preceded the static extension to the other two matrices; affine's algorithm was unchanged by that extension. Hosted speedup remains unmeasured.
- Final exact-head focused Nextest run: **5/5 pass**, including all three changed matrix tests plus affine-bounds and sparse-invalid-domain controls. Measured case durations were 29.07 s affine, 36.45 s permute/expand, 23.26 s sparse. These ran concurrently in Nextest, and the latter two lack a matched local baseline, so do not infer their speedup from prior hosted JUnits.
- `python3 scripts/gate.py --fast`: **6/6 stages pass**, **83 tests pass**, 1 skipped, zero tracked changes, 361.0 s (cold GMP/MPFR/MPC build). Phase 3 and Phase 4B change reports pass with zero required acknowledgement lines; saved PR body passes acknowledgement enforcement. `cargo fmt --all -- --check` and `git diff --check` passed.
- PR hosted checks began after opening and were pending at 18:19 UTC. Exact-head package expansion and red-team review remain for the orchestrator. The PR is not merge-ready yet.

At 18:19 UTC, `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/c-sanitizer-batch-20260927/target` reported `VERDICT: FREE`, clean tree at the pushed head, no scoped process, and the fast-gate PASS as history. The worktree and warm target are available for a fresh review; no writer will touch them until handoff.
