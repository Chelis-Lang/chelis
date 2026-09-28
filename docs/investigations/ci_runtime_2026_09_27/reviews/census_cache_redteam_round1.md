# PR #2685 red-team round 1

Reviewed exact pushed head: `0bcc0ea09f081081e61f807a2abc08d4223a515b`.
Verdict: satisfied; round closed. No in-scope P0, P1, P2, or P3 findings. No out-of-scope defect was identified.

## Executed evidence

All commands below ran in `/Users/robertronan/chelis-worktrees/census-cache-review-20260927`.

- `.venv/bin/python -m unittest discover -s scripts -p test_capacity_census_wire_calls.py`: exit 0, 32 tests passed. This includes positive and negative driver receipt/source controls, scope namespace controls, defining-artifact origin/identity controls, and construction dependency selection/path-escape controls.
- `python3 scripts/reap_orphans.py`: no repo build/test processes before the build. The author retained the CPU slot until root explicitly handed it off; no review build or selected-source mutation started before that handoff.
- `.venv/bin/python /Users/robertronan/chelis-worktrees/census-cache-review-probe.py`: exit 0, terminal PASS. Durable probe script records the complete reproduction. It ran compiler-json successfully (format 2), changed the first `mod` in `crates/chelis-python/src/lib.rs` to `???` while preserving exact file size and mtime, then invoked native-bindings against the shared warm target. Cargo rejected the actual invalid source with exit 101 and `expected item, found ?`; no prior scope packet was returned. The probe restored the original source and timestamp, then successfully collected native-bindings (format 4) and compiler-json (format 2). The three successes used distinct temporary `-o` paths, the same binding Cargo directory, and current matching source hashes. This tests scope switching, failure after a cached success, recovery, and reverse scope switching. Source restoration runs in a `finally` block.
- Probe receipts: `/Users/robertronan/chelis-worktrees/census-cache-review-20260927/target/review-probe-results.json`. Successful collection durations were 156.964s, 5.823s, and 1.236s; these are probe timings, not a baseline comparison or CI performance claim.

## Claim coverage

Reviewed all five changed files against the candidate's base. `capacity_census_typed.py` orchestrates compiler-json then native sequentially. Both now use one source-bound wrapper and binding target; wire retains its separate target. The namespace unit tests execute that separation and reject unknown scopes. The wrapper delegates non-selected dependencies to ordinary rustc. Each selected compilation still receives a unique output path, requires a fresh packet, checks the scope format, verifies selected input source hashes, validates wrapper receipt/binary/source stability, and derives dependency provenance from that invocation's Cargo stream. Existing artifact-selection controls passed against the new binding path.

No saved ownership verdict or evidence input was introduced. The adversarial syntax change demonstrates that a successful prior scope cannot suppress recompilation of the selected crate. The restored format-4 then format-2 results demonstrate fresh collection in both directions through the shared target.

The parent supplied the author's full `chelis-python::capacity_census_bindings::registered_pyfunctions_match_the_reviewed_rustdoc_signatures` terminal PASS on this exact head (1/1, 27 skipped, 1289.577s). Per the brief, I did not rederive that acceptance run or the author's baseline timing. Parent also reported hosted checks green; I did not independently certify hosted CI or package expansion. I did not rerun the whole census obligation suite or test concurrent collectors; the change claims sequential use. There is no new user-facing CLI, example, or language semantic contract in this diff.

## Restoration and handoff

`git status --porcelain`: empty. `python3 scripts/reap_orphans.py`: no repo build/test processes. `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/census-cache-review-20260927/target` at 2026-09-27T19:40:16Z: FREE, exact reviewed head, clean tree (0 modified/staged/untracked/unmerged), lease free, no scoped processes. All selected-source probe changes were restored. CPU slot released to parent. Available for any repair verification.
