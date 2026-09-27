# Numeric census profiling and PR handoff (2026-09-27)

## Delivered

- PR [#2682](https://github.com/Chelis-Lang/chelis/pull/2682), head `d30770571bc4237703626cf9224369de1efb0bd3`, branch `agent/census-perf-20260927`, worktree `/Users/robertronan/chelis-worktrees/census-perf-20260927` (clean, no owned build processes at handoff). Based on `origin/main` `af5dd14c7ac6187a63e435b73551bbe3376865ce`; fetched main immediately before push and it was still that head. Internal PR has `no-changelog`.
- Both required change-owned and manual package-expansion shard executors now set `CHELIS_CI_TIMING_DIR` inside their existing uploaded output directory. A workflow test checks that the timing directory is under the exact `--output` path that the `if: always()` artifact step uploads, then reads a real JSONL record from the existing timing hook. Captured child stdout and environment values do not enter the record. Documentation identifies it as diagnostic data outside verdict authority.
- **This is telemetry only.** It makes no runtime-reduction claim and neither caches nor reuses a census pass verdict. The JSONL files are outside the authoritative receipt digest.

## Verification and live state

- `.venv/bin/python -m unittest scripts.test_change_owned_workflow scripts.test_ci_script_tests`: 18 passed.
- `python3 scripts/gate.py --fast`: PASS, 5 stages, 83 tests passed, 1 skipped, zero tracked changes; 335.5 seconds due to cold native dependencies.
- Phase 3 and Phase 4B change/acknowledgement reports passed with zero changed protected identities.
- At handoff, PR #2682 was open, GitHub `PR Contract Acknowledgements` and `PR Base Retarget Validation` succeeded, while initial CI detection jobs were still running. **No fresh red-team round or manual package expansion has run yet.** Root is taking ownership of review and final exact-head CI/expansion inspection. Do not merge from this report alone.

## Census observations

- A complete direct `capacity_census_typed.py wire` run on this Mac exited 0 and emitted a 13.3 MB report with 97 numeric authority rows. It took 522.8 seconds in a partly warmed local target. The largest timed nested commands were fresh compiler-API Rust compilation (138.9 s), CLI test build (95.5 s), mutation-control supervisor (94.9 s, including a 78.6 s publication probe), cache compatibility (73.5 s), and Tide (65.6 s). Local timing is not a hosted speedup estimate; machine contention and warm state were not controlled. JSON and timing files are under this worktree’s `target/census-profile/wire-full*`.
- Current code uses three separate Cargo targets for wire, compiler-JSON, and native-bindings invocation inventories. The wire invocation target alone occupied about 2.4 GB, with 1.6 GB under `deps`; duplicate dependency artifacts exist across private targets. Each scope still forces a unique fresh selected-crate `-o .../fresh.rmeta` callback and verifies source/report/Cargo artifact provenance. The schema codec path runs a deliberate `cargo clean -p ...` to defeat preserved-size/mtime stale-source mutation; do not remove that control.
- The nightly private-target cache is currently absent from retained GitHub cache inventory, and nightly and expansion use different runner setup. A restore-only cross-run cache step would likely miss or rebuild. No cross-run cache change was made.

## Separate measured cache experiment

Use a new dedicated worktree and isolated target. Share dependency artifacts only between compiler-JSON and native-bindings invocation scopes inside the binding verifier; keep wire separate so concurrent wire/binding execution stays independent. Retain each selected crate’s unique fresh callback, current-source hash, report-format, and defining-Cargo-artifact checks. Measure a complete cold and warm binding run before/after on one quiet machine, inspect Cargo rebuild/fingerprint and source identities, rerun the preserved-size/mtime mutation controls, then run concurrent wire and bindings to expose artifact races. Sharing may invalidate hashes in the first scope when the second runs; that must fail closed. Propose a separate speedup PR only if controls pass and savings are material. Root asked to reassign this experiment after this telemetry handoff.
