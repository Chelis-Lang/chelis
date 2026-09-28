# Binding census within-run Cargo artifact reuse

**PR:** [#2685](https://github.com/Chelis-Lang/chelis/pull/2685) at `0bcc0ea09f081081e61f807a2abc08d4223a515b` (`agent/census-cache-exp-20260927`). Dedicated worktree: `/Users/robertronan/chelis-worktrees/census-cache-exp-20260927`. The branch started from `origin/main` `092b30974b2cb3b37efae2227fc3a54663ef5b8b`. It is pushed, labelled `no-changelog`, and remains unmerged.

## Change and invariant

The sequential compiler-JSON and native-bindings collectors now use one source-bound wrapper directory and private Cargo target during a single bindings verifier run. Wire collection retains its separate target. Each binding scope still invokes selected-crate `cargo rustc` with its own fresh `-o fresh.rmeta`; the wrapper callback, format checks, current-source hash, Cargo artifact provenance, native execution, and numeric-authority checks remain. No prior verdict is cached or reused. The PR also routes two formerly unclassified census scripts to their existing nightly dtype-oracle owner in `.config/ci-test-targets.toml`; this does not change the test cadence.

## Measured effect

On one Mac, at unchanged base `af5dd14c7ac6187a63e435b73551bbe3376865ce`, two isolated cold collector targets took **153.386 + 157.972 = 311.358 seconds** separately, versus **154.530 + 5.747 = 160.277 seconds** with the shared binding target: **151.081 seconds saved (48.5%)** in these two collectors. The separate runs built two ~2.9 GiB Cargo trees; the candidate built one. The second candidate stream marked 261 dependency artifacts fresh and compiled only the selected `chelis_python` crate. Both scopes produced format 2/4 evidence, stable driver SHA and crate provenance, and first-scope defining/fixture artifact hashes persisted through the second. These are local collector timings, not a measured hosted or full-PR runtime reduction.

## Correctness evidence

- A same-size source edit preserving `chelis-python/src/lib.rs` mtime changed the native collector's current-source SHA; restoring the source restored the original SHA. The tracked file was restored.
- Focused Python collector, compiler-JSON, typed census, and native-authority tests: 54 passed.
- Complete final-head fast gate: five stages passed, 83 tests passed, one skipped, zero file changes. An initial gate failure exposed missing CI path routes; the committed routes passed the full rerun. Phase 3 and Phase 4B reports required zero acknowledgement lines.
- Exact pushed-head owner test `registered_pyfunctions_match_the_reviewed_rustdoc_signatures` in `capacity_census_bindings` passed **1/1 selected in 1289.577 seconds**. It runs live registration and `bindings-discovery`, including compiler-JSON, native execution, artifact, and authority checks. Its nested timing trace recorded fresh wire compiler-API collection at 112.606 seconds, first `chelis_python` binding collection at 123.721 seconds, and second binding collection at 2.880 seconds, all exit zero. The exact selector skipped 27 other tests in that binary.
- Hosted required PR checks on the pushed head are green. Description-triggered Changelog and PR Contract Acknowledgements reruns also passed. Fresh red-team round 1 closed satisfied on this exact head with no findings. The reviewer passed 32 unit tests and an adversarial same-size/mtime syntax mutation through the warm shared target: the selected compilation failed on invalid current source, then recovered in both scope orders after restoration, each with a distinct fresh output. Exact-head package expansion [run 36345264264](https://github.com/Chelis-Lang/chelis/actions/runs/36345264264) is pending.

## Residual limits

Hosted transfer of the local 151-second saving is unmeasured. The initial collector comparison preceded a clean rebase onto `092b3097`, while the full owner test and hosted checks are on the pushed head. The PR deliberately leaves wire collection and any cross-run cache unchanged. The exact-head package-expansion receipt and final required-check readback must finish before merge.
