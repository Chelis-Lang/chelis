# Performance Baseline

This is the current active index for performance baselines and manual performance gates.
Detailed point-in-time measurements live under [`archive/perf/`](archive/perf/).

## Phase J Compiled Artifact Caching

Current historical baseline: [`archive/perf/perf_baseline_phase_j.md`](archive/perf/perf_baseline_phase_j.md).

Investigation detail: [`archive/perf/perf_baseline_investigation.md`](archive/perf/perf_baseline_investigation.md).

Summary of the Phase J measurement:

- Coral `chelis test tests/` post-cache wall clock was `9 m 17.8 s` on this workstation.
- The pre-cache comparison was `6 m 40.1 s` on the same workstation and corpus.
- The `< 30 s` headline target was not met.
- The result is a historical baseline, not a regression bound.
- The documented bottleneck was worker-side re-entry into the legacy `prepare_eval` path
  plus compiled-context serialization and decode overhead.

The reproducible manual gate remains `phase4_perf_baseline` in
[`manual_gates.md`](manual_gates.md). Update this file only with concise current status;
move long dated measurement logs into `archive/perf/`.

## Node-Local Test Parallelism

Current baseline from a Coral checkout on one workstation with
`target/release/chelis`:

- `chelis test tests/ --jobs auto`: `30.64 s` for 65 tests, 0 failures.
- `chelis test tests/ --jobs 1`: `40.28 s` for the same corpus, 0 failures.

`--jobs auto` uses `min(selected_test_files, available_parallelism())`; it has no
fixed cap. Output remains deterministic in discovery order, so CI logs do not depend
on worker completion order.

## Package-context prove latency (chelis#924)

Reef graph preparation is persisted across processes in a versioned,
integrity-checked cache. Its determinant includes the compiler version,
canonical package root, every source and manifest, `reef.lock`, and published
archive/shell identities. `LocalRegistry` packages participate in both this
cache and the compiled-context cache. “Every source” means a live inventory of
relative paths and exact bytes, not only modules remembered by a cached graph;
additions, deletions, and renames invalidate it. Cold construction requires
matching snapshots before and after graph preparation so a concurrent edit
cannot publish a mismatched graph/hash pair.

After a property verdict, `chelis prove` checks every declaration in the
selected entry module and follows linker-resolved references transitively.
Unreachable declarations elsewhere in an installed shell are not rechecked.
The manual Shoals cold/warm oracle is recorded in
[`manual_gates.md`](manual_gates.md).

On 2026-07-30, the final development binary for the chelis#924 change
(including complete source-inventory and mutation-race hardening) completed
that oracle against the published Shoals 0.24.1 artifacts in 1.38 seconds
cold and 0.75 seconds warm. Both runs exited successfully immediately after
emitting identical passing property and summary NDJSON. These measurements
characterize the development change; the release gate must repeat them with
the published binary.
