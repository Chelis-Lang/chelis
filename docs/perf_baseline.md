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

Current baseline from `/home/jeff/Documents/scratch/coral` on this workstation with
`target/release/chelis`:

- `chelis test tests/ --jobs auto`: `30.64 s` for 65 tests, 0 failures.
- `chelis test tests/ --jobs 1`: `40.28 s` for the same corpus, 0 failures.

`--jobs auto` uses `min(selected_test_files, available_parallelism())`; it has no
fixed cap. Output remains deterministic in discovery order, so CI logs do not depend
on worker completion order. LocalRegistry packages bypass the disk-cache hash probe
and still compile one in-memory context that is shared by every worker.
