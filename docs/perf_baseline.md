# Performance checks

Performance measurements depend on the compiler build, package corpus, cache
state, and host. Record those inputs with a measurement rather than treating
a workstation timing as a product guarantee.

## Run the acceptance commands

The [manual gates](manual_gates.md) define the commands, prerequisites, and
success criteria for compiled-artifact caching, test parallelism, and
package-context proof latency. Use the gate that owns the behavior being
measured. The [evaluator performance oracle](eval_performance.md) covers
operation-specific evaluation costs.

## Package caches and test workers

Reef graph preparation uses an integrity-checked cache keyed by the compiler
build, package root, source and manifest inventory, lockfile, and resolved
package identities. Source additions, deletions, and renames invalidate it.
Compiled contexts reuse checked library state when their identity matches.

`chelis test --jobs auto` chooses the smaller of the selected file count and
available parallelism. `--jobs 1` runs serially. Results print in discovery
order, independent of worker completion order.

## Measurement records

[Archived baseline notes](archive/perf/performance_baseline_notes.md) collect
workstation measurements and their limitations. The detailed
[compiled-cache measurement](archive/perf/perf_baseline_phase_j.md) and
[investigation](archive/perf/perf_baseline_investigation.md) retain their
original corpus and timing evidence.
