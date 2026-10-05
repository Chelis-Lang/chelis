# Axis contract oracle comparison

PR #2851 remains a draft discussion spike. This experiment does not decide whether
Chelis should adopt axis-core or either proof tool in production CI.

The baseline is frozen **after** merging `main` at
`b4cb6667abc86f2196108276f83cd4da641ee01f`, preserving its permutation diagnostics,
and admitting the axis-core leaf into the pipeline's approved transitive closure.
The campaign manifest records that post-merge commit and the content hashes of the
compiler, contracts, test inputs, harnesses, lockfile, runner, and witness probe.
No contracts or loop annotations are mutated. Each mutated executable is compiled
before it enters the viable denominator.

## Experiment contract

The authoritative experiment oracle is the recorded execution of
`scripts/axis_oracle_comparison.py`: `calibrate`, `cost`, `run`, then `vermilion`,
with a frozen manifest, per-mutant receipts, independent witnesses, and manual
classification of every unwitnessed survivor. Helper unit tests support this oracle;
they do not establish proof effectiveness.

`cargo-mutants 27.1.0 mutants --list --json -p chelis-axis-core -f
crates/chelis-axis-core/src/verified.rs` returns an empty list: the four functions
are inside `verus!`. The fallback mutates comparison, arithmetic, Boolean and
zero/one executable tokens directly. Its deterministic inventory preserves every
specification, invariant, comment and unmutated source span. This campaign tests
that operator inventory, not all possible compiler defects.

Every viable mutant runs through all three test suites: the axis contract tests,
the complete type-checker and IR unit and integration suites (`--all-targets
--no-fail-fast`). The
test column therefore measures integration detection. Kani and Verus measure the
four executable contracts. A failed proof is distinct from a contract violation:
Verus receives detection credit only with a concrete input independently evaluated
against a separate executable contract model. An unwitnessed proof rejection is
reported separately; it is evidence of proof fragility only after equivalence is
established. Tool errors, timeouts and insufficient unwinding receive no detection
credit. Panics on contract inputs are concrete violations of the executable safety
obligation. Nontermination without a reproduced finite violation is inconclusive.
Executable mutant tests have an 8 GiB process-group RSS ceiling and concrete
probes have a 512 MiB ceiling, so a mutant that continually appends cannot exhaust
the shared workstation. A resource-limit termination is inconclusive.

Kani restates the contracts as ordinary Rust assertions; it cannot read erased
Verus specifications. Rank is symbolic and bounded; axis and signed-axis values
retain their full machine widths. Slice lengths independently range from zero to
bound plus one, covering wrong arity too. Normalization has no allocation and is
calibrated beyond the slice harnesses' search ceiling. Each solver receives 300
seconds; the driver has a separate 420-second ceiling. Calibration uses increasing
bounds and then bisects the first timeout interval. All completed bounds, solve
logs, unwinds, wall times and sampled process-group RSS are retained. The common
campaign bound is the minimum of the four largest completed bounds. A search
ceiling reached without a timeout is a lower bound on capacity, not a discovered
maximum. The false permutation postcondition is a negative control for both tools.

The audited call graph determines Kani execution:

| Mutated function | Harnesses that can observe it |
| --- | --- |
| `is_permutation` | admission, inverse |
| `normalize_axis` | normalization |
| `reduction_survivors` | survivors |
| `checked_inverse` | inverse |

Vermilion runs on the full baseline and disagreements: witnessed Verus detection
without a Kani catch, unwitnessed Verus rejection, or a Verus pass with an
independent violation. It uses the same mutated Rust text and fresh generated Lean
statements. Baseline proof installation retains its statement-hash checks. For
mutants, existing tactics are replayed against the new statements and Lean must
check the resulting terms; an old statement hash is never accepted as a proof.
Adapter refusals and tool failures remain inconclusive. The source-to-Lean
translation remains part of the trusted boundary.

## Production domain and costs

There is no small compiler-wide tensor-rank ceiling. `ShapeMetadata::checked_rank`
in `crates/chelis-abi/src/metadata.rs` admits ranks through `i32::MAX`, consistent
with [04-NUM-11]. Metal movement's `MOVEMENT_MAX_DIM = 8` is a target-specific limit,
not the compiler input space. The result must state whether the campaign bound
clears the ABI ceiling. Witnesses beyond that ceiling establish kernel defects,
but do not establish production reachability. Rust caller extraction and complete
compiler behavior remain outside the proofs.
Axis scalars are `i32` under [05-DIM-1] and the movement signatures. The kernel's
`i64`/`usize` interface is wider; witnesses also record whether their axis values
fit the language domain. Normalization probes prefer small signed values before
the full-width extremes. Being within these scalar and rank domains is necessary
for production relevance, but does not prove a particular caller reaches the input.

Cold measurements use an empty task-owned Cargo target with already installed
tools and warm OS/download caches. Warm measurements reuse that target. Wall time
includes compilation and orchestration; solver times remain in the Kani logs.
Memory is the sampled sum of process-group RSS, which can count shared pages more
than once and can miss short peaks. Installation storage, setup receipts, runtime
microbenchmarks and isolated artifact sizes are reported separately. Both proof
systems erase specifications; algorithm changes can still affect runtime.

## Reproduction

Use an isolated checkout at the frozen commit and a uv-managed Python 3.11.
Installed Kani 0.68.0 and the pinned Verus release are required. For each stage:

```text
.venv/bin/python scripts/axis_oracle_comparison.py STAGE
  --checkout /absolute/disposable/checkout --output /absolute/receipts
  --kani /absolute/cargo-kani --verus /absolute/verus
```

For `vermilion`, also pass `--vermilion /absolute/pinned/vermilion` after building
its pinned upstream tools. Do not run two stages against the same checkout or
target concurrently. Resuming retains completed mutant rows and individual Kani
trials; changing frozen inputs or bound settings requires a new output directory.

Measurements and survivor classifications are pending the frozen campaign.
