# Axis contract oracle comparison

PR #2851 remains a draft discussion spike. This experiment does not decide whether
Chelis should adopt axis-core or either proof tool in production CI.

The baseline is frozen **after** merging `main` at
`b4cb6667abc86f2196108276f83cd4da641ee01f`, preserving its permutation diagnostics,
and admitting the axis-core leaf into the pipeline's approved transitive closure.
The campaign manifest records that post-merge commit and the content hashes of the
compiler, contracts, test inputs, harnesses, lockfile, runner, and witness probe.
The frozen experiment commit is
`d68ae1fc1cce832275977fecc3a705640703a99d`; later reporting, documentation and
runner hardening do not alter the measured kernel or compiler. The published
campaign uses that commit's runner snapshot, whose content hash is in the manifest.
The subsequent runner preserves an explicit Python interpreter, reaps command
groups on interruption and checks that Vermilion's structural verdict belongs to
the current case. It also inserts the unchanged Lean helper idempotently when
resuming a generated twin. The recorded Vermilion verdicts are audited against their case
paths and lowering results before reporting.
The frozen Verus parser also omitted the diagnostic `precondition not met` for
bounds safety. Reporting reclassifies those receipts from tool error to proof
rejection using the unchanged execution logs, retains the recorded label and
adjudication reason in the mutant table, and still requires an independent witness
for detection credit. Actual parser/tool failures are never promoted.
The frozen Kani parser likewise expected a failure description before its status,
whereas Kani 0.68.0 prints the status first. Reporting audits the failed check
blocks and downgrades mislabelled assertion catches to unwind failure or reached
unsupported construct. Recorded labels and adjudication reasons remain in the
table; this correction cannot add detection credit.
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

The frozen baseline passes 4 axis-contract tests, 2,522 type-checker tests across
166 targets and 1,670 IR tests across 87 targets. Eight pre-existing ignored tests
are outside this execution: the type-checker's exhaustive half-activation manual
gate and census regeneration helper, and six IR tests retaining superseded
BLAS-rejection expectations. Their ignored names and reasons remain in the logs;
this campaign does not claim those manual or historical checks ran.

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
These are observed completion limits under the stated budget on a shared host,
not hardware-independent limits or a guarantee that timings increase monotonically.

The survivor and inverse harnesses use a symbolic output index after checking
admission and length. Quantifying that index covers every position without an
additional output loop. A mutant returning an incorrectly empty survivor list
fails the length assertion before any assumption can discard its path. Admission
and normalization calibration receipts were retained from the preceding freeze:
their selected harness text, shared model, kernel, compiler inputs, tool versions,
commands and machine match exactly. `calibration-provenance.json` records these
checks and the original manifest; the changed survivor and inverse harnesses were
calibrated afresh.

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
The pinned Vermilion [VC policy](https://github.com/ilyasergey/vermilion/blob/696756d6b9bbaedd61d6cd743ec3165f2b639224/docs/vcgen.md)
describes a whole-function soundness target with termination side conditions.
The actual baseline emits five decreases checks as ordinary assertion VCs, all
kernel-checked in Lean. A focused zero-increment probe fails one of those VCs;
the other 74 close. This distinguishes checking the supplied progress assertions
from proving the Rust-to-VC translation sound. Liveness mutants without a finite
independent violation witness remain outside functional detection credit.

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

## Results

The complete campaign has 48 mutants: 47 compile, one is a Rust syntax error,
two are equivalent, 40 have independently reproduced finite contract violations,
and five are nontermination defects established by the fixed counters in source.
Those five have no finite violation witness and receive no functional detection
credit. Every unwitnessed survivor has an explicit
[manual classification](axis_oracle_comparison_results/manual-classifications.json).

| Oracle on the 40 witnessed defects | Caught | Conclusive pass | Inconclusive |
| --- | ---: | ---: | ---: |
| Tests | 36 (90%) | 4 | 0 |
| Kani at common rank 5 | 36 (90%) | 1 | 3 timeouts |
| Verus | 40 (100%) | 0 | 0 |
| Tests or Kani | 39 (97.5%) | 1 | 0 |

These are catch fractions within this operator inventory, not estimates of
general compiler-defect detection. Kani's three timeouts are not conclusive
misses. All three survivor defects involved in those timeouts are caught by
tests. Kani adds three catches over tests: incorrect wrong-arity admission,
skipping the first admission check, and accepting an axis equal to the rank.
Verus adds those three plus the high-rank normalization defect. Caller
prevalidation can mask leaf defects, so a kernel witness alone does not establish
a reachable compiler bug.

The only witnessed defect that passes both tests and common-bound Kani is
`b44bf8a3f4db`, changing the normalization rank guard from `>` to `>=`.
At rank `i64::MAX` and signed axis `-2`, the mutant returns `None` instead of
`Some(9223372036854775805)`. This rank exceeds the production ABI ceiling. A
supplementary run of the same Kani harness at full width catches the defect in
0.65 seconds. Thus the campaign demonstrates one common-bound Verus-only catch,
but **no production-domain functional defect requiring unbounded proof**.
That conclusion is restricted to these mutants; it does not make rank-5 checking
complete for production inputs.

The equivalent inverse mutants change the vector filler from zero to one,
or skip the first assignment, which would only write zero into an already-zero
entry. The filler mutant passes Verus and Kani. The skipped assignment
fails Verus's existing entry invariants despite equivalent behavior: this is the
confirmed proof-brittleness case. All five liveness mutants also fail proof,
but their unwitnessed rejections remain separate from that equivalence result.
Tests time out or reach their resource cap; Kani reports four unwind failures
and one timeout. The manually established liveness defects and rejected progress
obligations are useful qualitative evidence, outside the finite functional score.

Vermilion checks the baseline and 10 selected mutants. The baseline closes all
75 obligations. All selected mutants are rejected: four have independent finite
witnesses, while six remain unwitnessed (the five liveness defects and equivalent
skipped assignment). Every case lowers all four functions and emits 75
obligations, with no refused function. The selected comparison supplies no
additional defect detection over Verus. Replayed tactics are checked against
fresh statements; their failure can include proof-maintenance errors as well as
unsatisfied obligations. The independent witness requirement still applies.

The two primary tables are the
[per-mutant matrix](axis_oracle_comparison_results/mutants.csv) and
[cost table](axis_oracle_comparison_results/costs.csv). The
[bound trials](axis_oracle_comparison_results/bounds.csv),
[summary](axis_oracle_comparison_results/summary.json), and
[portable original receipts](axis_oracle_comparison_results/README.md) provide
the underlying evidence. Raw receipt labels are preserved: reporting corrects
two Verus bounds-precondition labels and four Kani unwind labels, with the
adjudication recorded in the matrix.

## Implication for deeper work

Verus buys inexpensive inductive assurance across the modeled machine domain;
the mutation inventory provides little marginal functional detection over tests
plus Kani, and its sole conclusive marginal catch lies outside production ranks.
It also finishes three allocating cases that exceed Kani's budget. Vermilion
provides Lean-checked terms for the translated obligations at substantially
greater setup and maintenance cost, without an additional catch here. Both
still trust their source-to-VC path and require correct caller contracts.

Before adopting this slice, the next investigation should prove a linear
permutation algorithm and measure actual compiler workloads. Broader mutation
classes and caller-boundary contracts are needed before generalizing these
catch fractions. For Kani, per-function bounds and a cheaper allocation model
deserve comparison with the deliberately uniform common bound used here.

## Bound calibration

The common campaign bound is **5**, with unwind 8 for the slice harnesses and 32
for normalization. It covers neither the compiler ABI rank ceiling of
2,147,483,647 nor Metal movement's target-specific ceiling of 8.

| Harness | Largest completed rank | First higher tested timeout | Interpretation |
| --- | ---: | ---: | --- |
| Admission | 13 | 14 | Observed five-minute limit |
| Normalization | 18,446,744,073,709,551,615 | None | Entire 64-bit rank domain completed; no loop or allocation |
| Survivors | 5 | 6 | Observed five-minute limit; allocation dominates |
| Inverse | 12 | 13 | Observed five-minute limit |

The normalization result matters when interpreting marginal catches: a guard
mutation at `i64::MAX` can survive the common rank-5 campaign yet be detected by
the same Kani harness at its independently measured full-width bound. That
kernel input is also beyond Chelis's ABI rank domain. The common-bound comparison
does not establish that such a defect intrinsically requires unbounded proof.

## Measured costs

Linux x86-64, Ryzen AI MAX+ 395, 32 logical CPUs and 128 GiB RAM; this is a shared
workstation. Cold/warm definitions and the RSS sampling limits above apply.
Installation figures include each proof tool's required additional Rust/Lean
toolchain, and exclude shared download/Cargo caches. They are active footprints,
not minimal install claims.

| Oracle | Cold seconds | Warm seconds | Cold / warm peak RSS, GiB | Additional installation |
| --- | ---: | ---: | ---: | --- |
| Integration tests | 226.13 | 131.32 | 6.50 / 0.31 | 6.15 MiB of new registry sources; ordinary Rust/Python excluded |
| Kani, four harnesses at rank 5 | 265.58 | 266.65 | 20.08 / 20.11 | 1.76 GiB, including CBMC and pinned nightly |
| Verus, four contracts | 0.79 | 0.67 | 0.32 / 0.29 | 2.78 GiB, including required Rust 1.98.1 |
| Vermilion, complete Lean twin | 50.68 | 27.94 | 3.34 / 2.72 | 14.43 GiB, including fork, Mathlib, Lean and required Rust 1.96 |

Both false-postcondition controls failed for their expected contract reason.
The test timing covers much more behavior than the proof timings. Kani repeats
solver work on a warm run: the Cargo cache does not cache verification results.
Direct Verus timing excludes tool installation and ordinary Cargo compilation.
Vermilion's cold case starts with a fresh generated proof twin, with upstream
tools and Mathlib already built.

Complete cold installation time was not captured for Kani or Verus. Vermilion's
observed active setup was 482.64 seconds including a cancellation and one
target-path failure/retry, excluding cloning and shared Cargo caches. Its rounded
10.4 GiB setup memory peak is a systemd cgroup figure including file cache, so it
is not directly comparable to the process-group RSS columns. There is no separate
test-oracle installation measurement. These missing and differently scoped setup
measurements remain explicit in `costs.csv` rather than being filled with zero.

The mutant loop records 8,609.69 seconds for tests, 2,537.53 for reachable Kani
harnesses, 23.88 for Verus, and 268.55 for the 10 selected Vermilion mutants.
These exclude calibration, baseline costs, installation, witness probes and
ordinary viability builds. After restricting Kani to reachable harnesses, full
integration testing dominates campaign wall time. The largest sampled Kani
mutant peak is 41.28 GiB, exceeding the baseline's approximately 20 GiB.

The ordinary release build of the axis-core crate in an empty Cargo target took
6.67 seconds. The stripped isolated benchmark binaries were 384,232 bytes for
plain Rust and 384,424 bytes for the verified implementation; each final `rustc`
compile took about 0.20 seconds. The benchmark retains the plain algorithm for
parity checks in both binaries, so the verified binary also links a separate
implementation. That 192-byte difference is not a measurement of ghost-code size
or of the complete compiler's artifact delta. The fresh Vermilion baseline case
used 416 KiB for its generated files and proof twin; shared Lean/Mathlib artifacts
are included in its installation footprint instead.

| Accepted permutation rank | Erased verified kernel, ns | Plain same algorithm, ns | Former linear seen-vector algorithm, ns |
| --- | ---: | ---: | ---: |
| 4 | 2.52 | 2.44 | 6.77 |
| 16 | 22.72 | 25.13 | 10.15 |
| 64 | 403.86 | 381.64 | 20.73 |
| 256 | 7,849.58 | 7,767.69 | 77.66 |
| 1,024 | 115,111.80 | 113,501.87 | 249.47 |

These are local medians from five batches, with exhaustive parity checks through
rank 5 before timing. They demonstrate the algorithmic tradeoff: the pairwise
predicate avoids allocation at small ranks but scales quadratically. The checker
constructor still performs its linear diagnostic pass before the predicate, so
the small-rank kernel timing does not establish a faster checker. Proof erasure
is consistent with similar timings for identical executable algorithms; this
spike does not measure whole-compiler throughput or binary size.
