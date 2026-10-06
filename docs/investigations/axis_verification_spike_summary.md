# Axis verification spike: final summary

## Purpose and status

PR [#2851](https://github.com/Chelis-Lang/chelis/pull/2851) is retained as a
reference and is not intended for merge. This document is the entry point to its
implementation, measurements and conclusions. The experiment is complete; its
frozen baseline and measured artifacts remain the evidence for this branch.
[Campaign scope and freeze](axis_oracle_comparison.md).

The spike tested whether Verus and Vermilion could prove the same four executable
axis-core functions used by Chelis, and whether those proofs detect defects that
integration tests and bounded Kani checking miss. Verus supplies automated
inductive verification. Vermilion translates the annotated Rust into obligations
whose proof terms Lean kernel-checks. The mutation campaign compares tests, Kani
and Verus on every viable mutant, then applies Vermilion to selected unexpected
Verus outcomes. [Proof comparison](verus_axis_poc.md#what-the-approaches-buy),
[experiment contract](axis_oracle_comparison.md#experiment-contract).

## What was built

The branch contains `chelis-axis-core`, with contracts for complete permutation
admission, signed-axis normalization, ordered reduction survivors and inverse
permutations. A checked `Permutation` value connects the executable kernel to the
type checker, IR lowering and validation, and automatic differentiation. Ordinary
Cargo builds compile the executable bodies with proof specifications erased.
[Kernel contracts and compiler connection](verus_axis_poc.md#proved-code-and-compiler-connection).

The opt-in Verus runner pins its verifier and checks the production source. The
Vermilion runner stages the same source, generates the complete Lean twin and
installs statement-hash-guarded manual proofs. Both require rejection of a planted
false postcondition for the expected proof reason. Translation refusals, parser
failures and toolchain failures cannot satisfy that control.
[Verus runner](verus_axis_poc.md#reproduction-and-baggage),
[Vermilion runner](verus_axis_poc.md#vermilion-comparison).

Kani has a harness for each function, restating the erased Verus contracts as
ordinary Rust assertions. The comparison runner freezes compiler and tool inputs,
calibrates bounds, records costs, generates executable mutations and resumes
keyed oracle runs. It runs the full axis, type-checker and IR test suites, and
restricts Kani to harnesses whose call graph reaches the mutated function. An
independent executable contract model supplies concrete witnesses. Proof rejection
earns detection credit only with such a witness; equivalence, proof brittleness,
timeouts, unwind failures and tool errors remain distinct outcomes.
[Harnesses and verdict rules](axis_oracle_comparison.md#experiment-contract).

## Findings

### Detection

Tests caught 36 of the 40 defects with independently reproduced finite contract
violations. Kani added three admission defects: accepting the wrong arity, skipping
the first admission check, and accepting an axis equal to the rank. Tests and Kani
together caught 39; Verus caught all 40. The test column measures integration
detection through callers, while Kani and Verus check the kernel contracts
directly. [Detection results](axis_oracle_comparison.md#results),
[per-mutant matrix](axis_oracle_comparison_results/mutants.csv).

Verus's sole additional catch was a normalization guard defect at a rank beyond
the production ABI domain. The same Kani harness catches it when run at full
width. Vermilion added no catch on its selected mutants. Unbounded proof therefore
bought no additional production-domain functional detection in this campaign;
its demonstrated advantage was inexpensive completeness over the modeled kernel
domain. [Marginal catch and supplementary check](axis_oracle_comparison.md#results),
[campaign summary](axis_oracle_comparison_results/summary.json).

### Verification cost

Once installed, Verus was the cheapest oracle to run: checking all four contracts
took less than a second warm. It also completed the allocating survivor cases
where Kani exhausted its budget. Kani's four-harness baseline used about 20 GiB
of sampled process-group RSS, and warming Cargo did not remove the solver work.
Allocation and loop structure dominate its practical reach.
[Measured costs](axis_oracle_comparison.md#measured-costs),
[cost table](axis_oracle_comparison_results/costs.csv).

Vermilion's warm complete-twin check took about 28 seconds, with roughly 14 GiB
of active tooling. Setup required a pinned frontend fork, Lean and Mathlib, while
regeneration required maintaining manual proofs. For this kernel, that combination
places it in a secondary audit workflow rather than the rapid edit-and-verify loop
that Verus supports. Cold checks start with installed tools; complete cold
installation times were not captured for Verus or Kani.
[Cost definitions and measurements](axis_oracle_comparison.md#measured-costs),
[proof engineering](verus_axis_poc.md#what-the-approaches-buy).

### Coverage and bounds

The common campaign rank bound of five does not cover the production ABI rank
domain. Individual harnesses reach very different bounds: normalization completes
the entire machine-width rank domain without loops or allocation, while survivors
reach only the small common bound. Admission and inverse lie between those
extremes. Bounded campaign coverage is incomplete for the slice functions even
though normalization can be checked completely at its own bound.
[Bound calibration](axis_oracle_comparison.md#bound-calibration),
[recorded trials](axis_oracle_comparison_results/bounds.csv),
[production domain](axis_oracle_comparison.md#production-domain-and-costs).

### Proof brittleness

One equivalent inverse mutant skips an assignment that would write zero into an
already-zero entry. Its output remains correct, but Verus rejects it because the
existing loop-entry invariants no longer fit. Tests and Kani pass it. This is
proof brittleness, not defect detection, and demonstrates why every credited
proof catch needs independent behavioural evidence. Nontermination mutants
without finite witnesses also remain outside the functional catch count.
[Equivalence and liveness analysis](axis_oracle_comparison.md#results),
[manual classifications](axis_oracle_comparison_results/manual-classifications.json).

### Executable algorithm cost

The verified permutation predicate compares positions pairwise and is quadratic;
the previous seen-position-vector predicate was linear. Proof specifications are
erased, and the verified executable and plain Rust version of the same algorithm
have similar measured timings. The large-rank regression comes from the algorithm
chosen for this proof, rather than executing proof annotations. The isolated
measurements do not establish whole-compiler throughput or artifact-size changes.
[Runtime and artifact measurements](axis_oracle_comparison.md#measured-costs),
[algorithm tradeoff](verus_axis_poc.md#what-the-approaches-buy).

### Tooling friction

`cargo-mutants` cannot see the functions inside `verus!`, so the campaign needed a
deterministic text mutator that leaves contracts and annotations intact. Both
verifier-output parsers also required post-hoc adjudication: the frozen Verus
parser missed a bounds-precondition diagnostic, while the Kani parser expected
failure fields in a different order. The reporter preserves original labels and
logs alongside corrections. Verus corrections still require witnesses; Kani
corrections remove false catch credit for unwind failures.
[Mutation and parser handling](axis_oracle_comparison.md),
[recorded labels and adjudications](axis_oracle_comparison_results/mutants.csv).

## Interpretation

The strongest result is the value of precise contracts. Directly checking
admission and position mappings produced the marginal gain over integration
tests. That gain was shared by bounded checking and inductive proof; the campaign
does not select one tool as the unique source of useful assurance.
[Detection comparison](axis_oracle_comparison.md#results).

Verus suits a fast verification hot path for kernels with exact integer and
sequence structure once their inductive invariants exist. Its recurring cost is
authoring those invariants and repairing them after executable changes. Kani
complements it without requiring user-authored loop invariants and supplies
concrete counterexamples. Kani still has explicit harness, rank and unwind
assumptions. For an agent, an executable counterexample is stronger feedback than
an unexplained proof failure, which can mean either incorrect code or an inadequate
proof structure. [Harness design](axis_oracle_comparison.md#experiment-contract),
[proof fit and maintenance](verus_axis_poc.md#what-the-approaches-buy),
[brittleness evidence](axis_oracle_comparison_results/manual-classifications.json).

Kernel-checked Lean terms did not earn their setup and maintenance weight for
this class of code in this experiment. Vermilion supplies a different assurance
boundary, but no extra detection here, and it shares the trusted Verus frontend
before its own lowering to Lean. Its stronger attraction is composition with
existing Lean mathematics when that is itself a requirement.
[Assurance boundary](verus_axis_poc.md#what-the-approaches-buy),
[selected-mutant results](axis_oracle_comparison.md#results).

The clearest ecosystem gap exposed by this spike is the verdict layer. Obtaining
useful agent feedback required orchestration across several oracles, independent
witnesses, explicit equivalence and brittleness classification, separation of
timeouts from misses, and output adjudication. That is an inference from the
runner this experiment required, rather than a survey claim that every other
tool lacks those features. [Experiment and verdict contract](axis_oracle_comparison.md#experiment-contract),
[parser adjudications](axis_oracle_comparison.md).

## Implications for future work

The next useful technologies are contract authoring and mutation-based measurement
of contract strength, invariant synthesis, proof repair after code changes,
conversion of verifier failures into runnable counterexamples, and a multi-oracle
verdict layer that agents can drive. These are research directions suggested by
the contrast between witnessed detection and annotation maintenance in this
campaign. [Deeper-work implications](axis_oracle_comparison.md#implication-for-deeper-work),
[proof engineering and scope](verus_axis_poc.md#what-the-approaches-buy).

For Chelis, promising Verus candidates are discrete compiler kernels such as shape
transforms, extent arithmetic and index plans. The expectation should be cheap
completeness for explicit contracts rather than additional catches over strong
tests and appropriately bounded checking. Efficient algorithms should be selected
and verified deliberately, beginning with a linear permutation algorithm. Contracts
at caller and representation boundaries are the next frontier: proving a leaf
does not establish that its callers construct the intended inputs. These are
proposed applications of the observed kernel pattern. Floating-point correctness
was untouched by this spike. [Next investigation](axis_oracle_comparison.md#implication-for-deeper-work),
[generalizability and caller boundary](verus_axis_poc.md#what-the-approaches-buy).

## Limits

The experiment covers one small axis kernel and a token-level operator inventory.
Its uniform common Kani bound makes the comparison consistent but hides the
greater reach of individual harnesses. The proof covers executable kernel
contracts; caller extraction of axes and mapping back onto compiler
representations lie outside it. Test outcomes include integration behaviour, so
they answer a different question from direct contract checks. Vermilion's
selected subset establishes no full-inventory catch rate.
[Experiment scope](axis_oracle_comparison.md#experiment-contract),
[proof boundary](verus_axis_poc.md#proved-code-and-compiler-connection),
[selection and results](axis_oracle_comparison.md#results).

## Evidence index and reproduction

[Axis oracle comparison](axis_oracle_comparison.md) owns the frozen experiment,
scoring rules, production domain, results and measured costs.

[Verus axis proof investigation](verus_axis_poc.md) owns the kernel contracts,
compiler connections, Verus/Vermilion implementation and assurance boundaries.

[Mutant matrix](axis_oracle_comparison_results/mutants.csv) records every mutation,
oracle verdict, witness, timing and adjudication.

[Cost table](axis_oracle_comparison_results/costs.csv) records setup footprints,
cold and warm checks, memory, rank coverage and explicitly missing measurements.

[Bound trials](axis_oracle_comparison_results/bounds.csv) records per-harness
calibration, loop unwinds, execution times and memory.

[Campaign summary](axis_oracle_comparison_results/summary.json) records aggregate
outcomes, common and individual bounds, and the selective Vermilion results.

[Manual classifications](axis_oracle_comparison_results/manual-classifications.json)
records the equivalence arguments and unwitnessed nontermination defects.

[Portable evidence instructions](axis_oracle_comparison_results/README.md) describe
the original receipt archive and regeneration of the tables without proof tools.

For a new measurement campaign, follow the
[reproduction section](axis_oracle_comparison.md#reproduction) at its frozen
baseline. For reading or auditing this reference, the committed results and
portable receipts are sufficient; this summary adds no new measurements.
