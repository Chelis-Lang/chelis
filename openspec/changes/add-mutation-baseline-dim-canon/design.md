# Design: add-mutation-baseline-dim-canon

## Context

`cargo-mutants` copies the source tree to a scratch directory, applies one mutation, and runs the test command. It reports four outcomes: **caught** (a test failed, which is the desired result), **missed** (no test failed), **unviable** (the mutant did not compile), and **timeout** (the mutant made the suite run too long). It runs an unmutated baseline first, and auto-sets the per-mutant timeout from the baseline's measured test duration.

Its exit codes are the single most important fact for this design: **0 means every tested mutant was caught**, 1 is a usage error, 2 is missed mutants, 3 is timeouts, 4 is a failing baseline, 5 and 6 relate to `--in-diff`, and 70 is an internal error.

That definition of 0 is a trap. If a file glob or a name regex matches nothing, zero mutants are tested, every tested mutant was therefore caught, and the tool exits 0. A wrapper that treats "exit 0" as success would report a broken filter as a flawless mutation score. This is the same failure shape that `add-coverage-baseline-chelis-ir` D4 corrected when an empty coverage measurement was reclassified from "0.0%" to a loud failure, and that `harden-lint-traversal-edges` corrected for an empty lint entry set. It recurs here in a more dangerous direction, because the misleading reading is flattering rather than alarming.

The subject is `crates/chelis-ir/src/dag.rs:283-508`: `normalized_key` plus `usize_gcd`, `flatten_product`, `push_product_atoms`, `assemble_product`, `normalize_dim_product`, `normalize_dim_quotient`, and `normalize_quotient_parts`.

The existing evidence over that span is example-based: `crates/chelis-ir/tests/dim_canonicalization.rs`, `crates/chelis-ir/tests/dim_canon_adversarial.rs`, and the unit module at `dag.rs:2374-2467`. The coverage change reports `dag.rs` at 84.9% region coverage with 349 uncovered regions, but that is the whole 3184-line file. The canonicalizer span is 230 lines of it, and its own coverage is not yet known.

## Goals / Non-Goals

**Goals:**

- Produce a triaged list of surviving mutants over one small, high-consequence span.
- Make every degenerate run — broken filter, renamed function, failing baseline, hanging mutant — a loud failure rather than a clean score.
- Separate survivors that indicate weak assertions from survivors that merely restate a coverage gap.
- Keep the whole surface off the per-PR critical path.

**Non-Goals:**

- No threshold, ratchet, or gate. See the proposal's Non-Goals.
- No crate-wide or workspace-wide run.
- No fixes to survivors within this change.
- No change to any file under `crates/`.

## Decisions

### D1: `cargo-mutants`, on the pinned stable toolchain

`cargo-mutants` works on stable Rust and needs no compiler plugin: it edits a copy of the source and rebuilds. That matters here because `rust-toolchain.toml` pins the toolchain for every invocation in this repository, and a tool requiring nightly would force either a second toolchain or a pin change. The alternative, `mutagen`, is a procedural-macro approach requiring nightly and is not maintained.

It is provided through `devenv.nix` `packages` with a version check in `enterTest`, exactly as `cargo-llvm-cov` and `cargo-nextest` are. It is a CLI tool, not a library dependency, so this is the correct route.

### D2: Scope by file glob plus a name regex over an explicit function list

`cargo-mutants` filters files with `-f` and mutant names with `--re`. The full mutant name includes the function name, the file name, and the description of the change, so an anchored name regex is how a 230-line span inside a 3184-line file is isolated.

The wrapper owns an explicit list of the eight function names. It composes the regex from that list rather than accepting a free-form pattern, so the scope is reviewable as a list of identifiers instead of as a regex.

The alternative — mutate all of `dag.rs` and filter the results afterwards — was rejected. Filtering after the run pays the full build-and-test cost for every mutant in the other 2950 lines, and mutation cost is dominated by the run, not the report. Filtering before the run is the only version that finishes.

### D3: An empty or under-populated scope is a loud failure

Two distinct degenerate cases must both fail, and neither does so on its own.

1. **Zero mutants generated.** The wrapper runs `cargo mutants --list --json` first, and fails non-zero if the list is empty, naming the file glob and the composed regex. Without this, exit 0 reads as a perfect score.
2. **A named function contributing zero mutants.** This is the case a simple non-empty check misses. If `assemble_product` is renamed and the wrapper's list is not updated, seven functions still produce mutants, the list is non-empty, and the eighth silently leaves scope forever. The wrapper therefore asserts that **every** name in its list appears in at least one listed mutant, and fails naming the absent ones.

The second check is the one that keeps the scope honest across future refactors, and it is cheap because `--list` runs without building anything.

### D4: Exit codes are mapped explicitly, never collapsed

The wrapper distinguishes all of `cargo-mutants`' exit codes rather than testing for zero.

- **0** — every tested mutant caught. Reported as success **only after** the D3 checks have passed.
- **1** — usage error. Loud failure naming the invocation.
- **2** — missed mutants. **Not a wrapper failure.** This is the expected outcome of a first baseline run and is the data the change exists to collect.
- **3** — timeouts. Reported separately, never as caught. See D5.
- **4** — the unmutated baseline failed its own tests. Loud failure, and distinct from everything else: it means the report describes nothing, because the suite was already red before any mutation.
- **70** — internal error. Loud failure.

**The baseline SHALL NOT be skipped.** `--baseline=skip` exists and makes runs faster, and it makes every result meaningless: without a green unmutated run there is no evidence that a "caught" mutant was caught by the mutation rather than by a pre-existing failure.

### D5: Timeouts are never counted as caught, and `usize_gcd` is the reason

`usize_gcd` (`dag.rs:298-306`) is a `while b != 0` Euclidean loop. A mutation that alters the loop body — replacing `a % b` with `a`, for instance — produces a loop that never terminates. This is not hypothetical for this span; it is the most likely non-terminating mutant in it.

A hung mutant that trips the timeout resembles a caught mutant in one respect: the test command did not succeed. Counting it as caught would be wrong. The test did not detect the mutation; the test never finished. The wrapper reports timeouts as their own class, and the committed baseline records them separately from both caught and missed.

The per-mutant timeout is left to the auto-setting derived from the baseline's measured duration, with an explicit override available. Auto-setting is preferred because a hand-picked constant becomes wrong as the suite grows.

### D6: Survivors are classified against coverage, and the dependency is on the mechanism, not the artifact

The coverage change's validation record sets the rule: a mutation run over unreached code "would waste effort on code no test executes". Its conclusion for `verify.rs` was to defer. The conclusion here is different, because the span is small enough to run either way — but the *report* must not conflate the two.

Each surviving mutant's line is cross-referenced against the region-coverage export for the same span. A survivor in an unreached region is reported as **unreached**, and a survivor in a reached region as **missed**. Only the second is evidence about assertion strength; the first is the coverage number restated at much higher cost.

**The dependency is narrower than it looks.** `add-coverage-baseline-chelis-ir` has completed groups 1 and 2 — `scripts/coverage.py` exists and produces the export — but its task 3.4, the committed baseline recorded on an x86_64 Linux host, is still open. This change consumes the **export mechanism**, which is done, and not the **committed baseline**, which is not. It therefore does not block on the other change finishing, and it must not be written to read the committed baseline file.

If the coverage export is unavailable, the wrapper fails rather than reporting survivors unclassified, because an unclassified survivor list invites exactly the misreading D6 exists to prevent.

### D7: The test scope includes the two backend crates, not just `chelis-ir`

The obvious scoping is to run `chelis-ir`'s tests for each mutant, since that is where the mutated file lives. It is wrong here.

The canonicalizer's consequence is realized in `capacity_fits` (`crates/chelis-backend-c/src/memory.rs:315-324`) and its HIP twin (`crates/chelis-backend-hip/src/memory.rs:360-362`), which decide buffer-slot reuse on key equality. A mutant that weakens the key but is caught only by a backend memory-planning test would be recorded as **missed** under a `chelis-ir`-only scope. That is not a conservative error; it is a false finding that would send a reader to write a redundant test in the wrong crate.

The test command therefore covers `chelis-ir`, `chelis-backend-c`, and `chelis-backend-hip`, run through `cargo-nextest` to match the `integration` stage's runner.

**The cost is accepted and is the dominant runtime term.** Three crates' suites per mutant, times the mutant count. This is the specific reason the span is 230 lines and not a whole file, and the reason the workflow is monthly rather than weekly.

### D8: Activation ordering

1. `scripts/mutants.py` and `scripts/test_mutants.py`, against fixture JSON built in temporary directories. No `cargo mutants` invocation and no `cargo-mutants` install needed for this step, matching how `scripts/test_coverage.py` handles its export fixtures.
2. Install `cargo-mutants` through `devenv.nix`; run `--list` only, and confirm the D3 checks against the real mutant population.
3. The first full run, and the triaged committed baseline.
4. Documentation.
5. `mutants.yml` and the `NON_GATE_WORKFLOWS` entry.

**Rollback:** delete the three new `scripts/` files, the workflow, the `NON_GATE_WORKFLOWS` entry, the `devenv.nix` package, and the doc. No crate, no spec, and no gate command is touched.

## Risks / Trade-offs

**[A broken filter reports a perfect score]** → The primary risk, and the reason D3 has two checks rather than one. The non-empty check catches a wrong glob; the per-function check catches a rename. Both run against `--list` output, so both are cheap and both are unit-testable against fixtures without invoking cargo.

**[Equivalent mutants inflate the survivor list]** → Real and unavoidable. Some mutations produce semantically identical code and can never be caught by any test. `saturating_mul` boundary mutations are likely candidates within this span. The mitigation is procedural: the committed baseline records a triage reason per survivor — equivalent, unreached, or genuinely untested — and an untriaged survivor list is not a finished artifact.

**[Runtime makes the run impractical]** → Bounded by D2's narrow scope, and measured in group 3 before the workflow lands. `cargo-mutants` supports sharding if the measured cost demands it; that is deferred rather than designed in, because a 230-line span probably does not need it and the shard configuration would be dead weight if it does not.

**[A mutation score becomes a target]** → The same risk the coverage change recorded, and the same answer: no threshold ships. A high catch rate over a span with weak assertions is achievable by testing implementation details, which is worse than the gap it hides. The runbook must say so, not only this design.

**[Flaky tests are recorded as caught]** → A test that fails intermittently will fail under some mutant and be credited with catching it. The green baseline required by D4 reduces but does not eliminate this. The committed baseline records the runner and the host so a suspicious result can be re-run.

**[The change produces a list nobody acts on]** → Accepted. The list has standalone value as a record of which parts of the canonicalizer the current tests do not pin, and the proposal deliberately routes fixes elsewhere rather than pretending they fit here.

## Open Questions

- Whether the survivor triage should be re-run after `add-dim-canon-property-tests` lands. It should, and it is the cheapest available measurement of whether those properties pin the implementation — but it is a follow-on run, not a task of either change.
- Whether `verify.rs` becomes the second target once its coverage improves. Deferred to whatever change acts on this baseline, exactly as the coverage change deferred it here.
- Whether the monthly schedule is the right cadence, or whether dispatch-only is enough until the first baseline is triaged. Settled during implementation; it does not affect the requirements.
