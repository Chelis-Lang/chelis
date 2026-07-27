# Proposal: add-mutation-baseline-dim-canon

## Why

`add-coverage-baseline-chelis-ir` measured how much of `chelis-ir` the 46 integration binaries reach. It cannot say whether reaching a line proves anything about it. Its own design records the limit: "Region coverage measures reach, not correctness. A test that executes `lower.rs` and asserts nothing scores identically to one that asserts the right IR." Mutation testing is the measurement that closes that gap, and the coverage change names it as the intended follow-on.

The subject is the dimension canonicalizer in `crates/chelis-ir/src/dag.rs:283-508`: `normalized_key` and its seven helpers. It is chosen over the alternatives for reasons the coverage data supports.

- Its output decides buffer-slot reuse. `capacity_fits` (`crates/chelis-backend-c/src/memory.rs:315-324`) compares concrete element counts when both sides are concrete, and otherwise falls back to `slot_key == req_key`. The HIP backend repeats it at `crates/chelis-backend-hip/src/memory.rs:360-362`. A weakened key is a wrong-capacity buffer reuse in generated code.
- It is dense in exactly the boundaries mutation operators target: `concrete == 0` (`dag.rs:340`), `concrete != 1` (`dag.rs:342`), the `atoms.len()` zero/one/many split (`dag.rs:346-350`), the `usize_gcd` loop (`dag.rs:298-306`), the `g > 1` guard (`dag.rs:494`), and the `saw_div` flag (`dag.rs:358`).
- It is small. Roughly 230 lines yields a mutant population that can be triaged by hand in one sitting, which is what makes a first mutation run useful rather than an unread report.
- `dag.rs` sits at 84.9% region coverage in the coverage change's validation record, its fifth-largest gap at 349 uncovered regions of 2310. That is a crate-file figure, not a figure for the canonicalizer span, and the difference matters — see below.

The coverage change's validation record also states the sequencing rule this change must obey. It rejected `verify.rs` as a mutation target on the grounds that "320 unreached regions means a mutation run there would waste effort on code no test executes. Coverage first, mutants second, in that order." The same rule applies here: a surviving mutant inside an unreached region is not a finding about assertion strength, it is the coverage number restated in a more expensive form.

## What Changes

- New `scripts/mutants.py`: a stdlib-first Python 3.11+ wrapper over `cargo mutants`, scoped to one named file and one named line span, driving `cargo-nextest` as the test runner so the measured suite matches the `integration` stage. Python, not shell, per the repository automation policy. It follows the shape of `scripts/coverage.py` deliberately.
- New `scripts/test_mutants.py`: the unit-test oracle. It is discovered by the existing `python -m unittest discover -s scripts -p 'test_*.py'` step in the `lint-and-unit` job, so it needs no new gate wiring.
- New `scripts/mutants_baseline_dim_canon.json`: a committed, hand-curated record of the mutant population and the surviving mutants, with provenance. Regenerated only by explicit maintainer action, never auto-updated on merge, following `scripts/coverage_baseline_chelis_ir.json` and `scripts/test_timing_baseline.json`.
- **Survivors are classified against coverage, not reported raw.** The wrapper SHALL cross-reference each surviving mutant's line against the region-coverage export for the same span, and SHALL report a survivor in an unreached region separately from a survivor in a reached one. Only the second class is evidence about assertion strength.
- New `.github/workflows/mutants.yml`: `workflow_dispatch` plus a monthly schedule. No `pull_request` trigger, not a required check, and added to `NON_GATE_WORKFLOWS` in `scripts/test_gate.py`, because `test_all_workflow_files_are_scope_classified` fails on any unclassified workflow file.
- New `docs/mutation_testing.md`: the runbook, mirroring `docs/coverage_setup.md`.
- `devenv.nix` gains `cargo-mutants` in `packages` with a version check in `enterTest`, matching how `cargo-llvm-cov` and `cargo-nextest` are provided.
- The wrapper SHALL fail loudly and non-zero when `cargo-mutants` is absent, when the target file or span does not exist, or when the run generates zero mutants. A zero-mutant run currently exits zero and reads as a perfect score; that reading must be impossible.

### Non-Goals

- **No mutation score threshold, and no gate.** No minimum caught percentage, no ratchet, no merge blocker. The argument for gating on a mutation score is separate and is not made here.
- **No workspace-wide or crate-wide mutation run.** `chelis-ir` is 36k lines. A crate-wide run multiplies the `chelis-ir` suite runtime by the mutant count and produces a report nobody reads. Scope is one file span.
- **No fixes to surviving mutants.** This change produces the list. Strengthening a test or correcting a defect that a survivor exposes is separate work, routed per finding. A survivor is not automatically a defect: it can be equivalent, unreachable, or genuinely untested, and the three demand different responses.
- **No change to `crates/chelis-ir/src/dag.rs`.** Mutation testing mutates a working copy; the committed source is untouched.
- **No product behavior change.** Compiler, runtime, CLI, backend, package, and generated-code behavior are untouched. This is a contributor-process change.
- **No `spec/**` change.** No normative specification is modified and no existing contract is weakened.
- **No claim on specification or atom coverage.** `spec/design/spec_provenance.md` owns coverage authority in that sense. This change measures whether Rust tests detect Rust source mutations. The terms collide in English only, exactly as recorded in the coverage change.

## Capabilities

### New Capabilities

- `mutation-baseline`: what the mutation wrapper runs, how scope is bounded, the fail-closed behavior required when zero mutants are generated or the toolchain is broken, how survivors are classified against coverage, how timeouts are counted, and what authority the committed baseline carries (advisory, never gating).

### Modified Capabilities

None. No existing requirement changes.

## Impact

- `scripts/mutants.py` — new wrapper; resolves the target span, invokes `cargo mutants`, parses the outcome JSON, cross-references coverage, renders the survivor report.
- `scripts/test_mutants.py` — new unit tests; the authoritative acceptance oracle for this change.
- `scripts/mutants_baseline_dim_canon.json` — new committed baseline.
- `scripts/test_gate.py` — `NON_GATE_WORKFLOWS` gains `mutants.yml`; without this the existing scope-classification test fails.
- `.github/workflows/mutants.yml` — new non-required workflow, dispatch plus monthly schedule.
- `devenv.nix` — `cargo-mutants` added to `packages`, with a version check in `enterTest`.
- `docs/mutation_testing.md`, `AGENTS.md` — runbook and a pointer to it.
- `CHANGELOG.md` — contributor-tooling entry.
- No crate under `crates/` is modified.
- Depends on `add-coverage-baseline-chelis-ir` for the region-coverage export that the survivor classification consumes. That change's groups 1 and 2 are complete; its committed baseline (task 3.4) is not. This change needs the export mechanism, which exists, not the committed baseline, which does not — see design D6.
- Complements `add-dim-canon-property-tests`, which strengthens the assertions over the same span. If both land, the mutation run measures the property suite too, and re-running after it lands is the cheapest way to learn whether the properties pin the implementation.
