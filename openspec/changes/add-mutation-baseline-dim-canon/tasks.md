# Tasks: add-mutation-baseline-dim-canon

**Authoritative completion oracle:** `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`, the step already run by the `lint-and-unit` job. Group 6 is not started until that oracle is green on groups 1 through 5. The mutation workflow stays out of `.github/workflows/` until group 5 completes, so no policy is active before its oracle passes.

## 1. Fixtures and failing tests

- [ ] 1.1 Mutant-listing fixtures: `_listing()` builds a `cargo mutants --list --json` payload in a temporary directory, with mutants attributed to each of the eight canonicalizer function names, plus variants that omit one function and that are empty. Built programmatically rather than checked in, matching `scripts/test_coverage.py`. No test in this group invokes cargo.
- [ ] 1.2 Outcome fixtures: `_outcomes()` builds payloads containing caught, missed, unviable, and timed-out mutants in known proportions.
- [ ] 1.3 Coverage-export fixtures reusing the export shape that `scripts/coverage.py` already emits, with lines inside the target span marked both reached and unreached.
- [ ] 1.4 `scripts/test_mutants.py` — assert an empty enumeration exits nonzero and names both filters, and that no catch rate is printed (spec: empty enumeration fails).
- [ ] 1.5 Assert a listing missing one named function exits nonzero and names that function (spec: a renamed function drops out of scope and fails). This is the check that survives future refactors; a bare non-empty test does not.
- [ ] 1.6 Assert the exit-code mapping: missed mutants are not a wrapper failure; a red baseline is a distinct loud failure; usage and internal errors are loud (spec: tool exit codes are mapped explicitly).
- [ ] 1.7 Assert the composed invocation contains no baseline-skipping option (spec: the baseline is always run).
- [ ] 1.8 Assert timeouts are counted in their own class and excluded from caught, and that the catch rate does not credit them (spec: timeouts are a distinct outcome).
- [ ] 1.9 Assert survivor classification against the coverage export: a survivor on a reached line is reported as missed, one on an unreached line as unreached, and an absent export exits nonzero (spec: survivors are classified against region coverage).
- [ ] 1.10 Assert the wrapper never opens `scripts/coverage_baseline_chelis_ir.json` (spec: the committed coverage baseline is not consulted). Assert by path, so the check survives a refactor of the export reader.
- [ ] 1.11 Assert a request to mutate a whole crate or the workspace exits nonzero (spec: crate-wide mutation is refused).
- [ ] 1.12 Assert a lower catch rate against the committed baseline exits zero (spec: a lower catch rate does not fail the run).
- [ ] 1.13 Confirm every task in this group fails red before group 2 begins.

## 2. Wrapper implementation

- [ ] 2.1 `scripts/mutants.py` — argument parsing: required target file, the source-controlled function list, optional `--coverage-export`, `--baseline`, `--json-out`, `--update-baseline`, `--timeout`. Stdlib only, Python 3.11+.
- [ ] 2.2 The eight canonicalizer function names as an explicit module-level list: `normalized_key`, `usize_gcd`, `flatten_product`, `push_product_atoms`, `assemble_product`, `normalize_dim_product`, `normalize_dim_quotient`, `normalize_quotient_parts`. Regex composed from the list and anchored; free-form patterns not accepted (spec: scope is an explicit function list).
- [ ] 2.3 `cargo-mutants` presence check; absence exits nonzero pointing at the devenv shell.
- [ ] 2.4 Enumeration pass via `cargo mutants --list --json` with the composed filters, followed by both group-1 scope checks before any mutant is run (spec: enumeration needs no build).
- [ ] 2.5 Run invocation: `--test-tool nextest`, test scope covering `chelis-ir`, `chelis-backend-c`, and `chelis-backend-hip` per design D7, no baseline-skipping option, auto timeout with an override available.
- [ ] 2.6 Outcome parser producing per-class counts and a survivor list with file, line, and mutant description.
- [ ] 2.7 Exit-code mapper per D4, with missed mutants passing through as data.
- [ ] 2.8 Coverage cross-reference producing the reached/unreached split; absent export exits nonzero.
- [ ] 2.9 Provenance capture from `rustc -vV`: `rustc_version`, `host_triple`, `recorded_at`, runner, and the crates in the test scope.
- [ ] 2.10 Survivor report sorted with reached-region survivors first, since those are the actionable class.
- [ ] 2.11 Run group 1 green, and the full `scripts/` discovery green, exit 0.

## 3. First run and the triaged baseline

- [ ] 3.1 Add `cargo-mutants` to `devenv.nix` packages with a `cargo mutants --version` check in `enterTest`.
- [ ] 3.2 Confirm `devenv shell` resolves the package and `enterTest` is green.
- [ ] 3.3 Enumeration-only run against the real tree. Record the mutant count per function. Confirm all eight functions appear. This step costs no build and is the first real evidence that D2's scoping works.
- [ ] 3.4 Produce a region-coverage export for `crates/chelis-ir/src/dag.rs` using the existing `scripts/coverage.py`, and record the coverage of the canonicalizer span specifically. The crate-file figure of 84.9% from the coverage change's validation record is not a figure for this 230-line span, and the difference determines how much of the survivor list is actionable.
- [ ] 3.5 First full mutation run. Record wall-clock cost, mutant count, and per-class counts. This is the measurement that decides whether the monthly cadence in group 5 is affordable.
- [ ] 3.6 Triage every survivor as equivalent, unreached, or untested, with a one-line reason each (spec: every survivor is triaged).
- [ ] 3.7 Commit `scripts/mutants_baseline_dim_canon.json` with its provenance block and the triage.
- [ ] 3.8 Record in the validation notes which canonicalizer branches the current tests fail to pin. This is the artifact the change exists to produce.

## 4. Documentation

- [ ] 4.1 `docs/mutation_testing.md` — install, local invocation, scope list maintenance, triage procedure, and baseline regeneration.
- [ ] 4.2 In the same doc, state that a catch rate carries no threshold, and that a high catch rate obtained by asserting implementation details is worse than the gap it hides. This is the third design risk and it must appear in the contributor-facing doc, not only in planning artifacts.
- [ ] 4.3 In the same doc, state the D7 scoping consequence plainly: backend tests are in scope because the canonicalizer's consequence is realized in backend memory planning, so a narrower run would mislabel caught mutants as missed.
- [ ] 4.4 Pointer from the `AGENTS.md` testing section, and a `CHANGELOG.md` entry under contributor tooling.

## 5. Workflow and scope classification

- [ ] 5.1 `.github/workflows/mutants.yml` — `workflow_dispatch` and a monthly `schedule` only; assert by inspection that no `pull_request` trigger is present.
- [ ] 5.2 Reuse `.github/actions/free-disk-space` and the `CARGO_PROFILE_DEV_DEBUG: 0` strip, per the chelis#392 mitigations. Mutation runs rebuild repeatedly and are a disk-pressure risk of the same family.
- [ ] 5.3 Install `cargo-mutants` through `taiki-e/install-action`, matching how `nextest` and `mdbook` are installed.
- [ ] 5.4 Publish the classified survivor summary to the job step summary; commit nothing.
- [ ] 5.5 Add `mutants.yml` to `NON_GATE_WORKFLOWS` in `scripts/test_gate.py` with a rule-id comment, matching the `GATE-SCOPE-SMT` precedent.
- [ ] 5.6 Confirm no mutation command was added to any `scripts/gate.py` stage, so `test_gate_parity` stays green.

## 6. Adversarial validation

- [ ] 6.1 Fresh subagent red team against the flattering-failure path: point the wrapper at a nonexistent file, at a file glob matching nothing, and at a function list with a misspelled name. Confirm each exits nonzero and that none reports a catch rate. This is the failure mode the whole design turns on, because the underlying tool exits zero when it tests nothing.
- [ ] 6.2 Confirm a red baseline cannot be presented as a mutation result: break a `chelis-ir` test deliberately, run, and confirm the wrapper reports a red baseline distinctly and emits no survivor list.
- [ ] 6.3 Confirm a hanging mutant is recorded as a timeout and not as caught, using a real non-terminating mutation of `usize_gcd` if the population contains one, or a synthetic outcome fixture if it does not.
- [ ] 6.4 Confirm the coverage cross-reference cannot silently degrade: remove the export, corrupt it, and hand it an export for a different file. Each must exit nonzero rather than emitting an unclassified list.
- [ ] 6.5 Confirm no path in the wrapper writes to `scripts/mutants_baseline_dim_canon.json` outside the explicit `--update-baseline` command.

## 7. Validation and acceptance

- [ ] 7.1 Authoritative oracle green: `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`.
- [ ] 7.2 `openspec validate --all --strict` passes.
- [ ] 7.3 `python3 scripts/gate.py --local` green.
- [ ] 7.4 Confirm every scenario in `specs/mutation-baseline/spec.md` maps to at least one test from group 1, or to an explicit inspection step in group 5 for the two workflow-shape scenarios.
- [ ] 7.5 Dispatch the workflow once on a branch and confirm it produces a summary, commits nothing, and reports no status context on any open pull request.

### Validation record

_Populated during implementation._
