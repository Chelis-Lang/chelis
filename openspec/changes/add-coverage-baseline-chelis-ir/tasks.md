# Tasks: add-coverage-baseline-chelis-ir

**Authoritative completion oracle:** `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`, the step already run by the `lint-and-unit` job. Group 6 is not started until that oracle is green on groups 1 through 5. The coverage workflow stays out of `.github/workflows/` until group 5 completes, so no policy is active before its oracle passes.

## 1. Fixtures and failing tests

- [x] 1.1 Export fixtures: `_export()` builds an llvm-cov JSON export in a temporary directory, with records for both `crates/chelis-ir/src/` and dependency paths. Built programmatically rather than checked in, matching `scripts/test_test_timing_check.py`. No test in this group invokes cargo.
- [x] 1.2 Baseline fixtures: `_baseline()` builds one well-formed with a full provenance block, plus variants with the provenance block missing, incomplete, corrupt in its counts, and not valid JSON.
- [x] 1.3 `scripts/test_coverage.py` — assert the parser keeps only paths under the measured package's `src/` and drops dependency records (spec: dependency sources excluded).
- [x] 1.4 Assert an export whose filter matches nothing exits nonzero and names both the filter and the export path, and that no percentage is printed (spec: filter matches nothing).
- [x] 1.5 Assert a matched file with zero covered regions reports 0.0% and exits zero, so 1.4 is distinguished from genuine zero coverage (spec: genuine zero coverage reported normally).
- [x] 1.6 Assert nonzero exit naming the offending input for: absent `cargo-llvm-cov`, unknown package name, absent baseline, malformed baseline, baseline missing its provenance block.
- [x] 1.7 Assert invocation with no package name, and with a workspace-wide request, exits nonzero (spec: workspace-wide measurement refused).
- [x] 1.8 Assert comparison against a baseline whose `host_triple` or `rustc_version` differs prints a warning, labels output cross-environment, and exits zero; and that a matching environment produces no warning.
- [x] 1.9 Assert a coverage decrease against the baseline exits zero (spec: no percentage is a threshold).
- [x] 1.10 Confirm every task in this group fails red before group 2 begins.

## 2. Wrapper implementation

- [x] 2.1 `scripts/coverage.py` — argument parsing: a required package name, optional `--baseline`, `--export`, `--json-out`, `--update-baseline`. Stdlib only, Python 3.11+.
- [x] 2.2 Workspace-membership check by reading the member manifests with `tomllib`, matching `gate.py`'s `workspace_member_packages`; unknown package exits nonzero naming the package. A member without a `src/` directory also fails loudly, so it cannot reach the filter and imitate lost coverage.
- [x] 2.3 `cargo-llvm-cov` presence check; absence exits nonzero pointing at the devenv shell.
- [x] 2.4 Invoke `cargo llvm-cov nextest -p <pkg> --json --summary-only --output-path <target/llvm-cov/...>`. The JSON export, not `--lcov`: region counts exist only in the export, and LCOV would have silently downgraded the change to line coverage.
- [x] 2.5 Export parser producing per-file covered and total region counts.
- [x] 2.6 Source-path filter restricted to the package's `src/`; empty result exits nonzero per D4.
- [x] 2.7 Provenance capture from `rustc -vV`: `rustc_version`, `host_triple`, `llvm_version`, `recorded_at`, and the runner used.
- [x] 2.8 Baseline comparison and the cross-environment warning path; comparison never changes the exit code.
- [x] 2.9 Per-file summary renderer, sorted by uncovered region count descending, so the largest gaps read first.
- [x] 2.10 Run group 1 green — 49 tests, and the full `scripts/` discovery at 406 tests, exit 0.

## 3. Record the baseline

- [x] 3.1 Add `cargo-llvm-cov` to `devenv.nix` packages, with a `cargo llvm-cov --version` check in `enterTest`.
- [x] 3.2 Confirm `devenv shell` resolves the package — `cargo-llvm-cov 0.8.5` from the Nix store, `enterTest` green.
- [ ] 3.3 Run the wrapper for `chelis-ir` on an `ubuntu-latest`-equivalent x86_64 host, matching the environment D5 names, on a clean worktree in an isolated `CARGO_TARGET_DIR`; record wall-clock time and peak `target/` size as the disk-budget evidence for D6.
- [ ] 3.4 Commit `scripts/coverage_baseline_chelis_ir.json` with its provenance block, written by `--update-baseline`.
- [ ] 3.5 Record in the validation notes below the region coverage of `lower.rs` and `eval.rs` specifically, and the five files with the most uncovered regions. This is the artifact the change exists to produce.
- [ ] 3.6 Confirm `target/llvm-cov/` needs no `.gitignore` entry, since `target/` is already ignored wholesale.

## 4. Documentation

- [ ] 4.1 `docs/coverage_setup.md` — install, local invocation, baseline regeneration procedure, and the D5 evidence limit stated plainly: the baseline is authoritative only for the environment in its provenance block.
- [ ] 4.2 In the same doc, state that region coverage measures reach and not assertion strength, and that the number carries no threshold. This is the mitigation recorded in the design's third risk; it must appear in the contributor-facing doc, not only in planning artifacts.
- [ ] 4.3 Add a pointer to the runbook in the `AGENTS.md` testing section.
- [ ] 4.4 `CHANGELOG.md` entry under contributor tooling.

## 5. Workflow and scope classification

- [ ] 5.1 `.github/workflows/coverage.yml` — `workflow_dispatch` and a weekly `schedule` only; assert by inspection that no `pull_request` trigger is present.
- [ ] 5.2 Reuse `.github/actions/free-disk-space` and set `CARGO_PROFILE_DEV_DEBUG: 0` and `CARGO_PROFILE_TEST_DEBUG: 0`, per the chelis#392 mitigations required by the spec.
- [ ] 5.3 Install `cargo-llvm-cov` through `taiki-e/install-action`, matching how the repository already installs `nextest` and `mdbook`.
- [ ] 5.4 Publish the per-file summary to the job step summary; commit nothing.
- [ ] 5.5 Add `coverage.yml` to `NON_GATE_WORKFLOWS` in `scripts/test_gate.py` with a rule-id comment explaining the scope decision, matching the `GATE-SCOPE-SMT` precedent.
- [ ] 5.6 Confirm no coverage command was added to any `scripts/gate.py` stage, so `test_gate_parity` stays green.

## 6. Validation

- [ ] 6.1 Authoritative oracle green: `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`.
- [ ] 6.2 `python3 scripts/gate.py --local` green.
- [ ] 6.3 Confirm each scenario in `specs/coverage-baseline/spec.md` maps to at least one test from group 1, or to an explicit inspection step in group 5 for the two workflow-shape scenarios.
- [ ] 6.4 Dispatch the workflow once on a branch and confirm it produces a summary, commits nothing, and reports no status context on any open pull request.

## 7. Adversarial validation

- [ ] 7.1 Fresh subagent red team against the fail-closed paths: rename the source directory and confirm 2.6 fails loudly rather than reporting 0.0%; truncate the LCOV mid-record; hand a baseline with a well-formed provenance block but a corrupt counts map; point the wrapper at a package that exists in `cargo metadata` but has no `src/` directory.
- [ ] 7.2 Confirm the wrapper cannot be coaxed into a workspace-wide run through package-name spelling, a repeated `-p` flag, or an empty string argument.
- [ ] 7.3 Confirm no path in the wrapper writes to `scripts/coverage_baseline_chelis_ir.json`; the baseline is regenerated only by an explicit maintainer command.

### Validation record

<!-- Filled in during implementation: oracle results, the 3.4 coverage findings, disk and timing evidence from 3.2, and red-team outcomes. -->
