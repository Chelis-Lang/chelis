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
- [x] 3.2a Add `llvm-tools-preview` to `rust-toolchain.toml` components; `cargo-llvm-cov` shells out to `llvm-cov` and `llvm-profdata` and fails without it. Placement is forced: devenv ignores `languages.rust.components` when `toolchainFile` is set. Cost is shared across all CI jobs — see design D8.
- [x] 3.2b Add `/.devenv` and `/.devenv.flake.nix` to `.gitignore`; commit `devenv.nix`, `devenv.yaml`, and `devenv.lock` as the tracked reproducible inputs.
- [x] 3.3 Provisional run on darwin-arm64 to prove the end-to-end path, with wall-clock and disk evidence for D6. The committed baseline still needs an x86_64 Linux run per D5; see the validation record.
- [ ] 3.4 Commit `scripts/coverage_baseline_chelis_ir.json` with its provenance block, written by `--update-baseline` **on an `ubuntu-latest`-equivalent x86_64 host**. Deliberately not written from the darwin run: a baseline whose provenance names the wrong host would make every Linux comparison cross-environment.
- [x] 3.5 Record in the validation notes below the region coverage of `lower.rs` and `eval.rs` specifically, and the files with the most uncovered regions. This is the artifact the change exists to produce.
- [x] 3.6 Confirm `target/llvm-cov/` needs no `.gitignore` entry, since `/target` is already ignored wholesale. Confirmed: `.gitignore:1`.

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

**Groups 1-2 (2026-07-27).** Authoritative oracle green: `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`, 406 tests, exit 0, of which 49 are new in `scripts/test_coverage.py`. `openspec validate --all --strict` passes, 5 items.

**Group 3, provisional reading (2026-07-27).** Not the committed baseline: this host is darwin-arm64, and D5 fixes the recording host as `ubuntu-latest` x86_64 to match the `integration` job. Recorded here as end-to-end evidence that the wrapper works against real `cargo llvm-cov` output.

Environment: rustc 1.97.1 / aarch64-apple-darwin / LLVM 22.1.6 / runner nextest.

Cost, against the D6 disk concern: 2m26s wall, and `target/llvm-cov-target` came to 1.3 GB. Both are far below what the design assumed. `cargo-llvm-cov` keeps its own target directory, so the existing 18 GB `target/` was not invalidated. The chelis#392 disk risk for a single-crate coverage job is smaller than estimated, though the Linux figure still has to be measured before the workflow lands.

**`chelis-ir` region coverage: 75.6% (38251/50589 regions, 17 files).**

| uncovered | regions | pct | file |
| ---: | ---: | ---: | --- |
| 5823 | 11800 | 50.7% | `host.rs` |
| 3928 | 15388 | 74.5% | `lower.rs` |
| 637 | 6540 | 90.3% | `grad.rs` |
| 476 | 4031 | 88.2% | `eval.rs` |
| 349 | 2310 | 84.9% | `dag.rs` |
| 320 | 2448 | 86.9% | `verify.rs` |
| 302 | 2691 | 88.8% | `tier2.rs` |
| 198 | 491 | 59.7% | `host_type_state.rs` |

The remaining nine files are each above 91% with under 90 uncovered regions.

Findings that change where the follow-on work should point:

- **`host.rs` is the gap, not `lower.rs`.** The proposal named `lower.rs` as the crate's main risk on the strength of its size. `host.rs` is smaller but has 48% more uncovered regions and sits at barely half coverage. Nearly 5.8k regions of host lowering never execute under the 46 integration binaries.
- **`host_type_state.rs` is the worst by proportion** at 59.7%. It is small enough (491 regions) that closing it is a bounded piece of work, unlike the two above.
- **`eval.rs` is at 88.2% with 476 uncovered regions.** For a file that serves as the bit-parity oracle for the C backend (chelis#770), those regions are worth reading individually rather than in aggregate.
- **`verify.rs` is at 86.9%.** It was the second cargo-mutants candidate; 320 unreached regions means a mutation run there would waste effort on code no test executes. Coverage first, mutants second, in that order.
