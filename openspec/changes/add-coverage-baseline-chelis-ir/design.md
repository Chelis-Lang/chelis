# Design: add-coverage-baseline-chelis-ir

## Context

The repository has no code-coverage tooling. `cargo llvm-cov`, `cargo-mutants`, `tarpaulin`, and `grcov` appear nowhere in any `Cargo.toml`, workflow, or Nix file.

The measurement target is `chelis-ir`: 18 source files, 36k lines of code, and 46 integration test binaries under `crates/chelis-ir/tests/`. Two files dominate. `lower.rs` is 11k lines, the largest single file in the repository. `eval.rs` is 2.6k lines and serves as the bit-parity oracle for the C backend, so its untested regions carry more risk than their size suggests.

Three existing constraints shape the design.

1. Linux CI already runs near the disk limit. The `lint-and-unit`, `integration`, `smt-build`, and `backend-sanitizers` jobs each set `CARGO_PROFILE_DEV_DEBUG: 0` and run the free-disk action to reclaim 25-30 GB. This is the recorded mitigation for chelis#392, a mid-link `ld` bus error caused by exhausting the runner disk. Source-based instrumentation adds significant object size, so a coverage job cannot be added casually to that budget.
2. `scripts/test_gate.py` enforces two rules that this change touches. `test_all_workflow_files_are_scope_classified` fails on any workflow file absent from `NON_GATE_WORKFLOWS`. `test_gate_parity` greps `ci.yml` and fails if a gate job runs a cargo command that `gate.py` did not produce.
3. Repository automation policy requires Python 3.11+, stdlib-first, with tests. Shell scripts are forbidden outside the published bootstrap exception.

`scripts/test_timing_baseline.json` is the closest existing precedent. It is a hand-curated baseline, consumed by `scripts/test_timing_check.py` in an `always()` and `continue-on-error: true` CI step, described in `ci.yml` as informational and explicitly never auto-updated on merge. This change copies that shape deliberately.

## Goals / Non-Goals

**Goals:**

- Produce a reproducible per-file region-coverage number for `crates/chelis-ir/src/`.
- Record that number as a committed artifact so a later change can be compared against it.
- Fail loudly when the toolchain or the inputs are broken, so that a broken run is never mistaken for a coverage collapse.
- Keep the whole surface off the per-PR critical path.

**Non-Goals:**

- No threshold, ratchet, or merge gate. See the proposal's Non-Goals.
- No workspace-wide instrumentation.
- No change to any crate under `crates/`.
- No claim on the coverage authority that `spec/design/spec_provenance.md` owns. That document governs specification and atom coverage. This change measures LLVM source regions in Rust code. The two terms collide in English only.

## Decisions

### D1: `cargo llvm-cov`, not tarpaulin or grcov

`cargo llvm-cov` drives LLVM source-based instrumentation, the same mechanism `rustc -C instrument-coverage` exposes. It reports **region** coverage, which distinguishes individual `match` arms and short-circuit branches.

That distinction is the reason to measure at all. `lower.rs` is an 11k-line dispatch over IR node shapes. Line coverage would score a large `match` as covered once any arm runs. Region coverage does not.

Alternatives rejected: `tarpaulin` is ptrace-based, Linux-only, and less accurate on generic and inlined code, of which a compiler crate has a great deal. `grcov` needs more assembly for the same instrumentation `cargo llvm-cov` already wraps.

**Output format follows from this.** The wrapper reads llvm-cov's own JSON export (`--json --summary-only`), not `--lcov`. LCOV's `DA` records are per-line hit counts and cannot express the arm-level distinction above; region counts appear only in the JSON export's `summary.regions` block. Choosing LCOV would have quietly reduced the whole change to line coverage. `--summary-only` drops the per-line segment arrays, which are large and unused here.

### D2: Drive it through nextest

The command is `cargo llvm-cov nextest -p chelis-ir`, not `cargo llvm-cov test`.

`cargo-nextest` is already installed in the `integration` and `macos-smoke` jobs, and `gate.py`'s `integration` stage is `cargo nextest run --workspace`. Measuring through the same runner means the coverage number describes the suite as CI actually schedules it. nextest runs each test in its own process; `cargo llvm-cov` handles the resulting multiple profraw files through its `LLVM_PROFILE_FILE` pattern.

### D3: Report only `crates/chelis-ir/src/`

`cargo llvm-cov -p chelis-ir` instruments the dependency graph, so raw output includes `chelis-deep`, `chelis-surf`, `chelis-types`, and more. The wrapper filters the parsed export to source paths under `crates/chelis-ir/src/` before summarizing.

Filtering happens in the wrapper, on parsed output, rather than through `--ignore-filename-regex`. The wrapper then owns the filter, the filter is unit-testable against a fixture export without invoking cargo, and D4 becomes enforceable.

### D4: An empty result is a failure, not zero percent

If the filter matches no source file, the wrapper exits non-zero and names the filter and the export path.

A path-filter typo, a crate rename, or a moved source root would otherwise render as 0.0% coverage. A future reader comparing against the baseline would see a total collapse and start debugging `chelis-ir` rather than the tool. This mirrors the fail-closed correction made in `harden-lint-traversal-edges`, where an explicit lint root that produced a silent empty entry set and exited 0 green was reclassified as a loud failure.

The same rule applies to a missing `cargo-llvm-cov` binary, an unknown package name, and a malformed baseline JSON. Every one exits non-zero with the offending input named.

### D5: The baseline records its provenance and warns, but does not fail, on drift

Region counts shift with the rustc version, the host target, and the LLVM version behind them. A baseline that hard-fails on a mismatched host would fail for every developer not on the recording host.

The baseline JSON therefore records `rustc_version`, `host_triple`, and `recorded_at` alongside the per-file counts. When the current environment differs from the recorded one, the wrapper prints a warning and continues, and the comparison output is labeled as cross-environment.

**Evidence limit, stated plainly:** the committed baseline is authoritative only for the environment named in its own provenance block. The recording host is `ubuntu-latest` x86_64, matching the `integration` job. A number produced on macOS arm64 is real but is not comparable to the committed baseline, and the wrapper says so in its output rather than silently reporting a delta.

### D6: Workflow is `workflow_dispatch` plus weekly schedule, never `pull_request`

`coverage.yml` carries no `pull_request` trigger, is not added to branch protection, and therefore cannot block a merge. It reuses the free-disk action and the `CARGO_PROFILE_DEV_DEBUG: 0` strip from the existing Linux jobs, because instrumentation compounds the chelis#392 disk pressure rather than relieving it.

It is added to `NON_GATE_WORKFLOWS` in `scripts/test_gate.py` in the same change. Omitting that step makes `test_all_workflow_files_are_scope_classified` fail, so the scope decision is forced to be explicit rather than implicit.

The precedent is `smt-full-prove.yml`: nightly and manual dispatch, not on PRs, with the fast smoke kept separate and required.

### D8: `llvm-tools` goes in `rust-toolchain.toml`, and that cost is shared

`cargo-llvm-cov` does not carry its own coverage tools. It shells out to `llvm-cov` and `llvm-profdata` from the `llvm-tools` rustup component, and fails with `failed to find llvm-tools-preview` without them.

The component is declared in `rust-toolchain.toml`, not in `devenv.nix`.

The scoped alternative was tried first and does not work: devenv's `languages.rust.components` is ignored when `languages.rust.toolchainFile` is set, which this project uses. Confirmed empirically — the toolchain built with `components` listed still shipped no `llvm-cov`. The remaining devenv-only route is to bypass `languages.rust` and build the toolchain by hand through the `rust-overlay` input. That decouples the devenv toolchain from the one CI installs and creates a second definition to keep in sync, for a tool whose whole purpose is measuring what CI runs.

**The trade-off is real and is accepted deliberately.** `rust-toolchain.toml` is honored by rustup for every invocation in this repository, so every CI job and every contributor now fetches `llvm-tools` (roughly 25 MB), not only the weekly coverage lane. Against the disk budget this repository actually fights — the 25-30 GB reclaimed per Linux job for chelis#392 — 25 MB is noise. Against the principle that the coverage lane should not tax the gate, it is a genuine, if small, exception. One toolchain definition shared by rustup, devenv, and CI is worth it.

### D7: Activation ordering

The order is fixed, because each step depends on the previous one.

1. `scripts/coverage.py` and `scripts/test_coverage.py`, with export fixtures built in temporary directories. The tests run against fixtures, so this step needs no cargo invocation and no `cargo-llvm-cov` install.
2. The recorded baseline. This step requires step 1 and a real instrumented run.
3. `coverage.yml` and the `NON_GATE_WORKFLOWS` entry. This step requires both.

Documentation lands with step 3.

**Rollback:** delete the three new `scripts/` files, the workflow, the `NON_GATE_WORKFLOWS` entry, the `.gitignore` line, and the doc. No crate, no spec, and no gate command is touched, so a revert needs no migration and leaves no residue.

## Risks / Trade-offs

**[The baseline goes stale and nobody notices]** → It is regenerated by hand, exactly like `test_timing_baseline.json`, so staleness is the expected steady state between deliberate refreshes. The `recorded_at` field in the provenance block makes the age visible in any diff or report. The weekly workflow run surfaces the current number independently of the committed one.

**[Instrumented builds break the Linux runner disk budget]** → The workflow is single-crate, not workspace, and reuses both existing mitigations. A failure here is contained: the workflow is non-required, so it cannot block a merge, and it is scheduled weekly rather than per-PR, so a failure is observed at most once a week rather than on every push.

**[A coverage number becomes a target]** → Region coverage measures reach, not correctness. A test that executes `lower.rs` and asserts nothing scores identically to one that asserts the right IR. This is a real and well-known failure mode, and it is the reason the change ships no threshold. The stated purpose of the number is to aim mutation and property testing, which do measure assertion strength. That framing belongs in `docs/coverage_setup.md`, not only in this design.

**[nextest and `cargo test` disagree on what runs]** → nextest does not run doctests. `gate.py`'s `DOCTEST_TYPES` stage runs doctests for `chelis-types` only, so `chelis-ir` has no doctest oracle to miss today. If `chelis-ir` gains doctests later, the coverage number will silently exclude them. The wrapper records the runner it used in the baseline provenance so this limit is legible.

**[The change stalls after the baseline and delivers no follow-through]** → Accepted. The artifact has standalone value: even a one-time reading of which `lower.rs` regions the 46 test binaries never reach is actionable, whether or not mutation testing follows.

## Open Questions

- Should the weekly run publish its summary as a job artifact or as a step summary? Step summary is cheaper and needs no retention policy. Resolved during implementation; it does not affect the spec.
- Whether `chelis-ir` remains the right long-term subject, or whether `verify.rs` deserves its own reading, is deferred to whatever change acts on the first baseline.
