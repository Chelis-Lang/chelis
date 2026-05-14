# CI Integration Test Runtime Diagnosis

Investigation of the "Integration Tests (Linux)" CI job bloat (~15 minutes wall,
pre-0.7.9 release). This note records the profiling data and the minimal
pre-release intervention. Deep cleanup (test consolidation, pareto-optimal-subset
trimming, reorganization) is deferred to a post-release workstream.

## Symptom

CI run 25854185613, job 75967825032 (`cargo test --workspace --tests`):

- Total job wall: 14m55s
- Compilation: 2m29s (cache hit via `Swatinem/rust-cache@v2`) — not the bottleneck
- Test execution: ~691s (~11.5 min) — the bottleneck

## Root cause

`cargo test --workspace --tests` runs the workspace's integration test binaries
**one binary at a time**. Tests within a binary parallelize across threads, but
the binaries themselves do not overlap. The workspace currently has 200 test
binaries; a handful are heavyweight (each test compiles/checks/builds chelis-std
or scans the whole repo), so the sequential per-binary schedule serializes long
tail-latency binaries behind each other.

## Profiling method

Built `cargo build --workspace --all-targets` first (so timing measures
execution, not compilation), then timed two runners on this workstation
(12+ cores):

| Runner | Wall clock |
|---|---|
| `cargo test --workspace --tests` | 241.45s |
| `cargo nextest run --workspace` | 99.71s |

`cargo nextest` schedules tests from **all** binaries in a single global thread
pool, so heavyweight binaries overlap instead of serializing. The sum of
individual test times is ~1046s; nextest collapses that to ~100s wall on a
multi-core box because it saturates all cores across binary boundaries.

## Slowest test binaries (by total wall time, from nextest run)

| Total | Tests | s/test | Binary |
|---|---|---|---|
| 207.74s | 37 | 5.615 | `chelis-cli::wsc_v3_stdlib_finish` |
| 162.35s | 8 | 20.294 | `chelis-cli::red_team_0_7_8` |
| 134.01s | 205 | 0.654 | `chelis-cli::cli` |
| 101.91s | 3 | 33.968 | `chelis-cli::phase3j_pre_std` |
| 73.04s | 8 | 9.130 | `chelis-cli::wsa8_monomorphization_build` |
| 62.17s | 150 | 0.414 | `chelis-backend-c` |
| 42.44s | 20 | 2.122 | `chelis-cli::wsc_stdlib_generalization` |
| 37.96s | 14 | 2.711 | `chelis-cli::rt3_adversarial` |
| 25.08s | 2 | 12.538 | `chelis-cli::phase_a_bundled_loader` |
| 22.66s | 19 | 1.192 | `chelis-backend-c::redteam_exec_compile` |

Note: `phase3j_pre_std`'s 3 visible non-ignored tests are 33-35s each; most of
that binary's body is already `#[ignore]`'d behind a documented manual gate.

## Slowest individual tests

| Time | Binary | Test |
|---|---|---|
| 101.83s | `chelis-cli::red_team_0_7_8` | `lint_subtree_invocation_matches_dot_for_doc_filename_convention` |
| 34.82s | `chelis-cli::phase3j_pre_std` | `cross_function_seed_stdlib_kaiming_uniform_uses_handler_seed` |
| 33.94s | `chelis-cli::cli` | `phase3a_reef_std_acceptance_oracle` |
| 33.82s | `chelis-cli::phase3j_pre_std` | `phase3j_pre_oracle_build_path_repros_uniform_like_seed_distinct_seeds_differ` |
| 33.27s | `chelis-cli::phase3j_pre_std` | `cross_function_seed_stdlib_normal_like_advances_rng_per_random_op` |
| 25.07s | `chelis-cli::phase_a_bundled_loader` | `phaseA_bundled_chelis_std_loader_property_oracle` |
| 23.80s | `chelis-cli::rt3_adversarial` | `production_stdlib_linear_now_builds_clean` |
| 23.32s | `chelis-cli::wsa8_monomorphization_build` | `build_stdlib_attention_succeeds` |

The outlier is `lint_subtree_invocation_matches_dot_for_doc_filename_convention`:
at 101.8s it is 3x the next slowest test. It runs `chelis lint --check` over the
**entire repository twice** (once as `.`, once as an explicit subtree list) and
compares `doc-filename-convention` error counts. It is the regression lock for
the 0.7.8 PR #93 path-canonicalization fix.

## Intervention (minimal, pre-release)

Two levers, both applied:

### Lever A — swap CI integration runners to `cargo nextest`

The `integration` and `macos-smoke` jobs swap `cargo test --workspace --tests`
for `cargo nextest run --workspace`. nextest is installed via
`taiki-e/install-action@nextest`. This ignores zero tests; it is purely a
scheduler change that overlaps heavyweight binaries. Other CI jobs that use
narrower invocations (`cargo test --workspace --lib`, `cargo test -p ...`) are
left on plain `cargo test` since they do not have the many-binary serialization
problem and `cargo test` must keep working for them.

### Lever B — `#[ignore]` the single pathological full-repo-lint test

`lint_subtree_invocation_matches_dot_for_doc_filename_convention` is
`#[ignore]`'d and wired as a documented manual gate in `docs/manual_gates.md`.
Rationale: even under nextest, a single 101s test sets a hard wall-clock floor
because nextest cannot parallelize within one test. It is also far over the
~60s inner-loop budget the agent contract sets for default `cargo test`. It is a
regression lock, so per the contract its coverage is preserved by the documented
manual gate command, not deleted.

Manual gate command:

```
cargo test -p chelis-cli --test red_team_0_7_8 \
  lint_subtree_invocation_matches_dot_for_doc_filename_convention -- --ignored --exact
```

Expected: exits 0; `doc-filename-convention` error counts from `chelis lint
--check .` and the explicit subtree walk are equal (PR #93 closure invariant).

The other heavyweight binaries (`wsc_v3_stdlib_finish`, `phase3j_pre_std` visible
tests, `wsa8_monomorphization_build`, etc.) are **not** ignored: they are heavy
but not pathological, and nextest's global pool parallelizes them across binary
boundaries. Trimming or consolidating them is the deferred post-release
workstream, not this change.

## Measured result

- Baseline (`cargo test --workspace --tests`, local): 241.45s
- After (`cargo nextest run --workspace`, local): 99.71s
- The authoritative before/after for the CI job itself is the PR's own
  "Integration Tests (Linux)" run time; see the PR description.
