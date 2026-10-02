# Test Toolchain Footgun Guards: Design Note

This note is the durable design record for three toolchain guards that
turn recurring CI footguns into standing enforcement:

1. a test-timing budget,
2. an em-dash-in-test-strings visibility fix, and
3. CI-gate parity.

It carries the conclusions of the design pass and records what was
actually built.

## Motivation

Three failure modes kept recurring in CI:

- Integration-test runtime crept up with no automatic signal until a
  job felt slow. There was no machine-readable per-test timing and no
  committed budget, so a regression was only ever caught by a human
  noticing.
- The `no-em-dash-in-public-strings` (§8.6) blocking lint rule fired
  correctly on em dashes in Rust test-function string literals, but
  the failure was misread twice (WS-B2 acceptance panic messages; the
  RT-1 adversarial panic message). Both times the actual cause was a
  debugging-visibility problem, not a rule gap.
- The "minimum repo gate" command list lived inline in two places that
  drifted: the `AGENTS.md` prose said `cargo test --workspace` and
  omitted `chelis lint --check .`; `.github/workflows/ci.yml` ran
  `cargo nextest run --workspace` and a `chelis lint --check .` step
  the docs never mentioned.

## Guard 1: test-timing budget

### Design

- One `ci-fast` worker builds and executes units plus the reviewed integration
  targets. Its `ci-fast` nextest profile publishes JUnit; `ci-fast-receipts`
  records the Cargo selection, binary listing, and build/list/run timings.
  The stable `integration` aggregate requires success. Full workspace
  execution and phase oracles run nightly; `docs/ci_validation.md` records
  their current owners.

### Why these defaults

`tolerance = 2.0`, `min_regression_delta = 0.05`, and
`absolute_ceiling = 30.0`s preserve useful diagnostics. The committed
two-shard hosted baseline contained 8,166 tests and no observation above 30
seconds. Exact-head run `33248009321` of unchanged test code then reported a
36.64-second workspace case and 34.87-to-83.02-second cases in the dtype and
generalization lanes, alongside dozens of relative outliers. The slow cases
were nested-build and compile/execute tests competing inside the runner, not
one consistent regression. That evidence makes both threshold classes useful
for diagnosis and unsuitable for a one-sample required check.

### Exit codes

By default, `scripts/test_timing_check.py` exits `1` for either kind of finding.
With `--informational-relative`, relative-only findings exit `0`, while any
absolute-ceiling finding still exits `1`. Hosted CI uses `--informational`,
which reports both classes and exits `0` for a valid report. Exit `2` is a
usage or IO error (missing or malformed JUnit XML, bad config) under every
mode. A malformed or empty JUnit file is a loud error, never a silent pass.

## Guard 2: em-dash-in-test-strings visibility

### Reproduction outcome

The §8.6 rule (`crates/chelis-lint/src/rules/no_em_dash_in_public_strings.rs`)
already catches em dashes in every relevant Rust test-string
construct. Reproduced against the current rule:

- em dash in a `\`-continued multi-line panic-message string literal
  (the exact shape of both historical misses): caught;
- em dash in a raw string literal (`r#"..."#`): caught;
- em dash in a `format!` macro string argument: caught;
- ASCII hyphen-minus in any of the above: correctly NOT flagged.

So this is **not a `quoted_spans` parser gap**. It is a **visibility
problem**: `cmd_lint` printed every violation inline as targets were
walked, interleaving a handful of blocking ERROR lines into several
hundred advisory `prefer-pipe-operator` lines. CI failure debugging
read the advisory noise and missed the blocking errors. The WS-B2 fix
commit (`8990b1c`) says so directly: "The four warnings were buried in
~600 lines of advisory `prefer-pipe-operator` warnings".

### Fix

`cmd_lint` in `crates/chelis-cli/src/main.rs` now buckets rendered
violation lines by severity instead of printing inline. Advisory and
warning lines print first; blocking errors print last under a
delimited `=== N blocking lint error(s) ===` header followed by a
`lint --check failed: N blocking error(s) above` summary line. CI logs
show the tail, so the blocking errors are now the last and most
visible thing on screen.

The fix stays inside the existing rule + CLI. No parallel Python
charset checker was added: that would be a second source of truth for
§8.6.

### Tests

- `crates/chelis-lint/src/rules/no_em_dash_in_public_strings.rs` gains
  `#[test]` cases mirroring the historical misses: em dash in a raw
  string, in a `format!` arg, and in a multi-line string, each inside
  a `#[test]` fn body, plus ASCII-hyphen negative parity.
- `crates/chelis-cli/tests/cli.rs` gains
  `lint_check_surfaces_blocking_error_below_advisory_noise`, which
  pairs an advisory-triggering Surf file with an em-dash Rust file and
  asserts the blocking error prints under the header and after any
  advisory/warning lines.

## Guard 3: CI-gate parity

### Design

- `scripts/gate.py` is the single source of truth for the per-PR
  developer-runnable gate. It defines the command list once, split by
  CI stage (`lint-and-unit`, `integration`, `runtime-representation`), with
  the union as the full gate.
  The independent `ci-fast` stage owns the PR subset and is not added to
  the full/manual stage union. Nightly `integration-support` workers invoke
  the frontend and domain support slices; `runtime-representation` remains
  the chelis#893 Phase 0 oracle alone: its release-profile reproducers and
  serial mutation re-scans cost about eleven hosted minutes, so the
  `runtime-representation-phase0-oracle` job runs it on a runner of its own
  rather than doubling a workspace shard. The stable `Workspace Tests
  (Linux)` context aggregates the shards, and `Integration Tests (Linux)`
  aggregates it with the parallel phase oracles. A stage name runs one
  subset; `--list` prints the canonical full list and `--validation` derives
  per-crate tests from the diff against `origin/main`.
- `python3 scripts/gate.py ...` is a bootstrap command, not permission to use
  the system interpreter for gate logic. Unless it is already running in
  Devenv, a uv-created venv, or `uv run`, the script re-executes itself as
  `uv run --managed-python --python 3.11 --no-project python
  scripts/gate.py ...`. It validates an explicit `PYO3_PYTHON` or exports the
  selected interpreter to every child command.
- Every child command's combined stdout/stderr streams live and is teed to a
  temporary transcript. Successful transcripts are removed. On failure the
  transcript moves under `target/gate-failures/`; diagnostics include the
  stage/index, duration, exit code or signal, host and relevant environment,
  exact rerun command, complete-log path, and a 200-line tail replay.
- The gate normalizes `CARGO_TARGET_DIR` inside the current worktree and
  rejects external paths. It sets `CARGO_HUSKY_DONT_INSTALL_HOOKS=1` so a
  build cannot mutate the clone's shared Git hooks while sibling worktrees
  are active.
- Every nextest profile sets `fail-fast = false`, and gate-owned nextest
  commands also pass `--no-fail-fast` explicitly. A failing assertion does
  not cancel later tests that may reveal independent failures.
- The canonical full list (`python3 scripts/gate.py --list` is authoritative;
  the two feature-matrix clippy commands are abbreviated here):
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo clippy --workspace --all-targets --features <solver-free features> -- -D warnings`
  - `cargo clippy --workspace --all-targets --no-default-features -- -D warnings`
  - `cargo fmt --all -- --check`
  - `cargo run -p chelis-cli --bin chelis --quiet -- lint --check .`
  - `<managed-python> scripts/check_std_bundle_reproducible.py`
  - `cargo test -p chelis-types --doc`
  - `cargo test -p chelis-compiler-api --doc`
  - `cargo test -p chelis-pipeline-core --doc`
  - `<managed-python> scripts/check_checkpoint_compile_fail.py`
  - `<managed-python> scripts/check_hash_order_phase_b_compile_fail.py`
  - `<managed-python> scripts/check_configuration_closure.py`
  - `<managed-python> scripts/pipeline_core_dependency_guard.py`
  - `<managed-python> scripts/pipeline_core_documentation_guard.py`
  - `<managed-python> scripts/check_pipeline_core_compile_fail.py`
  - `cargo nextest run --workspace --no-fail-fast`
  - `<managed-python> scripts/compiler_front_end_performance.py`
  - `<managed-python> scripts/unrepresentable_domain_oracle.py`
  - `<managed-python> scripts/runtime_representation_oracle.py --phase 0`
- CI substitutes `cargo nextest run --workspace --profile ci
  --no-fail-fast` for the workspace-nextest command and delegates the excluded
  capacity-census binaries to the required
  dtype oracle.
- `.github/workflows/ci.yml` gate steps call
  `python3 scripts/gate.py <stage>` instead of inlining
  cargo/chelis commands.
- The parity guard pins the complete ordered set of single-line `run:` scalars
  in each gate-owned worker. It does not try to emulate Bash or classify an
  executable from shell text; every added command requires an explicit review.
- `AGENTS.md` "Minimum repo gate" points at `python3 scripts/gate.py`
  plus a `--list` echo of the canonical list and documents uv routing and
  retained failure diagnostics.
- Scope is the per-PR developer-runnable gate ONLY. The dtype oracle and
  result aggregator, sanitizer, macOS-smoke, LOC-report, no-AI-authorship,
  and docs CI jobs are out of scope by design.

### The parity lock

`scripts/test_gate.py` asserts:

- the `--stage` subsets union exactly to the full list;
- a structural parity assertion that parses every single-line `run:` scalar in
  the gate-owned jobs and compares the complete ordered list with an exact
  allowlist. This makes future drift a test failure without depending on a
  partial Bash parser. Every other CI job is classified by name
  (`NON_GATE_JOBS`) so the exclusion is visible and reviewable;
- `--list` prints the canonical list.
- uv/Devenv detection, unmanaged re-exec, missing-uv guidance, child
  `PYO3_PYTHON` propagation, success cleanup, and complete failed-command
  diagnostics are covered by `scripts/test_gate_diagnostics.py`.

A `run: |` multi-line block in any gate-owned worker is outside the exact
scalar contract; `test_no_multiline_run_in_gate_jobs` also names that
prohibition directly.

## Ordering

Guard 3 owns the CI-step refactor (it touches the gate steps in
`.github/workflows/ci.yml`); Guard 1 slots its timing telemetry step
and the `ci` nextest profile in afterward. Guard 2 is independent of
both.

## Acceptance

- `cargo build --workspace --all-targets`
- `cargo nextest run --workspace --no-fail-fast`
- `cargo test --workspace --lib`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo run -p chelis-cli --bin chelis --quiet -- lint --check .`
  (exit 0)
- `uv run --managed-python --python 3.11 --no-project python -m unittest
  scripts.test_test_timing_check scripts.test_gate scripts.test_gate_local
  scripts.test_gate_diagnostics`
- `python3 scripts/gate.py` (runs and passes)
