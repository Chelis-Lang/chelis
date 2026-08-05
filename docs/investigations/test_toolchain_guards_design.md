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

- The CI workspace-tests job runs `cargo nextest run --workspace`. A `ci`
  nextest profile (`.config/nextest.toml`) adds a `[profile.ci.junit]`
  section so nextest writes machine-readable per-test JUnit XML to
  `target/nextest/ci/junit.xml`.
- `scripts/test_timing_check.py` parses that JUnit XML
  (`<testcase classname=... name=... time=...>`, keyed as
  `binary::test`) and flags tests over budget.
- Thresholds are config, never hardcoded:
  - `scripts/test_timing_config.json` holds `tolerance` (a multiplier)
    and `absolute_ceiling` (seconds).
  - `scripts/test_timing_baseline.json` maps `binary::test` -> seconds.
- The check flags a test when either:
  - it is NEW relative to the baseline AND runs longer than
    `absolute_ceiling`, or
  - it regressed past `tolerance` x its baseline time.
- The baseline is hand-curated and explicitly regenerated, NOT
  auto-regenerated on merge. Auto-regen would launder a real
  regression into the baseline. Regeneration is one documented
  command: `python3 scripts/test_timing_check.py --update-baseline`.
- CI wiring: an informational, non-failing step
  (`continue-on-error: true`) runs after the integration test step.
  It is promotable to blocking later by removing `continue-on-error`.

### Why these defaults

`tolerance = 2.0` and `absolute_ceiling = 30.0`s were chosen against
the current-state baseline: the slowest existing tests are the
`phase3j_pre_std` stdlib build oracles at ~42s. They are already in
the baseline, so they are only flagged if they more than double. A
genuinely new test that lands over 30s is worth a look. A separate
workstream (typecheck cache) is expected to shift the timing baseline
shortly; when it lands, the baseline is regenerated with the one
documented command above.

### Exit codes

`scripts/test_timing_check.py` exits `0` when nothing is over budget,
`1` when one or more tests are over budget, and `2` on usage / IO
error (missing or malformed JUnit XML, bad config). A malformed or
empty JUnit file is a loud error, never a silent pass.

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
  CI stage (`lint-and-unit`, `integration`), with the union as the
  full gate. The workflow's `workspace-tests` job invokes the integration
  stage; the stable `Integration Tests (Linux)` context aggregates that job
  with the parallel Phase 0-3 oracle. A stage name runs one subset; `--list`
  prints the canonical full list and `--local` derives per-crate tests from
  the diff against `origin/main`.
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
- The canonical full list:
  - `cargo build --workspace --all-targets`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo fmt --all -- --check`
  - `cargo run -p chelis-cli --bin chelis --quiet -- lint --check .`
  - `cargo test -p chelis-types --doc`
  - `cargo test -p chelis-compiler-api --doc`
  - `<managed-python> scripts/check_checkpoint_compile_fail.py`
  - `cargo nextest run --workspace --no-fail-fast`
- CI substitutes `cargo nextest run --workspace --profile ci
  --no-fail-fast` for the last
  command and delegates the excluded capacity-census binaries to the required
  dtype oracle.
- `.github/workflows/ci.yml` gate steps call
  `python3 scripts/gate.py <stage>` instead of inlining
  cargo/chelis commands.
- `AGENTS.md` "Minimum repo gate" points at `python3 scripts/gate.py`
  plus a `--list` echo of the canonical list and documents uv routing and
  retained failure diagnostics.
- Scope is the per-PR developer-runnable gate ONLY. The dtype oracle and
  result aggregator, sanitizer, macOS-smoke, LOC-report, no-AI-authorship,
  and docs CI jobs are out of scope by design.

### The parity lock

`scripts/test_gate.py` asserts:

- the `--stage` subsets union exactly to the full list;
- a parity assertion that greps `.github/workflows/ci.yml` and asserts
  every `cargo`/`chelis` invocation in a gate step is produced by
  `gate.py`. This is the lock that makes future drift a test failure.
  The parity test excludes the sanitizer / macOS-smoke / docs /
  LOC-report / no-AI-authorship jobs by name (`NON_GATE_JOBS`) so the
  exclusion is visible and reviewable;
- `--list` prints the canonical list.
- uv/Devenv detection, unmanaged re-exec, missing-uv guidance, child
  `PYO3_PYTHON` propagation, success cleanup, and complete failed-command
  diagnostics are covered by `scripts/test_gate_diagnostics.py`.

A `run: |` multi-line block in either gate job would hide its commands
from the line-based parity parser; `test_no_multiline_run_in_gate_jobs`
disallows it so parity stays enforceable.

## Ordering

Guard 3 owns the CI-step refactor (it touches the gate steps in
`.github/workflows/ci.yml`); Guard 1 slots its new informational step
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
