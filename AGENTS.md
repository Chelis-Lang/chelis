# Chelis Agent Contract

Canonical agent instructions for this repository.
`CLAUDE.md` should resolve to this file so Claude-style and Codex-style entry points do
not drift.

## Quality Standards

### Spec-First Development

- Before writing implementation, write test stubs derived from the owning spec.
- Every spec requirement should have a corresponding test before the code exists.
- If the spec says "X is a type error," write the failing test before implementing the checker.
- A phase is not done until every active spec requirement has both positive and negative
  coverage.

### Negative Test Parity

- For every test that checks something works, add the corresponding failure test.
- If you cannot name the failure case, the spec understanding is still weak.

### Do Not Trust Green

- Passing tests prove alignment with the tests, not necessarily with the spec.
- After green CI, check what active requirements still lack tests.
- Audit silent fallbacks, default values, empty error vectors, and `unwrap_or` paths.

### Phase Completion Criteria

- Do not claim a phase is done based on crate-local green or narrative progress.
- A phase is done when the executable acceptance surface is green, docs are honest, and
  the red team can only find minor residual issues.
- Budget for at least one adversarial validation pass that runs code, not just source review.

### One Acceptance Oracle Per Phase

- Every phase must name one authoritative completion oracle.
- That oracle may be a single command, a named suite, or a documented manual runner, but
  it must be explicit.
- Supporting evidence may exist, but it does not replace the oracle.
- If the oracle is manual or long-running, document it in the owning phase plan and the
  current-state docs.

## Red Team Protocol

### Baseline

- Fresh review context is preferred when practical.
- Red team against the spec, the code, the tests, the examples, and the CLI behavior.
- Execute tests and commands; do not treat source inspection as sufficient proof.
- For this repository, a requested "red team agent" means a fresh local subagent in a
  new context. A main-thread validation pass does **not** satisfy that request.

### Required Red-Team Behaviors

- Write and run adversarial tests when coverage is missing.
- Verify that inputs which should fail do fail, and with the right reason.
- Verify that inputs which should pass do pass, with exact outputs where applicable.
- Check docs and phase claims against the shipped behavior, not just intent.

### Fresh-Context Enforcement

- When asked to run a red team or "spawn a red team agent", first close any known stale
  or failed subagents from the current session and then spawn a new local subagent with
  fresh context.
- If the first spawn attempt routes to remote infrastructure, errors, or comes back in a
  broken state, close that handle and retry until you have either:
  1. a working fresh local subagent, or
  2. an explicit statement that red-team validation is blocked because fresh local
     subagent execution is unavailable.
- Do not substitute main-thread validation and call it a red team.
- Do not mark a phase as red-teamed unless the fresh-context subagent actually ran the
  validation work.

## Documentation And Spec Sync

### Documentation Hierarchy

1. `spec/design/chelis_canonical_reference.md`
2. `spec/00-12*.md`
3. `spec/design/chelis_project_plan.md`
4. `spec/design/archive/`

If active docs disagree, fix the disagreement instead of adding a third explanation.

### Public-Surface Change Rule

When behavior changes, update the owning code, tests, docs, and examples in the same
change set:

- parser / type system / IR / backend tests
- CLI integration tests
- executable examples in `examples/`
- active specs and current-state docs

## Contract Invariants

Machine-facing contracts should be expressed as invariants and locked with tests.

Examples:

- if a command reports perfect success, its error list must be empty
- formatter output must remain parseable on the supported corpus
- decompiler output must round-trip through the supported parser path
- executable examples must remain executable after canonical formatting
- status docs must not claim a stronger guarantee than the repo actually proves

When a public surface has an implicit invariant, make it explicit and test it.

## Example Corpus Policy

- `examples/` means executable Phase 0 examples that should survive `chelis fmt` and
  `chelis check` cleanly.
- `examples/illustrative/` is for syntax or design examples that are useful but not on
  the executable Phase 0 path.
- Do not mix those meanings in tests or docs.
- Decide the executable-vs-illustrative split early in a phase, not after examples have
  already been used as proof artifacts.

## Scripting Language Policy

- **Python** for all scripts, utilities, report generators, and automation helpers.
  Write tests for them.
- **Rust** where the task naturally fits a compiled workspace member.
- **Never shell.** Do not write `.sh` scripts. If a CI step needs a one-liner, invoke
  Python instead. Shell is fragile and untestable.
- **One exception — the published bootstrap installer.** Shell is permitted *only* for
  the chelisup bootstrap one-liner (`chelisup.sh`, the `curl -fsSL … | sh` installer). It
  runs on a bare machine *before* any chelis, cargo, or even Python exists, so Python is
  not an option — it is not guaranteed present either, which is the whole bootstrap
  problem. It MUST be minimal POSIX `sh`, `shellcheck`-clean, and covered by a test
  (`sh -n` parse plus `shellcheck` when available). This is a single-purpose carve-out for
  the one artifact that cannot be anything else; every other script remains Python.
- Existing `scripts/` directory uses Python; follow that convention.
- **Use the uv-managed Python** at `.venv/bin/python`, not the system Python.
  Create it once with `uv venv --python 3.11` from the repo root. `py/pyproject.toml`
  pins `requires-python = ">=3.11"`. See [`README.md`](README.md) for the full setup.

## Build Toolchain

`chelis-python` links against `libpython` and `.cargo/config.toml` sets
`PYO3_PYTHON` to `.venv/bin/python` so the link step finds the project-pinned
interpreter. This is a hard prerequisite on every platform: `cargo build` for
any crate that transitively pulls pyo3 will fail without a `.venv/`. On macOS
specifically, Apple's bundled `python3` is 3.9 and reports a stale
`sysconfig.LIBDIR` pointing at a non-existent Xcode framework path; using the
uv-managed interpreter sidesteps that. Run `uv venv --python 3.11` once before
building.

## Build And Gate Commands

Minimum repo gate:

```sh
python3 scripts/gate.py
```

`scripts/gate.py` is the single source of truth for the per-PR
developer-runnable gate. CI calls `python3 scripts/gate.py <stage>` for
each split job, and `scripts/test_gate.py` asserts the CI workflow
hand-inlines no gate command the script does not produce. To see the
canonical list:

```sh
python3 scripts/gate.py --list
# cargo build --workspace --all-targets  # ci-owned
# cargo clippy --workspace --all-targets -- -D warnings  # local + ci
# cargo fmt --all -- --check  # local + ci
# cargo run -p chelis-cli --bin chelis --quiet -- lint --check .  # local + ci
# cargo nextest run --workspace --profile ci  # ci-owned
# # --local also runs: cargo nextest run -p <crate> for each crate changed vs origin/main
```

The gate runs `cargo nextest run` (CI's actual runner), not `cargo test
--workspace`, and includes `chelis lint --check .` (the §8.6 / §12
naming gate). The sanitizer, macOS-smoke, LOC-report, no-AI-authorship,
docs, and smt-build CI jobs are out of scope for this script by design.

Local pre-push gate (chelis#360):

```sh
python3 scripts/gate.py --local
```

`--local` runs the developer pre-push subset: workspace clippy
(`-D warnings`, compile-only), `cargo fmt --check`,
`chelis lint --check .`, and `cargo nextest run -p <crate>` for each
crate changed vs `origin/main` (committed diff plus uncommitted work;
owning packages are resolved from each member's `Cargo.toml`, not the
directory name). The derived crate list is always printed; "no crate
changes detected" means the per-crate stage was skipped, not silently
empty. The workspace nextest stage is CI-owned: run `--local` before
pushing, open a draft PR early, and let CI (macOS Smoke is the
authoritative workspace oracle) run the full suite. See
[`docs/local_macos_environment.md`](docs/local_macos_environment.md)
for why the workspace suite does not belong in the local loop on
macOS.

Documentation-only changes (Markdown/prose with no code, fixture, or
example edits) are exempt from `--local`: skip the local gate, push,
and require green CI instead. The gate's clippy/build/test stages
cannot be affected by prose, and CI still runs the lint stage plus the
Docs job (mdBook build and the `skill_suite` example validator), which
cover everything a docs-only diff can break.

Default-gate discipline:

- `cargo test --workspace` is the inner development loop and should stay under roughly 60
  seconds on a machine without GPU/PyTorch
- tests that exceed that budget or require heavyweight local prerequisites should be
  `#[ignore]` by default and invoked through a documented manual gate
- every ignored test must have a concrete manual command and expected success condition in
  the owning phase docs

Phase-specific manual gates must be called out explicitly when they are not part of the
default workspace run.

## Build Concurrency And Process Hygiene

- Concurrent agents or subagents that build or test MUST use an isolated target
  directory: set `CARGO_TARGET_DIR=target/agents/<name>`, or use a separate git
  worktree with its own `target/`. A relative `CARGO_TARGET_DIR` resolves
  against the invocation cwd, not the workspace root, so run cargo from the
  repo root or use an absolute path. Never share the primary `target/` with a
  session that may be building concurrently; cargo's target-dir lock serializes
  the builds and feature/profile differences invalidate each other's caches.
- Before building, list orphaned cargo/rustc/cargo-nextest/chelis processes with
  `python3 scripts/reap_orphans.py` and reap them with
  `python3 scripts/reap_orphans.py --kill`. Review the dry-run listing first:
  ppid==1 cannot distinguish an abandoned build from a deliberately detached
  one (`nohup cargo build` you are still tailing). Run it from the checkout
  whose `target/` you are about to use; scoping is per-checkout. Orphaned runs
  keep burning CPU and hold the cargo lock across sessions.
- Contention diagnostic: several unrelated tests FAILing at near-identical
  wall-clock times (for example all ~217s, nextest's slow-kill) means CPU
  starvation, not code breakage. Measured 2026-06-10: the 25-test
  `rank_poly_tier3` suite took 2,434s under contention vs 24s on a quiet
  machine. Re-run on a quiet machine before treating those as real failures.
- Recommended inner loop: `cargo nextest run -p <crate> --test <file>` compiles
  only that test target.
- macOS workstation only: first-exec assessment can degrade under mass
  fresh-binary bursts and stall multi-binary test runs at ~0 CPU (chelis#356).
  Probe with `python3 scripts/preflight_exec_probe.py` (exit 1 wedged, exit 3
  slow) before trusting the gate's workspace nextest stage locally; when
  degraded, fall back to CI (macOS Smoke) for that stage per
  [`docs/local_macos_environment.md`](docs/local_macos_environment.md).

## Local HIP Environment

This workstation has a reconciled AMD/ROCm HIP setup, so HIP manual gates are locally
runnable. The authoritative runbook is [`docs/local_hip_environment.md`](docs/local_hip_environment.md).

**For any HIP manual gate (especially hipBLAS-linked tests), run via**
`scripts/hip_test.py` — e.g.
`scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1`.
The wrapper sets `HSA_OVERRIDE_GFX_VERSION=11.5.1`, the full `LD_LIBRARY_PATH`, and
the full `HIPCC_COMPILE_FLAGS_APPEND` (including `-L` to the gfx1151 wheel lib that
hipBLAS link resolution needs). Plain `cargo test --ignored` inherits only the
`environment.d/hip.conf` defaults, which segfault hipBLAS-linked binaries at process
exit with empty output — looks like a code regression but is purely environmental.

Key durable invariant: the wheel ROCm stack is authoritative, and
`~/.config/environment.d/hip.conf` provides the `HIPCC_COMPILE_FLAGS_APPEND` include
override so non-interactive shells and `cargo test --workspace` resolve HIP headers from
the `_rocm_sdk_core` wheel. Use `rocminfo` as the source of truth for local GPU probing;
do not assume `rocm-smi` is installed.

## Manual Gates

- Every manual acceptance gate must have a documented command, expected success condition,
  and owning phase.
- If default CI does not run the gate, the docs must say so directly.
- Ignored tests are allowed only when they clearly mirror a documented manual gate or an
  environment-dependent prerequisite.
- Phase summaries must not imply that a manual gate is part of the default workspace pass
  when it is not.

## CLI Surface Discipline

- CLI commands are part of the product surface, not wrappers around library tests.
- Formatter, decompiler, evaluator, checker, and build-command behavior should be tested
  against a corpus, not only single happy-path examples.
- For machine-facing CLI output, test both shape and semantic invariants.

## Style Gate

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` invoke `chelis fmt --check` and the full
blocking `chelis lint` rule set on the input file before the
front-end pipeline runs. Style failures block the build by default and
emit a one-issue-per-line diagnostic to stderr. Advisory lint warnings
report valid-but-non-preferred source and do not fail `lint --check` or
the built-in gate.

- Authoritative source of truth: `spec/01-nomenclature.md` (the rule
  spec) and `crates/chelis-lint/src/rules/` (the executable
  enforcement). The canonical formatter is `chelis_surf::format` for
  `.ch` and `chelis_deep::printer` for `.dp`.
- Override flag: `--allow-style-violations` bypasses the gate with a
  stderr warning. Use only for emergency local builds and one-off
  migrations. CI must not pass it. The flag bypasses only the style
  gate, not parse, type, effect, validation, evaluation, or backend
  errors.
- Test override env var: `CHELIS_STYLE_GATE_DISABLE=1` disables the
  gate process-wide. Reserved for the integration-test corpus that
  synthesizes ad-hoc Surf to exercise type/effect/linearity behavior;
  do not set it in production CI or in user-facing scripts.

When writing new code or fixtures, run `chelis fmt --inplace <file>`
and `chelis lint --check` before pushing. The gate replaces the older
manual checklist of "remember to run fmt"; if the gate is green and
`cargo test --workspace` passes, the change is ready.

## Surf Style Guide

When writing or rewriting Surf in this repository:

- prefer `def ... -> T = ...` over `def ... : T = ...` (enforced by the
  `surf-def-arrow-form` lint rule, §3.5)
- put types on function parameters, not on load-style top-level bindings
- use symbolic dimensions such as `batch` and `seq` for runtime-varying axes
- keep fixed architecture dimensions concrete
- do not annotate intermediate expressions when inference already determines the type
- keep meaningful intermediates like `h1`, `logits`, `probs`, and `loss`
- combine short tensor operations when the composed expression is clearer than over-decomposed single-op bindings
- pipe stages use first-argument insertion: `x |> f(y)` means
  `f(x, y)`. Use `x |> fn (v) -> f(y, v)` when the piped value belongs
  in a later argument position.
- treat decompiler-generated verbose load chains and checker-inserted ascriptions as debug output, not example style
- user code typically writes neither explicit `&` nor explicit `copy()` for fan-out
  into read-only primitives; auto-borrow handles it. Write `&x` when an exported
  API or dense signature benefits from clarity. Write `copy(x)` only when forking
  ownership for downstream consumption.
- existing fixtures and migration baselines may keep explicit `copy()`
  or `drop()` calls when they prove compatibility or preserve baseline
  evidence. `redundant-linearity-call` is advisory and should not be
  papered over with blocking-rule exceptions.
- lowered IR now carries compiler-inserted `Copy` and `Drop` nodes for implicit
  linearity. If auto-copy/auto-drop produces unexpected IR, treat it as a
  structural blocker and escalate against `spec/design/implicit_linearity.md`
  rather than papering over it as a routine fixture bug.
- type identifiers are PascalCase (`surf-type-pascal-case`, §3.1)
- function/value identifiers are snake_case (`surf-value-snake-case`, §3.2)
- functions carrying the `Test` effect are named `test_*` or `example_*`
  (`surf-test-name-prefix`, §10.1)

## Chelis-Specific Rules

### Deep AST

- Every Deep node is a 3-tuple: `(tag {} children...)`
- Metadata map is always present at element 1
- 62-tag closed vocabulary; see `spec/03-deep-syntax.md`
- Function application is `app`, names are `var`, literals are `lit`
- RISC primitives are built-in functions, not tags

### Type System

- No implicit precision promotion
- Named tensor dimensions match by name
- No implicit broadcasting; explicit `expand` only
- Integer literals default to `int32`, float literals to `f32`

### Build Reality

- `chelis build` emits C, header, and runtime artifacts plus compile flags (default target)
- `chelis build --target hip` emits C/HIP host code with embedded GPU kernel strings
- Neither target invokes the native compiler — the user runs `gcc`/`hipcc` manually
- On this workstation specifically, the local HIP toolchain was
  reconciled 2026-05-01 and HIP manual gates are runnable. See
  [`docs/local_hip_environment.md`](docs/local_hip_environment.md) for the authoritative
  runbook.

## Toolchain And Packaging Orchestration

Chelis has a rustup-style install/version layer and a one-command project
orchestrator. The authoritative design is
[`spec/design/chelis_packaging_and_install.md`](spec/design/chelis_packaging_and_install.md);
the user-facing guide is [`docs/book/src/install.md`](docs/book/src/install.md)
and [`docs/book/src/reef.md`](docs/book/src/reef.md).

- **`chelisup`** (`crates/chelisup`) is the toolchain installer and the
  pin-resolving `chelis` **shim**. It owns the toolchain lifecycle: bootstrap
  (`crates/chelisup/bootstrap/chelisup.sh`, the one permitted shell script),
  `install` / `default` / `list-installed` / `which` / `show` / `uninstall` /
  `self uninstall`. The store is `$CHELIS_HOME` (default `~/.chelis`):
  `toolchains/<ver>`, `bin/{chelis,chelisup}`, `reef/`, `src/`.
- **Shim resolution order** (first match wins): `+<ver>` arg → `CHELIS_TOOLCHAIN`
  → nearest `chelis-toolchain` file → nearest `reef.toml` `compiler =` pin →
  recorded default. A resolved-but-not-installed version is a loud error naming
  `chelisup install <ver>`; the shim never auto-installs on `cd` and never
  silently falls back. `+latest` is not a supported reference (concrete pins
  only).
- **`chelis reef setup [--path]`** (`cmd_reef_setup` in `crates/chelis-cli`) is
  the orchestrator: ensure the pinned toolchain (auto-install via the chelisup
  binary), `reef install --from-lockfile`, `reef src sync` when `[chelis-src]`
  is present, then a `reef doctor` summary. `reef doctor` is its read-only
  counterpart across all classes (toolchain, source crates, binary artifacts).
- **Shim-corruption trap (do not regress):** `reef setup`'s toolchain step MUST
  subprocess the real `chelisup` binary. Never call `chelisup::install::install`
  in-process from `chelis-cli`: that helper copies `current_exe()` into
  `<home>/bin/{chelis,chelisup}`, which from the `chelis` binary overwrites the
  shim with the compiler. There is a guard comment at the call site.
- **§5.4 invariant:** `setup`'s auto-install is *explicit* provisioning and is
  therefore exempt from the "no auto-install" rule, which governs only the
  *implicit* shim. The unknown-subcommand hint augments only clap's
  `InvalidSubcommand`.

Use the `packaging-install` skill when changing or validating any of this.

## Shared Local Skills

Project-local skills live in `agent-skills/`.
`.claude/skills` and `.codex/skills` should resolve to that same directory so both tool
surfaces load the same skill library.
Command wrappers should stay mirrored too: `.claude/commands/` and `.codex/commands/`
should stay behaviorally aligned so slash-command access does not drift between tool
surfaces. Keep a `red-team` alias wired to `redteam-exec`, and make that wrapper enforce
stale-agent cleanup plus a fresh local subagent before any validation is counted as a
red team.

Current shared skill set:

- `redteam-exec`
- `spec-sync`
- `phase-gate`
- `backend-numerics`
- `example-corpus`
- `cli-surface`
- `packaging-install`

## Downstream Shell Contract

Downstream shell repos (every repo in the canonical-reference §Shell
Ecosystem table) inherit this `AGENTS.md` verbatim (machine-local
environment sections excepted; see the contract §1) AND must satisfy
[`spec/design/shell_repo_contract.md`](spec/design/shell_repo_contract.md):
pin hygiene with a mechanical multi-location consistency guard, a
per-shell `docs/CHELIS_SURFACE.md` capability inventory, the
narrowing-citation rule (every workaround cites `chelis#NNN` at the site —
file upstream, never silently work around), expected-to-fail blocker
probes under `tests_blocked/` re-run at every pin bump, negative-test
sidecars, vendored shared skills, and uv-managed Python. The contract
names `Chelis-Lang/school` as its reference implementation. Changes to the
contract land here first and propagate to every shell per its scaffolding
drift rule.
