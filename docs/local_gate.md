# The local gate

`scripts/gate.py` is the single source of truth for the per-PR gate. This page
records how it behaves: the modes, what each mode runs, the interpreter it
resolves, the preflight, the lease, the reports it writes, and the regeneration
entry point behind it. `AGENTS.md` states only the rules an agent must follow;
the commands here are what those rules invoke.
[`ci_validation.md`](ci_validation.md) owns the hosted cadence, and
[`guard_changes_for_pr_authors.md`](guard_changes_for_pr_authors.md) owns the
pull-request lifecycle.

## Modes

```sh
python3 scripts/gate.py --fast                 # before every push: fixes in place, then checks
python3 scripts/gate.py --validation           # optional troubleshooting and extra validation
python3 scripts/gate.py --detach --validation  # optional run, detached
python3 scripts/gate.py --status [HANDLE]      # the detached run's real verdict
python3 scripts/gate.py --list                 # the canonical command list with ownership annotations
```

`--fast` is the pre-push gate: fix-in-place, run before every push. CI on the
pushed candidate owns routine PR validation and must pass before
ready-for-review. `--validation` is an optional way to reproduce checks on the
developer's machine; no per-PR run is required. The full workspace and broad
phase oracles run in `heavy-e2e.yml` daily at 03:17 UTC and on manual
dispatch. Passing PR checks does not certify those phase acceptance oracles;
dispatch them on the candidate when claiming completion. The PR test worker
calls `python3 scripts/gate.py ci-fast`: every default-feature lib/bin unit
target plus the reviewed integrations in `.config/ci-test-targets.toml`. The
legacy/full/manual gate selections remain available.

`scripts/test_gate.py` pins the complete ordered set of single-line `run:`
commands permitted in the gate-owned CI jobs, so shell syntax cannot hide an
unreviewed command. `--list` prints the canonical full list. Each printed
command carries one of four annotations: `fast + validation + ci` (the lint
and evaluator source-guard rows shared by `--fast` and `--validation`),
`validation + ci`, `ci-owned`, and `full gate; CI coverage split` (the workspace
nextest row). Two trailing `#` notes describe the dynamic stages: what
`--fast` runs, and the per-crate nextest `--validation` appends. The gate's own
output is the only authoritative list; no document transcribes it.

Before `--fast`, `--validation`, or another long local validation, fetch
`origin/main` so the changed-crate selection and inherited-failure comparison
use current evidence. If the branch is materially behind, reconcile it
deliberately before spending hours on a stale tree. When an unrelated failure
appears, reproduce or compare it on current `origin/main` before diagnosing it
as branch-owned.

## The Python bootstrap

`python3` is only the gate bootstrap. `scripts/gate.py` is stdlib-only and is
the one `python3` entry point that self-heals. Without an explicit
`PYO3_PYTHON`, it runs on this checkout's own interpreter whenever one exists:
the activated Devenv state venv under this checkout, else `.venv`. A launch
under any other interpreter re-executes through it, including another
checkout's venv that comes first on `PATH`; the capacity census
accepts only this checkout's interpreter and would otherwise refuse it at the
end of the runtime-representation stage. With no owned interpreter, a runtime
that is not already uv- or Devenv-managed re-executes through
`uv run --managed-python --python 3.11 --no-project` before running gate logic,
so `python3 scripts/gate.py` starts correctly in every environment. It exports
its selected interpreter as `PYO3_PYTHON` to every child command and computes
`PYO3_ENVIRONMENT_SIGNATURE` from that interpreter's build metadata. The
signature makes PyO3 refresh restored Cargo configuration when the interpreter
changes behind the same venv path. Its `--fast`
and `--validation` preflight warns when the worktree has no `.venv/bin/python`
(create it with `uv venv --python 3.11`): the census legs need it, and direct
cargo and nextest runs outside the gate fall back to it when `PYO3_PYTHON` is
unset. If uv is missing, the gate exits with installation and
Python-provisioning commands.

That re-exec drops `UV_PYTHON_PREFERENCE` from the child environment. uv reads
it as `--python-preference` and rejects it beside the `--managed-python` the
gate passes deliberately, so an ambient setting (Devenv exports `only-system`)
would otherwise turn the self-healing bootstrap into a hard
`error: the argument --managed-python cannot be used with --python-preference`.
Nothing else in the environment is altered: `UV_PYTHON_DOWNLOADS=never` still
fails loudly rather than letting the gate fetch an interpreter behind that
choice.

Inside Devenv, `chelis-gate` forwards every argument to `python3 scripts/gate.py`
and runs it under the activated `.devenv/state/venv` interpreter, the same one
`PYO3_PYTHON` names and the same one a developer gets by typing `python`. A
Devenv script can only name a Nix package, so the launcher starts under the
bare store CPython and hands the script the activated interpreter; without that
handoff the gate reads its own runtime as unmanaged and re-executes when it
should not.

## Interpreter resolution for cargo

Rust test and gate code resolves an interpreter through
`tests/support/managed_python.rs`: an explicit `PYO3_PYTHON` wins outright, and
the checkout's `.venv/bin/python` is the fallback. Outside Devenv,
`.cargo/config.toml` points `PYO3_PYTHON` at `.venv/bin/python`; Devenv
overrides it with `.devenv/state/venv/bin/python`. For direct Cargo commands in
a dedicated worktree, use that worktree's owned `.venv/bin/python`. An
explicit `PYO3_PYTHON` is authoritative: an invalid path must fail loudly
rather than fall back.

Hosted CI setup, hosted command publication, Devenv environment capture and
the gate compute the same `PYO3_ENVIRONMENT_SIGNATURE` from the selected
interpreter: executable and prefix paths, full version, ABI, library directory,
library name, pointer width and build flags. They reject `PYO3_CONFIG_FILE`,
`PYO3_NO_PYTHON` and `PYO3_CROSS*` overrides because those bypass native
interpreter discovery. CI supplies libpython only from that interpreter's
`LIBDIR`; a sibling installation is not a substitute.

Direct Cargo invocations bypass these entry points. After creating or replacing
the worktree venv, export its signature before reusing a Cargo target:

```sh
export PYO3_PYTHON="$PWD/.venv/bin/python"
export PYO3_ENVIRONMENT_SIGNATURE="$(.venv/bin/python -c 'from pathlib import Path; from scripts.ci_setup_uv_python import pyo3_environment_signature; print(pyo3_environment_signature(Path(".venv/bin/python")))')"
```

Two invocation forms follow, and the repository uses them consistently: every
script is `.venv/bin/python scripts/<name>.py`, and the gate is always
`python3 scripts/gate.py --fast`, `--validation`, or `--list`, never
`.venv/bin/python scripts/gate.py`. The
`uv run --managed-python --python 3.11 --no-project python ...` form appears
only where a document explains what the gate re-executes to, and as the
fallback when no `.venv` exists, never as a routine invocation. Inside an
activated Devenv shell, `.devenv/state/venv` serves the same purpose as `.venv`
and scripts are `python scripts/<name>.py`. Devenv is optional and, because
each command pays shell evaluation and activation unless it runs inside one
persistent shell, slower per command than the primary path; agent fleets use
the primary path.

**macOS:** Apple's bundled Python reports a stale `sysconfig.LIBDIR` path. Do
not route PyO3 to it.

**C front end.** The runtime-representation oracle and the `chelis-repr-inventory`
tests read the registered C and Objective-C headers through a `clang` binary on
PATH (any clang that prints `-ast-dump=json`), in addition to the `cc` the
capacity census already requires; a missing `clang` fails the scan loudly
rather than skipping it.

## What `--fast` runs

`--fast` is the inner-loop pass. It fixes in place and prints what it changed:
`scripts/regen_all.py --tier 0` (the rejection registry, the embedded
conformance skill assets, and the opaque-invariants corpus) and
`cargo fmt --all`, then `ci_change_owned.py classify-paths` over the changed
set, `chelis lint --check .`, `scripts/eval_system_guard.py` over the evaluator
source, `scripts/check_std_bundle_untracked.py` over the index,
`cargo clippy -p <crate> --tests -- -D warnings` for each changed crate,
one `cargo nextest run` over the drift tripwires (atom
partition, generated dtype header, compiler pins, opaque corpus,
loud-unsupported, payload census, bundled std loader, conformance manifest,
asset drift, skill-set uniformity, phase-3 gate inventory, stack-guard
coverage, runtime-extent target manifest), and, when a `packages/chelis-std/`
or `crates/chelis-std-bundle/` path changed,
`cargo nextest run -p chelis-std-bundle --lib` after the tripwires.

The chelis-std runtime each binary embeds is packed from `packages/chelis-std`
by `crates/chelis-std-bundle/build.rs` while the compiler builds, so a std edit
needs no regeneration step: the next build carries it. The bundled std loader
tripwire builds a fresh copy of the std sources with `chelis reef build` and
requires the embedded pair and the locks it writes to agree with those bytes,
and the bundle crate's tests check the input selection and the pinned archive
mtime. `scripts/check_std_bundle_untracked.py` refuses a tracked file under
`packages/chelis-std/dist/` or `crates/chelis-std-bundle/dist/`, and a tracked
`reef.lock` that records the bundled runtime: every lock reef writes names the
running binary's runtime, so a committed one goes stale with the first std
edit. The pre-commit hook runs it with `--tree` on a copy of the staged files
when a commit stages a path under either `dist/` directory or
`packages/chelis-std/reef.lock`; the gate and CI check every tracked lock.

Every writer runs before every check, and the path classification is the first
check because it is the cheapest row that can reject a push: it is the
planner's own rule lookup, and a new tracked file that no `[[path_rule]]`
routes fails `Plan Changed Integration Tests` in CI. It reports every unrouted path rather than the first
and prints the same sentence CI prints. It needs cargo on PATH, because it
reads the workspace package roots from `cargo metadata --no-deps --locked`,
and it costs a fraction of a second (see the measured figures in
[`ci_validation.md`](ci_validation.md)). It derives its set when it runs, as
the branch's diff from its merge base with `origin/main` plus uncommitted
tracked changes, so a branch behind `main` is not charged with paths `main`
changed after the fork; a missing `origin/main` or unrelated histories fail
the check. It matches the planner's rename handling, classifying both sides of
a move. It
classifies a path once git knows about it: untracked work is excluded, because
CI never sees it and nothing can route a scratch file. A modified artifact is
therefore classified in the run that changed it, and one a writer has just
created is classified by the next `--fast` after the artifact is staged.
Running `--fast` before every push runs it before that staging, so the create
case needs a second run that nothing requires; chelis#2262 has the fix and why
it is not here.

A local pass means "no unrouted path in my tree", not "CI will accept this".
It cannot see a divergence that comes from the package set itself: CI composes
base and candidate Cargo metadata inside the synthetic merge, while this reads
only the working tree, so an in-place crate rename can be ambiguous to the
planner and clean here.

`--fast` exits non-zero for any failing stage (fmt, regeneration, path
classification, lint, evaluator source guard, std-bundle tracking guard,
per-crate clippy, the tripwire run, or the bundle crate's tests) and never for
a file it fixed; a regenerated artifact is reported as a changed file to
commit, never as a failure. Changed files are reported from content hashes of
the porcelain set before and after the run, so a file that was already dirty
and that fmt changed further is still listed. It never runs a workspace clippy
row, the chelis#908 oracle, or the runtime-representation oracle, and it never
takes the lease.

## What `--validation` runs

`--validation` is an optional extra validation command, not the
pre-push gate; `--fast` is. It runs two of
the three workspace clippy configurations (`-D warnings`, compile-only): the
default row and the solver-free-features row. The `--no-default-features` row
is CI-owned through `gate.py lint-and-unit`, because
`check_configuration_closure.py` reconciles every repository `.rs` file against
the dep-info in the worktree's target, `crates/chelis-prove/src/clarabel_sos.rs`
is compiled per pull request only by the solver-free row, and the no-default
row compiles a strict subset of the default row. It then runs
`cargo fmt --check`, `chelis lint --check .`, the std-bundle reproducibility
check (`scripts/check_std_bundle_reproducible.py`, which runs the bundle build
script twice in one fresh target directory, each time in its own output
directory, and compares the packed bytes), the
explicit rustdoc commands, the checkpoint and
hash-order compile-fail fixtures, the configuration-closure check, both
pipeline-core guards, the chelis#908 unrepresentable-domain oracle, the
runtime-representation oracle, and
`cargo nextest run -p <crate> --no-fail-fast` for each crate changed vs
`origin/main` (committed diff plus uncommitted work; owning packages are
resolved from each member's `Cargo.toml`, not the directory name). The derived
crate list is always printed; "no crate changes detected" means the per-crate
stage was skipped, not silently empty. On macOS it compiles test-authored C
fixtures with Apple clang, which is no portability evidence;
[`ci_validation.md`](ci_validation.md) records where GCC sees them.

The workspace nextest stage is CI-owned. Mac workspace validation runs daily at
04:17 UTC in `macos-nightly.yml` and on manual dispatch; it is not a required
PR check, so dispatch it on the branch for Mac-specific changes.
[`local_macos_environment.md`](local_macos_environment.md) explains why the
workspace suite does not belong in the local loop on macOS.

If a cold `--validation` run is useful, launch it with
`python3 scripts/gate.py --detach --validation` and collect the result with
`python3 scripts/gate.py --status [HANDLE]`. The launcher's exit code is a
launch verdict and nothing more: the run's own exit code, exit 4 for a lease
timeout included, arrives through `--status`. Record the `--status` verdict and
the head it covered, never the launch.
[`investigations/agent_contract_rationale.md`](investigations/agent_contract_rationale.md)
§8 has the runs behind the evidence and invocation rules.

The gate runs `cargo nextest run --no-fail-fast` (CI's actual runner), not
`cargo test --workspace`, and includes `chelis lint --check .` (the §8.6 / §12
naming gate). The sanitizer, macOS-smoke, LOC-report, no-AI-authorship, docs,
and smt-build CI jobs are out of scope for this script by design. The separate
Secret scan job downloads its pinned detector and checks Git history; the
compiler gate does not run it.

### Doctests

The three explicit rustdoc stages exist because `cargo nextest` does not
execute doctests, and the gate deliberately does not use `--workspace --doc`.
`scripts/gate.py`'s module docstring records why, and what each stage covers.
Doctests only run where something invokes them: the canonical gate invokes
doctests for `chelis-types`, `chelis-compiler-api`, and `chelis-pipeline-core`.
The `backend-sanitizers` job also runs `cargo test -p chelis-backend-c --lib`
and an explicit `--doc` invocation. Full backend sanitizer integrations run
nightly in `heavy-e2e.yml`. A `compile_fail` oracle in another crate runs
nowhere until that crate gains an equivalent invocation in the same change set.

### The unrepresentable-domain oracle

`scripts/unrepresentable_domain_oracle.py` is chelis#908's authoritative
completion oracle, and the tracker requires every fix in that class to run it
continuously. The full command runs in `heavy-e2e.yml` daily at 03:17 UTC or on
manual dispatch, not in the required PR integration job. Acceptance is exit 0
with a final `ORACLE: PASS` line.

## Regeneration

`scripts/regen_all.py` is the regeneration entry point on its own as well.
Tier 0 writes by default; `--check` reports every stale artifact; `--full`
adds tier 1, the capacity census and the runtime-representation inventory.
The chelis-std runtime has no leg: the compiler build packs it. The
Python-binding baseline stores only stable reviewed rows, flags, authorities, and contracts; current
graph identities are execution evidence from the dedicated binding acceptance
test and are never persisted or regenerated. Full regeneration exits 2 naming
the manual action when a census row lands with citation `TODO` or the
regenerated inventory's digest differs from the reviewed `FREEZE_SHA256`. It
never writes a frozen digest, the loud-unsupported `BASELINE`, the binding or
wire census JSON, the dtype C header, or the tree-sitter parsers.

## Preflight

`--fast`, `--validation`, and the bare full gate run a preflight before the first
command: the environment checks (`PYO3_PYTHON`, `CARGO_TARGET_DIR`
containment, an explicit oracle handoff; exit 2 on failure), the git facts for
the run summary (never fatal), a warning when the worktree has no
`.venv/bin/python`, and, on macOS only, `scripts/preflight_exec_probe.py` as a
subprocess: probe exit 0 proceeds, exit 1 (wedged first exec) stops the gate
with exit 3 and the termination class `preflight-stop`, exit 3 (slow) and exit
2 (could not run) warn and proceed. On Linux the probe is skipped and the
summary records why; the environment checks, the `.venv` warning, the lease,
and the summary behave the same on both platforms. CI stage runs skip the
preflight.

The gate normalizes `CARGO_TARGET_DIR` to an absolute path inside the current
worktree and rejects paths outside it. It also sets
`CARGO_HUSKY_DONT_INSTALL_HOOKS=1` for child builds, preventing cargo-husky
from mutating the clone's shared `.git/hooks` while sibling worktrees run.
These controls isolate writable state; concurrent agents may still contend for
CPU and make each other slower.

## The lease

`--validation` and the bare full gate hold a workstation-wide `flock` on
`gate.lock` under `$CHELIS_GATE_LEASE_DIR`, else `$XDG_CACHE_HOME/chelis`, else
`~/.cache/chelis`. Updated runners acquire in ticket-registration order.
Cancellation, timeout, and process death release queue places; kernel locks
establish liveness, including after SIGKILL. Holder sidecars are descriptive
only.

Waits are indefinite by default, polling every 10 seconds; 60-second
heartbeats show holder details and queue position. `--no-wait` exits 4 on
lease/queue contention; `--lease-timeout SECONDS` caps the wait (exit 4);
`--no-lease` bypasses; queue errors exit 2. `--fast` only reports a holder.
Update active worktrees to honor the queue. Queue mechanics live in
`scripts/gate.py`; acceptance:
`.venv/bin/python -m unittest scripts.test_gate_queue`. The advisory lease
serializes `--validation` and full runs across worktrees on one workstation; it
never kills another process.

## Transcripts and reports

Every command's combined stdout and stderr streams live. On failure the gate
retains the complete transcript under `target/gate-failures/`, replays the
final 200 lines, and prints the stage/index, duration, exit code or signal,
relevant environment, exact rerun command, and transcript path. Successful
command transcripts are removed. Every run other than `--list`, pass or fail,
also writes `target/gate-reports/<utc-timestamp>-<pid>-<mode>.json` (or under
`$CHELIS_GATE_REPORT_DIR`): the mode, git facts, preflight and lease records,
per-stage seconds, the first failing stage, the termination class (`pass`,
`stage-failure`, `signal`, `environment`, `preflight-stop`, `lease-timeout`,
`user-cancel`, `internal-error`), and the files a `--fast` run changed; it then
prints one summary line with the stage count, seconds, verdict, and report
path. Record the seconds from that file when citing an optional `--validation` run
in the pull request.

## Documentation-only changes

Documentation-only changes require applicable CI on the candidate head. Hosted
docs-only classification is routing evidence, not proof that every changed
Markdown control artifact has an owning validator in that workflow.

Markdown parsed, embedded, mirrored, or used as agent instructions is a
control artifact, not inert prose. Its focused validators must run even when
CI reports `docs_only=true`. The always-run Docs job owns shared-agent-skill
validation: `scripts/check_agent_skills.py` validates metadata, the registered
shared set, source/embedded byte agreement, and Claude/Codex review-command
wrapper agreement; the validator and CI-routing tests exercise failure cases; and
`chelis-conformance`'s `asset_drift_tripwire` and `skill_set_uniformity` tests
exercise compiled assets and downstream distribution. Local reruns of these
checks are optional. Docs also builds mdBook and validates the package
skill/examples.

## Inner-loop discipline

- Run `cargo check -p <crate> --tests` before the first nextest of a change,
  then use focused `cargo nextest run -p <crate> --test <file>` commands for
  the inner development loop: each compiles only what it names. Do not
  substitute a workspace-wide `cargo test` run for the canonical gate.
- Tests that are slow or require heavyweight local prerequisites are
  `#[ignore]` by default and invoked through a documented manual gate;
  [`manual_gates.md`](manual_gates.md) is the inventory, and every ignored test
  has a concrete manual command and expected success condition there or in the
  owning phase docs.

## Build concurrency

- Concurrent agents or subagents that build or test MUST use an isolated
  target directory: set `CARGO_TARGET_DIR=target/agents/<name>`, or use a
  separate git worktree with its own `target/`. A relative `CARGO_TARGET_DIR`
  resolves against the invocation cwd, not the workspace root, so run cargo
  from the repo root or use an absolute path. Never share the primary
  `target/` with a session that may be building concurrently; cargo's
  target-dir lock serializes the builds and feature/profile differences
  invalidate each other's caches.
- Before building, list orphaned cargo/rustc/cargo-nextest/chelis processes
  with `python3 scripts/reap_orphans.py` and reap them with
  `python3 scripts/reap_orphans.py --kill` (`chelis-reap-orphans` inside
  Devenv). Review the dry-run listing first: ppid==1 cannot distinguish an
  abandoned build from a deliberately detached one you are still tailing. Run
  it from the checkout whose `target/` you are about to use; scoping is
  per-checkout. Orphaned runs keep burning CPU and hold the cargo lock across
  sessions.
- Before a subagent starts a heavyweight Cargo command, it reports the exact
  command and expected weight to the orchestrator, which checks active
  processes and load, then runs, staggers, or declines it. Isolated targets
  prevent state corruption but do not eliminate CPU starvation.
- Contention diagnostic: several unrelated tests FAILing at near-identical
  wall-clock times (for example all ~217s, nextest's slow-kill) means CPU
  starvation, not code breakage. Re-run on a quiet machine before treating
  those as real failures;
  [`investigations/agent_contract_rationale.md`](investigations/agent_contract_rationale.md)
  §2 has the measured factor.
- At session end, verify that task-owned background cargo, rustc, and nextest
  processes are gone. A stopped wrapper is not proof that its reparented
  children stopped; use the scoped `reap_orphans.py` dry run and kill only
  confirmed task-owned stragglers.
- macOS workstation only: first-exec assessment can degrade under mass
  fresh-binary bursts and stall multi-binary test runs at ~0 CPU.
  Probe with `python3 scripts/preflight_exec_probe.py` (exit 1 wedged, exit 3
  slow), or `chelis-exec-preflight` inside Devenv, before a local workspace
  nextest stage. If the probe reports degradation, use the manually dispatched
  `macos-nightly.yml` workflow per
  [`local_macos_environment.md`](local_macos_environment.md).

`scripts/reap_orphans.py` and `scripts/preflight_exec_probe.py` are invoked as
bare `python3` on purpose: they run before and independently of a working
project environment, so they stay standard-library only and keep parsing on
the oldest system Python a supported workstation ships (macOS still ships
3.9). `scripts/test_bootstrapless_scripts.py` locks both properties.
