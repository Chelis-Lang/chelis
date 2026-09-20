#!/usr/bin/env python3
"""Single source of truth for the per-PR developer-runnable repo gate.

Background: the "minimum repo gate" command list lived inline in two
places that drifted apart: the `AGENTS.md` prose ("cargo test
--workspace", no `chelis lint --check .`) and `.github/workflows/ci.yml`
(`cargo nextest run --workspace`, plus a `chelis lint --check .` step
the docs never mentioned). This script makes the list authoritative in
one file: CI calls `python3 scripts/gate.py <stage>`, `AGENTS.md`
points at `python3 scripts/gate.py`, and `scripts/test_gate.py` asserts
the CI workflow contains no hand-inlined gate command that this script
does not produce.

Scope: this is the per-PR developer-runnable gate ONLY. The sanitizer,
macOS-smoke, LOC-report, no-AI-authorship, and docs CI jobs are
deliberately out of scope; `scripts/test_gate.py` excludes those jobs
by name so the exclusion is visible and reviewable.

Usage (an unmanaged launcher is automatically re-executed through uv):
    python3 scripts/gate.py            # run every gate command
    python3 scripts/gate.py lint-and-unit   # run the Rust-policy subset
    python3 scripts/gate.py ci-fast         # hosted units + reviewed integrations
    python3 scripts/gate.py integration     # retain the legacy integration subset
    python3 scripts/gate.py integration --tests-only --partition hash:1/2
    python3 scripts/gate.py integration --support-only
    python3 scripts/gate.py runtime-representation  # run the #893 oracle stage
    python3 scripts/gate.py --list     # print the canonical full list,
                                       # annotated fast/local/CI-owned
    python3 scripts/gate.py --fast     # the pre-push gate: fix in place, then
                                       # lint, per-crate clippy, tripwires
    python3 scripts/gate.py --local    # optional troubleshooting and local
                                       # validation; CI owns PR readiness

Local/CI stage split: the complete developer gate and legacy integration stage
retain their existing selections. The separate `ci-fast` stage builds product
prerequisites and runs units plus the reviewed Cargo integration targets through
`scripts/ci_test_targets.py`; it is intentionally absent from STAGE_ORDER so the
full/manual gate does not execute a redundant subset. Full Linux and broad
phase-oracle validation runs in heavy-e2e.yml nightly or by manual dispatch;
Mac coverage runs in macos-nightly.yml. PR success does not certify those oracles.
The workspace execution stays out of `--local` because the mass first-exec burst
can wedge assessment on a Mac workstation (docs/local_macos_environment.md).
The local command runs two workspace clippy configurations
(compile-only, no mass exec), fmt, `chelis lint`, the regeneration and
compile-fail guards, both oracles, plus `cargo nextest run -p <crate>
--no-fail-fast` for each crate changed vs `origin/main` (committed diff plus
uncommitted work). The derived crate list is always printed so nothing is
silently skipped.

`--fast` is the pre-push gate, run before every push. It fixes in place
(`regen_all.py --tier 0`, plus `--tier 1` when a chelis-std path changed, then
`cargo fmt --all`), then runs `chelis lint --check .`, `cargo clippy -p <crate>
--tests -- -D warnings` for each changed crate, one nextest run over the drift
tripwires, and, when a chelis-std path changed, the bundle's self-consistency
test. Every writer runs before every check. It exits non-zero for any failing
stage (fmt, regeneration, lint, per-crate clippy, the tripwire run, or the
std-bundle self-test), never for a file it fixed, and it
prints every file the run changed (content hashes of the porcelain set before
and after, so a file that was already dirty and that fmt changed further is
still reported). It never runs the workspace clippy rows, the chelis#908
oracle, or the runtime-representation oracle, and it never takes the lease.

Why `--local` runs two of the three Clippy configurations
---------------------------------------------------------
`check_configuration_closure.py` (in `--local`) reconciles every repository
`.rs` file against rustc dep-info found under this worktree's target.
`crates/chelis-prove/src/clarabel_sos.rs` is a whole module behind the
`clarabel` feature, and the solver-free row is the only per-pull-request row
that compiles it, so on a fresh target one configuration followed by the closure
check fails. The `--no-default-features` row compiles a strict subset of the
default row (no whole file is gated on `cfg(not(feature = ...))`), so dropping
it from `--local` loses only the developer-side linting of the
`#[cfg(not(feature = "chelis-prove"))]` regions, which `gate.py lint-and-unit`
still lints on Linux for every pull request. `--list` marks that row `ci-owned`.

Preflight, lease, and run summary
---------------------------------
`--fast`, `--local`, and the bare full gate run a preflight before the first
command: the environment checks in `gate_environment` (exit 2 on failure), the
git facts for the summary (never fatal), a warning when the worktree has no
`.venv/bin/python` (the gate exports `PYO3_PYTHON`, so only direct cargo and
nextest runs outside the gate need it), and, on macOS only, the first-exec
probe `scripts/preflight_exec_probe.py` as a subprocess: exit 1 (wedged) stops
the gate with exit 3, exit 2 and 3 warn and proceed. On Linux the probe is
skipped and the summary records `{"verdict": "skipped", "reason": "not
darwin"}`; every other preflight, lease, and summary behavior is the same on
both platforms. CI stage runs skip all of it.

`--local` and the bare full gate then take an advisory workstation-wide lease,
`fcntl.flock` on `gate.lock` under `$CHELIS_GATE_LEASE_DIR`, else
`$XDG_CACHE_HOME/chelis`, else `~/.cache/chelis`,
held for the whole run so two cold full gates in different worktrees cannot
starve each other. The default is to wait indefinitely, polling every 10 s with
a heartbeat naming the holder and queue position every 60 s. Numbered tickets
in `gate.lock.queue/` are registered under a short `gate.lock.queue.lock` mutex;
only the oldest live ticket may acquire. Ticket liveness is kernel-backed, so
cancelled or killed waiters cannot leave a permanently blocking ticket. Queue
errors stop with exit 2. `--no-wait` exits 4 when the lease or queue is busy;
`--lease-timeout SECONDS` bounds the entire wait; `--no-lease` bypasses.
`--fast` only reports a holder. FIFO requires updated runners in each worktree.
The kernel releases the lock on holder death, SIGKILL included, so the JSON
sidecar beside the lock is descriptive, never authoritative; nothing is killed.

Every non-`--list` run, pass or fail, writes
`target/gate-reports/<utc>-<pid>-<mode>.json` (or under
`$CHELIS_GATE_REPORT_DIR`) with per-stage seconds, the first failing stage, the
termination class (pass, stage-failure, signal, environment, preflight-stop,
lease-timeout, user-cancel, internal-error), the preflight and lease records,
and the files a `--fast` run changed, then prints one human summary line.

Detached runs (chelis#1568)
---------------------------
`--detach` starts the run in its own session and returns at once, printing a
handle; `--status [HANDLE]` reports that run's verdict and exits with it. This
exists because the run outlives a caller's foreground command limit: the three
`--local` runs of the 2026-09-02 fleet took 7m51s, 9m56s and 10m02s against a
ten-minute limit.

The launcher is not a gate run. It creates no `GateReport`, writes no summary
and takes no lease, so "exactly one summary per run" still holds. The CHILD
runs `run_local`/`run_full` unchanged, so it takes the lease itself, holds the
flock for its whole life, and writes a sidecar naming its own pid.
`target/gate-failures/` is unaffected, because `run_commands` builds that path
from the module-level repo root and the child is spawned with `cwd=repo_root`.

`--no-wait`, `--no-lease` and `--lease-timeout` are forwarded to the child
untouched, which moves exit 4 out of the shell: with `--detach` the launcher's
exit code is a LAUNCH verdict (0 spawned, 2 could not spawn), and the lease
timeout surfaces as `--status`'s exit 4. `--status` returns 75 while the run is
alive, 1 if it died without writing a summary, and 2 for a missing or
malformed handle. `--detach --fast` is rejected: `--fast` fixes in place, and a
writer running unattended against a tree the agent is still editing is the
collision chelis#1568 is about.

The script is safe to run from any cwd: child commands use the repo root
(resolved relative to the script's own location) as their working directory.
Every child inherits one validated `PYO3_PYTHON`. Combined stdout/stderr is
streamed live; a failed command's complete transcript and a 200-line replay
are written under `target/gate-failures/`. Cargo output is pinned to this
worktree, and nextest continues after failures to expose the complete set.

Why four explicit rustdoc stages
---------------------------------
`cargo nextest` does not execute doctests, so a crate with a doctest
contract needs an explicit `cargo test -p <crate> --doc` command or that
contract runs nowhere. The `chelis-types` command runs the chelis#731
`ErrorWitness` contracts; `chelis-ir` runs the chelis#1286 verified-ownership
privacy contracts; the `chelis-compiler-api` and
`chelis-pipeline-core` commands run the compiler pipeline artifact
contracts. The `backend-sanitizers` CI job separately runs an unfiltered
`cargo test -p chelis-backend-c`, which picks up that crate's doctests.

The gate deliberately does not use `--workspace --doc`: that would make
every workspace doc example part of the gate without a reviewed scope
change. Adding a crate is a deliberate act, one command at a time.

Why the checkpoint script's own tests are not evidence
------------------------------------------------------
`check_checkpoint_compile_fail.py` checks the raw-offset fixture against
exact Rust diagnostics. Its Python unit tests use fake runners and never
execute the fixture, so they are evidence about decision logic only. The
same is true of `unrepresentable_domain_oracle.py`, whose unit tests patch
the command runners.

Why the unrepresentable-domain oracle runs in `integration`
-----------------------------------------------------------
Two of chelis#908's obligations run `cargo nextest`, which the Rust-policy
worker deliberately does not install, so the oracle cannot live in
`lint-and-unit`. `scripts/test_gate.py` locks both the stage membership and
the pairing between that stage and a nextest-installing job.

The CHELIS_ORACLE_BINARY handoff (chelis#1322)
----------------------------------------------
The oracle builds its own `chelis` before its first `.dp` fixture. Inside a
gate run that binary already exists, so the gate hands the built path over
in `CHELIS_ORACLE_BINARY` and the oracle skips the build. The gate sets it
only for a command list whose earlier `cargo run -p chelis-cli --bin chelis`
command provably builds that bin target; the support-only integration slice
used by hosted CI sets nothing and keeps the build-it-yourself behavior.
`scripts/test_gate.py` locks the build-before-oracle ordering the handoff
rests on, so a reorder cannot quietly turn the oracle cold again.

The variable is an explicit override and is therefore authoritative: a path
that is not an executable file is a loud failure, never a silent fall back
to a build, and an explicit setting from the caller is never replaced. That
is the same discipline applied to an explicit `PYO3_PYTHON`, timing
included -- `gate_environment` validates an explicit handoff, so a bad one
aborts before the first command rather than after the whole `--local` subset
has run.

Present-but-empty is a failure on both sides, not an off switch. Reading
`export CHELIS_ORACLE_BINARY=` as "unset" would disable the handoff with no
notice anywhere, so unset it entirely instead. The spelling itself lives in
exactly one place: the oracle declares it and this script imports it,
because two independent literals would let a rename keep every test green
while the handoff was dead.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The chelis#908 oracle owns its handoff variable's spelling, and this file
# must not carry a second copy of it (chelis#1322). Two independent literals
# would let a rename on either side keep the whole suite green while the
# handoff was dead: the oracle would see no variable, rebuild its own binary,
# and still print `ORACLE: PASS`. That silent degradation is the exact
# failure the handoff's fail-closed design exists to prevent, so the spelling
# is imported rather than repeated. `scripts/` goes on the path first because
# `scripts/test_gate.py` loads this file by path with the repo root, not
# `scripts/`, on `sys.path`. The oracle module is stdlib-only and imports
# cleanly on the macOS system Python 3.9, so this runs before the uv re-exec
# below without disturbing the bootstrap guidance.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from unrepresentable_domain_oracle import ORACLE_BINARY_ENV  # noqa: E402

# NB: `tomllib` is intentionally NOT imported at module top. It is stdlib
# only from Python 3.11. The gate bootstrap now re-executes an unmanaged
# launcher through uv before `main`, but tests and module consumers can import
# this file without taking that path. Keep the import deferred so those uses
# still receive focused guidance instead of a raw traceback (chelis#366).

REPO_ROOT = Path(__file__).resolve().parent.parent
MANAGED_PYTHON = "<managed-python>"
FAILURE_TAIL_LINES = 200
DIAGNOSTIC_ENVIRONMENT = (
    "PYO3_PYTHON",
    "VIRTUAL_ENV",
    "DEVENV_STATE",
    "CARGO_TARGET_DIR",
    "CARGO_HUSKY_DONT_INSTALL_HOOKS",
    ORACLE_BINARY_ENV,
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "RUSTFLAGS",
    "CARGO_BUILD_JOBS",
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "MACOSX_DEPLOYMENT_TARGET",
    "UV_RUN_RECURSION_DEPTH",
    "UV_PYTHON_INSTALL_DIR",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "CI",
    "GITHUB_ACTIONS",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "PATH",
)
PYTHON_VERSION_PROBE = (
    "import sys; "
    "raise SystemExit(0 if sys.version_info >= (3, 11) else 86)"
)

# The canonical CI-stage command list. The CI workflow
# has Rust-policy and workspace worker jobs. The workspace test workers run
# hash partitions of the integration stage's nextest command, while one
# support worker runs its non-test oracles exactly once.
# The full developer gate substitutes the complete default nextest profile for
# CI's split profile so the delegated census binaries are not dropped locally.
# Every
# command is a list of argv tokens (no shell).
#
# Keep this in lockstep with `.github/workflows/ci.yml`: the parity
# test in `scripts/test_gate.py` greps the workflow and fails if any
# `cargo`/`chelis` invocation in a gate step is not produced here.
BUILD_WORKSPACE: list[str] = ["cargo", "build", "--workspace", "--all-targets"]
CLIPPY_WORKSPACE: list[str] = [
    "cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings",
]
# The second half of the Clippy configuration matrix
# (`spec/design/hash_order_determinism.md` C2.3): every declared feature
# that needs no external solver toolchain. Clippy lints only the
# configuration it compiles, so a feature nobody compiles is a hole in the
# `disallowed_types` ban. `scripts/check_configuration_closure.py` owns the
# registry and fails if this command drifts from it.
# The off-state half of the matrix. An additive feature matrix never compiles
# `#[cfg(not(feature = ...))]`, and `chelis-cli`'s optional `chelis-prove`
# dependency is an implicit *default* feature guarding 25 such regions. Clippy
# lints only the configuration it compiles, so without this row those regions
# are linted by nothing at any cadence.
CLIPPY_NO_DEFAULT_FEATURES: list[str] = [
    "cargo", "clippy", "--workspace", "--all-targets", "--no-default-features",
    "--", "-D", "warnings",
]
CLIPPY_SOLVER_FREE_FEATURES: list[str] = [
    "cargo", "clippy", "--workspace", "--all-targets", "--features",
    "chelis-backend-c/sleef,"
    "chelis-compiler-api/compilation-trace,"
    "chelis-compiler-api/native-random-observer,"
    "chelis-e2e/hip-local-gpu,"
    "chelis-ir/lowering-trace,"
    "chelis-prove/clarabel,"
    "chelis-python/extension-module,"
    "chelis-runtime/ownership-ledger,"
    "chelis-types/checkpoint-compile-probe,"
    "chelis-types/generalize-sweep-oracle",
    "--", "-D", "warnings",
]
FMT_CHECK: list[str] = ["cargo", "fmt", "--all", "--", "--check"]
CHELIS_LINT_CHECK: list[str] = [
    "cargo",
    "run",
    "-p",
    "chelis-cli",
    "--bin",
    "chelis",
    "--quiet",
    "--",
    "lint",
    "--check",
    ".",
]
CHELIS_STD_BUNDLE_CHECK: list[str] = [
    MANAGED_PYTHON,
    "scripts/regenerate_chelis_std_bundle.py",
    "--debug",
    "--check",
]
# The default profile is the complete developer workspace suite. The `ci`
# profile writes JUnit XML and delegates two census binaries to the parallel
# dtype oracle.
NEXTEST_WORKSPACE: list[str] = [
    "cargo", "nextest", "run", "--workspace", "--no-fail-fast",
]
NEXTEST_WORKSPACE_CI: list[str] = [
    "cargo",
    "nextest",
    "run",
    "--workspace",
    "--profile",
    "ci",
    "--no-fail-fast",
]
NEXTEST_TARGETED_UNITS: list[str] = [
    "cargo",
    "nextest",
    "run",
    "--workspace",
    "--lib",
    "--bins",
    "--locked",
    "--profile",
    "ci-fast",
    "--ignore-default-filter",
    "--no-fail-fast",
]
TARGETED_PACKAGES_ENV = "CHELIS_TARGETED_PACKAGES"
# chelis#875: `cargo nextest` does not execute doctests. Each crate with
# a compile-fail contract needs an explicit rustdoc command. The current
# gate covers the type-system and compiler-pipeline contracts. The
# `backend-sanitizers` CI job also runs that crate's doctests through an
# unfiltered `cargo test -p chelis-backend-c` command.
#
# Do not replace these commands with `--workspace --doc`. That command
# makes every workspace doc example part of the gate without a reviewed
# scope change.
DOCTEST_TYPES: list[str] = ["cargo", "test", "-p", "chelis-types", "--doc"]
DOCTEST_IR: list[str] = ["cargo", "test", "-p", "chelis-ir", "--doc"]
DOCTEST_COMPILER_API: list[str] = [
    "cargo",
    "test",
    "-p",
    "chelis-compiler-api",
    "--doc",
]
DOCTEST_PIPELINE_CORE: list[str] = [
    "cargo",
    "test",
    "-p",
    "chelis-pipeline-core",
    "--doc",
]
TARGETED_DOCTESTS: dict[str, list[str]] = {
    "chelis-types": DOCTEST_TYPES,
    "chelis-ir": DOCTEST_IR,
    "chelis-compiler-api": DOCTEST_COMPILER_API,
    "chelis-pipeline-core": DOCTEST_PIPELINE_CORE,
}
# This script verifies the exact compiler diagnostic from the standalone
# raw-offset fixture. The marker is replaced with the same validated managed
# interpreter exported to child commands as PYO3_PYTHON.
CHECKPOINT_COMPILE_FAIL: list[str] = [
    MANAGED_PYTHON,
    "scripts/check_checkpoint_compile_fail.py",
]
# The liveness proof for the ban itself. `clippy.toml` is read by nothing else
# continuous, and no workspace source spells the banned types today, so
# deleting the two `disallowed-types` entries would leave every job green.
# This fixture compiles code that must be rejected and fails if it is not.
HASH_ORDER_PHASE_B_COMPILE_FAIL: list[str] = [
    MANAGED_PYTHON,
    "scripts/check_hash_order_phase_b_compile_fail.py",
]
# chelis#1341 Phase B completeness lock. Clippy's `disallowed_types` bans the
# raw hash collections, but lints only the configuration it compiles.
# Reconciles the repository's Rust sources against rustc's own dep-info from
# the Clippy stages above, so the compiler reports what it compiled instead
# of a script recomputing it. Must run after both Clippy commands.
CONFIGURATION_CLOSURE: list[str] = [
    MANAGED_PYTHON,
    "scripts/check_configuration_closure.py",
]
# The pipeline-core boundary guards. Before this, they ran only in the manual
# `compiler_pipeline_oracle.py`, so a forbidden dependency, a false no_std
# claim, or a broken facade compile-fail boundary passed hosted CI green. The
# dependency guard (one `cargo metadata`) and the documentation guard (pure
# Python) are cheap enough for the `--local` subset; the pipeline-artifact
# compile-fail fixture builds an out-of-workspace crate, so it stays in the
# per-PR gate stage (CI + full gate) alongside the checkpoint fixture.
PIPELINE_CORE_DEPENDENCY_GUARD: list[str] = [
    MANAGED_PYTHON,
    "scripts/pipeline_core_dependency_guard.py",
]
PIPELINE_CORE_DOCUMENTATION_GUARD: list[str] = [
    MANAGED_PYTHON,
    "scripts/pipeline_core_documentation_guard.py",
]
PIPELINE_CORE_COMPILE_FAIL: list[str] = [
    MANAGED_PYTHON,
    "scripts/check_pipeline_core_compile_fail.py",
]
# The chelis#908 unrepresentable-domain oracle. #908's "Constraint on every
# fix in this class" requires it to run in a continuous job: before this it
# was invoked by no workflow and no gate stage, so the only thing exercising
# it was its own unit tests, which patch the command runners and therefore
# never ran the behavioral oracle against a compiled binary. Every obligation
# it carries drives real compiled artifacts: the built `chelis` binary over
# `.dp` fixtures, and compiled test binaries through `cargo nextest`.
# Acceptance is exit 0 with a final `ORACLE: PASS` line.
#
# It belongs to the `integration` stage, not `lint-and-unit`, because those
# compiled obligations need `cargo nextest`. The Rust-policy worker
# deliberately does not install it, while workspace shard 2 installs it and
# has already built the workspace, so the oracle's two
# `nextest run` calls and its `cargo build -p chelis-cli` are warm. The
# separate `Verify nextest profile coverage` step also needs nextest, but runs
# on shard 1 to balance the hosted work. `--local` keeps both obligations: a
# developer machine running the gate already has nextest.
UNREPRESENTABLE_DOMAIN_ORACLE: list[str] = [
    MANAGED_PYTHON,
    "scripts/unrepresentable_domain_oracle.py",
]

# chelis#893 Phase 2, retaining the complete Phase 1 host/C receipt and Phase 0
# inventory/mutations before the generated ABI, device-owner, Python/DLPack,
# and backend-header census legs. The inherited release builds and serial
# mutation scans dominate the hosted cost, so the oracle remains its own gate
# stage and CI job. This preserves the workspace shards' partition wall and
# gives the cross-language receipt one failure boundary. The stage needs
# `cargo nextest`, clang, and the managed Python; `--local` keeps the same
# obligation on developer machines.
RUNTIME_REPRESENTATION_ORACLE: list[str] = [
    MANAGED_PYTHON,
    "scripts/runtime_representation_oracle.py",
    "--phase",
    "2",
]

# chelis#1205's authoritative front-end complexity and parity oracle. It
# reruns the focused structural counters after the workspace suite so their
# 20/40/80/160 growth evidence has a named continuous acceptance marker.
# Work counts, not wall time, own the threshold; hosted-runner CPU contention
# therefore cannot make the gate flaky.
COMPILER_FRONT_END_PERFORMANCE_ORACLE: list[str] = [
    MANAGED_PYTHON,
    "scripts/compiler_front_end_performance.py",
]

# chelis#1322. The oracle's `resolve_chelis_binary()` runs its own
# `cargo build -p chelis-cli --bin chelis` before its first `.dp` fixture.
# Inside a gate run that binary already exists: every command list this
# script runs the oracle in builds it earlier. Naming the built path in
# ORACLE_BINARY_ENV lets the oracle skip re-entering cargo for an artifact
# it was handed.
#
# The gate commands that leave a usable `chelis` at <target>/debug/chelis.
# Both are unconditional builds of that exact bin target, so their presence
# earlier in a list is a static guarantee rather than an assumption about
# cargo's behavior. `cargo nextest run --workspace` is deliberately NOT
# here: it happens to build the bin today because chelis-cli has
# integration tests, but that is an implicit consequence of cargo's test
# harness rules, not something this list states, and the handoff fails
# closed rather than falling back.
CHELIS_BINARY_PRODUCERS: tuple[tuple[str, ...], ...] = (
    tuple(BUILD_WORKSPACE),
    tuple(CHELIS_LINT_CHECK),
)

LOWERING_TRACE_TESTS: list[str] = [
    "cargo", "nextest", "run", "-p", "chelis-ir", "--features", "lowering-trace",
    "--lib", "--test", "lowering_trace", "--test", "helper_lowering_trace",
]

EMISSION_OBSERVER_TESTS: list[str] = [
    "cargo", "nextest", "run", "-p", "chelis-compiler-api", "--features",
    "compilation-trace,native-random-observer", "--lib", "--test", "emission_observer",
    "--test", "execution_artifact_metadata", "--test", "compilation_trace",
    "--test", "native_random_observer",
    "--test", "fixed_control_c",
]

STAGES: dict[str, list[list[str]]] = {
    "ci-fast": [[MANAGED_PYTHON, "scripts/ci_test_targets.py"]],
    "targeted-units": [NEXTEST_TARGETED_UNITS],
    "lint-and-unit": [
        CLIPPY_WORKSPACE,
        CLIPPY_SOLVER_FREE_FEATURES,
        CLIPPY_NO_DEFAULT_FEATURES,
        FMT_CHECK,
        CHELIS_LINT_CHECK,
        CHELIS_STD_BUNDLE_CHECK,
        DOCTEST_TYPES,
        DOCTEST_IR,
        DOCTEST_COMPILER_API,
        DOCTEST_PIPELINE_CORE,
        CHECKPOINT_COMPILE_FAIL,
        HASH_ORDER_PHASE_B_COMPILE_FAIL,
        CONFIGURATION_CLOSURE,
        PIPELINE_CORE_DEPENDENCY_GUARD,
        PIPELINE_CORE_DOCUMENTATION_GUARD,
        PIPELINE_CORE_COMPILE_FAIL,
    ],
    "integration": [
        NEXTEST_WORKSPACE_CI,
        LOWERING_TRACE_TESTS,
        EMISSION_OBSERVER_TESTS,
        COMPILER_FRONT_END_PERFORMANCE_ORACLE,
        UNREPRESENTABLE_DOMAIN_ORACLE,
        [MANAGED_PYTHON, "scripts/dtype_builtin_atom_closure_oracle.py"],
    ],
    "runtime-representation": [
        RUNTIME_REPRESENTATION_ORACLE,
    ],
}

STAGE_ORDER: list[str] = [
    "lint-and-unit",
    "integration",
    "runtime-representation",
]

HASH_PARTITION_RE = re.compile(
    r"^hash:(?P<shard>[1-9][0-9]*)/(?P<count>[1-9][0-9]*)$"
)

# The static subset for optional `--local` validation (chelis#360). Deliberately
# excludes BUILD_WORKSPACE (clippy already compiles everything; no mass
# first-exec burst) and NEXTEST_WORKSPACE (CI-owned; macOS Smoke is the
# authoritative workspace oracle). `--local` appends a dynamic
# `cargo nextest run -p <crate> --no-fail-fast` stage per changed crate; see
# `local_command_list`.
#
# Two of the three Clippy configurations, not one and not three.
# CONFIGURATION_CLOSURE's leg 3 reconciles every repository `.rs` file against
# the dep-info in this worktree's target, and
# `crates/chelis-prove/src/clarabel_sos.rs` is compiled per pull request only
# by CLIPPY_SOLVER_FREE_FEATURES, so that row must stay or `--local` fails on
# every fresh worktree. CLIPPY_NO_DEFAULT_FEATURES compiles a strict subset of
# CLIPPY_WORKSPACE (no whole file is gated on `cfg(not(feature = ...))`), so
# dropping it here loses only the developer-side lint of the 25
# `#[cfg(not(feature = "chelis-prove"))]` regions, which `lint-and-unit` still
# lints on Linux in CI for every pull request, whichever platform the
# developer runs `--local` on. It stays in STAGES so `--list` keeps publishing
# it (`ci-owned`) and `check_configuration_closure.py` still finds its owner.
LOCAL_STATIC_COMMANDS: list[list[str]] = [
    CLIPPY_WORKSPACE,
    CLIPPY_SOLVER_FREE_FEATURES,
    FMT_CHECK,
    CHELIS_LINT_CHECK,
    CHELIS_STD_BUNDLE_CHECK,
    DOCTEST_TYPES,
    DOCTEST_IR,
    DOCTEST_COMPILER_API,
    DOCTEST_PIPELINE_CORE,
    CHECKPOINT_COMPILE_FAIL,
    HASH_ORDER_PHASE_B_COMPILE_FAIL,
    CONFIGURATION_CLOSURE,
    PIPELINE_CORE_DEPENDENCY_GUARD,
    PIPELINE_CORE_DOCUMENTATION_GUARD,
    UNREPRESENTABLE_DOMAIN_ORACLE,
    RUNTIME_REPRESENTATION_ORACLE,
    LOWERING_TRACE_TESTS,
    EMISSION_OBSERVER_TESTS,
]

# The `--fast` inner-loop pass. Fix-in-place commands first, so the tree the
# read-only checks see is already normalized: regenerate the tier-0 artifacts
# (Python only, sub-second), then `cargo fmt --all` in write mode. The lint row
# is shared with `--local` and is also the `chelis` builder the tripwire
# nextest reuses. Per-crate clippy and the tripwire run are appended by
# `fast_command_list`.
FMT_WRITE: list[str] = ["cargo", "fmt", "--all"]
REGEN_TIER0_WRITE: list[str] = [
    MANAGED_PYTHON, "scripts/regen_all.py", "--tier", "0",
]
REGEN_TIER1_WRITE: list[str] = [
    MANAGED_PYTHON, "scripts/regen_all.py", "--tier", "1",
]


def classify_paths_command(changed_paths: list[str]) -> list[str]:
    """Reject a changed path the change-owned planner would refuse.

    The planner itself cannot run here: in `pull_request` mode it requires a
    two-parent synthetic merge, so against a working tree it exits before
    classifying anything. This calls the same
    `static_path_classification` the planner uses, and prints the same
    `unclassified changed path: <path>` sentence CI prints, so the local and
    hosted failures read alike (chelis#2250).
    """
    return [
        MANAGED_PYTHON, "scripts/ci_change_owned.py", "classify-paths",
        "--from-git",
    ]
# The bundle crate's in-crate `archive_self_consistency` test: the compile-time
# counterpart of the std-bundle regeneration check. It cannot join
# FAST_TRIPWIRE_NEXTEST because `--lib` would apply to every `-p` there and
# pull in `chelis-compiler-api`'s 25 s source-architecture test.
STD_BUNDLE_SELF_CONSISTENCY: list[str] = [
    "cargo", "nextest", "run", "-p", "chelis-std-bundle", "--lib",
    "--no-fail-fast",
]
# One nextest invocation, one test target per drift tripwire. Package and
# target selection compiles only these targets; the default profile's filter
# still drops the heavy bundled_chelis_std_loader property oracle. The
# multi-package `-p X --test A -p Y --test B` form is the one
# `scripts/loud_unsupported_phase3_oracle.py` already uses.
FAST_TRIPWIRE_NEXTEST: list[str] = [
    "cargo", "nextest", "run", "--no-fail-fast",
    "-p", "chelis-deep", "--test", "atom_partition_tripwire",
    "-p", "chelis-runtime", "--test", "runtime_dtype_generated_header",
    "-p", "chelis-cli",
    "--test", "compiler_pin_tripwire",
    "--test", "opaque_corpus_gate",
    "--test", "loud_unsupported_tripwire",
    "--test", "issue_729_payload_census",
    "--test", "bundled_chelis_std_loader",
    "-p", "chelis-conformance",
    "--test", "manifest_tripwire",
    "--test", "asset_drift_tripwire",
    "--test", "skill_set_uniformity",
    "-p", "chelis-compiler-api", "--test", "phase3_gate_inventory",
    "-p", "chelis-types",
    "--test", "stack_guard_coverage",
    "--test", "runtime_extent_target_manifest",
]
FAST_STATIC_COMMANDS: list[list[str]] = [
    REGEN_TIER0_WRITE,
    FMT_WRITE,
    CHELIS_LINT_CHECK,
]
# A change under either prefix appends the two std legs to `--fast`.
STD_PATH_PREFIXES: tuple[str, ...] = (
    "packages/chelis-std/",
    "crates/chelis-std-bundle/",
)

LOCAL_ANNOTATION = "local + ci"
FAST_ANNOTATION = "fast + local + ci"
CI_OWNED_ANNOTATION = "ci-owned"
FULL_GATE_SPLIT_ANNOTATION = "full gate; CI coverage split"
FAST_DYNAMIC_NOTE = (
    "# --fast runs, fixing in place: <managed-python> scripts/regen_all.py "
    "--tier 0 (and --tier 1 when a std path changed); cargo fmt --all; the "
    "chelis lint row above; cargo clippy -p <crate> --tests -- -D warnings "
    "per changed crate; one nextest run over the drift tripwires; and, when "
    "a std path changed, cargo nextest run -p chelis-std-bundle --lib"
)
LOCAL_DYNAMIC_NOTE = (
    "# --local also runs: cargo nextest run -p <crate> --no-fail-fast "
    "for each crate changed vs origin/main"
)

# Preflight, lease, and summary settings. Both directories are overridable
# so tests and operators can redirect files; the lease deliberately lives
# outside every worktree, because a `target/`-relative path is per-worktree
# and would defeat cross-worktree serialization.
REPORT_DIR_ENV = "CHELIS_GATE_REPORT_DIR"
LEASE_DIR_ENV = "CHELIS_GATE_LEASE_DIR"
LEASE_FILE_NAME = "gate.lock"
LEASE_POLL_SECONDS = 10.0
LEASE_HEARTBEAT_SECONDS = 60.0
TRANSIENT_RETRY_SECONDS = 0.1
PROBE_TIMEOUT_SECONDS = 180.0
PROBE_RUNBOOK = "docs/local_macos_environment.md"
SUMMARY_SCHEMA_VERSION = 1
EXIT_ENVIRONMENT = 2
EXIT_PREFLIGHT_STOP = 3
EXIT_LEASE_TIMEOUT = 4
EXIT_USER_CANCEL = 130

# Detached runs (chelis#1568). The handle and the combined log live beside the
# run summaries so `$CHELIS_GATE_REPORT_DIR` still controls placement and tests
# stay isolated.
DETACH_SUBDIR = "detached"
DETACH_SCHEMA_VERSION = 1
# sysexits EX_TEMPFAIL. No gate run can produce 75, and "not now, retry" is
# exactly what a poll against an unfinished run means, so `--status` can say
# "no verdict yet" without colliding with a verdict.
EXIT_STILL_RUNNING = 75


def is_managed_runtime(
    environ: dict[str, str],
    executable: Path,
    prefix: Path,
    base_prefix: Path,
) -> bool:
    """Return whether the current interpreter is managed by uv or Devenv."""
    if environ.get("UV_RUN_RECURSION_DEPTH"):
        return True

    devenv_state = environ.get("DEVENV_STATE")
    if devenv_state:
        devenv_prefix = Path(devenv_state) / "venv"
        if prefix == devenv_prefix or executable.parent.parent == devenv_prefix:
            return True

    if prefix != base_prefix:
        try:
            config = (prefix / "pyvenv.cfg").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return False
        if any(line.strip().startswith("uv =") for line in config.splitlines()):
            return True
    return False


def ensure_managed_runtime(
    argv: list[str],
    *,
    environ: dict[str, str] | None = None,
    executable: Path | None = None,
    prefix: Path | None = None,
    base_prefix: Path | None = None,
    find_uv=shutil.which,
    execvpe=os.execvpe,
    error_stream=None,
) -> int | None:
    """Re-exec an unmanaged gate launch through uv.

    Returns ``None`` when the current runtime is already managed. A successful
    re-exec never returns. Missing uv returns 127 after actionable setup
    guidance.
    """
    environment = dict(os.environ if environ is None else environ)
    current_executable = Path(sys.executable) if executable is None else executable
    current_prefix = Path(sys.prefix) if prefix is None else prefix
    current_base = Path(sys.base_prefix) if base_prefix is None else base_prefix
    error = sys.stderr if error_stream is None else error_stream

    if is_managed_runtime(
        environment,
        current_executable,
        current_prefix,
        current_base,
    ):
        return None

    uv = find_uv("uv")
    if uv is None:
        print(
            "gate: uv is required to select the repository-managed Python "
            "3.11 runtime, but `uv` was not found on PATH.",
            file=error,
        )
        print(
            "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
            file=error,
        )
        print(
            "Then verify and provision Python: `uv --version` and "
            "`uv python install 3.11`.",
            file=error,
        )
        return 127

    # uv reads UV_PYTHON_PREFERENCE as --python-preference and rejects it
    # alongside the --managed-python this command passes deliberately, so an
    # ambient setting turns the self-healing bootstrap into `error: the
    # argument --managed-python cannot be used with --python-preference`
    # (chelis#1421). Devenv exports only-system, but any uv user may set it,
    # and the bootstrap is what makes a bare `python3 scripts/gate.py` safe
    # from any shell. Drop it for this child only: the gate's own flag decides
    # which interpreter runs the gate. UV_PYTHON_DOWNLOADS is left alone, so a
    # workstation that forbids downloads still fails loudly rather than having
    # the gate fetch an interpreter behind that choice.
    environment.pop("UV_PYTHON_PREFERENCE", None)

    command = [
        uv,
        "run",
        "--managed-python",
        "--python",
        "3.11",
        "--no-project",
        "python",
        str(Path(__file__).resolve()),
        *argv,
    ]
    execvpe(uv, command, environment)
    raise RuntimeError("uv re-exec unexpectedly returned")


def gate_environment(
    environ: dict[str, str],
    *,
    executable: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Build the environment shared by every gate child command."""
    environment = dict(environ)
    configured = environment.get("PYO3_PYTHON")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.is_file():
            raise ValueError(
                "PYO3_PYTHON is set, but its configured interpreter does "
                f"not exist at {candidate}. The explicit setting will not "
                "be replaced. Run `uv python install 3.11`, then set "
                "`PYO3_PYTHON=\"$(uv python find 3.11)\"`, or unset it so "
                "gate.py can use its uv-selected interpreter."
            )
        validation_error = _python_validation_error(candidate)
        if validation_error is not None:
            raise ValueError(
                "PYO3_PYTHON is set, but its configured interpreter at "
                f"{candidate} is not a usable Python 3.11+ interpreter: "
                f"{validation_error}. The explicit setting will not be "
                "replaced. Run `uv python install 3.11`, then set "
                "`PYO3_PYTHON=\"$(uv python find 3.11)\"`, or unset it so "
                "gate.py can use its uv-selected interpreter."
            )
        environment["PYO3_PYTHON"] = str(candidate)
    else:
        if not executable.is_file():
            raise ValueError(
                "gate.py selected a managed Python interpreter that does "
                f"not exist at {executable}; run `uv python install 3.11` "
                "and retry."
            )
        validation_error = _python_validation_error(executable)
        if validation_error is not None:
            raise ValueError(
                "gate.py selected an interpreter at "
                f"{executable} that is not a usable Python 3.11+ "
                f"interpreter: {validation_error}. Run "
                "`uv python install 3.11` and retry."
            )
        environment["PYO3_PYTHON"] = str(executable)

    root = repo_root.resolve()
    configured_target = environment.get("CARGO_TARGET_DIR", "target")
    target = Path(configured_target)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(
            "CARGO_TARGET_DIR resolves outside the current worktree "
            f"({target}); unset it to use {root / 'target'} or choose a "
            "directory inside this worktree so concurrent agents cannot "
            "share Cargo build state."
        )
    environment["CARGO_TARGET_DIR"] = str(target)

    # An explicit handoff is diagnosed here, before the first command runs,
    # not when the oracle finally reaches it (chelis#1322). The oracle
    # rejects a bad value on its own, but that is command 10 of 10 in
    # `--local`: a typo would cost the whole workspace clippy, fmt, the
    # lint pass, three rustdoc stages and two Python guards before saying
    # so. PYO3_PYTHON above is diagnosed at command 0 of 10, and the docs
    # claim the two get the same discipline, so they now do. Only a value
    # the CALLER set is checked; the path `run_commands` computes for
    # itself names a binary an earlier command has yet to build.
    if ORACLE_BINARY_ENV in environment:
        error = _oracle_binary_validation_error(
            environment[ORACLE_BINARY_ENV], root
        )
        if error is not None:
            raise ValueError(
                f"{ORACLE_BINARY_ENV} is set, but {error}. The explicit "
                "setting will not be replaced. Point it at a `chelis` "
                "binary, or unset it entirely so the gate hands over the "
                "one its own commands build."
            )

    # cargo-husky's build script can write into the clone's shared .git
    # directory. The checked-in hook remains directly runnable; gate builds
    # must not mutate shared Git state behind sibling worktrees.
    environment["CARGO_HUSKY_DONT_INSTALL_HOOKS"] = "1"
    return environment


def _oracle_binary_validation_error(
    configured: str, repo_root: Path
) -> str | None:
    """Return why an explicit handoff value is unusable, or None.

    Mirrors `handed_over_binary` in
    `scripts/unrepresentable_domain_oracle.py`, including its rule that an
    empty value is a failure rather than an off switch. Both sides must
    agree on what "set" means: if the gate read an empty value as absent
    while the oracle read it as a handoff (or the reverse), an ambient
    `export CHELIS_ORACLE_BINARY=` would disable the handoff with no
    notice anywhere.
    """
    stripped = configured.strip()
    if not stripped:
        return (
            "its value is empty. An empty handoff is not an off switch: it "
            "would silently disable the handoff instead of naming a binary"
        )
    candidate = Path(stripped)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if candidate.is_dir():
        return f"{candidate} is a directory, not a `chelis` binary"
    if not candidate.is_file():
        return f"no file exists at {candidate}"
    if not os.access(candidate, os.X_OK):
        return f"{candidate} is not executable"
    return None


def _python_validation_error(candidate: Path) -> str | None:
    """Return why a path is not an executable Python 3.11+ interpreter."""
    try:
        result = subprocess.run(
            [str(candidate), "-c", PYTHON_VERSION_PROBE],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except OSError as exc:
        return str(exc)
    except subprocess.TimeoutExpired:
        return "the Python version probe timed out after 15 seconds"

    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip()
    if detail:
        return f"the Python version probe exited {result.returncode}: {detail}"
    return f"the Python version probe exited {result.returncode}"


def materialize_command(command: list[str], python: Path) -> list[str]:
    """Replace the canonical managed-Python marker for execution."""
    return [str(python) if part == MANAGED_PYTHON else part for part in command]


def oracle_binary_handoff(
    commands: list[list[str]], target_dir: str
) -> str | None:
    """The `chelis` this command list builds before it runs the oracle.

    Returns the path to hand over in `ORACLE_BINARY_ENV`, or None when this
    list does not run the chelis#908 oracle, or runs it without building
    `chelis` first. `target_dir` is the already-normalized
    `CARGO_TARGET_DIR` from `gate_environment`, so the handoff points into
    the same worktree-local target the child commands write to.

    Deciding this statically from the list, rather than probing the
    filesystem for a binary, is what keeps the handoff honest: the oracle
    treats the variable as authoritative and fails loudly on a bad path, so
    the gate may only set it where the list itself guarantees the build.
    That is also why `gate.py integration` on its own hands over nothing;
    hosted CI's support-only workspace-shard slice keeps the oracle's original
    build-it-yourself behavior.
    """
    try:
        oracle_index = commands.index(UNREPRESENTABLE_DOMAIN_ORACLE)
    except ValueError:
        return None
    for command in commands[:oracle_index]:
        if tuple(command) in CHELIS_BINARY_PRODUCERS:
            return str(Path(target_dir) / "debug" / "chelis")
    return None


def full_command_list() -> list[list[str]]:
    """The complete developer gate: every stage, in stage order.

    CI runs the same lint-and-unit list, but its integration stage uses the
    split `ci` profile and delegates two census binaries to the required dtype
    oracle. The developer command substitutes the default profile so those
    tests stay present without requiring a hosted-only parallel job. Every
    other stage member is taken verbatim, so a command added to any stage
    appears in `--list` without a second edit here.
    """
    commands: list[list[str]] = []
    for stage in STAGE_ORDER:
        for command in STAGES[stage]:
            commands.append(
                NEXTEST_WORKSPACE if command == NEXTEST_WORKSPACE_CI else command
            )
    return commands


def render(command: list[str]) -> str:
    return " ".join(command)


def list_annotation(command: list[str]) -> str:
    """The `--list` annotation for a canonical command: whether the
    `--fast` pre-push gate and the optional `--local` subset
    include it, only `--local` does, or CI owns it."""
    if command in FAST_STATIC_COMMANDS and command in LOCAL_STATIC_COMMANDS:
        return FAST_ANNOTATION
    if command in LOCAL_STATIC_COMMANDS:
        return LOCAL_ANNOTATION
    if command == NEXTEST_WORKSPACE:
        return FULL_GATE_SPLIT_ANNOTATION
    return CI_OWNED_ANNOTATION


def workspace_member_packages(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Map each workspace member directory (repo-relative, as written in
    the root `Cargo.toml` members list) to its `[package].name` from the
    member's own `Cargo.toml`. The directory name is NOT assumed to be
    the package name; `cargo nextest run -p` needs the package name."""
    try:
        import tomllib
    except ModuleNotFoundError:
        # `tomllib` is stdlib only from Python 3.11. The repo policy is to
        # run gate.py via the uv-managed `.venv/bin/python` (>=3.11); the
        # macOS system `python3` is 3.9. Surface that as guidance, not a
        # raw traceback (chelis#366).
        print(
            "gate.py --local needs Python 3.11+ (tomllib); invoke "
            "`python3 scripts/gate.py --local` so gate.py can route "
            "through uv per AGENTS.md",
            file=sys.stderr,
        )
        raise SystemExit(1)
    manifest = tomllib.loads((repo_root / "Cargo.toml").read_text())
    members: list[str] = manifest["workspace"]["members"]
    packages: dict[str, str] = {}
    for entry in members:
        if any(ch in entry for ch in "*?["):
            paths = sorted(p for p in repo_root.glob(entry) if p.is_dir())
        else:
            paths = [repo_root / entry]
        for path in paths:
            member_manifest = tomllib.loads((path / "Cargo.toml").read_text())
            rel = path.relative_to(repo_root).as_posix()
            packages[rel] = member_manifest["package"]["name"]
    return packages


def changed_paths_from_git(diff_output: str, status_output: str) -> list[str]:
    """Extract repo-relative changed paths from `git diff --name-only`
    output (committed work vs origin/main) plus `git status --porcelain`
    output (uncommitted and untracked work). Porcelain rename lines
    (`R  old -> new`) contribute both sides."""
    paths: list[str] = []
    for line in diff_output.splitlines():
        # git quotes paths containing non-ASCII/quote/backslash bytes
        # (core.quotePath); strip the quotes so the member-dir prefix
        # match still sees the path. Mirrors the porcelain branch below;
        # without this a committed-only change to such a file silently
        # excludes its crate from the --local nextest stage (PR #362
        # review finding 1).
        line = line.strip().strip('"')
        if line:
            paths.append(line)
    for line in status_output.splitlines():
        if len(line) < 4:
            continue
        # Porcelain v1: two status characters, a space, then the path
        # (or `orig -> dest` for renames/copies).
        body = line[3:]
        for part in body.split(" -> "):
            part = part.strip().strip('"')
            if part:
                paths.append(part)
    return paths


def changed_crates(
    paths: list[str], member_packages: dict[str, str]
) -> list[str]:
    """The sorted, deduplicated package names of workspace members that
    own at least one of `paths`. Paths outside every member directory
    (scripts/, docs/, spec/, ...) map to no crate."""
    found: set[str] = set()
    for path in paths:
        for member_dir, package in member_packages.items():
            if path == member_dir or path.startswith(member_dir + "/"):
                found.add(package)
    return sorted(found)


def local_command_list(crates: list[str]) -> list[list[str]]:
    """The optional `--local` command list: the static subset
    plus one `cargo nextest run -p <crate> --no-fail-fast` per changed crate."""
    commands = list(LOCAL_STATIC_COMMANDS)
    for crate in crates:
        commands.append(
            ["cargo", "nextest", "run", "-p", crate, "--no-fail-fast"]
        )
    return commands


def std_paths_changed(paths: list[str]) -> bool:
    """Whether any changed path lives under a chelis-std prefix, which is
    what makes the bundle's embedded bytes stale."""
    return any(
        path.startswith(prefix) for path in paths for prefix in STD_PATH_PREFIXES
    )


def fast_command_list(
    crates: list[str], *, std_changed: bool, changed_paths: list[str]
) -> list[list[str]]:
    """The `--fast` command list: fix-in-place regeneration and fmt, the
    changed-path classification, the lint row (which also builds `chelis`),
    `cargo clippy -p <crate> --tests` per changed crate, one nextest run over
    the drift tripwires, and, when a std path changed, the tier-1
    regeneration before the checks and the bundle self-consistency test after
    them. Every writer precedes every check: a changed `.ch` source makes the
    embedded bundle stale, and the `bundled_chelis_std_loader` tripwire would
    fail on it before a later regeneration could fix it.

    The classification is first among the checks because it is the cheapest
    thing here that can reject a push, and because until chelis#2250 an
    unrouted new file passed every local check and then failed `Plan Changed
    Integration Tests` in CI. It derives its own set when it runs rather than
    taking one built here; `AGENTS.md`'s `--fast` paragraph states once what
    that set covers and what it does not, and this docstring deliberately
    does not restate it.
    `docs/ci_validation.md` under Measured figures carries its cost with the
    conditions that produced it.

    Never the chelis#908 oracle: minutes of work whose `chelis check`
    timeouts under load are a known false-red source."""
    commands = [REGEN_TIER0_WRITE]
    if std_changed:
        commands.append(REGEN_TIER1_WRITE)
    commands.append(FMT_WRITE)
    commands.append(classify_paths_command(changed_paths))
    commands.append(CHELIS_LINT_CHECK)
    for crate in crates:
        commands.append(
            ["cargo", "clippy", "-p", crate, "--tests", "--", "-D", "warnings"]
        )
    commands.append(FAST_TRIPWIRE_NEXTEST)
    if std_changed:
        commands.append(STD_BUNDLE_SELF_CONSISTENCY)
    return commands


def _git_output(args: list[str]) -> str:
    """Run a git query from the repo root and return stdout. Raises
    `subprocess.CalledProcessError` (with stderr captured) on failure,
    e.g. when `origin/main` does not exist locally."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_facts() -> dict[str, str | None]:
    """The commit facts the run summary records. Never fatal: a shallow CI
    clone has no `origin/main`, and the `--local` and `--fast` paths keep
    their own loud failure for the diff they actually depend on."""
    facts: dict[str, str | None] = {}
    for key, args in (
        ("head", ["rev-parse", "HEAD"]),
        ("origin_main", ["rev-parse", "origin/main"]),
        ("merge_base", ["merge-base", "origin/main", "HEAD"]),
    ):
        try:
            facts[key] = _git_output(args).strip() or None
        except (subprocess.CalledProcessError, OSError):
            facts[key] = None
    return facts


def _porcelain_hashes(repo_root: Path = REPO_ROOT) -> dict[str, str | None]:
    """A content hash for every path `git status --porcelain` names, untracked
    files included one by one. Hashing the whole dirty set, not just its
    membership, is what lets `--fast` report a file that was already modified
    and that `cargo fmt` then changed further; a bare porcelain diff cannot see
    that. A path that no longer exists (deleted, or a rename source) hashes to
    None."""
    status_output = _git_output(["status", "--porcelain", "--untracked-files=all"])
    hashes: dict[str, str | None] = {}
    for path in changed_paths_from_git("", status_output):
        candidate = repo_root / path
        try:
            hashes[path] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except (OSError, IsADirectoryError):
            hashes[path] = None
    return hashes


def files_changed_between(
    before: dict[str, str | None], after: dict[str, str | None]
) -> list[str]:
    """Paths whose content hash differs between two porcelain snapshots,
    including paths that only appear in the second one."""
    changed = {path for path in after if path not in before}
    changed |= {path for path in before if after.get(path) != before[path]}
    return sorted(changed)


# --- per-run records ---------------------------------------------------------


# Plain classes, not dataclasses: `scripts/test_gate.py` execs this file's
# source under a module name that is not registered in `sys.modules`, and
# `dataclasses` resolves the string annotations `from __future__ import
# annotations` produces through `sys.modules[cls.__module__]`, which would
# make that harness fail on import.
class StageRecord:
    """One command's outcome inside a gate run."""

    def __init__(
        self,
        index: int,
        command: str,
        seconds: float,
        returncode: int,
        *,
        launch_error: str | None = None,
        transcript: str | None = None,
    ) -> None:
        self.index = index
        self.command = command
        self.seconds = seconds
        self.returncode = returncode
        self.launch_error = launch_error
        self.transcript = transcript


class GateReport:
    """Everything one gate run knows about itself, serialized by
    `write_summary` for pass and fail alike. Termination classes: pass,
    stage-failure, signal, environment, preflight-stop, lease-timeout,
    user-cancel, internal-error."""

    def __init__(self, mode: str, started_at: str) -> None:
        self.mode = mode
        self.started_at = started_at
        self.stages: list[StageRecord] = []
        self.termination = "pass"
        self.exit_code = 0
        self.preflight: dict = {}
        self.lease: dict = {}
        self.git: dict = {}
        self.files_changed_by_run: list[str] | None = []
        self.files_changed_note: str | None = None
        self.selected_python: str | None = None
        self.started_monotonic = time.monotonic()

    def first_failing_stage(self) -> dict | None:
        if self.termination not in ("stage-failure", "signal") or not self.stages:
            return None
        last = self.stages[-1]
        return {
            "index": last.index,
            "command": last.command,
            "returncode": last.returncode,
            "launch_error": last.launch_error,
            "transcript": last.transcript,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def report_directory(environ: dict[str, str], repo_root: Path = REPO_ROOT) -> Path:
    configured = environ.get(REPORT_DIR_ENV)
    if configured:
        directory = Path(configured)
        return directory if directory.is_absolute() else repo_root / directory
    return repo_root / "target" / "gate-reports"


def summary_payload(
    report: GateReport, *, ended: datetime, path: Path, repo_root: Path
) -> dict:
    try:
        report_path = str(path.relative_to(repo_root))
    except ValueError:
        report_path = str(path)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": report.mode,
        "started_at": report.started_at,
        "ended_at": _iso(ended),
        "seconds": round(time.monotonic() - report.started_monotonic, 3),
        "termination": report.termination,
        "exit_code": report.exit_code,
        "worktree": str(repo_root),
        # `python` is the interpreter every child ran (the normalized
        # PYO3_PYTHON); `runner_python` is the one that ran the gate itself.
        "python": report.selected_python,
        "runner_python": sys.executable,
        "git": report.git,
        "preflight": report.preflight,
        "lease": report.lease,
        "stages": [
            {
                "index": stage.index,
                "command": stage.command,
                "seconds": round(stage.seconds, 3),
                "returncode": stage.returncode,
                "launch_error": stage.launch_error,
            }
            for stage in report.stages
        ],
        "first_failing_stage": report.first_failing_stage(),
        "files_changed_by_run": report.files_changed_by_run,
        "files_changed_note": report.files_changed_note,
        "report_path": report_path,
    }


def write_summary(
    report: GateReport,
    *,
    environ: dict[str, str],
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Write `<report dir>/<utc>-<pid>-<mode>.json` and return its path.
    `target/` is gitignored at any depth, so the default needs no ignore
    edit."""
    directory = report_directory(environ, repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    ended = _utc_now()
    stamp = ended.strftime("%Y%m%dT%H%M%S.%fZ")
    mode = re.sub(r"[^A-Za-z0-9_.-]+", "-", report.mode).strip("-")
    path = directory / f"{stamp}-{os.getpid()}-{mode}.json"
    payload = summary_payload(report, ended=ended, path=path, repo_root=repo_root)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _mode_label(mode: str) -> str:
    if mode in ("fast", "local"):
        return f"gate --{mode}"
    if mode == "full":
        return "gate"
    return f"gate {mode}"


def human_summary(
    report: GateReport, path: Path | None, repo_root: Path = REPO_ROOT
) -> str:
    """One line: stage count, seconds, verdict, changed files, report path."""
    seconds = time.monotonic() - report.started_monotonic
    if report.termination == "pass":
        verdict = "PASS"
    elif report.termination in ("stage-failure", "signal"):
        failing = report.first_failing_stage() or {}
        detail = failing.get("command", "?")
        if failing.get("launch_error"):
            code = f"launch error: {failing['launch_error']}"
        else:
            code = describe_returncode(int(failing.get("returncode", 0)))
        verdict = (
            f"FAIL (stage {failing.get('index', '?')}: {detail}; {code})"
        )
    else:
        verdict = f"{report.termination.upper()} (exit {report.exit_code})"
    parts = [
        f"{_mode_label(report.mode)}: {len(report.stages)} stages, "
        f"{seconds:.1f} s, {verdict}"
    ]
    if report.mode == "fast":
        changed = report.files_changed_by_run
        if changed is None:
            parts.append("changed: unknown (git status failed after the run)")
        elif changed:
            shown = ", ".join(changed[:5])
            if len(changed) > 5:
                shown += f", and {len(changed) - 5} more"
            parts.append(f"changed: {len(changed)} file(s) ({shown})")
        else:
            parts.append("changed: 0 files")
    if path is not None:
        try:
            shown_path = str(path.relative_to(repo_root))
        except ValueError:
            shown_path = str(path)
        parts.append(f"report: {shown_path}")
    return "; ".join(parts)



# --- detached runs (chelis#1568) ----------------------------------------------


def detach_directory(
    environ: dict[str, str], repo_root: Path = REPO_ROOT
) -> Path:
    """Where handles and combined logs live: `<report dir>/detached`."""
    return report_directory(environ, repo_root) / DETACH_SUBDIR


def detach_child_argv(argv: list[str]) -> list[str]:
    """The caller's argv with `--detach` removed.

    Everything else passes through untouched, `--no-wait`, `--no-lease` and
    `--lease-timeout` included: the child is the run, so it owns the lease
    behaviour the caller asked for.
    """
    return [arg for arg in argv if arg != "--detach"]


def write_handle(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_handle(path: Path) -> dict:
    """Raises `OSError` or `ValueError` for a missing or malformed handle.

    `started_at` is validated here, not only `pid`, because `find_summary`
    proves a summary belongs to this run by comparing against it. A handle
    without a usable start instant would make that filter degrade OPEN, which
    is a guard that stops guarding under exactly the conditions it exists for,
    so the handle is rejected at the boundary instead.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pid" not in payload:
        raise ValueError(f"{path} is not a gate detach handle")
    if _parse_iso(payload.get("started_at") or "") is None:
        raise ValueError(
            f"{path} has no usable started_at, so a run summary could not be "
            "proven to belong to it"
        )
    return payload


def newest_handle(directory: Path) -> Path | None:
    try:
        handles = sorted(
            item for item in directory.glob("*.json") if item.is_file()
        )
    except OSError:
        return None
    return handles[-1] if handles else None


def _parse_iso(text: str) -> datetime | None:
    """Parse an `_iso` timestamp. `_iso` emits milliseconds, but accept the
    second-resolution spelling too so a handle written by any version reads."""
    for shape in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, shape)
        except (ValueError, TypeError):
            continue
    return None


def _summary_stamp(path: Path) -> datetime | None:
    """The UTC instant `write_summary` encoded in a summary's file name, which
    is when that run ENDED."""
    stamp = path.name.split("-", 1)[0]
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%S.%f%z")
    except ValueError:
        return None


def find_summary(
    report_dir: Path, pid: int, mode: str, *, not_before: str | None = None
) -> Path | None:
    """The run summary the detached child wrote, if it has finished.

    `write_summary` names its file `<stamp>-<pid>-<mode>.json`, so the child's
    own pid is what correlates the handle with the summary. That is why the
    spawn must not re-execute through uv: a grandchild would write a summary
    this glob could never find.

    A pid is not unique over time, so the pid alone is not enough, and it fails
    in BOTH directions. An older run in the same report directory that happened
    to get this pid is the only match for the whole window before this run
    finishes; a later run that reused the pid outranks this run's own summary
    once it exists. Either way polling reports someone else's verdict, which is
    a false PASS on the optional local gate.

    `not_before` is the handle's `started_at`, and the rule is: take the
    EARLIEST summary that ended at or after this run started. That one is
    provably this run's, because no other process can hold this pid between
    this run's start and its exit, so any impostor's summary must end later
    than this run's own. "Newest wins" cannot make that argument, and closing
    only the older direction leaves the newer one open.

    Without a usable floor no candidate can be proven to belong to this run, so
    none is returned. `read_handle` rejects a handle with no usable
    `started_at` for the same reason: a filter that degrades open is worse
    than no filter, because it looks like a guard.

    Returning the first hit in `sorted()` order assumes lexicographic order
    equals chronological order. That holds by construction, because
    `write_summary` stamps a fixed-width zero-padded UTC timestamp. It is
    written down here because an assumption that holds by construction is
    exactly the kind that breaks silently when someone changes the
    construction, and the construction lives in another function.

    The correlation is a pid plus a time window, which is not an identity, and
    two consequences follow. The floor is sampled before the spawn, so it is
    fractionally earlier than the child's true start, and a run that both
    ended and freed this pid inside that sub-millisecond window would still be
    accepted. And a run killed before it writes a summary, followed by a pid
    reuse, still returns the impostor. Both close the same way and neither is
    urgent, because this fails toward still-running rather than toward a false
    pass; chelis#1584 owns them.
    """
    floor = _parse_iso(not_before) if not_before else None
    if floor is None:
        return None
    safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "-", mode).strip("-")
    try:
        candidates = sorted(report_dir.glob(f"*-{pid}-{safe_mode}.json"))
    except OSError:
        return None
    for candidate in candidates:
        ended = _summary_stamp(candidate)
        # A name whose stamp cannot be read cannot be proven to be this run's,
        # and `write_summary` always produces a readable one, so it is not
        # something this gate wrote. `SummaryNamingTests` locks the writer and
        # this reader together so the format cannot drift apart silently.
        if ended is not None and ended >= floor:
            return candidate
    return None


def _pid_alive(pid: int) -> bool:
    """Whether a process with this pid exists. Fallible under pid reuse, the
    same caveat `reap_orphans.py` documents for `ppid == 1`, which is why the
    lease check below is preferred when it applies."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def spawn_detached(
    child_argv: list[str],
    *,
    environ: dict[str, str],
    executable: Path,
    directory: Path,
    mode: str,
    repo_root: Path = REPO_ROOT,
    now=_utc_now,
    popen=None,
    git_facts=None,
) -> dict:
    """Start the run in its own session and return the handle payload.

    `start_new_session=True` detaches the child from the caller's terminal and
    process group, so a foreground command timeout in an agent harness cannot
    take the run with it. That is the whole point.

    The obvious worry is that this makes the run reapable, since it then has
    `ppid == 1` and `scripts/reap_orphans.py --kill` calls that orphaned. It
    does not, for two measured reasons: the interpreter is not in that
    script's `BUILD_TOOL_NAMES` and does not live under `target/`, so the gate
    process is never matched at all; and the cargo, rustc and nextest children
    keep the live gate as their parent, so they are never classified orphaned
    either. Both properties are locked by `ReaperSafetyTests` in
    `scripts/test_gate_detach.py`, together with the negative twin: once the
    gate itself dies its children ARE orphans and the reaper still reaps them,
    which is what it is for. No protection list is needed, and adding one
    would risk shielding a genuinely abandoned build.
    """
    directory.mkdir(parents=True, exist_ok=True)
    started = now()
    stamp = started.strftime("%Y%m%dT%H%M%S.%fZ")
    base = f"{stamp}-{os.getpid()}"
    log_path = directory / f"{base}.log"
    handle_path = directory / f"{base}.json"
    facts = _git_facts() if git_facts is None else git_facts()
    spawn = subprocess.Popen if popen is None else popen
    script = str(Path(__file__).resolve())
    argv = [str(executable), script, *child_argv]
    log = open(log_path, "wb")
    try:
        child = spawn(
            argv,
            cwd=str(repo_root),
            env=environ,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    return {
        "schema_version": DETACH_SCHEMA_VERSION,
        "pid": child.pid,
        "mode": mode,
        "argv": argv,
        "log": str(log_path),
        "handle": str(handle_path),
        "report_dir": str(report_directory(environ, repo_root)),
        "lease_path": str(lease_dir(environ) / LEASE_FILE_NAME),
        "worktree": str(repo_root),
        "head": facts.get("head"),
        "started_at": _iso(started),
    }


def detached_state(
    handle: dict,
    *,
    alive=_pid_alive,
    peek=None,
    find=find_summary,
) -> dict:
    """Whether the detached run has finished, is running, or died.

    The order matters and avoids a pid-reuse false positive on the finished
    path. A matching summary is proof the run ENDED, so it is checked first.
    The lease is checked next because it is kernel-backed: a sidecar naming
    this pid, over a lock that is actually held, proves the run is alive. Only
    then does it fall back to `kill(pid, 0)`, which pid reuse can fool.
    """
    pid = int(handle["pid"])
    report_dir = Path(handle["report_dir"])
    summary_path = find(
        report_dir,
        pid,
        handle.get("mode", "local"),
        not_before=handle.get("started_at"),
    )
    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {
                "state": "finished",
                "summary_path": str(summary_path),
                "summary": None,
                "error": f"could not read the run summary: {exc}",
            }
        return {
            "state": "finished",
            "summary_path": str(summary_path),
            "summary": summary,
            "error": None,
        }
    probe = GateLease.peek if peek is None else peek
    holds_lease = False
    lease_path = handle.get("lease_path")
    if lease_path:
        try:
            holder = probe(Path(lease_path))
        except OSError:
            holder = None
        if holder and holder.get("pid") == pid:
            holds_lease = True
    if holds_lease:
        return {"state": "running", "evidence": "holds the gate lease", "error": None}
    if alive(pid):
        return {"state": "running", "evidence": "the process is alive", "error": None}
    return {"state": "died", "evidence": None, "error": None}


def run_detach(
    args: argparse.Namespace,
    *,
    argv: list[str],
    environ: dict[str, str],
    executable: Path,
    mode: str,
    repo_root: Path = REPO_ROOT,
    output_stream=None,
    error_stream=None,
    spawn=spawn_detached,
) -> int:
    """Launch the run detached and print its handle. The launcher's exit code
    is a LAUNCH verdict, never a gate verdict: `--status` is what restores the
    documented foreground exit contract."""
    output = sys.stdout if output_stream is None else output_stream
    error = sys.stderr if error_stream is None else error_stream
    directory = detach_directory(environ, repo_root)
    try:
        payload = spawn(
            detach_child_argv(argv),
            environ=environ,
            executable=executable,
            directory=directory,
            mode=mode,
            repo_root=repo_root,
        )
    except OSError as exc:
        print(f"gate: could not start a detached run: {exc}", file=error)
        return EXIT_ENVIRONMENT
    try:
        handle_path = write_handle(payload, Path(payload["handle"]))
    except OSError as exc:
        # The child is already running and detached. Without the handle it
        # cannot be found by `--status`, so name it here rather than exiting
        # with a bare "could not start" that is not even true.
        print(
            f"gate: the detached run started as pid {payload['pid']} but its "
            f"handle could not be written ({exc}); its log is "
            f"{payload['log']} and it holds the lease at {payload['lease_path']}",
            file=error,
        )
        return EXIT_ENVIRONMENT

    def shown(path: str) -> str:
        try:
            return str(Path(path).relative_to(repo_root))
        except ValueError:
            return path

    print(
        f"gate: detached {_mode_label(mode)} run started, pid {payload['pid']}",
        file=output,
    )
    print(f"gate: log:    {shown(payload['log'])}", file=output)
    print(f"gate: handle: {shown(str(handle_path))}", file=output)
    print(
        f"gate: status: python3 scripts/gate.py --status {shown(str(handle_path))}",
        file=output,
        flush=True,
    )
    return 0


def run_status(
    args: argparse.Namespace,
    *,
    environ: dict[str, str],
    repo_root: Path = REPO_ROOT,
    output_stream=None,
    error_stream=None,
    state_of=detached_state,
) -> int:
    """Report a detached run's verdict, and exit with it once it has one."""
    output = sys.stdout if output_stream is None else output_stream
    error = sys.stderr if error_stream is None else error_stream
    selector = args.status
    if selector == "latest":
        path = newest_handle(detach_directory(environ, repo_root))
        if path is None:
            print(
                "gate: no detached run handle found under "
                f"{detach_directory(environ, repo_root)}",
                file=error,
            )
            return EXIT_ENVIRONMENT
    else:
        path = Path(selector)
    try:
        handle = read_handle(path)
    except (OSError, ValueError) as exc:
        print(f"gate: cannot read the detach handle {path}: {exc}", file=error)
        return EXIT_ENVIRONMENT
    result = state_of(handle)
    if result["state"] == "running":
        print(
            f"gate: detached {_mode_label(handle.get('mode', '?'))} run "
            f"pid {handle['pid']} is still running ({result['evidence']}); "
            f"started {handle.get('started_at', '?')}",
            file=output,
        )
        print(f"gate: log: {handle.get('log', '?')}", file=output, flush=True)
        return EXIT_STILL_RUNNING
    if result["state"] == "died":
        print(
            f"gate: detached run pid {handle['pid']} is gone and wrote no run "
            f"summary; see {handle.get('log', '?')}",
            file=error,
            flush=True,
        )
        return 1
    summary = result.get("summary")
    if summary is None:
        print(f"gate: {result.get('error')}", file=error, flush=True)
        return EXIT_ENVIRONMENT
    exit_code = int(summary.get("exit_code", 1))
    termination = summary.get("termination", "?")
    print(
        f"gate: detached {_mode_label(summary.get('mode', '?'))} run finished: "
        f"{str(termination).upper()}, exit {exit_code}, "
        f"{summary.get('seconds', '?')} s; report {result['summary_path']}",
        file=output,
    )
    failing = summary.get("first_failing_stage")
    if failing:
        print(
            f"gate: first failing stage {failing.get('index', '?')}: "
            f"{failing.get('command', '?')}",
            file=output,
        )
        transcript = failing.get("transcript")
        if transcript:
            print(f"gate: transcript: {transcript}", file=output)
    if termination == "lease-timeout":
        print(
            "gate: exit 4 is the lease timeout, not a gate failure; another "
            "gate held the lease or was ahead in its queue",
            file=output,
        )
    print(f"gate: log: {handle.get('log', '?')}", file=output, flush=True)
    return exit_code

# --- preflight ---------------------------------------------------------------


def host_system() -> str:
    """The platform the preflight branches on, resolved at call time so tests
    (and the Linux script-unit CI job) can steer it."""
    return platform.system()


def run_probe(
    python: Path,
    repo_root: Path = REPO_ROOT,
    *,
    probe_runner=subprocess.run,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> dict:
    """Run `scripts/preflight_exec_probe.py` as a subprocess and classify its
    exit code. It is a subprocess, not an import, so the bootstrap-free
    carve-out in `scripts/test_bootstrapless_scripts.py` stays at two
    scripts. 0 -> ok; 1 -> wedged; 3 -> slow; 2 or anything else, a timeout,
    or a launch failure -> could-not-run."""
    command = [str(python), "scripts/preflight_exec_probe.py"]
    try:
        completed = probe_runner(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "verdict": "could-not-run",
            "exit_code": None,
            "output": f"the probe timed out after {timeout:.0f} s",
        }
    except OSError as exc:
        return {"verdict": "could-not-run", "exit_code": None, "output": str(exc)}
    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    verdict = {0: "ok", 1: "wedged", 2: "could-not-run", 3: "slow"}.get(
        completed.returncode, "could-not-run"
    )
    return {
        "verdict": verdict,
        "exit_code": completed.returncode,
        "output": text.splitlines()[0] if text else "",
        "full_output": text,
    }


def run_preflight(
    *,
    mode: str,
    report: GateReport,
    environ: dict[str, str],
    executable: Path,
    repo_root: Path = REPO_ROOT,
    system=None,
    probe=None,
    output_stream=None,
    error_stream=None,
) -> tuple[int | None, dict[str, str] | None]:
    """The checks before the first command of `--fast`, `--local`, or the
    bare full gate. Returns `(None, environment)` to proceed, or
    `(exit_code, None)` to stop. CI stage runs never call this. Everything
    the preflight says is a diagnostic, so it all goes to `error_stream`;
    `output_stream` is accepted for symmetry with the other runners."""
    del output_stream
    error = sys.stderr if error_stream is None else error_stream
    try:
        environment = gate_environment(
            dict(environ), executable=executable, repo_root=repo_root
        )
    except ValueError as exc:
        print(f"gate: environment setup failed: {exc}", file=error)
        report.termination = "environment"
        return EXIT_ENVIRONMENT, None
    report.selected_python = environment["PYO3_PYTHON"]

    report.git.update(_git_facts())

    venv_python = repo_root / ".venv" / "bin" / "python"
    venv_present = venv_python.is_file()
    report.preflight["venv_present"] = venv_present
    if not venv_present:
        print(
            f"gate: warning: {venv_python} is missing. The gate exports "
            "PYO3_PYTHON so its own commands do not need it; direct cargo and "
            "nextest runs in this worktree do. Create it with: uv venv "
            "--python 3.11",
            file=error,
        )

    current_system = (host_system if system is None else system)()
    if current_system == "Darwin":
        result = (run_probe if probe is None else probe)(executable, repo_root)
        report.preflight["probe"] = {
            key: result.get(key) for key in ("verdict", "exit_code", "output")
        }
        verdict = result["verdict"]
        if verdict == "wedged":
            print(
                "gate: preflight stop: first-exec assessment appears wedged "
                f"on this macOS host (probe exit {result['exit_code']}).",
                file=error,
            )
            if result.get("full_output"):
                print(result["full_output"], file=error)
            print(
                f"gate: see {PROBE_RUNBOOK} for the runbook; push and let "
                "macOS Smoke serve as the oracle. Exit 3 means preflight-stop, "
                f"not a {mode} failure.",
                file=error,
            )
            report.termination = "preflight-stop"
            return EXIT_PREFLIGHT_STOP, None
        if verdict == "slow":
            print(
                f"gate: warning: first exec is slow ({result['output']}); "
                f"see {PROBE_RUNBOOK}. Proceeding.",
                file=error,
            )
        elif verdict == "could-not-run":
            print(
                "gate: warning: the first-exec probe could not run "
                f"({result['output'] or 'no output'}); proceeding without a "
                "verdict.",
                file=error,
            )
    else:
        # The first-exec wedge is a macOS assessment behavior (chelis#356);
        # Linux has no equivalent, so the probe is skipped and the summary
        # says why rather than leaving the key absent.
        report.preflight["probe"] = {"verdict": "skipped", "reason": "not darwin"}

    # Every host fact is best effort: a missing value records null.
    host: dict = {"system": current_system}
    try:
        host["platform"] = platform.platform()
    except Exception:  # noqa: BLE001 - platform probing must never fail the gate
        host["platform"] = None
    host["cpu_count"] = os.cpu_count()
    try:
        host["load_average_1m"] = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        host["load_average_1m"] = None
    report.preflight["host"] = host
    return None, environment


# --- the advisory workstation-wide lease -------------------------------------


def lease_dir(environ: dict[str, str]) -> Path:
    """`$CHELIS_GATE_LEASE_DIR`, else `$XDG_CACHE_HOME/chelis`, else
    `~/.cache/chelis`. Deliberately outside every worktree."""
    configured = environ.get(LEASE_DIR_ENV)
    if configured:
        return Path(configured)
    xdg = environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "chelis"


def describe_holder(holder: dict | None) -> str:
    if not holder:
        return "an unknown holder (no readable sidecar)"
    return (
        f"pid {holder.get('pid', '?')} in {holder.get('worktree', '?')} "
        f"(mode {holder.get('mode', '?')}, head {holder.get('head') or '?'}, "
        f"since {holder.get('started_at', '?')})"
    )


class LeaseHeld(Exception):
    """Raised when the lease or queue is busy and the caller declined to keep
    waiting (`--no-wait`, or `--lease-timeout` elapsed)."""

    def __init__(
        self, holder: dict | None, waited: float, queue_position: int | None = None
    ) -> None:
        super().__init__(describe_holder(holder))
        self.holder = holder
        self.waited = waited
        self.queue_position = queue_position


class LeaseQueueError(OSError):
    """Queue bookkeeping failed; proceeding could overtake live waiters."""


class GateLease:
    """A FIFO admission queue ahead of the advisory flock and holder sidecar.

    A short queue lock serializes ticket registration and admission. Order is
    registration order, not process start time. Each waiter holds its ticket's
    flock; an unlocked ticket is abandoned, even after SIGKILL or PID reuse.
    The main lock still owns holder liveness and remains visible to old probes.
    """

    def __init__(
        self,
        path: Path,
        *,
        mode: str,
        worktree: Path,
        head: str | None,
        wait: bool,
        timeout: float | None,
        output=None,
        sleep=time.sleep,
        clock=time.monotonic,
        poll_seconds: float = LEASE_POLL_SECONDS,
        heartbeat_seconds: float = LEASE_HEARTBEAT_SECONDS,
    ) -> None:
        self.path = path
        self.sidecar_path = path.with_name(path.name + ".json")
        self.mode = mode
        self.worktree = worktree
        self.head = head
        self.wait = wait
        self.timeout = timeout
        self.output = sys.stdout if output is None else output
        self._sleep = sleep
        self._clock = clock
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._fd: int | None = None
        self.queue_path = path.with_name(path.name + ".queue")
        self._queue_guard_fd: int | None = None
        self._ticket_path: Path | None = None
        self._ticket_fd: int | None = None
        self.queue_position: int | None = None
        self.held = False
        self.wait_seconds = 0.0
        self.holder_seen: dict | None = None

    @staticmethod
    def current_holder(path: Path) -> dict | None:
        sidecar = path.with_name(path.name + ".json")
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _try_flock(fd: int, operation: int = fcntl.LOCK_EX) -> bool:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return False
            raise
        return True

    @classmethod
    def peek(cls, path: Path) -> dict | None:
        """Who holds the lease right now, without taking it: None when free,
        the sidecar (or an empty dict for an unreadable one) when held. The
        probe is a shared lock, so it never excludes anyone; an exclusive
        acquirer that lands in the same microseconds simply retries (see
        `acquire`)."""
        if not path.is_file():
            return None
        fd = os.open(path, os.O_RDONLY)
        try:
            if cls._try_flock(fd, fcntl.LOCK_SH):
                fcntl.flock(fd, fcntl.LOCK_UN)
                return None
            return cls.current_holder(path) or {}
        finally:
            os.close(fd)

    def acquire(self) -> None:
        # Acquisition can be interrupted before run_local/run_full stores this
        # object in main's cleanup state. Own every descriptor here until then.
        try:
            self._acquire()
        except OSError as exc:
            opened_main = self._fd is not None
            self.release()
            if opened_main:
                raise LeaseQueueError(str(exc)) from exc
            raise
        except BaseException:
            self.release()
            raise
        finally:
            if self._queue_guard_fd is not None:
                os.close(self._queue_guard_fd)
                self._queue_guard_fd = None

    def _try_turn(self) -> bool:
        # Never block on bookkeeping: --no-wait and the overall deadline apply
        # even if a process is stopped while holding this short-lived mutex.
        if not self._try_flock(self._queue_guard_fd):
            return False
        try:
            live: list[Path] = []
            # iterdir surfaces I/O errors; glob can suppress an unreadable
            # directory and would incorrectly describe the queue as empty.
            tickets = (p for p in self.queue_path.iterdir() if p.suffix == ".ticket")
            for ticket in sorted(tickets, key=lambda p: int(p.stem)):
                if ticket == self._ticket_path:
                    live.append(ticket)
                    continue
                try:
                    fd = os.open(ticket, os.O_RDONLY)
                except FileNotFoundError:
                    continue  # A cancelled waiter removed its ticket.
                try:
                    if self._try_flock(fd):
                        ticket.unlink(missing_ok=True)
                    else:
                        live.append(ticket)
                finally:
                    os.close(fd)
            if self._ticket_path is None:
                number = int(live[-1].stem) + 1 if live else 1
                ticket = self.queue_path / f"{number:020d}.ticket"
                self._ticket_fd = os.open(ticket, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
                self._ticket_path = ticket
                # Nobody can inspect this ticket until the queue lock drops.
                if not self._try_flock(self._ticket_fd):
                    raise RuntimeError("new gate queue ticket is already locked")
                live.append(ticket)
            self.queue_position = live.index(self._ticket_path) + 1
            if self.queue_position != 1 or not self._try_flock(self._fd):
                return False
            self._leave_queue()
            return True
        except (OSError, ValueError) as exc:
            raise LeaseQueueError(str(exc)) from exc
        finally:
            fcntl.flock(self._queue_guard_fd, fcntl.LOCK_UN)

    def _leave_queue(self) -> None:
        try:
            if self._ticket_path is not None:
                self._ticket_path.unlink(missing_ok=True)
        except OSError as exc:
            raise LeaseQueueError(str(exc)) from exc
        finally:
            self._ticket_path = None
            if self._ticket_fd is not None:
                os.close(self._ticket_fd)
                self._ticket_fd = None

    def _acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self.queue_path.mkdir(exist_ok=True)
            self._queue_guard_fd = os.open(
                self.path.with_name(self.path.name + ".queue.lock"),
                os.O_RDWR | os.O_CREAT, 0o644,
            )
        except OSError as exc:
            raise LeaseQueueError(str(exc)) from exc
        started = self._clock()
        last_heartbeat = started
        announced = False
        transient_retry = True
        attempted = False
        while True:
            waited = self._clock() - started
            if attempted and self.timeout is not None and waited >= self.timeout:
                self.wait_seconds = waited
                raise LeaseHeld(self.peek(self.path) or None, waited, self.queue_position)
            attempted = True
            if self._try_turn():
                break
            holder = self.peek(self.path)
            if holder:
                self.holder_seen = holder
            elif transient_retry and self.queue_position == 1:
                # No sidecar yet: either a holder is between its flock and
                # its sidecar write, or a `--fast` peek holds a shared lock
                # for a few microseconds. One short retry settles both
                # before this run announces an unknown holder.
                transient_retry = False
                pause = TRANSIENT_RETRY_SECONDS
                if self.timeout is not None:
                    pause = max(0.0, min(pause, self.timeout - waited))
                self._sleep(pause)
                continue
            now = self._clock()
            waited = now - started
            expired = self.timeout is not None and waited >= self.timeout
            if not self.wait or expired:
                self.wait_seconds = waited
                raise LeaseHeld(holder or None, waited, self.queue_position)
            position = (
                f"queue position {self.queue_position}"
                if self.queue_position is not None else "waiting to register a queue ticket"
            )
            holder_text = (
                f"held by {describe_holder(holder)}"
                if holder is not None else "no current holder"
            )
            if not announced:
                print(
                    f"gate: waiting for the gate lease {self.path}; {position}; "
                    f"{holder_text}; polling every "
                    f"{self.poll_seconds:.0f} s (pass --no-wait, "
                    "--lease-timeout SECONDS, or --no-lease to change this)",
                    file=self.output,
                    flush=True,
                )
                announced = True
                last_heartbeat = now
            elif now - last_heartbeat >= self.heartbeat_seconds:
                print(
                    f"gate: still waiting ({waited:.0f} s) for the gate lease "
                    f"{position}; {holder_text}",
                    file=self.output,
                    flush=True,
                )
                last_heartbeat = now
            # Never sleep past the deadline: a bounded wait is honoured to
            # the second, not to the next poll boundary.
            pause = (
                self.poll_seconds if self._ticket_path is not None
                else TRANSIENT_RETRY_SECONDS
            )
            if self.timeout is not None:
                pause = max(0.0, min(pause, self.timeout - waited))
            self._sleep(pause)
        self.wait_seconds = self._clock() - started
        self._write_sidecar()
        self.held = True

    def _write_sidecar(self) -> None:
        # We hold the lock, so any sidecar left behind is stale by definition.
        self.sidecar_path.unlink(missing_ok=True)
        descriptor = os.open(
            self.sidecar_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "worktree": str(self.worktree),
                    "head": self.head,
                    "mode": self.mode,
                    "started_at": _iso(_utc_now()),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def release(self) -> None:
        try:
            if self.held:
                self.sidecar_path.unlink(missing_ok=True)
        finally:
            try:
                self._leave_queue()
            finally:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
                self.held = False
                self.queue_position = None

    def __enter__(self) -> "GateLease":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def take_lease(
    *,
    mode: str,
    args: argparse.Namespace,
    report: GateReport,
    environ: dict[str, str],
    repo_root: Path = REPO_ROOT,
    output_stream=None,
    error_stream=None,
    lease_factory=None,
) -> tuple[int | None, GateLease | None]:
    """Take the lease for `--local` and the bare full gate; only report a
    holder for `--fast`. Returns `(exit_code, None)` when the lease is held
    and the caller declined to wait, else `(None, lease_or_None)`."""
    output = sys.stdout if output_stream is None else output_stream
    error = sys.stderr if error_stream is None else error_stream
    path = lease_dir(environ) / LEASE_FILE_NAME
    record: dict = {
        "mode": "not-taken",
        "path": str(path),
        "wait_seconds": 0.0,
        "holder_seen": None,
    }
    report.lease = record
    if getattr(args, "no_lease", False):
        record["mode"] = "bypassed"
        return None, None
    if mode == "fast":
        try:
            holder = GateLease.peek(path)
        except OSError as exc:
            print(f"gate: note: could not read the gate lease at {path}: {exc}", file=error)
            return None, None
        if holder is not None:
            record["holder_seen"] = holder or None
            print(
                f"gate: note: a full gate is running, {describe_holder(holder or None)}; "
                "expect slower compiles",
                file=output,
                flush=True,
            )
        return None, None
    factory = GateLease if lease_factory is None else lease_factory
    lease = factory(
        path,
        mode=mode,
        worktree=repo_root,
        head=report.git.get("head"),
        wait=not getattr(args, "no_wait", False),
        timeout=getattr(args, "lease_timeout", None),
        output=output,
    )
    try:
        lease.acquire()
    except LeaseQueueError as exc:
        print(
            f"gate: cannot use the gate lease queue at {path}: {exc}. "
            "Repair the queue directory or explicitly pass --no-lease to bypass it.",
            file=error,
        )
        record["mode"] = "error"
        report.termination = "environment"
        return EXIT_ENVIRONMENT, None
    except OSError as exc:
        # An unusable lease directory must not turn into a false red.
        print(
            f"gate: warning: could not take the gate lease at {path} ({exc}); "
            "proceeding without it",
            file=error,
        )
        record["mode"] = "bypassed"
        return None, None
    except LeaseHeld as held:
        record["mode"] = "timed-out"
        record["wait_seconds"] = round(held.waited, 3)
        record["holder_seen"] = held.holder
        reason = (
            "--no-wait was given"
            if getattr(args, "no_wait", False)
            else f"--lease-timeout {args.lease_timeout:g} elapsed"
        )
        position = (
            f"; queue position {held.queue_position}"
            if held.queue_position is not None else ""
        )
        holder_text = (
            f"held by {describe_holder(held.holder)}"
            if held.holder is not None else "no readable holder"
        )
        print(
            f"gate: the gate lease {path} is unavailable ({holder_text}{position}) "
            f"and {reason}. Rerun without the flag to wait, or pass --no-lease "
            "to bypass the lease. Exit 4 means lease-timeout, not a gate failure.",
            file=error,
        )
        report.termination = "lease-timeout"
        return EXIT_LEASE_TIMEOUT, None
    record["mode"] = "held"
    record["wait_seconds"] = round(lease.wait_seconds, 3)
    record["holder_seen"] = lease.holder_seen
    return None, lease


# --- the developer-facing runs ----------------------------------------------


def _derive_changed(label: str, report: GateReport) -> tuple[list[str], list[str]] | int:
    """Derive the changed paths and crates vs origin/main, print the crate
    list (or say explicitly that none were detected), record both in the
    report. Returns an exit code when git fails."""
    try:
        diff_output = _git_output(
            ["diff", "--name-only", "origin/main...HEAD"]
        )
        status_output = _git_output(["status", "--porcelain"])
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(
            f"gate {label}: git failed ({stderr}); cannot derive changed "
            f"crates vs origin/main",
            file=sys.stderr,
        )
        report.termination = "environment"
        return exc.returncode or 1
    paths = changed_paths_from_git(diff_output, status_output)
    crates = changed_crates(paths, workspace_member_packages())
    report.git["dirty"] = bool(status_output.strip())
    report.git["changed_paths"] = sorted(set(paths))
    report.git["selected_crates"] = crates
    if crates:
        print(
            f"gate {label}: changed crates vs origin/main: " + ", ".join(crates),
            flush=True,
        )
    else:
        stage = "nextest" if label == "--local" else "clippy"
        print(
            f"gate {label}: no crate changes detected vs origin/main; "
            f"skipping the per-crate {stage} stage. The workspace suite "
            "is CI-owned and was NOT run.",
            flush=True,
        )
    return paths, crates


def run_local(
    args: argparse.Namespace,
    *,
    report: GateReport,
    environ: dict[str, str],
    executable: Path,
    state: dict,
) -> int:
    """Run the optional `--local` gate: preflight, lease, derive
    the changed crates vs origin/main, then run the local command list."""
    code, environment = run_preflight(
        mode="local", report=report, environ=environ, executable=executable
    )
    if code is not None:
        return code
    assert environment is not None
    derived = _derive_changed("--local", report)
    if isinstance(derived, int):
        return derived
    _paths, crates = derived
    code, lease = take_lease(mode="local", args=args, report=report, environ=environment)
    state["lease"] = lease
    if code is not None:
        return code
    return run_commands(
        local_command_list(crates),
        stage_label="local",
        environ=environment,
        executable=executable,
        report=report,
    )


def run_fast(
    args: argparse.Namespace,
    *,
    report: GateReport,
    environ: dict[str, str],
    executable: Path,
    state: dict,
) -> int:
    """Run the `--fast` pre-push gate: preflight (holder note only), derive
    the changed crates and std paths, fix in place, then check. Exit is
    non-zero only when a stage fails."""
    code, environment = run_preflight(
        mode="fast", report=report, environ=environ, executable=executable
    )
    if code is not None:
        return code
    assert environment is not None
    derived = _derive_changed("--fast", report)
    if isinstance(derived, int):
        return derived
    paths, crates = derived
    std_changed = std_paths_changed(paths)
    report.git["std_changed"] = std_changed
    if std_changed:
        print(
            "gate --fast: chelis-std paths changed; appending the bundle "
            "self-consistency test and regen_all.py --tier 1",
            flush=True,
        )
    take_lease(mode="fast", args=args, report=report, environ=environment)
    state["lease"] = None
    before = _porcelain_hashes()
    code = run_commands(
        fast_command_list(
            crates, std_changed=std_changed, changed_paths=paths
        ),
        stage_label="fast",
        environ=environment,
        executable=executable,
        report=report,
    )
    try:
        after = _porcelain_hashes()
    except (subprocess.CalledProcessError, OSError) as exc:
        # The stages already ran and their verdict stands; only the
        # changed-file report is lost, and the summary says so.
        detail = getattr(exc, "stderr", None) or str(exc)
        report.files_changed_by_run = None
        report.files_changed_note = (
            f"git status failed after the run ({str(detail).strip()}); "
            "review `git status` by hand"
        )
        print(f"gate --fast: warning: {report.files_changed_note}", file=sys.stderr)
        return code
    report.files_changed_by_run = files_changed_between(before, after)
    if report.files_changed_by_run:
        print(
            f"gate --fast: changed {len(report.files_changed_by_run)} file(s) "
            "in place; review and commit them:",
            flush=True,
        )
        for path in report.files_changed_by_run:
            print(f"  {path}", flush=True)
    else:
        print("gate --fast: changed no files.", flush=True)
    return code


def run_full(
    args: argparse.Namespace,
    *,
    report: GateReport,
    environ: dict[str, str],
    executable: Path,
    state: dict,
) -> int:
    """The bare developer gate: preflight, lease, every stage in order."""
    code, environment = run_preflight(
        mode="full", report=report, environ=environ, executable=executable
    )
    if code is not None:
        return code
    assert environment is not None
    code, lease = take_lease(mode="full", args=args, report=report, environ=environment)
    state["lease"] = lease
    if code is not None:
        return code
    return run_commands(
        full_command_list(),
        stage_label="full",
        environ=environment,
        executable=executable,
        report=report,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the per-PR developer-runnable repo gate, or a single CI "
            "stage subset of it."
        ),
    )
    p.add_argument(
        "stage",
        nargs="?",
        choices=list(STAGES),
        help=(
            "Run only this CI stage's gate subset. Omit to run the complete "
            "developer gate."
        ),
    )
    p.add_argument(
        "--tests-only",
        action="store_true",
        help="Run only the integration stage's workspace nextest command.",
    )
    p.add_argument(
        "--support-only",
        action="store_true",
        help="Run only the integration stage's non-nextest support oracles.",
    )
    p.add_argument(
        "--support-slice", choices=("frontend", "domain"),
        help="Run one integration support subset on its existing workspace worker.",
    )
    p.add_argument(
        "--partition",
        metavar="HASH:N/M",
        help=(
            "Append one nextest hash partition to --tests-only integration; "
            "for example hash:1/2."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help=(
            "Print the canonical full gate command list, annotated with "
            "which commands the --local subset includes, and exit."
        ),
    )
    p.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run optional local validation (chelis#360): two workspace "
            "clippy configurations, fmt --check, chelis lint --check ., the "
            "regeneration and compile-fail guards, both oracles, and cargo "
            "nextest run -p <crate> for each crate changed vs origin/main. "
            "Takes the advisory gate lease. The workspace nextest stage is "
            "CI-owned."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Run the pre-push gate before every push: regen_all.py --tier 0 "
            "and cargo fmt --all fix in place, then chelis lint --check ., "
            "cargo clippy -p <crate> --tests per changed crate, and one nextest "
            "run over the drift tripwires. Prints the files it changed; never "
            "takes the lease."
        ),
    )
    p.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Start the run in its own session and return at once, printing a "
            "handle. For a caller with a foreground command timeout shorter "
            "than the run: the three completed --local runs of the 2026-09-02 "
            "fleet took 7m51s, 9m56s and 10m02s against a ten-minute limit. "
            "The exit code is a launch verdict; use --status for the gate's."
        ),
    )
    p.add_argument(
        "--status",
        nargs="?",
        const="latest",
        metavar="HANDLE",
        default=None,
        help=(
            "Report a detached run's verdict and exit with it: 75 while it is "
            "still running, otherwise the run's own exit code. Omit HANDLE for "
            "the newest handle in the report directory, which is this "
            "worktree's unless $CHELIS_GATE_REPORT_DIR points elsewhere."
        ),
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit 4 when the lease or its admission queue is busy.",
    )
    p.add_argument(
        "--no-lease",
        action="store_true",
        help="Do not take or wait for the advisory gate lease.",
    )
    p.add_argument(
        "--lease-timeout",
        metavar="SECONDS",
        type=float,
        default=None,
        help="Wait at most this long for the lease, then exit 4.",
    )
    args = p.parse_args(argv)
    if args.local and args.stage is not None:
        p.error("--local cannot be combined with a CI stage name")
    if args.local and args.list:
        p.error("--local cannot be combined with --list")
    if args.fast and args.stage is not None:
        p.error("--fast cannot be combined with a CI stage name")
    if args.fast and args.list:
        p.error("--fast cannot be combined with --list")
    if args.fast and args.local:
        p.error("--fast and --local are mutually exclusive")
    if args.detach and args.status is not None:
        p.error("--detach and --status are mutually exclusive")
    if args.detach and args.list:
        p.error("--detach cannot be combined with --list")
    if args.detach and args.stage is not None:
        p.error(
            "--detach cannot be combined with a CI stage name; CI stage runs "
            "are already supervised by the workflow"
        )
    if args.detach and args.fast:
        # --fast fixes in place (regen_all.py --tier 0, cargo fmt --all). A
        # writer running unattended against a tree the agent is still editing
        # is the collision chelis#1568 documents, and --fast is fast enough
        # that no foreground limit is at stake.
        p.error(
            "--detach cannot be combined with --fast; --fast writes to the "
            "worktree and must not run unattended"
        )
    if args.status is not None and (args.fast or args.local or args.list):
        p.error("--status cannot be combined with --fast/--local/--list")
    if args.status is not None and args.stage is not None:
        p.error("--status cannot be combined with a CI stage name")
    lease_flags = [
        name
        for name, given in (
            ("--no-wait", args.no_wait),
            ("--no-lease", args.no_lease),
            ("--lease-timeout", args.lease_timeout is not None),
        )
        if given
    ]
    if lease_flags and args.status is not None:
        p.error(f"{'/'.join(lease_flags)} cannot be combined with --status")
    if lease_flags and args.list:
        p.error(f"{'/'.join(lease_flags)} cannot be combined with --list")
    if lease_flags and args.stage is not None:
        p.error(
            f"{'/'.join(lease_flags)} cannot be combined with a CI stage name; "
            "stage runs never take the lease"
        )
    if args.no_wait and args.lease_timeout is not None:
        p.error("--no-wait and --lease-timeout are mutually exclusive")
    if args.no_lease and (args.no_wait or args.lease_timeout is not None):
        p.error("--no-lease cannot be combined with --no-wait/--lease-timeout")
    if args.lease_timeout is not None and args.lease_timeout <= 0:
        p.error("--lease-timeout must be a positive number of seconds")
    if args.support_slice is not None and not args.support_only:
        p.error("--support-slice requires integration --support-only")
    if args.tests_only and args.support_only:
        p.error("--tests-only and --support-only are mutually exclusive")
    if (args.tests_only or args.support_only) and args.stage != "integration":
        p.error("--tests-only/--support-only require the integration stage")
    if args.partition is not None and not args.tests_only:
        p.error("--partition requires integration --tests-only")
    if args.partition is not None:
        match = HASH_PARTITION_RE.fullmatch(args.partition)
        if match is None:
            p.error("--partition must have the form hash:N/M")
        if int(match.group("shard")) > int(match.group("count")):
            p.error("--partition shard N must not exceed partition count M")
    if (args.tests_only or args.support_only or args.partition) and (
        args.local or args.list or args.fast
    ):
        p.error(
            "integration selectors cannot be combined with --local/--fast/--list"
        )
    return args


def selected_stage_commands(
    stage: str,
    *,
    tests_only: bool,
    support_only: bool,
    partition: str | None,
    support_slice: str | None = None,
    environ: dict[str, str] | None = None,
) -> list[list[str]]:
    """Return one CI stage slice without duplicating canonical commands."""
    commands = STAGES[stage]
    if stage == "targeted-units":
        raw = (environ or {}).get(TARGETED_PACKAGES_ENV, "")
        if not raw:
            raise ValueError(
                f"{TARGETED_PACKAGES_ENV} must name the exact package frontier"
            )
        packages = raw.split(",")
        if any(
            not package
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", package) is None
            for package in packages
        ):
            raise ValueError(
                f"{TARGETED_PACKAGES_ENV} must be comma-separated Cargo package names"
            )
        if len(packages) != len(set(packages)):
            raise ValueError(f"{TARGETED_PACKAGES_ENV} contains duplicates")
        if packages:
            package_args = [
                argument
                for package in packages
                for argument in ("-p", package)
            ]
            targeted_clippy = [
                "cargo",
                "clippy",
                *package_args,
                "--lib",
                "--bins",
                "--tests",
                "--",
                "-D",
                "warnings",
            ]
            targeted_units = [
                part
                for part in NEXTEST_TARGETED_UNITS
                if part != "--workspace"
            ]
            targeted_units[3:3] = package_args
            commands = [FMT_CHECK, targeted_clippy, targeted_units]
            commands.extend(
                TARGETED_DOCTESTS[package]
                for package in packages
                if package in TARGETED_DOCTESTS
            )
    if tests_only:
        selected = [list(commands[0])]
    elif support_only:
        selected = [list(command) for command in commands[1:]]
    else:
        selected = [list(command) for command in commands]
    if support_slice is not None:
        selected = selected[:3] if support_slice == "frontend" else selected[3:]
    if partition is not None:
        selected[0].extend(["--partition", partition])
    return selected


def describe_returncode(returncode: int) -> str:
    if returncode >= 0:
        return f"exit code: {returncode}"
    number = -returncode
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = "UNKNOWN"
    return f"signal: {name} ({number})"


def _failure_log_path(
    failure_root: Path,
    stage_label: str,
    command: list[str],
    index: int,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage_label).strip("-")
    executable = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", Path(command[0]).name
    ).strip("-")
    return failure_root / (
        f"{timestamp}-{os.getpid()}-{index:02d}-{stage}-{executable}.log"
    )


def _rerun_command(
    command: list[str],
    environment: dict[str, str],
    repo_root: Path,
) -> str:
    assignments = [
        f"PYO3_PYTHON={shlex.quote(environment['PYO3_PYTHON'])}"
    ]
    for name in (
        "CARGO_TARGET_DIR",
        "CARGO_HUSKY_DONT_INSTALL_HOOKS",
        # Without this the printed rerun of a failed oracle stage would
        # build its own binary and so would not reproduce the failure.
        ORACLE_BINARY_ENV,
        "RUSTUP_TOOLCHAIN",
    ):
        value = environment.get(name)
        if value:
            assignments.append(f"{name}={shlex.quote(value)}")
    return (
        f"cd {shlex.quote(str(repo_root))} && env "
        + " ".join(assignments)
        + " "
        + shlex.join(command)
    )


def _print_failure_diagnostics(
    *,
    command: list[str],
    returncode: int,
    launch_error: OSError | None,
    duration: float,
    stage_label: str,
    index: int,
    total: int,
    repo_root: Path,
    failure_log: Path,
    environment: dict[str, str],
    tail: deque[str],
    line_count: int,
    error_stream,
) -> None:
    print(
        f"\ngate: {stage_label} command {index}/{total} failed",
        file=error_stream,
    )
    if launch_error is None:
        print(f"gate: {describe_returncode(returncode)}", file=error_stream)
    else:
        print(f"gate: launch error: {launch_error}", file=error_stream)
        print("gate: exit code: 127", file=error_stream)
    print(f"gate: duration: {duration:.3f}s", file=error_stream)
    print(f"gate: cwd: {repo_root}", file=error_stream)
    print(f"gate: command: {shlex.join(command)}", file=error_stream)
    print(
        f"gate: host: {platform.platform()} ({platform.machine()})",
        file=error_stream,
    )
    print(
        "gate: runner Python: "
        f"{sys.version.splitlines()[0]} at {sys.executable}",
        file=error_stream,
    )
    uv = shutil.which("uv", path=environment.get("PATH"))
    print(f"gate: uv: {uv or '<not found>'}", file=error_stream)
    print("gate: relevant environment:", file=error_stream)
    for name in DIAGNOSTIC_ENVIRONMENT:
        value = environment.get(name)
        rendered = shlex.quote(value) if value is not None else "<unset>"
        print(f"  {name}={rendered}", file=error_stream)
    print(f"gate: complete transcript: {failure_log}", file=error_stream)
    print(
        f"gate: rerun: {_rerun_command(command, environment, repo_root)}",
        file=error_stream,
    )
    # The rerun line pins the handoff on purpose: reproduce-the-failure is
    # what a failure diagnostic is for, and dropping the pin would rerun a
    # different binary than the one that failed. But the same line gets
    # pasted again after a fix, and then the pin is a trap: it re-runs the
    # binary from the failing run, so a Rust fix appears not to work, and
    # in the direction where the OLD binary passes it reports a false
    # `ORACLE: PASS`. Naming both uses costs one line (chelis#1322).
    if (
        ORACLE_BINARY_ENV in environment
        and UNREPRESENTABLE_DOMAIN_ORACLE[-1] in command
    ):
        print(
            f"gate: the rerun above pins {ORACLE_BINARY_ENV} to the "
            "`chelis` this run already built, which reproduces the "
            "failure exactly. After changing Rust source, drop that "
            "assignment so the oracle rebuilds; otherwise you are "
            "retesting the old binary.",
            file=error_stream,
        )

    omitted = max(0, line_count - len(tail))
    print(
        f"gate: final {len(tail)} lines of failed-command output "
        f"({omitted} earlier lines omitted):",
        file=error_stream,
    )
    for line in tail:
        error_stream.write(line)
    if tail and not tail[-1].endswith("\n"):
        error_stream.write("\n")
    error_stream.flush()


def run_commands(
    commands: list[list[str]],
    *,
    stage_label: str = "gate",
    repo_root: Path = REPO_ROOT,
    failure_root: Path | None = None,
    environ: dict[str, str] | None = None,
    executable: Path | None = None,
    output_stream=None,
    error_stream=None,
    report: GateReport | None = None,
) -> int:
    """Run commands serially with live output and retained failure evidence.

    When `report` is given, one `StageRecord` is appended per command run
    (success and failure alike) and the termination class is set on the
    failure paths; the integer return is unchanged."""
    output = sys.stdout if output_stream is None else output_stream
    error = sys.stderr if error_stream is None else error_stream
    current_executable = (
        Path(sys.executable) if executable is None else executable
    )
    try:
        environment = gate_environment(
            dict(os.environ if environ is None else environ),
            executable=current_executable,
            repo_root=repo_root,
        )
    except ValueError as exc:
        # Not "Python setup" any more: `gate_environment` also rejects a
        # cross-worktree CARGO_TARGET_DIR and an unusable explicit handoff.
        print(f"gate: environment setup failed: {exc}", file=error)
        if report is not None:
            report.termination = "environment"
            report.exit_code = EXIT_ENVIRONMENT
        return EXIT_ENVIRONMENT

    # chelis#1322: hand the oracle the `chelis` this list builds before it.
    # An explicit caller setting is authoritative and is never replaced,
    # the same way `gate_environment` treats an explicit PYO3_PYTHON.
    if ORACLE_BINARY_ENV not in environment:
        handoff = oracle_binary_handoff(
            commands, environment["CARGO_TARGET_DIR"]
        )
        if handoff is not None:
            environment[ORACLE_BINARY_ENV] = handoff

    persistent_root = (
        repo_root / "target/gate-failures"
        if failure_root is None
        else failure_root
    )
    selected_python = Path(environment["PYO3_PYTHON"])
    if report is not None:
        report.selected_python = str(selected_python)
    total = len(commands)
    for index, template in enumerate(commands, start=1):
        command = materialize_command(template, selected_python)
        print(
            f"+ [{stage_label} {index}/{total}] {shlex.join(command)}",
            file=output,
            flush=True,
        )
        tail: deque[str] = deque(maxlen=FAILURE_TAIL_LINES)
        line_count = 0
        launch_error = None
        started = time.monotonic()
        temporary_path: Path | None = None
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="replace",
            prefix="chelis-gate-",
            suffix=".log",
            delete=False,
        ) as transcript:
            temporary_path = Path(transcript.name)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                launch_error = exc
                returncode = 127
                line = f"launch error for {shlex.join(command)}: {exc}\n"
                transcript.write(line)
                transcript.flush()
                tail.append(line)
                line_count = 1
            else:
                assert process.stdout is not None
                with process.stdout:
                    for line in process.stdout:
                        transcript.write(line)
                        transcript.flush()
                        output.write(line)
                        output.flush()
                        tail.append(line)
                        line_count += 1
                returncode = process.wait()

        duration = time.monotonic() - started
        assert temporary_path is not None
        if returncode == 0:
            temporary_path.unlink(missing_ok=True)
            if report is not None:
                report.stages.append(
                    StageRecord(index, shlex.join(command), duration, 0)
                )
            continue

        persistent_root.mkdir(parents=True, exist_ok=True)
        failure_log = _failure_log_path(
            persistent_root,
            stage_label,
            command,
            index,
        )
        shutil.move(str(temporary_path), str(failure_log))
        mapped = returncode if returncode >= 0 else 128 - returncode
        if report is not None:
            report.stages.append(
                StageRecord(
                    index,
                    shlex.join(command),
                    duration,
                    returncode,
                    launch_error=None if launch_error is None else str(launch_error),
                    transcript=str(failure_log),
                )
            )
            report.termination = "signal" if returncode < 0 else "stage-failure"
            report.exit_code = mapped
        _print_failure_diagnostics(
            command=command,
            returncode=returncode,
            launch_error=launch_error,
            duration=duration,
            stage_label=stage_label,
            index=index,
            total=total,
            repo_root=repo_root,
            failure_log=failure_log,
            environment=environment,
            tail=tail,
            line_count=line_count,
            error_stream=error,
        )
        return mapped
    return 0


def main(
    argv: list[str],
    *,
    environ: dict[str, str] | None = None,
    executable: Path | None = None,
) -> int:
    """Dispatch one gate invocation. `environ` and `executable` default to
    the process's own so every existing `main([...])` call is unchanged;
    tests inject a minimal environment instead of inheriting a developer's
    `CARGO_TARGET_DIR` or writing under the real repository."""
    args = parse_args(argv)
    if args.list:
        for command in full_command_list():
            print(f"{render(command)}  # {list_annotation(command)}")
        print(FAST_DYNAMIC_NOTE)
        print(LOCAL_DYNAMIC_NOTE)
        return 0
    environment_in = dict(os.environ if environ is None else environ)
    current_executable = (
        Path(sys.executable) if executable is None else executable
    )
    if args.status is not None:
        return run_status(args, environ=environment_in)
    if args.detach:
        # Deliberately BEFORE the GateReport below: the launcher is not a gate
        # run, so it writes no summary and takes no lease. The child does both,
        # which keeps "exactly one summary per run" true.
        return run_detach(
            args,
            argv=list(argv),
            environ=environment_in,
            executable=current_executable,
            mode="local" if args.local else "full",
        )
    if args.fast:
        mode = "fast"
    elif args.local:
        mode = "local"
    elif args.stage is not None:
        mode = args.stage
    else:
        mode = "full"
    report = GateReport(mode=mode, started_at=_iso(_utc_now()))
    state: dict = {"lease": None}
    exit_code = 0
    try:
        if args.stage is not None:
            # CI stage runs: no preflight, no lease, summary only. A shallow
            # clone has no origin/main, so record what git can answer.
            report.git.update(_git_facts())
            commands = selected_stage_commands(
                args.stage,
                tests_only=args.tests_only,
                support_only=args.support_only,
                partition=args.partition,
                support_slice=args.support_slice,
                environ=environment_in,
            )
            exit_code = run_commands(
                commands,
                stage_label=args.stage,
                environ=environment_in,
                executable=current_executable,
                report=report,
            )
        else:
            runner = {"fast": run_fast, "local": run_local}.get(mode, run_full)
            exit_code = runner(
                args,
                report=report,
                environ=environment_in,
                executable=current_executable,
                state=state,
            )
    except KeyboardInterrupt:
        report.termination = "user-cancel"
        exit_code = EXIT_USER_CANCEL
        print("\ngate: cancelled by the user", file=sys.stderr)
    except BaseException:
        report.termination = "internal-error"
        exit_code = 1
        raise
    finally:
        lease = state.get("lease")
        if lease is not None:
            lease.release()
        report.exit_code = exit_code
        summary_path: Path | None = None
        try:
            summary_path = write_summary(report, environ=environment_in)
        except OSError as exc:
            print(f"gate: warning: could not write the run summary: {exc}", file=sys.stderr)
        print(human_summary(report, summary_path), flush=True)
    return exit_code


if __name__ == "__main__":
    managed_runtime_status = ensure_managed_runtime(sys.argv[1:])
    if managed_runtime_status is not None:
        sys.exit(managed_runtime_status)
    sys.exit(main(sys.argv[1:]))
