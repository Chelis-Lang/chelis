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
    python3 scripts/gate.py integration     # run the integration subset
    python3 scripts/gate.py integration --tests-only --partition hash:1/2
    python3 scripts/gate.py integration --support-only
    python3 scripts/gate.py --list     # print the canonical full list,
                                       # annotated local-vs-CI-owned
    python3 scripts/gate.py --local    # run the developer pre-push subset

Local/CI stage split (chelis#360): the full developer gate runs
`cargo nextest run --workspace --no-fail-fast` with the default profile, while the CI
integration stage uses the `ci` profile and delegates its two census binaries
to the required dtype oracle. The workspace execution stays out of `--local` --
macOS Smoke is the authoritative
workspace oracle, and on the macOS workstation the mass first-exec
burst it triggers can wedge assessment entirely (see
docs/local_macos_environment.md). `--local` is the pre-push
checkpoint: workspace clippy (compile-only, no mass exec), fmt,
`chelis lint`, plus `cargo nextest run -p <crate> --no-fail-fast` for each crate
changed vs `origin/main` (committed diff plus uncommitted work). The
derived crate list is always printed so nothing is silently skipped.

The script is safe to run from any cwd: child commands use the repo root
(resolved relative to the script's own location) as their working directory.
Every child inherits one validated `PYO3_PYTHON`. Combined stdout/stderr is
streamed live; a failed command's complete transcript and a 200-line replay
are written under `target/gate-failures/`. Cargo output is pinned to this
worktree, and nextest continues after failures to expose the complete set.

Why three explicit rustdoc stages
---------------------------------
`cargo nextest` does not execute doctests, so a crate with a doctest
contract needs an explicit `cargo test -p <crate> --doc` command or that
contract runs nowhere. The `chelis-types` command runs the chelis#731
`ErrorWitness` contracts; the `chelis-compiler-api` and
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
aborts before the first command rather than after the whole pre-push subset
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
# support worker runs its two non-test oracles exactly once.
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
# This script verifies the exact compiler diagnostic from the standalone
# raw-offset fixture. The marker is replaced with the same validated managed
# interpreter exported to child commands as PYO3_PYTHON.
CHECKPOINT_COMPILE_FAIL: list[str] = [
    MANAGED_PYTHON,
    "scripts/check_checkpoint_compile_fail.py",
]
# The pipeline-core boundary guards. Before this, they ran only in the manual
# `compiler_pipeline_oracle.py`, so a forbidden dependency, a false no_std
# claim, or a broken facade compile-fail boundary passed hosted CI green. The
# dependency guard (one `cargo metadata`) and the documentation guard (pure
# Python) are cheap enough for the local pre-push subset; the pipeline-artifact
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

STAGES: dict[str, list[list[str]]] = {
    "lint-and-unit": [
        CLIPPY_WORKSPACE,
        FMT_CHECK,
        CHELIS_LINT_CHECK,
        CHELIS_STD_BUNDLE_CHECK,
        DOCTEST_TYPES,
        DOCTEST_COMPILER_API,
        DOCTEST_PIPELINE_CORE,
        CHECKPOINT_COMPILE_FAIL,
        PIPELINE_CORE_DEPENDENCY_GUARD,
        PIPELINE_CORE_DOCUMENTATION_GUARD,
        PIPELINE_CORE_COMPILE_FAIL,
    ],
    "integration": [
        NEXTEST_WORKSPACE_CI,
        COMPILER_FRONT_END_PERFORMANCE_ORACLE,
        UNREPRESENTABLE_DOMAIN_ORACLE,
    ],
}

STAGE_ORDER: list[str] = ["lint-and-unit", "integration"]

HASH_PARTITION_RE = re.compile(
    r"^hash:(?P<shard>[1-9][0-9]*)/(?P<count>[1-9][0-9]*)$"
)

# The static `--local` pre-push subset (chelis#360). Deliberately
# excludes BUILD_WORKSPACE (clippy already compiles everything; no mass
# first-exec burst) and NEXTEST_WORKSPACE (CI-owned; macOS Smoke is the
# authoritative workspace oracle). `--local` appends a dynamic
# `cargo nextest run -p <crate> --no-fail-fast` stage per changed crate; see
# `local_command_list`.
LOCAL_STATIC_COMMANDS: list[list[str]] = [
    CLIPPY_WORKSPACE,
    FMT_CHECK,
    CHELIS_LINT_CHECK,
    CHELIS_STD_BUNDLE_CHECK,
    DOCTEST_TYPES,
    DOCTEST_COMPILER_API,
    DOCTEST_PIPELINE_CORE,
    CHECKPOINT_COMPILE_FAIL,
    PIPELINE_CORE_DEPENDENCY_GUARD,
    PIPELINE_CORE_DOCUMENTATION_GUARD,
    UNREPRESENTABLE_DOMAIN_ORACLE,
]

LOCAL_ANNOTATION = "local + ci"
CI_OWNED_ANNOTATION = "ci-owned"
FULL_GATE_SPLIT_ANNOTATION = "full gate; CI coverage split"
LOCAL_DYNAMIC_NOTE = (
    "# --local also runs: cargo nextest run -p <crate> --no-fail-fast "
    "for each crate changed vs origin/main"
)


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
    `--local` pre-push subset includes it or CI owns it."""
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
    """The `--local` pre-push command list: the static subset plus one
    `cargo nextest run -p <crate> --no-fail-fast` per changed crate."""
    commands = list(LOCAL_STATIC_COMMANDS)
    for crate in crates:
        commands.append(
            ["cargo", "nextest", "run", "-p", crate, "--no-fail-fast"]
        )
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


def run_local() -> int:
    """Run the `--local` pre-push gate: derive the changed crates vs
    origin/main, print the derived list (or say explicitly that none
    were detected), then run the local command list."""
    try:
        diff_output = _git_output(
            ["diff", "--name-only", "origin/main...HEAD"]
        )
        status_output = _git_output(["status", "--porcelain"])
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(
            f"gate --local: git failed ({stderr}); cannot derive changed "
            f"crates vs origin/main",
            file=sys.stderr,
        )
        return exc.returncode or 1
    paths = changed_paths_from_git(diff_output, status_output)
    crates = changed_crates(paths, workspace_member_packages())
    if crates:
        print(
            "gate --local: changed crates vs origin/main: "
            + ", ".join(crates),
            flush=True,
        )
    else:
        print(
            "gate --local: no crate changes detected vs origin/main; "
            "skipping the per-crate nextest stage. The workspace suite "
            "is CI-owned and was NOT run.",
            flush=True,
        )
    return run_commands(local_command_list(crates), stage_label="local")


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
        choices=STAGE_ORDER,
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
            "Run the developer pre-push subset: workspace clippy, fmt "
            "--check, chelis lint --check ., and cargo nextest run -p "
            "<crate> for each crate changed vs origin/main. The "
            "workspace nextest stage is CI-owned (chelis#360)."
        ),
    )
    args = p.parse_args(argv)
    if args.local and args.stage is not None:
        p.error("--local cannot be combined with a CI stage name")
    if args.local and args.list:
        p.error("--local cannot be combined with --list")
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
        args.local or args.list
    ):
        p.error("integration selectors cannot be combined with --local/--list")
    return args


def selected_stage_commands(
    stage: str,
    *,
    tests_only: bool,
    support_only: bool,
    partition: str | None,
) -> list[list[str]]:
    """Return one CI stage slice without duplicating canonical commands."""
    commands = STAGES[stage]
    if tests_only:
        selected = [list(commands[0])]
    elif support_only:
        selected = [list(command) for command in commands[1:]]
    else:
        selected = [list(command) for command in commands]
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
) -> int:
    """Run commands serially with live output and retained failure evidence."""
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
        return 2

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
            continue

        persistent_root.mkdir(parents=True, exist_ok=True)
        failure_log = _failure_log_path(
            persistent_root,
            stage_label,
            command,
            index,
        )
        shutil.move(str(temporary_path), str(failure_log))
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
        return returncode if returncode >= 0 else 128 - returncode
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.list:
        for command in full_command_list():
            print(f"{render(command)}  # {list_annotation(command)}")
        print(LOCAL_DYNAMIC_NOTE)
        return 0
    if args.local:
        return run_local()
    if args.stage is not None:
        commands = selected_stage_commands(
            args.stage,
            tests_only=args.tests_only,
            support_only=args.support_only,
            partition=args.partition,
        )
        stage_label = args.stage
    else:
        commands = full_command_list()
        stage_label = "full"
    return run_commands(commands, stage_label=stage_label)


if __name__ == "__main__":
    managed_runtime_status = ensure_managed_runtime(sys.argv[1:])
    if managed_runtime_status is not None:
        sys.exit(managed_runtime_status)
    sys.exit(main(sys.argv[1:]))
