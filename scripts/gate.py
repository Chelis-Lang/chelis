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
    python3 scripts/gate.py lint-and-unit   # run the lint-and-unit subset
    python3 scripts/gate.py integration     # run the integration subset
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

# The canonical CI-stage command list. The CI workflow
# has two developer-gate jobs, `lint-and-unit` and `workspace-tests`; each
# runs its own subset (`workspace-tests` invokes the `integration` stage).
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
# C-backend CI job also runs that crate's doctests through an unfiltered
# `cargo test -p chelis-backend-c` command.
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
# This script verifies the exact compiler diagnostic from the standalone
# raw-offset fixture. The marker is replaced with the same validated managed
# interpreter exported to child commands as PYO3_PYTHON.
CHECKPOINT_COMPILE_FAIL: list[str] = [
    MANAGED_PYTHON,
    "scripts/check_checkpoint_compile_fail.py",
]

STAGES: dict[str, list[list[str]]] = {
    "lint-and-unit": [
        BUILD_WORKSPACE,
        CLIPPY_WORKSPACE,
        FMT_CHECK,
        CHELIS_LINT_CHECK,
        DOCTEST_TYPES,
        DOCTEST_COMPILER_API,
        CHECKPOINT_COMPILE_FAIL,
    ],
    "integration": [
        NEXTEST_WORKSPACE_CI,
    ],
}

STAGE_ORDER: list[str] = ["lint-and-unit", "integration"]

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
    DOCTEST_TYPES,
    DOCTEST_COMPILER_API,
    CHECKPOINT_COMPILE_FAIL,
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
        environment["PYO3_PYTHON"] = str(candidate)
    else:
        if not executable.is_file():
            raise ValueError(
                "gate.py selected a managed Python interpreter that does "
                f"not exist at {executable}; run `uv python install 3.11` "
                "and retry."
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

    # cargo-husky's build script can write into the clone's shared .git
    # directory. The checked-in hook remains directly runnable; gate builds
    # must not mutate shared Git state behind sibling worktrees.
    environment["CARGO_HUSKY_DONT_INSTALL_HOOKS"] = "1"
    return environment


def materialize_command(command: list[str], python: Path) -> list[str]:
    """Replace the canonical managed-Python marker for execution."""
    return [str(python) if part == MANAGED_PYTHON else part for part in command]


def full_command_list() -> list[list[str]]:
    """The complete developer gate.

    CI runs the same lint-and-unit list, but its integration stage uses the
    split `ci` profile and delegates two census binaries to the required dtype
    oracle. The developer command uses the default profile so those tests stay
    present without requiring a hosted-only parallel job.
    """
    return [*STAGES["lint-and-unit"], NEXTEST_WORKSPACE]


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
    return args


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
    for name in ("CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"):
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
        print(f"gate: Python setup failed: {exc}", file=error)
        return 2

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
        commands = STAGES[args.stage]
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
