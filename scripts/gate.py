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

Usage:
    python3 scripts/gate.py            # run every gate command
    python3 scripts/gate.py lint-and-unit   # run the lint-and-unit subset
    python3 scripts/gate.py integration     # run the integration subset
    python3 scripts/gate.py --list     # print the canonical full list,
                                       # annotated local-vs-CI-owned
    python3 scripts/gate.py --local    # run the developer pre-push subset

Local/CI stage split (chelis#360): the full
`cargo nextest run --workspace --profile ci` stage stays in the
canonical list but is CI-owned -- macOS Smoke is the authoritative
workspace oracle, and on the macOS workstation the mass first-exec
burst it triggers can wedge assessment entirely (see
docs/local_macos_environment.md). `--local` is the pre-push
checkpoint: workspace clippy (compile-only, no mass exec), fmt,
`chelis lint`, plus `cargo nextest run -p <crate>` for each crate
changed vs `origin/main` (committed diff plus uncommitted work). The
derived crate list is always printed so nothing is silently skipped.

The script is safe to run from any cwd: it `chdir`s to the repo root
(resolved relative to the script's own location) before running any
command.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# NB: `tomllib` is intentionally NOT imported at module top. It is stdlib
# only from Python 3.11, and on macOS the system `python3` is 3.9, so a
# top-level import crashed EVERY gate.py invocation — including `--list`
# and the per-stage CI forms that never touch TOML — under the system
# interpreter (chelis#366). Only the `--local` changed-crate derivation
# parses Cargo.toml, so the import is deferred into
# `workspace_member_packages()` and guarded with a one-line guidance
# message instead of a raw traceback.

REPO_ROOT = Path(__file__).resolve().parent.parent

# The canonical gate command list, split by CI stage. The CI workflow
# has two developer-gate jobs, `lint-and-unit` and `integration`; each
# runs its own subset, and the union is the full per-PR gate. Every
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
# The `ci` nextest profile (.config/nextest.toml) writes per-test JUnit
# timing XML to target/nextest/ci/junit.xml, which
# scripts/test_timing_check.py consumes. Running it locally too keeps
# the dev gate and CI on one command.
NEXTEST_WORKSPACE: list[str] = [
    "cargo", "nextest", "run", "--workspace", "--profile", "ci",
]
# chelis#875: `cargo nextest` does not execute doctests, and every other
# gate stage runs under nextest. The chelis#731 Phase 2 `ErrorWitness`
# compile-fail oracles (crates/chelis-types/src/errors.rs) are rustdoc
# ```compile_fail blocks, so before this stage existed they ran in NO
# continuous job: the strongest artifact in that plan sat on the top rung
# of docs/agent_quality_architecture.md's ladder with nothing driving it.
#
# NARROWING, stated so a future widening is a conscious act: this is
# scoped to the two crates that own compile-fail oracles, not
# `--workspace --doc`. It buys the chelis-types oracles plus the
# chelis-compiler-api diagnostic producer privacy oracles; a workspace-wide
# doctest stage is a larger change (every crate's doc examples become gating)
# and should be argued on its own merits rather than smuggled in here.
#
# Doctests run in exactly two places in this repo: this stage (for these two
# packages), and the
# C-backend CI job's unfiltered `cargo test -p chelis-backend-c` (which
# picks up that crate's privacy compile-fail doctests as a side effect of
# having no `--lib`/`--test` filter). A `compile_fail` oracle added to any
# OTHER crate runs nowhere until one of those two is extended.
DOCTEST_ORACLES: list[str] = [
    "cargo",
    "test",
    "-p",
    "chelis-types",
    "-p",
    "chelis-compiler-api",
    "--doc",
]

STAGES: dict[str, list[list[str]]] = {
    "lint-and-unit": [
        BUILD_WORKSPACE,
        CLIPPY_WORKSPACE,
        FMT_CHECK,
        CHELIS_LINT_CHECK,
        DOCTEST_ORACLES,
    ],
    "integration": [
        NEXTEST_WORKSPACE,
    ],
}

STAGE_ORDER: list[str] = ["lint-and-unit", "integration"]

# The static `--local` pre-push subset (chelis#360). Deliberately
# excludes BUILD_WORKSPACE (clippy already compiles everything; no mass
# first-exec burst) and NEXTEST_WORKSPACE (CI-owned; macOS Smoke is the
# authoritative workspace oracle). `--local` appends a dynamic
# `cargo nextest run -p <crate>` stage per changed crate; see
# `local_command_list`.
LOCAL_STATIC_COMMANDS: list[list[str]] = [
    CLIPPY_WORKSPACE,
    FMT_CHECK,
    CHELIS_LINT_CHECK,
    DOCTEST_ORACLES,
]

LOCAL_ANNOTATION = "local + ci"
CI_OWNED_ANNOTATION = "ci-owned"
LOCAL_DYNAMIC_NOTE = (
    "# --local also runs: cargo nextest run -p <crate> "
    "for each crate changed vs origin/main"
)


def full_command_list() -> list[list[str]]:
    """The canonical full gate list: the union of every stage subset,
    in stage order. `cargo build` deliberately comes first so a compile
    failure surfaces before the slower clippy/test commands."""
    commands: list[list[str]] = []
    for stage in STAGE_ORDER:
        commands.extend(STAGES[stage])
    return commands


def render(command: list[str]) -> str:
    return " ".join(command)


def list_annotation(command: list[str]) -> str:
    """The `--list` annotation for a canonical command: whether the
    `--local` pre-push subset includes it or CI owns it."""
    if command in LOCAL_STATIC_COMMANDS:
        return LOCAL_ANNOTATION
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
            "gate.py --local needs Python 3.11+ (tomllib); run via "
            ".venv/bin/python per AGENTS.md",
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
    `cargo nextest run -p <crate>` per changed crate."""
    commands = list(LOCAL_STATIC_COMMANDS)
    for crate in crates:
        commands.append(["cargo", "nextest", "run", "-p", crate])
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
    return run_commands(local_command_list(crates))


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
            "Run only this CI stage's gate subset. Omit to run the full "
            "gate (the union of every stage)."
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


def run_commands(commands: list[list[str]]) -> int:
    """Run each command from the repo root, stopping at the first
    failure. Returns the exit code of the first failing command, or 0
    if every command succeeded."""
    for command in commands:
        print(f"+ {render(command)}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            print(
                f"gate: command failed with exit {result.returncode}: "
                f"{render(command)}",
                file=sys.stderr,
            )
            return result.returncode
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
    else:
        commands = full_command_list()
    return run_commands(commands)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
