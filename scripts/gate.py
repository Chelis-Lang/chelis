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
    python3 scripts/gate.py --list     # print the canonical full list

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

REPO_ROOT = Path(__file__).resolve().parent.parent

# The canonical gate command list, split by CI stage. The CI workflow
# has two developer-gate jobs, `lint-and-unit` and `integration`; each
# runs its own subset, and the union is the full per-PR gate. Every
# command is a list of argv tokens (no shell).
#
# Keep this in lockstep with `.github/workflows/ci.yml`: the parity
# test in `scripts/test_gate.py` greps the workflow and fails if any
# `cargo`/`chelis` invocation in a gate step is not produced here.
STAGES: dict[str, list[list[str]]] = {
    "lint-and-unit": [
        ["cargo", "build", "--workspace", "--all-targets"],
        ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
        ["cargo", "fmt", "--all", "--", "--check"],
        [
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
        ],
    ],
    "integration": [
        # The `ci` nextest profile (.config/nextest.toml) writes
        # per-test JUnit timing XML to target/nextest/ci/junit.xml,
        # which scripts/test_timing_check.py consumes. Running it
        # locally too keeps the dev gate and CI on one command.
        ["cargo", "nextest", "run", "--workspace", "--profile", "ci"],
    ],
}

STAGE_ORDER: list[str] = ["lint-and-unit", "integration"]


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
        help="Print the canonical full gate command list and exit.",
    )
    return p.parse_args(argv)


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
            print(render(command))
        return 0
    if args.stage is not None:
        commands = STAGES[args.stage]
    else:
        commands = full_command_list()
    return run_commands(commands)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
