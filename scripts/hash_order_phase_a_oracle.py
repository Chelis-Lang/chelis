#!/usr/bin/env python3
"""Run the complete Phase A hash-order determinism acceptance surface."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "scripts/check_hash_order_compile_fail.py"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "--lib",
        "hash_order_",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "cache_format_version_tracks_ordered_deferred_constraints",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "cli",
        "executable_examples",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "--test",
        "issue_942_inferred_tensor_cast",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "parity",
        "parity_hash_order_determinism",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "hash_order_stability",
        "--no-fail-fast",
    ),
)


class PhaseAOracleFailure(RuntimeError):
    """One component of the Phase A oracle failed."""


def validate(
    *, runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    commands: Sequence[Sequence[str]] = COMMANDS,
) -> None:
    for command in commands:
        completed = runner(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            rendered = " ".join(command)
            raise PhaseAOracleFailure(
                f"command exited {completed.returncode}: {rendered}"
            )
    print("HASH ORDER PHASE A ORACLE: PASS", flush=True)


def main() -> int:
    try:
        validate()
    except PhaseAOracleFailure as error:
        print(f"HASH ORDER PHASE A ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
