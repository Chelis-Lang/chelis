#!/usr/bin/env python3
"""Run the complete Phase A hash-order determinism acceptance surface.

One component was removed with its subject. `chelis-cli`'s
`hash_order_stability` target ran K=24 fresh processes over programs whose
settlement chose between two candidate `expand` shapes, which is the property
chelis#1338 reported as nondeterministic. `spec/04-type-system.md` section
4.7.2 gives `expand` and `insert` one result shape each, so there is nothing
left to settle and #1338 is resolved by construction rather than by this
oracle (chelis#1277 S2b). Every other component here is unrelated to that
model and still runs.
"""

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
