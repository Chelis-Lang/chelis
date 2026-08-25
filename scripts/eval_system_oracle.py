#!/usr/bin/env python3
"""Authoritative `add-eval-system-boundary` acceptance oracle.

The oracle runs the guard tests, the source guard, evaluator boundary tests,
invariant decode tests, and the live `process_run` CLI suite.

The ignored `std_io_pipeline` suite is not part of this oracle. Separate
compiled-lane evidence verifies the evaluator-only scope claim.

Usage:

    <managed-python> scripts/eval_system_oracle.py

Acceptance is exit 0 with the final line
``EVAL SYSTEM BOUNDARY ORACLE: PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


def oracle_legs(python: str) -> tuple[OracleLeg, ...]:
    """Return the frozen oracle command manifest in execution order."""

    return (
        OracleLeg(
            "source guard unit tests",
            (python, "scripts/test_eval_system_guard.py"),
        ),
        OracleLeg(
            "evaluator system boundary source guard",
            (python, "scripts/eval_system_guard.py"),
        ),
        OracleLeg(
            "compiler API evaluator boundary unit tests",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--lib",
                "-E",
                "test(runtime::system_tests::)",
                "--no-tests",
                "fail",
            ),
        ),
        OracleLeg(
            "opaque-type invariant decode conformance suite",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "invariant_decode",
                "--no-tests",
                "fail",
            ),
        ),
        OracleLeg(
            "process_run CLI parity suite",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "process_run_builtin",
                "--no-tests",
                "fail",
            ),
        ),
    )


def run_oracle(python: str = sys.executable) -> None:
    legs = oracle_legs(python)
    for index, leg in enumerate(legs, start=1):
        rendered = shlex.join(leg.argv)
        print(f"[{index}/{len(legs)}] {leg.name}: {rendered}", flush=True)
        try:
            completed = subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"EVAL SYSTEM BOUNDARY ORACLE: FAIL: {leg.name} (exit {error.returncode})"
            ) from error
        if completed.returncode != 0:
            raise SystemExit(
                f"EVAL SYSTEM BOUNDARY ORACLE: FAIL: {leg.name} (exit {completed.returncode})"
            )
    print("EVAL SYSTEM BOUNDARY ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
