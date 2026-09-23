#!/usr/bin/env python3
"""Authoritative chelis#730 Phase 3 acceptance oracle.

The first five legs cover the gate, diagnostic-kind, and rejection-authority
contracts owned by chelis#730. The final leg is the independently owned #912
root-realizability interlock required by ``spec/design/loud_unsupported.md``.
Until that leg is enabled and green, this oracle deliberately exits nonzero
and Phase 3 is not complete.

Acceptance is exit 0 with the final line
``LOUD UNSUPPORTED PHASE 3 ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/loud_unsupported_phase3_oracle.py
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
    """Return the frozen Phase 3 command manifest in execution order."""

    return (
        OracleLeg(
            "shared gate and rejected-cell contract",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "phase3_gate_contract",
                "--test",
                "issue_687_rejected_cells_corpus",
                "-p",
                "chelis-compiler-api",
                "--test",
                "phase3_gate_inventory",
            ),
        ),
        OracleLeg(
            "sealed diagnostic-kind contract",
            (python, "scripts/diagnostic_kind_oracle.py"),
        ),
        OracleLeg(
            "typed rejection-authority tests",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "--test",
                "rejection_authority",
            ),
        ),
        OracleLeg(
            "rejection-authority boundary",
            (python, "scripts/check_rejection_authority_boundary.py"),
        ),
        OracleLeg(
            "live issue-authority manifest",
            (python, "scripts/validate_rejection_issue_manifest.py"),
        ),
        OracleLeg(
            "root-realizability integration",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "issue_912_root_boundary",
                "--run-ignored",
                "all",
            ),
        ),
    )


def run_oracle(python: str = sys.executable) -> None:
    legs = oracle_legs(python)
    for index, leg in enumerate(legs, start=1):
        print(
            f"[{index}/{len(legs)}] {leg.name}: {shlex.join(leg.argv)}",
            flush=True,
        )
        try:
            subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                "LOUD UNSUPPORTED PHASE 3 ORACLE: FAIL: "
                f"{leg.name} (exit {error.returncode})"
            ) from error
    print("LOUD UNSUPPORTED PHASE 3 ORACLE: PASS")


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        run_oracle()
