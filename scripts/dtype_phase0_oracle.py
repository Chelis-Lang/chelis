#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 0 detector oracle.

This runner gives the detector work merged in PR #758 one named executable
receipt. It runs the frozen domain checker and every active matrix that owns a
Phase 0 detector chokepoint, the byte-exact executable-example comparator, and
the legacy floating agreement controls retained by PR #758.

Usage:

    .venv/bin/python scripts/dtype_phase0_oracle.py

Acceptance is exit 0 with the final line ``DTYPE PHASE 0 ORACLE: PASS``.
Known later-phase cells remain ignored by their owning matrix; this command
never opts into ignored rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


def oracle_legs() -> tuple[OracleLeg, ...]:
    """Return the frozen Phase 0 command manifest in execution order."""

    return (
        OracleLeg(
            "domain detector and active dtype matrices",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "domain_checker",
                "--test",
                "eval_tensor_narrowing_matrix",
                "--test",
                "narrow_dtype_matrix",
                "--test",
                "precision_matrix",
                "--test",
                "int_width_lane_matrix",
                "--test",
                "reduction_and_bitwise_matrix",
                "--test",
                "scalar_stub_matrix",
                "--test",
                "issue_680_int_exactness",
            ),
        ),
        OracleLeg(
            "byte-exact executable-corpus parity",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "parity",
            ),
        ),
        OracleLeg(
            "legacy floating agreement controls",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-e2e",
                "--test",
                "eval_agreement",
            ),
        ),
    )


def run_oracle() -> None:
    legs = oracle_legs()
    for index, leg in enumerate(legs, start=1):
        print(
            f"[{index}/{len(legs)}] {leg.name}: {shlex.join(leg.argv)}",
            flush=True,
        )
        try:
            subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"DTYPE PHASE 0 ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE PHASE 0 ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
