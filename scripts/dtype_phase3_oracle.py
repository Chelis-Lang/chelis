#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 3 acceptance oracle.

Phase 3 inherits the complete Phase 2 numeric contract and the generated
observation contract, then turns on the compiled-C dtype rows.  Acceptance is
exit 0 with the final line ``DTYPE PHASE 3 ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/dtype_phase3_oracle.py
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
            "inherited Phase 2 contract",
            (python, "scripts/dtype_phase2_oracle.py"),
        ),
        OracleLeg(
            "inherited observation Phase 3 contract",
            (python, "scripts/faithful_observation_phase3_oracle.py"),
        ),
        OracleLeg(
            "compiled C dtype matrix",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "narrow_dtype_matrix",
                "--test",
                "int_width_lane_matrix",
                "--test",
                "precision_matrix",
                "--test",
                "scalar_stub_matrix",
                "--test",
                "reduction_and_bitwise_matrix",
                "--test",
                "fold_static_cond_matrix",
                "--test",
                "issue_759_checked_cast_default",
                "--test",
                "issue_761_subnormal_ingress",
                "--test",
                "issue_734_tostring_placeholder",
                "--test",
                "observation_roundtrip_harness",
            ),
        ),
        OracleLeg(
            "compiled C structural locks",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-c",
                "--test",
                "host_emit_dtype_dispatch",
                "--test",
                "exec_compile",
            ),
        ),
        OracleLeg(
            "numeric surface censuses",
            (python, "scripts/capacity_census_liveness.py"),
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
                f"DTYPE PHASE 3 ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE PHASE 3 ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
