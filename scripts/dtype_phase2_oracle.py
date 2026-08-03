#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 2 acceptance oracle.

Phase 2 inherits the sealed Phase 1 surface, then proves that every host/IR
arithmetic consumer—including ordinary, window, and argument reductions—uses
the dtype-keyed kernels at the declared arithmetic width. It also freezes the
numeric-trap bytes and exact prover carriers.

Usage:

    .venv/bin/python scripts/dtype_phase2_oracle.py

Acceptance is exit 0 with the final line ``DTYPE PHASE 2 ORACLE: PASS``.
Compiled backend adoption remains Phase 3 and is deliberately absent.
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
    """Return the frozen Phase 2 command manifest in execution order."""

    return (
        OracleLeg(
            "inherited Phase 1 contract",
            (python, "scripts/dtype_phase1_oracle.py"),
        ),
        OracleLeg(
            "sealed numeric kernels and trap bytes",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "-E",
                "test(dtype_semantics::tests::) | binary(numeric_trap_contract)",
            ),
        ),
        OracleLeg(
            "IR kernel exclusivity",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "--test",
                "dtype_kernel_boundary",
            ),
        ),
        OracleLeg(
            "IR declared-width reduction behavior",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "-E",
                "test(phase2_)",
            ),
        ),
        OracleLeg(
            "host kernel exclusivity",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "dtype_kernel_boundary",
            ),
        ),
        OracleLeg(
            "host eval Phase 2 matrices",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "dtype_phase2_reduction_matrix",
                "--test",
                "issue_680_int_exactness",
                "--test",
                "precision_matrix",
                "--test",
                "prove_int64_exactness",
                "--test",
                "fold_static_cond_matrix",
            ),
        ),
        OracleLeg(
            "exact prover carriers",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-prove",
                "--test",
                "dtype_concrete_boundary",
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
                f"DTYPE PHASE 2 ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE PHASE 2 ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
