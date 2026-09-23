#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 1 acceptance oracle.

This runner turns `spec/design/dtype_semantics.md`'s Phase 1 oracle into one
executable command. It inherits the complete Phase 0 detector contract, then
covers the three frozen entry censuses, the sealed dtype/storage and
typed-ingress contracts, the exact execution wire, Phase 1-specific
payload/checker controls, and the Python/Hull readers of the tagged wire.

Usage:

    .venv/bin/python scripts/dtype_phase1_oracle.py

Acceptance is exit 0 with the final line ``PHASE 1 ORACLE: PASS``. Compiled-C
Phase 2/3 rows and ignored future rows are deliberately outside this command.
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
    """Return the frozen Phase 1 command manifest in execution order."""

    return (
        OracleLeg(
            "inherited Phase 0 contract",
            (python, "scripts/dtype_phase0_oracle.py"),
        ),
        OracleLeg(
            "covered-family Rust and C tripwire",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
            ),
        ),
        OracleLeg(
            "typed execution-wire census",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "capacity_census_wire",
            ),
        ),
        OracleLeg(
            "registered PyO3 signature census",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-python",
                "--test",
                "capacity_census_bindings",
            ),
        ),
        OracleLeg(
            "sealed dtype semantics",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "-E",
                "test(dtype_semantics::tests::)",
            ),
        ),
        OracleLeg(
            "typed Load ingress",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "-E",
                "test(load_ingress_)",
            ),
        ),
        OracleLeg(
            "execution-wire exactness",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "-E",
                "binary(execution_wire_v2) | test(reduced_float_) | "
                "test(eval_rejects_tagged_binding_dtype_substitution)",
            ),
        ),
        OracleLeg(
            "Phase 1 payload and checker contracts",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "issue_729_payload_census",
                "--test",
                "issue_759_checked_cast_default",
                "--test",
                "issue_860_checker_chokepoint",
            ),
        ),
        OracleLeg(
            "Python dtype ingress",
            (
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "bindings/python/tests",
                "-p",
                "test_dtype_ingress.py",
            ),
        ),
        OracleLeg(
            "Hull tagged-value reader",
            (
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests/conformance/hull",
                "-p",
                "test_run_conformance.py",
            ),
        ),
    )


def run_oracle(python: str = sys.executable) -> None:
    for index, leg in enumerate(oracle_legs(python), start=1):
        rendered = shlex.join(leg.argv)
        print(f"[{index}/{len(oracle_legs(python))}] {leg.name}: {rendered}", flush=True)
        try:
            completed = subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"PHASE 1 ORACLE: FAIL: {leg.name} (exit {error.returncode})") from error
        if completed.returncode != 0:
            raise SystemExit(
                f"PHASE 1 ORACLE: FAIL: {leg.name} (exit {completed.returncode})"
            )
    print("PHASE 1 ORACLE: PASS")


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        run_oracle()
