#!/usr/bin/env python3
"""Authoritative chelis#1287 first-class Count acceptance oracle.

The oracle covers the complete checker grammar, dedicated IR and evaluator,
exact WireDag v6 boundary, compiled C execution, HIP/Metal structural routing,
semantic/capacity registration, and the executable example corpus entry.

Usage:

    .venv/bin/python scripts/dtype_count_oracle.py

Acceptance is exit 0 with the final line ``DTYPE COUNT ORACLE: PASS``.
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
    """Return the frozen chelis#1287 command manifest in execution order."""

    return (
        OracleLeg(
            "checker axis and dtype contract",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "-E",
                "binary(issue_1287_count_checker) | test(count_tensor_groups_) | test(adjacent_pair_fold_preserves_canonical_tree) | test(count_add_traps_int64_overflow)",
            ),
        ),
        OracleLeg(
            "dedicated Count IR and evaluator",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "--test",
                "issue_1287_count",
            ),
        ),
        OracleLeg(
            "exact current WireDag and registered wire capacity",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "wire_dag_v6_count",
                "--test",
                "capacity_census_wire",
            ),
        ),
        OracleLeg(
            "compiled C exact multi-axis and empty execution",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-c",
                "--test",
                "exec_compile",
                "-E",
                "test(exec_count_)",
            ),
        ),
        OracleLeg(
            "device Count structural child oracle",
            (python, "scripts/dtype_count_device_oracle.py"),
        ),
        OracleLeg(
            "semantic registration and executable example parity",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
                "--test",
                "parity",
                "-E",
                "test(count_is_registered_against_its_exact_authority_atom) | "
                "test(parity_count_bool_axes) | test(parity_corpus_is_complete)",
            ),
        ),
        OracleLeg(
            "generated rejection registry agreement",
            (python, "scripts/generate_rejection_registries.py", "--check"),
        ),
    )


def run_oracle(python: str = sys.executable) -> None:
    legs = oracle_legs(python)
    for index, leg in enumerate(legs, start=1):
        print(f"[{index}/{len(legs)}] {leg.name}: {shlex.join(leg.argv)}", flush=True)
        try:
            subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"DTYPE COUNT ORACLE: FAIL: {leg.name} (exit {error.returncode})"
            ) from error
    print("DTYPE COUNT ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
