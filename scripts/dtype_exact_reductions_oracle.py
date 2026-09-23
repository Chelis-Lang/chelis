#!/usr/bin/env python3
"""Authoritative exact-reduction completion oracle for chelis#1281."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


class OracleFailure(RuntimeError):
    """A declared exact-reduction contract is absent or obsolete."""


@dataclass(frozen=True)
class SourceContract:
    name: str
    path: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


def source_contracts() -> tuple[SourceContract, ...]:
    return (
        SourceContract(
            "normative reduction atoms",
            "spec/05-risc-primitives.md",
            (
                "[05-OP-11]",
                "[05-OP-12]",
                "[05-OP-13]",
                "[05-OP-15]",
                "[05-OP-16]",
                "[05-RWIN-1]",
                "[05-RWIN-2]",
            ),
        ),
        SourceContract(
            "IR exact-reduction acceptance",
            "crates/chelis-ir/tests/issue_1281_exact_reductions.rs",
            ("first_nan", "runtime_empty", "tie", "infinity", "window"),
        ),
        SourceContract(
            "compiled C exact-reduction acceptance",
            "crates/chelis-backend-c/tests/issue_1281_exact_reductions.rs",
            ("first_nan", "runtime_empty", "tie", "infinity", "window"),
        ),
        SourceContract(
            "compiled C window extrema",
            "crates/chelis-backend-c/src/emit.rs",
            ("first NaN",),
            (
                "existing tie/NaN arithmetic remains tracked by #1298",
                "if (((const float*)t{x}_data)[dst_idx] == ext) { "
                "((float*)t{id}_data)[dst_idx] += gval; }",
            ),
        ),
    )


def validate_source_text(
    name: str,
    source: str,
    *,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> None:
    for token in required:
        if token not in source:
            raise OracleFailure(f"{name}: required contract `{token}` is absent")
    for token in forbidden:
        if token in source:
            raise OracleFailure(f"{name}: obsolete contract `{token}` remains")


def validate_source_contracts(repo_root: Path = REPO_ROOT) -> None:
    for contract in source_contracts():
        path = repo_root / contract.path
        try:
            source = path.read_text()
        except OSError as error:
            raise OracleFailure(
                f"{contract.name}: cannot read {contract.path}: {error}"
            ) from error
        validate_source_text(
            contract.name,
            source,
            required=contract.required,
            forbidden=contract.forbidden,
        )


def oracle_legs(python: str) -> tuple[OracleLeg, ...]:
    return (
        OracleLeg(
            "oracle self-tests",
            (python, "scripts/test_dtype_exact_reductions_oracle.py"),
        ),
        OracleLeg(
            "typed reduction kernels",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "-E",
                "test(/(issue_1281|reduce_tensor_groups|arg_reduce|window_grad)/)",
            ),
        ),
        OracleLeg(
            "IR evaluation and adjoints",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "--test",
                "issue_1281_exact_reductions",
            ),
        ),
        OracleLeg(
            "compiled C exact reductions",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-c",
                "--test",
                "issue_1281_exact_reductions",
            ),
        ),
        OracleLeg(
            "checker reduction domain",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "--test",
                "issue_230_argmax_reduce_output_dtype",
                "--test",
                "issue_259_nonliteral_axis_diagnostic",
            ),
        ),
        OracleLeg(
            "downstream exhaustive consumers",
            (
                "cargo",
                "check",
                "-p",
                "chelis-e2e",
                "-p",
                "chelis-prove",
                "-p",
                "chelis-backend-hip",
                "-p",
                "chelis-backend-metal",
                "--all-targets",
            ),
        ),
    )


def run_oracle(python: str = sys.executable) -> None:
    try:
        validate_source_contracts(REPO_ROOT)
    except OracleFailure as error:
        raise SystemExit(f"DTYPE EXACT REDUCTIONS ORACLE: FAIL: {error}") from error

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
                f"DTYPE EXACT REDUCTIONS ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE EXACT REDUCTIONS ORACLE: PASS")


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        run_oracle()
