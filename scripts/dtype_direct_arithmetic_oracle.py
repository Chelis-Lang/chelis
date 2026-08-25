#!/usr/bin/env python3
"""Authoritative direct-arithmetic completion oracle for chelis#1306.

This oracle proves the executable eval/C behavior, the HIP code-generation
contract, the explicit target dispositions, and the direct IR/WireDag
identities required by [05-OP-40] and [05-OP-41]. It deliberately does not run
ignored HIP hardware tests. The owning design document records that separate
manual gate and its exact expected result.

Acceptance is exit 0 with the final line
``DTYPE DIRECT ARITHMETIC ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/dtype_direct_arithmetic_oracle.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


class OracleFailure(RuntimeError):
    """A failed source-contract obligation."""


@dataclass(frozen=True)
class SourceContract:
    name: str
    path: str
    required: tuple[str, ...]


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


def source_contracts() -> tuple[SourceContract, ...]:
    """Return the direct-identity locks exercised by standing mutations."""

    return (
        SourceContract(
            "direct Sub lowering",
            "crates/chelis-ir/src/tier2.rs",
            (
                "add_synth(dag, RiscOp::Sub, vec![a, b], ty.clone(), parent_span)",
            ),
        ),
        SourceContract(
            "direct MinElem lowering",
            "crates/chelis-ir/src/tier2.rs",
            (
                "add_synth(dag, RiscOp::MinElem, vec![a, b], ty.clone(), parent_span)",
            ),
        ),
        SourceContract(
            "C checked subtraction",
            "crates/chelis-backend-c/src/emit.rs",
            (
                "chelis_int_checked_sub",
                '"-" => "sub",',
            ),
        ),
        SourceContract(
            "C stored-bit extrema",
            "crates/chelis-backend-c/src/emit.rs",
            (
                'let comparison = if func.contains("max") { ">=" } else { "<=" };',
                "(!isnan({rhs}) && ({lhs}) {comparison} ({rhs})) ? ({lhs}) : ({rhs})",
                "? __in_g_{id}[i] : UINT16_C(0);",
            ),
        ),
        SourceContract(
            "HIP stored-bit extrema",
            "crates/chelis-backend-hip/src/kernels.rs",
            (
                'let comparison = if is_max { ">=" } else { "<=" };',
                "bool select_left = isnan(av) || (!isnan(bv) && av {comparison} bv);",
                "out[i] = select_left ? av : bv;",
                "out[i] = {selected} ? g[idx_g] : {zero};",
            ),
        ),
        SourceContract(
            "target dispositions",
            "crates/chelis-compiler-api/src/compiler.rs",
            (
                "chelis_types::unimplemented_rejection!(\n                    1306,\n                    \"the Metal direct-subtraction/extrema",
                "checked signed-integer subtraction needs an exact HIP overflow-trap channel",
                "the HIP bf16/f16 direct-subtraction/extrema bit-preserving kernels are not implemented",
            ),
        ),
        SourceContract(
            "WireDag v6 identities",
            "crates/chelis-compiler-api/src/schema.rs",
            (
                "pub const WIRE_DAG_SCHEMA_VERSION: u32 = 6;",
                "pub enum WireFusedStepOp {\n    Add,\n    Sub,",
                "MaxElem,\n    MinElem,\n    ExtremaAdjoint {",
                'r#"{\"kind\":\"extrema_adjoint\",\"extrema\":\"max\",\"operand\":\"left\"}"#',
            ),
        ),
    )


def validate_source_contracts(repo_root: Path = REPO_ROOT) -> None:
    """Fail if any direct identity or exact-selection lock is absent."""

    for contract in source_contracts():
        path = repo_root / contract.path
        try:
            source = path.read_text()
        except OSError as error:
            raise OracleFailure(
                f"{contract.name}: cannot read {contract.path}: {error}"
            ) from error
        for required in contract.required:
            if required not in source:
                raise OracleFailure(
                    f"{contract.name}: required contract is absent from {contract.path}"
                )


def oracle_legs(python: str) -> tuple[OracleLeg, ...]:
    """Return the frozen chelis#1306 command manifest in execution order."""

    return (
        OracleLeg(
            "oracle self-tests and standing mutations",
            (python, "scripts/test_dtype_direct_arithmetic_oracle.py"),
        ),
        OracleLeg(
            "exact typed scalar and tensor kernels",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "-E",
                "test(/(direct_subtraction|extrema)/)",
            ),
        ),
        OracleLeg(
            "direct IR lowering, evaluation, and AD",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "--test",
                "issue_1306_direct_arithmetic",
            ),
        ),
        OracleLeg(
            "direct constant folding",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "-E",
                "test(/constant_fold_direct_/)",
            ),
        ),
        OracleLeg(
            "exact WireDag v6 decoder contract",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "wire_dag_v6_direct_arithmetic",
            ),
        ),
        OracleLeg(
            "wire and target disposition contracts",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "-E",
                "test(/direct_arithmetic/)",
            ),
        ),
        OracleLeg(
            "compiled C exact-value and trap behavior",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-c",
                "--test",
                "exec_compile",
                "direct_",
            ),
        ),
        OracleLeg(
            "HIP structural contracts",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-hip",
                "--test",
                "codegen_structure",
                "direct_",
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
        raise SystemExit(f"DTYPE DIRECT ARITHMETIC ORACLE: FAIL: {error}") from error

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
                f"DTYPE DIRECT ARITHMETIC ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE DIRECT ARITHMETIC ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
