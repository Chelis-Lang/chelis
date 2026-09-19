#!/usr/bin/env python3
"""Authoritative direct nonnumeric lowering oracle for chelis#1284/#630/#666.

The always-run legs prove direct IR identity, IEEE comparison and eager
selection semantics, AD behavior, the current wire contract, exact compiled C
execution, HIP structural kernels/capability admission, and stable typed Metal
rejection. Real HIP execution remains the separate manual gate documented in
``docs/manual_gates.md``.

Acceptance is exit zero with final line
``DTYPE NONNUMERIC LOWERING ORACLE: PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


class OracleFailure(RuntimeError):
    """A direct nonnumeric source contract is absent."""


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
            "direct identities and fusion barriers",
            "crates/chelis-ir/src/dag.rs",
            (
                "Compare(ComparisonKind)",
                "Logical(LogicalKind)",
                "Where,",
            ),
        ),
        SourceContract(
            "HIP semantic comparison",
            "crates/chelis-backend-hip/src/emit.rs",
            (
                '"chelis_f16_to_f32(a[idx_a])"',
                '"chelis_bf16_to_f32(a[idx_a])"',
                "kernels::comparison(",
            ),
        ),
        SourceContract(
            "HIP stored-bit where",
            "crates/chelis-backend-hip/src/emit.rs",
            (
                'Prim::F32 => "chelis_u32",',
                'Prim::F64 => "chelis_u64",',
                "kernels::where_stored(",
            ),
            ("RiscOp::Where => kernels::binary_elementwise",),
        ),
        SourceContract(
            "HIP Bool8 kernels",
            "crates/chelis-backend-hip/src/kernels.rs",
            (
                "unsigned char *out",
                "pub fn logical_binary(",
                "pub fn logical_not(",
                "out[i] = cond[idx_cond] != 0 ? a[idx_a] : b[idx_b];",
            ),
        ),
        SourceContract(
            "Metal target disposition",
            "crates/chelis-compiler-api/src/compiler.rs",
            (
                "RiscOp::Compare(kind) => Some(format!",
                "chelis_types::unimplemented_rejection!(\n                    1284,",
                "use `--target c` or `--target hip`",
            ),
        ),
        SourceContract(
            "Metal emitter defense",
            "crates/chelis-backend-metal/src/emit.rs",
            (
                "reject_direct_nonnumeric(dag)?;",
                "Metal direct nonnumeric node {id} reached emission after the #1284 typed capability rejection",
            ),
        ),
        SourceContract(
            "current WireDag identities",
            "crates/chelis-compiler-api/src/schema.rs",
            (
                "Compare {\n        comparison: WireComparisonKind,",
                "Logical {\n        logical: WireLogicalKind,",
                "Where {},",
            ),
        ),
    )


def validate_source_contracts(repo_root: Path = REPO_ROOT) -> None:
    for contract in source_contracts():
        source = (repo_root / contract.path).read_text()
        for required in contract.required:
            if required not in source:
                raise OracleFailure(f"{contract.name}: missing {required!r}")
        for forbidden in contract.forbidden:
            if forbidden in source:
                raise OracleFailure(f"{contract.name}: forbidden {forbidden!r}")


def oracle_legs(python: str) -> tuple[OracleLeg, ...]:
    return (
        OracleLeg(
            "oracle self-tests and standing mutations",
            (python, "scripts/test_dtype_nonnumeric_lowering_oracle.py"),
        ),
        OracleLeg(
            "direct IR lowering evaluation eager semantics and AD",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "--test",
                "logical_bool_semantics",
            ),
        ),
        OracleLeg(
            "current WireDag identities",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "logical_comparison_where_wire",
            ),
        ),
        OracleLeg(
            "compiled C exact semantics and ownership forms",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-c",
                "--test",
                "exec_compile",
                "-E",
                "test(typed_comparison_c_matrix_matches_evaluator_for_every_identity_and_dtype) | "
                "test(typed_logical_c_truth_tables_are_bool8) | "
                "test(typed_where_c_copies_selected_storage_bits_for_every_admitted_dtype) | "
                "test(issue_630_eq_neq_owned_copied_and_borrowed_tensors_match_ieee) | "
                "test(issue_666_typed_gte_where_forward_and_gradient_match_selected_branch)",
            ),
        ),
        OracleLeg(
            "HIP structural kernels and target admission",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-hip",
                "-p",
                "chelis-compiler-api",
                "-E",
                "binary(logical_comparison_where) | "
                "binary(logical_comparison_where_targets)",
            ),
        ),
        OracleLeg(
            "Metal typed target and emitter rejection",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-metal",
                "-p",
                "chelis-compiler-api",
                "-E",
                "binary(logical_comparison_where_rejection) | "
                "binary(logical_comparison_where_targets)",
            ),
        ),
    )


def run_oracle(python: str = sys.executable) -> None:
    validate_source_contracts()
    legs = oracle_legs(python)
    for index, leg in enumerate(legs, start=1):
        print(f"[{index}/{len(legs)}] {leg.name}: {shlex.join(leg.argv)}", flush=True)
        try:
            subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"DTYPE NONNUMERIC LOWERING ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE NONNUMERIC LOWERING ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
