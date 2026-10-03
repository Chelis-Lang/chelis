#!/usr/bin/env python3
"""Authoritative [05-OP-43] ReLU completion oracle for chelis#1313.

The always-run legs prove the exact typed semantics, dedicated IR/AD identity,
current WireDag contract, compiled C execution, and HIP/Metal code generation.
The owning design document names the separate hardware execution gates.

Acceptance is exit 0 with the final line ``DTYPE RELU ORACLE: PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


class OracleFailure(RuntimeError):
    """A structural ReLU contract is absent."""


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
            "dedicated lowering",
            "crates/chelis-ir/src/tier2.rs",
            ("add_synth(owner, dag, RiscOp::Relu, vec![x], ty.clone(), parent_span)",),
            ("lower_max_elem(owner, dag, x, zero",),
        ),
        SourceContract(
            "dedicated adjoint",
            "crates/chelis-ir/src/grad.rs",
            ("let dx = dag.add_node(node.owner, RiscOp::ReluAdjoint, vec![x, g]",),
            ("RiscOp::MaxElem => \"max_elem\",\n        RiscOp::Relu =>",),
        ),
        SourceContract(
            "sealed typed semantics",
            "crates/chelis-types/src/dtype_semantics.rs",
            (
                "pub fn float_relu(input: &TensorStorage)",
                "pub fn float_relu_adjoint(",
                "if $zero < input { g } else { $zero }",
            ),
        ),
        SourceContract(
            "C stored-bit selection",
            "crates/chelis-backend-c/src/emit.rs",
            (
                'self.emit_unary_func(id, "chelis_relu", inputs, ty)',
                'self.emit_binary(id, "chelis_relu_adjoint", inputs, ty)',
                'format!("({value}) < {zero} ? {zero} : ({value})")',
                'format!("__av < 0.0f ? UINT16_C(0) : {raw}")',
                'return format!("{zero} < ({lhs}) ? ({rhs}) : {zero}")',
                'format!("0.0f < __av ? {g_raw} : UINT16_C(0)")',
            ),
            ("fmaxf(__in_",),
        ),
        SourceContract(
            "exact numeric authority registrations",
            "crates/chelis-cli/tests/capacity_census_tripwire.rs",
            (
                "[compiler-builtin-numeric] relu(input: &tensor[D, p]) -> tensor[D, p]",
                "[compiler-risc-numeric] relu_adjoint(input: &tensor[D, p], cotangent:",
                "fn relu_identities_are_registered_against_their_exact_authority_atom()",
            ),
        ),
        SourceContract(
            "C scalar stored-value selection",
            "crates/chelis-backend-c/src/host_emit.rs",
            (
                "chelis_host_relu_{suffix}",
                "HostType::Float16 | HostType::BFloat16 => EmittedExpr::conditional(",
                "BinaryOperator::Less,\n                                numeric_arg(0),\n                                EmittedExpr::integer(0),",
                "EmittedExpr::integer(0),\n                            arg(0),",
                '"    return x < {zero} ? {zero} : x;"',
                'assert!(!body.contains("fmax")',
            ),
            ("stored_relu_decoder",),
        ),
        SourceContract(
            "HIP all-width selection",
            "crates/chelis-backend-hip/src/kernels.rs",
            (
                "pub fn relu(rank: usize, kernel_name: &str, kind: ElemKind)",
                "pub fn relu_reduced(",
                "pub fn relu_adjoint_reduced(",
                "out[i] = is_positive ? g[idx_g] : (unsigned short)0;",
            ),
        ),
        SourceContract(
            "Metal admitted-width selection",
            "crates/chelis-backend-metal/src/emit.rs",
            (
                "RiscOp::Add | RiscOp::Mul | RiscOp::ReluAdjoint",
                "out[tid] = a[tid] < ({msl_ty})0 ? ({msl_ty})0 : a[tid]",
                "out[tid] = ({msl_ty})0 < a[tid] ? b[tid] : ({msl_ty})0",
            ),
        ),
        SourceContract(
            "current exact wire identities",
            "crates/chelis-compiler-api/src/schema.rs",
            (
                "Relu,\n    ReluAdjoint,",
                "relu_wire_contract_rejects_wrong_arity_dtype_and_shape",
            ),
        ),
        SourceContract(
            "no stale closed-issue rejection receipt",
            "crates/chelis-compiler-api/src/compiler.rs",
            (),
            ("unimplemented chelis#1313", "unimplemented_rejection!(\n                    1313,"),
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
            (python, "scripts/test_dtype_relu_oracle.py"),
        ),
        OracleLeg(
            "sealed typed scalar/tensor semantics",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-types",
                "-E",
                "test(relu_)",
            ),
        ),
        OracleLeg(
            "dedicated IR lowering evaluation and AD",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-ir",
                "--test",
                "issue_1313_relu_adjoint",
            ),
        ),
        OracleLeg(
            "wire and target contracts",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "-E",
                "test(relu_)",
            ),
        ),
        OracleLeg(
            "compiled C exact bits and scalar selector",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-c",
                "-E",
                "test(direct_relu_) | test(host_scalar_relu_) | test(scalar_activation_names_have_closed_expression_identities_and_all_width_helpers)",
            ),
        ),
        OracleLeg(
            "HIP all-width structural kernels",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-hip",
                "--test",
                "codegen_structure",
                "-E",
                "test(dedicated_relu_)",
            ),
        ),
        OracleLeg(
            "Metal admitted-width structural kernels",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-metal",
                "--test",
                "dtype_matrix",
                "-E",
                "test(relu_)",
            ),
        ),
        OracleLeg(
            "numeric surface authority registrations",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
                "-E",
                "test(relu_identities_are_registered_)",
            ),
        ),
        OracleLeg(
            "generated rejection registry agreement",
            (python, "scripts/generate_rejection_registries.py", "--check"),
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
                f"DTYPE RELU ORACLE: FAIL: {leg.name} (exit {error.returncode})"
            ) from error
    print("DTYPE RELU ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
