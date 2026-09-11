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
    forbidden: tuple[str, ...] = ()


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
            "C Bool extrema avoid floating classification",
            "crates/chelis-backend-c/src/emit.rs",
            (
                "if ty.precision.is_integer() || matches!(ty.precision, Prim::Bool) {",
            ),
            (
                'if ty.precision.is_integer() {\n            format!("(({lhs})',
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
            "current WireDag identities",
            "crates/chelis-compiler-api/src/schema.rs",
            (
                "pub const WIRE_DAG_SCHEMA_VERSION: u32 = 11;",
                "pub enum WireFusedStepOp {\n    Add,\n    Sub,",
                "MaxElem,\n    MinElem,\n    ExtremaAdjoint {",
                'r#"{\"kind\":\"extrema_adjoint\",\"extrema\":\"max\",\"operand\":\"left\"}"#',
            ),
        ),
        SourceContract(
            "current direct arithmetic surface narrative",
            "docs/CHELIS_SURFACE.md",
            (
                "complete `g` to the exact operand selected by [05-OP-40]",
                "Tier-1 DAG:   add sub mul div floor_div trunc_div max_elem min_elem cmplt",
            ),
            (
                "(g*(x>=y), g*(x<y))",
                "| `sub` | `add(a, neg(b))` |",
                "| `min_elem` | `neg(max_elem(neg(a), neg(b)))` |",
            ),
        ),
        SourceContract(
            "current Deep direct subtraction narrative",
            "spec/03-deep-syntax.md",
            ("sub remains a direct Tier-1 RISC identity after lowering",),
            ("sub is a derived built-in, lowered to add(a, neg(b))",),
        ),
        SourceContract(
            "current captured direct arithmetic narrative",
            "openspec/changes/capture-risc-primitives/specs/risc-primitives/spec.md",
            (
                "it remains a direct `RiscOp::Sub` identity during IR construction",
                "it remains a direct `RiscOp::MinElem` selection identity",
            ),
            (
                "it becomes `add(a, neg(b))` during IR construction",
                "with `sub` decomposed rather than present as a node",
            ),
        ),
        SourceContract(
            "current canonical direct arithmetic narrative",
            "spec/design/chelis_canonical_reference.md",
            ("`sub` and `min_elem` are direct Tier-1 RISC identities",),
            ("`min_elem` is part of the specified derived built-in surface",),
        ),
        SourceContract(
            "current exactness regression narrative",
            "crates/chelis-cli/tests/issue_680_int_exactness.rs",
            ("direct `min_elem` compares the stored int64 operands",),
            ("`min_elem` lowers via `neg(max_elem(neg, neg))`",),
        ),
        SourceContract(
            "current precision matrix narrative",
            "crates/chelis-cli/tests/precision_matrix.rs",
            (
                "Direct `max_elem` compares both int64 operands at their declared width",
                "Direct `min_elem` compares both int64 operands at their declared width",
            ),
            (
                "Verified: returns the SMALLER operand",
                "`min_elem` passes today BY LUCK",
                "PASSES BY LUCK: operands collapse to one f64",
            ),
        ),
        SourceContract(
            "current inferred-shape direct arithmetic narrative",
            "crates/chelis-types/src/infer/validate.rs",
            ("`max_elem` and `min_elem` are direct Tier-1 identities",),
            ("`max_elem` (Tier 1) and `min_elem` (Tier 2)",),
        ),
        SourceContract(
            "current runtime direct arithmetic narrative",
            "crates/chelis-compiler-api/src/runtime/eval.rs",
            ("Tier-1 `max_elem` and `min_elem` are direct element-wise",),
            ("Tier-1 `max_elem` and Tier-2 `min_elem`",),
        ),
        SourceContract(
            "current builtin direct arithmetic grouping",
            "crates/chelis-types/src/builtins.rs",
            (
                'tensor_binop("add", &mut env, &mut vg);\n    tensor_binop("sub", &mut env, &mut vg);\n    tensor_binop("mul", &mut env, &mut vg);',
                'tensor_binop("max_elem", &mut env, &mut vg);\n    tensor_binop("min_elem", &mut env, &mut vg);',
            ),
            (
                '// Tier 2: Derived built-ins\n    tensor_binop("sub", &mut env, &mut vg);',
                'tensor_binop_to_out("matmul", &mut env, &mut vg);\n    tensor_binop("min_elem", &mut env, &mut vg);',
            ),
        ),
        SourceContract(
            "current lowering direct arithmetic grouping",
            "crates/chelis-ir/src/lower.rs",
            (
                "// Direct Tier-1 subtraction identity",
                "// Direct Tier-1 minimum selection identity",
                "// --- Direct Tier-1 MinElem lowering ---",
            ),
            ("// Tier 2 decompositions\n            \"sub\"",),
        ),
        SourceContract(
            "current Tier-2 helper module boundary",
            "crates/chelis-ir/src/tier2.rs",
            (
                "Most functions in this module decompose Tier 2 derived operations",
                "`lower_sub` and `lower_min_elem` emit direct Tier-1 identities",
            ),
            ("These functions decompose Tier 2 (derived) operations",),
        ),
        SourceContract(
            "current gradient direct subtraction narrative",
            "crates/chelis-ir/src/grad.rs",
            ("sign = pos - neg_cast (direct Tier-1 Sub)",),
            ("sign = pos - neg_cast  (tier2 sub)",),
        ),
        SourceContract(
            "current span-survival direct subtraction narrative",
            "crates/chelis-ir/tests/per_pass_span_propagation.rs",
            ("direct Tier-1 sub identity",),
            ("tier-2 sub (decomposes)",),
        ),
        SourceContract(
            "current host-runtime direct extrema narrative",
            "crates/chelis-compiler-api/tests/issue_185_host_runtime_binary.rs",
            (
                "direct Tier-1 extrema identities",
                "exact stored-operand selection required by [05-OP-40]",
            ),
            (
                "Binary Tier 2",
                "binary_map(.., f64::max)",
                "binary_map(.., f64::min)",
            ),
        ),
        SourceContract(
            "current fused direct-extrema narrative",
            "crates/chelis-ir/tests/fusion_adversarial.rs",
            (
                "ADV-10: MaxElem in fused chain preserves exact selected-operand semantics",
            ),
            (
                "ADV-10: MaxElem in fused chain uses fmaxf",
            ),
        ),
        SourceContract(
            "retired direct-arithmetic timing identities",
            "scripts/test_timing_baseline.json",
            (),
            (
                "max_elem_emits_fmaxf",
                "spec_sub_decomposes_to_add_neg",
                "lower_min_elem_decomposes",
                "lower_sub_decomposes",
                "min_elem_produces_neg_max_neg",
                "sub_produces_add_neg",
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
        for forbidden in contract.forbidden:
            if forbidden in source:
                raise OracleFailure(
                    f"{contract.name}: obsolete contract remains in {contract.path}"
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
            "exact current WireDag decoder contract",
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
