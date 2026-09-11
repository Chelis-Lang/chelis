#!/usr/bin/env python3
"""Structural chelis#1291 HIP and Metal Count acceptance oracle.

This oracle proves that direct Count entries and host-program Count helpers use
dedicated device kernels over the exact Bool8 carrier, never a Sum alias,
C-host fallback, or Metal stub. It does not claim numerical device acceptance:
that belongs to the named ignored ``count_`` tests in each backend's
``gpu_correctness.rs`` (the HIP and Metal hardware gates chelis#1291 owns).

Usage:

    .venv/bin/python scripts/dtype_count_device_oracle.py

Acceptance is exit 0 with the final line
``DTYPE COUNT DEVICE STRUCTURAL ORACLE: PASS``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
HIP_EMIT_SOURCE = Path("crates/chelis-backend-hip/src/emit.rs")
HIP_LIB_SOURCE = Path("crates/chelis-backend-hip/src/lib.rs")
METAL_EMIT_SOURCE = Path("crates/chelis-backend-metal/src/emit.rs")
METAL_LIB_SOURCE = Path("crates/chelis-backend-metal/src/lib.rs")
MUTATION_SOURCES = (
    HIP_EMIT_SOURCE,
    HIP_LIB_SOURCE,
    METAL_EMIT_SOURCE,
    METAL_LIB_SOURCE,
)
PASS_LINE = "DTYPE COUNT DEVICE STRUCTURAL ORACLE: PASS"


class OracleFailure(RuntimeError):
    """A failed chelis#1291 structural oracle obligation."""


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class MutationLeg:
    name: str
    source: Path
    mutate: Callable[[str], str]
    argv: tuple[str, ...]
    required_evidence: tuple[str, ...]


def oracle_legs() -> tuple[OracleLeg, ...]:
    """Return the default device-source acceptance manifest."""

    return (
        OracleLeg(
            "dedicated HIP Count kernel",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-hip",
                "--test",
                "hip_count_codegen",
            ),
        ),
        OracleLeg(
            "dedicated Metal Count kernel",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-metal",
                "--test",
                "metal_count_codegen",
            ),
        ),
        OracleLeg(
            "compiler API device Count artifacts",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "issue_1291_count_device_helpers_api",
            ),
        ),
        OracleLeg(
            "CLI device Count artifacts and example parity",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "issue_1291_count_device_helpers_cli",
                "--test",
                "issue_1291_count_device_entry",
                "--test",
                "parity",
                "-E",
                "binary(issue_1291_count_device_helpers_cli) | "
                "binary(issue_1291_count_device_entry) | "
                "test(parity_count_bool_device_entry_library_only) | "
                "test(parity_corpus_is_complete)",
            ),
        ),
    )


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _replace_exactly_once(source: str, old: str, new: str, owner: str) -> str:
    count = source.count(old)
    if count != 1:
        raise OracleFailure(
            f"{owner} mutation anchor drifted: expected 1, found {count}: {old!r}"
        )
    return source.replace(old, new, 1)


def mutate_hip_kernel_dispatch(source: str) -> str:
    """Alias HIP Count launch dispatch to Sum so the source test must fail."""

    return _replace_exactly_once(
        source,
        'Some(format!("kernel_count_{}", node.id.0))',
        'Some(format!("kernel_sum_{}", node.id.0))',
        "HIP Count dispatch",
    )


def mutate_metal_kernel_dispatch(source: str) -> str:
    """Alias Metal Count emission to Sum so the source test must fail."""

    return _replace_exactly_once(
        source,
        'format!("k_count_{}", node.id.0)',
        'format!("k_reduce_sum_{}", node.id.0)',
        "Metal Count dispatch",
    )


def mutate_host_helper_selection(source: str) -> str:
    """Drop Count from device helper selection, restoring C-host emission."""

    return _replace_exactly_once(
        source,
        ".any(|node| matches!(node.op, chelis_ir::dag::RiscOp::Count { .. }))",
        ".any(|_node| false)",
        "Count host-helper selection",
    )


@contextmanager
def temporary_mutation(path: Path, mutate: Callable[[str], str]) -> Iterator[None]:
    original = path.read_bytes()
    path.write_text(mutate(original.decode("utf-8")), encoding="utf-8")
    try:
        yield
    finally:
        path.write_bytes(original)
        if path.read_bytes() != original:
            raise OracleFailure(f"failed to restore controlled mutation: {path}")


def mutation_legs() -> tuple[MutationLeg, ...]:
    hip_kernel_test = "hip_count_emits_one_dedicated_multi_axis_checked_balanced_kernel"
    metal_kernel_test = (
        "metal_count_emits_one_dedicated_multi_axis_checked_balanced_kernel"
    )
    hip_host_test = "hip_host_program_externalizes_every_count_helper"
    metal_host_test = (
        "metal_host_program_externalizes_every_count_helper_without_a_stub"
    )
    return (
        MutationLeg(
            "HIP Count-to-Sum dispatch alias",
            HIP_EMIT_SOURCE,
            mutate_hip_kernel_dispatch,
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-hip",
                "--test",
                "hip_count_codegen",
                "-E",
                f"test({hip_kernel_test})",
            ),
            (hip_kernel_test,),
        ),
        MutationLeg(
            "Metal Count-to-Sum dispatch alias",
            METAL_EMIT_SOURCE,
            mutate_metal_kernel_dispatch,
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-metal",
                "--test",
                "metal_count_codegen",
                "-E",
                f"test({metal_kernel_test})",
            ),
            (metal_kernel_test,),
        ),
        MutationLeg(
            "HIP Count helper C-host fallback",
            HIP_LIB_SOURCE,
            mutate_host_helper_selection,
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-hip",
                "--test",
                "hip_count_codegen",
                "-E",
                f"test({hip_host_test})",
            ),
            (hip_host_test,),
        ),
        MutationLeg(
            "Metal Count helper C-host fallback",
            METAL_LIB_SOURCE,
            mutate_host_helper_selection,
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-backend-metal",
                "--test",
                "metal_count_codegen",
                "-E",
                f"test({metal_host_test})",
            ),
            (metal_host_test,),
        ),
    )


def oracle_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "CARGO_TARGET_DIR",
        str(REPO_ROOT / "target" / "agents" / "dtype-count-device-oracle"),
    )
    return env


def assert_mutation_sources_clean(env: dict[str, str]) -> None:
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--", *(str(path) for path in MUTATION_SOURCES)),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure("could not inspect Count device mutation source status")
    if completed.stdout.strip():
        raise OracleFailure(
            "refusing to mutate dirty Count device owner source; commit or stash "
            "the HIP and Metal emitter changes"
        )


def run_success(command: Sequence[str], env: dict[str, str]) -> None:
    print(f"+ {command_text(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise OracleFailure(
            f"focused command failed with exit {completed.returncode}: "
            f"{command_text(command)}"
        )


def run_mutation(leg: MutationLeg, env: dict[str, str]) -> None:
    path = REPO_ROOT / leg.source
    print(
        f"+ plant {leg.name}; expect {command_text(leg.argv)} to fail",
        flush=True,
    )
    with temporary_mutation(path, leg.mutate):
        completed = subprocess.run(
            leg.argv,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = completed.stdout
        if completed.returncode == 0:
            raise OracleFailure(f"{leg.name} survived its structural test")
        missing = [item for item in leg.required_evidence if item not in output]
        if missing:
            tail = "\n".join(output.splitlines()[-80:])
            raise OracleFailure(
                f"{leg.name} failed for the wrong reason; missing {missing}:\n{tail}"
            )
    print("  mutation produced the expected structural test failure", flush=True)


def main() -> int:
    env = oracle_environment()
    try:
        for leg in oracle_legs():
            run_success(leg.argv, env)
        assert_mutation_sources_clean(env)
        for leg in mutation_legs():
            run_mutation(leg, env)
    except OracleFailure as error:
        print(f"DTYPE COUNT DEVICE STRUCTURAL ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    print(PASS_LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
