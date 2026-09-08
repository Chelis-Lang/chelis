#!/usr/bin/env python3
"""Flatten the inherited dtype Phase 0-3 nextest contract into one union.

The individual phase runners remain independently reproducible acceptance
commands. The continuous Phase 3 job, however, inherits all of them. Running
their nested Python drivers used to launch 21 nextest processes and execute
overlapping binaries such as parity, eval_agreement, precision_matrix, and
dtype_semantics more than once. This module derives one filterset from those
authoritative manifests so nextest discovers, schedules, and executes each
matching test once.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import dtype_phase0_oracle
import dtype_phase1_oracle
import dtype_phase2_oracle
import faithful_observation_phase3_oracle


REPO_ROOT = Path(__file__).resolve().parents[1]
OWNERS = (
    "dtype-phase0",
    "dtype-phase1",
    "dtype-phase2",
    "observation-phase3",
    "dtype-phase3",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_NEXTEST_PREFIX = ("cargo", "nextest", "run")


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class OwnedOracleLeg:
    owner: str
    name: str
    argv: tuple[str, ...]


_INHERITED_SCRIPTS = {
    "scripts/dtype_phase0_oracle.py": "dtype-phase0",
    "scripts/dtype_phase1_oracle.py": "dtype-phase1",
    "scripts/dtype_phase2_oracle.py": "dtype-phase2",
    "scripts/faithful_observation_phase3_oracle.py": "observation-phase3",
}


def phase3_legacy_legs(python: str) -> tuple[OracleLeg, ...]:
    """The Phase 3 leaf manifest retained for independent contract tests."""
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


def _manifest_legs(owner: str, python: str) -> tuple[OracleLeg, ...]:
    if owner == "dtype-phase0":
        source = dtype_phase0_oracle.oracle_legs()
    elif owner == "dtype-phase1":
        source = dtype_phase1_oracle.oracle_legs(python)
    elif owner == "dtype-phase2":
        source = dtype_phase2_oracle.oracle_legs(python)
    elif owner == "observation-phase3":
        source = tuple(
            OracleLeg(name, tuple(argv))
            for name, argv in faithful_observation_phase3_oracle.SUITE_COMMANDS
        )
    elif owner == "dtype-phase3":
        source = phase3_legacy_legs(python)
    else:
        raise ValueError(f"unknown dtype phase owner {owner!r}")
    return tuple(
        OracleLeg(str(getattr(leg, "name")), tuple(getattr(leg, "argv")))
        for leg in source
    )


def _inherited_owner(leg: OracleLeg, python: str) -> str | None:
    if len(leg.argv) != 2 or leg.argv[0] != python:
        return None
    return _INHERITED_SCRIPTS.get(leg.argv[1])


def owned_leaf_legs(python: str = sys.executable) -> tuple[OwnedOracleLeg, ...]:
    """Every recursively inherited executable leaf with its phase owner."""
    leaves: list[OwnedOracleLeg] = []

    def visit(owner: str, active: tuple[str, ...]) -> None:
        if owner in active:
            cycle = " -> ".join((*active, owner))
            raise ValueError(f"dtype oracle inheritance cycle: {cycle}")
        for leg in _manifest_legs(owner, python):
            inherited_owner = _inherited_owner(leg, python)
            if inherited_owner is None:
                leaves.append(OwnedOracleLeg(owner, leg.name, leg.argv))
            else:
                visit(inherited_owner, (*active, owner))

    visit("dtype-phase3", ())
    return tuple(leaves)


def owned_nextest_legs(python: str = sys.executable) -> tuple[OwnedOracleLeg, ...]:
    """Every inherited nextest selection, attributed to exactly one phase."""
    return tuple(
        leg for leg in owned_leaf_legs(python) if leg.argv[:3] == _NEXTEST_PREFIX
    )


def non_test_legs(python: str = sys.executable) -> tuple[OracleLeg, ...]:
    """Inherited executable obligations that are not nextest selections."""
    unique: dict[tuple[str, ...], OracleLeg] = {}
    for leg in owned_leaf_legs(python):
        if leg.argv[:3] != _NEXTEST_PREFIX:
            unique.setdefault(leg.argv, OracleLeg(leg.name, leg.argv))
    return tuple(unique.values())


def _exact_predicate(predicate: str, name: str) -> str:
    if _SAFE_NAME.fullmatch(name) is None:
        raise ValueError(f"unsafe nextest {predicate} name {name!r}")
    return f"{predicate}(={name})"


def command_filter(argv: tuple[str, ...]) -> str:
    """Translate one existing cargo-selection command to a filterset."""
    if argv[:3] != _NEXTEST_PREFIX:
        raise ValueError(f"not a nextest run command: {argv!r}")
    packages: list[str] = []
    binaries: list[str] = []
    filterset: str | None = None
    index = 3
    while index < len(argv):
        argument = argv[index]
        if argument in ("-p", "--package"):
            packages.append(argv[index + 1])
            index += 2
        elif argument == "--test":
            binaries.append(argv[index + 1])
            index += 2
        elif argument in ("-E", "--filterset"):
            if filterset is not None:
                raise ValueError(f"multiple filtersets in {argv!r}")
            filterset = argv[index + 1]
            index += 2
        else:
            raise ValueError(
                f"unrecognized dtype-oracle nextest selector {argument!r} "
                f"in {argv!r}"
            )

    components: list[str] = []
    if packages:
        components.append(
            "(" + " | ".join(_exact_predicate("package", name) for name in packages) + ")"
        )
    if binaries:
        components.append(
            "(" + " | ".join(_exact_predicate("binary", name) for name in binaries) + ")"
        )
    if filterset is not None:
        components.append(f"({filterset})")
    if not components:
        raise ValueError(f"nextest command selects no tests: {argv!r}")
    return "(" + " & ".join(components) + ")"


def flattened_filter(python: str = sys.executable) -> str:
    """The stable-order union of every inherited nextest selection."""
    filters = dict.fromkeys(
        command_filter(leg.argv) for leg in owned_nextest_legs(python)
    )
    return " | ".join(filters)


def owner_filter(owner: str, python: str = sys.executable) -> str:
    if owner not in OWNERS:
        raise ValueError(f"unknown dtype phase owner {owner!r}")
    filters = dict.fromkeys(
        command_filter(leg.argv)
        for leg in owned_nextest_legs(python)
        if leg.owner == owner
    )
    return " | ".join(filters)


def flattened_nextest_command(
    python: str = sys.executable,
) -> tuple[str, ...]:
    return (
        "cargo",
        "nextest",
        "run",
        "--workspace",
        "--profile",
        "ci-full",
        "--ignore-default-filter",
        "--no-fail-fast",
        "-E",
        flattened_filter(python),
    )


def ownership_receipt(python: str = sys.executable) -> dict[str, object]:
    leaves = owned_leaf_legs(python)
    legs = tuple(leg for leg in leaves if leg.argv[:3] == _NEXTEST_PREFIX)
    non_test = tuple(leg for leg in leaves if leg.argv[:3] != _NEXTEST_PREFIX)
    filterset = flattened_filter(python)
    owners: dict[str, list[dict[str, str]]] = {owner: [] for owner in OWNERS}
    non_test_owners: dict[str, list[dict[str, object]]] = {
        owner: [] for owner in OWNERS
    }
    for leg in legs:
        owners[leg.owner].append(
            {"name": leg.name, "filter": command_filter(leg.argv)}
        )
    for leg in non_test:
        non_test_owners[leg.owner].append(
            {"name": leg.name, "argv": list(leg.argv)}
        )
    return {
        "schema_version": 2,
        "nextest_invocations_before_flattening": len(legs),
        "nextest_invocations_after_flattening": 1,
        "non_test_invocations_before_flattening": len(non_test),
        "non_test_invocations_after_flattening": len(non_test_legs(python)),
        "filter_sha256": hashlib.sha256(filterset.encode("utf-8")).hexdigest(),
        "owners": owners,
        "non_test_owners": non_test_owners,
    }


def write_ownership_receipt(
    path: Path, python: str = sys.executable
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ownership_receipt(python), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
