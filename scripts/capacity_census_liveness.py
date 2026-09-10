#!/usr/bin/env python3
"""Capacity census liveness gate (chelis#729, dtype_semantics.md section C6).

The owning tripwires verify discovered rows against their final authority
and, for families that still retain it, sealed foundation-era legacy debt.
This script checks that every issue cited by an active legacy disposition
is still OPEN. Wire baseline version 2 has no legacy admission: this script
validates its final-row shape before any issue lookup. The live wire verifier
owns graph completeness, semantic authority, and execution evidence; a valid
baseline shape does not prove those obligations.
The binding baseline likewise separates structural and executed CompilerJson
comparison rows from its four remaining native legacy signatures. Its shape
cannot replace the binding gate's current registration and codec execution.

Network gate (run by the change-gated and nightly liveness jobs; it also runs
manually at release cuts and during red-team passes):

    .venv/bin/python scripts/capacity_census_liveness.py

Success is exit 0 with the final line `CAPACITY CENSUS LIVENESS: PASS`.
Run it at every release cut and in red-team passes over the numeric surface.
Issue-bound legacy dispositions name at least one chelis#N reference. Exact
foundation-era dispositions are closed, family-specific strings backed by
immutable complete-row universes in the owning Rust tripwires. Active legacy
sets may shrink as rows acquire final authority, but no new or changed identity
may inherit one of these dispositions.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

REPO = "Chelis-Lang/chelis"
CENSUS_RELS = (
    Path("spec/design/capacity_census.json"),
    Path("spec/design/capacity_census_wire.json"),
    Path("spec/design/capacity_census_bindings.json"),
)
ISSUE_REF = re.compile(r"chelis#(\d+)")
LEGACY_TRANSITION_DISPOSITIONS_BY_FAMILY = {
    "bindings": frozenset(
        {
            "permanent-disposition(C6 registered PyO3 signature surface complete descriptor set ratified 2026-08-04)",
        }
    ),
}
LEGACY_TRANSITION_DISPOSITIONS = frozenset(
    disposition
    for dispositions in LEGACY_TRANSITION_DISPOSITIONS_BY_FAMILY.values()
    for disposition in dispositions
)
WIRE_ROW_KEYS = frozenset({"kind", "id", "flags", "authority", "contract"})


def wire_row_problem(row: object) -> str | None:
    """Validate persisted final-row shape, not the claimed contract's authority."""
    if not isinstance(row, dict) or row.keys() != WIRE_ROW_KEYS:
        return "WIRE BASELINE row requires exactly kind/id/flags/authority/contract; no legacy fields"
    identity = row["id"]
    if (
        row["kind"] != "wire-schema-numeric-field"
        or not isinstance(identity, str)
        or not identity.strip()
    ):
        return "WIRE BASELINE row must identify a wire-schema numeric field"
    if row["flags"] not in (["float-carrier"], ["numeric-field"]):
        return f"WIRE BASELINE row requires exactly one numeric capacity flag: {identity}"
    if row["authority"] not in ("TaggedTransport", "NumericOperation"):
        return f"WIRE BASELINE row requires TaggedTransport or NumericOperation: {identity}"
    contract = row["contract"]
    if not isinstance(contract, str) or not contract.strip():
        return f"WIRE BASELINE row requires a final-authority contract: {identity}"
    if row["authority"] == "NumericOperation" and not re.fullmatch(
        r"\[05-OP-[0-9]+\]", contract
    ):
        return f"WIRE BASELINE numeric operation contract requires [05-OP-N]: {identity}"
    return None


def validate_wire_baseline(payload: object) -> None:
    """Reject version/shape drift and duplicate identities without trusting a count.

    Source digests, graph identities, and execution receipts are fresh verifier
    output. They cannot be supplied by this persisted version-2 baseline.
    """
    if (
        not isinstance(payload, dict)
        or payload.keys() != {"version", "rows"}
        or type(payload["version"]) is not int
        or payload["version"] != 2
        or not isinstance(payload["rows"], list)
    ):
        raise ValueError("WIRE BASELINE requires exactly version: 2 and rows: [...]; no legacy fields")
    identities: set[str] = set()
    for row in payload["rows"]:
        problem = wire_row_problem(row)
        if problem:
            raise ValueError(problem)
        if row["id"] in identities:
            raise ValueError(f"WIRE BASELINE DUPLICATE identity: {row['id']}")
        identities.add(row["id"])


def wire_json_object(pairs: list[tuple[str, object]]) -> dict:
    """Do not let a repeated JSON key overwrite a version or row contract."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"WIRE BASELINE DUPLICATE JSON key: {key}")
        result[key] = value
    return result


def validate_binding_baseline(payload: object) -> None:
    """Check comparison/legacy shape; the Rust gate owns exact row admission."""
    if (not isinstance(payload, dict) or payload.keys() != {"version", "rows"}
            or type(payload["version"]) is not int or payload["version"] != 2
            or not isinstance(payload["rows"], list)):
        raise ValueError("BINDING BASELINE requires version: 2 and rows only")
    base = {"kind", "id", "flags"}
    native = {"chelis_python::CompiledModel::__call__", "chelis_python::NativeTensor::shape",
              "chelis_python::NativeTensor::__dlpack__", "chelis_python::NativeTensor::__dlpack_device__"}
    compiler_json = {"chelis_python::" + name for name in ("check_json", "compile_json", "desugar_json", "eval_json")}
    seen = set()
    for row in payload["rows"]:
        if not isinstance(row, dict) or not base <= row.keys():
            raise ValueError("BINDING BASELINE row requires kind/id/flags")
        name, kind, flags = row["id"], row["kind"], row["flags"]
        if (not isinstance(name, str) or not name or kind not in ("binding-pyfunction", "binding-pymethod")
                or not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags)
                or flags != sorted(set(flags))):
            raise ValueError("BINDING BASELINE invalid identity or capacity flags")
        if name in seen:
            raise ValueError("BINDING BASELINE duplicate identity")
        seen.add(name)
        if "citation" in row:
            if (row.keys() != base | {"citation"} or name.split("(", 1)[0] not in native
                    or kind != "binding-pymethod" or row["citation"] not in LEGACY_TRANSITION_DISPOSITIONS_BY_FAMILY["bindings"]):
                raise ValueError("BINDING BASELINE only unchanged native rows may retain legacy shape")
            continue
        keys = base | {"authority", "graph_identity"}
        if row.get("authority") == "TaggedTransport":
            owner = name.split("(", 1)[0]
            if (row.keys() != keys | {"contract"} or owner not in compiler_json
                    or kind != "binding-pyfunction" or row["contract"] != "compiler-json/" + owner
                    or not flags or not set(flags) <= {"float-carrier", "numeric-param", "numeric-return"}):
                raise ValueError("BINDING BASELINE invalid CompilerJson comparison row")
        elif row.keys() != keys or row.get("authority") != "nonnumeric" or flags:
            raise ValueError("BINDING BASELINE invalid structural comparison row")
        if not isinstance(row["graph_identity"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["graph_identity"]):
            raise ValueError("BINDING BASELINE missing graph identity")


def census_family(census_rel: Path) -> str:
    """Return the closed family identity for a checked census artifact."""
    if census_rel.name == "capacity_census.json":
        return "primary"
    if census_rel.name == "capacity_census_wire.json":
        return "wire"
    if census_rel.name == "capacity_census_bindings.json":
        return "bindings"
    return census_rel.stem


class IssueKind(Enum):
    """GitHub object kind returned by the REST issues endpoint."""

    ISSUE = "ISSUE"
    PULL_REQUEST = "PULL REQUEST"


class IssueState(Enum):
    """Only states relevant to the liveness contract."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class IssueRecord:
    """Typed liveness input: an OPEN pull request is not an OPEN issue."""

    kind: IssueKind
    state: IssueState


def extract_issue_refs(citation: str) -> list[int]:
    """Issue numbers referenced by a citation string, in order, deduplicated."""
    seen: list[int] = []
    for match in ISSUE_REF.finditer(citation):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def load_census_rows(
    root: Path,
    census_rels: tuple[Path, ...] = CENSUS_RELS,
) -> list[dict]:
    """Load census rows, validating wire final-only shape before issue lookup.

    The original covered-family baseline stores citations per row because
    individual legacy seams can have different owners. A legacy typed family
    may share one baseline-level citation; a row citation remains stronger.
    The wire baseline rejects either citation location, including empty ones.
    """
    rows: list[dict] = []
    for census_rel in census_rels:
        family = census_family(census_rel)
        payload = json.loads(
            (root / census_rel).read_text(),
            object_pairs_hook=wire_json_object if family in {"wire", "bindings"} else None,
        )
        if family == "wire":
            validate_wire_baseline(payload)
        elif family == "bindings":
            validate_binding_baseline(payload)
        inherited_citation = str(payload.get("citation", "")).strip()
        for source_row in payload["rows"]:
            row = dict(source_row)
            row["_census_family"] = family
            if not str(row.get("citation", "")).strip() and inherited_citation:
                row["citation"] = inherited_citation
            rows.append(row)
    return rows


def adjudicate(rows: list[dict], issues: dict[int, IssueRecord]) -> list[str]:
    """Pure verdict logic: return the list of problems (empty means pass).

    `issues` maps a cited GitHub number to its typed REST record. GitHub's
    issues endpoint also returns pull requests, identified by the
    `pull_request` field, so kind is part of the verdict rather than discarded.
    """
    problems: list[str] = []
    for row in rows:
        family = str(row.get("_census_family", "")).strip()
        if family == "wire":
            problem = wire_row_problem(
                {key: value for key, value in row.items() if key != "_census_family"}
            )
            if problem:
                problems.append(problem)
            continue
        citation = str(row.get("citation", "")).strip()
        row_id = f"[{row.get('kind', '?')}] {row.get('id', '?')}"
        if not citation:
            continue
        if citation == "TODO":
            problems.append(f"TODO legacy disposition (Rust tripwire should have caught this): {row_id}")
            continue
        family_dispositions = LEGACY_TRANSITION_DISPOSITIONS_BY_FAMILY.get(
            family, frozenset()
        )
        if citation in family_dispositions:
            continue
        if citation in LEGACY_TRANSITION_DISPOSITIONS:
            problems.append(
                f"WRONG CENSUS FAMILY: permanent disposition does not belong to "
                f"{family or '<missing>'}: {row_id}"
            )
            continue
        issue_numbers = extract_issue_refs(citation)
        if not issue_numbers:
            problems.append(
                f"UNRECOGNIZED disposition (expected an exact permanent disposition "
                f"or chelis#N): {row_id}"
            )
            continue
        for number in issue_numbers:
            issue = issues.get(number)
            if issue is None:
                problems.append(f"UNRESOLVABLE citation chelis#{number}: {row_id}")
            elif issue.kind is not IssueKind.ISSUE:
                problems.append(
                    f"WRONG OBJECT KIND: chelis#{number} is a "
                    f"{issue.kind.value}, not an OPEN issue: {row_id}"
                )
            elif issue.state is not IssueState.OPEN:
                problems.append(
                    f"STALE citation: chelis#{number} is {issue.state.value}, "
                    f"so this row's "
                    f"justification no longer stands and it must be re-adjudicated "
                    f"(un-census the surface, or re-cite live work): {row_id}"
                )
    return problems


def fetch_issue(
    number: int,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> IssueRecord | None:
    """Query GitHub's REST issues endpoint; return None on any invalid lookup.

    The injectable `run` seam is the unit-test oracle. `gh issue view` is not
    used because it accepts pull-request numbers while hiding object kind.
    """
    result = run(
        ["gh", "api", f"repos/{REPO}/issues/{number}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        state = IssueState(str(payload["state"]).upper())
        kind = (
            IssueKind.PULL_REQUEST
            if "pull_request" in payload
            else IssueKind.ISSUE
        )
        return IssueRecord(kind=kind, state=state)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        rows = load_census_rows(root)
    except (OSError, ValueError) as error:
        print(f"capacity census baseline violation: {error}")
        print("CAPACITY CENSUS LIVENESS: FAIL")
        return 1
    legacy_rows = [
        row for row in rows if str(row.get("citation", "")).strip()
    ]

    numbers = sorted(
        {
            number
            for row in legacy_rows
            for number in extract_issue_refs(str(row.get("citation", "")))
        }
    )
    resolved = {n: fetch_issue(n) for n in numbers}
    issues = {n: issue for n, issue in resolved.items() if issue is not None}
    problems = adjudicate(rows, issues)

    if problems:
        print(
            "capacity census liveness violation "
            "(spec/design/dtype_semantics.md section C6; chelis#729):"
        )
        for problem in problems:
            print(f"  {problem}")
        print("CAPACITY CENSUS LIVENESS: FAIL")
        return 1
    print(f"checked {len(legacy_rows)} active legacy rows, {len(numbers)} cited issues, all OPEN")
    print("CAPACITY CENSUS LIVENESS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
