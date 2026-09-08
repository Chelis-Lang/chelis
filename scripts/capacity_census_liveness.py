#!/usr/bin/env python3
"""Capacity census liveness gate (chelis#729, dtype_semantics.md section C6).

The Rust tripwires prove that every discovered row either has exactly one
final authority or belongs to the exact sealed foundation-era legacy set.
This script checks the leg CI cannot: that every issue cited by an active
legacy disposition is still OPEN. Rows with final authority omit a citation
and are deliberately outside this liveness ledger.

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
    "primary": frozenset(
        {
            "permanent-disposition(C6 initial non-seam complete descriptor set ratified 2026-08-04)",
            "permanent-disposition([05-OP-2] source-faithful prelude Json numeric split; exact descriptor ratified 2026-08-04)",
        }
    ),
    "wire": frozenset(
        {
            "permanent-disposition(C6 dtype-tagged wire schema complete descriptor set ratified 2026-08-04)",
        }
    ),
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
    """Load all frozen census rows and apply a baseline-level citation.

    The original covered-family baseline stores citations per row because
    individual legacy seams can have different owners. The generated typed
    baselines share one citation, kept at the top level so generator output is
    entirely structural. A row-level citation, when present, remains stronger.
    """
    rows: list[dict] = []
    for census_rel in census_rels:
        payload = json.loads((root / census_rel).read_text())
        family = census_family(census_rel)
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
        citation = str(row.get("citation", "")).strip()
        row_id = f"[{row.get('kind', '?')}] {row.get('id', '?')}"
        if not citation:
            continue
        if citation == "TODO":
            problems.append(f"TODO legacy disposition (Rust tripwire should have caught this): {row_id}")
            continue
        family = str(row.get("_census_family", "")).strip()
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
    rows = load_census_rows(root)
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
