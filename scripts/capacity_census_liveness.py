#!/usr/bin/env python3
"""Capacity census liveness gate (chelis#729, dtype_semantics.md section C6).

The CI tripwire (crates/chelis-cli/tests/capacity_census_tripwire.rs) proves
the census matches the enumerated surface and that no row is uncited. This
script checks the leg CI cannot: that every cited issue is still OPEN. A row
citing a closed issue has lost its justification - the seam it excused was
supposedly unwound - and must be re-adjudicated (un-censused, or re-cited to
live work) rather than grandfathered forever. This is the known-red-ledger
pattern of the chelis#732 Phase 2 oracle, applied to the census.

Network gate (run by the change-gated and nightly liveness jobs; it also runs
manually at release cuts and during red-team passes):

    .venv/bin/python scripts/capacity_census_liveness.py

Success is exit 0 with the final line `CAPACITY CENSUS LIVENESS: PASS`.
Run it at every release cut and in red-team passes over the numeric surface.
Every sanctioned citation names at least one chelis#N reference, so every row
is liveness-bound: the seam rows cite chelis#893 and the plain baselines cite
chelis#729, which means the entire inventory comes up for re-adjudication when
the plan closes. The typed wire and PyO3 baselines carry one top-level citation
that is inherited by every generated row.
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
        inherited_citation = str(payload.get("citation", "")).strip()
        for source_row in payload["rows"]:
            row = dict(source_row)
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
        if not citation or citation == "TODO":
            problems.append(f"UNCITED row (CI tripwire should have caught this): {row_id}")
            continue
        for number in extract_issue_refs(citation):
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
        if type(payload) is not dict:
            return None
        returned_number = payload.get("number")
        if type(returned_number) is not int or returned_number <= 0:
            return None
        if returned_number != number:
            return None
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

    numbers = sorted({n for row in rows for n in extract_issue_refs(str(row.get("citation", "")))})
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
    print(f"checked {len(rows)} rows, {len(numbers)} cited issues, all OPEN")
    print("CAPACITY CENSUS LIVENESS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
