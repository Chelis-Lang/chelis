#!/usr/bin/env python3
"""Live-validate [05-UNS-5] issue identities and open state.

Success is exit 0 with final line ``REJECTION ISSUE MANIFEST: PASS``.
The REST issues endpoint is required because it distinguishes issues from pull
requests. Tracker unavailability fails closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from capacity_census_liveness import IssueKind, IssueRecord, IssueState, fetch_issue
from generate_rejection_registries import MANIFEST_REL, load_issue_manifest


def adjudicate(
    rows: list[dict], records: dict[int, IssueRecord]
) -> list[str]:
    """Return live-validation problems; an empty list is success."""
    problems: list[str] = []
    for row in rows:
        number = int(row["number"])
        record = records.get(number)
        if record is None:
            problems.append(f"UNRESOLVABLE authority: chelis#{number}")
        elif record.kind is not IssueKind.ISSUE:
            problems.append(
                f"WRONG OBJECT KIND: chelis#{number} is a {record.kind.value}, "
                "not an OPEN issue"
            )
        elif record.state is not IssueState.OPEN:
            problems.append(
                f"STALE authority: chelis#{number} is {record.state.value}, not OPEN"
            )
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    manifest_path = root / MANIFEST_REL
    numbers = load_issue_manifest(manifest_path)
    rows = json.loads(manifest_path.read_text())["issues"]
    resolved = {number: fetch_issue(number) for number in numbers}
    records = {
        number: record for number, record in resolved.items() if record is not None
    }
    problems = adjudicate(rows, records)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("REJECTION ISSUE MANIFEST: FAIL", file=sys.stderr)
        return 1
    print(f"checked {len(numbers)} issue authorities; all are OPEN issues")
    print("REJECTION ISSUE MANIFEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
