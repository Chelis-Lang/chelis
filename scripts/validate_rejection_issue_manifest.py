#!/usr/bin/env python3
"""Validate source derivation, then live-check all [05-UNS-5] issue rows.

Success is exit 0 with final line ``REJECTION ISSUE MANIFEST: PASS``.
The REST issues endpoint is required because it distinguishes issues from pull
requests. Every standing row is fetched; tracker unavailability fails closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from capacity_census_liveness import IssueKind, IssueRecord, IssueState, fetch_issue
from generate_rejection_registries import (
    MANIFEST_REL,
    derive_issue_numbers,
    discover_production_sources,
    discover_production_workspace,
    load_issue_manifest,
    manifest_derivation_problems,
    parse_production_issue_citations,
    verify_compiler_source_closure,
)


class SourceManifestError(ValueError):
    """The checked-in manifest disagrees with fresh production sources."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


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


def validate_source_manifest(root: Path) -> tuple[list[int], list[dict]]:
    """Freshly validate source derivation and compiler closure."""
    manifest_path = root / MANIFEST_REL
    numbers = load_issue_manifest(manifest_path)
    workspace = discover_production_workspace(root)
    sources = discover_production_sources(root, workspace)
    verify_compiler_source_closure(root, sources, workspace)
    source_numbers = derive_issue_numbers(parse_production_issue_citations(sources))
    derivation_problems = manifest_derivation_problems(numbers, source_numbers)
    if derivation_problems:
        raise SourceManifestError(derivation_problems)
    rows = json.loads(manifest_path.read_text())["issues"]
    return numbers, rows


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        numbers, rows = validate_source_manifest(root)
    except SourceManifestError as error:
        for problem in error.problems:
            print(problem, file=sys.stderr)
        print("REJECTION ISSUE MANIFEST: FAIL", file=sys.stderr)
        return 1
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
