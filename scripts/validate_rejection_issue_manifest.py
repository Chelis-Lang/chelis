#!/usr/bin/env python3
"""Validate source derivation, then live-check [05-UNS-5] issue authorities.

Success is exit 0 with final line ``REJECTION ISSUE MANIFEST: PASS``.
The REST issues endpoint is required because it distinguishes issues from pull
requests. By default every standing row is fetched. PR mode admits only new
or modified rows against the exact synthetic merge's first parent. Source
disagreement, invalid comparison evidence, and tracker unavailability fail closed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from capacity_census_liveness import IssueKind, IssueRecord, IssueState, fetch_issue
from ci_change_owned import git_output, resolve_pr_commits
from generate_rejection_registries import (
    MANIFEST_REL,
    derive_issue_numbers,
    discover_production_sources,
    discover_production_workspace,
    load_issue_manifest,
    manifest_derivation_problems,
    parse_issue_manifest,
    parse_production_issue_citations,
    verify_compiler_source_closure,
)

ROOT = Path(__file__).resolve().parent.parent


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
    rows = [{"number": number, "kind": "issue", "state": "open"} for number in numbers]
    return numbers, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-head",
        help="event PR head SHA; require a matching synthetic merge and check changed rows only",
    )
    args = parser.parse_args(argv)
    try:
        base_numbers: set[int] | None = None
        if args.pr_head is not None:
            base, candidate = resolve_pr_commits(ROOT, "HEAD", args.pr_head)
            base_numbers = set(parse_issue_manifest(
                git_output(ROOT, ["show", f"{base}:{MANIFEST_REL.as_posix()}"]).decode()
            ))
            print(f"PR authority comparison: base={base} candidate={candidate}")
        numbers, rows = validate_source_manifest(ROOT)
        if base_numbers is not None:
            # Schema 1 fixes kind/state; a valid modified row is a new number.
            numbers = [number for number in numbers if number not in base_numbers]
            rows = [row for row in rows if row["number"] in numbers]
            print(f"selected changed issue authorities: {numbers}")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
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
    from observed_cargo import observed_cargo
    with observed_cargo():
        sys.exit(main())
