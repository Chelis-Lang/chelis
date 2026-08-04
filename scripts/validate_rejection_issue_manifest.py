#!/usr/bin/env python3
"""Live-validate [05-UNS-5] issue identities and open state.

Success is exit 0 with final line ``REJECTION ISSUE MANIFEST: PASS``.
The REST issues endpoint is required because it distinguishes issues from pull
requests. Tracker unavailability fails closed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from capacity_census_liveness import IssueKind, IssueRecord, IssueState, fetch_issue
from generate_rejection_registries import MANIFEST_REL, load_issue_manifest


REPO = "Chelis-Lang/chelis"
CLOSING_REFERENCE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:"
    r"#(?P<local>[1-9][0-9]*)|"
    r"Chelis-Lang/chelis#(?P<qualified>[1-9][0-9]*)|"
    r"https://github\.com/Chelis-Lang/chelis/issues/(?P<url>[1-9][0-9]*))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClosingSource:
    """One GitHub-controlled text surface that can close an issue."""

    label: str
    text: str


def _source_closing_references(source: ClosingSource) -> set[int]:
    references: set[int] = set()
    for match in CLOSING_REFERENCE.finditer(source.text):
        raw = match.group("local") or match.group("qualified") or match.group("url")
        references.add(int(raw))
    return references


def find_closing_references(sources: list[ClosingSource]) -> set[int]:
    """Return all issue numbers a PR title/body/commit would close."""
    return set().union(*(_source_closing_references(source) for source in sources), set())


def collect_pull_request_sources(
    event_path: Path | None = None,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> list[ClosingSource]:
    """Load closing-capable text from the current pull-request event.

    Issue comments and repository prose are intentionally absent: they are
    historical context, not merge inputs. Commit messages are fetched because
    GitHub can apply their closing keywords when the commit reaches ``main``.
    """
    if event_path is None:
        raw = os.environ.get("GITHUB_EVENT_PATH", "").strip()
        if not raw:
            return []
        event_path = Path(raw)
    try:
        event = json.loads(event_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read pull-request event: {error}") from error
    pull_request = event.get("pull_request")
    if not isinstance(pull_request, dict):
        return []
    number = pull_request.get("number") or event.get("number")
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise RuntimeError("pull-request event has no valid PR number")

    sources = [
        ClosingSource("pull request title", str(pull_request.get("title") or "")),
        ClosingSource("pull request body", str(pull_request.get("body") or "")),
    ]
    result = run(
        [
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPO}/pulls/{number}/commits?per_page=100",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot fetch PR #{number} commit messages: {result.stderr.strip()}"
        )
    try:
        pages = json.loads(result.stdout)
        for page in pages:
            for commit in page:
                sha = str(commit["sha"])[:12]
                message = str(commit["commit"]["message"])
                sources.append(ClosingSource(f"commit {sha}", message))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"malformed commit response for PR #{number}: {error}") from error
    return sources


def adjudicate_closing_references(
    rows: list[dict], sources: list[ClosingSource]
) -> list[str]:
    """Reject closing refs that would strand executable authorities."""
    by_number = {int(row["number"]): row for row in rows}
    problems: list[str] = []
    for source in sources:
        for number in sorted(_source_closing_references(source)):
            row = by_number.get(number)
            if row is None:
                continue
            for site in row.get("sites", []):
                problems.append(
                    f"CLOSING REFERENCE CONFLICT: {source.label} closes chelis#{number}, "
                    "but executable authority remains at "
                    f"{site['path']}:{site['line']}; rehome the constructor before merge"
                )
    return problems


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
    try:
        sources = collect_pull_request_sources()
    except RuntimeError as error:
        problems.append(f"UNRESOLVABLE pull-request closing references: {error}")
    else:
        problems.extend(adjudicate_closing_references(rows, sources))
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
