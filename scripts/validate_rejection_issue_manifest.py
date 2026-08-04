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
REPO_OWNER, REPO_NAME = REPO.split("/", 1)
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


CLOSING_ISSUES_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 100, after: $cursor) {
        nodes { number repository { nameWithOwner } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
""".strip()


def _source_closing_references(source: ClosingSource) -> set[int]:
    references: set[int] = set()
    for match in CLOSING_REFERENCE.finditer(source.text):
        raw = match.group("local") or match.group("qualified") or match.group("url")
        references.add(int(raw))
    return references


def find_closing_references(sources: list[ClosingSource]) -> set[int]:
    """Return all issue numbers a PR title, body, or commit would close."""
    return set().union(*(_source_closing_references(source) for source in sources), set())


def _read_pull_request_event(event_path: Path) -> tuple[dict, int]:
    try:
        event = json.loads(event_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read pull-request event: {error}") from error
    pull_request = event.get("pull_request")
    if not isinstance(pull_request, dict):
        raise RuntimeError("event is not a pull-request event")
    number = pull_request.get("number") or event.get("number")
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise RuntimeError("pull-request event has no valid PR number")
    return pull_request, number


def collect_pull_request_closing_issues(
    number: int,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> set[int]:
    """Read GitHub's authoritative, paginated closing-issue relation."""
    issues: set[int] = set()
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        args = [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={CLOSING_ISSUES_QUERY}",
            "-f",
            f"owner={REPO_OWNER}",
            "-f",
            f"name={REPO_NAME}",
            "-F",
            f"number={number}",
        ]
        if cursor is not None:
            args.extend(["-f", f"cursor={cursor}"])
        result = run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot fetch PR #{number} closing issues: {result.stderr.strip()}"
        )
        try:
            payload = json.loads(result.stdout)
            graphql_errors = payload.get("errors", [])
            if not isinstance(graphql_errors, list):
                raise TypeError("invalid GraphQL errors field")
            if graphql_errors:
                raise TypeError("GraphQL errors accompanied a partial response")
            connection = payload["data"]["repository"]["pullRequest"][
                "closingIssuesReferences"
            ]
            nodes = connection["nodes"]
            page_info = connection["pageInfo"]
            has_next = page_info["hasNextPage"]
            end_cursor = page_info["endCursor"]
            if not isinstance(nodes, list) or not isinstance(has_next, bool):
                raise TypeError("invalid nodes/pageInfo")
            for node in nodes:
                issue_number = node["number"]
                name_with_owner = node["repository"]["nameWithOwner"]
                if (
                    not isinstance(issue_number, int)
                    or isinstance(issue_number, bool)
                    or issue_number <= 0
                    or not isinstance(name_with_owner, str)
                ):
                    raise TypeError("invalid issue node")
                if name_with_owner.casefold() == REPO.casefold():
                    issues.add(issue_number)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"malformed closing-issue response for PR #{number}: {error}"
            ) from error
        if not has_next:
            return issues
        if (
            not isinstance(end_cursor, str)
            or not end_cursor
            or end_cursor in seen_cursors
        ):
            raise RuntimeError(
                f"malformed closing-issue response for PR #{number}: "
                "pagination cursor is missing or repeated"
            )
        seen_cursors.add(end_cursor)
        cursor = end_cursor


def collect_pull_request_sources(
    event_path: Path | None = None,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> list[ClosingSource]:
    """Load closing-capable text from the current pull-request event.

    Issue comments and repository prose are intentionally absent: they are
    historical context, not merge inputs. The title is included because this
    repository uses it as the squash commit title. Commit messages are fetched
    because GitHub can apply their closing keywords when they reach ``main``.
    """
    if event_path is None:
        raw = os.environ.get("GITHUB_EVENT_PATH", "").strip()
        if not raw:
            return []
        event_path = Path(raw)
    pull_request, number = _read_pull_request_event(event_path)
    title = pull_request.get("title")
    body = pull_request.get("body")
    if not isinstance(title, str):
        raise RuntimeError("pull-request event has no valid title")
    if body is not None and not isinstance(body, str):
        raise RuntimeError("pull-request event has no valid body")
    expected_commits = pull_request.get("commits")
    if (
        not isinstance(expected_commits, int)
        or isinstance(expected_commits, bool)
        or expected_commits < 0
    ):
        raise RuntimeError("pull-request event has no valid commit count")

    sources = [
        ClosingSource("pull request title", title),
        ClosingSource("pull request body", body or ""),
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
        fetched_commits = 0
        seen_shas: set[str] = set()
        if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
            raise TypeError("commit pages are not lists")
        for page in pages:
            for commit in page:
                full_sha = commit["sha"]
                if (
                    not isinstance(full_sha, str)
                    or re.fullmatch(r"[0-9a-fA-F]{40}", full_sha) is None
                    or full_sha in seen_shas
                ):
                    raise TypeError("invalid or duplicate commit SHA")
                seen_shas.add(full_sha)
                fetched_commits += 1
                sha = full_sha[:12]
                message = commit["commit"]["message"]
                if not isinstance(message, str):
                    raise TypeError("invalid commit message")
                sources.append(ClosingSource(f"commit {sha}", message))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"malformed commit response for PR #{number}: {error}") from error
    if fetched_commits != expected_commits:
        raise RuntimeError(
            f"incomplete commit response for PR #{number}: expected "
            f"{expected_commits} commits, received {fetched_commits}"
        )
    return sources


def adjudicate_closing_references(
    rows: list[dict], sources: list[ClosingSource]
) -> list[str]:
    """Reject closing refs that would strand executable authorities."""
    problems: list[str] = []
    for source in sources:
        problems.extend(
            adjudicate_closing_issue_numbers(
                rows, _source_closing_references(source), source.label
            )
        )
    return problems


def adjudicate_closing_issue_numbers(
    rows: list[dict], numbers: set[int], label: str
) -> list[str]:
    """Reject an authoritative closing set that strands live sites."""
    by_number = {int(row["number"]): row for row in rows}
    problems: list[str] = []
    for number in sorted(numbers):
        row = by_number.get(number)
        if row is None:
            continue
        for site in row.get("sites", []):
            problems.append(
                f"CLOSING REFERENCE CONFLICT: {label} closes chelis#{number}, "
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
    raw_event_path = os.environ.get("GITHUB_EVENT_PATH", "").strip()
    if raw_event_path:
        event_path = Path(raw_event_path)
        try:
            _, pull_request_number = _read_pull_request_event(event_path)
            sources = collect_pull_request_sources(event_path)
            closing_issues = collect_pull_request_closing_issues(pull_request_number)
        except RuntimeError as error:
            problems.append(f"UNRESOLVABLE pull-request closing references: {error}")
        else:
            problems.extend(adjudicate_closing_references(rows, sources))
            problems.extend(
                adjudicate_closing_issue_numbers(
                    rows, closing_issues, "GitHub closingIssuesReferences"
                )
            )
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
