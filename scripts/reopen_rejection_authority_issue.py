#!/usr/bin/env python3
"""Reopen a closed issue while executable rejection authorities still cite it.

The pull-request liveness check rejects closing relationships visible when that
check runs. GitHub does not emit a pull-request workflow activity when someone
adds a manual sidebar closing link, so this closed-issue guard is the
compensating backstop: it derives authorities from the current default branch
and restores the required OPEN state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from generate_rejection_registries import (
    AuthoritySite,
    discover_issue_authorities,
)


REPO = "Chelis-Lang/chelis"


def read_closed_issue_event(path: Path) -> int:
    """Return the issue number from a strict GitHub ``issues.closed`` event."""
    try:
        payload = json.loads(path.read_text())
        issue = payload["issue"]
        number = issue["number"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"cannot read closed-issue event: {error}") from error
    if payload.get("action") != "closed":
        raise RuntimeError("event action is not closed")
    if not isinstance(issue, dict) or "pull_request" in issue:
        raise RuntimeError("event object is not an issue")
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise RuntimeError("closed-issue event has no valid issue number")
    return number


def _run_gh(
    args: list[str], *, run: Callable[..., Any] = subprocess.run
) -> subprocess.CompletedProcess[str]:
    result = run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"GitHub command failed: {args}")
    return result


def _read_issue_state(
    number: int, *, run: Callable[..., Any] = subprocess.run
) -> str:
    result = _run_gh(
        ["gh", "api", f"repos/{REPO}/issues/{number}"],
        run=run,
    )
    try:
        payload = json.loads(result.stdout)
        state = payload["state"]
        returned_number = payload["number"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"malformed issue response for chelis#{number}: {error}"
        ) from error
    if returned_number != number or state not in {"open", "closed"}:
        raise RuntimeError(f"malformed issue response for chelis#{number}")
    if "pull_request" in payload:
        raise RuntimeError(f"chelis#{number} is a pull request, not an issue")
    return state


def _format_sites(sites: list[AuthoritySite]) -> str:
    rendered: list[str] = []
    for site in sites:
        if (
            not isinstance(site, AuthoritySite)
            or not isinstance(site.path, str)
            or not site.path
            or not isinstance(site.line, int)
            or isinstance(site.line, bool)
            or site.line <= 0
        ):
            raise TypeError(f"malformed authority site: {site!r}")
        rendered.append(f"- `{site.path}:{site.line}`")
    return "\n".join(rendered)


def _validate_authorities(
    authorities: object,
) -> dict[int, list[AuthoritySite]]:
    """Validate the complete discovery result before trusting absence."""
    if type(authorities) is not dict:
        raise TypeError("authority inventory is not an ordinary mapping")
    for authority_number, sites in authorities.items():
        if (
            not isinstance(authority_number, int)
            or isinstance(authority_number, bool)
            or authority_number <= 0
        ):
            raise TypeError(f"malformed authority number: {authority_number!r}")
        if not isinstance(sites, list):
            raise TypeError(
                f"authority sites for chelis#{authority_number} are not a list"
            )
        _format_sites(sites)
    return authorities


def _reopen(
    number: int,
    reason: str,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    result = _run_gh(
        [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{REPO}/issues/{number}",
            "-f",
            "state=open",
        ],
        run=run,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"malformed reopen response for chelis#{number}: {error}"
        ) from error
    if payload.get("number") != number or payload.get("state") != "open":
        raise RuntimeError(f"GitHub did not confirm chelis#{number} was reopened")

    body = (
        "Reopened automatically by the rejection-authority closure guard. "
        "This issue must remain open while executable rejection authorities "
        "still depend on it.\n\n"
        f"{reason}"
    )
    _run_gh(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{REPO}/issues/{number}/comments",
            "-f",
            f"body={body}",
        ],
        run=run,
    )


def guard_closed_issue(
    root: Path,
    number: int,
    *,
    discover: Callable[[Path], dict[int, list[AuthoritySite]]] = (
        discover_issue_authorities
    ),
    run: Callable[..., Any] = subprocess.run,
) -> bool:
    """Reopen ``number`` when current source still depends on it.

    Returns ``True`` only when this invocation performed the reopen. Inventory
    failures are unsafe ambiguity and therefore reopen the issue fail-closed.
    """
    try:
        authorities = _validate_authorities(discover(root))
        sites = authorities.get(number, [])
        if not sites:
            return False
        reason = "Executable authority sites still present:\n\n" + _format_sites(sites)
    except Exception as error:
        reason = f"Authority inventory failed closed: `{error}`"

    if _read_issue_state(number, run=run) == "open":
        return False
    _reopen(number, reason, run=run)
    return True


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    raw_event_path = os.environ.get("GITHUB_EVENT_PATH", "").strip()
    if not raw_event_path:
        print(
            "REJECTION AUTHORITY CLOSURE GUARD: missing GITHUB_EVENT_PATH",
            file=sys.stderr,
        )
        return 1
    try:
        number = read_closed_issue_event(Path(raw_event_path))
        reopened = guard_closed_issue(root, number)
    except RuntimeError as error:
        print(f"REJECTION AUTHORITY CLOSURE GUARD: FAIL: {error}", file=sys.stderr)
        return 1
    if reopened:
        print(f"REJECTION AUTHORITY CLOSURE GUARD: REOPENED chelis#{number}")
    else:
        print(f"REJECTION AUTHORITY CLOSURE GUARD: PASS chelis#{number}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
