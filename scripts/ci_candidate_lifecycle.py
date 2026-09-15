#!/usr/bin/env python3
"""Reject undeclared base updates and history rewrites on PR candidates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Protocol


SHA = re.compile(r"[0-9a-f]{40}\Z")


class GraphInspectionError(ValueError):
    """The event history cannot be inspected from the available Git objects."""


class Graph(Protocol):
    def is_ancestor(self, ancestor: str, descendant: str) -> bool: ...

    def merge_base(self, left: str, right: str) -> str: ...

    def new_merge_commits(
        self, before: str, head: str
    ) -> list[tuple[str, tuple[str, ...]]]: ...


class GitGraph:
    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = cwd

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.cwd,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        completed = self._run(
            "merge-base", "--is-ancestor", ancestor, descendant, check=False
        )
        if completed.returncode == 0:
            return True
        if completed.returncode == 1:
            return False
        detail = completed.stderr or completed.stdout
        raise GraphInspectionError(
            f"cannot compare candidate history {ancestor}..{descendant}: {detail}"
        )

    def merge_base(self, left: str, right: str) -> str:
        try:
            completed = self._run("merge-base", left, right)
        except subprocess.CalledProcessError as error:
            detail = error.stderr or error.stdout
            raise GraphInspectionError(
                f"cannot find merge base for {left} and {right}: {detail}"
            ) from error
        value = completed.stdout.strip()
        if not SHA.fullmatch(value):
            raise GraphInspectionError(
                f"git merge-base returned an invalid SHA: {value!r}"
            )
        return value

    def new_merge_commits(
        self, before: str, head: str
    ) -> list[tuple[str, tuple[str, ...]]]:
        try:
            completed = self._run(
                "rev-list", "--merges", "--parents", f"{before}..{head}"
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr or error.stdout
            raise GraphInspectionError(
                f"cannot enumerate candidate merges {before}..{head}: {detail}"
            ) from error
        rows: list[tuple[str, tuple[str, ...]]] = []
        for raw in completed.stdout.splitlines():
            fields = raw.split()
            if len(fields) < 3 or any(not SHA.fullmatch(value) for value in fields):
                raise GraphInspectionError(f"invalid merge-commit row: {raw!r}")
            rows.append((fields[0], tuple(fields[1:])))
        return rows


def _sha(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"pull_request synchronize payload requires {key}")
    return value


def _nested_sha(payload: Mapping[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            raise ValueError(
                "pull_request synchronize payload requires " + ".".join(keys)
            )
        current = current.get(key)
    if not isinstance(current, str) or not SHA.fullmatch(current):
        raise ValueError(
            "pull_request synchronize payload requires " + ".".join(keys)
        )
    return current


def classify_update(
    *, before: str, head: str, base: str, graph: Graph
) -> str:
    if graph.is_ancestor(before, head):
        for _commit, parents in graph.new_merge_commits(before, head):
            if any(graph.is_ancestor(parent, base) for parent in parents[1:]):
                return "base-merge"
        return "review-repair"

    old_base = graph.merge_base(before, base)
    new_base = graph.merge_base(head, base)
    if old_base != new_base and graph.is_ancestor(old_base, new_base):
        return "base-rebase"
    return "history-rewrite"


def _require_acknowledgement(
    body: str, *, prefix: str, head: str
) -> None:
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith(prefix)
    ]
    if len(lines) != 1:
        raise ValueError(
            f"{prefix} requires exactly one exact-head line in the PR body"
        )
    match = re.fullmatch(
        re.escape(prefix) + r" ([0-9a-f]{40}) (.+\S)",
        lines[0],
    )
    if match is None:
        raise ValueError(
            f"{prefix} must name the current head and a nonempty reason"
        )
    acknowledged_head = match.group(1)
    if acknowledged_head != head:
        raise ValueError(
            f"{prefix} names {acknowledged_head}, not the current head {head}"
        )


def validate_payload(
    payload: Mapping[str, Any],
    graph: Graph | None = None,
    *,
    current_pr: Mapping[str, Any] | None = None,
) -> str:
    action = payload.get("action")
    if action != "synchronize":
        return "initial-candidate" if action == "opened" else "unchanged-candidate"

    before = _sha(payload, "before")
    after = _sha(payload, "after")
    event_pr = payload.get("pull_request")
    if not isinstance(event_pr, Mapping):
        raise ValueError("pull_request synchronize payload requires pull_request")
    pr = current_pr if current_pr is not None else event_pr
    head = _nested_sha(pr, "head", "sha")
    base = _nested_sha(pr, "base", "sha")
    if after != head:
        raise ValueError(
            f"pull_request synchronize after {after} does not equal current PR head {head}"
        )
    body = pr.get("body")
    if body is None:
        body = ""
    if not isinstance(body, str):
        raise ValueError("pull_request body must be text or null")

    try:
        kind = classify_update(
            before=before,
            head=head,
            base=base,
            graph=graph or GitGraph(),
        )
    except GraphInspectionError:
        _require_acknowledgement(
            body,
            prefix="Candidate-history-rewrite:",
            head=head,
        )
        return "history-rewrite-unverifiable"
    if kind in {"base-merge", "base-rebase"}:
        _require_acknowledgement(
            body,
            prefix="Candidate-base-update:",
            head=head,
        )
    elif kind == "history-rewrite":
        _require_acknowledgement(
            body,
            prefix="Candidate-history-rewrite:",
            head=head,
        )
    return kind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument(
        "--current-pr",
        type=Path,
        help="optional live pull-request JSON used instead of the event snapshot",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(args.event.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a JSON object")
        current_pr = None
        if args.current_pr is not None:
            current_pr = json.loads(args.current_pr.read_text(encoding="utf-8"))
            if not isinstance(current_pr, dict):
                raise ValueError("current PR payload must be a JSON object")
        kind = validate_payload(payload, current_pr=current_pr)
    except (OSError, json.JSONDecodeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"CANDIDATE LIFECYCLE: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"CANDIDATE LIFECYCLE: PASS: {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
