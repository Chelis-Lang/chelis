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


def _acknowledgement_heads(body: str, *, prefix: str) -> list[str]:
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
    return [match.group(1)]


def _require_acknowledgement(
    body: str,
    *,
    prefix: str,
    head: str,
    graph: Graph,
    exact: bool = True,
) -> None:
    acknowledged_head = _acknowledgement_heads(body, prefix=prefix)[0]
    if exact and acknowledged_head != head:
        raise ValueError(
            f"{prefix} names {acknowledged_head}, not the current head {head}"
        )
    if not exact and not graph.is_ancestor(acknowledged_head, head):
        raise ValueError(
            f"{prefix} names {acknowledged_head}, which is not the current "
            f"head {head} or its ancestor"
        )


def _timeline_events(
    timeline: Sequence[Any] | None,
) -> list[Mapping[str, Any]]:
    if timeline is None:
        return []
    events: list[Mapping[str, Any]] = []
    for item in timeline:
        if isinstance(item, Mapping):
            events.append(item)
            continue
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for event in item:
                if not isinstance(event, Mapping):
                    raise ValueError("pull request timeline events must be objects")
                events.append(event)
            continue
        raise ValueError("pull request timeline must contain event objects or pages")
    return events


def _require_persistent_history_declaration(
    *,
    body: str,
    head: str,
    target_tip: str,
    timeline: Sequence[Any] | None,
    graph: Graph,
) -> None:
    base_merge = any(
        any(graph.is_ancestor(parent, target_tip) for parent in parents[1:])
        for _commit, parents in graph.new_merge_commits(target_tip, head)
    )
    if base_merge:
        _require_acknowledgement(
            body,
            prefix="Candidate-base-update:",
            head=head,
            graph=graph,
            exact=False,
        )

    force_push_seen = False
    for event in _timeline_events(timeline):
        if event.get("event") != "head_ref_force_pushed":
            continue
        commit_id = event.get("commit_id")
        if not isinstance(commit_id, str) or not SHA.fullmatch(commit_id):
            raise ValueError(
                "head_ref_force_pushed timeline event requires commit_id"
            )
        force_push_seen = True
    if not force_push_seen:
        return

    present = 0
    for prefix in ("Candidate-base-update:", "Candidate-history-rewrite:"):
        if not any(
            line.strip().startswith(prefix) for line in body.splitlines()
        ):
            continue
        present += 1
        _require_acknowledgement(
            body,
            prefix=prefix,
            head=head,
            graph=graph,
            exact=False,
        )
    if present == 0:
        raise ValueError(
            "a recorded force push requires a persistent "
            "Candidate-base-update: or Candidate-history-rewrite: line"
        )


def validate_payload(
    payload: Mapping[str, Any],
    graph: Graph | None = None,
    *,
    current_pr: Mapping[str, Any] | None = None,
    target_tip: str | None = None,
    timeline: Sequence[Any] | None = None,
) -> str:
    action = payload.get("action")
    event_pr = payload.get("pull_request")
    if not isinstance(event_pr, Mapping):
        if action is None:
            return "unchanged-candidate"
        raise ValueError("pull_request payload requires pull_request")
    pr = current_pr if current_pr is not None else event_pr
    head = _nested_sha(pr, "head", "sha")
    base = target_tip or _nested_sha(pr, "base", "sha")
    if not SHA.fullmatch(base):
        raise ValueError("target_tip must be a lowercase 40-character commit SHA")
    body = pr.get("body")
    if body is None:
        body = ""
    if not isinstance(body, str):
        raise ValueError("pull_request body must be text or null")
    active_graph = graph or GitGraph()

    if action != "synchronize":
        _require_persistent_history_declaration(
            body=body,
            head=head,
            target_tip=base,
            timeline=timeline,
            graph=active_graph,
        )
        return "initial-candidate" if action == "opened" else "unchanged-candidate"

    before = _sha(payload, "before")
    after = _sha(payload, "after")
    if after != head:
        raise ValueError(
            f"pull_request synchronize after {after} does not equal current PR head {head}"
        )
    try:
        kind = classify_update(
            before=before,
            head=head,
            base=base,
            graph=active_graph,
        )
    except GraphInspectionError as inspection:
        # Keep the fail-safe direction: an update nobody can classify still
        # owes the stricter declaration. Say why, though. Without this the
        # author reads "Candidate-history-rewrite: requires exactly one
        # exact-head line" and concludes they forgot to write one, when the
        # real state is that this checkout never obtained the pre-push head
        # (chelis#2229).
        try:
            _require_acknowledgement(
                body,
                prefix="Candidate-history-rewrite:",
                head=head,
                graph=active_graph,
            )
        except ValueError as missing:
            raise ValueError(
                f"the pre-push head {before} could not be inspected "
                f"({inspection}), so this update cannot be classified as a "
                "base update and the stricter declaration is required: "
                f"{missing}"
            ) from inspection
        return "history-rewrite-unverifiable"
    if kind in {"base-merge", "base-rebase"}:
        _require_acknowledgement(
            body,
            prefix="Candidate-base-update:",
            head=head,
            graph=active_graph,
        )
    elif kind == "history-rewrite":
        _require_acknowledgement(
            body,
            prefix="Candidate-history-rewrite:",
            head=head,
            graph=active_graph,
        )
    _require_persistent_history_declaration(
        body=body,
        head=head,
        target_tip=base,
        timeline=timeline,
        graph=active_graph,
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
    parser.add_argument("--target-tip")
    parser.add_argument(
        "--timeline",
        type=Path,
        help="optional paginated pull-request timeline JSON",
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
        timeline = None
        if args.timeline is not None:
            timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
            if not isinstance(timeline, list):
                raise ValueError("pull request timeline must be a JSON array")
        kind = validate_payload(
            payload,
            current_pr=current_pr,
            target_tip=args.target_tip,
            timeline=timeline,
        )
    except (OSError, json.JSONDecodeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"CANDIDATE LIFECYCLE: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"CANDIDATE LIFECYCLE: PASS: {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
