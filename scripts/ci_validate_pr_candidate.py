#!/usr/bin/env python3
"""Validate an open pull request's exact head and optional planned base."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
from typing import Any

if __package__:
    from . import ci_change_owned
else:
    import ci_change_owned


SHA = re.compile(r"[0-9a-f]{40}\Z")


def _nested_sha(payload: Mapping[str, Any], key: str) -> str:
    nested = payload.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"pull request payload requires {key}.sha")
    value = nested.get("sha")
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"pull request payload requires a full lowercase {key}.sha")
    return value


def validate_candidate(
    payload: Mapping[str, Any],
    *,
    expected_head_sha: str,
    expected_base_sha: str | None = None,
    plan: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return the current ``(head, base)`` after fail-closed validation."""
    if not SHA.fullmatch(expected_head_sha):
        raise ValueError(
            "expected_head_sha must be a lowercase 40-character commit SHA"
        )
    state = payload.get("state")
    if not isinstance(state, str):
        raise ValueError("pull request payload requires state")
    if state != "open":
        raise ValueError(f"pull request is not open: state={state!r}")
    head = _nested_sha(payload, "head")
    base = _nested_sha(payload, "base")
    if head != expected_head_sha:
        raise ValueError(
            f"stale head: expected {expected_head_sha}, current head is {head}"
        )
    if expected_base_sha is not None:
        if not SHA.fullmatch(expected_base_sha):
            raise ValueError(
                "expected_base_sha must be a lowercase 40-character commit SHA"
            )
        if base != expected_base_sha:
            raise ValueError(
                f"stale base: expected {expected_base_sha}, current base is {base}"
            )

    if plan is not None:
        ci_change_owned.verify_plan_digest(plan)
        if plan["mode"] != "pull_request":
            raise ValueError("final candidate validation requires a pull_request plan")
        if plan["event_pr_head"] != head:
            raise ValueError(
                f"plan head {plan['event_pr_head']} does not equal current head {head}"
            )
        if plan["base_sha"] != base:
            raise ValueError(
                f"stale base: planned {plan['base_sha']}, current base is {base}"
            )
    return head, base


def validate_checkout(
    *,
    expected_head_sha: str,
    expected_base_sha: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Return the candidate SHA after validating exact merge parents."""
    for label, value in (
        ("expected_head_sha", expected_head_sha),
        ("expected_base_sha", expected_base_sha),
    ):
        if not SHA.fullmatch(value):
            raise ValueError(
                f"{label} must be a lowercase 40-character commit SHA"
            )
    try:
        completed = runner(
            ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or str(error)
        raise ValueError(f"cannot inspect candidate checkout: {detail}") from error
    fields = completed.stdout.strip().split()
    if len(fields) != 3 or not all(SHA.fullmatch(value) for value in fields):
        raise ValueError(
            "candidate checkout must be one commit with exactly two parents"
        )
    candidate_sha, first_parent, second_parent = fields
    if first_parent != expected_base_sha:
        raise ValueError(
            f"candidate first parent is {first_parent}, expected base "
            f"{expected_base_sha}"
        )
    if second_parent != expected_head_sha:
        raise ValueError(
            f"candidate second parent is {second_parent}, expected head "
            f"{expected_head_sha}"
        )
    return candidate_sha


def run(
    *,
    repository: str,
    pr_number: int,
    expected_head_sha: str,
    expected_base_sha: str | None = None,
    validate_checkout_parents: bool = False,
    plan_path: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, str]:
    if type(pr_number) is not int or pr_number <= 0:
        raise ValueError("pr_number must be a positive integer")
    if not repository or repository.startswith("/") or repository.endswith("/"):
        raise ValueError("repository must be an owner/name identifier")
    try:
        completed = runner(
            ["gh", "api", f"repos/{repository}/pulls/{pr_number}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or str(error)
        raise ValueError(f"cannot read pull request #{pr_number}: {detail}") from error
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid pull request JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("pull request JSON must be an object")

    plan = None
    if plan_path is not None:
        plan = ci_change_owned.load_json(plan_path)
    head, base = validate_candidate(
        payload,
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
        plan=plan,
    )
    if validate_checkout_parents:
        if expected_base_sha is None:
            raise ValueError(
                "--validate-checkout requires --expected-base-sha"
            )
        validate_checkout(
            expected_head_sha=head,
            expected_base_sha=expected_base_sha,
            runner=runner,
        )
    return head, base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--expected-base-sha")
    parser.add_argument("--validate-checkout", action="store_true")
    parser.add_argument("--plan", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    head, base = run(
        repository=args.repository,
        pr_number=args.pr_number,
        expected_head_sha=args.expected_head_sha,
        expected_base_sha=args.expected_base_sha,
        validate_checkout_parents=args.validate_checkout,
        plan_path=args.plan,
    )
    scope = "head"
    if args.expected_base_sha is not None:
        scope = "head and base"
    if args.validate_checkout:
        scope += " and synthetic merge parents"
    print(
        f"PR CANDIDATE: PASS: exact {scope}; "
        f"head={head}, base={base}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
