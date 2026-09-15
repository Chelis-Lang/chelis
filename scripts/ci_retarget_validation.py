#!/usr/bin/env python3
"""Dispatch and receipt exact-base implementation validation for a PR retarget."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import re
import subprocess
import time
from typing import Any

if __package__:
    from . import ci_validate_pr_candidate
else:
    import ci_validate_pr_candidate


SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
CHECK_NAME = "PR Base Retarget Validation"
WORKFLOWS = {
    "CI": "ci.yml",
    "Hull": "conformance.yml",
}

Api = Callable[[str, str, dict[str, Any] | None], Mapping[str, Any] | None]


def _require_string(
    payload: Mapping[str, Any], key: str, pattern: re.Pattern[str]
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"pull request payload requires valid {key}")
    return value


def validate_retarget(
    payload: Mapping[str, Any],
    *,
    expected_head_sha: str,
    expected_base_sha: str,
    expected_base_ref: str,
) -> None:
    ci_validate_pr_candidate.validate_candidate(
        payload,
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
    )
    base = payload.get("base")
    if not isinstance(base, Mapping):
        raise ValueError("pull request payload requires base.ref")
    base_ref = _require_string(base, "ref", REF)
    if base_ref != expected_base_ref:
        raise ValueError(
            f"stale base ref: expected {expected_base_ref}, current base is "
            f"{base_ref}"
        )


def _check_output(title: str, summary: str) -> dict[str, str]:
    return {"title": title, "summary": summary}


def _create_check(
    api: Api,
    *,
    repository: str,
    head_sha: str,
    token: str,
) -> int:
    response = api(
        "POST",
        f"repos/{repository}/check-runs",
        {
            "name": CHECK_NAME,
            "head_sha": head_sha,
            "status": "in_progress",
            "external_id": token,
            "output": _check_output(
                "Fresh base-retarget validation is running",
                "Compiler CI and Hull must both finish successfully against "
                "the exact new synthetic merge.",
            ),
        },
    )
    if not isinstance(response, Mapping) or type(response.get("id")) is not int:
        raise ValueError("check-run creation returned no integer id")
    return response["id"]


def _update_check(
    api: Api,
    *,
    repository: str,
    check_id: int,
    conclusion: str,
    title: str,
    summary: str,
) -> None:
    api(
        "PATCH",
        f"repos/{repository}/check-runs/{check_id}",
        {
            "status": "completed",
            "conclusion": conclusion,
            "output": _check_output(title, summary),
        },
    )


def _dispatch(
    api: Api,
    *,
    repository: str,
    workflow: str,
    base_ref: str,
    pr_number: int,
    head_sha: str,
    base_sha: str,
    token: str,
) -> None:
    api(
        "POST",
        f"repos/{repository}/actions/workflows/{workflow}/dispatches",
        {
            "ref": base_ref,
            "inputs": {
                "pr_number": str(pr_number),
                "expected_head_sha": head_sha,
                "expected_base_sha": base_sha,
                "retarget_token": token,
            },
        },
    )


def _matching_run(
    payload: Mapping[str, Any],
    *,
    label: str,
    token: str,
    base_sha: str,
) -> Mapping[str, Any] | None:
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise ValueError(f"{label} workflow response requires workflow_runs")
    title = f"Retarget {label} {token}"
    matches = [
        run
        for run in runs
        if isinstance(run, Mapping)
        and run.get("display_title") == title
        and run.get("head_sha") == base_sha
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate {label} retarget workflow runs")
    return matches[0] if matches else None


def run(
    *,
    repository: str,
    pr_number: int,
    expected_head_sha: str,
    expected_base_sha: str,
    expected_base_ref: str,
    token: str,
    api: Api,
    sleeper: Callable[[float], None] = time.sleep,
    poll_seconds: float = 10.0,
    max_polls: int = 240,
) -> dict[str, str]:
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name identifier")
    if type(pr_number) is not int or pr_number <= 0:
        raise ValueError("pr_number must be a positive integer")
    for label, value in (
        ("expected_head_sha", expected_head_sha),
        ("expected_base_sha", expected_base_sha),
    ):
        if not SHA.fullmatch(value):
            raise ValueError(f"{label} must be a lowercase 40-character SHA")
    if not REF.fullmatch(expected_base_ref):
        raise ValueError("expected_base_ref must be a simple branch ref")
    if not TOKEN.fullmatch(token):
        raise ValueError("token must be a simple stable identifier")
    if type(max_polls) is not int or max_polls <= 0:
        raise ValueError("max_polls must be positive")

    pull = api("GET", f"repos/{repository}/pulls/{pr_number}", None)
    if not isinstance(pull, Mapping):
        raise ValueError("pull request API returned no object")
    validate_retarget(
        pull,
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
        expected_base_ref=expected_base_ref,
    )

    check_id = _create_check(
        api,
        repository=repository,
        head_sha=expected_head_sha,
        token=token,
    )
    try:
        for workflow in WORKFLOWS.values():
            _dispatch(
                api,
                repository=repository,
                workflow=workflow,
                base_ref=expected_base_ref,
                pr_number=pr_number,
                head_sha=expected_head_sha,
                base_sha=expected_base_sha,
                token=token,
            )

        completed: dict[str, str] = {}
        for _ in range(max_polls):
            for label, workflow in WORKFLOWS.items():
                if label in completed:
                    continue
                response = api(
                    "GET",
                    f"repos/{repository}/actions/workflows/{workflow}/runs"
                    f"?event=workflow_dispatch&branch={expected_base_ref}"
                    "&per_page=20",
                    None,
                )
                if not isinstance(response, Mapping):
                    raise ValueError(f"{label} workflow API returned no object")
                observed = _matching_run(
                    response,
                    label=label,
                    token=token,
                    base_sha=expected_base_sha,
                )
                if observed is None:
                    continue
                status = observed.get("status")
                if status != "completed":
                    continue
                conclusion = observed.get("conclusion")
                if conclusion != "success":
                    raise ValueError(
                        f"{label} retarget validation concluded {conclusion!r}"
                    )
                url = observed.get("html_url")
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise ValueError(f"{label} workflow run has no valid URL")
                completed[label] = url
            if len(completed) == len(WORKFLOWS):
                summary = "\n".join(
                    f"- {label}: {url}" for label, url in completed.items()
                )
                _update_check(
                    api,
                    repository=repository,
                    check_id=check_id,
                    conclusion="success",
                    title="Fresh base-retarget validation passed",
                    summary=summary,
                )
                return completed
            sleeper(poll_seconds)
        missing = sorted(set(WORKFLOWS) - set(completed))
        raise ValueError(
            "retarget workflow run did not appear or complete: "
            + ", ".join(missing)
        )
    except Exception as error:
        _update_check(
            api,
            repository=repository,
            check_id=check_id,
            conclusion="failure",
            title="Fresh base-retarget validation failed",
            summary=str(error),
        )
        raise


def _subprocess_api(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> Mapping[str, Any] | None:
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command.extend(["--input", "-"])
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            input=json.dumps(payload) if payload is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or str(error)
        raise ValueError(f"GitHub API {method} {path} failed: {detail}") from error
    if not completed.stdout.strip():
        return None
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"GitHub API returned invalid JSON: {error}") from error
    if not isinstance(decoded, Mapping):
        raise ValueError("GitHub API response must be an object")
    return decoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--expected-base-sha", required=True)
    parser.add_argument("--expected-base-ref", required=True)
    parser.add_argument("--token", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipts = run(
        repository=args.repository,
        pr_number=args.pr_number,
        expected_head_sha=args.expected_head_sha,
        expected_base_sha=args.expected_base_sha,
        expected_base_ref=args.expected_base_ref,
        token=args.token,
        api=_subprocess_api,
    )
    for label, url in receipts.items():
        print(f"{label}: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
